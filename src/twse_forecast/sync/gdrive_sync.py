"""Google Drive REST API v3 自動同步器（支援 OAuth Refresh Token 與 Service Account）。"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as OAuthCredentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials

from twse_forecast.config import ProjectPaths

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
DRIVE_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"


@dataclass(slots=True)
class SyncedFileResult:
    """單一檔案同步至 Google Drive 的結果。"""

    local_path: str
    drive_name: str
    action: str  # created | updated
    file_id: str


def build_drive_access_token() -> str:
    """從環境變數建置並重新整理 Google Drive OAuth2 Access Token。"""
    client_id = os.environ.get("GDRIVE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GDRIVE_CLIENT_SECRET", "").strip()
    refresh_token = os.environ.get("GDRIVE_REFRESH_TOKEN", "").strip()

    if client_id and client_secret and refresh_token:
        creds = OAuthCredentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=[DRIVE_SCOPE],
        )
        creds.refresh(Request())
        if not creds.token:
            raise RuntimeError("無法透過 GDRIVE_REFRESH_TOKEN 取得有效 Access Token。")
        return str(creds.token)

    sa_raw = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON", "").strip()
    if sa_raw:
        if sa_raw.startswith("{"):
            info = json.loads(sa_raw)
            creds = ServiceAccountCredentials.from_service_account_info(info, scopes=[DRIVE_SCOPE])
        else:
            creds = ServiceAccountCredentials.from_service_account_file(sa_raw, scopes=[DRIVE_SCOPE])
        creds.refresh(Request())
        if not creds.token:
            raise RuntimeError("無法透過 Service Account 取得有效 Access Token。")
        return str(creds.token)

    raise RuntimeError(
        "找不到 Google Drive 驗證憑證。請設定個人帳號 OAuth 環境變數"
        "（GDRIVE_CLIENT_ID、GDRIVE_CLIENT_SECRET、GDRIVE_REFRESH_TOKEN）"
        "或共用雲端硬碟 Service Account（GDRIVE_SERVICE_ACCOUNT_JSON）。"
    )


def has_gdrive_credentials_configured() -> bool:
    """判斷環境變數是否已完整設定 Google Drive 同步所需憑證與資料夾 ID。"""
    if not os.environ.get("GDRIVE_FOLDER_ID", "").strip():
        return False
    has_oauth = all(
        os.environ.get(k, "").strip()
        for k in ("GDRIVE_CLIENT_ID", "GDRIVE_CLIENT_SECRET", "GDRIVE_REFRESH_TOKEN")
    )
    has_sa = bool(os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON", "").strip())
    return has_oauth or has_sa


def _encode_multipart_related(
    metadata: dict[str, Any],
    content_bytes: bytes,
    content_mime: str,
) -> tuple[bytes, str]:
    boundary = f"twse_forecast_{uuid.uuid4().hex}"
    meta_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    body = (
        f"--{boundary}\r\n"
        f"Content-Type: application/json; charset=UTF-8\r\n\r\n"
    ).encode("utf-8") + meta_bytes + (
        f"\r\n--{boundary}\r\n"
        f"Content-Type: {content_mime}\r\n\r\n"
    ).encode("utf-8") + content_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/related; boundary={boundary}"


def _find_existing_file_id(
    client: httpx.Client,
    headers: dict[str, str],
    folder_id: str,
    drive_name: str,
) -> str | None:
    safe_name = drive_name.replace("'", "\\'")
    query = f"'{folder_id}' in parents and name='{safe_name}' and trashed=false"
    resp = client.get(
        DRIVE_FILES_URL,
        headers=headers,
        params={
            "q": query,
            "fields": "files(id,name,mimeType)",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        },
    )
    resp.raise_for_status()
    files = resp.json().get("files") or []
    return str(files[0]["id"]) if files else None


def upsert_file_to_drive(
    client: httpx.Client,
    *,
    access_token: str,
    folder_id: str,
    local_path: Path,
    drive_name: str,
    source_mime: str,
    target_google_mime: str | None = None,
) -> SyncedFileResult:
    """將本地檔案上傳至指定 Google Drive 資料夾（同名檔案原位覆寫更新）。"""
    auth_headers = {"Authorization": f"Bearer {access_token}"}
    existing_id = _find_existing_file_id(client, auth_headers, folder_id, drive_name)
    content_bytes = local_path.read_bytes()

    if existing_id:
        metadata: dict[str, Any] = {"name": drive_name}
        body, content_type = _encode_multipart_related(metadata, content_bytes, source_mime)
        resp = client.patch(
            f"{DRIVE_UPLOAD_URL}/{existing_id}",
            headers={**auth_headers, "Content-Type": content_type},
            params={"uploadType": "multipart", "supportsAllDrives": "true"},
            content=body,
        )
        resp.raise_for_status()
        file_id = str(resp.json().get("id") or existing_id)
        return SyncedFileResult(str(local_path), drive_name, "updated", file_id)

    create_meta: dict[str, Any] = {"name": drive_name, "parents": [folder_id]}
    if target_google_mime:
        create_meta["mimeType"] = target_google_mime
    body, content_type = _encode_multipart_related(create_meta, content_bytes, source_mime)
    resp = client.post(
        DRIVE_UPLOAD_URL,
        headers={**auth_headers, "Content-Type": content_type},
        params={"uploadType": "multipart", "supportsAllDrives": "true"},
        content=body,
    )
    resp.raise_for_status()
    file_id = str(resp.json()["id"])
    return SyncedFileResult(str(local_path), drive_name, "created", file_id)


def _latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def collect_sync_targets(paths: ProjectPaths) -> list[tuple[Path, str, str, str | None]]:
    """彙整要同步至 Google Drive 的最新週報、月報、三張視覺化圖表、儀表板與防竄改帳本。"""
    targets: list[tuple[Path, str, str, str | None]] = []

    latest_weekly = _latest_file(paths.weekly_reports_dir, "*.md")
    if latest_weekly:
        targets.append((
            latest_weekly,
            f"台股預測週報_{latest_weekly.stem}",
            "text/markdown",
            "application/vnd.google-apps.document",
        ))

    latest_monthly = _latest_file(paths.monthly_reports_dir, "*.md")
    if latest_monthly:
        targets.append((
            latest_monthly,
            f"台股月度績效稽核_{latest_monthly.stem}",
            "text/markdown",
            "application/vnd.google-apps.document",
        ))

    charts_dir = paths.dashboard_html.parent / "charts"
    chart_mapping = [
        (charts_dir / "horizons_forecast.png", "01_五週期預測機率與點位通道.png"),
        (charts_dir / "performance_tracking.png", "02_模擬投資組合淨值與回檔追蹤.png"),
        (charts_dir / "ai_accuracy_scorecard.png", "03_AI預測準確度與Brier計分卡.png"),
    ]
    for local_p, drive_n in chart_mapping:
        if local_p.exists():
            targets.append((local_p, drive_n, "image/png", None))

    if paths.dashboard_html.exists():
        targets.append((paths.dashboard_html, "台股預測互動儀表板.html", "text/html", None))

    if paths.ledger_jsonl.exists():
        targets.append((paths.ledger_jsonl, "predictions_ledger.jsonl", "application/json", None))

    return targets


def sync_project_outputs_to_gdrive(
    paths: ProjectPaths,
    *,
    folder_id: str | None = None,
    access_token: str | None = None,
    http_client: httpx.Client | None = None,
) -> list[SyncedFileResult]:
    """將最新報表、視覺化圖表與帳本同步至 Google Drive 指定資料夾。"""
    target_folder = (folder_id or os.environ.get("GDRIVE_FOLDER_ID", "")).strip()
    if not target_folder:
        raise RuntimeError("找不到環境變數 GDRIVE_FOLDER_ID，無法指定 Google Drive 同步資料夾。")

    token = access_token or build_drive_access_token()
    targets = collect_sync_targets(paths)
    if not targets:
        return []

    results: list[SyncedFileResult] = []
    if http_client is not None:
        for local_p, drive_n, src_mime, g_mime in targets:
            results.append(
                upsert_file_to_drive(
                    http_client,
                    access_token=token,
                    folder_id=target_folder,
                    local_path=local_p,
                    drive_name=drive_n,
                    source_mime=src_mime,
                    target_google_mime=g_mime,
                )
            )
        return results

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for local_p, drive_n, src_mime, g_mime in targets:
            results.append(
                upsert_file_to_drive(
                    client,
                    access_token=token,
                    folder_id=target_folder,
                    local_path=local_p,
                    drive_name=drive_n,
                    source_mime=src_mime,
                    target_google_mime=g_mime,
                )
            )
    return results
