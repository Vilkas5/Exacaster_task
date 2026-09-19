"""Fetch documents from a public Google Drive folder via the Drive API v3.

Read-only, API-key auth — works because the target folder is shared as
"Anyone with the link can view". No OAuth, no service account.
"""

from __future__ import annotations

import re

import requests

DRIVE_API = "https://www.googleapis.com/drive/v3"

# Native Google Docs/Slides have no raw bytes — they must be exported to a
# concrete format. We export to plain text, which document_utils already
# knows how to read as-is.
_EXPORT_MIME_TYPES = {
    "application/vnd.google-apps.document": ("text/plain", "txt"),
    "application/vnd.google-apps.presentation": ("text/plain", "txt"),
}

# Regular files: mimeType -> the extension document_utils.extract_text expects.
_SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


def extract_folder_id(url_or_id: str) -> str:
    """Accept either a full Drive folder URL or a bare folder ID."""
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url_or_id)
    return match.group(1) if match else url_or_id.strip()


def list_files(folder_id: str, api_key: str) -> list[dict]:
    """List files directly inside a public Drive folder (no recursion)."""
    resp = requests.get(
        f"{DRIVE_API}/files",
        params={
            "q": f"'{folder_id}' in parents and trashed = false",
            "key": api_key,
            "fields": "files(id,name,mimeType)",
            "pageSize": 1000,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("files", [])


def download_file(file: dict, api_key: str) -> tuple[str, bytes] | None:
    """Download one file's raw bytes, exporting native Google Docs to text.

    Returns (filename, content_bytes), or None for a type we don't extract
    text from (folders, images, sheets, ...) so the caller can skip it.
    """
    file_id = file["id"]
    mime_type = file["mimeType"]
    name = file["name"]

    if mime_type in _EXPORT_MIME_TYPES:
        export_mime, ext = _EXPORT_MIME_TYPES[mime_type]
        resp = requests.get(
            f"{DRIVE_API}/files/{file_id}/export",
            params={"mimeType": export_mime, "key": api_key},
            timeout=60,
        )
        resp.raise_for_status()
        return f"{name}.{ext}", resp.content

    if mime_type in _SUPPORTED_MIME_TYPES:
        ext = _SUPPORTED_MIME_TYPES[mime_type]
        resp = requests.get(
            f"{DRIVE_API}/files/{file_id}",
            params={"alt": "media", "key": api_key},
            timeout=60,
        )
        resp.raise_for_status()
        if not name.lower().endswith(f".{ext}"):
            name = f"{name}.{ext}"
        return name, resp.content

    return None
