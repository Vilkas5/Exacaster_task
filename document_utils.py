"""Text extraction helpers for uploaded documents."""

from __future__ import annotations

import io

import docx
from pypdf import PdfReader


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from an uploaded file's raw bytes.

    Supports .txt, .pdf, and .docx. Raises ValueError for anything else.
    """
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix == "txt":
        return data.decode("utf-8", errors="replace")

    if suffix == "pdf":
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    if suffix == "docx":
        document = docx.Document(io.BytesIO(data))
        paragraphs = [p.text for p in document.paragraphs]
        return "\n".join(paragraphs)

    raise ValueError(f"Unsupported file type: .{suffix}")
