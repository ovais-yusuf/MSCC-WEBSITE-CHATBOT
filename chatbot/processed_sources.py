"""
processed_sources.py

Tracks which URLs and PDF files have already been ingested into the index,
plus a content hash so we can detect when a page changes and re-ingest it.

Storage: JSON file at content/processed_sources.json

Schema:
{
  "pages": {
      "<url>": {
          "hash":          "<sha256 of cleaned text>",
          "title":         "...",
          "last_seen_at":  "2026-05-07T13:55:00Z",
          "last_changed_at": "2026-05-07T13:55:00Z"
      }
  },
  "pdfs":  {
      "<url-or-relative-path>": {
          "hash":          "<sha256 of file bytes>",
          "filename":      "MAR-26-AR.pdf",
          "month_label":   "March 2026 Action Report",
          "downloaded_at": "2026-05-07T13:55:00Z",
          "page_count":    8
      }
  }
}
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_PATH = os.path.join(
    os.path.dirname(__file__), "content", "processed_sources.json"
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ProcessedSources:
    """JSON-backed tracker for crawled URLs and downloaded PDFs."""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._data = {"pages": {}, "pdfs": {}}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._data["pages"] = data.get("pages", {}) or {}
                self._data["pdfs"] = data.get("pdfs", {}) or {}
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[processed] Could not read {self.path} ({exc}); starting fresh.")

    def save(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path + ".tmp"
        with self._lock:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)

    # ── Page tracking ─────────────────────────────────────────────────────

    def page_changed(self, url: str, content_hash: str) -> bool:
        """True if URL is new OR its hash differs from what we last stored."""
        rec = self._data["pages"].get(url)
        return rec is None or rec.get("hash") != content_hash

    def record_page(
        self,
        url: str,
        content_hash: str,
        title: str = "",
        changed: Optional[bool] = None,
    ) -> None:
        rec = self._data["pages"].get(url, {})
        now = _utcnow()
        if changed is None:
            changed = rec.get("hash") != content_hash
        rec["hash"] = content_hash
        rec["title"] = title or rec.get("title", "")
        rec["last_seen_at"] = now
        if changed:
            rec["last_changed_at"] = now
        self._data["pages"][url] = rec

    # ── PDF tracking ──────────────────────────────────────────────────────

    def pdf_seen(self, key: str) -> bool:
        """True if we've already processed this PDF (by URL or relative path)."""
        return key in self._data["pdfs"]

    def pdf_changed(self, key: str, file_hash: str) -> bool:
        rec = self._data["pdfs"].get(key)
        return rec is None or rec.get("hash") != file_hash

    def record_pdf(
        self,
        key: str,
        file_hash: str,
        filename: str,
        month_label: str = "",
        page_count: int = 0,
    ) -> None:
        self._data["pdfs"][key] = {
            "hash": file_hash,
            "filename": filename,
            "month_label": month_label,
            "downloaded_at": _utcnow(),
            "page_count": int(page_count),
        }

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "pages_tracked": len(self._data["pages"]),
            "pdfs_tracked": len(self._data["pdfs"]),
        }
