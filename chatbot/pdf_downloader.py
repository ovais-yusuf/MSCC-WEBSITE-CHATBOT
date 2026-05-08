"""
pdf_downloader.py

Identify newsletter / Action Report PDFs from a list of URLs found by the
crawler, download the new ones into content/Newsletters/, and record their
metadata (filename, month/year, hash, page count) in processed_sources.

The existing newsletter_ingest.py already understands filenames like
'MAR-26-AR.pdf' and 'JUL-25.pdf' — we keep that convention so newly
downloaded PDFs are picked up automatically.

Usage:
    from pdf_downloader import PDFDownloader, infer_filename
    dl = PDFDownloader(processed_sources)
    new_files = dl.download_new(pdf_urls)
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import unquote, urlparse

import requests

from processed_sources import ProcessedSources, sha256_bytes

NEWSLETTER_DIR = os.path.join(os.path.dirname(__file__), "content", "Newsletters")

DEFAULT_USER_AGENT = "MSCC-Chatbot-Indexer/1.0 (+https://metrosouthchamber.com/)"

# Map full month → 3-letter code used in existing filenames
MONTH_FULL_TO_CODE = {
    "january": "JAN", "february": "FEB", "march": "MAR", "april": "APR",
    "may": "MAY", "june": "JUN", "july": "JUL", "august": "AUG",
    "september": "SEP", "october": "OCT", "november": "NOV", "december": "DEC",
}
MONTH_ABBR_TO_CODE = {
    "jan": "JAN", "feb": "FEB", "mar": "MAR", "apr": "APR",
    "may": "MAY", "jun": "JUN", "jul": "JUL", "aug": "AUG",
    "sep": "SEP", "sept": "SEP", "oct": "OCT", "nov": "NOV", "dec": "DEC",
    # The Chamber publishes a combined July/August issue
    "julyaug": "JULYAUG", "jul-aug": "JULYAUG", "jul_aug": "JULYAUG",
    "july-august": "JULYAUG", "july_august": "JULYAUG",
}

NEWSLETTER_HINT_RE = re.compile(
    # Match common newsletter / Action Report markers, including filename
    # patterns like "MAY-26-AR" where -AR sits between two dashes.
    r"(action[\s_\-]*report|newsletter|monthly[\s_\-]*report|chamber[\s_\-]*update"
    r"|[\-_]ar[\-_\.]|[\-_]ar$|ar[\-_]\d|action[\-_]?report)",
    re.IGNORECASE,
)

# Standard months + the special "JULYAUG" / "JULY-AUG" / "JUL-AUG" combo
# the Chamber publishes once a year for the summer issue.
MONTH_YEAR_IN_TEXT_RE = re.compile(
    r"\b("
    r"julyaug|jul[\s_\-]?aug|july[\s_\-]?august|"
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
    r")\b[\s\-_]*?(\d{2,4})",
    re.IGNORECASE,
)


# ── PDF link classification ───────────────────────────────────────────────────


def looks_like_newsletter(url: str, anchor_text: str = "") -> bool:
    """
    Decide whether a PDF URL is likely a Chamber newsletter / Action Report.
    We use BOTH the URL and the anchor link text the crawler captured.
    """
    if not url.lower().endswith(".pdf"):
        return False
    blob = f"{url} {anchor_text}".lower()

    if NEWSLETTER_HINT_RE.search(blob):
        return True
    # Filenames like MAR-26-AR.pdf or feb_2026_action_report.pdf
    fname = os.path.basename(urlparse(url).path).lower()
    if MONTH_YEAR_IN_TEXT_RE.search(fname):
        return True
    return False


# ── Filename inference ────────────────────────────────────────────────────────


def _normalize_year(y: str) -> str:
    if len(y) == 4:
        return y[-2:]
    if len(y) == 2:
        return y
    return ""


_AR_FLAG_RE = re.compile(
    # Detects "Action Report" wording OR -AR suffix in the URL/anchor.
    # Patterns matched: "action report", "MAY-26-AR.pdf", "MAY-26-AR_.pdf",
    # "MAY-26 AR", "ar_-26".  We deliberately do NOT match a bare " ar "
    # because that produces false positives.
    r"action[\s_\-]*report"
    r"|[\-_]ar[\-_\.\s]"          # -AR followed by separator
    r"|[\-_]ar$",                 # -AR at end of string
    re.IGNORECASE,
)


def infer_filename(
    url: str,
    anchor_text: str = "",
    is_action_report: Optional[bool] = None,
) -> str:
    """
    Return a canonical filename in the same format newsletter_ingest understands:
        MAR-26-AR.pdf       (Action Report)
        JUL-25.pdf          (regular newsletter)
        JULYAUG-25-AR.pdf   (combined July/August Action Report)

    Falls back to the URL's original basename if month/year cannot be inferred.
    """
    raw_name = unquote(os.path.basename(urlparse(url).path))
    blob = f"{raw_name} {anchor_text}".lower()

    if is_action_report is None:
        is_action_report = bool(_AR_FLAG_RE.search(blob))

    m = MONTH_YEAR_IN_TEXT_RE.search(blob)
    if not m:
        return raw_name or "newsletter.pdf"

    month_token = m.group(1).lower().replace(" ", "")
    year_token = _normalize_year(m.group(2))
    code = MONTH_FULL_TO_CODE.get(month_token) or MONTH_ABBR_TO_CODE.get(month_token)
    if not code or not year_token:
        return raw_name or "newsletter.pdf"

    suffix = "-AR" if is_action_report else ""
    return f"{code}-{year_token}{suffix}.pdf"


def infer_month_label(filename: str) -> str:
    """Mirror newsletter_ingest.infer_month_label so metadata stays consistent."""
    # Lazy import to avoid pulling pdfplumber at module load if unused
    from newsletter_ingest import infer_month_label as _inf
    return _inf(filename)


# ── Downloader ────────────────────────────────────────────────────────────────


class PDFDownloader:
    """Downloads newsletter PDFs that we haven't already processed."""

    def __init__(
        self,
        processed: ProcessedSources,
        target_dir: str = NEWSLETTER_DIR,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_s: float = 60.0,
        max_bytes: int = 25 * 1024 * 1024,   # 25 MB cap per PDF
    ):
        self.processed = processed
        self.target_dir = target_dir
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        Path(self.target_dir).mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

    def download_new(
        self,
        pdf_links: Iterable[tuple[str, str]] | Iterable[str],
    ) -> list[dict]:
        """
        Download every newsletter PDF we haven't seen before (or whose bytes
        have changed). Accepts either:
            - iterable of url strings, OR
            - iterable of (url, anchor_text) tuples

        Returns list of records describing the new/updated files:
            { "url", "path", "filename", "month_label", "hash",
              "is_new": bool, "changed": bool }
        """
        records: list[dict] = []
        seen_urls: set[str] = set()

        for item in pdf_links:
            if isinstance(item, tuple):
                url, anchor = item
            else:
                url, anchor = item, ""

            if url in seen_urls:
                continue
            seen_urls.add(url)

            if not looks_like_newsletter(url, anchor):
                continue

            rec = self._download_one(url, anchor)
            if rec:
                records.append(rec)

        return records

    # ── internals ────────────────────────────────────────────────────────

    def _download_one(self, url: str, anchor: str) -> Optional[dict]:
        try:
            resp = self._session.get(url, timeout=self.timeout_s, stream=True)
        except requests.RequestException as exc:
            print(f"[pdf] FETCH ERROR {url}: {exc}")
            return None

        if resp.status_code != 200:
            print(f"[pdf] {resp.status_code} {url}")
            return None

        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "pdf" not in ctype and not url.lower().endswith(".pdf"):
            print(f"[pdf] not a PDF (content-type={ctype}): {url}")
            return None

        # Read the body with a hard byte cap
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > self.max_bytes:
                print(f"[pdf] too large (> {self.max_bytes} bytes): {url}")
                return None
            chunks.append(chunk)
        data = b"".join(chunks)

        if not data.startswith(b"%PDF"):
            print(f"[pdf] not a valid PDF body: {url}")
            return None

        file_hash = sha256_bytes(data)

        if not self.processed.pdf_changed(url, file_hash):
            return None  # unchanged → nothing to do

        filename = infer_filename(url, anchor)
        path = os.path.join(self.target_dir, filename)

        is_new = not os.path.exists(path)
        changed = not is_new
        with open(path, "wb") as f:
            f.write(data)

        page_count = _safe_page_count(path)
        month_label = infer_month_label(filename)

        self.processed.record_pdf(
            key=url,
            file_hash=file_hash,
            filename=filename,
            month_label=month_label,
            page_count=page_count,
        )

        print(
            f"[pdf] {'downloaded' if is_new else 'updated'}: {filename}  "
            f"({month_label}, {page_count} pages)  ←  {url}"
        )

        return {
            "url":          url,
            "path":         path,
            "filename":     filename,
            "month_label":  month_label,
            "hash":         file_hash,
            "page_count":   page_count,
            "is_new":       is_new,
            "changed":      changed,
        }


def _safe_page_count(path: str) -> int:
    """Count PDF pages without crashing the pipeline if pdfplumber chokes."""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0


__all__ = [
    "PDFDownloader",
    "looks_like_newsletter",
    "infer_filename",
    "infer_month_label",
    "NEWSLETTER_DIR",
]
