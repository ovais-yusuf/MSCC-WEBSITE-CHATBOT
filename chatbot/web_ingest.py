"""
web_ingest.py

Turns crawler page records (see web_crawler.extract_page) into chunk dicts
shaped IDENTICALLY to the ones produced by newsletter_ingest, so they can
be stored side-by-side in the same FAISS-style hybrid index.

Chunk schema (compatible with newsletter_index.build_index):
    {
        "text":         "...",                # used for embeddings
        "searchable":   "...",                # used for BM25 (extra keywords)
        "source":       "MSCC Website — <Title>",
        "file":         "<page slug>.html",
        "page":         <heading-section index, 1-based>,
        "section":      "<heading text>",
        "is_event":     False,

        # NEW metadata so the retriever / prompt can ground answers:
        "source_type":  "webpage",
        "source_url":   "https://...",
        "source_title": "...",
        "updated_at":   "...",
    }

Public functions:
    ingest_pages(pages)             -> list[chunk]
    page_to_chunks(page_record)     -> list[chunk]
"""

from __future__ import annotations

import os
import re
from typing import Iterable
from urllib.parse import urlparse

CHUNK_TARGET_CHARS = 900
CHUNK_OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 80


# Quick heuristic: a line that looks like a section heading we should split on
HEADING_LIKE_RE = re.compile(
    r"^(?:[A-Z][A-Z0-9 &\-’',\.]{4,80})$"
    r"|^[^\n]{3,80}:\s*$"
)


def _slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = (parsed.path or "/").strip("/")
    if not path:
        return "home.html"
    safe = re.sub(r"[^a-zA-Z0-9_\-]+", "-", path).strip("-")
    return f"{safe or 'page'}.html"


def _split_into_sections(text: str, headings: list[str]) -> list[tuple[str, str]]:
    """
    Group the page text into (heading, section_text) pairs.

    We look for headings present in the text (so we don't make up section
    boundaries). If no headings match, we return a single section.
    """
    if not text.strip():
        return []

    if not headings:
        return [("", text)]

    # Build a regex that matches any heading on its own line
    seen: set[str] = set()
    ordered: list[str] = []
    for h in headings:
        h_norm = h.strip()
        if not h_norm or len(h_norm) > 200:
            continue
        if h_norm in seen:
            continue
        seen.add(h_norm)
        ordered.append(h_norm)

    if not ordered:
        return [("", text)]

    pattern = re.compile(
        r"(?m)^(?:" + "|".join(re.escape(h) for h in ordered) + r")\s*$"
    )

    sections: list[tuple[str, str]] = []
    last_end = 0
    last_heading = ""

    for m in pattern.finditer(text):
        if m.start() > last_end:
            chunk = text[last_end:m.start()].strip()
            if chunk:
                sections.append((last_heading, chunk))
        last_heading = m.group(0).strip()
        last_end = m.end()

    tail = text[last_end:].strip()
    if tail:
        sections.append((last_heading, tail))

    if not sections:
        sections = [("", text)]
    return sections


def _split_section_into_chunks(section_text: str) -> list[str]:
    """Greedy paragraph-aware splitter targeting ~CHUNK_TARGET_CHARS."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
    chunks: list[str] = []
    current = ""
    prev_tail = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 > CHUNK_TARGET_CHARS and current:
            chunks.append((prev_tail + current).strip())
            prev_tail = current[-CHUNK_OVERLAP_CHARS:].strip() + "\n" if current else ""
            current = para + "\n"
        else:
            current = (current + para + "\n") if current else (para + "\n")

    if current.strip():
        chunks.append((prev_tail + current).strip())

    # Filter very short chunks (probably menu fragments)
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def page_to_chunks(page: dict) -> list[dict]:
    """Convert one crawler page record into one or more index chunks."""
    url = page.get("url") or ""
    title = page.get("title") or url
    text = page.get("text") or ""
    headings = page.get("headings") or []
    updated_at = page.get("updated_at") or page.get("fetched_at") or ""

    if not text.strip():
        return []

    file_label = _slug_from_url(url)
    source_label = f"MSCC Website — {title}".strip(" —") or "MSCC Website"

    out: list[dict] = []
    sections = _split_into_sections(text, headings)
    section_idx = 0

    for heading, section_text in sections:
        section_idx += 1
        for chunk_text in _split_section_into_chunks(section_text):
            searchable_extra = " ".join(filter(None, [
                title, heading, urlparse(url).path.replace("/", " "), "website mscc chamber",
            ]))
            out.append({
                "text":         chunk_text,
                "searchable":   f"{chunk_text}\n{searchable_extra}",
                "source":       source_label,
                "file":         file_label,
                "page":         section_idx,
                "section":      heading,
                "is_event":     False,
                # New metadata:
                "source_type":  "webpage",
                "source_url":   url,
                "source_title": title,
                "updated_at":   updated_at,
            })

    return out


def ingest_pages(pages: Iterable[dict]) -> list[dict]:
    """Turn many crawler page records into a flat list of chunks."""
    all_chunks: list[dict] = []
    for page in pages:
        all_chunks.extend(page_to_chunks(page))
    return all_chunks


__all__ = ["ingest_pages", "page_to_chunks"]
