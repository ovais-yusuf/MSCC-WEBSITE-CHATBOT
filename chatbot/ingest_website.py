"""
ingest_website.py

End-to-end auto-update for the MSCC chatbot knowledge base.

Pipeline:
  1. Crawl https://metrosouthchamber.com/  (same-domain, allowlisted paths)
  2. Track which pages changed (URL + content hash) via processed_sources.json
  3. Detect newsletter / Action Report PDFs linked anywhere on the site
  4. Download only the PDFs we haven't seen before (or whose bytes changed)
  5. Re-ingest ALL newsletters (newsletter_ingest.ingest_all) +
     all crawled pages (web_ingest.ingest_pages) into a single chunk list
  6. Rebuild the embeddings + BM25 index used at chat time
  7. Write a JSON run log to content/ingest_logs/<timestamp>.json

This script is safe to run repeatedly — unchanged pages and PDFs are skipped.
The chatbot picks up the new index the next time newsletter_retriever loads it
(on the next /chat request after restart, or after calling
NewsletterRetriever.reload()).

Manual run:
    python3 ingest_website.py
    python3 ingest_website.py --start-url https://metrosouthchamber.com/
    python3 ingest_website.py --max-pages 100 --no-rebuild

Environment variables:
    MSCC_START_URL          override the crawl seed (default homepage)
    MSCC_MAX_PAGES          cap on pages crawled (default 200)
    MSCC_USER_AGENT         override crawler User-Agent
    MSCC_RESPECT_ROBOTS     "0" to ignore robots.txt (default "1")
    OPENAI_API_KEY          required at chat time, not used by this script
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from processed_sources import ProcessedSources, sha256_text
from web_crawler import WebsiteCrawler, DEFAULT_START_URL, DEFAULT_USER_AGENT
from web_ingest import ingest_pages
from pdf_downloader import PDFDownloader

LOG_DIR = os.path.join(os.path.dirname(__file__), "content", "ingest_logs")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _print_section(title: str) -> None:
    bar = "═" * max(8, 60 - len(title))
    print(f"\n══ {title} {bar}")


# ── Steps ────────────────────────────────────────────────────────────────────


def crawl_site(start_url: str, max_pages: int, respect_robots: bool, user_agent: str):
    """Run the crawler; return (kept_pages, all_pdf_links_with_anchor_text)."""
    _print_section("CRAWL")
    print(f"Start URL : {start_url}")
    print(f"Max pages : {max_pages}")
    print(f"Robots    : {'respected' if respect_robots else 'IGNORED'}")

    crawler = WebsiteCrawler(
        start_url=start_url,
        max_pages=max_pages,
        respect_robots=respect_robots,
        user_agent=user_agent,
    )

    pages: list[dict] = []
    pdf_links_seen: dict[str, str] = {}   # url -> anchor text (best effort)

    for page in crawler.crawl():
        pages.append(page)
        for pdf_url in page.get("pdf_links") or ():
            # First time we see a PDF link, remember which page hosted it (used
            # later as anchor context when classifying newsletter PDFs).
            if pdf_url not in pdf_links_seen:
                pdf_links_seen[pdf_url] = page.get("title", "")

    print(f"\nPages fetched : {len(pages)}")
    print(f"PDF links seen: {len(pdf_links_seen)}")
    return pages, pdf_links_seen


def filter_changed_pages(pages: list[dict], processed: ProcessedSources):
    """
    Update processed_sources for every crawled page and return the subset that
    is NEW or whose content hash changed since last run.
    """
    new_or_changed: list[dict] = []
    for page in pages:
        url = page["url"]
        content_hash = sha256_text(page.get("text", ""))
        is_changed = processed.page_changed(url, content_hash)
        processed.record_page(
            url=url,
            content_hash=content_hash,
            title=page.get("title", ""),
            changed=is_changed,
        )
        if is_changed:
            new_or_changed.append(page)
    return new_or_changed


def download_new_pdfs(pdf_links: dict[str, str], processed: ProcessedSources):
    _print_section("PDFs")
    if not pdf_links:
        print("No PDF links found on crawled pages.")
        return []
    items = list(pdf_links.items())   # [(url, anchor_text), ...]
    downloader = PDFDownloader(processed=processed)
    new_records = downloader.download_new(items)
    print(f"New / updated PDFs: {len(new_records)}")
    return new_records


def rebuild_index(crawled_pages: list[dict]) -> dict:
    """
    Re-ingest ALL newsletters in content/Newsletters/ + all crawled pages into
    a fresh hybrid index (embeddings + BM25). Saves to newsletter_index.pkl.
    """
    _print_section("INDEX")

    # Lazy imports — ingest dependencies (pdfplumber, sentence-transformers)
    # are heavy and we want clean error messages if they're missing.
    from newsletter_ingest import ingest_all as ingest_newsletters
    from newsletter_index import build_index_from_chunks

    print("[index] Ingesting newsletter PDFs…")
    pdf_chunks = ingest_newsletters()
    print(f"[index] Newsletter chunks: {len(pdf_chunks)}")

    print("[index] Ingesting crawled webpages…")
    page_chunks = ingest_pages(crawled_pages)
    print(f"[index] Webpage chunks  : {len(page_chunks)}")

    chunks = pdf_chunks + page_chunks
    if not chunks:
        print("[index] Nothing to index — skipping rebuild.")
        return {"chunks": 0, "newsletter_chunks": 0, "webpage_chunks": 0}

    build_index_from_chunks(chunks, save=True)

    return {
        "chunks":            len(chunks),
        "newsletter_chunks": len(pdf_chunks),
        "webpage_chunks":    len(page_chunks),
    }


def write_run_log(entry: dict) -> str:
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    stamp = entry.get("started_at", _utcnow_iso()).replace(":", "-")
    log_path = os.path.join(LOG_DIR, f"{stamp}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, sort_keys=True)
    return log_path


# ── Main ─────────────────────────────────────────────────────────────────────


def run(
    start_url: str,
    max_pages: int,
    respect_robots: bool,
    user_agent: str,
    rebuild: bool = True,
) -> dict:
    started_at = _utcnow_iso()
    t0 = time.time()
    processed = ProcessedSources()

    pages, pdf_links = crawl_site(
        start_url=start_url,
        max_pages=max_pages,
        respect_robots=respect_robots,
        user_agent=user_agent,
    )

    _print_section("PAGE CHANGES")
    new_or_changed = filter_changed_pages(pages, processed)
    print(f"New or changed pages: {len(new_or_changed)}")
    for p in new_or_changed[:25]:
        print(f"  + {p['url']}")
    if len(new_or_changed) > 25:
        print(f"  …and {len(new_or_changed) - 25} more")

    new_pdfs = download_new_pdfs(pdf_links, processed)

    # Persist processed_sources before the (slower) reindex step so we don't
    # lose tracking state if rebuild fails halfway.
    processed.save()

    index_summary = {"chunks": 0, "newsletter_chunks": 0, "webpage_chunks": 0}
    if rebuild:
        index_summary = rebuild_index(crawled_pages=pages)
    else:
        print("[index] --no-rebuild set; skipping index rebuild.")

    finished_at = _utcnow_iso()

    summary = {
        "started_at":          started_at,
        "finished_at":         finished_at,
        "duration_seconds":    round(time.time() - t0, 2),
        "start_url":           start_url,
        "pages_crawled":       len(pages),
        "pages_new_or_changed": len(new_or_changed),
        "pages_changed_urls":  [p["url"] for p in new_or_changed],
        "pdf_links_seen":      len(pdf_links),
        "pdfs_new_or_updated": len(new_pdfs),
        "pdfs_added": [
            {
                "filename":    r["filename"],
                "month_label": r["month_label"],
                "url":         r["url"],
                "is_new":      r["is_new"],
                "page_count":  r["page_count"],
            }
            for r in new_pdfs
        ],
        "index_summary":       index_summary,
        "tracker_stats":       processed.stats(),
    }

    log_path = write_run_log(summary)
    print(f"\n[done] Run log: {log_path}")
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crawl MSCC, download new newsletters, and rebuild the chatbot index."
    )
    p.add_argument(
        "--start-url",
        default=os.getenv("MSCC_START_URL", DEFAULT_START_URL),
        help="Seed URL for the crawler (default: %(default)s)",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=int(os.getenv("MSCC_MAX_PAGES", "200")),
        help="Maximum pages to crawl (default: %(default)s)",
    )
    p.add_argument(
        "--user-agent",
        default=os.getenv("MSCC_USER_AGENT", DEFAULT_USER_AGENT),
        help="Override the HTTP User-Agent",
    )
    p.add_argument(
        "--ignore-robots",
        action="store_true",
        default=os.getenv("MSCC_RESPECT_ROBOTS", "1") == "0",
        help="Ignore robots.txt (default: respect it)",
    )
    p.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Skip the embeddings + BM25 rebuild step (useful for dry runs).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    summary = run(
        start_url=args.start_url,
        max_pages=args.max_pages,
        respect_robots=not args.ignore_robots,
        user_agent=args.user_agent,
        rebuild=not args.no_rebuild,
    )
    print("\n══ SUMMARY ══════════════════════════════════════════════════")
    print(json.dumps({
        "pages_crawled":       summary["pages_crawled"],
        "pages_new_or_changed": summary["pages_new_or_changed"],
        "pdfs_new_or_updated": summary["pdfs_new_or_updated"],
        "index_summary":       summary["index_summary"],
        "duration_seconds":    summary["duration_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
