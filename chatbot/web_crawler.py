"""
web_crawler.py

BFS crawler for the official MSCC website.

Responsibilities:
  - Start from https://metrosouthchamber.com/ and walk same-domain links
  - Honor robots.txt (best-effort — defaults to "allow" if file missing)
  - Stay within an allowlist of relevant URL prefixes (events, membership, etc.)
  - Skip irrelevant external links, calendar feeds, query-string pagination noise
  - Return one record per page:
        {
          "url":         str,
          "title":       str,
          "text":        str,   # cleaned main-content text
          "headings":    list[str],
          "updated_at":  str | "",
          "fetched_at":  str (UTC ISO),
          "html_status": int,
          "pdf_links":   list[str],   # absolute URLs to PDFs found on this page
        }

Designed to be safe to call repeatedly — caller decides what to do with
unchanged pages via processed_sources.ProcessedSources.

Network is the only side effect. We DO NOT touch the index here.
"""

from __future__ import annotations

import re
import time
from collections import deque
from datetime import datetime, timezone
from typing import Iterable, Iterator, Optional
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

DEFAULT_START_URL = "https://metrosouthchamber.com/"
DEFAULT_USER_AGENT = "MSCC-Chatbot-Indexer/1.0 (+https://metrosouthchamber.com/)"

# Path prefixes (lowercase, no leading slash) we consider RELEVANT for the
# chatbot's knowledge base. Anything else inside the same domain is skipped.
#
# These match the FIRST URL path segment (e.g. /news-media/action-report/
# matches because "news-media" is in this set).
RELEVANT_PATH_PREFIXES = (
    "",                       # the homepage itself
    "events",
    "event",
    "membership",
    "member",
    "news",
    "news-media",             # MSCC actually hosts /news-media/action-report/
    "blog",
    "press",
    "about",
    "staff",
    "team",
    "contact",
    "resources",
    "newsletter",
    "newsletters",
    "action-report",
    "e-update",
    "publications",
    "programs",
    "leadership",
    "leadership-metro-south",
    "gbyp",
    "young-professionals",
    "greater-brockton-young-professionals",
    "visitors",
    "visit-metro-south",
    "business-resources",
    "business-front-door",
    "economic-development",
    "one-stop",
    "community-one-stop",
    "community-one-stop-for-growth",
    "region",
    "metro-south-region",
    "partner-programs",
    "partners-programs",
)

# Suffixes / patterns we never want to fetch as HTML
SKIP_SUFFIXES = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".mp3", ".mp4", ".mov", ".avi", ".webm",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
)

# WordPress / common noise we want to skip
SKIP_PATH_FRAGMENTS = (
    "/wp-admin", "/wp-login", "/wp-json", "/feed",
    "/cdn-cgi/", "/xmlrpc.php",
    "?share=", "?replytocom=",
    "/tag/", "/author/", "/category/page/", "/page/",
)

# CSS selectors removed from each page before extraction
NOISE_SELECTORS = (
    "header", "footer", "nav",
    ".site-header", ".site-footer",
    ".elementor-location-header", ".elementor-location-footer",
    ".menu", ".main-menu", ".sub-menu", ".breadcrumb", ".breadcrumbs",
    ".widget_search", ".search-form",
    ".cookie", ".cookie-banner", ".gdpr",
    ".social-icons", ".social-links",
    "script", "style", "noscript", "iframe",
)

# Tag whose text we want to keep as a "heading"
HEADING_TAGS = ("h1", "h2", "h3", "h4")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── URL helpers ────────────────────────────────────────────────────────────────


def _normalize_url(url: str) -> str:
    """Strip fragment and trailing slash inconsistencies."""
    url, _ = urldefrag(url)
    # Remove a trailing slash on path-only URLs (but keep "/" for root)
    parsed = urlparse(url)
    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return parsed._replace(path=path, fragment="").geturl()


def _is_same_domain(a: str, b: str) -> bool:
    """
    Same domain ignoring an optional leading 'www.' so that
    https://metrosouthchamber.com/ and https://www.metrosouthchamber.com/
    are treated as one site.
    """
    def _strip(host: str) -> str:
        host = host.lower()
        return host[4:] if host.startswith("www.") else host
    return _strip(urlparse(a).netloc) == _strip(urlparse(b).netloc)


def _is_relevant_path(path: str) -> bool:
    p = (path or "/").lstrip("/").lower()
    if p == "":
        return True
    # First path segment must match an allowlisted prefix
    first = p.split("/", 1)[0]
    return first in RELEVANT_PATH_PREFIXES


def _should_skip(url: str) -> bool:
    low = url.lower()
    if any(low.endswith(suf) for suf in SKIP_SUFFIXES):
        return True
    if any(frag in low for frag in SKIP_PATH_FRAGMENTS):
        return True
    return False


def _looks_like_pdf(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


# ── Robots.txt handling ───────────────────────────────────────────────────────


class _Robots:
    """
    Cache one robots.txt decision per (scheme + netloc).

    We use Python's stdlib RobotFileParser to read the file, but we ALSO check
    the raw text ourselves: stdlib's parser has long-standing edge cases where
    it returns False for sites that actually allow everything (e.g. WordPress
    sites that emit `Disallow:` with no path inside a Yoast block). When the
    raw file contains no real Disallow rule for our user-agent, we trust the
    file and return True regardless of what stdlib decided.
    """

    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self._cache: dict[str, bool | None] = {}   # base -> "permissive?" or None
        self._raw: dict[str, str] = {}             # base -> raw robots.txt text

    def _fetch_raw(self, base: str) -> str:
        try:
            resp = requests.get(
                urljoin(base, "/robots.txt"),
                headers={"User-Agent": self.user_agent},
                timeout=10,
            )
            if resp.status_code == 200 and "html" not in (resp.headers.get("Content-Type") or "").lower():
                return resp.text
        except requests.RequestException:
            pass
        return ""

    @staticmethod
    def _is_permissive(raw: str, user_agent: str) -> bool:
        """
        Return True if the robots.txt has no DISALLOW rules that would block
        our crawler. We scan the section(s) for User-agent: * and for our UA
        prefix; an empty `Disallow:` (or no Disallow lines at all) means allow.
        """
        if not raw.strip():
            return True   # missing/empty file → allow

        ua_low = user_agent.lower().split("/", 1)[0]
        sections: list[tuple[list[str], list[str]]] = []   # (uas, disallow_paths)
        cur_uas: list[str] = []
        cur_dis: list[str] = []

        def flush():
            if cur_uas:
                sections.append((cur_uas[:], cur_dis[:]))

        for raw_line in raw.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip()
            if key == "user-agent":
                if cur_dis or cur_uas == []:
                    # New group — start fresh if previous group had rules
                    if cur_uas and cur_dis:
                        flush()
                        cur_uas = []
                        cur_dis = []
                cur_uas.append(val.lower())
            elif key == "disallow":
                cur_dis.append(val)
            # We intentionally ignore Allow / Crawl-delay / Sitemap for this check
        flush()

        applicable = []
        for uas, dis in sections:
            if "*" in uas or any(ua_low in u for u in uas):
                applicable.extend(dis)

        if not applicable:
            return True

        # If every Disallow value is empty, the file is allow-all.
        return all(p.strip() in ("", "/") and p.strip() != "/" for p in applicable) or \
               all(p.strip() == "" for p in applicable)

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"

        if base not in self._cache:
            raw = self._fetch_raw(base)
            self._raw[base] = raw
            permissive = self._is_permissive(raw, self.user_agent)
            self._cache[base] = permissive

        if self._cache[base]:
            return True   # robots.txt is allow-all for us

        # Otherwise fall back to stdlib for accurate per-path checks.
        rp = RobotFileParser()
        rp.parse(self._raw[base].splitlines())
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True


# ── HTML cleaning + extraction ────────────────────────────────────────────────


def _strip_noise(soup: BeautifulSoup) -> None:
    for sel in NOISE_SELECTORS:
        for el in soup.select(sel):
            el.decompose()


def _detect_main(soup: BeautifulSoup):
    """Pick the most likely main-content container."""
    for sel in (
        "main",
        "article",
        '[role="main"]',
        ".entry-content",
        ".post-content",
        ".page-content",
        "#content",
        "#main",
    ):
        node = soup.select_one(sel)
        if node and node.get_text(strip=True):
            return node
    return soup.body or soup


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_updated_at(soup: BeautifulSoup) -> str:
    """Best-effort hunt for an article published/modified timestamp."""
    candidates = []
    for sel in ("meta[property='article:modified_time']",
                "meta[property='article:published_time']",
                "meta[name='last-modified']",
                "time[datetime]"):
        node = soup.select_one(sel)
        if not node:
            continue
        val = node.get("content") or node.get("datetime") or ""
        if val:
            candidates.append(val.strip())
    return candidates[0] if candidates else ""


def extract_page(html: str, url: str) -> dict:
    """Pull title, headings, clean text, updated_at, and PDF/internal links."""
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)

    updated_at = _extract_updated_at(soup)

    # All anchors BEFORE we strip noise — the menu often links to important
    # pages too, and we need the PDF links from sidebars/widgets.
    pdf_links: list[str] = []
    page_links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        absolute = _normalize_url(urljoin(url, href))
        if _looks_like_pdf(absolute):
            pdf_links.append(absolute)
        else:
            page_links.append(absolute)

    # Now clean the soup so we don't carry menus/footers into the body text.
    _strip_noise(soup)
    main = _detect_main(soup)

    headings = []
    for tag in HEADING_TAGS:
        for h in main.find_all(tag):
            txt = h.get_text(" ", strip=True)
            if txt:
                headings.append(txt)

    text = _collapse_whitespace(main.get_text("\n", strip=True))

    return {
        "url":          url,
        "title":        title,
        "text":         text,
        "headings":     headings,
        "updated_at":   updated_at,
        "fetched_at":   _utcnow_iso(),
        "pdf_links":    sorted(set(pdf_links)),
        "page_links":   sorted(set(page_links)),
    }


# ── Crawler ───────────────────────────────────────────────────────────────────


class WebsiteCrawler:
    """
    BFS crawler limited to one domain + an allowlist of relevant path prefixes.

    Yields page records. The caller decides whether to ingest, skip (unchanged),
    or just collect PDF links.
    """

    def __init__(
        self,
        start_url: str = DEFAULT_START_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        max_pages: int = 200,
        request_delay_s: float = 0.5,
        timeout_s: float = 20.0,
        respect_robots: bool = True,
        path_prefixes: Iterable[str] = RELEVANT_PATH_PREFIXES,
    ):
        self.start_url = _normalize_url(start_url)
        self.user_agent = user_agent
        self.max_pages = max_pages
        self.request_delay_s = request_delay_s
        self.timeout_s = timeout_s
        self.respect_robots = respect_robots
        self.path_prefixes = tuple(p.lower() for p in path_prefixes)
        self._robots = _Robots(user_agent)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

    # Public API

    def crawl(self) -> Iterator[dict]:
        seen: set[str] = set()
        queue: deque[str] = deque([self.start_url])
        fetched_count = 0

        while queue and fetched_count < self.max_pages:
            url = queue.popleft()
            if url in seen:
                continue
            seen.add(url)

            if _should_skip(url):
                continue
            if not _is_same_domain(url, self.start_url):
                continue
            if not self._is_relevant(url):
                continue
            if self.respect_robots and not self._robots.can_fetch(url):
                print(f"[crawl] robots.txt blocks: {url}")
                continue

            try:
                resp = self._session.get(url, timeout=self.timeout_s, allow_redirects=True)
            except requests.RequestException as exc:
                print(f"[crawl] FETCH ERROR {url}: {exc}")
                continue

            time.sleep(self.request_delay_s)

            if resp.status_code != 200:
                print(f"[crawl] {resp.status_code} {url}")
                continue

            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "html" not in content_type:
                continue

            try:
                page = extract_page(resp.text, url)
            except Exception as exc:
                print(f"[crawl] EXTRACT ERROR {url}: {exc}")
                continue

            page["html_status"] = resp.status_code
            fetched_count += 1
            yield page

            # Enqueue same-domain page links
            for link in page.get("page_links") or ():
                norm = _normalize_url(link)
                if norm in seen:
                    continue
                if not _is_same_domain(norm, self.start_url):
                    continue
                if _should_skip(norm):
                    continue
                if not self._is_relevant(norm):
                    continue
                queue.append(norm)

    # Helpers

    def _is_relevant(self, url: str) -> bool:
        path = urlparse(url).path or "/"
        # Allow homepage explicitly
        if path in ("", "/"):
            return True
        first = path.lstrip("/").split("/", 1)[0].lower()
        return first in self.path_prefixes


__all__ = [
    "WebsiteCrawler",
    "extract_page",
    "DEFAULT_START_URL",
    "DEFAULT_USER_AGENT",
]
