"""
newsletter_ingest.py

Reads every PDF inside content/Newsletters/**
Extracts, cleans, and chunks text with metadata.

Special handling:
  - CHAMBER Calendar sections → one chunk per calendar event
  - Pipe-separated event lines → parsed into structured fields
  - Regular sections → paragraph-based chunks with overlap

Returns a list of chunk dicts ready for indexing.
"""

import os
import re
import pdfplumber

NEWSLETTER_ROOT = os.path.join(os.path.dirname(__file__), "content", "Newsletters")

MONTH_MAP = {
    "JAN": "January", "FEB": "February", "MAR": "March",
    "APR": "April",   "MAY": "May",      "JUN": "June",
    "JUL": "July",    "AUG": "August",   "SEP": "September",
    "SEPT": "September",
    "OCT": "October", "NOV": "November", "DEC": "December",
    # Combined summer issue (e.g. JULYAUG-25-AR.pdf → "July/August 2025 …")
    "JULYAUG": "July/August",
}

MONTH_NAMES = list(MONTH_MAP.values())   # for runtime detection

CHUNK_TARGET_CHARS = 900
CHUNK_OVERLAP_CHARS = 150

# Calendar column starts at ~55% of page width (verified: x≈400 on 720pt page)
CAL_COLUMN_START_RATIO = 0.54

# ── Regex patterns ────────────────────────────────────────────────────────────

# Matches: "Friday, January 30 | 11:45AM - 1:30PM | Venue | Address"
PIPE_EVENT_RE = re.compile(
    r"(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*)?"
    r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(?P<day>\d{1,2})"
    r"\s*\|\s*(?P<time>[\d:APMapm\s\-–]+)"
    r"\s*\|\s*(?P<rest>.+)",
    re.IGNORECASE,
)

# Matches a standalone month heading inside calendar block: "January", "February" etc.
MONTH_HEADING_RE = re.compile(
    r"^(?P<month>January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s*$",
    re.IGNORECASE,
)

# Matches lines like "14" or "05" (day number alone) or "05 Event Name 5:00PM"
CAL_DAY_RE = re.compile(
    r"^(?P<day>\d{1,2})\s+(?P<rest>.+)$|^(?P<day2>\d{1,2})\s*$"
)

# Time pattern anywhere in a line
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b")

# Heading-like lines: all caps or ends with colon only
HEADING_RE = re.compile(r"^([A-Z][A-Z &,'\-]{4,}|.{3,60}:)\s*$")

# Calendar section triggers
CAL_TRIGGERS = re.compile(
    r"chamber\s+calendar|calendar\s+sponsor|caalleennddaarr",
    re.IGNORECASE,
)


# ── Month inference ───────────────────────────────────────────────────────────

def infer_month_label(filename: str) -> str:
    """
    Build a human-readable label from a newsletter filename.

    Handles the canonical formats produced by pdf_downloader:
      MAR-26-AR.pdf       → "March 2026 Action Report"
      JUL-25.pdf          → "July 2025"
      JULYAUG-25-AR.pdf   → "July/August 2025 Action Report"

    Also tolerates older variants from the archive:
      julyaug19-ar.pdf    → "July/August 2019 Action Report"
      JULYAUGUST-22-AR.pdf → "July/August 2022 Action Report"
      Mar08ar.pdf         → "March 2008 Action Report"
    """
    stem = os.path.splitext(filename)[0]
    is_action_report = bool(re.search(r"(?i)(?<![a-z])ar(?![a-z])", stem))

    # Look for a 2-digit year inside the stem (preferred) or 4-digit year.
    year_m = re.search(r"(?<!\d)(\d{4}|\d{2})(?!\d)", stem)
    year = ""
    if year_m:
        y = year_m.group(1)
        year = y if len(y) == 4 else ("20" + y if int(y) < 80 else "19" + y)

    # Normalize month token: strip non-alphabetics, look up in extended map.
    upper = re.sub(r"[^A-Z]", "", stem.upper())
    month_token = ""
    # Longest-prefix match so "JULYAUGUST" wins over "JULY", and "SEPT" over "SEP".
    extended = {
        "JULYAUGUST": "July/August", "JULYAUG":   "July/August",
        "JANUARY":   "January",  "FEBRUARY":  "February", "MARCH":     "March",
        "APRIL":     "April",    "JUNE":      "June",     "JULY":      "July",
        "AUGUST":    "August",   "SEPTEMBER": "September", "OCTOBER":  "October",
        "NOVEMBER":  "November", "DECEMBER":  "December",
        "JAN": "January", "FEB": "February", "MAR": "March", "APR": "April",
        "MAY": "May",     "JUN": "June",     "JUL": "July",  "AUG": "August",
        "SEPT": "September", "SEP": "September", "OCT": "October",
        "NOV": "November",   "DEC": "December",
    }
    for token in sorted(extended, key=len, reverse=True):
        if upper.startswith(token):
            month_token = extended[token]
            break

    if not month_token:
        return os.path.splitext(filename)[0].capitalize()

    label = f"{month_token} {year}".strip()
    if is_action_report:
        label += " Action Report"
    return label


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_line(line: str) -> str:
    line = line.strip()
    if re.fullmatch(r"\d{1,3}", line):      # bare page numbers
        return ""
    if re.fullmatch(r"[.\-–_]{4,}", line):  # dot/dash filler
        return ""
    line = re.sub(r"\s{3,}", "  ", line)
    return line


def extract_pages(pdf_path: str) -> list[dict]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            lines = [clean_line(l) for l in raw.splitlines()]
            lines = [l for l in lines if l]

            # For calendar pages, also extract just the right column
            # so we get clean calendar text without Board of Directors noise
            cal_lines = None
            if _page_has_calendar_markers(lines):
                cal_lines = _extract_calendar_column(page)

            pages.append({
                "page_num":  i,
                "lines":     lines,
                "cal_lines": cal_lines,   # None if not a calendar page
            })
    return pages


def _page_has_calendar_markers(lines: list[str]) -> bool:
    """Quick check: does this page likely have a structured calendar block?"""
    text = " ".join(lines).lower()
    if "chamber calendar" in text or "caalleennddaarr" in text:
        return True
    day_hits = sum(1 for l in lines if re.fullmatch(r"\d{1,2}", l.strip()))
    return day_hits >= 3


def _extract_calendar_column(page) -> list[str]:
    """
    Crop to the right-hand calendar column and extract clean text.
    The calendar starts at ~54% of the page width based on layout analysis.
    """
    x_start = page.width * CAL_COLUMN_START_RATIO
    cropped = page.crop((x_start, 0, page.width, page.height))
    raw = cropped.extract_text(x_tolerance=3, y_tolerance=4) or ""
    lines = [clean_line(l) for l in raw.splitlines()]
    return [l for l in lines if l]


# ── Calendar event parsing ────────────────────────────────────────────────────

def make_event_chunk(event_name: str, date: str, time: str,
                     location: str, source_label: str,
                     page_num: int, filename: str) -> dict:
    """Build a structured chunk for a single calendar event."""
    parts = [p for p in [date, time, location] if p]
    text = f"EVENT: {event_name}\n" + "\n".join(parts)
    searchable = (
        f"{event_name} {date} {time} {location} "
        f"calendar event chamber {source_label}"
    ).strip()
    return {
        "text":      text,
        "searchable": searchable,   # extra field for BM25 boosting
        "source":    source_label,
        "file":      filename,
        "page":      page_num,
        "section":   "Calendar",
        "is_event":  True,
        "event_name": event_name,
        "date":       date,
        "time":       time,
        "location":   location,
        # Unified metadata so retriever can format prompt consistently
        "source_type":  "newsletter_pdf",
        "source_title": source_label,
    }


def parse_pipe_events(lines: list[str], source_label: str,
                      page_num: int, filename: str) -> list[dict]:
    """
    Find and parse pipe-separated event lines:
    'Friday, January 30 | 11:45AM - 1:30PM | Stonehill College | 320 Washington St, Easton'
    """
    chunks = []
    for line in lines:
        m = PIPE_EVENT_RE.search(line)
        if not m:
            continue
        month    = m.group("month")
        day      = m.group("day")
        time_str = m.group("time").strip()
        rest     = m.group("rest")

        # Split rest by | to get venue and address
        rest_parts = [p.strip() for p in rest.split("|")]
        venue   = rest_parts[0] if rest_parts else ""
        address = rest_parts[1] if len(rest_parts) > 1 else ""
        location = f"{venue}, {address}".strip(", ") if address else venue

        # Event name is usually on the same line before or after — use venue as fallback
        # Try to grab any text before the first | that isn't the date/weekday
        name_match = re.match(
            r"^(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*)?"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+\d{1,2}\s*\|",
            line, re.IGNORECASE,
        )
        event_name = venue  # default: use venue name

        chunks.append(make_event_chunk(
            event_name=event_name,
            date=f"{month} {day}",
            time=time_str,
            location=location,
            source_label=source_label,
            page_num=page_num,
            filename=filename,
        ))
    return chunks


def parse_structured_calendar(cal_lines: list[str], source_label: str,
                               page_num: int, filename: str) -> list[dict]:
    """
    Parse the CHAMBER Calendar block extracted from the cropped right column.

    After cropping, lines look like:
        'anuary'                                    ← partial month (first char clipped)
        '14 Ambassador Meeting 9:00AM'              ← DD + name + time
        'Virtual Meeting via Zoom'                  ← location
        '28 Board of Directors Meeting 12:00PM'
        'Metro South Chamber: 60 School Street'
        '12 Good Morning Metro South: FIFA'         ← no time on this line
        'Garner Hotel: 405 Westgate Drive 8:45AM'  ← location + time
    """
    # Suffix→month lookup (handles partial names like "anuary", "ebruary", etc.)
    SUFFIX_TO_MONTH = {
        "january":   "January",   "anuary":   "January",
        "february":  "February",  "ebruary":  "February",
        "march":     "March",     "arch":     "March",
        "april":     "April",     "pril":     "April",
        "may":       "May",       "ay":       "May",
        "june":      "June",      "une":      "June",
        "july":      "July",      "uly":      "July",
        "august":    "August",    "ugust":    "August",
        "september": "September", "eptember": "September",
        "october":   "October",   "ctober":   "October",
        "november":  "November",  "ovember":  "November",
        "december":  "December",  "ecember":  "December",
    }

    DAY_ENTRY_RE = re.compile(r"^(\d{1,2})\s+(.+)$")
    SKIP_RE = re.compile(
        r"^(www\.|action report|metrosouth|networking at|effective ways|"
        r"note:|sixty school|via zoom|caalleennddaarr|ddaarr|ssppoonnssoorr|"
        r"ee aavv|space avail)",
        re.IGNORECASE,
    )

    chunks = []
    current_month = ""
    i = 0

    while i < len(cal_lines):
        line = cal_lines[i].strip()

        # Skip noise lines
        if not line or SKIP_RE.match(line):
            i += 1
            continue

        # Month heading (full or partial suffix match)
        low = line.lower()
        if low in SUFFIX_TO_MONTH:
            current_month = SUFFIX_TO_MONTH[low]
            i += 1
            continue

        # Entry line: "DD Event Name [TIME]"
        m = DAY_ENTRY_RE.match(line)
        if m:
            day  = m.group(1)
            rest = m.group(2).strip()

            # Extract time from rest if present
            t = TIME_RE.search(rest)
            if t:
                time_str   = t.group().strip()
                event_name = (rest[:t.start()] + rest[t.end():]).strip()
            else:
                time_str   = ""
                event_name = rest

            # Next line = location (may also carry the time if not found above)
            location = ""
            i += 1
            if i < len(cal_lines):
                loc_line = cal_lines[i].strip()
                # Skip if it's another entry or month
                next_is_entry = bool(DAY_ENTRY_RE.match(loc_line))
                next_is_month = loc_line.lower() in SUFFIX_TO_MONTH
                if not next_is_entry and not next_is_month and not SKIP_RE.match(loc_line):
                    # Location may have a trailing time
                    lt = TIME_RE.search(loc_line)
                    if lt and not time_str:
                        time_str = lt.group().strip()
                        location = (loc_line[:lt.start()] + loc_line[lt.end():]).strip()
                    else:
                        location = TIME_RE.sub("", loc_line).strip().rstrip(",").strip()
                    i += 1

            if event_name:
                date_str = f"{current_month} {day}" if current_month else day
                chunks.append(make_event_chunk(
                    event_name=event_name,
                    date=date_str,
                    time=time_str,
                    location=location,
                    source_label=source_label,
                    page_num=page_num,
                    filename=filename,
                ))
            continue

        i += 1

    return chunks


def is_calendar_page(lines: list[str]) -> bool:
    """Return True if this page appears to contain a structured calendar block."""
    full_text = " ".join(lines).lower()
    if CAL_TRIGGERS.search(full_text):
        return True
    # Also flag if page has multiple month headings + day numbers
    month_hits = sum(1 for l in lines if MONTH_HEADING_RE.match(l))
    day_hits   = sum(1 for l in lines if re.fullmatch(r"\d{1,2}", l.strip()))
    return month_hits >= 2 and day_hits >= 3


# ── Paragraph reconstruction + standard chunking ─────────────────────────────

def reconstruct_paragraphs(lines: list[str]) -> list[str]:
    paragraphs = []
    current = ""
    for line in lines:
        is_heading   = bool(HEADING_RE.match(line))
        is_bullet    = line.startswith(("-", "•", "*", "·"))
        starts_cap   = bool(line) and line[0].isupper()
        is_short_end = current and not current.rstrip().endswith((".", "!", "?", ":"))

        if not current:
            current = line
        elif is_heading or is_bullet or (starts_cap and not is_short_end):
            paragraphs.append(current)
            current = line
        else:
            current = current.rstrip() + " " + line

    if current:
        paragraphs.append(current)
    return paragraphs


def split_into_chunks(paragraphs: list[str], source_label: str,
                      page_num: int, filename: str) -> list[dict]:
    chunks = []
    current_text  = ""
    current_section = ""
    prev_tail     = ""

    def flush(text: str):
        text = text.strip()
        if len(text) < 60:
            return
        chunks.append({
            "text":      text,
            "searchable": text,
            "source":    source_label,
            "file":      filename,
            "page":      page_num,
            "section":   current_section,
            "is_event":  False,
            # Unified metadata so retriever can format prompt consistently
            "source_type":  "newsletter_pdf",
            "source_title": source_label,
        })

    for para in paragraphs:
        if HEADING_RE.match(para) and len(para) < 80:
            current_section = para.rstrip(":")

        if len(current_text) + len(para) > CHUNK_TARGET_CHARS and current_text:
            flush(prev_tail + current_text)
            prev_tail    = current_text[-CHUNK_OVERLAP_CHARS:].strip() + "\n" if current_text else ""
            current_text = para + "\n"
        else:
            current_text += para + "\n"

    if current_text:
        flush(prev_tail + current_text)

    return chunks


# ── Main ingest entry point ───────────────────────────────────────────────────

def ingest_all() -> list[dict]:
    all_chunks = []

    for root, _, files in os.walk(NEWSLETTER_ROOT):
        for fname in sorted(files):
            if not fname.lower().endswith(".pdf"):
                continue

            pdf_path     = os.path.join(root, fname)
            source_label = infer_month_label(fname)
            print(f"  [ingest] Reading: {fname}  →  '{source_label}'")

            pages = extract_pages(pdf_path)
            cal_event_count  = 0
            pipe_event_count = 0

            for page_info in pages:
                lines     = page_info["lines"]
                cal_lines = page_info["cal_lines"]
                page_num  = page_info["page_num"]

                # ── 1. Pipe-separated events (scan full page text) ──
                pipe_events = parse_pipe_events(lines, source_label, page_num, fname)
                all_chunks.extend(pipe_events)
                pipe_event_count += len(pipe_events)

                # ── 2. Structured calendar block (use cropped column if available) ──
                parse_target = cal_lines if cal_lines is not None else lines
                if cal_lines is not None:
                    cal_events = parse_structured_calendar(
                        parse_target, source_label, page_num, fname
                    )
                    all_chunks.extend(cal_events)
                    cal_event_count += len(cal_events)

                # ── 3. Regular paragraph chunks (all pages) ──
                paras  = reconstruct_paragraphs(lines)
                chunks = split_into_chunks(paras, source_label, page_num, fname)
                all_chunks.extend(chunks)

            print(f"           ↳ Calendar events: {cal_event_count} | "
                  f"Pipe events: {pipe_event_count}")

    print(f"  [ingest] Total chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = ingest_all()
    events = [c for c in chunks if c.get("is_event")]
    print(f"\nAll calendar events found ({len(events)}):\n")
    for e in events:
        print(f"  [{e['source']} p{e['page']}] "
              f"{e.get('date','')} {e.get('time','')} | "
              f"{e.get('event_name','')} | {e.get('location','')}")
