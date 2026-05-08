# Auto-update pipeline (`ingest_website.py`)

The chatbot keeps itself up to date by crawling
[metrosouthchamber.com](https://metrosouthchamber.com/), downloading any new
newsletter / Action Report PDFs, and rebuilding the hybrid retrieval index in
one command.

```
ingest_website.py
   │
   ├─ web_crawler.py        BFS crawl, robots-aware, allowlisted paths
   ├─ pdf_downloader.py     find + dedupe newsletter PDFs by URL + sha256
   ├─ web_ingest.py         turn webpages into index chunks
   ├─ newsletter_ingest.py  turn PDFs into index chunks (existing)
   ├─ newsletter_index.py   rebuild embeddings + BM25 (existing, extended)
   └─ processed_sources.py  JSON-backed dedup tracker
```

A small JSON tracker at `content/processed_sources.json` records every URL we
crawl (with a content hash) and every PDF we download (with a file hash) so
re-runs only touch what actually changed.

## Run it manually

```bash
cd chatbot
python3 ingest_website.py
```

What you'll see:

1. `CRAWL` — pages fetched and PDFs discovered
2. `PAGE CHANGES` — list of new or updated URLs
3. `PDFs` — every newsletter that was downloaded or refreshed
4. `INDEX` — chunk counts and embedding rebuild progress
5. A run-log written to `content/ingest_logs/<UTC-timestamp>.json`

Useful flags:

```bash
python3 ingest_website.py --max-pages 100        # cap the crawl
python3 ingest_website.py --no-rebuild           # crawl + download only, skip reindex
python3 ingest_website.py --start-url https://metrosouthchamber.com/events/
python3 ingest_website.py --ignore-robots        # only if you're sure
```

Or set environment variables (`MSCC_START_URL`, `MSCC_MAX_PAGES`,
`MSCC_USER_AGENT`, `MSCC_RESPECT_ROBOTS`) for non-interactive runs.

## Verify the bot picked up new content

Two ways:

1. **Inspect the run log** in `content/ingest_logs/<timestamp>.json` —
   it lists every newly added PDF (`pdfs_added`) and every URL whose content
   hash changed (`pages_changed_urls`).
2. **Hot-reload the running server** without restarting:

   ```bash
   curl -X POST http://localhost:8000/admin/reload-index
   ```

   Then ask the bot a question whose answer is in the new content. The
   prompt now tags every retrieved passage like
   `[Source: newsletter_pdf | March 2026 Action Report | Page 3]`, and the
   model is instructed to cite newsletters naturally
   ("According to the March 2026 Action Report…") and prefer the most
   recent source when info conflicts.

## Schedule it on Railway

Railway supports two patterns:

### Option A — Railway scheduled job (recommended)

In the Railway dashboard for your service:

1. **Settings → Cron Schedule**
2. Add a new cron with command:
   ```
   python ingest_website.py
   ```
3. Pick a frequency, e.g. `0 7 * * *` for daily at 07:00 UTC,
   or `0 7 * * MON` for weekly Mondays.
4. Make sure the same env vars are set as the web service:
   - `OPENAI_API_KEY` (used by chat at runtime, not by ingest)
   - optional: `MSCC_MAX_PAGES`, `MSCC_USER_AGENT`

After each cron run, Railway logs will show the summary block printed at the
end of `ingest_website.py`.

### Option B — Run from a separate machine + reload remotely

If you'd rather run the heavy ingest job on your laptop or a personal VM:

```bash
python3 ingest_website.py
# upload newsletter_index.pkl + content/processed_sources.json to Railway
# (e.g. via `railway run` or an object store) then:
curl -X POST https://<your-app>.up.railway.app/admin/reload-index
```

The `/admin/reload-index` endpoint drops the in-memory index so the next chat
request picks up the new file from disk.

## Tests

The pipeline has fast offline tests that don't hit the network:

```bash
python3 -m unittest test_ingestion -v
```

Coverage includes:

- `processed_sources` round-trips, page-changed detection, PDF-changed
  detection
- `web_crawler.extract_page` clean-text extraction, menu/footer stripping,
  PDF link discovery, URL normalization, relevance filtering
- `pdf_downloader.looks_like_newsletter` classification +
  `infer_filename` canonicalization to the existing
  `MAR-26-AR.pdf` / `JUL-25.pdf` format
- `pdf_downloader.PDFDownloader.download_new` skipping already-processed
  PDFs by hash and ignoring non-newsletter PDFs entirely
- `newsletter_retriever.format_for_prompt` distinguishing
  `newsletter_pdf` vs `webpage` sources in the prompt
- `newsletter_retriever.retrieve` falling back gracefully when the index is
  empty or missing (so the chatbot's "I don't have that detail" path is
  reachable)
