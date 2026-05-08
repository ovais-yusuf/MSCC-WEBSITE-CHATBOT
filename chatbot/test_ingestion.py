"""
test_ingestion.py

Unit tests for the auto-update pipeline added in this change:

  - processed_sources: dedup tracker (URL hash + PDF hash)
  - web_crawler:       URL filtering, HTML extraction, PDF link discovery
  - web_ingest:        page → chunk conversion w/ unified metadata schema
  - pdf_downloader:    newsletter classification + filename inference
  - retriever:         format_for_prompt distinguishes webpage vs newsletter

These tests are FAST and OFFLINE. They never hit the real internet.

Run:
    python3 test_ingestion.py
    python3 -m unittest test_ingestion
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

# ── processed_sources ────────────────────────────────────────────────────────


class ProcessedSourcesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        self._tmp.close()
        os.unlink(self._tmp.name)   # we want a missing-file start

    def tearDown(self):
        if os.path.exists(self._tmp.name):
            os.unlink(self._tmp.name)

    def test_page_changed_detects_new_and_changed(self):
        from processed_sources import ProcessedSources, sha256_text

        ps = ProcessedSources(self._tmp.name)
        url = "https://example.com/foo"
        h1 = sha256_text("hello")
        h2 = sha256_text("hello world")

        self.assertTrue(ps.page_changed(url, h1))
        ps.record_page(url, h1, title="Foo")

        self.assertFalse(ps.page_changed(url, h1))
        self.assertTrue(ps.page_changed(url, h2))

    def test_save_and_reload_round_trip(self):
        from processed_sources import ProcessedSources

        ps = ProcessedSources(self._tmp.name)
        ps.record_page("https://example.com/a", "abc", "A")
        ps.record_pdf("https://example.com/x.pdf", "deadbeef",
                      filename="MAR-26-AR.pdf",
                      month_label="March 2026 Action Report",
                      page_count=8)
        ps.save()

        ps2 = ProcessedSources(self._tmp.name)
        self.assertFalse(ps2.page_changed("https://example.com/a", "abc"))
        self.assertTrue(ps2.pdf_seen("https://example.com/x.pdf"))
        self.assertEqual(ps2.stats(), {"pages_tracked": 1, "pdfs_tracked": 1})

    def test_pdf_changed_when_hash_differs(self):
        from processed_sources import ProcessedSources

        ps = ProcessedSources(self._tmp.name)
        ps.record_pdf("k", "h1", "f.pdf")
        self.assertFalse(ps.pdf_changed("k", "h1"))
        self.assertTrue(ps.pdf_changed("k", "h2"))


# ── web_crawler ──────────────────────────────────────────────────────────────


SAMPLE_HTML = """
<!doctype html>
<html><head>
  <title>About — Metro South Chamber</title>
  <meta property="article:modified_time" content="2026-04-30T12:00:00Z" />
</head><body>
  <header><nav><a href="/membership">Membership</a></nav></header>
  <main>
    <h1>About the Chamber</h1>
    <p>Welcome to the Metro South Chamber of Commerce.</p>
    <h2>Our Mission</h2>
    <p>We support local businesses across the Metro South region.</p>
    <p>Read our <a href="/newsletters/MAR-26-AR.pdf">March 2026 Action Report</a>.</p>
    <p>Also see <a href="https://other.example.com/x">an external link</a>.</p>
    <p>And <a href="/wp-admin/login">admin</a>.</p>
  </main>
  <footer>&copy; 2026 MSCC</footer>
</body></html>
"""


class WebCrawlerExtractionTests(unittest.TestCase):
    def test_extract_page_pulls_clean_text_and_pdf_links(self):
        from web_crawler import extract_page

        page = extract_page(SAMPLE_HTML, "https://metrosouthchamber.com/about")

        self.assertEqual(page["url"], "https://metrosouthchamber.com/about")
        self.assertIn("Metro South Chamber", page["title"])
        self.assertIn("Welcome to the Metro South Chamber", page["text"])
        self.assertNotIn("Membership", page["text"])  # menu was stripped
        self.assertNotIn("© 2026 MSCC", page["text"]) # footer was stripped
        self.assertIn("Our Mission", page["headings"])
        self.assertEqual(page["updated_at"], "2026-04-30T12:00:00Z")

        # Newsletter PDF link should be picked up as absolute URL
        self.assertIn(
            "https://metrosouthchamber.com/newsletters/MAR-26-AR.pdf",
            page["pdf_links"],
        )

    def test_url_normalization_and_relevance(self):
        from web_crawler import _is_relevant_path, _normalize_url, _should_skip

        self.assertTrue(_is_relevant_path("/"))
        self.assertTrue(_is_relevant_path("/membership"))
        self.assertTrue(_is_relevant_path("/events/upcoming"))
        self.assertFalse(_is_relevant_path("/cart"))
        self.assertFalse(_is_relevant_path("/random/path"))

        # Skips junk URLs
        self.assertTrue(_should_skip("https://x.com/a.jpg"))
        self.assertTrue(_should_skip("https://x.com/wp-admin/login"))
        self.assertFalse(_should_skip("https://x.com/about"))

        # Trailing slash + fragment normalization
        self.assertEqual(
            _normalize_url("https://x.com/foo/#bar"),
            "https://x.com/foo",
        )


# ── web_ingest ───────────────────────────────────────────────────────────────


class WebIngestTests(unittest.TestCase):
    def test_page_to_chunks_produces_unified_metadata(self):
        from web_ingest import page_to_chunks

        page = {
            "url": "https://metrosouthchamber.com/about",
            "title": "About — MSCC",
            "text": (
                "Welcome to the Chamber. "
                "We are an organization of more than 600 local businesses serving "
                "the Metro South region of Massachusetts including Brockton.\n\n"
                "Our Mission\n\n"
                "We support local businesses across the Metro South region. "
                "Our work covers advocacy, networking events, member resources, "
                "and economic development for the entire community."
            ),
            "headings": ["Our Mission"],
            "updated_at": "2026-04-30T12:00:00Z",
            "fetched_at": "2026-05-07T13:00:00Z",
        }

        chunks = page_to_chunks(page)
        self.assertGreaterEqual(len(chunks), 1)
        for c in chunks:
            self.assertEqual(c["source_type"], "webpage")
            self.assertEqual(c["source_url"], "https://metrosouthchamber.com/about")
            self.assertIn("MSCC Website", c["source"])
            self.assertEqual(c["is_event"], False)
            self.assertIn("text", c)
            self.assertIn("searchable", c)

    def test_short_pages_yield_no_microscopic_chunks(self):
        from web_ingest import page_to_chunks
        page = {"url": "https://x.com/y", "title": "Y", "text": "tiny",
                "headings": [], "updated_at": "", "fetched_at": ""}
        # Below the MIN_CHUNK_CHARS floor → should be filtered
        self.assertEqual(page_to_chunks(page), [])


# ── pdf_downloader ───────────────────────────────────────────────────────────


class PDFClassificationTests(unittest.TestCase):
    def test_looks_like_newsletter_classification(self):
        from pdf_downloader import looks_like_newsletter

        # URL/anchor strongly hints newsletter
        self.assertTrue(looks_like_newsletter(
            "https://metrosouthchamber.com/wp-content/uploads/2026/03/MAR-26-AR.pdf",
            anchor_text="Action Report",
        ))
        self.assertTrue(looks_like_newsletter(
            "https://metrosouthchamber.com/newsletters/feb_2026_action_report.pdf"
        ))

        # Generic PDF that has no month/year → not a newsletter
        self.assertFalse(looks_like_newsletter(
            "https://metrosouthchamber.com/forms/membership-application.pdf"
        ))

        # Not a PDF at all
        self.assertFalse(looks_like_newsletter("https://x.com/about"))

    def test_infer_filename_canonicalizes_to_existing_format(self):
        from pdf_downloader import infer_filename

        self.assertEqual(
            infer_filename(
                "https://metrosouthchamber.com/wp-content/uploads/2026/03/march-2026-action-report.pdf",
                anchor_text="March 2026 Action Report",
            ),
            "MAR-26-AR.pdf",
        )
        self.assertEqual(
            infer_filename(
                "https://example.com/jul_2025_newsletter.pdf",
                anchor_text="July 2025 Newsletter",
            ),
            "JUL-25.pdf",
        )

        # Unknown date → keep original basename
        self.assertEqual(
            infer_filename("https://example.com/something.pdf"),
            "something.pdf",
        )

    def test_real_mscc_filenames_are_handled_correctly(self):
        """Regression test: every URL pattern the live MSCC site uses must
        be detected, get a canonical filename that includes the -AR suffix
        for Action Reports, and produce a sensible month label."""
        from pdf_downloader import looks_like_newsletter, infer_filename
        from newsletter_ingest import infer_month_label

        cases = [
            # (url, expected_filename, expected_label)
            (
                "https://metrosouthchamber.com/wp-content/uploads/2026/05/MAY-26-AR.pdf",
                "MAY-26-AR.pdf",
                "May 2026 Action Report",
            ),
            (
                "https://metrosouthchamber.com/wp-content/uploads/2026/04/APR-26-AR.pdf",
                "APR-26-AR.pdf",
                "April 2026 Action Report",
            ),
            (
                "https://metrosouthchamber.com/wp-content/uploads/2026/04/JULYAUG-25-AR.pdf",
                "JULYAUG-25-AR.pdf",
                "July/August 2025 Action Report",
            ),
            (
                "https://metrosouthchamber.com/wp-content/uploads/2026/04/SEPT-25-AR.pdf",
                "SEP-25-AR.pdf",
                "September 2025 Action Report",
            ),
            (
                "http://www.metrosouthchamber.com/wp-content/uploads/2018/06/May-18-AR_.pdf",
                "MAY-18-AR.pdf",
                "May 2018 Action Report",
            ),
        ]
        for url, want_fname, want_label in cases:
            with self.subTest(url=url):
                self.assertTrue(looks_like_newsletter(url),
                                f"should detect {url}")
                got_fname = infer_filename(url)
                self.assertEqual(got_fname, want_fname)
                self.assertEqual(infer_month_label(got_fname), want_label)


class CrawlerSameDomainTests(unittest.TestCase):
    def test_www_and_apex_are_same_domain(self):
        from web_crawler import _is_same_domain
        self.assertTrue(_is_same_domain(
            "https://metrosouthchamber.com/about",
            "https://www.metrosouthchamber.com/wp-content/x.pdf",
        ))
        self.assertFalse(_is_same_domain(
            "https://metrosouthchamber.com/",
            "https://other.example.com/",
        ))

    def test_news_media_path_is_relevant(self):
        from web_crawler import _is_relevant_path
        self.assertTrue(_is_relevant_path("/news-media/action-report/"))
        self.assertTrue(_is_relevant_path("/visit-metro-south"))
        self.assertTrue(_is_relevant_path("/economic-development/leadership"))


class PDFDownloaderDedupTests(unittest.TestCase):
    """Verify download_new actually skips already-known PDFs."""

    def setUp(self):
        self._tmp_state = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        self._tmp_state.close()
        os.unlink(self._tmp_state.name)
        self._tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self._tmp_state.name):
            os.unlink(self._tmp_state.name)
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _fake_pdf_response(self):
        """A minimal valid PDF byte-stream the downloader will accept."""
        body = b"%PDF-1.4\n%MSCC test fixture\n%%EOF\n"

        def iter_chunks(chunk_size):
            yield body

        resp = mock.Mock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "application/pdf"}
        resp.iter_content = mock.Mock(side_effect=lambda chunk_size: iter_chunks(chunk_size))
        return resp, body

    def test_skips_already_processed_pdfs_by_hash(self):
        from processed_sources import ProcessedSources, sha256_bytes
        from pdf_downloader import PDFDownloader

        processed = ProcessedSources(self._tmp_state.name)
        # Downloader will lazy-call _safe_page_count which uses pdfplumber on
        # the saved file; for a fake PDF that returns 0, which is fine.
        dl = PDFDownloader(processed=processed, target_dir=self._tmp_dir)

        url = "https://metrosouthchamber.com/wp-content/uploads/2026/03/MAR-26-AR.pdf"

        resp, body = self._fake_pdf_response()
        with mock.patch.object(dl._session, "get", return_value=resp):
            first = dl.download_new([(url, "March 2026 Action Report")])
            self.assertEqual(len(first), 1)
            self.assertTrue(first[0]["is_new"])
            self.assertEqual(first[0]["filename"], "MAR-26-AR.pdf")

        # Pre-record the hash so the second call sees an unchanged PDF
        processed.record_pdf(url, sha256_bytes(body), "MAR-26-AR.pdf")

        resp2, _ = self._fake_pdf_response()
        with mock.patch.object(dl._session, "get", return_value=resp2):
            second = dl.download_new([(url, "March 2026 Action Report")])
            self.assertEqual(second, [])  # dedup hit

    def test_non_newsletter_pdf_is_ignored(self):
        from processed_sources import ProcessedSources
        from pdf_downloader import PDFDownloader

        processed = ProcessedSources(self._tmp_state.name)
        dl = PDFDownloader(processed=processed, target_dir=self._tmp_dir)

        url = "https://metrosouthchamber.com/forms/membership-application.pdf"
        with mock.patch.object(dl._session, "get") as mocked:
            out = dl.download_new([(url, "Membership Application")])
            self.assertEqual(out, [])
            mocked.assert_not_called()


# ── retriever prompt formatting ──────────────────────────────────────────────


class RetrieverFormatTests(unittest.TestCase):
    def test_format_for_prompt_distinguishes_webpage_and_newsletter(self):
        # We exercise the formatter directly without loading the index, so we
        # build a NewsletterRetriever instance and call format_for_prompt
        # with synthetic chunks.
        from newsletter_retriever import NewsletterRetriever

        retr = NewsletterRetriever()

        chunks = [
            {
                "text":         "EVENT: Good Morning Metro South\nMarch 5\n7:30AM",
                "source":       "March 2026 Action Report",
                "file":         "MAR-26-AR.pdf",
                "page":         3,
                "section":      "Calendar",
                "is_event":     True,
                "event_name":   "Good Morning Metro South",
                "date":         "March 5",
                "time":         "7:30AM",
                "location":     "Garner Hotel",
                "source_type":  "newsletter_pdf",
                "source_title": "March 2026 Action Report",
            },
            {
                "text":         "Membership at the Metro South Chamber unlocks…",
                "source":       "MSCC Website — Membership",
                "file":         "membership.html",
                "page":         1,
                "section":      "Why Join",
                "is_event":     False,
                "source_type":  "webpage",
                "source_url":   "https://metrosouthchamber.com/membership",
                "source_title": "Membership — MSCC",
                "updated_at":   "2026-04-30T12:00:00Z",
            },
        ]

        out = retr.format_for_prompt(chunks)
        self.assertIn("[Source: newsletter_pdf | March 2026 Action Report", out)
        self.assertIn("Calendar Event", out)
        self.assertIn("[Source: webpage | Membership — MSCC", out)
        self.assertIn("https://metrosouthchamber.com/membership", out)


# ── Smoke test for the full retrieval fallback path ──────────────────────────


class RetrievalFallbackTests(unittest.TestCase):
    """If the index is empty / missing, retrieve() must NOT crash and
    must return [] so the bot falls back to its 'I couldn't find that' line."""

    def test_retrieve_returns_empty_when_index_missing(self):
        from newsletter_retriever import NewsletterRetriever

        retr = NewsletterRetriever()
        # Force loaded=True with empty state so we don't actually read disk
        retr._loaded = True
        retr._chunks = []
        retr._embeddings = None
        retr._bm25 = None
        retr._model = None

        self.assertEqual(retr.retrieve("anything?", top_k=5), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
