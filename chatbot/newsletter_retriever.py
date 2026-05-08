"""
newsletter_retriever.py

Loads the pre-built index and provides hybrid retrieval:
  1. Semantic search  (sentence-transformer cosine similarity)
  2. Keyword search   (BM25 Okapi)
  3. Combined score   (weighted sum → top-k chunks)

Usage:
    from newsletter_retriever import NewsletterRetriever
    retriever = NewsletterRetriever()
    chunks = retriever.retrieve("What events were announced in March?", top_k=5)
"""

import os
import re
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = os.path.join(os.path.dirname(__file__), "newsletter_index.pkl")

# Weight given to semantic vs BM25 score (must sum to 1.0)
SEMANTIC_WEIGHT = 0.65
BM25_WEIGHT     = 0.35

# Keywords that strongly signal a newsletter question
NEWSLETTER_KEYWORDS = {
    "newsletter", "action report", "january", "february", "march",
    "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "issue", "calendar", "breakfast",
    "luncheon", "after hours", "business after hours", "good morning",
    "good day", "ribbon cutting", "multicultural", "athena", "award",
    "nominations", "panelist", "speaker", "sponsor", "member news",
    "welcome new", "intern", "bsu", "toy box", "graduate",
    "legislative", "workforce", "chamber update", "featured",
    "announcement",
}


MONTHS_LOWER = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

CALENDAR_QUERY_RE = re.compile(
    r"\b(calendar|event[s]?|happening|scheduled?|upcoming|when is|what'?s? on"
    r"|" + "|".join(MONTHS_LOWER) + r")\b",
    re.IGNORECASE,
)

def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class NewsletterRetriever:
    def __init__(self):
        self._loaded = False
        self._chunks = []
        self._embeddings = None
        self._bm25 = None
        self._model = None

    def _load(self):
        if self._loaded:
            return
        if not os.path.exists(INDEX_PATH):
            print("[retriever] Index not found. Run newsletter_index.py first.")
            self._loaded = True
            return
        with open(INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self._chunks     = data["chunks"]
        self._embeddings = data["embeddings"]   # already L2-normalised
        self._bm25       = data["bm25"]
        self._model      = SentenceTransformer(data.get("model_name", "all-MiniLM-L6-v2"))
        self._loaded = True
        n_news = sum(1 for c in self._chunks if c.get("source_type") == "newsletter_pdf")
        n_web  = sum(1 for c in self._chunks if c.get("source_type") == "webpage")
        print(f"[retriever] Index loaded — {len(self._chunks)} chunks "
              f"({n_news} newsletter, {n_web} webpage).")

    def reload(self) -> None:
        """Force the next retrieve() call to re-read the on-disk index."""
        self._loaded = False
        self._chunks = []
        self._embeddings = None
        self._bm25 = None
        self._model = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def is_newsletter_query(self, query: str) -> bool:
        """Heuristic: does this question likely relate to newsletter content?"""
        q = query.lower()
        return any(kw in q for kw in NEWSLETTER_KEYWORDS)

    def _is_calendar_query(self, query: str) -> bool:
        """True if the query is asking about events/calendar/a specific month."""
        return bool(CALENDAR_QUERY_RE.search(query))

    def _mentioned_months(self, query: str) -> list[str]:
        """Return any month names explicitly mentioned in the query."""
        q = query.lower()
        return [m for m in MONTHS_LOWER if m in q]

    def retrieve(self, query: str, top_k: int = 6) -> list[dict]:
        """
        Return the top_k most relevant chunks for the query.
        Calendar/event queries get boosted calendar chunks added to the top results.
        """
        self._load()
        if not self._chunks:
            return []

        # 1. Semantic scores (cosine similarity).
        #    Stick with float32 end-to-end — sentence-transformers returns
        #    float32 and the stored embeddings are float32. Casting to
        #    float64 before matmul caused transient overflow in BLAS on
        #    large indexes (16k+ chunks). All embeddings are L2-normalized
        #    so the dot product IS the cosine similarity.
        q_emb = self._model.encode([query]).astype(np.float32)
        q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_emb = q_emb / np.where(q_norm == 0, 1, q_norm)
        emb = np.asarray(self._embeddings, dtype=np.float32)
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            sem_scores = (emb @ q_emb.T).flatten()
        sem_scores = np.nan_to_num(sem_scores, nan=0.0, posinf=0.0, neginf=0.0)
        # Cosine similarity is in [-1, 1]; clip to a sane range.
        sem_scores = np.clip(sem_scores, -1.0, 1.0)

        # 2. BM25 scores (normalised to [0, 1])
        bm25_raw = np.array(self._bm25.get_scores(_tokenize(query)), dtype=float)
        bm25_max = bm25_raw.max()
        bm25_scores = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw

        # 3. Calendar boost: if query is about events/months, bump calendar chunks
        cal_boost = np.zeros(len(self._chunks))
        if self._is_calendar_query(query):
            mentioned = self._mentioned_months(query)
            for i, chunk in enumerate(self._chunks):
                if chunk.get("is_event"):
                    # Base boost for any calendar event
                    cal_boost[i] += 0.15
                    # Extra boost if the event's date matches a mentioned month
                    if mentioned:
                        chunk_date = (chunk.get("date", "") + " " +
                                      chunk.get("event_name", "")).lower()
                        if any(m in chunk_date for m in mentioned):
                            cal_boost[i] += 0.20

        # 4. Hybrid score
        hybrid = SEMANTIC_WEIGHT * sem_scores + BM25_WEIGHT * bm25_scores + cal_boost

        # 5. Top-k — always include at least 2 calendar events for event queries
        top_indices = np.argsort(hybrid)[::-1]

        results = []
        seen_ids = set()
        cal_count = 0

        for idx in top_indices:
            if len(results) >= top_k:
                break
            chunk = dict(self._chunks[idx])
            chunk["score"] = float(hybrid[idx])
            results.append(chunk)
            seen_ids.add(idx)
            if chunk.get("is_event"):
                cal_count += 1

        # If this is a calendar query and we got fewer than 2 calendar chunks, top them up
        if self._is_calendar_query(query) and cal_count < 2:
            for idx in top_indices:
                if idx in seen_ids:
                    continue
                chunk = self._chunks[idx]
                if chunk.get("is_event"):
                    c = dict(chunk)
                    c["score"] = float(hybrid[idx])
                    results.append(c)
                    cal_count += 1
                    if cal_count >= 3:
                        break

        return results

    def format_for_prompt(self, chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a block suitable for injection
        into the GPT system prompt.

        Each chunk is tagged with its origin so the model can ground citations:
            [Source: newsletter_pdf | March 2026 Action Report | Page 3]
            [Source: webpage | About — MSCC | https://metrosouthchamber.com/about]
        """
        if not chunks:
            return ""
        lines = ["RETRIEVED CONTENT (use ONLY this to answer):"]
        for c in chunks:
            stype = c.get("source_type", "newsletter_pdf")
            if c.get("is_event"):
                header = (
                    f"[Source: {stype} | {c.get('source','')} "
                    f"| Calendar Event | Page {c.get('page','?')}]"
                )
                lines.append(header)
                if c.get("event_name"):
                    lines.append(f"Event: {c['event_name']}")
                if c.get("date"):
                    lines.append(f"Date: {c['date']}")
                if c.get("time"):
                    lines.append(f"Time: {c['time']}")
                if c.get("location"):
                    lines.append(f"Location: {c['location']}")
            elif stype == "webpage":
                title = c.get("source_title") or c.get("source") or "MSCC Website"
                url   = c.get("source_url", "")
                header = f"[Source: webpage | {title}"
                if c.get("section"):
                    header += f" | Section: {c['section']}"
                if url:
                    header += f" | {url}"
                header += "]"
                lines.append(header)
                lines.append(c["text"].strip())
            else:
                header = f"[Source: {stype} | {c.get('source','')}"
                if c.get("section"):
                    header += f" | Section: {c['section']}"
                header += f" | Page {c.get('page','?')}]"
                lines.append(header)
                lines.append(c["text"].strip())
            lines.append("")
        return "\n".join(lines)


# Singleton used by main.py
_retriever = NewsletterRetriever()


def retrieve(query: str, top_k: int = 6) -> list[dict]:
    return _retriever.retrieve(query, top_k=top_k)


def is_newsletter_query(query: str) -> bool:
    return _retriever.is_newsletter_query(query)


def format_for_prompt(chunks: list[dict]) -> str:
    return _retriever.format_for_prompt(chunks)


def reload() -> None:
    """Drop the in-memory index so the next retrieve() reloads from disk."""
    _retriever.reload()
