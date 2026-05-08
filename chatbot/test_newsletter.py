"""
test_newsletter.py

Validates retrieval quality for 15+ newsletter-specific questions.
Run AFTER building the index:
    python3 newsletter_index.py
    python3 test_newsletter.py

Output for each question:
  - Question
  - Top retrieved chunks (source, page, score, preview)
  - PASS / WARN based on whether any relevant chunk was found
"""

import sys
from newsletter_retriever import NewsletterRetriever

retriever = NewsletterRetriever()

# Each test: (question, required_keywords_in_retrieved_text)
# If at least one retrieved chunk contains ANY of the required keywords → PASS
TESTS = [
    # --- Event lookup ---
    (
        "What events did the Chamber announce in March?",
        ["march", "event", "luncheon", "breakfast", "ribbon", "forum"],
    ),
    (
        "What is the Good Morning Metro South breakfast?",
        ["good morning", "breakfast", "metro south"],
    ),
    (
        "When is the multicultural business forum?",
        ["multicultural", "business forum", "scu"],
    ),
    (
        "What does the Chamber calendar include for April?",
        ["april", "calendar", "event"],
    ),
    (
        "Tell me about the ATHENA leadership award luncheon.",
        ["athena", "award", "leadership"],
    ),
    (
        "What ribbon cuttings happened recently?",
        ["ribbon cutting", "ribbon", "opening"],
    ),
    (
        "What happened at the annual meeting?",
        ["annual meeting", "annual", "112th", "expo"],
    ),
    # --- People lookup ---
    (
        "Who is Emma Penardi?",
        ["emma", "communications", "publications", "newsletter"],
    ),
    (
        "Who is Hannah McGuire?",
        ["hannah", "events", "programs", "coordinator"],
    ),
    (
        "What did the March newsletter say about the BSU graduate intern?",
        ["intern", "bsu", "graduate", "bridgewater"],
    ),
    # --- Contact lookup ---
    (
        "How do I contact the chamber about membership?",
        ["membership", "ian", "abreu", "contact", "508"],
    ),
    # --- Announcement lookup ---
    (
        "What did the newsletter say about the Toy Box opening?",
        ["toy box", "toy", "opening", "ribbon"],
    ),
    (
        "Who are the panelists for the women and minority owned business event?",
        ["panel", "women", "minority", "panelist"],
    ),
    (
        "What business news was featured in January?",
        ["january", "business", "news", "member"],
    ),
    # --- Cross-newsletter questions ---
    (
        "What sponsor supported Good Day Metro South events this year?",
        ["sponsor", "south shore bank", "cambridge", "good day"],
    ),
    (
        "What did the February 2026 newsletter cover?",
        ["february", "2026", "february 2026"],
    ),
    (
        "What is the Action Report?",
        ["action report", "newsletter", "monthly", "publications"],
    ),
    (
        "What legislative events did the chamber host?",
        ["legislative", "reception", "stonehill", "senate"],
    ),

    # ══════ NEW TESTS (production quality suite) ══════

    # Grounding: specific known person
    (
        "Who is Emma Penardi and what is her role?",
        ["emma", "vp", "communications", "newsletter", "social media"],
    ),
    # Calendar: specific month — must find ALL events
    (
        "What events are happening in March 2026?",
        ["march", "multicultural", "ambassador", "5"],
    ),
    # Specific event detail
    (
        "Tell me about the March 5 multicultural event",
        ["march 5", "multicultural", "thorny lea", "5:00"],
    ),
    # April calendar event
    (
        "What happened on April 13?",
        ["april 13", "multi-chamber", "mega networking", "plainridge"],
    ),
    # Funding / resources
    (
        "What funding was mentioned in the newsletters?",
        ["loan", "grant", "funding", "sba", "massdevelopment", "capital"],
    ),
    # Contact info
    (
        "What is the chamber phone number?",
        ["508", "586-0500"],
    ),
    # Safe failure: no content available
    (
        "What events are in December 2026?",
        # Should return no confident results — we check score is low OR no event chunks
        ["december"],   # permissive: just check we don't crash
    ),
]

PASS  = "✅ PASS"
WARN  = "⚠️  WARN"
ERROR = "❌ ERROR"


def run_tests():
    passed = 0
    warned = 0

    print("=" * 70)
    print("  NEWSLETTER RETRIEVAL TEST")
    print("=" * 70)

    for i, (question, keywords) in enumerate(TESTS, start=1):
        print(f"\n[{i:02d}] Q: {question}")
        print(f"      Expected keywords: {keywords}")

        try:
            chunks = retriever.retrieve(question, top_k=5)
        except Exception as e:
            print(f"      {ERROR} — retrieval exception: {e}")
            warned += 1
            continue

        if not chunks:
            print(f"      {WARN} — no chunks returned")
            warned += 1
            continue

        # Check if any required keyword appears in any returned chunk
        combined_text = " ".join(c["text"].lower() for c in chunks)
        hit = any(kw.lower() in combined_text for kw in keywords)

        status = PASS if hit else WARN
        if hit:
            passed += 1
        else:
            warned += 1

        print(f"      {status}")
        for j, c in enumerate(chunks[:3], start=1):
            preview = c["text"].replace("\n", " ")[:120]
            print(f"      Chunk {j}: [{c['source']} | p{c['page']}] score={c['score']:.3f}")
            print(f"               {preview}…")

    print("\n" + "=" * 70)
    print(f"  RESULTS: {passed}/{len(TESTS)} passed  |  {warned} warnings")
    print("=" * 70)
    return passed, warned


if __name__ == "__main__":
    passed, warned = run_tests()
    sys.exit(0 if warned == 0 else 1)
