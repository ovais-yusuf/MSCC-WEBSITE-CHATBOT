from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import os
from datetime import datetime
from pathlib import Path

load_dotenv()

# Newsletter retrieval — loads index lazily on first query
try:
    import newsletter_retriever as nr
    NEWSLETTER_RETRIEVAL = True
    print("[startup] Newsletter retriever loaded.")
except Exception as e:
    NEWSLETTER_RETRIEVAL = False
    print(f"[startup] Newsletter retriever unavailable: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_content():
    content_files = [
        ("Home", "content/home.txt"),
        ("Membership", "content/membership.txt"),
        ("Events", "content/events.txt"),
        ("News", "content/news.txt"),
        ("Resources", "content/resources.txt"),
        ("About", "content/about.txt"),
        ("Contact", "content/contact.txt"),
        ("Partner Programs", "content/programs.txt"),
        ("Metro South Region", "content/region.txt"),
        ("Leadership Metro South", "content/leadership.txt"),
        ("Greater Brockton Young Professionals", "content/gbyp.txt"),
        ("Visitors Guide", "content/visitors.txt"),
        ("Business Front Door", "content/business_front_door.txt"),
        ("Community One Stop for Growth", "content/one_stop.txt"),
    ]

    sections = []
    for section_name, file_path in content_files:
        try:
            with open(file_path, "r") as f:
                section_text = f.read().strip()
            if not section_text:
                print(f"[WARNING] {file_path} exists but is empty.")
        except FileNotFoundError:
            print(f"[WARNING] Content file not found: {file_path} — section will be empty.")
            section_text = ""
        sections.append(f"--- {section_name} ---\n{section_text}")

    return "\n\n".join(sections)


def log_chat(question, answer):
    timestamp = datetime.now().isoformat()
    line = f"[{timestamp}]\nQ: {question}\nA: {answer}\n\n"
    try:
        with open("logs.txt", "a") as log_file:
            log_file.write(line)
    except OSError:
        print(line, end="")


mscc_content = load_content()

_ROOT = Path(__file__).resolve().parent
_INDEX_HTML = _ROOT / "public" / "index.html"


class ChatRequest(BaseModel):
    message: str
    history: list = []


@app.get("/")
def home():
    # Local: serve HTML from disk. Vercel serverless often omits public/ from the
    # function bundle — CDN still hosts /index.html; redirect so the bubble UI loads.
    if _INDEX_HTML.is_file():
        return FileResponse(_INDEX_HTML, media_type="text/html")
    return RedirectResponse(url="/index.html", status_code=307)


@app.get("/health")
def health():
    return {"message": "MSCC Website Chatbot backend is running"}


@app.get("/debug-content")
def debug_content():
    sections = [line for line in mscc_content.split("\n") if line.startswith("---")]
    return {"loaded_sections": sections, "total_chars": len(mscc_content)}


@app.get("/debug-index")
def debug_index():
    """
    Show what's currently in the retrieval index so you can verify the
    auto-update pipeline is actually feeding the bot fresh content.
    """
    if not NEWSLETTER_RETRIEVAL:
        return {"ok": False, "error": "Retriever not loaded at startup."}
    try:
        nr._retriever._load()
        chunks = nr._retriever._chunks or []
    except Exception as e:
        return {"ok": False, "error": str(e)}

    by_type: dict = {}
    by_source: dict = {}
    sample_urls: list = []
    sample_pdfs: list = []
    for c in chunks:
        t = c.get("source_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
        s = c.get("source", "?")
        by_source[s] = by_source.get(s, 0) + 1
        if t == "webpage" and len(sample_urls) < 5:
            url = c.get("source_url")
            if url and url not in sample_urls:
                sample_urls.append(url)
        if t == "newsletter_pdf" and len(sample_pdfs) < 5:
            f = c.get("file")
            if f and f not in sample_pdfs:
                sample_pdfs.append(f)

    return {
        "ok": True,
        "total_chunks": len(chunks),
        "by_source_type": by_type,
        "by_source_top": dict(sorted(by_source.items(), key=lambda kv: -kv[1])[:10]),
        "sample_webpage_urls": sample_urls,
        "sample_newsletter_files": sample_pdfs,
    }


@app.get("/debug-retrieve")
def debug_retrieve(q: str, k: int = 5):
    """
    Run a query through the retriever and return the chunks it would feed
    to GPT, including which source each chunk came from. Use this to PROVE
    the bot's answer is grounded in the indexed content.

    Example:
        curl 'http://localhost:8000/debug-retrieve?q=membership%20benefits&k=5'
    """
    if not NEWSLETTER_RETRIEVAL:
        return {"ok": False, "error": "Retriever not loaded at startup."}
    try:
        chunks = nr.retrieve(q, top_k=k) or []
    except Exception as e:
        return {"ok": False, "error": str(e)}

    return {
        "ok":    True,
        "query": q,
        "count": len(chunks),
        "chunks": [
            {
                "score":         round(float(c.get("score", 0.0)), 4),
                "source_type":   c.get("source_type"),
                "source":        c.get("source"),
                "source_title":  c.get("source_title"),
                "source_url":    c.get("source_url"),
                "file":          c.get("file"),
                "page":          c.get("page"),
                "section":       c.get("section"),
                "is_event":      c.get("is_event", False),
                "preview":       (c.get("text") or "")[:240],
            }
            for c in chunks
        ],
    }


@app.post("/admin/reload-index")
def reload_index():
    """
    Force-reload the on-disk knowledge index so newly-ingested newsletters
    and webpages take effect without restarting the server.

    Protected by the optional ADMIN_TOKEN env var; if it's set, callers must
    send header `X-Admin-Token: <value>`. If unset, this endpoint is open
    (intended for local / single-tenant deployments).
    """
    if not NEWSLETTER_RETRIEVAL:
        return {"ok": False, "error": "Retriever not loaded at startup."}
    try:
        nr.reload()
        return {"ok": True, "message": "Retriever will reload index on next query."}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def build_newsletter_context(query: str) -> str:
    """
    Retrieve relevant newsletter + webpage chunks for the query.
    Returns a formatted string to inject into the system prompt,
    or an empty string if nothing relevant is found.

    Strategy:
      - Pull top-12 hybrid-scored chunks
      - Keep any with score > 0.05 (low threshold; retriever already ranked)
      - If nothing meets that, still keep the top 4 so the model has
        something to ground in instead of hallucinating "I don't know"
    """
    if not NEWSLETTER_RETRIEVAL:
        return ""
    try:
        chunks = nr.retrieve(query, top_k=12)
        if not chunks:
            return ""

        # Permissive threshold: with 16k+ chunks, even the right answer
        # often scores 0.10-0.20. The retriever's ranking is what matters.
        good = [c for c in chunks if c.get("score", 0.0) > 0.05]
        if not good:
            good = chunks[:4]   # fall back to top-N rather than nothing

        # Log what's about to be sent to GPT (helps debug grounding issues)
        types = [c.get("source_type", "?") for c in good]
        scores = [round(float(c.get("score", 0.0)), 3) for c in good]
        print(f"[retrieval] q={query!r}  →  {len(good)} chunks  "
              f"types={types}  scores={scores}")

        return nr.format_for_prompt(good)
    except Exception as e:
        print(f"[retrieval] Error: {e}")
        return ""


@app.post("/chat")
async def chat(request: ChatRequest):
    # Retrieve newsletter context before building the prompt
    newsletter_context = build_newsletter_context(request.message)

    system_prompt = (
        "You are the official Metro South Chamber of Commerce (MSCC) assistant. "
        "You are helpful, warm, and concise — like a knowledgeable staff member.\n\n"

        "═══ AVAILABLE CONTENT SECTIONS ═══\n"
        "The MSCC WEBSITE CONTENT block below contains ALL of these sections — search ALL of them before concluding something is absent:\n"
        "Home | Membership | Events | News | Resources | About | Contact | Partner Programs | "
        "Metro South Region | Leadership Metro South | Greater Brockton Young Professionals | Visitors Guide | Business Front Door | Community One Stop for Growth\n\n"

        "═══ STRICT GROUNDING — NO HALLUCINATIONS ═══\n"
        "- Answer ONLY from the MSCC content and newsletter excerpts provided below.\n"
        "- NEVER guess, assume, or fill in missing details.\n"
        "- NEVER use words like 'typically', 'usually', 'generally', or 'I believe'.\n"
        "- If a specific detail (time, speaker, location) is not in the provided content, "
        "say exactly: \"I don't have that detail in the available content.\"\n"
        "- Only say you cannot find something if you have checked ALL sections listed above and it is truly absent.\n"
        "- If the topic is entirely absent after checking all sections, say: \"I couldn't find that in the available newsletters or website content. "
        "You can reach the Chamber at (508) 586-0500 or [info@metrosouthchamber.com](mailto:info@metrosouthchamber.com)\"\n"
        "- Do NOT answer questions unrelated to MSCC, its members, events, or business resources.\n"
        "- Do NOT fabricate names, dates, prices, emails, or URLs.\n\n"

        "═══ TONE ═══\n"
        "- Sound natural and human — like a helpful colleague, not a search engine.\n"
        "- Use phrases like: 'Here's what I found:', 'This event is about...', 'You can reach them at...'\n"
        "- Avoid: 'Here are the following:', 'The organization provides:', 'According to the data:'\n"
        "- Keep responses concise by default. Go longer only when the user asks for full details.\n\n"

        "═══ MARKDOWN FORMATTING ═══\n"
        "- Use **bold** for: event names, people's names, section labels (Date, Location, Contact).\n"
        "- Example of correct bold use:\n"
        "    **Multicultural Business Forum**\n"
        "    - **Date:** March 5, 2026\n"
        "    - **Location:** Thorny Lea Golf Club, Brockton\n"
        "- Do NOT bold full sentences or paragraphs.\n"
        "- Use '- ' bullet points when listing 2+ items.\n"
        "- Use blank lines between sections so the response breathes.\n"
        "- Never merge sections into one run-on paragraph.\n\n"

        "═══ STAFF / PEOPLE QUESTIONS ═══\n"
        "Use this exact structure (with bold labels):\n"
        "  **[Name]** is the [title] at the Metro South Chamber of Commerce.\n\n"
        "  They handle:\n"
        "  - [responsibility 1]\n"
        "  - [responsibility 2]\n\n"
        "  **Contact:**\n"
        "  - **Phone:** (508) 586-0500 ext. [X]\n"
        "  - **Email:** [email](mailto:email)\n"
        "Keep role descriptions to one sentence. Do not copy long paragraphs.\n\n"

        "═══ EVENT QUESTIONS ═══\n"
        "When asked about events for a specific month:\n"
        "- List ALL events found in the retrieved content for that month — do not truncate.\n"
        "- Format each event as:\n"
        "    **[Event Name]**\n"
        "    - **Date:** [date]\n"
        "    - **Time:** [time] (only if in the content)\n"
        "    - **Location:** [location] (only if in the content)\n"
        "    - [one-line description if available]\n"
        "- If a detail is not in the retrieved content, omit that line entirely — do not guess.\n"
        "- After the list add: [View full event details and registration](https://metrosouthchamber.com/events/)\n"
        "When asked about a single specific event, give full details and end with the events link.\n"
        "When asked 'what events are coming up' (no month specified), show the next 2–3 and ask if they want more.\n\n"

        "═══ SOURCE GROUNDING ═══\n"
        "Each retrieved passage below is tagged with its origin:\n"
        "  [Source: newsletter_pdf | <month/year> | Page <n>]   ← from a Chamber newsletter\n"
        "  [Source: webpage | <title> | <url>]                  ← from metrosouthchamber.com\n"
        "Rules:\n"
        "- If the answer comes from a NEWSLETTER, naturally mention the issue once:\n"
        "  e.g. 'According to the March 2026 Action Report...' or 'The February newsletter notes...'\n"
        "- If the answer comes from the WEBSITE, you may cite the page subtly:\n"
        "  e.g. 'On the Membership page...' — but do not paste the URL inline unless the user asks.\n"
        "- If both website and newsletter content apply, combine them naturally and prefer the most\n"
        "  recent source for time-sensitive information (events, dates, prices).\n"
        "- NEVER fabricate newsletter, page, or URL content. If a detail (time, speaker, price)\n"
        "  is not in any retrieved passage, omit it or say you don't have that detail.\n\n"

        "═══ LINKS AND EMAILS ═══\n"
        "- Format ALL email addresses as markdown mailto links:\n"
        "  Correct: [events@metrosouthchamber.com](mailto:events@metrosouthchamber.com)\n"
        "  Wrong: events@metrosouthchamber.com\n"
        "- Format URLs as: [descriptive text](url)\n"
        "- Only use links and emails present in the provided content. Never fabricate them.\n"
        "- Include at most 2 links per response.\n\n"

        f"═══ MSCC WEBSITE CONTENT ═══\n{mscc_content}"
        + (f"\n\n═══ RETRIEVED NEWSLETTER CONTENT ═══\n{newsletter_context}" if newsletter_context else "")
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(request.history)
    messages.append({"role": "user", "content": request.message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=900
        )

        reply = (response.choices[0].message.content or "").strip()
        if not reply:
            reply = "I'm sorry, I couldn't generate a response right now. Please try again."

        log_chat(request.message, reply)
        return {"reply": reply}

    except Exception as e:
        fallback = "Sorry, I encountered an error and could not process your request. Please try again later."
        log_chat(request.message, fallback)
        return {"reply": fallback}
