# Metro South Chamber website chatbot

FastAPI backend plus a small web UI that answers visitor questions using Chamber website text, optional newsletter retrieval, and the OpenAI API.

## Repository layout

| Path | Purpose |
|------|---------|
| `chatbot/` | Application code, `content/`, `public/`, ingestion scripts |
| `*.docx` | Planning / scope notes (optional reference) |

## Quick start (local)

1. **Python** — Use the version in `chatbot/.python-version` if you use `pyenv`.

2. **Environment**

   ```bash
   cd chatbot
   cp env.sample .env
   ```

   Set `OPENAI_API_KEY` in `.env` (never commit `.env`; it is gitignored).

3. **Dependencies**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run the API**

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   - UI: [http://localhost:8000/](http://localhost:8000/)
   - Health: [http://localhost:8000/health](http://localhost:8000/health)

## Updating content

Static sections live under `chatbot/content/*.txt`. To crawl the live site and refresh newsletters / embeddings, see **`chatbot/INGESTION.md`**.

## Testing without cloning to your own computer

This project is a web app plus Python backend; **there is no official “try it in the browser” button in the repo alone.** Easiest paths for reviewers:

1. **Shared hosted URL (recommended)**  
   Someone with deploy access publishes the app (e.g. Vercel, Railway, Render, Fly.io, or a small VPS). Testers only open that URL in a browser. The maintainer sets `OPENAI_API_KEY` (and any other secrets) in the host’s dashboard—testers never need the repo or an API key.

2. **GitHub Codespaces**  
   On GitHub: **Code → Codespaces → Create codespace on main**. The repo lives in a cloud VM, not on the reviewer’s laptop. They still run `cd chatbot`, install deps, add `.env`, and start `uvicorn` in the Codespace terminal (or use port forwarding to open the UI in the browser). Heavier dependencies (e.g. sentence-transformers) make the first setup slower than a static site.

3. **Minimal local clone (optional)**  
   If they must run locally but want a smaller download: `git clone --depth 1 <repo-url>` gets history-truncated copy only; the repo is still large because of PDFs under `chatbot/content/Newsletters/`.

## Deploy notes

The app is written to work locally (`FileResponse` for `public/index.html`) and on serverless hosts that expose static files separately (see comments on `/` in `main.py`). Configure secrets (`OPENAI_API_KEY`) in your host’s environment, not in the repo.
