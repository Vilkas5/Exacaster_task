# GenAI Document Analysis — Feasibility POC

A Streamlit app that pulls a set of documents (from a public Google Drive
folder, or manual upload) and uses Claude to:

1. **Identify the primary topics discussed across the whole set.**
2. **Articulate each document's own perspective on those topics.**

Built as a feasibility POC for a market-research use case: automating what
would otherwise be a manual read-through of a large document set. Designed
to scale to hundreds of documents, not just the ~10-document sample set it
was validated against.

Live demo: https://exacaster-task.streamlit.app/

---

## How it works

Documents don't get pasted into one giant prompt — that approach breaks
down once you have more than a handful of documents, both on cost and on
context-window limits. Instead, `claude_client.analyze_documents` runs a
three-stage pipeline:

| Stage | What happens | Cost driver |
|---|---|---|
| **1. Map** | Claude reads each document *independently* (in parallel) and extracts its own local topics + stance | Scales linearly with document count, but each call is small and independent |
| **2. Cluster** | Local embeddings (no API call) group near-duplicate topic phrases across all documents — e.g. "AI Hallucinations" and "LLM Hallucinations" get merged | Free, near-instant, doesn't grow with corpus size |
| **3. Cleanup** | One small Claude call polishes final labels and catches any near-duplicates the embedding step missed | Sized by *topic count* (typically a few dozen), not document count |

A topic only gets promoted to the headline "primary topics" list if it's
covered by 2+ documents (`min_topic_coverage` in `analyze_documents`) —
single-document mentions are kept but shown separately as "minor topics,"
so the primary list reflects the whole set rather than one document's
tangent.

---

## Project structure

```
app.py                 Entry point — page setup, access gate, sidebar, renders the 3 tabs below
tab_demo.py             Tab 1 — Drive fetch / upload, "Analyze Documents", results

claude_client.py        The 3-stage pipeline described above (all Claude API calls live here)
topic_clustering.py     Stage 2 — embeds topic phrases and clusters them (scikit-learn, no API)
embeddings.py           Runs the local ONNX embedding model (used by topic_clustering.py)
model/                  The embedding model itself (~23MB), committed so no PyTorch is needed

drive_source.py         Google Drive API v3 — lists/downloads files from a public folder
document_utils.py       Extracts plain text from .txt / .pdf / .docx

api_key.py              Resolves ANTHROPIC_API_KEY / GOOGLE_API_KEY from .env or Streamlit secrets
auth.py                 ACCESS_PASSWORD gate — blocks the whole app until entered

tab_feasibility.py      Tab 2 — Feasibility Assessment, incl. validation findings from real testing
tab_solution.py         Tab 3 — Proposed Solution, incl. design alternatives considered and rejected

requirements.txt        Exact package list Streamlit Cloud installs
.env.example            Template of expected environment variables (no real values)
.gitignore              Keeps .env, venv/, and local junk out of git
```

---

## Local setup

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt      # Windows
# source venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

copy .env.example .env                            # Windows
# cp .env.example .env                             # macOS/Linux
```

Fill in `.env` with real values (see **Configuration** below), then:

```bash
venv\Scripts\python -m streamlit run app.py
```

Leave `ACCESS_PASSWORD` unset in your local `.env` — the gate no-ops when
it's not configured, so local dev is never blocked by it.

---

## Configuration

All secrets are environment-variable only — nothing is hardcoded, so
swapping to a different Anthropic account (e.g. a client's own key) means
changing a secret value, never the code.

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API calls (all 3 pipeline stages) |
| `GOOGLE_API_KEY` | Only for Drive fetch | Google Drive API v3 (read-only, works against publicly-shared folders — see `drive_source.py`) |
| `ACCESS_PASSWORD` | Recommended once deployed | Gates the whole app. Unset = no gate (fine for local dev, **not** fine for a public URL) |

**Getting an Anthropic key:** [console.anthropic.com](https://console.anthropic.com) → Settings → API Keys. Set a spend limit while you're there.

**Getting a Google API key:** [console.cloud.google.com](https://console.cloud.google.com) → enable the Drive API → Credentials → Create API key → restrict it to "Google Drive API" only.

---

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub (already done if you're reading this from the repo).
2. [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → New app → pick this repo, branch `main`, main file `app.py`.
3. App **Settings → Secrets**, paste (TOML format):
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   GOOGLE_API_KEY = "AIza..."
   ACCESS_PASSWORD = "choose-a-password"
   ```
4. Deploy. Share the resulting `*.streamlit.app` URL **and** the access password with whoever needs it.

**Rotating a key later** (e.g. it runs out of funds): edit it directly in
Streamlit Cloud's **Manage app → Secrets**. Note this requires **GitHub
write access to this repo** — Streamlit Cloud doesn't offer a lighter
"secrets only" role, so only grant that level of access to someone you'd
also trust with the code.

---

## Cost notes

- Model defaults to **Haiku 4.5** everywhere (cheapest tier) — configurable per-run from the sidebar, which controls stages 1 and 3 (stage 2 is free, local embeddings).
- Observed cost: roughly $0.01–0.02 per document for a full analysis run.
- The **access password isn't optional in practice** — an open URL with a working "Analyze Documents" button is a direct line to your API bill for anyone who finds it.
- If doing further development, consider testing via the Claude Code CLI (subscription-based, `claude_agent_sdk`) rather than burning the API key — see git history around the "local OAuth for testing" changes for how that was set up during this project.

---

## Known limitations (see `tab_feasibility.py` for the full write-up)

- Topic wording/granularity isn't perfectly deterministic between runs.
- Requires clean, extractable text — scanned/image-only PDFs need OCR first (not implemented).
- Model output isn't ground truth — worth a periodic human spot-check, especially in a new domain.
- Validated against a 10-document sample set; a larger, messier real corpus is the natural next test.
