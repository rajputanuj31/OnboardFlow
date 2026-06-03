# OnboardFlow v2

An agentic workspace that ingests a GitHub repository, answers questions about it through a **multi-agent Q&A** flow, and reviews pull requests through a parallel **multi-agent PR review** flow. Built on LangGraph + MCP + Qdrant, with an online LLM-as-judge logging eval scores to LangSmith.

---

## What's new in v2

The original OnboardFlow was a single ReAct agent over a pure-Python SQLite vector store. v2 is a different architecture under the same UI:

- **Qdrant** replaces SQLite cosine-similarity. Real ANN index, session-scoped via payload filter, runs in Docker (or in-process when Docker isn't available).
- **MCP** (Model Context Protocol) — two stdio servers spawned at FastAPI startup:
  - A custom **Codebase MCP server** we author, exposing `read_codebase_file`, `semantic_search`, `get_repo_structure`, `get_file_summary`.
  - The official **github/github-mcp-server** for PR / issue data (optional — only needed for PR review).
- **Multi-agent Q&A**: a router picks 1–3 specialists (`code_reader`, `arch_explainer`, `dep_tracer`) from a single user question; specialists run in parallel via the LangGraph `Send` API; a synthesizer fuses their outputs and streams the final answer.
- **Multi-agent PR review**: `fetch_pr` (GitHub MCP) → `code_explainer` + `impact_analyst` + `risk_assessor` in parallel → synthesizer.
- **Online LLM-as-judge**: every Q&A and PR review is scored by a separate judge LLM (accuracy / completeness / groundedness, 1–5), written to LangSmith as run feedback. Fire-and-forget — never blocks the response.

---

## Architecture

```
Q&A flow:
  POST /ask
    router ─┬─> code_reader ──────┐
            ├─> arch_explainer ───┼─> synthesizer (streams) ─> update_history
            └─> dep_tracer ───────┘                              │
                                                                 └─> evaluator (async)

PR review flow:
  POST /review-pr
    fetch_pr ─┬─> code_explainer ─┐
              ├─> impact_analyst ─┼─> synthesizer (streams)
              └─> risk_assessor ──┘                          │
                                                             └─> evaluator (async)

  Both flows:
   - Agents call tools via two MCP servers (codebase + github)
   - RAG retrieval hits Qdrant `code_chunks` collection filtered by session_id
   - SSE stream: per-agent status pills + final synthesized tokens
   - Run IDs captured for LangSmith feedback
```

---

## Tech stack

**Backend**
- **API**: FastAPI (lifespan-based startup; SSE for streaming)
- **Agents**: LangGraph (multi-agent topology, parallel `Send` API, `create_react_agent` for specialists)
- **Vector store**: Qdrant (Docker server, or embedded mode via local path)
- **MCP**: `mcp` Python SDK (stdio transport), `langchain-mcp-adapters` for tool binding
- **Sessions**: SQLite (`backend/.cache/sessions.db`)
- **Observability**: LangSmith (free tier OK)

**Frontend** — Next.js 16 (React 19) + Tailwind CSS 4 + Mermaid.js. The chat workspace has a Q&A | PR Review mode toggle; per-agent status pills render while agents are running.

---

## Prerequisites

- **Python 3.11+**
- **Node 20+ / npm**
- **Docker Desktop** — required to run Qdrant. (Or use embedded mode — see below.)
- **GitHub PAT** with read access to the repos you want to ingest
- **OpenAI API key** (`text-embedding-3-small` + `gpt-4o-mini`)
- *Optional*: `github-mcp-server` binary — only needed for the PR review flow

---

## Setup

### 1. Clone + env

```bash
git clone <this-repo>
cd oss-agent
cp .env.example .env
# edit .env: GITHUB_TOKEN, OPENAI_API_KEY, optional LangSmith keys
```

### 2. Start Qdrant

```bash
docker compose up -d qdrant
curl -s http://localhost:6333/  # sanity: returns qdrant JSON
```

If you don't have Docker, set `QDRANT_URL=./.cache/qdrant_data` in `.env` — the backend will run Qdrant in embedded mode (in-process, no daemon needed). Same code path, slightly slower at large scale.

### 3. Backend

```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Expected boot log:
```
INFO:mcp_clients:Codebase MCP: loaded 4 tools
INFO:     Application startup complete.
```

Hit `/health` to confirm:
```bash
curl -s http://localhost:8000/health
# mcp.codebase.connected: true, tool_count: 4
```

### 4. Frontend

```bash
cd ../frontend
npm install
npm run dev
```

Open `http://localhost:3000`. Paste your OpenAI key in the sidebar (or rely on `.env`), then ingest a repo.

---

## PR review setup (optional)

The official GitHub MCP server is a separate binary. Install it via Go:

```bash
brew install go        # if you don't have Go
go install github.com/github/github-mcp-server@latest
which github-mcp-server  # confirm it landed in $PATH (typically ~/go/bin)
```

Or grab a prebuilt binary from <https://github.com/github/github-mcp-server/releases> and put it on `$PATH`. Restart the backend; `/health` should now show `mcp.github.available: true`.

Without it, `POST /review-pr` returns 503 with an install hint — the rest of the app works.

---

## Configuration

`.env` keys:

| key | required | default | description |
| --- | --- | --- | --- |
| `GITHUB_TOKEN` | yes | — | GitHub PAT |
| `OPENAI_API_KEY` | yes | — | for embeddings + chat |
| `QDRANT_URL` | no | `http://localhost:6333` | server URL, or a local path / `:memory:` for embedded mode |
| `EVAL_ENABLED` | no | `true` | set `false` to skip the online judge entirely |
| `LANGCHAIN_TRACING_V2` | no | unset | `true` enables LangSmith tracing |
| `LANGCHAIN_API_KEY` | no | — | LangSmith key (needed for feedback writes) |
| `LANGCHAIN_PROJECT` | no | `default` | LangSmith project name |
| `FRONTEND_URL` | no | — | comma-separated extra CORS origins |

---

## Running tests

```bash
cd backend
../venv/bin/python -m pytest tests/ -v
```

Covers: Qdrant round-trip (mocked embeddings), router fallback + fan-out, PR URL parsing, Codebase MCP server's tool list over stdio. 17 tests, all hermetic — no network, no real LLM calls.

---

## Migration note (v1 → v2)

The first `/ingest` call against a v1 backend will re-embed the repo into Qdrant. The v1 SQLite `code_chunks` table is no longer read; you can drop the file (`backend/.cache/sessions.db`) if you want a fully clean slate, but the sessions table is still used for chat history and other state.

---

## Future extensions (not yet built)

- **Offline golden-set eval harness**: a fixed set of ~20 PRs and questions with expected findings, run via `pytest` + LangSmith datasets, to catch regressions in agent behavior over time. The online judge gives a real-time signal but doesn't replace a regression suite.
- **More MCP servers**: e.g. JIRA / Linear for cross-system context on PRs.
- **Specialist memory**: persist per-agent observations across questions so `dep_tracer` doesn't re-search for the same symbol every turn.

---

## Project layout

```
oss-agent/
├── docker-compose.yml          # Qdrant service
├── .env / .env.example
├── backend/
│   ├── main.py                 # FastAPI: /ingest, /ask, /review-pr, /health
│   ├── config.py               # Pydantic settings (QDRANT_URL, EVAL_ENABLED, ...)
│   ├── models.py               # request/response schemas
│   ├── prompts.py              # system + specialist + judge prompts
│   ├── onboard_github.py       # GitHub API fetch, tree formatter, PR URL parser
│   ├── graph/
│   │   ├── graph.py            # ingest_graph, qa_graph, pr_review_graph
│   │   ├── state.py            # RepoState, QAState, PRReviewState (with reducers)
│   │   ├── ingest_nodes.py     # fetch + summarize + architecture (unchanged from v1)
│   │   ├── qa_agent_nodes.py   # router + 3 specialists + synthesizer + history
│   │   ├── pr_review_nodes.py  # fetch_pr + 3 specialists + synthesizer
│   │   ├── evaluator.py        # online LLM-as-judge + LangSmith feedback
│   │   └── helpers.py
│   ├── mcp_clients/            # MCPRegistry: spawns + manages MCP subprocesses
│   ├── mcp_servers/
│   │   └── codebase_server.py  # custom MCP server (4 tools over stdio)
│   ├── utils/
│   │   ├── vector_db.py        # Qdrant client (same public API as v1 SQLite)
│   │   ├── session_db.py       # SQLite sessions table
│   │   └── http_cache.py       # SQLite cache for GitHub API
│   └── tests/                  # pytest smoke tests
├── frontend/
│   └── src/components/ChatWorkspace.tsx  # mode toggle + agent status pills
└── requirements.txt
```
