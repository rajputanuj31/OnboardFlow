# OnboardFlow

**Understand any repo and its incoming changes, in one place.**

> **v3 is under active development.** It is being rebuilt step by step, with every change
> measured by an evaluation harness. The previous version (v2: LangGraph multi-agent Q&A and
> PR review over Qdrant) is preserved at tag [`v2.0.0`](../../tree/v2.0.0).

## What v3 is
An engineer joining a codebase, or reviewing a pull request, asks: *how does this work, where
is X, what does this change break?* OnboardFlow v3 answers with citations to exact lines, using:

- **Whole-repo indexing per commit** (shallow clone, content-addressed, incremental)
- **Agentic search**: ripgrep, a tree-sitter symbol graph, and hybrid semantic search, chosen per question
- **PR review** with a finder → verifier pipeline
- **Evals first**: retrieval, answer, localization, review and security suites, with a calibrated
  LLM judge; every quality claim in this README will be a measured number

## Status
| Area | State |
|---|---|
| Scaffold: API, typed config, Postgres + Qdrant, CI | done |
| Eval harness (5 suites' worth of data, judge, paired CIs) | done; v2 baseline run pending |
| Ingestion, agent, retrieval, PR review, web app | planned |

## Run it locally
Requirements: Docker, [uv](https://docs.astral.sh/uv/), make.

```bash
cp .env.example .env          # then set DATABASE__URL (see the comment in the file)
make install                  # backend dependencies (locked)
make up                       # Postgres + Qdrant, waits until healthy
make dev                      # API on http://localhost:8000
curl localhost:8000/health/ready
```

Checks: `make lint typecheck test`, `make test-integration` (needs `make up`), `make audit`,
`make secrets-scan`. Run `make help` for everything.

## Repository layout
```
apps/api/         FastAPI backend (Python 3.13, package `onboardflow`)
evals/            eval harness, datasets, results
infra/            docker compose for Postgres + Qdrant
```
