# OnboardFlow v3 developer commands. Run `make help` for the list.

API      := apps/api
UV       := uv --directory $(API)
UVE      := uv --directory evals
EVAL     := $(UVE) run of-eval
RUN      ?= baseline
COMPOSE  := docker compose -f infra/docker-compose.yml

.DEFAULT_GOAL := help
.PHONY: help install up up-baseline down logs dev lint format typecheck test test-integration \
        check env-example audit secrets-scan eval-prepare eval-estimate eval-run eval-judge \
        eval-report eval-baseline eval-quick reproduce

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Install backend and eval-harness dependencies (locked)
	$(UV) sync --locked
	$(UVE) sync --locked

up: ## Start Postgres and Qdrant and wait until healthy
	$(COMPOSE) up -d --wait

up-baseline: ## Start the separate Qdrant used by the v2 baseline
	$(COMPOSE) --profile baseline up -d --wait qdrant-v2

down: ## Stop services (data volumes are kept)
	$(COMPOSE) --profile baseline down

logs: ## Tail service logs
	$(COMPOSE) logs -f

dev: ## Run the API with auto-reload on :8000
	$(UV) run uvicorn onboardflow.main:create_app --factory --reload --port 8000

lint: ## Ruff lint + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UVE) run ruff check .
	$(UVE) run ruff format --check .

format: ## Auto-format and fix lint where safe
	$(UV) run ruff format .
	$(UV) run ruff check --fix .
	$(UVE) run ruff format .
	$(UVE) run ruff check --fix .

typecheck: ## mypy (strict)
	$(UV) run mypy
	$(UVE) run mypy

test: ## Unit tests (no services, no network, no model calls)
	$(UV) run pytest
	$(UVE) run pytest

test-integration: ## Integration tests (needs `make up`)
	$(UV) run pytest -m integration

check: lint typecheck test ## Everything CI runs on the backend, except integration and security

env-example: ## Regenerate .env.example from the settings schema
	$(UV) run python -m onboardflow.settings > .env.example

audit: ## Scan locked dependencies for known vulnerabilities
	$(UV) export --frozen --no-dev --no-emit-project --format requirements-txt -o /tmp/onboardflow-requirements.txt
	uvx pip-audit --strict --disable-pip --require-hashes -r /tmp/onboardflow-requirements.txt
	$(UVE) export --frozen --no-dev --no-emit-project --format requirements-txt -o /tmp/onboardflow-evals-requirements.txt
	uvx pip-audit --strict --disable-pip --require-hashes -r /tmp/onboardflow-evals-requirements.txt

secrets-scan: ## Scan the git history for committed secrets (gitleaks via Docker)
	docker run --rm -v "$(CURDIR):/repo" zricethezav/gitleaks:v8.30.1 git /repo --redact -v

eval-prepare: ## Rebuild public eval datasets (SWE-QA, SWE-bench) from pinned sources
	$(EVAL) prepare

eval-estimate: ## Print the cost estimate for run RUN (spends nothing)
	$(EVAL) estimate --run $(RUN)

eval-run: ## Run SYSTEM (no-context|v2) over all suites in run RUN
	$(EVAL) run --run $(RUN) --system $(SYSTEM) $(ARGS)

eval-judge: ## Judge every system's outputs in run RUN
	$(EVAL) judge --run $(RUN) $(ARGS)

eval-report: ## Write evals/reports/RUN.md
	$(EVAL) report --run $(RUN)

eval-baseline: up-baseline ## Step 2 baseline: no-context + v2, judged, reported
	$(EVAL) baseline --run $(RUN) $(ARGS)

eval-quick: ## Small cached subset (20 answer items) used as the PR gate
	$(EVAL) quick $(ARGS)

reproduce: ## Regenerate every README results table from cached runs (step 12)
	@echo "reproduce: not implemented yet (build plan step 12)"; exit 1
