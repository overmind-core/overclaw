.PHONY: test test-parallel test-verbose test-cov test-fast test-serial lint-check lint-format help

# Python interpreter — prefers project venv, otherwise system python3
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

# When `uv` is available, run pytest/ruff with the matching dependency groups so
# `make test` works without a pre-synced venv (use: uv sync --group test --group dev).
UV := $(shell command -v uv 2>/dev/null)
ifneq ($(UV),)
PYTEST = $(UV) run --group test python -m pytest
RUFF = $(UV) run --group dev python -m ruff
else
PYTEST = $(PYTHON) -m pytest
RUFF = $(PYTHON) -m ruff
endif

# Number of parallel workers (auto = number of CPUs)
WORKERS ?= auto

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

test: ## Run all tests in parallel (default)
	$(PYTEST) tests/ -x -n $(WORKERS) --ignore=tests/test_spans.py --dist worksteal -q $(test_args)

test-cov: ## Run all tests with coverage report (sorted by coverage %)
	$(PYTEST) tests/ -n $(WORKERS) --ignore=tests/test_spans.py --dist worksteal --cov=overmind --cov-report=term-missing -q $(test_args)

test-serial: ## Run all tests serially (for debugging)
	$(PYTEST) tests/ -x --ignore=tests/test_spans.py -v $(test_args)

lint-check: ## Check lint + format (CI — no changes)
	$(RUFF) check
	$(RUFF) format --check

lint-format: ## Run ruff linter + formatter (auto-fix)
	$(RUFF) check --fix
	$(RUFF) format


.PHONY: generate_api_client generate_python_client schema backend dev

backend:
	cd bae && python manage.py runserver 8000

dev:
	cd frontend && bun run dev

schema:
	cd bae && python manage.py spectacular --file ../openapi.yaml --validate

generate_api_client: schema
	docker run --rm -v $(PWD):/workspace openapitools/openapi-generator-cli:v7.19.0 generate \
		-i /workspace/openapi.yaml \
		-g typescript-fetch \
		-o /workspace/frontend/src/api/generated \
		--additional-properties=typescriptThreePlus=true,supportsES6=true,enumPropertyNaming=original

