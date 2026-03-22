.PHONY: test test-parallel test-verbose test-cov test-fast test-serial lint-check lint-format help

# Python interpreter — uses the project venv if it exists, otherwise system python3
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

# Number of parallel workers (auto = number of CPUs)
WORKERS ?= auto

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

test: ## Run all tests in parallel (default)
	$(PYTHON) -m pytest tests/ -x -n $(WORKERS) --dist worksteal -q $(test_args)

test-cov: ## Run all tests with coverage report (sorted by coverage %)
	$(PYTHON) -m pytest tests/ -n $(WORKERS) --dist worksteal --cov=overclaw --cov-report=term-missing -q $(test_args)

test-serial: ## Run all tests serially (for debugging)
	$(PYTHON) -m pytest tests/ -x -v $(test_args)

lint-check: ## Check lint + format (CI — no changes)
	$(PYTHON) -m ruff check
	$(PYTHON) -m ruff format --check

lint-format: ## Run ruff linter + formatter (auto-fix)
	$(PYTHON) -m ruff check --fix
	$(PYTHON) -m ruff format


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

# Generate the Python client and embed it at overclaw/openapi_client/.
#
# The generator creates an `openapi_client` package with bare `from openapi_client.*`
# imports.  After copying the package into the overclaw namespace we rewrite every
# import so they resolve as `from overclaw.openapi_client.*` — consistent with
# how the rest of the codebase (client.py, etc.) references them.
generate_python_client: schema
	@echo "==> Generating Python API client (generator output → .tmp_python_client)"
	docker run --rm -v $(PWD):/workspace openapitools/openapi-generator-cli:v7.19.0 generate \
		-i /workspace/openapi.yaml \
		-g python \
		-o /workspace/.tmp_python_client \
		--additional-properties=packageName=overclaw.openapi_client,library=httpx,generateSourceCodeOnly=true
	@echo "==> Installing into overclaw/openapi_client/"
	rm -rf overclaw/openapi_client
	mv .tmp_python_client/overclaw/openapi_client overclaw/openapi_client
	rm -rf .tmp_python_client
	rm -rf overclaw/openapi_client/test
	rm -rf overclaw/openapi_client/docs
	@echo "==> Done."
