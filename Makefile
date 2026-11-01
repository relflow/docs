.DEFAULT_GOAL := help

QUARTO ?= uvx --from quarto-cli==1.10.18 quarto
HOST ?= 127.0.0.1
PORT ?= 4200

.PHONY: help prepare dev preview render check check-docs clean

help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; print "Targets:"} /^[a-zA-Z_-]+:.*##/ {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

prepare: ## Generate static evidence pages from local source and recorded results.
	uv run python -m evidence.render

dev: ## Preview the static site with live reload.
	$(QUARTO) preview docs --no-browser --host $(HOST) --port $(PORT)

preview: dev ## Alias for dev.

render: ## Render static HTML and Typst diagrams into docs/site/.
	$(QUARTO) render docs --no-execute

check: prepare ## Check static authoring, API references, and evidence reporting.
	uv run ruff format --check tests evidence
	uv run ruff check tests evidence
	uv run pytest
	uv run python -m tests.check_site

check-docs: render ## Validate the complete rendered site.
	uv run python -m tests.check_site --rendered

clean: ## Remove generated pages, site output, diagrams, and tool caches.
	uv run python -m evidence.render --clean
	rm -rf docs/site docs/.quarto docs/assets/diagrams .pytest_cache .ruff_cache
	rm -f docs/.gitignore
	rm -rf tests/__pycache__ evidence/__pycache__
	rm -rf evidence/tests/__pycache__ evidence/tests/runtime/__pycache__
	rm -rf evidence/experiments/__pycache__ evidence/experiments/*/__pycache__
