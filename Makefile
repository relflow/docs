.DEFAULT_GOAL := help

QUARTO ?= uvx --from quarto-cli==1.10.18 quarto
HOST ?= 127.0.0.1
PORT ?= 4200

.PHONY: help prepare dev preview render check check-docs clean

help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; print "Targets:"} /^[a-zA-Z_-]+:.*##/ {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

prepare: ## Generate static proof pages from local source and recorded results.
	uv run python proofs/render.py

dev: ## Preview the static site with live reload.
	$(QUARTO) preview docs --no-browser --host $(HOST) --port $(PORT)

preview: dev ## Alias for dev.

render: ## Render static HTML and Typst diagrams into docs/site/.
	$(QUARTO) render docs --no-execute

check: prepare ## Check static authoring, API references, and proof reporting.
	uv run ruff format --check scripts tests proofs/render.py proofs/source.py
	uv run ruff check scripts tests proofs/render.py proofs/source.py
	uv run pytest
	uv run python -m scripts.check

check-docs: render ## Validate the complete rendered site.
	uv run python -m scripts.check --rendered

clean: ## Remove generated proof pages and site output.
	uv run python proofs/render.py --clean
	rm -rf docs/site docs/.quarto docs/assets/diagrams
