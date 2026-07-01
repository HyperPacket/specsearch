.PHONY: sanity up index query test lint eval ui mcp

sanity: ## Scaffold checks (no services needed)
	@git init -q 2>/dev/null || true
	@git add -An | grep -Eq "personal/|data/raw" && { echo "LEAK: personal/ or data/raw not gitignored"; exit 1; } || echo "ok: gitignore protects personal/ and data/raw"
	@uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" && echo "ok: torch CUDA" || echo "WARN: torch CUDA unavailable — check pyproject CUDA index"
	@uv run pytest -q && echo "ok: tests pass"
	@uv run ruff check . && echo "ok: ruff clean"
	@for f in CLAUDE.md docs/DESIGN.md personal/interview-questions.md; do test -s "$$f" || { echo "MISSING: $$f"; exit 1; }; done && echo "ok: authored docs intact"

up: ## Start Elasticsearch
	docker compose up -d elasticsearch

index: ## Build ES + Chroma indexes
	uv run specsearch index data/raw

query: ## Sample query with explain
	uv run specsearch query "What is the required concrete compressive strength at 28 days?" --explain

test: ## Run tests
	uv run pytest

lint: ## Ruff linter
	uv run ruff check .

eval: ## Evaluation harness
	uv run specsearch eval

ui: ## Streamlit UI
	uv run streamlit run src/specsearch/app.py

mcp: ## MCP server
	uv run python -m specsearch.mcp_server
