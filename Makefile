.PHONY: install qdrant api inngest ui test lint format

install:
	uv sync --all-groups

qdrant:
	docker compose up -d qdrant

api:
	uv run uvicorn science_rag.app:app --reload

inngest:
	npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery

ui:
	uv run streamlit run ui/streamlit_app.py

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .
