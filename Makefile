.PHONY: data features train tune evaluate pipeline app test lint install

install:
	uv sync --all-extras

data:
	uv run python -m src.ingestion.download

features:
	uv run python -m src.features.build_features
	uv run python -m src.features.macro_features

train:
	uv run python -m src.models.pd_model
	uv run python -m src.models.lgd_model

evaluate:
	uv run python -m src.evaluate.metrics

pipeline: data features train evaluate

tune:
	uv run python -m src.models.tune_pd

app:
	uv run streamlit run app/Home.py

test:
	uv run pytest tests/ -v --tb=short

lint:
	uv run ruff check src/ tests/ app/
	uv run ruff format --check src/ tests/ app/

format:
	uv run ruff format src/ tests/ app/
