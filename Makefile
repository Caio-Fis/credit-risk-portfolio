.PHONY: data features train tune evaluate pipeline app test lint install \
        data-lc features-lc pipeline-lc

install:
	uv sync --all-extras

# ---- Legacy pipeline (Home Credit, kept as secondary static baseline) ----
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

# ---- v2 pipeline (LendingClub 2007-2018 + FRED macro + online learning) ----
data-lc:
	uv run python -m src.ingestion.download_lendingclub

features-lc:
	uv run python -m src.features.macro_features
	uv run python -m src.features.lendingclub_features

pipeline-lc: data-lc features-lc

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
