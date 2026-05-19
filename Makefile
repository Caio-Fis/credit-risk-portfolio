.PHONY: data features train tune evaluate pipeline app test lint install \
        data-lc features-lc pipeline-lc smoke-lc

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

# Hermetic smoke for the LC pipeline: synthetic 15-col frame + committed FRED
# cache, ~1s. Catches schema regressions without downloading the 167 MB CSV.
smoke-lc:
	uv run pytest tests/test_pipeline_lc.py -v

# ---- v3 API (FastAPI service) ----
api-dev:
	uv run uvicorn src.api.main:app --reload --port 7860

api-prod:
	CREDIT_RISK_ENVIRONMENT=production uv run uvicorn src.api.main:app --host 0.0.0.0 --port 7860 --proxy-headers

docker-build:
	docker build -t credit-risk-api:dev .

docker-run:
	docker run --rm -p 7860:7860 credit-risk-api:dev

docker-up:
	docker compose up -d --build
	@echo "API available at http://localhost:7860  /docs"

docker-down:
	docker compose down

sync-hf-space:
	bash scripts/sync_hf_space.sh

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
