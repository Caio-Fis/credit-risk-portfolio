# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# Stage 1 — builder: install deps into an isolated venv
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Bring in the uv binary (fast, deterministic, lock-aware)
COPY --from=ghcr.io/astral-sh/uv:0.5.5 /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Build-only system deps for LightGBM, NumPy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Resolve and install dependencies first (cached layer) — without project
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python 3.11 && \
    uv sync --no-install-project --no-dev

# Copy source and install the project itself (editable=False in container)
COPY src/ ./src/
COPY artifacts/pd_model_lc.joblib artifacts/pd_calibrator_lc.joblib ./artifacts/
COPY data/processed/macro_features.parquet ./data/processed/
COPY data/processed/arf_drifts_lc.csv data/processed/sliding_calibration_lc.csv ./data/processed/
COPY data/schemas/ ./data/schemas/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

# ---------------------------------------------------------------------------
# Stage 2 — runtime: minimal slim image with just the venv + source
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# libgomp is needed at runtime by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CREDIT_RISK_ENVIRONMENT=production \
    PORT=7860

WORKDIR /app

# Pull venv and app payload from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Non-root user for HF Spaces compatibility
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
      sys.exit(0 if urllib.request.urlopen(f'http://localhost:{__import__(\"os\").environ.get(\"PORT\",\"7860\")}/health').status==200 else 1)"

# Honour PORT (HF Spaces convention is 7860 but the env var is the contract).
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860} --log-level info --proxy-headers"]
