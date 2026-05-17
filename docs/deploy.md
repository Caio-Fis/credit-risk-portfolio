# Deploy guide — Credit Risk PD API

Two deployment paths. Both ship the same Docker image; the difference is
just where it runs.

## Path A — HuggingFace Spaces (recommended)

Free tier: 2 vCPU / 16 GB RAM / no cold start. Best fit for this service.

### One-off setup

1. Create a Space at <https://huggingface.co/new-space>:
   - **SDK**: Docker
   - **Hardware**: CPU basic (free)
   - **Visibility**: public
2. In the new Space repo, replace the auto-generated `README.md` with the
   one under [`hf_space/README.md`](../hf_space/README.md). The YAML
   frontmatter is what tells HF "Docker, port 7860".
3. Copy `Dockerfile` (plus everything `Dockerfile` references — the
   `.dockerignore` already excludes what shouldn't ship).

### Sync from this GitHub repo

Two options:

**Option 1 — push directly**

```bash
git remote add space https://huggingface.co/spaces/<your-user>/credit-risk-api
git push space master
```

You may need a token (`HF_TOKEN`) configured as a Git credential.

**Option 2 — auto-sync via webhook**

In the Space's Settings → "Webhooks", point HF at the GitHub repo so
every push to `master` triggers a Space rebuild.

### What HF builds

HF reads `Dockerfile` from the Space root, builds it, and runs `CMD`
with `PORT=7860` injected. The healthcheck (`/health`) decides when the
Space goes online.

## Path B — local Docker / Docker Compose

For local end-to-end testing.

```bash
# Build (multi-stage, layer-cached)
docker build -t credit-risk-api:dev .

# Run
docker run --rm -p 7860:7860 credit-risk-api:dev

# Or via compose (auto-restart, healthcheck, env vars)
docker compose up -d
docker compose logs -f api

# Tear down
docker compose down
```

After the container reports healthy:

```bash
curl http://localhost:7860/health
curl http://localhost:7860/docs       # Swagger UI
curl http://localhost:7860/metrics    # Prometheus exposition
```

## Path C — bare uvicorn (dev only)

```bash
make install
uv run uvicorn src.api.main:app --reload --port 7860
```

The reload flag watches `src/` for edits.

## Environment variables

All prefixed with `CREDIT_RISK_` and validated by Pydantic Settings in
`src/api/settings.py`. The most useful overrides:

| Var | Default | Notes |
|---|---|---|
| `CREDIT_RISK_ENVIRONMENT` | `development` | Set to `production` for JSON logs |
| `CREDIT_RISK_LOG_LEVEL` | `INFO` | |
| `CREDIT_RISK_MODEL_PATH` | `artifacts/pd_model_lc.joblib` | Bind-mount a different artefact |
| `CREDIT_RISK_RECALIBRATION_CADENCE_DAYS` | `7` | |
| `PORT` | `7860` | HF Spaces injects this |

## Smoke tests once deployed

```bash
BASE=https://<your-space>.hf.space  # or http://localhost:7860

curl -s "$BASE/health" | jq .
curl -s "$BASE/v1/models/info" | jq '.metrics'

curl -s -X POST "$BASE/v1/predict" \
  -H 'content-type: application/json' \
  -d '{"revenue":65000,"dti_n":18.5,"loan_amnt":15000,"fico_n":720,
       "experience_c":1,"emp_length":5,"purpose":"debt_consolidation",
       "home_ownership_n":"MORTGAGE","addr_state":"CA","zip_code":"900xx",
       "issue_d":"2017-06-01"}' | jq .

curl -s "$BASE/v1/monitor/drift" | jq '.by_year'
curl -s "$BASE/v1/monitor/calibration" | jq '.summary'
```

## Memory / size budget

Final image is ~750 MB-1.1 GB depending on platform. Most of the weight
is shap / numpy / scikit-learn / lightgbm — already required by the
training pipeline. Model artefacts add only a few MB.
