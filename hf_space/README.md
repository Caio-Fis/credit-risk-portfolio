---
title: Credit Risk PD API
emoji: 💳
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
short_description: Adaptive PD scoring with drift-aware explainability
---

# Credit Risk PD API

Production FastAPI service exposing the v2 adaptive PD pipeline from
[Caio-Fis/credit-risk-portfolio](https://github.com/Caio-Fis/credit-risk-portfolio).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/v1/models/info` | Trained model metadata + OOT metrics |
| POST | `/v1/predict` | Calibrated PD for one loan |
| POST | `/v1/predict/batch` | Up to 1000 loans |
| POST | `/v1/explain` | SHAP waterfall for one loan |
| GET | `/v1/monitor/drift` | ADWIN/KSWIN event timeline |
| GET | `/v1/monitor/calibration` | Rolling Brier / slope / refit |
| POST | `/v1/monitor/recalibrate` | Trigger background recalibration |
| GET | `/metrics` | Prometheus exposition |

Interactive Swagger UI lives at `/docs`; ReDoc at `/redoc`.

## Example

```bash
curl -X POST $SPACE_URL/v1/predict \
  -H 'content-type: application/json' \
  -d '{"revenue":65000,"dti_n":18.5,"loan_amnt":15000,"fico_n":720,
       "experience_c":1,"emp_length":5,"purpose":"debt_consolidation",
       "home_ownership_n":"MORTGAGE","addr_state":"CA","zip_code":"900xx",
       "issue_d":"2017-06-01"}'
```

## How this Space is built

This Space uses the `Dockerfile` from the upstream repo. Synchronise via the
Hugging Face web UI (Settings → "Pull from GitHub") or push manually:

```bash
git remote add space https://huggingface.co/spaces/<user>/credit-risk-api
git push space main
```

See `docs/deploy.md` upstream for the full deploy guide.
