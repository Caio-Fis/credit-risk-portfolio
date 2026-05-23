# Credit Risk PD — Next.js frontend

Frontend para a FastAPI live em
[Caio-Fis-credit-risk-api.hf.space](https://Caio-Fis-credit-risk-api.hf.space/docs).

Stack: **Next.js 16 (App Router) · TypeScript · Tailwind v4 · shadcn/ui ·
@tanstack/react-query · react-hook-form + zod · Recharts**.

## Rotas

Tabela regenerada por `scripts/sync_readmes.py` (no root do repo) a partir
de `web/src/app/**/page.tsx`.

<!-- AUTO:frontend-routes -->
| Path | UI | API consumed |
|---|---|---|
| `/` | Landing — bilingual PT/EN, static marketing | — |
| `/explain` | Advanced view — SHAP waterfall + macro context | `POST /v1/predict` · `POST /v1/explain` |
| `/insights` | Due-diligence — adaptive SHAP heatmap (month × feature), per-decile attribution, Ridge surrogate | `GET /v1/explain/adaptive-shap` |
| `/monitor` | Risk & ops — drift timeline, calibration trend, retraining uplift, champion vs challenger | `GET /v1/monitor/{drift,calibration,champion-vs-challenger,rolling-vs-frozen}` · `POST /v1/monitor/recalibrate` |
| `/origination` | Analyst wizard with plain-English narrative | `POST /v1/predict` · `POST /v1/explain` |
| `/portfolio` | Batch CSV upload (en-US / pt-BR) + vintage analysis | `POST /v1/predict/batch` |
<!-- /AUTO:frontend-routes -->

## Dev

```bash
cd web
cp .env.example .env.local           # ou exporte NEXT_PUBLIC_API_BASE_URL
npm install
npm run dev                          # http://localhost:3000
```

Para apontar para uma API local em vez do HF Space:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:7860 npm run dev
```

## Regerar tipos da API

A partir do OpenAPI ao vivo:

```bash
./scripts/gen-types.sh               # gera src/lib/api-types.ts
```

## Build & produção local

```bash
npm run build && npm start           # serve em :3000
```

## Deploy (Vercel)

1. Importar o repo `Caio-Fis/credit-risk-portfolio` em vercel.com.
2. **Root Directory:** `web/`
3. Env var: `NEXT_PUBLIC_API_BASE_URL = https://Caio-Fis-credit-risk-api.hf.space`
4. Deploy.

Como a API tem `allow_origins=["*"]` (ver `src/api/settings.py`),
chamadas browser-side do domínio Vercel funcionam sem mudança no backend.
