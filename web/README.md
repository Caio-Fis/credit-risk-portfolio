# Credit Risk PD — Next.js frontend

Frontend para a FastAPI live em
[Caio-Fis-credit-risk-api.hf.space](https://Caio-Fis-credit-risk-api.hf.space/docs).

Stack: **Next.js 16 (App Router) · TypeScript · Tailwind v4 · shadcn/ui ·
@tanstack/react-query · react-hook-form + zod · Recharts**.

## Rotas

| Path | Endpoint consumido | UI |
|---|---|---|
| `/` | `GET /v1/models/info` | Landing com métricas OOT |
| `/origination` | `POST /v1/predict` | Form 11 campos → PD calibrada + risk band + macro snapshot |
| `/explain` | `POST /v1/explain` | Form → SHAP waterfall horizontal + top 5 drivers |

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

## Próximos passos (out-of-scope do MVP)

- `/portfolio` — upload CSV → `POST /v1/predict/batch`
- `/monitor` — dashboard live com `GET /v1/monitor/drift/live` + `GET /v1/monitor/calibration`
- Botão de recalibrate (`POST /v1/monitor/recalibrate`)
