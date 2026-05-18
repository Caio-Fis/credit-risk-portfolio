#!/usr/bin/env bash
set -euo pipefail
API="${API_BASE_URL:-https://Caio-Fis-credit-risk-api.hf.space}"
echo "Fetching OpenAPI from $API/openapi.json"
curl -fsSL "$API/openapi.json" -o /tmp/credit-risk-openapi.json
npx --yes openapi-typescript /tmp/credit-risk-openapi.json -o src/lib/api-types.ts
echo "Wrote src/lib/api-types.ts"
