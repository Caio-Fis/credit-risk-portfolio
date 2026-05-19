# Backlog — para depois

Itens conscientemente adiados em 2026-05-18 após shippar o produto (backend FastAPI live + front Next.js bilíngue live). Decidi parar em "shippable" e priorizar tempo em outras coisas.

Quando voltar: pegar uma seção por vez, não tentar tudo de uma vez.

---

## Próximos passos (próxima sessão — 2026-05-19)

Online Learning v2 está LIVE em produção (Vercel + HF Space). O que sobrou do escopo original e o que apareceu durante o deploy:

### Tier 1 — fechar o ciclo Online Learning v2

- [ ] **Notebooks 12 e 13** (Fase 7.4 do `todo.md`) — `12_online_learning.ipynb` (ARF + drift + champion vs challenger) e `13_adaptive_shap.ipynb` (rebaselined SHAP + per-decile + Ridge surrogate) para reprodutibilidade. ~3-4h.
- [ ] **Testes unitários para `src/models/online_pd_model.py` e `src/explain/shap_adaptive.py`** (Fase 7.3) — a sessão de hoje cobriu só a API. ~2-3h.
- [ ] **CI smoke test do pipeline online** (Fase 7.5) — atualmente CI só roda testes unitários; adicionar um smoke que valida `make pipeline-lc` em sample reduzido. ~1-2h.

### Tier 2 — produto adiado

- [ ] **`/portfolio` — batch CSV upload** consumindo `POST /v1/predict/batch`. Para o usuário/analista subir uma planilha e ter score em massa. Estimativa: ~4-6h.
- [ ] **Domínio custom na Vercel** — está em `credit-risk-portfolio.vercel.app`. Se virar projeto real, comprar domínio.

### Tier 3 — débito técnico de hoje

- [ ] **Tracking de binários via Git LFS no repo do GitHub** — hoje `*.joblib`/`*.parquet`/`*.png` estão como blobs raw no GitHub. Funciona pra esse tamanho, mas se o joblib crescer (ex: ARF persistido) vai estourar limite. Considerar `git lfs migrate import --include="*.joblib,*.parquet,*.png" --everything`. **CUIDADO:** rewrite de história, força-push para `origin/master`. Não-bloqueante hoje, é higiene futura.
- [ ] **README das três versões (root, web/, hf_space/)** — divergiram um pouco hoje (hf_space/README.md ainda lista 9 endpoints, deveria ter 17). Mecanismo automático ou manual de sincronização.
- [ ] **Atualizar versões Node.js em GitHub Actions** — warnings de deprecação de Node 20 (forced para Node 24 em 2026-06-02). `astral-sh/setup-uv@v3`, `actions/checkout@v4`, etc. precisam upgrade.

### Tier 4 — decisões arquiteturais

- [ ] **Streamlit `app/`** — ainda usa o modelo antigo v1 Home Credit, mas o projeto pivotou pra LendingClub v2. Decidir: (a) aposentar Streamlit, (b) migrar pra v2 (duplica esforço com Next.js), (c) manter como "vitrine acadêmica" v1.
- [ ] **CSVs em `data/processed/` versionados vs gerados** — hoje os 8 CSVs que back-eiam `/monitor` e `/insights` são commitados raw. Alternativa: gerar via `make pipeline-lc` em CI, salvar como GitHub Artifacts, baixar no build do Docker. Mais limpo mas mais frágil.

---

## Front — quick polish

- [x] **OG / Twitter / favicon próprio** _(feito 2026-05-19)_ — gerados via file-conventions do Next 16 em vez de PNG estáticos:
  - `web/src/app/opengraph-image.tsx` (ImageResponse 1200×630, gradient violet→emerald no design tokens da landing)
  - `web/src/app/twitter-image.tsx` (re-exporta o OG, `card: summary_large_image`)
  - `web/src/app/icon.tsx` (32×32) + `web/src/app/apple-icon.tsx` (180×180), mark "AC" sobre gradient violet→emerald. Default `favicon.ico` do Next removido.
  - `layout.tsx`: `metadataBase` + `openGraph` (com locale pt_BR e alternateLocale en_US) + `twitter` + `robots`. Build emite `/icon`, `/apple-icon`, `/opengraph-image`, `/twitter-image` como rotas estáticas prerendered.

- [x] **Smoke test em produção pós-deploy** _(feito 2026-05-19, parcial)_ — verificado via WebFetch + curl no live:
  - ✓ Landing PT-BR default (`html lang="pt-BR"`, hero "Decisões de crédito mais inteligentes em segundos.")
  - ✓ Toggle PT/EN presente na nav, com aria-label e visible text
  - ✓ Wordmark "Análise de Crédito" presente
  - ✓ Wizard 3-step: "Etapa 1 de 3 · Sobre o tomador" + "Carregar exemplo"
  - ✓ Empty state "Preencha para ver a análise"
  - ✓ `/explain` retorna 200 com "Modo avançado — análise detalhada"
  - ⚠ Falta verificar em browser real: toggle realmente troca wordmark, gauge+narrativa após análise, card violet visível, waterfall humanizado em /explain, mobile 360px. (Não dá pra fazer headless.)
  - ⚠ OG/Twitter meta no live: zero ainda — só aparece após Vercel rebuild com as mudanças desta sessão.

- [x] **Lighthouse pass** _(feito 2026-05-19, via chrome-for-testing local)_ — rodado em mobile + desktop. Resultados landing:
  - Mobile: Performance 63 / Accessibility 96 / Best Practices 100 / SEO 100. LCP 3.2s, TBT **4080ms** (script eval 4.5s)
  - Desktop: Performance 57 (curvas mais apertadas), LCP 2.4s, TBT 600ms
  - **Veredito sobre blur/gradients**: NÃO são o gargalo. São static + low-alpha + pointer-events-none, custo de paint <5ms. Manter como estão.
  - **Gargalo real**: JS execution. Chunk `043wleu_f8edf.js` (196 KB) tem 83% código não usado na landing — provavelmente recharts/radix sendo arrastado pra rota errada. Fix está fora do escopo "polish" mas vale como follow-up.
  - **Fixes aplicados**: A11y `color-contrast` (zinc-500 → zinc-400 em footer, page.tsx, trust-signals, lang-toggle, step numbers do HowItWorks) e `label-content-name-mismatch` (lang-toggle aria-label trocado por sr-only span). A11y final: **96 → 100**.
  - Reports salvos temporariamente em `/tmp/lh-mobile.json` e `/tmp/lh-desktop.json` (não commitados).

- [x] **JS bundle audit + code-split** _(feito 2026-05-19, follow-up do Lighthouse)_ — duas fontes de waste corrigidas:
  - `Toaster` (`sonner`) vivia no `app/layout.tsx`, então toda rota incluía sonner + next-themes mesmo sem chamar `toast()`. Movido pras duas únicas páginas que tostam (`/origination` e `/explain`). next-themes inteiro removido como dependência (o wrapper só pegava o tema via `useTheme()` que ficava em `"system"` ignorado, já que o html é forçadamente `dark`). Theme passou a ser hardcoded.
  - `ShapWaterfall` puxava `recharts` (~700 KB) por import estático dentro de `risk-details.tsx`, que é usado por ambas as rotas — mas o waterfall só renderiza quando `showTechnical && explanation` (i.e. /explain). Trocado por `next/dynamic` com `ssr: false`. Mesma coisa no import direto em `explain/page.tsx`.
  - **Resultado mensurado**: landing total-byte-weight **535 → 414 KiB** (Lighthouse mobile, local prod build); a11y 96 → **100**; LCP local 2.5s.
  - **TBT continua alto** (~4.1s mobile). Isso é custo de framework runtime (Next 16 + React 19 + Turbopack), não código nosso. Fica fora de escopo até voltar a virar gargalo.

- [x] **Frontend tests** _(feito 2026-05-19)_ — Vitest + RTL + jsdom configurados. 29 testes / 4 arquivos:
  - `src/lib/feature-labels.test.ts`: `humanizeKey`, `getFeatureLabel` PT+EN + fallback, `formatFeatureValue` para USD/pct/anos/etc.
  - `src/lib/narrative.test.ts`: `buildNarrative` ordering por |shap| desc, direction up/down, topN, imutabilidade, `riskBandFromPd` thresholds.
  - `src/lib/i18n/provider.test.tsx`: default PT, detecção via `navigator.language`, hydrate de `localStorage`, toggle persiste, junk em localStorage é ignorado.
  - `src/components/loan-wizard.test.tsx`: starts em step 1, não avança com campo inválido, avança com fields válidos, "Carregar exemplo" pula pra step 3, botão só aparece em step 1.
  - Scripts: `npm test` (run) e `npm run test:watch`.

---

## Backend / modelo — pivô online learning (parado em 2026-05-16)

Todo o `tasks/todo.md` (63 itens em 6 fases). Resumo do que está lá:

- **Fase 1 — Fundação de dados** — baixar LendingClub do Zenodo, schema versionado, feature engineering temporal-safe, estender macro pra FRED ao invés de só BCB
- **Fase 2 — Baseline estático refeito** — split temporal estrito (train ≤2014, val 2015, holdout 2016-2018), rolling OOT trimestral, capturar degradação na crise 2008
- **Fase 3 — Challenger online** — `ARFClassifier` (River) com warm-start, ADWIN+KSWIN, label delay simulado de 90d, plot champion vs challenger
- **Fase 4 — Calibração online** — sliding-window isotônica de 6 meses, comparar Brier antes/depois
- **Fase 5 — Adaptive SHAP** — replicar arXiv:2511.03807: SHAP background rebaseado por mês, per-slice reweighting, incremental Ridge surrogate. **Killer chart:** heatmap mensal de SHAP × feature em `artifacts/shap_drift_heatmap.png`
- **Fase 6 — Streamlit + integração**

**Decisão sobre esse pivô (a tomar quando voltar):**
- (a) executar inteiro — ~40h+, alto retorno técnico (esse é o diferencial vs. portfólios genéricos de PD)
- (b) escopo reduzido — só Fase 2.3 (rolling OOT do baseline atual, sem ARF). Captura o número da crise. ~4-6h
- (c) arquivar oficialmente e seguir em frente

---

## Features de produto adiadas

- [ ] **/portfolio — batch CSV upload** consumindo `POST /v1/predict/batch`. Para o usuário/analista subir uma planilha e ter score em massa.
- [x] **/monitor — dashboard live de drift** _(feito 2026-05-19)_ — consome `GET /v1/monitor/{drift,calibration,champion-vs-challenger,rolling-vs-frozen}` + `POST /v1/monitor/recalibrate`. 4 seções: drift events, calibração yearly (Brier raw/static/sliding), champion vs challenger (ARF AUROC/KS), rolling vs frozen (LGBM retreinado vs congelado em 2013). Bilingual PT/EN.
- [x] **/insights — adaptive SHAP no front** _(feito 2026-05-19, novo)_ — rota dedicada para due-diligence: heatmap CSS-grid mês × feature (top-12), per-decile bars com low/mid/high, Ridge surrogate coefs over time com toggle de features. Consome novo `GET /v1/explain/adaptive-shap`.
- [ ] **Domínio custom na Vercel** — está em `credit-risk-portfolio.vercel.app`. Se virar projeto real, comprar domínio.

---

## Higiene de repo

- [ ] **README das três versões (root, web/, hf_space/)** estão coerentes hoje, mas vão divergir conforme o produto evoluir. Considerar consolidar quando for repaginar de novo.
- [ ] **Streamlit `app/`** ainda usa o modelo antigo (v1 Home Credit). Decidir se mantém como "vitrine acadêmica" ou se aposenta após o pivô online learning.
