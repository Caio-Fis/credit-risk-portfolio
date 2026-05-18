# Backlog — para depois

Itens conscientemente adiados em 2026-05-18 após shippar o produto (backend FastAPI live + front Next.js bilíngue live). Decidi parar em "shippable" e priorizar tempo em outras coisas.

Quando voltar: pegar uma seção por vez, não tentar tudo de uma vez.

---

## Front — quick polish

- [ ] **OG / Twitter / favicon próprio** — `web/src/app/layout.tsx` só tem `title` + `description` no `Metadata`. Sem `openGraph` nem `twitter` o link compartilhado em LinkedIn/WhatsApp/Twitter mostra preview vazio. Falta:
  - `openGraph.title`, `openGraph.description`, `openGraph.images: [{ url: "/og.png", width: 1200, height: 630 }]`
  - `twitter.card: "summary_large_image"`
  - Criar `public/og.png` (1200×630) com tipografia do produto + screenshot da landing
  - Criar `public/favicon.ico` próprio (hoje é o default do Next 16)

- [ ] **Smoke test em produção pós-deploy** — após os pushes recentes (commits `39e4484`, `5c2b895`, `ef841b7`, `5490e86`) confirmar no live:
  - Landing em PT-BR por default
  - Toggle PT/EN no canto direito troca tudo (incluindo wordmark "Análise de Crédito" ↔ "Credit Analysis")
  - Wizard 3-step + botão "Carregar exemplo"
  - Gauge + narrativa renderizam após análise
  - Card violet "Modo avançado" aparece visível no fim do painel
  - `/explain` mostra waterfall com labels humanizadas e tooltip sem "shap : ..."
  - Mobile (DevTools 360px) navegável

- [ ] **Lighthouse pass** — gradients radiais + blur 3xl no `layout.tsx` podem pesar em LCP. Rodar lighthouse no deploy de produção e ver se vale reduzir blur ou trocar por `bg-image` estático.

- [ ] **Frontend tests** — zero cobertura hoje. Vitest + React Testing Library para travar:
  - Wizard valida cada step antes do continue
  - "Carregar exemplo" pula para step 3
  - Toggle EN/PT persiste em localStorage
  - `buildNarrative()` ordena por |shap| desc e gera bullets com direção correta
  - `getFeatureLabel` faz fallback quando key não está na dict

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
- [ ] **/monitor — dashboard live de drift** consumindo `GET /v1/monitor/drift/live` + `GET /v1/monitor/calibration`. Botão de recalibrate.
- [ ] **Domínio custom na Vercel** — está em `credit-risk-portfolio.vercel.app`. Se virar projeto real, comprar domínio.

---

## Higiene de repo

- [ ] **README das três versões (root, web/, hf_space/)** estão coerentes hoje, mas vão divergir conforme o produto evoluir. Considerar consolidar quando for repaginar de novo.
- [ ] **Streamlit `app/`** ainda usa o modelo antigo (v1 Home Credit). Decidir se mantém como "vitrine acadêmica" ou se aposenta após o pivô online learning.
