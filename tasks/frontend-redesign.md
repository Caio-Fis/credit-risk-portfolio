# Frontend Redesign — Recruiter-friendly UI

**Decisão (2026-05-18):** Repaginar o front Next.js em `web/` para público recrutador/gestor de portfólio (não-credit-expert, mas tech-savvy). Direção visual Mercury/Linear (dark elegante, acentos sutis). Form vira wizard 3-step. Resultado vira gauge + narrativa em linguagem natural. SHAP cru fica em "ver detalhes".

## Princípios

- **Storytelling > jargão.** Cada termo técnico aparece com tooltip explicando.
- **Mobile-first.** Wizard e gauge funcionam em 360px.
- **Sem perder profundidade.** Toda info técnica continua acessível via expander/tooltip.
- **Zero breaking changes no backend.** Mesmas chamadas para `/v1/predict`, `/v1/explain`, `/v1/models/info`.

## Fase 1 — Foundation

- [ ] **1.1** Instalar shadcn components que faltam: `tooltip`, `progress`, `accordion`, `radio-group` (manual em `src/components/ui/`, sem rodar CLI do shadcn — já está estável)
- [ ] **1.2** Refinar `app/globals.css`: adicionar tokens de cor de acento (violet `#a78bfa` e emerald `#34d399` em opacidades baixas), gradient stops para hero, manter dark zinc como base
- [ ] **1.3** Atualizar `app/layout.tsx`: background com radial gradient sutil (violet + emerald em 5% opacity nos cantos), max-width ajustado para `max-w-5xl` em conteúdo (mais focado), `max-w-6xl` em hero/landing
- [ ] **1.4** Criar `src/lib/feature-labels.ts`: dicionário `feature_key → { label, helper, tooltip, format }` mapeando `dti_n` → "Debt-to-income", `fico_n` → "FICO credit score", `revenue` → "Annual income (USD)" etc. Inclui o que cada métrica mede e a faixa esperada.
- [ ] **1.5** Criar `src/lib/narrative.ts`: função `buildNarrative(contributions)` que converte SHAP em bullets em inglês claro com sinal e magnitude relativa ("Strong credit score reduced risk by ~12 points; high DTI added ~5"). Magnitude em "risk points" (SHAP * 100 arredondado).

## Fase 2 — Nav + footer

- [ ] **2.1** Refresh `components/nav.tsx`: logo com pontinho pulsante "Live" ao lado, hover underline animado (border-bottom transition), reorder dos links: Home → Score → API → GitHub
- [ ] **2.2** Footer mais quieto: uma linha só, links sutis, sem perder credenciais técnicas

## Fase 3 — Landing (/)

- [ ] **3.1** Hero: headline em duas linhas com gradient sutil na segunda metade, parágrafo de positioning humanizado (sem "ADWIN + KSWIN", explicar o valor: "predicts default risk, adapts to economic shifts, explains every decision"), 2 CTAs primários ("Score a loan" + "See how it works"), badge "Live · FastAPI on HF Spaces" mais sutil
- [ ] **3.2** Substituir cards de endpoints (`POST /v1/predict` etc) por **value cards**: 3 cards com ícone + título amigável + descrição:
  - "Score a new loan" — wizard de 30 segundos
  - "Understand any decision" — SHAP traduzido para inglês simples
  - "Adapts to drift" — auto-recalibration quando o macro muda
- [ ] **3.3** Card "Model in production" humanizado: cada métrica vira `MetricWithTooltip` (label friendly + valor + `?` que explica em uma frase). `AUROC_TEST_CALIBRATED` → "Discriminative power" com tooltip "How well the model separates defaulters from non-defaulters on out-of-time test data. 1.0 = perfect, 0.5 = coin flip."
- [ ] **3.4** Nova seção "How it works" com 3 steps visuais (icon + label + 1 sentence): Inputs → Calibrated PD → Explanation. Connectores sutis entre eles.

## Fase 4 — Score wizard (/origination)

- [ ] **4.1** Criar `components/loan-wizard.tsx` substituindo `loan-form.tsx`:
  - Step 1 — "About the borrower": revenue, employment length, experience class (label: "10+ years employed"), home ownership, state, ZIP
  - Step 2 — "The loan": amount, purpose, issue date
  - Step 3 — "Credit profile": FICO, DTI
  - Cada campo com helper text amigável + `?` tooltip com a definição técnica
  - Progress bar (1/3, 2/3, 3/3) no topo do card
  - Botões Back / Continue, último step troca Continue por "Score this loan"
  - Botão "Load sample loan" só visível no Step 1 (preenche `sampleLoan` e pula direto pro Step 3)
- [ ] **4.2** Refatorar `app/origination/page.tsx`: dispara `/predict` **e** `/explain` em paralelo no submit (Promise.all). Mostra skeleton da gauge enquanto carrega.
- [ ] **4.3** Layout em duas colunas no desktop (wizard à esquerda, result à direita); stack no mobile com result aparecendo abaixo
- [ ] **4.4** Estado "before submit": preview card explicando o que vai acontecer ("Submit to see the calibrated risk, explanation, and macro context") em vez do code-flavored placeholder atual

## Fase 5 — Result panel (gauge + narrative)

- [ ] **5.1** Criar `components/risk-gauge.tsx`: semi-circular gauge SVG custom (180°, arc cheio = 100% PD). Cores em zonas: emerald 0-5%, amber 5-15%, orange 15-30%, red 30%+. Big number central com PD%. Risk band label abaixo (`low` / `medium` / `high` / `very high`).
- [ ] **5.2** Criar `components/risk-narrative.tsx`: lista de 4-5 bullets renderizando `buildNarrative()` da Fase 1.5. Bullets verde quando reduzem risco, vermelho quando aumentam. Sentinela: "Your profile is above average / typical / below average for the test population" no topo.
- [ ] **5.3** Criar `components/risk-details.tsx`: Accordion expandido por padrão em desktop, colapsado em mobile, com:
  - Tab "Score breakdown": PD raw, PD calibrated, Score 0-1000 (cada um com tooltip)
  - Tab "Macro context": macro snapshot (FRED features) com tooltips
  - Tab "Full SHAP chart": o `ShapWaterfall` existente, com labels humanizadas usando `feature-labels.ts`
- [ ] **5.4** Refatorar `prediction-card.tsx` para deprecation (deletar depois que o novo result panel estiver no lugar) ou converter em wrapper

## Fase 6 — Explain page (/explain) — simplificar

- [ ] **6.1** Decisão: manter `/explain` como rota dedicada para quem quer focar só no SHAP, mas sem duplicar form (reusa `LoanWizard`)
- [ ] **6.2** Resultado: waterfall em destaque + top 5 drivers em narrativa, sem gauge (já está em /origination)
- [ ] **6.3** Subtitle com 1 frase humanizando TreeSHAP: "Each bar shows how much that detail moved the risk up or down vs. an average loan"

## Fase 7 — Polish + verificação

- [ ] **7.1** Loading states: skeleton da gauge match exato (mesma altura, mesmo svg shape) → zero CLS
- [ ] **7.2** Cold-start friendly copy: erro de fetch no `/models/info` mostra "Waking up the model… first request can take up to 20s on HuggingFace's free tier" com retry button
- [ ] **7.3** Tooltip em todo termo técnico: AUROC, Brier, PD, SHAP, DTI, FICO
- [ ] **7.4** Verificar contrast ratio AA em todas cores (especialmente acentos violet/emerald em fundo zinc-950)
- [ ] **7.5** Rodar `npm run build` no `web/` — zero TS errors, zero lint warnings
- [ ] **7.6** Rodar `npm run dev` e fazer um pass manual: landing → wizard step-by-step → load sample → see gauge → expand details → /explain → ver waterfall
- [ ] **7.7** Testar em mobile real (DevTools 360px width) — wizard navegável, gauge legível, narrativa flui

## Fora de escopo (talvez próxima rodada)

- Light mode (audiência primary é dark-tolerante)
- `/portfolio` (batch CSV upload) — fora do MVP
- `/monitor` dashboard (drift live) — fora do MVP
- i18n PT-BR (audiência é recrutador, inglês é certo)
- Animações Framer Motion além de transitions CSS — overhead não justifica
- Tema customizável (acento user-pickable)

## Review (preenchido depois)

_TBD_
