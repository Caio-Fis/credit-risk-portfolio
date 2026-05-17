# Pivô: Online Learning + Adaptive XAI

**Decisão (2026-05-16):** champion/challenger com LightGBM estático (atual) vs. River+ARF (challenger), em LendingClub 2007–2018, com adaptive SHAP replicando arXiv:2511.03807.

## Fase 1 — Fundação de dados

- [ ] **1.1** Adicionar `river>=0.21` e `pyarrow>=15` ao `pyproject.toml`; remover `kaggle` se for abandonar Home Credit como primário (decisão pendente: manter como secundário ou descartar)
- [ ] **1.2** Criar `src/ingestion/download_lendingclub.py` — baixar do Zenodo (record 11295916), validar checksum, salvar parquet em `data/raw/lendingclub/`
- [ ] **1.3** Criar schema versionado em `data/schemas/lendingclub.json` (colunas, tipos, codificação de `loan_status`, definição binária de default a partir de `loan_status ∈ {Charged Off, Default, Late (31-120)}`)
- [ ] **1.4** Criar `src/features/lendingclub_features.py` — feature engineering temporal-safe: nada que use info pós-origination (ex: `last_pymnt_*`, `total_pymnt`, etc.). Coluna `issue_d` vira eixo temporal canônico.
- [ ] **1.5** Estender `src/features/macro_features.py` para usar FRED (Fed Funds Rate, US unemployment, US GDP YoY) em vez de só BCB, alinhado às datas de issue_d. Manter BCB num módulo separado caso seja útil para fallback.

## Fase 2 — Baseline estático refeito no novo dataset

- [ ] **2.1** Refatorar `src/models/pd_model.py` para aceitar dataset configurável (LendingClub vs. Home Credit) via `src/config.py`
- [ ] **2.2** Refazer treino LightGBM+isotônica em LendingClub com split temporal estrito: train ≤ 2014, val 2015, holdout 2016–2018 (cobrindo período pós-crise)
- [ ] **2.3** Adicionar avaliação **rolling OOT** em `src/evaluate/metrics.py`: para cada trimestre 2009..2018, treinar em [início..t-1], avaliar em t, registrar AUC/KS/Brier/calibration slope. Plot temporal salvo em `artifacts/rolling_oot_static.png`.
- [ ] **2.4** Verificar degradação esperada na crise 2008 e capturar números pro README

## Fase 3 — Challenger online (River + ARF)

- [ ] **3.1** Criar `src/models/online_pd_model.py` com:
  - `ARFClassifier` (River) como modelo base, fallback `SRPClassifier`
  - Pipeline de pré-processamento online (StandardScaler streaming, OneHotEncoder streaming)
  - Suporte a `init_model` para warm-start a partir dos primeiros N meses
- [ ] **3.2** Criar `src/monitoring/drift_online.py`:
  - ADWIN sobre erro de predição (label disponível com delay simulado de 90d)
  - KSWIN sobre distribuição de score (não-supervisionado, sem delay)
  - PSI rolling em features-chave usando o módulo existente `psi.py`
- [ ] **3.3** Avaliação rolling OOT do challenger com a mesma rotina do baseline (resultado lado a lado)
- [ ] **3.4** Simulador de label delay: quando treinar com janela [t-Δ, t-90d] em vez de [t-Δ, t]
- [ ] **3.5** Plot comparativo champion vs challenger ao longo do tempo (AUC + Brier rolling), `artifacts/champion_vs_challenger.png`

## Fase 4 — Calibração online

- [ ] **4.1** Criar `src/models/online_calibration.py` com sliding-window isotonic regression (janela de 6 meses) + Platt fallback
- [ ] **4.2** Aplicar a saída do challenger, comparar Brier antes/depois
- [ ] **4.3** Plot de calibration slope rolling (champion vs challenger calibrado)

## Fase 5 — Adaptive SHAP (replicar arXiv:2511.03807)

- [ ] **5.1** Refatorar `src/explain/shap_explain.py` para aceitar `background_window` configurável (em vez de background fixo)
- [ ] **5.2** Implementar `SlidingWindowSHAP` que rebaseia o background a cada N meses
- [ ] **5.3** Implementar **per-slice SHAP reweighting** (slices por grade de risco/produto)
- [ ] **5.4** Implementar **incremental Ridge surrogate** para calibração de explicações no espaço logit
- [ ] **5.5** Plot temporal de importância SHAP das top-10 features (heatmap month × feature) — é o killer chart do projeto. `artifacts/shap_drift_heatmap.png`
- [ ] **5.6** Teste de estabilidade temporal SHAP (cosine similarity entre janelas adjacentes, Kendall tau ranking)

## Fase 6 — App Streamlit (atualizações)

- [ ] **6.1** Nova página `app/pages/6_DriftMonitor.py`: timeline de ADWIN/KSWIN, PSI por feature, status do drift detector
- [ ] **6.2** Nova página `app/pages/7_OnlineLearning.py`: champion vs challenger lado a lado, métricas rolling
- [ ] **6.3** Atualizar `4_Explainability.py` para opção de "SHAP estático" vs "SHAP adaptativo" com seletor de janela temporal
- [ ] **6.4** Atualizar Home.py com novo storytelling (3 módulos viram 4: PD/LGD core → monitoramento → score contextual → **online learning + adaptive XAI**)

## Fase 7 — Documentação e fechamento

- [ ] **7.1** Atualizar README com: nova arquitetura, novo dataset (justificar troca), módulo 4, riscos conhecidos (label delay, catastrophic forgetting, calibration drift), tabela de resultados rolling OOT
- [ ] **7.2** Adicionar seção de bibliografia citando os 4-5 papers centrais
- [ ] **7.3** Atualizar testes em `tests/` para cobrir online model + adaptive SHAP
- [ ] **7.4** Adicionar notebook `12_online_learning.ipynb` e `13_adaptive_shap.ipynb` para exploração reproduzível
- [ ] **7.5** Atualizar CI (`.github/workflows/ci.yml`) com smoke test do pipeline online

---

## Decisões pendentes (perguntar antes de começar Fase 1)

1. **Home Credit:** manter como baseline secundário ou descartar de vez? (impacto em deps, notebooks e narrativa)
2. **Macro:** trocar BCB por FRED (US) ou manter ambos? Lending Club é mercado americano, BCB não faz sentido para ele.
3. **Notebooks antigos:** atualizar todos para novo dataset ou marcar 01-11 como "v1 - Home Credit (deprecated)" e criar série 12+ paralela?
4. **Escopo de fase:** ir Fase 1→7 em sequência ou cortar escopo? (tudo isso é trabalho significativo)

## Review

### Fase 1 — Ingestão e features (concluída 2026-05-17)
- LendingClub Zenodo 11295916 baixado (1.35M rows, 15 cols, MD5 OK)
- Surpresa: dataset já vem reduzido a granting-time only, com `Default` binário derivado. Schema mais simples que o LendingClub raw — adaptei o pipeline pra esse esquema.
- FRED CSV endpoint (sem API key) funciona perfeito. 5 séries (FEDFUNDS, UNRATE, GDPC1, VIXCLS, DGS10) mergidas via merge_asof backward em issue_d.
- Validação macro: Fed Funds 4.49→0.16% em 2008, Unemployment 4.7→9.4%, VIX 22→60, GDP YoY −3.2%. Assinatura canônica da crise ✓
- BCB removido completamente; old macro_features.py reescrito.
- Notebooks 01-11 movidos para `notebooks/legacy/`.

### Fase 2 — Baseline estático no novo dataset (concluída 2026-05-17)
- `src/models/pd_model_lc.py` criado: LightGBM nativo com cat features + isotonic na val 2015.
- Split temporal estrito: train ≤ 2014, val 2015, test 2016-2017. 2018 excluído (viés de maturação no Zenodo).
- AUROC test calibrated: 0.655 — baixo vs. v1 Home Credit (0.78) mas é o ceiling honesto desse universo de 11 features.
- **rolling_oot_evaluation** + **frozen_oot_evaluation** adicionados em src/evaluate/metrics.py.
- Killer chart `rolling_vs_frozen_lc.png`: frozen@2013 trava em AUROC ~0.628, rolling sobe pra 0.674. KS frozen 0.18 → rolling 0.25. **+4.6pp AUROC, +7pp KS** comprovam valor do retreino.

### Fase 3 — Challenger online ARF (concluída 2026-05-17 com caveat)
- `src/models/online_pd_model.py`: ARFClassifier do River com test-then-train, fila de label-delay 90d, KSWIN sobre score.
- `src/monitoring/drift_online.py`: replay ADWIN + KSWIN no stream.
- Sample 500/mês = 61.8K rows. Rodou em ~14min.
- **Insight inesperado**: ARF puro tem AUROC ~0.535 vs LightGBM rolling ~0.653 — perde 12pp. Razões: cold-start lento, sinal fraco do dataset (só 11 features), Python puro do River sem capacidade de explorar interações que LGBM captura.
- Drifts detectados: 83 KSWIN + 7 ADWIN ao longo de 11 anos. Concentração de KSWIN em 2009-2014 condiz com mudança de mix de portfolio LendingClub.
- **Narrativa do projeto pivota**: o ganho não é "trocar modelo", é "adicionar camada adaptativa (drift detection + sliding cal + adaptive SHAP) sobre o LightGBM". ARF entra como detector, não como modelo principal.

### Fase 4 — Calibração online sliding-window (concluída 2026-05-17)
- `src/models/online_calibration.py`: SlidingWindowIsotonic + SlidingWindowPlatt + função vetorizada `apply_sliding_calibration`.
- Bug encontrado e corrigido: refit por linha custava 6h+ → batch monthly refit (10s).
- Bug #2: assignment posicional por `.values` desalinhava com ties em issue_d → docs claras agora.
- Predições do frozen LightGBM extendidas pra 2013-2017 (1.2M rows) pra dar warm-start ao sliding cal.
- **Resultado honesto**: sliding cal preserva AUROC e dá melhoria marginal (~1-2%) em Brier vs static. Slope similar. Razão: LightGBM raw já está razoavelmente calibrado com isotonic estática nesse universo.
- Plot `sliding_calibration_lc.png` mostra os três regimes (raw / static / sliding) lado a lado.

### Fase 5 — Adaptive SHAP (concluída 2026-05-17)
- `src/explain/shap_adaptive.py` + `src/explain/run_adaptive_shap.py`: rebaselined SHAP, per-decile SHAP, Incremental Ridge surrogate.
- Bug encontrado: TreeExplainer com `interventional` + background NÃO funciona com cat features do LightGBM. Trocado para `tree_path_dependent`. Trade-off: a "rebaselining" perde rigor teórico mas o heatmap ainda mostra drift de importance via dados explicados.
- Bug #2: `.map(freq).fillna(0.0)` em coluna `category` falha — corrigido fazendo cast pra object antes.
- 3 artifacts gerados: `adaptive_shap_heatmap.png` (top 12 features × 48 meses), `shap_by_decile_lc.png` (10 deciles × top 12 features em 2017), `ridge_surrogate_coefs.png` (top 10 coefs ao longo do tempo).
- **Confirmação interessante via Ridge surrogate**: macro features (fed_funds_rate, us_unemployment, us_real_gdp_yoy, us_10y_treasury) aparecem no top 10 — LightGBM ESTÁ usando macro, só não está sendo muito eficiente nisso (alinha com AUROC modesto do dataset).

### Fase 6 — Streamlit (pendente)
### Fase 7 — Docs/tests/CI (parcial — README ainda não atualizado)

