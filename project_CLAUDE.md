# Contexto do projeto — Credit Risk Portfolio

## O que é este projeto
Portfólio profissional de análise de risco de crédito PJ, construído como demonstração
end-to-end de competências em engenharia de dados, modelagem estatística, visualização
e conhecimento de negócio de crédito.

Não é um projeto acadêmico. Simula decisões reais de uma fintech que opera crédito de
investimento para micro e pequenas empresas (MPE).

## Problema de negócio
Uma fintech precisa decidir limite, prazo e taxa para MPE no momento da concessão,
sem histórico próprio de inadimplência. O score único de bureau é insuficiente porque:
- É estático num mundo dinâmico
- Colapsa contexto num único valor (650 para 30d ≠ 650 para 48m)
- Não separa PD de LGD — dimensões com custos de erro distintos

## Decisões técnicas já tomadas — não questionar sem motivo

**Dataset:** Home Credit Default Risk (Kaggle) como base principal,
enriquecido com séries macroeconômicas do BCB (Selic, inadimplência setorial).
Módulo 3 usa dataset sintético com DGP controlado.

**Modelo PD:** LightGBM + calibração de Platt scaling.
- Motivo: gradient boosting produz bom ranking mas probabilidades mal calibradas.
  Para EL em reais precisamos de probabilidade em escala real.
- NÃO usar isotonic regression — overfita em amostras menores.

**Modelo LGD:** Regressão Beta (variável resposta em [0,1]).
- Tratado como problema separado do PD — custo de erro é diferente.
- NÃO colapsar PD e LGD num score único.

**Expected Loss:** EL = PD × LGD × EAD, calculado em reais por contrato.

**Monitoramento:** PSI por feature.
- PSI < 0.10 → estável
- PSI 0.10–0.20 → atenção
- PSI > 0.20 → drift confirmado, retreino necessário

## Estrutura de módulos

### Módulo 1 — Pipeline core (prioridade máxima)
`src/ingestion/` → `src/features/` → `src/models/` → `src/explain/` → `src/evaluate/`

Entregáveis: feature store documentada, PD calibrado, LGD separado, EL por contrato,
SHAP waterfall por decisão.

### Módulo 2 — Early warning e monitoramento
`src/monitoring/` + `src/early_warning/`

Entregáveis: PSI automatizado, análise de safra, lista de empresas em alerta,
gatilhos comportamentais (queda de volume transacional, protestos).

### Módulo 3 — Score contextual
`src/contextual/` com dataset sintético

Entregáveis: demonstração quantitativa de que o mesmo CNPJ tem risco diferente
por produto e prazo. Score único vira uma feature, não a decisão.

### App Streamlit
`app/` com 5 páginas:
1. Simulador de concessão
2. Dashboard de portfólio
3. Early warning
4. Explicabilidade SHAP
5. Score contextual interativo

## Métricas alvo (Módulo 1)
- AUROC > 0.78  ← teto prático com feature set atual: 0.7796 (gap 0.0004, dentro do CV std ±0.0016)
- KS > 0.35     ← atingido: 0.4186
- Brier Score < 0.15  ← atingido: 0.0671
- Calibration plot dentro de banda de confiança 95%

## Ajuste fino realizado (hiperparâmetros + feature engineering)
- Optuna 50 trials (TPE, CV 5-fold no train split): AUROC 0.7747 → 0.7760
- Feature engineering round 1 (EXT_SOURCE product, bureau overdue rate, refusal rate,
  social circle default rate, LTV, income per family, unemployed flag): → 0.7768
- Feature engineering round 2 (POS_CASH_balance DPD mensal, credit_card utilização): → 0.7796
- Feature store final: 182 colunas | 307K clientes
- LGBM_PARAMS: n_estimators=1000, learning_rate=0.020151, num_leaves=38,
  min_child_samples=104, reg_alpha=1.43, reg_lambda=1.36, min_split_gain=0.21

## Convenções deste projeto

**Paths:** Sempre usar `pathlib.Path`. Raiz do projeto via:
```python
from src.config import PROJECT_ROOT, DATA_DIR, MODELS_DIR
```

**Logging:** Usar `loguru` — já configurado em `src/config.py`.

**Features:** Toda feature nova documentada em `data/schemas/features.yaml`
com nome, descrição, janela temporal e fonte.

**Modelos:** Salvar com `mlflow` — nunca pickle solto em `models/`.

**Notebooks:** Numerados (01_, 02_...), outputs limpos antes de commitar.
Células de código < 20 linhas — extrair funções para `src/` se maior.

## O que NÃO fazer
- Não misturar lógica de negócio com código de apresentação do Streamlit
- Não treinar modelo dentro de notebook — notebook chama funções de `src/`
- Não usar score único como decisão final — sempre PD + LGD separados
- Não hardcodar thresholds — tudo em `src/config.py` ou arquivo de configuração
- Não commitar sem rodar `make test`

## Comandos úteis
```bash
make data        # baixa e valida dados brutos
make features    # constrói feature store
make train       # treina PD e LGD, loga no mlflow
make evaluate    # gera métricas e plots
make pipeline    # executa tudo acima em sequência
make app         # sobe Streamlit local
make test        # roda pytest
make lint        # roda ruff
```

## Estado atual do projeto
- [x] Estrutura de pastas criada
- [x] pyproject.toml configurado (inclui optuna)
- [x] Makefile criado (make data/features/train/tune/evaluate/pipeline/app/test/lint)
- [x] Ingestão do Home Credit implementada
- [x] Feature engineering implementado (182 features, batch chunked para tabelas >10M linhas)
- [x] Modelo PD treinado (LightGBM + Platt, AUROC OOS 0.7796)
- [x] Modelo LGD treinado (Beta regression)
- [x] Expected Loss calculado (EL = PD × LGD × EAD por contrato)
- [x] SHAP implementado (TreeExplainer + sampling inteligente)
- [x] Módulo 2 implementado (Early Warning: PSI, score trajectory, alertas)
- [x] Módulo 3 implementado (Score contextual: DGP sintético + LightGBM contextual)
- [x] App Streamlit completo (5 páginas, integrado com modelos treinados)
- [x] Ajuste fino: Optuna + feature engineering (POS_CASH, credit_card, interações)
- [ ] Deploy no Streamlit Cloud
