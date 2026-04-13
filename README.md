# Credit Risk Portfolio — Pipeline end-to-end para crédito PJ

## O problema

Instituições financeiras que operam crédito para micro e pequenas empresas enfrentam um paradoxo: precisam decidir limite, prazo e taxa no momento da concessão, mas os dados mais relevantes sobre o tomador — comportamento de pagamento, sazonalidade, dependência de clientes — só aparecem meses depois.

O modelo de score único agrava o problema ao colapsar toda essa complexidade num número entre 0 e 1000 que ignora o contexto da operação. Um score 650 pode ser adequado para capital de giro de 30 dias e completamente inadequado para um investimento de 48 meses. O mesmo CNPJ, riscos radicalmente diferentes.

Este projeto demonstra uma arquitetura alternativa: modelos separados de PD e LGD, enriquecimento com dados macroeconômicos, score contextual por produto e prazo, e monitoramento contínuo de drift — tudo com explicabilidade por contrato.

---

## A solução

Pipeline end-to-end que transforma dados cadastrais, transacionais e macroeconômicos em:

- **Probabilidade de default calibrada** (PD) — não apenas ranking, mas probabilidade em escala real
- **Estimativa de perda dado default** (LGD) — quanto se perde se o default ocorrer
- **Expected loss por contrato** — EL = PD × LGD × EAD, em reais
- **Score contextual por produto e prazo** — o mesmo cliente avaliado diferente para produtos diferentes
- **Monitoramento de drift** — alertas automáticos quando a distribuição dos dados muda

---

## Estrutura do projeto

```
credit-risk-portfolio/
│
├── README.md                        ← você está aqui
├── pyproject.toml                   ← dependências versionadas
├── Makefile                         ← make pipeline | make app | make test
├── .github/workflows/ci.yml         ← testes automáticos em cada push
│
├── data/
│   ├── raw/                         ← .gitignore — não versionado
│   ├── processed/                   ← features prontas para treino
│   └── schemas/                     ← contratos de dado versionados
│
├── src/
│   ├── ingestion/                   ← download, validação de schema, particionamento
│   ├── features/
│   │   ├── build_features.py        ← feature store batch (janelas 30/90/365d)
│   │   └── macro_features.py        ← enriquecimento com séries do BCB
│   ├── models/
│   │   ├── pd_model.py              ← LightGBM + calibração de Platt
│   │   ├── lgd_model.py             ← regressão Beta / Tobit
│   │   └── expected_loss.py         ← EL = PD × LGD × EAD
│   ├── explain/                     ← SHAP values, waterfall por contrato
│   ├── evaluate/                    ← AUROC, KS, Brier Score, calibration plot
│   ├── monitoring/
│   │   ├── psi.py                   ← Population Stability Index por feature
│   │   ├── drift_detector.py        ← alerta quando PSI > 0.20
│   │   └── vintage_analysis.py      ← inadimplência por safra e maturidade
│   ├── early_warning/
│   │   ├── score_trajectory.py      ← queda de score > N pts em 30d
│   │   └── behavioral_signals.py    ← gatilhos: volume transacional, protestos
│   └── contextual/
│       ├── data_generator.py        ← dataset sintético com DGP controlado
│       ├── context_features.py      ← prazo, produto, garantia como features
│       └── interaction_model.py     ← interações cliente × contexto
│
├── notebooks/
│   ├── 01_eda.ipynb                 ← exploração + storytelling de negócio
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_pd.ipynb
│   ├── 04_modeling_lgd.ipynb
│   ├── 05_expected_loss.ipynb
│   ├── 06_vintage_analysis.ipynb
│   ├── 07_drift_monitoring.ipynb
│   ├── 08_early_warning.ipynb
│   ├── 09_synthetic_data.ipynb      ← transparência total do DGP
│   ├── 10_contextual_score.ipynb    ← mesmo CNPJ, produtos diferentes
│   └── 11_score_unico_falha.ipynb   ← demonstração quantitativa da limitação
│
├── app/
│   ├── Home.py                      ← visão geral e narrativa do projeto
│   └── pages/
│       ├── 1_Concessao.py           ← simulador: entrada → score → EL
│       ├── 2_Portfolio.py           ← dashboard de carteira com KPIs
│       ├── 3_EarlyWarning.py        ← lista de alertas + trajetória de score
│       ├── 4_Explicabilidade.py     ← SHAP waterfall interativo por contrato
│       └── 5_ScoreContextual.py     ← mesmo cliente, produtos diferentes ao vivo
│
└── tests/
    ├── test_features.py
    ├── test_models.py
    └── test_monitoring.py
```

---

## Módulos

### Módulo 1 — Pipeline end-to-end PD + LGD (core)

**Dataset:** Home Credit Default Risk (Kaggle) enriquecido com séries macroeconômicas do Banco Central do Brasil (Selic, inadimplência setorial, índice de atividade econômica).

**O que demonstra:**
- Engenharia de features sobre múltiplas tabelas relacionais com janelas temporais
- Feature store documentada e reproduzível
- Modelo de PD (LightGBM) com calibração de probabilidade via Platt scaling
- Modelo de LGD separado (regressão Beta) — tratado como problema distinto
- Cálculo de Expected Loss em valor absoluto por contrato
- Explicabilidade por SHAP com waterfall individual

**Métricas alvo:**
- AUROC > 0.78
- KS > 0.35
- Brier Score < 0.15
- Calibration plot dentro de banda de confiança de 95%

---

### Módulo 2 — Early warning e monitoramento de carteira

**O que demonstra:**
- Population Stability Index (PSI) automatizado por feature com alertas
- Análise de safra — inadimplência acumulada por coorte de originação
- Modelo de early warning baseado em trajetória de score (queda > 50pts em 30d)
- Gatilhos comportamentais: queda de volume transacional, emissão de NF-e, protestos
- Separação clara entre deterioração de dado (drift de população) e deterioração de modelo

**Limites operacionais implementados:**
- PSI < 0.10 → estável
- PSI 0.10–0.20 → atenção, investigar
- PSI > 0.20 → drift confirmado, retreino necessário

---

### Módulo 3 — Score contextual (diferencial)

**Dataset:** Sintético com data generating process (DGP) totalmente documentado e controlado.

**O que demonstra:**
- Por que o mesmo CNPJ tem risco radicalmente diferente para capital de giro de 30 dias vs investimento de 48 meses
- Como produto, prazo e garantia devem entrar como features — não como filtros de pré-processamento
- Demonstração quantitativa da limitação do score único: simulação de decisões com score único vs score contextual, com diferença de expected loss calculada
- O score único não é errado — é insuficiente. Ele vira uma feature entre muitas.

---

## Resultados

> *Preencher após treino e validação dos modelos.*

| Modelo | AUROC | KS | Brier Score |
|--------|-------|----|-------------|
| PD — baseline logística | — | — | — |
| PD — LightGBM sem calibração | — | — | — |
| PD — LightGBM + Platt | — | — | — |
| LGD — regressão Beta | R² = — | — | — |

---

## Como rodar

```bash
# 1. clonar e instalar
git clone https://github.com/seu-usuario/credit-risk-portfolio
cd credit-risk-portfolio
pip install -e ".[dev]"

# 2. baixar dados (requer login Kaggle)
make data

# 3. rodar pipeline completo
make pipeline

# 4. subir o app
make app
```

> O Makefile encadeia ingestão → features → treino → avaliação → exportação de artefatos.
> Cada etapa é idempotente: pode ser reexecutada sem efeito colateral.

---

## Decisões técnicas e trade-offs

**PD e LGD como modelos separados**
O custo de errar em cada dimensão é diferente. PD alta com LGD baixa (garantia real) tem expected loss menor que PD média com LGD alta (sem garantia). Colapsar tudo num score único perde essa informação no momento da precificação. O trade-off é complexidade operacional — dois modelos para manter, monitorar e retreinar.

**LightGBM com calibração de Platt**
Gradient boosting produz bom ranking mas probabilidades mal calibradas — o modelo tende a comprimir as probabilidades para longe de 0 e 1. Para calcular EL em reais, você precisa de probabilidade em escala real. O Platt scaling resolve isso com custo computacional mínimo. Alternativa considerada e descartada: isotonic regression (overfita em amostras menores).

**Home Credit + enriquecimento BCB**
O dataset Home Credit tem riqueza relacional suficiente para demonstrar feature engineering real. O enriquecimento com BCB simula o que uma empresa real faria: adicionar contexto macroeconômico à decisão individual. Dado de bureau sozinho é insuficiente para prazo longo — a inadimplência de uma MPE em recessão é estruturalmente diferente da mesma MPE em expansão.

**Dataset sintético no módulo 3**
Nenhum dataset público permite controlar produto e prazo como variável experimental de forma limpa. O sintético com DGP documentado é mais honesto que tentar extrair esse efeito de dados observacionais onde produto e prazo são correlacionados com o perfil do tomador. A transparência do DGP é parte do argumento — o leitor pode verificar que o efeito não foi fabricado.

**Streamlit para o app**
Trade-off deliberado entre velocidade de entrega e robustez. Para portfólio, Streamlit entrega uma interface interativa deployável em horas. O código do app é mantido separado de `src/` para deixar explícito que é camada de apresentação, não lógica de negócio.

---

## Contexto e motivação

Este projeto nasceu de uma pergunta prática: como uma fintech que opera crédito de investimento para MPE deveria construir seu pipeline de risco do zero, sem histórico próprio de inadimplência?

A resposta passa por admitir o que não se sabe (sem histórico próprio, o primeiro modelo é um scorecard de regras, não ML), construir a infraestrutura de coleta desde o dia 1, e separar explicitamente as decisões de negócio das decisões técnicas — porque o custo assimétrico do erro define o threshold, não o modelo.

Os três módulos refletem essa progressão: concessão → monitoramento → crítica ao modelo vigente.

---

## Autor

> *Preencher com nome, LinkedIn, contato.*
