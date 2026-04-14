# Credit Risk Portfolio — End-to-end pipeline for SME credit risk

## The problem

Financial institutions that operate credit for micro and small businesses face a paradox: they must decide credit limit, tenor, and rate at origination, but the most relevant data about the borrower — payment behavior, seasonality, customer concentration — only surfaces months later.

The single-score model compounds the problem by collapsing all this complexity into a number between 0 and 1000 that ignores the operation's context. A score of 650 may be adequate for 30-day working capital and completely inadequate for a 48-month investment loan. Same CNPJ, radically different risks.

This project demonstrates an alternative architecture: separate PD and LGD models, enrichment with macroeconomic data, contextual scoring by product and tenor, and continuous drift monitoring — all with per-contract explainability.

---

## The solution

End-to-end pipeline that transforms registration, transactional, and macroeconomic data into:

- **Calibrated probability of default** (PD) — not just ranking, but probability on a real scale
- **Loss given default estimate** (LGD) — how much is lost if default occurs
- **Expected loss per contract** — EL = PD × LGD × EAD, in currency units
- **Contextual score by product and tenor** — the same client assessed differently for different products
- **Drift monitoring** — automatic alerts when the data distribution shifts

---

## Project structure

```
credit-risk-portfolio/
│
├── README.md                        ← you are here
├── pyproject.toml                   ← versioned dependencies
├── Makefile                         ← make pipeline | make app | make test
├── .github/workflows/ci.yml         ← automated tests on each push
│
├── data/
│   ├── raw/                         ← .gitignore — not versioned
│   ├── processed/                   ← features ready for training
│   └── schemas/                     ← versioned data contracts
│
├── src/
│   ├── ingestion/                   ← download, schema validation, partitioning
│   ├── features/
│   │   ├── build_features.py        ← batch feature store (30/90/365d windows)
│   │   └── macro_features.py        ← enrichment with BCB time series
│   ├── models/
│   │   ├── pd_model.py              ← LightGBM + isotonic calibration
│   │   ├── lgd_model.py             ← Beta / Tobit regression
│   │   └── expected_loss.py         ← EL = PD × LGD × EAD
│   ├── explain/                     ← SHAP values, waterfall per contract
│   ├── evaluate/                    ← AUROC, KS, Brier Score, calibration plot
│   ├── monitoring/
│   │   ├── psi.py                   ← Population Stability Index per feature
│   │   ├── drift_detector.py        ← alert when PSI > 0.20
│   │   └── vintage_analysis.py      ← default rate by cohort and maturity
│   ├── early_warning/
│   │   ├── score_trajectory.py      ← score drop > N pts in 30d
│   │   └── behavioral_signals.py    ← triggers: transaction volume, protests
│   └── contextual/
│       ├── data_generator.py        ← synthetic dataset with controlled DGP
│       ├── context_features.py      ← tenor, product, collateral as features
│       └── interaction_model.py     ← client × context interactions
│
├── notebooks/
│   ├── 01_eda.ipynb                 ← exploration + business storytelling
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_pd.ipynb
│   ├── 04_modeling_lgd.ipynb
│   ├── 05_expected_loss.ipynb
│   ├── 06_vintage_analysis.ipynb
│   ├── 07_drift_monitoring.ipynb
│   ├── 08_early_warning.ipynb
│   ├── 09_synthetic_data.ipynb      ← full DGP transparency
│   ├── 10_contextual_score.ipynb    ← same CNPJ, different products
│   └── 11_score_unico_falha.ipynb   ← quantitative demonstration of single-score limitations
│
├── app/
│   ├── Home.py                      ← project overview and narrative
│   └── pages/
│       ├── 1_Concessao.py           ← simulator: input → score → EL
│       ├── 2_Portfolio.py           ← portfolio dashboard with KPIs
│       ├── 3_EarlyWarning.py        ← alert list + score trajectory
│       ├── 4_Explicabilidade.py     ← interactive SHAP waterfall per contract
│       └── 5_ScoreContextual.py     ← same client, different products live
│
└── tests/
    ├── test_features.py
    ├── test_models.py
    └── test_monitoring.py
```

---

## Modules

### Module 1 — End-to-end PD + LGD pipeline (core)

**Dataset:** Home Credit Default Risk (Kaggle) enriched with macroeconomic time series from the Brazilian Central Bank (Selic rate, sectoral default rate, economic activity index).

**What it demonstrates:**
- Feature engineering across multiple relational tables with temporal windows
- Documented and reproducible feature store
- PD model (LightGBM) with probability calibration via isotonic regression
- Separate LGD model (Beta regression) — treated as a distinct problem
- Expected Loss calculation in absolute currency per contract
- Per-contract explainability via SHAP waterfall

**Target metrics:**
- AUROC > 0.78
- KS > 0.35
- Brier Score < 0.15
- Calibration plot within 95% confidence band

---

### Module 2 — Early warning and portfolio monitoring

**What it demonstrates:**
- Automated Population Stability Index (PSI) per feature with alerts
- Vintage analysis — cumulative default rate by origination cohort
- Early warning model based on score trajectory (drop > 50 pts in 30 days)
- Behavioral triggers: transaction volume drop, invoice issuance, protests
- Clear separation between data deterioration (population drift) and model deterioration

**Operational thresholds implemented:**
- PSI < 0.10 → stable
- PSI 0.10–0.20 → caution, investigate
- PSI > 0.20 → drift confirmed, retraining required

---

### Module 3 — Contextual score (differentiator)

**Dataset:** Synthetic with a fully documented and controlled data generating process (DGP).

**What it demonstrates:**
- Why the same CNPJ has radically different risk for 30-day working capital vs a 48-month investment loan
- How product, tenor, and collateral should enter as features — not as pre-processing filters
- Quantitative demonstration of the single-score limitation: simulation of decisions with single score vs contextual score, with calculated expected loss difference
- The single score is not wrong — it is insufficient. It becomes one feature among many.

---

## Results

OOS hold-out (20% stratified split, never seen during training):

| Model | AUROC | KS | Brier Score |
|-------|-------|----|-------------|
| PD — logistic baseline | 0.7747 | 0.41 | 0.0671 |
| PD — LightGBM without calibration | 0.7794 | 0.42 | 0.0671 |
| PD — LightGBM + isotonic regression | **0.7794** | **0.4186** | **0.0663** |
| LGD — Beta regression | R² = 0.82 | — | — |

---

## How to run

```bash
# 1. clone and install
git clone https://github.com/Caio-Fis/credit-risk-portfolio
cd credit-risk-portfolio
pip install -e ".[dev]"

# 2. download data (requires Kaggle login)
make data

# 3. run full pipeline
make pipeline

# 4. start the app
make app
```

> The Makefile chains ingestion → features → training → evaluation → artifact export.
> Each step is idempotent: can be re-run without side effects.

---

## Technical decisions and trade-offs

**PD and LGD as separate models**
The cost of errors in each dimension differs. High PD with low LGD (real collateral) has a lower expected loss than medium PD with high LGD (no collateral). Collapsing everything into a single score loses this information at pricing time. The trade-off is operational complexity — two models to maintain, monitor, and retrain.

**LightGBM with isotonic calibration**
Gradient boosting produces good ranking but poorly calibrated probabilities — the model tends to compress probabilities away from 0 and 1. To calculate EL in currency units, you need probability on a real scale. Isotonic regression solves this with minimal computational cost. Alternative considered and discarded: Platt scaling (showed systematic miscalibration on this dataset per Hosmer-Lemeshow tests).

**Home Credit + BCB enrichment**
The Home Credit dataset has sufficient relational richness to demonstrate real feature engineering. The BCB enrichment simulates what a real company would do: add macroeconomic context to individual decisions. Bureau data alone is insufficient for long tenors — the default rate of an SME in recession is structurally different from the same SME during expansion.

**Synthetic dataset in module 3**
No public dataset allows controlling product and tenor as a clean experimental variable. The synthetic dataset with a documented DGP is more honest than trying to extract this effect from observational data where product and tenor are correlated with borrower profile. The transparency of the DGP is part of the argument — the reader can verify that the effect was not fabricated.

**Streamlit for the app**
Deliberate trade-off between delivery speed and robustness. For a portfolio, Streamlit delivers a deployable interactive interface in hours. The app code is kept separate from `src/` to make explicit that it is a presentation layer, not business logic.

---

## Context and motivation

This project was born from a practical question: how should a fintech that operates investment credit for SMEs build its risk pipeline from scratch, without its own default history?

The answer involves admitting what is unknown (without own history, the first model is a rules scorecard, not ML), building the data collection infrastructure from day one, and explicitly separating business decisions from technical decisions — because the asymmetric cost of error defines the threshold, not the model.

The three modules reflect this progression: origination → monitoring → critique of the current model.

---

## Author

**Caio Silva**
[linkedin.com/in/caio-silva-027974220](https://www.linkedin.com/in/caio-silva-027974220)
