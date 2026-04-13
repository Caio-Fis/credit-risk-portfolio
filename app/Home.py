"""Home — Project narrative and overview."""

import streamlit as st

st.set_page_config(
    page_title="Credit Risk Portfolio",
    page_icon="📊",
    layout="wide",
)

st.title("Credit Risk Portfolio")
st.subheader("End-to-end credit risk pipeline for SMEs")

st.markdown("""
---

## The problem

Financial institutions operating credit for micro and small enterprises face a paradox:
they must decide limits, tenors and rates **at the time of origination**, but the most relevant
data about the borrower — payment behaviour, seasonality, client dependency — only emerge
months later.

The **single score** aggravates the problem by collapsing all this complexity into a number between 0 and 1000
that ignores the context of the operation.

> A score of 650 may be adequate for 30-day working capital and completely inadequate
> for a 48-month investment. **Same client, radically different risks.**

---

## The solution

End-to-end pipeline that transforms registration, transactional and macroeconomic data into:

| Output | Description |
|---|---|
| **Calibrated PD** | Probability of default — not a ranking, but a probability on a real scale |
| **Estimated LGD** | How much is lost if default occurs |
| **Expected Loss** | EL = PD × LGD × EAD in currency units per contract |
| **Contextual score** | Same client assessed differently for different products |
| **Drift alerts** | Automated PSI — warning when the population changes |

---

## Modules
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    **Module 1 — Core Pipeline**

    LightGBM + isotonic calibration for PD.
    Beta regression for LGD.
    Expected Loss per contract.
    SHAP per decision.
    """)

with col2:
    st.warning("""
    **Module 2 — Early Warning**

    PSI per feature with automatic alerts.
    Vintage analysis by cohort.
    Score drop > 50pts in 30 days.
    Behavioural triggers.
    """)

with col3:
    st.success("""
    **Module 3 — Contextual Score**

    Synthetic dataset with documented DGP.
    Product and tenor as features, not filters.
    Quantitative demonstration of the limitations
    of the single score.
    """)

st.markdown("""
---

## Navigate the pages

Use the left menu to explore each module:

- **Origination**: Credit analysis simulator
- **Portfolio**: Portfolio dashboard with KPIs
- **Early Warning**: Deterioration alerts
- **Explainability**: SHAP waterfall per contract
- **Contextual Score**: Same client, different products

---

## Technical decisions

**PD and LGD as separate models**: the cost of error in each dimension is different.
High PD with low LGD (real collateral) has lower expected loss than medium PD with high LGD.

**LightGBM with isotonic calibration**: gradient boosting produces good ranking but poorly
calibrated probabilities. To compute EL in currency units, you need probability on a real scale.

**Single score is not wrong — it is insufficient**: it becomes one feature among many in the
contextual model.
""")
