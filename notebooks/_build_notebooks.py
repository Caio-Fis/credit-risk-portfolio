"""Build the two reproducibility notebooks (12 + 13) from source.

Run with ``uv run python notebooks/_build_notebooks.py`` to regenerate
``12_online_learning.ipynb`` and ``13_adaptive_shap.ipynb``. Both notebooks
consume artefacts already on disk (``data/processed/*`` and
``artifacts/*``) — they do **not** retrain models from scratch, so they
finish in seconds and can ship in CI.

Keeping the source as code (instead of editing JSON by hand) means each
cell has a clear name and the structure stays diffable in git.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


def write(nb: nbf.NotebookNode, name: str) -> None:
    nb.metadata.setdefault("kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"})
    nb.metadata.setdefault("language_info", {"name": "python", "pygments_lexer": "ipython3"})
    target = NB_DIR / name
    nbf.write(nb, target)
    print(f"  wrote {target}")


# ---------------------------------------------------------------------------
# Notebook 12 — Online learning challenger (ARF + drift + champion-vs-challenger)
# ---------------------------------------------------------------------------
def build_nb12() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            """
# 12 — Online learning challenger

Reproducibility notebook for the v2 challenger built on top of LendingClub 2007-2018.
This notebook **loads pre-computed artefacts** (no training from scratch) — the
full ARF run takes ~14 min on a laptop and is driven by `src.models.online_pd_model`.

Three things in scope:

1. **ARF (Adaptive Random Forest)** as a streaming challenger to the frozen LightGBM
   champion, with **test-then-train** and a 90-day **label-delay** queue.
2. **Drift detection** on the live stream (KSWIN on the score distribution, ADWIN on
   the prediction error feed).
3. **Champion vs. challenger** comparison — yearly AUROC / Brier — to ground the
   narrative that the lift comes from the **adaptive layer**, not from replacing the
   model.

The companion notebook `13_adaptive_shap.ipynb` covers explanations under drift.
"""
        ),
        code(
            """
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Resolve the repo root regardless of where Jupyter is launched from.
ROOT = Path.cwd()
while ROOT != ROOT.parent and not (ROOT / 'pyproject.toml').exists():
    ROOT = ROOT.parent
PROCESSED = ROOT / 'data' / 'processed'
ARTIFACTS = ROOT / 'artifacts'

plt.rcParams['figure.dpi'] = 110
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
"""
        ),
        md(
            """
## 1. Champion (frozen LightGBM @ 2013) baseline

The static champion was trained on data up to 2014, calibrated on 2015 and frozen.
We loaded its predictions for 2013-2017 — that gives us a fair comparison window
against the streaming challenger.
"""
        ),
        code(
            """
frozen = pd.read_parquet(PROCESSED / 'frozen_lgbm_predictions_lc.parquet')
frozen['issue_d'] = pd.to_datetime(frozen['issue_d'])
print('frozen LGBM predictions:', len(frozen), 'rows | columns:', list(frozen.columns))
frozen.head()
"""
        ),
        code(
            """
# Rolling vs frozen — the killer chart from Fase 2.
rolling = pd.read_csv(PROCESSED / 'rolling_oot_lc.csv')
frozen_oot = pd.read_csv(PROCESSED / 'frozen_oot_lc.csv')
print('rolling OOT (retrained yearly):')
display(rolling)
print('frozen OOT (no retraining since 2013):')
display(frozen_oot)
"""
        ),
        md(
            """
## 2. Challenger artefacts — ARF stream

The ARF challenger is driven by `src.models.online_pd_model.stream_evaluate`. It
consumes the stream chronologically, predicts each record before learning from it
(test-then-train) and queues the label for release 90 days later.

We load the artefacts produced by the production run:

- `arf_predictions_lc.parquet` — one row per sample (issue_d, y, p).
- `arf_yearly_lc.csv` — yearly AUROC / KS / Brier / calibration slope.
- `arf_drifts_lc.csv` — drift events (KSWIN over the score, ADWIN over the error).
"""
        ),
        code(
            """
arf_preds = pd.read_parquet(PROCESSED / 'arf_predictions_lc.parquet')
arf_yearly = pd.read_csv(PROCESSED / 'arf_yearly_lc.csv')
arf_drifts = pd.read_csv(PROCESSED / 'arf_drifts_lc.csv')

arf_preds['issue_d'] = pd.to_datetime(arf_preds['issue_d'])
arf_drifts['timestamp'] = pd.to_datetime(arf_drifts['timestamp'])

print(f'ARF stream: {len(arf_preds):,} predictions across '
      f'{arf_preds["issue_d"].dt.year.nunique()} years')
print(f'Drift events: total={len(arf_drifts)} | '
      f'KSWIN={(arf_drifts.detector == "KSWIN").sum()} | '
      f'ADWIN={(arf_drifts.detector == "ADWIN").sum()}')
arf_yearly
"""
        ),
        md(
            """
## 3. Champion vs challenger — yearly side by side

Both models are evaluated on the *same* yearly windows. The picture confirms the
narrative recorded in `tasks/todo.md`: ARF underperforms LightGBM by ~10pp AUROC in
this universe (small feature set, no interaction modelling). The point of the
challenger is **drift detection + adaptive recalibration**, not replacement.
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

# AUROC
ax = axes[0]
ax.plot(rolling['year'], rolling['auroc'], 'o-', label='LightGBM rolling', lw=2)
ax.plot(frozen_oot['year'], frozen_oot['auroc'], 's--', label='LightGBM frozen@2013', lw=2, alpha=0.8)
ax.plot(arf_yearly['year'], arf_yearly['auroc'], 'D-.', label='ARF challenger', lw=2, color='#c44e52')
ax.set_title('AUROC over time')
ax.set_xlabel('issue year')
ax.set_ylabel('AUROC')
ax.legend(fontsize=8)
ax.set_ylim(0.45, 0.75)

# Brier
ax = axes[1]
ax.plot(rolling['year'], rolling['brier'], 'o-', label='LightGBM rolling', lw=2)
ax.plot(frozen_oot['year'], frozen_oot['brier'], 's--', label='LightGBM frozen@2013', lw=2, alpha=0.8)
ax.plot(arf_yearly['year'], arf_yearly['brier'], 'D-.', label='ARF challenger', lw=2, color='#c44e52')
ax.set_title('Brier score over time (lower is better)')
ax.set_xlabel('issue year')
ax.set_ylabel('Brier')
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
The pre-rendered version of this comparison lives at
`artifacts/champion_vs_challenger.png`. Feel free to skip the cell above and
inline the image if you're running this on a machine without the parquet files.
"""
        ),
        md(
            """
## 4. Drift events timeline

KSWIN reacts to changes in the **score distribution** — no labels needed, fires
immediately when the input mix shifts. ADWIN reacts to the **prediction error**
itself, so it lags by the label delay but tells you whether the shift hurt.
"""
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(11, 3.5))

# All ARF predictions become a low-alpha rug so the timeline is obvious.
ax.scatter(arf_preds['issue_d'], [0.02] * len(arf_preds), s=0.4, alpha=0.05, color='gray', label='_nolegend_')

# Drift events as colour-coded ticks.
for detector, marker, color, y in [('KSWIN', 'v', '#dd8452', 0.55), ('ADWIN', '^', '#c44e52', 0.85)]:
    events = arf_drifts[arf_drifts['detector'] == detector]
    ax.scatter(events['timestamp'], [y] * len(events), marker=marker, color=color,
               s=40, label=f'{detector} ({len(events)})')

ax.set_yticks([0.02, 0.55, 0.85])
ax.set_yticklabels(['stream', 'KSWIN', 'ADWIN'])
ax.set_title('Drift detection on the ARF stream')
ax.set_xlabel('issue date')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
The concentration of KSWIN flags between 2009 and 2014 lines up with the documented
shift in LendingClub's product mix during that period (more 60-month loans, higher
average loan amount). This is the kind of signal that should fire a recalibration
job in production.
"""
        ),
        md(
            """
## 5. Mini live demo — re-run the stream on a tiny subsample

Sanity check the API and confirm the artefacts came from the same code path. We
shrink to 50 records per month so this finishes in <30s on a laptop.
"""
        ),
        code(
            """
import sys
sys.path.insert(0, str(ROOT))

from src.config import LENDINGCLUB_FEATURES
from src.models.online_pd_model import TARGET_COL, DATE_COL, stream_evaluate

df = pd.read_parquet(LENDINGCLUB_FEATURES)
feature_cols = [c for c in df.columns if c not in {TARGET_COL, DATE_COL}]
print('feature_cols:', feature_cols)
print('rows:', len(df))
"""
        ),
        code(
            """
demo = stream_evaluate(
    df,
    feature_cols=feature_cols,
    samples_per_month=50,   # ~7k rows total; ~30s
    n_models=5,
    label_delay_days=90,
    log_every=2000,
)
demo.yearly
"""
        ),
        code(
            """
print(f'demo predictions: {len(demo.predictions):,}')
print(f'demo drift events: {len(demo.drift_events)}')
demo.predictions.head()
"""
        ),
        md(
            """
## 6. Takeaways

- **Static champion still wins on raw discrimination** in this dataset — only 11
  features and weak signal, which limits the tree-based online learner.
- **Challenger earns its keep through drift telemetry**: KSWIN fires early
  (no label needed), ADWIN confirms damage post-hoc when the delayed labels arrive.
- **Production architecture** uses both: LightGBM (champion) for scoring, ARF
  (challenger) for monitoring + as the recalibration trigger.

Next: notebook 13 looks at **how the importance landscape itself moves** under
drift — the killer chart of the project.
"""
        ),
    ]
    write(nb, "12_online_learning.ipynb")


# ---------------------------------------------------------------------------
# Notebook 13 — Adaptive SHAP (rebaselined SHAP, per-decile, Ridge surrogate)
# ---------------------------------------------------------------------------
def build_nb13() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            """
# 13 — Adaptive SHAP under drift

Reproducibility notebook for `src.explain.shap_adaptive` — the v2 explanation
layer. Replicates the three ideas from
*Fair and Explainable Credit Scoring under Concept Drift* (Shivogo John,
arXiv:2511.03807, 2025):

1. **Sliding-window rebaselined SHAP** — recompute TreeSHAP each month with a
   rolling background so the baseline reflects the recent population.
2. **Per-risk-decile SHAP** — same model, but the importance mix changes when
   you stratify by predicted PD.
3. **Incremental Ridge surrogate** — an online linear approximation of the tree
   model whose coefficients are easy to track over time.

The pre-computed artefacts already on disk drive the figures. Re-run the source
script at `src.explain.run_adaptive_shap` if you want to regenerate them.
"""
        ),
        code(
            """
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd()
while ROOT != ROOT.parent and not (ROOT / 'pyproject.toml').exists():
    ROOT = ROOT.parent
PROCESSED = ROOT / 'data' / 'processed'
ARTIFACTS = ROOT / 'artifacts'

plt.rcParams['figure.dpi'] = 110
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
"""
        ),
        md(
            """
## 1. Monthly rebaselined SHAP — month × feature heatmap

For each month from 2014 to 2017, TreeSHAP is run with a rolling 6-month
background. We aggregate `mean(|SHAP|)` per feature and pivot to a heatmap —
**the shifting columns of heat make drift visible**.
"""
        ),
        code(
            """
monthly = pd.read_csv(PROCESSED / 'adaptive_shap_monthly.csv')
print(f'months: {monthly["month"].nunique()} | features: {monthly["feature"].nunique()}')
monthly.head()
"""
        ),
        code(
            """
pivot = monthly.pivot(index='month', columns='feature', values='mean_abs_shap')
top_k = 12
top = pivot.mean(axis=0).sort_values(ascending=False).head(top_k).index.tolist()
pivot = pivot[top]

fig, ax = plt.subplots(figsize=(13, 7))
im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=7)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=35, ha='right')
ax.set_title(f'Adaptive SHAP — top {top_k} features × month (rebaselined background)')
fig.colorbar(im, ax=ax, label='mean |SHAP|')
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
The macro block (`fed_funds_rate`, `us_unemployment`, `us_10y_treasury`,
`vix_close`, `us_real_gdp_yoy`) holds steady weight across most of 2014-2017 and
visibly **lifts in mid-2016 and early-2017** — the period when the Fed funds rate
started its post-crisis hike cycle. Tree-based models that see macro features
*are* using them, even if the marginal lift in AUROC on this 11-feature universe
is modest.
"""
        ),
        md(
            """
## 2. Per-risk-decile SHAP — 2017 cohort

Same model, same 2017 cohort, but stratified by **predicted PD decile**. The
importance mix is not stationary across the risk grade: low-risk loans are
explained more by borrower factors, high-risk loans more by macro.
"""
        ),
        code(
            """
by_dec = pd.read_csv(PROCESSED / 'adaptive_shap_by_decile.csv')
print(f'deciles: {by_dec["decile"].nunique()} | features: {by_dec["feature"].nunique()}')
by_dec.head()
"""
        ),
        code(
            """
pivot_dec = by_dec.pivot(index='decile', columns='feature', values='mean_abs_shap')
top_dec = pivot_dec.mean(axis=0).sort_values(ascending=False).head(12).index.tolist()
pivot_dec = pivot_dec[top_dec]

fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(pivot_dec.values, aspect='auto', cmap='magma')
ax.set_yticks(range(len(pivot_dec.index)))
ax.set_yticklabels([f'decile {d}' for d in pivot_dec.index])
ax.set_xticks(range(len(pivot_dec.columns)))
ax.set_xticklabels(pivot_dec.columns, rotation=35, ha='right')
ax.set_title('SHAP by predicted-risk decile — 2017 cohort')
fig.colorbar(im, ax=ax, label='mean |SHAP|')
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
## 3. Incremental Ridge surrogate — coefficient drift

The Ridge surrogate is fit each month on a 6-month sliding window of
(features → logit(p̂)) pairs. It gives a **linear, signed** view of which
features the tree leaned on at any point in time.

The Ridge confirms a useful detail flagged in the project notes: macro features
appear in the top 10 coefficient set, with sign flips at points where the regime
changed (rate hikes, VIX spikes). Useful as a sanity layer alongside SHAP.
"""
        ),
        code(
            """
coefs = pd.read_csv(PROCESSED / 'ridge_surrogate_coefs.csv')
months = coefs['month'].tolist()
features = [c for c in coefs.columns if c != 'month']
print(f'surrogate months: {len(months)} | features: {len(features)}')
coefs.head()
"""
        ),
        code(
            """
abs_mean = coefs[features].abs().mean().sort_values(ascending=False)
top_abs = abs_mean.head(10).index.tolist()

fig, ax = plt.subplots(figsize=(11, 5))
for f in top_abs:
    ax.plot(months, coefs[f], label=f, lw=1.4)

step = max(1, len(months) // 12)
ax.set_xticks(range(0, len(months), step))
ax.set_xticklabels([months[i] for i in range(0, len(months), step)], rotation=45, ha='right', fontsize=8)
ax.set_title('Incremental Ridge surrogate — top 10 coefficients over time')
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
## 4. Stability metrics — adjacent-month similarity

The monthly heatmap is striking but qualitative. A simple **cosine similarity** of
the per-feature SHAP vector between adjacent months gives a stability score: a
dip means the importance mix changed sharply.
"""
        ),
        code(
            """
from numpy.linalg import norm

months_sorted = sorted(monthly['month'].unique())
mat = monthly.pivot(index='month', columns='feature', values='mean_abs_shap').loc[months_sorted].fillna(0)

cos = []
for i in range(1, len(mat)):
    a, b = mat.iloc[i - 1].values, mat.iloc[i].values
    denom = (norm(a) * norm(b)) or 1e-9
    cos.append(float(np.dot(a, b) / denom))

fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(months_sorted[1:], cos, lw=1.5, color='#4c72b0')
ax.axhline(0.99, ls='--', color='gray', lw=0.8, label='0.99 reference')
ax.set_ylim(0.95, 1.005)
step = max(1, len(months_sorted) // 12)
ax.set_xticks(range(1, len(months_sorted), step))
ax.set_xticklabels([months_sorted[i] for i in range(1, len(months_sorted), step)], rotation=45, ha='right', fontsize=8)
ax.set_title('Cosine similarity of |SHAP| vector between adjacent months')
ax.set_ylabel('cosine')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
## 5. Takeaways

- **Importance drifts even when the model is frozen.** The killer chart shows
  the macro block heating up around 2016-2017, well before any retraining would
  have moved the LightGBM tree splits.
- **Per-decile SHAP is non-decorative.** The mix at decile 0 (lowest risk) is
  not the mix at decile 9 (highest risk); a single global SHAP plot hides that.
- **The Ridge surrogate is a cheap second opinion.** It's not as accurate as the
  tree, but its signed, monthly coefficients are easy to monitor and to alert on.

This closes the v2 explanation story. The front-end consumes the same CSVs at
`/insights` (`web/src/app/insights/page.tsx`) so the analyst-facing UI and the
notebook stay in lockstep.
"""
        ),
    ]
    write(nb, "13_adaptive_shap.ipynb")


if __name__ == "__main__":
    print("Building reproducibility notebooks...")
    build_nb12()
    build_nb13()
    print("Done.")
