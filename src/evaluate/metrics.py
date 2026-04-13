"""PD model evaluation metrics.

Target metrics:
- AUROC > 0.78
- KS > 0.35
- Brier Score < 0.15
- Calibration plot within 95% confidence band

Main functions:
- auroc: Area Under ROC Curve
- ks_stat: Kolmogorov-Smirnov statistic
- brier_score: Brier Score
- gini: Gini coefficient (= 2×AUROC − 1), credit industry standard
- hosmer_lemeshow: formal calibration test (H-L chi-squared)
- binomial_test_by_bucket: binomial test by PD bucket (Basel standard)
- calibration_plot: generates and saves the calibration plot
- bucket_calibration_plot: bucket calibration plot with Basel traffic light
- full_evaluation: runs all metrics and returns a dictionary
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from scipy.stats import binom
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.config import ARTIFACTS_DIR


def auroc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculates AUROC.

    Args:
        y_true: True labels (0/1).
        y_proba: Predicted probabilities.

    Returns:
        AUROC score in [0.5, 1.0].
    """
    score = roc_auc_score(y_true, y_proba)
    logger.info(f"AUROC: {score:.4f}")
    return score


def ks_stat(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculates KS statistic (Kolmogorov-Smirnov).

    KS = max(|CDF_default - CDF_non-default|)
    Measures separation between the score distributions of the two groups.

    Args:
        y_true: True labels (0/1).
        y_proba: Predicted probabilities.

    Returns:
        KS statistic in [0, 1].
    """
    df = pd.DataFrame({"y": y_true, "p": y_proba}).sort_values("p")
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    cumpos = (df["y"] == 1).cumsum() / n_pos
    cumneg = (df["y"] == 0).cumsum() / n_neg

    ks = float((cumpos - cumneg).abs().max())
    logger.info(f"KS: {ks:.4f}")
    return ks


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculates Brier Score (lower = better calibration).

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.

    Returns:
        Brier Score in [0, 1].
    """
    score = brier_score_loss(y_true, y_proba)
    logger.info(f"Brier Score: {score:.4f}")
    return score


def gini(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Gini coefficient: discrimination metric, credit industry standard.

    Gini = 2 × AUROC − 1, mapped to [−1, 1].
    Practical interpretation:
    - < 0.30: weak
    - 0.30–0.50: fair
    - 0.50–0.70: good (market standard for corporate credit)
    - > 0.70: excellent

    Args:
        y_true: True labels (0/1).
        y_proba: Predicted probabilities.

    Returns:
        Gini coefficient in [−1, 1].
    """
    score = 2 * roc_auc_score(y_true, y_proba) - 1
    logger.info(f"Gini: {score:.4f}")
    return score


def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float, pd.DataFrame]:
    """Hosmer-Lemeshow test: formally validates PD calibration.

    Groups observations into deciles of predicted probability and applies
    a chi-squared test comparing observed vs. expected defaults.

    H0: model is well calibrated (failing to reject is desirable).
    p-value > 0.05 → acceptable calibration.
    p-value < 0.05 → evidence of systematic miscalibration.

    Relevance for EL: if H-L rejects, predicted PDs are systematically
    wrong in some range — the EL calculated in currency units will not be reliable.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        n_bins: Number of groups (deciles by default).

    Returns:
        Tuple (hl_statistic, p_value, per_group_table).
    """
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")

    grouped = (
        df.groupby("bin", observed=True)
        .agg(n=("y", "count"), observed=("y", "sum"), expected=("p", "sum"))
        .reset_index()
    )

    # H-L statistic: sum((O - E)^2 / (E × (1 - E/n)))
    denom = grouped["expected"] * (1 - grouped["expected"] / grouped["n"])
    hl_stat = float(((grouped["observed"] - grouped["expected"]) ** 2 / denom).sum())
    df_freedom = len(grouped) - 2
    p_value = float(1 - stats.chi2.cdf(hl_stat, df_freedom))

    grouped["pd_mean"] = grouped["expected"] / grouped["n"]
    grouped["default_rate_obs"] = grouped["observed"] / grouped["n"]
    grouped["ratio_obs_exp"] = grouped["observed"] / grouped["expected"].clip(lower=1e-6)

    result = grouped.rename(columns={"bin": "faixa_pd", "n": "contratos"})
    result["faixa_pd"] = result["faixa_pd"].apply(
        lambda iv: f"({iv.left:.2%}, {iv.right:.2%}]"
    )

    logger.info(
        f"Hosmer-Lemeshow: H={hl_stat:.2f}, df={df_freedom}, p={p_value:.4f} "
        f"→ {'calibration OK' if p_value > 0.05 else 'WARNING: systematic miscalibration detected'}"
    )
    return hl_stat, p_value, result


def binomial_test_by_bucket(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    buckets: list[float] | None = None,
) -> pd.DataFrame:
    """Binomial test by PD bucket — Basel III traffic light approach.

    For each predicted PD bucket, tests whether the number of observed defaults
    is consistent with the mean predicted PD via a one-sided binomial test.

    Traffic light:
    - Green  (p > 0.05): acceptable calibration for this bucket
    - Yellow (0.01 < p ≤ 0.05): investigate — significant deviation
    - Red    (p ≤ 0.01): confirmed miscalibration — retraining required

    Used by banks to validate IRB (Internal Ratings-Based) models
    before the Central Bank (Basel III, art. 143 of Resolution 4.557).

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        buckets: PD bucket boundaries (e.g.: [0, 0.02, 0.05, 0.10, 0.20, 1.0]).

    Returns:
        DataFrame with test result per bucket.
    """
    if buckets is None:
        buckets = [0.0, 0.02, 0.05, 0.10, 0.20, 1.0]

    labels = [
        f"{a:.0%}–{b:.0%}" for a, b in zip(buckets[:-1], buckets[1:])
    ]

    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df["bucket"] = pd.cut(df["p"], bins=buckets, labels=labels, include_lowest=True)

    rows = []
    for bucket, group in df.groupby("bucket", observed=True):
        n = len(group)
        if n == 0:
            continue
        observed = int(group["y"].sum())
        pd_mean = float(group["p"].mean())
        expected = pd_mean * n

        # One-sided test: P(X ≥ observed | n, pd_mean)
        p_value = float(binom.sf(observed - 1, n, pd_mean))

        if p_value > 0.05:
            semaforo = "Verde"
        elif p_value > 0.01:
            semaforo = "Amarelo"
        else:
            semaforo = "Vermelho"

        rows.append(
            {
                "faixa_pd": str(bucket),
                "contratos": n,
                "defaults_observados": observed,
                "defaults_esperados": round(expected, 1),
                "pd_media_predita": round(pd_mean, 4),
                "taxa_obs": round(observed / n, 4),
                "p_value": round(p_value, 4),
                "semaforo": semaforo,
            }
        )

    result = pd.DataFrame(rows)
    n_verde = (result["semaforo"] == "Verde").sum()
    n_amarelo = (result["semaforo"] == "Amarelo").sum()
    n_vermelho = (result["semaforo"] == "Vermelho").sum()
    logger.info(
        f"Binomial test by bucket: Green={n_verde} | Yellow={n_amarelo} | Red={n_vermelho}"
    )
    return result


def bucket_calibration_plot(
    result: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """Bucket calibration plot with Basel traffic light.

    Args:
        result: DataFrame returned by binomial_test_by_bucket().
        save_path: If provided, saves the figure.

    Returns:
        Matplotlib figure.
    """
    colors = {"Verde": "seagreen", "Amarelo": "goldenrod", "Vermelho": "firebrick"}
    bar_colors = [colors[s] for s in result["semaforo"]]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(len(result))
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        result["taxa_obs"] * 100,
        width,
        label="Observed default rate (%)",
        color=bar_colors,
        alpha=0.85,
    )
    ax.bar(
        [i + width / 2 for i in x],
        result["pd_media_predita"] * 100,
        width,
        label="Mean predicted PD (%)",
        color="steelblue",
        alpha=0.6,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(result["faixa_pd"], rotation=20, ha="right")
    ax.set_ylabel("Default rate (%)")
    ax.set_title("Calibration by Bucket — Basel Binomial Test (traffic light = observed)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # p-value and traffic light annotations
    for i, row in result.iterrows():
        ax.text(
            i - width / 2,
            row["taxa_obs"] * 100 + 0.1,
            f"p={row['p_value']:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=colors[row["semaforo"]],
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Bucket calibration plot saved at {save_path}")

    return fig


def calibration_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    save_path: Path | None = None,
) -> plt.Figure:
    """Generates calibration plot with 95% confidence band.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        n_bins: Number of bins for grouping.
        save_path: If provided, saves the plot to this path.

    Returns:
        Matplotlib figure.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    # 95% confidence band via bootstrap
    n_boot = 200
    rng = np.random.default_rng(42)
    boot_curves = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pt, pp = calibration_curve(y_true[idx], y_proba[idx], n_bins=n_bins)
        # Interpolate to align points
        boot_curves.append(np.interp(prob_pred, pp, pt))

    boot_arr = np.array(boot_curves)
    lower = np.percentile(boot_arr, 2.5, axis=0)
    upper = np.percentile(boot_arr, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", lw=1.5)
    ax.fill_between(
        prob_pred, lower, upper, alpha=0.2, color="steelblue", label="95% CI"
    )
    ax.plot(prob_pred, prob_true, "o-", color="steelblue", label="PD Model (Isotonic)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Fraction of actual positives")
    ax.set_title("Calibration Curve — PD Model")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Calibration plot saved at {save_path}")

    return fig


def roc_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path | None = None,
) -> plt.Figure:
    """Generates ROC curve.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        save_path: If provided, saves the plot.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="PD LightGBM+Isotonic")
    ax.set_title(f"ROC Curve — AUROC={auroc(y_true, y_proba):.4f}")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC plot saved at {save_path}")

    return fig


def full_evaluation(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_plots: bool = True,
    artifact_dir: Path = ARTIFACTS_DIR,
) -> dict[str, float]:
    """Runs all metrics and optionally saves plots.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        save_plots: If True, saves plots to artifact_dir.
        artifact_dir: Directory to save plots.

    Returns:
        Dictionary with all metrics.
    """
    metrics = {
        "auroc": auroc(y_true, y_proba),
        "gini": gini(y_true, y_proba),
        "ks": ks_stat(y_true, y_proba),
        "brier_score": brier_score(y_true, y_proba),
        "avg_precision": average_precision_score(y_true, y_proba),
    }

    # Check targets
    targets = {"auroc": (0.78, ">="), "ks": (0.35, ">="), "brier_score": (0.15, "<=")}
    for metric, (target, direction) in targets.items():
        val = metrics[metric]
        ok = (val >= target) if direction == ">=" else (val <= target)
        status = "OK" if ok else "BELOW TARGET"
        logger.info(f"  {metric}: {val:.4f} (target {direction} {target}) → {status}")

    # Statistical calibration tests
    hl_stat, hl_pvalue, _ = hosmer_lemeshow_test(y_true, y_proba)
    metrics["hl_statistic"] = hl_stat
    metrics["hl_pvalue"] = hl_pvalue

    bucket_result = binomial_test_by_bucket(y_true, y_proba)
    metrics["buckets_verdes"] = int((bucket_result["semaforo"] == "Verde").sum())
    metrics["buckets_amarelos"] = int((bucket_result["semaforo"] == "Amarelo").sum())
    metrics["buckets_vermelhos"] = int((bucket_result["semaforo"] == "Vermelho").sum())

    if save_plots:
        calibration_plot(
            y_true, y_proba, save_path=artifact_dir / "calibration_plot.png"
        )
        roc_plot(y_true, y_proba, save_path=artifact_dir / "roc_curve.png")
        bucket_calibration_plot(
            bucket_result, save_path=artifact_dir / "bucket_calibration.png"
        )

    return metrics


def oos_comparison_plot(
    metrics_train: dict[str, float],
    metrics_oos: dict[str, float],
    save_path: Path | None = None,
) -> plt.Figure:
    """Bar chart comparing train vs. OOS hold-out metrics.

    Args:
        metrics_train: Metrics calculated on the training set.
        metrics_oos: Metrics calculated on the hold-out (OOS).
        save_path: If provided, saves the plot.

    Returns:
        Matplotlib figure.
    """
    keys = ["auroc", "ks", "brier_score"]
    labels = ["AUROC", "KS", "Brier Score"]
    train_vals = [metrics_train[k] for k in keys]
    oos_vals = [metrics_oos[k] for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_train = ax.bar(x - width / 2, train_vals, width, label="Train (in-sample)", color="steelblue", alpha=0.8)
    bars_oos = ax.bar(x + width / 2, oos_vals, width, label="Test OOS (hold-out)", color="coral", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Train vs. OOS — PD Model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in [*bars_train, *bars_oos]:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Train vs. OOS comparison saved at {save_path}")

    return fig


if __name__ == "__main__":
    from src.features.build_features import load_feature_store
    from src.models.pd_model import OOS_PATH, load_pd_model, predict_pd

    df = load_feature_store()
    model = load_pd_model()

    # OOS evaluation (hold-out) — honest metric
    if OOS_PATH.exists():
        oos = pd.read_parquet(OOS_PATH)
        y_true_oos = oos["y_true"].values
        y_pred_oos = oos["y_pred"].values
        logger.info(f"OOS evaluation — {len(y_true_oos):,} samples (20% hold-out)")
        metrics_oos = full_evaluation(
            y_true_oos,
            y_pred_oos,
            save_plots=True,
            artifact_dir=ARTIFACTS_DIR,
        )

        # Train vs. OOS comparison
        y_pred_all = predict_pd(model, df)
        y_true_all = df["TARGET"].values
        metrics_train = {
            "auroc": auroc(y_true_all, y_pred_all),
            "ks": ks_stat(y_true_all, y_pred_all),
            "brier_score": brier_score(y_true_all, y_pred_all),
            "avg_precision": average_precision_score(y_true_all, y_pred_all),
        }
        overfit_gap = metrics_train["auroc"] - metrics_oos["auroc"]
        logger.info(f"AUROC train-OOS gap: {overfit_gap:+.4f} ({'OK' if overfit_gap < 0.05 else 'WARNING: possible overfitting'})")

        oos_comparison_plot(
            metrics_train,
            metrics_oos,
            save_path=ARTIFACTS_DIR / "train_vs_oos.png",
        )
    else:
        # Fallback: evaluate on full dataset (in-sample)
        logger.warning("OOS predictions not found — using full dataset (in-sample).")
        y_pred = predict_pd(model, df)
        y_true = df["TARGET"].values
        full_evaluation(y_true, y_pred)
