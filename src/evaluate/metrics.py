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

    result = grouped.rename(columns={"bin": "pd_range", "n": "contracts"})
    result["pd_range"] = result["pd_range"].apply(
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
            traffic_light = "Green"
        elif p_value > 0.01:
            traffic_light = "Yellow"
        else:
            traffic_light = "Red"

        rows.append(
            {
                "pd_range": str(bucket),
                "contracts": n,
                "observed_defaults": observed,
                "expected_defaults": round(expected, 1),
                "mean_predicted_pd": round(pd_mean, 4),
                "observed_rate": round(observed / n, 4),
                "p_value": round(p_value, 4),
                "traffic_light": traffic_light,
            }
        )

    result = pd.DataFrame(rows)
    n_green = (result["traffic_light"] == "Green").sum()
    n_yellow = (result["traffic_light"] == "Yellow").sum()
    n_red = (result["traffic_light"] == "Red").sum()
    logger.info(
        f"Binomial test by bucket: Green={n_green} | Yellow={n_yellow} | Red={n_red}"
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
    colors = {"Green": "seagreen", "Yellow": "goldenrod", "Red": "firebrick"}
    bar_colors = [colors[s] for s in result["traffic_light"]]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(len(result))
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        result["observed_rate"] * 100,
        width,
        label="Observed default rate (%)",
        color=bar_colors,
        alpha=0.85,
    )
    ax.bar(
        [i + width / 2 for i in x],
        result["mean_predicted_pd"] * 100,
        width,
        label="Mean predicted PD (%)",
        color="steelblue",
        alpha=0.6,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(result["pd_range"], rotation=20, ha="right")
    ax.set_ylabel("Default rate (%)")
    ax.set_title("Calibration by Bucket — Basel Binomial Test (traffic light = observed)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # p-value and traffic light annotations
    for i, row in result.iterrows():
        ax.text(
            i - width / 2,
            row["observed_rate"] * 100 + 0.1,
            f"p={row['p_value']:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=colors[row["traffic_light"]],
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
    metrics["green_buckets"] = int((bucket_result["traffic_light"] == "Green").sum())
    metrics["yellow_buckets"] = int((bucket_result["traffic_light"] == "Yellow").sum())
    metrics["red_buckets"] = int((bucket_result["traffic_light"] == "Red").sum())

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


def rolling_oot_evaluation(
    df: pd.DataFrame,
    train_params: dict | None = None,
    categorical_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
    date_col: str = "issue_d",
    target_col: str = "target",
    freq: str = "Y",
    start_year: int = 2009,
    end_year: int = 2017,
    use_calibration: bool = True,
) -> pd.DataFrame:
    """Rolling out-of-time evaluation: retrain on [start..t-1], evaluate on t.

    For each period t (yearly by default), trains a LightGBM on all data
    issued strictly before t and evaluates on data issued during t. This is
    the honest measure of how a static model degrades over real time.

    Returns a DataFrame with one row per period, including AUROC, KS, Brier,
    calibration slope, and base rate.
    """
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LinearRegression

    if train_params is None:
        train_params = {
            "n_estimators": 800,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 200,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "objective": "binary",
            "metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in {target_col, date_col}]
    if categorical_cols:
        for c in categorical_cols:
            if c in df.columns:
                df[c] = df[c].astype("category")

    rows = []
    for year in range(start_year, end_year + 1):
        train_mask = df[date_col].dt.year < year
        test_mask = df[date_col].dt.year == year
        if train_mask.sum() < 5000 or test_mask.sum() < 500:
            logger.warning(f"  Year {year}: skipped (train={train_mask.sum()}, test={test_mask.sum()})")
            continue

        # Take last 6 months of train as inner-val for calibration + early stop
        train = df[train_mask]
        cutoff = train[date_col].max() - pd.DateOffset(months=6)
        inner_train = train[train[date_col] <= cutoff]
        inner_val = train[train[date_col] > cutoff]
        if len(inner_val) < 500:
            inner_train, inner_val = train, train.sample(frac=0.1, random_state=42)

        X_tr, y_tr = inner_train[feature_cols], inner_train[target_col].astype("int8")
        X_va, y_va = inner_val[feature_cols], inner_val[target_col].astype("int8")
        X_te, y_te = df[test_mask][feature_cols], df[test_mask][target_col].astype("int8")

        model = LGBMClassifier(**train_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[early_stopping(40), log_evaluation(period=0)],
            categorical_feature=categorical_cols or "auto",
        )
        p_va_raw = model.predict_proba(X_va)[:, 1]
        p_te_raw = model.predict_proba(X_te)[:, 1]

        if use_calibration:
            cal = IsotonicRegression(out_of_bounds="clip").fit(p_va_raw, y_va.values)
            p_te = cal.transform(p_te_raw)
        else:
            p_te = p_te_raw

        # Calibration slope: regress y on logit(p)
        eps = 1e-6
        logit = np.log(np.clip(p_te, eps, 1 - eps) / (1 - np.clip(p_te, eps, 1 - eps)))
        slope = float(LinearRegression().fit(logit.reshape(-1, 1), y_te.values).coef_[0])

        rows.append({
            "year": year,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "base_rate_train": float(y_tr.mean()),
            "base_rate_test": float(y_te.mean()),
            "auroc": float(roc_auc_score(y_te, p_te)),
            "ks": float(_ks_from_arrays(y_te.values, p_te)),
            "brier": float(brier_score_loss(y_te, p_te)),
            "calib_slope": slope,
            "best_iteration": int(model.best_iteration_),
        })
        logger.info(
            f"  Year {year}: AUROC={rows[-1]['auroc']:.4f}, KS={rows[-1]['ks']:.4f}, "
            f"Brier={rows[-1]['brier']:.4f}, slope={rows[-1]['calib_slope']:.3f}"
        )

    return pd.DataFrame(rows)


def frozen_oot_evaluation(
    df: pd.DataFrame,
    train_until_year: int,
    test_start_year: int,
    test_end_year: int,
    train_params: dict | None = None,
    categorical_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
    date_col: str = "issue_d",
    target_col: str = "target",
) -> pd.DataFrame:
    """Train once on data ≤ `train_until_year`, then evaluate on each
    subsequent year (frozen model). This is the canonical test of how a
    static model degrades when never retrained — the motivation for the
    online challenger.
    """
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LinearRegression

    if train_params is None:
        train_params = {
            "n_estimators": 1500,
            "learning_rate": 0.04,
            "num_leaves": 63,
            "min_child_samples": 200,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "objective": "binary",
            "metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in {target_col, date_col}]
    if categorical_cols:
        for c in categorical_cols:
            if c in df.columns:
                df[c] = df[c].astype("category")

    train_full = df[df[date_col].dt.year <= train_until_year]
    cutoff = train_full[date_col].max() - pd.DateOffset(months=6)
    inner_train = train_full[train_full[date_col] <= cutoff]
    inner_val = train_full[train_full[date_col] > cutoff]
    if len(inner_val) < 500:
        inner_train, inner_val = train_full, train_full.sample(frac=0.1, random_state=42)

    logger.info(
        f"FROZEN training — ≤ {train_until_year}: {len(inner_train):,} train + {len(inner_val):,} inner-val"
    )
    model = LGBMClassifier(**train_params)
    model.fit(
        inner_train[feature_cols], inner_train[target_col].astype("int8"),
        eval_set=[(inner_val[feature_cols], inner_val[target_col].astype("int8"))],
        eval_metric="auc",
        callbacks=[early_stopping(50), log_evaluation(period=0)],
        categorical_feature=categorical_cols or "auto",
    )
    p_inner_val = model.predict_proba(inner_val[feature_cols])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        p_inner_val, inner_val[target_col].astype("int8").values
    )

    eps = 1e-6
    rows = []
    for year in range(test_start_year, test_end_year + 1):
        test = df[df[date_col].dt.year == year]
        if len(test) < 500:
            continue
        y_te = test[target_col].astype("int8").values
        p_raw = model.predict_proba(test[feature_cols])[:, 1]
        p = calibrator.transform(p_raw)
        logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
        slope = float(LinearRegression().fit(logit.reshape(-1, 1), y_te).coef_[0])
        rows.append({
            "year": year,
            "n_test": int(len(y_te)),
            "base_rate_test": float(y_te.mean()),
            "auroc": float(roc_auc_score(y_te, p)),
            "ks": float(_ks_from_arrays(y_te, p)),
            "brier": float(brier_score_loss(y_te, p)),
            "calib_slope": slope,
        })
        logger.info(
            f"  Year {year}: AUROC={rows[-1]['auroc']:.4f}, KS={rows[-1]['ks']:.4f}, "
            f"Brier={rows[-1]['brier']:.4f}, slope={rows[-1]['calib_slope']:.3f}"
        )
    return pd.DataFrame(rows)


def rolling_vs_frozen_plot(
    rolling: pd.DataFrame,
    frozen: pd.DataFrame,
    train_until_year: int,
    save_path: Path | None = None,
) -> plt.Figure:
    """Side-by-side plot: yearly-retrained vs frozen model."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, col, label, lo_hi in zip(
        axes,
        ["auroc", "ks", "brier"],
        ["AUROC", "KS", "Brier score"],
        [(0.50, 0.80), (0.0, 0.4), (0.10, 0.25)],
    ):
        ax.plot(rolling["year"], rolling[col], "o-", color="steelblue", lw=2,
                label="Retrained yearly")
        ax.plot(frozen["year"], frozen[col], "s--", color="firebrick", lw=2,
                label=f"Frozen at {train_until_year}")
        ax.axvline(train_until_year + 0.5, color="black", alpha=0.3, lw=1, ls=":")
        ax.set_title(label)
        ax.set_xlabel("Year (held-out)")
        ax.set_ylim(*lo_hi)
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Static PD model — retrained yearly vs frozen — LendingClub 2010-2017")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Rolling vs frozen plot saved at {save_path}")
    return fig


def _ks_from_arrays(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    df = pd.DataFrame({"y": y_true, "p": y_proba}).sort_values("p")
    n_pos = int(y_true.sum()) or 1
    n_neg = int(len(y_true) - y_true.sum()) or 1
    cumpos = (df["y"] == 1).cumsum() / n_pos
    cumneg = (df["y"] == 0).cumsum() / n_neg
    return float((cumpos - cumneg).abs().max())


def rolling_oot_plot(results: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Plots AUROC, KS, Brier over the rolling OOT years."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, col, label, lo_hi in zip(
        axes,
        ["auroc", "ks", "brier"],
        ["AUROC", "KS", "Brier score"],
        [(0.5, 0.85), (0.0, 0.5), (0.10, 0.25)],
    ):
        ax.plot(results["year"], results[col], "o-", color="steelblue", lw=2)
        ax.set_title(label)
        ax.set_xlabel("Year (held-out)")
        ax.set_ylim(*lo_hi)
        ax.grid(alpha=0.3)
        # Highlight 2008–2010 stress
        ax.axvspan(2008, 2010, color="firebrick", alpha=0.08, label="2008 crisis")
        if ax is axes[0]:
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Rolling Out-of-Time evaluation — static PD model retrained each year")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Rolling OOT plot saved at {save_path}")
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
