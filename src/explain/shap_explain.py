"""Explainability via SHAP values.

Generates waterfall plots per individual contract and global feature importance.

Main functions:
- compute_shap: calculates SHAP values for a set of predictions
- waterfall_plot: generates SHAP waterfall for a specific contract
- top_features: returns the N most globally important features
- summary_plot: global importance summary plot
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger


def compute_shap(
    model,
    X: pd.DataFrame,
    background_sample: int = 100,
    max_explain_samples: int = 2000,
    seed: int = 42,
) -> shap.Explanation:
    """Calculates SHAP values for the dataset X.

    For CalibratedClassifierCV, extracts the base LightGBM estimator
    and uses TreeExplainer (more efficient than KernelExplainer).

    Args:
        model: Trained model (CalibratedClassifierCV or LGBMClassifier).
        X: Features DataFrame.
        background_sample: Background sample size for KernelExplainer.
        max_explain_samples: Row limit for SHAP. If X > limit, uses uniform
            random sampling (preserves temporal/risk distribution).
            TreeExplainer is O(n × features × depth) — 307K rows takes ~10min;
            2000 rows takes ~2s.
        seed: Seed for sampling.

    Returns:
        shap.Explanation with values for the positive class (default).
    """
    from sklearn.calibration import CalibratedClassifierCV

    # Subsample when X is too large for SHAP to be interactive.
    # TreeExplainer computes SHAP point-by-point (no inter-sample interaction),
    # so sampling is equivalent to choosing which contracts to explain.
    # Global importance = mean(|SHAP|) converges with n~500 (CLT); 2000 is conservative.
    # We use uniform random sampling — no class stratification because
    # the synthetic contextual dataset is balanced (~43% default).
    # If used in production with imbalanced dataset (<5% default), pass y=target
    # and implement explicit stratified sampling.
    n_original = len(X)
    if n_original > max_explain_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_original, size=max_explain_samples, replace=False)
        idx.sort()  # preserve original order
        X = X.iloc[idx].reset_index(drop=True)
        logger.info(
            f"SHAP: {n_original:,} → {max_explain_samples:,} rows. "
            f"Individual values: identical. Global importance: error ≈ σ/√{max_explain_samples}."
        )

    # Extract base estimator from calibration wrapper
    if isinstance(model, CalibratedClassifierCV):
        base_estimator = model.calibrated_classifiers_[0].estimator
    else:
        base_estimator = model

    logger.info(f"Computing SHAP values (TreeExplainer) for {len(X)} samples...")

    try:
        explainer = shap.TreeExplainer(base_estimator)
        shap_values = explainer(X)
        # For binary classification, take the positive class (index 1)
        if shap_values.values.ndim == 3:
            explanation = shap.Explanation(
                values=shap_values.values[:, :, 1],
                base_values=shap_values.base_values[:, 1]
                if shap_values.base_values.ndim > 1
                else shap_values.base_values,
                data=shap_values.data,
                feature_names=list(X.columns),
            )
        else:
            explanation = shap_values
    except Exception as e:
        logger.warning(
            f"TreeExplainer failed ({e}). Falling back to KernelExplainer (slower)."
        )
        background = shap.sample(X, background_sample)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_matrix = explainer.shap_values(X)
        if isinstance(shap_matrix, list):
            shap_matrix = shap_matrix[1]
        explanation = shap.Explanation(
            values=shap_matrix,
            base_values=np.full(
                len(X),
                explainer.expected_value[1]
                if hasattr(explainer.expected_value, "__len__")
                else explainer.expected_value,
            ),
            data=X.values,
            feature_names=list(X.columns),
        )

    logger.success(f"SHAP values computed: {explanation.values.shape}")
    return explanation


def waterfall_plot(
    explanation: shap.Explanation,
    idx: int,
    max_display: int = 15,
    save_path: Path | None = None,
) -> plt.Figure:
    """Generates SHAP waterfall plot for a specific contract.

    Args:
        explanation: shap.Explanation for the full dataset.
        idx: Contract index in the DataFrame.
        max_display: Maximum number of features to display.
        save_path: If provided, saves the figure to this path.

    Returns:
        Matplotlib figure.
    """
    shap.plots.waterfall(explanation[idx], max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Waterfall plot saved at {save_path}")

    return fig


def top_features(
    explanation: shap.Explanation,
    n: int = 10,
) -> pd.DataFrame:
    """Returns the N most important features by global SHAP importance.

    Global importance = mean(|SHAP values|) per feature.

    Args:
        explanation: shap.Explanation for the dataset.
        n: Number of features to return.

    Returns:
        DataFrame with columns: feature, mean_abs_shap (sorted descending).
    """
    mean_abs = np.abs(explanation.values).mean(axis=0)
    feature_names = explanation.feature_names or [f"f{i}" for i in range(len(mean_abs))]

    df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    logger.info(f"Top {n} features by SHAP importance:\n{df.to_string(index=False)}")
    return df


def summary_plot(
    explanation: shap.Explanation,
    max_display: int = 20,
    save_path: Path | None = None,
) -> plt.Figure:
    """Generates global importance summary plot.

    Args:
        explanation: shap.Explanation for the dataset.
        max_display: Maximum number of features to display.
        save_path: If provided, saves the figure.

    Returns:
        Matplotlib figure.
    """
    shap.summary_plot(
        explanation.values,
        explanation.data,
        feature_names=explanation.feature_names,
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Summary plot saved at {save_path}")

    return fig
