"""Métricas de avaliação do modelo PD.

Métricas alvo:
- AUROC > 0.78
- KS > 0.35
- Brier Score < 0.15
- Calibration plot dentro de banda de confiança de 95%

Funções principais:
- auroc: Area Under ROC Curve
- ks_stat: Kolmogorov-Smirnov statistic
- brier_score: Brier Score
- calibration_plot: gera e salva o gráfico de calibração
- full_evaluation: roda todas as métricas e retorna dicionário
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.config import ARTIFACTS_DIR


def auroc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calcula AUROC.

    Args:
        y_true: Labels verdadeiros (0/1).
        y_proba: Probabilidades preditas.

    Returns:
        AUROC score em [0.5, 1.0].
    """
    score = roc_auc_score(y_true, y_proba)
    logger.info(f"AUROC: {score:.4f}")
    return score


def ks_stat(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calcula estatística KS (Kolmogorov-Smirnov).

    KS = max(|CDF_default - CDF_adimplente|)
    Mede a separação entre as distribuições de score dos dois grupos.

    Args:
        y_true: Labels verdadeiros (0/1).
        y_proba: Probabilidades preditas.

    Returns:
        KS statistic em [0, 1].
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
    """Calcula Brier Score (menor = melhor calibração).

    Args:
        y_true: Labels verdadeiros.
        y_proba: Probabilidades preditas.

    Returns:
        Brier Score em [0, 1].
    """
    score = brier_score_loss(y_true, y_proba)
    logger.info(f"Brier Score: {score:.4f}")
    return score


def calibration_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    save_path: Path | None = None,
) -> plt.Figure:
    """Gera gráfico de calibração com banda de confiança 95%.

    Args:
        y_true: Labels verdadeiros.
        y_proba: Probabilidades preditas.
        n_bins: Número de bins para agrupamento.
        save_path: Se fornecido, salva o plot neste caminho.

    Returns:
        Figura matplotlib.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    # Banda de confiança 95% via bootstrap
    n_boot = 200
    rng = np.random.default_rng(42)
    boot_curves = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pt, pp = calibration_curve(y_true[idx], y_proba[idx], n_bins=n_bins)
        # Interpola para alinhar pontos
        boot_curves.append(np.interp(prob_pred, pp, pt))

    boot_arr = np.array(boot_curves)
    lower = np.percentile(boot_arr, 2.5, axis=0)
    upper = np.percentile(boot_arr, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Calibração perfeita", lw=1.5)
    ax.fill_between(
        prob_pred, lower, upper, alpha=0.2, color="steelblue", label="IC 95%"
    )
    ax.plot(prob_pred, prob_true, "o-", color="steelblue", label="Modelo PD (Platt)")
    ax.set_xlabel("Probabilidade predita")
    ax.set_ylabel("Fração de positivos reais")
    ax.set_title("Curva de Calibração — Modelo PD")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Calibration plot salvo em {save_path}")

    return fig


def roc_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path | None = None,
) -> plt.Figure:
    """Gera curva ROC.

    Args:
        y_true: Labels verdadeiros.
        y_proba: Probabilidades preditas.
        save_path: Se fornecido, salva o plot.

    Returns:
        Figura matplotlib.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="PD LightGBM+Platt")
    ax.set_title(f"Curva ROC — AUROC={auroc(y_true, y_proba):.4f}")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC plot salvo em {save_path}")

    return fig


def full_evaluation(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_plots: bool = True,
    artifact_dir: Path = ARTIFACTS_DIR,
) -> dict[str, float]:
    """Roda todas as métricas e opcionalmente salva os plots.

    Args:
        y_true: Labels verdadeiros.
        y_proba: Probabilidades preditas.
        save_plots: Se True, salva gráficos em artifact_dir.
        artifact_dir: Diretório para salvar gráficos.

    Returns:
        Dicionário com todas as métricas.
    """
    metrics = {
        "auroc": auroc(y_true, y_proba),
        "ks": ks_stat(y_true, y_proba),
        "brier_score": brier_score(y_true, y_proba),
        "avg_precision": average_precision_score(y_true, y_proba),
    }

    # Verifica metas
    targets = {"auroc": (0.78, ">="), "ks": (0.35, ">="), "brier_score": (0.15, "<=")}
    for metric, (target, direction) in targets.items():
        val = metrics[metric]
        ok = (val >= target) if direction == ">=" else (val <= target)
        status = "OK" if ok else "ABAIXO DA META"
        logger.info(f"  {metric}: {val:.4f} (meta {direction} {target}) → {status}")

    if save_plots:
        calibration_plot(
            y_true, y_proba, save_path=artifact_dir / "calibration_plot.png"
        )
        roc_plot(y_true, y_proba, save_path=artifact_dir / "roc_curve.png")

    return metrics


def oos_comparison_plot(
    metrics_train: dict[str, float],
    metrics_oos: dict[str, float],
    save_path: Path | None = None,
) -> plt.Figure:
    """Gráfico de barras comparando métricas train vs. OOS hold-out.

    Args:
        metrics_train: Métricas calculadas no conjunto de treino.
        metrics_oos: Métricas calculadas no hold-out (OOS).
        save_path: Se fornecido, salva o plot.

    Returns:
        Figura matplotlib.
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
    ax.set_title("Train vs. OOS — Modelo PD")
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
        logger.info(f"Comparativo train vs. OOS salvo em {save_path}")

    return fig


if __name__ == "__main__":
    from src.features.build_features import load_feature_store
    from src.models.pd_model import OOS_PATH, load_pd_model, predict_pd

    df = load_feature_store()
    model = load_pd_model()

    # Avaliação OOS (hold-out) — métrica honesta
    if OOS_PATH.exists():
        oos = pd.read_parquet(OOS_PATH)
        y_true_oos = oos["y_true"].values
        y_pred_oos = oos["y_pred"].values
        logger.info(f"Avaliação OOS — {len(y_true_oos):,} amostras (hold-out 20%)")
        metrics_oos = full_evaluation(
            y_true_oos,
            y_pred_oos,
            save_plots=True,
            artifact_dir=ARTIFACTS_DIR,
        )

        # Comparativo train vs. OOS
        y_pred_all = predict_pd(model, df)
        y_true_all = df["TARGET"].values
        metrics_train = {
            "auroc": auroc(y_true_all, y_pred_all),
            "ks": ks_stat(y_true_all, y_pred_all),
            "brier_score": brier_score(y_true_all, y_pred_all),
            "avg_precision": average_precision_score(y_true_all, y_pred_all),
        }
        overfit_gap = metrics_train["auroc"] - metrics_oos["auroc"]
        logger.info(f"Gap AUROC train-OOS: {overfit_gap:+.4f} ({'OK' if overfit_gap < 0.05 else 'ATENÇÃO: possível overfitting'})")

        oos_comparison_plot(
            metrics_train,
            metrics_oos,
            save_path=ARTIFACTS_DIR / "train_vs_oos.png",
        )
    else:
        # Fallback: avalia no dataset completo (in-sample)
        logger.warning("OOS predictions não encontradas — usando dataset completo (in-sample).")
        y_pred = predict_pd(model, df)
        y_true = df["TARGET"].values
        full_evaluation(y_true, y_pred)
