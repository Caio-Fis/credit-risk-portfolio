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
- gini: coeficiente de Gini (= 2×AUROC − 1), padrão da indústria de crédito
- hosmer_lemeshow: teste formal de calibração (H-L chi-squared)
- binomial_test_by_bucket: teste binomial por faixa de PD (padrão Basel)
- calibration_plot: gera e salva o gráfico de calibração
- bucket_calibration_plot: gráfico de calibração por bucket com semáforo Basel
- full_evaluation: roda todas as métricas e retorna dicionário
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


def gini(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Coeficiente de Gini: discriminação no padrão da indústria de crédito.

    Gini = 2 × AUROC − 1, mapeado para [−1, 1].
    Interpretação prática:
    - < 0.30: fraco
    - 0.30–0.50: razoável
    - 0.50–0.70: bom (padrão de mercado para crédito PJ)
    - > 0.70: excelente

    Args:
        y_true: Labels verdadeiros (0/1).
        y_proba: Probabilidades preditas.

    Returns:
        Coeficiente de Gini em [−1, 1].
    """
    score = 2 * roc_auc_score(y_true, y_proba) - 1
    logger.info(f"Gini: {score:.4f}")
    return score


def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float, pd.DataFrame]:
    """Teste de Hosmer-Lemeshow: valida calibração da PD formalmente.

    Agrupa observações em decis de probabilidade predita e aplica
    teste chi-squared comparando defaults observados vs. esperados.

    H0: modelo está bem calibrado (não rejeitar é desejável).
    p-value > 0.05 → calibração aceitável.
    p-value < 0.05 → evidência de miscalibração sistemática.

    Relevância para EL: se H-L rejeita, as PDs preditas estão sistematicamente
    erradas em alguma faixa — o EL calculado em reais não será confiável.

    Args:
        y_true: Labels verdadeiros.
        y_proba: Probabilidades preditas.
        n_bins: Número de grupos (decis por padrão).

    Returns:
        Tupla (hl_statistic, p_value, tabela_por_grupo).
    """
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")

    grouped = (
        df.groupby("bin", observed=True)
        .agg(n=("y", "count"), observed=("y", "sum"), expected=("p", "sum"))
        .reset_index()
    )

    # Estatística H-L: sum((O - E)^2 / (E × (1 - E/n)))
    denom = grouped["expected"] * (1 - grouped["expected"] / grouped["n"])
    hl_stat = float(((grouped["observed"] - grouped["expected"]) ** 2 / denom).sum())
    df_freedom = len(grouped) - 2
    p_value = float(1 - stats.chi2.cdf(hl_stat, df_freedom))

    grouped["pd_mean"] = grouped["expected"] / grouped["n"]
    grouped["default_rate_obs"] = grouped["observed"] / grouped["n"]
    grouped["ratio_obs_exp"] = grouped["observed"] / grouped["expected"].clip(lower=1e-6)

    result = grouped.rename(columns={"bin": "faixa_pd", "n": "contratos"})

    logger.info(
        f"Hosmer-Lemeshow: H={hl_stat:.2f}, df={df_freedom}, p={p_value:.4f} "
        f"→ {'calibração OK' if p_value > 0.05 else 'ATENÇÃO: miscalibração detectada'}"
    )
    return hl_stat, p_value, result


def binomial_test_by_bucket(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    buckets: list[float] | None = None,
) -> pd.DataFrame:
    """Teste binomial por faixa de PD — abordagem Basel III traffic light.

    Para cada bucket de PD predita, testa se o número de defaults observados
    é compatível com a PD média predita via teste binomial unilateral.

    Semáforo:
    - Verde  (p > 0.05): calibração aceitável para essa faixa
    - Amarelo (0.01 < p ≤ 0.05): investigar — desvio significativo
    - Vermelho (p ≤ 0.01): miscalibração confirmada — retreino necessário

    Usado por bancos para validar modelos IRB (Internal Ratings-Based)
    perante o Banco Central (Basileia III, art. 143 da Resolução 4.557).

    Args:
        y_true: Labels verdadeiros.
        y_proba: Probabilidades preditas.
        buckets: Limites das faixas de PD (ex: [0, 0.02, 0.05, 0.10, 0.20, 1.0]).

    Returns:
        DataFrame com resultado do teste por faixa.
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

        # Teste unilateral: P(X ≥ observado | n, pd_mean)
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
        f"Teste binomial por bucket: Verde={n_verde} | Amarelo={n_amarelo} | Vermelho={n_vermelho}"
    )
    return result


def bucket_calibration_plot(
    result: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """Gráfico de calibração por bucket com semáforo Basel.

    Args:
        result: DataFrame retornado por binomial_test_by_bucket().
        save_path: Se fornecido, salva a figura.

    Returns:
        Figura matplotlib.
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
        label="Default rate observada (%)",
        color=bar_colors,
        alpha=0.85,
    )
    ax.bar(
        [i + width / 2 for i in x],
        result["pd_media_predita"] * 100,
        width,
        label="PD média predita (%)",
        color="steelblue",
        alpha=0.6,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(result["faixa_pd"], rotation=20, ha="right")
    ax.set_ylabel("Taxa de default (%)")
    ax.set_title("Calibração por Bucket — Teste Binomial Basel (semáforo = observado)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Anotações de p-value e semáforo
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
        logger.info(f"Bucket calibration plot salvo em {save_path}")

    return fig


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
        "gini": gini(y_true, y_proba),
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

    # Testes estatísticos de calibração
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
