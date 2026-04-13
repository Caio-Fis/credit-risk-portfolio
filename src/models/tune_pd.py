"""PD model hyperparameter fine-tuning via Optuna.

Strategy:
- Objective: AUROC via 5-fold CV on the train split (80% of data)
- The OOS split (20%) remains untouched during the search — evaluated only at the end
- Focus: regularisation parameters, since the train/OOS gap is ~0.13 (overfitting)
- 50 Bayesian trials (TPE sampler), pruning with MedianPruner
- Result: new LGBM_PARAMS to replace in pd_model.py

Usage:
    uv run python -m src.models.tune_pd
    make tune
"""

import mlflow
import mlflow.sklearn
import optuna
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.config import MLFLOW_EXPERIMENT_PD, MLFLOW_TRACKING_URI
from src.features.build_features import load_feature_store
from src.models.pd_model import TEST_SIZE, _prepare_features

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 50
CV_FOLDS = 5
SEED = 42


def _objective(trial: optuna.Trial, X_train, y_train) -> float:
    """Optuna objective: CV AUROC on the train split."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "random_state": SEED,
        "n_jobs": 2,
        "verbose": -1,
    }

    lgbm = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scores = cross_val_score(lgbm, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
    return scores.mean()


def tune_pd_hyperparams(n_trials: int = N_TRIALS) -> dict:
    """Runs Bayesian hyperparameter search and returns the best params.

    Args:
        n_trials: Number of Optuna trials.

    Returns:
        Dictionary with best hyperparameters for LGBMClassifier.
    """
    df = load_feature_store()
    X, _ = _prepare_features(df)
    y = df["TARGET"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    logger.info(
        f"Tuning — train: {len(y_train):,} | OOS: {len(y_test):,} | "
        f"default rate: {y.mean():.2%}"
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(
        lambda trial: _objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    best_cv = study.best_value
    logger.success(f"Best CV AUROC (train): {best_cv:.4f}")
    logger.info(f"Best params:\n{best}")

    # Evaluate with calibration on OOS — honest estimate
    best_lgbm = LGBMClassifier(**{**best, "random_state": SEED, "n_jobs": 2, "verbose": -1})
    calibrated = CalibratedClassifierCV(best_lgbm, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)
    y_pred_oos = calibrated.predict_proba(X_test)[:, 1]
    auroc_oos = roc_auc_score(y_test, y_pred_oos)
    logger.success(f"OOS AUROC with best params: {auroc_oos:.4f} (target ≥ 0.78)")

    # Log result to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_PD)
    with mlflow.start_run(run_name="optuna_tuning"):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_params(best)
        mlflow.log_metric("best_cv_auroc", best_cv)
        mlflow.log_metric("auroc_oos_tuned", auroc_oos)

    return best, auroc_oos


if __name__ == "__main__":
    best_params, auroc_oos = tune_pd_hyperparams()

    print("\n" + "=" * 60)
    print("REPLACE LGBM_PARAMS in src/models/pd_model.py with:")
    print("=" * 60)
    for k, v in sorted(best_params.items()):
        if isinstance(v, float):
            print(f'    "{k}": {v:.6f},')
        else:
            print(f'    "{k}": {v},')
    print(f"\nExpected OOS AUROC: {auroc_oos:.4f}")
    if auroc_oos >= 0.78:
        print("✓ Target 0.78 reached")
    else:
        print(f"✗ Still below target — gap: {0.78 - auroc_oos:.4f}")
