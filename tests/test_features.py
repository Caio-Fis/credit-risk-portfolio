"""Testes para o módulo de feature engineering."""

from src.features.build_features import build_temporal_features


def test_build_temporal_features_adds_columns(small_credit_df):
    result = build_temporal_features(small_credit_df)
    new_cols = {
        "credit_income_ratio",
        "annuity_income_ratio",
        "age_years",
        "credit_annuity_ratio",
        "employed_years",
        "employed_to_age_ratio",
    }
    assert new_cols.issubset(set(result.columns)), (
        f"Colunas ausentes: {new_cols - set(result.columns)}"
    )


def test_build_temporal_features_no_nan_in_ratios(small_credit_df):
    result = build_temporal_features(small_credit_df)
    ratio_cols = ["credit_income_ratio", "annuity_income_ratio", "credit_annuity_ratio"]
    for col in ratio_cols:
        assert result[col].isna().sum() == 0, f"NaN encontrado em {col}"


def test_build_temporal_features_ext_source_mean(small_credit_df):
    result = build_temporal_features(small_credit_df)
    assert "ext_source_mean" in result.columns
    # ext_source_mean deve estar entre min e max individuais
    assert (
        result["ext_source_mean"]
        >= result[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1) - 1e-6
    ).all()


def test_build_temporal_features_preserves_shape(small_credit_df):
    result = build_temporal_features(small_credit_df)
    assert len(result) == len(small_credit_df), "Número de linhas alterado"
    assert result.shape[1] >= small_credit_df.shape[1], "Colunas foram removidas"


def test_age_years_positive(small_credit_df):
    result = build_temporal_features(small_credit_df)
    assert (result["age_years"] > 0).all(), "age_years deve ser positivo"
