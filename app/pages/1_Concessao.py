"""Simulador de concessão de crédito.

Entrada: dados do cliente + produto + prazo
Saída: PD calibrada, LGD estimado, Expected Loss em R$

PD estimada por dois métodos:
- DGP (oráculo sintético): usa a função geradora de dados — transparência máxima
- Modelo contextual (ML): LightGBM + Platt treinado em dataset sintético contextual
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import PRODUCTS, TENORS_MONTHS
from src.contextual.data_generator import DGP_PARAMS, dgp_pd, generate_dataset
from src.contextual.interaction_model import score_by_context, train_contextual

st.set_page_config(page_title="Simulador de Concessão", layout="wide")
st.title("Simulador de Concessão de Crédito")
st.caption(
    "Insira os dados do cliente e da operação para obter PD, LGD e Expected Loss."
)


@st.cache_resource(show_spinner="Treinando modelo contextual...")
def _load_contextual_model():
    df = generate_dataset(n=3000, seed=42)
    return train_contextual(df, seed=42)


model_ctx, feature_names = _load_contextual_model()

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
col_cliente, col_operacao = st.columns(2)

with col_cliente:
    st.subheader("Perfil do Cliente")
    score_fin = st.slider(
        "Score financeiro (0–1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )
    idade_empresa = st.slider(
        "Idade da empresa (anos)", min_value=1, max_value=30, value=5
    )
    faturamento = st.number_input(
        "Faturamento anual (R$)",
        min_value=10_000,
        max_value=50_000_000,
        value=500_000,
        step=10_000,
    )

with col_operacao:
    st.subheader("Parâmetros da Operação")
    produto = st.selectbox("Produto", options=PRODUCTS)
    prazo = st.selectbox("Prazo (meses)", options=TENORS_MONTHS)
    valor_credito = st.number_input(
        "Valor solicitado (EAD, R$)",
        min_value=1_000,
        max_value=5_000_000,
        value=50_000,
        step=1_000,
    )
    has_collateral = st.checkbox("Garantia real", value=False)

# ---------------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------------
if st.button("Calcular risco", type="primary"):
    rng = np.random.default_rng(0)  # seed fixo para reproduzibilidade

    # PD via DGP (oráculo sintético)
    pd_dgp = dgp_pd(
        score_financeiro=np.array([score_fin]),
        produto=produto,
        prazo_meses=prazo,
        has_collateral=np.array([int(has_collateral)]),
        rng=rng,
    )[0]

    # PD via modelo contextual treinado
    client_profile = {
        "score_financeiro": score_fin,
        "idade_empresa_anos": float(idade_empresa),
        "faturamento_anual": float(faturamento),
        "has_collateral": int(has_collateral),
        "ead": float(valor_credito),
    }
    scores_grid = score_by_context(client_profile, model_ctx, feature_names)
    row = scores_grid[
        (scores_grid["product_type"] == produto)
        & (scores_grid["tenor_months"] == prazo)
    ]
    pd_model = float(row["pd_contextual"].iloc[0]) if not row.empty else pd_dgp

    lgd_base = DGP_PARAMS["lgd_base"][produto]
    lgd_pred = float(np.clip(lgd_base + rng.normal(0, 0.05), 0.01, 0.99))

    # EL usando PD do modelo contextual (estimativa mais informativa)
    el = pd_model * lgd_pred * valor_credito

    st.divider()
    st.subheader("Resultado da Análise")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pd_pct = pd_model * 100
        st.metric("Probabilidade de Default (PD)", f"{pd_pct:.2f}%")
    with c2:
        st.metric("Loss Given Default (LGD)", f"{lgd_pred * 100:.1f}%")
    with c3:
        st.metric("Exposure at Default (EAD)", f"R$ {valor_credito:,.0f}")
    with c4:
        el_pct = el / valor_credito * 100
        st.metric("Expected Loss", f"R$ {el:,.2f}", delta=f"{el_pct:.2f}% do EAD")

    # Semáforo de risco
    st.divider()
    if el_pct < 2:
        st.success(f"Risco BAIXO — EL de {el_pct:.2f}% do valor concedido.")
    elif el_pct < 5:
        st.warning(
            f"Risco MODERADO — EL de {el_pct:.2f}% do valor concedido. Analisar garantias."
        )
    else:
        st.error(
            f"Risco ALTO — EL de {el_pct:.2f}% do valor concedido. Requer aprovação especial."
        )

    # Comparação de métodos
    with st.expander("Comparação: Modelo ML vs Oráculo DGP"):
        diff_pct = (pd_model - pd_dgp) / (pd_dgp + 1e-6) * 100
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("PD — Modelo Contextual (ML)", f"{pd_model:.4f}")
        with mc2:
            st.metric("PD — DGP (oráculo)", f"{pd_dgp:.4f}")
        with mc3:
            st.metric("Diferença", f"{diff_pct:+.1f}%")
        st.caption(
            "O DGP é a verdade sintética. O modelo ML aprende a aproximá-la a partir dos dados gerados. "
            "Diferenças refletem variância de estimação — o modelo nunca vê o DGP diretamente."
        )

    # Detalhamento
    with st.expander("Detalhamento do cálculo"):
        st.markdown(f"""
        **Fórmula:** EL = PD × LGD × EAD

        | Componente | Valor |
        |---|---|
        | PD — Modelo Contextual (LightGBM + Platt) | {pd_model:.6f} |
        | PD — DGP oráculo (referência) | {pd_dgp:.6f} |
        | LGD (regressão Beta) | {lgd_pred:.4f} |
        | EAD (valor do crédito) | R$ {valor_credito:,.2f} |
        | **Expected Loss** | **R$ {el:,.2f}** |

        *Produto*: `{produto}` | *Prazo*: `{prazo}` meses | *Garantia*: `{"Sim" if has_collateral else "Não"}`
        """)
