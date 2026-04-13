"""Score Contextual — mesmo cliente, produtos diferentes ao vivo."""

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.contextual.data_generator import generate_dataset
from src.contextual.interaction_model import score_by_context, train_contextual

st.set_page_config(page_title="Score Contextual", layout="wide")
st.title("Score Contextual")
st.caption(
    "Demonstração ao vivo: o mesmo cliente tem risco radicalmente diferente "
    "dependendo do produto e do prazo solicitado."
)


@st.cache_resource(show_spinner="Treinando modelo contextual...")
def load_contextual_model():
    df = generate_dataset(n=3000, seed=42)
    model, feature_names = train_contextual(df, seed=42)
    return model, feature_names


model_ctx, feature_names = load_contextual_model()

# ---------------------------------------------------------------------------
# Perfil do cliente
# ---------------------------------------------------------------------------
st.subheader("Configure o Perfil do Cliente")
col_a, col_b, col_c = st.columns(3)

with col_a:
    score_fin = st.slider("Score financeiro (0=fraco, 1=forte)", 0.0, 1.0, 0.5, 0.05)
with col_b:
    faturamento = st.number_input(
        "Faturamento anual (R$)", 50_000, 10_000_000, 500_000, 50_000
    )
with col_c:
    has_collateral = int(st.checkbox("Tem garantia real"))

ead = st.number_input("Valor da operação (EAD, R$)", 5_000, 2_000_000, 100_000, 5_000)

client_profile = {
    "score_financeiro": score_fin,
    "idade_empresa_anos": 5,
    "faturamento_anual": faturamento,
    "has_collateral": has_collateral,
    "ead": ead,
}

# ---------------------------------------------------------------------------
# Cálculo ao vivo
# ---------------------------------------------------------------------------
scores = score_by_context(client_profile, model_ctx, feature_names)
scores["el_R$"] = scores["el_contextual"]
scores["pd_pct"] = scores["pd_contextual"] * 100

# Heatmap PD
st.subheader("PD Contextual — Produto × Prazo (mesmo cliente)")
pivot_pd = scores.pivot(
    index="tenor_months", columns="product_type", values="pd_contextual"
)

fig_heat = px.imshow(
    pivot_pd,
    text_auto=".3f",
    color_continuous_scale="RdYlGn_r",
    zmin=0,
    zmax=min(0.5, pivot_pd.values.max() * 1.5),
    labels={"x": "Produto", "y": "Prazo (meses)", "color": "PD"},
    title="PD por Produto × Prazo (score único ignoraria essas diferenças)",
)
st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------------------------
# Gráfico de EL por produto
# ---------------------------------------------------------------------------
st.subheader("Expected Loss (R$) por Produto e Prazo")
fig_el = px.line(
    scores.sort_values("tenor_months"),
    x="tenor_months",
    y="el_R$",
    color="product_type",
    markers=True,
    labels={
        "tenor_months": "Prazo (meses)",
        "el_R$": "Expected Loss (R$)",
        "product_type": "Produto",
    },
    title="Como o EL evolui com o prazo — diferente por produto",
)
st.plotly_chart(fig_el, use_container_width=True)

# ---------------------------------------------------------------------------
# Conclusão
# ---------------------------------------------------------------------------
min_el = scores["el_R$"].min()
max_el = scores["el_R$"].max()
ratio = max_el / (min_el + 0.01)

st.divider()
st.subheader("O argumento central")
st.info(f"""
Para este cliente com score financeiro **{score_fin:.2f}** e EAD de **R$ {ead:,.0f}**:

- Expected Loss **mínimo**: R$ {min_el:,.0f} (operação de menor risco)
- Expected Loss **máximo**: R$ {max_el:,.0f} (operação de maior risco)
- **Razão máx/mín**: {ratio:.1f}x

Usar score único significa tratar todos esses casos como equivalentes.
O modelo contextual **diferencia {ratio:.1f}x** o risco entre produtos/prazos do mesmo cliente.
""")
