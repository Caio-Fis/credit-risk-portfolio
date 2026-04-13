"""Home — Narrativa do projeto e visão geral."""

import streamlit as st

st.set_page_config(
    page_title="Credit Risk Portfolio",
    page_icon="📊",
    layout="wide",
)

st.title("Credit Risk Portfolio")
st.subheader("Pipeline end-to-end de risco de crédito para MPE")

st.markdown("""
---

## O problema

Instituições financeiras que operam crédito para micro e pequenas empresas enfrentam um paradoxo:
precisam decidir limite, prazo e taxa **no momento da concessão**, mas os dados mais relevantes
sobre o tomador — comportamento de pagamento, sazonalidade, dependência de clientes — só aparecem
meses depois.

O **score único** agrava o problema ao colapsar toda essa complexidade num número entre 0 e 1000
que ignora o contexto da operação.

> Um score 650 pode ser adequado para capital de giro de 30 dias e completamente inadequado
> para um investimento de 48 meses. **O mesmo CNPJ, riscos radicalmente diferentes.**

---

## A solução

Pipeline end-to-end que transforma dados cadastrais, transacionais e macroeconômicos em:

| Saída | Descrição |
|---|---|
| **PD calibrada** | Probabilidade de default — não ranking, mas probabilidade em escala real |
| **LGD estimado** | Quanto se perde se o default ocorrer |
| **Expected Loss** | EL = PD × LGD × EAD em reais por contrato |
| **Score contextual** | Mesmo cliente avaliado diferente para produtos diferentes |
| **Alertas de drift** | PSI automatizado — aviso quando a população muda |

---

## Módulos
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    **Módulo 1 — Pipeline Core**

    LightGBM + Platt scaling para PD.
    Regressão Beta para LGD.
    Expected Loss em R$ por contrato.
    SHAP por decisão.
    """)

with col2:
    st.warning("""
    **Módulo 2 — Early Warning**

    PSI por feature com alertas automáticos.
    Análise de safra por coorte.
    Queda de score > 50pts em 30 dias.
    Gatilhos comportamentais.
    """)

with col3:
    st.success("""
    **Módulo 3 — Score Contextual**

    Dataset sintético com DGP documentado.
    Produto e prazo como features, não filtros.
    Demonstração quantitativa da limitação
    do score único.
    """)

st.markdown("""
---

## Navegue pelas páginas

Use o menu à esquerda para explorar cada módulo:

- **Concessão**: Simulador de análise de crédito
- **Portfolio**: Dashboard de carteira com KPIs
- **Early Warning**: Alertas de deterioração
- **Explicabilidade**: SHAP waterfall por contrato
- **Score Contextual**: Mesmo cliente, produtos diferentes

---

## Decisões técnicas

**PD e LGD como modelos separados**: o custo de errar em cada dimensão é diferente.
PD alta com LGD baixa (garantia real) tem expected loss menor que PD média com LGD alta.

**LightGBM com calibração de Platt**: gradient boosting produz bom ranking mas probabilidades
mal calibradas. Para calcular EL em reais, você precisa de probabilidade em escala real.

**Score único não é errado — é insuficiente**: ele vira uma feature entre muitas no modelo
contextual.
""")
