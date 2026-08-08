# =============================================================================
# COMPONENTE
# Tarjetas de Indicadores
# =============================================================================

import streamlit as st


def metric_card(title, value, delta=None):

    st.metric(
        label=title,
        value=value,
        delta=delta
    )