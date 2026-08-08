# =============================================================================
# DASHBOARD PRINCIPAL
# =============================================================================

import streamlit as st

from geoai.components.metrics import metric_card
from geoai.services.dataset_service import dataset_service
from geoai.services.graph_service import graph_service
from geoai.services.model_service import model_service
from geoai.services.forecast_service import forecast_service
from geoai.services.system_service import system_service

def show_dashboard():

    st.subheader("Centro de Control GeoAI")

    st.write(
        "Estado general de la Plataforma Inteligente GeoAI."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    metric_card(
        "Municipios",
        dataset_service.get_num_municipios()
    )

    with col2:
        metric_card(
            "GraphData",
            "13"
        )

    with col3:
        metric_card(
            "Modelo Oficial",
            "GraphSAGE"
        )

    with col4:
        metric_card(
            "Sistema",
            "Operativo"
        )

    st.divider()

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        metric_card(
            "Variables",
            "36"
        )

    with col6:
        metric_card(
            "Forecast",
            "Disponible"
        )

    with col7:
        metric_card(
            "Biblioteca IA",
            "20"
        )

    with col8:
        metric_card(
            "Versión",
            "1.0"
        )