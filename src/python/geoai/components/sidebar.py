# =============================================================================
# COMPONENTE
# Menú Lateral
# =============================================================================

import streamlit as st


def show_sidebar():

    with st.sidebar:

        st.title("🌎 GeoAI")

        st.markdown("---")

        page = st.radio(

            "Navegación",

            [

                "🏠 Inicio",

                "📊 Dataset Científico",

                "🕸️ GraphData",

                "📈 Benchmark",

                "🧠 Modelo Oficial",

                "📑 Evaluación",

                "🔮 Forecasting",

                "🤖 Agente IA",

                "📄 Reportes"

            ]

        )

        st.markdown("---")

        st.caption("Versión 1.0")

    return page