# =============================================================================
# PLATAFORMA INTELIGENTE GEOAI
# Punto de Entrada
# =============================================================================

import streamlit as st

from geoai.components.header import show_header
from geoai.components.sidebar import show_sidebar
from geoai.components.theme import apply_theme
from geoai.dashboard.dashboard import show_dashboard

# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================

st.set_page_config(
    page_title="GeoAI",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# COMPONENTES
# =============================================================================

apply_theme()

show_header()

page = show_sidebar()

if page == "🏠 Inicio":

    show_dashboard()

else:

    st.info(f"Módulo: {page}")