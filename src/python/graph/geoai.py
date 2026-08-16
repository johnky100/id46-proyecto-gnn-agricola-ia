# graph-06_geoai.py

# BLOQUE 1. Importaciones --------------------------------------------------
## Objetivo: Importar las dependencias necesarias para construir la Plataforma Inteligente GeoAI utilizando el modelo oficial, los resultados del forecasting espacio-temporal, la infraestructura geográfica, los componentes de visualización, los sistemas de reporte científico y el asistente inteligente del proyecto.
### Producto:
# - Librerías cargadas correctamente.
### Responde:
# ¿Qué dependencias requiere la Plataforma Inteligente GeoAI para integrar los resultados científicos, generar mapas, construir el Dashboard, elaborar reportes y asistir la toma de decisiones?

# Funciones del sistema
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

# Librerías científicas
import numpy as np
import pandas as pd
import torch

# Librerías geoespaciales
import geopandas as gpd

# Librerías de visualización
import folium
import plotly.express as px
import plotly.graph_objects as go

# Configuración oficial del proyecto
from src.python.config.config_project import (
    PROJECT_SEED,
    DASHBOARD_CONFIG,
    REPORT_CONFIG,
    API_CONFIG,
    AI_AGENT_CONFIG,
    GEOAI_OUTPUTS
)

# Rutas oficiales
from src.python.config.paths import (
    DATASET_FILE,
    GRAPH_DATA_DIR,
    FORECAST_REPORTS_DIR
)

# Utilidades
from src.python.utils.results import (
    build_benchmark_result
)

# Configuración del entorno
warnings.filterwarnings("ignore")

print("-" * 80)

# BLOQUE 2. Configuración --------------------------------------------------
## Objetivo: Construir y validar la configuración oficial de la Plataforma Inteligente GeoAI a partir de la configuración central del proyecto, garantizando la reproducibilidad, consistencia e integridad del proceso de visualización, análisis, generación de reportes y asistencia inteligente.
### Producto:
# - GEOAI_CONFIG
### Responde:
# ¿La Plataforma Inteligente GeoAI dispone de una configuración oficial, reproducible y consistente para integrar los resultados científicos del proyecto?

def build_geoai_configuration() -> dict:
    """
    Construye y valida la configuración oficial de la Plataforma GeoAI.

    Returns
    -------
    dict
        Configuración oficial de la Plataforma GeoAI.
    """

    # Construcción ---------------------------------------------------------
    geoai_config = {
        "platform_name": "geoai_platform",
        "platform_version": "1.0",
        "prediction_source": "official_forecasting",
        "dashboard_config": DASHBOARD_CONFIG,
        "report_config": REPORT_CONFIG,
        "api_config": API_CONFIG,
        "ai_agent_config": AI_AGENT_CONFIG,
        "geoai_outputs": GEOAI_OUTPUTS,
        "random_state": PROJECT_SEED
    }

    # Validación -----------------------------------------------------------
    required_keys = [
        "platform_name",
        "platform_version",
        "prediction_source",
        "dashboard_config",
        "report_config",
        "api_config",
        "ai_agent_config",
        "geoai_outputs",
        "random_state"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in geoai_config
    ]

    if missing_keys:
        raise RuntimeError(
            "La configuración oficial de la Plataforma GeoAI está incompleta: "
            f"{missing_keys}"
        )

    if not isinstance(geoai_config["platform_name"], str):
        raise TypeError("platform_name debe ser una cadena.")

    if not isinstance(geoai_config["platform_version"], str):
        raise TypeError("platform_version debe ser una cadena.")

    if not isinstance(geoai_config["prediction_source"], str):
        raise TypeError("prediction_source debe ser una cadena.")

    if not isinstance(geoai_config["dashboard_config"], dict):
        raise TypeError("dashboard_config debe ser un diccionario.")

    if not isinstance(geoai_config["report_config"], dict):
        raise TypeError("report_config debe ser un diccionario.")

    if not isinstance(geoai_config["api_config"], dict):
        raise TypeError("api_config debe ser un diccionario.")

    if not isinstance(geoai_config["ai_agent_config"], dict):
        raise TypeError("ai_agent_config debe ser un diccionario.")

    if not isinstance(geoai_config["geoai_outputs"], dict):
        raise TypeError("geoai_outputs debe ser un diccionario.")

    if not isinstance(geoai_config["random_state"], int):
        raise TypeError("random_state debe ser un entero.")

    if not geoai_config["platform_name"].strip():
        raise ValueError("platform_name está vacío.")

    if not geoai_config["platform_version"].strip():
        raise ValueError("platform_version está vacío.")

    if not geoai_config["prediction_source"].strip():
        raise ValueError("prediction_source está vacío.")

    return geoai_config


GEOAI_CONFIG = build_geoai_configuration()

print("-" * 80)

# BLOQUE 3. Recursos Científicos --------------------------------------------
## Objetivo: Recuperar y validar los activos científicos oficiales generados durante todo el pipeline para garantizar la disponibilidad, integridad, trazabilidad y consistencia de la información utilizada por la Plataforma Inteligente GeoAI.
### Producto:
# - GEOAI_ASSETS
### Responde:
# ¿Los activos científicos generados por el pipeline están disponibles, son consistentes y pueden ser consumidos por la Plataforma Inteligente GeoAI?

def load_geoai_assets(geoai_config: dict) -> dict:
    """
    Recupera y valida los activos científicos oficiales de la Plataforma GeoAI.

    Parameters
    ----------
    geoai_config : dict
        Configuración oficial de la Plataforma GeoAI.

    Returns
    -------
    dict
        Activos científicos oficiales.
    """
    # Validación -----------------------------------------------------------
    if not isinstance(geoai_config, dict):
        raise TypeError("geoai_config debe ser un diccionario.")
    required_keys=[
        "prediction_source",
        "dashboard_config",
        "report_config",
        "api_config",
        "ai_agent_config"
    ]
    missing_keys=[key for key in required_keys if key not in geoai_config]
    if missing_keys:
        raise RuntimeError(f"La configuración oficial está incompleta: {missing_keys}")
    # Recuperación ---------------------------------------------------------
    geoai_assets={
        "dataset":{
            "scientific_dataset":SCIENTIFIC_DATASET
        },
        "graph":{
            "graph_data":GRAPH_DATA
        },
        "benchmark":{
            "benchmark_result":BENCHMARK_RESULT
        },
        "training":{
            "train_result":TRAIN_RESULT
        },
        "evaluation":{
            "evaluation_result":EVALUATION_RESULT,
            "evaluation_metrics":EVALUATION_METRICS
        },
        "forecast":{
            "forecast_result":FORECAST_RESULT,
            "forecast_summary":FORECAST_SUMMARY,
            "forecast_explainability":FORECAST_EXPLAINABILITY,
            "forecast_output":FORECAST_OUTPUT
        }
    }
    # Validación -----------------------------------------------------------
    required_assets=[
        "dataset",
        "graph",
        "benchmark",
        "training",
        "evaluation",
        "forecast"
    ]
    missing_assets=[asset for asset in required_assets if asset not in geoai_assets]
    if missing_assets:
        raise RuntimeError(f"Faltan activos científicos: {missing_assets}")
    for group_name,group in geoai_assets.items():
        if not isinstance(group,dict):
            raise TypeError(f"El grupo '{group_name}' debe ser un diccionario.")
        if not group:
            raise RuntimeError(f"El grupo '{group_name}' está vacío.")
        for asset_name,asset_value in group.items():
            if asset_value is None:
                raise RuntimeError(f"El activo científico '{asset_name}' es None.")
    # Retorno --------------------------------------------------------------
    return geoai_assets

GEOAI_ASSETS=load_geoai_assets(
    geoai_config=GEOAI_CONFIG
)

print("-"*80)

# BLOQUE 4. Serie Espacio-Temporal ------------------------------------------
## Objetivo: Construir y validar la serie espacio-temporal oficial integrando los datos históricos y los resultados del forecasting para disponer de una estructura científica única utilizada por todos los componentes de la Plataforma Inteligente GeoAI.
### Producto:
# - SPATIOTEMPORAL_SERIES
### Responde:
# ¿La Plataforma GeoAI dispone de una serie espacio-temporal oficial que integre de forma consistente la información histórica y proyectada?

def build_spatiotemporal_series(geoai_assets: dict)->pd.DataFrame:
    """
    Construye la serie espacio-temporal oficial de la Plataforma GeoAI.

    Parameters
    ----------
    geoai_assets : dict
        Activos científicos oficiales.

    Returns
    -------
    pd.DataFrame
        Serie espacio-temporal oficial.
    """
    # Validación -----------------------------------------------------------
    if not isinstance(geoai_assets,dict):
        raise TypeError("geoai_assets debe ser un diccionario.")
    required_groups=["forecast"]
    missing_groups=[group for group in required_groups if group not in geoai_assets]
    if missing_groups:
        raise RuntimeError(f"Faltan grupos de activos científicos: {missing_groups}")
    forecast_assets=geoai_assets["forecast"]
    required_assets=["forecast_result"]
    missing_assets=[asset for asset in required_assets if asset not in forecast_assets]
    if missing_assets:
        raise RuntimeError(f"Faltan activos del forecasting: {missing_assets}")
    # Recuperación ---------------------------------------------------------
    historical_data=forecast_assets["forecast_result"]["historical_dataset"].copy()
    forecast_data=forecast_assets["forecast_result"]["forecast_dataset"].copy()
    # Construcción ---------------------------------------------------------
    historical_data["data_type"]="historical"
    forecast_data["data_type"]="forecast"
    spatiotemporal_series=pd.concat([historical_data,forecast_data],ignore_index=True)
    spatiotemporal_series=spatiotemporal_series.sort_values(by=["cod_mpio","anio"]).reset_index(drop=True)
    # Validación -----------------------------------------------------------
    if spatiotemporal_series.empty:
        raise RuntimeError("La serie espacio-temporal está vacía.")
    required_columns=["cod_mpio","anio","data_type"]
    missing_columns=[column for column in required_columns if column not in spatiotemporal_series.columns]
    if missing_columns:
        raise RuntimeError(f"Faltan columnas obligatorias: {missing_columns}")
    if spatiotemporal_series.duplicated(subset=["cod_mpio","anio"]).any():
        raise RuntimeError("Existen registros municipio-año duplicados.")
    # Retorno --------------------------------------------------------------
    return spatiotemporal_series

SPATIOTEMPORAL_SERIES=build_spatiotemporal_series(
    geoai_assets=GEOAI_ASSETS
)

print("-"*80)

# BLOQUE 5. Analítica GeoAI ------------------------------------------------
## Objetivo: Construir y validar la analítica oficial de la Plataforma GeoAI mediante la generación de indicadores científicos, métricas e información resumida utilizada por los componentes de visualización, análisis territorial, reportes y asistencia inteligente.
### Producto:
# - GEOAI_ANALYTICS
### Responde:
# ¿La Plataforma GeoAI dispone de indicadores científicos consistentes para soportar el análisis del comportamiento histórico y proyectado del territorio?

def build_geoai_analytics(spatiotemporal_series: pd.DataFrame,geoai_assets: dict,geoai_config: dict)->dict:
    """
    Construye la analítica oficial de la Plataforma GeoAI.

    Parameters
    ----------
    spatiotemporal_series : pd.DataFrame
        Serie espacio-temporal oficial.

    geoai_assets : dict
        Activos científicos oficiales.

    geoai_config : dict
        Configuración oficial de la Plataforma.

    Returns
    -------
    dict
        Analítica oficial de la Plataforma GeoAI.
    """
    # Validación -----------------------------------------------------------
    if not isinstance(spatiotemporal_series,pd.DataFrame):
        raise TypeError("spatiotemporal_series debe ser un DataFrame.")
    if spatiotemporal_series.empty:
        raise ValueError("spatiotemporal_series está vacío.")
    if not isinstance(geoai_assets,dict):
        raise TypeError("geoai_assets debe ser un diccionario.")
    if not isinstance(geoai_config,dict):
        raise TypeError("geoai_config debe ser un diccionario.")
    # Recuperación ---------------------------------------------------------
    forecast_assets=geoai_assets["forecast"]
    # Construcción ---------------------------------------------------------
    geoai_analytics={
        "series":spatiotemporal_series,
        "configuration":geoai_config["dashboard_config"],
        "model":forecast_assets["forecast_result"]["best_model"],
        "municipalities":spatiotemporal_series["cod_mpio"].nunique(),
        "departments":spatiotemporal_series["cod_depto"].nunique() if "cod_depto" in spatiotemporal_series.columns else None,
        "historical_years":sorted(spatiotemporal_series.loc[spatiotemporal_series["data_type"]=="historical","anio"].unique().tolist()),
        "forecast_years":sorted(spatiotemporal_series.loc[spatiotemporal_series["data_type"]=="forecast","anio"].unique().tolist()),
        "historical_records":int(spatiotemporal_series["data_type"].eq("historical").sum()),
        "forecast_records":int(spatiotemporal_series["data_type"].eq("forecast").sum()),
        "total_records":len(spatiotemporal_series),
        "variables":len(spatiotemporal_series.columns),
        "created_at":datetime.now()
    }
    # Validación -----------------------------------------------------------
    required_keys=[
        "series",
        "configuration",
        "model",
        "municipalities",
        "departments",
        "historical_years",
        "forecast_years",
        "historical_records",
        "forecast_records",
        "total_records",
        "variables",
        "created_at"
    ]
    missing_keys=[key for key in required_keys if key not in geoai_analytics]
    if missing_keys:
        raise RuntimeError(f"La analítica oficial está incompleta: {missing_keys}")
    if geoai_analytics["municipalities"]<=0:
        raise RuntimeError("No existen municipios disponibles.")
    if geoai_analytics["total_records"]<=0:
        raise RuntimeError("No existen registros disponibles.")
    # Retorno --------------------------------------------------------------
    return geoai_analytics

GEOAI_ANALYTICS=build_geoai_analytics(
    spatiotemporal_series=SPATIOTEMPORAL_SERIES,
    geoai_assets=GEOAI_ASSETS,
    geoai_config=GEOAI_CONFIG
)

print("-"*80)

# BLOQUE 6. Capa Geoespacial GeoAI -----------------------------------------
## Objetivo: Construir y validar la capa geoespacial oficial de la Plataforma GeoAI integrando la serie espacio-temporal, la información territorial y los resultados científicos para soportar el análisis espacial, la visualización geográfica y la toma de decisiones.
### Producto:
# - GEOAI_GEOSPATIAL
### Responde:
# ¿La Plataforma GeoAI dispone de una capa geoespacial consistente para representar y analizar el comportamiento histórico y proyectado del territorio?

def build_geoai_geospatial(spatiotemporal_series: pd.DataFrame,geoai_assets: dict,geoai_config: dict)->dict:
    """
    Construye la capa geoespacial oficial de la Plataforma GeoAI.

    Parameters
    ----------
    spatiotemporal_series : pd.DataFrame
        Serie espacio-temporal oficial.

    geoai_assets : dict
        Activos científicos oficiales.

    geoai_config : dict
        Configuración oficial de la Plataforma.

    Returns
    -------
    dict
        Capa geoespacial oficial.
    """
    # Validación -----------------------------------------------------------
    if not isinstance(spatiotemporal_series,pd.DataFrame):
        raise TypeError("spatiotemporal_series debe ser un DataFrame.")
    if spatiotemporal_series.empty:
        raise ValueError("spatiotemporal_series está vacío.")
    if not isinstance(geoai_assets,dict):
        raise TypeError("geoai_assets debe ser un diccionario.")
    if not isinstance(geoai_config,dict):
        raise TypeError("geoai_config debe ser un diccionario.")
    required_columns=["cod_mpio","anio","latitud","longitud","data_type"]
    missing_columns=[column for column in required_columns if column not in spatiotemporal_series.columns]
    if missing_columns:
        raise RuntimeError(f"Faltan columnas geoespaciales obligatorias: {missing_columns}")
    # Recuperación ---------------------------------------------------------
    forecast_assets=geoai_assets["forecast"]
    # Construcción ---------------------------------------------------------
    geoai_geospatial={
        "series":spatiotemporal_series,
        "forecast_summary":forecast_assets["forecast_summary"],
        "configuration":geoai_config["dashboard_config"],
        "municipalities":spatiotemporal_series["cod_mpio"].nunique(),
        "departments":spatiotemporal_series["cod_depto"].nunique() if "cod_depto" in spatiotemporal_series.columns else None,
        "historical_years":sorted(spatiotemporal_series.loc[spatiotemporal_series["data_type"]=="historical","anio"].unique().tolist()),
        "forecast_years":sorted(spatiotemporal_series.loc[spatiotemporal_series["data_type"]=="forecast","anio"].unique().tolist()),
        "historical_records":int(spatiotemporal_series["data_type"].eq("historical").sum()),
        "forecast_records":int(spatiotemporal_series["data_type"].eq("forecast").sum()),
        "geometry_columns":["latitud","longitud"],
        "coordinate_system":"EPSG:4326",
        "created_at":datetime.now()
    }
    # Validación -----------------------------------------------------------
    required_keys=[
        "series",
        "forecast_summary",
        "configuration",
        "municipalities",
        "departments",
        "historical_years",
        "forecast_years",
        "historical_records",
        "forecast_records",
        "geometry_columns",
        "coordinate_system",
        "created_at"
    ]
    missing_keys=[key for key in required_keys if key not in geoai_geospatial]
    if missing_keys:
        raise RuntimeError(f"La capa geoespacial está incompleta: {missing_keys}")
    if geoai_geospatial["municipalities"]<=0:
        raise RuntimeError("No existen municipios disponibles.")
    if len(geoai_geospatial["historical_years"])==0:
        raise RuntimeError("No existen años históricos.")
    # Retorno --------------------------------------------------------------
    return geoai_geospatial

GEOAI_GEOSPATIAL=build_geoai_geospatial(
    spatiotemporal_series=SPATIOTEMPORAL_SERIES,
    geoai_assets=GEOAI_ASSETS,
    geoai_config=GEOAI_CONFIG
)

print("-"*80)

# BLOQUE 7. Base de Conocimiento GeoAI -------------------------------------
## Objetivo: Construir y validar la Base de Conocimiento oficial de la Plataforma GeoAI integrando los activos científicos, la documentación técnica, la metodología, los resultados del pipeline y las fuentes documentales utilizadas por el asistente inteligente.
### Producto:
# - GEOAI_KNOWLEDGE_BASE
### Responde:
# ¿La Plataforma GeoAI dispone de una Base de Conocimiento científica consistente para soportar los procesos de búsqueda, recuperación de información y razonamiento del asistente inteligente?

def build_geoai_knowledge_base(geoai_assets:dict,geoai_config:dict)->dict:
    """
    Construye la Base de Conocimiento oficial de la Plataforma GeoAI.

    Parameters
    ----------
    geoai_assets : dict
        Activos científicos oficiales.

    geoai_config : dict
        Configuración oficial de la Plataforma.

    Returns
    -------
    dict
        Base de Conocimiento oficial.
    """
    # Validación -----------------------------------------------------------
    if not isinstance(geoai_assets,dict):
        raise TypeError("geoai_assets debe ser un diccionario.")
    if not isinstance(geoai_config,dict):
        raise TypeError("geoai_config debe ser un diccionario.")
    required_groups=["dataset","graph","benchmark","training","evaluation","forecast"]
    missing_groups=[group for group in required_groups if group not in geoai_assets]
    if missing_groups:
        raise RuntimeError(f"Faltan activos científicos: {missing_groups}")
    # Recuperación ---------------------------------------------------------
    dataset_assets=geoai_assets["dataset"]
    graph_assets=geoai_assets["graph"]
    benchmark_assets=geoai_assets["benchmark"]
    training_assets=geoai_assets["training"]
    evaluation_assets=geoai_assets["evaluation"]
    forecast_assets=geoai_assets["forecast"]
    # Construcción ---------------------------------------------------------
    geoai_knowledge_base={
        "dataset":dataset_assets,
        "graph":graph_assets,
        "benchmark":benchmark_assets,
        "training":training_assets,
        "evaluation":evaluation_assets,
        "forecast":forecast_assets,
        "scientific_documents":{},
        "technical_documents":{},
        "methodology":{},
        "metadata":{
            "platform":geoai_config["platform_name"],
            "version":geoai_config["platform_version"],
            "created_at":datetime.now()
        }
    }
    # Validación -----------------------------------------------------------
    required_keys=[
        "dataset",
        "graph",
        "benchmark",
        "training",
        "evaluation",
        "forecast",
        "scientific_documents",
        "technical_documents",
        "methodology",
        "metadata"
    ]
    missing_keys=[key for key in required_keys if key not in geoai_knowledge_base]
    if missing_keys:
        raise RuntimeError(f"La Base de Conocimiento está incompleta: {missing_keys}")
    # Retorno --------------------------------------------------------------
    return geoai_knowledge_base

GEOAI_KNOWLEDGE_BASE=build_geoai_knowledge_base(
    geoai_assets=GEOAI_ASSETS,
    geoai_config=GEOAI_CONFIG
)

print("-"*80)

# BLOQUE 8. Asistente Inteligente GeoAI ------------------------------------
## Objetivo: Construir y validar el Asistente Inteligente oficial de la Plataforma GeoAI integrando la Base de Conocimiento, la serie espacio-temporal, la analítica científica y la capa geoespacial para soportar procesos de consulta, razonamiento, interpretación y generación de recomendaciones territoriales.
### Producto:
# - GEOAI_ASSISTANT
### Responde:
# ¿El Asistente Inteligente dispone de toda la información científica necesaria para responder consultas y generar recomendaciones confiables?

def build_geoai_assistant(spatiotemporal_series:pd.DataFrame,geoai_analytics:dict,geoai_geospatial:dict,geoai_knowledge_base:dict,geoai_config:dict)->dict:
    """
    Construye el Asistente Inteligente oficial de la Plataforma GeoAI.

    Parameters
    ----------
    spatiotemporal_series : pd.DataFrame
        Serie espacio-temporal oficial.

    geoai_analytics : dict
        Analítica oficial.

    geoai_geospatial : dict
        Capa geoespacial oficial.

    geoai_knowledge_base : dict
        Base de Conocimiento oficial.

    geoai_config : dict
        Configuración oficial.

    Returns
    -------
    dict
        Asistente Inteligente oficial.
    """
    # Validación -----------------------------------------------------------
    if not isinstance(spatiotemporal_series,pd.DataFrame):
        raise TypeError("spatiotemporal_series debe ser un DataFrame.")
    if spatiotemporal_series.empty:
        raise ValueError("spatiotemporal_series está vacío.")
    if not isinstance(geoai_analytics,dict):
        raise TypeError("geoai_analytics debe ser un diccionario.")
    if not isinstance(geoai_geospatial,dict):
        raise TypeError("geoai_geospatial debe ser un diccionario.")
    if not isinstance(geoai_knowledge_base,dict):
        raise TypeError("geoai_knowledge_base debe ser un diccionario.")
    if not isinstance(geoai_config,dict):
        raise TypeError("geoai_config debe ser un diccionario.")
    # Recuperación ---------------------------------------------------------
    metadata=geoai_knowledge_base["metadata"]
    # Construcción ---------------------------------------------------------
    geoai_assistant={
        "knowledge_base":geoai_knowledge_base,
        "analytics":geoai_analytics,
        "geospatial":geoai_geospatial,
        "series":spatiotemporal_series,
        "configuration":geoai_config["ai_agent_config"],
        "platform":metadata["platform"],
        "version":metadata["version"],
        "municipalities":spatiotemporal_series["cod_mpio"].nunique(),
        "historical_years":sorted(spatiotemporal_series.loc[spatiotemporal_series["data_type"]=="historical","anio"].unique().tolist()),
        "forecast_years":sorted(spatiotemporal_series.loc[spatiotemporal_series["data_type"]=="forecast","anio"].unique().tolist()),
        "created_at":datetime.now()
    }
    # Validación -----------------------------------------------------------
    required_keys=[
        "knowledge_base",
        "analytics",
        "geospatial",
        "series",
        "configuration",
        "platform",
        "version",
        "municipalities",
        "historical_years",
        "forecast_years",
        "created_at"
    ]
    missing_keys=[key for key in required_keys if key not in geoai_assistant]
    if missing_keys:
        raise RuntimeError(f"La estructura del Asistente Inteligente está incompleta: {missing_keys}")
    if geoai_assistant["municipalities"]<=0:
        raise RuntimeError("No existen municipios disponibles.")
    # Retorno --------------------------------------------------------------
    return geoai_assistant

GEOAI_ASSISTANT=build_geoai_assistant(
    spatiotemporal_series=SPATIOTEMPORAL_SERIES,
    geoai_analytics=GEOAI_ANALYTICS,
    geoai_geospatial=GEOAI_GEOSPATIAL,
    geoai_knowledge_base=GEOAI_KNOWLEDGE_BASE,
    geoai_config=GEOAI_CONFIG
)

print("-"*80)

# BLOQUE 9. Salida Oficial GeoAI -------------------------------------------
## Objetivo: Integrar y validar los componentes oficiales de la Plataforma GeoAI para construir una salida científica única, consistente y reutilizable por las aplicaciones de visualización, análisis territorial, inteligencia artificial y toma de decisiones.
### Producto:
# - GEOAI_OUTPUT
### Responde:
# ¿La Plataforma GeoAI dispone de una salida oficial integrada con todos los componentes científicos necesarios para su funcionamiento?

def build_geoai_output(geoai_analytics:dict,geoai_geospatial:dict,geoai_knowledge_base:dict,geoai_assistant:dict,geoai_config:dict)->dict:
    """
    Construye la salida oficial de la Plataforma GeoAI.

    Parameters
    ----------
    geoai_analytics : dict
        Analítica oficial.

    geoai_geospatial : dict
        Capa geoespacial oficial.

    geoai_knowledge_base : dict
        Base de Conocimiento oficial.

    geoai_assistant : dict
        Asistente Inteligente oficial.

    geoai_config : dict
        Configuración oficial de la Plataforma.

    Returns
    -------
    dict
        Salida oficial de la Plataforma GeoAI.
    """
    # Validación -----------------------------------------------------------
    objects={
        "geoai_analytics":geoai_analytics,
        "geoai_geospatial":geoai_geospatial,
        "geoai_knowledge_base":geoai_knowledge_base,
        "geoai_assistant":geoai_assistant,
        "geoai_config":geoai_config
    }
    for name,obj in objects.items():
        if not isinstance(obj,dict):
            raise TypeError(f"{name} debe ser un diccionario.")
        if not obj:
            raise ValueError(f"{name} está vacío.")
    # Recuperación ---------------------------------------------------------
    metadata=geoai_knowledge_base["metadata"]
    # Construcción ---------------------------------------------------------
    geoai_output={
        "analytics":geoai_analytics,
        "geospatial":geoai_geospatial,
        "knowledge_base":geoai_knowledge_base,
        "assistant":geoai_assistant,
        "metadata":{
            "platform":metadata["platform"],
            "version":metadata["version"],
            "created_at":datetime.now()
        }
    }
    # Validación -----------------------------------------------------------
    required_keys=[
        "analytics",
        "geospatial",
        "knowledge_base",
        "assistant",
        "metadata"
    ]
    missing_keys=[key for key in required_keys if key not in geoai_output]
    if missing_keys:
        raise RuntimeError(f"La salida oficial está incompleta: {missing_keys}")
    required_metadata=["platform","version","created_at"]
    missing_metadata=[key for key in required_metadata if key not in geoai_output["metadata"]]
    if missing_metadata:
        raise RuntimeError(f"Los metadatos oficiales están incompletos: {missing_metadata}")
    # Retorno --------------------------------------------------------------
    return geoai_output

GEOAI_OUTPUT=build_geoai_output(
    geoai_analytics=GEOAI_ANALYTICS,
    geoai_geospatial=GEOAI_GEOSPATIAL,
    geoai_knowledge_base=GEOAI_KNOWLEDGE_BASE,
    geoai_assistant=GEOAI_ASSISTANT,
    geoai_config=GEOAI_CONFIG
)

print("-"*80)

# BLOQUE 10. Resumen de Ejecución GeoAI ------------------------------------
## Objetivo: Validar la salida oficial de la Plataforma GeoAI y presentar el resumen final de ejecución verificando la integridad de todos los componentes científicos construidos durante el pipeline.
### Producto:
# - Resumen Oficial de Ejecución
### Responde:
# ¿La Plataforma GeoAI fue construida correctamente y todos sus componentes científicos están disponibles para su utilización?

def report_geoai_results(geoai_output:dict)->None:
    """
    Valida la salida oficial y presenta el resumen de ejecución de la Plataforma GeoAI.

    Parameters
    ----------
    geoai_output : dict
        Salida oficial de la Plataforma GeoAI.

    Returns
    -------
    None
    """
    # Validación -----------------------------------------------------------
    if not isinstance(geoai_output,dict):
        raise TypeError("geoai_output debe ser un diccionario.")
    if not geoai_output:
        raise ValueError("geoai_output está vacío.")
    required_keys=[
        "analytics",
        "geospatial",
        "knowledge_base",
        "assistant",
        "metadata"
    ]
    missing_keys=[key for key in required_keys if key not in geoai_output]
    if missing_keys:
        raise RuntimeError(f"La salida oficial está incompleta: {missing_keys}")
    # Recuperación ---------------------------------------------------------
    analytics=geoai_output["analytics"]
    metadata=geoai_output["metadata"]
    # Construcción ---------------------------------------------------------
    summary={
        "platform":metadata["platform"],
        "version":metadata["version"],
        "municipalities":analytics["municipalities"],
        "departments":analytics["departments"],
        "historical_records":analytics["historical_records"],
        "forecast_records":analytics["forecast_records"],
        "total_records":analytics["total_records"],
        "created_at":metadata["created_at"]
    }
    # Validación -----------------------------------------------------------
    required_summary=[
        "platform",
        "version",
        "municipalities",
        "departments",
        "historical_records",
        "forecast_records",
        "total_records",
        "created_at"
    ]
    missing_summary=[key for key in required_summary if key not in summary]
    if missing_summary:
        raise RuntimeError(f"El resumen oficial está incompleto: {missing_summary}")
    # Retorno --------------------------------------------------------------
    print("="*80)
    print("PLATAFORMA INTELIGENTE GEOAI")
    print("="*80)
    print(f"Plataforma             : {summary['platform']}")
    print(f"Versión                : {summary['version']}")
    print(f"Municipios             : {summary['municipalities']}")
    print(f"Departamentos          : {summary['departments']}")
    print(f"Registros históricos   : {summary['historical_records']}")
    print(f"Registros proyectados  : {summary['forecast_records']}")
    print(f"Total de registros     : {summary['total_records']}")
    print(f"Fecha de ejecución     : {summary['created_at']}")
    print("="*80)
    print("Estado                 : EJECUCIÓN FINALIZADA CORRECTAMENTE")
    print("="*80)

report_geoai_results(
    geoai_output=GEOAI_OUTPUT
)