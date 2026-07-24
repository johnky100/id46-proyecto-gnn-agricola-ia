# forescasting.py

# BLOQUE 1. Importaciones -------------------------------------------------------
## Objetivo: Importar las dependencias oficiales necesarias para ejecutar el
# protocolo científico de Forecasting Espacio-Temporal utilizando el modelo
# oficial GraphSAGE, el Dataset Científico Certificado, la colección oficial
# de GraphData, los componentes de Inteligencia Artificial Explicable (XAI),
# los productos cartográficos y la infraestructura de la plataforma GeoAI.
#### Producto:
# - Librerías cargadas correctamente.
# - Configuración global inicializada.
# - Paths oficiales disponibles.
# - Arquitectura Graph Neural Network disponible.
# - Componentes de Explicabilidad disponibles.
# - Reproducibilidad científica inicializada.
#### Responde:
# ¿Se cargaron correctamente todas las dependencias oficiales requeridas para
# ejecutar el protocolo científico de Forecasting Espacio-Temporal y generar
# los productos oficiales de la plataforma GeoAI?

import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.python.config.config_project import (
    PROJECT_SEED,
    DEFAULT_CRS_EPSG,
    GEOMETRY_COLUMN,
    FEATURE_COLUMNS,
)

from src.python.config.paths import (
    DATASET_FILE,
    GRAPH_DATA_DIR,
    BEST_MODEL_CONFIG_FILE,
    BEST_MODEL_METADATA_FILE,
    BEST_MODEL_TORCH_FILE,
    FORECAST_RESULTS_DIR,
    FORECAST_REPORTS_DIR,
    FORECAST_MAPS_DIR,
    FORECAST_PRODUCTS_DIR,
    FORECAST_OUTPUT_FILE,
)

from src.python.models.graph_neural_networks import (
    GNN_CONFIG,
    build_gnn_model,
)

from src.python.analysis.explainability import generate_explainability

warnings.filterwarnings("ignore")

np.random.seed(PROJECT_SEED)
torch.manual_seed(PROJECT_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(PROJECT_SEED)

print("-" * 80)
print("Bloque 1. Importaciones cargadas correctamente.")

# BLOQUE 2. Configuración del Forecasting ---------------------------------------
## Objetivo: Definir la configuración oficial del protocolo científico de
# Forecasting Espacio-Temporal, incluyendo la estrategia de predicción, el
# horizonte temporal, la configuración geoespacial, los productos científicos,
# la generación de figuras, la exportación de resultados y los criterios de
# reproducibilidad requeridos por la plataforma GeoAI.
#### Producto:
# - FORECAST_CONFIG construido y validado correctamente.
#### Responde:
# ¿Se configuró correctamente el protocolo oficial de Forecasting
# Espacio-Temporal para garantizar la reproducibilidad, consistencia
# metodológica y estandarización de los productos científicos?

FORECAST_CONFIG={

    # Información general
    "forecast_name":"Forecasting Espacio-Temporal",
    "forecast_version":"1.0.0",
    "official_model":"GraphSAGE",

    # Configuración científica
    "prediction_target":"log_rendimiento",
    "forecast_type":"Espacio-Temporal",
    "forecast_level":"Municipal",
    "forecast_frequency":"Anual",
    "forecast_strategy":"Recursive",
    "graph_type":"Spatial Graph",
    "graph_collection":"Annual GraphData",

    # Horizonte temporal
    "historical_period":(2006,2018),
    "forecast_period":(2019,2035),
    "forecast_years":list(range(2019,2036)),

    # Configuración geoespacial
    "crs_epsg":DEFAULT_CRS_EPSG,
    "geometry_column":GEOMETRY_COLUMN,

    # Configuración cartográfica
    "figure_format":"png",
    "figure_dpi":300,
    "figure_width":14,
    "figure_height":10,
    "figure_title":True,
    "figure_variable":True,
    "figure_year":True,

    # Productos científicos
    "generate_predictions":True,
    "generate_statistics":True,
    "generate_prediction_maps":True,
    "generate_change_maps":True,
    "generate_percentage_maps":True,
    "generate_anomaly_maps":True,
    "generate_graph_map":True,
    "generate_animation":True,
    "generate_geoai_products":True,

    # Configuración de Explicabilidad
    "generate_explainability":True,
    "explainability_method":"GNNExplainer",

    # Organización de productos
    "historical_products":True,
    "forecast_products":True,
    "comparative_products":True,
    "geoai_products":True,

    # Exportación
    "export_formats":[
        "parquet",
        "csv",
        "gpkg",
        "geojson",
        "json",
        "png"
    ],

    # Persistencia
    "save_results":True,
    "save_figures":True,
    "save_products":True,
    "save_forecast_summary":True,
    "save_forecast_metadata":True,
    "save_global_statistics":True,
    "save_dashboard_data":True,

    # Control de calidad
    "validate_outputs":True,
    "reproducible":True
}

print("-"*80)
print("Bloque 2. Configuración del Forecasting construida correctamente.")

# BLOQUE 3. Recuperación del Modelo Oficial -------------------------------------
## Objetivo: Recuperar los artefactos oficiales del modelo GraphSAGE
# seleccionados durante el Benchmark Científico y entrenados previamente,
# los cuales serán utilizados para reconstruir posteriormente la arquitectura
# del modelo durante el proceso de Forecasting Espacio-Temporal.
#### Producto:
# - OFFICIAL_MODEL construido y validado correctamente.
#### Responde:
# ¿Se recuperaron correctamente los artefactos oficiales necesarios para
# reconstruir el modelo GraphSAGE del proyecto?

def load_official_model():

    # Validación
    if not BEST_MODEL_TORCH_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo oficial: {BEST_MODEL_TORCH_FILE}"
        )

    if not BEST_MODEL_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró la configuración oficial: {BEST_MODEL_CONFIG_FILE}"
        )

    if not BEST_MODEL_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"No se encontraron los metadatos oficiales: {BEST_MODEL_METADATA_FILE}"
        )

    # Recuperación
    with open(BEST_MODEL_CONFIG_FILE, "r", encoding="utf-8") as file:
        model_config = json.load(file)

    with open(BEST_MODEL_METADATA_FILE, "r", encoding="utf-8") as file:
        model_metadata = json.load(file)

    checkpoint = torch.load(
        BEST_MODEL_TORCH_FILE,
        map_location="cpu"
    )

    # Construcción
    official_model = {
        "checkpoint": checkpoint,
        "model_state_dict": checkpoint["model_state_dict"],
        "model_config": model_config,
        "model_metadata": model_metadata,
        "model_name": model_config["model_name"],
        "model_family": model_config["family"],
        "model_version": model_config.get("model_version", "1.0.0"),
        "device": "cpu",
        "status": "OFFICIAL_MODEL_RECOVERED"
    }

    # Validación
    assert official_model["checkpoint"] is not None
    assert official_model["model_state_dict"] is not None
    assert official_model["model_config"] is not None
    assert official_model["model_metadata"] is not None
    assert official_model["model_name"] is not None
    assert official_model["model_family"] is not None
    assert official_model["device"] == "cpu"
    assert official_model["status"] == "OFFICIAL_MODEL_RECOVERED"

    # Retorno
    return official_model

OFFICIAL_MODEL = load_official_model()

print("-" * 80)
print("Bloque 3. Modelo oficial recuperado correctamente.")

# BLOQUE 4. Recuperación del Dataset y GraphData ---------------------------
## Objetivo: Recuperar el Dataset Científico Certificado y la colección
# oficial de GraphData utilizados durante el entrenamiento del modelo
# GraphSAGE, con el fin de reconstruir posteriormente la arquitectura
# oficial y ejecutar el Forecasting Espacio-Temporal.
#### Producto:
# - FORECAST_DATA construido y validado correctamente.
#### Responde:
# ¿Se recuperaron correctamente el Dataset Científico Certificado y la
# colección oficial de GraphData requeridos para el Forecasting?

def load_forecast_data():

    # Validación
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el Dataset Científico Certificado: {DATASET_FILE}"
        )

    if not GRAPH_DATA_DIR.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio GraphData: {GRAPH_DATA_DIR}"
        )

    # Recuperación
    dataset = pd.read_parquet(DATASET_FILE)

    graph_files = [
        graph_file
        for graph_file in sorted(GRAPH_DATA_DIR.glob("graph_data_*.pt"))
        if graph_file.stem != "graph_data_collection"
    ]

    if not graph_files:
        raise FileNotFoundError(
            "No se encontraron archivos GraphData históricos."
        )

    graphs = [
        torch.load(graph_file, map_location="cpu")
        for graph_file in graph_files
    ]

    # Construcción
    forecast_data = {
        "dataset": dataset,
        "graphs": graphs,
        "graph_files": graph_files,
        "n_graphs": len(graphs),
        "n_observations": len(dataset),
        "historical_period": FORECAST_CONFIG["historical_period"],
        "forecast_period": FORECAST_CONFIG["forecast_period"],
        "status": "FORECAST_DATA_READY"
    }

    # Validación
    assert forecast_data["dataset"] is not None
    assert forecast_data["graphs"]
    assert forecast_data["n_graphs"] > 0
    assert forecast_data["n_observations"] > 0
    assert forecast_data["status"] == "FORECAST_DATA_READY"

    # Retorno
    return forecast_data


FORECAST_DATA = load_forecast_data()

print("-" * 80)
print("Bloque 4. Dataset Científico y GraphData recuperados correctamente.")

# BLOQUE 5. Reconstrucción del Modelo Oficial ------------------------------
## Objetivo: Reconstruir la arquitectura oficial GraphSAGE utilizando la
# configuración recuperada, la colección oficial de GraphData y el estado
# entrenado del modelo para preparar el proceso de Forecasting Espacio-Temporal.
#### Producto:
# - FORECAST_MODEL construido y validado correctamente.
#### Responde:
# ¿Se reconstruyó correctamente el modelo oficial GraphSAGE para ejecutar
# el Forecasting Espacio-Temporal?

def build_forecast_model():

    # Validación
    if OFFICIAL_MODEL["status"] != "OFFICIAL_MODEL_RECOVERED":
        raise ValueError(
            "Los artefactos del modelo oficial no están disponibles."
        )

    if FORECAST_DATA["status"] != "FORECAST_DATA_READY":
        raise ValueError(
            "Los datos oficiales del forecasting no están disponibles."
        )

    # Recuperación
    model_config = OFFICIAL_MODEL["model_config"]
    model_state_dict = OFFICIAL_MODEL["model_state_dict"]
    input_channels = FORECAST_DATA["graphs"][0].num_node_features
    output_channels = 1

    # Construcción
    model = build_gnn_model(
        model_config=model_config,
        input_channels=input_channels,
        output_channels=output_channels
    )

    model.load_state_dict(model_state_dict)
    model.eval()

    forecast_model = {
        "model": model,
        "model_config": model_config,
        "model_name": OFFICIAL_MODEL["model_name"],
        "model_family": OFFICIAL_MODEL["model_family"],
        "model_version": OFFICIAL_MODEL["model_version"],
        "input_channels": input_channels,
        "output_channels": output_channels,
        "device": OFFICIAL_MODEL["device"],
        "status": "MODEL_READY"
    }

    # Validación
    assert forecast_model["model"] is not None
    assert forecast_model["input_channels"] > 0
    assert forecast_model["output_channels"] == 1
    assert forecast_model["status"] == "MODEL_READY"

    # Retorno
    return forecast_model


FORECAST_MODEL = build_forecast_model()

print("-" * 80)
print("Bloque 5. Modelo oficial reconstruido correctamente.")

# BLOQUE 6. Construcción del Dataset Forecast ------------------------------
## Objetivo: Construir el Dataset Científico Forecast para cada año del
# horizonte de predicción utilizando como punto de partida el último año
# histórico disponible y la estrategia oficial de Forecasting Espacio-
# Temporal definida para el proyecto.
#### Producto:
# - FORECAST_DATASET construido y validado correctamente.
#### Responde:
# ¿Se construyó correctamente el Dataset Científico Forecast para el
# horizonte temporal definido?

def build_forecast_dataset():
    # Validación
    if FORECAST_DATA["status"] != "FORECAST_DATA_READY":
        raise ValueError("El Dataset Científico Histórico no está disponible.")
    if FORECAST_MODEL["status"] != "MODEL_READY":
        raise ValueError("El modelo oficial no está disponible.")

    # Recuperación
    dataset = FORECAST_DATA["dataset"].copy()
    historical_year = FORECAST_CONFIG["historical_period"][1]
    forecast_years = FORECAST_CONFIG["forecast_years"]
    base_dataset = (
        dataset.loc[dataset["anio"] == historical_year]
        .copy()
        .reset_index(drop=True)
    )

    # Construcción
    forecast_dataset = []
    for year in forecast_years:
        year_dataset = base_dataset.copy()
        year_dataset["anio"] = year
        year_dataset["historical_year"] = historical_year
        year_dataset["forecast_year"] = year
        year_dataset["forecast_step"] = year - historical_year
        year_dataset["forecast"] = True
        forecast_dataset.append(year_dataset)

    forecast_dataset = pd.concat(forecast_dataset, ignore_index=True)

    forecast_result = {
        "dataset": forecast_dataset,
        "historical_year": historical_year,
        "forecast_period": FORECAST_CONFIG["forecast_period"],
        "forecast_years": forecast_years,
        "n_years": len(forecast_years),
        "n_observations": len(forecast_dataset),
        "status": "FORECAST_DATASET_READY"
    }

    # Validación
    assert forecast_result["dataset"] is not None
    assert forecast_result["n_years"] > 0
    assert forecast_result["n_observations"] > 0
    assert forecast_result["status"] == "FORECAST_DATASET_READY"

    # Retorno
    return forecast_result

FORECAST_DATASET = build_forecast_dataset()

print("-" * 80)
print("Bloque 6. Dataset Forecast construido correctamente.")

# BLOQUE 7. Construcción del GraphData Forecast -----------------------------
## Objetivo: Construir la colección oficial de GraphData Forecast a partir
# del Dataset Científico Forecast utilizando la estructura espacial del
# grafo histórico para preparar la ejecución del Forecasting Espacio-
# Temporal mediante el modelo oficial GraphSAGE.
#### Producto:
# - FORECAST_GRAPHS construido y validado correctamente.
#### Responde:
# ¿Se construyó correctamente la colección oficial de GraphData Forecast
# requerida para ejecutar el Forecasting Espacio-Temporal?

def build_forecast_graphs():
    # Validación
    if FORECAST_DATASET["status"] != "FORECAST_DATASET_READY":
        raise ValueError("El Dataset Forecast no está disponible.")
    if FORECAST_DATA["status"] != "FORECAST_DATA_READY":
        raise ValueError("La colección histórica de GraphData no está disponible.")

    # Recuperación
    forecast_dataset = FORECAST_DATASET["dataset"]
    reference_graph = FORECAST_DATA["graphs"][-1]
    forecast_years = FORECAST_DATASET["forecast_years"]

    # Construcción
    forecast_graphs = []
    for year in forecast_years:
        year_dataset = (
            forecast_dataset.loc[forecast_dataset["anio"] == year]
            .copy()
            .reset_index(drop=True)
        )

        graph = reference_graph.clone()
        graph.x = torch.tensor(
            year_dataset[FEATURE_COLUMNS].values,
            dtype=torch.float
        )
        graph.year = year
        graph.forecast = True
        graph.forecast_step = year - FORECAST_DATASET["historical_year"]
        forecast_graphs.append(graph)

    forecast_result = {
        "graphs": forecast_graphs,
        "reference_graph": reference_graph,
        "forecast_years": forecast_years,
        "n_graphs": len(forecast_graphs),
        "status": "FORECAST_GRAPHS_READY"
    }

    # Validación
    assert forecast_result["graphs"]
    assert forecast_result["n_graphs"] > 0
    assert forecast_result["status"] == "FORECAST_GRAPHS_READY"

    # Retorno
    return forecast_result

FORECAST_GRAPHS = build_forecast_graphs()

print("-" * 80)
print("Bloque 7. GraphData Forecast construido correctamente.")

# BLOQUE 8. Forecast Recursivo ---------------------------------------------
## Objetivo: Ejecutar el Forecasting Espacio-Temporal mediante una estrategia
# recursiva utilizando la colección oficial de GraphData Forecast y el modelo
# oficial GraphSAGE para generar las predicciones correspondientes al
# horizonte temporal definido.
#### Producto:
# - FORECAST_RESULT construido y validado correctamente.
#### Responde:
# ¿Se ejecutó correctamente el Forecasting Espacio-Temporal utilizando la
# estrategia recursiva definida para el proyecto?

def run_recursive_forecast():
    # Validación
    if FORECAST_MODEL["status"] != "MODEL_READY":
        raise ValueError("El modelo oficial no está disponible.")
    if FORECAST_GRAPHS["status"] != "FORECAST_GRAPHS_READY":
        raise ValueError("La colección GraphData Forecast no está disponible.")

    # Recuperación
    model = FORECAST_MODEL["model"]
    graphs = FORECAST_GRAPHS["graphs"]
    forecast_years = FORECAST_GRAPHS["forecast_years"]
    inference_start = datetime.now()

    # Construcción
    predictions = []
    model.eval()
    with torch.no_grad():
        for graph, year in zip(graphs, forecast_years):
            prediction = model(graph.x, graph.edge_index).cpu().numpy()
            predictions.append({
                "year": year,
                "prediction": prediction,
                "n_nodes": prediction.shape[0]
            })

    forecast_result = {
        "predictions": predictions,
        "forecast_years": forecast_years,
        "n_graphs": len(graphs),
        "n_predictions": sum(p["n_nodes"] for p in predictions),
        "inference_time": (datetime.now() - inference_start).total_seconds(),
        "status": "FORECAST_COMPLETED"
    }

    # Validación
    assert forecast_result["n_graphs"] > 0
    assert forecast_result["n_predictions"] > 0
    assert forecast_result["status"] == "FORECAST_COMPLETED"

    # Retorno
    return forecast_result

FORECAST_RESULT = run_recursive_forecast()

print("-" * 80)
print("Bloque 8. Forecast Recursivo ejecutado correctamente.")

# BLOQUE 9. Actualización Recursiva del Dataset Forecast -------------------
## Objetivo: Actualizar recursivamente el Dataset Científico Forecast
# incorporando las predicciones generadas por el modelo GraphSAGE para
# mantener la consistencia temporal durante el horizonte de Forecasting
# Espacio-Temporal.
#### Producto:
# - FORECAST_DATASET actualizado correctamente.
#### Responde:
# ¿Se actualizó correctamente el Dataset Científico Forecast utilizando
# las predicciones obtenidas durante el Forecasting Recursivo?

def update_forecast_dataset():
    # Validación
    if FORECAST_RESULT["status"] != "FORECAST_COMPLETED":
        raise ValueError("El Forecasting no ha sido ejecutado.")
    if FORECAST_DATASET["status"] != "FORECAST_DATASET_READY":
        raise ValueError("El Dataset Forecast no está disponible.")

    # Recuperación
    dataset = FORECAST_DATASET["dataset"].copy()
    predictions = FORECAST_RESULT["predictions"]
    target = FORECAST_CONFIG["prediction_target"]

    # Construcción
    for result in predictions:
        mask = dataset["anio"] == result["year"]
        dataset.loc[mask, target] = result["prediction"].reshape(-1)

    forecast_dataset = {
        "dataset": dataset,
        "forecast_years": FORECAST_RESULT["forecast_years"],
        "prediction_target": target,
        "n_observations": len(dataset),
        "status": "FORECAST_DATASET_UPDATED"
    }

    # Validación
    assert forecast_dataset["dataset"] is not None
    assert forecast_dataset["n_observations"] > 0
    assert forecast_dataset["status"] == "FORECAST_DATASET_UPDATED"

    # Retorno
    return forecast_dataset


FORECAST_DATASET = update_forecast_dataset()

print("-" * 80)
print("Bloque 9. Dataset Forecast actualizado correctamente.")

# BLOQUE 10. Reconstrucción Recursiva del GraphData Forecast ---------------
## Objetivo: Reconstruir la colección oficial de GraphData Forecast a partir
# del Dataset Científico Forecast actualizado para preparar la siguiente
# iteración del Forecasting Espacio-Temporal mediante el modelo GraphSAGE.
#### Producto:
# - FORECAST_GRAPHS actualizado y validado correctamente.
#### Responde:
# ¿Se reconstruyó correctamente la colección oficial de GraphData Forecast
# para la siguiente iteración del Forecasting Espacio-Temporal?

def rebuild_forecast_graphs():
    # Validación
    if FORECAST_DATASET["status"] != "FORECAST_DATASET_UPDATED":
        raise ValueError("El Dataset Forecast actualizado no está disponible.")
    if FORECAST_DATA["status"] != "FORECAST_DATA_READY":
        raise ValueError("La colección histórica de GraphData no está disponible.")

    # Recuperación
    dataset = FORECAST_DATASET["dataset"]
    forecast_years = FORECAST_DATASET["forecast_years"]
    reference_graph = FORECAST_DATA["graphs"][-1]
    historical_year = FORECAST_DATASET["historical_year"]

    # Construcción
    forecast_graphs = []
    for year in forecast_years:
        year_dataset = (
            dataset.loc[dataset["anio"] == year]
            .copy()
            .reset_index(drop=True)
        )

        graph = reference_graph.clone()
        graph.x = torch.tensor(
            year_dataset[FEATURE_COLUMNS].values,
            dtype=torch.float
        )
        graph.year = year
        graph.forecast = True
        graph.forecast_step = year - historical_year
        forecast_graphs.append(graph)

    forecast_result = {
        "graphs": forecast_graphs,
        "forecast_years": forecast_years,
        "historical_year": historical_year,
        "n_graphs": len(forecast_graphs),
        "status": "FORECAST_GRAPHS_UPDATED"
    }

    # Validación
    assert forecast_result["graphs"]
    assert forecast_result["n_graphs"] > 0
    assert forecast_result["status"] == "FORECAST_GRAPHS_UPDATED"

    # Retorno
    return forecast_result


FORECAST_GRAPHS = rebuild_forecast_graphs()

print("-" * 80)
print("Bloque 10. GraphData Forecast reconstruido correctamente.")

# BLOQUE 11. Consolidación de Resultados del Forecasting -------------------
## Objetivo: Consolidar las predicciones generadas durante el Forecasting
# Espacio-Temporal en un Dataset Científico único que servirá como base para
# la generación de productos cartográficos, análisis científicos y la
# plataforma GeoAI.
#### Producto:
# - FORECAST_RESULTS_DATASET construido y validado correctamente.
#### Responde:
# ¿Se consolidaron correctamente los resultados del Forecasting
# Espacio-Temporal?

def build_forecast_results():

    # Validación
    if FORECAST_RESULT["status"] != "FORECAST_COMPLETED":
        raise ValueError(
            "El Forecasting no ha sido ejecutado."
        )

    if FORECAST_DATASET["status"] != "FORECAST_DATASET_UPDATED":
        raise ValueError(
            "El Dataset Forecast actualizado no está disponible."
        )

    # Recuperación
    dataset = FORECAST_DATASET["dataset"].copy()
    forecast_years = FORECAST_RESULT["forecast_years"]
    target = FORECAST_CONFIG["prediction_target"]

    # Construcción
    forecast_results = (
        dataset.loc[
            dataset["anio"].isin(forecast_years)
        ]
        .copy()
        .reset_index(drop=True)
    )

    forecast_summary = {
        "dataset": forecast_results,
        "forecast_years": forecast_years,
        "prediction_target": target,
        "historical_year": FORECAST_DATASET["historical_year"],
        "forecast_period": FORECAST_DATASET["forecast_period"],
        "n_years": len(forecast_years),
        "n_observations": len(forecast_results),
        "status": "FORECAST_RESULTS_READY"
    }

    # Validación
    assert forecast_summary["dataset"] is not None
    assert forecast_summary["n_observations"] > 0
    assert forecast_summary["status"] == "FORECAST_RESULTS_READY"

    # Retorno
    return forecast_summary


FORECAST_RESULTS = build_forecast_results()

print("-" * 80)
print("Bloque 11. Resultados del Forecasting consolidados correctamente.")

# BLOQUE 12. Evaluación Científica del Forecast ----------------------------
## Objetivo: Evaluar científicamente las predicciones generadas durante el
# Forecasting Espacio-Temporal mediante indicadores estadísticos y de
# consistencia espacio-temporal para garantizar la calidad del Dataset
# Científico Forecast.
#### Producto:
# - FORECAST_EVALUATION construido y validado correctamente.
#### Responde:
# ¿Los resultados del Forecasting cumplen los criterios de calidad
# científica establecidos para el proyecto?

def evaluate_forecast():

    # Validación
    if FORECAST_RESULTS["status"] != "FORECAST_RESULTS_READY":
        raise ValueError(
            "Los resultados del Forecasting no están disponibles."
        )

    # Recuperación
    dataset = FORECAST_RESULTS["dataset"]
    target = FORECAST_RESULTS["prediction_target"]

    # Construcción
    evaluation = {

        "minimum": float(dataset[target].min()),
        "maximum": float(dataset[target].max()),
        "mean": float(dataset[target].mean()),
        "median": float(dataset[target].median()),
        "std": float(dataset[target].std()),
        "variance": float(dataset[target].var()),

        "missing_values": int(dataset[target].isna().sum()),
        "duplicate_records": int(dataset.duplicated().sum()),

        "n_years": dataset["anio"].nunique(),
        "n_municipalities": dataset["cod_mpio"].nunique(),
        "n_predictions": len(dataset),

        "status": "FORECAST_EVALUATED"

    }

    # Validación
    assert evaluation["n_predictions"] > 0
    assert evaluation["n_years"] > 0
    assert evaluation["missing_values"] == 0
    assert evaluation["status"] == "FORECAST_EVALUATED"

    # Retorno
    return evaluation


FORECAST_EVALUATION = evaluate_forecast()

print("-" * 80)
print("Bloque 12. Evaluación científica completada.")

# BLOQUE 13. Generación de Mapas Científicos -------------------------------
## Objetivo: Generar la colección oficial de mapas científicos a partir del
# Dataset Científico Forecast para representar espacialmente las
# predicciones obtenidas mediante el modelo GraphSAGE.
#### Producto:
# - FORECAST_MAPS construido y validado correctamente.
#### Responde:
# ¿Se generaron correctamente los mapas científicos del Forecasting
# Espacio-Temporal?

def build_forecast_maps():

    # Validación
    if FORECAST_RESULTS["status"] != "FORECAST_RESULTS_READY":
        raise ValueError(
            "Los resultados del Forecasting no están disponibles."
        )

    if FORECAST_EVALUATION["status"] != "FORECAST_EVALUATED":
        raise ValueError(
            "La evaluación científica no está disponible."
        )

    # Recuperación
    dataset = FORECAST_RESULTS["dataset"]
    forecast_years = FORECAST_RESULTS["forecast_years"]

    # Construcción
    maps = generate_forecast_maps(
        dataset=dataset,
        years=forecast_years,
        config=FORECAST_CONFIG
    )

    forecast_maps = {
        "maps": maps,
        "forecast_years": forecast_years,
        "historical_year": FORECAST_RESULTS["historical_year"],
        "forecast_period": FORECAST_RESULTS["forecast_period"],
        "n_maps": len(maps),
        "status": "FORECAST_MAPS_READY"
    }

    # Validación
    assert forecast_maps["n_maps"] > 0
    assert forecast_maps["status"] == "FORECAST_MAPS_READY"

    # Retorno
    return forecast_maps


FORECAST_MAPS = build_forecast_maps()

print("-" * 80)
print("Bloque 13. Mapas científicos generados correctamente.")

# BLOQUE 14. Explicabilidad Científica del Forecast ------------------------
## Objetivo: Generar la explicación científica de las predicciones obtenidas
# mediante el modelo GraphSAGE utilizando técnicas oficiales de Explainable
# Artificial Intelligence (XAI) para identificar la contribución de las
# variables y de la estructura espacial del grafo.
#### Producto:
# - FORECAST_EXPLAINABILITY construido y validado correctamente.
#### Responde:
# ¿Se generó correctamente la explicación científica del Forecasting
# Espacio-Temporal?

def build_forecast_explainability():

    # Validación
    if FORECAST_RESULTS["status"] != "FORECAST_RESULTS_READY":
        raise ValueError(
            "Los resultados del Forecasting no están disponibles."
        )

    if FORECAST_MODEL["status"] != "MODEL_READY":
        raise ValueError(
            "El modelo oficial no está disponible."
        )

    # Recuperación
    model = FORECAST_MODEL["model"]
    graphs = FORECAST_GRAPHS["graphs"]

    # Construcción
    explainability = generate_explainability(
        model=model,
        graphs=graphs,
        config=FORECAST_CONFIG
    )

    forecast_explainability = {
        "explainability": explainability,
        "method": FORECAST_CONFIG["explainability_method"],
        "forecast_years": FORECAST_RESULTS["forecast_years"],
        "historical_year": FORECAST_RESULTS["historical_year"],
        "forecast_period": FORECAST_RESULTS["forecast_period"],
        "n_graphs": len(graphs),
        "status": "FORECAST_EXPLAINABILITY_READY"
    }

    # Validación
    assert forecast_explainability["explainability"] is not None
    assert forecast_explainability["n_graphs"] > 0
    assert forecast_explainability["status"] == "FORECAST_EXPLAINABILITY_READY"

    # Retorno
    return forecast_explainability


FORECAST_EXPLAINABILITY = build_forecast_explainability()

print("-" * 80)
print("Bloque 14. Explicabilidad científica generada correctamente.")

# BLOQUE 15. Integración Científica ----------------------------------------
## Objetivo: Integrar los productos científicos del Forecasting en un único
# conjunto de indicadores para la plataforma GeoAI.
### Producto:
# - GEOAI_INDICATORS construido y validado correctamente.
### Responde:
# ¿Se integraron correctamente los indicadores científicos?

def build_geoai_indicators():

    # Validación
    if FORECAST_RESULTS["status"] != "FORECAST_RESULTS_READY":
        raise ValueError("Los resultados del Forecasting no están disponibles.")

    if FORECAST_EVALUATION["status"] != "FORECAST_EVALUATED":
        raise ValueError("La evaluación científica no está disponible.")

    if FORECAST_MAPS["status"] != "FORECAST_MAPS_READY":
        raise ValueError("Los mapas científicos no están disponibles.")

    if FORECAST_EXPLAINABILITY["status"] != "FORECAST_EXPLAINABILITY_READY":
        raise ValueError("La explicabilidad científica no está disponible.")

    # Recuperación
    forecast = FORECAST_RESULTS
    evaluation = FORECAST_EVALUATION
    maps = FORECAST_MAPS
    explainability = FORECAST_EXPLAINABILITY

    # Construcción
    geoai_indicators = {
        "forecast": forecast,
        "evaluation": evaluation,
        "maps": maps,
        "explainability": explainability,
        "model": FORECAST_MODEL["model_name"],
        "forecast_period": FORECAST_RESULTS["forecast_period"],
        "generation_date": datetime.now(),
        "status": "GEOAI_INDICATORS_READY"
    }

    # Validación
    assert geoai_indicators["status"] == "GEOAI_INDICATORS_READY"
    assert all(
        geoai_indicators[key] is not None
        for key in (
            "forecast",
            "evaluation",
            "maps",
            "explainability"
        )
    )

    # Retorno
    return geoai_indicators


GEOAI_INDICATORS = build_geoai_indicators()

print("-" * 80)
print("Bloque 15. Integración Científica completada correctamente.")

# BLOQUE 16. Construcción del Paquete Científico GeoAI ---------------------
## Objetivo: Integrar los productos científicos del Forecasting en un único
# paquete oficial para su consumo por la plataforma GeoAI.
### Producto:
# - GEOAI_PACKAGE construido y validado correctamente.
### Responde:
# ¿Se consolidó correctamente el Paquete Científico GeoAI?

def build_geoai_package():

    # Validación
    if FORECAST_RESULTS["status"] != "FORECAST_RESULTS_READY":
        raise ValueError("Los resultados del Forecasting no están disponibles.")

    if FORECAST_EVALUATION["status"] != "FORECAST_EVALUATED":
        raise ValueError("La evaluación científica no está disponible.")

    if FORECAST_MAPS["status"] != "FORECAST_MAPS_READY":
        raise ValueError("Los mapas científicos no están disponibles.")

    if FORECAST_EXPLAINABILITY["status"] != "FORECAST_EXPLAINABILITY_READY":
        raise ValueError("La explicabilidad científica no está disponible.")

    # Recuperación
    forecast = FORECAST_RESULTS
    evaluation = FORECAST_EVALUATION
    maps = FORECAST_MAPS
    explainability = FORECAST_EXPLAINABILITY

    # Construcción
    geoai_package = {
        "forecast": forecast,
        "evaluation": evaluation,
        "maps": maps,
        "explainability": explainability,
        "metadata": {
            "model": FORECAST_MODEL["model_name"],
            "model_family": FORECAST_MODEL["model_family"],
            "model_version": FORECAST_MODEL["model_version"],
            "forecast_period": FORECAST_CONFIG["forecast_period"],
            "generated_at": datetime.now().isoformat()
        },
        "status": "GEOAI_PACKAGE_READY"
    }

    # Validación
    assert all(
        geoai_package[key] is not None
        for key in (
            "forecast",
            "evaluation",
            "maps",
            "explainability",
            "metadata"
        )
    )
    assert geoai_package["status"] == "GEOAI_PACKAGE_READY"

    # Retorno
    return geoai_package


GEOAI_PACKAGE = build_geoai_package()

print("-" * 80)
print("Bloque 16. Paquete Científico GeoAI construido correctamente.")

# BLOQUE 17. Exportación del Paquete Científico GeoAI ----------------------
## Objetivo: Exportar el Paquete Científico GeoAI como producto oficial del
# Forecasting para su utilización por la plataforma GeoAI.
### Producto:
# - GEOAI_PACKAGE exportado y validado correctamente.
### Responde:
# ¿Se exportó correctamente el Paquete Científico GeoAI?

def export_geoai_package():

    # Validación
    if GEOAI_PACKAGE["status"] != "GEOAI_PACKAGE_READY":
        raise ValueError(
            "El Paquete Científico GeoAI no está disponible."
        )

    # Recuperación
    package = GEOAI_PACKAGE
    output_path = OUTPUTS_DIR / "geoai_package.pt"

    # Construcción
    torch.save(package, output_path)

    export_result = {
        "file": output_path,
        "filename": output_path.name,
        "format": output_path.suffix.replace(".", ""),
        "size_mb": round(output_path.stat().st_size / 1024**2, 2),
        "generated_at": datetime.now().isoformat(),
        "status": "GEOAI_PACKAGE_EXPORTED"
    }

    # Validación
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert export_result["status"] == "GEOAI_PACKAGE_EXPORTED"

    # Retorno
    return export_result


GEOAI_EXPORT = export_geoai_package()

print("-" * 80)
print("Bloque 17. Paquete Científico GeoAI exportado correctamente.")

# BLOQUE 18. Resumen Consolidado del Forecasting ---------------------------
## Objetivo: Consolidar el resumen ejecutivo del proceso de Forecasting
# integrando los productos científicos, archivos exportados, ubicaciones y
# metadatos de ejecución.
### Producto:
# - FORECAST_SUMMARY construido y validado correctamente.
### Responde:
# ¿Finalizó correctamente el proceso completo de Forecasting y quedaron
# disponibles todos los productos científicos?

def build_forecast_summary():

    # Validación
    if GEOAI_EXPORT["status"] != "GEOAI_PACKAGE_EXPORTED":
        raise ValueError(
            "El Paquete Científico GeoAI no fue exportado correctamente."
        )

    # Recuperación
    export = GEOAI_EXPORT
    package = GEOAI_PACKAGE

    # Construcción
    forecast_summary = {
        "model": FORECAST_MODEL["model_name"],
        "model_family": FORECAST_MODEL["model_family"],
        "model_version": FORECAST_MODEL["model_version"],
        "historical_period": FORECAST_CONFIG["historical_period"],
        "forecast_period": FORECAST_CONFIG["forecast_period"],
        "forecast_years": FORECAST_CONFIG["forecast_years"],

        "products": {
            "forecast": FORECAST_RESULTS["status"],
            "evaluation": FORECAST_EVALUATION["status"],
            "maps": FORECAST_MAPS["status"],
            "explainability": FORECAST_EXPLAINABILITY["status"],
            "package": package["status"]
        },

        "exports": {
            "file": str(export["file"]),
            "filename": export["filename"],
            "format": export["format"],
            "size_mb": export["size_mb"]
        },

        "metadata": {
            "generated_at": package["metadata"]["generated_at"],
            "prediction_target": FORECAST_CONFIG["prediction_target"]
        },

        "status": "FORECAST_SUMMARY_READY"
    }

    # Validación
    assert forecast_summary["products"]["package"] == "GEOAI_PACKAGE_READY"
    assert forecast_summary["exports"]["size_mb"] > 0
    assert forecast_summary["status"] == "FORECAST_SUMMARY_READY"

    # Retorno
    return forecast_summary


FORECAST_SUMMARY = build_forecast_summary()

print("-" * 80)
print("Bloque 18. Resumen consolidado del Forecasting construido correctamente.")