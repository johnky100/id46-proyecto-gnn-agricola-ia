# graph-05_forescasting.py

# BLOQUE 1. Importaciones -------------------------------------------------------
## Objetivo: Importar las dependencias oficiales necesarias para ejecutar el
# protocolo científico de Forecasting Espacio-Temporal utilizando el Modelo
# Oficial GraphSAGE, el Dataset Científico Certificado y la colección oficial
# de GraphData.
#### Producto:
# - Librerías científicas cargadas correctamente.
# - Configuración global inicializada.
# - Rutas oficiales disponibles.
# - Arquitectura Graph Neural Network disponible.
# - Reproducibilidad científica inicializada.
#### Responde:
# ¿Se cargaron correctamente todas las dependencias oficiales requeridas para
# ejecutar el protocolo científico de Forecasting Espacio-Temporal?
print("-" * 80)
print("Bloque 1. Importaciones.")

# Librerías estándar
import json
import warnings
from pathlib import Path

# Librerías científicas
import joblib
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# Configuración del proyecto
from src.python.config.config_project import (
    PROJECT_SEED,
    FEATURE_COLUMNS
)

# Variables oficiales del Forecast
TARGET_COLUMNS = [
    "log_rendimiento"
]

# Rutas oficiales
from src.python.config.paths import (
    BENCHMARK_DATA_FILE,
    OFFICIAL_MODEL_TORCH_FILE,
    OFFICIAL_MODEL_CONFIG_FILE,
    OFFICIAL_MODEL_METADATA_FILE
)

# Directorios oficiales del Forecast
FORECAST_DIR = Path(
    "src/python/outputs/forecast"
)

FORECAST_RESULTS_DIR = (
    FORECAST_DIR / "results"
)

FORECAST_REPORTS_DIR = (
    FORECAST_DIR / "reports"
)

FORECAST_PRODUCTS_DIR = (
    FORECAST_DIR / "products"
)

FORECAST_MAPS_DIR = (
    FORECAST_DIR / "maps"
)

# Arquitectura Graph Neural Network
from src.python.models.graph_neural_networks import (
    build_gnn_model
)

# Configuración del entorno
warnings.filterwarnings("ignore")

np.random.seed(PROJECT_SEED)

torch.manual_seed(PROJECT_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(PROJECT_SEED)

print("-" * 80)
print("Bloque 1. Importaciones completado correctamente.")

# BLOQUE 2. Configuración del Forecasting ---------------------------------------
## Objetivo: Definir la configuración oficial del protocolo científico de
# Forecasting Espacio-Temporal utilizando el Modelo Oficial GraphSAGE y la
# colección oficial de GraphData para generar predicciones futuras.
#### Producto:
# - FORECAST_CONFIG construido y validado correctamente.
#### Responde:
# ¿Se configuró correctamente el protocolo oficial de Forecasting
# Espacio-Temporal para garantizar la reproducibilidad y consistencia
# metodológica del proceso de predicción?

print("-" * 80)
print("Bloque 2. Configuración del Forecasting.")

FORECAST_CONFIG = {

    # Información general
    "forecast_name": "Forecasting Espacio-Temporal",
    "forecast_version": "1.0.0",
    "official_model": "GraphSAGE",

    # Configuración científica
    "prediction_target": TARGET_COLUMNS[0],
    "forecast_type": "Espacio-Temporal",
    "forecast_level": "Municipal",
    "forecast_frequency": "Anual",

    # Estrategia de Forecasting
    "forecast_strategy": "Direct",
    "forecast_mode": "Baseline",

    # Horizonte temporal
    "historical_period": (2006, 2018),
    "forecast_period": (2019, 2035),
    "forecast_years": list(range(2019, 2036)),
    "forecast_horizon": 17,

    # Variables oficiales
    "feature_columns": FEATURE_COLUMNS,
    "target_columns": TARGET_COLUMNS,

    # Productos oficiales
    "generate_predictions": True,
    "generate_statistics": True,
    "save_results": True,
    "save_forecast_summary": True,

    # Formatos de exportación
    "export_formats": [
        "parquet",
        "csv",
        "json"
    ],

    # Control de calidad
    "validate_outputs": True,
    "reproducible": True

}

# Validación
required_products = [
    "forecast_name",
    "forecast_version",
    "official_model",
    "prediction_target",
    "forecast_strategy",
    "forecast_mode",
    "historical_period",
    "forecast_period",
    "forecast_years",
    "forecast_horizon",
    "feature_columns",
    "target_columns"
]

missing_products = [
    product
    for product in required_products
    if product not in FORECAST_CONFIG
]

if missing_products:
    raise ValueError(
        "FORECAST_CONFIG está incompleto: "
        f"{missing_products}"
    )

for product in required_products:

    if FORECAST_CONFIG[product] is None:
        raise ValueError(
            f"'{product}' es inválido."
        )

if len(FORECAST_CONFIG["feature_columns"]) == 0:
    raise ValueError(
        "No existen variables predictoras."
    )

if len(FORECAST_CONFIG["target_columns"]) == 0:
    raise ValueError(
        "No existen variables objetivo."
    )

historical_start, historical_end = (
    FORECAST_CONFIG["historical_period"]
)

forecast_start, forecast_end = (
    FORECAST_CONFIG["forecast_period"]
)

if historical_end >= forecast_start:
    raise ValueError(
        "El período histórico debe finalizar antes del "
        "inicio del Forecast."
    )

if len(FORECAST_CONFIG["forecast_years"]) != FORECAST_CONFIG["forecast_horizon"]:
    raise ValueError(
        "El horizonte del Forecast no coincide con los años definidos."
    )

print("-" * 80)
print("Bloque 2. Configuración del Forecasting construida correctamente.")

# BLOQUE 3. Recuperación del Modelo Oficial -------------------------------------
## Objetivo: Recuperar los artefactos oficiales del Modelo GraphSAGE
# seleccionados durante el Benchmark Científico para reconstruir el Modelo
# Oficial que será utilizado durante el proceso de Forecasting Espacio-
# Temporal.
#### Producto:
# - OFFICIAL_MODEL construido y validado correctamente.
#### Responde:
# ¿Se recuperó correctamente el Modelo Oficial requerido para ejecutar el
# protocolo científico de Forecasting Espacio-Temporal?

print("-" * 80)
print("Bloque 3. Recuperación del Modelo Oficial.")

def load_official_model():

    # Validación de archivos
    if not OFFICIAL_MODEL_TORCH_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el Modelo Oficial: {OFFICIAL_MODEL_TORCH_FILE}"
        )

    if not OFFICIAL_MODEL_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró la configuración oficial: {OFFICIAL_MODEL_CONFIG_FILE}"
        )

    if not OFFICIAL_MODEL_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"No se encontraron los metadatos oficiales: "
            f"{OFFICIAL_MODEL_METADATA_FILE}"
        )

    # Recuperación de la configuración
    with open(
        OFFICIAL_MODEL_CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        model_config = json.load(file)

    # Validación de la configuración
    if not isinstance(model_config, dict):
        raise TypeError(
            "model_config debe ser un diccionario."
        )

    required_config_keys = [
    "model_name",
    "family",
    "model_config"
]
    
    missing_config_keys = [
        key
        for key in required_config_keys
        if key not in model_config
    ]

    if missing_config_keys:
        raise ValueError(
            "La configuración del Modelo Oficial está incompleta: "
            f"{missing_config_keys}"
        )

    # Recuperación de metadatos
    with open(
        OFFICIAL_MODEL_METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        model_metadata = json.load(file)

    # Validación de metadatos
    if not isinstance(model_metadata, dict):
        raise TypeError(
            "model_metadata debe ser un diccionario."
        )

    # Recuperación del checkpoint
    checkpoint = torch.load(
        OFFICIAL_MODEL_TORCH_FILE,
        map_location="cpu",
        weights_only=False
    )

    # Validación del checkpoint
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "El checkpoint debe ser un diccionario."
        )

    required_checkpoint_keys = [
        "model_state_dict"
    ]

    if len(checkpoint["model_state_dict"]) == 0:
        raise ValueError(
            "model_state_dict está vacío."
        )


    missing_checkpoint_keys = [
        key
        for key in required_checkpoint_keys
        if key not in checkpoint
    ]

    if missing_checkpoint_keys:
        raise ValueError(
            "El checkpoint del Modelo Oficial está incompleto: "
            f"{missing_checkpoint_keys}"
        )

    # Dispositivo
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Construcción
    official_model = {

        "checkpoint": checkpoint,

        "trained_model": checkpoint["model_state_dict"],

        "model_config": model_config,

        "model_metadata": model_metadata,

        "model_name": model_config["model_name"],

        "model_family": model_config["family"],

        "model_version": model_config.get(
            "model_version",
            "1.0.0"
        ),

        "device": device,

        "status": "OFFICIAL_MODEL_RECOVERED"

    }

    # Validación
    if not isinstance(official_model, dict):
        raise TypeError(
            "OFFICIAL_MODEL debe ser un diccionario."
        )

    required_products = [
        "checkpoint",
        "trained_model",
        "model_config",
        "model_metadata",
        "model_name",
        "model_family",
        "model_version",
        "device",
        "status"
    ]

    missing_products = [
        product
        for product in required_products
        if product not in official_model
    ]

    if missing_products:
        raise ValueError(
            "OFFICIAL_MODEL está incompleto: "
            f"{missing_products}"
        )

    for product in required_products:

        if official_model[product] is None:
            raise ValueError(
                f"'{product}' es inválido."
            )

    if len(official_model["trained_model"]) == 0:
        raise ValueError(
            "El modelo entrenado no contiene parámetros."
        )

    if not isinstance(
        official_model["device"],
        torch.device
    ):
        raise TypeError(
            "device debe ser un objeto torch.device."
        )

    if official_model["status"] != "OFFICIAL_MODEL_RECOVERED":
        raise ValueError(
            "El estado del Modelo Oficial es inválido."
        )

    return official_model


OFFICIAL_MODEL = load_official_model()

print("-" * 80)
print("Bloque 3. Modelo Oficial recuperado correctamente.")

with open(
    OFFICIAL_MODEL_CONFIG_FILE,
    "r",
    encoding="utf-8"
) as file:

    model_config = json.load(file)

print(model_config)

# BLOQUE 4. Recuperación de los Datos Oficiales ---------------------------------
## Objetivo: Recuperar los datos oficiales generados durante el Benchmark
# Científico requeridos para ejecutar el proceso de Forecasting
# Espacio-Temporal mediante el Modelo Oficial GraphSAGE.
#### Producto:
# - FORECAST_DATA construido y validado correctamente.
#### Responde:
# ¿Se recuperaron correctamente los datos oficiales requeridos para ejecutar
# el protocolo científico de Forecasting Espacio-Temporal?

print("-" * 80)
print("Bloque 4. Recuperación de los Datos Oficiales.")

def load_forecast_data():

    # Validación
    if not BENCHMARK_DATA_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró BenchmarkData: {BENCHMARK_DATA_FILE}"
        )

    # Recuperación
    benchmark_data = joblib.load(
        BENCHMARK_DATA_FILE
    )

    # Validación
    if not isinstance(
        benchmark_data,
        dict
    ):
        raise TypeError(
            "BenchmarkData debe ser un diccionario."
        )

    required_keys = [
        "graphs",
        "x_train",
        "y_train",
        "x_validation",
        "y_validation",
        "x_test",
        "y_test",
        "train_index",
        "validation_index",
        "test_index",
        "scaler"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in benchmark_data
    ]

    if missing_keys:
        raise ValueError(
            "BenchmarkData está incompleto: "
            f"{missing_keys}"
        )

    for key in required_keys:

        if benchmark_data[key] is None:
            raise ValueError(
                f"'{key}' es inválido."
            )

    x_test = benchmark_data["x_test"]

    if hasattr(
        x_test,
        "shape"
    ):
        n_observations = x_test.shape[0]

    else:
        n_observations = len(x_test)

    # Construcción
    forecast_data = {

        "benchmark_data": benchmark_data,

        "graphs": benchmark_data["graphs"],

        "x_test": x_test,

        "y_test": benchmark_data["y_test"],

        "test_index": benchmark_data["test_index"],

        "n_graphs": len(
            benchmark_data["graphs"]
        ),

        "n_observations": n_observations,

        "historical_period":
            FORECAST_CONFIG["historical_period"],

        "forecast_period":
            FORECAST_CONFIG["forecast_period"],

        "status":
            "FORECAST_DATA_READY"

    }

    # Validación
    if not isinstance(
        forecast_data,
        dict
    ):
        raise TypeError(
            "FORECAST_DATA debe ser un diccionario."
        )

    required_products = [
        "benchmark_data",
        "graphs",
        "x_test",
        "y_test",
        "test_index",
        "n_graphs",
        "n_observations",
        "historical_period",
        "forecast_period",
        "status"
    ]

    missing_products = [
        product
        for product in required_products
        if product not in forecast_data
    ]

    if missing_products:
        raise ValueError(
            "FORECAST_DATA está incompleto: "
            f"{missing_products}"
        )

    for product in required_products:

        if forecast_data[product] is None:
            raise ValueError(
                f"'{product}' es inválido."
            )

    if forecast_data["n_graphs"] <= 0:
        raise ValueError(
            "La colección GraphData está vacía."
        )

    if forecast_data["n_observations"] <= 0:
        raise ValueError(
            "No existen observaciones para Forecasting."
        )

    if forecast_data["status"] != "FORECAST_DATA_READY":
        raise ValueError(
            "El estado de FORECAST_DATA es inválido."
        )

    return forecast_data


FORECAST_DATA = load_forecast_data()

print("-" * 80)
print("Bloque 4. Datos Oficiales recuperados correctamente.")

# BLOQUE 5. Reconstrucción del Modelo Oficial ------------------------------
## Objetivo: Reconstruir la arquitectura oficial del Modelo GraphSAGE
# utilizando la configuración recuperada y el estado entrenado del Modelo
# Oficial para ejecutar el proceso de Forecasting Espacio-Temporal.
#### Producto:
# - FORECAST_MODEL construido y validado correctamente.
#### Responde:
# ¿Se reconstruyó correctamente el Modelo Oficial requerido para ejecutar
# el protocolo científico de Forecasting Espacio-Temporal?

print("-" * 80)
print("Bloque 5. Reconstrucción del Modelo Oficial.")

def build_forecast_model():

    # Validación
    if OFFICIAL_MODEL["status"] != "OFFICIAL_MODEL_RECOVERED":
        raise ValueError(
            "El Modelo Oficial no está disponible."
        )

    if FORECAST_DATA["status"] != "FORECAST_DATA_READY":
        raise ValueError(
            "Los datos oficiales del Forecasting no están disponibles."
        )

    # Recuperación
    model_config = OFFICIAL_MODEL["model_config"]

    trained_model = OFFICIAL_MODEL["trained_model"]

    graphs = FORECAST_DATA["graphs"]

    input_channels = graphs[0].num_node_features

    output_channels = len(TARGET_COLUMNS)

    # Adaptación de la configuración oficial
    gnn_config = {

        "model_name":
            model_config["model_name"],

        "hidden_channels":
            model_config["model_config"]["hidden_channels"],

        "dropout":
            model_config["model_config"]["dropout"]

    }

    # Construcción
    model = build_gnn_model(
        model_config=gnn_config,
        input_channels=input_channels,
        output_channels=output_channels
    )

    model.load_state_dict(
        trained_model
    )

    model.to(
        OFFICIAL_MODEL["device"]
    )

    model.eval()

    forecast_model = {

        "model": model,

        "model_config": gnn_config,

        "model_name": OFFICIAL_MODEL["model_name"],

        "model_family": OFFICIAL_MODEL["model_family"],

        "model_version": OFFICIAL_MODEL["model_version"],

        "input_channels": input_channels,

        "output_channels": output_channels,

        "device": OFFICIAL_MODEL["device"],

        "status": "MODEL_READY"

    }

    # Validación
    if not isinstance(
        forecast_model,
        dict
    ):
        raise TypeError(
            "FORECAST_MODEL debe ser un diccionario."
        )

    required_products = [
        "model",
        "model_config",
        "model_name",
        "model_family",
        "model_version",
        "input_channels",
        "output_channels",
        "device",
        "status"
    ]

    missing_products = [
        product
        for product in required_products
        if product not in forecast_model
    ]

    if missing_products:
        raise ValueError(
            "FORECAST_MODEL está incompleto: "
            f"{missing_products}"
        )

    for product in required_products:

        if forecast_model[product] is None:
            raise ValueError(
                f"'{product}' es inválido."
            )

    if forecast_model["input_channels"] <= 0:
        raise ValueError(
            "input_channels debe ser mayor que cero."
        )

    if forecast_model["output_channels"] <= 0:
        raise ValueError(
            "output_channels debe ser mayor que cero."
        )

    if not isinstance(
        forecast_model["device"],
        torch.device
    ):
        raise TypeError(
            "device debe ser un objeto torch.device."
        )

    if forecast_model["status"] != "MODEL_READY":
        raise ValueError(
            "El estado de FORECAST_MODEL es inválido."
        )

    return forecast_model


FORECAST_MODEL = build_forecast_model()

print("-" * 80)
print("Bloque 5. Modelo Oficial reconstruido correctamente.")

# BLOQUE 6. Construcción del Dataset Forecast ------------------------------
## Objetivo: Construir el Dataset Oficial de Forecast utilizando el último
# GraphData histórico recuperado durante el Benchmark Científico para
# ejecutar el proceso de predicción del Modelo Oficial GraphSAGE.
#### Producto:
# - FORECAST_DATASET construido y validado correctamente.
#### Responde:
# ¿Se construyó correctamente el Dataset Oficial requerido para ejecutar
# el protocolo científico de Forecasting Espacio-Temporal?

print("-" * 80)
print("Bloque 6. Construcción del Dataset Forecast.")

def build_forecast_dataset():

    # Validación
    if FORECAST_DATA["status"] != "FORECAST_DATA_READY":
        raise ValueError(
            "Los datos oficiales del Forecasting no están disponibles."
        )

    if FORECAST_MODEL["status"] != "MODEL_READY":
        raise ValueError(
            "El Modelo Oficial no está disponible."
        )

    # Recuperación
    reference_graph = FORECAST_DATA["graphs"][-1]

    x_forecast = reference_graph.x

    edge_index = reference_graph.edge_index

    forecast_years = FORECAST_CONFIG["forecast_years"]

    n_observations = x_forecast.shape[0]

    n_features = x_forecast.shape[1]

    # Construcción
    forecast_dataset = {

        "x_forecast": x_forecast,

        "edge_index": edge_index,

        "forecast_years": forecast_years,

        "forecast_period":
            FORECAST_CONFIG["forecast_period"],

        "forecast_horizon":
            FORECAST_CONFIG["forecast_horizon"],

        "n_years":
            len(forecast_years),

        "n_observations":
            n_observations,

        "n_features":
            n_features,

        "status":
            "FORECAST_DATASET_READY"

    }

    # Validación
    if not isinstance(
        forecast_dataset,
        dict
    ):
        raise TypeError(
            "FORECAST_DATASET debe ser un diccionario."
        )

    required_products = [
        "x_forecast",
        "edge_index",
        "forecast_years",
        "forecast_period",
        "forecast_horizon",
        "n_years",
        "n_observations",
        "n_features",
        "status"
    ]

    missing_products = [
        product
        for product in required_products
        if product not in forecast_dataset
    ]

    if missing_products:
        raise ValueError(
            "FORECAST_DATASET está incompleto: "
            f"{missing_products}"
        )

    for product in required_products:

        if forecast_dataset[product] is None:
            raise ValueError(
                f"'{product}' es inválido."
            )

    if forecast_dataset["n_years"] <= 0:
        raise ValueError(
            "El horizonte de Forecasting es inválido."
        )

    if (
        forecast_dataset["forecast_horizon"]
        != forecast_dataset["n_years"]
    ):
        raise ValueError(
            "forecast_horizon no coincide con el número de años."
        )

    if forecast_dataset["n_observations"] <= 0:
        raise ValueError(
            "No existen nodos para Forecasting."
        )

    if forecast_dataset["n_features"] <= 0:
        raise ValueError(
            "No existen variables predictoras."
        )

    if (
        forecast_dataset["edge_index"].shape[0]
        != 2
    ):
        raise ValueError(
            "edge_index posee una estructura inválida."
        )

    if forecast_dataset["status"] != "FORECAST_DATASET_READY":
        raise ValueError(
            "El estado de FORECAST_DATASET es inválido."
        )

    return forecast_dataset


FORECAST_DATASET = build_forecast_dataset()

print("-" * 80)
print("Bloque 6. Dataset Forecast construido correctamente.")

# BLOQUE 7. Construcción del GraphData Forecast -----------------------------
## Objetivo: Construir la colección oficial de GraphData Forecast utilizando
# la estructura espacial del último GraphData histórico y los datos oficiales
# recuperados para preparar el proceso de predicción mediante el Modelo
# Oficial GraphSAGE.
#### Producto:
# - FORECAST_GRAPHS construido y validado correctamente.
#### Responde:
# ¿Se construyó correctamente la colección oficial de GraphData Forecast
# requerida para ejecutar el protocolo científico de Forecasting
# Espacio-Temporal?

print("-" * 80)
print("Bloque 7. Construcción del GraphData Forecast.")

def build_forecast_graphs():

    # Validación
    if FORECAST_DATASET["status"] != "FORECAST_DATASET_READY":
        raise ValueError(
            "El Dataset Forecast no está disponible."
        )

    if FORECAST_DATA["status"] != "FORECAST_DATA_READY":
        raise ValueError(
            "Los datos oficiales del Forecasting no están disponibles."
        )

    # Recuperación
    x_forecast = FORECAST_DATASET["x_forecast"]

    reference_graph = FORECAST_DATA["graphs"][-1]

    forecast_years = FORECAST_DATASET["forecast_years"]

    expected_nodes = reference_graph.num_nodes

    # Construcción
    graphs = []

    for year in forecast_years:

        graph = reference_graph.clone()

        if isinstance(
            x_forecast,
            torch.Tensor
        ):

            graph.x = x_forecast.clone().float()

        else:

            graph.x = torch.tensor(
                x_forecast,
                dtype=torch.float32
            )

        received_nodes = graph.x.shape[0]

        if expected_nodes != received_nodes:
            raise ValueError(
                "El número de nodos del GraphData no coincide "
                "con las observaciones del Dataset Forecast."
            )

        graph.year = year

        graph.forecast = True

        graphs.append(graph)

    forecast_graphs = {

        "graphs": graphs,

        "reference_graph": reference_graph,

        "forecast_years": forecast_years,

        "n_graphs": len(graphs),

        "n_nodes": expected_nodes,

        "n_features": graph.x.shape[1],

        "status": "FORECAST_GRAPHS_READY"

    }

    # Validación
    if not isinstance(
        forecast_graphs,
        dict
    ):
        raise TypeError(
            "FORECAST_GRAPHS debe ser un diccionario."
        )

    required_products = [
        "graphs",
        "reference_graph",
        "forecast_years",
        "n_graphs",
        "n_nodes",
        "n_features",
        "status"
    ]

    missing_products = [
        product
        for product in required_products
        if product not in forecast_graphs
    ]

    if missing_products:
        raise ValueError(
            "FORECAST_GRAPHS está incompleto: "
            f"{missing_products}"
        )

    for product in required_products:

        if forecast_graphs[product] is None:
            raise ValueError(
                f"'{product}' es inválido."
            )

    if forecast_graphs["n_graphs"] <= 0:
        raise ValueError(
            "No se construyeron GraphData para Forecasting."
        )

    if forecast_graphs["n_nodes"] <= 0:
        raise ValueError(
            "El número de nodos es inválido."
        )

    if forecast_graphs["n_features"] <= 0:
        raise ValueError(
            "El número de variables predictoras es inválido."
        )

    if forecast_graphs["status"] != "FORECAST_GRAPHS_READY":
        raise ValueError(
            "El estado de FORECAST_GRAPHS es inválido."
        )

    return forecast_graphs


FORECAST_GRAPHS = build_forecast_graphs()

print("-" * 80)
print("Bloque 7. GraphData Forecast construido correctamente.")

# BLOQUE 8. Ejecución del Forecast ----------------------------------------------
## Objetivo: Ejecutar el protocolo científico de Forecasting Espacio-Temporal
# utilizando el Modelo Oficial GraphSAGE y la colección oficial de GraphData
# Forecast para generar las predicciones correspondientes al horizonte
# temporal definido.
#### Producto:
# - FORECAST_RESULT construido y validado correctamente.
#### Responde:
# ¿Se ejecutó correctamente el protocolo científico de Forecasting
# Espacio-Temporal?

print("-" * 80)
print("Bloque 8. Ejecución del Forecast.")

def run_forecast():

    # Validación
    if FORECAST_MODEL["status"] != "MODEL_READY":
        raise ValueError(
            "El Modelo Oficial no está disponible."
        )

    if FORECAST_GRAPHS["status"] != "FORECAST_GRAPHS_READY":
        raise ValueError(
            "La colección GraphData Forecast no está disponible."
        )

    # Recuperación
    model = FORECAST_MODEL["model"]

    device = FORECAST_MODEL["device"]

    graphs = FORECAST_GRAPHS["graphs"]

    forecast_years = FORECAST_GRAPHS["forecast_years"]

    # Construcción
    predictions = []

    model.to(device)

    model.eval()

    inference_start = datetime.now()

    with torch.no_grad():

        for graph, year in zip(
            graphs,
            forecast_years
        ):

            graph = graph.to(device)

            prediction = model(
                graph.x,
                graph.edge_index
            )

            expected_nodes = graph.num_nodes

            received_nodes = prediction.shape[0]

            if expected_nodes != received_nodes:
                raise ValueError(
                    "El número de predicciones no coincide "
                    "con el número de nodos."
                )

            prediction_cpu = prediction.detach().cpu()

            predictions.append({

                "year": year,

                "prediction_tensor": prediction_cpu,

                "prediction": prediction_cpu.numpy(),

                "n_nodes": received_nodes

            })

    inference_time = (
        datetime.now() - inference_start
    ).total_seconds()

    forecast_result = {

        "forecast_strategy":
            FORECAST_CONFIG["forecast_strategy"],

        "forecast_mode":
            FORECAST_CONFIG["forecast_mode"],

        "predictions":
            predictions,

        "forecast_years":
            forecast_years,

        "n_graphs":
            len(graphs),

        "n_predictions":
            sum(
                prediction["n_nodes"]
                for prediction in predictions
            ),

        "inference_time":
            inference_time,

        "status":
            "FORECAST_COMPLETED"

    }

    # Validación
    if not isinstance(
        forecast_result,
        dict
    ):
        raise TypeError(
            "FORECAST_RESULT debe ser un diccionario."
        )

    required_products = [
        "forecast_strategy",
        "forecast_mode",
        "predictions",
        "forecast_years",
        "n_graphs",
        "n_predictions",
        "inference_time",
        "status"
    ]

    missing_products = [
        product
        for product in required_products
        if product not in forecast_result
    ]

    if missing_products:
        raise ValueError(
            "FORECAST_RESULT está incompleto: "
            f"{missing_products}"
        )

    for product in required_products:

        if forecast_result[product] is None:
            raise ValueError(
                f"'{product}' es inválido."
            )

    if forecast_result["n_graphs"] <= 0:
        raise ValueError(
            "No existen GraphData para Forecasting."
        )

    if forecast_result["n_predictions"] <= 0:
        raise ValueError(
            "No se generaron predicciones."
        )

    if forecast_result["inference_time"] < 0:
        raise ValueError(
            "El tiempo de inferencia es inválido."
        )

    if forecast_result["status"] != "FORECAST_COMPLETED":
        raise ValueError(
            "El estado de FORECAST_RESULT es inválido."
        )

    return forecast_result


FORECAST_RESULT = run_forecast()

print("-" * 80)
print("Bloque 8. Forecast ejecutado correctamente.")

# BLOQUE 9. Consolidación del Forecast ------------------------------------------
## Objetivo: Consolidar las predicciones generadas durante el proceso oficial
# de Forecasting Espacio-Temporal para construir el producto científico
# oficial que será utilizado por los módulos de exportación, reportes,
# Dashboard y la plataforma GeoAI.
#### Producto:
# - FORECAST_SUMMARY construido y validado correctamente.
#### Responde:
# ¿Se consolidaron correctamente las predicciones generadas durante el
# Forecasting Espacio-Temporal?

print("-" * 80)
print("Bloque 9. Consolidación del Forecast.")

def build_forecast_summary():

    # Validación
    if FORECAST_RESULT["status"] != "FORECAST_COMPLETED":
        raise ValueError(
            "El Forecasting no ha sido ejecutado."
        )

    # Recuperación
    predictions = FORECAST_RESULT["predictions"]

    forecast_years = FORECAST_RESULT["forecast_years"]

    # Construcción
    forecast_summary = {

        "forecast_name":
            FORECAST_CONFIG["forecast_name"],

        "forecast_version":
            FORECAST_CONFIG["forecast_version"],

        "official_model":
            FORECAST_CONFIG["official_model"],

        "forecast_strategy":
            FORECAST_RESULT["forecast_strategy"],

        "forecast_mode":
            FORECAST_RESULT["forecast_mode"],

        "prediction_target":
            FORECAST_CONFIG["prediction_target"],

        "forecast_period":
            FORECAST_CONFIG["forecast_period"],

        "forecast_years":
            forecast_years,

        "forecast_horizon":
            len(forecast_years),

        "predictions":
            predictions,

        "n_graphs":
            FORECAST_RESULT["n_graphs"],

        "n_predictions":
            FORECAST_RESULT["n_predictions"],

        "inference_time":
            FORECAST_RESULT["inference_time"],

        "status":
            "FORECAST_SUMMARY_READY"

    }

    # Validación
    if not isinstance(
        forecast_summary,
        dict
    ):
        raise TypeError(
            "FORECAST_SUMMARY debe ser un diccionario."
        )

    required_products = [
        "forecast_name",
        "forecast_version",
        "official_model",
        "forecast_strategy",
        "forecast_mode",
        "prediction_target",
        "forecast_period",
        "forecast_years",
        "forecast_horizon",
        "predictions",
        "n_graphs",
        "n_predictions",
        "inference_time",
        "status"
    ]

    missing_products = [
        product
        for product in required_products
        if product not in forecast_summary
    ]

    if missing_products:
        raise ValueError(
            "FORECAST_SUMMARY está incompleto: "
            f"{missing_products}"
        )

    for product in required_products:

        if forecast_summary[product] is None:
            raise ValueError(
                f"'{product}' es inválido."
            )

    if forecast_summary["forecast_horizon"] <= 0:
        raise ValueError(
            "El horizonte de Forecasting es inválido."
        )

    if forecast_summary["n_graphs"] <= 0:
        raise ValueError(
            "El número de GraphData es inválido."
        )

    if forecast_summary["n_predictions"] <= 0:
        raise ValueError(
            "No existen predicciones para consolidar."
        )

    if forecast_summary["inference_time"] < 0:
        raise ValueError(
            "El tiempo de inferencia es inválido."
        )

    if forecast_summary["status"] != "FORECAST_SUMMARY_READY":
        raise ValueError(
            "El estado de FORECAST_SUMMARY es inválido."
        )

    return forecast_summary


FORECAST_SUMMARY = build_forecast_summary()

print("-" * 80)
print("Bloque 9. Forecast consolidado correctamente.")

# BLOQUE 10. Exportación de Resultados ------------------------------------------
## Objetivo: Exportar los productos oficiales generados durante el protocolo
# científico de Forecasting Espacio-Temporal para su utilización por los
# módulos de reportes, Dashboard y la plataforma GeoAI.
#### Producto:
# - FORECAST_EXPORT construido y validado correctamente.
#### Responde:
# ¿Se exportaron correctamente los productos oficiales del Forecasting
# Espacio-Temporal?

print("-" * 80)
print("Bloque 10. Exportación de Resultados.")

def export_forecast_results():

    # Validación
    if FORECAST_SUMMARY["status"] != "FORECAST_SUMMARY_READY":
        raise ValueError(
            "El resumen del Forecast no está disponible."
        )

    # Crear directorio
    FORECAST_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Recuperación
    predictions = FORECAST_SUMMARY["predictions"]

    # Construcción
    records = []

    for prediction in predictions:

        prediction_values = (
            prediction["prediction"]
            .reshape(-1)
            .tolist()
        )

        for node_id, value in enumerate(prediction_values):

            records.append({

                "forecast_year": prediction["year"],

                "node_id": node_id,

                "prediction": float(value)

            })

    forecast_dataframe = pd.DataFrame(
        records
    )

    # Archivos de salida
    parquet_file = (
        FORECAST_RESULTS_DIR
        / "forecast_predictions.parquet"
    )

    csv_file = (
        FORECAST_RESULTS_DIR
        / "forecast_predictions.csv"
    )

    xlsx_file = (
        FORECAST_RESULTS_DIR
        / "forecast_predictions.xlsx"
    )

    json_file = (
        FORECAST_RESULTS_DIR
        / "forecast_summary.json"
    )

    # Exportación
    forecast_dataframe.to_parquet(
        parquet_file,
        index=False
    )

    forecast_dataframe.to_csv(
        csv_file,
        index=False,
        encoding="utf-8-sig"
    )

    forecast_dataframe.to_excel(
        xlsx_file,
        index=False
    )

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            FORECAST_SUMMARY,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    # Construcción
    forecast_export = {

        "parquet_file": parquet_file,

        "csv_file": csv_file,

        "xlsx_file": xlsx_file,

        "json_file": json_file,

        "n_predictions": len(
            forecast_dataframe
        ),

        "status": "FORECAST_RESULTS_EXPORTED"

    }

    # Validación
    if not isinstance(
        forecast_export,
        dict
    ):
        raise TypeError(
            "FORECAST_EXPORT debe ser un diccionario."
        )

    required_products = [
        "parquet_file",
        "csv_file",
        "xlsx_file",
        "json_file",
        "n_predictions",
        "status"
    ]

    missing_products = [
        product
        for product in required_products
        if product not in forecast_export
    ]

    if missing_products:
        raise ValueError(
            "FORECAST_EXPORT está incompleto: "
            f"{missing_products}"
        )

    for product in required_products:

        if forecast_export[product] is None:
            raise ValueError(
                f"'{product}' es inválido."
            )

    if not forecast_export["parquet_file"].exists():
        raise FileNotFoundError(
            "No fue posible exportar el archivo Parquet."
        )

    if not forecast_export["csv_file"].exists():
        raise FileNotFoundError(
            "No fue posible exportar el archivo CSV."
        )

    if not forecast_export["xlsx_file"].exists():
        raise FileNotFoundError(
            "No fue posible exportar el archivo Excel."
        )

    if not forecast_export["json_file"].exists():
        raise FileNotFoundError(
            "No fue posible exportar el archivo JSON."
        )

    if forecast_export["n_predictions"] <= 0:
        raise ValueError(
            "No existen predicciones para exportar."
        )

    if forecast_export["status"] != "FORECAST_RESULTS_EXPORTED":
        raise ValueError(
            "El estado de FORECAST_EXPORT es inválido."
        )

    return forecast_export


FORECAST_EXPORT = export_forecast_results()

print("-" * 80)
print("Bloque 10. Resultados exportados correctamente.")

# BLOQUE 11. Resumen Científico del Forecast -------------------------------
## Objetivo: Presentar el resumen científico del protocolo oficial de
# Forecasting Espacio-Temporal, consolidando la información del Modelo
# Oficial, la configuración utilizada, los resultados obtenidos y los
# productos científicos generados para la plataforma GeoAI.
#### Producto:
# - FORECAST_REPORT construido y validado correctamente.
#### Responde:
# ¿Finalizó correctamente el protocolo científico de Forecasting
# Espacio-Temporal?

print("-" * 80)
print("Bloque 11. Resumen Científico del Forecast.")

def build_forecast_report():

    # Validación
    if FORECAST_EXPORT["status"] != "FORECAST_RESULTS_EXPORTED":
        raise ValueError(
            "Los resultados del Forecast no fueron exportados."
        )

    # Recuperación
    forecast_years = FORECAST_SUMMARY["forecast_years"]

    forecast_report = {

        "forecast_name":
            FORECAST_CONFIG["forecast_name"],

        "forecast_version":
            FORECAST_CONFIG["forecast_version"],

        "official_model":
            FORECAST_CONFIG["official_model"],

        "forecast_strategy":
            FORECAST_CONFIG["forecast_strategy"],

        "forecast_mode":
            FORECAST_CONFIG["forecast_mode"],

        "prediction_target":
            FORECAST_CONFIG["prediction_target"],

        "historical_period":
            FORECAST_CONFIG["historical_period"],

        "forecast_period":
            FORECAST_CONFIG["forecast_period"],

        "forecast_years":
            forecast_years,

        "forecast_horizon":
            len(forecast_years),

        "n_graphs":
            FORECAST_SUMMARY["n_graphs"],

        "n_predictions":
            FORECAST_SUMMARY["n_predictions"],

        "inference_time":
            FORECAST_SUMMARY["inference_time"],

        "export_formats":
            FORECAST_CONFIG["export_formats"],

        "parquet_file":
            FORECAST_EXPORT["parquet_file"],

        "csv_file":
            FORECAST_EXPORT["csv_file"],

        "xlsx_file":
            FORECAST_EXPORT["xlsx_file"],

        "json_file":
            FORECAST_EXPORT["json_file"],

        "status":
            "FORECAST_FINISHED"

    }

    # Validación
    if not isinstance(
        forecast_report,
        dict
    ):
        raise TypeError(
            "FORECAST_REPORT debe ser un diccionario."
        )

    required_products = [
        "forecast_name",
        "forecast_version",
        "official_model",
        "forecast_strategy",
        "forecast_mode",
        "prediction_target",
        "historical_period",
        "forecast_period",
        "forecast_years",
        "forecast_horizon",
        "n_graphs",
        "n_predictions",
        "inference_time",
        "export_formats",
        "parquet_file",
        "csv_file",
        "xlsx_file",
        "json_file",
        "status"
    ]

    missing_products = [
        product
        for product in required_products
        if product not in forecast_report
    ]

    if missing_products:
        raise ValueError(
            "FORECAST_REPORT está incompleto: "
            f"{missing_products}"
        )

    for product in required_products:

        if forecast_report[product] is None:
            raise ValueError(
                f"'{product}' es inválido."
            )

    if forecast_report["forecast_horizon"] <= 0:
        raise ValueError(
            "El horizonte de Forecast es inválido."
        )

    if forecast_report["n_graphs"] <= 0:
        raise ValueError(
            "El número de GraphData es inválido."
        )

    if forecast_report["n_predictions"] <= 0:
        raise ValueError(
            "No existen predicciones."
        )

    if forecast_report["inference_time"] < 0:
        raise ValueError(
            "El tiempo de inferencia es inválido."
        )

    if forecast_report["status"] != "FORECAST_FINISHED":
        raise ValueError(
            "El estado del reporte es inválido."
        )

    return forecast_report


FORECAST_REPORT = build_forecast_report()

print("-" * 80)
print("RESUMEN CIENTÍFICO DEL FORECAST")
print("-" * 80)

print("PROTOCOLO CIENTÍFICO")
print(f"Modelo Oficial            : {FORECAST_REPORT['official_model']}")
print(f"Estrategia Forecast       : {FORECAST_REPORT['forecast_strategy']}")
print(f"Modo de Forecast          : {FORECAST_REPORT['forecast_mode']}")
print(f"Variable Objetivo         : {FORECAST_REPORT['prediction_target']}")

print("-" * 80)

print("COBERTURA TEMPORAL")
print(
    f"Período Histórico         : "
    f"{FORECAST_REPORT['historical_period'][0]} - "
    f"{FORECAST_REPORT['historical_period'][1]}"
)

print(
    f"Horizonte Forecast        : "
    f"{FORECAST_REPORT['forecast_period'][0]} - "
    f"{FORECAST_REPORT['forecast_period'][1]}"
)

print(
    f"Años Pronosticados        : "
    f"{FORECAST_REPORT['forecast_horizon']}"
)

print("-" * 80)

print("RESULTADOS DEL FORECAST")

print(
    f"GraphData Procesados      : "
    f"{FORECAST_REPORT['n_graphs']}"
)

print(
    f"Predicciones Generadas    : "
    f"{FORECAST_REPORT['n_predictions']}"
)

print(
    f"Tiempo de Inferencia (s)  : "
    f"{FORECAST_REPORT['inference_time']:.4f}"
)

print(
    f"Predicciones por Grafo    : "
    f"{FORECAST_REPORT['n_predictions'] / FORECAST_REPORT['n_graphs']:.0f}"
)

print("-" * 80)

print("PRODUCTOS GENERADOS")

print(
    f"Parquet                   : "
    f"{FORECAST_REPORT['parquet_file'].name}"
)

print(
    f"CSV                       : "
    f"{FORECAST_REPORT['csv_file'].name}"
)

print(
    f"Excel                     : "
    f"{FORECAST_REPORT['xlsx_file'].name}"
)

print(
    f"JSON                      : "
    f"{FORECAST_REPORT['json_file'].name}"
)

print("-" * 80)

print("INTERPRETACIÓN CIENTÍFICA")

print(
    f"El Modelo Oficial {FORECAST_REPORT['official_model']} "
    f"ejecutó correctamente el protocolo de Forecasting "
    f"Espacio-Temporal para el horizonte "
    f"{FORECAST_REPORT['forecast_period'][0]}-"
    f"{FORECAST_REPORT['forecast_period'][1]}, "
    f"procesando {FORECAST_REPORT['n_graphs']} GraphData "
    f"y generando {FORECAST_REPORT['n_predictions']} "
    f"predicciones espacio-temporales, las cuales fueron "
    f"exportadas en formatos Parquet, CSV, Excel y JSON "
    f"para su integración con la plataforma GeoAI."
)

print("-" * 80)

print("ESTADO DEL PROCESO")
print(f"Estado Final              : {FORECAST_REPORT['status']}")

print("-" * 80)
print("PROTOCOLO DE FORECASTING FINALIZADO EXITOSAMENTE")
print("-" * 80)

print("Bloque 11. Resumen Científico generado correctamente.")