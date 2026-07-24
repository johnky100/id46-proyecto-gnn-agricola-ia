# graph-03_train_model.py

# BLOQUE 1. Importaciones --------------------------------------------------
## Objetivo: Importar las dependencias necesarias para ejecutar el forecasting
# espacio-temporal utilizando el modelo oficial GraphSAGE, el Dataset
# Científico Certificado, la colección oficial de GraphData y los componentes
# científicos requeridos para generar predicciones, productos cartográficos,
# análisis geoespacial, explicabilidad y validación del forecasting.
### Producto:
# - Librerías cargadas correctamente.
### Responde:
# ¿Qué dependencias requiere el protocolo de forecasting para reconstruir el
# modelo oficial, generar predicciones espacio-temporales, producir mapas
# científicos, interpretar los resultados y consolidar la salida oficial del
# forecasting?

print("-" * 80)
print("Bloque 1. Importaciones.")

# Funciones del sistema
import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib

# Librerías científicas
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

# Configuración del proyecto
from src.python.config.paths import (
    DATASET_FILE,
    GRAPH_DATA_DIR,
    BEST_MODEL_CONFIG_FILE,
    BEST_MODEL_METADATA_FILE,
    BEST_MODEL_TORCH_FILE,
    BENCHMARK_RESULTS_FILE,
    TRAINING_SUMMARY_FILE,
)

from src.python.config.config_project import (
    PROJECT_SEED,
    TRAIN_CONFIG,
)

# Modelos Graph Neural Networks
from src.python.models.graph_neural_networks import (
    GNN_CONFIG,
    build_gnn_model,
    build_training_components,
    train_gnn,
    evaluate_gnn
)

warnings.filterwarnings("ignore")

torch.manual_seed(PROJECT_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(PROJECT_SEED)

print("-" * 80)
print("Bloque 1. Importaciones cargadas correctamente.")

# BLOQUE 2. Configuración --------------------------------------------------
## Objetivo: Definir la configuración oficial que gobernará el entrenamiento
# definitivo del modelo oficial seleccionado durante el Benchmark Científico,
# garantizando la reproducibilidad y consistencia del protocolo experimental.
#### Producto:
# - train_configuration
#### Responde:
# ¿El entrenamiento definitivo dispone de una configuración oficial,
# reproducible y consistente para generar el modelo oficial del proyecto?

print("-" * 80)
print("Bloque 2. Configuración.")

def build_train_configuration() -> dict:
    """
    Construye y valida la configuración oficial del entrenamiento definitivo.

    Returns
    -------
    dict
        Configuración oficial del entrenamiento.
    """

    # Construcción de la configuración
    train_configuration = {
        "training_name": "final_training",
        "training_version": "1.0",
        "use_full_dataset": True,
        "save_trained_model": True,
        "save_training_summary": True,
        "save_training_metrics": True,
        "save_training_metadata": True,
        "overwrite_existing_model": True,
        "random_state": PROJECT_SEED,
        "verbose": True
    }

    # Validación de la estructura
    required_keys = [
        "training_name",
        "training_version",
        "use_full_dataset",
        "save_trained_model",
        "save_training_summary",
        "save_training_metrics",
        "save_training_metadata",
        "overwrite_existing_model",
        "random_state",
        "verbose"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in train_configuration
    ]

    if missing_keys:
        raise RuntimeError(
            "La configuración oficial del entrenamiento está incompleta: "
            f"{missing_keys}"
        )

    # Validación de tipos
    if not isinstance(train_configuration["training_name"], str):
        raise TypeError("training_name debe ser una cadena.")

    if not isinstance(train_configuration["training_version"], str):
        raise TypeError("training_version debe ser una cadena.")

    if not isinstance(train_configuration["random_state"], int):
        raise TypeError("random_state debe ser un entero.")

    boolean_keys = [
        "use_full_dataset",
        "save_trained_model",
        "save_training_summary",
        "save_training_metrics",
        "save_training_metadata",
        "overwrite_existing_model",
        "verbose"
    ]

    for key in boolean_keys:

        if not isinstance(train_configuration[key], bool):
            raise TypeError(
                f"{key} debe ser un valor booleano."
            )

    # Validación del contenido
    if not train_configuration["training_name"].strip():
        raise ValueError(
            "training_name está vacío."
        )

    if not train_configuration["training_version"].strip():
        raise ValueError(
            "training_version está vacío."
        )

    return train_configuration

# Construcción del producto
TRAIN_CONFIGURATION = build_train_configuration()

print("-" * 80)
print("Bloque 2. Configuración cargada correctamente.")

# BLOQUE 3. Carga del Modelo Oficial GraphSAGE ------------------------------
## Objetivo: Recuperar la configuración oficial y los resultados del modelo
# GraphSAGE seleccionado durante el Benchmark Científico para ejecutar el
# entrenamiento definitivo del modelo oficial del proyecto.
#### Producto:
# - official_model
#### Responde:
# ¿La configuración y los resultados oficiales del modelo GraphSAGE fueron
# recuperados correctamente para iniciar el entrenamiento definitivo?

print("-" * 80)
print("Bloque 3. Carga del Modelo Oficial GraphSAGE.")

def load_official_model() -> dict:
    """
    Recupera y valida la configuración oficial y los resultados del modelo
    GraphSAGE seleccionado durante el Benchmark Científico.

    Returns
    -------
    dict
        Información oficial del modelo GraphSAGE.
    """

    # Validación de archivos
    if not BEST_MODEL_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"No fue posible localizar {BEST_MODEL_CONFIG_FILE.name}."
        )

    if not BENCHMARK_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"No fue posible localizar {BENCHMARK_RESULTS_FILE.name}."
        )

    # Recuperación de la configuración oficial
    try:

        with open(
            BEST_MODEL_CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            benchmark_model = json.load(file)

    except Exception as error:

        raise RuntimeError(
            f"Error al cargar la configuración oficial del Benchmark: {error}"
        )

    # Validación
    for key in ("model_code", "model_name", "family"):

        if key not in benchmark_model:
            raise ValueError(
                f"La configuración del Benchmark no contiene '{key}'."
            )

    if benchmark_model["family"] != "graph_neural_networks":
        raise ValueError(
            "El entrenamiento definitivo únicamente admite Graph Neural Networks."
        )

    if benchmark_model["model_name"] != "graphsage":
        raise ValueError(
            "El modelo oficial del proyecto debe ser GraphSAGE."
        )

    if "graphsage" not in GNN_CONFIG:
        raise KeyError(
            "No se encontró la configuración oficial de GraphSAGE."
        )

    official_model_config = GNN_CONFIG["graphsage"]

    # Recuperación de resultados
    try:

        benchmark_results = joblib.load(
            BENCHMARK_RESULTS_FILE
        )

    except Exception as error:

        raise RuntimeError(
            f"Error al cargar los resultados del Benchmark: {error}"
        )

    if not benchmark_results:
        raise ValueError(
            "Los resultados del Benchmark están vacíos."
        )

    # Recuperación del modelo oficial
    try:

        official_model_result = next(
            result
            for result in benchmark_results
            if result["model_code"] == official_model_config["model_code"]
        )

    except StopIteration:

        raise ValueError(
            "No fue posible localizar los resultados oficiales de GraphSAGE."
        )

    # Validación
    for key in ("model_code", "model_name", "family", "model"):

        if key not in official_model_result:
            raise ValueError(
                f"El resultado oficial de GraphSAGE no contiene '{key}'."
            )

    # Retorno
    return {
        "official_model_config": official_model_config,
        "official_model_result": official_model_result,
        "benchmark_results": benchmark_results,
        "benchmark_summary": {
            "model_code": official_model_config["model_code"],
            "model_name": official_model_config["model_name"],
            "family": official_model_config["family"]
        }
    }

# Construcción del producto
OFFICIAL_MODEL = load_official_model()

official_model_config = OFFICIAL_MODEL["official_model_config"]
official_model_result = OFFICIAL_MODEL["official_model_result"]
benchmark_results = OFFICIAL_MODEL["benchmark_results"]
benchmark_summary = OFFICIAL_MODEL["benchmark_summary"]

print("-" * 80)
print("Bloque 3. Modelo Oficial GraphSAGE validado correctamente.")

# BLOQUE 4. Carga de los Datos Oficiales -----------------------------------
## Objetivo: Recuperar y validar el Dataset Científico y la colección oficial
# de GraphData que servirán como entradas para el entrenamiento definitivo
# del modelo oficial GraphSAGE.
#### Producto:
# - training_data
#### Responde:
# ¿El Dataset Científico y la colección oficial de GraphData fueron
# recuperados y validados correctamente para iniciar el entrenamiento
# definitivo del modelo oficial GraphSAGE?

print("-" * 80)
print("Bloque 4. Carga de los Datos Oficiales.")

def load_training_data() -> dict:
    """
    Recupera y valida el Dataset Científico y la colección oficial de
    GraphData utilizados durante el entrenamiento definitivo.

    Returns
    -------
    dict
        Datos oficiales del entrenamiento.
    """

    # Validación de archivos
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"No fue posible localizar el Dataset Científico: {DATASET_FILE}"
        )

    if not GRAPH_DATA_DIR.exists():
        raise FileNotFoundError(
            f"No fue posible localizar el directorio GraphData: {GRAPH_DATA_DIR}"
        )

    # Recuperación del Dataset Científico
    try:

        dataset = pd.read_parquet(DATASET_FILE)

    except Exception as error:

        raise RuntimeError(
            f"Error al cargar el Dataset Científico: {error}"
        )

    # Recuperación de los GraphData
    graph_files = sorted(
        file
        for file in GRAPH_DATA_DIR.glob("graph_data_*.pt")
        if file.stem != "graph_data_collection"
    )

    if not graph_files:
        raise FileNotFoundError(
            "No se encontraron GraphData oficiales para el entrenamiento."
        )

    graphs = []

    for graph_file in graph_files:

        try:

            graphs.append(
                torch.load(
                    graph_file,
                    weights_only=False
                )
            )

        except Exception as error:

            raise RuntimeError(
                f"Error al cargar {graph_file.name}: {error}"
            )

    # Validación
    if dataset.empty:
        raise ValueError(
            "El Dataset Científico está vacío."
        )

    if not graphs:
        raise ValueError(
            "La colección oficial de GraphData está vacía."
        )

    for index, graph in enumerate(graphs, start=1):

        if graph is None:
            raise ValueError(
                f"GraphData #{index} no fue cargado correctamente."
            )

        if not hasattr(graph, "num_node_features"):
            raise TypeError(
                f"GraphData #{index} no corresponde a un objeto válido de PyTorch Geometric."
            )

    # Retorno
    return {
        "dataset": dataset,
        "graphs": graphs
    }


# Construcción del producto
TRAINING_DATA = load_training_data()

training_data = TRAINING_DATA

print("-" * 80)
print("Bloque 4. Carga de los Datos Oficiales validado correctamente.")

# BLOQUE 5. Construcción de las Entradas del Entrenamiento -----------------
## Objetivo: Construir y validar las entradas oficiales requeridas para el
# entrenamiento definitivo del modelo oficial GraphSAGE.
#### Producto:
# - training_inputs
#### Responde:
# ¿Las entradas oficiales del entrenamiento fueron construidas y validadas
# correctamente para iniciar el entrenamiento del modelo oficial GraphSAGE?

print("-" * 80)
print("Bloque 5. Construcción de las Entradas del Entrenamiento.")

def build_training_inputs(
    official_model_config: dict,
    training_data: dict
) -> dict:
    """
    Construye y valida las entradas oficiales utilizadas durante el
    entrenamiento definitivo del modelo GraphSAGE.

    Parameters
    ----------
    official_model_config : dict
        Configuración oficial del modelo GraphSAGE.

    training_data : dict
        Datos oficiales del entrenamiento.

    Returns
    -------
    dict
        Entradas oficiales del entrenamiento.
    """

    # Validación de entradas
    if official_model_config is None:
        raise ValueError(
            "official_model_config no puede ser nulo."
        )

    if training_data is None:
        raise ValueError(
            "training_data no puede ser nulo."
        )

    # Recuperación de información
    model_family = official_model_config["family"]
    model_name = official_model_config["model_name"]

    # Validación del modelo oficial
    if model_family != "graph_neural_networks":
        raise ValueError(
            "El entrenamiento definitivo únicamente admite Graph Neural Networks."
        )

    if model_name != "graphsage":
        raise ValueError(
            "El modelo oficial del proyecto es GraphSAGE."
        )

    # Construcción del producto
    training_inputs = {
        "model_config": official_model_config,
        "graphs": training_data["graphs"]
    }

    # Validación del producto
    required_keys = [
        "model_config",
        "graphs"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in training_inputs
    ]

    if missing_keys:
        raise ValueError(
            f"Faltan parámetros en training_inputs: {missing_keys}"
        )

    if not training_inputs["graphs"]:
        raise ValueError(
            "La colección oficial de GraphData está vacía."
        )

    return training_inputs

# Construcción del producto
TRAINING_INPUTS = build_training_inputs(
    official_model_config=official_model_config,
    training_data=TRAINING_DATA
)

print("-" * 80)
print("Bloque 5. Construcción del Entrenamiento realizada correctamente.")

# BLOQUE 6. Preparación de las Entradas del Modelo -------------------------
## Objetivo: Preparar y validar las entradas oficiales utilizadas por el
# modelo GraphSAGE durante el entrenamiento definitivo.
#### Producto:
# - training_features
#### Responde:
# ¿Las entradas oficiales del modelo GraphSAGE fueron preparadas y
# validadas correctamente para iniciar el entrenamiento?

print("-" * 80)
print("Bloque 6. Preparación de las Entradas del Modelo.")

def build_training_features(
    training_inputs: dict
) -> dict:
    """
    Prepara y valida las entradas oficiales requeridas por el modelo
    GraphSAGE durante el entrenamiento definitivo.

    Parameters
    ----------
    training_inputs : dict
        Entradas oficiales del entrenamiento.

    Returns
    -------
    dict
        Entradas oficiales del modelo.
    """

    # Validación de entrada
    if training_inputs is None:
        raise ValueError(
            "training_inputs no puede ser nulo."
        )

    required_keys = [
        "model_config",
        "graphs"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in training_inputs
    ]

    if missing_keys:
        raise ValueError(
            f"Faltan parámetros en training_inputs: {missing_keys}"
        )

    graphs = training_inputs["graphs"]

    if not graphs:
        raise ValueError(
            "La colección oficial de GraphData está vacía."
        )

    # Construcción del producto
    training_features = {
        "model_config": training_inputs["model_config"],
        "graphs": graphs
    }

    # Validación del producto
    if not training_features["graphs"]:
        raise ValueError(
            "No existen GraphData para el entrenamiento."
        )

    return training_features

# Construcción del producto
TRAINING_FEATURES = build_training_features(
    training_inputs=TRAINING_INPUTS
)

print("-" * 80)
print("Bloque 6. Entradas del Modelo correctas.")

# BLOQUE 7. Entrenamiento del Modelo Oficial -------------------------------
## Objetivo: Construir y entrenar el modelo oficial GraphSAGE utilizando la
# colección oficial de GraphData del proyecto.
#### Producto:
# - training_result
#### Responde:
# ¿El modelo oficial GraphSAGE fue construido, entrenado y validado
# correctamente utilizando los datos oficiales del proyecto?

print("-" * 80)
print("Bloque 7. Entrenamiento del Modelo Oficial.")

def train_official_model(
    training_features: dict
) -> dict:
    """
    Construye, entrena y valida el modelo oficial GraphSAGE.

    Parameters
    ----------
    training_features : dict
        Entradas oficiales del entrenamiento.

    Returns
    -------
    dict
        Resultado oficial del entrenamiento.
    """

    # Validación de entrada
    if training_features is None:
        raise ValueError(
            "training_features no puede ser nulo."
        )

    required_keys = [
        "model_config",
        "graphs"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in training_features
    ]

    if missing_keys:
        raise ValueError(
            f"Faltan parámetros en training_features: {missing_keys}"
        )

    # Recuperación de información
    model_config = training_features["model_config"]
    graphs = training_features["graphs"]

    if not graphs:
        raise ValueError(
            "La colección oficial de GraphData está vacía."
        )

    # Validación del modelo oficial
    required_config_keys = [
        "model_code",
        "model_name",
        "family"
    ]

    missing_config_keys = [
        key
        for key in required_config_keys
        if key not in model_config
    ]

    if missing_config_keys:
        raise ValueError(
            "La configuración oficial del modelo está incompleta: "
            f"{missing_config_keys}"
        )

    if model_config["family"] != "graph_neural_networks":
        raise ValueError(
            "El entrenamiento definitivo únicamente admite Graph Neural Networks."
        )

    if model_config["model_name"] != "graphsage":
        raise ValueError(
            "El modelo oficial del proyecto es GraphSAGE."
        )

    # Construcción del modelo oficial
    model = build_gnn_model(
        model_config=model_config,
        input_channels=graphs[0].num_node_features,
        output_channels=1
    )

    # Construcción de los componentes de entrenamiento
    training_components = build_training_components(
        model=model,
        model_config=model_config
    )
    
    # Entrenamiento del modelo oficial
    training_result = train_gnn(
        model=model,
        graphs=graphs,
        criterion=training_components["criterion"],
        optimizer=training_components["optimizer"],
        model_config=model_config
    )

    # Validación del producto
    if training_result is None:
        raise RuntimeError(
            "El entrenamiento definitivo no produjo resultados."
        )

    required_result_keys = [
        "model",
        "training_time",
        "loss",
        "loss_history"
    ]

    missing_result_keys = [
        key
        for key in required_result_keys
        if key not in training_result
    ]

    if missing_result_keys:
        raise ValueError(
            "El resultado del entrenamiento está incompleto: "
            f"{missing_result_keys}"
        )

    if training_result["model"] is None:
        raise RuntimeError(
            "El modelo entrenado es nulo."
        )

    return training_result

# Construcción del producto
TRAINING_RESULT = train_official_model(
    training_features=TRAINING_FEATURES
)

print("-" * 80)
print("Bloque 7. Entrenamiento del Modelo Oficial correcto.")

# BLOQUE 8. Exportación del Modelo Oficial ---------------------------------
## Objetivo: Exportar el modelo oficial GraphSAGE entrenado y los metadatos
# generados durante el entrenamiento definitivo para garantizar la
# reproducibilidad del proyecto.
#### Producto:
# - export_result
#### Responde:
# ¿El modelo oficial GraphSAGE fue exportado correctamente para las
# siguientes etapas del proyecto?

print("-" * 80)
print("Bloque 8. Exportación del Modelo Oficial.")

def export_official_model(
    training_inputs: dict,
    training_result: dict
) -> dict:
    """
    Exporta el modelo oficial GraphSAGE entrenado y sus metadatos.
    """

    # Validación
    if training_inputs is None:
        raise ValueError(
            "training_inputs no puede ser nulo."
        )

    if training_result is None:
        raise ValueError(
            "training_result no puede ser nulo."
        )

    required_input_keys = [
        "model_config"
    ]

    missing_input_keys = [
        key
        for key in required_input_keys
        if key not in training_inputs
    ]

    if missing_input_keys:
        raise ValueError(
            f"Faltan parámetros en training_inputs: {missing_input_keys}"
        )

    if "model" not in training_result:
        raise ValueError(
            "No se encontró el modelo entrenado."
        )

    # Recuperación
    trained_model = training_result["model"]
    model_config = training_inputs["model_config"]
    loss_history = training_result.get("loss_history")

    if model_config["family"] != "graph_neural_networks":
        raise ValueError(
            "La exportación únicamente admite Graph Neural Networks."
        )

    if model_config["model_name"] != "graphsage":
        raise ValueError(
            "El modelo oficial del proyecto es GraphSAGE."
        )

    if not hasattr(trained_model, "state_dict"):
        raise TypeError(
            "El modelo entrenado no corresponde a un objeto válido de PyTorch."
        )

    model_file = BEST_MODEL_TORCH_FILE
    config_file = BEST_MODEL_CONFIG_FILE
    metadata_file = BEST_MODEL_METADATA_FILE

    export_format = "torch"

    # Construcción
    try:

        torch.save(
            {
                "model_state_dict": trained_model.state_dict(),
                "model_config": model_config
            },
            model_file
        )

    except Exception as error:

        raise RuntimeError(
            f"Error al exportar el modelo oficial: {error}"
        )

    if not model_file.exists():
        raise RuntimeError(
            "No fue posible exportar el modelo oficial."
        )

    try:

        with open(
            config_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                model_config,
                file,
                indent=4,
                ensure_ascii=False
            )

    except Exception as error:

        raise RuntimeError(
            f"Error al exportar la configuración oficial: {error}"
        )

    if not config_file.exists():
        raise RuntimeError(
            "No fue posible exportar la configuración oficial."
        )

    training_metadata = {
        "model_code": model_config["model_code"],
        "model_name": model_config["model_name"],
        "family": model_config["family"],
        "training_name": TRAIN_CONFIGURATION["training_name"],
        "training_version": TRAIN_CONFIGURATION["training_version"],
        "training_time": training_result.get("training_time"),
        "training_loss": training_result.get("loss"),
        "loss_history": loss_history,
        "epochs": model_config.get("epochs"),
        "training_date": datetime.now().isoformat(),
        "export_format": export_format
    }

    epochs = training_metadata["epochs"]

    if epochs is None:
        raise ValueError(
            "No se encontró el número de épocas del entrenamiento."
        )

    if not isinstance(epochs, int):
        raise TypeError(
            "epochs debe ser un entero."
        )

    if epochs <= 0:
        raise ValueError(
            "epochs debe ser mayor que cero."
        )

    if loss_history is None:
        raise ValueError(
            "No se encontró el historial de entrenamiento."
        )

    if not isinstance(loss_history, list):
        raise TypeError(
            "loss_history debe ser una lista."
        )

    if len(loss_history) != epochs:
        raise ValueError(
            "El número de pérdidas registradas no coincide con el número de épocas."
        )

    try:

        with open(
            metadata_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                training_metadata,
                file,
                indent=4,
                ensure_ascii=False
            )

    except Exception as error:

        raise RuntimeError(
            f"Error al exportar los metadatos: {error}"
        )

    if not metadata_file.exists():
        raise RuntimeError(
            "No fue posible exportar los metadatos del entrenamiento."
        )

    # Retorno
    export_result = {
        "status": "SUCCESS",
        "model_code": model_config["model_code"],
        "model_name": model_config["model_name"],
        "family": model_config["family"],
        "model_file": str(model_file),
        "config_file": str(config_file),
        "metadata_file": str(metadata_file),
        "export_format": export_format,
        "training_time": training_result.get("training_time"),
        "training_loss": training_result.get("loss")
    }

    return export_result


EXPORT_RESULT = export_official_model(
    training_inputs=TRAINING_INPUTS,
    training_result=TRAINING_RESULT
)

print("-" * 80)
print("Bloque 8. Modelo Oficial exportado.")

# BLOQUE 9. Validación Final -----------------------------------------------
## Objetivo: Verificar la integridad del entrenamiento definitivo y de todos
# los productos oficiales generados durante el proceso.
#### Producto:
# - validation_result
#### Responde:
# ¿El entrenamiento definitivo fue ejecutado correctamente y todos los
# productos oficiales fueron generados de forma íntegra?

print("-" * 80)
print("Bloque 9. Validación Final.")

# BLOQUE 9.1. Validación de Entradas ---------------------------------------

def validate_training_inputs(
    training_inputs: dict,
    training_result: dict,
    export_result: dict
) -> None:
    """
    Valida las entradas requeridas para ejecutar la validación final
    del entrenamiento.
    """

    # Validación
    if training_inputs is None:
        raise RuntimeError(
            "No se encontraron las entradas oficiales del entrenamiento."
        )

    if training_result is None:
        raise RuntimeError(
            "El entrenamiento definitivo no generó resultados."
        )

    if export_result is None:
        raise RuntimeError(
            "La exportación del modelo no produjo resultados."
        )

    required_input_keys = [
        "model_config"
    ]

    missing_input_keys = [
        key
        for key in required_input_keys
        if key not in training_inputs
    ]

    if missing_input_keys:
        raise ValueError(
            f"Faltan parámetros en training_inputs: {missing_input_keys}"
        )

    model_config = training_inputs["model_config"]

    required_model_keys = [
        "model_code",
        "model_name",
        "family"
    ]

    missing_model_keys = [
        key
        for key in required_model_keys
        if key not in model_config
    ]

    if missing_model_keys:
        raise ValueError(
            f"Faltan parámetros en model_config: {missing_model_keys}"
        )

    if model_config["family"] != "graph_neural_networks":
        raise ValueError(
            "La validación únicamente admite Graph Neural Networks."
        )

    if model_config["model_name"] != "graphsage":
        raise ValueError(
            "El modelo oficial del proyecto es GraphSAGE."
        )

    required_training_keys = [
        "model",
        "training_time",
        "loss",
        "loss_history"
    ]

    missing_training_keys = [
        key
        for key in required_training_keys
        if key not in training_result
    ]

    if missing_training_keys:
        raise ValueError(
            "El resultado del entrenamiento está incompleto: "
            f"{missing_training_keys}"
        )

    if training_result["model"] is None:
        raise RuntimeError(
            "El modelo entrenado es nulo."
        )

    required_export_keys = [
        "status",
        "model_file",
        "config_file",
        "metadata_file",
        "export_format"
    ]

    missing_export_keys = [
        key
        for key in required_export_keys
        if key not in export_result
    ]

    if missing_export_keys:
        raise ValueError(
            "El resultado de la exportación está incompleto: "
            f"{missing_export_keys}"
        )

    if export_result["status"] != "SUCCESS":
        raise RuntimeError(
            "La exportación del modelo no finalizó correctamente."
        )

    # Retorno
    return

# BLOQUE 9.2. Recuperación de Datos ----------------------------------------
## Objetivo: Recuperar la información requerida para ejecutar la validación
# final del entrenamiento definitivo.
#### Producto:
# - validation_data
#### Responde:
# ¿Toda la información requerida para la validación fue recuperada
# correctamente?

print("-" * 80)
print("Bloque 9.2. Recuperación de Datos.")

def recover_validation_data(
    training_inputs: dict,
    export_result: dict
) -> dict:
    """
    Recupera la información requerida para ejecutar la validación
    final del entrenamiento.
    """

    # Recuperación
    model_config = training_inputs["model_config"]

    model_file = Path(
        export_result["model_file"]
    )

    config_file = Path(
        export_result["config_file"]
    )

    metadata_file = Path(
        export_result["metadata_file"]
    )

    export_format = export_result["export_format"]

    validation_data = {
        "model_config": model_config,
        "model_file": model_file,
        "config_file": config_file,
        "metadata_file": metadata_file,
        "export_format": export_format
    }

    # Retorno
    return validation_data


VALIDATION_DATA = recover_validation_data(
    training_inputs=TRAINING_INPUTS,
    export_result=EXPORT_RESULT
)

print("-" * 80)
print("Bloque 9.2. Recuperación de Datos realizada.")

# BLOQUE 9.3. Validación de Archivos Exportados -----------------------------
## Objetivo: Verificar la existencia e integridad de los archivos oficiales
# generados durante la exportación del modelo.
#### Producto:
# - exported_files
#### Responde:
# ¿Los archivos oficiales del modelo fueron exportados correctamente?

print("-" * 80)
print("Bloque 9.3. Validación de Archivos Exportados.")

def validate_exported_files(
    validation_data: dict
) -> dict:
    """
    Valida la integridad de los archivos oficiales exportados.
    """

    # Validación
    if validation_data is None:
        raise RuntimeError(
            "validation_data no puede ser nulo."
        )

    required_keys = [
        "model_file",
        "config_file",
        "metadata_file"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in validation_data
    ]

    if missing_keys:
        raise ValueError(
            f"Faltan parámetros en validation_data: {missing_keys}"
        )

    # Recuperación
    model_file = validation_data["model_file"]
    config_file = validation_data["config_file"]
    metadata_file = validation_data["metadata_file"]

    # Validación
    for file in [
        model_file,
        config_file,
        metadata_file
    ]:

        if not file.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo: {file.name}"
            )

        if file.stat().st_size == 0:
            raise RuntimeError(
                f"El archivo está vacío: {file.name}"
            )

    try:

        checkpoint = torch.load(
            model_file,
            map_location="cpu"
        )

    except Exception as error:

        raise RuntimeError(
            f"No fue posible cargar el checkpoint: {error}"
        )

    required_checkpoint_keys = [
        "model_state_dict",
        "model_config"
    ]

    missing_checkpoint_keys = [
        key
        for key in required_checkpoint_keys
        if key not in checkpoint
    ]

    if missing_checkpoint_keys:
        raise ValueError(
            "El checkpoint está incompleto: "
            f"{missing_checkpoint_keys}"
        )

    # Retorno
    exported_files = {
        "checkpoint": checkpoint,
        "model_file": model_file,
        "config_file": config_file,
        "metadata_file": metadata_file
    }

    return exported_files

EXPORTED_FILES = validate_exported_files(
    validation_data=VALIDATION_DATA
)

print("-" * 80)
print("Bloque 9.3. Validación de Archivos Exportados realizada.")

# BLOQUE 9.4. Validación de Metadatos --------------------------------------
## Objetivo: Validar la estructura e integridad de los archivos oficiales de
# configuración y metadatos generados durante la exportación.
#### Producto:
# - validation_metadata
#### Responde:
# ¿Los archivos oficiales de configuración y metadatos son íntegros y
# contienen toda la información requerida?

print("-" * 80)
print("Bloque 9.4. Validación de Metadatos.")

def validate_metadata(
    exported_files: dict
) -> dict:
    """
    Valida la estructura de los archivos oficiales de configuración
    y metadatos.
    """

    # Validación
    if exported_files is None:
        raise RuntimeError(
            "exported_files no puede ser nulo."
        )

    required_keys = [
        "config_file",
        "metadata_file"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in exported_files
    ]

    if missing_keys:
        raise ValueError(
            f"Faltan parámetros en exported_files: {missing_keys}"
        )

    # Recuperación
    config_file = exported_files["config_file"]
    metadata_file = exported_files["metadata_file"]

    try:

        with open(
            config_file,
            "r",
            encoding="utf-8"
        ) as file:

            exported_config = json.load(file)

    except Exception as error:

        raise RuntimeError(
            f"Error al leer la configuración exportada: {error}"
        )

    try:

        with open(
            metadata_file,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)

    except Exception as error:

        raise RuntimeError(
            f"Error al leer los metadatos: {error}"
        )

    required_config_keys = [
        "model_code",
        "model_name",
        "family"
    ]

    missing_config_keys = [
        key
        for key in required_config_keys
        if key not in exported_config
    ]

    if missing_config_keys:
        raise ValueError(
            "La configuración exportada está incompleta: "
            f"{missing_config_keys}"
        )

    required_metadata_keys = [
        "model_code",
        "model_name",
        "family",
        "training_name",
        "training_version",
        "training_loss",
        "loss_history",
        "export_format"
    ]

    missing_metadata_keys = [
        key
        for key in required_metadata_keys
        if key not in metadata
    ]

    if missing_metadata_keys:
        raise ValueError(
            "Los metadatos están incompletos: "
            f"{missing_metadata_keys}"
        )

    if not isinstance(
        metadata["loss_history"],
        list
    ):
        raise TypeError(
            "loss_history debe ser una lista."
        )

    if len(metadata["loss_history"]) == 0:
        raise ValueError(
            "loss_history no puede estar vacío."
        )

    if not all(
        isinstance(loss, (int, float))
        for loss in metadata["loss_history"]
    ):
        raise TypeError(
            "Todos los elementos de loss_history deben ser numéricos."
        )

    # Retorno
    validation_metadata = {
        "exported_config": exported_config,
        "metadata": metadata
    }

    return validation_metadata

VALIDATION_METADATA = validate_metadata(
    exported_files=EXPORTED_FILES
)

print("-" * 80)
print("Bloque 9.4. Validación de Metadatos realizada.")

# BLOQUE 9.5. Validación de Consistencia Científica -------------------------
## Objetivo: Verificar la consistencia científica entre la configuración del
# entrenamiento, la configuración exportada y los metadatos oficiales.
#### Producto:
# - scientific_validation
#### Responde:
# ¿Los productos oficiales exportados son científicamente consistentes con el
# entrenamiento definitivo?

print("-" * 80)
print("Bloque 9.5. Validación de Consistencia Científica.")

def validate_model_consistency(
    validation_data: dict,
    validation_metadata: dict
) -> dict:
    """
    Valida la consistencia científica entre la configuración utilizada
    durante el entrenamiento y los productos oficiales exportados.
    """

    # Validación
    if validation_data is None:
        raise RuntimeError(
            "validation_data no puede ser nulo."
        )

    if validation_metadata is None:
        raise RuntimeError(
            "validation_metadata no puede ser nulo."
        )

    required_validation_keys = [
        "model_config",
        "export_format"
    ]

    missing_validation_keys = [
        key
        for key in required_validation_keys
        if key not in validation_data
    ]

    if missing_validation_keys:
        raise ValueError(
            f"Faltan parámetros en validation_data: "
            f"{missing_validation_keys}"
        )

    required_metadata_keys = [
        "exported_config",
        "metadata"
    ]

    missing_metadata_keys = [
        key
        for key in required_metadata_keys
        if key not in validation_metadata
    ]

    if missing_metadata_keys:
        raise ValueError(
            f"Faltan parámetros en validation_metadata: "
            f"{missing_metadata_keys}"
        )

    # Recuperación
    model_config = validation_data["model_config"]

    export_format = validation_data["export_format"]

    exported_config = validation_metadata["exported_config"]

    metadata = validation_metadata["metadata"]

    # Validación
    if exported_config["model_code"] != model_config["model_code"]:
        raise ValueError(
            "El código del modelo exportado no coincide con la configuración del entrenamiento."
        )

    if exported_config["model_name"] != model_config["model_name"]:
        raise ValueError(
            "El nombre del modelo exportado no coincide con la configuración del entrenamiento."
        )

    if exported_config["family"] != model_config["family"]:
        raise ValueError(
            "La familia del modelo exportado no coincide con la configuración del entrenamiento."
        )

    if metadata["model_code"] != model_config["model_code"]:
        raise ValueError(
            "El código del modelo no coincide con los metadatos."
        )

    if metadata["model_name"] != model_config["model_name"]:
        raise ValueError(
            "El nombre del modelo no coincide con los metadatos."
        )

    if metadata["family"] != model_config["family"]:
        raise ValueError(
            "La familia del modelo no coincide con los metadatos."
        )

    if metadata["training_name"] != TRAIN_CONFIGURATION["training_name"]:
        raise ValueError(
            "El nombre del entrenamiento no coincide."
        )

    if metadata["training_version"] != TRAIN_CONFIGURATION["training_version"]:
        raise ValueError(
            "La versión del entrenamiento no coincide."
        )

    if export_format != "torch":
        raise ValueError(
            "El formato oficial de exportación debe ser Torch."
        )

    if metadata["export_format"] != export_format:
        raise ValueError(
            "El formato de exportación no coincide con los metadatos."
        )

    # Retorno
    scientific_validation = {
        "status": "SUCCESS",
        "model_code": model_config["model_code"],
        "model_name": model_config["model_name"],
        "family": model_config["family"],
        "training_name": metadata["training_name"],
        "training_version": metadata["training_version"],
        "export_format": export_format
    }

    return scientific_validation

SCIENTIFIC_VALIDATION = validate_model_consistency(
    validation_data=VALIDATION_DATA,
    validation_metadata=VALIDATION_METADATA
)

print("-" * 80)
print("Bloque 9.5. Validación de Consistencia Científica realizada.")

# BLOQUE 9.6. Construcción del Producto ------------------------------------
## Objetivo: Construir el producto oficial de la validación final del
# entrenamiento definitivo.
#### Producto:
# - validation_result
#### Responde:
# ¿El producto oficial de la validación fue construido correctamente?

print("-" * 80)
print("Bloque 9.6. Construcción del Producto.")

def build_validation_result(
    validation_data: dict,
    validation_metadata: dict,
    scientific_validation: dict
) -> dict:
    """
    Construye el producto oficial de la validación final.
    """

    # Validación
    if validation_data is None:
        raise RuntimeError(
            "validation_data no puede ser nulo."
        )

    if validation_metadata is None:
        raise RuntimeError(
            "validation_metadata no puede ser nulo."
        )

    if scientific_validation is None:
        raise RuntimeError(
            "scientific_validation no puede ser nulo."
        )

    required_validation_keys = [
        "model_file",
        "config_file",
        "metadata_file"
    ]

    missing_validation_keys = [
        key
        for key in required_validation_keys
        if key not in validation_data
    ]

    if missing_validation_keys:
        raise ValueError(
            f"Faltan parámetros en validation_data: "
            f"{missing_validation_keys}"
        )

    required_metadata_keys = [
        "metadata"
    ]

    missing_metadata_keys = [
        key
        for key in required_metadata_keys
        if key not in validation_metadata
    ]

    if missing_metadata_keys:
        raise ValueError(
            f"Faltan parámetros en validation_metadata: "
            f"{missing_metadata_keys}"
        )

    # Recuperación
    model_file = validation_data["model_file"]

    config_file = validation_data["config_file"]

    metadata_file = validation_data["metadata_file"]

    metadata = validation_metadata["metadata"]

    # Construcción
    validation_result = {
        "status": scientific_validation["status"],
        "training_name": metadata["training_name"],
        "training_version": metadata["training_version"],
        "model_code": scientific_validation["model_code"],
        "model_name": scientific_validation["model_name"],
        "family": scientific_validation["family"],
        "model_file": str(model_file),
        "config_file": str(config_file),
        "metadata_file": str(metadata_file),
        "training_time": metadata.get("training_time"),
        "training_loss": metadata.get("training_loss"),
        "validation_date": datetime.now().isoformat()
    }

    # Retorno
    return validation_result

VALIDATION_RESULT = build_validation_result(
    validation_data=VALIDATION_DATA,
    validation_metadata=VALIDATION_METADATA,
    scientific_validation=SCIENTIFIC_VALIDATION
)

print("-" * 80)
print("Bloque 9.6. Construcción del Producto realizada.")

# BLOQUE 9.7. Orquestación de la Validación Final ---------------------------
## Objetivo: Ejecutar de forma secuencial el proceso oficial de validación
# final del entrenamiento definitivo.
#### Producto:
# - VALIDATION_RESULT
#### Responde:
# ¿La validación final del entrenamiento fue ejecutada correctamente?

print("-" * 80)
print("Bloque 9.7. Orquestación de la Validación Final.")

def validate_training(
    training_inputs: dict,
    training_result: dict,
    export_result: dict
) -> dict:
    """
    Ejecuta el proceso oficial de validación final del entrenamiento.
    """

    # Validación
    validate_training_inputs(
        training_inputs=training_inputs,
        training_result=training_result,
        export_result=export_result
    )

    # Recuperación
    validation_data = recover_validation_data(
        training_inputs=training_inputs,
        export_result=export_result
    )

    # Validación
    exported_files = validate_exported_files(
        validation_data=validation_data
    )

    validation_metadata = validate_metadata(
        exported_files=exported_files
    )

    scientific_validation = validate_model_consistency(
        validation_data=validation_data,
        validation_metadata=validation_metadata
    )

    # Construcción
    validation_result = build_validation_result(
        validation_data=validation_data,
        validation_metadata=validation_metadata,
        scientific_validation=scientific_validation
    )

    # Retorno
    return validation_result


VALIDATION_RESULT = validate_training(
    training_inputs=TRAINING_INPUTS,
    training_result=TRAINING_RESULT,
    export_result=EXPORT_RESULT
)

print("-" * 80)
print("Bloque 9.7. Validación Final realizada.")

# BLOQUE 10. Producto Oficial del Entrenamiento ----------------------------
## Objetivo: Consolidar los productos oficiales generados durante el
# entrenamiento definitivo y construir el producto oficial que será
# consumido por la etapa de evaluación del pipeline científico.
#### Producto:
# - training_output
#### Responde:
# ¿Cuál es el producto oficial generado por el entrenamiento definitivo que
# será utilizado por las siguientes etapas del pipeline científico?

print("-" * 80)
print("Bloque 10. Producto Oficial del Entrenamiento.")

def build_training_output(
    training_inputs: dict,
    training_result: dict,
    export_result: dict,
    validation_result: dict
) -> dict:
    """
    Consolida los productos oficiales generados durante el entrenamiento
    definitivo y construye el artefacto oficial del módulo.

    Parameters
    ----------
    training_inputs : dict
    training_result : dict
    export_result : dict
    validation_result : dict

    Returns
    -------
    dict
        Resultado oficial del entrenamiento.
    """

    # Validación de entradas
    if training_inputs is None:
        raise ValueError(
            "training_inputs no puede ser nulo."
        )

    if training_result is None:
        raise ValueError(
            "training_result no puede ser nulo."
        )

    if export_result is None:
        raise ValueError(
            "export_result no puede ser nulo."
        )

    if validation_result is None:
        raise ValueError(
            "validation_result no puede ser nulo."
        )

    # Recuperación de información
    model_config = training_inputs["model_config"]

    # Construcción de productos oficiales
    generated_products = {
        "official_model": export_result["model_file"],
        "training_metadata": export_result["metadata_file"],
        "model_configuration": export_result["config_file"],
        "benchmark_results": str(BENCHMARK_RESULTS_FILE)
    }

    # Validación de productos
    for product_name, product_path in generated_products.items():

        if not Path(product_path).exists():
            raise FileNotFoundError(
                f"No fue posible localizar el producto oficial: {product_name}"
            )

    # Construcción del resumen oficial
    training_summary = {
        "status": "SUCCESS",
        "training_name": validation_result["training_name"],
        "training_version": validation_result["training_version"],
        "model_code": model_config["model_code"],
        "model_name": model_config["model_name"],
        "family": model_config["family"],
        "training_time": training_result.get("training_time"),
        "training_loss": training_result.get("loss"),
        "loss_history": training_result.get("loss_history"),
        "generated_products": generated_products,
        "validation_date": validation_result["validation_date"],
        "generation_date": datetime.now().isoformat()
    }

    # Exportación del resumen
    try:

        with open(
            TRAINING_SUMMARY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                training_summary,
                file,
                indent=4,
                ensure_ascii=False
            )

    except Exception as error:

        raise RuntimeError(
            f"Error al exportar el resumen del entrenamiento: {error}"
        )

    if not TRAINING_SUMMARY_FILE.exists():
        raise RuntimeError(
            "No fue posible exportar el resumen oficial del entrenamiento."
        )

    # Incorporación del resumen a los productos oficiales
    generated_products["training_summary"] = str(
        TRAINING_SUMMARY_FILE
    )

    # Construcción del producto oficial
    training_output = {
        "status": "SUCCESS",
        "model": training_result["model"],
        "model_config": model_config,
        "config_file": export_result["config_file"],

        # Métricas del entrenamiento
        "training_time": training_result.get("training_time"),
        "training_loss": training_result.get("loss"),
        "loss_history": training_result.get("loss_history"),

        # Archivos generados
        "model_file": export_result["model_file"],
        "metadata_file": export_result["metadata_file"],
        "summary_file": str(TRAINING_SUMMARY_FILE),

        # Productos oficiales
        "generated_products": generated_products,

        # Validación
        "validation": validation_result,

        # Resumen ejecutivo
        "summary": training_summary
    }

    return training_output

# Construcción del producto
TRAINING_OUTPUT = build_training_output(
    training_inputs=TRAINING_INPUTS,
    training_result=TRAINING_RESULT,
    export_result=EXPORT_RESULT,
    validation_result=VALIDATION_RESULT
)

print("-" * 80)
print("Bloque 10. Producto Oficial del Entrenamiento realizado.")

# BLOQUE 11. Reporte Final del Entrenamiento -------------------------------
## Objetivo: Presentar el resumen ejecutivo del entrenamiento definitivo y
# de los productos oficiales generados durante el proceso.
#### Producto:
# - Reporte de ejecución en consola.
#### Responde:
# ¿Cuál fue el resultado final del entrenamiento definitivo y qué
# productos oficiales fueron generados?

print("-" * 80)
print("Bloque 11. Reporte Final del Entrenamiento.")

def report_training_output(
    training_output: dict
) -> None:
    """
    Presenta el resumen ejecutivo del entrenamiento definitivo y de los
    productos oficiales generados durante el proceso.

    Parameters
    ----------
    training_output : dict
        Producto oficial del entrenamiento.

    Returns
    -------
    None
    """

    # Validación de entrada
    if training_output is None:
        raise ValueError(
            "training_output no puede ser nulo."
        )

    required_keys = [
        "status",
        "model_config",
        "model_file",
        "config_file",
        "metadata_file",
        "summary_file",
        "generated_products",
        "validation",
        "summary"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in training_output
    ]

    if missing_keys:
        raise ValueError(
            "El producto oficial del entrenamiento está incompleto: "
            f"{missing_keys}"
        )

    # Recuperación de información
    model_config = training_output["model_config"]
    summary = training_output["summary"]
    generated_products = training_output["generated_products"]

    if model_config is None:
        raise ValueError(
            "model_config no puede ser nulo."
        )

    if summary is None:
        raise ValueError(
            "summary no puede ser nulo."
        )

    if generated_products is None:
        raise ValueError(
            "generated_products no puede ser nulo."
        )

    # Reporte ejecutivo
    print("\n" + "=" * 80)
    print("REPORTE FINAL DEL ENTRENAMIENTO")
    print("=" * 80)

    print(f"Estado                 : {training_output['status']}")
    print(f"Modelo Oficial         : {model_config['model_name']}")
    print(f"Código                 : {model_config['model_code']}")
    print(f"Familia                : {model_config['family']}")
    print(f"Versión Entrenamiento  : {summary['training_version']}")
    print(f"Tiempo Entrenamiento   : {summary['training_time']}")
    print(f"Loss Final             : {summary['training_loss']}")

    print("\nArchivos Oficiales Exportados")
    print("-" * 80)

    print(
        f"Modelo GraphSAGE (.pt) : "
        f"{Path(training_output['model_file']).name}"
    )
    print(
        f"Ruta                   : "
        f"{training_output['model_file']}"
    )

    print()

    print(
        f"Configuración (.json)  : "
        f"{Path(training_output['config_file']).name}"
    )
    print(
        f"Ruta                   : "
        f"{training_output['config_file']}"
    )

    print()

    print(
        f"Metadatos (.json)      : "
        f"{Path(training_output['metadata_file']).name}"
    )
    print(
        f"Ruta                   : "
        f"{training_output['metadata_file']}"
    )

    print()

    print(
        f"Resumen (.json)        : "
        f"{Path(training_output['summary_file']).name}"
    )
    print(
        f"Ruta                   : "
        f"{training_output['summary_file']}"
    )

    print("\nProductos Oficiales Generados")
    print("-" * 80)

    for product_name, product_path in generated_products.items():

        if product_path is None:

            print(f"{product_name:<30}: No generado")

        else:

            print(
                f"{product_name:<30}: "
                f"{Path(product_path).name}"
            )

    print("\nPróxima Etapa")
    print("-" * 80)
    print("Archivo : graph-04_evaluation.py")
    print("Proceso : Evaluación Científica")

# Ejecución del reporte
report_training_output(
    training_output=TRAINING_OUTPUT
)

print("-" * 80)
print("Bloque 11. Reporte Final del Entrenamiento realizado.")