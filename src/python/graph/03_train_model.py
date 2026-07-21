# graph-03_train_model.py

# BLOQUE 1. Importaciones --------------------------------------------------
## Objetivo: Importar las dependencias necesarias para ejecutar el entrenamiento
# definitivo del modelo oficial GraphSAGE utilizando el Dataset Científico
# y la colección oficial de GraphData del proyecto.
### Producto:
# - Librerías cargadas correctamente.
### Responde:
# ¿Qué dependencias requiere el protocolo de entrenamiento para
# reconstruir, entrenar y exportar el modelo oficial del proyecto?

# Funciones del sistema
import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib

# Librerías científicas
import pandas as pd
import torch

# Configuración del proyecto
from src.python.config.paths import (
    DATASET_FILE,
    GRAPH_DATA_DIR,
    BEST_MODEL_CONFIG_FILE,
    BENCHMARK_RESULTS_FILE,
    BEST_MODEL_TORCH_FILE,
    BEST_MODEL_METADATA_FILE,
    TRAINING_SUMMARY_FILE
)

from src.python.config.config_project import (
    PROJECT_SEED
)

# Modelos Graph Neural Networks
from src.python.models.graph_neural_networks import (
    GNN_CONFIG,
    build_gnn_model,
    build_training_components,
    train_gnn
)

warnings.filterwarnings("ignore")

print("-" * 80)

# BLOQUE 2. Configuración --------------------------------------------------
## Objetivo: Definir la configuración oficial que gobernará el entrenamiento
# definitivo del modelo oficial seleccionado durante el Benchmark Científico,
# garantizando la reproducibilidad y consistencia del protocolo experimental.
#### Producto:
# - TRAINING_CONFIG
#### Responde:
# ¿El entrenamiento definitivo dispone de una configuración oficial,
# reproducible y consistente para generar el modelo oficial del proyecto?

def build_training_configuration() -> dict:
    """
    Construye y valida la configuración oficial del entrenamiento definitivo.

    Returns
    -------
    dict
        Configuración oficial del entrenamiento.
    """

    # Construcción de la configuración
    training_config = {
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
        if key not in training_config
    ]

    if missing_keys:
        raise RuntimeError(
            "La configuración oficial del entrenamiento está incompleta: "
            f"{missing_keys}"
        )

    # Validación de tipos
    if not isinstance(training_config["training_name"], str):
        raise TypeError("training_name debe ser una cadena.")

    if not isinstance(training_config["training_version"], str):
        raise TypeError("training_version debe ser una cadena.")

    if not isinstance(training_config["random_state"], int):
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
        if not isinstance(training_config[key], bool):
            raise TypeError(f"{key} debe ser un valor booleano.")

    # Validación del contenido
    if not training_config["training_name"].strip():
        raise ValueError("training_name está vacío.")

    if not training_config["training_version"].strip():
        raise ValueError("training_version está vacío.")

    return training_config

# BLOQUE 3. Carga del Modelo Oficial GraphSAGE ------------------------------
## Objetivo: Recuperar la configuración oficial y los resultados del modelo
# GraphSAGE seleccionado durante el Benchmark Científico para ejecutar el
# entrenamiento definitivo del modelo oficial del proyecto.
#### Producto:
# - official_model_config
# - benchmark_results
# - official_model_result
#### Responde:
# ¿La configuración y los resultados oficiales del modelo GraphSAGE fueron
# recuperados correctamente para iniciar el entrenamiento definitivo?

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

    # Validación de la configuración
    required_keys = [
        "model_code",
        "model_name",
        "family"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in benchmark_model
    ]

    if missing_keys:
        raise ValueError(
            f"La configuración del Benchmark está incompleta: {missing_keys}"
        )

    # Validación del modelo oficial
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

    # Recuperación de los resultados del Benchmark
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

    # Recuperación del resultado oficial
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

    # Validación del resultado
    required_keys = [
        "model_code",
        "model_name",
        "family",
        "model"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in official_model_result
    ]

    if missing_keys:
        raise ValueError(
            f"El resultado oficial de GraphSAGE está incompleto: {missing_keys}"
        )

    benchmark_summary = {
        "model_code": official_model_config["model_code"],
        "model_name": official_model_config["model_name"],
        "family": official_model_config["family"]
    }

    return {
        "official_model_config": official_model_config,
        "official_model_result": official_model_result,
        "benchmark_results": benchmark_results,
        "benchmark_summary": benchmark_summary
    }

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
            graph = torch.load(
                graph_file,
                weights_only=False
            )

        except Exception as error:
            raise RuntimeError(
                f"Error al cargar {graph_file.name}: {error}"
            )

        graphs.append(graph)

    # Construcción del producto
    training_data = {
        "dataset": dataset,
        "graphs": graphs
    }

    # Validación del Dataset Científico
    if training_data["dataset"].empty:
        raise ValueError(
            "El Dataset Científico está vacío."
        )

    # Validación de GraphData
    if not training_data["graphs"]:
        raise ValueError(
            "La colección oficial de GraphData está vacía."
        )

    for index, graph in enumerate(
        training_data["graphs"],
        start=1
    ):

        if graph is None:
            raise ValueError(
                f"GraphData #{index} no fue cargado correctamente."
            )

        if not hasattr(graph, "num_node_features"):
            raise TypeError(
                f"GraphData #{index} no corresponde a un objeto válido de PyTorch Geometric."
            )

    return training_data

# BLOQUE 5. Construcción de las Entradas del Entrenamiento -----------------
## Objetivo: Construir y validar las entradas oficiales requeridas para el
# entrenamiento definitivo del modelo oficial GraphSAGE.
#### Producto:
# - training_inputs
#### Responde:
# ¿Las entradas oficiales del entrenamiento fueron construidas y validadas
# correctamente para iniciar el entrenamiento del modelo oficial GraphSAGE?

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

# BLOQUE 6. Preparación de las Entradas del Modelo -------------------------
## Objetivo: Preparar y validar las entradas oficiales utilizadas por el
# modelo GraphSAGE durante el entrenamiento definitivo.
#### Producto:
# - training_features
#### Responde:
# ¿Las entradas oficiales del modelo GraphSAGE fueron preparadas y
# validadas correctamente para iniciar el entrenamiento?

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

# BLOQUE 7. Entrenamiento del Modelo Oficial -------------------------------
## Objetivo: Construir y entrenar el modelo oficial GraphSAGE utilizando la
# colección oficial de GraphData del proyecto.
#### Producto:
# - training_result
#### Responde:
# ¿El modelo oficial GraphSAGE fue construido, entrenado y validado
# correctamente utilizando los datos oficiales del proyecto?

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
        "model"
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

# BLOQUE 8. Exportación del Modelo Oficial ---------------------------------
## Objetivo: Exportar el modelo oficial GraphSAGE entrenado y los metadatos
# generados durante el entrenamiento definitivo para garantizar la
# reproducibilidad del proyecto.
#### Producto:
# - export_result
#### Responde:
# ¿El modelo oficial GraphSAGE fue exportado correctamente para las
# siguientes etapas del proyecto?

def export_official_model(
    training_inputs: dict,
    training_result: dict
) -> dict:
    """
    Exporta el modelo oficial GraphSAGE entrenado y sus metadatos.

    Parameters
    ----------
    training_inputs : dict
        Entradas oficiales del entrenamiento.

    training_result : dict
        Resultado oficial del entrenamiento.

    Returns
    -------
    dict
        Resultado oficial de la exportación.
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

    # Recuperación de información
    trained_model = training_result["model"]
    model_config = training_inputs["model_config"]

    # Validación del modelo oficial
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

    # Configuración de la exportación
    model_file = BEST_MODEL_TORCH_FILE
    export_format = "torch"

    # Exportación del modelo
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

    # Construcción de los metadatos
    training_metadata = {
        "model_code": model_config["model_code"],
        "model_name": model_config["model_name"],
        "family": model_config["family"],
        "training_name": TRAINING_CONFIG["training_name"],
        "training_version": TRAINING_CONFIG["training_version"],
        "training_time": training_result.get("training_time"),
        "training_loss": training_result.get("loss"),
        "epochs": model_config.get("epochs"),
        "training_date": datetime.now().isoformat(),
        "export_format": export_format
    }

    # Exportación de metadatos
    try:

        with open(
            BEST_MODEL_METADATA_FILE,
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

    if not BEST_MODEL_METADATA_FILE.exists():
        raise RuntimeError(
            "No fue posible exportar los metadatos del entrenamiento."
        )

    # Construcción del producto
    export_result = {
        "status": "SUCCESS",
        "model_code": model_config["model_code"],
        "model_name": model_config["model_name"],
        "family": model_config["family"],
        "model_file": str(model_file),
        "metadata_file": str(BEST_MODEL_METADATA_FILE),
        "export_format": export_format,
        "training_time": training_result.get("training_time"),
        "training_loss": training_result.get("loss")
    }

    return export_result

# BLOQUE 9. Validación Final -----------------------------------------------
## Objetivo: Verificar la integridad del entrenamiento definitivo y de todos
# los productos oficiales generados durante el proceso.
#### Producto:
# - validation_result
#### Responde:
# ¿El entrenamiento definitivo fue ejecutado correctamente y todos los
# productos oficiales fueron generados de forma íntegra?

def validate_training(
    training_inputs: dict,
    training_result: dict,
    export_result: dict
) -> dict:
    """
    Valida la integridad del entrenamiento definitivo y de todos los
    productos oficiales generados durante el proceso.

    Parameters
    ----------
    training_inputs : dict
        Entradas oficiales del entrenamiento.

    training_result : dict
        Resultado oficial del entrenamiento.

    export_result : dict
        Resultado oficial de la exportación.

    Returns
    -------
    dict
        Resultado oficial de la validación.
    """

    # Validación de entradas
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

    # Recuperación de información
    model_config = training_inputs["model_config"]

    # Validación del modelo oficial
    if model_config["family"] != "graph_neural_networks":
        raise ValueError(
            "La validación únicamente admite Graph Neural Networks."
        )

    if model_config["model_name"] != "graphsage":
        raise ValueError(
            "El modelo oficial del proyecto es GraphSAGE."
        )

    # Validación del entrenamiento
    required_training_keys = [
        "model"
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

    # Validación de la exportación
    required_export_keys = [
        "status",
        "model_file",
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

    # Validación de archivos
    model_file = Path(export_result["model_file"])
    metadata_file = Path(export_result["metadata_file"])

    if not model_file.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo exportado: {model_file.name}"
        )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"No se encontraron los metadatos: {metadata_file.name}"
        )

    if model_file.stat().st_size == 0:
        raise RuntimeError(
            "El archivo del modelo está vacío."
        )

    if metadata_file.stat().st_size == 0:
        raise RuntimeError(
            "El archivo de metadatos está vacío."
        )

    # Recuperación de metadatos
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

    required_metadata_keys = [
        "model_code",
        "model_name",
        "family",
        "training_name",
        "training_version",
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

    # Consistencia científica
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

    if metadata["training_name"] != TRAINING_CONFIG["training_name"]:
        raise ValueError(
            "El nombre del entrenamiento no coincide."
        )

    if metadata["training_version"] != TRAINING_CONFIG["training_version"]:
        raise ValueError(
            "La versión del entrenamiento no coincide."
        )

    if export_result["export_format"] != "torch":
        raise ValueError(
            "El formato oficial de exportación debe ser Torch."
        )

    if metadata["export_format"] != export_result["export_format"]:
        raise ValueError(
            "El formato de exportación no coincide con los metadatos."
        )

    # Construcción del producto
    validation_result = {
        "status": "SUCCESS",
        "training_name": metadata["training_name"],
        "training_version": metadata["training_version"],
        "model_code": model_config["model_code"],
        "model_name": model_config["model_name"],
        "family": model_config["family"],
        "model_file": str(model_file),
        "metadata_file": str(metadata_file),
        "training_time": metadata.get("training_time"),
        "training_loss": metadata.get("training_loss"),
        "validation_date": datetime.now().isoformat()
    }

    return validation_result

# BLOQUE 10. Producto Oficial del Entrenamiento ----------------------------
## Objetivo: Consolidar los productos oficiales generados durante el
# entrenamiento definitivo y construir el producto oficial que será
# consumido por la etapa de evaluación del pipeline científico.
#### Producto:
# - training_output
#### Responde:
# ¿Cuál es el producto oficial generado por el entrenamiento definitivo que
# será utilizado por las siguientes etapas del pipeline científico?

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
        "benchmark_configuration": str(BEST_MODEL_CONFIG_FILE),
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
        "model_file": export_result["model_file"],
        "metadata_file": export_result["metadata_file"],
        "summary_file": str(TRAINING_SUMMARY_FILE),
        "generated_products": generated_products,
        "validation": validation_result,
        "summary": training_summary
    }

    return training_output

# BLOQUE 11. Reporte Final del Entrenamiento -------------------------------
## Objetivo: Presentar el resumen ejecutivo del entrenamiento definitivo y
# de los productos oficiales generados durante el proceso.
#### Producto:
# - Reporte de ejecución en consola.
#### Responde:
# ¿Cuál fue el resultado final del entrenamiento definitivo y qué
# productos oficiales fueron generados?

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

    # Reporte ejecutivo
    print("\n" + "=" * 80)
    print("REPORTE FINAL DEL ENTRENAMIENTO")
    print("=" * 80)

    print(f"Estado                 : {training_output['status']}")
    print(f"Modelo Oficial         : {model_config['model_name']}")
    print(f"Código                 : {model_config['model_code']}")
    print(f"Familia                : {model_config['family']}")
    print(f"Versión Entrenamiento  : {summary['training_version']}")
    print(f"Tiempo Entrenamiento   : {summary.get('training_time')}")
    print(f"Loss Final             : {summary.get('training_loss')}")

    print("\nProductos Oficiales")

    for product_name, product_path in training_output["generated_products"].items():
        print(f"{product_name:<25}: {Path(product_path).name}")

    print("\nPróxima Etapa")
    print("04_evaluation.py")

    print("=" * 80)