# graph-04_evaluation.py

# BLOQUE 1. Importaciones --------------------------------------------------
## Objetivo: Importar las dependencias necesarias para ejecutar la evaluación
# científica del modelo oficial GraphSAGE utilizando el Dataset Científico,
# la colección oficial de GraphData y los componentes de explicabilidad
# del proyecto.
### Producto:
# - Librerías cargadas correctamente.
### Responde:
# ¿Qué dependencias requiere el protocolo de evaluación para reconstruir,
# evaluar, interpretar y validar el modelo oficial del proyecto?

print("-" * 80)
print("Bloque 1. Importaciones.")

# Funciones del sistema
import json
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path

# Librerías científicas
import numpy as np
import pandas as pd
import torch

# Configuración del proyecto
from src.python.config.config_project import (
    PROJECT_SEED,
    FEATURE_COLUMNS
)

# Rutas oficiales
from src.python.config.paths import (
    DATASET_FILE,
    GRAPH_DATA_DIR,
    EVALUATION_REPORTS_DIR,
    BEST_MODEL_TORCH_FILE,
    BEST_MODEL_CONFIG_FILE,
    BEST_MODEL_METADATA_FILE
)

# Utilidades
from src.python.utils.results import (
    build_benchmark_result
)

# Modelos Graph Neural Networks
from src.python.models.graph_neural_networks import (
    GNN_CONFIG,
    build_gnn_model,
    evaluate_gnn,
    predict_gnn
)

# Explicabilidad
from src.python.analysis.explainability import (
    aggregate_feature_importance,
    analyze_target_correlation,
    audit_feature_importance,
    build_explainer,
    build_explanation_targets,
    build_feature_ranking,
    build_scientific_summary,
    export_explainability_results,
    extract_explanation_masks,
    finalize_explainability,
    generate_explainability_plots,
    generate_node_explanations,
    initialize_explainability_context,
    initialize_global_explainability,
    select_explainability_method,
    validate_global_explainability
)

print("initialize_global_explainability =", initialize_global_explainability)

# Configuración del entorno
warnings.filterwarnings("ignore")

print("-" * 80)
print("Bloque 1. Importaciones completado.")

# BLOQUE 2. Configuración --------------------------------------------------
## Objetivo: Definir la configuración oficial que gobernará la evaluación
# científica del modelo oficial GraphSAGE, garantizando la reproducibilidad,
# consistencia e integridad del protocolo experimental.
### Producto:
# - EVALUATION_CONFIG
### Responde:
# ¿La evaluación científica dispone de una configuración oficial,
# reproducible y consistente para validar el modelo oficial del proyecto?

def build_evaluation_configuration() -> dict:
    """
    Construye y valida la configuración oficial de la evaluación científica.

    Returns
    -------
    dict
        Configuración oficial de la evaluación.
    """

    # Construcción de la configuración
    evaluation_config = {
        "evaluation_name": "scientific_evaluation",
        "evaluation_version": "1.0",
        "prediction_source": "official_model",
        "calculate_metrics": True,
        "calculate_residuals": True,
        "calculate_global_explainability": True,
        "calculate_local_explainability": True,
        "calculate_feature_importance": True,
        "shap_batch_size": 512,
        "show_progress": True,
        "export_gnn_embeddings": True,
        "save_evaluation_results": True,
        "save_evaluation_summary": True,
        "save_feature_importance": True,
        "save_shap_values": True,
        "save_validation_results": True,
        "save_evaluation_report": True,
        "random_state": PROJECT_SEED
    }

    # Validación de la estructura
    required_keys = [
        "evaluation_name",
        "evaluation_version",
        "prediction_source",
        "calculate_metrics",
        "calculate_residuals",
        "calculate_global_explainability",
        "calculate_local_explainability",
        "calculate_feature_importance",
        "shap_batch_size",
        "show_progress",
        "export_gnn_embeddings",
        "save_evaluation_results",
        "save_evaluation_summary",
        "save_feature_importance",
        "save_shap_values",
        "save_validation_results",
        "save_evaluation_report",
        "random_state"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in evaluation_config
    ]

    if missing_keys:
        raise RuntimeError(
            "La configuración oficial de la evaluación está incompleta: "
            f"{missing_keys}"
        )

    # Validación de tipos
    if not isinstance(evaluation_config["evaluation_name"], str):
        raise TypeError("evaluation_name debe ser una cadena.")

    if not isinstance(evaluation_config["evaluation_version"], str):
        raise TypeError("evaluation_version debe ser una cadena.")

    if not isinstance(evaluation_config["prediction_source"], str):
        raise TypeError("prediction_source debe ser una cadena.")

    if not isinstance(evaluation_config["shap_batch_size"], int):
        raise TypeError("shap_batch_size debe ser un entero.")

    if not isinstance(evaluation_config["random_state"], int):
        raise TypeError("random_state debe ser un entero.")

    boolean_keys = [
        "calculate_metrics",
        "calculate_residuals",
        "calculate_global_explainability",
        "calculate_local_explainability",
        "calculate_feature_importance",
        "show_progress",
        "export_gnn_embeddings",
        "save_evaluation_results",
        "save_evaluation_summary",
        "save_feature_importance",
        "save_shap_values",
        "save_validation_results",
        "save_evaluation_report"
    ]

    for key in boolean_keys:
        if not isinstance(evaluation_config[key], bool):
            raise TypeError(f"{key} debe ser un valor booleano.")

    # Validación del contenido
    if not evaluation_config["evaluation_name"].strip():
        raise ValueError("evaluation_name está vacío.")

    if not evaluation_config["evaluation_version"].strip():
        raise ValueError("evaluation_version está vacío.")

    if not evaluation_config["prediction_source"].strip():
        raise ValueError("prediction_source está vacío.")

    if evaluation_config["shap_batch_size"] <= 0:
        raise ValueError("shap_batch_size debe ser mayor que cero.")

    return evaluation_config

EVALUATION_CONFIG = build_evaluation_configuration()

print("-" * 80)
print("Bloque 2. Configuración Finalizada.")

# BLOQUE 3. Carga del Modelo Oficial ---------------------------------------
## Objetivo: Recuperar el modelo oficial GraphSAGE entrenado, sus metadatos
# y la configuración oficial utilizada durante el entrenamiento definitivo
# para iniciar la evaluación científica del proyecto.
#### Producto:
# - official_model
#### Responde:
# ¿El modelo oficial GraphSAGE fue recuperado correctamente para iniciar
# la evaluación científica?

def load_official_model() -> dict:
    """
    Recupera y valida el modelo oficial GraphSAGE entrenado junto con
    su configuración oficial y sus metadatos exportados durante el
    entrenamiento definitivo.

    Returns
    -------
    dict
        Información oficial del modelo GraphSAGE.
    """

    # Validación
    if not BEST_MODEL_TORCH_FILE.exists():
        raise FileNotFoundError(
            f"No fue posible localizar {BEST_MODEL_TORCH_FILE.name}."
        )

    if not BEST_MODEL_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"No fue posible localizar {BEST_MODEL_CONFIG_FILE.name}."
        )

    if not BEST_MODEL_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"No fue posible localizar {BEST_MODEL_METADATA_FILE.name}."
        )

    # Recuperación
    try:
        with open(
            BEST_MODEL_CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            model_config = json.load(file)

    except Exception as error:
        raise RuntimeError(
            f"Error al cargar la configuración oficial: {error}"
        )

    try:
        with open(
            BEST_MODEL_METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            model_metadata = json.load(file)

    except Exception as error:
        raise RuntimeError(
            f"Error al cargar los metadatos del modelo oficial: {error}"
        )

    try:
        checkpoint = torch.load(
            BEST_MODEL_TORCH_FILE,
            weights_only=False
        )

        # Recuperación del modelo entrenado
        input_channels = checkpoint["model_state_dict"][
            "conv1.lin_l.weight"
        ].shape[1]

        trained_model = build_gnn_model(
            model_config=model_config,
            input_channels=input_channels,
            output_channels=1
        )

        trained_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        trained_model.eval()

    except Exception as error:
        raise RuntimeError(
            f"Error al cargar el modelo oficial: {error}"
        )

    # Validación
    required_config_keys = [
        "model_code",
        "model_name",
        "family",
        "hidden_channels",
        "dropout",
        "learning_rate",
        "weight_decay",
        "epochs"
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

    required_metadata_keys = [
        "model_code",
        "model_name",
        "family",
        "training_name",
        "training_version",
        "training_time",
        "training_loss",
        "loss_history",
        "epochs",
        "training_date",
        "export_format"
    ]

    missing_metadata_keys = [
        key
        for key in required_metadata_keys
        if key not in model_metadata
    ]

    if missing_metadata_keys:
        raise ValueError(
            "Los metadatos del modelo están incompletos: "
            f"{missing_metadata_keys}"
        )

    if model_config["family"] != "graph_neural_networks":
        raise ValueError(
            "La evaluación científica únicamente admite Graph Neural Networks."
        )

    if model_config["model_name"] != "graphsage":
        raise ValueError(
            "El modelo oficial del proyecto debe ser GraphSAGE."
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
            "El checkpoint del modelo está incompleto: "
            f"{missing_checkpoint_keys}"
        )

    official_model = {
        "checkpoint": checkpoint,
        "trained_model": trained_model,
        "model_config": model_config,
        "model_metadata": model_metadata
    }

    return official_model

OFFICIAL_MODEL = load_official_model()

print("-" * 80)
print("Bloque 3. Carga del Modelo Oficial Finalizada.")

# BLOQUE 4. Carga de los Datos ---------------------------------------------
## Objetivo: Recuperar el Dataset Científico Certificado y la colección
# oficial de GraphData espacio-temporal utilizados durante la evaluación
# científica del modelo oficial GraphSAGE.
#### Producto:
# - evaluation_data
#### Responde:
# ¿Los datos oficiales del proyecto fueron recuperados correctamente para
# iniciar la evaluación científica?

def load_evaluation_data() -> dict:
    """
    Recupera y valida el Dataset Científico Certificado y la colección
    oficial de GraphData utilizados durante la evaluación científica.

    Returns
    -------
    dict
        Datos oficiales de evaluación.
    """

    # Validación de rutas
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"No fue posible localizar {DATASET_FILE.name}."
        )

    if not GRAPH_DATA_DIR.exists():
        raise FileNotFoundError(
            f"No fue posible localizar el directorio de GraphData."
        )

    # Recuperación del Dataset Científico
    try:

        dataset = pd.read_parquet(
            DATASET_FILE
        )

    except Exception as error:

        raise RuntimeError(
            f"Error al cargar el Dataset Científico: {error}"
        )

    # Validación del Dataset
    if dataset.empty:

        raise ValueError(
            "El Dataset Científico está vacío."
        )

    # Recuperación de GraphData
    graph_files = sorted(
        GRAPH_DATA_DIR.glob(
            "graph_data_*.pt"
        )
    )

    graph_files = [
        graph_file
        for graph_file in graph_files
        if graph_file.stem != "graph_data_collection"
    ]

    if len(graph_files) == 0:

        raise FileNotFoundError(
            "No se encontraron GraphData oficiales."
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
                f"Error al cargar '{graph_file.name}': {error}"
            )

        graphs.append(graph)

    # Validación de la colección
    if len(graphs) == 0:

        raise ValueError(
            "La colección oficial de GraphData está vacía."
        )

    for index, graph in enumerate(
        graphs,
        start=1
    ):

        if graph is None:

            raise ValueError(
                f"GraphData #{index} no fue cargado correctamente."
            )

        if not hasattr(
            graph,
            "num_node_features"
        ):

            raise TypeError(
                f"GraphData #{index} no corresponde a un objeto válido "
                "de PyTorch Geometric."
            )

    # Construcción del producto oficial
    evaluation_data = {
        "dataset": dataset,
        "graphs": graphs
    }

    # Validación del producto
    required_keys = [
        "dataset",
        "graphs"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in evaluation_data
    ]

    if missing_keys:

        raise ValueError(
            f"El producto evaluation_data está incompleto: {missing_keys}"
        )

    return evaluation_data

EVALUATION_DATA = load_evaluation_data()

print("-" * 80)
print("Bloque 4. Carga de los Datos Finalizada.")

# BLOQUE 5. Generación de Predicciones -------------------------------------
## Objetivo: Generar las predicciones oficiales del modelo GraphSAGE
# utilizando la colección oficial de GraphData del proyecto.
#### Producto:
# - prediction_result
#### Responde:
# ¿El modelo oficial generó correctamente las predicciones del conjunto
# oficial de evaluación?

def generate_predictions(
    official_model: dict,
    evaluation_data: dict
) -> dict:
    """
    Genera y valida las predicciones oficiales del modelo GraphSAGE.

    Parameters
    ----------
    official_model : dict
        Modelo oficial recuperado.
    evaluation_data : dict
        Datos oficiales de evaluación.

    Returns
    -------
    dict
        Resultado oficial de la inferencia.
    """

    # Validación
    if not isinstance(official_model, dict):
        raise TypeError(
            "official_model debe ser un diccionario."
        )

    if not isinstance(evaluation_data, dict):
        raise TypeError(
            "evaluation_data debe ser un diccionario."
        )

    required_model_keys = [
        "checkpoint",
        "trained_model",
        "model_metadata",
        "model_config"
    ]

    missing_model_keys = [
        key
        for key in required_model_keys
        if key not in official_model
    ]

    if missing_model_keys:
        raise ValueError(
            f"official_model está incompleto: {missing_model_keys}"
        )

    required_data_keys = [
        "dataset",
        "graphs"
    ]

    missing_data_keys = [
        key
        for key in required_data_keys
        if key not in evaluation_data
    ]

    if missing_data_keys:
        raise ValueError(
            f"evaluation_data está incompleto: {missing_data_keys}"
        )

    # Recuperación
    
    trained_model = official_model["trained_model"]

    if not hasattr(trained_model, "eval"):
        raise TypeError(
            "El modelo oficial no corresponde a un módulo válido de PyTorch."
        )

    graphs = evaluation_data["graphs"]

    # Inferencia
    try:

        prediction_output = predict_gnn(
            model=trained_model,
            graphs=graphs
        )

    except Exception as error:

        raise RuntimeError(
            f"Error durante la generación de predicciones: {error}"
        )

    # Validación
    if prediction_output is None:
        raise RuntimeError(
            "La función oficial de predicción no devolvió resultados."
        )

    required_prediction_keys = [
        "y_true",
        "y_pred"
    ]

    missing_prediction_keys = [
        key
        for key in required_prediction_keys
        if key not in prediction_output
    ]

    if missing_prediction_keys:
        raise ValueError(
            "El resultado oficial de la inferencia está incompleto: "
            f"{missing_prediction_keys}"
        )

    y_true = np.asarray(
        prediction_output["y_true"]
    )

    y_pred = np.asarray(
        prediction_output["y_pred"]
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Las dimensiones de y_true y y_pred no coinciden."
        )

    # Construcción
    prediction_result = {

        "y_true": y_true,

        "y_pred": y_pred,

        "inference_time": prediction_output.get(
            "inference_time",
            None
        )
    }

    # Validación
    required_result_keys = [
        "y_true",
        "y_pred",
        "inference_time"
    ]

    missing_result_keys = [
        key
        for key in required_result_keys
        if key not in prediction_result
    ]

    if missing_result_keys:
        raise ValueError(
            "prediction_result está incompleto: "
            f"{missing_result_keys}"
        )

    return prediction_result


PREDICTION_RESULT = generate_predictions(
    official_model=OFFICIAL_MODEL,
    evaluation_data=EVALUATION_DATA
)

print("-" * 80)
print("Bloque 5. Generación de Predicciones Finalizada.")

# BLOQUE 6. Evaluación del Modelo ------------------------------------------
## Objetivo: Evaluar el desempeño predictivo del modelo oficial GraphSAGE
# utilizando las funciones oficiales de evaluación del proyecto.
#### Producto:
# - evaluation_result
#### Responde:
# ¿El modelo oficial fue evaluado correctamente?

def evaluate_official_model(
    prediction_result: dict
) -> dict:
    """
    Evalúa el desempeño predictivo del modelo oficial GraphSAGE.

    Parameters
    ----------
    prediction_result : dict
        Resultado oficial de la inferencia.

    Returns
    -------
    dict
        Resultado oficial de la evaluación.
    """

    # Validación
    if not isinstance(
        prediction_result,
        dict
    ):
        raise TypeError(
            "prediction_result debe ser un diccionario."
        )

    required_prediction_keys = [
        "y_true",
        "y_pred"
    ]

    missing_prediction_keys = [
        key
        for key in required_prediction_keys
        if key not in prediction_result
    ]

    if missing_prediction_keys:
        raise ValueError(
            "prediction_result está incompleto: "
            f"{missing_prediction_keys}"
        )

    # Recuperación
    y_true = prediction_result["y_true"]
    y_pred = prediction_result["y_pred"]

    # Validación
    if y_true is None:
        raise ValueError(
            "y_true no puede ser nulo."
        )

    if y_pred is None:
        raise ValueError(
            "y_pred no puede ser nulo."
        )

    if len(y_true) == 0:
        raise ValueError(
            "y_true está vacío."
        )

    if len(y_pred) == 0:
        raise ValueError(
            "y_pred está vacío."
        )

    # Evaluación
    try:

        evaluation_result = evaluate_gnn(
            y_true=y_true,
            y_pred=y_pred
        )

    except Exception as error:

        raise RuntimeError(
            f"Error durante la evaluación del modelo: {error}"
        )

    # Validación
    if evaluation_result is None:

        raise RuntimeError(
            "La función oficial de evaluación no devolvió resultados."
        )

    required_evaluation_keys = [
        "rmse",
        "mae",
        "mape",
        "r2"
    ]

    missing_evaluation_keys = [
        key
        for key in required_evaluation_keys
        if key not in evaluation_result
    ]

    if missing_evaluation_keys:

        raise ValueError(
            "evaluation_result está incompleto: "
            f"{missing_evaluation_keys}"
        )

    return evaluation_result


EVALUATION_RESULT = evaluate_official_model(
    prediction_result=PREDICTION_RESULT
)

print("-" * 80)
print("Bloque 6. Evaluación del Modelo Finalizado.")

# BLOQUE 7. Construcción del Resultado Oficial -----------------------------
## Objetivo: Consolidar el resultado oficial de la evaluación científica
# del modelo oficial GraphSAGE utilizando la estructura oficial del proyecto.
#### Producto:
# - evaluation_summary
#### Responde:
# ¿El resultado oficial de la evaluación fue construido correctamente?

def build_evaluation_summary(
    official_model: dict,
    prediction_result: dict,
    evaluation_result: dict
) -> dict:
    """
    Construye y valida el resultado oficial de la evaluación científica.

    Parameters
    ----------
    official_model : dict
        Modelo oficial recuperado.

    prediction_result : dict
        Resultado oficial de la inferencia.

    evaluation_result : dict
        Resultado oficial de la evaluación.

    Returns
    -------
    dict
        Resumen oficial de la evaluación.
    """

    # Validación -----------------------------------------------------------
    if not isinstance(official_model, dict):
        raise TypeError(
            "official_model debe ser un diccionario."
        )

    if not isinstance(prediction_result, dict):
        raise TypeError(
            "prediction_result debe ser un diccionario."
        )

    if not isinstance(evaluation_result, dict):
        raise TypeError(
            "evaluation_result debe ser un diccionario."
        )

    required_model_keys = [
        "checkpoint",
        "trained_model",
        "model_metadata",
        "model_config"
    ]

    missing_model_keys = [
        key
        for key in required_model_keys
        if key not in official_model
    ]

    if missing_model_keys:
        raise ValueError(
            f"official_model está incompleto: {missing_model_keys}"
        )

    # Recuperación ---------------------------------------------------------
    model_config = official_model["model_config"]
    model_metadata = official_model["model_metadata"]

    training_result = {
        "model": official_model["trained_model"],
        "training_time": model_metadata["training_time"],
        "loss": model_metadata["training_loss"],
        "loss_history": model_metadata["loss_history"],
        "epochs": model_metadata["epochs"]
    }

    # Construcción ---------------------------------------------------------
    try:

        evaluation_summary = build_benchmark_result(
            model_config=model_config,
            training_result=training_result,
            prediction_result=prediction_result,
            evaluation_result=evaluation_result
        )

        if "loss" not in evaluation_summary:
            raise ValueError(
                "El resultado del Benchmark no contiene la métrica 'loss'."
            )

        evaluation_summary["training_loss"] = evaluation_summary.pop("loss")
        evaluation_summary["epochs"] = model_metadata["epochs"]

    except Exception as error:

        raise RuntimeError(
            f"Error durante la construcción del resultado oficial: {error}"
        )

    # Validación -----------------------------------------------------------
    if evaluation_summary is None:

        raise RuntimeError(
            "No fue posible construir el resultado oficial."
        )

    required_summary_keys = [
        "model_code",
        "model_name",
        "family",
        "training_time",
        "training_loss",
        "epochs",
        "inference_time",
        "rmse",
        "mae",
        "mape",
        "r2"
    ]

    missing_summary_keys = [
        key
        for key in required_summary_keys
        if key not in evaluation_summary
    ]

    if missing_summary_keys:
        raise ValueError(
            f"El resumen oficial está incompleto: {missing_summary_keys}"
        )

    return evaluation_summary

EVALUATION_SUMMARY = build_evaluation_summary(
    official_model=OFFICIAL_MODEL,
    prediction_result=PREDICTION_RESULT,
    evaluation_result=EVALUATION_RESULT
)

print("-" * 80)
print("Bloque 7. Construcción del Resultado Oficial Finalizado.")

# BLOQUE 8. Explicabilidad Global ------------------------------------------
## Objetivo: Interpretar el comportamiento del modelo oficial GraphSAGE
# mediante el proceso oficial de Inteligencia Artificial Explicable (XAI),
# identificando la importancia global de las variables, generando
# visualizaciones, resúmenes científicos y productos reproducibles para la
# interpretación y validación del modelo.
#### Producto:
# - global_explainability
#### Responde:
# ¿El proceso oficial de explicabilidad global del modelo GraphSAGE fue
# ejecutado, validado y documentado correctamente?

print("-" * 80)
print("Bloque 8. Explicabilidad Global.")

# BLOQUE 8.1. Validación de Entradas ---------------------------------------
## Objetivo: Validar la estructura y consistencia de las entradas requeridas
# para ejecutar el proceso oficial de explicabilidad global del modelo
# GraphSAGE.
#### Producto:
# - Entradas validadas.
#### Responde:
# ¿Las entradas requeridas para la explicabilidad global son válidas?

def validate_global_explainability_inputs(
    official_model: dict,
    evaluation_data: dict,
    prediction_result: dict,
    evaluation_config: dict
) -> None:
    """
    Valida las entradas requeridas para ejecutar el proceso oficial de
    explicabilidad global.

    Parameters
    ----------
    official_model : dict
        Modelo oficial recuperado.

    evaluation_data : dict
        Datos oficiales de evaluación.

    prediction_result : dict
        Resultado oficial de la inferencia.

    evaluation_config : dict
        Configuración oficial de la evaluación.

    Returns
    -------
    None
    """

    # Validación
    if not isinstance(official_model, dict):
        raise TypeError(
            "official_model debe ser un diccionario."
        )

    if not isinstance(evaluation_data, dict):
        raise TypeError(
            "evaluation_data debe ser un diccionario."
        )

    if not isinstance(prediction_result, dict):
        raise TypeError(
            "prediction_result debe ser un diccionario."
        )

    if not isinstance(evaluation_config, dict):
        raise TypeError(
            "evaluation_config debe ser un diccionario."
        )

    # Validación
    required_model_keys = [
        "trained_model",
        "model_metadata",
        "model_config"
    ]

    missing_model_keys = [
        key
        for key in required_model_keys
        if key not in official_model
    ]

    if missing_model_keys:
        raise ValueError(
            "official_model está incompleto: "
            f"{missing_model_keys}"
        )

    # Validación
    required_data_keys = [
        "dataset",
        "graphs"
    ]

    missing_data_keys = [
        key
        for key in required_data_keys
        if key not in evaluation_data
    ]

    if missing_data_keys:
        raise ValueError(
            "evaluation_data está incompleto: "
            f"{missing_data_keys}"
        )

    # Validación
    required_prediction_keys = [
        "y_true",
        "y_pred"
    ]

    missing_prediction_keys = [
        key
        for key in required_prediction_keys
        if key not in prediction_result
    ]

    if missing_prediction_keys:
        raise ValueError(
            "prediction_result está incompleto: "
            f"{missing_prediction_keys}"
        )

    # Validación
    required_config_keys = [
        "calculate_global_explainability",
        "calculate_feature_importance",
        "calculate_local_explainability",
        "shap_batch_size"
    ]

    missing_config_keys = [
        key
        for key in required_config_keys
        if key not in evaluation_config
    ]

    if missing_config_keys:
        raise ValueError(
            "evaluation_config está incompleto: "
            f"{missing_config_keys}"
        )

    return None

# BLOQUE 8.2. Recuperación de Datos ----------------------------------------
## Objetivo: Recuperar la información oficial requerida para ejecutar el
# proceso de explicabilidad global del modelo GraphSAGE.
#### Producto:
# - explainability_data
#### Responde:
# ¿Se recuperó correctamente la información requerida para ejecutar la
# explicabilidad global?

def recover_global_explainability_data(
    official_model: dict,
    evaluation_data: dict,
    prediction_result: dict
) -> dict:
    """
    Recupera la información oficial requerida para ejecutar el proceso de
    explicabilidad global.

    Parameters
    ----------
    official_model : dict
        Modelo oficial recuperado.

    evaluation_data : dict
        Datos oficiales de evaluación.

    prediction_result : dict
        Resultado oficial de la inferencia.

    Returns
    -------
    dict
        Información oficial requerida para la explicabilidad global.
    """

    # Recuperación
    explainability_data = {

        "trained_model":
            official_model["trained_model"],

        "model_metadata":
            official_model["model_metadata"],

        "model_config":
            official_model["model_config"],

        "dataset":
            evaluation_data["dataset"],

        "graphs":
            evaluation_data["graphs"],

        "y_true":
            prediction_result["y_true"],

        "y_pred":
            prediction_result["y_pred"]

    }

    # Validación
    missing_values = [

        key
        for key, value in explainability_data.items()
        if value is None

    ]

    if missing_values:

        raise RuntimeError(
            "No fue posible recuperar la siguiente información de "
            f"explicabilidad: {missing_values}"
        )

    # Retorno
    return explainability_data

# BLOQUE 8.3. Ejecución de la Explicabilidad Global -------------------------
## Objetivo: Ejecutar el flujo oficial de explicabilidad global del modelo
# GraphSAGE utilizando los componentes oficiales del módulo de
# explicabilidad.
#### Producto:
# - global_explainability
#### Responde:
# ¿El proceso oficial de explicabilidad global fue ejecutado correctamente?

def execute_global_explainability(
    explainability_data: dict,
    evaluation_config: dict
) -> dict:
    """
    Ejecuta el flujo oficial de explicabilidad global.

    Parameters
    ----------
    explainability_data : dict
        Información oficial para la explicabilidad.

    evaluation_config : dict
        Configuración oficial de evaluación.

    Returns
    -------
    dict
        Resultado oficial de la explicabilidad global.
    """

    # Validación
    if not isinstance(explainability_data, dict):
        raise TypeError(
            "explainability_data debe ser un diccionario."
        )

    if not isinstance(evaluation_config, dict):
        raise TypeError(
            "evaluation_config debe ser un diccionario."
        )

    required_keys = [
        "trained_model",
        "model_metadata",
        "dataset",
        "graphs",
        "y_true"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in explainability_data
    ]

    if missing_keys:
        raise ValueError(
            "explainability_data está incompleto: "
            f"{missing_keys}"
        )

    # Recuperación
    trained_model = explainability_data["trained_model"]
    model_metadata = explainability_data["model_metadata"]
    dataset = explainability_data["dataset"]
    graphs = explainability_data["graphs"]
    y_true = explainability_data["y_true"]
    feature_names = FEATURE_COLUMNS.copy()

    if len(feature_names) == 0:
        raise RuntimeError(
            "No existen variables para la explicabilidad."
        )

    # Construcción
    global_explainability = (
        initialize_global_explainability(
            model_metadata=model_metadata,
            reports_dir=EVALUATION_REPORTS_DIR
        )
    )

    select_explainability_method(
        global_explainability
    )

    global_explainability = (
        build_explainer(
            global_explainability=global_explainability,
            trained_model=trained_model
        )
    )

    global_explainability = (
        initialize_explainability_context(
            global_explainability
        )
    )

    global_explainability = (
        build_explanation_targets(
            global_explainability=global_explainability,
            graphs=graphs
        )
    )

    global_explainability = (
        generate_node_explanations(
            global_explainability=global_explainability,
            graphs=graphs
        )
    )

    global_explainability = (
        extract_explanation_masks(
            global_explainability
        )
    )

    global_explainability = (
        aggregate_feature_importance(
            global_explainability
        )
    )

    global_explainability = (
        finalize_explainability(
            global_explainability
        )
    )

    # -------------
    print("-" * 80)
    print("Número de variables :", len(feature_names))

    feature_importance = (
        global_explainability["feature_importance"]
    )

    print(
        "Número de importancias:",
        len(feature_importance)
    )
    print("-" * 80)
    # ------------------

    global_explainability = (
        build_feature_ranking(
            global_explainability=global_explainability,
            feature_names=feature_names
        )
    )

    global_explainability = (
        generate_explainability_plots(
            global_explainability
        )
    )

    global_explainability = (
        build_scientific_summary(
            global_explainability
        )
    )

    global_explainability = (
        export_explainability_results(
            global_explainability
        )
    )

    validation_result = (
        validate_global_explainability(
            global_explainability
        )
    )

    if not validation_result:
        raise RuntimeError(
            "La validación oficial de la explicabilidad falló."
        )

    global_explainability = (
        audit_feature_importance(
            global_explainability
        )
    )

    global_explainability = (
        analyze_target_correlation(
            global_explainability=global_explainability,
            feature_data=dataset,
            target_data=y_true
        )
    )

    if global_explainability is None:
        raise RuntimeError(
            "No fue posible construir la explicabilidad global."
        )

    if global_explainability.get("status") != "VALIDATED":
        raise RuntimeError(
            "La explicabilidad global no finalizó correctamente."
        )

    # Retorno
    return global_explainability

# BLOQUE 8.4. Validación del Resultado -------------------------------------
## Objetivo: Validar la estructura del resultado oficial generado durante el
# proceso de explicabilidad global.
#### Producto:
# - global_explainability validado.
#### Responde:
# ¿El resultado oficial de la explicabilidad global posee la estructura
# requerida para continuar con la evaluación?

def validate_global_explainability_output(
    global_explainability: dict
) -> None:
    """
    Valida la estructura del resultado oficial de la explicabilidad global.

    Parameters
    ----------
    global_explainability : dict
        Resultado oficial de la explicabilidad global.

    Returns
    -------
    None
    """

    # Validación
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    # Validación
    required_keys = [
        "method",
        "explainer",
        "feature_importance",
        "feature_ranking",
        "scientific_summary",
        "plots",
        "exported_files",
        "importance_audit",
        "target_correlation",
        "target_correlation_summary",
        "status"
    ]

    missing_keys = [

        key
        for key in required_keys
        if key not in global_explainability

    ]

    if missing_keys:
        raise RuntimeError(
            "global_explainability está incompleto: "
            f"{missing_keys}"
        )

    if global_explainability["target_correlation_summary"] is None:
        raise RuntimeError(
            "target_correlation_summary no fue generado."
        )

    # Validación
    if global_explainability["feature_ranking"].empty:
        raise RuntimeError(
            "feature_ranking está vacío."
        )

    if global_explainability["scientific_summary"] is None:

        raise RuntimeError(
            "scientific_summary no fue generado."
        )

    if global_explainability["importance_audit"] is None:

        raise RuntimeError(
            "importance_audit no fue generado."
        )

    if global_explainability["target_correlation"].empty:
        raise RuntimeError(
            "target_correlation está vacío."
        )
   
    if global_explainability["status"] != "TARGET_CORRELATION_ANALYZED":

        raise RuntimeError(
            "La explicabilidad global no finalizó correctamente."
        )

    if len(global_explainability["plots"]) == 0:
        raise RuntimeError(
            "plots está vacío."
        )

    if len(global_explainability["exported_files"]) == 0:
        raise RuntimeError(
            "exported_files está vacío."
        )

    # Retorno
    return None

# BLOQUE 8.5. Construcción de la Explicabilidad Global ----------------------
## Objetivo: Orquestar el proceso oficial de explicabilidad global del modelo
# GraphSAGE mediante la ejecución secuencial de los componentes oficiales.
#### Producto:
# - global_explainability
#### Responde:
# ¿La explicabilidad global del modelo oficial fue construida
# correctamente?

def build_global_explainability(
    official_model: dict,
    evaluation_data: dict,
    prediction_result: dict,
    evaluation_config: dict
) -> dict:
    """
    Construye la explicabilidad global oficial del modelo GraphSAGE.

    Parameters
    ----------
    official_model : dict
        Modelo oficial recuperado.

    evaluation_data : dict
        Datos oficiales de evaluación.

    prediction_result : dict
        Resultado oficial de la inferencia.

    evaluation_config : dict
        Configuración oficial de evaluación.

    Returns
    -------
    dict
        Resultado oficial de la explicabilidad global.
    """

    # Validación
    validate_global_explainability_inputs(
        official_model=official_model,
        evaluation_data=evaluation_data,
        prediction_result=prediction_result,
        evaluation_config=evaluation_config
    )

    # Recuperación
    explainability_data = (
        recover_global_explainability_data(
            official_model=official_model,
            evaluation_data=evaluation_data,
            prediction_result=prediction_result
        )
    )

    # Construcción
    global_explainability = (
        execute_global_explainability(
            explainability_data=explainability_data,
            evaluation_config=evaluation_config
        )
    )

    # Validación
    validate_global_explainability_output(
        global_explainability=global_explainability
    )

    # Retorno
    return global_explainability

GLOBAL_EXPLAINABILITY = (
    build_global_explainability(
        official_model=OFFICIAL_MODEL,
        evaluation_data=EVALUATION_DATA,
        prediction_result=PREDICTION_RESULT,
        evaluation_config=EVALUATION_CONFIG
    )
)

# BLOQUE 8.6. Exportación de la Explicabilidad Global -----------------------
## Objetivo: Exportar el resultado oficial de la explicabilidad global para
# su reutilización en los módulos científicos del proyecto.
#### Producto:
# - Archivo oficial de explicabilidad global.
#### Responde:
# ¿La explicabilidad global fue exportada correctamente?

def export_global_explainability(
    global_explainability: dict,
    export_file: Path
) -> dict:
    """
    Exporta el resultado oficial de la explicabilidad global.

    Parameters
    ----------
    global_explainability : dict
        Resultado oficial de la explicabilidad global.

    export_file : Path
        Ruta del archivo de exportación.

    Returns
    -------
    dict
        Resultado de la exportación.
    """

    # Validación
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    if not isinstance(export_file, Path):
        raise TypeError(
            "export_file debe ser un objeto Path."
        )

    # Construcción
    export_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        export_file,
        "wb"
    ) as file:

        pickle.dump(
            global_explainability,
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    # Validación
    if not export_file.exists():
        raise FileNotFoundError(
            "No fue posible exportar la explicabilidad global."
        )

    export_result = {
        "status": "EXPORTED",
        "file": export_file,
        "size_bytes": export_file.stat().st_size
    }

    # Retorno
    return export_result

# BLOQUE 8.7. Recuperación de la Explicabilidad Global ----------------------
## Objetivo: Recuperar el resultado oficial de la explicabilidad global
# previamente exportado para su reutilización en los módulos científicos.
#### Producto:
# - global_explainability
#### Responde:
# ¿La explicabilidad global fue recuperada correctamente?

def recover_global_explainability(
    export_file: Path
) -> dict:
    """
    Recupera el resultado oficial de la explicabilidad global.

    Parameters
    ----------
    export_file : Path
        Ruta del archivo oficial de explicabilidad.

    Returns
    -------
    dict
        Resultado oficial de la explicabilidad global.
    """

    # Validación
    if not isinstance(export_file, Path):
        raise TypeError(
            "export_file debe ser un objeto Path."
        )

    if not export_file.exists():
        raise FileNotFoundError(
            f"No existe el archivo:\n{export_file}"
        )

    # Recuperación
    with open(
        export_file,
        "rb"
    ) as file:

        global_explainability = (
            pickle.load(file)
        )

    # Validación
    if not isinstance(
        global_explainability,
        dict
    ):
        raise TypeError(
            "La explicabilidad recuperada debe ser un "
            "diccionario."
        )

    required_keys = [
        "feature_ranking",
        "feature_importance",
        "scientific_summary"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in global_explainability
    ]

    if missing_keys:
        raise ValueError(
            "La explicabilidad recuperada está incompleta: "
            f"{missing_keys}"
        )

    # Retorno
    return global_explainability

print("-" * 80)
print("Bloque 8. Explicabilidad Global Finalizada.")

# BLOQUE 9. Construcción del Resultado Final -------------------------------
## Objetivo: Consolidar el resultado científico oficial del proceso de
# evaluación del modelo GraphSAGE, integrando las predicciones, métricas,
# explicabilidad, validaciones y productos generados.
#### Producto:
# - evaluation_output
#### Responde:
# ¿El proceso completo de evaluación científica finalizó correctamente?

print("-" * 80)
print("Bloque 9. Construcción del Resultado Final.")

def build_evaluation_output(
    official_model: dict,
    prediction_result: dict,
    evaluation_result: dict,
    evaluation_summary: dict,
    global_explainability: dict,
    evaluation_config: dict
) -> dict:
    """
    Consolida el resultado oficial del proceso de evaluación científica.

    Parameters
    ----------
    official_model : dict
        Modelo oficial recuperado.

    prediction_result : dict
        Resultado oficial de la inferencia.

    evaluation_result : dict
        Resultado oficial de la evaluación.

    evaluation_summary : dict
        Resumen oficial de la evaluación.

    global_explainability : dict
        Resultado oficial de la explicabilidad.

    evaluation_config : dict
        Configuración oficial de la evaluación.

    Returns
    -------
    dict
        Resultado oficial del proceso completo de evaluación.
    """

    # Validación -----------------------------------------------------------
    if not isinstance(official_model, dict):
        raise TypeError(
            "official_model debe ser un diccionario."
        )

    if not isinstance(prediction_result, dict):
        raise TypeError(
            "prediction_result debe ser un diccionario."
        )

    if not isinstance(evaluation_result, dict):
        raise TypeError(
            "evaluation_result debe ser un diccionario."
        )

    if not isinstance(evaluation_summary, dict):
        raise TypeError(
            "evaluation_summary debe ser un diccionario."
        )

    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    if not isinstance(evaluation_config, dict):
        raise TypeError(
            "evaluation_config debe ser un diccionario."
        )

    # ---------------------------------------------------------------------
    # official_model
    # ---------------------------------------------------------------------

    required_model_keys = [
        "trained_model",
        "model_metadata",
        "model_config"
    ]

    missing_model_keys = [
        key
        for key in required_model_keys
        if key not in official_model
    ]

    if missing_model_keys:
        raise ValueError(
            f"official_model está incompleto: {missing_model_keys}"
        )

    # ---------------------------------------------------------------------
    # evaluation_summary
    # ---------------------------------------------------------------------

    required_summary_keys = [
        "model_code",
        "model_name",
        "family",
        "model",
        "training_time",
        "training_loss",
        "epochs",
        "inference_time",
        "rmse",
        "mae",
        "mape",
        "r2"
    ]

    missing_summary_keys = [
        key
        for key in required_summary_keys
        if key not in evaluation_summary
    ]

    if missing_summary_keys:
        raise ValueError(
            "evaluation_summary está incompleto: "
            f"{missing_summary_keys}"
        )

    # ---------------------------------------------------------------------
    # global_explainability
    # ---------------------------------------------------------------------

    required_explainability_keys = [
        "method",
        "feature_ranking",
        "scientific_summary",
        "status",
        "target_correlation",
        "target_correlation_summary"
    ]

    missing_explainability_keys = [
        key
        for key in required_explainability_keys
        if key not in global_explainability
    ]

    if missing_explainability_keys:
        raise ValueError(
            "global_explainability está incompleto: "
            f"{missing_explainability_keys}"
        )

    # Recuperación ---------------------------------------------------------
    model_metadata = official_model["model_metadata"]

    feature_ranking = global_explainability[
        "feature_ranking"
    ]

    scientific_summary = global_explainability[
        "scientific_summary"
    ]

    explainability_method = global_explainability[
        "method"
    ]

    explainability_status = global_explainability[
        "status"
    ]

    status = (
        "SUCCESS"
        if explainability_status == "TARGET_CORRELATION_ANALYZED"
        else "FAILED"
    )

    # Construcción ---------------------------------------------------------
    evaluation_output = {
        "status": status,
        "official_model": official_model,
        "prediction_result": prediction_result,
        "evaluation_result": evaluation_result,
        "evaluation_summary": evaluation_summary,
        "global_explainability": global_explainability,

        "summary": {
            "model_code": evaluation_summary["model_code"],
            "model_name": evaluation_summary["model_name"],
            "family": evaluation_summary["family"],

            # NUEVOS CAMPOS
            "training_time": evaluation_summary["training_time"],
            "training_loss": evaluation_summary["training_loss"],
            "epochs": evaluation_summary["epochs"],

            # Ya existentes
            "inference_time": evaluation_summary["inference_time"],
            "rmse": evaluation_summary["rmse"],
            "mae": evaluation_summary["mae"],
            "mape": evaluation_summary["mape"],
            "r2": evaluation_summary["r2"],

            "explainability_method": explainability_method,
            "top_10_variables": feature_ranking.head(10),
            "explainability_status": explainability_status
        },

        "generated_products": {
            "reports_directory":
                EVALUATION_REPORTS_DIR,

            "scientific_summary":
                scientific_summary,

            "feature_ranking":
                feature_ranking

        },

        "validation": {
            "prediction_completed":
                prediction_result is not None,

            "evaluation_completed":
                evaluation_result is not None,

            "summary_completed":
                evaluation_summary is not None,

            "explainability_completed":
                explainability_status == "TARGET_CORRELATION_ANALYZED"

        },

        "metadata": {
            "model_metadata":
                model_metadata,

            "evaluation_config":
                evaluation_config
        }
    }

    # Validación -----------------------------------------------------------
    required_output_keys = [
        "status",
        "official_model",
        "prediction_result",
        "evaluation_result",
        "evaluation_summary",
        "global_explainability",
        "summary",
        "generated_products",
        "validation",
        "metadata"
    ]

    missing_output_keys = [

        key
        for key in required_output_keys
        if key not in evaluation_output

    ]

    if missing_output_keys:

        raise ValueError(
            "evaluation_output está incompleto: "
            f"{missing_output_keys}"
        )

    if evaluation_output["status"] != "SUCCESS":

        raise RuntimeError(
            "El proceso de evaluación científica no finalizó correctamente."
        )

    return evaluation_output


EVALUATION_OUTPUT = build_evaluation_output(
    official_model=OFFICIAL_MODEL,
    prediction_result=PREDICTION_RESULT,
    evaluation_result=EVALUATION_RESULT,
    evaluation_summary=EVALUATION_SUMMARY,
    global_explainability=GLOBAL_EXPLAINABILITY,
    evaluation_config=EVALUATION_CONFIG
)

print("-" * 80)
print("Bloque 9. Resultado Finalizado.")

# BLOQUE 10. Reporte Ejecutivo ---------------------------------------------
## Objetivo: Presentar el resultado ejecutivo del proceso oficial de
# evaluación científica del modelo GraphSAGE.
#### Producto:
# - Reporte Ejecutivo
#### Responde:
# ¿Cuáles fueron los resultados finales del proceso de evaluación científica?

print("-" * 80)
print("Bloque 10. Reporte Ejecutivo.")

def report_evaluation_output(
    evaluation_output: dict
) -> None:
    """
    Presenta el reporte ejecutivo del proceso oficial de evaluación.

    Parameters
    ----------
    evaluation_output : dict
        Resultado oficial del proceso de evaluación.

    Returns
    -------
    None
    """

    # Validación -----------------------------------------------------------
    if not isinstance(evaluation_output, dict):
        raise TypeError(
            "evaluation_output debe ser un diccionario."
        )

    required_keys = [
        "status",
        "summary",
        "validation",
        "generated_products",
        "metadata"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in evaluation_output
    ]

    if missing_keys:
        raise ValueError(
            "evaluation_output está incompleto: "
            f"{missing_keys}"
        )

    # Recuperación ---------------------------------------------------------
    summary = evaluation_output["summary"]

    validation = evaluation_output["validation"]

    generated_products = evaluation_output[
        "generated_products"
    ]

    metadata = evaluation_output["metadata"]

    # Reporte --------------------------------------------------------------
    print("-" * 80)
    print("REPORTE EJECUTIVO DE LA EVALUACIÓN CIENTÍFICA")
    print("-" * 80)

    print(f"Estado                 : {evaluation_output['status']}")
    print(f"Modelo                 : {summary['model_name']}")
    print(f"Familia                : {summary['family']}")

    print("-" * 80)

    print("MÉTRICAS DE DESEMPEÑO")
    print(f"Tiempo de entrenamiento : {summary['training_time']}")
    print(f"Pérdida final           : {summary['training_loss']:.6f}")
    print(f"Épocas                  : {summary['epochs']}")

    print("-" * 80)

    print(f"RMSE                    : {summary['rmse']:.6f}")
    print(f"MAE                     : {summary['mae']:.6f}")
    print(f"MAPE                    : {summary['mape']:.6f}")
    print(f"R²                      : {summary['r2']:.6f}")
    print(f"Tiempo de inferencia    : {summary['inference_time']}")

    print("-" * 80)

    print("EXPLICABILIDAD")
    print(f"Método                 : {summary['explainability_method']}")
    print(f"Estado                 : {summary['explainability_status']}")

    print("Top 10 variables:")

    top_variables = summary["top_10_variables"]
    for index, row in enumerate(
        top_variables.itertuples(index=False),
        start=1
    ):

        print(
            f"{index:02d}. "
            f"{row.variable} "
            f"({row.importance:.6f})"
        )

    print("-" * 80)

    print("VALIDACIÓN DEL PROCESO")

    for key, value in validation.items():
        print(f"{key:<30}: {'OK' if value else 'ERROR'}")

    print("-" * 80)

    print("PRODUCTOS GENERADOS")

    for key, value in generated_products.items():

        if key == "feature_ranking":

            print(
                f"{key:<30}: "
                f"{len(value)} variables"
            )

        else:

            print(
                f"{key:<30}: {value}"
            )

    print("-" * 80)

    print("METADATOS")

    for key, value in metadata.items():

        if isinstance(value, dict):
            print(
                f"{key:<30}: "
                f"{len(value)} elementos"
            )
        else:
            print(
                f"{key:<30}: {value}"
            )

    print("-" * 80)

report_evaluation_output(
    EVALUATION_OUTPUT
)

print("-" * 80)
print("Bloque 10. Reporte Ejecutivo Finalizado.")
