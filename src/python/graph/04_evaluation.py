# graph-04_evaluation.py

# ==============================================================================
# BLOQUE 1. IMPORTACIONES
# ==============================================================================
## Objetivo:
# Importar las dependencias necesarias para ejecutar la evaluación científica
# del Modelo Oficial GraphSAGE utilizando el Dataset Científico Certificado,
# la colección oficial de GraphData y los componentes oficiales de
# explicabilidad del proyecto.
#
## Producto:
# - Dependencias científicas cargadas correctamente.
# - Entorno de evaluación configurado.
#
## Responde:
# ¿El protocolo científico dispone de todas las dependencias necesarias para
# reconstruir, evaluar, interpretar y validar el Modelo Oficial del proyecto?
# ==============================================================================

print("-" * 80)
print("Bloque 1. Importaciones.")

# ------------------------------------------------------------------------------
# 1. Librerías estándar de Python
# ------------------------------------------------------------------------------

import json
import pickle
import warnings
import joblib
from pathlib import Path

# ------------------------------------------------------------------------------
# 2. Librerías científicas
# ------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ------------------------------------------------------------------------------
# 3. Configuración oficial del proyecto
# ------------------------------------------------------------------------------

from src.python.config.config_project import (
    PROJECT_SEED,
    FEATURE_COLUMNS,
)

# ------------------------------------------------------------------------------
# 4. Rutas oficiales del proyecto
# ------------------------------------------------------------------------------

from src.python.config.paths import (
    DATASET_FILE,
    BENCHMARK_DATA_FILE,
    EVALUATION_REPORTS_DIR,
    OFFICIAL_MODEL_TORCH_FILE,
    OFFICIAL_MODEL_CONFIG_FILE,
    OFFICIAL_MODEL_METADATA_FILE,
)

# ------------------------------------------------------------------------------
# 5. Utilidades oficiales
# ------------------------------------------------------------------------------

from src.python.utils.results import (
    build_benchmark_result,
)

# ------------------------------------------------------------------------------
# 6. Modelos Graph Neural Networks
# ------------------------------------------------------------------------------

from src.python.models.graph_neural_networks import (
    build_gnn_model,
    evaluate_gnn,
    predict_gnn,
)

# ------------------------------------------------------------------------------
# 7. Componentes oficiales de Explicabilidad
# ------------------------------------------------------------------------------

from src.python.analysis.explainability import (
    initialize_global_explainability,
    initialize_explainability_context,

    select_explainability_method,
    build_explainer,
    build_explanation_targets,

    generate_node_explanations,
    extract_explanation_masks,
    aggregate_feature_importance,

    build_feature_ranking,
    generate_explainability_plots,
    build_scientific_summary,

    export_explainability_results,

    validate_global_explainability,

    finalize_explainability,

    audit_feature_importance,
    analyze_target_correlation,
)

# ------------------------------------------------------------------------------
# 8. Configuración del entorno
# ------------------------------------------------------------------------------

warnings.filterwarnings("ignore")

print("-" * 80)
print("Bloque 1. Importaciones completado.")

# ==============================================================================
# BLOQUE 2. CONFIGURACIÓN DE LA EVALUACIÓN CIENTÍFICA
# ==============================================================================
## Objetivo:
# Construir y validar la configuración oficial que gobernará el protocolo de
# evaluación científica del Modelo Oficial GraphSAGE, garantizando la
# reproducibilidad, consistencia e integridad del proceso experimental.
#
## Producto:
# - EVALUATION_CONFIG
#
## Responde:
# ¿La evaluación científica dispone de una configuración oficial, validada y
# reproducible para ejecutar el protocolo de evaluación del Modelo Oficial?
# ==============================================================================

# BLOQUE 2.1. Construcción de la Configuración -------------------------------

def build_evaluation_configuration() -> dict:
    """
    Construye la configuración oficial de la evaluación científica.

    Returns
    -------
    dict
        Configuración oficial de la evaluación científica.
    """

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
        "random_state": PROJECT_SEED,
    }

    required_products = [
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
        "random_state",
    ]

    missing_products = [
        product
        for product in required_products
        if product not in evaluation_config
    ]

    if missing_products:
        raise RuntimeError(
            "EvaluationConfiguration está incompleto: "
            f"{missing_products}"
        )

    return evaluation_config

# BLOQUE 2.2. Validación de la Estructura ------------------------------------

def validate_evaluation_configuration_structure(
    evaluation_config: dict
) -> None:
    """
    Valida la estructura de la configuración oficial de la evaluación
    científica.

    Parameters
    ----------
    evaluation_config : dict
        Configuración oficial de la evaluación científica.

    Returns
    -------
    None
    """

    if evaluation_config is None:
        raise ValueError(
            "La configuración de la evaluación no puede ser nula."
        )

    if not isinstance(evaluation_config, dict):
        raise TypeError(
            "evaluation_config debe ser un diccionario."
        )

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
        "random_state",
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

    for key in required_keys:

        if evaluation_config[key] is None:
            raise ValueError(
                f"'{key}' es inválido."
            )

    return None

# BLOQUE 2.3. Validación de Tipos --------------------------------------------

def validate_evaluation_configuration_types(
    evaluation_config: dict
) -> None:
    """
    Valida los tipos de datos de la configuración oficial de la evaluación
    científica.

    Parameters
    ----------
    evaluation_config : dict
        Configuración oficial de la evaluación científica.

    Returns
    -------
    None
    """

    if evaluation_config is None:
        raise ValueError(
            "La configuración de la evaluación no puede ser nula."
        )

    if not isinstance(evaluation_config, dict):
        raise TypeError(
            "evaluation_config debe ser un diccionario."
        )

    expected_types = {
        "evaluation_name": str,
        "evaluation_version": str,
        "prediction_source": str,
        "shap_batch_size": int,
        "random_state": int,
    }

    for key, expected_type in expected_types.items():

        if not isinstance(
            evaluation_config[key],
            expected_type
        ):
            raise TypeError(
                f"{key} debe ser de tipo "
                f"{expected_type.__name__}."
            )

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
        "save_evaluation_report",
    ]

    for key in boolean_keys:

        if not isinstance(
            evaluation_config[key],
            bool
        ):
            raise TypeError(
                f"{key} debe ser un valor booleano."
            )

    return None

# BLOQUE 2.4. Validación del Contenido ---------------------------------------

def validate_evaluation_configuration_values(
    evaluation_config: dict
) -> None:
    """
    Valida el contenido de la configuración oficial de la evaluación
    científica.

    Parameters
    ----------
    evaluation_config : dict
        Configuración oficial de la evaluación científica.

    Returns
    -------
    None
    """

    if evaluation_config is None:
        raise ValueError(
            "La configuración de la evaluación no puede ser nula."
        )

    if not isinstance(evaluation_config, dict):
        raise TypeError(
            "evaluation_config debe ser un diccionario."
        )

    validate_evaluation_configuration_structure(
        evaluation_config
    )

    validate_evaluation_configuration_types(
        evaluation_config
    )

    if not evaluation_config["evaluation_name"].strip():
        raise ValueError(
            "evaluation_name está vacío."
        )

    if not evaluation_config["evaluation_version"].strip():
        raise ValueError(
            "evaluation_version está vacío."
        )

    if not evaluation_config["prediction_source"].strip():
        raise ValueError(
            "prediction_source está vacío."
        )

    if evaluation_config["prediction_source"] != "official_model":
        raise ValueError(
            "prediction_source debe ser 'official_model'."
        )

    if evaluation_config["shap_batch_size"] <= 0:
        raise ValueError(
            "shap_batch_size debe ser mayor que cero."
        )

    if evaluation_config["random_state"] < 0:
        raise ValueError(
            "random_state debe ser mayor o igual a cero."
        )

    return None

# BLOQUE 2.5. Construcción de la Configuración Oficial ------------------------

def get_evaluation_configuration() -> dict:
    """
    Construye y valida la configuración oficial de la evaluación científica.

    Returns
    -------
    dict
        Configuración oficial de la evaluación científica validada.
    """

    try:

        # Construcción
        evaluation_config = build_evaluation_configuration()

        # Validación de la estructura
        validate_evaluation_configuration_structure(
            evaluation_config=evaluation_config
        )

        # Validación de los tipos de datos
        validate_evaluation_configuration_types(
            evaluation_config=evaluation_config
        )

        # Validación del contenido
        validate_evaluation_configuration_values(
            evaluation_config=evaluation_config
        )

        required_products = [
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
            "random_state",
        ]

        missing_products = [
            product
            for product in required_products
            if product not in evaluation_config
        ]

        if missing_products:
            raise RuntimeError(
                "EvaluationConfiguration está incompleto: "
                f"{missing_products}"
            )

        return evaluation_config

    except Exception as error:

        raise RuntimeError(
            "No fue posible construir la configuración oficial de la evaluación."
        ) from error

# ------------------------------------------------------------------------------
# Construcción del producto oficial
# ------------------------------------------------------------------------------

EVALUATION_CONFIG = get_evaluation_configuration()

print("-" * 80)
print("Configuración Oficial de la Evaluación Científica")
print("-" * 80)

print(
    f"Evaluación                : "
    f"{EVALUATION_CONFIG['evaluation_name']}"
)

print(
    f"Versión                   : "
    f"{EVALUATION_CONFIG['evaluation_version']}"
)

print(
    f"Fuente de predicción      : "
    f"{EVALUATION_CONFIG['prediction_source']}"
)

print(
    f"Explicabilidad Global     : "
    f"{EVALUATION_CONFIG['calculate_global_explainability']}"
)

print(
    f"Explicabilidad Local      : "
    f"{EVALUATION_CONFIG['calculate_local_explainability']}"
)

print(
    f"Importancia de Variables  : "
    f"{EVALUATION_CONFIG['calculate_feature_importance']}"
)

print(
    f"Semilla Científica        : "
    f"{EVALUATION_CONFIG['random_state']}"
)

print("-" * 80)
print("Bloque 2. Configuración oficial construida correctamente.")

# BLOQUE 3. Carga del Modelo Oficial ----------------------------------------
## Objetivo: Recuperar el modelo oficial GraphSAGE entrenado y sus recursos
# científicos para iniciar la evaluación científica del proyecto.
#
## Producto:
# - official_model
#
## Responde:
# ¿El modelo oficial GraphSAGE fue recuperado correctamente para iniciar la
# evaluación científica?

# BLOQUE 3.1. Validación de Archivos -----------------------------------------

def validate_official_model_files() -> None:
    """
    Valida la existencia de los archivos oficiales del modelo GraphSAGE.

    Returns
    -------
    None
    """

    required_files = {
        "OFFICIAL_MODEL_TORCH_FILE": OFFICIAL_MODEL_TORCH_FILE,
        "OFFICIAL_MODEL_CONFIG_FILE": OFFICIAL_MODEL_CONFIG_FILE,
        "OFFICIAL_MODEL_METADATA_FILE": OFFICIAL_MODEL_METADATA_FILE,
    }

    for product, file_path in required_files.items():

        if file_path is None:
            raise ValueError(
                f"{product} no puede ser nulo."
            )

        if not file_path.exists():
            raise FileNotFoundError(
                f"No fue posible localizar {file_path.name}."
            )

    return None

# BLOQUE 3.2. Recuperación de la Configuración Oficial -----------------------

def load_official_model_configuration() -> dict:
    """
    Recupera la configuración oficial del modelo GraphSAGE.

    Returns
    -------
    dict
        Configuración oficial completa del modelo.
    """

    try:

        with open(
            OFFICIAL_MODEL_CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            configuration = json.load(file)

        if not isinstance(configuration, dict):
            raise TypeError(
                "La configuración oficial debe ser un diccionario."
            )

        required_keys = [
            "model_code",
            "model_name",
            "family",
            "model_config"
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in configuration
        ]

        if missing_keys:
            raise ValueError(
                "La configuración oficial está incompleta: "
                f"{missing_keys}"
            )

        for key in required_keys:

            if configuration[key] is None:
                raise ValueError(
                    f"'{key}' es inválido."
                )

        if not isinstance(
            configuration["model_config"],
            dict
        ):
            raise TypeError(
                "model_config debe ser un diccionario."
            )

        model_config = configuration["model_config"].copy()

        model_config["model_code"] = configuration["model_code"]
        model_config["model_name"] = configuration["model_name"]
        model_config["family"] = configuration["family"]

        required_products = [
            "model_code",
            "model_name",
            "family",
        ]

        missing_products = [
            product
            for product in required_products
            if product not in model_config
        ]

        if missing_products:
            raise RuntimeError(
                "ModelConfiguration está incompleto: "
                f"{missing_products}"
            )

        return model_config

    except Exception as error:

        raise RuntimeError(
            "No fue posible recuperar la configuración oficial del modelo."
        ) from error

# BLOQUE 3.3. Recuperación de los Metadatos Oficiales ------------------------

def load_official_model_metadata() -> dict:
    """
    Recupera los metadatos oficiales del modelo GraphSAGE.

    Returns
    -------
    dict
        Metadatos oficiales del modelo.
    """

    validate_official_model_files()

    try:

        with open(
            OFFICIAL_MODEL_METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            model_metadata = json.load(file)

        if not isinstance(model_metadata, dict):
            raise TypeError(
                "Los metadatos oficiales deben ser un diccionario."
            )

        required_products = [
            "model_code",
            "model_name",
            "family",
        ]

        missing_products = [
            product
            for product in required_products
            if product not in model_metadata
        ]

        if missing_products:
            raise ValueError(
                "ModelMetadata está incompleto: "
                f"{missing_products}"
            )

        for product in required_products:

            if model_metadata[product] is None:
                raise ValueError(
                    f"'{product}' es inválido."
                )

        return model_metadata

    except Exception as error:

        raise RuntimeError(
            "No fue posible recuperar los metadatos oficiales del modelo."
        ) from error
    
# BLOQUE 3.4. Recuperación del Modelo Oficial -------------------------------

def load_trained_official_model(
    model_config: dict
) -> tuple[dict, torch.nn.Module]:
    """
    Recupera el modelo oficial GraphSAGE entrenado.

    Parameters
    ----------
    model_config : dict
        Configuración oficial del modelo.

    Returns
    -------
    tuple[dict, torch.nn.Module]
        Checkpoint oficial y modelo GraphSAGE entrenado.
    """

    if model_config is None:
        raise ValueError(
            "La configuración del modelo no puede ser nula."
        )

    if not isinstance(model_config, dict):
        raise TypeError(
            "model_config debe ser un diccionario."
        )

    validate_official_model_files()

    try:

        checkpoint = torch.load(
            OFFICIAL_MODEL_TORCH_FILE,
            weights_only=False
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "El checkpoint debe ser un diccionario."
            )

        required_products = [
            "model_state_dict",
        ]

        missing_products = [
            product
            for product in required_products
            if product not in checkpoint
        ]

        if missing_products:
            raise ValueError(
                "Checkpoint incompleto: "
                f"{missing_products}"
            )

        state_dict = checkpoint["model_state_dict"]

        if "conv1.lin_l.weight" not in state_dict:
            raise ValueError(
                "No fue posible localizar "
                "'conv1.lin_l.weight' en el checkpoint."
            )

        input_channels = state_dict[
            "conv1.lin_l.weight"
        ].shape[1]

        trained_model = build_gnn_model(
            model_config=model_config,
            input_channels=input_channels,
            output_channels=1
        )

        if not isinstance(
            trained_model,
            nn.Module
        ):
            raise TypeError(
                "El modelo construido no corresponde a nn.Module."
            )

        trained_model.load_state_dict(
            state_dict
        )

        trained_model.eval()

        return checkpoint, trained_model

    except Exception as error:

        raise RuntimeError(
            "No fue posible recuperar el modelo oficial GraphSAGE."
        ) from error
    
# BLOQUE 3.5. Validación de la Configuración Oficial -------------------------

def validate_official_model_configuration(
    model_config: dict
) -> None:
    """
    Valida la configuración oficial del modelo GraphSAGE.

    Parameters
    ----------
    model_config : dict
        Configuración oficial del modelo.

    Returns
    -------
    None
    """

    if model_config is None:
        raise ValueError(
            "La configuración del modelo no puede ser nula."
        )

    if not isinstance(model_config, dict):
        raise TypeError(
            "model_config debe ser un diccionario."
        )

    required_config_keys = [
        "model_code",
        "model_name",
        "family",
        "hidden_channels",
        "dropout",
        "learning_rate",
        "weight_decay",
        "epochs",
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

    for key in required_config_keys:

        if model_config[key] is None:
            raise ValueError(
                f"'{key}' es inválido."
            )

    if not isinstance(model_config["model_code"], str):
        raise TypeError(
            "model_code debe ser una cadena."
        )

    if not isinstance(model_config["model_name"], str):
        raise TypeError(
            "model_name debe ser una cadena."
        )

    if not isinstance(model_config["family"], str):
        raise TypeError(
            "family debe ser una cadena."
        )

    if not model_config["model_code"].strip():
        raise ValueError(
            "model_code está vacío."
        )

    if not model_config["model_name"].strip():
        raise ValueError(
            "model_name está vacío."
        )

    if not model_config["family"].strip():
        raise ValueError(
            "family está vacía."
        )

    if model_config["hidden_channels"] <= 0:
        raise ValueError(
            "hidden_channels debe ser mayor que cero."
        )

    if not 0 <= model_config["dropout"] <= 1:
        raise ValueError(
            "dropout debe estar entre 0 y 1."
        )

    if model_config["learning_rate"] <= 0:
        raise ValueError(
            "learning_rate debe ser mayor que cero."
        )

    if model_config["weight_decay"] < 0:
        raise ValueError(
            "weight_decay no puede ser negativo."
        )

    if model_config["epochs"] <= 0:
        raise ValueError(
            "epochs debe ser mayor que cero."
        )

    return None

# BLOQUE 3.6. Validación de los Metadatos Oficiales --------------------------

def validate_official_model_metadata(
    model_metadata: dict
) -> None:
    """
    Valida los metadatos oficiales del modelo GraphSAGE.

    Parameters
    ----------
    model_metadata : dict
        Metadatos oficiales del modelo.

    Returns
    -------
    None
    """

    if model_metadata is None:
        raise ValueError(
            "Los metadatos del modelo no pueden ser nulos."
        )

    if not isinstance(model_metadata, dict):
        raise TypeError(
            "model_metadata debe ser un diccionario."
        )

    required_metadata_keys = [
        "model_code",
        "model_name",
        "family",
        "model_config",
        "graphs",
        "training_time",
        "training_loss",
        "loss_history",
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
            "Los metadatos del modelo oficial están incompletos: "
            f"{missing_metadata_keys}"
        )

    for key in required_metadata_keys:

        if model_metadata[key] is None:
            raise ValueError(
                f"'{key}' es inválido."
            )

    if not isinstance(model_metadata["model_code"], str):
        raise TypeError(
            "model_code debe ser una cadena."
        )

    if not isinstance(model_metadata["model_name"], str):
        raise TypeError(
            "model_name debe ser una cadena."
        )

    if not isinstance(model_metadata["family"], str):
        raise TypeError(
            "family debe ser una cadena."
        )

    if not isinstance(model_metadata["model_config"], dict):
        raise TypeError(
            "model_config debe ser un diccionario."
        )

    if not isinstance(model_metadata["graphs"], int):
        raise TypeError(
            "graphs debe ser un entero."
        )

    if not isinstance(model_metadata["training_time"], (int, float)):
        raise TypeError(
            "training_time debe ser numérico."
        )

    if not isinstance(model_metadata["training_loss"], (int, float)):
        raise TypeError(
            "training_loss debe ser numérico."
        )

    if not isinstance(model_metadata["loss_history"], list):
        raise TypeError(
            "loss_history debe ser una lista."
        )

    if not isinstance(model_metadata["training_date"], str):
        raise TypeError(
            "training_date debe ser una cadena."
        )

    if not isinstance(model_metadata["export_format"], str):
        raise TypeError(
            "export_format debe ser una cadena."
        )

    if not model_metadata["model_code"].strip():
        raise ValueError(
            "model_code está vacío."
        )

    if not model_metadata["model_name"].strip():
        raise ValueError(
            "model_name está vacío."
        )

    if not model_metadata["family"].strip():
        raise ValueError(
            "family está vacía."
        )

    if model_metadata["graphs"] <= 0:
        raise ValueError(
            "graphs debe ser mayor que cero."
        )

    if model_metadata["training_time"] < 0:
        raise ValueError(
            "training_time no puede ser negativo."
        )

    if model_metadata["training_loss"] < 0:
        raise ValueError(
            "training_loss no puede ser negativo."
        )

    if len(model_metadata["loss_history"]) == 0:
        raise ValueError(
            "loss_history está vacío."
        )

    if not model_metadata["training_date"].strip():
        raise ValueError(
            "training_date está vacío."
        )

    if not model_metadata["export_format"].strip():
        raise ValueError(
            "export_format está vacío."
        )

    return None

# BLOQUE 3.7. Validación del Modelo Oficial ---------------------------------

def validate_official_model(
    checkpoint: dict,
    model_config: dict
) -> None:
    """
    Valida el modelo oficial GraphSAGE recuperado.

    Parameters
    ----------
    checkpoint : dict
        Checkpoint oficial del modelo.

    model_config : dict
        Configuración oficial del modelo.

    Returns
    -------
    None
    """

    if checkpoint is None:
        raise ValueError(
            "El checkpoint no puede ser nulo."
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "checkpoint debe ser un diccionario."
        )

    if model_config is None:
        raise ValueError(
            "La configuración del modelo no puede ser nula."
        )

    if not isinstance(model_config, dict):
        raise TypeError(
            "model_config debe ser un diccionario."
        )

    validate_official_model_configuration(
        model_config=model_config
    )

    if model_config["family"] != "graph_neural_networks":
        raise ValueError(
            "La evaluación científica únicamente admite Graph Neural Networks."
        )

    if model_config["model_name"].lower() != "graphsage":
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

    for key in required_checkpoint_keys:

        if checkpoint[key] is None:
            raise ValueError(
                f"'{key}' es inválido."
            )

    if not isinstance(
        checkpoint["model_state_dict"],
        dict
    ):
        raise TypeError(
            "model_state_dict debe ser un diccionario."
        )

    if not isinstance(
        checkpoint["model_config"],
        dict
    ):
        raise TypeError(
            "model_config del checkpoint debe ser un diccionario."
        )

    return None

# BLOQUE 3.8. Construcción del Modelo Oficial -------------------------------

def get_official_model() -> dict:
    """
    Construye el modelo oficial GraphSAGE para la evaluación científica.

    Returns
    -------
    dict
        Modelo oficial GraphSAGE.
    """

    try:

        # Validación de archivos
        validate_official_model_files()

        # Recuperación de productos
        model_config = load_official_model_configuration()

        model_metadata = load_official_model_metadata()

        checkpoint, trained_model = load_trained_official_model(
            model_config=model_config
        )

        # Validación de productos
        validate_official_model_configuration(
            model_config=model_config
        )

        validate_official_model_metadata(
            model_metadata=model_metadata
        )

        validate_official_model(
            checkpoint=checkpoint,
            model_config=model_config
        )

        # Construcción del producto oficial
        official_model = {
            "checkpoint": checkpoint,
            "trained_model": trained_model,
            "model_config": model_config,
            "model_metadata": model_metadata
        }

        required_products = [
            "checkpoint",
            "trained_model",
            "model_config",
            "model_metadata",
        ]

        missing_products = [
            product
            for product in required_products
            if product not in official_model
        ]

        if missing_products:
            raise RuntimeError(
                "OfficialModel está incompleto: "
                f"{missing_products}"
            )

        if not isinstance(
            official_model["trained_model"],
            nn.Module
        ):
            raise TypeError(
                "trained_model debe ser una instancia de nn.Module."
            )

        return official_model

    except Exception as error:

        raise RuntimeError(
            "No fue posible construir el Modelo Oficial."
        ) from error


OFFICIAL_MODEL = get_official_model()

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
    Recupera y valida la colección oficial BenchmarkData utilizada durante
    la evaluación científica del Modelo Oficial.

    Returns
    -------
    dict
        Datos oficiales de evaluación.
    """

    # --------------------------------------------------------------------------
    # Validación de rutas
    # --------------------------------------------------------------------------

    if not BENCHMARK_DATA_FILE.exists():
        raise FileNotFoundError(
            f"No fue posible localizar '{BENCHMARK_DATA_FILE.name}'."
        )

    # --------------------------------------------------------------------------
    # Recuperación de BenchmarkData
    # --------------------------------------------------------------------------

    try:

        benchmark_data = joblib.load(
            BENCHMARK_DATA_FILE
        )

    except Exception as error:

        raise RuntimeError(
            f"Error al recuperar BenchmarkData: {error}"
        )

    # --------------------------------------------------------------------------
    # Validación de BenchmarkData
    # --------------------------------------------------------------------------

    if not isinstance(
        benchmark_data,
        dict
    ):
        raise TypeError(
            "BenchmarkData debe ser un diccionario."
        )

    required_benchmark = [
        "graphs",
        "x_test",
        "y_test",
        "test_index",
    ]

    missing_benchmark = [
        key
        for key in required_benchmark
        if key not in benchmark_data
    ]

    if missing_benchmark:

        raise ValueError(
            "BenchmarkData está incompleto: "
            f"{missing_benchmark}"
        )

    graphs = benchmark_data["graphs"]

    dataset = pd.DataFrame(
        benchmark_data["x_test"],
        columns=FEATURE_COLUMNS
    )

    if dataset.empty:
        raise ValueError(
            "El DataFrame de variables predictoras está vacío."
        )

    if not isinstance(
        graphs,
        list
    ):
        raise TypeError(
            "'graphs' debe ser una lista."
        )

    if len(graphs) == 0:

        raise ValueError(
            "La colección oficial de GraphData está vacía."
        )

    # --------------------------------------------------------------------------
    # Validación de la colección GraphData
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # Construcción del producto oficial
    # --------------------------------------------------------------------------
    dataset = pd.DataFrame(
        benchmark_data["x_test"],
        columns=FEATURE_COLUMNS
    )

    if dataset.empty:
        raise ValueError(
            "El DataFrame de variables predictoras está vacío."
        )


    evaluation_data = {
        "dataset": dataset,
        "benchmark_data": benchmark_data,
        "graphs": graphs,
    }

    # --------------------------------------------------------------------------
    # Validación del producto
    # --------------------------------------------------------------------------

    required_keys = [
        "benchmark_data",
        "dataset",
        "graphs",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in evaluation_data
    ]

    if missing_keys:

        raise ValueError(
            "El producto evaluation_data está incompleto: "
            f"{missing_keys}"
        )

    # --------------------------------------------------------------------------
    # Retorno
    # --------------------------------------------------------------------------

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
    Genera y valida las predicciones oficiales del Modelo Oficial.

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

    # --------------------------------------------------------------------------
    # Validación
    # --------------------------------------------------------------------------

    if not isinstance(
        official_model,
        dict
    ):
        raise TypeError(
            "official_model debe ser un diccionario."
        )

    if not isinstance(
        evaluation_data,
        dict
    ):
        raise TypeError(
            "evaluation_data debe ser un diccionario."
        )

    required_model_keys = [
        "checkpoint",
        "trained_model",
        "model_metadata",
        "model_config",
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

    required_data_keys = [
        "benchmark_data",
        "graphs",
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

    # --------------------------------------------------------------------------
    # Recuperación
    # --------------------------------------------------------------------------

    trained_model = official_model[
        "trained_model"
    ]

    if not hasattr(
        trained_model,
        "eval"
    ):
        raise TypeError(
            "El modelo oficial no corresponde a un módulo válido de PyTorch."
        )

    benchmark_data = evaluation_data[
        "benchmark_data"
    ]

    graphs = evaluation_data[
        "graphs"
    ]

    y_true = benchmark_data[
        "y_test"
    ]

    test_index = benchmark_data[
        "test_index"
    ]

    # --------------------------------------------------------------------------
    # Inferencia
    # --------------------------------------------------------------------------

    try:

        prediction_output = predict_gnn(
            model=trained_model,
            graphs=graphs
        )

    except Exception as error:

        raise RuntimeError(
            f"Error durante la generación de predicciones: {error}"
        )

    # --------------------------------------------------------------------------
    # Validación
    # --------------------------------------------------------------------------

    if prediction_output is None:
        raise RuntimeError(
            "La función oficial de predicción no devolvió resultados."
        )

    required_prediction_keys = [
        "y_pred",
        "inference_time",
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

    y_pred = np.asarray(
        prediction_output["y_pred"]
    )[test_index]

    y_true = np.asarray(
        y_true
    )

    print("\n" + "=" * 80)
    print("AUDITORÍA DE PREDICCIÓN")
    print("=" * 80)

    print(f"y_true.shape : {tuple(y_true.shape)}")
    print(f"y_pred.shape : {tuple(y_pred.shape)}")

    print(f"len(y_true)  : {len(y_true)}")
    print(f"len(y_pred)  : {len(y_pred)}")
    print(type(y_true))
    print(type(y_pred))

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Las dimensiones de y_true y y_pred no coinciden."
        )

    # --------------------------------------------------------------------------
    # Construcción
    # --------------------------------------------------------------------------

    prediction_result = {

        "y_true": y_true,

        "y_pred": y_pred,

        "inference_time": prediction_output[
            "inference_time"
        ],

    }

    # --------------------------------------------------------------------------
    # Validación
    # --------------------------------------------------------------------------

    required_result_keys = [
        "y_true",
        "y_pred",
        "inference_time",
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

print("-" * 80)
print("AUDITORÍA BLOQUE 8")

print("OFFICIAL_MODEL")
print(OFFICIAL_MODEL.keys())

print("EVALUATION_DATA")
print(EVALUATION_DATA.keys())

print("PREDICTION_RESULT")
print(PREDICTION_RESULT.keys())

print("EVALUATION_CONFIG")
print(EVALUATION_CONFIG.keys())

print("-" * 80)

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
        "model_name",
        "model_code",
        "family",
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
