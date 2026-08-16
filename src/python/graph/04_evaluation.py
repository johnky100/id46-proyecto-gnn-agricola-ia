# 05_evaluation.py

# BLOQUE 1. IMPORTACIÓN DE DEPENDENCIAS Y CONFIGURACIÓN CIENTÍFICA
# Objetivo: Importar las dependencias oficiales del proyecto, cargar la configuración científica
# centralizada y registrar los módulos especializados responsables de la evaluación científica del Modelo Oficial GraphSAGE.
# Arquitectura científica
# Entradas: Librerías estándar, librerías científicas, configuración oficial y módulos especializados de evaluación.
# Producto: Entorno científico inicializado y configuración oficial cargada.
# Pregunta científica: ¿El entorno dispone de todas las dependencias, configuraciones y módulos necesarios
# para evaluar el Modelo Oficial GraphSAGE de forma reproducible?

print("\nBLOQUE 1. IMPORTACIONES") # Mostrar encabezado del bloque

import json
from pathlib import Path

import joblib
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from src.python.config.config_project import (
    PROJECT_SEED, # Cargar semilla científica oficial
    OFF
)
from src.python.config.paths import (
    OFFICIAL_MODEL_TORCH_FILE,
    OFFICIAL_MODEL_CONFIG_FILE,
    OFFICIAL_MODEL_JOBLIB_FILE,
    BENCHMARK_EXPERIMENT_FILE,
    EVALUATION_DIR,
) # Cargar rutas oficiales existentes

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
) # Componentes oficiales de explicabilidad

print("Importaciones científicas : CARGADAS") # Confirmar librerías
print("Rutas oficiales           : CARGADAS") # Confirmar rutas
print("BLOQUE 1                  : VALIDADO") # Confirmar bloque

# BLOQUE 2. CONFIGURACIÓN DE LA EVALUACIÓN CIENTÍFICA
# Objetivo: Construir y validar la configuración oficial que gobernará el protocolo de
# evaluación científica del Modelo Oficial GraphSAGE, garantizando reproducibilidad,
# consistencia e integridad del proceso experimental.
# Arquitectura científica
# Entradas: PROJECT_SEED y componentes definidos para la evaluación científica.
# Producto: EVALUATION_CONFIG validada.
# Pregunta científica: ¿La evaluación científica dispone de una configuración oficial,
# validada y reproducible para ejecutar el protocolo de evaluación del Modelo Oficial?

# BLOQUE 2.1. CONSTRUCCIÓN DE LA CONFIGURACIÓN
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
    } # Construir configuración oficial de evaluación

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
    ] # Definir campos obligatorios

    missing_products = [
        product
        for product in required_products
        if product not in evaluation_config
    ] # Identificar campos faltantes

    if missing_products:
        raise RuntimeError(
            "EvaluationConfiguration está incompleto: "
            f"{missing_products}"
        ) # Detener ejecución ante configuración incompleta

    return evaluation_config


# BLOQUE 2.2. VALIDACIÓN DE LA ESTRUCTURA

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
        ) # Validar existencia de la configuración

    if not isinstance(evaluation_config, dict):
        raise TypeError(
            "evaluation_config debe ser un diccionario."
        ) # Validar tipo de la configuración

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
    ] # Definir estructura obligatoria

    missing_keys = [
        key
        for key in required_keys
        if key not in evaluation_config
    ] # Identificar claves faltantes

    if missing_keys:
        raise RuntimeError(
            "La configuración oficial de la evaluación está incompleta: "
            f"{missing_keys}"
        ) # Detener ejecución ante claves faltantes

    for key in required_keys:
        if evaluation_config[key] is None:
            raise ValueError(
                f"'{key}' es inválido."
            ) # Validar ausencia de valores nulos

    return None

# BLOQUE 2.3. VALIDACIÓN DE TIPOS

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
        ) # Validar existencia de la configuración

    if not isinstance(evaluation_config, dict):
        raise TypeError(
            "evaluation_config debe ser un diccionario."
        ) # Validar tipo de la configuración

    expected_types = {
        "evaluation_name": str,
        "evaluation_version": str,
        "prediction_source": str,
        "shap_batch_size": int,
        "random_state": int,
    } # Definir tipos esperados

    for key, expected_type in expected_types.items():
        if not isinstance(
            evaluation_config[key],
            expected_type
        ):
            raise TypeError(
                f"{key} debe ser de tipo "
                f"{expected_type.__name__}."
            ) # Validar tipo de cada parámetro

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
    ] # Definir parámetros booleanos

    for key in boolean_keys:
        if not isinstance(
            evaluation_config[key],
            bool
        ):
            raise TypeError(
                f"{key} debe ser un valor booleano."
            ) # Validar tipo booleano

    return None

# BLOQUE 2.4. VALIDACIÓN DEL CONTENIDO

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
        ) # Validar existencia de la configuración

    if not isinstance(evaluation_config, dict):
        raise TypeError(
            "evaluation_config debe ser un diccionario."
        ) # Validar tipo de la configuración

    validate_evaluation_configuration_structure(
        evaluation_config
    ) # Validar estructura

    validate_evaluation_configuration_types(
        evaluation_config
    ) # Validar tipos

    if not evaluation_config["evaluation_name"].strip():
        raise ValueError(
            "evaluation_name está vacío."
        ) # Validar nombre de evaluación

    if not evaluation_config["evaluation_version"].strip():
        raise ValueError(
            "evaluation_version está vacío."
        ) # Validar versión de evaluación

    if not evaluation_config["prediction_source"].strip():
        raise ValueError(
            "prediction_source está vacío."
        ) # Validar fuente de predicción

    if evaluation_config["prediction_source"] != "official_model":
        raise ValueError(
            "prediction_source debe ser 'official_model'."
        ) # Validar fuente oficial del modelo

    if evaluation_config["shap_batch_size"] <= 0:
        raise ValueError(
            "shap_batch_size debe ser mayor que cero."
        ) # Validar tamaño del lote SHAP

    if evaluation_config["random_state"] < 0:
        raise ValueError(
            "random_state debe ser mayor o igual a cero."
        ) # Validar semilla científica

    return None


# BLOQUE 2.5. CONSTRUCCIÓN DE LA CONFIGURACIÓN OFICIAL

def get_evaluation_configuration() -> dict:
    """
    Construye y valida la configuración oficial de la evaluación científica.

    Returns
    -------
    dict
        Configuración oficial de la evaluación científica validada.
    """

    try:

        evaluation_config = build_evaluation_configuration() # Construir configuración

        validate_evaluation_configuration_structure(
            evaluation_config=evaluation_config
        ) # Validar estructura

        validate_evaluation_configuration_types(
            evaluation_config=evaluation_config
        ) # Validar tipos

        validate_evaluation_configuration_values(
            evaluation_config=evaluation_config
        ) # Validar contenido

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
        ] # Definir productos obligatorios

        missing_products = [
            product
            for product in required_products
            if product not in evaluation_config
        ] # Verificar productos obligatorios

        if missing_products:
            raise RuntimeError(
                "EvaluationConfiguration está incompleto: "
                f"{missing_products}"
            ) # Detener ejecución ante configuración incompleta

        return evaluation_config

    except Exception as error:

        raise RuntimeError(
            "No fue posible construir la configuración oficial de la evaluación."
        ) from error

# Construcción del producto oficial
EVALUATION_CONFIG = get_evaluation_configuration() # Construir y validar configuración oficial
print("-" * 80)
print("Configuración Oficial de la Evaluación Científica")
print("-" * 80)

print(f"Evaluación                : {EVALUATION_CONFIG['evaluation_name']}") # Mostrar nombre
print(f"Versión                   : {EVALUATION_CONFIG['evaluation_version']}") # Mostrar versión
print(f"Fuente de predicción      : {EVALUATION_CONFIG['prediction_source']}") # Mostrar fuente
print(f"Explicabilidad Global     : {EVALUATION_CONFIG['calculate_global_explainability']}") # Mostrar configuración global
print(f"Explicabilidad Local      : {EVALUATION_CONFIG['calculate_local_explainability']}") # Mostrar configuración local
print(f"Importancia de Variables  : {EVALUATION_CONFIG['calculate_feature_importance']}") # Mostrar importancia
print(f"Semilla Científica        : {EVALUATION_CONFIG['random_state']}") # Mostrar semilla

# BLOQUE 3. RECUPERACIÓN DE ARTEFACTOS CIENTÍFICOS PERSISTIDOS
# Objetivo: Recuperar exclusivamente desde disco los artefactos oficiales necesarios
# para iniciar la evaluación independiente del Modelo Oficial GraphSAGE.
# Arquitectura científica
# Entradas: Modelo PyTorch, Modelo JOBLIB, BenchmarkData, GraphData, test_index,
# scaler, configuración del modelo y metadatos del entrenamiento.
# Producto: Artefactos científicos persistidos recuperados y disponibles para evaluación independiente.
# Pregunta científica: ¿Los artefactos científicos oficiales fueron recuperados correctamente
# desde disco para iniciar una evaluación independiente y reproducible?

# BLOQUE 3.1. VALIDACIÓN DE ARCHIVOS OFICIALES
def validate_evaluation_artifact_files() -> None:
    """
    Valida la existencia de los archivos oficiales requeridos para la evaluación.
    Returns
    -------
    None
    """
    required_files = {
        "OFFICIAL_MODEL_TORCH_FILE": OFFICIAL_MODEL_TORCH_FILE,
        "OFFICIAL_MODEL_CONFIG_FILE": OFFICIAL_MODEL_CONFIG_FILE,
        "OFFICIAL_MODEL_JOBLIB_FILE": OFFICIAL_MODEL_JOBLIB_FILE,
        "BENCHMARK_EXPERIMENT_FILE": BENCHMARK_EXPERIMENT_FILE,
    } # Definir archivos oficiales requeridos

    for product, file_path in required_files.items():
        if file_path is None:
            raise ValueError(
                f"{product} no está definido."
            ) # Validar definición de la ruta

        if not isinstance(
            file_path,
            Path
        ):
            raise TypeError(
                f"{product} debe ser una instancia de pathlib.Path."
            ) # Validar tipo de ruta

        if not file_path.exists():
            raise FileNotFoundError(
                f"No fue posible localizar '{file_path.name}'."
            ) # Validar existencia física del artefacto

        if not file_path.is_file():
            raise RuntimeError(
                f"'{file_path}' no corresponde a un archivo."
            ) # Validar naturaleza del artefacto

        if file_path.stat().st_size <= 0:
            raise RuntimeError(
                f"'{file_path.name}' existe pero está vacío."
            ) # Validar contenido físico
    print("Archivos oficiales requeridos : VALIDADOS") # Confirmar archivos
    return None

# BLOQUE 3.2. RECUPERACIÓN DEL EXPERIMENTO CIENTÍFICO DEL BENCHMARK
def load_benchmark_experiment() -> dict:
    """
    Recupera el experimento científico persistido del Benchmark.
    Returns
    -------
    dict
        Experimento científico persistido.
    """

    try:
        benchmark_experiment = joblib.load(
            BENCHMARK_EXPERIMENT_FILE
        ) # Recuperar experimento científico desde disco

    except Exception as error:
        raise RuntimeError(
            "No fue posible recuperar benchmark_experiment.joblib."
        ) from error

    if not isinstance(
        benchmark_experiment,
        dict
    ):
        raise TypeError(
            "El experimento científico del Benchmark debe ser un diccionario."
        ) # Validar estructura

    required_experiment_fields = [
        "benchmark_data",
        "official_model",
        "status",
    ] # Definir campos mínimos del experimento

    missing_experiment_fields = [
        field
        for field in required_experiment_fields
        if field not in benchmark_experiment
    ] # Identificar campos faltantes

    if missing_experiment_fields:
        raise RuntimeError(
            "El experimento científico del Benchmark está incompleto: "
            f"{missing_experiment_fields}"
        ) # Validar contrato mínimo

    if benchmark_experiment["status"] != "VALIDATED":
        raise RuntimeError(
            "El experimento científico del Benchmark no presenta estado VALIDATED."
        ) # Validar estado científico
    print("Experimento Benchmark       : RECUPERADO Y VALIDADO") # Confirmar recuperación
    return benchmark_experiment

# BLOQUE 3.3. RECUPERACIÓN DE BENCHMARK_DATA
def load_benchmark_data(
    benchmark_experiment: dict
) -> dict:
    """
    Recupera los datos científicos utilizados durante el Benchmark.
    Parameters
    ----------
    benchmark_experiment : dict
        Experimento científico persistido.
    Returns
    -------
    dict
        BenchmarkData persistido.
    """

    if "benchmark_data" not in benchmark_experiment:
        raise RuntimeError(
            "El experimento científico no contiene benchmark_data."
        ) # Validar disponibilidad

    benchmark_data = benchmark_experiment[
        "benchmark_data"
    ] # Recuperar datos científicos

    if not isinstance(
        benchmark_data,
        dict
    ):
        raise TypeError(
            "benchmark_data debe ser un diccionario."
        ) # Validar estructura

    required_benchmark_data_fields = [
        "graphs",
        "train_index",
        "validation_index",
        "test_index",
        "scaler",
    ] # Definir entradas requeridas

    missing_benchmark_data_fields = [
        field
        for field in required_benchmark_data_fields
        if field not in benchmark_data
    ] # Identificar entradas faltantes

    if missing_benchmark_data_fields:
        raise RuntimeError(
            "benchmark_data no contiene todas las entradas requeridas: "
            f"{missing_benchmark_data_fields}"
        ) # Validar cobertura
    print("BenchmarkData                : RECUPERADO Y VALIDADO") # Confirmar datos
    return benchmark_data

# BLOQUE 3.4. RECUPERACIÓN DEL MODELO PYTORCH
def load_pytorch_artifact() -> dict:
    """
    Recupera el checkpoint persistido del Modelo Oficial GraphSAGE.

    Returns
    -------
    dict
        Checkpoint oficial persistido.
    """

    try:

        checkpoint = torch.load(
            OFFICIAL_MODEL_TORCH_FILE,
            weights_only=False
        ) # Recuperar checkpoint PyTorch desde disco

    except Exception as error:

        raise RuntimeError(
            "No fue posible recuperar el checkpoint PyTorch del Modelo Oficial."
        ) from error

    if not isinstance(
        checkpoint,
        dict
    ):
        raise TypeError(
            "El checkpoint PyTorch debe ser un diccionario."
        ) # Validar estructura del checkpoint

    if "model_state_dict" in checkpoint:

        model_state_dict = checkpoint[
            "model_state_dict"
        ] # Recuperar pesos del checkpoint

    else:

        model_state_dict = checkpoint # Utilizar checkpoint como state_dict directo

    if not isinstance(
        model_state_dict,
        dict
    ):
        raise TypeError(
            "Los pesos recuperados del Modelo Oficial deben ser un diccionario."
        ) # Validar estructura de pesos

    if len(
        model_state_dict
    ) == 0:
        raise RuntimeError(
            "El checkpoint del Modelo Oficial no contiene pesos."
        ) # Validar contenido

    print("Checkpoint PyTorch           : RECUPERADO Y VALIDADO") # Confirmar recuperación
    return checkpoint

# BLOQUE 3.5. RECUPERACIÓN DE LA CONFIGURACIÓN DEL MODELO
def load_evaluation_model_configuration() -> dict:
    """
    Recupera y valida la configuración persistida del Modelo Oficial.
    Returns
    -------
    dict
        Configuración oficial recuperada.
    """
    try:
        with OFFICIAL_MODEL_CONFIG_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:
            model_config = json.load(file) # Recuperar configuración desde disco
    except Exception as error:
        raise RuntimeError(
            "No fue posible recuperar la configuración del Modelo Oficial."
        ) from error

    if not isinstance(
        model_config,
        dict
    ):
        raise TypeError(
            "La configuración del modelo debe ser un diccionario."
        ) # Validar estructura

    print(f"Campos de configuración recuperados : {list(model_config.keys())}") # Mostrar estructura real

    if len(model_config) == 0:
        raise RuntimeError(
            "La configuración oficial del Modelo está vacía."
        ) # Validar contenido
    print("Configuración del Modelo : RECUPERADA") # Confirmar recuperación
    return model_config

# BLOQUE 3.6. RECUPERACIÓN DEL PAQUETE JOBLIB DEL MODELO OFICIAL
def load_joblib_model_artifact() -> dict:
    """
    Recupera el paquete serializado del Modelo Oficial.

    Returns
    -------
    dict
        Paquete JOBLIB del Modelo Oficial.
    """

    try:

        model_package = joblib.load(
            OFFICIAL_MODEL_JOBLIB_FILE
        ) # Recuperar paquete JOBLIB desde disco

    except Exception as error:

        raise RuntimeError(
            "No fue posible recuperar official_model.joblib."
        ) from error

    if not isinstance(
        model_package,
        dict
    ):
        raise TypeError(
            "El contenido de official_model.joblib debe ser un diccionario."
        ) # Validar estructura

    required_joblib_fields = [
        "model_code",
        "model_name",
        "family",
        "model_state_dict",
        "model_config",
        "input_channels",
        "output_channels",
        "project_seed",
        "status",
    ] # Definir contrato JOBLIB

    missing_joblib_fields = [
        field
        for field in required_joblib_fields
        if field not in model_package
    ] # Identificar campos faltantes

    if missing_joblib_fields:
        raise RuntimeError(
            "official_model.joblib está incompleto: "
            f"{missing_joblib_fields}"
        ) # Validar contrato JOBLIB

    if model_package["status"] != "VALIDATED":
        raise RuntimeError(
            "official_model.joblib no presenta estado VALIDATED."
        ) # Validar estado

    if not isinstance(
        model_package["model_state_dict"],
        dict
    ):
        raise TypeError(
            "El model_state_dict de official_model.joblib debe ser un diccionario."
        ) # Validar pesos JOBLIB

    if len(
        model_package["model_state_dict"]
    ) == 0:
        raise RuntimeError(
            "official_model.joblib contiene un model_state_dict vacío."
        ) # Validar pesos

    print("Modelo JOBLIB                : RECUPERADO Y VALIDADO") # Confirmar recuperación
    return model_package

# BLOQUE 3.7. EJECUCIÓN DE LA RECUPERACIÓN OFICIAL
validate_evaluation_artifact_files() # Validar artefactos físicos
benchmark_experiment_loaded = load_benchmark_experiment() # Recuperar experimento Benchmark
benchmark_data_loaded = load_benchmark_data(
    benchmark_experiment_loaded
) # Recuperar BenchmarkData

official_checkpoint_loaded = load_pytorch_artifact() # Recuperar checkpoint PyTorch
evaluation_model_config = load_evaluation_model_configuration() # Recuperar configuración oficial
official_model_joblib_loaded = load_joblib_model_artifact() # Recuperar paquete JOBLIB

print("BLOQUE 3                    : VALIDADO") # Confirmar recuperación completa

# BLOQUE 4. VALIDACIÓN DE LOS ARTEFACTOS DE EVALUACIÓN
# Objetivo: Validar la existencia, estructura, identidad, configuración y compatibilidad de los artefactos
# científicos recuperados para la evaluación independiente del Modelo Oficial GraphSAGE.
# Arquitectura científica
# Entradas: Modelo Oficial GraphSAGE, configuración oficial, BenchmarkData persistido, GraphData,
# particiones train_index, validation_index, test_index y scaler.
# Producto: Entradas de evaluación validadas.
# Pregunta científica: ¿Los artefactos persistidos presentan la identidad, estructura, dimensionalidad,
# particiones y componentes necesarios para evaluar el Modelo Oficial GraphSAGE de forma independiente y reproducible?

joblib_model_config = official_model_joblib_loaded[
    "model_config"
] # Recuperar configuración persistida en JOBLIB

if not isinstance(
    joblib_model_config,
    dict
):
    raise TypeError(
        "joblib_model_config debe ser un diccionario."
    ) # Validar estructura

print(f"Campos JSON                    : {list(evaluation_model_config.keys())}") # Mostrar campos JSON
print(f"Campos JOBLIB                  : {list(joblib_model_config.keys())}") # Mostrar campos JOBLIB



print("\nDIAGNÓSTICO DEL ESTADO DE LA CONFIGURACIÓN") # Mostrar diagnóstico
print(f"Status configuración JSON      : {evaluation_model_config.get('status')}") # Mostrar estado
print(f"Status configuración JOBLIB    : {official_model_joblib_loaded.get('status')}") # Mostrar estado JOBLIB
print(f"Status experimento Benchmark   : {benchmark_experiment_loaded.get('status')}") # Mostrar estado Benchmark


print("\nBLOQUE 4. VALIDACIÓN DE LOS ARTEFACTOS DE EVALUACIÓN") # Mostrar encabezado del bloque

# BLOQUE 4.1. VALIDACIÓN DE EXISTENCIA Y TIPO DE LOS ARTEFACTOS RECUPERADOS
required_evaluation_artifacts = {
    "benchmark_experiment_loaded": benchmark_experiment_loaded,
    "benchmark_data_loaded": benchmark_data_loaded,
    "official_checkpoint_loaded": official_checkpoint_loaded,
    "evaluation_model_config": evaluation_model_config,
    "official_model_joblib_loaded": official_model_joblib_loaded,
} # Definir artefactos requeridos

for artifact_name, artifact in required_evaluation_artifacts.items():
    if artifact is None:
        raise RuntimeError(
            f"El artefacto '{artifact_name}' no está disponible."
        ) # Validar disponibilidad
    print(f"{artifact_name:<32}: {type(artifact).__name__}") # Mostrar tipo recuperado

print("Artefactos recuperados          : VALIDADOS") # Confirmar recuperación

# BLOQUE 4.2. VALIDACIÓN DE ESTRUCTURA DEL EXPERIMENTO
if not isinstance(
    benchmark_experiment_loaded,
    dict
):
    raise TypeError(
        "benchmark_experiment_loaded debe ser un diccionario."
    ) # Validar estructura

if not isinstance(
    benchmark_data_loaded,
    dict
):
    raise TypeError(
        "benchmark_data_loaded debe ser un diccionario."
    ) # Validar estructura

if "benchmark_data" not in benchmark_experiment_loaded:
    raise RuntimeError(
        "El experimento científico no contiene benchmark_data."
    ) # Validar estructura

if "official_model" not in benchmark_experiment_loaded:
    raise RuntimeError(
        "El experimento científico no contiene official_model."
    ) # Validar estructura

if "status" not in benchmark_experiment_loaded:
    raise RuntimeError(
        "El experimento científico no contiene status."
    ) # Validar estado

if benchmark_experiment_loaded["status"] != "VALIDATED":
    raise RuntimeError(
        "El experimento científico no presenta estado VALIDATED."
    ) # Validar estado científico

print("Estructura del experimento      : VALIDADA") # Confirmar estructura

# BLOQUE 4.3. VALIDACIÓN DE IDENTIDAD DEL MODELO OFICIAL
required_model_identity_fields = [
    "model_code",
    "model_name",
    "family",
] # Definir identidad mínima del Modelo Oficial

for field in required_model_identity_fields:
    if field not in official_model_joblib_loaded:
        raise RuntimeError(
            f"official_model.joblib no contiene '{field}'."
        ) # Validar identidad

    if field not in benchmark_experiment_loaded["official_model"]:
        raise RuntimeError(
            f"El experimento Benchmark no contiene '{field}' "
            "en official_model."
        ) # Validar identidad Benchmark

joblib_model_code = str(
    official_model_joblib_loaded["model_code"]
) # Recuperar código oficial

joblib_model_name = str(
    official_model_joblib_loaded["model_name"]
) # Recuperar nombre oficial

joblib_model_family = str(
    official_model_joblib_loaded["family"]
) # Recuperar familia oficial

benchmark_model_code = str(
    benchmark_experiment_loaded["official_model"]["model_code"]
) # Recuperar código del Benchmark

benchmark_model_name = str(
    benchmark_experiment_loaded["official_model"]["model_name"]
) # Recuperar nombre del Benchmark

benchmark_model_family = str(
    benchmark_experiment_loaded["official_model"]["family"]
) # Recuperar familia del Benchmark

if joblib_model_code != benchmark_model_code:
    raise RuntimeError(
        "El código del Modelo Oficial no coincide entre JOBLIB y Benchmark."
    ) # Validar código

if joblib_model_name.strip().lower() != benchmark_model_name.strip().lower():
    raise RuntimeError(
        "El nombre del Modelo Oficial no coincide entre JOBLIB y Benchmark."
    ) # Validar nombre

if joblib_model_family.strip().lower() != benchmark_model_family.strip().lower():
    raise RuntimeError(
        "La familia del Modelo Oficial no coincide entre JOBLIB y Benchmark."
    ) # Validar familia

print(f"Código del modelo               : {joblib_model_code}") # Mostrar identidad
print(f"Nombre del modelo               : {joblib_model_name}") # Mostrar identidad
print(f"Familia del modelo              : {joblib_model_family}") # Mostrar familia
print("Identidad del Modelo Oficial    : VALIDADA") # Confirmar identidad

# BLOQUE 4.4. VALIDACIÓN DE LA CONFIGURACIÓN DEL MODELO OFICIAL

if not isinstance(
    evaluation_model_config,
    dict
):
    raise TypeError(
        "evaluation_model_config debe ser un diccionario."
    ) # Validar estructura de configuración

if not isinstance(
    official_model_joblib_loaded.get("model_config"),
    dict
):
    raise TypeError(
        "La configuración almacenada en official_model.joblib debe ser un diccionario."
    ) # Validar configuración JOBLIB

joblib_model_config = official_model_joblib_loaded[
    "model_config"
] # Recuperar configuración arquitectónica y de entrenamiento

required_selection_fields = [
    "model_code",
    "model_name",
    "family",
    "model_config",
    "selection_source",
    "selection_scope",
    "ranking_metric",
    "ranking_direction",
    "status",
] # Definir campos científicos mínimos de selección

missing_selection_fields = [
    field
    for field in required_selection_fields
    if field not in evaluation_model_config
] # Identificar campos de selección faltantes

if missing_selection_fields:
    raise RuntimeError(
        "La configuración científica del Modelo Oficial está incompleta: "
        f"{missing_selection_fields}"
    ) # Validar contrato científico

required_training_fields = [
    "model_code",
    "model_name",
    "family",
    "hidden_channels",
    "dropout",
    "learning_rate",
    "weight_decay",
    "epochs",
] # Definir campos arquitectónicos y de entrenamiento

missing_training_fields = [
    field
    for field in required_training_fields
    if field not in joblib_model_config
] # Identificar parámetros faltantes

if missing_training_fields:
    raise RuntimeError(
        "La configuración JOBLIB del Modelo Oficial está incompleta: "
        f"{missing_training_fields}"
    ) # Validar configuración de entrenamiento

if str(
    evaluation_model_config["model_code"]
) != joblib_model_code:
    raise RuntimeError(
        "El código de la configuración científica no coincide "
        "con el Modelo Oficial."
    ) # Validar código

if str(
    evaluation_model_config["model_name"]
).strip().lower() != joblib_model_name.strip().lower():
    raise RuntimeError(
        "El nombre de la configuración científica no coincide "
        "con el Modelo Oficial."
    ) # Validar nombre

if str(
    evaluation_model_config["family"]
).strip().lower() != joblib_model_family.strip().lower():
    raise RuntimeError(
        "La familia de la configuración científica no coincide "
        "con el Modelo Oficial."
    ) # Validar familia

if str(
    joblib_model_config["model_code"]
) != joblib_model_code:
    raise RuntimeError(
        "El código de la configuración JOBLIB no coincide "
        "con el Modelo Oficial."
    ) # Validar código JOBLIB

if str(
    joblib_model_config["model_name"]
).strip().lower() != joblib_model_name.strip().lower():
    raise RuntimeError(
        "El nombre de la configuración JOBLIB no coincide "
        "con el Modelo Oficial."
    ) # Validar nombre JOBLIB

if str(
    joblib_model_config["family"]
).strip().lower() != joblib_model_family.strip().lower():
    raise RuntimeError(
        "La familia de la configuración JOBLIB no coincide "
        "con el Modelo Oficial."
    ) # Validar familia JOBLIB

if not isinstance(
    evaluation_model_config["model_config"],
    dict
):
    raise TypeError(
        "El campo model_config del registro científico debe ser un diccionario."
    ) # Validar configuración anidada

selection_model_config = evaluation_model_config[
    "model_config"
] # Recuperar configuración seleccionada por Benchmark

if not isinstance(
    selection_model_config,
    dict
):
    raise TypeError(
        "La configuración seleccionada por Benchmark debe ser un diccionario."
    ) # Validar configuración Benchmark

training_hyperparameters = {
    "hidden_channels": joblib_model_config["hidden_channels"],
    "dropout": joblib_model_config["dropout"],
    "learning_rate": joblib_model_config["learning_rate"],
    "weight_decay": joblib_model_config["weight_decay"],
    "epochs": joblib_model_config["epochs"],
} # Recuperar hiperparámetros oficiales del entrenamiento

for parameter_name, parameter_value in training_hyperparameters.items():

    if parameter_name in selection_model_config:

        if selection_model_config[parameter_name] != parameter_value:
            raise RuntimeError(
                f"El parámetro '{parameter_name}' no coincide entre "
                "la selección del Benchmark y JOBLIB."
            ) # Validar consistencia del hiperparámetro

if evaluation_model_config["status"] != "OFFICIAL":
    raise RuntimeError(
        "La configuración científica del Modelo Oficial no presenta estado OFFICIAL."
    ) # Validar estado oficial

if official_model_joblib_loaded["status"] != "VALIDATED":
    raise RuntimeError(
        "La configuración JOBLIB del Modelo Oficial no presenta estado VALIDATED."
    ) # Validar estado JOBLIB

print(f"Configuración Benchmark       : {evaluation_model_config['status']}") # Mostrar estado oficial
print(f"Configuración JOBLIB           : {official_model_joblib_loaded['status']}") # Mostrar estado JOBLIB
print(f"Hiperparámetros oficiales      : {training_hyperparameters}") # Mostrar configuración
print("Configuración del modelo       : VALIDADA") # Confirmar validación

# BLOQUE 4.5. VALIDACIÓN DE DIMENSIONES OFICIALES
if "input_channels" not in official_model_joblib_loaded:
    raise RuntimeError(
        "official_model.joblib no contiene input_channels."
    ) # Validar dimensión de entrada

if "output_channels" not in official_model_joblib_loaded:
    raise RuntimeError(
        "official_model.joblib no contiene output_channels."
    ) # Validar dimensión de salida

evaluation_input_channels = int(
    official_model_joblib_loaded["input_channels"]
) # Recuperar dimensión de entrada

evaluation_output_channels = int(
    official_model_joblib_loaded["output_channels"]
) # Recuperar dimensión de salida

if evaluation_input_channels <= 0:
    raise ValueError(
        "input_channels debe ser mayor que cero."
    ) # Validar dimensión

if evaluation_output_channels != 1:
    raise RuntimeError(
        "El Modelo Oficial debe presentar una salida de dimensión 1."
    ) # Validar salida

print(f"Variables de entrada            : {evaluation_input_channels}") # Mostrar dimensión
print(f"Variables de salida             : {evaluation_output_channels}") # Mostrar dimensión
print("Dimensiones oficiales           : VALIDADAS") # Confirmar dimensiones

# BLOQUE 4.6. VALIDACIÓN DE LOS GRAPHDATA
if not isinstance(
    benchmark_data_loaded["graphs"],
    (list, tuple)
):
    raise TypeError(
        "graphs debe ser una lista o tupla."
    ) # Validar colección GraphData

evaluation_graphs = benchmark_data_loaded[
    "graphs"
] # Recuperar GraphData oficiales

if len(evaluation_graphs) == 0:
    raise RuntimeError(
        "La colección de GraphData está vacía."
    ) # Validar existencia

for graph_position, graph in enumerate(
    evaluation_graphs,
    start=1
):
    if not isinstance(
        graph,
        Data
    ):
        raise TypeError(
            f"GraphData {graph_position} no es una instancia de torch_geometric.data.Data."
        ) # Validar tipo GraphData

    if not hasattr(
        graph,
        "x"
    ):
        raise RuntimeError(
            f"GraphData {graph_position} no contiene x."
        ) # Validar características

    if not hasattr(
        graph,
        "edge_index"
    ):
        raise RuntimeError(
            f"GraphData {graph_position} no contiene edge_index."
        ) # Validar estructura

    if not hasattr(
        graph,
        "y"
    ):
        raise RuntimeError(
            f"GraphData {graph_position} no contiene y."
        ) # Validar objetivo

    if graph.x.ndim != 2:
        raise RuntimeError(
            f"GraphData {graph_position} presenta x con dimensión "
            f"{graph.x.ndim}; se esperaban 2."
        ) # Validar características

    if graph.edge_index.ndim != 2:
        raise RuntimeError(
            f"GraphData {graph_position} presenta edge_index con "
            f"{graph.edge_index.ndim} dimensiones."
        ) # Validar estructura

    if graph.edge_index.shape[0] != 2:
        raise RuntimeError(
            f"GraphData {graph_position} presenta edge_index con "
            f"forma {graph.edge_index.shape}; se esperaba [2, num_edges]."
        ) # Validar formato topológico

    if int(graph.x.shape[1]) != evaluation_input_channels:
        raise RuntimeError(
            f"GraphData {graph_position} presenta "
            f"{graph.x.shape[1]} variables, pero el Modelo Oficial "
            f"requiere {evaluation_input_channels}."
        ) # Validar compatibilidad

    if int(graph.y.reshape(-1).shape[0]) != int(graph.x.shape[0]):
        raise RuntimeError(
            f"GraphData {graph_position} no conserva correspondencia "
            "entre nodos y variable objetivo."
        ) # Validar correspondencia

print(f"GraphData disponibles           : {len(evaluation_graphs)}") # Mostrar cantidad
print("Estructura de los GraphData     : VALIDADA") # Confirmar estructura

# BLOQUE 4.7. VALIDACIÓN DE LAS PARTICIONES
required_partition_names = [
    "train_index",
    "validation_index",
    "test_index",
] # Definir particiones requeridas

for partition_name in required_partition_names:
    if partition_name not in benchmark_data_loaded:
        raise RuntimeError(
            f"No existe la partición '{partition_name}'."
        ) # Validar existencia

    partition = benchmark_data_loaded[
        partition_name
    ] # Recuperar partición

    if not isinstance(
        partition,
        np.ndarray
    ):
        raise TypeError(
            f"{partition_name} debe ser un numpy.ndarray."
        ) # Validar tipo

    if partition.ndim != 1:
        raise RuntimeError(
            f"{partition_name} debe ser un vector unidimensional."
        ) # Validar dimensionalidad

    if len(partition) == 0:
        raise RuntimeError(
            f"{partition_name} está vacío."
        ) # Validar contenido

    if not np.issubdtype(
        partition.dtype,
        np.integer
    ):
        raise TypeError(
            f"{partition_name} debe contener índices enteros."
        ) # Validar índices

    if np.any(
        partition < 0
    ):
        raise ValueError(
            f"{partition_name} contiene índices negativos."
        ) # Validar límites inferiores

    if np.any(
        partition >= len(evaluation_graphs)
    ):
        raise ValueError(
            f"{partition_name} contiene índices fuera del rango de GraphData."
        ) # Validar límites superiores

train_index = benchmark_data_loaded[
    "train_index"
] # Recuperar entrenamiento

validation_index = benchmark_data_loaded[
    "validation_index"
] # Recuperar validación

test_index = benchmark_data_loaded[
    "test_index"
] # Recuperar prueba

if len(
    np.unique(train_index)
) != len(train_index):
    raise RuntimeError(
        "train_index contiene índices duplicados."
    ) # Validar unicidad

if len(
    np.unique(validation_index)
) != len(validation_index):
    raise RuntimeError(
        "validation_index contiene índices duplicados."
    ) # Validar unicidad

if len(
    np.unique(test_index)
) != len(test_index):
    raise RuntimeError(
        "test_index contiene índices duplicados."
    ) # Validar unicidad

# BLOQUE 4.8. VALIDACIÓN DE INDEPENDENCIA DE LAS PARTICIONES
train_set = set(
    train_index.tolist()
) # Convertir entrenamiento a conjunto

validation_set = set(
    validation_index.tolist()
) # Convertir validación a conjunto

test_set = set(
    test_index.tolist()
) # Convertir prueba a conjunto

if train_set.intersection(
    validation_set
):
    raise RuntimeError(
        "Existe solapamiento entre entrenamiento y validación."
    ) # Validar independencia

if train_set.intersection(
    test_set
):
    raise RuntimeError(
        "Existe solapamiento entre entrenamiento y prueba."
    ) # Validar independencia

if validation_set.intersection(
    test_set
):
    raise RuntimeError(
        "Existe solapamiento entre validación y prueba."
    ) # Validar independencia

partition_total = (
    len(train_set)
    + len(validation_set)
    + len(test_set)
) # Verificar cobertura de particiones

if partition_total != len(
    evaluation_graphs
):
    raise RuntimeError(
        "Las particiones no cubren exactamente todos los GraphData."
    ) # Validar cobertura

print(f"GraphData de entrenamiento      : {len(train_index)}") # Mostrar partición
print(f"GraphData de validación         : {len(validation_index)}") # Mostrar partición
print(f"GraphData de prueba             : {len(test_index)}") # Mostrar partición
print("Particiones                     : VALIDADAS") # Confirmar particiones

# BLOQUE 4.9. VALIDACIÓN ESPECÍFICA DEL TEST_INDEX
if len(
    test_index
) == 0:
    raise RuntimeError(
        "test_index está vacío."
    ) # Validar conjunto de prueba

print("test_index                      : DISPONIBLE Y VÁLIDO") # Confirmar prueba

# BLOQUE 4.10. VALIDACIÓN DEL SCALER
if "scaler" not in benchmark_data_loaded:
    raise RuntimeError(
        "benchmark_data no contiene scaler."
    ) # Validar existencia

evaluation_scaler = benchmark_data_loaded[
    "scaler"
] # Recuperar scaler oficial

if evaluation_scaler is None:
    raise RuntimeError(
        "El scaler recuperado es None."
    ) # Validar disponibilidad

if not hasattr(
    evaluation_scaler,
    "transform"
):
    raise TypeError(
        "El scaler recuperado no dispone del método transform."
    ) # Validar capacidad de transformación

if not hasattr(
    evaluation_scaler,
    "mean_"
):
    raise RuntimeError(
        "El scaler recuperado no contiene mean_."
    ) # Validar parámetros

if not hasattr(
    evaluation_scaler,
    "scale_"
):
    raise RuntimeError(
        "El scaler recuperado no contiene scale_."
    ) # Validar parámetros

if len(
    evaluation_scaler.mean_
) != evaluation_input_channels:
    raise RuntimeError(
        "La dimensión de mean_ del scaler no coincide con "
        "las variables de entrada del Modelo Oficial."
    ) # Validar dimensionalidad

if len(
    evaluation_scaler.scale_
) != evaluation_input_channels:
    raise RuntimeError(
        "La dimensión de scale_ del scaler no coincide con "
        "las variables de entrada del Modelo Oficial."
    ) # Validar dimensionalidad

print(f"Scaler               : {type(evaluation_scaler).__name__}") # Mostrar tipo
print("Scaler                : VALIDADO") # Confirmar scaler

# BLOQUE 4.11. CONSTRUCCIÓN DEL PRODUCTO DE VALIDACIÓN
evaluation_inputs_validated = {
    "graphs": evaluation_graphs,
    "train_index": train_index,
    "validation_index": validation_index,
    "test_index": test_index,
    "scaler": evaluation_scaler,
    "model_code": joblib_model_code,
    "model_name": joblib_model_name,
    "family": joblib_model_family,
    "model_config": evaluation_model_config,
    "input_channels": evaluation_input_channels,
    "output_channels": evaluation_output_channels,
} # Construir producto de entradas validadas

required_validated_fields = [
    "graphs",
    "train_index",
    "validation_index",
    "test_index",
    "scaler",
    "model_code",
    "model_name",
    "family",
    "model_config",
    "input_channels",
    "output_channels",
] # Definir contrato del producto

missing_validated_fields = [
    field
    for field in required_validated_fields
    if field not in evaluation_inputs_validated
] # Identificar campos faltantes

if missing_validated_fields:
    raise RuntimeError(
        "El producto de entradas validadas está incompleto: "
        f"{missing_validated_fields}"
    ) # Validar producto

print("Entradas de evaluación          : VALIDADAS") # Confirmar producto
print("BLOQUE 4                        : VALIDADO") # Confirmar bloque

# BLOQUE 5. DIAGNÓSTICO PREVIO PARA LA RECONSTRUCCIÓN DEL MODELO OFICIAL
# Objetivo: Verificar que la arquitectura, configuración y pesos persistidos necesarios para reconstruir
# de forma independiente el Modelo Oficial GraphSAGE se encuentran disponibles y son compatibles.
# Arquitectura científica
# Entradas: Configuración oficial del Modelo GraphSAGE, pesos persistidos y dimensión de entrada validada.
# Producto: Información diagnóstica suficiente para reconstruir evaluation_graphsage.
# Pregunta científica: ¿Se dispone de toda la información estructural y paramétrica necesaria para reconstruir
# de forma independiente el Modelo Oficial GraphSAGE sin utilizar la instancia original del entrenamiento?

print("\nBLOQUE 5. DIAGNÓSTICO PREVIO PARA LA RECONSTRUCCIÓN DEL MODELO OFICIAL") # Mostrar encabezado
print(f"Modelo Oficial    : {joblib_model_name}") # Mostrar modelo
print(f"Código Oficial    : {joblib_model_code}") # Mostrar código
print(f"Familia           : {joblib_model_family}") # Mostrar familia
print(f"Input channels    : {evaluation_input_channels}") # Mostrar entrada
print(f"Output channels   : {evaluation_output_channels}") # Mostrar salida
print("Clase del modelo   : PENDIENTE DE RECUPERACIÓN") # Indicar dependencia pendiente

print("\nBLOQUE 5. RECONSTRUCCIÓN DEL MODELO OFICIAL") # Mostrar encabezado

# BLOQUE 5.1. DEFINICIÓN DE LA ARQUITECTURA INDEPENDIENTE
class EvaluationGraphSAGE(nn.Module):
    """
    Arquitectura independiente utilizada exclusivamente para la evaluación
    científica del Modelo Oficial GraphSAGE.
    """
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        dropout: float,
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            input_channels,
            hidden_channels,
        ) # Primera capa GraphSAGE

        self.conv2 = SAGEConv(
            hidden_channels,
            hidden_channels,
        ) # Segunda capa GraphSAGE

        self.output_layer = nn.Linear(
            hidden_channels,
            output_channels,
        ) # Capa de salida oficial

        self.dropout = float(
            dropout
        ) # Configurar dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        x = self.conv1(
            x,
            edge_index,
        ) # Primera propagación GraphSAGE

        x = F.relu(
            x
        ) # Activación ReLU

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        ) # Aplicar dropout

        x = self.conv2(
            x,
            edge_index,
        ) # Segunda propagación GraphSAGE

        x = F.relu(
            x
        ) # Activación ReLU

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        ) # Aplicar dropout

        x = self.output_layer(
            x
        ) # Generar predicción
        return x

print("\n5.2 VALIDACIÓN DE LA CONFIGURACIÓN ARQUITECTÓNICA") # Mostrar encabezado
if evaluation_input_channels != 36:
    raise RuntimeError(
        f"El modelo requiere 36 variables de entrada y se recibieron "
        f"{evaluation_input_channels}."
    ) # Validar entrada oficial

if evaluation_output_channels != 1:
    raise RuntimeError(
        f"El modelo requiere una salida de dimensión 1 y se recibió "
        f"{evaluation_output_channels}."
    ) # Validar salida oficial

evaluation_hidden_channels = int(
    joblib_model_config["hidden_channels"]
) # Recuperar dimensión oculta oficial

evaluation_dropout = float(
    joblib_model_config["dropout"]
) # Recuperar dropout oficial

if evaluation_hidden_channels != 64:
    raise RuntimeError(
        f"La arquitectura oficial requiere 64 unidades ocultas y se "
        f"recibieron {evaluation_hidden_channels}."
    ) # Validar dimensión oculta

print(f"Input channels     : {evaluation_input_channels}") # Mostrar entrada
print(f"Hidden channels    : {evaluation_hidden_channels}") # Mostrar dimensión oculta
print(f"Output channels    : {evaluation_output_channels}") # Mostrar salida
print(f"Dropout            : {evaluation_dropout}") # Mostrar dropout
print("Arquitectura       : 36 -> 64 -> 64 -> 1") # Mostrar arquitectura
print("Configuración      : VALIDADA") # Confirmar configuración

# BLOQUE 5.3. CONSTRUCCIÓN DE LA INSTANCIA INDEPENDIENTE
evaluation_device = torch.device(
    "cpu"
) # Utilizar el dispositivo oficial registrado durante el entrenamiento

evaluation_graphsage = EvaluationGraphSAGE(
    input_channels=evaluation_input_channels,
    hidden_channels=evaluation_hidden_channels,
    output_channels=evaluation_output_channels,
    dropout=evaluation_dropout,
) # Construir instancia independiente del Modelo Oficial

evaluation_graphsage = evaluation_graphsage.to(
    evaluation_device
) # Transferir modelo al dispositivo oficial de evaluación

evaluation_graphsage.eval() # Establecer modo evaluación

print(f"Clase reconstruida : {type(evaluation_graphsage).__name__}") # Mostrar clase
print(f"Dispositivo         : {evaluation_device}") # Mostrar dispositivo
print("Instancia           : CONSTRUIDA") # Confirmar construcción

print("\nBLOQUE 5.4. CARGA DE LOS PESOS PERSISTIDOS") # Mostrar encabezado
if not isinstance(evaluation_graphsage, nn.Module):
    raise TypeError(
        "evaluation_graphsage debe ser una instancia de nn.Module."
    ) # Validar instancia independiente

if not isinstance(official_model_joblib_loaded, dict):
    raise TypeError(
        "official_model_joblib_loaded debe ser un diccionario."
    ) # Validar artefacto JOBLIB

if "model_state_dict" not in official_model_joblib_loaded:
    raise RuntimeError(
        "official_model.joblib no contiene model_state_dict."
    ) # Validar disponibilidad de pesos

evaluation_state_dict = official_model_joblib_loaded[
    "model_state_dict"
] # Recuperar pesos oficiales persistidos

if not isinstance(evaluation_state_dict, dict):
    raise TypeError(
        "model_state_dict debe ser un diccionario."
    ) # Validar estructura de pesos

if len(evaluation_state_dict) == 0:
    raise RuntimeError(
        "model_state_dict está vacío."
    ) # Validar disponibilidad de parámetros

evaluation_graphsage.load_state_dict(
    evaluation_state_dict,
    strict=True
) # Cargar pesos con coincidencia estructural estricta

evaluation_graphsage.eval() # Mantener modelo en modo evaluación

print(f"Parámetros recuperados : {len(evaluation_state_dict)}") # Mostrar cantidad de parámetros
print(f"Dispositivo            : {evaluation_device}") # Mostrar dispositivo
print("Carga de pesos         : VALIDADA") # Confirmar carga estricta
print("evaluation_graphsage   : PESOS CARGADOS") # Confirmar producto

print("\nBLOQUE 5.5. VALIDACIÓN NUMÉRICA DE LOS PESOS RECONSTRUIDOS") # Mostrar encabezado
if not isinstance(
    evaluation_graphsage,
    nn.Module
):
    raise TypeError(
        "evaluation_graphsage debe ser una instancia de nn.Module."
    ) # Validar modelo reconstruido

if not isinstance(
    evaluation_state_dict,
    dict
):
    raise TypeError(
        "evaluation_state_dict debe ser un diccionario."
    ) # Validar pesos persistidos

reconstructed_state_dict = (
    evaluation_graphsage.state_dict()
) # Recuperar pesos del modelo reconstruido

persisted_state_keys = set(
    evaluation_state_dict.keys()
) # Recuperar claves persistidas

reconstructed_state_keys = set(
    reconstructed_state_dict.keys()
) # Recuperar claves reconstruidas

if persisted_state_keys != reconstructed_state_keys:
    raise RuntimeError(
        "Las claves de los pesos persistidos y reconstruidos no coinciden."
    ) # Validar estructura de parámetros

weight_comparison_results = [] # Inicializar resultados de comparación

for parameter_name in sorted(
    persisted_state_keys
):

    persisted_parameter = (
        evaluation_state_dict[
            parameter_name
        ]
        .detach()
        .cpu()
    ) # Recuperar parámetro persistido

    reconstructed_parameter = (
        reconstructed_state_dict[
            parameter_name
        ]
        .detach()
        .cpu()
    ) # Recuperar parámetro reconstruido

    if persisted_parameter.shape != reconstructed_parameter.shape:
        raise RuntimeError(
            f"La dimensión del parámetro '{parameter_name}' no coincide: "
            f"{persisted_parameter.shape} != "
            f"{reconstructed_parameter.shape}"
        ) # Validar dimensiones

    persisted_parameter_float = (
        persisted_parameter.to(torch.float64)
    ) # Convertir parámetro persistido

    reconstructed_parameter_float = (
        reconstructed_parameter.to(torch.float64)
    ) # Convertir parámetro reconstruido

    if not torch.isfinite(
        persisted_parameter_float
    ).all():
        raise RuntimeError(
            f"El parámetro persistido '{parameter_name}' "
            "contiene valores no finitos."
        ) # Validar estabilidad numérica

    if not torch.isfinite(
        reconstructed_parameter_float
    ).all():
        raise RuntimeError(
            f"El parámetro reconstruido '{parameter_name}' "
            "contiene valores no finitos."
        ) # Validar estabilidad numérica

    absolute_difference = torch.abs(
        persisted_parameter_float
        - reconstructed_parameter_float
    ) # Calcular diferencia absoluta

    maximum_difference = float(
        absolute_difference.max().item()
    ) # Obtener diferencia máxima

    mean_difference = float(
        absolute_difference.mean().item()
    ) # Obtener diferencia media

    if not torch.allclose(
        persisted_parameter_float,
        reconstructed_parameter_float,
        rtol=1e-7,
        atol=1e-8
    ):
        raise RuntimeError(
            f"El parámetro '{parameter_name}' "
            "no presenta equivalencia numérica."
        ) # Validar equivalencia numérica

    weight_comparison_results.append(
        {
            "parameter_name": parameter_name,
            "shape": tuple(
                persisted_parameter.shape
            ),
            "max_absolute_difference": maximum_difference,
            "mean_absolute_difference": mean_difference,
            "numerically_equivalent": True,
        }
    ) # Registrar comparación

if len(weight_comparison_results) != len(
    persisted_state_keys
):
    raise RuntimeError(
        "No se compararon todos los parámetros del Modelo Oficial."
    ) # Validar cobertura

maximum_weight_difference = max(
    result["max_absolute_difference"]
    for result in weight_comparison_results
) # Obtener diferencia máxima global

mean_weight_difference = float(
    np.mean(
        [
            result["mean_absolute_difference"]
            for result in weight_comparison_results
        ]
    )
) # Obtener diferencia media global

if maximum_weight_difference > 1e-8:
    raise RuntimeError(
        "La diferencia máxima entre los pesos persistidos y reconstruidos "
        f"es {maximum_weight_difference:.12e}."
    ) # Validar tolerancia global

print(f"Parámetros comparados : {len(weight_comparison_results)}") # Mostrar parámetros comparados
print(f"Diferencia máxima     : {maximum_weight_difference:.12e}") # Mostrar diferencia máxima
print(f"Diferencia media      : {mean_weight_difference:.12e}") # Mostrar diferencia media
print("Estructura de pesos   : COINCIDENTE") # Confirmar estructura
print("Dimensiones           : COINCIDENTES") # Confirmar dimensiones
print("Valores numéricos     : EQUIVALENTES") # Confirmar equivalencia
print("evaluation_graphsage  : VALIDADO") # Confirmar reconstrucción

print("\nBLOQUE 5.6. VALIDACIÓN FUNCIONAL DEL MODELO RECONSTRUIDO") # Mostrar encabezado
if not isinstance(
    evaluation_graphsage,
    nn.Module
):
    raise TypeError(
        "evaluation_graphsage debe ser una instancia de nn.Module."
    ) # Validar modelo reconstruido

if not isinstance(
    benchmark_data_loaded,
    dict
):
    raise TypeError(
        "benchmark_data_loaded debe ser un diccionario."
    ) # Validar datos persistidos

if "graphs" not in benchmark_data_loaded:
    raise RuntimeError(
        "benchmark_data_loaded no contiene graphs."
    ) # Validar GraphData persistidos

if "test_index" not in benchmark_data_loaded:
    raise RuntimeError(
        "benchmark_data_loaded no contiene test_index."
    ) # Validar partición de prueba persistida

evaluation_graphs_persisted = benchmark_data_loaded[
    "graphs"
] # Recuperar GraphData persistidos

evaluation_test_index_persisted = benchmark_data_loaded[
    "test_index"
] # Recuperar índice de prueba persistido

if not isinstance(
    evaluation_graphs_persisted,
    (list, tuple)
):
    raise TypeError(
        "Los GraphData persistidos deben estar contenidos en una lista o tupla."
    ) # Validar colección de grafos

if len(
    evaluation_graphs_persisted
) == 0:
    raise RuntimeError(
        "La colección de GraphData persistidos está vacía."
    ) # Validar disponibilidad de datos

if not isinstance(
    evaluation_test_index_persisted,
    np.ndarray
):
    raise TypeError(
        "test_index persistido debe ser un numpy.ndarray."
    ) # Validar partición de prueba

if len(
    evaluation_test_index_persisted
) == 0:
    raise RuntimeError(
        "test_index persistido está vacío."
    ) # Validar partición de prueba

evaluation_test_position = int(
    evaluation_test_index_persisted[0]
) # Recuperar primer índice de prueba

if evaluation_test_position < 0:
    raise ValueError(
        "El índice de prueba no puede ser negativo."
    ) # Validar índice

if evaluation_test_position >= len(
    evaluation_graphs_persisted
):
    raise IndexError(
        "El índice de prueba está fuera del rango de GraphData persistidos."
    ) # Validar correspondencia del índice

evaluation_graph = evaluation_graphs_persisted[
    evaluation_test_position
] # Recuperar GraphData de prueba

if not hasattr(
    evaluation_graph,
    "x"
):
    raise RuntimeError(
        "El GraphData de prueba no contiene x."
    ) # Validar características

if not hasattr(
    evaluation_graph,
    "edge_index"
):
    raise RuntimeError(
        "El GraphData de prueba no contiene edge_index."
    ) # Validar estructura topológica

evaluation_x = evaluation_graph.x.to(
    evaluation_device
) # Transferir características al dispositivo

evaluation_edge_index = evaluation_graph.edge_index.to(
    evaluation_device
) # Transferir topología al dispositivo

if evaluation_x.ndim != 2:
    raise RuntimeError(
        f"x presenta {evaluation_x.ndim} dimensiones; se esperaban 2."
    ) # Validar dimensionalidad

if evaluation_x.shape[1] != evaluation_input_channels:
    raise RuntimeError(
        f"El GraphData presenta {evaluation_x.shape[1]} variables y "
        f"el modelo requiere {evaluation_input_channels}."
    ) # Validar compatibilidad de entrada

if evaluation_edge_index.ndim != 2:
    raise RuntimeError(
        "edge_index debe presentar dos dimensiones."
    ) # Validar estructura topológica

if evaluation_edge_index.shape[0] != 2:
    raise RuntimeError(
        f"edge_index presenta forma {evaluation_edge_index.shape}; se esperaba [2, num_edges]."
    ) # Validar formato topológico

evaluation_graphsage.eval() # Establecer modo evaluación

with torch.no_grad():
    evaluation_output = evaluation_graphsage(
        evaluation_x,
        evaluation_edge_index
    ) # Ejecutar inferencia funcional

if not isinstance(
    evaluation_output,
    torch.Tensor
):
    raise TypeError(
        "La salida del Modelo Oficial reconstruido debe ser un tensor."
    ) # Validar salida

if evaluation_output.ndim != 2:
    raise RuntimeError(
        f"La salida presenta {evaluation_output.ndim} dimensiones; se esperaban 2."
    ) # Validar dimensionalidad de salida

if evaluation_output.shape[1] != evaluation_output_channels:
    raise RuntimeError(
        f"La salida presenta {evaluation_output.shape[1]} canales y "
        f"el modelo requiere {evaluation_output_channels}."
    ) # Validar canales de salida

if evaluation_output.shape[0] != evaluation_x.shape[0]:
    raise RuntimeError(
        f"La salida contiene {evaluation_output.shape[0]} predicciones para "
        f"{evaluation_x.shape[0]} nodos."
    ) # Validar correspondencia nodo-predicción

if not torch.isfinite(
    evaluation_output
).all():
    raise RuntimeError(
        "La salida del modelo reconstruido contiene valores no finitos."
    ) # Validar estabilidad numérica

print(f"GraphData utilizado    : {evaluation_test_position}") # Mostrar GraphData
print(f"Nodos procesados       : {evaluation_x.shape[0]}") # Mostrar nodos
print(f"Variables de entrada   : {evaluation_x.shape[1]}") # Mostrar variables
print(f"Aristas procesadas     : {evaluation_edge_index.shape[1]}") # Mostrar aristas
print(f"Predicciones generadas : {evaluation_output.shape[0]}") # Mostrar predicciones
print(f"Dimensión de salida    : {tuple(evaluation_output.shape)}") # Mostrar salida
print("Valores no finitos     : 0") # Confirmar estabilidad numérica
print("Inferencia funcional   : VALIDADA") # Confirmar funcionalidad
print("evaluation_graphsage   : LISTO PARA EVALUACIÓN") # Confirmar producto

print("\nBLOQUE 5.7. CIERRE Y VALIDACIÓN DE LA RECONSTRUCCIÓN DEL MODELO OFICIAL") # Mostrar encabezado
if not isinstance(
    evaluation_graphsage,
    nn.Module
):
    raise TypeError(
        "evaluation_graphsage debe ser una instancia de nn.Module."
    ) # Validar instancia final

if not isinstance(
    evaluation_state_dict,
    dict
):
    raise TypeError(
        "evaluation_state_dict debe ser un diccionario."
    ) # Validar pesos oficiales

evaluation_final_state_dict = (
    evaluation_graphsage.state_dict()
) # Recuperar estado final del modelo

if set(
    evaluation_final_state_dict.keys()
) != set(
    evaluation_state_dict.keys()
):
    raise RuntimeError(
        "La estructura final de evaluation_graphsage "
        "no coincide con los pesos oficiales."
    ) # Validar estructura final

for parameter_name in evaluation_state_dict:

    persisted_parameter = (
        evaluation_state_dict[
            parameter_name
        ]
        .detach()
        .cpu()
        .to(torch.float64)
    ) # Recuperar parámetro oficial

    final_parameter = (
        evaluation_final_state_dict[
            parameter_name
        ]
        .detach()
        .cpu()
        .to(torch.float64)
    ) # Recuperar parámetro final

    if not torch.equal(
        persisted_parameter,
        final_parameter
    ):
        raise RuntimeError(
            f"El parámetro '{parameter_name}' "
            "no coincide con los pesos oficiales."
        ) # Validar identidad paramétrica final

if evaluation_graphsage.training:
    raise RuntimeError(
        "evaluation_graphsage no se encuentra en modo evaluación."
    ) # Validar modo evaluación

if evaluation_input_channels != 36:
    raise RuntimeError(
        "La instancia final no presenta las 36 variables de entrada esperadas."
    ) # Validar entrada

if evaluation_hidden_channels != 64:
    raise RuntimeError(
        "La instancia final no presenta las 64 unidades ocultas esperadas."
    ) # Validar arquitectura oculta

if evaluation_output_channels != 1:
    raise RuntimeError(
        "La instancia final no presenta una salida de dimensión 1."
    ) # Validar salida

print(f"Modelo Oficial         : {joblib_model_name}") # Mostrar modelo
print(f"Código Oficial         : {joblib_model_code}") # Mostrar código
print(f"Familia                : {joblib_model_family}") # Mostrar familia
print("Arquitectura            : 36 -> 64 -> 64 -> 1") # Mostrar arquitectura
print(f"Parámetros validados   : {len(evaluation_state_dict)}") # Mostrar parámetros
print(f"Dispositivo            : {evaluation_device}") # Mostrar dispositivo
print("Modo del modelo         : EVALUACIÓN") # Confirmar modo
print("Estructura              : VALIDADA") # Confirmar estructura
print("Pesos                   : VALIDADOS") # Confirmar pesos
print("Inferencia              : VALIDADA") # Confirmar funcionalidad
print("evaluation_graphsage    : LISTO PARA EVALUACIÓN") # Confirmar producto

# BLOQUE 6. DIAGNÓSTICO PREVIO PARA LA VALIDACIÓN ESTRICTA DEL MODELO RECONSTRUIDO
# Objetivo: Verificar la disponibilidad de identidad, arquitectura, configuración y pesos necesarios para validar evaluation_graphsage.
# Entradas: evaluation_graphsage, artefactos oficiales persistidos y configuración validada.
# Producto: Información diagnóstica disponible para construir la validación estricta.
# Pregunta científica: ¿Se encuentran disponibles todos los elementos necesarios para validar independientemente el Modelo Oficial reconstruido?

print("\nBLOQUE 6. DIAGNÓSTICO PREVIO PARA LA VALIDACIÓN ESTRICTA DEL MODELO RECONSTRUIDO") # Mostrar encabezado
print(f"Modelo reconstruido     : {type(evaluation_graphsage).__name__}") # Mostrar clase reconstruida
print(f"Código oficial          : {joblib_model_code}") # Mostrar código oficial
print(f"Nombre oficial          : {joblib_model_name}") # Mostrar nombre oficial
print(f"Familia oficial         : {joblib_model_family}") # Mostrar familia oficial
print(f"Input channels          : {evaluation_input_channels}") # Mostrar dimensión de entrada
print(f"Hidden channels         : {evaluation_hidden_channels}") # Mostrar dimensión oculta
print(f"Output channels         : {evaluation_output_channels}") # Mostrar dimensión de salida
print(f"Dropout                 : {evaluation_dropout}") # Mostrar dropout
print(f"Dispositivo             : {evaluation_device}") # Mostrar dispositivo

evaluation_parameter_count = sum(
    parameter.numel()
    for parameter in evaluation_graphsage.parameters()
) # Calcular cantidad de parámetros de la instancia reconstruida

evaluation_state_dict_keys = list(
    evaluation_graphsage.state_dict().keys()
) # Recuperar estructura del estado del modelo

print(f"Parámetros totales       : {evaluation_parameter_count}") # Mostrar cantidad de parámetros
print(f"Tensores state_dict     : {len(evaluation_state_dict_keys)}") # Mostrar cantidad de tensores
print(f"Modo entrenamiento      : {evaluation_graphsage.training}") # Mostrar estado del modelo
print(f"Pesos oficiales         : {len(evaluation_state_dict)}") # Mostrar pesos persistidos
print(f"Strict loading          : {True}") # Registrar criterio estricto
print("Diagnóstico              : COMPLETADO") # Confirmar diagnóstico

print("\nBLOQUE 6.1. VALIDACIÓN DE IDENTIDAD DEL MODELO RECONSTRUIDO") # Mostrar encabezado
if not isinstance(
    official_model_joblib_loaded,
    dict
):
    raise TypeError(
        "official_model_joblib_loaded debe ser un diccionario."
    ) # Validar artefacto oficial

required_identity_fields = [
    "model_code",
    "model_name",
    "family",
] # Definir campos oficiales de identidad

for field in required_identity_fields:
    if field not in official_model_joblib_loaded:
        raise RuntimeError(
            f"official_model.joblib no contiene '{field}'."
        ) # Validar identidad persistida

evaluation_model_code = str(
    official_model_joblib_loaded["model_code"]
) # Recuperar código oficial

evaluation_model_name = str(
    official_model_joblib_loaded["model_name"]
) # Recuperar nombre oficial

evaluation_model_family = str(
    official_model_joblib_loaded["family"]
) # Recuperar familia oficial

if evaluation_model_code != joblib_model_code:
    raise RuntimeError(
        "El código utilizado durante la evaluación "
        "no coincide con el código oficial persistido."
    ) # Validar código

if evaluation_model_name.strip().lower() != joblib_model_name.strip().lower():
    raise RuntimeError(
        "El nombre utilizado durante la evaluación "
        "no coincide con el nombre oficial persistido."
    ) # Validar nombre

if evaluation_model_family.strip().lower() != joblib_model_family.strip().lower():
    raise RuntimeError(
        "La familia utilizada durante la evaluación "
        "no coincide con la familia oficial persistida."
    ) # Validar familia

if evaluation_model_code != "GNN02":
    raise RuntimeError(
        "El código del Modelo Oficial debe ser GNN02."
    ) # Validar código científico oficial

if evaluation_model_name.strip().lower() != "graphsage":
    raise RuntimeError(
        "El nombre del Modelo Oficial debe ser graphsage."
    ) # Validar nombre científico oficial

if evaluation_model_family.strip().lower() != "graph_neural_networks":
    raise RuntimeError(
        "La familia del Modelo Oficial debe ser graph_neural_networks."
    ) # Validar familia científica oficial

print(f"Código del modelo      : {evaluation_model_code}") # Mostrar código
print(f"Nombre del modelo      : {evaluation_model_name}") # Mostrar nombre
print(f"Familia del modelo     : {evaluation_model_family}") # Mostrar familia
print("Identidad del modelo   : VALIDADA") # Confirmar identidad

print("\nBLOQUE 6.2. VALIDACIÓN DE LAS DIMENSIONES Y CONFIGURACIÓN ARQUITECTÓNICA") # Mostrar encabezado
if not hasattr(
    evaluation_graphsage,
    "conv1"
):
    raise RuntimeError(
        "evaluation_graphsage no contiene conv1."
    ) # Validar primera capa

if not hasattr(
    evaluation_graphsage,
    "conv2"
):
    raise RuntimeError(
        "evaluation_graphsage no contiene conv2."
    ) # Validar segunda capa

if not hasattr(
    evaluation_graphsage,
    "output_layer"
):
    raise RuntimeError(
        "evaluation_graphsage no contiene output_layer."
    ) # Validar capa de salida

if not hasattr(
    evaluation_graphsage,
    "dropout"
):
    raise RuntimeError(
        "evaluation_graphsage no contiene dropout."
    ) # Validar configuración dropout

evaluation_conv1_input = int(
    evaluation_graphsage.conv1.in_channels
) # Recuperar entrada de conv1

evaluation_conv1_output = int(
    evaluation_graphsage.conv1.out_channels
) # Recuperar salida de conv1

evaluation_conv2_input = int(
    evaluation_graphsage.conv2.in_channels
) # Recuperar entrada de conv2

evaluation_conv2_output = int(
    evaluation_graphsage.conv2.out_channels
) # Recuperar salida de conv2

evaluation_output_input = int(
    evaluation_graphsage.output_layer.in_features
) # Recuperar entrada de la capa de salida

evaluation_output_output = int(
    evaluation_graphsage.output_layer.out_features
) # Recuperar salida de la capa de salida

evaluation_dropout_value = float(
    evaluation_graphsage.dropout
) # Recuperar dropout de la instancia

if evaluation_conv1_input != evaluation_input_channels:
    raise RuntimeError(
        f"conv1 requiere {evaluation_input_channels} entradas y "
        f"presenta {evaluation_conv1_input}."
    ) # Validar entrada oficial

if evaluation_conv1_output != evaluation_hidden_channels:
    raise RuntimeError(
        f"conv1 debe producir {evaluation_hidden_channels} unidades y "
        f"presenta {evaluation_conv1_output}."
    ) # Validar primera dimensión oculta

if evaluation_conv2_input != evaluation_hidden_channels:
    raise RuntimeError(
        f"conv2 debe recibir {evaluation_hidden_channels} unidades y "
        f"recibe {evaluation_conv2_input}."
    ) # Validar entrada de segunda capa

if evaluation_conv2_output != evaluation_hidden_channels:
    raise RuntimeError(
        f"conv2 debe producir {evaluation_hidden_channels} unidades y "
        f"produce {evaluation_conv2_output}."
    ) # Validar segunda dimensión oculta

if evaluation_output_input != evaluation_hidden_channels:
    raise RuntimeError(
        f"output_layer debe recibir {evaluation_hidden_channels} unidades y "
        f"recibe {evaluation_output_input}."
    ) # Validar entrada de salida

if evaluation_output_output != evaluation_output_channels:
    raise RuntimeError(
        f"output_layer debe producir {evaluation_output_channels} salida y "
        f"produce {evaluation_output_output}."
    ) # Validar salida oficial

if not np.isclose(
    evaluation_dropout_value,
    evaluation_dropout,
    rtol=0.0,
    atol=1e-12
):
    raise RuntimeError(
        f"El dropout reconstruido {evaluation_dropout_value} "
        f"no coincide con el dropout oficial {evaluation_dropout}."
    ) # Validar dropout

if not np.isclose(
    evaluation_dropout_value,
    0.3,
    rtol=0.0,
    atol=1e-12
):
    raise RuntimeError(
        "El dropout del Modelo Oficial debe ser 0.3."
    ) # Validar configuración científica

print(f"Entrada conv1          : {evaluation_conv1_input}") # Mostrar entrada
print(f"Salida conv1           : {evaluation_conv1_output}") # Mostrar salida
print(f"Entrada conv2          : {evaluation_conv2_input}") # Mostrar entrada
print(f"Salida conv2           : {evaluation_conv2_output}") # Mostrar salida
print(f"Entrada output_layer   : {evaluation_output_input}") # Mostrar entrada
print(f"Salida output_layer    : {evaluation_output_output}") # Mostrar salida
print(f"Dropout                : {evaluation_dropout_value}") # Mostrar dropout
print("Arquitectura           : 36 -> 64 -> 64 -> 1") # Mostrar arquitectura
print("Configuración           : VALIDADA") # Confirmar configuración

print("\nBLOQUE 6.3. VALIDACIÓN DE PARÁMETROS Y ESTRUCTURA DEL STATE DICT") # Mostrar encabezado
if not isinstance(
    evaluation_graphsage,
    nn.Module
):
    raise TypeError(
        "evaluation_graphsage debe ser una instancia de nn.Module."
    ) # Validar modelo reconstruido

if not isinstance(
    evaluation_state_dict,
    dict
):
    raise TypeError(
        "evaluation_state_dict debe ser un diccionario."
    ) # Validar pesos oficiales

evaluation_parameter_count = sum(
    parameter.numel()
    for parameter in evaluation_graphsage.parameters()
) # Calcular cantidad total de parámetros

if evaluation_parameter_count != 12993:
    raise RuntimeError(
        f"El Modelo Oficial debe contener 12993 parámetros y "
        f"evaluation_graphsage contiene {evaluation_parameter_count}."
    ) # Validar cantidad oficial de parámetros

evaluation_state_dict = evaluation_graphsage.state_dict() # Recuperar state_dict reconstruido

official_state_dict_keys = set(
    official_model_joblib_loaded["model_state_dict"].keys()
) # Recuperar claves oficiales

evaluation_state_dict_keys = set(
    evaluation_state_dict.keys()
) # Recuperar claves reconstruidas

if evaluation_state_dict_keys != official_state_dict_keys:
    missing_state_dict_keys = sorted(
        official_state_dict_keys - evaluation_state_dict_keys
    ) # Identificar claves faltantes

    unexpected_state_dict_keys = sorted(
        evaluation_state_dict_keys - official_state_dict_keys
    ) # Identificar claves inesperadas

    raise RuntimeError(
        "La estructura del state_dict no coincide con los pesos oficiales. "
        f"Claves faltantes: {missing_state_dict_keys}. "
        f"Claves inesperadas: {unexpected_state_dict_keys}."
    ) # Validar estructura del state_dict

if len(
    evaluation_state_dict_keys
) != 8:
    raise RuntimeError(
        f"El state_dict debe contener 8 tensores y contiene "
        f"{len(evaluation_state_dict_keys)}."
    ) # Validar cantidad de tensores

for parameter_name in sorted(
    official_state_dict_keys
):

    official_parameter = official_model_joblib_loaded[
        "model_state_dict"
    ][
        parameter_name
    ] # Recuperar parámetro oficial

    evaluation_parameter = evaluation_state_dict[
        parameter_name
    ] # Recuperar parámetro reconstruido

    if official_parameter.shape != evaluation_parameter.shape:
        raise RuntimeError(
            f"El parámetro '{parameter_name}' presenta dimensiones "
            f"diferentes: {official_parameter.shape} != "
            f"{evaluation_parameter.shape}."
        ) # Validar dimensión del parámetro

    if official_parameter.numel() != evaluation_parameter.numel():
        raise RuntimeError(
            f"El parámetro '{parameter_name}' presenta una cantidad "
            "diferente de elementos."
        ) # Validar cantidad de elementos

print(f"Parámetros totales     : {evaluation_parameter_count}") # Mostrar parámetros
print(f"Tensores state_dict    : {len(evaluation_state_dict_keys)}") # Mostrar tensores
print("Claves state_dict      : COINCIDENTES") # Confirmar claves
print("Dimensiones            : COINCIDENTES") # Confirmar dimensiones
print("Cantidad de parámetros : VALIDADA") # Confirmar parámetros
print("State_dict             : VALIDADO") # Confirmar estructura

print("\nBLOQUE 6.4. CIERRE DE LA VALIDACIÓN ESTRICTA DEL MODELO RECONSTRUIDO") # Mostrar encabezado
if not isinstance(
    evaluation_graphsage,
    nn.Module
):
    raise TypeError(
        "evaluation_graphsage debe ser una instancia de nn.Module."
    ) # Validar modelo independiente

if not isinstance(
    evaluation_state_dict,
    dict
):
    raise TypeError(
        "evaluation_state_dict debe ser un diccionario."
    ) # Validar pesos oficiales

if evaluation_graphsage.training:
    raise RuntimeError(
        "evaluation_graphsage debe permanecer en modo evaluación."
    ) # Validar modo evaluación

evaluation_strict_test_model = EvaluationGraphSAGE(
    input_channels=evaluation_input_channels,
    hidden_channels=evaluation_hidden_channels,
    output_channels=evaluation_output_channels,
    dropout=evaluation_dropout,
) # Construir instancia temporal para prueba estricta

evaluation_strict_test_model = evaluation_strict_test_model.to(
    evaluation_device
) # Transferir instancia temporal al dispositivo

evaluation_strict_test_model.load_state_dict(
    evaluation_state_dict,
    strict=True
) # Comprobar compatibilidad estricta con los pesos oficiales

evaluation_strict_test_model.eval() # Establecer modo evaluación

strict_validation_state_dict = (
    evaluation_strict_test_model.state_dict()
) # Recuperar estado después de carga estricta

for parameter_name in evaluation_state_dict:

    official_parameter = (
        evaluation_state_dict[
            parameter_name
        ]
        .detach()
        .cpu()
    ) # Recuperar parámetro oficial

    strict_parameter = (
        strict_validation_state_dict[
            parameter_name
        ]
        .detach()
        .cpu()
    ) # Recuperar parámetro cargado estrictamente

    if not torch.equal(
        official_parameter,
        strict_parameter
    ):
        raise RuntimeError(
            f"El parámetro '{parameter_name}' no coincide después "
            "de la carga strict=True."
        ) # Validar integridad de carga estricta

if len(
    evaluation_state_dict
) != 8:
    raise RuntimeError(
        f"Se esperaban 8 tensores y se recuperaron "
        f"{len(evaluation_state_dict)}."
    ) # Validar cantidad de tensores

if evaluation_parameter_count != 12993:
    raise RuntimeError(
        f"Se esperaban 12993 parámetros y se recuperaron "
        f"{evaluation_parameter_count}."
    ) # Validar cantidad de parámetros

print(f"Modelo independiente   : {type(evaluation_graphsage).__name__}") # Mostrar modelo
print(f"Arquitectura            : {evaluation_input_channels} -> {evaluation_hidden_channels} -> {evaluation_hidden_channels} -> {evaluation_output_channels}") # Mostrar arquitectura
print(f"Dropout                 : {evaluation_dropout}") # Mostrar dropout
print(f"Parámetros totales      : {evaluation_parameter_count}") # Mostrar parámetros
print(f"Tensores state_dict     : {len(evaluation_state_dict)}") # Mostrar tensores
print("Carga strict=True       : VALIDADA") # Confirmar carga estricta
print("Integridad de pesos     : VALIDADA") # Confirmar integridad
print("Modo evaluación         : VALIDADO") # Confirmar modo
print("Modelo independiente    : VALIDADO") # Confirmar producto

evaluation_block_6_status = "VALIDATED" # Registrar aprobación final del Bloque 6
evaluation_block_6_stage = "VALIDATED" # Registrar etapa final aprobada

print(f"Estado Bloque 6       : {evaluation_block_6_status}") # Mostrar estado final
print("BLOQUE 6              : VALIDATED") # Confirmar cierre del bloque

# BLOQUE 7. DIAGNÓSTICO PREVIO PARA LA COMPARACIÓN NUMÉRICA DE LOS PESOS
# Objetivo: Verificar la disponibilidad y estructura de los pesos JOBLIB y PyTorch antes de ejecutar la comparación tensor por tensor.
# Entradas: official_model_joblib_loaded y official_checkpoint_loaded.
# Producto: Información diagnóstica disponible para la comparación numérica.
# Pregunta científica: ¿Se encuentran disponibles los dos estados de parámetros oficiales necesarios para realizar una comparación numérica independiente?

print("\nBLOQUE 7. DIAGNÓSTICO PREVIO PARA LA COMPARACIÓN NUMÉRICA DE LOS PESOS") # Mostrar encabezado
if not isinstance(
    official_model_joblib_loaded,
    dict
):
    raise TypeError(
        "official_model_joblib_loaded debe ser un diccionario."
    ) # Validar artefacto JOBLIB

if "model_state_dict" not in official_model_joblib_loaded:
    raise RuntimeError(
        "official_model_joblib_loaded no contiene model_state_dict."
    ) # Validar pesos JOBLIB

joblib_state_dict = official_model_joblib_loaded[
    "model_state_dict"
] # Recuperar pesos JOBLIB

if not isinstance(
    joblib_state_dict,
    dict
):
    raise TypeError(
        "joblib_state_dict debe ser un diccionario."
    ) # Validar estructura JOBLIB

if not isinstance(
    official_checkpoint_loaded,
    dict
):
    raise TypeError(
        "official_checkpoint_loaded debe ser un diccionario."
    ) # Validar checkpoint PyTorch

if "model_state_dict" in official_checkpoint_loaded:
    torch_state_dict = official_checkpoint_loaded[
        "model_state_dict"
    ] # Recuperar pesos PyTorch desde checkpoint
else:
    torch_state_dict = official_checkpoint_loaded # Utilizar checkpoint como state_dict cuando corresponde

if not isinstance(
    torch_state_dict,
    dict
):
    raise TypeError(
        "torch_state_dict debe ser un diccionario."
    ) # Validar estructura PyTorch

joblib_state_keys = set(
    joblib_state_dict.keys()
) # Recuperar claves JOBLIB

torch_state_keys = set(
    torch_state_dict.keys()
) # Recuperar claves PyTorch

print(f"Pesos JOBLIB            : {len(joblib_state_dict)}") # Mostrar cantidad JOBLIB
print(f"Pesos PyTorch           : {len(torch_state_dict)}") # Mostrar cantidad PyTorch
print(f"Claves JOBLIB           : {len(joblib_state_keys)}") # Mostrar claves JOBLIB
print(f"Claves PyTorch          : {len(torch_state_keys)}") # Mostrar claves PyTorch
print(f"Claves coincidentes     : {joblib_state_keys == torch_state_keys}") # Mostrar coincidencia estructural
print("Diagnóstico              : COMPLETADO") # Confirmar diagnóstico

print("\nBLOQUE 7.1. VALIDACIÓN ESTRUCTURAL DE LOS PESOS JOBLIB Y PYTORCH") # Mostrar encabezado
if joblib_state_keys != torch_state_keys:
    raise RuntimeError(
        "JOBLIB y PyTorch no contienen las mismas claves."
    ) # Validar coincidencia de claves

weight_structure_results = [] # Inicializar resultados estructurales
for parameter_name in sorted(
    joblib_state_keys
):
    joblib_parameter = joblib_state_dict[
        parameter_name
    ] # Recuperar parámetro JOBLIB

    torch_parameter = torch_state_dict[
        parameter_name
    ] # Recuperar parámetro PyTorch

    if not isinstance(
        joblib_parameter,
        torch.Tensor
    ):
        raise TypeError(
            f"El parámetro '{parameter_name}' de JOBLIB "
            "no es un tensor de PyTorch."
        ) # Validar tipo JOBLIB

    if not isinstance(
        torch_parameter,
        torch.Tensor
    ):
        raise TypeError(
            f"El parámetro '{parameter_name}' de PyTorch "
            "no es un tensor."
        ) # Validar tipo PyTorch

    joblib_shape = tuple(
        joblib_parameter.shape
    ) # Recuperar dimensión JOBLIB

    torch_shape = tuple(
        torch_parameter.shape
    ) # Recuperar dimensión PyTorch

    if joblib_shape != torch_shape:
        raise RuntimeError(
            f"El parámetro '{parameter_name}' presenta dimensiones "
            f"diferentes: {joblib_shape} != {torch_shape}."
        ) # Validar dimensiones

    if joblib_parameter.numel() != torch_parameter.numel():
        raise RuntimeError(
            f"El parámetro '{parameter_name}' presenta diferente "
            "cantidad de elementos."
        ) # Validar cantidad de elementos

    if not torch.isfinite(
        joblib_parameter.detach().cpu()
    ).all():
        raise RuntimeError(
            f"El parámetro '{parameter_name}' de JOBLIB "
            "contiene valores no finitos."
        ) # Validar estabilidad JOBLIB

    if not torch.isfinite(
        torch_parameter.detach().cpu()
    ).all():
        raise RuntimeError(
            f"El parámetro '{parameter_name}' de PyTorch "
            "contiene valores no finitos."
        ) # Validar estabilidad PyTorch

    weight_structure_results.append(
        {
            "parameter_name": parameter_name,
            "shape": joblib_shape,
            "numel": int(joblib_parameter.numel()),
            "structure_status": "VALIDATED",
        }
    ) # Registrar validación estructural

if len(
    weight_structure_results
) != len(
    joblib_state_keys
):
    raise RuntimeError(
        "No se validaron todos los tensores de los pesos."
    ) # Validar cobertura

print(f"Tensores validados      : {len(weight_structure_results)}") # Mostrar cantidad
print("Claves                  : COINCIDENTES") # Confirmar claves
print("Dimensiones             : COINCIDENTES") # Confirmar dimensiones
print("Cantidad de elementos   : VALIDADA") # Confirmar elementos
print("Valores finitos         : VALIDADOS") # Confirmar estabilidad
print("Estructura de pesos     : VALIDADA") # Confirmar estructura

print("\nBLOQUE 7.2. COMPARACIÓN NUMÉRICA TENSOR POR TENSOR") # Mostrar encabezado
weight_comparison_results = [] # Inicializar resultados de comparación
for parameter_name in sorted(
    joblib_state_keys
):
    joblib_parameter = joblib_state_dict[
        parameter_name
    ] # Recuperar tensor JOBLIB

    torch_parameter = torch_state_dict[
        parameter_name
    ] # Recuperar tensor PyTorch

    joblib_parameter_float = (
        joblib_parameter
        .detach()
        .cpu()
        .to(torch.float64)
    ) # Normalizar tensor JOBLIB

    torch_parameter_float = (
        torch_parameter
        .detach()
        .cpu()
        .to(torch.float64)
    ) # Normalizar tensor PyTorch

    absolute_difference = torch.abs(
        joblib_parameter_float
        - torch_parameter_float
    ) # Calcular diferencia absoluta

    max_abs_difference = float(
        absolute_difference.max().item()
    ) # Obtener diferencia máxima

    mean_abs_difference = float(
        absolute_difference.mean().item()
    ) # Obtener diferencia media

    if not torch.allclose(
        joblib_parameter_float,
        torch_parameter_float,
        rtol=1e-7,
        atol=1e-8
    ):
        raise RuntimeError(
            f"El parámetro '{parameter_name}' presenta diferencias "
            "numéricas superiores a la tolerancia."
        ) # Validar equivalencia numérica

    weight_comparison_results.append(
        {
            "parameter_name": parameter_name,
            "shape": tuple(
                joblib_parameter.shape
            ),
            "max_abs_difference": max_abs_difference,
            "mean_abs_difference": mean_abs_difference,
            "comparison_status": "VALIDATED",
        }
    ) # Registrar comparación tensor por tensor

if len(
    weight_comparison_results
) != len(
    joblib_state_keys
):
    raise RuntimeError(
        "No se compararon todos los tensores de los pesos oficiales."
    ) # Validar cobertura

weight_comparison_df = pd.DataFrame(
    weight_comparison_results
) # Construir tabla de comparación

max_abs_difference = float(
    weight_comparison_df[
        "max_abs_difference"
    ].max()
) # Obtener diferencia máxima global

mean_abs_difference = float(
    weight_comparison_df[
        "mean_abs_difference"
    ].mean()
) # Obtener diferencia media global

if not np.isfinite(
    max_abs_difference
):
    raise RuntimeError(
        "max_abs_difference no es finito."
    ) # Validar diferencia máxima

if not np.isfinite(
    mean_abs_difference
):
    raise RuntimeError(
        "mean_abs_difference no es finito."
    ) # Validar diferencia media

if max_abs_difference > 1e-8:
    comparison_status = "FAILED"
    raise RuntimeError(
        f"La diferencia máxima {max_abs_difference:.12e} "
        "supera la tolerancia establecida."
    ) # Validar equivalencia global

comparison_status = "VALIDATED" # Registrar estado global

print(f"Tensores comparados     : {len(weight_comparison_results)}") # Mostrar cantidad comparada
print(f"max_abs_difference      : {max_abs_difference:.12e}") # Mostrar diferencia máxima
print(f"mean_abs_difference     : {mean_abs_difference:.12e}") # Mostrar diferencia media
print(f"comparison_status       : {comparison_status}") # Mostrar estado

print("\nBLOQUE 7.3. CONSOLIDACIÓN DE LA COMPARACIÓN NUMÉRICA") # Mostrar encabezado
if not isinstance(
    weight_comparison_df,
    pd.DataFrame
):
    raise TypeError(
        "weight_comparison_df debe ser un DataFrame."
    ) # Validar resultados de comparación

required_comparison_columns = [
    "parameter_name",
    "shape",
    "max_abs_difference",
    "mean_abs_difference",
    "comparison_status",
] # Definir contrato de comparación

missing_comparison_columns = [
    column
    for column in required_comparison_columns
    if column not in weight_comparison_df.columns
] # Identificar columnas faltantes

if missing_comparison_columns:
    raise RuntimeError(
        "Faltan columnas en la comparación numérica: "
        f"{missing_comparison_columns}"
    ) # Validar contrato

if len(
    weight_comparison_df
) != len(
    joblib_state_dict
):
    raise RuntimeError(
        "La cantidad de resultados no coincide con la cantidad de pesos."
    ) # Validar cobertura

if not np.isfinite(
    weight_comparison_df[
        "max_abs_difference"
    ].to_numpy(
        dtype=np.float64
    )
).all():
    raise RuntimeError(
        "Existen valores no finitos en max_abs_difference."
    ) # Validar estabilidad

if not np.isfinite(
    weight_comparison_df[
        "mean_abs_difference"
    ].to_numpy(
        dtype=np.float64
    )
).all():
    raise RuntimeError(
        "Existen valores no finitos en mean_abs_difference."
    ) # Validar estabilidad

if not (
    weight_comparison_df[
        "comparison_status"
    ] == "VALIDATED"
).all():
    raise RuntimeError(
        "No todos los tensores presentan estado VALIDATED."
    ) # Validar estado tensorial

if max_abs_difference != 0.0:
    raise RuntimeError(
        f"La diferencia máxima global es {max_abs_difference:.12e} y "
        "no es exactamente cero."
    ) # Validar equivalencia máxima

if mean_abs_difference != 0.0:
    raise RuntimeError(
        f"La diferencia media global es {mean_abs_difference:.12e} y "
        "no es exactamente cero."
    ) # Validar equivalencia media

if comparison_status != "VALIDATED":
    raise RuntimeError(
        "La comparación global no presenta estado VALIDATED."
    ) # Validar estado global

print(f"Tensores validados      : {len(weight_comparison_df)}") # Mostrar tensores
print(f"max_abs_difference      : {max_abs_difference:.12e}") # Mostrar diferencia máxima
print(f"mean_abs_difference     : {mean_abs_difference:.12e}") # Mostrar diferencia media
print(f"comparison_status       : {comparison_status}") # Mostrar estado global
print("Comparación JOBLIB-PyTorch : VALIDADA") # Confirmar comparación

print("\nBLOQUE 7.4. CIERRE DE LA VALIDACIÓN NUMÉRICA") # Mostrar encabezado
if comparison_status != "VALIDATED":
    raise RuntimeError(
        "La comparación JOBLIB-PyTorch no presenta estado VALIDATED."
    ) # Validar estado global

if max_abs_difference != 0.0:
    raise RuntimeError(
        f"max_abs_difference debe ser 0.0 y presenta {max_abs_difference:.12e}."
    ) # Validar diferencia máxima

if mean_abs_difference != 0.0:
    raise RuntimeError(
        f"mean_abs_difference debe ser 0.0 y presenta {mean_abs_difference:.12e}."
    ) # Validar diferencia media

if len(weight_comparison_df) != 8:
    raise RuntimeError(
        f"Se esperaban 8 tensores comparados y se obtuvieron {len(weight_comparison_df)}."
    ) # Validar cobertura de comparación

if not (
    weight_comparison_df["comparison_status"] == "VALIDATED"
).all():
    raise RuntimeError(
        "No todos los tensores presentan estado VALIDATED."
    ) # Validar estado tensorial

comparison_validation_summary = {
    "tensors_compared": int(len(weight_comparison_df)),
    "max_abs_difference": float(max_abs_difference),
    "mean_abs_difference": float(mean_abs_difference),
    "comparison_status": comparison_status,
} # Construir resumen de validación

print(f"Tensores comparados     : {comparison_validation_summary['tensors_compared']}") # Mostrar cobertura
print(f"max_abs_difference      : {comparison_validation_summary['max_abs_difference']:.12e}") # Mostrar diferencia máxima
print(f"mean_abs_difference     : {comparison_validation_summary['mean_abs_difference']:.12e}") # Mostrar diferencia media
print(f"comparison_status       : {comparison_validation_summary['comparison_status']}") # Mostrar estado
print("Equivalencia numérica   : VALIDADA") # Confirmar equivalencia
print("Persistencia JOBLIB     : VALIDADA") # Confirmar persistencia
print("BLOQUE 7                : COMPLETADO") # Confirmar cierre

# BLOQUE 8. DIAGNÓSTICO PREVIO PARA LA RECUPERACIÓN DEL CONJUNTO DE PRUEBA
# Objetivo: Verificar las entradas oficiales necesarias para recuperar exclusivamente el conjunto TEST.
# Entradas: benchmark_data_loaded.
# Producto: Estado diagnóstico del Bloque 8.
# Pregunta científica: ¿Están disponibles todas las entradas oficiales necesarias para recuperar reproduciblemente TEST?

print("\nBLOQUE 8. DIAGNÓSTICO PREVIO PARA LA RECUPERACIÓN DEL CONJUNTO DE PRUEBA") # Mostrar encabezado
evaluation_block_8_status = "ERROR" # Inicializar estado en ERROR hasta completar todas las validaciones
evaluation_block_8_stage = "DIAGNOSTICO" # Registrar etapa actual
evaluation_test_ready = False # Impedir declarar TEST listo prematuramente

if not isinstance(
    benchmark_data_loaded,
    dict
):
    raise TypeError(
        "benchmark_data_loaded debe ser un diccionario."
    ) # Validar estructura

required_test_inputs = [
    "graphs",
    "test_index",
    "scaler",
] # Definir entradas oficiales requeridas

missing_test_inputs = [
    field
    for field in required_test_inputs
    if field not in benchmark_data_loaded
] # Identificar entradas faltantes

if missing_test_inputs:
    raise RuntimeError(
        "BenchmarkData no contiene las entradas necesarias para TEST: "
        f"{missing_test_inputs}"
    ) # Detener bloque ante entradas faltantes

evaluation_graphs_persisted = benchmark_data_loaded[
    "graphs"
] # Recuperar GraphData persistidos

evaluation_test_index_persisted = benchmark_data_loaded[
    "test_index"
] # Recuperar índice TEST persistido

evaluation_scaler_persisted = benchmark_data_loaded[
    "scaler"
] # Recuperar scaler persistido

if not isinstance(
    evaluation_graphs_persisted,
    (list, tuple)
):
    raise TypeError(
        "evaluation_graphs_persisted debe ser una lista o tupla."
    ) # Validar GraphData

if len(
    evaluation_graphs_persisted
) != 13:
    raise RuntimeError(
        "Se esperaban exactamente 13 GraphData persistidos."
    ) # Validar cobertura temporal

if not isinstance(
    evaluation_test_index_persisted,
    np.ndarray
):
    raise TypeError(
        "evaluation_test_index_persisted debe ser un ndarray."
    ) # Validar índice TEST

if evaluation_test_index_persisted.ndim != 1:
    raise ValueError(
        "evaluation_test_index_persisted debe ser un vector unidimensional."
    ) # Validar dimensionalidad

if len(
    evaluation_test_index_persisted
) != 3:
    raise RuntimeError(
        "La partición TEST oficial debe contener exactamente 3 índices."
    ) # Validar tamaño TEST

if evaluation_scaler_persisted is None:
    raise RuntimeError(
        "evaluation_scaler_persisted no está disponible."
    ) # Validar scaler

evaluation_block_8_status = "DIAGNOSTIC_VALIDATED" # Registrar diagnóstico aprobado
evaluation_block_8_stage = "DIAGNOSTICO_VALIDADO" # Registrar etapa aprobada

print(f"GraphData disponibles   : {len(evaluation_graphs_persisted)}") # Mostrar GraphData
print(f"Índices TEST            : {evaluation_test_index_persisted.tolist()}") # Mostrar índices
print(f"GraphData TEST          : {len(evaluation_test_index_persisted)}") # Mostrar tamaño TEST
print(f"Scaler                  : {type(evaluation_scaler_persisted).__name__}") # Mostrar scaler
print("Diagnóstico              : VALIDADO") # Confirmar diagnóstico

print("\nBLOQUE 8.1. RECUPERACIÓN DEL TEST INDEX OFICIAL") # Mostrar encabezado
if evaluation_block_8_status != "DIAGNOSTIC_VALIDATED":
    raise RuntimeError(
        "El Bloque 8 no puede continuar porque el diagnóstico no fue validado."
    ) # Impedir continuidad después de un diagnóstico fallido

evaluation_block_8_stage = "TEST_INDEX" # Registrar etapa actual

if not isinstance(
    evaluation_test_index_persisted,
    np.ndarray
):
    raise TypeError(
        "evaluation_test_index_persisted debe ser un ndarray."
    ) # Validar tipo

if evaluation_test_index_persisted.ndim != 1:
    raise ValueError(
        "evaluation_test_index_persisted debe ser un vector unidimensional."
    ) # Validar dimensionalidad

if len(
    evaluation_test_index_persisted
) != 3:
    raise RuntimeError(
        "El conjunto TEST oficial debe contener exactamente 3 índices."
    ) # Validar tamaño

if not np.issubdtype(
    evaluation_test_index_persisted.dtype,
    np.integer
):
    raise TypeError(
        "Los índices TEST deben ser enteros."
    ) # Validar tipo numérico

evaluation_test_index = (
    evaluation_test_index_persisted.copy()
) # Crear índice independiente

if len(
    np.unique(evaluation_test_index)
) != len(
    evaluation_test_index
):
    raise RuntimeError(
        "El índice TEST contiene posiciones duplicadas."
    ) # Validar unicidad

if np.any(
    evaluation_test_index < 0
):
    raise ValueError(
        "El índice TEST contiene valores negativos."
    ) # Validar rango inferior

if np.any(
    evaluation_test_index >= len(evaluation_graphs_persisted)
):
    raise IndexError(
        "El índice TEST contiene posiciones fuera del conjunto de GraphData."
    ) # Validar rango superior

if not np.array_equal(
    evaluation_test_index,
    evaluation_test_index_persisted
):
    raise RuntimeError(
        "evaluation_test_index no coincide con el índice TEST persistido."
    ) # Validar integridad

evaluation_block_8_status = "TEST_INDEX_VALIDATED" # Registrar etapa aprobada
evaluation_block_8_stage = "TEST_INDEX_VALIDADO" # Registrar etapa aprobada

print(f"Índices TEST             : {evaluation_test_index.tolist()}") # Mostrar índices
print(f"GraphData seleccionados  : {len(evaluation_test_index)}") # Mostrar cantidad
print("Origen                   : PERSISTIDO") # Confirmar origen
print("Partición TEST           : VALIDADA") # Confirmar partición

print("\nBLOQUE 8.2. RECUPERACIÓN DE LOS GRAPHDATA DE TEST") # Mostrar encabezado
if evaluation_block_8_status != "TEST_INDEX_VALIDATED":
    raise RuntimeError(
        "El Bloque 8.1 no fue validado. No se pueden recuperar GraphData TEST."
    ) # Impedir continuidad

evaluation_block_8_stage = "TEST_GRAPHS" # Registrar etapa actual
evaluation_test_graphs = [
    evaluation_graphs_persisted[
        int(graph_index)
    ]
    for graph_index in evaluation_test_index
] # Recuperar exclusivamente GraphData TEST

if len(
    evaluation_test_graphs
) != len(
    evaluation_test_index
):
    raise RuntimeError(
        "La cantidad de GraphData recuperados no coincide con test_index."
    ) # Validar correspondencia

if len(
    evaluation_test_graphs
) != 3:
    raise RuntimeError(
        "La evaluación TEST debe contener exactamente 3 GraphData."
    ) # Validar tamaño

for test_position, graph in enumerate(
    evaluation_test_graphs,
    start=1
):

    if not isinstance(
        graph,
        Data
    ):
        raise TypeError(
            f"El GraphData TEST {test_position} no es una instancia de Data."
        ) # Validar tipo

    if not hasattr(
        graph,
        "x"
    ):
        raise RuntimeError(
            f"El GraphData TEST {test_position} no contiene x."
        ) # Validar características

    if not hasattr(
        graph,
        "y"
    ):
        raise RuntimeError(
            f"El GraphData TEST {test_position} no contiene y."
        ) # Validar objetivo

    if not hasattr(
        graph,
        "edge_index"
    ):
        raise RuntimeError(
            f"El GraphData TEST {test_position} no contiene edge_index."
        ) # Validar estructura topológica

    if graph.x.ndim != 2:
        raise ValueError(
            f"El GraphData TEST {test_position} presenta x con "
            f"{graph.x.ndim} dimensiones."
        ) # Validar X

    if graph.x.shape[1] != evaluation_input_channels:
        raise RuntimeError(
            f"El GraphData TEST {test_position} presenta "
            f"{graph.x.shape[1]} variables y se esperaban "
            f"{evaluation_input_channels}."
        ) # Validar compatibilidad con modelo

    if graph.x.shape[0] != graph.y.reshape(-1).shape[0]:
        raise RuntimeError(
            f"El GraphData TEST {test_position} presenta diferente "
            "cantidad de nodos y objetivos."
        ) # Validar correspondencia

evaluation_block_8_status = "TEST_GRAPHS_VALIDATED" # Registrar etapa aprobada
evaluation_block_8_stage = "TEST_GRAPHS_VALIDADO" # Registrar etapa aprobada

print(f"Índices TEST             : {evaluation_test_index.tolist()}") # Mostrar índices
print(f"GraphData TEST           : {len(evaluation_test_graphs)}") # Mostrar cantidad
print(f"Variables por GraphData  : {evaluation_test_graphs[0].x.shape[1]}") # Mostrar variables
print("Origen                   : PERSISTIDO") # Confirmar origen
print("GraphData TEST           : VALIDADOS") # Confirmar recuperación

print("\nBLOQUE 8.3. CONSTRUCCIÓN DE X_TEST Y Y_TEST") # Mostrar encabezado
if evaluation_block_8_status != "TEST_GRAPHS_VALIDATED":
    raise RuntimeError(
        "El Bloque 8.2 no fue validado. No se pueden construir X_test y y_test."
    ) # Impedir continuidad

evaluation_block_8_stage = "TEST_DATA" # Registrar etapa actual
evaluation_test_x_parts = [] # Inicializar X
evaluation_test_y_parts = [] # Inicializar y

for test_position, graph in enumerate(
    evaluation_test_graphs,
    start=1
):
    graph_y = graph.y.reshape(
        -1
    ) # Normalizar objetivo

    if not torch.isfinite(
        graph.x.detach().cpu()
    ).all():
        raise RuntimeError(
            f"X del GraphData TEST {test_position} contiene valores no finitos."
        ) # Validar X

    if not torch.isfinite(
        graph_y.detach().cpu()
    ).all():
        raise RuntimeError(
            f"y del GraphData TEST {test_position} contiene valores no finitos."
        ) # Validar y

    evaluation_test_x_parts.append(
        graph.x.detach().clone()
    ) # Recuperar X

    evaluation_test_y_parts.append(
        graph_y.detach().clone()
    ) # Recuperar y

X_test = torch.cat(
    evaluation_test_x_parts,
    dim=0
) # Consolidar X_TEST

evaluation_y_test = torch.cat(
    evaluation_test_y_parts,
    dim=0
) # Consolidar y_TEST

if X_test.ndim != 2:
    raise RuntimeError(
        "X_test debe ser bidimensional."
    ) # Validar estructura

if X_test.shape[1] != evaluation_input_channels:
    raise RuntimeError(
        "X_test no coincide con la dimensión de entrada del Modelo Oficial."
    ) # Validar dimensión

if X_test.shape[0] != evaluation_y_test.shape[0]:
    raise RuntimeError(
        "X_test y evaluation_y_test no contienen la misma cantidad de observaciones."
    ) # Validar alineación

if not torch.isfinite(
    X_test.detach().cpu()
).all():
    raise RuntimeError(
        "X_test contiene valores no finitos."
    ) # Validar estabilidad

if not torch.isfinite(
    evaluation_y_test.detach().cpu()
).all():
    raise RuntimeError(
        "evaluation_y_test contiene valores no finitos."
    ) # Validar estabilidad

evaluation_block_8_status = "TEST_DATA_VALIDATED" # Registrar etapa aprobada
evaluation_block_8_stage = "TEST_DATA_VALIDADO" # Registrar etapa aprobada

print(f"GraphData TEST          : {len(evaluation_test_graphs)}") # Mostrar grafos
print(f"Observaciones X_test    : {X_test.shape[0]}") # Mostrar observaciones
print(f"Variables X_test        : {X_test.shape[1]}") # Mostrar variables
print(f"Observaciones y_test    : {evaluation_y_test.shape[0]}") # Mostrar objetivos
print("Transformación scaler   : NO APLICADA") # Confirmar que no se transforma nuevamente
print("X_test                  : VALIDADO") # Confirmar X
print("evaluation_y_test       : VALIDADO") # Confirmar y

print("\nBLOQUE 8.4. VALIDACIÓN DEL SCALER OFICIAL") # Mostrar encabezado
if evaluation_block_8_status != "TEST_DATA_VALIDATED":
    raise RuntimeError(
        "El Bloque 8.3 no fue validado. No se puede validar el scaler."
    ) # Impedir continuidad

evaluation_block_8_stage = "SCALER" # Registrar etapa actual
if evaluation_scaler_persisted is None:
    raise RuntimeError(
        "El scaler oficial no está disponible."
    ) # Validar existencia

if not hasattr(
    evaluation_scaler_persisted,
    "mean_"
):
    raise TypeError(
        "El scaler oficial no contiene mean_."
    ) # Validar estructura

if not hasattr(
    evaluation_scaler_persisted,
    "scale_"
):
    raise TypeError(
        "El scaler oficial no contiene scale_."
    ) # Validar estructura

if not hasattr(
    evaluation_scaler_persisted,
    "n_features_in_"
):
    raise TypeError(
        "El scaler oficial no contiene n_features_in_."
    ) # Validar dimensionalidad

evaluation_scaler_features = int(
    evaluation_scaler_persisted.n_features_in_
) # Recuperar dimensionalidad

evaluation_scaler_mean = np.asarray(
    evaluation_scaler_persisted.mean_,
    dtype=np.float64
) # Recuperar medias

evaluation_scaler_scale = np.asarray(
    evaluation_scaler_persisted.scale_,
    dtype=np.float64
) # Recuperar escalas

if evaluation_scaler_features != evaluation_input_channels:
    raise RuntimeError(
        f"El scaler contiene {evaluation_scaler_features} variables y "
        f"el Modelo Oficial requiere {evaluation_input_channels}."
    ) # Validar compatibilidad

if evaluation_scaler_mean.ndim != 1:
    raise ValueError(
        "mean_ del scaler debe ser un vector unidimensional."
    ) # Validar estructura

if evaluation_scaler_scale.ndim != 1:
    raise ValueError(
        "scale_ del scaler debe ser un vector unidimensional."
    ) # Validar estructura

if len(
    evaluation_scaler_mean
) != evaluation_scaler_features:
    raise RuntimeError(
        "La cantidad de medias del scaler no coincide con sus variables."
    ) # Validar medias

if len(
    evaluation_scaler_scale
) != evaluation_scaler_features:
    raise RuntimeError(
        "La cantidad de escalas del scaler no coincide con sus variables."
    ) # Validar escalas

if not np.isfinite(
    evaluation_scaler_mean
).all():
    raise RuntimeError(
        "El scaler contiene medias no finitas."
    ) # Validar estabilidad

if not np.isfinite(
    evaluation_scaler_scale
).all():
    raise RuntimeError(
        "El scaler contiene escalas no finitas."
    ) # Validar estabilidad

if np.any(
    evaluation_scaler_scale <= 0
):
    raise RuntimeError(
        "El scaler contiene escalas menores o iguales a cero."
    ) # Validar escalas positivas

if X_test.shape[1] != evaluation_scaler_features:
    raise RuntimeError(
        f"X_test contiene {X_test.shape[1]} variables y el scaler "
        f"espera {evaluation_scaler_features}."
    ) # Validar compatibilidad X_TEST-Scaler

evaluation_block_8_status = "SCALER_VALIDATED" # Registrar etapa aprobada
evaluation_block_8_stage = "SCALER_VALIDADO" # Registrar etapa aprobada

print(f"Tipo de scaler           : {type(evaluation_scaler_persisted).__name__}") # Mostrar tipo
print(f"Variables del scaler     : {evaluation_scaler_features}") # Mostrar variables
print(f"Variables de X_test      : {X_test.shape[1]}") # Mostrar variables
print("Medias del scaler        : VALIDAS") # Confirmar medias
print("Escalas del scaler       : VALIDAS") # Confirmar escalas
print("Transformación aplicada  : NO") # Confirmar que no se transformó X_TEST
print("Compatibilidad           : VALIDADA") # Confirmar compatibilidad
print("Scaler oficial           : VALIDADO") # Confirmar scaler

print("\nBLOQUE 8.5. CIERRE Y VALIDACIÓN DEL CONJUNTO DE PRUEBA") # Mostrar encabezado
if evaluation_block_8_status != "SCALER_VALIDATED":
    evaluation_block_8_status = "ERROR"
    evaluation_test_ready = False
    raise RuntimeError(
        "El Bloque 8 no puede cerrarse porque una de sus etapas anteriores no fue validada."
    ) # Impedir certificación incompleta

if len(
    evaluation_test_index
) != 3:
    evaluation_block_8_status = "ERROR"
    evaluation_test_ready = False
    raise RuntimeError(
        "El conjunto TEST oficial debe contener 3 índices."
    ) # Validar partición

if len(
    evaluation_test_graphs
) != len(
    evaluation_test_index
):
    evaluation_block_8_status = "ERROR"
    evaluation_test_ready = False
    raise RuntimeError(
        "La cantidad de GraphData TEST no coincide con test_index."
    ) # Validar correspondencia

evaluation_test_graph_nodes = [
    int(graph.x.shape[0])
    for graph in evaluation_test_graphs
] # Registrar nodos TEST

evaluation_test_total_nodes = int(
    sum(evaluation_test_graph_nodes)
) # Calcular observaciones TEST

if evaluation_test_total_nodes != X_test.shape[0]:
    evaluation_block_8_status = "ERROR"
    evaluation_test_ready = False
    raise RuntimeError(
        "La cantidad total de nodos TEST no coincide con X_test."
    ) # Validar correspondencia

if evaluation_test_total_nodes != evaluation_y_test.shape[0]:
    evaluation_block_8_status = "ERROR"
    evaluation_test_ready = False
    raise RuntimeError(
        "La cantidad total de nodos TEST no coincide con evaluation_y_test."
    ) # Validar correspondencia

if evaluation_scaler_persisted.n_features_in_ != evaluation_input_channels:
    evaluation_block_8_status = "ERROR"
    evaluation_test_ready = False
    raise RuntimeError(
        "El scaler oficial no es compatible con el Modelo Oficial."
    ) # Validar scaler

evaluation_test_ready = True # Declarar TEST listo solamente después de todas las validaciones
evaluation_block_8_status = "VALIDATED" # Registrar estado final aprobado
evaluation_block_8_stage = "COMPLETADO" # Registrar cierre

print(f"Índices TEST             : {evaluation_test_index.tolist()}") # Mostrar partición
print(f"GraphData TEST           : {len(evaluation_test_graphs)}") # Mostrar grafos
print(f"Nodos TEST               : {evaluation_test_total_nodes}") # Mostrar observaciones
print(f"Variables X_test         : {X_test.shape[1]}") # Mostrar variables
print(f"Observaciones y_test     : {evaluation_y_test.shape[0]}") # Mostrar objetivos
print(f"Scaler                   : {type(evaluation_scaler_persisted).__name__}") # Mostrar scaler
print("Partición TEST           : VALIDADA") # Confirmar partición
print("X_test                   : VALIDADO") # Confirmar X
print("y_test                   : VALIDADO") # Confirmar y
print("Scaler                   : VALIDADO") # Confirmar scaler
print("Evaluación final         : EXCLUSIVAMENTE SOBRE TEST") # Declarar conjunto
print("evaluation_test_graphs   : LISTO") # Confirmar producto
print("evaluation_test_index    : LISTO") # Confirmar producto
print("evaluation_y_test        : LISTO") # Confirmar producto
print("BLOQUE 8                 : VALIDATED") # Confirmar certificación

# BLOQUE 9. DIAGNÓSTICO PREVIO PARA LA GENERACIÓN DE PREDICCIONES INDEPENDIENTES
# Objetivo: Verificar que el modelo independiente y los GraphData TEST necesarios para generar predicciones están disponibles.
# Entradas: evaluation_graphsage y evaluation_test_graphs.
# Producto: Entorno diagnóstico preparado para la inferencia independiente.
# Pregunta científica: ¿La generación de predicciones puede realizarse exclusivamente mediante evaluation_graphsage sobre el conjunto TEST validado?

print("\nBLOQUE 9.0. DIAGNÓSTICO PREVIO PARA LA GENERACIÓN DE PREDICCIONES INDEPENDIENTES") # Mostrar encabezado
evaluation_block_9_status = "ERROR" # Inicializar estado en ERROR hasta completar todas las validaciones
evaluation_block_9_stage = "DIAGNOSTICO" # Registrar etapa actual
if evaluation_block_8_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 8 no está VALIDATED. Estado actual: {evaluation_block_8_status}"
    ) # Impedir inferencia sobre TEST no validado

if evaluation_test_ready is not True:
    raise RuntimeError(
        "evaluation_test_ready no está establecido en True."
    ) # Impedir inferencia sobre TEST no preparado

if not isinstance(
    evaluation_graphsage,
    nn.Module
):
    raise TypeError(
        "evaluation_graphsage debe ser una instancia de nn.Module."
    ) # Validar modelo independiente

if not isinstance(
    evaluation_test_graphs,
    (list, tuple)
):
    raise TypeError(
        "evaluation_test_graphs debe ser una lista o tupla."
    ) # Validar GraphData TEST

if len(
    evaluation_test_graphs
) == 0:
    raise RuntimeError(
        "evaluation_test_graphs está vacío."
    ) # Validar disponibilidad TEST

if not isinstance(
    evaluation_test_index,
    np.ndarray
):
    raise TypeError(
        "evaluation_test_index debe ser un ndarray."
    ) # Validar índice TEST

if len(
    evaluation_test_graphs
) != len(
    evaluation_test_index
):
    raise RuntimeError(
        "La cantidad de GraphData TEST no coincide con evaluation_test_index."
    ) # Validar correspondencia TEST

if not isinstance(
    evaluation_y_test,
    torch.Tensor
):
    raise TypeError(
        "evaluation_y_test debe ser un tensor de PyTorch."
    ) # Validar objetivo TEST

if evaluation_y_test.ndim != 1:
    raise RuntimeError(
        "evaluation_y_test debe ser un vector unidimensional."
    ) # Validar estructura del objetivo

if evaluation_y_test.shape[0] == 0:
    raise RuntimeError(
        "evaluation_y_test está vacío."
    ) # Validar disponibilidad del objetivo

if evaluation_graphsage.training:
    raise RuntimeError(
        "evaluation_graphsage debe encontrarse en modo evaluación."
    ) # Validar modo inferencia

evaluation_test_node_counts = [] # Inicializar conteo de nodos TEST

for test_position, graph in enumerate(
    evaluation_test_graphs,
    start=1
):

    if not isinstance(
        graph,
        Data
    ):
        raise TypeError(
            f"El GraphData TEST {test_position} no es una instancia de Data."
        ) # Validar tipo GraphData

    if not hasattr(
        graph,
        "x"
    ):
        raise RuntimeError(
            f"El GraphData TEST {test_position} no contiene x."
        ) # Validar características

    if not hasattr(
        graph,
        "edge_index"
    ):
        raise RuntimeError(
            f"El GraphData TEST {test_position} no contiene edge_index."
        ) # Validar topología

    if not hasattr(
        graph,
        "y"
    ):
        raise RuntimeError(
            f"El GraphData TEST {test_position} no contiene y."
        ) # Validar objetivo

    if graph.x.ndim != 2:
        raise ValueError(
            f"El GraphData TEST {test_position} presenta x con "
            f"{graph.x.ndim} dimensiones."
        ) # Validar dimensionalidad de entrada

    if graph.x.shape[1] != evaluation_input_channels:
        raise RuntimeError(
            f"El GraphData TEST {test_position} presenta "
            f"{graph.x.shape[1]} variables y se esperaban "
            f"{evaluation_input_channels}."
        ) # Validar compatibilidad con el modelo

    if graph.edge_index.ndim != 2:
        raise ValueError(
            f"El edge_index del GraphData TEST {test_position} "
            "debe ser bidimensional."
        ) # Validar estructura topológica

    if graph.edge_index.shape[0] != 2:
        raise ValueError(
            f"El edge_index del GraphData TEST {test_position} "
            "debe presentar forma [2, num_edges]."
        ) # Validar formato topológico

    graph_y = graph.y.reshape(
        -1
    ) # Normalizar objetivo del GraphData

    if graph.x.shape[0] != graph_y.shape[0]:
        raise RuntimeError(
            f"El GraphData TEST {test_position} presenta diferente "
            "cantidad de nodos y objetivos."
        ) # Validar correspondencia nodo-target

    evaluation_test_node_counts.append(
        int(graph.x.shape[0])
    ) # Registrar nodos del GraphData TEST

evaluation_test_total_nodes = int(
    sum(evaluation_test_node_counts)
) # Calcular observaciones TEST

if evaluation_test_total_nodes != evaluation_y_test.shape[0]:
    raise RuntimeError(
        f"Los GraphData TEST contienen {evaluation_test_total_nodes} nodos y "
        f"evaluation_y_test contiene {evaluation_y_test.shape[0]} observaciones."
    ) # Validar correspondencia global TEST

if not torch.isfinite(
    evaluation_y_test.detach().cpu()
).all():
    raise RuntimeError(
        "evaluation_y_test contiene valores no finitos."
    ) # Validar estabilidad del objetivo

evaluation_model_parameter_device = next(
    evaluation_graphsage.parameters()
).device # Recuperar dispositivo del modelo

if evaluation_model_parameter_device != evaluation_device:
    raise RuntimeError(
        f"evaluation_graphsage se encuentra en {evaluation_model_parameter_device} "
        f"y se esperaba {evaluation_device}."
    ) # Validar dispositivo

evaluation_block_9_status = "DIAGNOSTIC_VALIDATED" # Registrar diagnóstico aprobado
evaluation_block_9_stage = "DIAGNOSTICO_VALIDADO" # Registrar etapa aprobada

print(f"Estado Bloque 8         : {evaluation_block_8_status}") # Mostrar dependencia
print(f"TEST preparado          : {evaluation_test_ready}") # Mostrar disponibilidad TEST
print(f"Modelo de inferencia    : {type(evaluation_graphsage).__name__}") # Mostrar modelo
print(f"Dispositivo              : {evaluation_model_parameter_device}") # Mostrar dispositivo
print(f"GraphData TEST           : {len(evaluation_test_graphs)}") # Mostrar cantidad
print(f"Índices TEST             : {evaluation_test_index.tolist()}") # Mostrar partición
print(f"Nodos TEST               : {evaluation_test_total_nodes}") # Mostrar observaciones
print(f"Variables de entrada     : {evaluation_input_channels}") # Mostrar variables
print(f"Observaciones y_test     : {evaluation_y_test.shape[0]}") # Mostrar objetivos
print(f"Modo evaluación          : {not evaluation_graphsage.training}") # Mostrar modo
print("Fuente de inferencia     : evaluation_graphsage") # Confirmar fuente
print("Modelo Benchmark         : NO UTILIZADO") # Confirmar aislamiento
print("Modelo entrenamiento     : NO UTILIZADO") # Confirmar aislamiento
print("Diagnóstico Bloque 9.0   : VALIDATED") # Confirmar diagnóstico

print("\nBLOQUE 9.1. PREPARACIÓN DEL MODELO PARA INFERENCIA") # Mostrar encabezado
if evaluation_block_9_status != "DIAGNOSTIC_VALIDATED":
    raise RuntimeError(
        "El Bloque 9.0 no fue validado. No se puede preparar el modelo para inferencia."
    ) # Impedir continuidad

evaluation_block_9_stage = "MODEL_PREPARATION" # Registrar etapa actual
if not isinstance(
    evaluation_graphsage,
    nn.Module
):
    raise TypeError(
        "evaluation_graphsage debe ser una instancia de nn.Module."
    ) # Validar tipo del modelo

if evaluation_device.type != "cpu":
    raise RuntimeError(
        f"El dispositivo de evaluación debe ser CPU y actualmente es {evaluation_device}."
    ) # Validar dispositivo oficial

evaluation_graphsage = evaluation_graphsage.to(
    evaluation_device
) # Transferir modelo a CPU

evaluation_graphsage.eval() # Establecer modo evaluación
for parameter_name, parameter in evaluation_graphsage.named_parameters():
    if parameter.device.type != "cpu":
        raise RuntimeError(
            f"El parámetro '{parameter_name}' no se encuentra en CPU."
        ) # Validar dispositivo de parámetros

    if not torch.isfinite(
        parameter.detach().cpu()
    ).all():
        raise RuntimeError(
            f"El parámetro '{parameter_name}' contiene valores no finitos."
        ) # Validar estabilidad de parámetros

evaluation_parameter_count_9_1 = int(
    sum(
        parameter.numel()
        for parameter in evaluation_graphsage.parameters()
    )
) # Calcular parámetros del modelo preparado

if evaluation_parameter_count_9_1 != 12993:
    raise RuntimeError(
        f"El modelo presenta {evaluation_parameter_count_9_1} parámetros "
        "y se esperaban 12993."
    ) # Validar arquitectura persistida

evaluation_model_device_9_1 = next(
    evaluation_graphsage.parameters()
).device # Recuperar dispositivo final

if evaluation_model_device_9_1 != evaluation_device:
    raise RuntimeError(
        f"El modelo se encuentra en {evaluation_model_device_9_1} "
        f"y se esperaba {evaluation_device}."
    ) # Validar dispositivo final

if evaluation_graphsage.training:
    raise RuntimeError(
        "evaluation_graphsage no se encuentra en modo evaluación."
    ) # Validar modo eval

evaluation_block_9_status = "MODEL_PREPARED" # Registrar preparación aprobada
evaluation_block_9_stage = "MODEL_PREPARED" # Registrar etapa aprobada

print(f"Clase del modelo         : {type(evaluation_graphsage).__name__}") # Mostrar clase
print(f"Dispositivo              : {evaluation_model_device_9_1}") # Mostrar dispositivo
print(f"Modo evaluación          : {not evaluation_graphsage.training}") # Mostrar modo
print(f"Parámetros totales       : {evaluation_parameter_count_9_1}") # Mostrar parámetros
print("Parámetros en CPU        : VALIDADO") # Confirmar parámetros
print("Valores de parámetros   : FINITOS") # Confirmar estabilidad
print("Modelo para inferencia   : PREPARADO") # Confirmar preparación
print("BLOQUE 9.1               : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 9.2. INFERENCIA INDEPENDIENTE SOBRE TEST") # Mostrar encabezado
if evaluation_block_9_status != "MODEL_PREPARED":
    raise RuntimeError(
        "El Bloque 9.1 no fue validado. No se puede iniciar la inferencia."
    ) # Impedir continuidad

evaluation_block_9_stage = "INFERENCE" # Registrar etapa actual

if not isinstance(
    evaluation_graphsage,
    nn.Module
):
    raise TypeError(
        "evaluation_graphsage debe ser una instancia de nn.Module."
    ) # Validar modelo

if evaluation_graphsage.training:
    raise RuntimeError(
        "evaluation_graphsage debe encontrarse en modo evaluación."
    ) # Validar modo inferencia

if not isinstance(
    evaluation_test_graphs,
    (list, tuple)
):
    raise TypeError(
        "evaluation_test_graphs debe ser una lista o tupla."
    ) # Validar TEST

if len(
    evaluation_test_graphs
) != len(
    evaluation_test_index
):
    raise RuntimeError(
        "La cantidad de GraphData TEST no coincide con evaluation_test_index."
    ) # Validar correspondencia TEST

evaluation_predictions = [] # Inicializar predicciones TEST
evaluation_inference_node_counts = [] # Inicializar conteo de nodos inferidos
evaluation_inference_start = time.perf_counter() # Iniciar medición de tiempo

with torch.no_grad():
    for test_position, graph in enumerate(
        evaluation_test_graphs,
        start=1
    ):

        graph_x = graph.x.to(
            evaluation_device
        ) # Transferir características al dispositivo de evaluación

        graph_edge_index = graph.edge_index.to(
            evaluation_device
        ) # Transferir topología al dispositivo de evaluación

        prediction = evaluation_graphsage(
            graph_x,
            graph_edge_index
        ) # Generar predicción exclusivamente con evaluation_graphsage

        if not isinstance(
            prediction,
            torch.Tensor
        ):
            raise TypeError(
                f"La predicción del GraphData TEST {test_position} "
                "no es un tensor de PyTorch."
            ) # Validar salida

        prediction = prediction.reshape(
            -1
        ) # Normalizar salida

        if prediction.shape[0] != graph_x.shape[0]:
            raise RuntimeError(
                f"La predicción del GraphData TEST {test_position} contiene "
                f"{prediction.shape[0]} observaciones y se esperaban "
                f"{graph_x.shape[0]}."
            ) # Validar correspondencia nodo-predicción

        if not torch.isfinite(
            prediction.detach().cpu()
        ).all():
            raise RuntimeError(
                f"La predicción del GraphData TEST {test_position} "
                "contiene valores no finitos."
            ) # Validar estabilidad numérica

        evaluation_predictions.append(
            prediction.detach().cpu()
        ) # Almacenar predicción independiente

        evaluation_inference_node_counts.append(
            int(graph_x.shape[0])
        ) # Registrar nodos procesados

evaluation_inference_end = time.perf_counter() # Finalizar medición de tiempo
inference_time = float(
    evaluation_inference_end
    - evaluation_inference_start
) # Calcular tiempo total de inferencia

if inference_time < 0:
    raise RuntimeError(
        "El tiempo de inferencia no puede ser negativo."
    ) # Validar medición

if len(
    evaluation_predictions
) != len(
    evaluation_test_graphs
):
    raise RuntimeError(
        "La cantidad de predicciones no coincide con la cantidad de GraphData TEST."
    ) # Validar cobertura

y_pred = torch.cat(
    evaluation_predictions,
    dim=0
) # Consolidar predicciones TEST

evaluation_total_inference_nodes = int(
    sum(evaluation_inference_node_counts)
) # Calcular observaciones procesadas

if y_pred.shape[0] != evaluation_total_inference_nodes:
    raise RuntimeError(
        "La cantidad de predicciones no coincide con los nodos procesados."
    ) # Validar cobertura de predicciones

if not torch.isfinite(
    y_pred.detach().cpu()
).all():
    raise RuntimeError(
        "y_pred contiene valores no finitos."
    ) # Validar estabilidad final

evaluation_block_9_status = "INFERENCE_VALIDATED" # Registrar inferencia aprobada
evaluation_block_9_stage = "INFERENCE_VALIDATED" # Registrar etapa aprobada

print(f"GraphData evaluados     : {len(evaluation_test_graphs)}") # Mostrar grafos
print(f"Índices TEST             : {evaluation_test_index.tolist()}") # Mostrar partición
print(f"Nodos procesados         : {evaluation_total_inference_nodes}") # Mostrar observaciones
print(f"Predicciones generadas   : {y_pred.shape[0]}") # Mostrar predicciones
print(f"Tiempo de inferencia     : {inference_time:.6f} segundos") # Mostrar tiempo
print("torch.no_grad()          : ACTIVADO") # Confirmar ausencia de gradientes
print("Modelo utilizado         : evaluation_graphsage") # Confirmar modelo
print("Modelo Benchmark         : NO UTILIZADO") # Confirmar aislamiento
print("Modelo entrenamiento     : NO UTILIZADO") # Confirmar aislamiento
print("y_pred                   : GENERADO") # Confirmar producto
print("BLOQUE 9.2               : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 9.3. VALIDACIÓN NUMÉRICA DE LAS PREDICCIONES") # Mostrar encabezado
if evaluation_block_9_status != "INFERENCE_VALIDATED":
    raise RuntimeError(
        "El Bloque 9.2 no fue validado. No se pueden validar las predicciones."
    ) # Impedir continuidad

evaluation_block_9_stage = "PREDICTION_VALIDATION" # Registrar etapa actual
if not isinstance(
    y_pred,
    torch.Tensor
):
    raise TypeError(
        "y_pred debe ser un tensor de PyTorch."
    ) # Validar tipo de predicción

if not isinstance(
    evaluation_y_test,
    torch.Tensor
):
    raise TypeError(
        "evaluation_y_test debe ser un tensor de PyTorch."
    ) # Validar tipo del target

if y_pred.ndim != 1:
    raise RuntimeError(
        f"y_pred debe ser un vector unidimensional y presenta {y_pred.ndim} dimensiones."
    ) # Validar dimensionalidad de predicción

if evaluation_y_test.ndim != 1:
    raise RuntimeError(
        f"evaluation_y_test debe ser un vector unidimensional y presenta {evaluation_y_test.ndim} dimensiones."
    ) # Validar dimensionalidad del target

evaluation_test_node_counts = [
    int(graph.x.shape[0])
    for graph in evaluation_test_graphs
] # Recuperar cantidad de nodos por GraphData TEST

evaluation_test_total_nodes = int(
    sum(evaluation_test_node_counts)
) # Calcular cantidad total de nodos TEST

if len(
    evaluation_test_graphs
) != len(
    evaluation_test_index
):
    raise RuntimeError(
        "La cantidad de GraphData TEST no coincide con evaluation_test_index."
    ) # Validar cobertura TEST

if y_pred.shape[0] != evaluation_test_total_nodes:
    raise RuntimeError(
        f"y_pred contiene {y_pred.shape[0]} observaciones y TEST contiene "
        f"{evaluation_test_total_nodes} nodos."
    ) # Validar cobertura de nodos

if evaluation_y_test.shape[0] != evaluation_test_total_nodes:
    raise RuntimeError(
        f"evaluation_y_test contiene {evaluation_y_test.shape[0]} observaciones y TEST contiene "
        f"{evaluation_test_total_nodes} nodos."
    ) # Validar cobertura del target

if y_pred.shape[0] != evaluation_y_test.shape[0]:
    raise RuntimeError(
        f"y_pred contiene {y_pred.shape[0]} observaciones y evaluation_y_test contiene "
        f"{evaluation_y_test.shape[0]} observaciones."
    ) # Validar alineación predicción-target

if y_pred.shape[0] == 0:
    raise RuntimeError(
        "y_pred está vacío."
    ) # Validar disponibilidad

if not torch.isfinite(
    y_pred.detach().cpu()
).all():
    raise RuntimeError(
        "y_pred contiene valores no finitos."
    ) # Validar estabilidad de predicciones

if not torch.isfinite(
    evaluation_y_test.detach().cpu()
).all():
    raise RuntimeError(
        "evaluation_y_test contiene valores no finitos."
    ) # Validar estabilidad del target

if not np.isfinite(
    inference_time
):
    raise RuntimeError(
        "inference_time contiene un valor no finito."
    ) # Validar tiempo

if inference_time < 0:
    raise RuntimeError(
        "inference_time no puede ser negativo."
    ) # Validar tiempo

evaluation_prediction_min = float(
    y_pred.min().item()
) # Calcular mínimo de predicciones

evaluation_prediction_max = float(
    y_pred.max().item()
) # Calcular máximo de predicciones

evaluation_prediction_mean = float(
    y_pred.mean().item()
) # Calcular media de predicciones

evaluation_target_min = float(
    evaluation_y_test.min().item()
) # Calcular mínimo del target

evaluation_target_max = float(
    evaluation_y_test.max().item()
) # Calcular máximo del target

evaluation_target_mean = float(
    evaluation_y_test.mean().item()
) # Calcular media del target

if not np.isfinite(
    evaluation_prediction_min
) or not np.isfinite(
    evaluation_prediction_max
) or not np.isfinite(
    evaluation_prediction_mean
):
    raise RuntimeError(
        "Los estadísticos de y_pred contienen valores no finitos."
    ) # Validar estadísticos de predicción

if not np.isfinite(
    evaluation_target_min
) or not np.isfinite(
    evaluation_target_max
) or not np.isfinite(
    evaluation_target_mean
):
    raise RuntimeError(
        "Los estadísticos de evaluation_y_test contienen valores no finitos."
    ) # Validar estadísticos del target

evaluation_block_9_status = "PREDICTIONS_VALIDATED" # Registrar validación aprobada
evaluation_block_9_stage = "PREDICTIONS_VALIDATED" # Registrar etapa aprobada

print(f"Índices TEST             : {evaluation_test_index.tolist()}") # Mostrar partición
print(f"GraphData TEST           : {len(evaluation_test_graphs)}") # Mostrar grafos
print(f"Nodos TEST               : {evaluation_test_total_nodes}") # Mostrar nodos
print(f"Dimensión y_pred         : {tuple(y_pred.shape)}") # Mostrar dimensión
print(f"Dimensión y_test         : {tuple(evaluation_y_test.shape)}") # Mostrar dimensión target
print(f"Predicciones             : {y_pred.shape[0]}") # Mostrar cantidad predicciones
print(f"Targets                  : {evaluation_y_test.shape[0]}") # Mostrar cantidad targets
print(f"Mínimo y_pred            : {evaluation_prediction_min:.12e}") # Mostrar mínimo
print(f"Máximo y_pred            : {evaluation_prediction_max:.12e}") # Mostrar máximo
print(f"Media y_pred             : {evaluation_prediction_mean:.12e}") # Mostrar media
print(f"Mínimo y_test            : {evaluation_target_min:.12e}") # Mostrar mínimo target
print(f"Máximo y_test            : {evaluation_target_max:.12e}") # Mostrar máximo target
print(f"Media y_test             : {evaluation_target_mean:.12e}") # Mostrar media target
print(f"Tiempo inferencia        : {inference_time:.6f} segundos") # Mostrar tiempo
print("Dimensiones              : VALIDADAS") # Confirmar dimensiones
print("Cantidad de nodos        : VALIDADA") # Confirmar nodos
print("Correspondencia target   : VALIDADA") # Confirmar alineación
print("Finitud numérica         : VALIDADA") # Confirmar estabilidad
print("Tiempo de inferencia     : VALIDADO") # Confirmar tiempo
print("BLOQUE 9.3               : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 9.4. CONSOLIDACIÓN DE LA INFERENCIA INDEPENDIENTE") # Mostrar encabezado
if evaluation_block_9_status != "PREDICTIONS_VALIDATED":
    raise RuntimeError(
        "El Bloque 9.3 no fue validado. No se puede consolidar la inferencia."
    ) # Impedir consolidación incompleta

evaluation_block_9_stage = "CONSOLIDATION" # Registrar etapa actual

if not isinstance(
    joblib_model_code,
    str
) or not joblib_model_code.strip():
    raise RuntimeError(
        "El código del Modelo Oficial no está disponible."
    ) # Validar identidad

if not isinstance(
    joblib_model_name,
    str
) or not joblib_model_name.strip():
    raise RuntimeError(
        "El nombre del Modelo Oficial no está disponible."
    ) # Validar identidad

if not isinstance(
    joblib_model_family,
    str
) or not joblib_model_family.strip():
    raise RuntimeError(
        "La familia del Modelo Oficial no está disponible."
    ) # Validar identidad

evaluation_model_code = joblib_model_code # Registrar código oficial
evaluation_model_name = joblib_model_name # Registrar nombre oficial
evaluation_model_family = joblib_model_family # Registrar familia oficial

if evaluation_model_code != "GNN02":
    raise RuntimeError(
        f"El código recuperado es {evaluation_model_code} y se esperaba GNN02."
    ) # Validar código oficial

if evaluation_model_name.strip().lower() != "graphsage":
    raise RuntimeError(
        f"El nombre recuperado es {evaluation_model_name} y se esperaba graphsage."
    ) # Validar nombre oficial

if evaluation_model_family.strip().lower() != "graph_neural_networks":
    raise RuntimeError(
        f"La familia recuperada es {evaluation_model_family} y se esperaba graph_neural_networks."
    ) # Validar familia oficial

if evaluation_block_8_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 8 no está VALIDATED. Estado actual: {evaluation_block_8_status}"
    ) # Validar procedencia TEST

if evaluation_test_ready is not True:
    raise RuntimeError(
        "evaluation_test_ready no está establecido en True."
    ) # Validar disponibilidad TEST

if not isinstance(
    evaluation_test_index,
    np.ndarray
):
    raise TypeError(
        "evaluation_test_index debe ser un ndarray."
    ) # Validar índice TEST

if not isinstance(
    evaluation_test_graphs,
    (list, tuple)
):
    raise TypeError(
        "evaluation_test_graphs debe ser una lista o tupla."
    ) # Validar GraphData TEST

if len(
    evaluation_test_graphs
) != len(
    evaluation_test_index
):
    raise RuntimeError(
        "La cantidad de GraphData TEST no coincide con evaluation_test_index."
    ) # Validar correspondencia TEST

if not isinstance(
    y_pred,
    torch.Tensor
):
    raise TypeError(
        "y_pred debe ser un tensor de PyTorch."
    ) # Validar predicciones

if y_pred.shape[0] != evaluation_y_test.shape[0]:
    raise RuntimeError(
        "La cantidad de predicciones no coincide con la cantidad de targets."
    ) # Validar predicciones

if y_pred.shape[0] != evaluation_test_total_nodes:
    raise RuntimeError(
        "La cantidad de predicciones no coincide con los nodos TEST."
    ) # Validar cobertura TEST

if not torch.isfinite(
    y_pred.detach().cpu()
).all():
    raise RuntimeError(
        "y_pred contiene valores no finitos."
    ) # Validar estabilidad final

if not np.isfinite(
    inference_time
):
    raise RuntimeError(
        "inference_time contiene un valor no finito."
    ) # Validar tiempo

if inference_time < 0:
    raise RuntimeError(
        "inference_time no puede ser negativo."
    ) # Validar tiempo

evaluation_prediction_source = "evaluation_graphsage" # Registrar fuente exclusiva de predicción
evaluation_test_indices_final = evaluation_test_index.tolist() # Registrar índices TEST
evaluation_test_graph_count = int(
    len(evaluation_test_graphs)
) # Registrar cantidad de GraphData TEST
evaluation_prediction_count = int(
    y_pred.shape[0]
) # Registrar cantidad de predicciones
evaluation_target_count = int(
    evaluation_y_test.shape[0]
) # Registrar cantidad de targets
evaluation_inference_nodes_final = int(
    evaluation_test_total_nodes
) # Registrar nodos evaluados
evaluation_inference_time = float(
    inference_time
) # Registrar tiempo de inferencia

if evaluation_prediction_source != "evaluation_graphsage":
    raise RuntimeError(
        "La fuente de predicción no corresponde a evaluation_graphsage."
    ) # Validar fuente

evaluation_independent_prediction_result = {
    "model_code": evaluation_model_code,
    "model_name": evaluation_model_name,
    "family": evaluation_model_family,
    "test_index": evaluation_test_indices_final,
    "test_graphs": evaluation_test_graph_count,
    "test_nodes": evaluation_inference_nodes_final,
    "prediction_count": evaluation_prediction_count,
    "target_count": evaluation_target_count,
    "prediction_source": evaluation_prediction_source,
    "inference_time": evaluation_inference_time,
    "status": "VALIDATED",
} # Construir registro consolidado de inferencia

required_prediction_result_fields = [
    "model_code",
    "model_name",
    "family",
    "test_index",
    "test_graphs",
    "test_nodes",
    "prediction_count",
    "target_count",
    "prediction_source",
    "inference_time",
    "status",
] # Definir campos obligatorios del resultado

missing_prediction_result_fields = [
    field
    for field in required_prediction_result_fields
    if field not in evaluation_independent_prediction_result
] # Identificar campos faltantes

if missing_prediction_result_fields:
    raise RuntimeError(
        f"El resultado consolidado no contiene los campos requeridos: {missing_prediction_result_fields}"
    ) # Validar estructura final

if evaluation_independent_prediction_result["status"] != "VALIDATED":
    raise RuntimeError(
        "El resultado consolidado no presenta estado VALIDATED."
    ) # Validar estado final

evaluation_block_9_status = "VALIDATED" # Registrar aprobación final del Bloque 9
evaluation_block_9_stage = "VALIDATED" # Registrar etapa final aprobada

print(f"Código del modelo        : {evaluation_model_code}") # Mostrar identidad
print(f"Nombre del modelo        : {evaluation_model_name}") # Mostrar identidad
print(f"Familia                  : {evaluation_model_family}") # Mostrar familia
print(f"Índices TEST             : {evaluation_test_indices_final}") # Mostrar partición
print(f"GraphData TEST           : {evaluation_test_graph_count}") # Mostrar grafos
print(f"Nodos TEST               : {evaluation_inference_nodes_final}") # Mostrar nodos
print(f"Predicciones             : {evaluation_prediction_count}") # Mostrar predicciones
print(f"Targets                  : {evaluation_target_count}") # Mostrar targets
print(f"Fuente                   : {evaluation_prediction_source}") # Mostrar fuente
print(f"Tiempo inferencia        : {evaluation_inference_time:.6f} segundos") # Mostrar tiempo
print("Identidad                : VALIDADA") # Confirmar identidad
print("Conjunto TEST            : VALIDADO") # Confirmar TEST
print("Fuente de predicción     : VALIDADA") # Confirmar fuente
print("Predicciones             : VALIDADAS") # Confirmar predicciones
print("Estado final Bloque 9    : VALIDATED") # Confirmar estado final

# BLOQUE 10. DIAGNÓSTICO PREVIO PARA LA AUDITORÍA DE PREDICCIONES
# Objetivo: Verificar la disponibilidad y estructura de los objetos necesarios para auditar las predicciones independientes.
# Entradas: evaluation_y_test, y_pred, evaluation_test_index y evaluation_test_graphs.
# Producto: Diagnóstico preparado para construir PREDICTION_RESULT.
# Pregunta científica: ¿Se encuentran disponibles y correctamente estructurados los valores reales y predichos para realizar una auditoría independiente de TEST?

print("\nBLOQUE 10. DIAGNÓSTICO PREVIO PARA LA AUDITORÍA DE PREDICCIONES") # Mostrar encabezado

print("\nBLOQUE 10.0. DIAGNÓSTICO PREVIO PARA LA AUDITORÍA DE PREDICCIONES") # Mostrar encabezado
evaluation_block_10_status = "ERROR" # Inicializar estado en ERROR hasta completar las validaciones
evaluation_block_10_stage = "DIAGNOSTICO" # Registrar etapa actual

if evaluation_block_9_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 9 no está VALIDATED. Estado actual: {evaluation_block_9_status}"
    ) # Impedir auditoría sobre inferencia no validada

if not isinstance(
    evaluation_y_test,
    torch.Tensor
):
    raise TypeError(
        "evaluation_y_test debe ser un tensor de PyTorch."
    ) # Validar valores reales

if not isinstance(
    y_pred,
    torch.Tensor
):
    raise TypeError(
        "y_pred debe ser un tensor de PyTorch."
    ) # Validar predicciones

if not isinstance(
    evaluation_test_index,
    np.ndarray
):
    raise TypeError(
        "evaluation_test_index debe ser un ndarray."
    ) # Validar partición TEST

if not isinstance(
    evaluation_test_graphs,
    (list, tuple)
):
    raise TypeError(
        "evaluation_test_graphs debe ser una lista o tupla."
    ) # Validar GraphData TEST

if len(
    evaluation_test_graphs
) == 0:
    raise RuntimeError(
        "evaluation_test_graphs está vacío."
    ) # Validar disponibilidad TEST

if len(
    evaluation_test_index
) == 0:
    raise RuntimeError(
        "evaluation_test_index está vacío."
    ) # Validar disponibilidad de índices

y_true = evaluation_y_test.reshape(
    -1
) # Normalizar vector de valores reales

y_pred_audit = y_pred.reshape(
    -1
) # Normalizar vector de predicciones

if y_true.shape[0] == 0:
    raise RuntimeError(
        "y_true está vacío."
    ) # Validar disponibilidad de valores reales

if y_pred_audit.shape[0] == 0:
    raise RuntimeError(
        "y_pred está vacío."
    ) # Validar disponibilidad de predicciones

if len(
    evaluation_test_graphs
) != len(
    evaluation_test_index
):
    raise RuntimeError(
        "La cantidad de GraphData TEST no coincide con evaluation_test_index."
    ) # Validar correspondencia de partición

evaluation_test_node_counts = [] # Inicializar conteo de nodos TEST

for test_position, graph in enumerate(
    evaluation_test_graphs,
    start=1
):

    if not isinstance(
        graph,
        Data
    ):
        raise TypeError(
            f"El GraphData TEST {test_position} no es una instancia de Data."
        ) # Validar tipo GraphData

    if not hasattr(
        graph,
        "x"
    ):
        raise RuntimeError(
            f"El GraphData TEST {test_position} no contiene x."
        ) # Validar características

    if not hasattr(
        graph,
        "y"
    ):
        raise RuntimeError(
            f"El GraphData TEST {test_position} no contiene y."
        ) # Validar objetivo

    evaluation_test_node_counts.append(
        int(graph.x.shape[0])
    ) # Registrar nodos TEST

evaluation_test_total_nodes = int(
    sum(evaluation_test_node_counts)
) # Calcular nodos totales TEST

if y_true.shape[0] != evaluation_test_total_nodes:
    raise RuntimeError(
        f"y_true contiene {y_true.shape[0]} observaciones y TEST contiene "
        f"{evaluation_test_total_nodes} nodos."
    ) # Validar cobertura del target

if y_pred_audit.shape[0] != evaluation_test_total_nodes:
    raise RuntimeError(
        f"y_pred contiene {y_pred_audit.shape[0]} observaciones y TEST contiene "
        f"{evaluation_test_total_nodes} nodos."
    ) # Validar cobertura de predicciones

evaluation_block_10_status = "DIAGNOSTIC_VALIDATED" # Registrar diagnóstico aprobado
evaluation_block_10_stage = "DIAGNOSTICO_VALIDADO" # Registrar etapa aprobada

print(f"Estado Bloque 9         : {evaluation_block_9_status}") # Mostrar dependencia
print(f"GraphData TEST          : {len(evaluation_test_graphs)}") # Mostrar cantidad
print(f"Índices TEST            : {evaluation_test_index.tolist()}") # Mostrar partición
print(f"Nodos por GraphData     : {evaluation_test_node_counts}") # Mostrar distribución
print(f"Nodos TEST              : {evaluation_test_total_nodes}") # Mostrar nodos
print(f"Observaciones y_true    : {len(y_true)}") # Mostrar valores reales
print(f"Observaciones y_pred    : {len(y_pred_audit)}") # Mostrar predicciones
print(f"Tipo y_true             : {type(y_true).__name__}") # Mostrar tipo
print(f"Tipo y_pred             : {type(y_pred_audit).__name__}") # Mostrar tipo
print(f"Shape y_true            : {tuple(y_true.shape)}") # Mostrar dimensión
print(f"Shape y_pred            : {tuple(y_pred_audit.shape)}") # Mostrar dimensión
print("Fuente                  : evaluation_graphsage") # Confirmar procedencia
print("Diagnóstico Bloque 10.0 : VALIDATED") # Confirmar diagnóstico

print("\nBLOQUE 10.1. VALIDACIÓN ESTRUCTURAL DE LAS PREDICCIONES") # Mostrar encabezado
if evaluation_block_10_status != "DIAGNOSTIC_VALIDATED":
    raise RuntimeError(
        "El Bloque 10.0 no fue validado. No se puede validar la estructura de las predicciones."
    ) # Impedir continuidad

evaluation_block_10_stage = "STRUCTURAL_VALIDATION" # Registrar etapa actual
if not isinstance(
    y_true,
    torch.Tensor
):
    raise TypeError(
        "y_true debe ser un tensor de PyTorch."
    ) # Validar tipo de valores reales

if not isinstance(
    y_pred_audit,
    torch.Tensor
):
    raise TypeError(
        "y_pred_audit debe ser un tensor de PyTorch."
    ) # Validar tipo de predicciones

if y_true.ndim != 1:
    raise ValueError(
        f"y_true presenta {y_true.ndim} dimensiones y se esperaba un vector unidimensional."
    ) # Validar dimensionalidad de valores reales

if y_pred_audit.ndim != 1:
    raise ValueError(
        f"y_pred_audit presenta {y_pred_audit.ndim} dimensiones y se esperaba un vector unidimensional."
    ) # Validar dimensionalidad de predicciones

if y_true.shape != y_pred_audit.shape:
    raise RuntimeError(
        f"y_true presenta forma {tuple(y_true.shape)} y y_pred presenta forma {tuple(y_pred_audit.shape)}."
    ) # Validar correspondencia dimensional

if y_true.shape[0] != evaluation_test_total_nodes:
    raise RuntimeError(
        f"y_true contiene {y_true.shape[0]} observaciones y TEST contiene {evaluation_test_total_nodes} nodos."
    ) # Validar cobertura de valores reales

if y_pred_audit.shape[0] != evaluation_test_total_nodes:
    raise RuntimeError(
        f"y_pred contiene {y_pred_audit.shape[0]} observaciones y TEST contiene {evaluation_test_total_nodes} nodos."
    ) # Validar cobertura de predicciones

if y_true.shape[0] != y_pred_audit.shape[0]:
    raise RuntimeError(
        "La cantidad de valores reales no coincide con la cantidad de predicciones."
    ) # Validar cantidad de observaciones

if y_true.dtype != y_pred_audit.dtype:
    raise RuntimeError(
        f"y_true presenta dtype {y_true.dtype} y y_pred presenta dtype {y_pred_audit.dtype}."
    ) # Validar compatibilidad de dtype

if not torch.is_floating_point(
    y_true
):
    raise TypeError(
        f"y_true presenta dtype {y_true.dtype}; se requiere un tipo numérico de punto flotante."
    ) # Validar naturaleza numérica

if not torch.is_floating_point(
    y_pred_audit
):
    raise TypeError(
        f"y_pred presenta dtype {y_pred_audit.dtype}; se requiere un tipo numérico de punto flotante."
    ) # Validar naturaleza numérica

if y_true.shape[0] == 0:
    raise RuntimeError(
        "y_true no puede estar vacío."
    ) # Validar disponibilidad de observaciones

if y_pred_audit.shape[0] == 0:
    raise RuntimeError(
        "y_pred no puede estar vacío."
    ) # Validar disponibilidad de predicciones

evaluation_y_true_dtype = str(
    y_true.dtype
) # Registrar dtype de valores reales

evaluation_y_pred_dtype = str(
    y_pred_audit.dtype
) # Registrar dtype de predicciones

evaluation_prediction_observations = int(
    y_pred_audit.shape[0]
) # Registrar cantidad de predicciones

evaluation_target_observations = int(
    y_true.shape[0]
) # Registrar cantidad de valores reales

evaluation_block_10_status = "STRUCTURE_VALIDATED" # Registrar validación estructural aprobada
evaluation_block_10_stage = "STRUCTURE_VALIDATED" # Registrar etapa aprobada

print(f"Tipo y_true             : {type(y_true).__name__}") # Mostrar tipo real
print(f"Tipo y_pred             : {type(y_pred_audit).__name__}") # Mostrar tipo predicción
print(f"Shape y_true            : {tuple(y_true.shape)}") # Mostrar dimensión real
print(f"Shape y_pred            : {tuple(y_pred_audit.shape)}") # Mostrar dimensión predicción
print(f"Dtype y_true            : {evaluation_y_true_dtype}") # Mostrar dtype real
print(f"Dtype y_pred            : {evaluation_y_pred_dtype}") # Mostrar dtype predicción
print(f"Observaciones y_true    : {evaluation_target_observations}") # Mostrar cantidad real
print(f"Observaciones y_pred    : {evaluation_prediction_observations}") # Mostrar cantidad predicción
print(f"Nodos TEST              : {evaluation_test_total_nodes}") # Mostrar nodos TEST
print("Tipo                    : VALIDADO") # Confirmar tipo
print("Dimensionalidad         : VALIDADA") # Confirmar dimensionalidad
print("Correspondencia shape   : VALIDADA") # Confirmar correspondencia dimensional
print("Cantidad observaciones  : VALIDADA") # Confirmar cantidad
print("Compatibilidad dtype    : VALIDADA") # Confirmar dtype
print("BLOQUE 10.1             : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 10.2. AUDITORÍA NUMÉRICA DE LAS PREDICCIONES") # Mostrar encabezado
if evaluation_block_10_status != "STRUCTURE_VALIDATED":
    raise RuntimeError(
        "El Bloque 10.1 no fue validado. No se puede realizar la auditoría numérica."
    ) # Impedir continuidad

evaluation_block_10_stage = "NUMERICAL_AUDIT" # Registrar etapa actual
y_true_cpu = y_true.detach().cpu() # Copiar valores reales a CPU para auditoría
y_pred_cpu = y_pred_audit.detach().cpu() # Copiar predicciones a CPU para auditoría

if not torch.isfinite(
    y_true_cpu
).all():
    raise RuntimeError(
        "y_true contiene valores NaN o infinitos."
    ) # Validar finitud de valores reales

if not torch.isfinite(
    y_pred_cpu
).all():
    raise RuntimeError(
        "y_pred contiene valores NaN o infinitos."
    ) # Validar finitud de predicciones

evaluation_y_true_nan_count = int(
    torch.isnan(y_true_cpu).sum().item()
) # Contar valores NaN reales

evaluation_y_pred_nan_count = int(
    torch.isnan(y_pred_cpu).sum().item()
) # Contar valores NaN predichos

evaluation_y_true_inf_count = int(
    torch.isinf(y_true_cpu).sum().item()
) # Contar infinitos reales

evaluation_y_pred_inf_count = int(
    torch.isinf(y_pred_cpu).sum().item()
) # Contar infinitos predichos

if evaluation_y_true_nan_count != 0:
    raise RuntimeError(
        f"y_true contiene {evaluation_y_true_nan_count} valores NaN."
    ) # Validar ausencia de NaN reales

if evaluation_y_pred_nan_count != 0:
    raise RuntimeError(
        f"y_pred contiene {evaluation_y_pred_nan_count} valores NaN."
    ) # Validar ausencia de NaN predichos

if evaluation_y_true_inf_count != 0:
    raise RuntimeError(
        f"y_true contiene {evaluation_y_true_inf_count} valores infinitos."
    ) # Validar ausencia de infinitos reales

if evaluation_y_pred_inf_count != 0:
    raise RuntimeError(
        f"y_pred contiene {evaluation_y_pred_inf_count} valores infinitos."
    ) # Validar ausencia de infinitos predichos

evaluation_y_true_min = float(
    y_true_cpu.min().item()
) # Calcular mínimo real

evaluation_y_true_max = float(
    y_true_cpu.max().item()
) # Calcular máximo real

evaluation_y_true_mean = float(
    y_true_cpu.mean().item()
) # Calcular media real

evaluation_y_pred_min = float(
    y_pred_cpu.min().item()
) # Calcular mínimo predicho

evaluation_y_pred_max = float(
    y_pred_cpu.max().item()
) # Calcular máximo predicho

evaluation_y_pred_mean = float(
    y_pred_cpu.mean().item()
) # Calcular media predicha

if not np.isfinite(
    evaluation_y_true_min
):
    raise RuntimeError(
        "El mínimo de y_true no es finito."
    ) # Validar estadístico real

if not np.isfinite(
    evaluation_y_true_max
):
    raise RuntimeError(
        "El máximo de y_true no es finito."
    ) # Validar estadístico real

if not np.isfinite(
    evaluation_y_true_mean
):
    raise RuntimeError(
        "La media de y_true no es finita."
    ) # Validar estadístico real

if not np.isfinite(
    evaluation_y_pred_min
):
    raise RuntimeError(
        "El mínimo de y_pred no es finito."
    ) # Validar estadístico predicho

if not np.isfinite(
    evaluation_y_pred_max
):
    raise RuntimeError(
        "El máximo de y_pred no es finito."
    ) # Validar estadístico predicho

if not np.isfinite(
    evaluation_y_pred_mean
):
    raise RuntimeError(
        "La media de y_pred no es finita."
    ) # Validar estadístico predicho

evaluation_block_10_status = "NUMERICAL_VALIDATED" # Registrar auditoría numérica aprobada
evaluation_block_10_stage = "NUMERICAL_VALIDATED" # Registrar etapa aprobada

print(f"NaN en y_true          : {evaluation_y_true_nan_count}") # Mostrar NaN reales
print(f"NaN en y_pred          : {evaluation_y_pred_nan_count}") # Mostrar NaN predichos
print(f"Inf en y_true          : {evaluation_y_true_inf_count}") # Mostrar infinitos reales
print(f"Inf en y_pred          : {evaluation_y_pred_inf_count}") # Mostrar infinitos predichos
print(f"Mínimo y_true          : {evaluation_y_true_min:.12e}") # Mostrar mínimo real
print(f"Máximo y_true          : {evaluation_y_true_max:.12e}") # Mostrar máximo real
print(f"Media y_true           : {evaluation_y_true_mean:.12e}") # Mostrar media real
print(f"Mínimo y_pred          : {evaluation_y_pred_min:.12e}") # Mostrar mínimo predicho
print(f"Máximo y_pred          : {evaluation_y_pred_max:.12e}") # Mostrar máximo predicho
print(f"Media y_pred           : {evaluation_y_pred_mean:.12e}") # Mostrar media predicha
print("Finitud y_true         : VALIDADA") # Confirmar finitud real
print("Finitud y_pred         : VALIDADA") # Confirmar finitud predicha
print("Integridad numérica    : VALIDADA") # Confirmar integridad
print("BLOQUE 10.2            : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 10.3. VALIDACIÓN DE ALINEACIÓN ENTRE TEST, TARGET Y PREDICCIONES") # Mostrar encabezado
if evaluation_block_10_status != "NUMERICAL_VALIDATED":
    raise RuntimeError(
        "El Bloque 10.2 no fue validado. No se puede validar la alineación."
    ) # Impedir continuidad

evaluation_block_10_stage = "ALIGNMENT_VALIDATION" # Registrar etapa actual

if len(
    evaluation_test_graphs
) != len(
    evaluation_test_index
):
    raise RuntimeError(
        "La cantidad de GraphData TEST no coincide con evaluation_test_index."
    ) # Validar cantidad de grafos e índices

evaluation_alignment_node_counts = [] # Inicializar nodos por GraphData
evaluation_alignment_offsets = [] # Inicializar offsets de concatenación
evaluation_alignment_offset = 0 # Inicializar posición acumulada

for test_position, graph in enumerate(
    evaluation_test_graphs,
    start=1
):
    if not isinstance(
        graph,
        Data
    ):
        raise TypeError(
            f"El GraphData TEST {test_position} no es una instancia de Data."
        ) # Validar tipo GraphData

    if not hasattr(
        graph,
        "x"
    ):
        raise RuntimeError(
            f"El GraphData TEST {test_position} no contiene x."
        ) # Validar características

    if not hasattr(
        graph,
        "y"
    ):
        raise RuntimeError(
            f"El GraphData TEST {test_position} no contiene y."
        ) # Validar target

    graph_node_count = int(
        graph.x.shape[0]
    ) # Recuperar nodos del GraphData

    graph_target_count = int(
        graph.y.reshape(-1).shape[0]
    ) # Recuperar targets del GraphData

    if graph_node_count != graph_target_count:
        raise RuntimeError(
            f"El GraphData TEST {test_position} contiene "
            f"{graph_node_count} nodos y {graph_target_count} targets."
        ) # Validar nodo-target

    evaluation_alignment_node_counts.append(
        graph_node_count
    ) # Registrar nodos

    evaluation_alignment_offsets.append(
        evaluation_alignment_offset
    ) # Registrar offset inicial

    evaluation_alignment_offset += graph_node_count # Actualizar posición acumulada

evaluation_alignment_total_nodes = int(
    sum(evaluation_alignment_node_counts)
) # Calcular total de nodos

if evaluation_alignment_total_nodes != len(
    y_true
):
    raise RuntimeError(
        f"TEST contiene {evaluation_alignment_total_nodes} observaciones y "
        f"y_true contiene {len(y_true)}."
    ) # Validar cobertura de target

if evaluation_alignment_total_nodes != len(
    y_pred_audit
):
    raise RuntimeError(
        f"TEST contiene {evaluation_alignment_total_nodes} observaciones y "
        f"y_pred contiene {len(y_pred_audit)}."
    ) # Validar cobertura de predicción

evaluation_alignment_end = int(
    evaluation_alignment_offset
) # Registrar posición final

if evaluation_alignment_end != len(
    y_true
):
    raise RuntimeError(
        "El offset final de TEST no coincide con la longitud de y_true."
    ) # Validar offset target

if evaluation_alignment_end != len(
    y_pred_audit
):
    raise RuntimeError(
        "El offset final de TEST no coincide con la longitud de y_pred."
    ) # Validar offset predicción

if y_true.shape != y_pred_audit.shape:
    raise RuntimeError(
        f"y_true presenta forma {tuple(y_true.shape)} y "
        f"y_pred presenta forma {tuple(y_pred_audit.shape)}."
    ) # Validar forma conjunta

if evaluation_alignment_node_counts != [
    int(graph.x.shape[0])
    for graph in evaluation_test_graphs
]:
    raise RuntimeError(
        "La estructura de nodos TEST cambió durante la validación."
    ) # Validar estabilidad de TEST

if evaluation_test_index.tolist() != [
    10,
    11,
    12
]:
    raise RuntimeError(
        f"Los índices TEST recuperados son {evaluation_test_index.tolist()} "
        "y se esperaban [10, 11, 12]."
    ) # Validar partición oficial

evaluation_alignment_status = "VALIDATED" # Registrar alineación aprobada
evaluation_block_10_status = "ALIGNMENT_VALIDATED" # Registrar estado del bloque
evaluation_block_10_stage = "ALIGNMENT_VALIDATED" # Registrar etapa aprobada

print(f"Índices TEST             : {evaluation_test_index.tolist()}") # Mostrar índices
print(f"Nodos por GraphData      : {evaluation_alignment_node_counts}") # Mostrar nodos
print(f"Offsets TEST              : {evaluation_alignment_offsets}") # Mostrar offsets
print(f"Total nodos TEST          : {evaluation_alignment_total_nodes}") # Mostrar total
print(f"Observaciones y_true     : {len(y_true)}") # Mostrar targets
print(f"Observaciones y_pred     : {len(y_pred_audit)}") # Mostrar predicciones
print(f"Shape y_true             : {tuple(y_true.shape)}") # Mostrar forma target
print(f"Shape y_pred             : {tuple(y_pred_audit.shape)}") # Mostrar forma predicción
print("Correspondencia GraphData : VALIDADA") # Confirmar GraphData
print("Correspondencia target    : VALIDADA") # Confirmar target
print("Correspondencia predicción: VALIDADA") # Confirmar predicción
print("Offsets de concatenación  : VALIDADOS") # Confirmar offsets
print("Partición TEST            : VALIDADA") # Confirmar partición
print("BLOQUE 10.3               : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 10.4. CONSTRUCCIÓN DEL RESULTADO CONSOLIDADO DE PREDICCIONES") # Mostrar encabezado
if evaluation_block_10_status != "ALIGNMENT_VALIDATED":
    raise RuntimeError(
        "El Bloque 10.3 no fue validado. No se puede construir PREDICTION_RESULT."
    ) # Impedir construcción sin alineación validada

evaluation_block_10_stage = "RESULT_CONSTRUCTION" # Registrar etapa actual

if evaluation_block_9_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 9 no está VALIDATED. Estado actual: {evaluation_block_9_status}"
    ) # Validar procedencia de inferencia

if evaluation_model_code != "GNN02":
    raise RuntimeError(
        f"El código del modelo es {evaluation_model_code} y se esperaba GNN02."
    ) # Validar identidad oficial

if evaluation_model_name.strip().lower() != "graphsage":
    raise RuntimeError(
        f"El nombre del modelo es {evaluation_model_name} y se esperaba graphsage."
    ) # Validar nombre oficial

if evaluation_model_family.strip().lower() != "graph_neural_networks":
    raise RuntimeError(
        f"La familia del modelo es {evaluation_model_family} y se esperaba graph_neural_networks."
    ) # Validar familia oficial

if evaluation_test_index.tolist() != [
    10,
    11,
    12
]:
    raise RuntimeError(
        f"La partición TEST es {evaluation_test_index.tolist()} y se esperaba [10, 11, 12]."
    ) # Validar partición oficial

if evaluation_prediction_source != "evaluation_graphsage":
    raise RuntimeError(
        f"La fuente de predicción es {evaluation_prediction_source} y se esperaba evaluation_graphsage."
    ) # Validar fuente de inferencia

if not np.isfinite(
    evaluation_inference_time
):
    raise RuntimeError(
        "evaluation_inference_time no es finito."
    ) # Validar tiempo de inferencia

if evaluation_inference_time < 0:
    raise RuntimeError(
        "evaluation_inference_time no puede ser negativo."
    ) # Validar tiempo de inferencia

if len(
    y_true
) != len(
    y_pred_audit
):
    raise RuntimeError(
        "y_true y y_pred no contienen la misma cantidad de observaciones."
    ) # Validar correspondencia

if len(
    y_true
) != evaluation_test_total_nodes:
    raise RuntimeError(
        "La cantidad de valores reales no coincide con los nodos TEST."
    ) # Validar cobertura TEST

if len(
    y_pred_audit
) != evaluation_test_total_nodes:
    raise RuntimeError(
        "La cantidad de predicciones no coincide con los nodos TEST."
    ) # Validar cobertura TEST

evaluation_y_true_cpu = y_true.detach().cpu().clone() # Crear copia independiente de los valores reales
evaluation_y_pred_cpu = y_pred_audit.detach().cpu().clone() # Crear copia independiente de las predicciones

if not torch.isfinite(
    evaluation_y_true_cpu
).all():
    raise RuntimeError(
        "La copia consolidada de y_true contiene valores no finitos."
    ) # Validar integridad del target consolidado

if not torch.isfinite(
    evaluation_y_pred_cpu
).all():
    raise RuntimeError(
        "La copia consolidada de y_pred contiene valores no finitos."
    ) # Validar integridad de predicciones consolidadas

PREDICTION_RESULT = {
    "model_code": evaluation_model_code,
    "model_name": evaluation_model_name,
    "family": evaluation_model_family,
    "test_index": evaluation_test_index.copy(),
    "test_graphs": int(len(evaluation_test_graphs)),
    "test_nodes": int(evaluation_test_total_nodes),
    "y_true": evaluation_y_true_cpu,
    "y_pred": evaluation_y_pred_cpu,
    "n_observations": int(evaluation_y_pred_cpu.shape[0]),
    "prediction_source": evaluation_prediction_source,
    "inference_time": float(evaluation_inference_time),
    "status": "VALIDATED",
} # Construir resultado consolidado de predicciones

if not isinstance(
    PREDICTION_RESULT,
    dict
):
    raise TypeError(
        "PREDICTION_RESULT debe ser un diccionario."
    ) # Validar estructura del resultado

required_prediction_result_fields = [
    "model_code",
    "model_name",
    "family",
    "test_index",
    "test_graphs",
    "test_nodes",
    "y_true",
    "y_pred",
    "n_observations",
    "prediction_source",
    "inference_time",
    "status",
] # Definir campos obligatorios

missing_prediction_result_fields = [
    field
    for field in required_prediction_result_fields
    if field not in PREDICTION_RESULT
] # Identificar campos faltantes

if missing_prediction_result_fields:
    raise RuntimeError(
        f"PREDICTION_RESULT presenta campos faltantes: {missing_prediction_result_fields}"
    ) # Validar cobertura del resultado

if PREDICTION_RESULT["y_true"].shape != PREDICTION_RESULT["y_pred"].shape:
    raise RuntimeError(
        "y_true y y_pred dentro de PREDICTION_RESULT presentan formas diferentes."
    ) # Validar forma consolidada

if PREDICTION_RESULT["n_observations"] != PREDICTION_RESULT["y_true"].shape[0]:
    raise RuntimeError(
        "n_observations no coincide con la cantidad de valores reales."
    ) # Validar cantidad consolidada

if PREDICTION_RESULT["n_observations"] != PREDICTION_RESULT["y_pred"].shape[0]:
    raise RuntimeError(
        "n_observations no coincide con la cantidad de predicciones."
    ) # Validar cantidad consolidada

if PREDICTION_RESULT["n_observations"] != PREDICTION_RESULT["test_nodes"]:
    raise RuntimeError(
        "n_observations no coincide con los nodos TEST."
    ) # Validar cobertura TEST

if PREDICTION_RESULT["test_graphs"] != len(
    PREDICTION_RESULT["test_index"]
):
    raise RuntimeError(
        "La cantidad de GraphData TEST no coincide con la cantidad de índices TEST."
    ) # Validar partición

if PREDICTION_RESULT["status"] != "VALIDATED":
    raise RuntimeError(
        "PREDICTION_RESULT no presenta estado VALIDATED."
    ) # Validar estado

evaluation_block_10_status = "RESULT_CONSTRUCTED" # Registrar construcción aprobada
evaluation_block_10_stage = "RESULT_CONSTRUCTED" # Registrar etapa aprobada

print(f"Código del modelo        : {PREDICTION_RESULT['model_code']}") # Mostrar código
print(f"Nombre del modelo        : {PREDICTION_RESULT['model_name']}") # Mostrar nombre
print(f"Familia                  : {PREDICTION_RESULT['family']}") # Mostrar familia
print(f"Índices TEST             : {PREDICTION_RESULT['test_index'].tolist()}") # Mostrar TEST
print(f"GraphData TEST           : {PREDICTION_RESULT['test_graphs']}") # Mostrar grafos
print(f"Nodos TEST               : {PREDICTION_RESULT['test_nodes']}") # Mostrar nodos
print(f"Observaciones            : {PREDICTION_RESULT['n_observations']}") # Mostrar observaciones
print(f"Shape y_true             : {tuple(PREDICTION_RESULT['y_true'].shape)}") # Mostrar target
print(f"Shape y_pred             : {tuple(PREDICTION_RESULT['y_pred'].shape)}") # Mostrar predicción
print(f"Fuente                   : {PREDICTION_RESULT['prediction_source']}") # Mostrar fuente
print(f"Tiempo inferencia        : {PREDICTION_RESULT['inference_time']:.6f} segundos") # Mostrar tiempo
print(f"Estado                   : {PREDICTION_RESULT['status']}") # Mostrar estado
print("PREDICTION_RESULT       : CONSTRUIDO") # Confirmar construcción
print("BLOQUE 10.4             : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 10.5. CIERRE Y VALIDACIÓN FINAL DEL BLOQUE 10") # Mostrar encabezado
if evaluation_block_10_status != "RESULT_CONSTRUCTED":
    raise RuntimeError(
        "El Bloque 10.4 no fue validado. No se puede cerrar el Bloque 10."
    ) # Impedir cierre incompleto

if evaluation_block_9_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 9 no está VALIDATED. Estado actual: {evaluation_block_9_status}"
    ) # Validar dependencia de inferencia

if not isinstance(
    PREDICTION_RESULT,
    dict
):
    raise TypeError(
        "PREDICTION_RESULT debe ser un diccionario."
    ) # Validar resultado consolidado

required_prediction_result_fields = [
    "model_code",
    "model_name",
    "family",
    "test_index",
    "test_graphs",
    "test_nodes",
    "y_true",
    "y_pred",
    "n_observations",
    "prediction_source",
    "inference_time",
    "status",
] # Definir campos obligatorios

missing_prediction_result_fields = [
    field
    for field in required_prediction_result_fields
    if field not in PREDICTION_RESULT
] # Identificar campos faltantes

if missing_prediction_result_fields:
    raise RuntimeError(
        f"PREDICTION_RESULT presenta campos faltantes: {missing_prediction_result_fields}"
    ) # Validar estructura completa

if PREDICTION_RESULT["model_code"] != evaluation_model_code:
    raise RuntimeError(
        "El código del modelo no coincide con la identidad oficial."
    ) # Validar identidad

if PREDICTION_RESULT["model_name"].strip().lower() != evaluation_model_name.strip().lower():
    raise RuntimeError(
        "El nombre del modelo no coincide con la identidad oficial."
    ) # Validar identidad

if PREDICTION_RESULT["family"].strip().lower() != evaluation_model_family.strip().lower():
    raise RuntimeError(
        "La familia del modelo no coincide con la identidad oficial."
    ) # Validar identidad

if PREDICTION_RESULT["model_code"] != "GNN02":
    raise RuntimeError(
        "PREDICTION_RESULT no corresponde al Modelo Oficial GNN02."
    ) # Validar código oficial

if PREDICTION_RESULT["model_name"].strip().lower() != "graphsage":
    raise RuntimeError(
        "PREDICTION_RESULT no corresponde a GraphSAGE."
    ) # Validar nombre oficial

if PREDICTION_RESULT["family"].strip().lower() != "graph_neural_networks":
    raise RuntimeError(
        "PREDICTION_RESULT no corresponde a la familia GNN."
    ) # Validar familia oficial

if PREDICTION_RESULT["prediction_source"] != "evaluation_graphsage":
    raise RuntimeError(
        "La fuente de predicción no corresponde a evaluation_graphsage."
    ) # Validar fuente independiente

if PREDICTION_RESULT["test_index"].tolist() != evaluation_test_index.tolist():
    raise RuntimeError(
        "Los índices TEST de PREDICTION_RESULT no coinciden con evaluation_test_index."
    ) # Validar partición TEST

if PREDICTION_RESULT["test_graphs"] != len(
    evaluation_test_graphs
):
    raise RuntimeError(
        "La cantidad de GraphData TEST no coincide con PREDICTION_RESULT."
    ) # Validar cantidad de grafos

if PREDICTION_RESULT["test_nodes"] != evaluation_test_total_nodes:
    raise RuntimeError(
        "La cantidad de nodos TEST no coincide con PREDICTION_RESULT."
    ) # Validar nodos

if not isinstance(
    PREDICTION_RESULT["y_true"],
    torch.Tensor
):
    raise TypeError(
        "PREDICTION_RESULT['y_true'] debe ser un tensor."
    ) # Validar target

if not isinstance(
    PREDICTION_RESULT["y_pred"],
    torch.Tensor
):
    raise TypeError(
        "PREDICTION_RESULT['y_pred'] debe ser un tensor."
    ) # Validar predicción

if PREDICTION_RESULT["y_true"].shape != y_true.shape:
    raise RuntimeError(
        "La forma consolidada de y_true no coincide con la forma validada."
    ) # Validar target consolidado

if PREDICTION_RESULT["y_pred"].shape != y_pred_audit.shape:
    raise RuntimeError(
        "La forma consolidada de y_pred no coincide con la forma validada."
    ) # Validar predicción consolidada

if PREDICTION_RESULT["n_observations"] != evaluation_test_total_nodes:
    raise RuntimeError(
        "n_observations no coincide con los nodos TEST."
    ) # Validar observaciones

if PREDICTION_RESULT["n_observations"] != PREDICTION_RESULT["y_true"].shape[0]:
    raise RuntimeError(
        "n_observations no coincide con y_true."
    ) # Validar target

if PREDICTION_RESULT["n_observations"] != PREDICTION_RESULT["y_pred"].shape[0]:
    raise RuntimeError(
        "n_observations no coincide con y_pred."
    ) # Validar predicción

if not torch.equal(
    PREDICTION_RESULT["y_true"],
    y_true.detach().cpu()
):
    raise RuntimeError(
        "El y_true consolidado no coincide con el y_true validado."
    ) # Validar integridad del target

if not torch.equal(
    PREDICTION_RESULT["y_pred"],
    y_pred_audit.detach().cpu()
):
    raise RuntimeError(
        "El y_pred consolidado no coincide con el y_pred validado."
    ) # Validar integridad de la predicción

if not torch.isfinite(
    PREDICTION_RESULT["y_true"]
).all():
    raise RuntimeError(
        "El y_true consolidado contiene valores no finitos."
    ) # Validar finitud final

if not torch.isfinite(
    PREDICTION_RESULT["y_pred"]
).all():
    raise RuntimeError(
        "El y_pred consolidado contiene valores no finitos."
    ) # Validar finitud final

if not np.isfinite(
    PREDICTION_RESULT["inference_time"]
):
    raise RuntimeError(
        "El tiempo de inferencia consolidado no es finito."
    ) # Validar tiempo

if PREDICTION_RESULT["inference_time"] < 0:
    raise RuntimeError(
        "El tiempo de inferencia consolidado no puede ser negativo."
    ) # Validar tiempo

if PREDICTION_RESULT["status"] != "VALIDATED":
    raise RuntimeError(
        "PREDICTION_RESULT no presenta estado VALIDATED."
    ) # Validar estado del resultado

evaluation_block_10_status = "VALIDATED" # Registrar aprobación final
evaluation_block_10_stage = "VALIDATED" # Registrar etapa final aprobada

print(f"Estado Bloque 9         : {evaluation_block_9_status}") # Mostrar dependencia
print(f"Código del modelo       : {PREDICTION_RESULT['model_code']}") # Mostrar código
print(f"Nombre del modelo       : {PREDICTION_RESULT['model_name']}") # Mostrar nombre
print(f"Familia                 : {PREDICTION_RESULT['family']}") # Mostrar familia
print(f"Índices TEST            : {PREDICTION_RESULT['test_index'].tolist()}") # Mostrar partición
print(f"GraphData TEST          : {PREDICTION_RESULT['test_graphs']}") # Mostrar grafos
print(f"Nodos TEST              : {PREDICTION_RESULT['test_nodes']}") # Mostrar nodos
print(f"Observaciones           : {PREDICTION_RESULT['n_observations']}") # Mostrar observaciones
print(f"Shape y_true            : {tuple(PREDICTION_RESULT['y_true'].shape)}") # Mostrar target
print(f"Shape y_pred            : {tuple(PREDICTION_RESULT['y_pred'].shape)}") # Mostrar predicción
print(f"Fuente                  : {PREDICTION_RESULT['prediction_source']}") # Mostrar fuente
print(f"Tiempo inferencia       : {PREDICTION_RESULT['inference_time']:.6f} segundos") # Mostrar tiempo
print(f"Estado resultado        : {PREDICTION_RESULT['status']}") # Mostrar estado
print("Identidad               : VALIDADA") # Confirmar identidad
print("Partición TEST          : VALIDADA") # Confirmar TEST
print("Integridad y_true       : VALIDADA") # Confirmar target
print("Integridad y_pred       : VALIDADA") # Confirmar predicción
print("Finitud numérica        : VALIDADA") # Confirmar finitud
print("Tiempo de inferencia    : VALIDADO") # Confirmar tiempo
print("BLOQUE 10               : VALIDATED") # Confirmar cierre final

# BLOQUE 11. DIAGNÓSTICO PREVIO PARA EL CÁLCULO DE MÉTRICAS DE EVALUACIÓN
# Objetivo: Verificar la disponibilidad de PREDICTION_RESULT y determinar las métricas respaldadas por el contrato actual.
# Entradas: PREDICTION_RESULT y artefactos de configuración de la evaluación.
# Producto: Diagnóstico preparado para calcular las métricas oficiales sobre TEST.
# Pregunta científica: ¿Se encuentran disponibles los valores reales y predichos y existe respaldo contractual para cada métrica de evaluación?

print("\nBLOQUE 11. DIAGNÓSTICO PREVIO PARA EL CÁLCULO DE MÉTRICAS DE EVALUACIÓN") # Mostrar encabezado

print("\nBLOQUE 11.0. DIAGNÓSTICO PREVIO PARA EL CÁLCULO DE MÉTRICAS DE EVALUACIÓN") # Mostrar encabezado
evaluation_block_11_status = "ERROR" # Inicializar estado en ERROR
evaluation_block_11_stage = "DIAGNOSTICO" # Registrar etapa actual

if evaluation_block_10_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 10 no está VALIDATED. Estado actual: {evaluation_block_10_status}"
    ) # Impedir cálculo sobre resultado no validado

if not isinstance(
    PREDICTION_RESULT,
    dict
):
    raise TypeError(
        "PREDICTION_RESULT debe ser un diccionario."
    ) # Validar estructura

if PREDICTION_RESULT.get(
    "status"
) != "VALIDATED":
    raise RuntimeError(
        "PREDICTION_RESULT debe presentar estado VALIDATED."
    ) # Validar estado

required_prediction_fields = [
    "model_code",
    "model_name",
    "family",
    "test_index",
    "test_graphs",
    "test_nodes",
    "y_true",
    "y_pred",
    "n_observations",
    "prediction_source",
    "inference_time",
    "status",
] # Definir campos requeridos

missing_prediction_fields = [
    field
    for field in required_prediction_fields
    if field not in PREDICTION_RESULT
] # Identificar campos faltantes

if missing_prediction_fields:
    raise RuntimeError(
        f"PREDICTION_RESULT presenta campos faltantes: {missing_prediction_fields}"
    ) # Validar cobertura

evaluation_y_true = PREDICTION_RESULT[
    "y_true"
].detach().cpu().reshape(
    -1
).to(
    torch.float64
) # Recuperar valores reales

evaluation_y_pred = PREDICTION_RESULT[
    "y_pred"
].detach().cpu().reshape(
    -1
).to(
    torch.float64
) # Recuperar predicciones

if evaluation_y_true.shape != evaluation_y_pred.shape:
    raise RuntimeError(
        f"y_true presenta forma {tuple(evaluation_y_true.shape)} y "
        f"y_pred presenta forma {tuple(evaluation_y_pred.shape)}."
    ) # Validar dimensiones

if len(
    evaluation_y_true
) == 0:
    raise RuntimeError(
        "No existen observaciones disponibles para la evaluación."
    ) # Validar disponibilidad

if not torch.isfinite(
    evaluation_y_true
).all():
    raise RuntimeError(
        "evaluation_y_true contiene valores no finitos."
    ) # Validar estabilidad

if not torch.isfinite(
    evaluation_y_pred
).all():
    raise RuntimeError(
        "evaluation_y_pred contiene valores no finitos."
    ) # Validar estabilidad

if PREDICTION_RESULT[
    "n_observations"
] != len(
    evaluation_y_true
):
    raise RuntimeError(
        "La cantidad de observaciones registrada no coincide con y_true."
    ) # Validar trazabilidad

if PREDICTION_RESULT[
    "n_observations"
] != len(
    evaluation_y_pred
):
    raise RuntimeError(
        "La cantidad de observaciones registrada no coincide con y_pred."
    ) # Validar trazabilidad

if PREDICTION_RESULT[
    "test_nodes"
] != len(
    evaluation_y_true
):
    raise RuntimeError(
        "La cantidad de nodos TEST no coincide con las observaciones."
    ) # Validar cobertura TEST

if PREDICTION_RESULT[
    "prediction_source"
] != "evaluation_graphsage":
    raise RuntimeError(
        "La fuente de predicción no corresponde a evaluation_graphsage."
    ) # Validar independencia

if PREDICTION_RESULT[
    "test_index"
].tolist() != evaluation_test_index.tolist():
    raise RuntimeError(
        "El test_index de PREDICTION_RESULT no coincide con evaluation_test_index."
    ) # Validar partición

if PREDICTION_RESULT[
    "test_graphs"
] != len(
    evaluation_test_graphs
):
    raise RuntimeError(
        "La cantidad de GraphData TEST no coincide con PREDICTION_RESULT."
    ) # Validar cantidad de grafos

required_evaluation_metrics = [
    "RMSE",
    "MAE",
    "MAPE",
    "R2",
] # Definir métricas oficiales

optional_evaluation_metrics = [
    "MSE",
] # Registrar métrica opcional sin incluirla todavía

evaluation_block_11_status = "DIAGNOSTIC_VALIDATED" # Registrar diagnóstico aprobado
evaluation_block_11_stage = "DIAGNOSTICO_VALIDADO" # Registrar etapa aprobada

print(f"Estado Bloque 10         : {evaluation_block_10_status}") # Mostrar dependencia
print(f"Modelo                   : {PREDICTION_RESULT['model_name']}") # Mostrar modelo
print(f"Código                   : {PREDICTION_RESULT['model_code']}") # Mostrar código
print(f"Familia                  : {PREDICTION_RESULT['family']}") # Mostrar familia
print(f"Índices TEST             : {PREDICTION_RESULT['test_index'].tolist()}") # Mostrar partición
print(f"GraphData TEST           : {PREDICTION_RESULT['test_graphs']}") # Mostrar grafos
print(f"Nodos TEST               : {PREDICTION_RESULT['test_nodes']}") # Mostrar nodos
print(f"Observaciones y_true     : {len(evaluation_y_true)}") # Mostrar observaciones reales
print(f"Observaciones y_pred     : {len(evaluation_y_pred)}") # Mostrar observaciones predichas
print(f"Shape y_true             : {tuple(evaluation_y_true.shape)}") # Mostrar forma real
print(f"Shape y_pred             : {tuple(evaluation_y_pred.shape)}") # Mostrar forma predicha
print(f"Tipo cálculo             : {evaluation_y_true.dtype}") # Mostrar precisión numérica
print(f"Fuente de predicción     : {PREDICTION_RESULT['prediction_source']}") # Mostrar procedencia
print(f"Métricas obligatorias    : {required_evaluation_metrics}") # Mostrar métricas oficiales
print(f"Métricas opcionales      : {optional_evaluation_metrics}") # Mostrar métricas opcionales
print("y_true                   : FINITO") # Confirmar estabilidad
print("y_pred                   : FINITO") # Confirmar estabilidad
print("Correspondencia          : VALIDADA") # Confirmar correspondencia
print("Diagnóstico Bloque 11.0  : VALIDATED") # Confirmar diagnóstico

print("\nBLOQUE 11.1. VALIDACIÓN DEL CONTRATO DE MÉTRICAS") # Mostrar encabezado
if evaluation_block_11_status != "DIAGNOSTIC_VALIDATED":
    raise RuntimeError(
        f"El Bloque 11.0 no fue validado. Estado actual: {evaluation_block_11_status}"
    ) # Impedir continuidad sin diagnóstico validado

evaluation_block_11_stage = "METRIC_CONTRACT" # Registrar etapa actual
contract_required_metrics = [
    "RMSE",
    "MAE",
    "MAPE",
    "R2",
] # Definir métricas obligatorias del contrato

contract_optional_metrics = [
    "MSE",
] # Registrar métrica opcional sin incluirla

if not isinstance(
    required_evaluation_metrics,
    list
):
    raise TypeError(
        "required_evaluation_metrics debe ser una lista."
    ) # Validar estructura

if not isinstance(
    optional_evaluation_metrics,
    list
):
    raise TypeError(
        "optional_evaluation_metrics debe ser una lista."
    ) # Validar estructura

if required_evaluation_metrics != contract_required_metrics:
    raise RuntimeError(
        "Las métricas obligatorias no coinciden con el contrato oficial."
    ) # Validar contrato obligatorio

if optional_evaluation_metrics != contract_optional_metrics:
    raise RuntimeError(
        "Las métricas opcionales no coinciden con el contrato definido."
    ) # Validar contrato opcional

if len(
    set(contract_required_metrics)
) != len(
    contract_required_metrics
):
    raise RuntimeError(
        "El contrato contiene métricas obligatorias duplicadas."
    ) # Validar unicidad

if len(
    set(contract_optional_metrics)
) != len(
    contract_optional_metrics
):
    raise RuntimeError(
        "El conjunto opcional contiene métricas duplicadas."
    ) # Validar unicidad

if "RMSE" not in required_evaluation_metrics:
    raise RuntimeError(
        "RMSE debe formar parte de la evaluación oficial."
    ) # Validar RMSE

if "MAE" not in required_evaluation_metrics:
    raise RuntimeError(
        "MAE debe formar parte de la evaluación oficial."
    ) # Validar MAE

if "MAPE" not in required_evaluation_metrics:
    raise RuntimeError(
        "MAPE debe formar parte de la evaluación oficial."
    ) # Validar MAPE

if "R2" not in required_evaluation_metrics:
    raise RuntimeError(
        "R2 debe formar parte de la evaluación oficial."
    ) # Validar R2

evaluation_metric_contract = {
    "required": contract_required_metrics.copy(),
    "optional_pending": contract_optional_metrics.copy(),
    "mse_confirmed": False,
    "status": "VALIDATED",
} # Construir contrato oficial de métricas

if evaluation_metric_contract[
    "mse_confirmed"
]:
    evaluation_metrics_to_calculate = (
        contract_required_metrics
        + ["MSE"]
    ) # Incluir MSE solamente con respaldo contractual
else:
    evaluation_metrics_to_calculate = (
        contract_required_metrics.copy()
    ) # Mantener exclusivamente métricas obligatorias

if evaluation_metrics_to_calculate != [
    "RMSE",
    "MAE",
    "MAPE",
    "R2",
]:
    raise RuntimeError(
        "El conjunto de métricas a calcular no coincide con el contrato oficial."
    ) # Validar conjunto final

evaluation_block_11_status = "METRIC_CONTRACT_VALIDATED" # Registrar contrato aprobado
evaluation_block_11_stage = "METRIC_CONTRACT_VALIDATED" # Registrar etapa aprobada

print(f"Métricas obligatorias    : {contract_required_metrics}") # Mostrar métricas obligatorias
print(f"Métricas opcionales      : {contract_optional_metrics}") # Mostrar métricas opcionales
print(f"MSE confirmado           : {evaluation_metric_contract['mse_confirmed']}") # Mostrar estado MSE
print(f"Métricas a calcular      : {evaluation_metrics_to_calculate}") # Mostrar conjunto final
print(f"Estado del contrato      : {evaluation_metric_contract['status']}") # Mostrar estado
print("RMSE                     : CONFIRMADA") # Confirmar RMSE
print("MAE                      : CONFIRMADA") # Confirmar MAE
print("MAPE                     : CONFIRMADA") # Confirmar MAPE
print("R2                       : CONFIRMADA") # Confirmar R2
print("MSE                      : NO INCLUIDA") # Confirmar exclusión
print("Contrato de métricas     : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 11.2. PREPARACIÓN Y VALIDACIÓN DE LOS VECTORES DE EVALUACIÓN") # Mostrar encabezado
if evaluation_block_11_status != "METRIC_CONTRACT_VALIDATED":
    raise RuntimeError(
        f"El Bloque 11.1 no fue validado. Estado actual: {evaluation_block_11_status}"
    ) # Impedir continuidad sin contrato validado

evaluation_block_11_stage = "VECTOR_PREPARATION" # Registrar etapa actual
if not isinstance(
    evaluation_y_true,
    torch.Tensor
):
    raise TypeError(
        "evaluation_y_true debe ser un tensor de PyTorch."
    ) # Validar valores reales

if not isinstance(
    evaluation_y_pred,
    torch.Tensor
):
    raise TypeError(
        "evaluation_y_pred debe ser un tensor de PyTorch."
    ) # Validar predicciones

evaluation_y_true = evaluation_y_true.detach().cpu().reshape(
    -1
).to(
    torch.float64
) # Preparar valores reales en precisión numérica común

evaluation_y_pred = evaluation_y_pred.detach().cpu().reshape(
    -1
).to(
    torch.float64
) # Preparar predicciones en precisión numérica común

if evaluation_y_true.ndim != 1:
    raise ValueError(
        "evaluation_y_true debe ser un vector unidimensional."
    ) # Validar dimensionalidad real

if evaluation_y_pred.ndim != 1:
    raise ValueError(
        "evaluation_y_pred debe ser un vector unidimensional."
    ) # Validar dimensionalidad predicha

if evaluation_y_true.shape != evaluation_y_pred.shape:
    raise RuntimeError(
        f"evaluation_y_true presenta forma {tuple(evaluation_y_true.shape)} y "
        f"evaluation_y_pred presenta forma {tuple(evaluation_y_pred.shape)}."
    ) # Validar correspondencia dimensional

if evaluation_y_true.numel() == 0:
    raise RuntimeError(
        "evaluation_y_true está vacío."
    ) # Validar disponibilidad real

if evaluation_y_pred.numel() == 0:
    raise RuntimeError(
        "evaluation_y_pred está vacío."
    ) # Validar disponibilidad predicha

if evaluation_y_true.numel() != PREDICTION_RESULT[
    "n_observations"
]:
    raise RuntimeError(
        "La cantidad de observaciones de evaluation_y_true no coincide con PREDICTION_RESULT."
    ) # Validar trazabilidad real

if evaluation_y_pred.numel() != PREDICTION_RESULT[
    "n_observations"
]:
    raise RuntimeError(
        "La cantidad de observaciones de evaluation_y_pred no coincide con PREDICTION_RESULT."
    ) # Validar trazabilidad predicha

if evaluation_y_true.numel() != PREDICTION_RESULT[
    "test_nodes"
]:
    raise RuntimeError(
        "La cantidad de valores reales no coincide con los nodos TEST."
    ) # Validar cobertura TEST

if evaluation_y_pred.numel() != PREDICTION_RESULT[
    "test_nodes"
]:
    raise RuntimeError(
        "La cantidad de predicciones no coincide con los nodos TEST."
    ) # Validar cobertura TEST

if evaluation_y_true.dtype != torch.float64:
    raise TypeError(
        f"evaluation_y_true presenta dtype {evaluation_y_true.dtype}."
    ) # Validar precisión real

if evaluation_y_pred.dtype != torch.float64:
    raise TypeError(
        f"evaluation_y_pred presenta dtype {evaluation_y_pred.dtype}."
    ) # Validar precisión predicha

if not torch.isfinite(
    evaluation_y_true
).all():
    raise RuntimeError(
        "evaluation_y_true contiene valores no finitos."
    ) # Validar estabilidad real

if not torch.isfinite(
    evaluation_y_pred
).all():
    raise RuntimeError(
        "evaluation_y_pred contiene valores no finitos."
    ) # Validar estabilidad predicha

evaluation_observation_count = int(
    evaluation_y_true.numel()
) # Registrar cantidad final de observaciones

evaluation_y_true_shape = tuple(
    evaluation_y_true.shape
) # Registrar forma final real

evaluation_y_pred_shape = tuple(
    evaluation_y_pred.shape
) # Registrar forma final predicha

evaluation_block_11_2_status = "VALIDATED" # Registrar 11.2 aprobado
evaluation_block_11_2_stage = "VECTORS_VALIDATED" # Registrar etapa aprobada

print(f"Shape y_true             : {evaluation_y_true_shape}") # Mostrar forma real
print(f"Shape y_pred             : {evaluation_y_pred_shape}") # Mostrar forma predicha
print(f"Dtype y_true             : {evaluation_y_true.dtype}") # Mostrar precisión real
print(f"Dtype y_pred             : {evaluation_y_pred.dtype}") # Mostrar precisión predicha
print(f"Observaciones            : {evaluation_observation_count}") # Mostrar cantidad
print(f"Nodos TEST               : {PREDICTION_RESULT['test_nodes']}") # Mostrar nodos TEST
print("Valores reales           : FINITOS") # Confirmar finitud real
print("Predicciones             : FINITAS") # Confirmar finitud predicha
print("Dimensionalidad          : VALIDADA") # Confirmar dimensionalidad
print("Cantidad                 : VALIDADA") # Confirmar cantidad
print("Precisión numérica       : FLOAT64") # Confirmar precisión
print("Vectores de evaluación   : VALIDATED") # Confirmar preparación

print("\nBLOQUE 11.3. CÁLCULO DE RMSE Y MAE") # Mostrar encabezado
evaluation_block_11_3_status = "ERROR" # Inicializar estado específico de 11.3
evaluation_block_11_3_stage = "RMSE_MAE" # Registrar etapa actual

if evaluation_block_11_2_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 11.2 no fue validado. Estado actual: {evaluation_block_11_2_status}"
    ) # Impedir cálculo sin vectores validados

if not isinstance(
    evaluation_y_true,
    torch.Tensor
):
    raise TypeError(
        "evaluation_y_true debe ser un tensor de PyTorch."
    ) # Validar valores reales

if not isinstance(
    evaluation_y_pred,
    torch.Tensor
):
    raise TypeError(
        "evaluation_y_pred debe ser un tensor de PyTorch."
    ) # Validar predicciones

if evaluation_y_true.ndim != 1:
    raise RuntimeError(
        "evaluation_y_true debe ser un vector unidimensional."
    ) # Validar dimensionalidad real

if evaluation_y_pred.ndim != 1:
    raise RuntimeError(
        "evaluation_y_pred debe ser un vector unidimensional."
    ) # Validar dimensionalidad predicha

if evaluation_y_true.shape != evaluation_y_pred.shape:
    raise RuntimeError(
        f"evaluation_y_true presenta forma {tuple(evaluation_y_true.shape)} y "
        f"evaluation_y_pred presenta forma {tuple(evaluation_y_pred.shape)}."
    ) # Validar correspondencia dimensional

if evaluation_y_true.numel() == 0:
    raise RuntimeError(
        "No existen observaciones para calcular RMSE y MAE."
    ) # Validar disponibilidad de observaciones

if not torch.isfinite(
    evaluation_y_true
).all():
    raise RuntimeError(
        "evaluation_y_true contiene valores no finitos."
    ) # Validar estabilidad de valores reales

if not torch.isfinite(
    evaluation_y_pred
).all():
    raise RuntimeError(
        "evaluation_y_pred contiene valores no finitos."
    ) # Validar estabilidad de predicciones

if evaluation_y_true.dtype != torch.float64:
    raise TypeError(
        f"evaluation_y_true presenta dtype {evaluation_y_true.dtype} y se esperaba torch.float64."
    ) # Validar precisión real

if evaluation_y_pred.dtype != torch.float64:
    raise TypeError(
        f"evaluation_y_pred presenta dtype {evaluation_y_pred.dtype} y se esperaba torch.float64."
    ) # Validar precisión predicha

evaluation_residuals = (
    evaluation_y_pred - evaluation_y_true
) # Calcular residuos predicción-real

if not torch.isfinite(
    evaluation_residuals
).all():
    raise RuntimeError(
        "Los residuos contienen valores no finitos."
    ) # Validar estabilidad de residuos

evaluation_squared_errors = (
    evaluation_residuals ** 2
) # Calcular errores cuadráticos

evaluation_absolute_errors = (
    torch.abs(evaluation_residuals)
) # Calcular errores absolutos

if not torch.isfinite(
    evaluation_squared_errors
).all():
    raise RuntimeError(
        "Los errores cuadráticos contienen valores no finitos."
    ) # Validar errores cuadráticos

if not torch.isfinite(
    evaluation_absolute_errors
).all():
    raise RuntimeError(
        "Los errores absolutos contienen valores no finitos."
    ) # Validar errores absolutos

evaluation_mse_internal_tensor = torch.mean(
    evaluation_squared_errors
) # Calcular MSE interno

evaluation_rmse_tensor = torch.sqrt(
    evaluation_mse_internal_tensor
) # Calcular RMSE

evaluation_mae_tensor = torch.mean(
    evaluation_absolute_errors
) # Calcular MAE

if not torch.isfinite(
    evaluation_mse_internal_tensor
):
    raise RuntimeError(
        "El MSE interno no es finito."
    ) # Validar MSE interno

if not torch.isfinite(
    evaluation_rmse_tensor
):
    raise RuntimeError(
        "RMSE no es finito."
    ) # Validar RMSE

if not torch.isfinite(
    evaluation_mae_tensor
):
    raise RuntimeError(
        "MAE no es finito."
    ) # Validar MAE

evaluation_mse_internal = float(
    evaluation_mse_internal_tensor.item()
) # Convertir MSE interno a escalar Python

evaluation_rmse = float(
    evaluation_rmse_tensor.item()
) # Convertir RMSE a escalar Python

evaluation_mae = float(
    evaluation_mae_tensor.item()
) # Convertir MAE a escalar Python

if evaluation_mse_internal < 0:
    raise RuntimeError(
        "El MSE interno no puede ser negativo."
    ) # Validar rango MSE

if evaluation_rmse < 0:
    raise RuntimeError(
        "RMSE no puede ser negativo."
    ) # Validar rango RMSE

if evaluation_mae < 0:
    raise RuntimeError(
        "MAE no puede ser negativo."
    ) # Validar rango MAE

if not np.isfinite(
    evaluation_mse_internal
):
    raise RuntimeError(
        "El MSE interno final no es finito."
    ) # Validar estabilidad final MSE

if not np.isfinite(
    evaluation_rmse
):
    raise RuntimeError(
        "El RMSE final no es finito."
    ) # Validar estabilidad final RMSE

if not np.isfinite(
    evaluation_mae
):
    raise RuntimeError(
        "El MAE final no es finito."
    ) # Validar estabilidad final MAE

evaluation_block_11_3_status = "VALIDATED" # Registrar 11.3 aprobado
evaluation_block_11_3_stage = "RMSE_MAE_VALIDATED" # Registrar etapa aprobada

print(f"Observaciones            : {evaluation_observation_count}") # Mostrar observaciones
print(f"RMSE                     : {evaluation_rmse:.12f}") # Mostrar RMSE
print(f"MAE                      : {evaluation_mae:.12f}") # Mostrar MAE
print(f"MSE interno              : {evaluation_mse_internal:.12f}") # Mostrar MSE auxiliar
print("Residuos                 : FINITOS") # Confirmar residuos
print("RMSE                     : VALIDADO") # Confirmar RMSE
print("MAE                      : VALIDADO") # Confirmar MAE
print("Métricas finitas         : VALIDADO") # Confirmar estabilidad
print("BLOQUE 11.3              : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 11.4. CÁLCULO DE MAPE") # Mostrar encabezado
evaluation_block_11_4_status = "ERROR" # Inicializar estado específico de 11.4
evaluation_block_11_4_stage = "MAPE" # Registrar etapa actual

if evaluation_block_11_3_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 11.3 no fue validado. Estado actual: {evaluation_block_11_3_status}"
    ) # Impedir cálculo sin RMSE y MAE validados

if not isinstance(
    evaluation_y_true,
    torch.Tensor
):
    raise TypeError(
        "evaluation_y_true debe ser un tensor de PyTorch."
    ) # Validar valores reales

if not isinstance(
    evaluation_y_pred,
    torch.Tensor
):
    raise TypeError(
        "evaluation_y_pred debe ser un tensor de PyTorch."
    ) # Validar predicciones

if evaluation_y_true.shape != evaluation_y_pred.shape:
    raise RuntimeError(
        "evaluation_y_true y evaluation_y_pred deben presentar la misma forma."
    ) # Validar correspondencia

if evaluation_y_true.numel() == 0:
    raise RuntimeError(
        "No existen observaciones para calcular MAPE."
    ) # Validar disponibilidad

if not torch.isfinite(
    evaluation_y_true
).all():
    raise RuntimeError(
        "evaluation_y_true contiene valores no finitos."
    ) # Validar estabilidad de valores reales

if not torch.isfinite(
    evaluation_y_pred
).all():
    raise RuntimeError(
        "evaluation_y_pred contiene valores no finitos."
    ) # Validar estabilidad de predicciones

evaluation_zero_target_mask = (
    evaluation_y_true == 0
) # Identificar valores reales iguales a cero

evaluation_zero_target_count = int(
    evaluation_zero_target_mask.sum().item()
) # Contar valores reales iguales a cero

if evaluation_zero_target_count > 0:
    raise RuntimeError(
        f"MAPE no puede calcularse de forma estricta porque "
        f"evaluation_y_true contiene {evaluation_zero_target_count} valores iguales a cero."
    ) # Impedir división por cero

evaluation_absolute_percentage_errors = (
    torch.abs(
        evaluation_y_pred - evaluation_y_true
    )
    / torch.abs(
        evaluation_y_true
    )
) # Calcular errores porcentuales relativos

if not torch.isfinite(
    evaluation_absolute_percentage_errors
).all():
    raise RuntimeError(
        "Los errores porcentuales relativos contienen valores no finitos."
    ) # Validar estabilidad numérica

evaluation_mape_tensor = torch.mean(
    evaluation_absolute_percentage_errors
) # Calcular MAPE en escala relativa

if not torch.isfinite(
    evaluation_mape_tensor
):
    raise RuntimeError(
        "MAPE no es finito."
    ) # Validar MAPE

if evaluation_mape_tensor.item() < 0:
    raise RuntimeError(
        "MAPE no puede ser negativo."
    ) # Validar rango MAPE

evaluation_mape = float(
    evaluation_mape_tensor.item()
) # Convertir MAPE a escalar Python

if not np.isfinite(
    evaluation_mape
):
    raise RuntimeError(
        "El valor final de MAPE no es finito."
    ) # Validar MAPE final

evaluation_block_11_4_status = "VALIDATED" # Registrar 11.4 aprobado
evaluation_block_11_4_stage = "MAPE_VALIDATED" # Registrar etapa aprobada

print(f"Observaciones            : {evaluation_observation_count}") # Mostrar observaciones
print(f"Ceros en y_true         : {evaluation_zero_target_count}") # Mostrar ceros
print(f"MAPE                     : {evaluation_mape:.12f}") # Mostrar MAPE relativo
print("Escala MAPE              : RELATIVA") # Confirmar escala
print("Multiplicación por 100   : NO APLICADA") # Confirmar compatibilidad con Benchmark
print("División por cero        : NO DETECTADA") # Confirmar seguridad
print("Errores porcentuales     : FINITOS") # Confirmar estabilidad
print("MAPE                     : VALIDADO") # Confirmar MAPE
print("BLOQUE 11.4              : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 11.5. CÁLCULO DE R2") # Mostrar encabezado
evaluation_block_11_5_status = "ERROR" # Inicializar estado específico de 11.5
evaluation_block_11_5_stage = "R2" # Registrar etapa actual

if evaluation_block_11_4_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 11.4 no fue validado. Estado actual: {evaluation_block_11_4_status}"
    ) # Impedir cálculo sin MAPE validado

if not isinstance(
    evaluation_y_true,
    torch.Tensor
):
    raise TypeError(
        "evaluation_y_true debe ser un tensor de PyTorch."
    ) # Validar valores reales

if not isinstance(
    evaluation_y_pred,
    torch.Tensor
):
    raise TypeError(
        "evaluation_y_pred debe ser un tensor de PyTorch."
    ) # Validar predicciones

if evaluation_y_true.shape != evaluation_y_pred.shape:
    raise RuntimeError(
        "evaluation_y_true y evaluation_y_pred deben presentar la misma forma."
    ) # Validar correspondencia

if evaluation_y_true.numel() < 2:
    raise RuntimeError(
        "Se requieren al menos dos observaciones para calcular R2."
    ) # Validar tamaño muestral

if not torch.isfinite(
    evaluation_y_true
).all():
    raise RuntimeError(
        "evaluation_y_true contiene valores no finitos."
    ) # Validar estabilidad de valores reales

if not torch.isfinite(
    evaluation_y_pred
).all():
    raise RuntimeError(
        "evaluation_y_pred contiene valores no finitos."
    ) # Validar estabilidad de predicciones

evaluation_y_true_mean = torch.mean(
    evaluation_y_true
) # Calcular media de los valores reales

evaluation_total_sum_squares = torch.sum(
    (
        evaluation_y_true - evaluation_y_true_mean
    ) ** 2
) # Calcular suma total de cuadrados

evaluation_residual_sum_squares = torch.sum(
    (
        evaluation_y_true - evaluation_y_pred
    ) ** 2
) # Calcular suma residual de cuadrados

if not torch.isfinite(
    evaluation_y_true_mean
):
    raise RuntimeError(
        "La media de y_true no es finita."
    ) # Validar estabilidad de la media

if not torch.isfinite(
    evaluation_total_sum_squares
):
    raise RuntimeError(
        "La suma total de cuadrados no es finita."
    ) # Validar estabilidad de SST

if not torch.isfinite(
    evaluation_residual_sum_squares
):
    raise RuntimeError(
        "La suma residual de cuadrados no es finita."
    ) # Validar estabilidad de SSE

if evaluation_total_sum_squares.item() <= 0:
    raise RuntimeError(
        "La varianza total de y_true es cero. R2 no está definido."
    ) # Validar denominador de R2

evaluation_r2_tensor = (
    1.0
    - (
        evaluation_residual_sum_squares
        / evaluation_total_sum_squares
    )
) # Calcular coeficiente de determinación

if not torch.isfinite(
    evaluation_r2_tensor
):
    raise RuntimeError(
        "R2 no es finito."
    ) # Validar estabilidad de R2

evaluation_r2 = float(
    evaluation_r2_tensor.item()
) # Convertir R2 a escalar Python

if not np.isfinite(
    evaluation_r2
):
    raise RuntimeError(
        "El valor final de R2 no es finito."
    ) # Validar R2 final

evaluation_block_11_5_status = "VALIDATED" # Registrar 11.5 aprobado
evaluation_block_11_5_stage = "R2_VALIDATED" # Registrar etapa aprobada

print(f"Observaciones            : {evaluation_observation_count}") # Mostrar observaciones
print(f"Media y_true             : {evaluation_y_true_mean.item():.12f}") # Mostrar media real
print(f"Suma total cuadrados     : {evaluation_total_sum_squares.item():.12f}") # Mostrar SST
print(f"Suma residual cuadrados  : {evaluation_residual_sum_squares.item():.12f}") # Mostrar SSE
print(f"R2                       : {evaluation_r2:.12f}") # Mostrar R2
print("SST                      : FINITA") # Confirmar SST
print("SSE                      : FINITA") # Confirmar SSE
print("R2                       : FINITO") # Confirmar estabilidad
print("R2                       : VALIDADO") # Confirmar R2
print("BLOQUE 11.5              : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 11.6. VALIDACIÓN CONJUNTA DE LAS MÉTRICAS") # Mostrar encabezado
evaluation_block_11_6_status = "ERROR" # Inicializar estado específico de 11.6
evaluation_block_11_6_stage = "METRICS_VALIDATION" # Registrar etapa actual

if evaluation_block_11_5_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 11.5 no fue validado. Estado actual: {evaluation_block_11_5_status}"
    ) # Impedir continuidad sin R2 validado

evaluation_metrics_result = {
    "RMSE": evaluation_rmse,
    "MAE": evaluation_mae,
    "MAPE": evaluation_mape,
    "R2": evaluation_r2,
} # Consolidar métricas oficiales

required_metric_names = [
    "RMSE",
    "MAE",
    "MAPE",
    "R2",
] # Definir métricas obligatorias

missing_metrics = [
    metric_name
    for metric_name in required_metric_names
    if metric_name not in evaluation_metrics_result
] # Identificar métricas faltantes

if missing_metrics:
    raise RuntimeError(
        f"Faltan métricas oficiales: {missing_metrics}"
    ) # Validar cobertura

for metric_name, metric_value in evaluation_metrics_result.items():
    if not isinstance(
        metric_value,
        (float, int, np.floating, np.integer)
    ):
        raise TypeError(
            f"{metric_name} debe ser un valor numérico escalar."
        ) # Validar tipo numérico

    if not np.isfinite(
        float(metric_value)
    ):
        raise RuntimeError(
            f"{metric_name} no es finita."
        ) # Validar finitud

if evaluation_rmse < 0:
    raise RuntimeError(
        "RMSE no puede ser negativo."
    ) # Validar rango RMSE

if evaluation_mae < 0:
    raise RuntimeError(
        "MAE no puede ser negativo."
    ) # Validar rango MAE

if evaluation_mape < 0:
    raise RuntimeError(
        "MAPE no puede ser negativo."
    ) # Validar rango MAPE

if evaluation_observation_count != PREDICTION_RESULT[
    "n_observations"
]:
    raise RuntimeError(
        "La cantidad de observaciones de las métricas no coincide con PREDICTION_RESULT."
    ) # Validar trazabilidad

if evaluation_observation_count != PREDICTION_RESULT[
    "test_nodes"
]:
    raise RuntimeError(
        "La cantidad de observaciones de las métricas no coincide con los nodos TEST."
    ) # Validar cobertura TEST

if PREDICTION_RESULT[
    "test_index"
].tolist() != evaluation_test_index.tolist():
    raise RuntimeError(
        "La partición utilizada por las métricas no coincide con evaluation_test_index."
    ) # Validar partición TEST

evaluation_metrics_test_index = PREDICTION_RESULT[
    "test_index"
].copy() # Registrar partición asociada a las métricas

evaluation_metrics_observations = int(
    evaluation_observation_count
) # Registrar observaciones evaluadas

evaluation_metrics_status = "VALIDATED" # Registrar conjunto de métricas aprobado
evaluation_block_11_6_status = "VALIDATED" # Registrar 11.6 aprobado
evaluation_block_11_6_stage = "METRICS_VALIDATED" # Registrar etapa aprobada

print(f"RMSE                     : {evaluation_rmse:.12f}") # Mostrar RMSE
print(f"MAE                      : {evaluation_mae:.12f}") # Mostrar MAE
print(f"MAPE                     : {evaluation_mape:.12f}") # Mostrar MAPE relativo
print("Escala MAPE              : RELATIVA") # Confirmar escala MAPE
print(f"R2                       : {evaluation_r2:.12f}") # Mostrar R2
print(f"Observaciones            : {evaluation_metrics_observations}") # Mostrar observaciones
print(f"Índices TEST             : {evaluation_metrics_test_index.tolist()}") # Mostrar partición
print("RMSE                     : VALIDADO") # Confirmar RMSE
print("MAE                      : VALIDADO") # Confirmar MAE
print("MAPE                     : VALIDADO") # Confirmar MAPE
print("R2                       : VALIDADO") # Confirmar R2
print("Finitud                  : VALIDADA") # Confirmar finitud
print("Cobertura TEST           : VALIDADA") # Confirmar cobertura
print("Conjunto de métricas     : VALIDATED") # Confirmar conjunto
print("BLOQUE 11.6              : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 11.7. CONSTRUCCIÓN DEL RESULTADO FINAL DE EVALUACIÓN") # Mostrar encabezado
evaluation_block_11_7_status = "ERROR" # Inicializar estado específico de 11.7
evaluation_block_11_7_stage = "RESULT_CONSTRUCTION" # Registrar etapa actual

if evaluation_block_11_6_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 11.6 no fue validado. Estado actual: {evaluation_block_11_6_status}"
    ) # Impedir construcción sin métricas validadas

if not isinstance(
    PREDICTION_RESULT,
    dict
):
    raise TypeError(
        "PREDICTION_RESULT debe ser un diccionario."
    ) # Validar resultado de predicciones

if PREDICTION_RESULT.get(
    "status"
) != "VALIDATED":
    raise RuntimeError(
        "PREDICTION_RESULT debe presentar estado VALIDATED."
    ) # Validar estado de predicciones

if evaluation_metrics_status != "VALIDATED":
    raise RuntimeError(
        "El conjunto de métricas no presenta estado VALIDATED."
    ) # Validar conjunto de métricas

if evaluation_model_code != "GNN02":
    raise RuntimeError(
        f"El código del modelo es {evaluation_model_code} y se esperaba GNN02."
    ) # Validar identidad oficial

if evaluation_model_name.strip().lower() != "graphsage":
    raise RuntimeError(
        f"El nombre del modelo es {evaluation_model_name} y se esperaba graphsage."
    ) # Validar identidad oficial

if evaluation_model_family.strip().lower() != "graph_neural_networks":
    raise RuntimeError(
        f"La familia del modelo es {evaluation_model_family} y se esperaba graph_neural_networks."
    ) # Validar familia oficial

if PREDICTION_RESULT[
    "prediction_source"
] != "evaluation_graphsage":
    raise RuntimeError(
        "La fuente de predicción no corresponde a evaluation_graphsage."
    ) # Validar independencia

if PREDICTION_RESULT[
    "test_index"
].tolist() != evaluation_test_index.tolist():
    raise RuntimeError(
        "La partición TEST de PREDICTION_RESULT no coincide con evaluation_test_index."
    ) # Validar partición

if evaluation_metrics_test_index.tolist() != evaluation_test_index.tolist():
    raise RuntimeError(
        "La partición asociada a las métricas no coincide con evaluation_test_index."
    ) # Validar trazabilidad TEST

if evaluation_observation_count != PREDICTION_RESULT[
    "n_observations"
]:
    raise RuntimeError(
        "La cantidad de observaciones evaluadas no coincide con PREDICTION_RESULT."
    ) # Validar trazabilidad observacional

if evaluation_observation_count != len(
    evaluation_y_true
):
    raise RuntimeError(
        "La cantidad de observaciones no coincide con evaluation_y_true."
    ) # Validar correspondencia observacional

if evaluation_observation_count != len(
    evaluation_y_pred
):
    raise RuntimeError(
        "La cantidad de observaciones no coincide con evaluation_y_pred."
    ) # Validar correspondencia predictiva

EVALUATION_RESULT = {
    "model_code": evaluation_model_code,
    "model_name": evaluation_model_name,
    "family": evaluation_model_family,
    "test_index": evaluation_test_index.copy(),
    "test_graphs": int(len(evaluation_test_graphs)),
    "test_nodes": int(evaluation_test_total_nodes),
    "n_observations": int(evaluation_observation_count),
    "prediction_source": PREDICTION_RESULT["prediction_source"],
    "inference_time": float(PREDICTION_RESULT["inference_time"]),
    "RMSE": float(evaluation_rmse),
    "MAE": float(evaluation_mae),
    "MAPE": float(evaluation_mape),
    "R2": float(evaluation_r2),
    "metrics": {
        "RMSE": float(evaluation_rmse),
        "MAE": float(evaluation_mae),
        "MAPE": float(evaluation_mape),
        "R2": float(evaluation_r2),
    },
    "status": "VALIDATED",
} # Construir resultado final de evaluación

if not isinstance(
    EVALUATION_RESULT,
    dict
):
    raise TypeError(
        "EVALUATION_RESULT debe ser un diccionario."
    ) # Validar estructura final

required_evaluation_result_fields = [
    "model_code",
    "model_name",
    "family",
    "test_index",
    "test_graphs",
    "test_nodes",
    "n_observations",
    "prediction_source",
    "inference_time",
    "RMSE",
    "MAE",
    "MAPE",
    "R2",
    "metrics",
    "status",
] # Definir campos obligatorios

missing_evaluation_result_fields = [
    field
    for field in required_evaluation_result_fields
    if field not in EVALUATION_RESULT
] # Identificar campos faltantes

if missing_evaluation_result_fields:
    raise RuntimeError(
        f"EVALUATION_RESULT presenta campos faltantes: {missing_evaluation_result_fields}"
    ) # Validar cobertura

if EVALUATION_RESULT[
    "test_graphs"
] != len(
    EVALUATION_RESULT["test_index"]
):
    raise RuntimeError(
        "La cantidad de GraphData TEST no coincide con test_index."
    ) # Validar partición

if EVALUATION_RESULT[
    "n_observations"
] != evaluation_observation_count:
    raise RuntimeError(
        "La cantidad de observaciones de EVALUATION_RESULT no coincide con el conteo evaluado."
    ) # Validar trazabilidad observacional

if EVALUATION_RESULT[
    "prediction_source"
] != PREDICTION_RESULT[
    "prediction_source"
]:
    raise RuntimeError(
        "La fuente de predicción no coincide entre EVALUATION_RESULT y PREDICTION_RESULT."
    ) # Validar procedencia

for metric_name in [
    "RMSE",
    "MAE",
    "MAPE",
    "R2",
]:
    metric_value = EVALUATION_RESULT[
        metric_name
    ] # Recuperar métrica

    if not isinstance(
        metric_value,
        (float, int, np.floating, np.integer)
    ):
        raise TypeError(
            f"{metric_name} debe ser un valor numérico escalar."
        ) # Validar tipo numérico

    if not np.isfinite(
        float(metric_value)
    ):
        raise RuntimeError(
            f"La métrica {metric_name} no es finita."
        ) # Validar estabilidad numérica

    if EVALUATION_RESULT[
        "metrics"
    ][metric_name] != metric_value:
        raise RuntimeError(
            f"La métrica {metric_name} no coincide entre los niveles del resultado."
        ) # Validar consistencia

if EVALUATION_RESULT[
    "RMSE"
] < 0:
    raise RuntimeError(
        "RMSE no puede ser negativo."
    ) # Validar rango RMSE

if EVALUATION_RESULT[
    "MAE"
] < 0:
    raise RuntimeError(
        "MAE no puede ser negativo."
    ) # Validar rango MAE

if EVALUATION_RESULT[
    "MAPE"
] < 0:
    raise RuntimeError(
        "MAPE no puede ser negativo."
    ) # Validar rango MAPE

if not np.isfinite(
    EVALUATION_RESULT[
        "inference_time"
    ]
):
    raise RuntimeError(
        "El tiempo de inferencia no es finito."
    ) # Validar estabilidad temporal

if EVALUATION_RESULT[
    "inference_time"
] < 0:
    raise RuntimeError(
        "El tiempo de inferencia no puede ser negativo."
    ) # Validar rango temporal

if EVALUATION_RESULT[
    "status"
] != "VALIDATED":
    raise RuntimeError(
        "EVALUATION_RESULT no presenta estado VALIDATED."
    ) # Validar estado final

evaluation_block_11_7_status = "VALIDATED" # Registrar 11.7 aprobado
evaluation_block_11_7_stage = "RESULT_CONSTRUCTED" # Registrar etapa aprobada

print(f"Código del modelo        : {EVALUATION_RESULT['model_code']}") # Mostrar código
print(f"Nombre del modelo        : {EVALUATION_RESULT['model_name']}") # Mostrar modelo
print(f"Familia                  : {EVALUATION_RESULT['family']}") # Mostrar familia
print(f"Índices TEST             : {EVALUATION_RESULT['test_index'].tolist()}") # Mostrar partición
print(f"GraphData TEST           : {EVALUATION_RESULT['test_graphs']}") # Mostrar grafos
print(f"Nodos TEST               : {EVALUATION_RESULT['test_nodes']}") # Mostrar nodos
print(f"Observaciones            : {EVALUATION_RESULT['n_observations']}") # Mostrar observaciones
print(f"Fuente de predicción     : {EVALUATION_RESULT['prediction_source']}") # Mostrar fuente
print(f"Tiempo de inferencia     : {EVALUATION_RESULT['inference_time']:.6f} segundos") # Mostrar tiempo
print(f"RMSE                     : {EVALUATION_RESULT['RMSE']:.12f}") # Mostrar RMSE
print(f"MAE                      : {EVALUATION_RESULT['MAE']:.12f}") # Mostrar MAE
print(f"MAPE                     : {EVALUATION_RESULT['MAPE']:.12f}") # Mostrar MAPE relativo
print("Escala MAPE              : RELATIVA") # Confirmar escala MAPE
print(f"R2                       : {EVALUATION_RESULT['R2']:.12f}") # Mostrar R2
print(f"Estado                   : {EVALUATION_RESULT['status']}") # Mostrar estado
print("Resultado de evaluación  : CONSTRUIDO") # Confirmar construcción
print("BLOQUE 11.7              : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 11.8. AUDITORÍA FINAL Y CIERRE DEL BLOQUE 11") # Mostrar encabezado
evaluation_block_11_8_status = "ERROR" # Inicializar estado específico de 11.8
evaluation_block_11_8_stage = "FINAL_AUDIT" # Registrar etapa actual

if evaluation_block_11_7_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 11.7 no fue validado. Estado actual: {evaluation_block_11_7_status}"
    ) # Impedir cierre sin resultado construido

if evaluation_block_10_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 10 no está VALIDATED. Estado actual: {evaluation_block_10_status}"
    ) # Validar dependencia anterior

if not isinstance(
    EVALUATION_RESULT,
    dict
):
    raise TypeError(
        "EVALUATION_RESULT debe ser un diccionario."
    ) # Validar estructura final

required_final_fields = [
    "model_code",
    "model_name",
    "family",
    "test_index",
    "test_graphs",
    "test_nodes",
    "n_observations",
    "prediction_source",
    "inference_time",
    "RMSE",
    "MAE",
    "MAPE",
    "R2",
    "metrics",
    "status",
] # Definir campos finales obligatorios

missing_final_fields = [
    field
    for field in required_final_fields
    if field not in EVALUATION_RESULT
] # Identificar campos faltantes

if missing_final_fields:
    raise RuntimeError(
        f"EVALUATION_RESULT presenta campos faltantes: {missing_final_fields}"
    ) # Validar estructura completa

if EVALUATION_RESULT[
    "model_code"
] != PREDICTION_RESULT[
    "model_code"
]:
    raise RuntimeError(
        "El código del modelo no coincide entre EVALUATION_RESULT y PREDICTION_RESULT."
    ) # Validar identidad

if EVALUATION_RESULT[
    "model_name"
].strip().lower() != PREDICTION_RESULT[
    "model_name"
].strip().lower():
    raise RuntimeError(
        "El nombre del modelo no coincide entre EVALUATION_RESULT y PREDICTION_RESULT."
    ) # Validar identidad

if EVALUATION_RESULT[
    "family"
].strip().lower() != PREDICTION_RESULT[
    "family"
].strip().lower():
    raise RuntimeError(
        "La familia del modelo no coincide entre EVALUATION_RESULT y PREDICTION_RESULT."
    ) # Validar identidad

if EVALUATION_RESULT[
    "model_code"
] != "GNN02":
    raise RuntimeError(
        "EVALUATION_RESULT no corresponde al Modelo Oficial GNN02."
    ) # Validar código oficial

if EVALUATION_RESULT[
    "model_name"
].strip().lower() != "graphsage":
    raise RuntimeError(
        "EVALUATION_RESULT no corresponde a GraphSAGE."
    ) # Validar nombre oficial

if EVALUATION_RESULT[
    "family"
].strip().lower() != "graph_neural_networks":
    raise RuntimeError(
        "EVALUATION_RESULT no corresponde a la familia oficial GNN."
    ) # Validar familia oficial

if EVALUATION_RESULT[
    "prediction_source"
] != "evaluation_graphsage":
    raise RuntimeError(
        "EVALUATION_RESULT no registra evaluation_graphsage como fuente de predicción."
    ) # Validar procedencia independiente

if EVALUATION_RESULT[
    "test_index"
].tolist() != PREDICTION_RESULT[
    "test_index"
].tolist():
    raise RuntimeError(
        "La partición TEST no coincide entre EVALUATION_RESULT y PREDICTION_RESULT."
    ) # Validar partición

if EVALUATION_RESULT[
    "test_index"
].tolist() != evaluation_test_index.tolist():
    raise RuntimeError(
        "La partición TEST no coincide con evaluation_test_index."
    ) # Validar partición oficial

if EVALUATION_RESULT[
    "test_graphs"
] != PREDICTION_RESULT[
    "test_graphs"
]:
    raise RuntimeError(
        "La cantidad de GraphData TEST no coincide entre resultados."
    ) # Validar cantidad de grafos

if EVALUATION_RESULT[
    "test_nodes"
] != PREDICTION_RESULT[
    "test_nodes"
]:
    raise RuntimeError(
        "La cantidad de nodos TEST no coincide entre resultados."
    ) # Validar nodos

if EVALUATION_RESULT[
    "n_observations"
] != PREDICTION_RESULT[
    "n_observations"
]:
    raise RuntimeError(
        "La cantidad de observaciones no coincide entre resultados."
    ) # Validar observaciones

if EVALUATION_RESULT[
    "n_observations"
] != evaluation_observation_count:
    raise RuntimeError(
        "La cantidad de observaciones no coincide con los vectores evaluados."
    ) # Validar cobertura observacional

if EVALUATION_RESULT[
    "n_observations"
] != len(
    evaluation_y_true
):
    raise RuntimeError(
        "La cantidad de observaciones no coincide con evaluation_y_true."
    ) # Validar valores reales

if EVALUATION_RESULT[
    "n_observations"
] != len(
    evaluation_y_pred
):
    raise RuntimeError(
        "La cantidad de observaciones no coincide con evaluation_y_pred."
    ) # Validar predicciones

if EVALUATION_RESULT[
    "test_graphs"
] != len(
    EVALUATION_RESULT["test_index"]
):
    raise RuntimeError(
        "La cantidad de GraphData TEST no coincide con test_index."
    ) # Validar partición

for metric_name in [
    "RMSE",
    "MAE",
    "MAPE",
    "R2",
]:
    metric_value = EVALUATION_RESULT[
        metric_name
    ] # Recuperar métrica

    if not isinstance(
        metric_value,
        (float, int, np.floating, np.integer)
    ):
        raise TypeError(
            f"{metric_name} debe ser un valor numérico escalar."
        ) # Validar tipo numérico

    if not np.isfinite(
        float(metric_value)
    ):
        raise RuntimeError(
            f"{metric_name} no es finita."
        ) # Validar finitud

    if EVALUATION_RESULT[
        "metrics"
    ][metric_name] != metric_value:
        raise RuntimeError(
            f"{metric_name} no coincide entre los campos principales y metrics."
        ) # Validar consistencia

if EVALUATION_RESULT[
    "RMSE"
] < 0:
    raise RuntimeError(
        "RMSE no puede ser negativo."
    ) # Validar rango RMSE

if EVALUATION_RESULT[
    "MAE"
] < 0:
    raise RuntimeError(
        "MAE no puede ser negativo."
    ) # Validar rango MAE

if EVALUATION_RESULT[
    "MAPE"
] < 0:
    raise RuntimeError(
        "MAPE no puede ser negativo."
    ) # Validar rango MAPE

if not np.isfinite(
    EVALUATION_RESULT[
        "inference_time"
    ]
):
    raise RuntimeError(
        "El tiempo de inferencia no es finito."
    ) # Validar estabilidad temporal

if EVALUATION_RESULT[
    "inference_time"
] < 0:
    raise RuntimeError(
        "El tiempo de inferencia no puede ser negativo."
    ) # Validar rango temporal

if EVALUATION_RESULT[
    "status"
] != "VALIDATED":
    raise RuntimeError(
        "EVALUATION_RESULT no presenta estado VALIDATED."
    ) # Validar estado final

evaluation_audit_fields = {
    "identity": (
        EVALUATION_RESULT["model_code"] == PREDICTION_RESULT["model_code"]
        and EVALUATION_RESULT["model_name"].strip().lower() == PREDICTION_RESULT["model_name"].strip().lower()
        and EVALUATION_RESULT["family"].strip().lower() == PREDICTION_RESULT["family"].strip().lower()
    ),
    "test_partition": (
        EVALUATION_RESULT["test_index"].tolist() == evaluation_test_index.tolist()
    ),
    "observation_count": (
        EVALUATION_RESULT["n_observations"] == evaluation_observation_count
        and EVALUATION_RESULT["n_observations"] == len(evaluation_y_true)
        and EVALUATION_RESULT["n_observations"] == len(evaluation_y_pred)
    ),
    "prediction_source": (
        EVALUATION_RESULT["prediction_source"] == "evaluation_graphsage"
    ),
    "metrics": all(
        np.isfinite(
            float(EVALUATION_RESULT[metric_name])
        )
        for metric_name in [
            "RMSE",
            "MAE",
            "MAPE",
            "R2",
        ]
    ),
    "inference_time": (
        np.isfinite(
            float(EVALUATION_RESULT["inference_time"])
        )
        and EVALUATION_RESULT["inference_time"] >= 0
    ),
    "result_status": (
        EVALUATION_RESULT["status"] == "VALIDATED"
    ),
} # Registrar resultados reales de auditoría

evaluation_audit_failures = [
    audit_name
    for audit_name, audit_passed in evaluation_audit_fields.items()
    if not audit_passed
] # Identificar componentes no validados

if evaluation_audit_failures:
    raise RuntimeError(
        f"La auditoría final presenta componentes no validados: {evaluation_audit_failures}"
    ) # Impedir cierre con auditoría incompleta

evaluation_block_11_8_status = "VALIDATED" # Registrar auditoría final aprobada
evaluation_block_11_8_stage = "FINAL_AUDIT_VALIDATED" # Registrar etapa aprobada

evaluation_block_11_status = "VALIDATED" # Registrar aprobación final
evaluation_block_11_stage = "VALIDATED" # Registrar cierre definitivo

print(f"Código del modelo        : {EVALUATION_RESULT['model_code']}") # Mostrar código
print(f"Nombre del modelo        : {EVALUATION_RESULT['model_name']}") # Mostrar nombre
print(f"Familia                  : {EVALUATION_RESULT['family']}") # Mostrar familia
print(f"Índices TEST             : {EVALUATION_RESULT['test_index'].tolist()}") # Mostrar partición
print(f"GraphData TEST           : {EVALUATION_RESULT['test_graphs']}") # Mostrar grafos
print(f"Nodos TEST               : {EVALUATION_RESULT['test_nodes']}") # Mostrar nodos
print(f"Observaciones            : {EVALUATION_RESULT['n_observations']}") # Mostrar observaciones
print(f"Fuente                   : {EVALUATION_RESULT['prediction_source']}") # Mostrar fuente
print(f"RMSE                     : {EVALUATION_RESULT['RMSE']:.12f}") # Mostrar RMSE
print(f"MAE                      : {EVALUATION_RESULT['MAE']:.12f}") # Mostrar MAE
print(f"MAPE                     : {EVALUATION_RESULT['MAPE']:.12f}") # Mostrar MAPE relativo
print("Escala MAPE              : RELATIVA") # Confirmar escala
print(f"R2                       : {EVALUATION_RESULT['R2']:.12f}") # Mostrar R2
print(f"Tiempo inferencia        : {EVALUATION_RESULT['inference_time']:.6f} segundos") # Mostrar tiempo
print(f"Estado resultado         : {EVALUATION_RESULT['status']}") # Mostrar estado
print("Identidad del modelo     : VALIDADA") # Confirmar identidad
print("Partición TEST           : VALIDADA") # Confirmar TEST
print("Cobertura observacional  : VALIDADA") # Confirmar cobertura
print("Fuente de predicción     : VALIDADA") # Confirmar procedencia
print("Métricas oficiales       : VALIDADAS") # Confirmar métricas
print("Tiempo de inferencia     : VALIDADO") # Confirmar tiempo
print("Auditoría final          : VALIDADA") # Confirmar auditoría
print("BLOQUE 11.8              : VALIDATED") # Confirmar auditoría
print("BLOQUE 11                : VALIDATED") # Confirmar cierre definitivo

# BLOQUE 12. DIAGNÓSTICO PREVIO PARA EL ANÁLISIS DE RESIDUOS
# Objetivo: Verificar la disponibilidad y consistencia de los valores reales y predichos necesarios para construir los residuos.
# Entradas: PREDICTION_RESULT y EVALUATION_RESULT.
# Producto: Diagnóstico preparado para la construcción de residuals.
# Pregunta científica: ¿Se encuentran disponibles y correctamente alineados los valores reales y predichos necesarios para construir los residuos del Modelo Oficial sobre TEST?

print("\nBLOQUE 12. DIAGNÓSTICO PREVIO PARA EL ANÁLISIS DE RESIDUOS") # Mostrar encabezado

print("\nBLOQUE 12.0. DIAGNÓSTICO PREVIO PARA EL ANÁLISIS DE RESIDUOS") # Mostrar encabezado
evaluation_block_12_status = "ERROR" # Inicializar estado en ERROR hasta completar las validaciones
evaluation_block_12_stage = "DIAGNOSTICO" # Registrar etapa actual

if evaluation_block_11_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 11 no está VALIDATED. Estado actual: {evaluation_block_11_status}"
    ) # Impedir análisis sobre evaluación no validada

if not isinstance(
    PREDICTION_RESULT,
    dict
):
    raise TypeError(
        "PREDICTION_RESULT debe ser un diccionario."
    ) # Validar estructura de predicciones

if not isinstance(
    EVALUATION_RESULT,
    dict
):
    raise TypeError(
        "EVALUATION_RESULT debe ser un diccionario."
    ) # Validar estructura de evaluación

if PREDICTION_RESULT.get(
    "status"
) != "VALIDATED":
    raise RuntimeError(
        "PREDICTION_RESULT debe presentar estado VALIDATED."
    ) # Validar estado de predicciones

if EVALUATION_RESULT.get(
    "status"
) != "VALIDATED":
    raise RuntimeError(
        "EVALUATION_RESULT debe presentar estado VALIDATED."
    ) # Validar estado de evaluación

required_residual_prediction_fields = [
    "model_code",
    "model_name",
    "family",
    "test_index",
    "test_nodes",
    "y_true",
    "y_pred",
    "n_observations",
    "prediction_source",
    "status",
] # Definir campos requeridos

missing_residual_prediction_fields = [
    field
    for field in required_residual_prediction_fields
    if field not in PREDICTION_RESULT
] # Identificar campos faltantes

if missing_residual_prediction_fields:
    raise RuntimeError(
        f"PREDICTION_RESULT presenta campos faltantes: {missing_residual_prediction_fields}"
    ) # Validar cobertura

if "y_true" not in PREDICTION_RESULT:
    raise RuntimeError(
        "PREDICTION_RESULT no contiene y_true."
    ) # Validar valores reales

if "y_pred" not in PREDICTION_RESULT:
    raise RuntimeError(
        "PREDICTION_RESULT no contiene y_pred."
    ) # Validar predicciones

evaluation_residual_y_true = PREDICTION_RESULT[
    "y_true"
].detach().cpu().reshape(
    -1
) # Recuperar valores reales

evaluation_residual_y_pred = PREDICTION_RESULT[
    "y_pred"
].detach().cpu().reshape(
    -1
) # Recuperar valores predichos

if evaluation_residual_y_true.shape != evaluation_residual_y_pred.shape:
    raise RuntimeError(
        "y_true y y_pred no presentan la misma dimensión para el análisis de residuos."
    ) # Validar dimensiones

if len(
    evaluation_residual_y_true
) != len(
    evaluation_residual_y_pred
):
    raise RuntimeError(
        "y_true y y_pred no presentan la misma longitud para el análisis de residuos."
    ) # Validar longitud

if len(
    evaluation_residual_y_true
) == 0:
    raise RuntimeError(
        "No existen observaciones para construir los residuos."
    ) # Validar disponibilidad

if not torch.isfinite(
    evaluation_residual_y_true
).all():
    raise RuntimeError(
        "y_true contiene valores no finitos."
    ) # Validar estabilidad de y_true

if not torch.isfinite(
    evaluation_residual_y_pred
).all():
    raise RuntimeError(
        "y_pred contiene valores no finitos."
    ) # Validar estabilidad de y_pred

if PREDICTION_RESULT[
    "n_observations"
] != len(
    evaluation_residual_y_true
):
    raise RuntimeError(
        "La cantidad de observaciones no coincide con PREDICTION_RESULT."
    ) # Validar trazabilidad

evaluation_residual_test_index = PREDICTION_RESULT[
    "test_index"
].copy() # Recuperar partición TEST

if EVALUATION_RESULT[
    "test_index"
].tolist() != evaluation_residual_test_index.tolist():
    raise RuntimeError(
        "La partición TEST no coincide entre PREDICTION_RESULT y EVALUATION_RESULT."
    ) # Validar trazabilidad TEST

if PREDICTION_RESULT[
    "prediction_source"
] != "evaluation_graphsage":
    raise RuntimeError(
        "La fuente de predicción no corresponde a evaluation_graphsage."
    ) # Validar procedencia independiente

evaluation_residual_observation_count = int(
    len(evaluation_residual_y_true)
) # Registrar observaciones disponibles

evaluation_block_12_status = "DIAGNOSTIC_VALIDATED" # Registrar diagnóstico aprobado
evaluation_block_12_stage = "DIAGNOSTICO_VALIDADO" # Registrar etapa aprobada

print(f"Estado Bloque 11        : {evaluation_block_11_status}") # Mostrar dependencia
print(f"Modelo                  : {PREDICTION_RESULT['model_name']}") # Mostrar modelo
print(f"Código                  : {PREDICTION_RESULT['model_code']}") # Mostrar código
print(f"Familia                 : {PREDICTION_RESULT['family']}") # Mostrar familia
print(f"Índices TEST            : {evaluation_residual_test_index.tolist()}") # Mostrar partición
print(f"Observaciones TEST      : {evaluation_residual_observation_count}") # Mostrar observaciones
print(f"Shape y_true            : {tuple(evaluation_residual_y_true.shape)}") # Mostrar shape real
print(f"Shape y_pred            : {tuple(evaluation_residual_y_pred.shape)}") # Mostrar shape predicho
print(f"Fuente de predicción    : {PREDICTION_RESULT['prediction_source']}") # Mostrar procedencia
print("Dimensiones             : COINCIDENTES") # Confirmar dimensiones
print("Longitud                : COINCIDENTE") # Confirmar longitud
print("Valores numéricos       : FINITOS") # Confirmar estabilidad
print("Partición TEST          : VALIDADA") # Confirmar partición
print("Diagnóstico Bloque 12.0 : VALIDATED") # Confirmar diagnóstico

print("\nBLOQUE 12.1. CONSTRUCCIÓN DE LOS RESIDUOS") # Mostrar encabezado
if evaluation_block_12_status != "DIAGNOSTIC_VALIDATED":
    raise RuntimeError(
        f"El Bloque 12.0 no fue validado. Estado actual: {evaluation_block_12_status}"
    ) # Impedir construcción sin diagnóstico validado

evaluation_block_12_stage = "RESIDUAL_CONSTRUCTION" # Registrar etapa actual
if not isinstance(
    evaluation_residual_y_true,
    torch.Tensor
):
    raise TypeError(
        "evaluation_residual_y_true debe ser un tensor de PyTorch."
    ) # Validar valores reales

if not isinstance(
    evaluation_residual_y_pred,
    torch.Tensor
):
    raise TypeError(
        "evaluation_residual_y_pred debe ser un tensor de PyTorch."
    ) # Validar predicciones

if evaluation_residual_y_true.shape != evaluation_residual_y_pred.shape:
    raise RuntimeError(
        "y_true y y_pred deben presentar la misma forma."
    ) # Validar correspondencia dimensional

if evaluation_residual_y_true.numel() == 0:
    raise RuntimeError(
        "evaluation_residual_y_true está vacío."
    ) # Validar disponibilidad

if evaluation_residual_y_pred.numel() == 0:
    raise RuntimeError(
        "evaluation_residual_y_pred está vacío."
    ) # Validar disponibilidad

residuals = (
    evaluation_residual_y_true - evaluation_residual_y_pred
) # Construir residuos como y_true - y_pred

if residuals.shape != evaluation_residual_y_true.shape:
    raise RuntimeError(
        "Los residuos no conservan la dimensión de y_true."
    ) # Validar dimensión resultante

if residuals.numel() != evaluation_residual_observation_count:
    raise RuntimeError(
        "La cantidad de residuos no coincide con las observaciones TEST."
    ) # Validar cobertura

if not torch.isfinite(
    residuals
).all():
    raise RuntimeError(
        "Los residuos contienen valores no finitos."
    ) # Validar estabilidad numérica

residuals = residuals.detach().cpu() # Consolidar residuos en CPU
evaluation_residual_mean = float(
    torch.mean(
        residuals
    ).item()
) # Calcular media descriptiva de residuos

evaluation_residual_std = float(
    torch.std(
        residuals,
        unbiased=False
    ).item()
) # Calcular desviación estándar descriptiva

evaluation_residual_min = float(
    torch.min(
        residuals
    ).item()
) # Obtener residuo mínimo

evaluation_residual_max = float(
    torch.max(
        residuals
    ).item()
) # Obtener residuo máximo

if not np.isfinite(
    evaluation_residual_mean
):
    raise RuntimeError(
        "La media de los residuos no es finita."
    ) # Validar media

if not np.isfinite(
    evaluation_residual_std
):
    raise RuntimeError(
        "La desviación estándar de los residuos no es finita."
    ) # Validar desviación estándar

if not np.isfinite(
    evaluation_residual_min
):
    raise RuntimeError(
        "El residuo mínimo no es finito."
    ) # Validar mínimo

if not np.isfinite(
    evaluation_residual_max
):
    raise RuntimeError(
        "El residuo máximo no es finito."
    ) # Validar máximo

evaluation_block_12_status = "RESIDUALS_VALIDATED" # Registrar residuos aprobados
evaluation_block_12_stage = "RESIDUALS_VALIDATED" # Registrar etapa aprobada

print(f"Observaciones            : {evaluation_residual_observation_count}") # Mostrar observaciones
print(f"Shape residuals          : {tuple(residuals.shape)}") # Mostrar dimensión
print(f"Media residual           : {evaluation_residual_mean:.12f}") # Mostrar media
print(f"Desviación estándar      : {evaluation_residual_std:.12f}") # Mostrar desviación
print(f"Residuo mínimo           : {evaluation_residual_min:.12f}") # Mostrar mínimo
print(f"Residuo máximo           : {evaluation_residual_max:.12f}") # Mostrar máximo
print("Definición               : y_true - y_pred") # Mostrar definición
print("Valores no finitos       : 0") # Confirmar finitud
print("Cobertura TEST           : VALIDADA") # Confirmar cobertura
print("Residuos                 : CONSTRUIDOS") # Confirmar construcción
print("BLOQUE 12.1              : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 12.2. ESTADÍSTICOS DESCRIPTIVOS DE LOS RESIDUOS") # Mostrar encabezado
if evaluation_block_12_status != "RESIDUALS_VALIDATED":
    raise RuntimeError(
        f"El Bloque 12.1 no fue validado. Estado actual: {evaluation_block_12_status}"
    ) # Impedir análisis sin residuos validados

evaluation_block_12_stage = "RESIDUAL_DESCRIPTIVES" # Registrar etapa actual
if not isinstance(
    residuals,
    torch.Tensor
):
    raise TypeError(
        "residuals debe ser un tensor de PyTorch."
    ) # Validar estructura

if residuals.ndim != 1:
    raise RuntimeError(
        "residuals debe ser un vector unidimensional."
    ) # Validar dimensionalidad

if residuals.numel() != evaluation_residual_observation_count:
    raise RuntimeError(
        "La cantidad de residuos no coincide con las observaciones TEST."
    ) # Validar cobertura

if not torch.isfinite(
    residuals
).all():
    raise RuntimeError(
        "residuals contiene valores no finitos."
    ) # Validar estabilidad

evaluation_residual_mean = float(
    torch.mean(
        residuals
    ).item()
) # Calcular media

evaluation_residual_median = float(
    torch.median(
        residuals
    ).item()
) # Calcular mediana

evaluation_residual_std = float(
    torch.std(
        residuals,
        unbiased=False
    ).item()
) # Calcular desviación estándar poblacional

evaluation_residual_variance = float(
    torch.var(
        residuals,
        unbiased=False
    ).item()
) # Calcular varianza poblacional

evaluation_residual_min = float(
    torch.min(
        residuals
    ).item()
) # Calcular mínimo

evaluation_residual_max = float(
    torch.max(
        residuals
    ).item()
) # Calcular máximo

evaluation_residual_q25 = float(
    torch.quantile(
        residuals,
        0.25
    ).item()
) # Calcular primer cuartil

evaluation_residual_q75 = float(
    torch.quantile(
        residuals,
        0.75
    ).item()
) # Calcular tercer cuartil

evaluation_residual_iqr = (
    evaluation_residual_q75
    - evaluation_residual_q25
) # Calcular rango intercuartílico

evaluation_residual_abs_mean = float(
    torch.mean(
        torch.abs(
            residuals
        )
    ).item()
) # Calcular media absoluta descriptiva

evaluation_residual_positive_count = int(
    torch.sum(
        residuals > 0
    ).item()
) # Contar residuos positivos

evaluation_residual_negative_count = int(
    torch.sum(
        residuals < 0
    ).item()
) # Contar residuos negativos

evaluation_residual_zero_count = int(
    torch.sum(
        residuals == 0
    ).item()
) # Contar residuos exactamente iguales a cero

evaluation_residual_positive_rate = (
    evaluation_residual_positive_count
    / evaluation_residual_observation_count
) # Calcular proporción de residuos positivos

evaluation_residual_negative_rate = (
    evaluation_residual_negative_count
    / evaluation_residual_observation_count
) # Calcular proporción de residuos negativos

evaluation_residual_zero_rate = (
    evaluation_residual_zero_count
    / evaluation_residual_observation_count
) # Calcular proporción de residuos cero

evaluation_residual_descriptive_values = {
    "mean": evaluation_residual_mean,
    "median": evaluation_residual_median,
    "std": evaluation_residual_std,
    "variance": evaluation_residual_variance,
    "min": evaluation_residual_min,
    "q25": evaluation_residual_q25,
    "q75": evaluation_residual_q75,
    "iqr": evaluation_residual_iqr,
    "max": evaluation_residual_max,
    "absolute_mean": evaluation_residual_abs_mean,
    "positive_count": evaluation_residual_positive_count,
    "negative_count": evaluation_residual_negative_count,
    "zero_count": evaluation_residual_zero_count,
} # Consolidar estadísticos descriptivos

for statistic_name, statistic_value in evaluation_residual_descriptive_values.items():
    if isinstance(
        statistic_value,
        (float, int)
    ):
        if not np.isfinite(
            float(statistic_value)
        ):
            raise RuntimeError(
                f"El estadístico {statistic_name} no es finito."
            ) # Validar estabilidad de estadísticos

if evaluation_residual_positive_count + evaluation_residual_negative_count + evaluation_residual_zero_count != evaluation_residual_observation_count:
    raise RuntimeError(
        "La clasificación de residuos positivos, negativos y cero no cubre todas las observaciones."
    ) # Validar partición de residuos

if evaluation_residual_iqr < 0:
    raise RuntimeError(
        "El rango intercuartílico no puede ser negativo."
    ) # Validar IQR

evaluation_block_12_status = "DESCRIPTIVES_VALIDATED" # Registrar estadísticos aprobados
evaluation_block_12_stage = "DESCRIPTIVES_VALIDATED" # Registrar etapa aprobada

print(f"Observaciones            : {evaluation_residual_observation_count}") # Mostrar observaciones
print(f"Media                    : {evaluation_residual_mean:.12f}") # Mostrar media
print(f"Mediana                  : {evaluation_residual_median:.12f}") # Mostrar mediana
print(f"Desviación estándar      : {evaluation_residual_std:.12f}") # Mostrar desviación
print(f"Varianza                 : {evaluation_residual_variance:.12f}") # Mostrar varianza
print(f"Mínimo                   : {evaluation_residual_min:.12f}") # Mostrar mínimo
print(f"Q25                      : {evaluation_residual_q25:.12f}") # Mostrar primer cuartil
print(f"Q75                      : {evaluation_residual_q75:.12f}") # Mostrar tercer cuartil
print(f"IQR                      : {evaluation_residual_iqr:.12f}") # Mostrar rango intercuartílico
print(f"Máximo                   : {evaluation_residual_max:.12f}") # Mostrar máximo
print(f"Media absoluta           : {evaluation_residual_abs_mean:.12f}") # Mostrar media absoluta
print(f"Residuos positivos       : {evaluation_residual_positive_count}") # Mostrar positivos
print(f"Residuos negativos       : {evaluation_residual_negative_count}") # Mostrar negativos
print(f"Residuos cero            : {evaluation_residual_zero_count}") # Mostrar ceros
print(f"Tasa positivos           : {evaluation_residual_positive_rate:.6f}") # Mostrar proporción positiva
print(f"Tasa negativos           : {evaluation_residual_negative_rate:.6f}") # Mostrar proporción negativa
print(f"Tasa cero                : {evaluation_residual_zero_rate:.6f}") # Mostrar proporción cero
print("Estadísticos             : FINITOS") # Confirmar estabilidad
print("Cobertura                : VALIDADA") # Confirmar cobertura
print("BLOQUE 12.2              : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 12.3. ANÁLISIS DE SESGO DE LOS RESIDUOS") # Mostrar encabezado
if evaluation_block_12_status != "DESCRIPTIVES_VALIDATED":
    raise RuntimeError(
        f"El Bloque 12.2 no fue validado. Estado actual: {evaluation_block_12_status}"
    ) # Impedir análisis sin estadísticos descriptivos validados

evaluation_block_12_stage = "RESIDUAL_BIAS" # Registrar etapa actual

if not isinstance(
    residuals,
    torch.Tensor
):
    raise TypeError(
        "residuals debe ser un tensor de PyTorch."
    ) # Validar residuos

if residuals.ndim != 1:
    raise ValueError(
        "residuals debe ser un vector unidimensional."
    ) # Validar dimensión

if len(
    residuals
) == 0:
    raise RuntimeError(
        "residuals está vacío."
    ) # Validar disponibilidad

if len(
    residuals
) != evaluation_residual_observation_count:
    raise RuntimeError(
        "La cantidad de residuos no coincide con las observaciones TEST."
    ) # Validar cobertura

if not torch.isfinite(
    residuals
).all():
    raise RuntimeError(
        "residuals contiene valores no finitos."
    ) # Validar estabilidad

residual_positive_mask = (
    residuals > 0
) # Identificar residuos positivos

residual_negative_mask = (
    residuals < 0
) # Identificar residuos negativos

residual_zero_mask = (
    residuals == 0
) # Identificar residuos exactamente nulos

residual_positive_count = int(
    residual_positive_mask.sum().item()
) # Contar subestimaciones

residual_negative_count = int(
    residual_negative_mask.sum().item()
) # Contar sobreestimaciones

residual_zero_count = int(
    residual_zero_mask.sum().item()
) # Contar predicciones exactas

residual_positive_proportion = (
    residual_positive_count
    / len(residuals)
) # Calcular proporción positiva

residual_negative_proportion = (
    residual_negative_count
    / len(residuals)
) # Calcular proporción negativa

residual_zero_proportion = (
    residual_zero_count
    / len(residuals)
) # Calcular proporción nula

if residual_positive_count > 0:
    residual_positive_mean = float(
        residuals[
            residual_positive_mask
        ].mean().item()
    ) # Calcular magnitud media de subestimación
else:
    residual_positive_mean = 0.0 # Registrar ausencia de residuos positivos

if residual_negative_count > 0:
    residual_negative_mean = float(
        residuals[
            residual_negative_mask
        ].mean().item()
    ) # Calcular magnitud media de sobreestimación
else:
    residual_negative_mean = 0.0 # Registrar ausencia de residuos negativos

residual_bias = float(
    residuals.mean().item()
) # Estimar sesgo medio

if residual_bias > 0:
    residual_bias_direction = "SUBESTIMACIÓN MEDIA" # Clasificar dirección positiva
elif residual_bias < 0:
    residual_bias_direction = "SOBREESTIMACIÓN MEDIA" # Clasificar dirección negativa
else:
    residual_bias_direction = "SIN SESGO MEDIO" # Clasificar sesgo nulo

residual_bias_analysis = {
    "bias": residual_bias,
    "bias_direction": residual_bias_direction,
    "positive_count": residual_positive_count,
    "negative_count": residual_negative_count,
    "zero_count": residual_zero_count,
    "positive_proportion": residual_positive_proportion,
    "negative_proportion": residual_negative_proportion,
    "zero_proportion": residual_zero_proportion,
    "positive_mean": residual_positive_mean,
    "negative_mean": residual_negative_mean,
} # Consolidar análisis de sesgo

for analysis_name, analysis_value in residual_bias_analysis.items():
    if isinstance(
        analysis_value,
        (int, float)
    ):
        if not np.isfinite(
            analysis_value
        ):
            raise RuntimeError(
                f"El resultado '{analysis_name}' no es finito."
            ) # Validar estabilidad

if (
    residual_positive_count
    + residual_negative_count
    + residual_zero_count
) != len(
    residuals
):
    raise RuntimeError(
        "La clasificación de residuos no cubre todas las observaciones."
    ) # Validar cobertura

if not np.isclose(
    residual_positive_proportion
    + residual_negative_proportion
    + residual_zero_proportion,
    1.0,
    atol=1e-12
):
    raise RuntimeError(
        "Las proporciones de residuos no suman uno."
    ) # Validar partición proporcional

residual_bias_status = "VALIDATED" # Registrar estado del análisis

evaluation_block_12_status = "BIAS_VALIDATED" # Registrar sesgo aprobado
evaluation_block_12_stage = "BIAS_VALIDATED" # Registrar etapa aprobada

print(f"Sesgo medio              : {residual_bias:.12f}") # Mostrar sesgo
print(f"Dirección del sesgo      : {residual_bias_direction}") # Mostrar dirección
print(f"Residuos positivos       : {residual_positive_count}") # Mostrar positivos
print(f"Residuos negativos       : {residual_negative_count}") # Mostrar negativos
print(f"Residuos nulos           : {residual_zero_count}") # Mostrar nulos
print(f"Proporción positivos     : {residual_positive_proportion:.12f}") # Mostrar proporción positiva
print(f"Proporción negativos     : {residual_negative_proportion:.12f}") # Mostrar proporción negativa
print(f"Proporción nulos         : {residual_zero_proportion:.12f}") # Mostrar proporción nula
print(f"Media positivos          : {residual_positive_mean:.12f}") # Mostrar magnitud positiva
print(f"Media negativos          : {residual_negative_mean:.12f}") # Mostrar magnitud negativa
print("Clasificación            : COMPLETA") # Confirmar cobertura
print("Proporciones             : VALIDADAS") # Confirmar proporciones
print("Estado del análisis      : VALIDATED") # Mostrar estado
print("BLOQUE 12.3              : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 12.4. ANÁLISIS DE DISPERSIÓN Y EXTREMOS DE LOS RESIDUOS") # Mostrar encabezado
if evaluation_block_12_status != "BIAS_VALIDATED":
    raise RuntimeError(
        f"El Bloque 12.3 no fue validado. Estado actual: {evaluation_block_12_status}"
    ) # Impedir continuidad sin sesgo validado

evaluation_block_12_stage = "DISPERSION_EXTREMES" # Registrar etapa actual

if not isinstance(
    residuals,
    torch.Tensor
):
    raise TypeError(
        "residuals debe ser un tensor de PyTorch."
    ) # Validar residuos

if residuals.ndim != 1:
    raise ValueError(
        "residuals debe ser un vector unidimensional."
    ) # Validar dimensión

if len(
    residuals
) == 0:
    raise RuntimeError(
        "residuals está vacío."
    ) # Validar disponibilidad

if len(
    residuals
) != evaluation_residual_observation_count:
    raise RuntimeError(
        "La cantidad de residuos no coincide con las observaciones TEST."
    ) # Validar cobertura

if not torch.isfinite(
    residuals
).all():
    raise RuntimeError(
        "residuals contiene valores no finitos."
    ) # Validar estabilidad

residual_abs = torch.abs(
    residuals
) # Calcular magnitud absoluta de los residuos

if not torch.isfinite(
    residual_abs
).all():
    raise RuntimeError(
        "La magnitud absoluta de los residuos contiene valores no finitos."
    ) # Validar estabilidad

residual_abs_sorted, residual_abs_order = torch.sort(
    residual_abs,
    descending=True
) # Ordenar residuos por magnitud absoluta

residual_extreme_count = min(
    10,
    len(residuals)
) # Determinar cantidad máxima de extremos

residual_extreme_indices = residual_abs_order[
    :residual_extreme_count
] # Recuperar índices de residuos extremos

residual_extreme_values = residuals[
    residual_extreme_indices
] # Recuperar valores extremos

residual_extreme_absolute_values = residual_abs[
    residual_extreme_indices
] # Recuperar magnitudes extremas

residual_abs_mean = float(
    residual_abs.mean().item()
) # Calcular magnitud absoluta media

residual_abs_median = float(
    residual_abs.median().item()
) # Calcular mediana absoluta

residual_abs_max = float(
    residual_abs.max().item()
) # Obtener máximo absoluto

residual_abs_q95 = float(
    torch.quantile(
        residual_abs,
        torch.tensor(
            0.95,
            dtype=residual_abs.dtype
        )
    ).item()
) # Calcular percentil 95 absoluto

residual_abs_q99 = float(
    torch.quantile(
        residual_abs,
        torch.tensor(
            0.99,
            dtype=residual_abs.dtype
        )
    ).item()
) # Calcular percentil 99 absoluto

if not np.isfinite(
    residual_abs_mean
):
    raise RuntimeError(
        "La media absoluta no es finita."
    ) # Validar media absoluta

if not np.isfinite(
    residual_abs_median
):
    raise RuntimeError(
        "La mediana absoluta no es finita."
    ) # Validar mediana absoluta

if not np.isfinite(
    residual_abs_max
):
    raise RuntimeError(
        "El máximo absoluto no es finito."
    ) # Validar máximo absoluto

if not np.isfinite(
    residual_abs_q95
):
    raise RuntimeError(
        "El percentil 95 absoluto no es finito."
    ) # Validar percentil 95

if not np.isfinite(
    residual_abs_q99
):
    raise RuntimeError(
        "El percentil 99 absoluto no es finito."
    ) # Validar percentil 99

if residual_extreme_count == 0:
    raise RuntimeError(
        "No se pudieron identificar residuos extremos."
    ) # Validar disponibilidad de extremos

if residual_extreme_values.shape[0] != residual_extreme_count:
    raise RuntimeError(
        "La cantidad de residuos extremos no coincide con la cantidad esperada."
    ) # Validar extremos

if not torch.isfinite(
    residual_extreme_values
).all():
    raise RuntimeError(
        "Los valores extremos contienen valores no finitos."
    ) # Validar extremos

if not torch.isfinite(
    residual_extreme_absolute_values
).all():
    raise RuntimeError(
        "Las magnitudes extremas contienen valores no finitos."
    ) # Validar extremos

if residual_abs_q95 > residual_abs_q99:
    raise RuntimeError(
        "El percentil 95 absoluto no puede superar al percentil 99 absoluto."
    ) # Validar orden de percentiles

if residual_abs_q99 > residual_abs_max:
    raise RuntimeError(
        "El percentil 99 absoluto no puede superar el máximo absoluto."
    ) # Validar orden estadístico

residual_dispersion_analysis = {
    "absolute_mean": residual_abs_mean,
    "absolute_median": residual_abs_median,
    "absolute_max": residual_abs_max,
    "absolute_q95": residual_abs_q95,
    "absolute_q99": residual_abs_q99,
    "extreme_count": residual_extreme_count,
} # Consolidar análisis de dispersión

for analysis_name, analysis_value in residual_dispersion_analysis.items():
    if not np.isfinite(
        float(analysis_value)
    ):
        raise RuntimeError(
            f"El resultado '{analysis_name}' no es finito."
        ) # Validar estabilidad de resultados

residual_dispersion_status = "VALIDATED" # Registrar estado del análisis
evaluation_block_12_status = "DISPERSION_VALIDATED" # Registrar dispersión aprobada
evaluation_block_12_stage = "DISPERSION_VALIDATED" # Registrar etapa aprobada

print(f"Observaciones            : {len(residuals)}") # Mostrar observaciones
print(f"Media absoluta           : {residual_abs_mean:.12f}") # Mostrar magnitud media
print(f"Mediana absoluta         : {residual_abs_median:.12f}") # Mostrar mediana absoluta
print(f"Máximo absoluto          : {residual_abs_max:.12f}") # Mostrar máximo absoluto
print(f"Q95 absoluto             : {residual_abs_q95:.12f}") # Mostrar percentil 95
print(f"Q99 absoluto             : {residual_abs_q99:.12f}") # Mostrar percentil 99
print(f"Extremos identificados   : {residual_extreme_count}") # Mostrar cantidad de extremos
print(f"Mayor residuo            : {float(residual_extreme_values[0].item()):.12f}") # Mostrar mayor residuo firmado
print(f"Mayor magnitud           : {float(residual_extreme_absolute_values[0].item()):.12f}") # Mostrar mayor error absoluto
print("Valores extremos        : FINITOS") # Confirmar estabilidad
print("Orden estadístico       : VALIDADO") # Confirmar orden de percentiles
print("Análisis de dispersión  : VALIDADO") # Confirmar producto
print("BLOQUE 12.4             : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 12.5. CONSOLIDACIÓN DEL ANÁLISIS DE RESIDUOS") # Mostrar encabezado
if evaluation_block_12_status != "DISPERSION_VALIDATED":
    raise RuntimeError(
        f"El Bloque 12.4 no fue validado. Estado actual: {evaluation_block_12_status}"
    ) # Impedir consolidación sin dispersión validada

if residual_bias_status != "VALIDATED":
    raise RuntimeError(
        f"El análisis de sesgo no está VALIDATED. Estado actual: {residual_bias_status}"
    ) # Validar análisis de sesgo

if residual_dispersion_status != "VALIDATED":
    raise RuntimeError(
        f"El análisis de dispersión no está VALIDATED. Estado actual: {residual_dispersion_status}"
    ) # Validar análisis de dispersión

evaluation_block_12_stage = "CONSOLIDATION" # Registrar etapa actual

if not isinstance(
    residual_bias_analysis,
    dict
):
    raise TypeError(
        "residual_bias_analysis debe ser un diccionario."
    ) # Validar análisis de sesgo

if not isinstance(
    residual_dispersion_analysis,
    dict
):
    raise TypeError(
        "residual_dispersion_analysis debe ser un diccionario."
    ) # Validar análisis de dispersión

if len(
    residuals
) != evaluation_residual_observation_count:
    raise RuntimeError(
        "La cantidad de residuos no coincide con las observaciones TEST."
    ) # Validar cobertura

if evaluation_residual_test_index.tolist() != evaluation_test_index.tolist():
    raise RuntimeError(
        "La partición TEST de los residuos no coincide con evaluation_test_index."
    ) # Validar partición

if PREDICTION_RESULT[
    "prediction_source"
] != "evaluation_graphsage":
    raise RuntimeError(
        "La fuente de predicción no corresponde a evaluation_graphsage."
    ) # Validar procedencia

RESIDUAL_ANALYSIS_RESULT = {
    "model_code": PREDICTION_RESULT["model_code"],
    "model_name": PREDICTION_RESULT["model_name"],
    "family": PREDICTION_RESULT["family"],
    "test_index": evaluation_residual_test_index.copy(),
    "test_nodes": int(evaluation_residual_observation_count),
    "n_observations": int(evaluation_residual_observation_count),
    "prediction_source": PREDICTION_RESULT["prediction_source"],
    "residual_definition": "y_true - y_pred",
    "residual_mean": float(evaluation_residual_mean),
    "residual_median": float(evaluation_residual_median),
    "residual_std": float(evaluation_residual_std),
    "residual_variance": float(evaluation_residual_variance),
    "residual_min": float(evaluation_residual_min),
    "residual_q25": float(evaluation_residual_q25),
    "residual_q75": float(evaluation_residual_q75),
    "residual_iqr": float(evaluation_residual_iqr),
    "residual_max": float(evaluation_residual_max),
    "residual_absolute_mean": float(residual_abs_mean),
    "residual_absolute_median": float(residual_abs_median),
    "residual_absolute_max": float(residual_abs_max),
    "residual_absolute_q95": float(residual_abs_q95),
    "residual_absolute_q99": float(residual_abs_q99),
    "residual_positive_count": int(residual_positive_count),
    "residual_negative_count": int(residual_negative_count),
    "residual_zero_count": int(residual_zero_count),
    "residual_positive_proportion": float(residual_positive_proportion),
    "residual_negative_proportion": float(residual_negative_proportion),
    "residual_zero_proportion": float(residual_zero_proportion),
    "residual_positive_mean": float(residual_positive_mean),
    "residual_negative_mean": float(residual_negative_mean),
    "bias_analysis": residual_bias_analysis,
    "dispersion_analysis": residual_dispersion_analysis,
    "extreme_count": int(residual_extreme_count),
    "status": "VALIDATED",
} # Consolidar análisis residual

required_residual_result_fields = [
    "model_code",
    "model_name",
    "family",
    "test_index",
    "test_nodes",
    "n_observations",
    "prediction_source",
    "residual_definition",
    "residual_mean",
    "residual_median",
    "residual_std",
    "residual_variance",
    "residual_min",
    "residual_q25",
    "residual_q75",
    "residual_iqr",
    "residual_max",
    "residual_absolute_mean",
    "residual_absolute_median",
    "residual_absolute_max",
    "residual_absolute_q95",
    "residual_absolute_q99",
    "residual_positive_count",
    "residual_negative_count",
    "residual_zero_count",
    "residual_positive_proportion",
    "residual_negative_proportion",
    "residual_zero_proportion",
    "bias_analysis",
    "dispersion_analysis",
    "extreme_count",
    "status",
] # Definir campos obligatorios

missing_residual_result_fields = [
    field
    for field in required_residual_result_fields
    if field not in RESIDUAL_ANALYSIS_RESULT
] # Identificar campos faltantes

if missing_residual_result_fields:
    raise RuntimeError(
        f"RESIDUAL_ANALYSIS_RESULT presenta campos faltantes: {missing_residual_result_fields}"
    ) # Validar cobertura

if RESIDUAL_ANALYSIS_RESULT[
    "test_nodes"
] != RESIDUAL_ANALYSIS_RESULT[
    "n_observations"
]:
    raise RuntimeError(
        "Los nodos TEST no coinciden con las observaciones."
    ) # Validar cobertura

if RESIDUAL_ANALYSIS_RESULT[
    "test_index"
].tolist() != evaluation_residual_test_index.tolist():
    raise RuntimeError(
        "La partición consolidada no coincide con la partición de residuos."
    ) # Validar trazabilidad

for field_name in [
    "residual_mean",
    "residual_median",
    "residual_std",
    "residual_variance",
    "residual_min",
    "residual_q25",
    "residual_q75",
    "residual_iqr",
    "residual_max",
    "residual_absolute_mean",
    "residual_absolute_median",
    "residual_absolute_max",
    "residual_absolute_q95",
    "residual_absolute_q99",
]:
    if not np.isfinite(
        RESIDUAL_ANALYSIS_RESULT[field_name]
    ):
        raise RuntimeError(
            f"El campo {field_name} no es finito."
        ) # Validar estabilidad

if RESIDUAL_ANALYSIS_RESULT[
    "residual_iqr"
] < 0:
    raise RuntimeError(
        "El IQR de los residuos no puede ser negativo."
    ) # Validar rango

if RESIDUAL_ANALYSIS_RESULT[
    "residual_absolute_q95"
] > RESIDUAL_ANALYSIS_RESULT[
    "residual_absolute_q99"
]:
    raise RuntimeError(
        "Q95 absoluto no puede superar Q99 absoluto."
    ) # Validar percentiles

if RESIDUAL_ANALYSIS_RESULT[
    "residual_absolute_q99"
] > RESIDUAL_ANALYSIS_RESULT[
    "residual_absolute_max"
]:
    raise RuntimeError(
        "Q99 absoluto no puede superar el máximo absoluto."
    ) # Validar orden estadístico

if (
    RESIDUAL_ANALYSIS_RESULT["residual_positive_count"]
    + RESIDUAL_ANALYSIS_RESULT["residual_negative_count"]
    + RESIDUAL_ANALYSIS_RESULT["residual_zero_count"]
) != RESIDUAL_ANALYSIS_RESULT[
    "n_observations"
]:
    raise RuntimeError(
        "La clasificación de residuos no cubre todas las observaciones."
    ) # Validar cobertura

if RESIDUAL_ANALYSIS_RESULT[
    "prediction_source"
] != "evaluation_graphsage":
    raise RuntimeError(
        "La fuente consolidada no corresponde a evaluation_graphsage."
    ) # Validar independencia

if RESIDUAL_ANALYSIS_RESULT[
    "status"
] != "VALIDATED":
    raise RuntimeError(
        "RESIDUAL_ANALYSIS_RESULT no presenta estado VALIDATED."
    ) # Validar estado final

evaluation_block_12_status = "VALIDATED" # Registrar aprobación final del Bloque 12
evaluation_block_12_stage = "VALIDATED" # Registrar cierre definitivo

print(f"Código del modelo        : {RESIDUAL_ANALYSIS_RESULT['model_code']}") # Mostrar código
print(f"Nombre del modelo        : {RESIDUAL_ANALYSIS_RESULT['model_name']}") # Mostrar modelo
print(f"Familia                  : {RESIDUAL_ANALYSIS_RESULT['family']}") # Mostrar familia
print(f"Índices TEST             : {RESIDUAL_ANALYSIS_RESULT['test_index'].tolist()}") # Mostrar partición
print(f"Observaciones            : {RESIDUAL_ANALYSIS_RESULT['n_observations']}") # Mostrar observaciones
print(f"Fuente                   : {RESIDUAL_ANALYSIS_RESULT['prediction_source']}") # Mostrar fuente
print(f"Definición residuos      : {RESIDUAL_ANALYSIS_RESULT['residual_definition']}") # Mostrar definición
print(f"Sesgo medio              : {RESIDUAL_ANALYSIS_RESULT['residual_mean']:.12f}") # Mostrar sesgo
print(f"Media absoluta           : {RESIDUAL_ANALYSIS_RESULT['residual_absolute_mean']:.12f}") # Mostrar magnitud
print(f"Q95 absoluto             : {RESIDUAL_ANALYSIS_RESULT['residual_absolute_q95']:.12f}") # Mostrar Q95
print(f"Q99 absoluto             : {RESIDUAL_ANALYSIS_RESULT['residual_absolute_q99']:.12f}") # Mostrar Q99
print(f"Máximo absoluto          : {RESIDUAL_ANALYSIS_RESULT['residual_absolute_max']:.12f}") # Mostrar máximo
print(f"Residuos positivos       : {RESIDUAL_ANALYSIS_RESULT['residual_positive_count']}") # Mostrar positivos
print(f"Residuos negativos       : {RESIDUAL_ANALYSIS_RESULT['residual_negative_count']}") # Mostrar negativos
print(f"Residuos nulos           : {RESIDUAL_ANALYSIS_RESULT['residual_zero_count']}") # Mostrar nulos
print(f"Extremos identificados   : {RESIDUAL_ANALYSIS_RESULT['extreme_count']}") # Mostrar extremos
print(f"Estado                   : {RESIDUAL_ANALYSIS_RESULT['status']}") # Mostrar estado
print("Trazabilidad             : VALIDADA") # Confirmar trazabilidad
print("Cobertura TEST           : VALIDADA") # Confirmar cobertura
print("Análisis de residuos     : CONSOLIDADO") # Confirmar consolidación
print("BLOQUE 12                : VALIDATED") # Confirmar cierre

# BLOQUE 13. VALIDACIÓN Y ESTRUCTURACIÓN TEMPORAL DE LA EVALUACIÓN
# Objetivo: Verificar y estructurar la dimensión temporal del conjunto TEST para permitir la evaluación del desempeño del Modelo Oficial por año.
# Entradas: evaluation_test_graphs, evaluation_test_index, PREDICTION_RESULT y GraphData TEST.
# Producto: Estructura temporal validada para la construcción posterior de metrics_by_year.
# Pregunta científica: ¿El conjunto TEST conserva una estructura temporal válida y suficiente para evaluar el desempeño del Modelo Oficial durante los años 2016, 2017 y 2018?

print("\nBLOQUE 13. DIAGNÓSTICO PREVIO PARA LAS MÉTRICAS TEMPORALES") # Mostrar encabezado

print("\nBLOQUE 13.0. DIAGNÓSTICO PREVIO DE LA ESTRUCTURA TEMPORAL") # Mostrar encabezado

evaluation_block_13_0_status = "ERROR" # Inicializar estado del subbloque
evaluation_block_13_0_stage = "DIAGNOSTICO" # Registrar etapa actual

if evaluation_block_12_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 12 no está VALIDATED. Estado actual: {evaluation_block_12_status}"
    ) # Validar dependencia del Bloque 12

if not isinstance(evaluation_test_graphs, (list, tuple)):
    raise TypeError(
        "evaluation_test_graphs debe ser una lista o tupla."
    ) # Validar colección TEST

if len(evaluation_test_graphs) == 0:
    raise RuntimeError(
        "evaluation_test_graphs está vacío."
    ) # Validar disponibilidad de GraphData TEST

if not isinstance(evaluation_test_index, np.ndarray):
    raise TypeError(
        "evaluation_test_index debe ser un ndarray."
    ) # Validar índice TEST

if len(evaluation_test_index) == 0:
    raise RuntimeError(
        "evaluation_test_index está vacío."
    ) # Validar disponibilidad del índice TEST

if len(evaluation_test_graphs) != len(evaluation_test_index):
    raise RuntimeError(
        "La cantidad de GraphData TEST no coincide con evaluation_test_index."
    ) # Validar correspondencia entre GraphData e índices

if not isinstance(PREDICTION_RESULT, dict):
    raise TypeError(
        "PREDICTION_RESULT debe ser un diccionario."
    ) # Validar estructura de predicciones

if PREDICTION_RESULT.get("status") != "VALIDATED":
    raise RuntimeError(
        "PREDICTION_RESULT debe presentar estado VALIDATED."
    ) # Validar estado de las predicciones

required_temporal_prediction_fields = [
    "model_code",
    "model_name",
    "family",
    "test_index",
    "y_true",
    "y_pred",
    "n_observations",
    "prediction_source",
    "status",
] # Definir campos temporales requeridos

missing_temporal_prediction_fields = [
    field
    for field in required_temporal_prediction_fields
    if field not in PREDICTION_RESULT
] # Identificar campos temporales faltantes

if missing_temporal_prediction_fields:
    raise RuntimeError(
        f"PREDICTION_RESULT presenta campos faltantes: {missing_temporal_prediction_fields}"
    ) # Validar cobertura del contrato

if PREDICTION_RESULT["test_index"].tolist() != evaluation_test_index.tolist():
    raise RuntimeError(
        "El test_index de PREDICTION_RESULT no coincide con evaluation_test_index."
    ) # Validar trazabilidad de la partición TEST

if len(PREDICTION_RESULT["y_true"]) != len(PREDICTION_RESULT["y_pred"]):
    raise RuntimeError(
        "y_true y y_pred no presentan la misma longitud."
    ) # Validar correspondencia entre observaciones y predicciones

if PREDICTION_RESULT["n_observations"] != len(PREDICTION_RESULT["y_true"]):
    raise RuntimeError(
        "n_observations no coincide con la cantidad de y_true."
    ) # Validar trazabilidad de observaciones

if PREDICTION_RESULT["n_observations"] != len(PREDICTION_RESULT["y_pred"]):
    raise RuntimeError(
        "n_observations no coincide con la cantidad de y_pred."
    ) # Validar trazabilidad de predicciones

if PREDICTION_RESULT["prediction_source"] != "evaluation_graphsage":
    raise RuntimeError(
        "La fuente de predicción no corresponde a evaluation_graphsage."
    ) # Validar procedencia oficial de las predicciones

evaluation_temporal_graph_years = [] # Inicializar años TEST
evaluation_temporal_graph_node_counts = [] # Inicializar nodos por GraphData

for test_position, graph in enumerate(evaluation_test_graphs, start=1):
    if not isinstance(graph, Data):
        raise TypeError(
            f"El GraphData TEST {test_position} no es una instancia de Data."
        ) # Validar tipo GraphData

    if not hasattr(graph, "x"):
        raise RuntimeError(
            f"El GraphData TEST {test_position} no contiene x."
        ) # Validar variables predictoras

    if not hasattr(graph, "y"):
        raise RuntimeError(
            f"El GraphData TEST {test_position} no contiene y."
        ) # Validar variable objetivo

    if not hasattr(graph, "current_year"):
        raise RuntimeError(
            f"El GraphData TEST {test_position} no contiene current_year."
        ) # Validar fuente temporal oficial

    graph_node_count = int(graph.x.shape[0]) # Recuperar número de nodos

    if graph_node_count <= 0:
        raise RuntimeError(
            f"El GraphData TEST {test_position} no contiene nodos."
        ) # Validar disponibilidad de nodos

    graph_year = graph.current_year # Recuperar año temporal oficial

    if isinstance(graph_year, torch.Tensor):
        if graph_year.numel() != 1:
            raise RuntimeError(
                f"current_year del GraphData TEST {test_position} no es escalar."
            ) # Validar año escalar

        graph_year = int(graph_year.item()) # Normalizar año tensor

    elif isinstance(graph_year, np.ndarray):
        if graph_year.size != 1:
            raise RuntimeError(
                f"current_year del GraphData TEST {test_position} no es escalar."
            ) # Validar año escalar

        graph_year = int(graph_year.item()) # Normalizar año ndarray

    elif isinstance(graph_year, (int, np.integer)):
        graph_year = int(graph_year) # Normalizar año entero

    else:
        raise TypeError(
            f"current_year del GraphData TEST {test_position} presenta tipo no válido: {type(graph_year).__name__}"
        ) # Validar tipo temporal

    if graph_year < 2006 or graph_year > 2018:
        raise RuntimeError(
            f"El año {graph_year} está fuera del periodo científico 2006-2018."
        ) # Validar periodo científico

    evaluation_temporal_graph_years.append(graph_year) # Registrar año TEST
    evaluation_temporal_graph_node_counts.append(graph_node_count) # Registrar nodos TEST

evaluation_temporal_total_nodes = int(
    sum(evaluation_temporal_graph_node_counts)
) # Calcular observaciones temporales

if evaluation_temporal_total_nodes != len(PREDICTION_RESULT["y_true"]):
    raise RuntimeError(
        "La cantidad total de nodos TEST no coincide con y_true."
    ) # Validar cobertura de observaciones

if evaluation_temporal_total_nodes != len(PREDICTION_RESULT["y_pred"]):
    raise RuntimeError(
        "La cantidad total de nodos TEST no coincide con y_pred."
    ) # Validar cobertura de predicciones

evaluation_temporal_years = sorted(
    set(evaluation_temporal_graph_years)
) # Obtener años temporales únicos

if len(evaluation_temporal_years) != len(evaluation_temporal_graph_years):
    raise RuntimeError(
        "Existen años duplicados entre los GraphData TEST."
    ) # Validar unicidad temporal

if not np.isfinite(
    np.asarray(evaluation_temporal_graph_years, dtype=np.float64)
).all():
    raise RuntimeError(
        "Los años TEST contienen valores no finitos."
    ) # Validar estabilidad temporal

evaluation_block_13_0_status = "VALIDATED" # Registrar diagnóstico validado
evaluation_block_13_0_stage = "DIAGNOSTICO_VALIDADO" # Registrar etapa aprobada
evaluation_block_13_status = "DIAGNOSTIC_VALIDATED" # Registrar estado provisional del Bloque 13
evaluation_block_13_stage = "DIAGNOSTICO_VALIDADO" # Registrar etapa global provisional

print(f"Estado Bloque 12        : {evaluation_block_12_status}") # Mostrar dependencia
print(f"GraphData TEST          : {len(evaluation_test_graphs)}") # Mostrar cantidad
print(f"Índices TEST            : {evaluation_test_index.tolist()}") # Mostrar índices
print(f"Años TEST               : {evaluation_temporal_graph_years}") # Mostrar años
print(f"Años disponibles        : {evaluation_temporal_years}") # Mostrar años únicos
print(f"Nodos por GraphData     : {evaluation_temporal_graph_node_counts}") # Mostrar nodos
print(f"Nodos TEST              : {evaluation_temporal_total_nodes}") # Mostrar observaciones
print(f"Observaciones y_true    : {len(PREDICTION_RESULT['y_true'])}") # Mostrar valores reales
print(f"Observaciones y_pred    : {len(PREDICTION_RESULT['y_pred'])}") # Mostrar predicciones
print(f"Fuente de predicción    : {PREDICTION_RESULT['prediction_source']}") # Mostrar procedencia
print("Correspondencia temporal : VALIDADA") # Confirmar correspondencia
print(f"Estado Bloque 13.0      : {evaluation_block_13_0_status}") # Mostrar estado
print("BLOQUE 13.0             : VALIDATED") # Confirmar aprobación

print("\nBLOQUE 13.1. DIAGNÓSTICO DE LA ESTRUCTURA TEMPORAL DE GRAPH DATA") # Mostrar encabezado
evaluation_block_13_1_status = "ERROR" # Inicializar estado del subbloque
evaluation_block_13_1_stage = "DIAGNOSTICO_GRAPH_DATA" # Registrar etapa actual

if evaluation_block_13_0_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 13.0 no está VALIDATED. Estado actual: {evaluation_block_13_0_status}"
    ) # Validar dependencia del Bloque 13.0

if len(evaluation_test_graphs) != len(evaluation_temporal_graph_years):
    raise RuntimeError(
        "La cantidad de GraphData no coincide con la cantidad de años temporales."
    ) # Validar correspondencia temporal

if len(evaluation_test_graphs) != len(evaluation_test_index):
    raise RuntimeError(
        "La cantidad de GraphData no coincide con evaluation_test_index."
    ) # Validar correspondencia con índice TEST

evaluation_graph_temporal_structure = [] # Inicializar estructura temporal

for position, (test_index, graph, graph_year) in enumerate(
    zip(
        evaluation_test_index,
        evaluation_test_graphs,
        evaluation_temporal_graph_years
    ),
    start=1
):
    if not isinstance(graph, Data):
        raise TypeError(
            f"El GraphData {position} no corresponde a torch_geometric.data.Data."
        ) # Validar tipo GraphData

    if not hasattr(graph, "x"):
        raise RuntimeError(
            f"El GraphData {position} no contiene x."
        ) # Validar variables predictoras

    if not hasattr(graph, "y"):
        raise RuntimeError(
            f"El GraphData {position} no contiene y."
        ) # Validar variable objetivo

    if not hasattr(graph, "current_year"):
        raise RuntimeError(
            f"El GraphData {position} no contiene current_year."
        ) # Validar dimensión temporal

    if graph.x.ndim != 2:
        raise RuntimeError(
            f"El GraphData {position} presenta x con {graph.x.ndim} dimensiones; se requieren 2."
        ) # Validar estructura de variables

    if graph.y.ndim == 0:
        graph_target_count = 1 # Registrar objetivo escalar
    else:
        graph_target_count = int(graph.y.numel()) # Registrar cantidad de objetivos

    graph_node_count = int(graph.x.shape[0]) # Recuperar número de nodos
    graph_feature_count = int(graph.x.shape[1]) # Recuperar número de variables

    if graph_node_count <= 0:
        raise RuntimeError(
            f"El GraphData {position} no contiene nodos."
        ) # Validar nodos

    if graph_feature_count <= 0:
        raise RuntimeError(
            f"El GraphData {position} no contiene variables predictoras."
        ) # Validar variables predictoras

    if graph_target_count != graph_node_count:
        raise RuntimeError(
            f"El GraphData {position} presenta {graph_target_count} objetivos para {graph_node_count} nodos."
        ) # Validar correspondencia objetivo nodos

    graph_current_year = graph.current_year # Recuperar año almacenado

    if isinstance(graph_current_year, torch.Tensor):
        if graph_current_year.numel() != 1:
            raise RuntimeError(
                f"current_year del GraphData {position} no es escalar."
            ) # Validar año escalar

        graph_current_year = int(graph_current_year.item()) # Normalizar año

    elif isinstance(graph_current_year, np.ndarray):
        if graph_current_year.size != 1:
            raise RuntimeError(
                f"current_year del GraphData {position} no es escalar."
            ) # Validar año escalar

        graph_current_year = int(graph_current_year.item()) # Normalizar año

    elif isinstance(graph_current_year, (int, np.integer)):
        graph_current_year = int(graph_current_year) # Normalizar año entero

    else:
        raise TypeError(
            f"current_year del GraphData {position} presenta tipo no válido: {type(graph_current_year).__name__}"
        ) # Validar tipo del año

    if graph_current_year != graph_year:
        raise RuntimeError(
            f"El año del GraphData {position} ({graph_current_year}) no coincide con el año registrado ({graph_year})."
        ) # Validar trazabilidad temporal

    evaluation_graph_temporal_structure.append(
        {
            "position": position,
            "test_index": int(test_index),
            "year": graph_year,
            "n_nodes": graph_node_count,
            "n_features": graph_feature_count,
            "n_targets": graph_target_count,
        }
    ) # Registrar estructura temporal del GraphData

evaluation_graph_years = [
    item["year"]
    for item in evaluation_graph_temporal_structure
] # Recuperar años

evaluation_graph_node_counts = [
    item["n_nodes"]
    for item in evaluation_graph_temporal_structure
] # Recuperar nodos

evaluation_graph_feature_counts = [
    item["n_features"]
    for item in evaluation_graph_temporal_structure
] # Recuperar variables

evaluation_graph_target_counts = [
    item["n_targets"]
    for item in evaluation_graph_temporal_structure
] # Recuperar objetivos

if len(set(evaluation_graph_years)) != len(evaluation_graph_years):
    raise RuntimeError(
        "La estructura temporal contiene años duplicados."
    ) # Validar unicidad temporal

if len(set(evaluation_graph_node_counts)) != 1:
    raise RuntimeError(
        "Los GraphData TEST no presentan una cantidad uniforme de nodos."
    ) # Validar estabilidad estructural

if len(set(evaluation_graph_feature_counts)) != 1:
    raise RuntimeError(
        "Los GraphData TEST no presentan una cantidad uniforme de variables."
    ) # Validar estabilidad de características

if evaluation_graph_node_counts != evaluation_graph_target_counts:
    raise RuntimeError(
        "La cantidad de nodos no coincide con la cantidad de objetivos por GraphData."
    ) # Validar correspondencia nodos objetivos

evaluation_block_13_1_status = "VALIDATED" # Registrar subbloque validado
evaluation_block_13_1_stage = "ESTRUCTURA_TEMPORAL_VALIDADA" # Registrar etapa validada

print(f"GraphData TEST             : {len(evaluation_graph_temporal_structure)}") # Mostrar GraphData
print(f"Índices TEST               : {evaluation_test_index.tolist()}") # Mostrar índices
print(f"Años                       : {evaluation_graph_years}") # Mostrar años
print(f"Nodos por GraphData        : {evaluation_graph_node_counts}") # Mostrar nodos
print(f"Variables por GraphData    : {evaluation_graph_feature_counts}") # Mostrar variables
print(f"Objetivos por GraphData    : {evaluation_graph_target_counts}") # Mostrar objetivos
print("Correspondencia año índice : VALIDADA") # Confirmar correspondencia temporal
print("Estructura de GraphData    : VALIDADA") # Confirmar estructura
print("Nodos entre años            : CONSISTENTES") # Confirmar estabilidad
print("Variables entre años       : CONSISTENTES") # Confirmar estabilidad
print("Objetivos por nodo          : CONSISTENTES") # Confirmar correspondencia
print(f"Estado Bloque 13.1         : {evaluation_block_13_1_status}") # Mostrar estado


print("\nBLOQUE 13.2. RECUPERACIÓN Y VALIDACIÓN DEL AÑO TEMPORAL") # Mostrar encabezado

evaluation_block_13_2_status = "ERROR" # Inicializar estado del subbloque
evaluation_block_13_2_stage = "RECUPERACION_ANO" # Registrar etapa actual

if evaluation_block_13_1_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 13.1 no está VALIDATED. Estado actual: {evaluation_block_13_1_status}"
    ) # Validar dependencia del Bloque 13.1

evaluation_temporal_year_by_test_index = {} # Inicializar correspondencia índice año
evaluation_temporal_year_records = [] # Inicializar registros temporales

for test_index, graph, validated_year in zip(
    evaluation_test_index,
    evaluation_test_graphs,
    evaluation_temporal_graph_years
):
    graph_year = graph.current_year # Recuperar año oficial del GraphData

    if isinstance(graph_year, torch.Tensor):
        if graph_year.numel() != 1:
            raise RuntimeError(
                f"current_year del índice TEST {int(test_index)} no es escalar."
            ) # Validar año escalar

        graph_year = int(graph_year.item()) # Normalizar año Tensor

    elif isinstance(graph_year, np.ndarray):
        if graph_year.size != 1:
            raise RuntimeError(
                f"current_year del índice TEST {int(test_index)} no es escalar."
            ) # Validar año escalar

        graph_year = int(graph_year.item()) # Normalizar año ndarray

    elif isinstance(graph_year, (int, np.integer)):
        graph_year = int(graph_year) # Normalizar año entero

    else:
        raise TypeError(
            f"current_year del índice TEST {int(test_index)} presenta tipo no válido: {type(graph_year).__name__}"
        ) # Validar tipo del año

    if graph_year != int(validated_year):
        raise RuntimeError(
            f"El año recuperado {graph_year} no coincide con el año validado {int(validated_year)} para el índice TEST {int(test_index)}."
        ) # Validar consistencia temporal

    if graph_year < 2006 or graph_year > 2018:
        raise RuntimeError(
            f"El año {graph_year} está fuera del periodo científico 2006-2018."
        ) # Validar rango temporal

    if int(test_index) in evaluation_temporal_year_by_test_index:
        raise RuntimeError(
            f"El índice TEST {int(test_index)} aparece más de una vez."
        ) # Validar unicidad del índice

    evaluation_temporal_year_by_test_index[int(test_index)] = graph_year # Registrar correspondencia

    evaluation_temporal_year_records.append(
        {
            "test_index": int(test_index),
            "year": graph_year,
        }
    ) # Registrar relación temporal

evaluation_temporal_year_records = sorted(
    evaluation_temporal_year_records,
    key=lambda record: record["test_index"]
) # Ordenar registros temporalmente por índice TEST

evaluation_temporal_test_indices = [
    record["test_index"]
    for record in evaluation_temporal_year_records
] # Recuperar índices ordenados

evaluation_temporal_year_values = [
    record["year"]
    for record in evaluation_temporal_year_records
] # Recuperar años ordenados

if evaluation_temporal_test_indices != evaluation_test_index.tolist():
    raise RuntimeError(
        "La secuencia de índices temporales no coincide con evaluation_test_index."
    ) # Validar orden de índices

if len(set(evaluation_temporal_year_values)) != len(
    evaluation_temporal_year_values
):
    raise RuntimeError(
        "Existen años temporales duplicados."
    ) # Validar unicidad de años

if evaluation_temporal_year_values != sorted(
    evaluation_temporal_year_values
):
    raise RuntimeError(
        "Los años TEST no están ordenados cronológicamente."
    ) # Validar orden cronológico

evaluation_block_13_2_status = "VALIDATED" # Registrar subbloque validado
evaluation_block_13_2_stage = "ANO_TEMPORAL_VALIDADO" # Registrar etapa validada

print(f"Índices TEST              : {evaluation_temporal_test_indices}") # Mostrar índices
print(f"Años recuperados          : {evaluation_temporal_year_values}") # Mostrar años
print(f"Correspondencia temporal  : {evaluation_temporal_year_by_test_index}") # Mostrar correspondencia
print("Años dentro del periodo   : VALIDADO") # Confirmar rango temporal
print("Unicidad de índices       : VALIDADA") # Confirmar unicidad
print("Unicidad de años          : VALIDADA") # Confirmar unicidad temporal
print("Orden cronológico         : VALIDADO") # Confirmar orden
print(f"Estado Bloque 13.2        : {evaluation_block_13_2_status}") # Mostrar estado

print("\nBLOQUE 13.3. VALIDACIÓN DE LA ASIGNACIÓN TEMPORAL ÍNDICE TEST AÑO") # Mostrar encabezado
evaluation_block_13_3_status = "ERROR" # Inicializar estado del subbloque
evaluation_block_13_3_stage = "ASIGNACION_TEMPORAL" # Registrar etapa actual

if evaluation_block_13_2_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 13.2 no está VALIDATED. Estado actual: {evaluation_block_13_2_status}"
    ) # Validar dependencia del Bloque 13.2

if not isinstance(evaluation_temporal_year_by_test_index, dict):
    raise TypeError(
        "evaluation_temporal_year_by_test_index debe ser un diccionario."
    ) # Validar estructura de asignación temporal

if not evaluation_temporal_year_by_test_index:
    raise RuntimeError(
        "evaluation_temporal_year_by_test_index está vacío."
    ) # Validar disponibilidad de asignaciones

prediction_test_indices = [
    int(index)
    for index in PREDICTION_RESULT["test_index"]
] # Recuperar índices de predicción

if prediction_test_indices != evaluation_test_index.tolist():
    raise RuntimeError(
        "Los índices TEST de PREDICTION_RESULT no coinciden con evaluation_test_index."
    ) # Validar trazabilidad de índices

missing_temporal_assignments = [
    index
    for index in prediction_test_indices
    if index not in evaluation_temporal_year_by_test_index
] # Identificar índices sin asignación temporal

if missing_temporal_assignments:
    raise RuntimeError(
        f"Existen índices TEST sin asignación temporal: {missing_temporal_assignments}"
    ) # Validar cobertura temporal

evaluation_temporal_assignment = [
    {
        "test_index": index,
        "year": evaluation_temporal_year_by_test_index[index],
    }
    for index in prediction_test_indices
] # Construir asignación oficial índice año

evaluation_temporal_prediction_years = [
    record["year"]
    for record in evaluation_temporal_assignment
] # Recuperar años asociados a predicciones

if len(evaluation_temporal_assignment) != len(prediction_test_indices):
    raise RuntimeError(
        "La cantidad de asignaciones temporales no coincide con la cantidad de índices TEST."
    ) # Validar cobertura de asignaciones

if evaluation_temporal_prediction_years != evaluation_temporal_year_values:
    raise RuntimeError(
        "Los años asignados a las predicciones no coinciden con los años temporales validados."
    ) # Validar consistencia temporal

if len(set(evaluation_temporal_prediction_years)) != len(
    evaluation_temporal_prediction_years
):
    raise RuntimeError(
        "Existen años duplicados en la asignación temporal de predicciones."
    ) # Validar unicidad temporal

evaluation_temporal_index_year_map = {
    record["test_index"]: record["year"]
    for record in evaluation_temporal_assignment
} # Construir mapa final índice año

evaluation_block_13_3_status = "VALIDATED" # Registrar subbloque validado
evaluation_block_13_3_stage = "ASIGNACION_TEMPORAL_VALIDADA" # Registrar etapa validada

print(f"Índices TEST               : {prediction_test_indices}") # Mostrar índices
print(f"Años asignados             : {evaluation_temporal_prediction_years}") # Mostrar años
print(f"Asignación índice año      : {evaluation_temporal_index_year_map}") # Mostrar asignación
print("Cobertura de índices       : VALIDADA") # Confirmar cobertura
print("Trazabilidad de predicción : VALIDADA") # Confirmar trazabilidad
print("Correspondencia índice año : VALIDADA") # Confirmar asignación
print("Unicidad temporal          : VALIDADA") # Confirmar unicidad
print(f"Estado Bloque 13.3         : {evaluation_block_13_3_status}") # Mostrar estado

print("\nBLOQUE 13.4. VALIDACIÓN DE LA ALINEACIÓN TEMPORAL DE Y_TRUE Y Y_PRED") # Mostrar encabezado

evaluation_block_13_4_status = "ERROR" # Inicializar estado del subbloque
evaluation_block_13_4_stage = "ALINEACION_TEMPORAL" # Registrar etapa actual

if evaluation_block_13_3_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 13.3 no está VALIDATED. Estado actual: {evaluation_block_13_3_status}"
    ) # Validar dependencia del Bloque 13.3

y_true_temporal = PREDICTION_RESULT["y_true"] # Recuperar valores observados
y_pred_temporal = PREDICTION_RESULT["y_pred"] # Recuperar predicciones

if len(y_true_temporal) != len(y_pred_temporal):
    raise RuntimeError(
        "y_true y y_pred no presentan la misma cantidad de observaciones."
    ) # Validar correspondencia global

evaluation_temporal_total_nodes = sum(
    item["n_nodes"]
    for item in evaluation_graph_temporal_structure
) # Calcular total de nodos TEST

if len(y_true_temporal) != evaluation_temporal_total_nodes:
    raise RuntimeError(
        f"y_true contiene {len(y_true_temporal)} observaciones y los GraphData TEST contienen {evaluation_temporal_total_nodes} nodos."
    ) # Validar cobertura de y_true

if len(y_pred_temporal) != evaluation_temporal_total_nodes:
    raise RuntimeError(
        f"y_pred contiene {len(y_pred_temporal)} observaciones y los GraphData TEST contienen {evaluation_temporal_total_nodes} nodos."
    ) # Validar cobertura de y_pred

evaluation_temporal_alignment = [] # Inicializar estructura de alineación

temporal_start = 0 # Inicializar posición inicial

for record in evaluation_graph_temporal_structure:
    test_index = record["test_index"] # Recuperar índice TEST
    year = evaluation_temporal_index_year_map[test_index] # Recuperar año asignado
    n_nodes = record["n_nodes"] # Recuperar nodos del GraphData

    temporal_end = temporal_start + n_nodes # Calcular posición final

    if temporal_end > len(y_true_temporal):
        raise RuntimeError(
            f"El rango temporal del año {year} excede la longitud de y_true."
        ) # Validar rango de y_true

    if temporal_end > len(y_pred_temporal):
        raise RuntimeError(
            f"El rango temporal del año {year} excede la longitud de y_pred."
        ) # Validar rango de y_pred

    evaluation_temporal_alignment.append(
        {
            "test_index": test_index,
            "year": year,
            "start": temporal_start,
            "end": temporal_end,
            "n_observations": n_nodes,
        }
    ) # Registrar alineación temporal

    temporal_start = temporal_end # Avanzar posición temporal

if temporal_start != len(y_true_temporal):
    raise RuntimeError(
        "La suma de observaciones temporales no cubre completamente y_true."
    ) # Validar cobertura completa de y_true

if temporal_start != len(y_pred_temporal):
    raise RuntimeError(
        "La suma de observaciones temporales no cubre completamente y_pred."
    ) # Validar cobertura completa de y_pred

evaluation_temporal_y_true_by_year = {} # Inicializar valores observados por año
evaluation_temporal_y_pred_by_year = {} # Inicializar predicciones por año

for alignment in evaluation_temporal_alignment:
    year = alignment["year"] # Recuperar año
    start = alignment["start"] # Recuperar posición inicial
    end = alignment["end"] # Recuperar posición final

    evaluation_temporal_y_true_by_year[year] = y_true_temporal[
        start:end
    ] # Asignar y_true al año

    evaluation_temporal_y_pred_by_year[year] = y_pred_temporal[
        start:end
    ] # Asignar y_pred al año

for year in evaluation_temporal_y_true_by_year:
    if len(evaluation_temporal_y_true_by_year[year]) != len(
        evaluation_temporal_y_pred_by_year[year]
    ):
        raise RuntimeError(
            f"y_true y y_pred no están alineados para el año {year}."
        ) # Validar alineación por año

    expected_count = next(
        item["n_observations"]
        for item in evaluation_temporal_alignment
        if item["year"] == year
    ) # Recuperar cantidad esperada

    if len(evaluation_temporal_y_true_by_year[year]) != expected_count:
        raise RuntimeError(
            f"La cantidad de observaciones del año {year} no coincide con su GraphData."
        ) # Validar cobertura por año

evaluation_temporal_observation_counts = {
    year: len(values)
    for year, values in evaluation_temporal_y_true_by_year.items()
} # Construir distribución temporal

evaluation_block_13_4_status = "VALIDATED" # Registrar subbloque validado
evaluation_block_13_4_stage = "ALINEACION_TEMPORAL_VALIDADA" # Registrar etapa validada

print(f"Observaciones totales      : {len(y_true_temporal)}") # Mostrar observaciones
print(f"Predicciones totales       : {len(y_pred_temporal)}") # Mostrar predicciones
print(f"Alineaciones temporales    : {len(evaluation_temporal_alignment)}") # Mostrar años
print(f"Observaciones por año      : {evaluation_temporal_observation_counts}") # Mostrar distribución
print("y_true vs y_pred           : ALINEADOS") # Confirmar alineación
print("Cobertura temporal         : COMPLETA") # Confirmar cobertura
print("Correspondencia GraphData  : VALIDADA") # Confirmar GraphData
print("Asignación por año         : VALIDADA") # Confirmar asignación
print(f"Estado Bloque 13.4         : {evaluation_block_13_4_status}") # Mostrar estado

print("\nBLOQUE 13.5. CONSOLIDACIÓN DE LA ESTRUCTURA TEMPORAL") # Mostrar encabezado

evaluation_block_13_5_status = "ERROR" # Inicializar estado del subbloque
evaluation_block_13_5_stage = "CONSOLIDACION_TEMPORAL" # Registrar etapa actual

if evaluation_block_13_4_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 13.4 no está VALIDATED. Estado actual: {evaluation_block_13_4_status}"
    ) # Validar dependencia del Bloque 13.4

required_temporal_products = {
    "graph_structure": evaluation_graph_temporal_structure,
    "year_by_test_index": evaluation_temporal_year_by_test_index,
    "index_year_map": evaluation_temporal_index_year_map,
    "alignment": evaluation_temporal_alignment,
    "y_true_by_year": evaluation_temporal_y_true_by_year,
    "y_pred_by_year": evaluation_temporal_y_pred_by_year,
    "observation_counts": evaluation_temporal_observation_counts,
} # Definir productos temporales requeridos

missing_temporal_products = [
    name
    for name, value in required_temporal_products.items()
    if value is None
] # Identificar productos faltantes

if missing_temporal_products:
    raise RuntimeError(
        f"Faltan productos temporales requeridos: {missing_temporal_products}"
    ) # Validar disponibilidad de productos

evaluation_temporal_by_year = {} # Inicializar estructura temporal consolidada

for alignment in evaluation_temporal_alignment:
    test_index = alignment["test_index"] # Recuperar índice TEST
    year = alignment["year"] # Recuperar año
    start = alignment["start"] # Recuperar posición inicial
    end = alignment["end"] # Recuperar posición final
    n_observations = alignment["n_observations"] # Recuperar observaciones

    if year not in evaluation_temporal_y_true_by_year:
        raise RuntimeError(
            f"No existe y_true para el año {year}."
        ) # Validar valores observados

    if year not in evaluation_temporal_y_pred_by_year:
        raise RuntimeError(
            f"No existe y_pred para el año {year}."
        ) # Validar predicciones

    y_true_year = evaluation_temporal_y_true_by_year[year] # Recuperar y_true anual
    y_pred_year = evaluation_temporal_y_pred_by_year[year] # Recuperar y_pred anual

    if len(y_true_year) != n_observations:
        raise RuntimeError(
            f"y_true del año {year} no coincide con el número esperado de observaciones."
        ) # Validar cantidad anual

    if len(y_pred_year) != n_observations:
        raise RuntimeError(
            f"y_pred del año {year} no coincide con el número esperado de observaciones."
        ) # Validar cantidad anual

    evaluation_temporal_by_year[year] = {
        "test_index": test_index,
        "year": year,
        "start": start,
        "end": end,
        "n_observations": n_observations,
        "y_true": y_true_year,
        "y_pred": y_pred_year,
    } # Consolidar información temporal por año

evaluation_temporal_year_order = sorted(
    evaluation_temporal_by_year.keys()
) # Construir orden cronológico

if evaluation_temporal_year_order != evaluation_temporal_year_values:
    raise RuntimeError(
        "El orden temporal consolidado no coincide con el orden validado."
    ) # Validar orden temporal

evaluation_temporal_total_observations = sum(
    item["n_observations"]
    for item in evaluation_temporal_by_year.values()
) # Calcular observaciones consolidadas

if evaluation_temporal_total_observations != len(
    PREDICTION_RESULT["y_true"]
):
    raise RuntimeError(
        "La estructura temporal consolidada no cubre todas las observaciones."
    ) # Validar cobertura total

evaluation_temporal_result = {
    "model_code": PREDICTION_RESULT["model_code"],
    "model_name": PREDICTION_RESULT["model_name"],
    "family": PREDICTION_RESULT["family"],
    "test_index": evaluation_test_index.tolist(),
    "years": evaluation_temporal_year_order,
    "year_by_test_index": evaluation_temporal_year_by_test_index,
    "index_year_map": evaluation_temporal_index_year_map,
    "graph_structure": evaluation_graph_temporal_structure,
    "temporal_alignment": evaluation_temporal_alignment,
    "data_by_year": evaluation_temporal_by_year,
    "observation_counts": evaluation_temporal_observation_counts,
    "total_observations": evaluation_temporal_total_observations,
    "prediction_source": PREDICTION_RESULT["prediction_source"],
    "status": "VALIDATED",
} # Construir producto temporal consolidado

evaluation_block_13_5_status = "VALIDATED" # Registrar subbloque validado
evaluation_block_13_5_stage = "ESTRUCTURA_TEMPORAL_CONSOLIDADA" # Registrar etapa validada

print(f"Modelo                     : {evaluation_temporal_result['model_name']}") # Mostrar modelo
print(f"Código                     : {evaluation_temporal_result['model_code']}") # Mostrar código
print(f"Familia                    : {evaluation_temporal_result['family']}") # Mostrar familia
print(f"Años consolidados          : {evaluation_temporal_result['years']}") # Mostrar años
print(f"Índices TEST               : {evaluation_temporal_result['test_index']}") # Mostrar índices
print(f"Observaciones por año      : {evaluation_temporal_result['observation_counts']}") # Mostrar distribución
print(f"Observaciones totales      : {evaluation_temporal_result['total_observations']}") # Mostrar total
print(f"Fuente de predicción       : {evaluation_temporal_result['prediction_source']}") # Mostrar fuente
print("Estructura temporal        : CONSOLIDADA") # Confirmar consolidación
print("Cobertura de observaciones : COMPLETA") # Confirmar cobertura
print("Orden cronológico          : VALIDADO") # Confirmar orden
print(f"Estado Bloque 13.5         : {evaluation_block_13_5_status}") # Mostrar estado

print("\nBLOQUE 13.6. AUDITORÍA FINAL DE LA ESTRUCTURA TEMPORAL") # Mostrar encabezado
evaluation_block_13_6_status = "ERROR" # Inicializar estado del subbloque
evaluation_block_13_6_stage = "AUDITORIA_FINAL" # Registrar etapa actual

if evaluation_block_13_5_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 13.5 no está VALIDATED. Estado actual: {evaluation_block_13_5_status}"
    ) # Validar dependencia del Bloque 13.5

if not isinstance(evaluation_temporal_result, dict):
    raise TypeError(
        "evaluation_temporal_result debe ser un diccionario."
    ) # Validar producto temporal

required_temporal_result_fields = [
    "model_code",
    "model_name",
    "family",
    "test_index",
    "years",
    "year_by_test_index",
    "index_year_map",
    "graph_structure",
    "temporal_alignment",
    "data_by_year",
    "observation_counts",
    "total_observations",
    "prediction_source",
    "status",
] # Definir campos obligatorios

missing_temporal_result_fields = [
    field
    for field in required_temporal_result_fields
    if field not in evaluation_temporal_result
] # Identificar campos faltantes

if missing_temporal_result_fields:
    raise RuntimeError(
        f"Faltan campos en evaluation_temporal_result: {missing_temporal_result_fields}"
    ) # Validar contrato del producto

if evaluation_temporal_result["status"] != "VALIDATED":
    raise RuntimeError(
        "evaluation_temporal_result no presenta estado VALIDATED."
    ) # Validar estado del producto

if evaluation_temporal_result["model_code"] != PREDICTION_RESULT["model_code"]:
    raise RuntimeError(
        "El código del modelo temporal no coincide con PREDICTION_RESULT."
    ) # Validar identidad del modelo

if evaluation_temporal_result["model_name"] != PREDICTION_RESULT["model_name"]:
    raise RuntimeError(
        "El nombre del modelo temporal no coincide con PREDICTION_RESULT."
    ) # Validar identidad del modelo

if evaluation_temporal_result["family"] != PREDICTION_RESULT["family"]:
    raise RuntimeError(
        "La familia del modelo temporal no coincide con PREDICTION_RESULT."
    ) # Validar familia del modelo

if evaluation_temporal_result["prediction_source"] != PREDICTION_RESULT["prediction_source"]:
    raise RuntimeError(
        "La fuente de predicción no coincide con PREDICTION_RESULT."
    ) # Validar trazabilidad de predicciones

if evaluation_temporal_result["test_index"] != evaluation_test_index.tolist():
    raise RuntimeError(
        "Los índices TEST del producto temporal no coinciden con los índices oficiales."
    ) # Validar índices TEST

if evaluation_temporal_result["years"] != evaluation_temporal_year_values:
    raise RuntimeError(
        "Los años del producto temporal no coinciden con los años validados."
    ) # Validar años temporales

if evaluation_temporal_result["year_by_test_index"] != evaluation_temporal_year_by_test_index:
    raise RuntimeError(
        "La correspondencia índice año no coincide con la validada en 13.2."
    ) # Validar trazabilidad temporal

if evaluation_temporal_result["index_year_map"] != evaluation_temporal_index_year_map:
    raise RuntimeError(
        "El mapa índice año no coincide con la asignación validada en 13.3."
    ) # Validar asignación temporal

if evaluation_temporal_result["temporal_alignment"] != evaluation_temporal_alignment:
    raise RuntimeError(
        "La alineación temporal no coincide con la validada en 13.4."
    ) # Validar alineación temporal

if evaluation_temporal_result["observation_counts"] != evaluation_temporal_observation_counts:
    raise RuntimeError(
        "La distribución de observaciones no coincide con la validada."
    ) # Validar distribución temporal

if evaluation_temporal_result["total_observations"] != len(
    PREDICTION_RESULT["y_true"]
):
    raise RuntimeError(
        "El total de observaciones temporales no coincide con PREDICTION_RESULT."
    ) # Validar cobertura total

if evaluation_temporal_result["total_observations"] != len(
    PREDICTION_RESULT["y_pred"]
):
    raise RuntimeError(
        "El total de predicciones temporales no coincide con PREDICTION_RESULT."
    ) # Validar cobertura total

if sum(
    evaluation_temporal_result["observation_counts"].values()
) != evaluation_temporal_result["total_observations"]:
    raise RuntimeError(
        "La suma de observaciones por año no coincide con el total temporal."
    ) # Validar suma temporal

if len(evaluation_temporal_result["years"]) != len(
    evaluation_temporal_result["data_by_year"]
):
    raise RuntimeError(
        "La cantidad de años no coincide con la cantidad de estructuras anuales."
    ) # Validar cobertura anual

for year in evaluation_temporal_result["years"]:
    if year not in evaluation_temporal_result["data_by_year"]:
        raise RuntimeError(
            f"No existe estructura temporal para el año {year}."
        ) # Validar presencia anual

    annual_record = evaluation_temporal_result["data_by_year"][year] # Recuperar registro anual

    if annual_record["year"] != year:
        raise RuntimeError(
            f"El registro anual {year} contiene un año inconsistente."
        ) # Validar identidad temporal

    if annual_record["n_observations"] != evaluation_temporal_result["observation_counts"][year]:
        raise RuntimeError(
            f"La cantidad de observaciones del año {year} es inconsistente."
        ) # Validar observaciones anuales

    if len(annual_record["y_true"]) != annual_record["n_observations"]:
        raise RuntimeError(
            f"y_true del año {year} presenta una cantidad incorrecta de observaciones."
        ) # Validar y_true anual

    if len(annual_record["y_pred"]) != annual_record["n_observations"]:
        raise RuntimeError(
            f"y_pred del año {year} presenta una cantidad incorrecta de observaciones."
        ) # Validar y_pred anual

evaluation_temporal_audit = {
    "model_code": evaluation_temporal_result["model_code"],
    "model_name": evaluation_temporal_result["model_name"],
    "family": evaluation_temporal_result["family"],
    "test_index": evaluation_temporal_result["test_index"],
    "years": evaluation_temporal_result["years"],
    "total_graphs": len(evaluation_temporal_result["graph_structure"]),
    "total_observations": evaluation_temporal_result["total_observations"],
    "observation_counts": evaluation_temporal_result["observation_counts"],
    "prediction_source": evaluation_temporal_result["prediction_source"],
    "temporal_structure": "VALIDATED",
    "temporal_alignment": "VALIDATED",
    "coverage": "COMPLETE",
    "status": "VALIDATED",
} # Construir auditoría temporal final

evaluation_block_13_6_status = "VALIDATED" # Registrar subbloque validado
evaluation_block_13_6_stage = "AUDITORIA_TEMPORAL_VALIDADA" # Registrar etapa validada

print(f"Modelo Oficial            : {evaluation_temporal_audit['model_name']}") # Mostrar modelo
print(f"Código Oficial            : {evaluation_temporal_audit['model_code']}") # Mostrar código
print(f"Familia                   : {evaluation_temporal_audit['family']}") # Mostrar familia
print(f"GraphData TEST             : {evaluation_temporal_audit['total_graphs']}") # Mostrar GraphData
print(f"Índices TEST              : {evaluation_temporal_audit['test_index']}") # Mostrar índices
print(f"Años evaluados            : {evaluation_temporal_audit['years']}") # Mostrar años
print(f"Observaciones totales     : {evaluation_temporal_audit['total_observations']}") # Mostrar observaciones
print(f"Observaciones por año     : {evaluation_temporal_audit['observation_counts']}") # Mostrar distribución
print(f"Fuente de predicción      : {evaluation_temporal_audit['prediction_source']}") # Mostrar fuente
print("Estructura temporal       : VALIDADA") # Confirmar estructura
print("Alineación temporal       : VALIDADA") # Confirmar alineación
print("Cobertura temporal        : COMPLETA") # Confirmar cobertura
print("Trazabilidad              : VALIDADA") # Confirmar trazabilidad
print(f"Estado Bloque 13.6        : {evaluation_block_13_6_status}") # Mostrar estado

if evaluation_block_13_6_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 13.6 no está VALIDATED. Estado actual: {evaluation_block_13_6_status}"
    ) # Validar auditoría final del Bloque 13

evaluation_block_13_status = "VALIDATED" # Cerrar formalmente el Bloque 13
evaluation_block_13_stage = "TEMPORAL_VALIDATION_COMPLETED" # Registrar cierre del Bloque 13

print(f"Estado Bloque 13.6      : {evaluation_block_13_6_status}") # Mostrar estado de auditoría
print(f"Estado Bloque 13        : {evaluation_block_13_status}") # Mostrar estado global
print(f"Etapa Bloque 13         : {evaluation_block_13_stage}") # Mostrar etapa final
print("Bloque 13               : VALIDATED") # Confirmar cierre

# BLOQUE 14. DIAGNÓSTICO PREVIO PARA LA COMPARACIÓN CON EL BENCHMARK
# Objetivo: Verificar que existen resultados independientes y resultados oficiales del Benchmark para GraphSAGE.
# Entradas: temporal_evaluation_result y benchmark_experiment_loaded.
# Producto: Diagnóstico preparado para benchmark_comparison.
# Pregunta científica: ¿Existen resultados compatibles y trazables para comparar la evaluación independiente de GraphSAGE con el Benchmark oficial?

print("\nBLOQUE 14.0. DIAGNÓSTICO PREVIO PARA LA COMPARACIÓN CON EL BENCHMARK") # Mostrar encabezado
evaluation_block_14_status = "ERROR" # Inicializar estado global del Bloque 14
evaluation_block_14_stage = "DIAGNOSTICO" # Registrar etapa global
evaluation_block_14_0_status = "ERROR" # Inicializar estado específico de 14.0
evaluation_block_14_0_stage = "DIAGNOSTICO" # Registrar etapa actual

if evaluation_block_13_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 13 no está VALIDATED. Estado actual: {evaluation_block_13_status}"
    ) # Validar dependencia temporal

if not isinstance(EVALUATION_RESULT, dict):
    raise TypeError(
        "EVALUATION_RESULT debe ser un diccionario."
    ) # Validar resultado independiente

if EVALUATION_RESULT.get("status") != "VALIDATED":
    raise RuntimeError(
        "EVALUATION_RESULT debe presentar estado VALIDATED."
    ) # Validar estado de evaluación

if not isinstance(benchmark_experiment_loaded, dict):
    raise TypeError(
        "benchmark_experiment_loaded debe ser un diccionario."
    ) # Validar estructura Benchmark

if benchmark_experiment_loaded.get("status") != "VALIDATED":
    raise RuntimeError(
        "benchmark_experiment_loaded debe presentar estado VALIDATED."
    ) # Validar estado Benchmark

required_benchmark_inputs = [
    "official_model",
    "benchmark_data",
] # Definir componentes requeridos del Benchmark

missing_benchmark_inputs = [
    field
    for field in required_benchmark_inputs
    if field not in benchmark_experiment_loaded
] # Identificar componentes faltantes

if missing_benchmark_inputs:
    raise RuntimeError(
        "benchmark_experiment_loaded presenta componentes faltantes: "
        f"{missing_benchmark_inputs}"
    ) # Validar cobertura Benchmark

benchmark_official_model = benchmark_experiment_loaded[
    "official_model"
] # Recuperar identidad oficial del Benchmark

benchmark_official_data = benchmark_experiment_loaded[
    "benchmark_data"
] # Recuperar datos oficiales del Benchmark

if not isinstance(benchmark_official_model, dict):
    raise TypeError(
        "official_model del Benchmark debe ser un diccionario."
    ) # Validar estructura del modelo oficial

if not isinstance(benchmark_official_data, dict):
    raise TypeError(
        "benchmark_data del Benchmark debe ser un diccionario."
    ) # Validar estructura de datos Benchmark

required_benchmark_identity_fields = [
    "model_code",
    "model_name",
    "family",
] # Definir identidad oficial requerida

missing_benchmark_identity_fields = [
    field
    for field in required_benchmark_identity_fields
    if field not in benchmark_official_model
] # Identificar identidad faltante

if missing_benchmark_identity_fields:
    raise RuntimeError(
        "official_model presenta campos faltantes: "
        f"{missing_benchmark_identity_fields}"
    ) # Validar identidad completa

required_evaluation_identity_fields = [
    "model_code",
    "model_name",
    "family",
] # Definir identidad requerida de evaluación

missing_evaluation_identity_fields = [
    field
    for field in required_evaluation_identity_fields
    if field not in EVALUATION_RESULT
] # Identificar identidad faltante

if missing_evaluation_identity_fields:
    raise RuntimeError(
        "EVALUATION_RESULT presenta campos de identidad faltantes: "
        f"{missing_evaluation_identity_fields}"
    ) # Validar identidad de evaluación

evaluation_model_code = str(
    EVALUATION_RESULT["model_code"]
).strip() # Recuperar código de evaluación

benchmark_model_code = str(
    benchmark_official_model["model_code"]
).strip() # Recuperar código Benchmark

evaluation_model_name = str(
    EVALUATION_RESULT["model_name"]
).strip() # Recuperar nombre de evaluación

benchmark_model_name = str(
    benchmark_official_model["model_name"]
).strip() # Recuperar nombre Benchmark

evaluation_model_family = str(
    EVALUATION_RESULT["family"]
).strip() # Recuperar familia de evaluación

benchmark_model_family = str(
    benchmark_official_model["family"]
).strip() # Recuperar familia Benchmark

if evaluation_model_code != benchmark_model_code:
    raise RuntimeError(
        "El código del modelo independiente no coincide con el Benchmark."
    ) # Validar identidad

if evaluation_model_name.lower() != benchmark_model_name.lower():
    raise RuntimeError(
        "El nombre del modelo independiente no coincide con el Benchmark."
    ) # Validar identidad

if evaluation_model_family.lower() != benchmark_model_family.lower():
    raise RuntimeError(
        "La familia del modelo independiente no coincide con el Benchmark."
    ) # Validar familia

evaluation_benchmark_identity_status = "VALIDATED" # Registrar identidad validada
evaluation_block_14_0_status = "VALIDATED" # Registrar diagnóstico aprobado
evaluation_block_14_0_stage = "DIAGNOSTIC_VALIDATED" # Registrar etapa aprobada

print(f"Modelo Evaluation        : {evaluation_model_name}") # Mostrar modelo independiente
print(f"Modelo Benchmark         : {benchmark_model_name}") # Mostrar modelo Benchmark
print(f"Código Evaluation        : {evaluation_model_code}") # Mostrar código independiente
print(f"Código Benchmark         : {benchmark_model_code}") # Mostrar código Benchmark
print(f"Familia Evaluation       : {evaluation_model_family}") # Mostrar familia independiente
print(f"Familia Benchmark        : {benchmark_model_family}") # Mostrar familia Benchmark
print("Identidad del modelo     : VALIDADA") # Confirmar identidad
print("EVALUATION_RESULT        : VALIDADO") # Confirmar evaluación
print("Benchmark                : VALIDADO") # Confirmar Benchmark
print("Entradas de comparación  : DISPONIBLES") # Confirmar disponibilidad
print(f"Estado Bloque 14.0       : {evaluation_block_14_0_status}") # Mostrar estado

print("\nBLOQUE 14.1. VALIDACIÓN DE LA ESTRUCTURA DEL BENCHMARK") # Mostrar encabezado
evaluation_block_14_1_status = "ERROR" # Inicializar estado
evaluation_block_14_1_stage = "BENCHMARK_STRUCTURE" # Registrar etapa actual

if evaluation_block_14_0_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 14.0 no está VALIDATED. Estado actual: {evaluation_block_14_0_status}"
    ) # Validar dependencia

if not isinstance(benchmark_official_model, dict):
    raise TypeError(
        "benchmark_official_model debe ser un diccionario."
    ) # Validar estructura del modelo

if not isinstance(benchmark_official_data, dict):
    raise TypeError(
        "benchmark_official_data debe ser un diccionario."
    ) # Validar estructura de datos

if not benchmark_official_model:
    raise RuntimeError(
        "benchmark_official_model está vacío."
    ) # Validar contenido

if not benchmark_official_data:
    raise RuntimeError(
        "benchmark_official_data está vacío."
    ) # Validar contenido

required_official_model_fields = [
    "model_code",
    "model_name",
    "family",
] # Definir identidad mínima requerida

missing_official_model_fields = [
    field
    for field in required_official_model_fields
    if field not in benchmark_official_model
] # Identificar campos faltantes

if missing_official_model_fields:
    raise RuntimeError(
        f"Faltan campos del Modelo Oficial: {missing_official_model_fields}"
    ) # Validar contrato del modelo

official_model_field_types = {
    field: type(benchmark_official_model[field]).__name__
    for field in benchmark_official_model
} # Registrar tipos de campos del modelo

benchmark_data_field_types = {
    field: type(benchmark_official_data[field]).__name__
    for field in benchmark_official_data
} # Registrar tipos de campos Benchmark

if not all(
    str(benchmark_official_model[field]).strip()
    for field in required_official_model_fields
):
    raise RuntimeError(
        "Uno o más campos obligatorios del Modelo Oficial están vacíos."
    ) # Validar contenido de identidad

evaluation_benchmark_structure = {
    "official_model": benchmark_official_model,
    "benchmark_data": benchmark_official_data,
    "official_model_fields": list(benchmark_official_model.keys()),
    "benchmark_data_fields": list(benchmark_official_data.keys()),
    "official_model_field_types": official_model_field_types,
    "benchmark_data_field_types": benchmark_data_field_types,
    "status": "VALIDATED",
} # Construir estructura validada

evaluation_benchmark_structure_status = "VALIDATED" # Registrar estructura validada
evaluation_block_14_1_status = "VALIDATED" # Registrar subbloque validado
evaluation_block_14_1_stage = "BENCHMARK_STRUCTURE_VALIDATED" # Registrar etapa validada

print(f"Campos Modelo Oficial   : {evaluation_benchmark_structure['official_model_fields']}") # Mostrar campos
print(f"Campos Benchmark Data   : {evaluation_benchmark_structure['benchmark_data_fields']}") # Mostrar campos
print(f"Tipos Modelo Oficial    : {evaluation_benchmark_structure['official_model_field_types']}") # Mostrar tipos
print(f"Tipos Benchmark Data    : {evaluation_benchmark_structure['benchmark_data_field_types']}") # Mostrar tipos
print("Campos obligatorios      : VALIDADOS") # Confirmar contrato
print("Contenido estructural    : VALIDADO") # Confirmar contenido
print("Estructura Benchmark     : VALIDADA") # Confirmar estructura
print(f"Estado Bloque 14.1       : {evaluation_block_14_1_status}") # Mostrar estado

print("\nBLOQUE 14.2. VALIDACIÓN DE LA IDENTIDAD DEL MODELO OFICIAL") # Mostrar encabezado
evaluation_block_14_2_status = "ERROR" # Inicializar estado
evaluation_block_14_2_stage = "MODEL_IDENTITY" # Registrar etapa actual

if evaluation_block_14_1_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 14.1 no está VALIDATED. Estado actual: {evaluation_block_14_1_status}"
    ) # Validar dependencia

if evaluation_benchmark_structure_status != "VALIDATED":
    raise RuntimeError(
        f"La estructura Benchmark no está VALIDATED. Estado actual: {evaluation_benchmark_structure_status}"
    ) # Validar estructura

benchmark_model_code = str(
    benchmark_official_model["model_code"]
).strip() # Recuperar código Benchmark

benchmark_model_name = str(
    benchmark_official_model["model_name"]
).strip() # Recuperar nombre Benchmark

benchmark_model_family = str(
    benchmark_official_model["family"]
).strip() # Recuperar familia Benchmark

evaluation_model_code = str(
    EVALUATION_RESULT["model_code"]
).strip() # Recuperar código de evaluación

evaluation_model_name = str(
    EVALUATION_RESULT["model_name"]
).strip() # Recuperar nombre de evaluación

evaluation_model_family = str(
    EVALUATION_RESULT["family"]
).strip() # Recuperar familia de evaluación

if not benchmark_model_code:
    raise RuntimeError(
        "El código del Modelo Oficial del Benchmark está vacío."
    ) # Validar código Benchmark

if not benchmark_model_name:
    raise RuntimeError(
        "El nombre del Modelo Oficial del Benchmark está vacío."
    ) # Validar nombre Benchmark

if not benchmark_model_family:
    raise RuntimeError(
        "La familia del Modelo Oficial del Benchmark está vacía."
    ) # Validar familia Benchmark

if not evaluation_model_code:
    raise RuntimeError(
        "El código del modelo de evaluación está vacío."
    ) # Validar código evaluación

if not evaluation_model_name:
    raise RuntimeError(
        "El nombre del modelo de evaluación está vacío."
    ) # Validar nombre evaluación

if not evaluation_model_family:
    raise RuntimeError(
        "La familia del modelo de evaluación está vacía."
    ) # Validar familia evaluación

if evaluation_model_code != benchmark_model_code:
    raise RuntimeError(
        f"El código del modelo no coincide: Evaluation={evaluation_model_code}, Benchmark={benchmark_model_code}"
    ) # Validar identidad por código

if evaluation_model_name.lower() != benchmark_model_name.lower():
    raise RuntimeError(
        f"El nombre del modelo no coincide: Evaluation={evaluation_model_name}, Benchmark={benchmark_model_name}"
    ) # Validar identidad por nombre

if evaluation_model_family.lower() != benchmark_model_family.lower():
    raise RuntimeError(
        f"La familia del modelo no coincide: Evaluation={evaluation_model_family}, Benchmark={benchmark_model_family}"
    ) # Validar identidad por familia

evaluation_official_model_identity = {
    "model_code": benchmark_model_code,
    "model_name": benchmark_model_name,
    "family": benchmark_model_family,
    "evaluation_model_code": evaluation_model_code,
    "evaluation_model_name": evaluation_model_name,
    "evaluation_model_family": evaluation_model_family,
    "identity_status": "VALIDATED",
} # Construir identidad oficial validada

evaluation_official_model_identity_status = "VALIDATED" # Registrar identidad validada
evaluation_block_14_2_status = "VALIDATED" # Registrar subbloque validado
evaluation_block_14_2_stage = "MODEL_IDENTITY_VALIDATED" # Registrar etapa validada

print(f"Código Oficial           : {benchmark_model_code}") # Mostrar código oficial
print(f"Modelo Oficial           : {benchmark_model_name}") # Mostrar modelo oficial
print(f"Familia Oficial          : {benchmark_model_family}") # Mostrar familia oficial
print(f"Código Evaluation        : {evaluation_model_code}") # Mostrar código evaluación
print(f"Modelo Evaluation        : {evaluation_model_name}") # Mostrar modelo evaluación
print(f"Familia Evaluation       : {evaluation_model_family}") # Mostrar familia evaluación
print("Código                    : VALIDADO") # Confirmar código
print("Nombre                    : VALIDADO") # Confirmar nombre
print("Familia                   : VALIDADA") # Confirmar familia
print("Identidad Modelo Oficial  : VALIDADA") # Confirmar identidad
print(f"Estado Bloque 14.2        : {evaluation_block_14_2_status}") # Mostrar estado

print("\nBLOQUE 14.3. RECUPERACIÓN Y VALIDACIÓN DE LAS MÉTRICAS DEL BENCHMARK") # Mostrar encabezado
evaluation_block_14_3_status = "ERROR" # Inicializar estado
evaluation_block_14_3_stage = "BENCHMARK_METRICS_RECOVERY" # Registrar etapa actual

if evaluation_block_14_2_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 14.2 no está VALIDATED. Estado actual: {evaluation_block_14_2_status}"
    ) # Validar dependencia

if evaluation_official_model_identity_status != "VALIDATED":
    raise RuntimeError(
        f"La identidad del Modelo Oficial no está VALIDATED. Estado actual: {evaluation_official_model_identity_status}"
    ) # Validar identidad oficial

required_benchmark_metric_names = [
    "rmse",
    "mae",
    "mape",
    "r2",
] # Definir métricas oficiales requeridas

benchmark_metric_source = None # Inicializar fuente de métricas
benchmark_metric_container = None # Inicializar contenedor de métricas

if all(
    metric in benchmark_official_model
    for metric in required_benchmark_metric_names
):
    benchmark_metric_source = "official_model" # Identificar fuente directa
    benchmark_metric_container = benchmark_official_model # Seleccionar contenedor

elif all(
    metric in benchmark_official_data
    for metric in required_benchmark_metric_names
):
    benchmark_metric_source = "benchmark_data" # Identificar fuente de datos
    benchmark_metric_container = benchmark_official_data # Seleccionar contenedor

else:
    raise RuntimeError(
        "No se encontraron las cuatro métricas oficiales "
        "RMSE, MAE, MAPE y R2 en las estructuras disponibles."
    ) # Validar disponibilidad de métricas

benchmark_official_metrics = {
    metric: float(benchmark_metric_container[metric])
    for metric in required_benchmark_metric_names
} # Recuperar métricas oficiales

for metric, value in benchmark_official_metrics.items():
    if not np.isfinite(value):
        raise RuntimeError(
            f"La métrica {metric.upper()} del Benchmark no es finita."
        ) # Validar estabilidad numérica

if benchmark_official_metrics["rmse"] < 0:
    raise RuntimeError(
        "RMSE del Benchmark no puede ser negativa."
    ) # Validar RMSE

if benchmark_official_metrics["mae"] < 0:
    raise RuntimeError(
        "MAE del Benchmark no puede ser negativa."
    ) # Validar MAE

if benchmark_official_metrics["mape"] < 0:
    raise RuntimeError(
        "MAPE del Benchmark no puede ser negativa."
    ) # Validar MAPE

if not -1 <= benchmark_official_metrics["r2"] <= 1:
    raise RuntimeError(
        f"R2 del Benchmark fuera del rango esperado: {benchmark_official_metrics['r2']}"
    ) # Validar R2

evaluation_benchmark_metrics = {
    "model_code": evaluation_official_model_identity["model_code"],
    "model_name": evaluation_official_model_identity["model_name"],
    "family": evaluation_official_model_identity["family"],
    "rmse": benchmark_official_metrics["rmse"],
    "mae": benchmark_official_metrics["mae"],
    "mape": benchmark_official_metrics["mape"],
    "r2": benchmark_official_metrics["r2"],
    "source": benchmark_metric_source,
    "status": "VALIDATED",
} # Construir producto de métricas Benchmark

evaluation_benchmark_metrics_status = "VALIDATED" # Registrar métricas validadas
evaluation_block_14_3_status = "VALIDATED" # Registrar subbloque validado
evaluation_block_14_3_stage = "BENCHMARK_METRICS_VALIDATED" # Registrar etapa validada

print(f"Fuente de métricas       : {benchmark_metric_source}") # Mostrar procedencia
print(f"Modelo                   : {evaluation_benchmark_metrics['model_name']}") # Mostrar modelo
print(f"Código                   : {evaluation_benchmark_metrics['model_code']}") # Mostrar código
print(f"Familia                  : {evaluation_benchmark_metrics['family']}") # Mostrar familia
print(f"RMSE                     : {evaluation_benchmark_metrics['rmse']:.15f}") # Mostrar RMSE
print(f"MAE                      : {evaluation_benchmark_metrics['mae']:.15f}") # Mostrar MAE
print(f"MAPE                     : {evaluation_benchmark_metrics['mape']:.15f}") # Mostrar MAPE
print(f"R2                       : {evaluation_benchmark_metrics['r2']:.15f}") # Mostrar R2
print("RMSE                     : VALIDADO") # Confirmar RMSE
print("MAE                      : VALIDADO") # Confirmar MAE
print("MAPE                     : VALIDADO") # Confirmar MAPE
print("R2                       : VALIDADO") # Confirmar R2
print("Procedencia              : VALIDADA") # Confirmar procedencia
print("Métricas Benchmark       : RECUPERADAS Y VALIDADAS") # Confirmar producto
print(f"Estado Bloque 14.3       : {evaluation_block_14_3_status}") # Mostrar estado

print("\nBLOQUE 14.4. VALIDACIÓN DEL CONTRATO DE MÉTRICAS DEL BENCHMARK") # Mostrar encabezado
evaluation_block_14_4_status = "ERROR" # Inicializar estado
evaluation_block_14_4_stage = "BENCHMARK_METRICS_CONTRACT" # Registrar etapa actual

if evaluation_block_14_3_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 14.3 no está VALIDATED. Estado actual: {evaluation_block_14_3_status}"
    ) # Validar dependencia

if evaluation_benchmark_metrics_status != "VALIDATED":
    raise RuntimeError(
        f"Las métricas Benchmark no están VALIDATED. Estado actual: {evaluation_benchmark_metrics_status}"
    ) # Validar métricas

if not isinstance(evaluation_benchmark_metrics, dict):
    raise TypeError(
        "evaluation_benchmark_metrics debe ser un diccionario."
    ) # Validar estructura

required_metric_contract_fields = [
    "model_code",
    "model_name",
    "family",
    "rmse",
    "mae",
    "mape",
    "r2",
    "source",
    "status",
] # Definir contrato requerido

missing_metric_contract_fields = [
    field
    for field in required_metric_contract_fields
    if field not in evaluation_benchmark_metrics
] # Identificar campos faltantes

if missing_metric_contract_fields:
    raise RuntimeError(
        f"Faltan campos del contrato de métricas: {missing_metric_contract_fields}"
    ) # Validar contrato

if evaluation_benchmark_metrics["status"] != "VALIDATED":
    raise RuntimeError(
        "El producto de métricas Benchmark no presenta estado VALIDATED."
    ) # Validar estado

if evaluation_benchmark_metrics["source"] not in {
    "official_model",
    "benchmark_data",
}:
    raise RuntimeError(
        f"Fuente de métricas no reconocida: {evaluation_benchmark_metrics['source']}"
    ) # Validar procedencia

if evaluation_benchmark_metrics["model_code"] != evaluation_official_model_identity["model_code"]:
    raise RuntimeError(
        "El código de las métricas no coincide con el Modelo Oficial."
    ) # Validar identidad

if evaluation_benchmark_metrics["model_name"].lower() != evaluation_official_model_identity["model_name"].lower():
    raise RuntimeError(
        "El nombre de las métricas no coincide con el Modelo Oficial."
    ) # Validar identidad

if evaluation_benchmark_metrics["family"].lower() != evaluation_official_model_identity["family"].lower():
    raise RuntimeError(
        "La familia de las métricas no coincide con el Modelo Oficial."
    ) # Validar familia

benchmark_metric_values = {
    "rmse": float(evaluation_benchmark_metrics["rmse"]),
    "mae": float(evaluation_benchmark_metrics["mae"]),
    "mape": float(evaluation_benchmark_metrics["mape"]),
    "r2": float(evaluation_benchmark_metrics["r2"]),
} # Recuperar valores métricos

for metric, value in benchmark_metric_values.items():
    if not np.isfinite(value):
        raise RuntimeError(
            f"La métrica {metric.upper()} no es finita."
        ) # Validar estabilidad numérica

if benchmark_metric_values["rmse"] < 0:
    raise RuntimeError(
        "RMSE no puede ser negativa."
    ) # Validar RMSE

if benchmark_metric_values["mae"] < 0:
    raise RuntimeError(
        "MAE no puede ser negativa."
    ) # Validar MAE

if benchmark_metric_values["mape"] < 0:
    raise RuntimeError(
        "MAPE no puede ser negativa."
    ) # Validar MAPE

if not -1 <= benchmark_metric_values["r2"] <= 1:
    raise RuntimeError(
        f"R2 está fuera del rango esperado: {benchmark_metric_values['r2']}"
    ) # Validar R2

evaluation_benchmark_metric_contract = {
    "model_code": evaluation_benchmark_metrics["model_code"],
    "model_name": evaluation_benchmark_metrics["model_name"],
    "family": evaluation_benchmark_metrics["family"],
    "metrics": benchmark_metric_values,
    "source": evaluation_benchmark_metrics["source"],
    "status": "VALIDATED",
} # Construir contrato validado

evaluation_benchmark_metric_contract_status = "VALIDATED" # Registrar contrato validado
evaluation_block_14_4_status = "VALIDATED" # Registrar subbloque validado
evaluation_block_14_4_stage = "BENCHMARK_METRIC_CONTRACT_VALIDATED" # Registrar etapa validada

print(f"Modelo Oficial           : {evaluation_benchmark_metric_contract['model_name']}") # Mostrar modelo
print(f"Código Oficial           : {evaluation_benchmark_metric_contract['model_code']}") # Mostrar código
print(f"Familia Oficial          : {evaluation_benchmark_metric_contract['family']}") # Mostrar familia
print(f"Fuente                   : {evaluation_benchmark_metric_contract['source']}") # Mostrar procedencia
print(f"RMSE                     : {benchmark_metric_values['rmse']:.15f}") # Mostrar RMSE
print(f"MAE                      : {benchmark_metric_values['mae']:.15f}") # Mostrar MAE
print(f"MAPE                     : {benchmark_metric_values['mape']:.15f}") # Mostrar MAPE
print(f"R2                       : {benchmark_metric_values['r2']:.15f}") # Mostrar R2
print("Contrato de campos       : VALIDADO") # Confirmar contrato
print("Identidad de métricas    : VALIDADA") # Confirmar identidad
print("Rangos métricos          : VALIDADOS") # Confirmar rangos
print("Procedencia              : VALIDADA") # Confirmar procedencia
print("Contrato Benchmark       : VALIDADO") # Confirmar contrato final
print(f"Estado Bloque 14.4       : {evaluation_block_14_4_status}") # Mostrar estado

print("\nBLOQUE 14.5. COMPARACIÓN NUMÉRICA ENTRE EVALUACIÓN INDEPENDIENTE Y BENCHMARK") # Mostrar encabezado
evaluation_block_14_5_status = "ERROR" # Inicializar estado
evaluation_block_14_5_stage = "NUMERIC_COMPARISON" # Registrar etapa actual

if evaluation_block_14_4_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 14.4 no está VALIDATED. Estado actual: {evaluation_block_14_4_status}"
    ) # Validar dependencia

if evaluation_benchmark_metric_contract_status != "VALIDATED":
    raise RuntimeError(
        f"El contrato de métricas Benchmark no está VALIDATED. Estado actual: {evaluation_benchmark_metric_contract_status}"
    ) # Validar contrato Benchmark

required_evaluation_metric_fields = [
    "RMSE",
    "MAE",
    "MAPE",
    "R2",
] # Definir métricas requeridas de evaluación

missing_evaluation_metric_fields = [
    field
    for field in required_evaluation_metric_fields
    if field not in EVALUATION_RESULT
] # Identificar métricas faltantes

if missing_evaluation_metric_fields:
    raise RuntimeError(
        f"Faltan métricas en EVALUATION_RESULT: {missing_evaluation_metric_fields}"
    ) # Validar cobertura de evaluación

evaluation_metrics_for_comparison = {
    "rmse": float(EVALUATION_RESULT["RMSE"]),
    "mae": float(EVALUATION_RESULT["MAE"]),
    "mape": float(EVALUATION_RESULT["MAPE"]),
    "r2": float(EVALUATION_RESULT["R2"]),
} # Recuperar métricas independientes

benchmark_metrics_for_comparison = (
    evaluation_benchmark_metric_contract["metrics"]
) # Recuperar métricas Benchmark

for metric, value in evaluation_metrics_for_comparison.items():
    if not np.isfinite(value):
        raise RuntimeError(
            f"La métrica Evaluation {metric.upper()} no es finita."
        ) # Validar estabilidad numérica

for metric, value in benchmark_metrics_for_comparison.items():
    if not np.isfinite(value):
        raise RuntimeError(
            f"La métrica Benchmark {metric.upper()} no es finita."
        ) # Validar estabilidad numérica

evaluation_metric_comparison = {} # Inicializar estructura de comparación

for metric in required_benchmark_metric_names:
    evaluation_value = evaluation_metrics_for_comparison[metric] # Recuperar valor Evaluation
    benchmark_value = benchmark_metrics_for_comparison[metric] # Recuperar valor Benchmark
    absolute_difference = abs(evaluation_value - benchmark_value) # Calcular diferencia absoluta

    if benchmark_value != 0:
        relative_difference = (
            absolute_difference / abs(benchmark_value)
        ) # Calcular diferencia relativa
    else:
        relative_difference = (
            0.0 if evaluation_value == 0 else np.inf
        ) # Resolver diferencia relativa con Benchmark cero

    evaluation_metric_comparison[metric] = {
        "evaluation_value": evaluation_value,
        "benchmark_value": benchmark_value,
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
    } # Registrar comparación

evaluation_benchmark_comparison = {
    "model_code": evaluation_official_model_identity["model_code"],
    "model_name": evaluation_official_model_identity["model_name"],
    "family": evaluation_official_model_identity["family"],
    "evaluation_metrics": evaluation_metrics_for_comparison,
    "benchmark_metrics": benchmark_metrics_for_comparison,
    "metric_comparison": evaluation_metric_comparison,
    "status": "VALIDATED",
} # Construir producto de comparación

evaluation_block_14_5_status = "VALIDATED" # Registrar comparación validada
evaluation_block_14_5_stage = "NUMERIC_COMPARISON_VALIDATED" # Registrar etapa validada

print(f"Modelo Oficial           : {evaluation_benchmark_comparison['model_name']}") # Mostrar modelo
print(f"Código Oficial           : {evaluation_benchmark_comparison['model_code']}") # Mostrar código
print(f"Familia Oficial          : {evaluation_benchmark_comparison['family']}") # Mostrar familia
print(f"RMSE Evaluation          : {evaluation_metrics_for_comparison['rmse']:.15f}") # Mostrar RMSE Evaluation
print(f"RMSE Benchmark           : {benchmark_metrics_for_comparison['rmse']:.15f}") # Mostrar RMSE Benchmark
print(f"Diferencia RMSE          : {evaluation_metric_comparison['rmse']['absolute_difference']:.15f}") # Mostrar diferencia RMSE
print(f"MAE Evaluation           : {evaluation_metrics_for_comparison['mae']:.15f}") # Mostrar MAE Evaluation
print(f"MAE Benchmark            : {benchmark_metrics_for_comparison['mae']:.15f}") # Mostrar MAE Benchmark
print(f"Diferencia MAE           : {evaluation_metric_comparison['mae']['absolute_difference']:.15f}") # Mostrar diferencia MAE
print(f"MAPE Evaluation          : {evaluation_metrics_for_comparison['mape']:.15f}") # Mostrar MAPE Evaluation
print(f"MAPE Benchmark           : {benchmark_metrics_for_comparison['mape']:.15f}") # Mostrar MAPE Benchmark
print(f"Diferencia MAPE          : {evaluation_metric_comparison['mape']['absolute_difference']:.15f}") # Mostrar diferencia MAPE
print(f"R2 Evaluation            : {evaluation_metrics_for_comparison['r2']:.15f}") # Mostrar R2 Evaluation
print(f"R2 Benchmark             : {benchmark_metrics_for_comparison['r2']:.15f}") # Mostrar R2 Benchmark
print(f"Diferencia R2            : {evaluation_metric_comparison['r2']['absolute_difference']:.15f}") # Mostrar diferencia R2
print("Métricas Evaluation      : RECUPERADAS") # Confirmar métricas Evaluation
print("Métricas Benchmark       : RECUPERADAS") # Confirmar métricas Benchmark
print("Comparación numérica     : COMPLETADA") # Confirmar comparación
print("Diferencias              : CUANTIFICADAS") # Confirmar diferencias
print(f"Estado Bloque 14.5       : {evaluation_block_14_5_status}") # Mostrar estado

print("\nBLOQUE 14.6. DIAGNÓSTICO DE LAS DISCREPANCIAS ENTRE EVALUACIÓN Y BENCHMARK") # Mostrar encabezado
evaluation_block_14_6_status = "ERROR" # Inicializar estado
evaluation_block_14_6_stage = "DISCREPANCY_DIAGNOSTIC" # Registrar etapa actual

if evaluation_block_14_5_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 14.5 no está VALIDATED. Estado actual: {evaluation_block_14_5_status}"
    ) # Validar dependencia

if not isinstance(evaluation_benchmark_comparison, dict):
    raise TypeError(
        "evaluation_benchmark_comparison debe ser un diccionario."
    ) # Validar producto de comparación

required_comparison_fields = [
    "model_code",
    "model_name",
    "family",
    "evaluation_metrics",
    "benchmark_metrics",
    "metric_comparison",
    "status",
] # Definir contrato requerido

missing_comparison_fields = [
    field
    for field in required_comparison_fields
    if field not in evaluation_benchmark_comparison
] # Identificar campos faltantes

if missing_comparison_fields:
    raise RuntimeError(
        f"Faltan campos en evaluation_benchmark_comparison: {missing_comparison_fields}"
    ) # Validar contrato

if evaluation_benchmark_comparison["status"] != "VALIDATED":
    raise RuntimeError(
        "evaluation_benchmark_comparison no presenta estado VALIDATED."
    ) # Validar estado

required_comparison_metrics = [
    "rmse",
    "mae",
    "mape",
    "r2",
] # Definir métricas requeridas

evaluation_metric_discrepancy = {} # Inicializar diagnóstico de discrepancias
for metric in required_comparison_metrics:
    comparison = evaluation_benchmark_comparison[
        "metric_comparison"
    ][metric] # Recuperar comparación de métrica

    evaluation_value = float(
        comparison["evaluation_value"]
    ) # Recuperar valor Evaluation

    benchmark_value = float(
        comparison["benchmark_value"]
    ) # Recuperar valor Benchmark

    absolute_difference = float(
        comparison["absolute_difference"]
    ) # Recuperar diferencia absoluta

    relative_difference = float(
        comparison["relative_difference"]
    ) # Recuperar diferencia relativa

    if not np.isfinite(evaluation_value):
        raise RuntimeError(
            f"El valor Evaluation de {metric.upper()} no es finito."
        ) # Validar valor Evaluation

    if not np.isfinite(benchmark_value):
        raise RuntimeError(
            f"El valor Benchmark de {metric.upper()} no es finito."
        ) # Validar valor Benchmark

    if not np.isfinite(absolute_difference):
        raise RuntimeError(
            f"La diferencia absoluta de {metric.upper()} no es finita."
        ) # Validar diferencia absoluta

    if not np.isfinite(relative_difference) and not (
        benchmark_value == 0 and evaluation_value != 0
    ):
        raise RuntimeError(
            f"La diferencia relativa de {metric.upper()} no es válida."
        ) # Validar diferencia relativa

    signed_difference = evaluation_value - benchmark_value # Calcular diferencia con signo

    if np.isclose(
        evaluation_value,
        benchmark_value,
        rtol=0.0,
        atol=1e-10,
    ):
        numerical_status = "EQUIVALENTE" # Clasificar equivalencia numérica
    else:
        numerical_status = "DIFERENTE" # Clasificar diferencia numérica

    if signed_difference > 0:
        direction = "MAYOR_EN_EVALUATION" # Determinar dirección
    elif signed_difference < 0:
        direction = "MENOR_EN_EVALUATION" # Determinar dirección
    else:
        direction = "IGUAL" # Determinar igualdad exacta

    evaluation_metric_discrepancy[metric] = {
        "evaluation_value": evaluation_value,
        "benchmark_value": benchmark_value,
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
        "signed_difference": signed_difference,
        "numerical_status": numerical_status,
        "direction": direction,
    } # Registrar diagnóstico

evaluation_equivalent_metrics = [
    metric
    for metric, result in evaluation_metric_discrepancy.items()
    if result["numerical_status"] == "EQUIVALENTE"
] # Identificar métricas equivalentes

evaluation_different_metrics = [
    metric
    for metric, result in evaluation_metric_discrepancy.items()
    if result["numerical_status"] == "DIFERENTE"
] # Identificar métricas diferentes

evaluation_benchmark_discrepancy_count = len(
    evaluation_different_metrics
) # Contabilizar discrepancias

if evaluation_benchmark_discrepancy_count == 0:
    evaluation_benchmark_comparison_status = "NUMERICALLY_EQUIVALENT" # Clasificar equivalencia global
else:
    evaluation_benchmark_comparison_status = "NUMERICALLY_DIFFERENT" # Clasificar diferencia global

evaluation_benchmark_discrepancy_diagnostic = {
    "model_code": evaluation_benchmark_comparison["model_code"],
    "model_name": evaluation_benchmark_comparison["model_name"],
    "family": evaluation_benchmark_comparison["family"],
    "metric_diagnostics": evaluation_metric_discrepancy,
    "equivalent_metrics": evaluation_equivalent_metrics,
    "different_metrics": evaluation_different_metrics,
    "different_metric_count": evaluation_benchmark_discrepancy_count,
    "comparison_status": evaluation_benchmark_comparison_status,
    "interpretation": (
        "Las diferencias numéricas deben interpretarse según la "
        "coincidencia del protocolo experimental; no constituyen por sí "
        "mismas evidencia de error."
    ),
    "status": "VALIDATED",
} # Construir diagnóstico de discrepancias

evaluation_block_14_6_status = "VALIDATED" # Registrar diagnóstico validado
evaluation_block_14_6_stage = "DISCREPANCY_DIAGNOSTIC_VALIDATED" # Registrar etapa validada

print(f"Modelo Oficial           : {evaluation_benchmark_discrepancy_diagnostic['model_name']}") # Mostrar modelo
print(f"Código Oficial           : {evaluation_benchmark_discrepancy_diagnostic['model_code']}") # Mostrar código
print(f"Métricas equivalentes    : {evaluation_equivalent_metrics}") # Mostrar equivalencias
print(f"Métricas diferentes      : {evaluation_different_metrics}") # Mostrar diferencias
print(f"Cantidad de diferencias  : {evaluation_benchmark_discrepancy_count}") # Mostrar cantidad
print(f"Estado numérico          : {evaluation_benchmark_comparison_status}") # Mostrar estado
print("Interpretación           : CONTEXTUAL") # Confirmar interpretación
print("Causa de discrepancias   : NO ATRIBUIDA AUTOMÁTICAMENTE") # Evitar inferencia indebida
print("Diagnóstico              : COMPLETADO") # Confirmar diagnóstico
print(f"Estado Bloque 14.6       : {evaluation_block_14_6_status}") # Mostrar estado

print("\nBLOQUE 14.7. CONSOLIDACIÓN Y AUDITORÍA FINAL DE LA COMPARACIÓN") # Mostrar encabezado
evaluation_block_14_7_status = "ERROR" # Inicializar estado
evaluation_block_14_7_stage = "FINAL_AUDIT" # Registrar etapa actual
if evaluation_block_14_6_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 14.6 no está VALIDATED. Estado actual: {evaluation_block_14_6_status}"
    ) # Validar dependencia

if evaluation_block_14_2_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 14.2 no está VALIDATED. Estado actual: {evaluation_block_14_2_status}"
    ) # Validar identidad

if evaluation_block_14_4_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 14.4 no está VALIDATED. Estado actual: {evaluation_block_14_4_status}"
    ) # Validar contrato de métricas

if evaluation_block_14_5_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 14.5 no está VALIDATED. Estado actual: {evaluation_block_14_5_status}"
    ) # Validar comparación

if evaluation_block_14_6_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 14.6 no está VALIDATED. Estado actual: {evaluation_block_14_6_status}"
    ) # Validar diagnóstico

required_final_products = {
    "official_model_identity": evaluation_official_model_identity,
    "benchmark_metric_contract": evaluation_benchmark_metric_contract,
    "benchmark_comparison": evaluation_benchmark_comparison,
    "discrepancy_diagnostic": evaluation_benchmark_discrepancy_diagnostic,
} # Definir productos requeridos

missing_final_products = [
    name
    for name, value in required_final_products.items()
    if value is None
] # Identificar productos faltantes

if missing_final_products:
    raise RuntimeError(
        f"Faltan productos requeridos para la auditoría final: {missing_final_products}"
    ) # Validar disponibilidad

if evaluation_official_model_identity["model_code"] != evaluation_benchmark_comparison["model_code"]:
    raise RuntimeError(
        "La identidad del modelo no coincide con la comparación final."
    ) # Validar trazabilidad del modelo

if evaluation_benchmark_metric_contract["model_code"] != evaluation_benchmark_comparison["model_code"]:
    raise RuntimeError(
        "El contrato de métricas no coincide con la comparación final."
    ) # Validar trazabilidad de métricas

if len(evaluation_benchmark_comparison["metric_comparison"]) != 4:
    raise RuntimeError(
        "La comparación final no contiene las cuatro métricas oficiales."
    ) # Validar cobertura de métricas

if len(evaluation_benchmark_discrepancy_diagnostic["metric_diagnostics"]) != 4:
    raise RuntimeError(
        "El diagnóstico final no contiene las cuatro métricas oficiales."
    ) # Validar cobertura del diagnóstico

evaluation_metric_comparison_audit = {
    metric: {
        "evaluation": result["evaluation_value"],
        "benchmark": result["benchmark_value"],
        "absolute_difference": result["absolute_difference"],
        "relative_difference": result["relative_difference"],
        "status": evaluation_benchmark_discrepancy_diagnostic[
            "metric_diagnostics"
        ][metric]["numerical_status"],
    }
    for metric, result in evaluation_benchmark_comparison[
        "metric_comparison"
    ].items()
} # Consolidar auditoría por métrica

BENCHMARK_COMPARISON_RESULT = {
    "model_code": evaluation_official_model_identity["model_code"],
    "model_name": evaluation_official_model_identity["model_name"],
    "family": evaluation_official_model_identity["family"],
    "evaluation_metrics": evaluation_benchmark_comparison["evaluation_metrics"],
    "benchmark_metrics": evaluation_benchmark_comparison["benchmark_metrics"],
    "metric_comparison": evaluation_metric_comparison_audit,
    "equivalent_metrics": evaluation_benchmark_discrepancy_diagnostic[
        "equivalent_metrics"
    ],
    "different_metrics": evaluation_benchmark_discrepancy_diagnostic[
        "different_metrics"
    ],
    "different_metric_count": evaluation_benchmark_discrepancy_diagnostic[
        "different_metric_count"
    ],
    "comparison_status": evaluation_benchmark_discrepancy_diagnostic[
        "comparison_status"
    ],
    "benchmark_metric_source": evaluation_benchmark_metric_contract[
        "source"
    ],
    "interpretation": evaluation_benchmark_discrepancy_diagnostic[
        "interpretation"
    ],
    "status": "VALIDATED",
} # Construir producto oficial de comparación

if BENCHMARK_COMPARISON_RESULT["model_code"] != EVALUATION_RESULT["model_code"]:
    raise RuntimeError(
        "El código del Modelo Oficial no coincide con EVALUATION_RESULT."
    ) # Validar identidad final

if BENCHMARK_COMPARISON_RESULT["model_code"] != benchmark_official_model["model_code"]:
    raise RuntimeError(
        "El código del Modelo Oficial no coincide con el Benchmark."
    ) # Validar identidad cruzada

if BENCHMARK_COMPARISON_RESULT["model_name"].lower() != str(
    EVALUATION_RESULT["model_name"]
).lower():
    raise RuntimeError(
        "El nombre del Modelo Oficial no coincide con EVALUATION_RESULT."
    ) # Validar nombre final

if BENCHMARK_COMPARISON_RESULT["family"].lower() != str(
    EVALUATION_RESULT["family"]
).lower():
    raise RuntimeError(
        "La familia del Modelo Oficial no coincide con EVALUATION_RESULT."
    ) # Validar familia final

evaluation_block_14_7_status = "VALIDATED" # Registrar auditoría final validada
evaluation_block_14_7_stage = "FINAL_AUDIT_VALIDATED" # Registrar etapa final

evaluation_block_14_status = "VALIDATED" # Cerrar formalmente el Bloque 14
evaluation_block_14_stage = "BENCHMARK_COMPARISON_COMPLETED" # Registrar cierre global

print(f"Modelo Oficial           : {BENCHMARK_COMPARISON_RESULT['model_name']}") # Mostrar modelo
print(f"Código Oficial           : {BENCHMARK_COMPARISON_RESULT['model_code']}") # Mostrar código
print(f"Familia Oficial          : {BENCHMARK_COMPARISON_RESULT['family']}") # Mostrar familia
print(f"Métricas comparadas      : {len(BENCHMARK_COMPARISON_RESULT['metric_comparison'])}") # Mostrar cobertura
print(f"Métricas equivalentes    : {BENCHMARK_COMPARISON_RESULT['equivalent_metrics']}") # Mostrar equivalencias
print(f"Métricas diferentes      : {BENCHMARK_COMPARISON_RESULT['different_metrics']}") # Mostrar diferencias
print(f"Diferencias detectadas   : {BENCHMARK_COMPARISON_RESULT['different_metric_count']}") # Mostrar cantidad
print(f"Estado comparación       : {BENCHMARK_COMPARISON_RESULT['comparison_status']}") # Mostrar estado
print(f"Fuente métricas Benchmark: {BENCHMARK_COMPARISON_RESULT['benchmark_metric_source']}") # Mostrar fuente
print("Identidad                : VALIDADA") # Confirmar identidad
print("Cobertura de métricas    : VALIDADA") # Confirmar cobertura
print("Trazabilidad             : VALIDADA") # Confirmar trazabilidad
print("Auditoría final          : VALIDADA") # Confirmar auditoría
print(f"Estado Bloque 14.7       : {evaluation_block_14_7_status}") # Mostrar estado
print(f"Estado Bloque 14         : {evaluation_block_14_status}") # Mostrar estado global

# BLOQUE 15. CONSOLIDACIÓN Y VALIDACIÓN CIENTÍFICA DE LA EVALUACIÓN
# Objetivo: Consolidar los resultados de la evaluación independiente, el diagnóstico temporal y la comparación con el Benchmark oficial.
# Entradas: temporal_evaluation_result y BENCHMARK_COMPARISON_RESULT.
# Producto: EVALUATION_FINAL_RESULT.
# Pregunta científica: ¿El resultado consolidado de la evaluación del Modelo Oficial es completo, trazable y científicamente consistente?

print("\nBLOQUE 15.0. DIAGNÓSTICO PREVIO PARA LA CONSOLIDACIÓN DE LA EVALUACIÓN") # Mostrar encabezado
evaluation_block_15_status = "ERROR" # Inicializar estado global del Bloque 15
evaluation_block_15_stage = "DIAGNOSTICO" # Registrar etapa global
evaluation_block_15_0_status = "ERROR" # Inicializar estado específico de 15.0
evaluation_block_15_0_stage = "DIAGNOSTICO" # Registrar etapa actual

if evaluation_block_13_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 13 no está VALIDATED. Estado actual: {evaluation_block_13_status}"
    ) # Validar dependencia temporal

if evaluation_block_14_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 14 no está VALIDATED. Estado actual: {evaluation_block_14_status}"
    ) # Validar dependencia Benchmark

if not isinstance(EVALUATION_RESULT, dict):
    raise TypeError(
        "EVALUATION_RESULT debe ser un diccionario."
    ) # Validar estructura de evaluación

if EVALUATION_RESULT.get("status") != "VALIDATED":
    raise RuntimeError(
        "EVALUATION_RESULT debe presentar estado VALIDATED."
    ) # Validar estado de evaluación

if not isinstance(evaluation_temporal_result, dict):
    raise TypeError(
        "evaluation_temporal_result debe ser un diccionario."
    ) # Validar estructura temporal

if evaluation_temporal_result.get("status") != "VALIDATED":
    raise RuntimeError(
        "evaluation_temporal_result debe presentar estado VALIDATED."
    ) # Validar estado temporal

if not isinstance(BENCHMARK_COMPARISON_RESULT, dict):
    raise TypeError(
        "BENCHMARK_COMPARISON_RESULT debe ser un diccionario."
    ) # Validar estructura Benchmark

if BENCHMARK_COMPARISON_RESULT.get("status") != "VALIDATED":
    raise RuntimeError(
        "BENCHMARK_COMPARISON_RESULT debe presentar estado VALIDATED."
    ) # Validar estado Benchmark

required_evaluation_fields = [
    "model_code",
    "model_name",
    "family",
] # Definir identidad mínima de evaluación

missing_evaluation_fields = [
    field
    for field in required_evaluation_fields
    if field not in EVALUATION_RESULT
] # Identificar campos faltantes de evaluación

if missing_evaluation_fields:
    raise RuntimeError(
        f"Faltan campos en EVALUATION_RESULT: {missing_evaluation_fields}"
    ) # Validar contrato de evaluación

if "years" not in evaluation_temporal_result:
    raise RuntimeError(
        "evaluation_temporal_result no contiene la dimensión temporal requerida."
    ) # Validar disponibilidad temporal

required_benchmark_fields = [
    "model_code",
    "model_name",
    "family",
] # Definir identidad mínima Benchmark

missing_benchmark_fields = [
    field
    for field in required_benchmark_fields
    if field not in BENCHMARK_COMPARISON_RESULT
] # Identificar campos faltantes Benchmark

if missing_benchmark_fields:
    raise RuntimeError(
        f"Faltan campos en BENCHMARK_COMPARISON_RESULT: {missing_benchmark_fields}"
    ) # Validar contrato Benchmark

evaluation_model_code = str(EVALUATION_RESULT["model_code"]).strip() # Recuperar código de evaluación
evaluation_model_name = str(EVALUATION_RESULT["model_name"]).strip() # Recuperar modelo de evaluación
evaluation_model_family = str(EVALUATION_RESULT["family"]).strip() # Recuperar familia de evaluación
benchmark_model_code = str(BENCHMARK_COMPARISON_RESULT["model_code"]).strip() # Recuperar código Benchmark
benchmark_model_name = str(BENCHMARK_COMPARISON_RESULT["model_name"]).strip() # Recuperar modelo Benchmark
benchmark_model_family = str(BENCHMARK_COMPARISON_RESULT["family"]).strip() # Recuperar familia Benchmark

if evaluation_model_code != benchmark_model_code:
    raise RuntimeError(
        "El código del Modelo Oficial no coincide entre evaluación y Benchmark."
    ) # Validar identidad cruzada

if evaluation_model_name.lower() != benchmark_model_name.lower():
    raise RuntimeError(
        "El nombre del Modelo Oficial no coincide entre evaluación y Benchmark."
    ) # Validar identidad cruzada

if evaluation_model_family.lower() != benchmark_model_family.lower():
    raise RuntimeError(
        "La familia del Modelo Oficial no coincide entre evaluación y Benchmark."
    ) # Validar familia cruzada

evaluation_block_15_inputs = {
    "evaluation_result": EVALUATION_RESULT,
    "evaluation_temporal_result": evaluation_temporal_result,
    "benchmark_comparison_result": BENCHMARK_COMPARISON_RESULT,
    "model_code": evaluation_model_code,
    "model_name": evaluation_model_name,
    "family": evaluation_model_family,
    "status": "VALIDATED",
} # Construir diagnóstico de entradas

evaluation_block_15_0_status = "VALIDATED" # Registrar diagnóstico validado
evaluation_block_15_0_stage = "DIAGNOSTIC_VALIDATED" # Registrar etapa validada

print(f"Modelo Oficial            : {evaluation_model_name}") # Mostrar modelo
print(f"Código Oficial            : {evaluation_model_code}") # Mostrar código
print(f"Familia Oficial           : {evaluation_model_family}") # Mostrar familia
print("EVALUATION_RESULT         : VALIDADO") # Confirmar evaluación
print("Evaluación temporal       : VALIDADA") # Confirmar información temporal
print("Benchmark Comparison      : VALIDADO") # Confirmar comparación
print("Identidad del modelo      : VALIDADA") # Confirmar identidad
print("Compatibilidad entradas   : VALIDADA") # Confirmar compatibilidad
print("Dimensión temporal        : DISPONIBLE") # Confirmar dimensión temporal
print("Entradas de consolidación : DISPONIBLES") # Confirmar disponibilidad
print(f"Estado Bloque 15.0        : {evaluation_block_15_0_status}") # Mostrar estado

print("\nBLOQUE 15.1. CONSOLIDACIÓN DE LA IDENTIDAD Y RESULTADOS DE LA EVALUACIÓN") # Mostrar encabezado
evaluation_block_15_1_status = "ERROR" # Inicializar estado
evaluation_block_15_1_stage = "EVALUATION_CONSOLIDATION" # Registrar etapa actual

if evaluation_block_15_0_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.0 no está VALIDATED. Estado actual: {evaluation_block_15_0_status}"
    ) # Validar dependencia

if not isinstance(EVALUATION_RESULT, dict):
    raise TypeError(
        "EVALUATION_RESULT debe ser un diccionario."
    ) # Validar estructura

if EVALUATION_RESULT.get("status") != "VALIDATED":
    raise RuntimeError(
        "EVALUATION_RESULT debe presentar estado VALIDATED."
    ) # Validar estado

required_evaluation_identity_fields = [
    "model_code",
    "model_name",
    "family",
] # Definir identidad requerida

missing_evaluation_identity_fields = [
    field
    for field in required_evaluation_identity_fields
    if field not in EVALUATION_RESULT
] # Identificar campos faltantes

if missing_evaluation_identity_fields:
    raise RuntimeError(
        f"Faltan campos de identidad en EVALUATION_RESULT: {missing_evaluation_identity_fields}"
    ) # Validar identidad

evaluation_model_code = str(
    EVALUATION_RESULT["model_code"]
).strip() # Recuperar código oficial

evaluation_model_name = str(
    EVALUATION_RESULT["model_name"]
).strip() # Recuperar nombre oficial

evaluation_model_family = str(
    EVALUATION_RESULT["family"]
).strip() # Recuperar familia oficial

if evaluation_model_code != evaluation_block_15_inputs["model_code"]:
    raise RuntimeError(
        "El código del modelo no coincide con las entradas consolidadas de 15.0."
    ) # Validar trazabilidad

if evaluation_model_name.lower() != evaluation_block_15_inputs["model_name"].lower():
    raise RuntimeError(
        "El nombre del modelo no coincide con las entradas consolidadas de 15.0."
    ) # Validar trazabilidad

if evaluation_model_family.lower() != evaluation_block_15_inputs["family"].lower():
    raise RuntimeError(
        "La familia del modelo no coincide con las entradas consolidadas de 15.0."
    ) # Validar trazabilidad

evaluation_identity_result = {
    "model_code": evaluation_model_code,
    "model_name": evaluation_model_name,
    "family": evaluation_model_family,
} # Consolidar identidad oficial

evaluation_metrics_source = {
    key: value
    for key, value in EVALUATION_RESULT.items()
    if key.lower() in {"rmse", "mae", "mape", "r2"}
} # Recuperar únicamente métricas existentes

evaluation_set = EVALUATION_RESULT.get(
    "evaluation_set"
) # Recuperar conjunto de evaluación si existe

evaluation_consolidated_result = {
    "model_code": evaluation_identity_result["model_code"],
    "model_name": evaluation_identity_result["model_name"],
    "family": evaluation_identity_result["family"],
    "evaluation_set": evaluation_set,
    "metrics": evaluation_metrics_source,
    "status": "VALIDATED",
} # Construir resultado consolidado

evaluation_block_15_1_status = "VALIDATED" # Registrar consolidación validada
evaluation_block_15_1_stage = "EVALUATION_CONSOLIDATED" # Registrar etapa validada

print(f"Modelo Oficial           : {evaluation_identity_result['model_name']}") # Mostrar modelo
print(f"Código Oficial           : {evaluation_identity_result['model_code']}") # Mostrar código
print(f"Familia Oficial          : {evaluation_identity_result['family']}") # Mostrar familia
print(f"Conjunto Evaluación      : {evaluation_set}") # Mostrar conjunto
print(f"Métricas disponibles     : {list(evaluation_metrics_source.keys())}") # Mostrar métricas
print("Identidad                : VALIDADA") # Confirmar identidad
print("Resultados Evaluation    : CONSOLIDADOS") # Confirmar consolidación
print("Trazabilidad             : VALIDADA") # Confirmar trazabilidad
print(f"Estado Bloque 15.1       : {evaluation_block_15_1_status}") # Mostrar estado

print("\nBLOQUE 15.2. CONSOLIDACIÓN DE LOS RESULTADOS TEMPORALES") # Mostrar encabezado
evaluation_block_15_2_status = "ERROR" # Inicializar estado
evaluation_block_15_2_stage = "TEMPORAL_CONSOLIDATION" # Registrar etapa actual

if evaluation_block_15_1_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.1 no está VALIDATED. Estado actual: {evaluation_block_15_1_status}"
    ) # Validar dependencia de 15.1

if evaluation_block_13_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 13 no está VALIDATED. Estado actual: {evaluation_block_13_status}"
    ) # Validar dependencia temporal

if not isinstance(evaluation_temporal_result, dict):
    raise TypeError(
        "evaluation_temporal_result debe ser un diccionario."
    ) # Validar estructura temporal

if evaluation_temporal_result.get("status") != "VALIDATED":
    raise RuntimeError(
        "evaluation_temporal_result debe presentar estado VALIDATED."
    ) # Validar estado temporal

temporal_result_status = evaluation_temporal_result.get(
    "status"
) # Recuperar estado temporal

temporal_result_years = evaluation_temporal_result.get(
    "years"
) # Recuperar dimensión temporal existente

if temporal_result_years is None:
    raise RuntimeError(
        "evaluation_temporal_result no contiene la dimensión temporal 'years'."
    ) # Validar dimensión temporal

if not isinstance(temporal_result_years, (list, tuple, np.ndarray)):
    raise TypeError(
        "La dimensión temporal 'years' debe ser una estructura secuencial."
    ) # Validar estructura temporal

if len(temporal_result_years) == 0:
    raise RuntimeError(
        "La dimensión temporal 'years' está vacía."
    ) # Validar cobertura temporal

evaluation_temporal_consolidated_result = dict(
    evaluation_temporal_result
) # Copiar producto temporal validado

evaluation_temporal_consolidated_result["model_code"] = (
    evaluation_consolidated_result["model_code"]
) # Asociar código oficial

evaluation_temporal_consolidated_result["model_name"] = (
    evaluation_consolidated_result["model_name"]
) # Asociar modelo oficial

evaluation_temporal_consolidated_result["family"] = (
    evaluation_consolidated_result["family"]
) # Asociar familia oficial

evaluation_temporal_consolidated_result["status"] = "VALIDATED" # Confirmar estado consolidado

evaluation_block_15_2_status = "VALIDATED" # Registrar consolidación validada
evaluation_block_15_2_stage = "TEMPORAL_CONSOLIDATED" # Registrar etapa validada

print(f"Modelo Oficial           : {evaluation_temporal_consolidated_result['model_name']}") # Mostrar modelo
print(f"Código Oficial           : {evaluation_temporal_consolidated_result['model_code']}") # Mostrar código
print(f"Familia Oficial          : {evaluation_temporal_consolidated_result['family']}") # Mostrar familia
print(f"Años disponibles         : {list(temporal_result_years)}") # Mostrar años
print(f"Cantidad de años         : {len(temporal_result_years)}") # Mostrar cobertura temporal
print(f"Estado temporal original : {temporal_result_status}") # Mostrar estado original
print("Dimensión temporal       : VALIDADA") # Confirmar dimensión
print("Integración temporal     : VALIDADA") # Confirmar integración
print("Trazabilidad modelo      : VALIDADA") # Confirmar trazabilidad
print(f"Estado Bloque 15.2       : {evaluation_block_15_2_status}") # Mostrar estado

print("\nBLOQUE 15.3. INTEGRACIÓN DE LA COMPARACIÓN CON EL BENCHMARK") # Mostrar encabezado
evaluation_block_15_3_status = "ERROR" # Inicializar estado
evaluation_block_15_3_stage = "BENCHMARK_INTEGRATION" # Registrar etapa actual

if evaluation_block_15_2_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.2 no está VALIDATED. Estado actual: {evaluation_block_15_2_status}"
    ) # Validar dependencia temporal

if evaluation_block_14_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 14 no está VALIDATED. Estado actual: {evaluation_block_14_status}"
    ) # Validar dependencia Benchmark

if not isinstance(BENCHMARK_COMPARISON_RESULT, dict):
    raise TypeError(
        "BENCHMARK_COMPARISON_RESULT debe ser un diccionario."
    ) # Validar estructura Benchmark

if BENCHMARK_COMPARISON_RESULT.get("status") != "VALIDATED":
    raise RuntimeError(
        "BENCHMARK_COMPARISON_RESULT debe presentar estado VALIDATED."
    ) # Validar estado Benchmark

required_benchmark_integration_fields = ["model_code", "model_name", "family", "evaluation_metrics", "benchmark_metrics", "metric_comparison", "comparison_status"] # Definir contrato de integración

missing_benchmark_integration_fields = [field for field in required_benchmark_integration_fields if field not in BENCHMARK_COMPARISON_RESULT] # Identificar campos faltantes

if missing_benchmark_integration_fields:
    raise RuntimeError(
        f"Faltan campos requeridos en BENCHMARK_COMPARISON_RESULT: {missing_benchmark_integration_fields}"
    ) # Validar contrato Benchmark

if BENCHMARK_COMPARISON_RESULT["model_code"] != evaluation_consolidated_result["model_code"]:
    raise RuntimeError(
        "El código del Benchmark no coincide con la evaluación consolidada."
    ) # Validar identidad

if BENCHMARK_COMPARISON_RESULT["model_name"].lower() != evaluation_consolidated_result["model_name"].lower():
    raise RuntimeError(
        "El nombre del Benchmark no coincide con la evaluación consolidada."
    ) # Validar identidad

if BENCHMARK_COMPARISON_RESULT["family"].lower() != evaluation_consolidated_result["family"].lower():
    raise RuntimeError(
        "La familia del Benchmark no coincide con la evaluación consolidada."
    ) # Validar familia

evaluation_benchmark_consolidated_result = {
    "model_code": evaluation_consolidated_result["model_code"],
    "model_name": evaluation_consolidated_result["model_name"],
    "family": evaluation_consolidated_result["family"],
    "evaluation_metrics": BENCHMARK_COMPARISON_RESULT["evaluation_metrics"],
    "benchmark_metrics": BENCHMARK_COMPARISON_RESULT["benchmark_metrics"],
    "metric_comparison": BENCHMARK_COMPARISON_RESULT["metric_comparison"],
    "comparison_status": BENCHMARK_COMPARISON_RESULT["comparison_status"],
    "equivalent_metrics": BENCHMARK_COMPARISON_RESULT.get("equivalent_metrics", []),
    "different_metrics": BENCHMARK_COMPARISON_RESULT.get("different_metrics", []),
    "different_metric_count": BENCHMARK_COMPARISON_RESULT.get("different_metric_count", 0),
    "benchmark_metric_source": BENCHMARK_COMPARISON_RESULT.get("benchmark_metric_source"),
    "status": "VALIDATED",
} # Construir consolidación Benchmark

evaluation_block_15_3_status = "VALIDATED" # Registrar integración validada
evaluation_block_15_3_stage = "BENCHMARK_INTEGRATED" # Registrar etapa validada

print(f"Modelo Oficial           : {evaluation_benchmark_consolidated_result['model_name']}") # Mostrar modelo
print(f"Código Oficial           : {evaluation_benchmark_consolidated_result['model_code']}") # Mostrar código
print(f"Familia Oficial          : {evaluation_benchmark_consolidated_result['family']}") # Mostrar familia
print(f"Métricas Evaluation      : {len(evaluation_benchmark_consolidated_result['evaluation_metrics'])}") # Mostrar métricas Evaluation
print(f"Métricas Benchmark       : {len(evaluation_benchmark_consolidated_result['benchmark_metrics'])}") # Mostrar métricas Benchmark
print(f"Métricas comparadas      : {len(evaluation_benchmark_consolidated_result['metric_comparison'])}") # Mostrar cobertura comparativa
print(f"Métricas diferentes      : {evaluation_benchmark_consolidated_result['different_metrics']}") # Mostrar discrepancias
print(f"Diferencias detectadas   : {evaluation_benchmark_consolidated_result['different_metric_count']}") # Mostrar cantidad de discrepancias
print(f"Estado comparación       : {evaluation_benchmark_consolidated_result['comparison_status']}") # Mostrar estado comparativo
print("Identidad Benchmark      : VALIDADA") # Confirmar identidad
print("Comparación Benchmark    : INTEGRADA") # Confirmar integración
print("Trazabilidad             : VALIDADA") # Confirmar trazabilidad
print(f"Estado Bloque 15.3       : {evaluation_block_15_3_status}") # Mostrar estado

print("\nBLOQUE 15.4. DIAGNÓSTICO CIENTÍFICO DE CONSISTENCIA") # Mostrar encabezado
evaluation_block_15_4_status = "ERROR" # Inicializar estado
evaluation_block_15_4_stage = "SCIENTIFIC_CONSISTENCY_DIAGNOSTIC" # Registrar etapa actual

if evaluation_block_15_3_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.3 no está VALIDATED. Estado actual: {evaluation_block_15_3_status}"
    ) # Validar dependencia Benchmark

if evaluation_block_15_2_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.2 no está VALIDATED. Estado actual: {evaluation_block_15_2_status}"
    ) # Validar dependencia temporal

if not isinstance(evaluation_consolidated_result, dict):
    raise TypeError(
        "evaluation_consolidated_result debe ser un diccionario."
    ) # Validar estructura de evaluación

if not isinstance(evaluation_temporal_consolidated_result, dict):
    raise TypeError(
        "evaluation_temporal_consolidated_result debe ser un diccionario."
    ) # Validar estructura temporal

if not isinstance(evaluation_benchmark_consolidated_result, dict):
    raise TypeError(
        "evaluation_benchmark_consolidated_result debe ser un diccionario."
    ) # Validar estructura Benchmark

identity_fields = ["model_code", "model_name", "family"] # Definir campos de identidad

for field in identity_fields:
    if field not in evaluation_consolidated_result:
        raise RuntimeError(
            f"Falta el campo de identidad {field} en evaluation_consolidated_result."
        ) # Validar identidad de evaluación

    if field not in evaluation_temporal_consolidated_result:
        raise RuntimeError(
            f"Falta el campo de identidad {field} en evaluation_temporal_consolidated_result."
        ) # Validar identidad temporal

    if field not in evaluation_benchmark_consolidated_result:
        raise RuntimeError(
            f"Falta el campo de identidad {field} en evaluation_benchmark_consolidated_result."
        ) # Validar identidad Benchmark

structural_identity_consistent = (
    evaluation_consolidated_result["model_code"]
    == evaluation_temporal_consolidated_result["model_code"]
    == evaluation_benchmark_consolidated_result["model_code"]
    and
    evaluation_consolidated_result["model_name"].lower()
    == evaluation_temporal_consolidated_result["model_name"].lower()
    == evaluation_benchmark_consolidated_result["model_name"].lower()
    and
    evaluation_consolidated_result["family"].lower()
    == evaluation_temporal_consolidated_result["family"].lower()
    == evaluation_benchmark_consolidated_result["family"].lower()
) # Evaluar consistencia estructural

if not structural_identity_consistent:
    raise RuntimeError(
        "La identidad del Modelo Oficial no es consistente entre los productos consolidados."
    ) # Validar consistencia estructural

required_evaluation_metrics = ["rmse", "mae", "mape", "r2"] # Definir métricas requeridas

evaluation_metrics_normalized = {
    str(key).lower(): value
    for key, value in evaluation_consolidated_result["metrics"].items()
} # Normalizar nombres de métricas para validación

evaluation_metrics_available = all(
    metric in evaluation_metrics_normalized
    for metric in required_evaluation_metrics
) # Verificar cobertura de métricas Evaluation

benchmark_metrics_normalized = {
    str(key).lower(): value
    for key, value in evaluation_benchmark_consolidated_result["benchmark_metrics"].items()
} # Normalizar métricas Benchmark

benchmark_metrics_available = all(
    metric in benchmark_metrics_normalized
    for metric in required_evaluation_metrics
) # Verificar cobertura Benchmark

comparison_metrics_normalized = {
    str(key).lower(): value
    for key, value in evaluation_benchmark_consolidated_result["metric_comparison"].items()
} # Normalizar comparación de métricas

comparison_metrics_available = all(
    metric in comparison_metrics_normalized
    for metric in required_evaluation_metrics
) # Verificar cobertura comparativa

if not evaluation_metrics_available:
    raise RuntimeError(
        "La evaluación consolidada no contiene todas las métricas requeridas."
    ) # Validar cobertura Evaluation

if not benchmark_metrics_available:
    raise RuntimeError(
        "La consolidación Benchmark no contiene todas las métricas requeridas."
    ) # Validar cobertura Benchmark

if not comparison_metrics_available:
    raise RuntimeError(
        "La comparación consolidada no contiene todas las métricas requeridas."
    ) # Validar cobertura comparativa

comparison_status = evaluation_benchmark_consolidated_result["comparison_status"] # Recuperar estado numérico

if comparison_status not in {
    "NUMERICALLY_EQUIVALENT",
    "NUMERICALLY_DIFFERENT",
}:
    raise RuntimeError(
        f"Estado de comparación no reconocido: {comparison_status}"
    ) # Validar clasificación numérica

if comparison_status == "NUMERICALLY_EQUIVALENT":
    numerical_consistency_status = "EQUIVALENTE" # Clasificar equivalencia numérica
else:
    numerical_consistency_status = "DIFERENTE" # Clasificar diferencia numérica

different_metric_count = int(
    evaluation_benchmark_consolidated_result.get("different_metric_count", 0)
) # Recuperar cantidad de discrepancias

different_metrics = [
    str(metric).lower()
    for metric in evaluation_benchmark_consolidated_result.get("different_metrics", [])
] # Normalizar métricas diferentes

if numerical_consistency_status == "DIFERENTE" and different_metric_count == 0:
    raise RuntimeError(
        "La comparación indica diferencias, pero no registra métricas diferentes."
    ) # Validar coherencia diagnóstica

if numerical_consistency_status == "EQUIVALENTE" and different_metric_count != 0:
    raise RuntimeError(
        "La comparación indica equivalencia, pero registra métricas diferentes."
    ) # Validar coherencia diagnóstica

scientific_consistency_diagnostic = {
    "model_code": evaluation_consolidated_result["model_code"],
    "model_name": evaluation_consolidated_result["model_name"],
    "family": evaluation_consolidated_result["family"],
    "structural_consistency": {
        "status": "VALIDATED",
        "identity_consistent": structural_identity_consistent,
    },
    "metric_consistency": {
        "evaluation_metrics_available": evaluation_metrics_available,
        "benchmark_metrics_available": benchmark_metrics_available,
        "comparison_metrics_available": comparison_metrics_available,
        "status": "VALIDATED",
    },
    "numerical_consistency": {
        "comparison_status": comparison_status,
        "status": numerical_consistency_status,
        "different_metric_count": different_metric_count,
        "different_metrics": different_metrics,
    },
    "interpretation": "Las diferencias numéricas quedan documentadas y no se atribuye causalidad sin evidencia experimental.",
    "status": "VALIDATED",
} # Construir diagnóstico científico

evaluation_block_15_4_status = "VALIDATED" # Registrar diagnóstico validado
evaluation_block_15_4_stage = "SCIENTIFIC_CONSISTENCY_VALIDATED" # Registrar etapa validada

print(f"Modelo Oficial           : {scientific_consistency_diagnostic['model_name']}") # Mostrar modelo
print(f"Código Oficial           : {scientific_consistency_diagnostic['model_code']}") # Mostrar código
print(f"Familia Oficial          : {scientific_consistency_diagnostic['family']}") # Mostrar familia
print("Consistencia estructural : VALIDADA") # Confirmar identidad
print("Cobertura de métricas    : VALIDADA") # Confirmar métricas
print(f"Consistencia numérica    : {numerical_consistency_status}") # Mostrar consistencia numérica
print(f"Métricas diferentes      : {different_metrics}") # Mostrar discrepancias
print(f"Diferencias detectadas   : {different_metric_count}") # Mostrar cantidad
print("Interpretación causal    : NO ATRIBUIDA") # Evitar inferencia causal
print("Diagnóstico científico   : VALIDADO") # Confirmar diagnóstico
print(f"Estado Bloque 15.4       : {evaluation_block_15_4_status}") # Mostrar estado

print("\nBLOQUE 15.5. CONSTRUCCIÓN DEL RESULTADO CIENTÍFICO FINAL") # Mostrar encabezado
evaluation_block_15_5_status = "ERROR" # Inicializar estado
evaluation_block_15_5_stage = "FINAL_RESULT_CONSTRUCTION" # Registrar etapa actual

if evaluation_block_15_4_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.4 no está VALIDATED. Estado actual: {evaluation_block_15_4_status}"
    ) # Validar dependencia científica

if not isinstance(evaluation_consolidated_result, dict):
    raise TypeError(
        "evaluation_consolidated_result debe ser un diccionario."
    ) # Validar resultado de evaluación

if not isinstance(evaluation_temporal_consolidated_result, dict):
    raise TypeError(
        "evaluation_temporal_consolidated_result debe ser un diccionario."
    ) # Validar resultado temporal

if not isinstance(evaluation_benchmark_consolidated_result, dict):
    raise TypeError(
        "evaluation_benchmark_consolidated_result debe ser un diccionario."
    ) # Validar resultado Benchmark

if not isinstance(scientific_consistency_diagnostic, dict):
    raise TypeError(
        "scientific_consistency_diagnostic debe ser un diccionario."
    ) # Validar diagnóstico científico

required_final_components = {
    "evaluation": evaluation_consolidated_result,
    "temporal": evaluation_temporal_consolidated_result,
    "benchmark": evaluation_benchmark_consolidated_result,
    "scientific_diagnostic": scientific_consistency_diagnostic,
} # Definir componentes requeridos

missing_final_components = [
    name
    for name, value in required_final_components.items()
    if not value
] # Identificar componentes faltantes

if missing_final_components:
    raise RuntimeError(
        f"Faltan componentes para construir el resultado final: {missing_final_components}"
    ) # Validar cobertura

final_model_code = evaluation_consolidated_result["model_code"] # Recuperar código oficial
final_model_name = evaluation_consolidated_result["model_name"] # Recuperar modelo oficial
final_model_family = evaluation_consolidated_result["family"] # Recuperar familia oficial

if final_model_code != evaluation_temporal_consolidated_result["model_code"]:
    raise RuntimeError(
        "El código del modelo no coincide con el resultado temporal."
    ) # Validar trazabilidad temporal

if final_model_code != evaluation_benchmark_consolidated_result["model_code"]:
    raise RuntimeError(
        "El código del modelo no coincide con el resultado Benchmark."
    ) # Validar trazabilidad Benchmark

if final_model_code != scientific_consistency_diagnostic["model_code"]:
    raise RuntimeError(
        "El código del modelo no coincide con el diagnóstico científico."
    ) # Validar trazabilidad científica

EVALUATION_FINAL_RESULT = {
    "model_identity": {
        "model_code": final_model_code,
        "model_name": final_model_name,
        "family": final_model_family,
    },
    "evaluation": evaluation_consolidated_result,
    "temporal": evaluation_temporal_consolidated_result,
    "benchmark_comparison": evaluation_benchmark_consolidated_result,
    "scientific_diagnosis": scientific_consistency_diagnostic,
    "status": "VALIDATED",
} # Construir resultado científico final

evaluation_block_15_5_status = "VALIDATED" # Registrar construcción validada
evaluation_block_15_5_stage = "FINAL_RESULT_CONSTRUCTED" # Registrar etapa validada

print(f"Modelo Oficial           : {EVALUATION_FINAL_RESULT['model_identity']['model_name']}") # Mostrar modelo
print(f"Código Oficial           : {EVALUATION_FINAL_RESULT['model_identity']['model_code']}") # Mostrar código
print(f"Familia Oficial          : {EVALUATION_FINAL_RESULT['model_identity']['family']}") # Mostrar familia
print("Evaluación               : INTEGRADA") # Confirmar evaluación
print("Resultados temporales    : INTEGRADOS") # Confirmar temporalidad
print("Benchmark                : INTEGRADO") # Confirmar Benchmark
print("Diagnóstico científico   : INTEGRADO") # Confirmar diagnóstico
print("Identidad                : VALIDADA") # Confirmar identidad
print("Trazabilidad             : VALIDADA") # Confirmar trazabilidad
print("Resultado científico     : CONSTRUIDO") # Confirmar construcción
print(f"Estado Bloque 15.5       : {evaluation_block_15_5_status}") # Mostrar estado

print("\nBLOQUE 15.6. AUDITORÍA Y VALIDACIÓN FINAL DEL RESULTADO CIENTÍFICO") # Mostrar encabezado
evaluation_block_15_6_status = "ERROR" # Inicializar estado
evaluation_block_15_6_stage = "FINAL_AUDIT" # Registrar etapa actual

if evaluation_block_15_5_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.5 no está VALIDATED. Estado actual: {evaluation_block_15_5_status}"
    ) # Validar dependencia

if not isinstance(EVALUATION_FINAL_RESULT, dict):
    raise TypeError(
        "EVALUATION_FINAL_RESULT debe ser un diccionario."
    ) # Validar estructura final

if EVALUATION_FINAL_RESULT.get("status") != "VALIDATED":
    raise RuntimeError(
        "EVALUATION_FINAL_RESULT debe presentar estado VALIDATED."
    ) # Validar estado final

required_final_fields = [
    "model_identity",
    "evaluation",
    "temporal",
    "benchmark_comparison",
    "scientific_diagnosis",
    "status",
] # Definir contrato final

missing_final_fields = [
    field
    for field in required_final_fields
    if field not in EVALUATION_FINAL_RESULT
] # Identificar campos faltantes

if missing_final_fields:
    raise RuntimeError(
        f"Faltan componentes en EVALUATION_FINAL_RESULT: {missing_final_fields}"
    ) # Validar cobertura final

required_identity_fields = [
    "model_code",
    "model_name",
    "family",
] # Definir identidad requerida

missing_identity_fields = [
    field
    for field in required_identity_fields
    if field not in EVALUATION_FINAL_RESULT["model_identity"]
] # Identificar identidad faltante

if missing_identity_fields:
    raise RuntimeError(
        f"Faltan campos de identidad en EVALUATION_FINAL_RESULT: {missing_identity_fields}"
    ) # Validar identidad

final_model_code = EVALUATION_FINAL_RESULT["model_identity"]["model_code"] # Recuperar código
final_model_name = EVALUATION_FINAL_RESULT["model_identity"]["model_name"] # Recuperar modelo
final_model_family = EVALUATION_FINAL_RESULT["model_identity"]["family"] # Recuperar familia

if final_model_code != evaluation_consolidated_result["model_code"]:
    raise RuntimeError(
        "La identidad final no coincide con evaluation_consolidated_result."
    ) # Validar trazabilidad Evaluation

if final_model_code != evaluation_temporal_consolidated_result["model_code"]:
    raise RuntimeError(
        "La identidad final no coincide con evaluation_temporal_consolidated_result."
    ) # Validar trazabilidad temporal

if final_model_code != evaluation_benchmark_consolidated_result["model_code"]:
    raise RuntimeError(
        "La identidad final no coincide con evaluation_benchmark_consolidated_result."
    ) # Validar trazabilidad Benchmark

if final_model_code != scientific_consistency_diagnostic["model_code"]:
    raise RuntimeError(
        "La identidad final no coincide con scientific_consistency_diagnostic."
    ) # Validar trazabilidad científica

if EVALUATION_FINAL_RESULT["evaluation"].get("status") != "VALIDATED":
    raise RuntimeError(
        "El componente de evaluación no está VALIDATED."
    ) # Validar evaluación

if EVALUATION_FINAL_RESULT["temporal"].get("status") != "VALIDATED":
    raise RuntimeError(
        "El componente temporal no está VALIDATED."
    ) # Validar temporalidad

if EVALUATION_FINAL_RESULT["benchmark_comparison"].get("status") != "VALIDATED":
    raise RuntimeError(
        "El componente Benchmark no está VALIDATED."
    ) # Validar Benchmark

if EVALUATION_FINAL_RESULT["scientific_diagnosis"].get("status") != "VALIDATED":
    raise RuntimeError(
        "El diagnóstico científico no está VALIDATED."
    ) # Validar diagnóstico

evaluation_final_components_status = {
    "model_identity": "VALIDATED",
    "evaluation": "VALIDATED",
    "temporal": "VALIDATED",
    "benchmark_comparison": "VALIDATED",
    "scientific_diagnosis": "VALIDATED",
} # Consolidar estados de auditoría

evaluation_final_audit = {
    "model_code": final_model_code,
    "model_name": final_model_name,
    "family": final_model_family,
    "components": evaluation_final_components_status,
    "benchmark_comparison_status": EVALUATION_FINAL_RESULT[
        "benchmark_comparison"
    ]["comparison_status"],
    "scientific_diagnosis_status": EVALUATION_FINAL_RESULT[
        "scientific_diagnosis"
    ]["status"],
    "status": "VALIDATED",
} # Construir auditoría final

evaluation_block_15_6_status = "VALIDATED" # Registrar auditoría validada
evaluation_block_15_6_stage = "FINAL_AUDIT_VALIDATED" # Registrar etapa validada

evaluation_block_15_status = "VALIDATED" # Cerrar formalmente el Bloque 15
evaluation_block_15_stage = "SCIENTIFIC_EVALUATION_COMPLETED" # Registrar cierre global

print(f"Modelo Oficial           : {final_model_name}") # Mostrar modelo
print(f"Código Oficial           : {final_model_code}") # Mostrar código
print(f"Familia Oficial          : {final_model_family}") # Mostrar familia
print("Identidad final          : VALIDADA") # Confirmar identidad
print("Evaluación               : VALIDADA") # Confirmar evaluación
print("Temporalidad             : VALIDADA") # Confirmar temporalidad
print("Benchmark                : VALIDADO") # Confirmar Benchmark
print("Diagnóstico científico   : VALIDADO") # Confirmar diagnóstico
print(f"Estado comparación       : {evaluation_final_audit['benchmark_comparison_status']}") # Mostrar comparación
print("Trazabilidad             : VALIDADA") # Confirmar trazabilidad
print("Auditoría final          : VALIDADA") # Confirmar auditoría
print(f"Estado Bloque 15.6       : {evaluation_block_15_6_status}") # Mostrar estado
print(f"Estado Bloque 15         : {evaluation_block_15_status}") # Mostrar estado global

print("\nBLOQUE 15.7.0. DIAGNÓSTICO PREVIO DE PERSISTENCIA Y RECUPERACIÓN") # Mostrar encabezado
evaluation_block_15_7_status = "ERROR" # Inicializar estado
evaluation_block_15_7_stage = "PERSISTENCE_DIAGNOSTIC" # Registrar etapa actual
evaluation_block_15_7_0_status = "ERROR" # Inicializar estado específico

if evaluation_block_15_6_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.6 no está VALIDATED. Estado actual: {evaluation_block_15_6_status}"
    ) # Validar dependencia

if not isinstance(EVALUATION_FINAL_RESULT, dict):
    raise TypeError(
        "EVALUATION_FINAL_RESULT debe ser un diccionario."
    ) # Validar resultado final

if EVALUATION_FINAL_RESULT.get("status") != "VALIDATED":
    raise RuntimeError(
        "EVALUATION_FINAL_RESULT debe presentar estado VALIDATED."
    ) # Validar estado final

if not isinstance(evaluation_final_audit, dict):
    raise TypeError(
        "evaluation_final_audit debe ser un diccionario."
    ) # Validar auditoría final

if evaluation_final_audit.get("status") != "VALIDATED":
    raise RuntimeError(
        "evaluation_final_audit debe presentar estado VALIDATED."
    ) # Validar auditoría

required_persistence_objects = {
    "EVALUATION_CONFIG": EVALUATION_CONFIG,
    "EVALUATION_RESULT": EVALUATION_RESULT,
    "evaluation_temporal_result": evaluation_temporal_result,
    "evaluation_benchmark_consolidated_result": evaluation_benchmark_consolidated_result,
    "evaluation_consolidated_result": evaluation_consolidated_result,
    "evaluation_temporal_consolidated_result": evaluation_temporal_consolidated_result,
    "evaluation_benchmark_consolidated_result": evaluation_benchmark_consolidated_result,
    "scientific_consistency_diagnostic": scientific_consistency_diagnostic,
    "EVALUATION_FINAL_RESULT": EVALUATION_FINAL_RESULT,
    "evaluation_final_audit": evaluation_final_audit,
} # Definir productos científicos requeridos

missing_persistence_objects = [
    name
    for name, value in required_persistence_objects.items()
    if value is None
] # Identificar productos faltantes

if missing_persistence_objects:
    raise RuntimeError(
        f"Faltan productos para persistencia: {missing_persistence_objects}"
    ) # Validar cobertura

persistence_model_identity = EVALUATION_FINAL_RESULT["model_identity"] # Recuperar identidad final

if persistence_model_identity["model_code"] != "GNN02":
    raise RuntimeError(
        f"Código inesperado para persistencia: {persistence_model_identity['model_code']}"
    ) # Validar modelo oficial

if persistence_model_identity["model_name"].lower() != "graphsage":
    raise RuntimeError(
        f"Modelo inesperado para persistencia: {persistence_model_identity['model_name']}"
    ) # Validar modelo oficial

if persistence_model_identity["family"].lower() != "graph_neural_networks":
    raise RuntimeError(
        f"Familia inesperada para persistencia: {persistence_model_identity['family']}"
    ) # Validar familia oficial

evaluation_block_15_7_0_status = "VALIDATED" # Registrar diagnóstico validado
evaluation_block_15_7_stage = "PERSISTENCE_DIAGNOSTIC_VALIDATED" # Registrar etapa validada

print(f"Modelo Oficial           : {persistence_model_identity['model_name']}") # Mostrar modelo
print(f"Código Oficial           : {persistence_model_identity['model_code']}") # Mostrar código
print(f"Familia Oficial          : {persistence_model_identity['family']}") # Mostrar familia
print(f"Productos requeridos     : {len(required_persistence_objects)}") # Mostrar productos
print(f"Productos disponibles    : {len(required_persistence_objects) - len(missing_persistence_objects)}") # Mostrar disponibilidad
print("EVALUATION_FINAL_RESULT  : VALIDADO") # Confirmar resultado final
print("Auditoría final          : VALIDADA") # Confirmar auditoría
print("Identidad oficial        : VALIDADA") # Confirmar identidad
print("Productos de persistencia: DISPONIBLES") # Confirmar disponibilidad
print(f"Estado Bloque 15.7.0     : {evaluation_block_15_7_0_status}") # Mostrar estado

print("\nBLOQUE 15.7.1. DIAGNÓSTICO DE LAS RUTAS OFICIALES DE PERSISTENCIA") # Mostrar encabezado
evaluation_block_15_7_1_status = "ERROR" # Inicializar estado
evaluation_block_15_7_1_stage = "PERSISTENCE_PATH_DIAGNOSTIC" # Registrar etapa actual

if evaluation_block_15_7_0_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.7.0 no está VALIDATED. Estado actual: {evaluation_block_15_7_0_status}"
    ) # Validar dependencia

import src.python.config.paths as evaluation_paths # Cargar módulo oficial de rutas

evaluation_path_names = [
    name
    for name in dir(evaluation_paths)
    if "EVALUATION" in name
] # Identificar constantes oficiales de evaluación

if not evaluation_path_names:
    raise RuntimeError(
        "No se encontraron constantes EVALUATION en src.python.config.paths."
    ) # Validar disponibilidad de rutas

evaluation_path_values = {
    name: getattr(evaluation_paths, name)
    for name in evaluation_path_names
} # Recuperar valores de las rutas

evaluation_file_paths = {
    name: value
    for name, value in evaluation_path_values.items()
    if isinstance(value, (str, Path))
    and Path(value).suffix
} # Identificar rutas de archivos

evaluation_directory_paths = {
    name: value
    for name, value in evaluation_path_values.items()
    if isinstance(value, (str, Path))
    and not Path(value).suffix
} # Identificar rutas de directorios

required_persistence_path_candidates = [
    name
    for name in evaluation_path_names
    if any(
        token in name
        for token in [
            "RESULT",
            "SUMMARY",
            "REPORT",
            "METADATA",
            "MANIFEST",
            "AUDIT",
            "CERTIFICATE",
        ]
    )
] # Identificar candidatos de persistencia

if not required_persistence_path_candidates:
    raise RuntimeError(
        "No se encontraron rutas candidatas para persistencia de evaluación."
    ) # Validar candidatos

evaluation_block_15_7_1_status = "VALIDATED" # Registrar diagnóstico validado
evaluation_block_15_7_1_stage = "PERSISTENCE_PATHS_VALIDATED" # Registrar etapa validada

print(f"Constantes EVALUATION   : {len(evaluation_path_names)}") # Mostrar cantidad
print(f"Rutas de archivos       : {len(evaluation_file_paths)}") # Mostrar archivos
print(f"Rutas de directorios    : {len(evaluation_directory_paths)}") # Mostrar directorios
print(f"Candidatos persistencia : {len(required_persistence_path_candidates)}") # Mostrar candidatos
print(f"Constantes encontradas  : {evaluation_path_names}") # Mostrar constantes
print(f"Estado Bloque 15.7.1    : {evaluation_block_15_7_1_status}") # Mostrar estado

print("\nBLOQUE 15.7.2. DIAGNÓSTICO DEL DESTINO OFICIAL DE LOS RESULTADOS") # Mostrar encabezado
evaluation_block_15_7_2_status = "ERROR" # Inicializar estado
evaluation_block_15_7_2_stage = "RESULTS_PATH_DIAGNOSTIC" # Registrar etapa actual

if evaluation_block_15_7_1_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.7.1 no está VALIDATED. Estado actual: {evaluation_block_15_7_1_status}"
    ) # Validar dependencia

if not hasattr(evaluation_paths, "EVALUATION_RESULTS_FILE"):
    raise RuntimeError(
        "EVALUATION_RESULTS_FILE no existe en src.python.config.paths."
    ) # Validar ruta oficial

evaluation_results_path = Path(evaluation_paths.EVALUATION_RESULTS_FILE) # Recuperar ruta oficial de resultados

if not evaluation_results_path.suffix:
    raise RuntimeError(
        "EVALUATION_RESULTS_FILE no corresponde a un archivo."
    ) # Validar destino

evaluation_results_parent = evaluation_results_path.parent # Recuperar directorio padre
evaluation_results_suffix = evaluation_results_path.suffix.lower() # Recuperar extensión
evaluation_results_exists = evaluation_results_path.exists() # Verificar existencia actual
evaluation_results_parent_exists = evaluation_results_parent.exists() # Verificar directorio

supported_result_formats = {
    ".parquet",
    ".json",
    ".joblib",
    ".pkl",
    ".pickle",
} # Definir formatos soportados

if evaluation_results_suffix not in supported_result_formats:
    raise RuntimeError(
        f"Formato de resultados no reconocido: {evaluation_results_suffix}"
    ) # Validar formato soportado

if not evaluation_results_parent_exists:
    raise RuntimeError(
        f"El directorio de resultados no existe: {evaluation_results_parent}"
    ) # Validar directorio destino

evaluation_results_format = evaluation_results_suffix.replace(".", "").upper() # Normalizar formato
evaluation_block_15_7_2_status = "VALIDATED" # Registrar diagnóstico validado
evaluation_block_15_7_2_stage = "RESULTS_DESTINATION_VALIDATED" # Registrar etapa validada

print(f"Ruta resultados          : {evaluation_results_path}") # Mostrar ruta oficial
print(f"Extensión                : {evaluation_results_suffix}") # Mostrar formato
print(f"Formato                  : {evaluation_results_format}") # Mostrar formato normalizado
print(f"Directorio padre         : {evaluation_results_parent}") # Mostrar directorio
print(f"Directorio disponible    : {evaluation_results_parent_exists}") # Mostrar disponibilidad
print(f"Archivo existente        : {evaluation_results_exists}") # Mostrar existencia
print("Destino oficial          : VALIDADO") # Confirmar destino
print("Formato de persistencia  : VALIDADO") # Confirmar formato
print("Formato oficial          : PARQUET") # Confirmar formato oficial
print(f"Estado Bloque 15.7.2     : {evaluation_block_15_7_2_status}") # Mostrar estado

print("\nBLOQUE 15.7.3.0. DIAGNÓSTICO DE LA ESTRUCTURA REAL DE EVALUATION_RESULT") # Mostrar encabezado

evaluation_block_15_7_3_0_status = "ERROR" # Inicializar estado
evaluation_block_15_7_3_0_stage = "EVALUATION_RESULT_STRUCTURE_DIAGNOSTIC" # Registrar etapa actual

if evaluation_block_15_7_2_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.7.2 no está VALIDATED. Estado actual: {evaluation_block_15_7_2_status}"
    ) # Validar dependencia

if not isinstance(EVALUATION_RESULT, dict):
    raise TypeError(
        "EVALUATION_RESULT debe ser un diccionario."
    ) # Validar estructura

evaluation_result_keys = list(EVALUATION_RESULT.keys()) # Recuperar campos reales

evaluation_result_metric_keys = [
    key
    for key in evaluation_result_keys
    if str(key).lower() in {"rmse", "mae", "mape", "r2"}
] # Identificar métricas directas

evaluation_result_nested_metric_keys = [] # Inicializar métricas anidadas

if isinstance(EVALUATION_RESULT.get("metrics"), dict):
    evaluation_result_nested_metric_keys = list(
        EVALUATION_RESULT["metrics"].keys()
    ) # Recuperar métricas anidadas

evaluation_result_status = EVALUATION_RESULT.get("status") # Recuperar estado
evaluation_result_model_code = EVALUATION_RESULT.get("model_code") # Recuperar código
evaluation_result_model_name = EVALUATION_RESULT.get("model_name") # Recuperar modelo
evaluation_result_family = EVALUATION_RESULT.get("family") # Recuperar familia

print(f"Campos EVALUATION_RESULT : {evaluation_result_keys}") # Mostrar campos reales
print(f"Métricas directas        : {evaluation_result_metric_keys}") # Mostrar métricas directas
print(f"Métricas anidadas        : {evaluation_result_nested_metric_keys}") # Mostrar métricas anidadas
print(f"Modelo                   : {evaluation_result_model_name}") # Mostrar modelo
print(f"Código                   : {evaluation_result_model_code}") # Mostrar código
print(f"Familia                  : {evaluation_result_family}") # Mostrar familia
print(f"Estado                   : {evaluation_result_status}") # Mostrar estado
print(f"Índice TEST disponible   : {'test_index' in EVALUATION_RESULT}") # Verificar índice
print(f"GraphData TEST disponible: {'test_graphs' in EVALUATION_RESULT}") # Verificar grafos
print(f"Observaciones disponibles: {'n_observations' in EVALUATION_RESULT}") # Verificar observaciones
print(f"Conjunto disponible      : {'evaluation_set' in EVALUATION_RESULT}") # Verificar conjunto
print("Escritura en disco       : NO EJECUTADA") # Confirmar ausencia de persistencia
print(f"Estado Bloque 15.7.3.0   : {evaluation_block_15_7_3_0_status}") # Mostrar estado

print("\nBLOQUE 15.7.3. PERSISTENCIA DE LOS RESULTADOS DE EVALUACIÓN") # Mostrar encabezado
evaluation_block_15_7_3_status = "ERROR" # Inicializar estado
evaluation_block_15_7_3_stage = "EVALUATION_RESULTS_PERSISTENCE" # Registrar etapa actual

if evaluation_block_15_7_2_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.7.2 no está VALIDATED. Estado actual: {evaluation_block_15_7_2_status}"
    ) # Validar dependencia

if evaluation_block_15_7_3_0_status != "ERROR":
    raise RuntimeError(
        f"El diagnóstico 15.7.3.0 debe conservar estado ERROR por detección de estructura incompatible. Estado actual: {evaluation_block_15_7_3_0_status}"
    ) # Validar diagnóstico previo

if not isinstance(EVALUATION_RESULT, dict):
    raise TypeError(
        "EVALUATION_RESULT debe ser un diccionario."
    ) # Validar estructura

required_result_fields = [
    "model_code",
    "model_name",
    "family",
    "test_index",
    "test_graphs",
    "test_nodes",
    "n_observations",
    "prediction_source",
    "inference_time",
    "RMSE",
    "MAE",
    "MAPE",
    "R2",
    "metrics",
    "status",
] # Definir contrato real

missing_result_fields = [
    field
    for field in required_result_fields
    if field not in EVALUATION_RESULT
] # Identificar campos faltantes

if missing_result_fields:
    raise RuntimeError(
        f"Faltan campos requeridos en EVALUATION_RESULT: {missing_result_fields}"
    ) # Validar contrato real

if EVALUATION_RESULT["status"] != "VALIDATED":
    raise RuntimeError(
        "EVALUATION_RESULT debe presentar estado VALIDATED."
    ) # Validar estado

evaluation_metrics = EVALUATION_RESULT["metrics"] # Recuperar métricas

if not isinstance(evaluation_metrics, dict):
    raise TypeError(
        "EVALUATION_RESULT['metrics'] debe ser un diccionario."
    ) # Validar métricas

required_metrics = ["RMSE", "MAE", "MAPE", "R2"] # Definir métricas requeridas

missing_metrics = [
    metric
    for metric in required_metrics
    if metric not in evaluation_metrics
] # Identificar métricas faltantes

if missing_metrics:
    raise RuntimeError(
        f"Faltan métricas en EVALUATION_RESULT['metrics']: {missing_metrics}"
    ) # Validar métricas

evaluation_results_record = {
    "model_code": EVALUATION_RESULT["model_code"],
    "model_name": EVALUATION_RESULT["model_name"],
    "family": EVALUATION_RESULT["family"],
    "test_graphs": EVALUATION_RESULT["test_graphs"],
    "test_nodes": EVALUATION_RESULT["test_nodes"],
    "n_observations": EVALUATION_RESULT["n_observations"],
    "prediction_source": EVALUATION_RESULT["prediction_source"],
    "inference_time": EVALUATION_RESULT["inference_time"],
    "RMSE": EVALUATION_RESULT["RMSE"],
    "MAE": EVALUATION_RESULT["MAE"],
    "MAPE": EVALUATION_RESULT["MAPE"],
    "R2": EVALUATION_RESULT["R2"],
    "status": EVALUATION_RESULT["status"],
} # Construir registro tabular

evaluation_results_dataframe = pd.DataFrame(
    [evaluation_results_record]
) # Construir DataFrame

evaluation_results_dataframe.to_parquet(
    evaluation_results_path,
    index=False
) # Persistir resultados en Parquet

if not evaluation_results_path.exists():
    raise RuntimeError(
        "El archivo evaluation_results.parquet no fue creado."
    ) # Verificar creación

evaluation_results_size = evaluation_results_path.stat().st_size # Obtener tamaño físico

if evaluation_results_size <= 0:
    raise RuntimeError(
        "El archivo evaluation_results.parquet fue creado pero está vacío."
    ) # Validar contenido

evaluation_results_reloaded = pd.read_parquet(
    evaluation_results_path
) # Recuperar resultados persistidos

if evaluation_results_reloaded.empty:
    raise RuntimeError(
        "El archivo persistido no contiene registros."
    ) # Validar recuperación

if len(evaluation_results_reloaded) != 1:
    raise RuntimeError(
        f"Se esperaba un registro persistido y se recuperaron {len(evaluation_results_reloaded)}."
    ) # Validar cardinalidad

evaluation_block_15_7_3_status = "VALIDATED" # Registrar persistencia validada
evaluation_block_15_7_3_stage = "EVALUATION_RESULTS_PERSISTED_AND_VERIFIED" # Registrar etapa validada

print(f"Archivo                  : {evaluation_results_path}") # Mostrar archivo
print("Formato                  : PARQUET") # Mostrar formato
print(f"Registros                : {len(evaluation_results_reloaded)}") # Mostrar registros
print(f"Columnas                 : {list(evaluation_results_reloaded.columns)}") # Mostrar columnas
print(f"Tamaño bytes             : {evaluation_results_size}") # Mostrar tamaño
print(f"Modelo                   : {evaluation_results_reloaded.iloc[0]['model_name']}") # Mostrar modelo
print(f"Código                   : {evaluation_results_reloaded.iloc[0]['model_code']}") # Mostrar código
print(f"RMSE                     : {evaluation_results_reloaded.iloc[0]['RMSE']:.6f}") # Mostrar RMSE
print(f"MAE                      : {evaluation_results_reloaded.iloc[0]['MAE']:.6f}") # Mostrar MAE
print(f"MAPE                     : {evaluation_results_reloaded.iloc[0]['MAPE']:.6f}") # Mostrar MAPE
print(f"R2                       : {evaluation_results_reloaded.iloc[0]['R2']:.6f}") # Mostrar R2
print("Persistencia             : COMPLETADA") # Confirmar persistencia
print("Recuperación             : VALIDADA") # Confirmar recuperación
print("Integridad               : VALIDADA") # Confirmar integridad
print(f"Estado Bloque 15.7.3     : {evaluation_block_15_7_3_status}") # Mostrar estado

print("\nBLOQUE 15.7.4. PERSISTENCIA DEL PRODUCTO CIENTÍFICO COMPLETO") # Mostrar encabezado
evaluation_block_15_7_4_status = "ERROR" # Inicializar estado
evaluation_block_15_7_4_stage = "SCIENTIFIC_PRODUCT_PERSISTENCE" # Registrar etapa actual

if evaluation_block_15_7_3_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.7.3 no está VALIDATED. Estado actual: {evaluation_block_15_7_3_status}"
    ) # Validar dependencia

required_scientific_objects = {
    "EVALUATION_CONFIG": EVALUATION_CONFIG,
    "EVALUATION_RESULT": EVALUATION_RESULT,
    "evaluation_temporal_result": evaluation_temporal_result,
    "evaluation_consolidated_result": evaluation_consolidated_result,
    "evaluation_temporal_consolidated_result": evaluation_temporal_consolidated_result,
    "evaluation_benchmark_consolidated_result": evaluation_benchmark_consolidated_result,
    "scientific_consistency_diagnostic": scientific_consistency_diagnostic,
    "EVALUATION_FINAL_RESULT": EVALUATION_FINAL_RESULT,
    "evaluation_final_audit": evaluation_final_audit,
} # Definir productos científicos requeridos

missing_scientific_objects = [
    name
    for name, value in required_scientific_objects.items()
    if value is None
] # Identificar productos faltantes

if missing_scientific_objects:
    raise RuntimeError(
        f"Faltan productos científicos: {missing_scientific_objects}"
    ) # Validar cobertura

if not hasattr(evaluation_paths, "EVALUATION_METADATA_FILE"):
    raise RuntimeError(
        "EVALUATION_METADATA_FILE no existe en src.python.config.paths."
    ) # Validar ruta oficial

evaluation_metadata_path = Path(
    evaluation_paths.EVALUATION_METADATA_FILE
) # Recuperar ruta oficial de metadatos

if not evaluation_metadata_path.suffix:
    raise RuntimeError(
        "EVALUATION_METADATA_FILE no corresponde a un archivo."
    ) # Validar destino

evaluation_metadata_suffix = evaluation_metadata_path.suffix.lower() # Recuperar formato

if evaluation_metadata_suffix not in {".json", ".joblib", ".pkl", ".pickle"}:
    raise RuntimeError(
        f"Formato no compatible para producto científico estructurado: {evaluation_metadata_suffix}"
    ) # Validar formato

evaluation_metadata_path.parent.mkdir(
    parents=True,
    exist_ok=True
) # Crear directorio oficial si es necesario

scientific_product = {
    "evaluation_config": EVALUATION_CONFIG,
    "evaluation_result": EVALUATION_RESULT,
    "evaluation_temporal_result": evaluation_temporal_result,
    "evaluation_consolidated_result": evaluation_consolidated_result,
    "evaluation_temporal_consolidated_result": evaluation_temporal_consolidated_result,
    "evaluation_benchmark_consolidated_result": evaluation_benchmark_consolidated_result,
    "scientific_consistency_diagnostic": scientific_consistency_diagnostic,
    "evaluation_final_result": EVALUATION_FINAL_RESULT,
    "evaluation_final_audit": evaluation_final_audit,
    "model_identity": {
        "model_code": EVALUATION_RESULT["model_code"],
        "model_name": EVALUATION_RESULT["model_name"],
        "family": EVALUATION_RESULT["family"],
    },
    "status": "VALIDATED",
} # Construir producto científico completo

if evaluation_metadata_suffix == ".json":
    import json

    serializable_scientific_product = json.loads(
        json.dumps(
            scientific_product,
            default=lambda value: value.tolist()
            if hasattr(value, "tolist")
            else str(value)
        )
    ) # Preparar estructuras serializables

    with open(
        evaluation_metadata_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            serializable_scientific_product,
            file,
            ensure_ascii=False,
            indent=2
        ) # Persistir producto estructurado

elif evaluation_metadata_suffix in {".joblib", ".pkl", ".pickle"}:
    import joblib

    joblib.dump(
        scientific_product,
        evaluation_metadata_path
    ) # Persistir producto científico estructurado

if not evaluation_metadata_path.exists():
    raise RuntimeError(
        "El producto científico completo no fue persistido."
    ) # Verificar persistencia

evaluation_metadata_size = evaluation_metadata_path.stat().st_size # Obtener tamaño

if evaluation_metadata_size <= 0:
    raise RuntimeError(
        "El producto científico persistido está vacío."
    ) # Validar contenido físico

evaluation_block_15_7_4_status = "VALIDATED" # Registrar persistencia validada
evaluation_block_15_7_4_stage = "SCIENTIFIC_PRODUCT_PERSISTED" # Registrar etapa validada

print(f"Archivo                  : {evaluation_metadata_path}") # Mostrar archivo
print(f"Formato                  : {evaluation_metadata_suffix.upper().replace('.', '')}") # Mostrar formato
print(f"Productos científicos    : {len(required_scientific_objects)}") # Mostrar cantidad
print(f"Tamaño bytes             : {evaluation_metadata_size}") # Mostrar tamaño
print(f"Modelo                   : {scientific_product['model_identity']['model_name']}") # Mostrar modelo
print(f"Código                   : {scientific_product['model_identity']['model_code']}") # Mostrar código
print(f"Familia                  : {scientific_product['model_identity']['family']}") # Mostrar familia
print("Producto científico     : PERSISTIDO") # Confirmar persistencia
print("Integridad estructural   : VALIDADA") # Confirmar integridad
print(f"Estado Bloque 15.7.4     : {evaluation_block_15_7_4_status}") # Mostrar estado

print("\nBLOQUE 15.7.5. VERIFICACIÓN DE RECUPERACIÓN DEL PRODUCTO CIENTÍFICO") # Mostrar encabezado
evaluation_block_15_7_5_status = "ERROR" # Inicializar estado
evaluation_block_15_7_5_stage = "SCIENTIFIC_PRODUCT_RECOVERY" # Registrar etapa actual

if evaluation_block_15_7_4_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.7.4 no está VALIDATED. Estado actual: {evaluation_block_15_7_4_status}"
    ) # Validar dependencia

if not evaluation_metadata_path.exists():
    raise RuntimeError(
        f"No existe el producto científico persistido: {evaluation_metadata_path}"
    ) # Validar archivo

if evaluation_metadata_suffix != ".json":
    raise RuntimeError(
        f"El bloque espera JSON según la persistencia realizada: {evaluation_metadata_suffix}"
    ) # Validar formato

import json # Importar lector JSON

with open(
    evaluation_metadata_path,
    "r",
    encoding="utf-8"
) as file:
    recovered_scientific_product = json.load(file) # Recuperar producto científico

if not isinstance(recovered_scientific_product, dict):
    raise TypeError(
        "El producto científico recuperado debe ser un diccionario."
    ) # Validar recuperación

required_recovered_products = [
    "evaluation_config",
    "evaluation_result",
    "evaluation_temporal_result",
    "evaluation_consolidated_result",
    "evaluation_temporal_consolidated_result",
    "evaluation_benchmark_consolidated_result",
    "scientific_consistency_diagnostic",
    "evaluation_final_result",
    "evaluation_final_audit",
    "model_identity",
    "status",
] # Definir productos esperados

missing_recovered_products = [
    field
    for field in required_recovered_products
    if field not in recovered_scientific_product
] # Identificar productos faltantes

if missing_recovered_products:
    raise RuntimeError(
        f"Faltan productos después de la recuperación: {missing_recovered_products}"
    ) # Validar cobertura recuperada

recovered_identity = recovered_scientific_product["model_identity"] # Recuperar identidad
if recovered_identity["model_code"] != EVALUATION_RESULT["model_code"]:
    raise RuntimeError(
        "El código del modelo recuperado no coincide con EVALUATION_RESULT."
    ) # Validar código

if recovered_identity["model_name"].lower() != EVALUATION_RESULT["model_name"].lower():
    raise RuntimeError(
        "El nombre del modelo recuperado no coincide con EVALUATION_RESULT."
    ) # Validar nombre

if recovered_identity["family"].lower() != EVALUATION_RESULT["family"].lower():
    raise RuntimeError(
        "La familia del modelo recuperado no coincide con EVALUATION_RESULT."
    ) # Validar familia

recovered_evaluation_result = recovered_scientific_product["evaluation_result"] # Recuperar evaluación
required_recovered_evaluation_fields = [
    "model_code",
    "model_name",
    "family",
    "test_index",
    "test_graphs",
    "test_nodes",
    "n_observations",
    "prediction_source",
    "inference_time",
    "RMSE",
    "MAE",
    "MAPE",
    "R2",
    "metrics",
    "status",
] # Definir contrato recuperado

missing_recovered_evaluation_fields = [
    field
    for field in required_recovered_evaluation_fields
    if field not in recovered_evaluation_result
] # Identificar campos faltantes

if missing_recovered_evaluation_fields:
    raise RuntimeError(
        f"Faltan campos en evaluation_result recuperado: {missing_recovered_evaluation_fields}"
    ) # Validar evaluación recuperada

recovered_status = recovered_scientific_product["status"] # Recuperar estado científico
recovered_evaluation_status = recovered_evaluation_result["status"] # Recuperar estado Evaluation
if recovered_status != "VALIDATED":
    raise RuntimeError(
        f"Estado del producto recuperado no válido: {recovered_status}"
    ) # Validar estado global

if recovered_evaluation_status != "VALIDATED":
    raise RuntimeError(
        f"Estado de EVALUATION_RESULT recuperado no válido: {recovered_evaluation_status}"
    ) # Validar estado Evaluation

original_test_graphs_type = type(
    EVALUATION_RESULT["test_graphs"]
).__name__ # Identificar tipo original

recovered_test_graphs_type = type(
    recovered_evaluation_result["test_graphs"]
).__name__ # Identificar tipo recuperado

test_graphs_recovery_type_match = (
    original_test_graphs_type == recovered_test_graphs_type
) # Comparar tipos

evaluation_block_15_7_5_status = "VALIDATED" # Registrar recuperación validada
evaluation_block_15_7_5_stage = "SCIENTIFIC_PRODUCT_RECOVERY_VALIDATED" # Registrar etapa validada

print(f"Archivo recuperado       : {evaluation_metadata_path}") # Mostrar archivo
print(f"Productos recuperados    : {len(required_recovered_products)}") # Mostrar cantidad
print(f"Modelo recuperado        : {recovered_identity['model_name']}") # Mostrar modelo
print(f"Código recuperado        : {recovered_identity['model_code']}") # Mostrar código
print(f"Familia recuperada       : {recovered_identity['family']}") # Mostrar familia
print(f"Estado producto          : {recovered_status}") # Mostrar estado
print(f"Estado Evaluation        : {recovered_evaluation_status}") # Mostrar estado Evaluation
print(f"Tipo GraphData original  : {original_test_graphs_type}") # Mostrar tipo original
print(f"Tipo test_graphs recuperado: {recovered_test_graphs_type}") # Mostrar tipo recuperado
print(f"Tipo GraphData conservado: {test_graphs_recovery_type_match}") # Mostrar equivalencia de tipo
print("Identidad                : VALIDADA") # Confirmar identidad
print("Estructura científica    : RECUPERADA") # Confirmar estructura
print(f"Estado Bloque 15.7.5     : {evaluation_block_15_7_5_status}") # Mostrar estado

print("\nBLOQUE 15.7.6. DIAGNÓSTICO DE DISPONIBILIDAD REAL DE GRAPH DATA TEST") # Mostrar encabezado

evaluation_block_15_7_6_status = "ERROR" # Inicializar estado
evaluation_block_15_7_6_stage = "GRAPHDATA_TEST_DIAGNOSTIC" # Registrar etapa actual

if evaluation_block_15_7_5_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.7.5 no está VALIDATED. Estado actual: {evaluation_block_15_7_5_status}"
    ) # Validar dependencia

evaluation_graphdata_candidates = {
    "EVALUATION_RESULT.test_graphs": EVALUATION_RESULT.get("test_graphs"),
    "EVALUATION_RESULT.test_nodes": EVALUATION_RESULT.get("test_nodes"),
    "EVALUATION_RESULT.test_index": EVALUATION_RESULT.get("test_index"),
    "evaluation_temporal_result": evaluation_temporal_result,
    "evaluation_consolidated_result": evaluation_consolidated_result,
    "evaluation_temporal_consolidated_result": evaluation_temporal_consolidated_result,
} # Identificar fuentes candidatas

evaluation_graphdata_candidate_types = {
    name: type(value).__name__
    for name, value in evaluation_graphdata_candidates.items()
} # Identificar tipos reales

evaluation_graphdata_collection_candidates = {
    name: value
    for name, value in evaluation_graphdata_candidates.items()
    if isinstance(value, (list, tuple, dict))
} # Identificar estructuras potencialmente coleccionables

evaluation_graphdata_object_candidates = {
    name: value
    for name, value in evaluation_graphdata_candidates.items()
    if isinstance(value, (list, tuple))
    and len(value) > 0
} # Identificar colecciones no vacías

evaluation_graphdata_int_fields = [
    name
    for name, value in evaluation_graphdata_candidates.items()
    if isinstance(value, int)
] # Identificar campos que son contadores

print(f"test_graphs tipo          : {type(EVALUATION_RESULT.get('test_graphs')).__name__}") # Mostrar tipo
print(f"test_graphs valor         : {EVALUATION_RESULT.get('test_graphs')}") # Mostrar valor
print(f"test_nodes tipo           : {type(EVALUATION_RESULT.get('test_nodes')).__name__}") # Mostrar tipo
print(f"test_index tipo           : {type(EVALUATION_RESULT.get('test_index')).__name__}") # Mostrar tipo
print(f"Fuentes candidatas        : {list(evaluation_graphdata_candidates.keys())}") # Mostrar fuentes
print(f"Tipos candidatos          : {evaluation_graphdata_candidate_types}") # Mostrar tipos
print(f"Contadores identificados  : {evaluation_graphdata_int_fields}") # Mostrar contadores
print(f"Colecciones identificadas : {list(evaluation_graphdata_collection_candidates.keys())}") # Mostrar colecciones
print(f"Objetos potenciales       : {list(evaluation_graphdata_object_candidates.keys())}") # Mostrar candidatos
print("GraphData recuperados     : NO DETERMINADO") # Evitar conclusión prematura
print("Persistencia modificada   : NO") # Confirmar ausencia de escritura
print(f"Estado Bloque 15.7.6      : {evaluation_block_15_7_6_status}") # Mostrar estado

print("\nBLOQUE 15.7.6. DIAGNÓSTICO DE DISPONIBILIDAD REAL DE GRAPH DATA TEST") # Mostrar encabezado
evaluation_block_15_7_6_status = "ERROR" # Inicializar estado
evaluation_block_15_7_6_stage = "GRAPHDATA_TEST_DIAGNOSTIC" # Registrar etapa actual

if evaluation_block_15_7_5_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.7.5 no está VALIDATED. Estado actual: {evaluation_block_15_7_5_status}"
    ) # Validar dependencia

evaluation_graphdata_candidates = {
    "EVALUATION_RESULT.test_graphs": EVALUATION_RESULT.get("test_graphs"),
    "EVALUATION_RESULT.test_nodes": EVALUATION_RESULT.get("test_nodes"),
    "EVALUATION_RESULT.test_index": EVALUATION_RESULT.get("test_index"),
    "evaluation_temporal_result": evaluation_temporal_result,
    "evaluation_consolidated_result": evaluation_consolidated_result,
    "evaluation_temporal_consolidated_result": evaluation_temporal_consolidated_result,
} # Identificar fuentes candidatas

evaluation_graphdata_candidate_types = {
    name: type(value).__name__
    for name, value in evaluation_graphdata_candidates.items()
} # Identificar tipos reales

evaluation_graphdata_collection_candidates = {
    name: value
    for name, value in evaluation_graphdata_candidates.items()
    if isinstance(value, (list, tuple, dict))
} # Identificar estructuras potencialmente coleccionables

evaluation_graphdata_object_candidates = {
    name: value
    for name, value in evaluation_graphdata_candidates.items()
    if isinstance(value, (list, tuple))
    and len(value) > 0
} # Identificar colecciones no vacías

evaluation_graphdata_int_fields = [
    name
    for name, value in evaluation_graphdata_candidates.items()
    if isinstance(value, int)
] # Identificar campos que son contadores

print(f"test_graphs tipo          : {type(EVALUATION_RESULT.get('test_graphs')).__name__}") # Mostrar tipo
print(f"test_graphs valor         : {EVALUATION_RESULT.get('test_graphs')}") # Mostrar valor
print(f"test_nodes tipo           : {type(EVALUATION_RESULT.get('test_nodes')).__name__}") # Mostrar tipo
print(f"test_index tipo           : {type(EVALUATION_RESULT.get('test_index')).__name__}") # Mostrar tipo
print(f"Fuentes candidatas        : {list(evaluation_graphdata_candidates.keys())}") # Mostrar fuentes
print(f"Tipos candidatos          : {evaluation_graphdata_candidate_types}") # Mostrar tipos
print(f"Contadores identificados  : {evaluation_graphdata_int_fields}") # Mostrar contadores
print(f"Colecciones identificadas : {list(evaluation_graphdata_collection_candidates.keys())}") # Mostrar colecciones
print(f"Objetos potenciales       : {list(evaluation_graphdata_object_candidates.keys())}") # Mostrar candidatos
print("GraphData recuperados     : NO DETERMINADO") # Evitar conclusión prematura
print("Persistencia modificada   : NO") # Confirmar ausencia de escritura
print(f"Estado Bloque 15.7.6      : {evaluation_block_15_7_6_status}") # Mostrar estado

print("\nBLOQUE 15.7.7. DIAGNÓSTICO DE LA TRAZABILIDAD DE LAS ESTRUCTURAS TEST") # Mostrar encabezado
evaluation_block_15_7_7_status = "ERROR" # Inicializar estado
evaluation_block_15_7_7_stage = "TEST_STRUCTURE_TRACEABILITY_DIAGNOSTIC" # Registrar etapa actual

if evaluation_block_15_7_6_status != "ERROR":
    raise RuntimeError(
        f"El Bloque 15.7.6 debe conservar estado ERROR por tratarse de un diagnóstico no concluyente. Estado actual: {evaluation_block_15_7_6_status}"
    ) # Validar dependencia

if "test_index" not in EVALUATION_RESULT:
    raise RuntimeError(
        "EVALUATION_RESULT no contiene test_index."
    ) # Validar índice TEST

evaluation_test_index = EVALUATION_RESULT["test_index"] # Recuperar índice TEST
if not isinstance(evaluation_test_index, np.ndarray):
    raise TypeError(
        f"test_index debe ser ndarray. Tipo encontrado: {type(evaluation_test_index).__name__}"
    ) # Validar estructura del índice

evaluation_test_index_shape = evaluation_test_index.shape # Recuperar dimensión
evaluation_test_index_dtype = evaluation_test_index.dtype # Recuperar tipo de datos
evaluation_test_index_size = evaluation_test_index.size # Recuperar cantidad de elementos
evaluation_test_index_sample = evaluation_test_index[:10].tolist() # Obtener muestra del índice

evaluation_test_index_unique = np.unique(
    evaluation_test_index
) # Identificar valores únicos

evaluation_test_index_unique_count = len(
    evaluation_test_index_unique
) # Contar valores únicos

if evaluation_test_index_size == 0:
    raise RuntimeError(
        "test_index existe pero está vacío."
    ) # Validar contenido

evaluation_test_index_traceability = (
    evaluation_test_index_unique_count == evaluation_test_index_size
) # Evaluar unicidad del índice

print(f"Tipo test_index          : {type(evaluation_test_index).__name__}") # Mostrar tipo
print(f"Shape test_index        : {evaluation_test_index_shape}") # Mostrar dimensión
print(f"Dtype test_index        : {evaluation_test_index_dtype}") # Mostrar tipo de datos
print(f"Elementos test_index    : {evaluation_test_index_size}") # Mostrar cantidad
print(f"Valores únicos           : {evaluation_test_index_unique_count}") # Mostrar valores únicos
print(f"Muestra test_index      : {evaluation_test_index_sample}") # Mostrar muestra
print(f"Índice TEST único        : {evaluation_test_index_traceability}") # Mostrar unicidad
print(f"GraphData TEST           : {EVALUATION_RESULT.get('test_graphs')}") # Mostrar cantidad de grafos
print(f"Nodos TEST               : {EVALUATION_RESULT.get('test_nodes')}") # Mostrar nodos
print(f"Observaciones TEST       : {EVALUATION_RESULT.get('n_observations')}") # Mostrar observaciones
print("Objetos GraphData        : NO PRESENTES EN EVALUATION_RESULT") # Confirmar estructura
print("Referencia TEST          : DISPONIBLE MEDIANTE test_index") # Confirmar referencia
print("Persistencia modificada  : NO") # Confirmar ausencia de escritura

evaluation_block_15_7_7_status = "VALIDATED" # Registrar diagnóstico validado
evaluation_block_15_7_7_stage = "TEST_STRUCTURE_TRACEABILITY_VALIDATED" # Registrar etapa validada

print("Trazabilidad TEST        : VALIDADA") # Confirmar trazabilidad
print(f"Estado Bloque 15.7.7     : {evaluation_block_15_7_7_status}") # Mostrar estado

print("\nBLOQUE 15.7.8. PERSISTENCIA DE LA REFERENCIA TEST PARA FORECASTING") # Mostrar encabezado
evaluation_block_15_7_8_status = "ERROR" # Inicializar estado
evaluation_block_15_7_8_stage = "TEST_REFERENCE_PERSISTENCE" # Registrar etapa actual

if evaluation_block_15_7_7_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.7.7 no está VALIDATED. Estado actual: {evaluation_block_15_7_7_status}"
    ) # Validar dependencia

if not isinstance(evaluation_test_index, np.ndarray):
    raise TypeError(
        "evaluation_test_index debe ser un ndarray."
    ) # Validar índice TEST

if evaluation_test_index.size == 0:
    raise RuntimeError(
        "evaluation_test_index está vacío."
    ) # Validar contenido

test_reference = {
    "test_index": evaluation_test_index.tolist(),
    "n_test_graphs": int(EVALUATION_RESULT["test_graphs"]),
    "n_test_nodes": int(EVALUATION_RESULT["test_nodes"]),
    "n_test_observations": int(EVALUATION_RESULT["n_observations"]),
    "model_code": str(EVALUATION_RESULT["model_code"]),
    "model_name": str(EVALUATION_RESULT["model_name"]),
    "family": str(EVALUATION_RESULT["family"]),
    "status": "VALIDATED",
} # Construir referencia TEST

if len(test_reference["test_index"]) != test_reference["n_test_graphs"]:
    raise RuntimeError(
        "La cantidad de índices TEST no coincide con n_test_graphs."
    ) # Validar cardinalidad

if len(set(test_reference["test_index"])) != len(test_reference["test_index"]):
    raise RuntimeError(
        "Los índices TEST no son únicos."
    ) # Validar unicidad

if test_reference["model_code"] != "GNN02":
    raise RuntimeError(
        f"Código de modelo inesperado: {test_reference['model_code']}"
    ) # Validar modelo oficial

if test_reference["model_name"].lower() != "graphsage":
    raise RuntimeError(
        f"Modelo inesperado: {test_reference['model_name']}"
    ) # Validar modelo oficial

if test_reference["family"].lower() != "graph_neural_networks":
    raise RuntimeError(
        f"Familia inesperada: {test_reference['family']}"
    ) # Validar familia oficial

scientific_product["test_reference"] = test_reference # Incorporar referencia TEST al producto
with open(
    evaluation_metadata_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        scientific_product,
        file,
        ensure_ascii=False,
        indent=2,
        default=lambda value: value.tolist()
        if hasattr(value, "tolist")
        else str(value)
    ) # Actualizar producto científico persistido

if not evaluation_metadata_path.exists():
    raise RuntimeError(
        "El archivo de metadatos no existe después de actualizar la referencia TEST."
    ) # Verificar persistencia

evaluation_metadata_size = evaluation_metadata_path.stat().st_size # Obtener tamaño actualizado
if evaluation_metadata_size <= 0:
    raise RuntimeError(
        "El archivo de metadatos quedó vacío."
    ) # Validar contenido físico

with open(
    evaluation_metadata_path,
    "r",
    encoding="utf-8"
) as file:
    forecasting_reference_recovered = json.load(file) # Recuperar referencia persistida

if "test_reference" not in forecasting_reference_recovered:
    raise RuntimeError(
        "La referencia TEST no fue recuperada desde el archivo persistido."
    ) # Validar recuperación

recovered_test_reference = forecasting_reference_recovered["test_reference"] # Recuperar referencia TEST

if recovered_test_reference["test_index"] != test_reference["test_index"]:
    raise RuntimeError(
        "El test_index recuperado no coincide con el test_index original."
    ) # Validar índice

if recovered_test_reference["n_test_graphs"] != test_reference["n_test_graphs"]:
    raise RuntimeError(
        "n_test_graphs recuperado no coincide con el valor original."
    ) # Validar cantidad de grafos

if recovered_test_reference["n_test_nodes"] != test_reference["n_test_nodes"]:
    raise RuntimeError(
        "n_test_nodes recuperado no coincide con el valor original."
    ) # Validar nodos

if recovered_test_reference["n_test_observations"] != test_reference["n_test_observations"]:
    raise RuntimeError(
        "n_test_observations recuperado no coincide con el valor original."
    ) # Validar observaciones

evaluation_block_15_7_8_status = "VALIDATED" # Registrar persistencia validada
evaluation_block_15_7_8_stage = "TEST_REFERENCE_PERSISTED_AND_VERIFIED" # Registrar etapa validada

print(f"Archivo                  : {evaluation_metadata_path}") # Mostrar archivo
print(f"test_index               : {recovered_test_reference['test_index']}") # Mostrar índices TEST
print(f"GraphData TEST           : {recovered_test_reference['n_test_graphs']}") # Mostrar cantidad de grafos
print(f"Nodos TEST               : {recovered_test_reference['n_test_nodes']}") # Mostrar nodos
print(f"Observaciones TEST       : {recovered_test_reference['n_test_observations']}") # Mostrar observaciones
print(f"Modelo Oficial           : {recovered_test_reference['model_name']}") # Mostrar modelo
print(f"Código Oficial           : {recovered_test_reference['model_code']}") # Mostrar código
print(f"Familia Oficial          : {recovered_test_reference['family']}") # Mostrar familia
print("Referencia TEST          : PERSISTIDA") # Confirmar persistencia
print("Recuperación TEST        : VALIDADA") # Confirmar recuperación
print("Compatibilidad Forecasting: PREPARADA") # Confirmar disponibilidad
print(f"Estado Bloque 15.7.8     : {evaluation_block_15_7_8_status}") # Mostrar estado

print("\nBLOQUE 15.7.9. VERIFICACIÓN FINAL DE TRAZABILIDAD PARA FORECASTING") # Mostrar encabezado
evaluation_block_15_7_9_status = "ERROR" # Inicializar estado
evaluation_block_15_7_9_stage = "FORECASTING_TRACEABILITY_DIAGNOSTIC" # Registrar etapa actual

if evaluation_block_15_7_8_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.7.8 no está VALIDATED. Estado actual: {evaluation_block_15_7_8_status}"
    ) # Validar dependencia

if not isinstance(forecasting_reference_recovered, dict):
    raise TypeError(
        "forecasting_reference_recovered debe ser un diccionario."
    ) # Validar producto recuperado

required_forecasting_products = [
    "evaluation_config",
    "evaluation_result",
    "evaluation_temporal_result",
    "evaluation_consolidated_result",
    "evaluation_temporal_consolidated_result",
    "evaluation_benchmark_consolidated_result",
    "scientific_consistency_diagnostic",
    "evaluation_final_result",
    "evaluation_final_audit",
    "model_identity",
    "test_reference",
    "status",
] # Definir productos requeridos

missing_forecasting_products = [
    field
    for field in required_forecasting_products
    if field not in forecasting_reference_recovered
] # Identificar productos faltantes

if missing_forecasting_products:
    raise RuntimeError(
        f"Faltan productos requeridos para Forecasting: {missing_forecasting_products}"
    ) # Validar cobertura

forecasting_identity = forecasting_reference_recovered["model_identity"] # Recuperar identidad
forecasting_test_reference = forecasting_reference_recovered["test_reference"] # Recuperar referencia TEST
required_test_reference_fields = [
    "test_index",
    "n_test_graphs",
    "n_test_nodes",
    "n_test_observations",
    "model_code",
    "model_name",
    "family",
    "status",
] # Definir contrato TEST

missing_test_reference_fields = [
    field
    for field in required_test_reference_fields
    if field not in forecasting_test_reference
] # Identificar campos TEST faltantes

if missing_test_reference_fields:
    raise RuntimeError(
        f"Faltan campos en test_reference: {missing_test_reference_fields}"
    ) # Validar contrato TEST

if forecasting_identity["model_code"] != "GNN02":
    raise RuntimeError(
        f"Código del Modelo Oficial inesperado: {forecasting_identity['model_code']}"
    ) # Validar código oficial

if forecasting_identity["model_name"].lower() != "graphsage":
    raise RuntimeError(
        f"Modelo Oficial inesperado: {forecasting_identity['model_name']}"
    ) # Validar modelo oficial

if forecasting_identity["family"].lower() != "graph_neural_networks":
    raise RuntimeError(
        f"Familia del Modelo Oficial inesperada: {forecasting_identity['family']}"
    ) # Validar familia oficial

if forecasting_test_reference["test_index"] != [10, 11, 12]:
    raise RuntimeError(
        f"Referencia TEST inesperada: {forecasting_test_reference['test_index']}"
    ) # Validar índices TEST

if forecasting_test_reference["n_test_graphs"] != 3:
    raise RuntimeError(
        f"Cantidad inesperada de GraphData TEST: {forecasting_test_reference['n_test_graphs']}"
    ) # Validar cantidad de grafos

if forecasting_test_reference["n_test_nodes"] != 3363:
    raise RuntimeError(
        f"Cantidad inesperada de nodos TEST: {forecasting_test_reference['n_test_nodes']}"
    ) # Validar nodos TEST

if forecasting_test_reference["n_test_observations"] != 3363:
    raise RuntimeError(
        f"Cantidad inesperada de observaciones TEST: {forecasting_test_reference['n_test_observations']}"
    ) # Validar observaciones TEST

if forecasting_test_reference["model_code"] != forecasting_identity["model_code"]:
    raise RuntimeError(
        "El código del modelo no coincide entre model_identity y test_reference."
    ) # Validar identidad cruzada

if forecasting_test_reference["model_name"].lower() != forecasting_identity["model_name"].lower():
    raise RuntimeError(
        "El nombre del modelo no coincide entre model_identity y test_reference."
    ) # Validar identidad cruzada

if forecasting_test_reference["family"].lower() != forecasting_identity["family"].lower():
    raise RuntimeError(
        "La familia del modelo no coincide entre model_identity y test_reference."
    ) # Validar identidad cruzada

if forecasting_test_reference["status"] != "VALIDATED":
    raise RuntimeError(
        "La referencia TEST no presenta estado VALIDATED."
    ) # Validar estado TEST

if forecasting_reference_recovered["status"] != "VALIDATED":
    raise RuntimeError(
        "El producto científico recuperado no presenta estado VALIDATED."
    ) # Validar estado global

forecasting_traceability_result = {
    "model_identity": forecasting_identity,
    "test_reference": forecasting_test_reference,
    "scientific_products_available": len(required_forecasting_products),
    "scientific_products_missing": len(missing_forecasting_products),
    "forecasting_ready": True,
    "status": "VALIDATED",
} # Construir diagnóstico final de trazabilidad

evaluation_block_15_7_9_status = "VALIDATED" # Registrar trazabilidad validada
evaluation_block_15_7_9_stage = "FORECASTING_TRACEABILITY_VALIDATED" # Registrar etapa validada

print(f"Modelo Oficial           : {forecasting_identity['model_name']}") # Mostrar modelo
print(f"Código Oficial           : {forecasting_identity['model_code']}") # Mostrar código
print(f"Familia Oficial          : {forecasting_identity['family']}") # Mostrar familia
print(f"test_index               : {forecasting_test_reference['test_index']}") # Mostrar referencia TEST
print(f"GraphData TEST           : {forecasting_test_reference['n_test_graphs']}") # Mostrar cantidad
print(f"Nodos TEST               : {forecasting_test_reference['n_test_nodes']}") # Mostrar nodos
print(f"Observaciones TEST       : {forecasting_test_reference['n_test_observations']}") # Mostrar observaciones
print(f"Productos requeridos     : {len(required_forecasting_products)}") # Mostrar productos
print(f"Productos faltantes      : {len(missing_forecasting_products)}") # Mostrar faltantes
print("Identidad Forecasting    : VALIDADA") # Confirmar identidad
print("Referencia TEST          : VALIDADA") # Confirmar referencia
print("Trazabilidad científica  : VALIDADA") # Confirmar trazabilidad
print("Forecasting preparado    : SI") # Confirmar preparación
print(f"Estado Bloque 15.7.9     : {evaluation_block_15_7_9_status}") # Mostrar estado

print("\nBLOQUE 15.7.10. CIERRE Y CERTIFICACIÓN DE PERSISTENCIA PARA FORECASTING") # Mostrar encabezado
evaluation_block_15_7_10_status = "ERROR" # Inicializar estado
evaluation_block_15_7_10_stage = "PERSISTENCE_CERTIFICATION" # Registrar etapa actual

if evaluation_block_15_7_9_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.7.9 no está VALIDATED. Estado actual: {evaluation_block_15_7_9_status}"
    ) # Validar dependencia

if not isinstance(forecasting_traceability_result, dict):
    raise TypeError(
        "forecasting_traceability_result debe ser un diccionario."
    ) # Validar diagnóstico

if forecasting_traceability_result.get("status") != "VALIDATED":
    raise RuntimeError(
        "forecasting_traceability_result debe presentar estado VALIDATED."
    ) # Validar estado

if not isinstance(forecasting_reference_recovered, dict):
    raise TypeError(
        "forecasting_reference_recovered debe ser un diccionario."
    ) # Validar producto recuperado

if "test_reference" not in forecasting_reference_recovered:
    raise RuntimeError(
        "El producto persistido no contiene test_reference."
    ) # Validar referencia TEST

final_test_reference = forecasting_reference_recovered["test_reference"] # Recuperar referencia final
if final_test_reference["test_index"] != [10, 11, 12]:
    raise RuntimeError(
        f"Referencia TEST final inesperada: {final_test_reference['test_index']}"
    ) # Validar índice final

if final_test_reference["n_test_graphs"] != 3:
    raise RuntimeError(
        f"Cantidad final de GraphData TEST inesperada: {final_test_reference['n_test_graphs']}"
    ) # Validar cantidad final

if final_test_reference["n_test_nodes"] != 3363:
    raise RuntimeError(
        f"Cantidad final de nodos TEST inesperada: {final_test_reference['n_test_nodes']}"
    ) # Validar nodos finales

if final_test_reference["n_test_observations"] != 3363:
    raise RuntimeError(
        f"Cantidad final de observaciones TEST inesperada: {final_test_reference['n_test_observations']}"
    ) # Validar observaciones finales

if not evaluation_metadata_path.exists():
    raise RuntimeError(
        "El archivo de metadatos de evaluación no existe."
    ) # Validar persistencia física

if not evaluation_results_path.exists():
    raise RuntimeError(
        "El archivo de resultados de evaluación no existe."
    ) # Validar resultados persistidos

evaluation_metadata_final_size = evaluation_metadata_path.stat().st_size # Obtener tamaño final de metadatos
evaluation_results_final_size = evaluation_results_path.stat().st_size # Obtener tamaño final de resultados
if evaluation_metadata_final_size <= 0:
    raise RuntimeError(
        "El archivo de metadatos de evaluación está vacío."
    ) # Validar contenido

if evaluation_results_final_size <= 0:
    raise RuntimeError(
        "El archivo de resultados de evaluación está vacío."
    ) # Validar contenido

evaluation_persistence_certificate = {
    "model_identity": forecasting_reference_recovered["model_identity"],
    "test_reference": final_test_reference,
    "evaluation_metadata_file": str(evaluation_metadata_path),
    "evaluation_results_file": str(evaluation_results_path),
    "evaluation_metadata_size": evaluation_metadata_final_size,
    "evaluation_results_size": evaluation_results_final_size,
    "persistence_status": "VALIDATED",
    "recovery_status": "VALIDATED",
    "forecasting_traceability_status": "VALIDATED",
    "graphdata_loading_status": "PENDING",
    "status": "VALIDATED",
} # Construir certificado de persistencia

evaluation_block_15_7_status = "VALIDATED" # Cerrar estado global del Bloque 15.7
evaluation_block_15_7_stage = "PERSISTENCE_AND_FORECASTING_TRACEABILITY_COMPLETED" # Registrar cierre global

evaluation_block_15_7_10_status = "VALIDATED" # Registrar certificación validada
evaluation_block_15_7_10_stage = "PERSISTENCE_CERTIFICATION_VALIDATED" # Registrar etapa final

print(f"Modelo Oficial           : {evaluation_persistence_certificate['model_identity']['model_name']}") # Mostrar modelo
print(f"Código Oficial           : {evaluation_persistence_certificate['model_identity']['model_code']}") # Mostrar código
print(f"test_index               : {evaluation_persistence_certificate['test_reference']['test_index']}") # Mostrar índice TEST
print(f"GraphData TEST           : {evaluation_persistence_certificate['test_reference']['n_test_graphs']}") # Mostrar cantidad
print(f"Nodos TEST               : {evaluation_persistence_certificate['test_reference']['n_test_nodes']}") # Mostrar nodos
print(f"Observaciones TEST       : {evaluation_persistence_certificate['test_reference']['n_test_observations']}") # Mostrar observaciones
print(f"Metadatos persistidos    : {evaluation_metadata_path}") # Mostrar metadatos
print(f"Resultados persistidos   : {evaluation_results_path}") # Mostrar resultados
print(f"Tamaño metadatos         : {evaluation_metadata_final_size}") # Mostrar tamaño
print(f"Tamaño resultados        : {evaluation_results_final_size}") # Mostrar tamaño
print("Persistencia             : VALIDADA") # Confirmar persistencia
print("Recuperación             : VALIDADA") # Confirmar recuperación
print("Trazabilidad Forecasting : VALIDADA") # Confirmar trazabilidad
print("Carga GraphData          : PENDIENTE") # Diferenciar carga efectiva
print("Bloque 15.7              : VALIDATED") # Confirmar cierre global
print(f"Estado Bloque 15.7.10    : {evaluation_block_15_7_10_status}") # Mostrar estado

print("\nBLOQUE 15.7.11. VERIFICACIÓN FÍSICA DE GRAPH DATA TEST PARA FORECASTING") # Mostrar encabezado
evaluation_block_15_7_11_status = "ERROR" # Inicializar estado
evaluation_block_15_7_11_stage = "GRAPHDATA_PHYSICAL_VERIFICATION" # Registrar etapa actual

if evaluation_block_15_7_10_status != "VALIDATED":
    raise RuntimeError(
        f"El Bloque 15.7.10 no está VALIDATED. Estado actual: {evaluation_block_15_7_10_status}"
    ) # Validar dependencia

if not isinstance(forecasting_test_reference, dict):
    raise TypeError(
        "forecasting_test_reference debe ser un diccionario."
    ) # Validar referencia TEST

required_test_indices = forecasting_test_reference["test_index"] # Recuperar índices TEST

if required_test_indices != [10, 11, 12]:
    raise RuntimeError(
        f"Índices TEST inesperados: {required_test_indices}"
    ) # Validar referencia oficial

if forecasting_test_reference["n_test_graphs"] != 3:
    raise RuntimeError(
        f"Cantidad inesperada de GraphData TEST: {forecasting_test_reference['n_test_graphs']}"
    ) # Validar cantidad de grafos

project_root_candidates = [
    Path.cwd(),
    Path.cwd().parent,
    Path.cwd().parent.parent,
    Path.cwd().parent.parent.parent,
] # Definir candidatos de raíz del proyecto

project_root = None # Inicializar raíz

for candidate in project_root_candidates:
    if (
        (candidate / "src").exists()
        and (candidate / "src" / "Python").exists()
    ):
        project_root = candidate
        break # Identificar raíz real del proyecto

if project_root is None:
    raise RuntimeError(
        "No fue posible identificar la raíz del proyecto mediante src/Python."
    ) # Validar raíz del proyecto

print(f"Raíz proyecto           : {project_root}") # Mostrar raíz

import src.python.config.paths as evaluation_paths # Cargar rutas oficiales

path_module_names = [
    name
    for name in dir(evaluation_paths)
    if any(
        token in name.upper()
        for token in [
            "GRAPH",
            "DATA",
            "DATASET",
            "OUTPUT",
            "MODEL",
            "TRAIN",
            "TEST",
        ]
    )
] # Identificar constantes potencialmente relacionadas

path_module_values = {
    name: getattr(evaluation_paths, name)
    for name in path_module_names
} # Recuperar valores de rutas

path_candidates = {
    name: Path(value)
    for name, value in path_module_values.items()
    if isinstance(value, (str, Path))
} # Convertir rutas válidas

print(f"Constantes candidatas   : {len(path_candidates)}") # Mostrar cantidad
print(f"Constantes encontradas  : {list(path_candidates.keys())}") # Mostrar constantes

graph_file_extensions = {
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
    ".joblib",
    ".bin",
    ".ptz",
} # Definir formatos potenciales de GraphData

graph_file_candidates = [] # Inicializar archivos candidatos

search_directories = [
    project_root / "src" / "Python",
    project_root / "src" / "Python" / "outputs",
    project_root / "src" / "Python" / "data",
    project_root / "src" / "Python" / "models",
    project_root / "src" / "Python" / "artifacts",
    project_root / "data",
    project_root / "outputs",
    project_root / "artifacts",
] # Definir directorios de búsqueda

search_directories = [
    directory
    for directory in search_directories
    if directory.exists()
] # Conservar directorios existentes

for directory in search_directories:
    try:
        for file_path in directory.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in graph_file_extensions
            ):
                graph_file_candidates.append(file_path)
    except (PermissionError, OSError):
        continue # Ignorar directorios no accesibles

graph_file_candidates = sorted(
    set(graph_file_candidates)
) # Eliminar duplicados y ordenar

print(f"Archivos Graph candidatos: {len(graph_file_candidates)}") # Mostrar cantidad

if not graph_file_candidates:
    raise RuntimeError(
        "No se encontraron archivos candidatos que puedan contener GraphData."
    ) # Validar disponibilidad física

graph_test_candidates = [] # Inicializar candidatos TEST

for file_path in graph_file_candidates:
    file_name_lower = file_path.name.lower()

    if any(
        token in file_name_lower
        for token in [
            "graph",
            "grapdata",
            "graph_data",
            "dataset",
            "data",
        ]
    ):
        graph_test_candidates.append(file_path) # Filtrar archivos relacionados con grafos

print(f"Candidatos GraphData    : {len(graph_test_candidates)}") # Mostrar cantidad

if not graph_test_candidates:
    raise RuntimeError(
        "Se encontraron archivos serializados, pero ninguno presenta una nomenclatura compatible con GraphData."
    ) # Validar nomenclatura

test_index_strings = [
    str(index)
    for index in required_test_indices
] # Preparar índices para búsqueda

indexed_graph_candidates = {
    index: []
    for index in required_test_indices
} # Inicializar candidatos por índice

for file_path in graph_test_candidates:
    file_name_lower = file_path.stem.lower()

    for index in required_test_indices:
        index_tokens = [
            f"{index}",
            f"_{index}",
            f"-{index}",
            f"year_{index}",
            f"graph_{index}",
            f"graphdata_{index}",
            f"data_{index}",
        ] # Definir patrones de índice

        if any(token in file_name_lower for token in index_tokens):
            indexed_graph_candidates[index].append(file_path) # Asociar archivo con índice

print(
    f"Candidatos por índice   : "
    f"{ {index: len(files) for index, files in indexed_graph_candidates.items()} }"
) # Mostrar cobertura

missing_graph_indices = [
    index
    for index, files in indexed_graph_candidates.items()
    if not files
] # Identificar índices sin candidato físico

if missing_graph_indices:
    raise RuntimeError(
        f"No se encontró un archivo físico candidato para los GraphData TEST: {missing_graph_indices}"
    ) # Validar cobertura física

print("Cobertura física TEST    : COMPLETA") # Confirmar cobertura

import torch # Importar PyTorch

loaded_graphdata = {} # Inicializar objetos recuperados
graphdata_load_errors = {} # Inicializar errores de carga

for index in required_test_indices:
    candidates = indexed_graph_candidates[index] # Recuperar candidatos del índice
    loaded = False # Control de carga
    for candidate in candidates:
        try:
            loaded_object = torch.load(
                candidate,
                map_location="cpu",
                weights_only=False
            ) # Intentar recuperar objeto serializado

            loaded_graphdata[index] = {
                "path": candidate,
                "object": loaded_object,
            } # Registrar objeto recuperado

            loaded = True
            break # Detener búsqueda al encontrar un objeto cargable

        except Exception as error:
            graphdata_load_errors.setdefault(index, []).append(
                {
                    "path": str(candidate),
                    "error": str(error),
                }
            ) # Registrar error de carga

    if not loaded:
        raise RuntimeError(
            f"No fue posible cargar físicamente el GraphData TEST asociado al índice {index}."
        ) # Validar recuperación física

print(f"GraphData cargados: {len(loaded_graphdata)}") # Mostrar cantidad

if len(loaded_graphdata) != len(required_test_indices):
    raise RuntimeError(
        "La cantidad de GraphData cargados no coincide con la cantidad esperada."
    ) # Validar cardinalidad

graphdata_structure_validation = {} # Inicializar diagnóstico estructural
for index, payload in loaded_graphdata.items():
    graph_object = payload["object"] # Recuperar objeto
    graph_type = type(graph_object).__name__ # Identificar tipo
    has_x = hasattr(graph_object, "x") or (
        isinstance(graph_object, dict) and "x" in graph_object
    ) # Verificar características

    has_edge_index = hasattr(graph_object, "edge_index") or (
        isinstance(graph_object, dict) and "edge_index" in graph_object
    ) # Verificar estructura topológica

    has_y = hasattr(graph_object, "y") or (
        isinstance(graph_object, dict) and "y" in graph_object
    ) # Verificar variable objetivo

    graphdata_structure_validation[index] = {
        "type": graph_type,
        "has_x": has_x,
        "has_edge_index": has_edge_index,
        "has_y": has_y,
    } # Registrar estructura

    if not has_x:
        raise RuntimeError(
            f"GraphData TEST {index} no contiene atributo x."
        ) # Validar características

    if not has_edge_index:
        raise RuntimeError(
            f"GraphData TEST {index} no contiene edge_index."
        ) # Validar topología

    if not has_y:
        raise RuntimeError(
            f"GraphData TEST {index} no contiene y."
        ) # Validar variable objetivo

print(f"Estructuras GraphData: {graphdata_structure_validation}") # Mostrar diagnóstico estructural

evaluation_graphdata_test_reference = {
    "test_index": required_test_indices,
    "n_test_graphs": len(loaded_graphdata),
    "loaded_graphdata": {
        str(index): {
            "path": str(payload["path"]),
            "type": graphdata_structure_validation[index]["type"],
        }
        for index, payload in loaded_graphdata.items()
    },
    "structure_validated": True,
    "status": "VALIDATED",
} # Construir referencia física TEST

scientific_product["graphdata_test_reference"] = (
    evaluation_graphdata_test_reference
) # Incorporar referencia física al producto científico

with open(
    evaluation_metadata_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        scientific_product,
        file,
        ensure_ascii=False,
        indent=2,
        default=lambda value: value.tolist()
        if hasattr(value, "tolist")
        else str(value)
    ) # Persistir referencia física GraphData

if not evaluation_metadata_path.exists():
    raise RuntimeError(
        "El archivo de metadatos no existe después de persistir la referencia GraphData."
    ) # Validar persistencia

with open(
    evaluation_metadata_path,
    "r",
    encoding="utf-8"
) as file:
    graphdata_reference_recovered = json.load(file) # Recuperar referencia física

if "graphdata_test_reference" not in graphdata_reference_recovered:
    raise RuntimeError(
        "La referencia física GraphData no pudo recuperarse."
    ) # Validar recuperación

recovered_graphdata_reference = (
    graphdata_reference_recovered["graphdata_test_reference"]
) # Recuperar referencia persistida

if recovered_graphdata_reference["test_index"] != required_test_indices:
    raise RuntimeError(
        "Los índices TEST recuperados no coinciden con los índices originales."
    ) # Validar índices

if recovered_graphdata_reference["n_test_graphs"] != 3:
    raise RuntimeError(
        "La cantidad de GraphData recuperados no coincide con 3."
    ) # Validar cantidad

evaluation_block_15_7_11_status = "VALIDATED" # Registrar verificación validada
evaluation_block_15_7_11_stage = "GRAPHDATA_PHYSICAL_VERIFICATION_VALIDATED" # Registrar etapa validada

print(f"Índices TEST             : {required_test_indices}") # Mostrar índices
print(f"GraphData encontrados    : {len(graph_file_candidates)}") # Mostrar archivos
print(f"GraphData TEST cargados  : {len(loaded_graphdata)}") # Mostrar cargados
print(f"GraphData TEST esperados : {forecasting_test_reference['n_test_graphs']}") # Mostrar esperados
print(f"Referencia física        : PERSISTIDA") # Confirmar persistencia
print(f"Recuperación física      : VALIDADA") # Confirmar recuperación
print(f"Estructura GraphData     : VALIDADA") # Confirmar estructura
print("Carga Forecasting        : DISPONIBLE") # Confirmar disponibilidad
print(f"Estado Bloque 15.7.11    : {evaluation_block_15_7_11_status}") # Mostrar estado