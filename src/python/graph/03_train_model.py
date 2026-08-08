# graph-03_train_model.py

# ==============================================================================
# BLOQUE 1. IMPORTACIONES
# Objetivo:
# Importar las dependencias oficiales requeridas para ejecutar el entrenamiento
# definitivo del Modelo Oficial seleccionado durante el Benchmark Científico,
# incluyendo las librerías científicas, la configuración del proyecto, las
# rutas institucionales y el pipeline oficial de entrenamiento.
#
# Producto:
# - Dependencias científicas cargadas correctamente.
# - Configuración oficial disponible.
# - Pipeline oficial de entrenamiento disponible.
#
# Responde:
# ¿Se encuentran disponibles todas las dependencias necesarias para ejecutar
# el entrenamiento reproducible del Modelo Oficial del proyecto?
# ==============================================================================

print("-" * 80)
print("Bloque 1. Importaciones.")

# ------------------------------------------------------------------------------
# Librerías del sistema
# ------------------------------------------------------------------------------

import json
import random
import warnings
from datetime import datetime
from pathlib import Path

import joblib

# ------------------------------------------------------------------------------
# Librerías científicas
# ------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------------------------
# Configuración del proyecto
# ------------------------------------------------------------------------------

from src.python.config.paths import (
    OFFICIAL_MODEL_CONFIG_FILE,
    OFFICIAL_MODEL_METADATA_FILE,
    OFFICIAL_MODEL_TORCH_FILE,

    BENCHMARK_DATA_FILE,
    TRAINING_SUMMARY_FILE,
)

from src.python.config.config_project import (
    PROJECT_SEED,
    BENCHMARK_CONFIG,
    BENCHMARK_MODELS,
    BENCHMARK_METRICS,
    BENCHMARK_REPRODUCIBILITY,
    OFFICIAL_MODEL_FAMILY,
    OFFICIAL_MODEL_NAME,
)

# ------------------------------------------------------------------------------
# Modelos Graph Neural Networks
# ------------------------------------------------------------------------------

from src.python.models.graph_neural_networks import (
    GNN_CONFIG,
    run_gnn_training,
)

# ------------------------------------------------------------------------------
# Configuración de reproducibilidad
# ------------------------------------------------------------------------------

warnings.filterwarnings("ignore")

random.seed(PROJECT_SEED)
np.random.seed(PROJECT_SEED)
torch.manual_seed(PROJECT_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(PROJECT_SEED)
    torch.cuda.manual_seed_all(PROJECT_SEED)

print("-" * 80)
print("Bloque 1. Importaciones cargadas correctamente.")

# ==============================================================================
# BLOQUE 2. CONFIGURACIÓN DEL ENTORNO DE ENTRENAMIENTO
# Objetivo:
# Configurar el entorno de ejecución, validar la configuración oficial del
# entrenamiento definitivo, establecer las condiciones de reproducibilidad y
# seleccionar el dispositivo de procesamiento para entrenar el Modelo Oficial.
#
# Producto:
# - Entorno de entrenamiento configurado.
# - Configuración oficial validada.
# - Reproducibilidad establecida.
# - Dispositivo de procesamiento seleccionado.
#
# Pregunta científica:
# ¿El entorno de ejecución cumple las condiciones necesarias para entrenar el
# Modelo Oficial de forma reproducible y consistente?
# ==============================================================================

print("\n" + "-" * 80)
print("BLOQUE 2. CONFIGURACIÓN DEL ENTORNO DE ENTRENAMIENTO")
print("-" * 80)

# ------------------------------------------------------------------------------
# 2.1 Validación de la configuración oficial
# ------------------------------------------------------------------------------

if PROJECT_SEED is None:
    raise ValueError(
        "La semilla oficial del proyecto es inválida."
    )

if not isinstance(BENCHMARK_REPRODUCIBILITY, dict):
    raise TypeError(
        "'BENCHMARK_REPRODUCIBILITY' debe ser un diccionario."
    )

if "deterministic" not in BENCHMARK_REPRODUCIBILITY:
    raise KeyError(
        "La configuración no contiene 'deterministic'."
    )

# ------------------------------------------------------------------------------
# 2.2 Configuración de la reproducibilidad
# ------------------------------------------------------------------------------

random.seed(PROJECT_SEED)
np.random.seed(PROJECT_SEED)
torch.manual_seed(PROJECT_SEED)

CUDA_AVAILABLE = torch.cuda.is_available()

if (
    BENCHMARK_REPRODUCIBILITY["deterministic"]
    and CUDA_AVAILABLE
):

    torch.cuda.manual_seed(PROJECT_SEED)
    torch.cuda.manual_seed_all(PROJECT_SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------------------------
# 2.3 Selección del dispositivo
# ------------------------------------------------------------------------------

DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

# ------------------------------------------------------------------------------
# 2.4 Certificación del entorno
# ------------------------------------------------------------------------------

print()

print("-" * 80)
print("CERTIFICACIÓN DEL ENTORNO DE ENTRENAMIENTO")
print("-" * 80)

print(f"Semilla del proyecto          : {PROJECT_SEED}")

print(
    "Modo determinístico           : "
    f"{BENCHMARK_REPRODUCIBILITY['deterministic']}"
)

print(f"Dispositivo de procesamiento  : {DEVICE.upper()}")

if CUDA_AVAILABLE:

    print(
        f"GPU detectada                 : "
        f"{torch.cuda.get_device_name(0)}"
    )

print("Estado                        : OK")

print("-" * 80)
print()

print("Bloque 2. Configuración completada correctamente.")

# ==============================================================================
# BLOQUE 3. RECUPERACIÓN DEL MODELO OFICIAL
# Objetivo:
# Recuperar y validar la configuración oficial del Modelo Oficial seleccionado
# durante el Benchmark Científico, verificando que corresponde a la familia
# oficial definida por el proyecto y que dispone de toda la información
# necesaria para ejecutar el entrenamiento definitivo.
#
# Producto:
# - official_model
#
# Pregunta científica:
# ¿El Modelo Oficial fue recuperado correctamente para iniciar el
# entrenamiento definitivo?
# ==============================================================================

print("\n" + "-" * 80)
print("BLOQUE 3. RECUPERACIÓN DEL MODELO OFICIAL")
print("-" * 80)

# ------------------------------------------------------------------------------
# load_official_model
# ------------------------------------------------------------------------------

def load_official_model() -> dict:
    """
    Recupera y valida la configuración oficial del Modelo Oficial.

    Returns
    -------
    dict
        Configuración oficial del Modelo Oficial.
    """

    # --------------------------------------------------------------------------
    # Validación
    # --------------------------------------------------------------------------

    if not OFFICIAL_MODEL_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"No fue posible localizar '{OFFICIAL_MODEL_CONFIG_FILE.name}'."
        )

    # --------------------------------------------------------------------------
    # Recuperación
    # --------------------------------------------------------------------------

    try:

        with open(
            OFFICIAL_MODEL_CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            official_model = json.load(file)

    except Exception as error:

        raise RuntimeError(
            f"Error al recuperar el Modelo Oficial: {error}"
        )

    # --------------------------------------------------------------------------
    # Validación del Modelo Oficial
    # --------------------------------------------------------------------------

    required_keys = [
        "model_code",
        "model_name",
        "family",
        "model_config",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in official_model
    ]

    if missing_keys:
        raise ValueError(
            "La configuración del Modelo Oficial está incompleta: "
            f"{missing_keys}"
        )

    if official_model["family"] != OFFICIAL_MODEL_FAMILY:
        raise ValueError(
            "La familia del Modelo Oficial no coincide con la configuración "
            "del proyecto."
        )

    if official_model["model_name"] != OFFICIAL_MODEL_NAME:
        raise ValueError(
            "El Modelo Oficial no coincide con la configuración del proyecto."
        )

    if official_model["model_config"] is None:
        raise ValueError(
            "La configuración del Modelo Oficial es inválida."
        )

    # --------------------------------------------------------------------------
    # Retorno
    # --------------------------------------------------------------------------

    return official_model

# ------------------------------------------------------------------------------
# Construcción del producto oficial
# ------------------------------------------------------------------------------

OFFICIAL_MODEL = load_official_model()

print(f"Modelo oficial             : {OFFICIAL_MODEL['model_name']}")
print(f"Familia                    : {OFFICIAL_MODEL['family']}")
print(f"Código                     : {OFFICIAL_MODEL['model_code']}")

print("\nModelo Oficial recuperado correctamente.")

# ==============================================================================
# BLOQUE 4. RECUPERACIÓN DE LA COLECCIÓN OFICIAL BENCHMARKDATA
# Objetivo:
# Recuperar y validar la Colección Oficial BenchmarkData generada durante el
# Benchmark Científico, verificando que todos los productos requeridos para el
# entrenamiento definitivo del Modelo Oficial se encuentren disponibles.
#
# Producto:
# - benchmark_data
#
# Pregunta científica:
# ¿La Colección Oficial BenchmarkData fue recuperada correctamente para
# ejecutar el entrenamiento definitivo del Modelo Oficial?
# ==============================================================================

print("\n" + "-" * 80)
print("BLOQUE 4. RECUPERACIÓN DE BENCHMARKDATA")
print("-" * 80)

# ------------------------------------------------------------------------------
# load_benchmark_data
# ------------------------------------------------------------------------------

def load_benchmark_data() -> dict:
    """
    Recupera la Colección Oficial BenchmarkData.

    Returns
    -------
    dict
        Colección Oficial BenchmarkData.
    """

    # --------------------------------------------------------------------------
    # Validación
    # --------------------------------------------------------------------------

    if not BENCHMARK_DATA_FILE.exists():
        raise FileNotFoundError(
            f"No fue posible localizar '{BENCHMARK_DATA_FILE.name}'."
        )

    # --------------------------------------------------------------------------
    # Recuperación
    # --------------------------------------------------------------------------

    try:

        benchmark_data = joblib.load(BENCHMARK_DATA_FILE)

    except Exception as error:

        raise RuntimeError(
            f"Error al recuperar BenchmarkData: {error}"
        )

    # --------------------------------------------------------------------------
    # Validación
    # --------------------------------------------------------------------------

    if not isinstance(benchmark_data, dict):
        raise TypeError(
            "BenchmarkData debe ser un diccionario."
        )

    required_products = [
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
        "scaler",
    ]

    missing_products = [
        product
        for product in required_products
        if product not in benchmark_data
    ]

    if missing_products:
        raise ValueError(
            "BenchmarkData está incompleto: "
            f"{missing_products}"
        )

    for product in required_products:

        if benchmark_data[product] is None:
            raise ValueError(
                f"'{product}' es inválido."
            )

    # --------------------------------------------------------------------------
    # Retorno
    # --------------------------------------------------------------------------

    return benchmark_data

# ------------------------------------------------------------------------------
# Construcción del producto oficial
# ------------------------------------------------------------------------------

BENCHMARK_DATA = load_benchmark_data()

print(f"GraphData                  : {len(BENCHMARK_DATA['graphs'])}")
print(f"Entrenamiento              : {len(BENCHMARK_DATA['train_index'])}")
print(f"Validación                 : {len(BENCHMARK_DATA['validation_index'])}")
print(f"Prueba                     : {len(BENCHMARK_DATA['test_index'])}")

print("\nColección Oficial BenchmarkData recuperada correctamente.")

# ==============================================================================
# BLOQUE 5. CONSTRUCCIÓN DE LAS ENTRADAS OFICIALES DEL ENTRENAMIENTO
# Objetivo:
# Construir la estructura oficial TrainingInputs, integrando el Modelo Oficial y la Colección Oficial 
# BenchmarkData, preservando el contrato científico definido por el Benchmark para garantizar la 
# trazabilidad y reproducibilidad del entrenamiento.
#
# Producto:
# - training_inputs
#
# Pregunta científica:
# ¿La estructura oficial TrainingInputs preserva íntegramente el contrato científico establecido por 
# el Benchmark para el entrenamiento definitivo del Modelo Oficial?
# ==============================================================================

print("\n" + "-" * 80)
print("BLOQUE 5. CONSTRUCCIÓN DE TRAININGINPUTS")
print("-" * 80)


# ------------------------------------------------------------------------------
# build_training_inputs
# ------------------------------------------------------------------------------

def build_training_inputs(
    official_model: dict,
    benchmark_data: dict
) -> dict:
    """
    Construye la estructura oficial TrainingInputs integrando el
    Modelo Oficial y la Colección Oficial BenchmarkData,
    preservando el contrato científico definido por el Benchmark
    para garantizar la trazabilidad y reproducibilidad del
    entrenamiento.

    Parameters
    ----------
    official_model : dict
        Configuración oficial del Modelo Oficial.

    benchmark_data : dict
        Colección Oficial BenchmarkData.

    Returns
    -------
    dict
        Estructura oficial TrainingInputs.
    """

    # --------------------------------------------------------------------------
    # Validación
    # --------------------------------------------------------------------------

    if not isinstance(official_model, dict):
        raise TypeError(
            "'official_model' debe ser un diccionario."
        )

    if not isinstance(benchmark_data, dict):
        raise TypeError(
            "'benchmark_data' debe ser un diccionario."
        )

    # --------------------------------------------------------------------------
    # Validación del Modelo Oficial
    # --------------------------------------------------------------------------

    required_model = [
        "model_code",
        "model_name",
        "family",
        "model_config",
    ]

    missing_model = [
        key
        for key in required_model
        if key not in official_model
    ]

    if missing_model:
        raise ValueError(
            "Modelo Oficial incompleto: "
            f"{missing_model}"
        )

    # --------------------------------------------------------------------------
    # Validación de BenchmarkData
    # --------------------------------------------------------------------------

    required_benchmark = [
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
        "scaler",
    ]

    missing_benchmark = [
        key
        for key in required_benchmark
        if key not in benchmark_data
    ]

    if missing_benchmark:
        raise ValueError(
            "BenchmarkData incompleto: "
            f"{missing_benchmark}"
        )

    # --------------------------------------------------------------------------
    # Construcción del contrato científico
    # --------------------------------------------------------------------------

    training_inputs = {
        "model": official_model,
        "benchmark_data": benchmark_data,
    }

    # --------------------------------------------------------------------------
    # Validación del contrato científico
    # --------------------------------------------------------------------------

    required_products = [
        "model",
        "benchmark_data",
    ]

    for product in required_products:

        if product not in training_inputs:
            raise RuntimeError(
                f"TrainingInputs no contiene '{product}'."
            )

        if training_inputs[product] is None:
            raise RuntimeError(
                f"'{product}' es inválido."
            )

    # --------------------------------------------------------------------------
    # Retorno
    # --------------------------------------------------------------------------

    return training_inputs


# ------------------------------------------------------------------------------
# Construcción del contrato científico
# ------------------------------------------------------------------------------

TRAINING_INPUTS = build_training_inputs(
    official_model=OFFICIAL_MODEL,
    benchmark_data=BENCHMARK_DATA,
)

print(
    f"GraphData : "
    f"{len(TRAINING_INPUTS['benchmark_data']['graphs'])}"
)

print(f"Modelo Oficial             : {TRAINING_INPUTS['model']['model_name']}")
print(f"GraphData                  : {len(TRAINING_INPUTS['benchmark_data']['graphs'])}")
print(f"Contrato científico        : BenchmarkData → TrainingInputs")
print("\nTrainingInputs construido correctamente.")

# ==============================================================================
# BLOQUE 6. PREPARACIÓN DE LAS ENTRADAS DEL MODELO
# Objetivo:
# Construir, validar y certificar las entradas oficiales del entrenamiento a partir de la Colección 
# Oficial BenchmarkData, preservando la integridad del contrato científico entre el Benchmark Científico 
# y el Entrenamiento del Modelo Oficial.
# Producto:
# - training_features
#
# Pregunta científica:
# ¿Las entradas oficiales del entrenamiento preservan íntegramente el contrato científico definido por 
# BenchmarkData y garantizan la reproducibilidad del Modelo Oficial?
# ==============================================================================

print("\n" + "-" * 80)
print("BLOQUE 6. PREPARACIÓN DE LAS ENTRADAS DEL MODELO")
print("-" * 80)


# ------------------------------------------------------------------------------
# build_training_features
# ------------------------------------------------------------------------------

def build_training_features(
    training_inputs: dict
) -> dict:
    """
    Construye, valida y certifica las entradas oficiales del entrenamiento
    a partir de TrainingInputs, preservando el contrato científico
    establecido entre el Benchmark Científico y el Entrenamiento
    del Modelo Oficial.

    Parameters
    ----------
    training_inputs : dict
        Entradas oficiales del entrenamiento.

    Returns
    -------
    dict
        Entradas oficiales del Modelo Oficial.
    """

    # --------------------------------------------------------------------------
    # Validación del contrato científico
    # --------------------------------------------------------------------------

    if not isinstance(training_inputs, dict):
        raise TypeError(
            "'training_inputs' debe ser un diccionario."
        )

    required_keys = [
        "model",
        "benchmark_data",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in training_inputs
    ]

    if missing_keys:
        raise ValueError(
            "TrainingInputs está incompleto: "
            f"{missing_keys}"
        )

    model = training_inputs["model"]
    benchmark_data = training_inputs["benchmark_data"]

    # --------------------------------------------------------------------------
    # Construcción del contrato científico
    # --------------------------------------------------------------------------

    training_features = {
        "model_config": model["model_config"],
        "graphs": benchmark_data["graphs"],
        "x_train": benchmark_data["x_train"],
        "y_train": benchmark_data["y_train"],
        "x_validation": benchmark_data["x_validation"],
        "y_validation": benchmark_data["y_validation"],
        "train_index": benchmark_data["train_index"],
        "validation_index": benchmark_data["validation_index"],
        "scaler": benchmark_data["scaler"],
    }

    # --------------------------------------------------------------------------
    # Validación del contrato científico
    # --------------------------------------------------------------------------

    required_products = [
        "model_config",
        "graphs",
        "x_train",
        "y_train",
        "x_validation",
        "y_validation",
        "train_index",
        "validation_index",
        "scaler",
    ]

    missing_products = [
        key
        for key in required_products
        if key not in training_features
    ]

    if missing_products:
        raise RuntimeError(
            "TrainingFeatures está incompleto: "
            f"{missing_products}"
        )

    # Validación de la configuración oficial del modelo
    if training_features["model_config"] is None:
        raise RuntimeError(
            "La configuración oficial del modelo es inválida."
        )

    # Validación de la colección oficial GraphData
    if training_features["graphs"] is None:
        raise ValueError(
            "La colección oficial GraphData es nula."
        )

    if len(training_features["graphs"]) == 0:
        raise ValueError(
            "La colección oficial GraphData está vacía."
        )

    # --------------------------------------------------------------------------
    # Retorno
    # --------------------------------------------------------------------------

    return training_features

# ------------------------------------------------------------------------------
# Construcción del producto oficial
# ------------------------------------------------------------------------------

TRAINING_FEATURES = build_training_features(
    training_inputs=TRAINING_INPUTS
)

print(f"Variables de entrenamiento : {TRAINING_FEATURES['x_train'].shape}")
print(f"GraphData                  : {len(TRAINING_FEATURES['graphs'])}")

print("Contrato científico        : VALIDADO")

# ==============================================================================
# BLOQUE 7. ENTRENAMIENTO DEL MODELO OFICIAL
# Objetivo:
# Ejecutar el pipeline oficial de entrenamiento utilizando las TrainingFeatures certificadas, 
# construidas a partir de la Colección Oficial BenchmarkData y de la configuración oficial del Modelo.
#
# Producto:
# - training_result
#
# Pregunta científica:
# ¿El Modelo Oficial fue entrenado correctamente utilizando las TrainingFeatures certificadas, 
# preservando el contrato científico establecido por BenchmarkData?
# ==============================================================================

print("\n" + "-" * 80)
print("BLOQUE 7. ENTRENAMIENTO DEL MODELO OFICIAL")
print("-" * 80)


# ------------------------------------------------------------------------------
# run_official_training
# ------------------------------------------------------------------------------

def run_official_training(
    training_features: dict
) -> dict:
    """
    Ejecuta el entrenamiento oficial del Modelo Oficial utilizando
    las TrainingFeatures certificadas, preservando el contrato
    científico establecido entre el Benchmark Científico y el
    Entrenamiento.

    Parameters
    ----------
    training_features : dict
        Entradas oficiales del entrenamiento.

    Returns
    -------
    dict
        Resultado oficial del entrenamiento.
    """

    # --------------------------------------------------------------------------
    # Validación
    # --------------------------------------------------------------------------

    if not isinstance(training_features, dict):
        raise TypeError(
            "'training_features' debe ser un diccionario."
        )

    required_keys = [
        "model_config",
        "graphs",
        "x_train",
        "y_train",
        "x_validation",
        "y_validation",
        "train_index",
        "validation_index",
        "scaler",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in training_features
    ]

    if missing_keys:
        raise ValueError(
            "TrainingFeatures está incompleto: "
            f"{missing_keys}"
        )

    # --------------------------------------------------------------------------
    # Recuperación
    # --------------------------------------------------------------------------

    # Recuperación de productos
    graphs = training_features["graphs"]

    # Copia de la configuración del modelo
    model_config = training_features["model_config"].copy()

    # Completar la información requerida por graph_neural_networks.py
    model_config["model_name"] = OFFICIAL_MODEL["model_name"]
    model_config["model_code"] = OFFICIAL_MODEL["model_code"]
    model_config["family"] = OFFICIAL_MODEL["family"]

    # --------------------------------------------------------------------------
    # Entrenamiento
    # --------------------------------------------------------------------------

    training_result = run_gnn_training(
        model_config=model_config,
        graphs=graphs,
    )

    # --------------------------------------------------------------------------
    # Validación del producto
    # --------------------------------------------------------------------------

    if training_result is None:
        raise RuntimeError(
            "El entrenamiento no produjo resultados."
        )

    if not isinstance(training_result, dict):
        raise TypeError(
            "run_gnn_training() debe retornar un diccionario."
        )

    required_products = [
        "model",
        "loss",
        "loss_history",
        "training_time",
        "model_config",
        "prediction_result",
        "evaluation_result",
    ]

    missing_products = [
        key
        for key in required_products
        if key not in training_result
    ]

    if missing_products:
        raise ValueError(
            "TrainingResult está incompleto: "
            f"{missing_products}"
        )

    if training_result["model"] is None:
        raise RuntimeError(
            "El modelo entrenado es inválido."
        )

    if training_result["loss"] is None:
        raise RuntimeError(
            "La pérdida final del entrenamiento es inválida."
        )

    if training_result["training_time"] <= 0:
        raise RuntimeError(
            "El tiempo oficial del entrenamiento es inválido."
        )

    # --------------------------------------------------------------------------
    # Certificación
    # --------------------------------------------------------------------------

    print()

    print("-" * 80)
    print("CERTIFICACIÓN DEL ENTRENAMIENTO")
    print("-" * 80)

    print(f"Modelo                  : {model_config['model_name']}")
    print(f"Tiempo de entrenamiento : {training_result['training_time']:.4f} s")
    print(f"Loss final              : {training_result['loss']:.6f}")
    print("Contrato científico   : VALIDADO")
    print("-" * 80)

    # --------------------------------------------------------------------------
    # Retorno
    # --------------------------------------------------------------------------

    return training_result


# ------------------------------------------------------------------------------
# Construcción del producto oficial
# ------------------------------------------------------------------------------

TRAINING_RESULT = run_official_training(
    training_features=TRAINING_FEATURES
)

print("\nBloque 7. Entrenamiento completado correctamente.")

# ==============================================================================
# BLOQUE 8. EXPORTACIÓN DEL MODELO ENTRENADO
# Objetivo:
# Exportar el Modelo Oficial entrenado y los metadatos científicos generados
# durante el entrenamiento definitivo, garantizando la reproducibilidad y
# trazabilidad del proyecto.
#
# Producto:
# - export_result
#
# Pregunta científica:
# ¿El Modelo Oficial entrenado fue exportado correctamente para las siguientes
# etapas del proyecto, preservando la reproducibilidad del proyecto?
# ==============================================================================

print("\n" + "-" * 80)
print("BLOQUE 8. EXPORTACIÓN DEL MODELO ENTRENADO")
print("-" * 80)


# ------------------------------------------------------------------------------
# export_official_model
# ------------------------------------------------------------------------------
def export_official_model(
    official_model: dict,
    training_result: dict,
    training_features: dict
) -> dict:
    """
    Exporta el Modelo Oficial entrenado y los metadatos
    científicos generados durante el entrenamiento,
    garantizando la reproducibilidad y trazabilidad del
    proyecto.

    Parameters
    ----------
    official_model : dict
        Configuración oficial del modelo.

    training_result : dict
        Resultado oficial del entrenamiento.

    training_features : dict
        Entradas oficiales del entrenamiento.

    Returns
    -------
    dict
        Resultado oficial de la exportación.
    """

    # --------------------------------------------------------------------------
    # Validación
    # --------------------------------------------------------------------------

    if not isinstance(official_model, dict):
        raise TypeError(
            "'official_model' debe ser un diccionario."
        )

    if not isinstance(training_result, dict):
        raise TypeError(
            "'training_result' debe ser un diccionario."
        )

    if not isinstance(training_features, dict):
        raise TypeError(
            "'training_features' debe ser un diccionario."
        )

    required_model = [
        "model_code",
        "model_name",
        "family",
        "model_config",
    ]

    missing_model = [
        key
        for key in required_model
        if key not in official_model
    ]

    if missing_model:
        raise ValueError(
            f"Modelo Oficial incompleto: {missing_model}"
        )

    required_training = [
        "model",
        "loss",
        "loss_history",
        "training_time",
    ]

    missing_training = [
        key
        for key in required_training
        if key not in training_result
    ]

    if missing_training:
        raise ValueError(
            f"TrainingResult incompleto: {missing_training}"
        )

    if "graphs" not in training_features:
        raise ValueError(
            "TrainingFeatures no contiene 'graphs'."
        )

    # --------------------------------------------------------------------------
    # Recuperación de productos oficiales
    # --------------------------------------------------------------------------

    trained_model = training_result["model"]

    model_config = official_model["model_config"].copy()

    # Completar la información de identificación del modelo
    model_config["model_code"] = official_model["model_code"]
    model_config["model_name"] = official_model["model_name"]
    model_config["family"] = official_model["family"]

    # --------------------------------------------------------------------------
    # Exportación del Modelo Oficial
    # --------------------------------------------------------------------------

    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "model_config": model_config,
        },
        OFFICIAL_MODEL_TORCH_FILE,
    )

    if not OFFICIAL_MODEL_TORCH_FILE.exists():
        raise RuntimeError(
            "No fue posible exportar el modelo entrenado."
        )

    if OFFICIAL_MODEL_TORCH_FILE.stat().st_size == 0:
        raise RuntimeError(
            "El archivo del modelo entrenado está vacío."
        )

    # --------------------------------------------------------------------------
    # Construcción de metadatos
    # --------------------------------------------------------------------------

    metadata = {
        "model_code": official_model["model_code"],
        "model_name": official_model["model_name"],
        "family": official_model["family"],
        "model_config": model_config,
        "graphs": len(training_features["graphs"]),
        "training_time": training_result["training_time"],
        "training_loss": training_result["loss"],
        "loss_history": training_result["loss_history"],

        # <-- Agregar esta línea
        "epochs": model_config["epochs"],

        "training_date": datetime.now().isoformat(),
        "export_format": "torch",
    }

    with open(
        OFFICIAL_MODEL_METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    if not OFFICIAL_MODEL_METADATA_FILE.exists():
        raise RuntimeError(
            "No fue posible exportar los metadatos."
        )

    if OFFICIAL_MODEL_METADATA_FILE.stat().st_size == 0:
        raise RuntimeError(
            "El archivo de metadatos está vacío."
        )

    # --------------------------------------------------------------------------
    # Construcción del producto oficial
    # --------------------------------------------------------------------------

    export_result = {
        "status": "SUCCESS",
        "model_file": str(
            OFFICIAL_MODEL_TORCH_FILE
        ),
        "metadata_file": str(
            OFFICIAL_MODEL_METADATA_FILE
        ),
        "training_time": training_result[
            "training_time"
        ],
        "training_loss": training_result[
            "loss"
        ],
    }

    # --------------------------------------------------------------------------
    # Validación del producto
    # --------------------------------------------------------------------------

    required_products = [
        "status",
        "model_file",
        "metadata_file",
        "training_time",
        "training_loss",
    ]

    missing_products = [
        key
        for key in required_products
        if key not in export_result
    ]

    if missing_products:
        raise RuntimeError(
            "ExportResult está incompleto: "
            f"{missing_products}"
        )

    # --------------------------------------------------------------------------
    # Certificación
    # --------------------------------------------------------------------------

    print()

    print("-" * 80)
    print("CERTIFICACIÓN DE LA EXPORTACIÓN")
    print("-" * 80)

    print(
        f"Modelo exportado         : "
        f"{OFFICIAL_MODEL_TORCH_FILE.name}"
    )

    print(
        f"Metadatos exportados     : "
        f"{OFFICIAL_MODEL_METADATA_FILE.name}"
    )

    print(
        "Reproducibilidad         : GARANTIZADA"
    )

    print("-" * 80)

    # --------------------------------------------------------------------------
    # Retorno
    # --------------------------------------------------------------------------

    return export_result

# ------------------------------------------------------------------------------
# Construcción del producto oficial
# ------------------------------------------------------------------------------

EXPORT_RESULT = export_official_model(
    official_model=OFFICIAL_MODEL,
    training_result=TRAINING_RESULT,
    training_features=TRAINING_FEATURES,
)

print(
    f"Modelo exportado         : "
    f"{EXPORT_RESULT['model_file']}"
)

print(
    f"Metadatos                : "
    f"{EXPORT_RESULT['metadata_file']}"
)

print(
    "\nBloque 8. Exportación certificada correctamente."
)

# ==============================================================================
# BLOQUE 9. VALIDACIÓN FINAL
#
# Objetivo:
# Verificar la integridad del entrenamiento definitivo y de todos los
# productos oficiales generados durante el proceso.
#
# Producto:
# - validation_result
#
# Pregunta científica:
# ¿El entrenamiento definitivo fue ejecutado correctamente y todos los
# productos oficiales fueron generados de forma íntegra, preservando el
# contrato científico del proyecto?
# ==============================================================================

print("\n" + "-" * 80)
print("BLOQUE 9. VALIDACIÓN FINAL")
print("-" * 80)


# ------------------------------------------------------------------------------
# validate_training_products
# ------------------------------------------------------------------------------

def validate_training_products(
    official_model: dict,
    training_result: dict,
    export_result: dict
) -> dict:
    """
    Valida el Modelo Oficial, el resultado del entrenamiento y la
    exportación oficial, certificando la integridad de todos los
    productos científicos generados durante el entrenamiento.

    Parameters
    ----------
    official_model : dict
        Modelo Oficial recuperado del Benchmark.

    training_result : dict
        Resultado oficial del entrenamiento.

    export_result : dict
        Resultado oficial de la exportación.

    Returns
    -------
    dict
        Resultado oficial de la validación.
    """

    # --------------------------------------------------------------------------
    # Validación
    # --------------------------------------------------------------------------

    if not isinstance(official_model, dict):
        raise TypeError(
            "'official_model' debe ser un diccionario."
        )

    if not isinstance(training_result, dict):
        raise TypeError(
            "'training_result' debe ser un diccionario."
        )

    if not isinstance(export_result, dict):
        raise TypeError(
            "'export_result' debe ser un diccionario."
        )

    required_model = [
        "model_code",
        "model_name",
        "family",
        "model_config",
    ]

    missing_model = [
        key
        for key in required_model
        if key not in official_model
    ]

    if missing_model:
        raise ValueError(
            "Modelo Oficial incompleto: "
            f"{missing_model}"
        )

    required_training = [
        "model",
        "loss",
        "loss_history",
        "training_time",
        "model_config",
        "prediction_result",
        "evaluation_result",
    ]

    missing_training = [
        key
        for key in required_training
        if key not in training_result
    ]

    if missing_training:
        raise ValueError(
            "TrainingResult incompleto: "
            f"{missing_training}"
        )

    required_export = [
        "status",
        "model_file",
        "metadata_file",
        "training_time",
        "training_loss",
    ]

    missing_export = [
        key
        for key in required_export
        if key not in export_result
    ]

    if missing_export:
        raise ValueError(
            "ExportResult incompleto: "
            f"{missing_export}"
        )

    # --------------------------------------------------------------------------
    # Validación del contrato científico
    # --------------------------------------------------------------------------

    if training_result["model"] is None:
        raise RuntimeError(
            "El modelo entrenado es inválido."
        )

    if training_result["loss"] is None:
        raise RuntimeError(
            "La pérdida final del entrenamiento es inválida."
        )

    if training_result["training_time"] <= 0:
        raise RuntimeError(
            "El tiempo de entrenamiento es inválido."
        )

    if export_result["status"] != "SUCCESS":
        raise RuntimeError(
            "La exportación del Modelo Oficial no fue exitosa."
        )

    # --------------------------------------------------------------------------
    # Construcción del producto oficial
    # --------------------------------------------------------------------------

    validation_result = {
        "status": "SUCCESS",
        "official_model": official_model["model_name"],
        "family": official_model["family"],
        "training_time": training_result["training_time"],
        "training_loss": training_result["loss"],
        "export_status": export_result["status"],
    }

    # --------------------------------------------------------------------------
    # Validación del producto
    # --------------------------------------------------------------------------

    required_products = [
        "status",
        "official_model",
        "family",
        "training_time",
        "training_loss",
        "export_status",
    ]

    missing_products = [
        key
        for key in required_products
        if key not in validation_result
    ]

    if missing_products:
        raise RuntimeError(
            "ValidationResult está incompleto: "
            f"{missing_products}"
        )

    # --------------------------------------------------------------------------
    # Certificación
    # --------------------------------------------------------------------------

    print()

    print("-" * 80)
    print("CERTIFICACIÓN DE LA VALIDACIÓN FINAL")
    print("-" * 80)

    print(f"Modelo Oficial          : {official_model['model_name']}")
    print(f"Familia                 : {official_model['family']}")
    print(f"Estado de exportación   : {export_result['status']}")
    print("Contrato científico     : VALIDADO")

    print("-" * 80)

    # --------------------------------------------------------------------------
    # Retorno
    # --------------------------------------------------------------------------

    return validation_result


# ------------------------------------------------------------------------------
# Construcción del producto oficial
# ------------------------------------------------------------------------------

VALIDATION_RESULT = validate_training_products(
    official_model=OFFICIAL_MODEL,
    training_result=TRAINING_RESULT,
    export_result=EXPORT_RESULT,
)

print("\nBloque 9. Validación certificada correctamente.")

# ==============================================================================
# BLOQUE 9.2. RECUPERACIÓN Y VALIDACIÓN DE LOS ARCHIVOS EXPORTADOS
# ==============================================================================

print("\n" + "-" * 80)
print("BLOQUE 9.2. RECUPERACIÓN Y VALIDACIÓN DE LOS ARCHIVOS EXPORTADOS")
print("-" * 80)


# ------------------------------------------------------------------------------
# load_exported_files
# ------------------------------------------------------------------------------

def load_exported_files(
    export_result: dict
) -> dict:
    """
    Recupera y valida los archivos oficiales exportados durante el
    entrenamiento, garantizando su disponibilidad para las siguientes
    etapas del proyecto.

    Parameters
    ----------
    export_result : dict
        Resultado oficial de la exportación.

    Returns
    -------
    dict
        Archivos oficiales recuperados.
    """

    # --------------------------------------------------------------------------
    # Validación
    # --------------------------------------------------------------------------

    if not isinstance(export_result, dict):
        raise TypeError(
            "'export_result' debe ser un diccionario."
        )

    required_keys = [
        "status",
        "model_file",
        "metadata_file",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in export_result
    ]

    if missing_keys:
        raise ValueError(
            "ExportResult está incompleto: "
            f"{missing_keys}"
        )

    if export_result["status"] != "SUCCESS":
        raise RuntimeError(
            "La exportación no finalizó correctamente."
        )

    # --------------------------------------------------------------------------
    # Recuperación de archivos
    # --------------------------------------------------------------------------

    model_file = Path(export_result["model_file"])
    metadata_file = Path(export_result["metadata_file"])

    # --------------------------------------------------------------------------
    # Validación de los archivos
    # --------------------------------------------------------------------------

    if not model_file.exists():
        raise FileNotFoundError(
            f"No fue posible localizar '{model_file.name}'."
        )

    if model_file.stat().st_size == 0:
        raise RuntimeError(
            "El archivo del modelo exportado está vacío."
        )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"No fue posible localizar '{metadata_file.name}'."
        )

    if metadata_file.stat().st_size == 0:
        raise RuntimeError(
            "El archivo de metadatos está vacío."
        )

    # --------------------------------------------------------------------------
    # Construcción del producto oficial
    # --------------------------------------------------------------------------

    recovered_files = {
        "model_file": model_file,
        "metadata_file": metadata_file,
        "status": "SUCCESS",
    }

    # --------------------------------------------------------------------------
    # Validación del producto
    # --------------------------------------------------------------------------

    required_products = [
        "model_file",
        "metadata_file",
        "status",
    ]

    missing_products = [
        key
        for key in required_products
        if key not in recovered_files
    ]

    if missing_products:
        raise RuntimeError(
            "RecoveredFiles está incompleto: "
            f"{missing_products}"
        )

    # --------------------------------------------------------------------------
    # Certificación
    # --------------------------------------------------------------------------

    print()

    print("-" * 80)
    print("CERTIFICACIÓN DE LA RECUPERACIÓN")
    print("-" * 80)

    print(f"Modelo exportado         : {model_file.name}")
    print(f"Metadatos exportados     : {metadata_file.name}")
    print("Integridad de archivos   : VALIDADA")

    print("-" * 80)

    # --------------------------------------------------------------------------
    # Retorno
    # --------------------------------------------------------------------------

    return recovered_files


# ------------------------------------------------------------------------------
# Construcción del producto oficial
# ------------------------------------------------------------------------------

RECOVERED_FILES = load_exported_files(
    export_result=EXPORT_RESULT
)

print("\nBloque 9.2. Recuperación certificada correctamente.")

# ==============================================================================
# BLOQUE 9.3. VALIDACIÓN DEL CHECKPOINT DEL MODELO OFICIAL
# ==============================================================================

print("\n" + "-" * 80)
print("BLOQUE 9.3. VALIDACIÓN DEL CHECKPOINT")
print("-" * 80)


# ------------------------------------------------------------------------------
# validate_checkpoint
# ------------------------------------------------------------------------------

def validate_checkpoint(
    recovered_files: dict
) -> dict:
    """
    Recupera y valida el checkpoint oficial del Modelo Oficial.

    Parameters
    ----------
    recovered_files : dict
        Archivos oficiales recuperados.

    Returns
    -------
    dict
        Checkpoint oficial validado.
    """

    # --------------------------------------------------------------------------
    # Validación
    # --------------------------------------------------------------------------

    if not isinstance(recovered_files, dict):
        raise TypeError(
            "'recovered_files' debe ser un diccionario."
        )

    required_keys = [
        "model_file",
        "metadata_file",
        "status",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in recovered_files
    ]

    if missing_keys:
        raise ValueError(
            "RecoveredFiles está incompleto: "
            f"{missing_keys}"
        )

    model_file = recovered_files["model_file"]

    if not isinstance(model_file, Path):
        raise TypeError(
            "'model_file' debe ser un objeto Path."
        )

    if not model_file.exists():
        raise FileNotFoundError(
            f"No fue posible localizar '{model_file.name}'."
        )

    # --------------------------------------------------------------------------
    # Recuperación del checkpoint
    # --------------------------------------------------------------------------

    try:

        checkpoint = torch.load(
            model_file,
            map_location="cpu",
            weights_only=False,
        )

    except Exception as error:

        raise RuntimeError(
            "No fue posible recuperar el checkpoint oficial."
        ) from error

    # --------------------------------------------------------------------------
    # Validación del contrato científico
    # --------------------------------------------------------------------------

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "El checkpoint debe ser un diccionario."
        )

    required_checkpoint = [
        "model_state_dict",
        "model_config",
    ]

    missing_checkpoint = [
        key
        for key in required_checkpoint
        if key not in checkpoint
    ]

    if missing_checkpoint:
        raise ValueError(
            "Checkpoint incompleto: "
            f"{missing_checkpoint}"
        )

    model_state_dict = checkpoint["model_state_dict"]
    model_config = checkpoint["model_config"]

    if not isinstance(model_state_dict, dict):
        raise TypeError(
            "'model_state_dict' debe ser un diccionario."
        )

    if len(model_state_dict) == 0:
        raise ValueError(
            "El modelo entrenado no contiene parámetros."
        )

    if not isinstance(model_config, dict):
        raise TypeError(
            "'model_config' debe ser un diccionario."
        )

    required_config = [
        "model_code",
        "model_name",
        "family",
    ]

    missing_config = [
        key
        for key in required_config
        if key not in model_config
    ]

    if missing_config:
        raise ValueError(
            "La configuración oficial del modelo está incompleta: "
            f"{missing_config}"
        )

    # --------------------------------------------------------------------------
    # Construcción del producto oficial
    # --------------------------------------------------------------------------

    validated_checkpoint = {
        "checkpoint": checkpoint,
        "model_state_dict": model_state_dict,
        "model_config": model_config,
    }

    # --------------------------------------------------------------------------
    # Validación del producto
    # --------------------------------------------------------------------------

    required_products = [
        "checkpoint",
        "model_state_dict",
        "model_config",
    ]

    missing_products = [
        key
        for key in required_products
        if key not in validated_checkpoint
    ]

    if missing_products:
        raise RuntimeError(
            "ValidatedCheckpoint está incompleto: "
            f"{missing_products}"
        )

    # --------------------------------------------------------------------------
    # Certificación
    # --------------------------------------------------------------------------

    print()

    print("-" * 80)
    print("CERTIFICACIÓN DEL CHECKPOINT")
    print("-" * 80)

    print(f"Modelo                  : {model_config['model_name']}")
    print(f"Familia                 : {model_config['family']}")
    print(f"Parámetros almacenados  : {len(model_state_dict)}")
    print("Checkpoint              : VALIDADO")

    print("-" * 80)

    # --------------------------------------------------------------------------
    # Retorno
    # --------------------------------------------------------------------------

    return validated_checkpoint


# ------------------------------------------------------------------------------
# Construcción del producto oficial
# ------------------------------------------------------------------------------

VALIDATED_CHECKPOINT = validate_checkpoint(
    recovered_files=RECOVERED_FILES,
)

print("\nBloque 9.3. Checkpoint certificado correctamente.")

# ==============================================================================
# BLOQUE 9.4. VALIDACIÓN DE LOS METADATOS DEL ENTRENAMIENTO
# ==============================================================================

print("\n" + "-" * 80)
print("BLOQUE 9.4. VALIDACIÓN DE LOS METADATOS")
print("-" * 80)


# ------------------------------------------------------------------------------
# validate_metadata
# ------------------------------------------------------------------------------

def validate_metadata(
    recovered_files: dict,
    official_model: dict,
    training_result: dict
) -> dict:
    """
    Recupera y valida los metadatos oficiales del entrenamiento.

    Parameters
    ----------
    recovered_files : dict
        Archivos oficiales recuperados.

    official_model : dict
        Configuración oficial del Modelo Oficial.

    training_result : dict
        Resultado oficial del entrenamiento.

    Returns
    -------
    dict
        Metadatos oficiales validados.
    """

    # --------------------------------------------------------------------------
    # Validación
    # --------------------------------------------------------------------------

    if not isinstance(recovered_files, dict):
        raise TypeError(
            "'recovered_files' debe ser un diccionario."
        )

    if not isinstance(official_model, dict):
        raise TypeError(
            "'official_model' debe ser un diccionario."
        )

    if not isinstance(training_result, dict):
        raise TypeError(
            "'training_result' debe ser un diccionario."
        )

    required_files = [
        "model_file",
        "metadata_file",
        "status",
    ]

    missing_files = [
        key
        for key in required_files
        if key not in recovered_files
    ]

    if missing_files:
        raise ValueError(
            "RecoveredFiles está incompleto: "
            f"{missing_files}"
        )

    metadata_file = recovered_files["metadata_file"]

    if not isinstance(metadata_file, Path):
        raise TypeError(
            "'metadata_file' debe ser un objeto Path."
        )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"No fue posible localizar '{metadata_file.name}'."
        )

    if metadata_file.stat().st_size == 0:
        raise RuntimeError(
            "El archivo de metadatos está vacío."
        )

    # --------------------------------------------------------------------------
    # Recuperación de metadatos
    # --------------------------------------------------------------------------

    try:

        with open(
            metadata_file,
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

    except Exception as error:

        raise RuntimeError(
            "No fue posible recuperar los metadatos oficiales."
        ) from error

    # --------------------------------------------------------------------------
    # Validación del contrato científico
    # --------------------------------------------------------------------------

    required_metadata = [
        "model_code",
        "model_name",
        "family",
        "model_config",
        "graphs",
        "training_time",
        "training_loss",
        "loss_history",
        "epochs",
        "training_date",
        "export_format",
    ]

    missing_metadata = [
        key
        for key in required_metadata
        if key not in metadata
    ]

    if missing_metadata:
        raise ValueError(
            "Los metadatos están incompletos: "
            f"{missing_metadata}"
        )

    if metadata["model_code"] != official_model["model_code"]:
        raise ValueError("model_code inconsistente.")

    if metadata["model_name"] != official_model["model_name"]:
        raise ValueError("model_name inconsistente.")

    if metadata["family"] != official_model["family"]:
        raise ValueError("family inconsistente.")

    if metadata["training_time"] != training_result["training_time"]:
        raise ValueError("training_time inconsistente.")

    if metadata["training_loss"] != training_result["loss"]:
        raise ValueError("training_loss inconsistente.")

    if metadata["loss_history"] != training_result["loss_history"]:
        raise ValueError("loss_history inconsistente.")

    if not isinstance(metadata["epochs"], int):
        raise TypeError(
            "'epochs' debe ser un entero."
        )

    if metadata["epochs"] <= 0:
        raise ValueError(
            "'epochs' debe ser mayor que cero."
        )

    if not isinstance(metadata["loss_history"], list):
        raise TypeError(
            "'loss_history' debe ser una lista."
        )

    if len(metadata["loss_history"]) != metadata["epochs"]:
        raise ValueError(
            "La longitud de 'loss_history' no coincide "
            "con el número de épocas."
        )

    if metadata["export_format"] != "torch":
        raise ValueError(
            "Formato de exportación inválido."
        )

    # --------------------------------------------------------------------------
    # Construcción del producto oficial
    # --------------------------------------------------------------------------

    validated_metadata = {
        "metadata": metadata,
        "metadata_file": metadata_file,
    }

    # --------------------------------------------------------------------------
    # Validación del producto
    # --------------------------------------------------------------------------

    required_products = [
        "metadata",
        "metadata_file",
    ]

    missing_products = [
        key
        for key in required_products
        if key not in validated_metadata
    ]

    if missing_products:
        raise RuntimeError(
            "ValidatedMetadata está incompleto: "
            f"{missing_products}"
        )

    # --------------------------------------------------------------------------
    # Certificación
    # --------------------------------------------------------------------------

    print()

    print("-" * 80)
    print("CERTIFICACIÓN DE LOS METADATOS")
    print("-" * 80)

    print(f"Modelo                  : {metadata['model_name']}")
    print(f"Épocas                  : {metadata['epochs']}")
    print(f"Tiempo entrenamiento    : {metadata['training_time']:.4f} s")
    print(f"Formato                 : {metadata['export_format']}")
    print("Metadatos               : VALIDADOS")

    print("-" * 80)

    # --------------------------------------------------------------------------
    # Retorno
    # --------------------------------------------------------------------------

    return validated_metadata


# ------------------------------------------------------------------------------
# Construcción del producto oficial
# ------------------------------------------------------------------------------

VALIDATED_METADATA = validate_metadata(
    recovered_files=RECOVERED_FILES,
    official_model=OFFICIAL_MODEL,
    training_result=TRAINING_RESULT,
)

print("\nBloque 9.4. Metadatos certificados correctamente.")

# ==============================================================================
# BLOQUE 9.5. CERTIFICACIÓN CIENTÍFICA DEL ENTRENAMIENTO
# ==============================================================================

print("\n" + "-" * 80)
print("BLOQUE 9.5. CERTIFICACIÓN CIENTÍFICA")
print("-" * 80)


# ------------------------------------------------------------------------------
# certify_training
# ------------------------------------------------------------------------------

def certify_training(
    official_model: dict,
    training_result: dict,
    export_result: dict,
    validated_checkpoint: dict,
    validated_metadata: dict
) -> dict:
    """
    Certifica científicamente el entrenamiento del Modelo Oficial.

    Parameters
    ----------
    official_model : dict
        Configuración oficial del Modelo Oficial.

    training_result : dict
        Resultado oficial del entrenamiento.

    export_result : dict
        Resultado oficial de la exportación.

    validated_checkpoint : dict
        Checkpoint oficial validado.

    validated_metadata : dict
        Metadatos oficiales validados.

    Returns
    -------
    dict
        Certificación científica del entrenamiento.
    """

    # --------------------------------------------------------------------------
    # Validación
    # --------------------------------------------------------------------------

    required_objects = {
        "official_model": official_model,
        "training_result": training_result,
        "export_result": export_result,
        "validated_checkpoint": validated_checkpoint,
        "validated_metadata": validated_metadata,
    }

    for name, obj in required_objects.items():

        if not isinstance(obj, dict):

            raise TypeError(
                f"'{name}' debe ser un diccionario."
            )

    required_model = [
        "model_code",
        "model_name",
        "family",
    ]

    missing_model = [
        key
        for key in required_model
        if key not in official_model
    ]

    if missing_model:
        raise ValueError(
            "Modelo Oficial incompleto: "
            f"{missing_model}"
        )

    required_training = [
        "model",
        "loss",
        "loss_history",
        "training_time",
        "model_config",
        "prediction_result",
        "evaluation_result",
    ]

    missing_training = [
        key
        for key in required_training
        if key not in training_result
    ]

    if missing_training:
        raise ValueError(
            "TrainingResult incompleto: "
            f"{missing_training}"
        )

    if export_result.get("status") != "SUCCESS":
        raise RuntimeError(
            "La exportación del Modelo Oficial no fue exitosa."
        )

    if "model_config" not in validated_checkpoint:
        raise ValueError(
            "ValidatedCheckpoint está incompleto."
        )

    if "metadata" not in validated_metadata:
        raise ValueError(
            "ValidatedMetadata está incompleto."
        )

    # --------------------------------------------------------------------------
    # Validación del contrato científico
    # --------------------------------------------------------------------------

    checkpoint_config = validated_checkpoint["model_config"]
    metadata = validated_metadata["metadata"]

    if checkpoint_config["model_code"] != official_model["model_code"]:
        raise ValueError(
            "Inconsistencia en 'model_code'."
        )

    if checkpoint_config["model_name"] != official_model["model_name"]:
        raise ValueError(
            "Inconsistencia en 'model_name'."
        )

    if metadata["model_code"] != official_model["model_code"]:
        raise ValueError(
            "Los metadatos no corresponden al Modelo Oficial."
        )

    if metadata["training_loss"] != training_result["loss"]:
        raise ValueError(
            "La pérdida registrada es inconsistente."
        )

    if metadata["training_time"] != training_result["training_time"]:
        raise ValueError(
            "El tiempo de entrenamiento es inconsistente."
        )

    # --------------------------------------------------------------------------
    # Construcción del producto oficial
    # --------------------------------------------------------------------------

    certification_result = {
        "status": "CERTIFIED",
        "model_code": official_model["model_code"],
        "model_name": official_model["model_name"],
        "family": official_model["family"],
        "training_time": training_result["training_time"],
        "training_loss": training_result["loss"],
        "epochs": metadata["epochs"],
        "export_format": metadata["export_format"],
        "certification_date": datetime.now().isoformat(),
    }

    # --------------------------------------------------------------------------
    # Validación del producto
    # --------------------------------------------------------------------------

    required_products = [
        "status",
        "model_code",
        "model_name",
        "family",
        "training_time",
        "training_loss",
        "epochs",
        "export_format",
        "certification_date",
    ]

    missing_products = [
        key
        for key in required_products
        if key not in certification_result
    ]

    if missing_products:
        raise RuntimeError(
            "CertificationResult está incompleto: "
            f"{missing_products}"
        )

    # --------------------------------------------------------------------------
    # Certificación
    # --------------------------------------------------------------------------

    print()

    print("-" * 80)
    print("CERTIFICACIÓN CIENTÍFICA DEL ENTRENAMIENTO")
    print("-" * 80)

    print("Estado                  : CERTIFIED")
    print(f"Modelo                  : {certification_result['model_name']}")
    print(f"Familia                 : {certification_result['family']}")
    print(f"Tiempo entrenamiento    : {certification_result['training_time']:.4f} s")
    print(f"Loss final              : {certification_result['training_loss']:.6f}")
    print(f"Épocas                  : {certification_result['epochs']}")
    print(f"Formato                 : {certification_result['export_format']}")
    print("Contrato científico     : VALIDADO")

    print("-" * 80)

    # --------------------------------------------------------------------------
    # Retorno
    # --------------------------------------------------------------------------

    return certification_result


# ------------------------------------------------------------------------------
# Construcción del producto oficial
# ------------------------------------------------------------------------------

CERTIFICATION_RESULT = certify_training(
    official_model=OFFICIAL_MODEL,
    training_result=TRAINING_RESULT,
    export_result=EXPORT_RESULT,
    validated_checkpoint=VALIDATED_CHECKPOINT,
    validated_metadata=VALIDATED_METADATA,
)

print("\nBloque 9.5. Certificación científica completada correctamente.")

# ==============================================================================
# BLOQUE 10. RESUMEN CIENTÍFICO DEL ENTRENAMIENTO
# Objetivo:
# Consolidar y presentar un resumen científico del entrenamiento definitivo del
# Modelo Oficial, integrando la información más relevante del proceso para
# documentar los resultados obtenidos y facilitar la trazabilidad del proyecto.
#
# Producto:
# - training_summary
# Fuente oficial del entrenamiento : BenchmarkData
#
# Pregunta científica:
# ¿El entrenamiento del Modelo Oficial fue resumido correctamente para
# documentar los resultados oficiales del proyecto?
# ==============================================================================

print("\n" + "-" * 80)
print("BLOQUE 10. RESUMEN CIENTÍFICO DEL ENTRENAMIENTO")
print("-" * 80)


# ------------------------------------------------------------------------------
# build_training_summary
# ------------------------------------------------------------------------------

def build_training_summary(
    official_model: dict,
    training_result: dict,
    export_result: dict,
    certification_result: dict
) -> dict:
    """
    Construye el resumen científico del entrenamiento.

    Parameters
    ----------
    official_model : dict
        Configuración oficial del Modelo Oficial.

    training_result : dict
        Resultado oficial del entrenamiento.

    export_result : dict
        Resultado oficial de la exportación.

    certification_result : dict
        Resultado oficial de la certificación científica.

    Returns
    -------
    dict
        Resumen científico del entrenamiento.
    """

    # --------------------------------------------------------------------------
    # Validación
    # --------------------------------------------------------------------------

    required_objects = {
        "official_model": official_model,
        "training_result": training_result,
        "export_result": export_result,
        "certification_result": certification_result,
    }

    for name, obj in required_objects.items():

        if not isinstance(obj, dict):

            raise TypeError(
                f"'{name}' debe ser un diccionario."
            )

    # --------------------------------------------------------------------------
    # Construcción
    # --------------------------------------------------------------------------

    training_summary = {

        "model_code": official_model["model_code"],

        "model_name": official_model["model_name"],

        "family": official_model["family"],

        "training_time": training_result["training_time"],

        "training_loss": training_result["loss"],

        "model_file": export_result["model_file"],

        "metadata_file": export_result["metadata_file"],

        "status": certification_result["status"]

    }

    # --------------------------------------------------------------------------
    # Resumen científico
    # --------------------------------------------------------------------------

    print(f"Modelo Oficial           : {training_summary['model_name']}")
    print(f"Familia                  : {training_summary['family']}")
    print(f"Tiempo entrenamiento     : {training_summary['training_time']:.4f} s")
    print(f"Loss final               : {training_summary['training_loss']:.6f}")
    print(f"Estado científico        : {training_summary['status']}")

    print("\nResumen científico construido correctamente.")

    # --------------------------------------------------------------------------
    # Retorno
    # --------------------------------------------------------------------------

    return training_summary


# ------------------------------------------------------------------------------
# Construcción del producto oficial
# ------------------------------------------------------------------------------

TRAINING_SUMMARY = build_training_summary(
    official_model=OFFICIAL_MODEL,
    training_result=TRAINING_RESULT,
    export_result=EXPORT_RESULT,
    certification_result=CERTIFICATION_RESULT,
)

print("\nBloque 10. Resumen científico completado correctamente.")

# ==============================================================================
# BLOQUE 11. CONFIRMACIÓN FINAL DEL ENTRENAMIENTO
# Objetivo:
# Confirmar la finalización satisfactoria del entrenamiento del Modelo Oficial,
# registrando el estado final del proceso y certificando que los productos
# oficiales fueron generados correctamente para continuar con la etapa de
# evaluación del proyecto.
#
# Producto:
# - execution_status
#
# Pregunta científica:
# ¿El entrenamiento del Modelo Oficial finalizó correctamente y los productos
# oficiales se encuentran disponibles para la etapa de evaluación?
# ==============================================================================

print("\n" + "-" * 80)
print("BLOQUE 11. CONFIRMACIÓN FINAL DEL ENTRENAMIENTO")
print("-" * 80)


# ------------------------------------------------------------------------------
# confirm_training_execution
# ------------------------------------------------------------------------------

def confirm_training_execution(
    certification_result: dict
) -> dict:
    """
    Confirma la finalización oficial del entrenamiento del
    Modelo Oficial, certificando que el proceso concluyó
    correctamente y que los productos oficiales se encuentran
    disponibles para la etapa de evaluación.

    Parameters
    ----------
    certification_result : dict
        Resultado oficial de la certificación científica.

    Returns
    -------
    dict
        Estado final de ejecución.
    """

    # --------------------------------------------------------------------------
    # Validación
    # --------------------------------------------------------------------------

    if not isinstance(certification_result, dict):
        raise TypeError(
            "'certification_result' debe ser un diccionario."
        )

    required_keys = [
        "status",
        "model_code",
        "model_name",
        "family",
        "training_time",
        "training_loss",
        "epochs",
        "export_format",
        "certification_date",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in certification_result
    ]

    if missing_keys:
        raise ValueError(
            "CertificationResult está incompleto: "
            f"{missing_keys}"
        )

    if certification_result["status"] != "CERTIFIED":
        raise RuntimeError(
            "El entrenamiento no fue certificado."
        )

    # --------------------------------------------------------------------------
    # Construcción del producto oficial
    # --------------------------------------------------------------------------

    execution_status = {

        "status": "SUCCESS",

        "script": "03_train_model.py",

        "official_model": certification_result["model_name"],

        "family": certification_result["family"],

        "next_stage": "04_evaluation.py",

        "execution_date": datetime.now().isoformat(),

    }

    # --------------------------------------------------------------------------
    # Validación del producto
    # --------------------------------------------------------------------------

    required_products = [
        "status",
        "script",
        "official_model",
        "family",
        "next_stage",
        "execution_date",
    ]

    missing_products = [
        key
        for key in required_products
        if key not in execution_status
    ]

    if missing_products:
        raise RuntimeError(
            "ExecutionStatus está incompleto: "
            f"{missing_products}"
        )

    # --------------------------------------------------------------------------
    # Confirmación
    # --------------------------------------------------------------------------

    print()

    print("-" * 80)
    print("CONFIRMACIÓN FINAL DEL ENTRENAMIENTO")
    print("-" * 80)

    print("Estado final             : SUCCESS")
    print(f"Script                   : {execution_status['script']}")
    print(f"Modelo Oficial           : {execution_status['official_model']}")
    print(f"Familia                  : {execution_status['family']}")
    print(f"Siguiente etapa          : {execution_status['next_stage']}")
    print("Entrenamiento            : CERTIFICADO")

    print("-" * 80)

    # --------------------------------------------------------------------------
    # Retorno
    # --------------------------------------------------------------------------

    return execution_status


# ------------------------------------------------------------------------------
# Construcción del producto oficial
# ------------------------------------------------------------------------------

EXECUTION_STATUS = confirm_training_execution(
    certification_result=CERTIFICATION_RESULT,
)

print("\n" + "-" * 80)
print("SCRIPT 03_TRAIN_MODEL.PY FINALIZADO CORRECTAMENTE")
print("-" * 80)