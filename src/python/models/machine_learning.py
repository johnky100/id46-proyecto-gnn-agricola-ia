# machine_learning.py

# BLOQUE 1. Importaciones --------------------------------------------------

# Funciones del sistema
import time  # Medición del tiempo de entrenamiento

# Tipado
from typing import Any

# Librerías científicas
import numpy as np  # Operaciones numéricas

# Scikit-Learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)

# XGBoost
from xgboost import XGBRegressor

# LightGBM
from lightgbm import LGBMRegressor

# CatBoost
from catboost import CatBoostRegressor

# Utilidades del proyecto
from src.python.utils.results import (
    build_benchmark_result
)

from src.python.config.config_project import (
    BENCHMARK_MODEL_CODES,
    PROJECT_SEED
)

# BLOQUE 2. Configuración --------------------------------------------------
## Objetivo: Definir la configuración oficial de los modelos de Machine
## Learning utilizados durante el Benchmark Científico.
##
## Producto:
## - MACHINE_LEARNING_CONFIG
##
## Responde:
## ¿Los modelos de Machine Learning disponen de una configuración oficial,
## reproducible y consistente con el Benchmark Científico?

# Configuración oficial de los modelos -------------------------------------
MACHINE_LEARNING_CONFIG = {

    "random_forest": {

        # Identificación
        "model_code": "ML01",
        "model_name": "random_forest",
        "family": "machine_learning",

        # Implementación
        "estimator": RandomForestRegressor,

        # Hiperparámetros
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": PROJECT_SEED,
        "n_jobs": -1
    },

    "xgboost": {

        # Identificación
        "model_code": "ML02", # Código oficial del modelo
        "model_name": "xgboost",
        "family": "machine_learning",

        # Implementación
        "estimator": XGBRegressor,

        # Hiperparámetros
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": PROJECT_SEED,
        "verbosity": 0
    },

    "lightgbm": {

        # Identificación
        "model_code": "ML03", # Código oficial del modelo
        "model_name": "lightgbm",
        "family": "machine_learning",

        # Implementación
        "estimator": LGBMRegressor,

        # Hiperparámetros
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": -1,
        "random_state": PROJECT_SEED,
        "verbose": -1
    },

    "catboost": {

        # Identificación
        "model_code": "ML04", # Código oficial del modelo
        "model_name": "catboost",
        "family": "machine_learning",

        # Implementación
        "estimator": CatBoostRegressor,

        # Hiperparámetros
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": PROJECT_SEED,
        "verbose": False
    }

}  # Configuración oficial de Machine Learning

# BLOQUE 3. Entrenamiento del Modelo ---------------------------------------
## Objetivo: Construir y entrenar un modelo de Machine Learning utilizando la
## configuración oficial definida para el Benchmark Científico.
##
## Entradas:
## - model_config
## - x_train
## - y_train
##
## Producto:
## - training_result
##
## Responde:
## ¿El modelo de Machine Learning fue construido y entrenado correctamente
## sobre el conjunto de entrenamiento?

def train_machine_learning_model(
    model_config: dict,
    x_train: np.ndarray,
    y_train: np.ndarray
) -> dict:
    """
    Construye y entrena un modelo oficial de Machine Learning.

    Parameters
    ----------
    model_config : dict
        Configuración oficial del modelo.

    x_train : np.ndarray
        Variables predictoras del conjunto de entrenamiento.

    y_train : np.ndarray
        Variable objetivo del conjunto de entrenamiento.

    Returns
    -------
    dict
        Modelo entrenado y tiempo oficial de entrenamiento.
    """

    if model_config is None:
        raise ValueError(
            "La configuración del modelo no puede ser nula."
        )

    if x_train is None or y_train is None:
        raise ValueError(
            "Los datos de entrenamiento no pueden ser nulos."
        )

    if len(x_train) == 0 or len(y_train) == 0:
        raise ValueError(
            "Los datos de entrenamiento están vacíos."
        )

    if len(x_train) != len(y_train):
        raise ValueError(
            "Las variables predictoras y la variable objetivo deben tener el mismo número de observaciones."
        )

    estimator = model_config["estimator"]  # Clase del estimador

    model_parameters = {
        key: value
        for key, value in model_config.items()
        if key not in [
            "model_code",
            "model_name",
            "family",
            "estimator"
        ]
    }  # Parámetros del modelo

    model = estimator(
        **model_parameters
    )  # Modelo oficial

    training_start = time.perf_counter()  # Inicio del entrenamiento

    try:

        model.fit(
            x_train,
            y_train
        )  # Entrenamiento del modelo

    except Exception as error:

        raise RuntimeError(
            f"Error durante el entrenamiento del modelo "
            f"{model_config['model_name']}: {error}"
        ) from error

    training_time = (
        time.perf_counter() - training_start
    )  # Tiempo oficial de entrenamiento

    return {
        "model": model,
        "training_time": training_time,
        "loss": None
    }  # Resultado oficial del entrenamiento

# BLOQUE 4. Predicción -----------------------------------------------------
## Objetivo: Generar las predicciones del modelo de Machine Learning
## entrenado sobre el conjunto de prueba y registrar el tiempo oficial
## de inferencia.
##
## Entradas:
## - model
## - x_test
##
## Producto:
## - prediction_result
##
## Responde:
## ¿El modelo de Machine Learning genera correctamente las predicciones
## sobre el conjunto de prueba?

def predict_machine_learning_model(
    model: Any,
    x_test: np.ndarray
) -> dict:
    """
    Genera las predicciones utilizando un modelo oficial de
    Machine Learning.

    Parameters
    ----------
    model : Any
        Modelo previamente entrenado.

    x_test : np.ndarray
        Variables predictoras del conjunto de prueba.

    Returns
    -------
    dict
        Predicciones y tiempo oficial de inferencia.
    """

    if model is None:
        raise ValueError(
            "El modelo entrenado no puede ser nulo."
        )

    if x_test is None:
        raise ValueError(
            "El conjunto de prueba no puede ser nulo."
        )

    if len(x_test) == 0:
        raise ValueError(
            "El conjunto de prueba está vacío."
        )

    inference_start = time.perf_counter()  # Inicio de la inferencia

    try:

        y_pred = model.predict(
            x_test
        )  # Predicciones del modelo

    except Exception as error:

        raise RuntimeError(
            f"Error durante la inferencia: {error}"
        ) from error

    inference_time = (
        time.perf_counter() - inference_start
    )  # Tiempo oficial de inferencia

    return {
        "y_pred": y_pred,
        "y_true": None,
        "inference_time": inference_time
    }  # Resultado oficial de la inferencia

# BLOQUE 5. Evaluación del Modelo ------------------------------------------
## Objetivo: Calcular las métricas oficiales de desempeño predictivo para
## un modelo de Machine Learning utilizando el conjunto de prueba.
##
## Entradas:
## - y_true
## - y_pred
##
## Producto:
## - evaluation_result
##
## Responde:
## ¿Cuál es el desempeño predictivo del modelo de Machine Learning sobre
## el conjunto de prueba?

def evaluate_machine_learning_model(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Calcula las métricas oficiales del Benchmark para un modelo de
    Machine Learning.

    Parameters
    ----------
    y_true : np.ndarray
        Valores observados.

    y_pred : np.ndarray
        Valores predichos.

    Returns
    -------
    dict
        Métricas oficiales de evaluación.
    """

    if y_true is None or y_pred is None:
        raise ValueError(
            "Los datos de evaluación no pueden ser nulos."
        )

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError(
            "Los datos de evaluación están vacíos."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Los valores observados y predichos deben tener el mismo número de observaciones."
        )

    try:

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )  # Error cuadrático medio

        mae = mean_absolute_error(
            y_true,
            y_pred
        )  # Error absoluto medio

        mape = mean_absolute_percentage_error(
            y_true,
            y_pred
        )  # Error porcentual absoluto medio

        r2 = r2_score(
            y_true,
            y_pred
        )  # Coeficiente de determinación

    except Exception as error:

        raise RuntimeError(
            f"Error durante la evaluación del modelo: {error}"
        ) from error

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2
    }  # Resultado oficial de la evaluación

# BLOQUE 6. Construcción del Resultado Oficial -----------------------------
## Objetivo: Construir la estructura oficial de resultados de un modelo de
## Machine Learning compatible con el Benchmark Científico.
##
## Entradas:
## - model_config
## - prediction_result
## - evaluation_result
## - training_result
##
## Producto:
## - benchmark_result
##
## Responde:
## ¿Los resultados del modelo fueron consolidados correctamente para el
## Benchmark Científico?

def build_machine_learning_results(
    model_config: dict,
    prediction_result: dict,
    evaluation_result: dict,
    training_result: dict | None = None
) -> dict:
    """
    Construye el resultado oficial del Benchmark para un modelo de
    Machine Learning.

    Parameters
    ----------
    model_config : dict
        Configuración oficial del modelo.

    prediction_result : dict
        Resultado oficial de la inferencia.

    evaluation_result : dict
        Resultado oficial de la evaluación.

    training_result : dict, optional
        Resultado oficial del entrenamiento.

    Returns
    -------
    dict
        Resultado oficial del Benchmark.
    """

    return build_benchmark_result(
        model_config=model_config,
        prediction_result=prediction_result,
        evaluation_result=evaluation_result,
        training_result=training_result
    )  # Resultado oficial del Benchmark

# BLOQUE 7. Ejecución Completa del Modelo ----------------------------------
## Objetivo: Ejecutar de forma secuencial el flujo completo de un modelo de
## Machine Learning, incluyendo entrenamiento, predicción, evaluación y
## construcción del resultado oficial del Benchmark.
##
## Entradas:
## - model_name
## - x_train
## - y_train
## - x_test
## - y_test
##
## Producto:
## - benchmark_result
##
## Responde:
## ¿El modelo de Machine Learning fue ejecutado correctamente bajo el
## protocolo oficial del Benchmark Científico?

def run_machine_learning(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Ejecuta el flujo completo de un modelo oficial de Machine Learning.

    Parameters
    ----------
    model_name : str
        Nombre oficial del modelo de Machine Learning.

    x_train : np.ndarray
        Variables predictoras del conjunto de entrenamiento.

    y_train : np.ndarray
        Variable objetivo del conjunto de entrenamiento.

    x_test : np.ndarray
        Variables predictoras del conjunto de prueba.

    y_test : np.ndarray
        Variable objetivo del conjunto de prueba.

    Returns
    -------
    dict
        Resultado oficial del Benchmark.
    """

    if model_name not in MACHINE_LEARNING_CONFIG:
        raise ValueError(
            f"Modelo de Machine Learning no soportado: {model_name}"
        )

    if any(
        value is None
        for value in (
            x_train,
            y_train,
            x_test,
            y_test
        )
    ):
        raise ValueError(
            "Los conjuntos de entrenamiento y prueba no pueden ser nulos."
        )

    # Configuración oficial -------------------------------------------------
    model_config = MACHINE_LEARNING_CONFIG[
        model_name
    ]  # Configuración oficial del modelo

    # Entrenamiento ---------------------------------------------------------
    training_result = train_machine_learning_model(
        model_config=model_config,
        x_train=x_train,
        y_train=y_train
    )  # Resultado oficial del entrenamiento

    # Predicción ------------------------------------------------------------
    prediction_result = predict_machine_learning_model(
        model=training_result["model"],
        x_test=x_test
    )  # Resultado oficial de la inferencia

    # Evaluación ------------------------------------------------------------
    evaluation_result = evaluate_machine_learning_model(
        y_true=y_test,
        y_pred=prediction_result["y_pred"]
    )  # Resultado oficial de la evaluación

    # Resultado oficial -----------------------------------------------------
    benchmark_result = build_machine_learning_results(
        model_config=model_config,
        prediction_result=prediction_result,
        evaluation_result=evaluation_result,
        training_result=training_result
    )

    return benchmark_result