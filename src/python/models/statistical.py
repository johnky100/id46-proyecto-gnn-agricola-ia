# models-statistical.py
# BLOQUE 1. Importaciones --------------------------------------------------
# Funciones del sistema
import time

# Librerías científicas
import numpy as np

# Scikit-Learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)

# Utilidades del proyecto
from src.python.utils.results import (
    build_benchmark_result
)

# Configuración oficial
from src.python.config.config_project import (
    BENCHMARK_MODEL_CODES
)

# BLOQUE 2. Configuración del Modelo ----------------------------------------
## Objetivo: Definir la configuración oficial del modelo de Regresión Lineal
## utilizada durante el Benchmark Científico.
##
## Producto:
## - STATISTICAL_CONFIG
##
## Responde:
## ¿El modelo dispone de una configuración oficial, reproducible y
## consistente con el Benchmark Científico?

# 2.1. Configuración oficial del modelo -------------------------------------

STATISTICAL_CONFIG = {

    # Identificación --------------------------------------------------------
    "model_code": BENCHMARK_MODEL_CODES["ST01"],
    "model_name": "linear_regression",
    "family": "statistical",

    # Implementación --------------------------------------------------------
    "library": "scikit-learn",
    "estimator": LinearRegression,

    # Hiperparámetros -------------------------------------------------------
    "fit_intercept": True,
    "copy_X": True,
    "positive": False

}  # Configuración oficial del modelo de Regresión Lineal

# BLOQUE 3. Entrenamiento del Modelo ---------------------------------------
## Objetivo: Construir y entrenar el modelo oficial de Regresión Lineal
## utilizando el conjunto de entrenamiento definido por el Benchmark
## Científico.
##
## Entradas:
## - x_train
## - y_train
##
## Producto:
## - training_result
##
## Responde:
## ¿El modelo fue construido y entrenado correctamente sobre el conjunto
## de entrenamiento?

def train_linear_regression(
    x_train: np.ndarray,
    y_train: np.ndarray
) -> dict:
    """
    Construye y entrena el modelo oficial de Regresión Lineal.

    Parameters
    ----------
    x_train : np.ndarray
        Variables predictoras del conjunto de entrenamiento.

    y_train : np.ndarray
        Variable objetivo del conjunto de entrenamiento.

    Returns
    -------
    dict
        Modelo entrenado y tiempo oficial de entrenamiento.
    """

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
            "Las variables predictoras y la variable objetivo "
            "deben tener el mismo número de observaciones."
        )

    estimator = STATISTICAL_CONFIG.get("estimator") # Clase del estimador

    if estimator is None:
        raise ValueError(
            "No se encontró un estimador válido."
        )

    model_parameters = {
        key: value
        for key, value in STATISTICAL_CONFIG.items()
        if key not in [
            "model_code",
            "model_name",
            "family",
            "library",
            "estimator"
        ]
    }  # Parámetros oficiales del modelo

    model = estimator(
        **model_parameters
    )  # Construir modelo oficial

    training_start = time.perf_counter()  # Inicio del entrenamiento

    try:

        model.fit(
            x_train,
            y_train
        )  # Entrenar modelo

    except Exception as error:

        raise RuntimeError(
            f"Error durante el entrenamiento: {error}"
        ) from error

    training_time = (
        time.perf_counter() - training_start
    )  # Tiempo oficial de entrenamiento

    return {
        "model": model,
        "training_time": training_time,
        "loss": None
    }  # Resultado oficial del entrenamiento

# BLOQUE 4. Predicción ------------------------------------------------------
## Objetivo: Generar las predicciones del modelo oficial de Regresión Lineal
## sobre el conjunto de prueba y registrar el tiempo oficial de inferencia.
##
## Entradas:
## - model
## - x_test
##
## Producto:
## - prediction_result
##
## Responde:
## ¿El modelo entrenado genera correctamente las predicciones sobre el
## conjunto de prueba?

def predict_linear_regression(
    model: LinearRegression,
    x_test: np.ndarray
) -> dict:
    """
    Genera las predicciones utilizando el modelo oficial de
    Regresión Lineal.

    Parameters
    ----------
    model : LinearRegression
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
## Objetivo: Calcular las métricas oficiales de desempeño predictivo del
## modelo de Regresión Lineal utilizando el conjunto de prueba.
##
## Entradas:
## - y_true
## - y_pred
##
## Producto:
## - evaluation_result
##
## Responde:
## ¿Cuál es el desempeño predictivo del modelo sobre datos no utilizados
## durante el entrenamiento?

def evaluate_linear_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Calcula las métricas oficiales del Benchmark para el modelo de
    Regresión Lineal.

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
## Objetivo: Construir la estructura oficial de resultados del modelo de
## Regresión Lineal compatible con el Benchmark Científico.
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

def build_linear_regression_results(
    model_config: dict,
    prediction_result: dict,
    evaluation_result: dict,
    training_result: dict | None = None
) -> dict:
    """
    Construye el resultado oficial del Benchmark para el modelo de
    Regresión Lineal.

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
## Objetivo: Ejecutar de forma secuencial el flujo completo del modelo de
## Regresión Lineal, incluyendo entrenamiento, predicción, evaluación y
## construcción del resultado oficial del Benchmark.
##
## Entradas:
## - x_train
## - y_train
## - x_test
## - y_test
##
## Producto:
## - benchmark_result
##
## Responde:
## ¿El modelo de Regresión Lineal fue ejecutado correctamente bajo el
## protocolo oficial del Benchmark Científico?

def run_linear_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Ejecuta el flujo completo del modelo oficial de Regresión Lineal.

    Parameters
    ----------
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
    model_config = STATISTICAL_CONFIG  # Configuración oficial del modelo

    # Entrenamiento ---------------------------------------------------------
    training_result = train_linear_regression(
        x_train=x_train,
        y_train=y_train
    )  # Resultado oficial del entrenamiento

    # Predicción ------------------------------------------------------------
    prediction_result = predict_linear_regression(
        model=training_result["model"],
        x_test=x_test
    )  # Resultado oficial de la inferencia

    # Evaluación ------------------------------------------------------------
    evaluation_result = evaluate_linear_regression(
        y_true=y_test,
        y_pred=prediction_result["y_pred"]
    )  # Resultado oficial de la evaluación

    # Resultado oficial -----------------------------------------------------
    benchmark_result = build_linear_regression_results(
        model_config=model_config,
        prediction_result=prediction_result,
        evaluation_result=evaluation_result,
        training_result=training_result
    )

    return benchmark_result