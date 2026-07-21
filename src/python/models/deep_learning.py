# models-deep_learning.py

## BLOQUE 1. Importaciones --------------------------------------------------

# Funciones del sistema
import time  # Medición del tiempo de entrenamiento
from typing import Any

# Librerías científicas
import numpy as np  # Operaciones numéricas

# Scikit-Learn
from sklearn.neural_network import (
    MLPRegressor
)  # Multi-Layer Perceptron

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)  # Métricas oficiales

# Utilidades del proyecto
from src.python.utils.results import (
    build_benchmark_result
)  # Construcción del resultado oficial del Benchmark

from src.python.config.config_project import (
    BENCHMARK_MODEL_CODES,
    PROJECT_SEED
)  # Configuración oficial del Benchmark

# BLOQUE 2. Configuración --------------------------------------------------
## Objetivo: Definir la configuración oficial del modelo de Deep Learning
## utilizado durante el Benchmark Científico.
##
## Producto:
## - DEEP_LEARNING_CONFIG
##
## Responde:
## ¿El modelo de Deep Learning dispone de una configuración oficial,
## reproducible y consistente con el Benchmark Científico?

# Configuración oficial del modelo -----------------------------------------
DEEP_LEARNING_CONFIG = {

    # Identificación
    "model_code": BENCHMARK_MODEL_CODES["DL01"],
    "model_name": "mlp",
    "family": "deep_learning",

    # Implementación
    "estimator": MLPRegressor,

    # Hiperparámetros
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": "auto",
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "max_iter": 500,
    "shuffle": True,
    "random_state": PROJECT_SEED,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 20,
    "tol": 1e-4

}  # Configuración oficial del modelo MLP

# BLOQUE 3. Entrenamiento del Modelo ---------------------------------------
## Objetivo: Construir y entrenar el modelo oficial de Deep Learning
## utilizando la configuración oficial definida para el Benchmark Científico.
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
## ¿El modelo de Deep Learning fue construido y entrenado correctamente
## sobre el conjunto de entrenamiento?

def train_mlp(
    model_config: dict,
    x_train: np.ndarray,
    y_train: np.ndarray
) -> dict:
    """
    Construye y entrena el modelo oficial MLP.

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
            f"Error durante el entrenamiento del modelo MLP: {error}"
        ) from error

    training_time = (
        time.perf_counter() - training_start
    )  # Tiempo oficial de entrenamiento

    return {
        "model": model,
        "training_time": training_time,
        "loss": getattr(model, "loss_", None)
    }  # Resultado oficial del entrenamiento

# BLOQUE 4. Predicción -----------------------------------------------------
## Objetivo: Generar las predicciones del modelo MLP entrenado sobre el
## conjunto de prueba y registrar el tiempo oficial de inferencia.
##
## Entradas:
## - model
## - x_test
##
## Producto:
## - prediction_result
##
## Responde:
## ¿El modelo de Deep Learning genera correctamente las predicciones
## sobre el conjunto de prueba?

def predict_mlp(
    model: MLPRegressor,
    x_test: np.ndarray
) -> dict:
    """
    Genera las predicciones utilizando el modelo oficial MLP.

    Parameters
    ----------
    model : MLPRegressor
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
            f"Error durante la inferencia del modelo MLP: {error}"
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
## Objetivo: Calcular las métricas oficiales de desempeño predictivo para el
## modelo MLP utilizando el conjunto de prueba.
##
## Entradas:
## - y_true
## - y_pred
##
## Producto:
## - evaluation_result
##
## Responde:
## ¿Cuál es el desempeño predictivo del modelo MLP sobre el conjunto de prueba?

def evaluate_mlp(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Calcula las métricas oficiales del Benchmark para el modelo MLP.

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
            "Los valores observados y predichos no pueden ser nulos."
        )

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError(
            "Los valores observados y predichos están vacíos."
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
            f"Error durante la evaluación del modelo MLP: {error}"
        ) from error

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2
    }  # Resultado oficial de la evaluación

# BLOQUE 6. Construcción del Resultado Oficial -----------------------------
## Objetivo: Construir la estructura oficial de resultados del modelo MLP
## compatible con el Benchmark Científico.
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
## ¿Los resultados del modelo MLP fueron consolidados correctamente para el
## Benchmark Científico?

def build_mlp_results(
    model_config: dict,
    prediction_result: dict,
    evaluation_result: dict,
    training_result: dict | None = None
) -> dict:
    """
    Construye el resultado oficial del Benchmark para el modelo MLP.

    Parameters
    ----------
    model_config : dict
        Configuración oficial del modelo.

    prediction_result : dict
        Resultado oficial de la predicción.

    evaluation_result : dict
        Resultado oficial de la evaluación.

    training_result : dict | None, optional
        Resultado oficial del entrenamiento.

    Returns
    -------
    dict
        Resultado oficial del Benchmark Científico.
    """

    return build_benchmark_result(
        model_config=model_config,
        prediction_result=prediction_result,
        evaluation_result=evaluation_result,
        training_result=training_result
    )  # Resultado oficial del Benchmark

# BLOQUE 7. Ejecución Completa del Modelo ----------------------------------
## Objetivo: Ejecutar de forma secuencial el flujo completo del modelo oficial
## de Deep Learning, incluyendo entrenamiento, predicción, evaluación y
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
## ¿El modelo de Deep Learning fue ejecutado correctamente bajo el
## protocolo oficial del Benchmark Científico?

def run_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Ejecuta el flujo completo del modelo oficial MLP.

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
    model_config = DEEP_LEARNING_CONFIG  # Configuración oficial del modelo

    # Entrenamiento ---------------------------------------------------------
    training_result = train_mlp(
        model_config=model_config,
        x_train=x_train,
        y_train=y_train
    )  # Resultado oficial del entrenamiento

    # Predicción ------------------------------------------------------------
    prediction_result = predict_mlp(
        model=training_result["model"],
        x_test=x_test
    )  # Resultado oficial de la inferencia

    # Evaluación ------------------------------------------------------------
    evaluation_result = evaluate_mlp(
        y_true=y_test,
        y_pred=prediction_result["y_pred"]
    )  # Resultado oficial de la evaluación

    # Resultado oficial -----------------------------------------------------
    benchmark_result = build_mlp_results(
        model_config=model_config,
        prediction_result=prediction_result,
        evaluation_result=evaluation_result,
        training_result=training_result
    )

    return benchmark_result