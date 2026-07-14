# machine_learning.py
# BLOQUE 1. Importaciones --------------------------------------------------
# Funciones del sistema
import time # Medición del tiempo de entrenamiento
import warnings # Control de advertencias

# Librerías científicas
import numpy as np # Operaciones numéricas

# Scikit-Learn
from sklearn.ensemble import RandomForestRegressor # Random Forest
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
) # Métricas oficiales

# XGBoost
from xgboost import XGBRegressor # Extreme Gradient Boosting

# LightGBM
from lightgbm import LGBMRegressor # Light Gradient Boosting Machine

# CatBoost
from catboost import CatBoostRegressor # CatBoost

# Utilidades del proyecto
from src.python.utils.results import (
    build_benchmark_result
) # Construcción del resultado oficial del Benchmark

# Configuración oficial del proyecto
from src.python.config.config_project import (
    MODEL_CODES
) # Configuración oficial del Benchmark

warnings.filterwarnings(
    "ignore"
) # Ocultar advertencias no críticas

# BLOQUE 2. Configuración --------------------------------------------------
## Objetivo: # Definir la configuración oficial de los modelos de Machine Learning
# utilizados durante el Benchmark Científico.
### Producto: - MACHINE_LEARNING_CONFIG
### Responde: ¿Los modelos de Machine Learning disponen de una configuración oficial,
# reproducible y consistente con el Benchmark Científico?
# Configuración oficial de los modelos -------------------------------------
MACHINE_LEARNING_CONFIG = {
    "random_forest": {
        "model_code": MODEL_CODES["ML01"],
        "model_name": "random_forest",
        "family": "machine_learning",
        "estimator": RandomForestRegressor,
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": -1
    },

    "xgboost": {
        "model_code": MODEL_CODES["ML02"],
        "model_name": "xgboost",
        "family": "machine_learning",
        "estimator": XGBRegressor,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0
    },

    "lightgbm": {
        "model_code": MODEL_CODES["ML03"],
        "model_name": "lightgbm",
        "family": "machine_learning",
        "estimator": LGBMRegressor,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": -1,
        "random_state": 42,
        "verbose": -1
    },

    "catboost": {
        "model_code": MODEL_CODES["ML04"],
        "model_name": "catboost",
        "family": "machine_learning",
        "estimator": CatBoostRegressor,
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": 42,
        "verbose": False
    }
} # Configuración oficial de Machine Learning

# BLOQUE 3. Entrenamiento del Modelo ---------------------------------------
## Objetivo: Construir y entrenar un modelo de Machine Learning utilizando la
# configuración oficial definida para el Benchmark Científico.
### Entradas: - model_config - y_train- y_train
### Producto: - trained_model - training_time
### Responde: ¿El modelo de Machine Learning fue construido y entrenado correctamente
# sobre el conjunto de entrenamiento?
def train_machine_learning_model(
    model_config,
    x_train,
    y_train
):
    
    """
    Construye y entrena un modelo oficial de Machine Learning.

    Parameters
    ----------
    model_config : dict
        Configuración oficial del modelo.

    x_train : ndarray
        Variables predictoras del conjunto de entrenamiento.

    y_train : ndarray
        Variable objetivo del conjunto de entrenamiento.

    Returns
    -------
    dict
        Modelo entrenado y tiempo oficial de entrenamiento.
    """

    estimator = model_config["estimator"] # Clase del estimador
    model_parameters = {
        key: value
        for key, value in model_config.items()
        if key not in [
            "model_code",
            "model_name",
            "family",
            "estimator"
        ]
    } # Parámetros del modelo

    try:
        model = estimator(
            **model_parameters
        ) # Construcción del modelo

    except Exception as error:
        raise RuntimeError(
            f"Error al construir el modelo "
            f"{model_config['model_name']}: {error}"
        )

    training_start = time.time() # Inicio del entrenamiento

    try:
        model.fit(
            x_train,
            y_train
        ) # Entrenamiento del modelo

    except Exception as error:

        raise RuntimeError(
            f"Error durante el entrenamiento del modelo "
            f"{model_config['model_name']}: {error}"
        )

    training_time = (
        time.time() - training_start
    ) # Tiempo de entrenamiento

    training_result = {

        "model": model,

        "training_time": training_time

    } # Resultado oficial del entrenamiento

    return training_result

# BLOQUE 4. Predicción -----------------------------------------------------
## Objetivo: Generar las predicciones del modelo de Machine Learning entrenado sobre
# el conjunto de prueba y registrar el tiempo oficial de inferencia.
### Entradas: - model - x_test
### Producto: - y_pred - inference_time
### Responde: ¿El modelo de Machine Learning genera correctamente las predicciones
# sobre el conjunto de prueba?
def predict_machine_learning_model(
    model,
    x_test
):
    """
    Genera las predicciones utilizando un modelo de Machine Learning.

    Parameters
    ----------
    model : object
        Modelo previamente entrenado.

    x_test : ndarray
        Variables predictoras del conjunto de prueba.

    Returns
    -------
    dict
        Predicciones y tiempo oficial de inferencia.
    """

    inference_start = time.time() # Inicio de la inferencia

    try:
        y_pred = model.predict(
            x_test
        ) # Predicciones del modelo

    except Exception as error:
        raise RuntimeError(
            f"Error durante la inferencia: {error}"
        )

    inference_time = (
        time.time() - inference_start
    ) # Tiempo oficial de inferencia

    return {
        "y_pred": y_pred,
        "inference_time": inference_time
    }

# BLOQUE 5. Evaluación del Modelo ------------------------------------------
## Objetivo: Calcular las métricas oficiales de desempeño predictivo para un modelo
# de Machine Learning utilizando el conjunto de prueba.
### Entradas: - y_test - y_pred
### Producto: - evaluation_result
### Responde: ¿Cuál es el desempeño predictivo del modelo de Machine Learning sobre el conjunto de prueba?
def evaluate_machine_learning_model(
    y_true,
    y_pred
):
    """
    Calcula las métricas oficiales del Benchmark para un modelo de
    Machine Learning.

    Parameters
    ----------
    y_test : ndarray
        Valores observados.

    y_pred : ndarray
        Valores predichos.

    Returns
    -------
    dict
        Métricas oficiales de evaluación.
    """

    try:
        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        ) # Error cuadrático medio

        mae = mean_absolute_error(
            y_true,
            y_pred
        ) # Error absoluto medio

        mape = mean_absolute_percentage_error(
            y_true,
            y_pred
        ) # Error porcentual absoluto medio

        r2 = r2_score(
            y_true,
            y_pred
        ) # Coeficiente de determinación

        adjusted_r2 = np.nan # Se calculará posteriormente

    except Exception as error:
        raise RuntimeError(
            f"Error durante la evaluación del modelo: {error}"
        )

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "adjusted_r2": adjusted_r2
    }

# BLOQUE 6. Construcción del Resultado Oficial -----------------------------
## Objetivo: Construir la estructura oficial de resultados de un modelo de Machine
# Learning compatible con el Benchmark Científico.
### Entradas: - model_config - training_result - prediction_result - evaluation_result
### Producto: - benchmark_result
### Responde: ¿Los resultados del modelo fueron consolidados correctamente para el Benchmark Científico?
def build_machine_learning_results(
    model_config,
    prediction_result,
    evaluation_result,
    training_result = None
):
    """
    Construye el resultado oficial del Benchmark para el modelo de
    Machine Learning.

    Parameters
    ----------
    model_config : dict
        Configuración oficial del modelo.

    prediction_result : dict
        Resultado de la predicción.

    evaluation_result : dict
        Resultado de la evaluación.

    training_result : dict, optional
        Resultado del entrenamiento.

    Returns
    -------
    dict
        Resultado oficial del Benchmark.
    """

    return build_benchmark_result(

        model_config = model_config,

        prediction_result = prediction_result,

        evaluation_result = evaluation_result,

        training_result = training_result

    )

# BLOQUE 7. Ejecución Completa del Modelo ----------------------------------
## Objetivo:
# Ejecutar de forma secuencial el flujo completo de un modelo de Machine
# Learning, incluyendo entrenamiento, predicción, evaluación y
# construcción del resultado oficial del Benchmark.
##
# Entradas:
# - model_name
# - x_train
# - y_train
# - x_test
# - y_test
##
# Producto:
# - benchmark_result
##
# Responde:
# ¿El modelo de Machine Learning fue ejecutado correctamente bajo el
# protocolo oficial del Benchmark Científico?
def run_machine_learning(
    model_name,
    x_train,
    y_train,
    x_test,
    y_test
):
    """
    Ejecuta el flujo completo del modelo oficial de Machine Learning.
    """

    # Configuración oficial -------------------------------------------------
    model_config = MACHINE_LEARNING_CONFIG[
        model_name
    ] # Configuración oficial del modelo

    # Entrenamiento ---------------------------------------------------------
    training_result = train_machine_learning_model(
        model_config = model_config,
        x_train = x_train,
        y_train = y_train
    ) # Entrenamiento oficial

    # Predicción ------------------------------------------------------------
    prediction_result = predict_machine_learning_model(
        model = training_result["model"],
        x_test = x_test
    ) # Predicción oficial

    # Evaluación ------------------------------------------------------------
    evaluation_result = evaluate_machine_learning_model(
        y_true = y_test,
        y_pred = prediction_result["y_pred"]
    ) # Evaluación oficial

    # Resultado oficial -----------------------------------------------------
    return build_machine_learning_results(
        model_config = model_config,
        prediction_result = prediction_result,
        evaluation_result = evaluation_result,
        training_result = training_result
    )