# statistical.py
# BLOQUE 1. Importaciones --------------------------------------------------
# Funciones del sistema
import time # Medición del tiempo de entrenamiento
import warnings # Control de advertencias

# Librerías científicas
import numpy as np # Operaciones numéricas

# Scikit-Learn
from sklearn.linear_model import LinearRegression # Regresión Lineal
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
) # Métricas oficiales

# Utilidades del proyecto
from src.python.utils.results import (
    build_benchmark_result
) # Construcción del resultado oficial del Benchmark

# Configuración oficial del proyecto
from src.python.config.config_project import (
    MODEL_CODES
) # Configuración del Benchmark

warnings.filterwarnings(
    "ignore"
) # Ocultar advertencias no críticas

# BLOQUE 2. Configuración --------------------------------------------------
## Objetivo: Definir la configuración oficial utilizada por los modelos estadísticos durante el Benchmark Científico.
### Producto: - STATISTICAL_CONFIG
### Responde: ¿El modelo estadístico dispone de una configuración oficial, reproducible y consistente con el Benchmark Científico?
# Configuración oficial del modelo -----------------------------------------
STATISTICAL_CONFIG = {
    "model_code": MODEL_CODES["ST01"],
    "model_name": "linear_regression",
    "family": "statistical",
    "library": "scikit-learn",
    "estimator": LinearRegression,
    "fit_intercept": True,
    "copy_X": True,
    "positive": False
} # Configuración oficial de Linear Regression

# Validación de la configuración -------------------------------------------
required_keys = [
    "model_code",
    "model_name",
    "family",
    "fit_intercept",
    "copy_X",
    "positive"
] # Parámetros obligatorios

missing_keys = [
    key
    for key in required_keys
    if key not in STATISTICAL_CONFIG
] # Parámetros faltantes

if missing_keys:
    raise ValueError(
        f"Faltan parámetros en STATISTICAL_CONFIG: {missing_keys}"
    )

# BLOQUE 3. Entrenamiento del Modelo ---------------------------------------
## Objetivo: Construir y entrenar el modelo oficial de Regresión Lineal utilizando
# el conjunto de entrenamiento definido por el Benchmark Científico.
### Entradas: - x_train - y_train
### Producto: - trained_model - training_time
### Responde: ¿El modelo fue construido y entrenado correctamente sobre el conjunto de entrenamiento?
def train_linear_regression(
    x_train,
    y_train
):
    """
    Construye y entrena el modelo oficial de Regresión Lineal.

    Parameters
    ----------
    x_train : ndarray
        Variables predictoras del conjunto de entrenamiento.
    y_train : ndarray
        Variable objetivo del conjunto de entrenamiento.

    Returns
    -------
    dict
        Modelo entrenado y tiempo de entrenamiento.
    """

    # Construcción del modelo ----------------------------------------------
    try:
        model = LinearRegression(
            fit_intercept = STATISTICAL_CONFIG["fit_intercept"],
            copy_X = STATISTICAL_CONFIG["copy_X"],
            positive = STATISTICAL_CONFIG["positive"]
        ) # Modelo oficial

    except Exception as error:
        raise RuntimeError(
            f"Error al construir el modelo: {error}"
        )

    # Entrenamiento ---------------------------------------------------------
    training_start = time.time() # Inicio del entrenamiento
    try:
        model.fit(
            x_train,
            y_train
        ) # Entrenamiento del modelo

    except Exception as error:
        raise RuntimeError(
            f"Error durante el entrenamiento: {error}"
        )

    training_time = (
        time.time() - training_start
    ) # Tiempo de entrenamiento

    return {
        "model": model,
        "training_time": training_time
    }

# BLOQUE 4. Predicción -----------------------------------------------------
## Objetivo: Generar las predicciones del modelo entrenado sobre el conjunto de prueba y registrar el tiempo oficial de inferencia.
### Entradas: - model - x_test
### Producto: - y_pred - inference_time
### Responde: ¿El modelo entrenado genera correctamente las predicciones sobre el conjunto de prueba?
def predict_linear_regression(
    model,
    x_test
):
    """
    Genera las predicciones utilizando el modelo oficial de
    Regresión Lineal.

    Parameters
    ----------
    model : LinearRegression
        Modelo previamente entrenado.

    x_test : ndarray
        Variables predictoras del conjunto de prueba.

    Returns
    -------
    dict
        Predicciones y tiempo de inferencia.
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
## Objetivo: Calcular las métricas oficiales de desempeño predictivo del modelo de
# Regresión Lineal utilizando el conjunto de prueba.
## Entradas: - y_test - y_pred
## Producto: - evaluation_metrics
### Responde: ¿Cuál es el desempeño predictivo del modelo sobre datos no utilizados
# durante el entrenamiento?
def evaluate_linear_regression(
    y_true,
    y_pred
):
    """
    Calcula las métricas oficiales del Benchmark para el modelo de
    Regresión Lineal.

    Parameters
    ----------
    y_true : ndarray
        Valores observados.

    y_pred : ndarray
        Valores predichos.

    Returns
    -------
    dict
        Métricas oficiales de evaluación.
    """

    # Evaluación -----------------------------------------------------------
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

        adjusted_r2 = (
            np.nan
        ) # Se calculará posteriormente

    except Exception as error:

        raise RuntimeError(
            f"Error durante la evaluación del modelo: {error}"
        )

    # Resultado ------------------------------------------------------------
    evaluation_result = {

        "rmse": rmse,

        "mae": mae,

        "mape": mape,

        "r2": r2,

        "adjusted_r2": adjusted_r2

    } # Resultado oficial de la evaluación

    return evaluation_result

# BLOQUE 6. Construcción del Resultado Oficial -----------------------------
## Objetivo: Construir la estructura oficial de resultados del modelo de Regresión
# Lineal compatible con el Benchmark Científico.
### Entradas: - training_result - prediction_result - evaluation_result
### Producto: - benchmark_result
### Responde: ¿Los resultados del modelo fueron consolidados correctamente para el Benchmark Científico?
def build_linear_regression_results(
    model_config,
    prediction_result,
    evaluation_result,
    training_result = None
):
    """
    Construye el resultado oficial del Benchmark para el modelo de
    Regresión Lineal.

    Returns
    -------
    dict
        Resultado oficial del Benchmark.
    """

    benchmark_result = build_benchmark_result(
        model_config = model_config,
        prediction_result = prediction_result,
        evaluation_result = evaluation_result,
        training_result = training_result
    ) # Resultado oficial del Benchmark

    return benchmark_result

# BLOQUE 7. Ejecución Completa del Modelo ----------------------------------
## Objetivo: Ejecutar de forma secuencial el flujo completo del modelo de Regresión
# Lineal, incluyendo entrenamiento, predicción, evaluación y construcción
# del resultado oficial del Benchmark.
### Entradas: - x_train - y_train - x_test - y_test
### Producto: - benchmark_result
### Responde: ¿El modelo de Regresión Lineal fue ejecutado correctamente bajo el
# protocolo oficial del Benchmark Científico?
def run_linear_regression(
    x_train,
    y_train,
    x_test,
    y_test
):
    """
    Ejecuta el flujo completo del modelo oficial de Regresión Lineal.

    Parameters
    ----------
    x_train : ndarray
        Variables predictoras del conjunto de entrenamiento.

    y_train : ndarray
        Variable objetivo del conjunto de entrenamiento.

    x_test : ndarray
        Variables predictoras del conjunto de prueba.

    y_test : ndarray
        Variable objetivo del conjunto de prueba.

    Returns
    -------
    dict
        Resultado oficial del Benchmark.
    """

    # Configuración oficial -------------------------------------------------
    model_config = STATISTICAL_CONFIG # Configuración oficial del modelo

    # Entrenamiento ---------------------------------------------------------
    training_result = train_linear_regression(
        x_train = x_train,
        y_train = y_train
    ) # Entrenamiento del modelo

    # Predicción ------------------------------------------------------------
    prediction_result = predict_linear_regression(
        model = training_result["model"],
        x_test = x_test
    ) # Predicciones oficiales

    # Evaluación ------------------------------------------------------------
    evaluation_result = evaluate_linear_regression(
        y_true = y_test,
        y_pred = prediction_result["y_pred"]
    ) # Evaluación oficial

    # Resultado oficial -----------------------------------------------------
    return build_linear_regression_results(
        model_config = model_config,
        prediction_result = prediction_result,
        evaluation_result = evaluation_result,
        training_result = training_result
    )