# deep_learning.py

# BLOQUE 1. Importaciones --------------------------------------------------
# Funciones del sistema
import time # Medición del tiempo de entrenamiento
import warnings # Control de advertencias

# Librerías científicas
import numpy as np # Operaciones numéricas

# Scikit-Learn
from sklearn.neural_network import (
    MLPRegressor
) # Multi-Layer Perceptron

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

# Configuración oficial
from src.python.config.config_project import (
    MODEL_CODES
) # Configuración oficial del Benchmark

warnings.filterwarnings(
    "ignore"
) # Ocultar advertencias no críticas

# BLOQUE 2. Configuración --------------------------------------------------
## Objetivo: Definir la configuración oficial del modelo de Deep Learning utilizado durante el Benchmark Científico.
### Producto: - DEEP_LEARNING_CONFIG
### Responde: ¿El modelo de Deep Learning dispone de una configuración oficial, reproducible y consistente con el Benchmark Científico?

# Configuración oficial del modelo -----------------------------------------
DEEP_LEARNING_CONFIG = {
    "model_code": MODEL_CODES["DL01"],
    "model_name": "mlp",
    "family": "deep_learning",
    "estimator": MLPRegressor,
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": "auto",
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "max_iter": 500,
    "shuffle": True,
    "random_state": 42,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 20,
    "tol": 1e-4
} # Configuración oficial del modelo MLP

# Validación de la configuración -------------------------------------------
required_keys = [
    "model_code",
    "model_name",
    "family",
    "estimator"
] # Parámetros obligatorios

missing_keys = [
    key
    for key in required_keys
    if key not in DEEP_LEARNING_CONFIG
] # Parámetros faltantes

if missing_keys:
    raise ValueError(
        f"Faltan parámetros en DEEP_LEARNING_CONFIG: {missing_keys}"
    )

# BLOQUE 3. Entrenamiento del Modelo ---------------------------------------
## Objetivo: Construir y entrenar el modelo oficial de Deep Learning utilizando
# la configuración oficial definida para el Benchmark Científico.
### Entradas: - model_config - x_train - y_train
### Producto: - trained_model - training_time
### Responde: ¿El modelo de Deep Learning fue construido y entrenado correctamente sobre el conjunto de entrenamiento?

def train_mlp(
    model_config,
    x_train,
    y_train
):
    """
    Construye y entrena el modelo oficial MLP.

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
            f"Error al construir el modelo MLP: {error}"
        )

    training_start = time.time() # Inicio del entrenamiento

    try:

        model.fit(

            x_train,

            y_train

        ) # Entrenamiento

    except Exception as error:

        raise RuntimeError(
            f"Error durante el entrenamiento del modelo MLP: {error}"
        )

    training_time = (
        time.time() - training_start
    ) # Tiempo oficial

    return {

        "model": model,

        "training_time": training_time

    }

# BLOQUE 4. Predicción -----------------------------------------------------
## Objetivo: Generar las predicciones del modelo MLP entrenado sobre el conjunto de
# prueba y registrar el tiempo oficial de inferencia.
## # Entradas: - model - x_test
### Producto: - y_pred - inference_time
### Responde: ¿El modelo de Deep Learning genera correctamente las predicciones sobre el conjunto de prueba?
def predict_mlp(
    model,
    x_test
):
    """
    Genera las predicciones utilizando el modelo oficial MLP.

    Parameters
    ----------
    model : MLPRegressor
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
            f"Error durante la inferencia del modelo MLP: {error}"
        )

    inference_time = (
        time.time() - inference_start
    ) # Tiempo oficial de inferencia

    return {
        "y_pred": y_pred,
        "inference_time": inference_time
    }

# BLOQUE 5. Evaluación del Modelo ------------------------------------------
## Objetivo: Calcular las métricas oficiales de desempeño predictivo para el modelo
# MLP utilizando el conjunto de prueba.
### Entradas: - y_test - y_pred
### Producto: - evaluation_result
### Responde: ¿Cuál es el desempeño predictivo del modelo MLP sobre el conjunto de prueba?
def evaluate_mlp(
    y_test,
    y_pred
):
    """
    Calcula las métricas oficiales del Benchmark para el modelo MLP.

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
                y_test,
                y_pred
            )
        ) # Error cuadrático medio

        mae = mean_absolute_error(
            y_test,
            y_pred
        ) # Error absoluto medio

        mape = mean_absolute_percentage_error(
            y_test,
            y_pred
        ) # Error porcentual absoluto medio

        r2 = r2_score(
            y_test,
            y_pred
        ) # Coeficiente de determinación

        adjusted_r2 = np.nan # Se calculará posteriormente
    except Exception as error:
        raise RuntimeError(
            f"Error durante la evaluación del modelo MLP: {error}"
        )

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "adjusted_r2": adjusted_r2
    }

# BLOQUE 6. Construcción del Resultado Oficial -----------------------------
## Objetivo: Construir la estructura oficial de resultados del modelo MLP compatible con el Benchmark Científico.
### Entradas: - training_result - prediction_result - evaluation_result
### Producto: - benchmark_result
### Responde: ¿Los resultados del modelo MLP fueron consolidados correctamente para el Benchmark Científico?
def build_mlp_results(
    model_config,
    prediction_result,
    evaluation_result,
    training_result = None
):
    """
    Construye el resultado oficial del Benchmark para el modelo MLP.

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

def run_mlp(
    x_train,
    y_train,
    x_test,
    y_test
):
    """
    Ejecuta el flujo completo del modelo oficial MLP.
    """

    # Configuración oficial -------------------------------------------------
    model_config = DEEP_LEARNING_CONFIG # Configuración oficial

    # Entrenamiento ---------------------------------------------------------
    training_result = train_mlp(
        model_config = model_config,
        x_train = x_train,
        y_train = y_train
    ) # Entrenamiento oficial

    # Predicción ------------------------------------------------------------
    prediction_result = predict_mlp(
        model = training_result["model"],
        x_test = x_test
    ) # Predicción oficial

    # Evaluación ------------------------------------------------------------
    evaluation_result = evaluate_mlp(
        y_true = y_test,
        y_pred = prediction_result["y_pred"]
    ) # Evaluación oficial

    # Resultado oficial -----------------------------------------------------
    return build_mlp_results(
        model_config = model_config,
        prediction_result = prediction_result,
        evaluation_result = evaluation_result,
        training_result = training_result
    )