# 04_evaluation.py

# BLOQUE 1. Importaciones --------------------------------------------------
## Objetivo: Importar las librerías, configuraciones y funciones oficiales
# necesarias para ejecutar la evaluación científica del modelo oficial
# del proyecto.

# Funciones del sistema ----------------------------------------------------
import json # Lectura y escritura de archivos JSON
import joblib # Serialización de objetos Python
import warnings # Control de advertencias
import numpy as np # Operaciones numéricas
import shap # Interpretabilidad mediante SHAP
import os # Operaciones sobre archivos y directorios
import matplotlib.pyplot as plt # Visualizaciones

# Gestión de archivos ------------------------------------------------------
from pathlib import Path # Gestión de rutas
from datetime import datetime # Fecha y hora del entrenamiento
import time # Medición de tiempos
from tqdm.auto import tqdm # Barra de progreso

# Librerías científicas ----------------------------------------------------
import pandas as pd # Manipulación del dataset científico
import torch # Modelos Deep Learning y Graph Neural Networks
import pyarrow.parquet as pq # Lectura de archivos Parquet

# Utilidades ---------------------------------------------------------------
from src.python.utils.data_preparation import (
    prepare_tabular_features
) # Preparación oficial del dataset

# Configuración oficial del proyecto ---------------------------------------
from src.python.config.config_project import (
    DATASET_CONFIG
) # Configuración oficial del dataset

# Rutas oficiales ----------------------------------------------------------
from src.python.config.paths import (
    DATASET_FILE,
    GRAPH_DATA_FILE,
    EVALUATION_REPORTS_DIR,
    BEST_MODEL_JOBLIB_FILE,
    BEST_MODEL_TORCH_FILE,
    BEST_MODEL_METADATA_FILE
) # Productos oficiales de la evaluación

from src.python.utils.results import (
    build_benchmark_result
) # Construcción oficial del Benchmark

# Modelos estadísticos -----------------------------------------------------
from src.python.models.statistical import (
    STATISTICAL_CONFIG,
    predict_linear_regression,
    evaluate_linear_regression
) # Modelos estadísticos

# Modelos de Machine Learning ----------------------------------------------
from src.python.models.machine_learning import (
    MACHINE_LEARNING_CONFIG,
    predict_machine_learning_model,
    evaluate_machine_learning_model
) # Modelos de Machine Learning

# Modelos Deep Learning ----------------------------------------------------
from src.python.models.deep_learning import (
    DEEP_LEARNING_CONFIG,
    predict_mlp,
    evaluate_mlp
) # Modelos Deep Learning

# Modelos Graph Neural Networks --------------------------------------------
from src.python.models.graph_neural_networks import (
    GNN_CONFIG,
    predict_gnn,
    evaluate_gnn
) # Modelos Graph Neural Networks

# Configuración del entorno ------------------------------------------------
warnings.filterwarnings(
    "ignore"
) # Ocultar advertencias no críticas


print("-" * 80)

# BLOQUE 2. Configuración --------------------------------------------------
## Objetivo: Definir la configuración oficial utilizada durante la evaluación
# científica, la validación estadística, la interpretabilidad y la generación
# de productos oficiales del proyecto.
#### Producto: - EVALUATION_CONFIG
#### Responde: ¿La evaluación científica dispone de una configuración oficial,
# reproducible y consistente con el protocolo experimental?

# Configuración oficial de la evaluación -----------------------------------
EVALUATION_CONFIG = {
    # Identificación -------------------------------------------------------
    "evaluation_name": "scientific_evaluation",
    "evaluation_version": "1.0",

    # Evaluación predictiva ------------------------------------------------
    "prediction_source": "trained_model",
    "calculate_metrics": True,
    "calculate_residuals": True,

    # Interpretabilidad ----------------------------------------------------
    "calculate_global_explainability": True,
    "calculate_local_explainability": True,
    "calculate_feature_importance": True,
    "shap_batch_size": 512,
    "show_progress": True,

    # Graph Neural Networks ------------------------------------------------
    "export_gnn_embeddings": True,

    # Exportación ----------------------------------------------------------
    "save_evaluation_results": True,
    "save_evaluation_summary": True,
    "save_feature_importance": True,
    "save_shap_values": True,
    "save_validation_results": True,
    "save_evaluation_report": True,

    # Reproducibilidad -----------------------------------------------------
    "random_state": 42

} # Configuración oficial de la evaluación

# Validación ---------------------------------------------------------------
required_keys = [

    "evaluation_name",
    "evaluation_version",

    "prediction_source",
    "calculate_metrics",
    "calculate_residuals",

    "calculate_global_explainability",
    "calculate_local_explainability",
    "calculate_feature_importance",

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

] # Parámetros obligatorios

missing_keys = [
    key
    for key in required_keys
    if key not in EVALUATION_CONFIG
] # Parámetros faltantes

if missing_keys:

    raise ValueError(
        f"Faltan parámetros en EVALUATION_CONFIG: {missing_keys}"
    )

print("-" * 80)

# BLOQUE 3. Carga del Modelo Oficial ---------------------------------------
## Objetivo: Recuperar el modelo oficial entrenado, sus metadatos y reconstruir
# la configuración oficial utilizada durante el entrenamiento definitivo.
#### Entradas: - BEST_MODEL_METADATA_FILE - BEST_MODEL_JOBLIB_FILE - BEST_MODEL_TORCH_FILE
#### Producto: - trained_model - model_metadata - model_config
#### Responde: ¿El modelo oficial fue recuperado correctamente para iniciar la evaluación científica?

# BLOQUE 3. Carga del Modelo -----------------------------------------------
## Objetivo: Recuperar el modelo oficial, sus metadatos y la configuración
# correspondiente para realizar la evaluación científica.
### Entradas: - BEST_MODEL_METADATA_FILE - BEST_MODEL_JOBLIB_FILE - BEST_MODEL_TORCH_FILE
### Producto: - trained_model - model_metadata - model_config
### Responde: ¿El modelo oficial fue recuperado correctamente?

# Carga de los metadatos ---------------------------------------------------
try:
    with open(
        BEST_MODEL_METADATA_FILE,
        "r",
        encoding = "utf-8"
    ) as file:
        model_metadata = json.load(
            file
        ) # Metadatos oficiales

except Exception as error:
    raise RuntimeError(
        f"Error al cargar los metadatos del modelo: {error}"
    )

# Validación de los metadatos ----------------------------------------------
required_metadata_keys = [
    "model_code",
    "model_name",
    "family"
] # Parámetros obligatorios

missing_metadata_keys = [
    key
    for key in required_metadata_keys
    if key not in model_metadata
] # Parámetros faltantes

if missing_metadata_keys:
    raise ValueError(
        f"Faltan parámetros en model_metadata: {missing_metadata_keys}"
    )

# Recuperación de información ----------------------------------------------
model_family = (
    model_metadata[
        "family"
    ]
) # Familia del modelo

model_name = (
    model_metadata[
        "model_name"
    ]
) # Nombre oficial

# Reconstrucción de la configuración oficial -------------------------------
try:
    if model_family == "statistical":
        model_config = (
            STATISTICAL_CONFIG
        ) # Configuración oficial

    elif model_family == "machine_learning":
        model_config = (
            MACHINE_LEARNING_CONFIG[
                model_name
            ]
        ) # Configuración oficial

    elif model_family == "deep_learning":
        model_config = (
            DEEP_LEARNING_CONFIG[
                model_name
            ]
        ) # Configuración oficial

    elif model_family == "graph_neural_networks":
        model_config = (
            GNN_CONFIG[
                model_name
            ]
        ) # Configuración oficial

    else:
        raise ValueError(
            f"Familia de modelo no soportada: {model_family}"
        )

except Exception as error:
    raise RuntimeError(
        f"Error al reconstruir la configuración oficial: {error}"
    )

# Carga del modelo ---------------------------------------------------------
try:
    if model_family in [
        "statistical",
        "machine_learning"
    ]:
        trained_model = joblib.load(
            BEST_MODEL_JOBLIB_FILE
        ) # Modelo Scikit-Learn

    elif model_family in [
        "deep_learning",
        "graph_neural_networks"
    ]:
        trained_model = torch.load(
            BEST_MODEL_TORCH_FILE,
            weights_only = False
        ) # Modelo PyTorch

    else:
        raise ValueError(
            f"Familia de modelo no soportada: {model_family}"
        )

    trained_model_name = type(
        trained_model
    ).__name__ # Nombre de la clase del modelo

    print(f"Modelo cargado: {trained_model_name}")

except Exception as error:
    raise RuntimeError(
        f"Error al cargar el modelo oficial: {error}"
    )

# Validación del modelo ----------------------------------------------------
if trained_model is None:
    raise RuntimeError(
        "No fue posible recuperar el modelo oficial."
    )

# Validación de la configuración -------------------------------------------
if model_config is None:
    raise RuntimeError(
        "No fue posible recuperar la configuración oficial."
    )

if not isinstance(
    model_config,
    dict
):
    raise RuntimeError(
        "La configuración oficial no tiene un formato válido."
    )

if len(
    model_config
) == 0:
    raise RuntimeError(
        "La configuración oficial está vacía."
    )

print("-" * 80)

# BLOQUE 4. Carga de los Datos ---------------------------------------------
## Objetivo: Cargar el dataset científico y el grafo oficial utilizados para
# ejecutar la evaluación científica del modelo oficial del proyecto.
#### Entradas: - DATASET_FILE - GRAPH_DATA_FILE
#### Producto: - evaluation_data
#### Responde: ¿Los datos oficiales del proyecto fueron recuperados correctamente para iniciar la evaluación científica?

# Carga del dataset científico ---------------------------------------------
try:
    dataset = pd.read_parquet(
        DATASET_FILE
    ) # Dataset científico oficial

except Exception as error:
    raise RuntimeError(
        f"Error al cargar el dataset científico: {error}"
    )

# Validación del dataset ---------------------------------------------------
if dataset.empty:
    raise ValueError(
        "El dataset científico está vacío."
    )

# Carga del grafo oficial --------------------------------------------------
try:
    graph_data = torch.load(
        GRAPH_DATA_FILE,
        weights_only = False
    ) # Grafo oficial

except Exception as error:
    raise RuntimeError(
        f"Error al cargar el GraphData: {error}"
    )

# Validación del grafo -----------------------------------------------------
if graph_data is None:
    raise ValueError(
        "El GraphData oficial no fue cargado correctamente."
    )

required_graph_attributes = [
    "x",
    "edge_index",
    "num_node_features"
] # Atributos obligatorios

missing_graph_attributes = [
    attribute
    for attribute in required_graph_attributes
    if not hasattr(
        graph_data,
        attribute
    )
] # Atributos faltantes

if missing_graph_attributes:
    raise ValueError(
        f"El GraphData está incompleto: {missing_graph_attributes}"
    )

# Consolidación de los datos -----------------------------------------------
evaluation_data = {
    "dataset": dataset,
    "graph_data": graph_data
} # Datos oficiales de la evaluación

# Validación ---------------------------------------------------------------
required_data_keys = [
    "dataset",
    "graph_data"
] # Parámetros obligatorios

missing_data_keys = [
    key
    for key in required_data_keys
    if key not in evaluation_data
] # Parámetros faltantes

if missing_data_keys:
    raise ValueError(
        f"Faltan parámetros en evaluation_data: {missing_data_keys}"
    )

# Resumen ------------------------------------------------------------------
print(f"Observaciones: {len(dataset):,}")
print(f"Variables: {dataset.shape[1]}")

print("-" * 80)

# BLOQUE 5. Preparación e Inferencia ---------------------------------------
## Objetivo: Preparar las entradas oficiales y generar las predicciones del
# modelo ganador utilizando las funciones oficiales del proyecto.
#### Entradas: - trained_model - model_metadata - evaluation_data
#### Producto: - prediction_result - tabular_features
#### Responde: ¿El modelo oficial generó correctamente las predicciones del conjunto de evaluación?

# Recuperación de información ----------------------------------------------
model_family = model_metadata[
    "family"
] # Familia del modelo

# Funciones oficiales -------------------------------------------------------
model_functions = {
    "statistical": {
        "predict": predict_linear_regression,
        "evaluate": evaluate_linear_regression
    },

    "machine_learning": {
        "predict": predict_machine_learning_model,
        "evaluate": evaluate_machine_learning_model
    },

    "deep_learning": {
        "predict": predict_mlp,
        "evaluate": evaluate_mlp
    },

    "graph_neural_networks": {
        "predict": predict_gnn,
        "evaluate": evaluate_gnn
    }
} # Funciones oficiales por familia

# Recuperar funciones -------------------------------------------------------
model_handler = model_functions.get(
    model_family
)

if model_handler is None:
    raise RuntimeError(
        f"No existe un manejador oficial para '{model_family}'."
    )

prediction_function = model_handler[
    "predict"
] # Función oficial de predicción

evaluation_function = model_handler[
    "evaluate"
] # Función oficial de evaluación

# Inicialización ------------------------------------------------------------
tabular_features = None # Variables tabulares
x_data = None # Variables predictoras
y_true = None # Variable objetivo
prediction_output = None # Resultado de la inferencia

# Preparación de datos ------------------------------------------------------
if model_family != "graph_neural_networks":
    tabular_features = prepare_tabular_features(
        dataset = evaluation_data[
            "dataset"
        ],

        target_variable = DATASET_CONFIG[
            "target_variable"
        ],

        excluded_columns = DATASET_CONFIG[
            "excluded_columns"
        ]
    ) # Preparación oficial

    x_data = tabular_features[
        "x_data"
    ] # Variables predictoras

    y_true = tabular_features[
        "y_true"
    ] # Variable objetivo

# Inferencia ---------------------------------------------------------------
try:
    if model_family == "graph_neural_networks":
        prediction_output = prediction_function(
            model = trained_model,
            graph_data = evaluation_data[
                "graph_data"
            ]
        ) # Inferencia GNN

        y_true = prediction_output[
            "y_true"
        ] # Valores observados

    else:
        prediction_output = prediction_function(
            model = trained_model,
            x_test = x_data
        ) # Inferencia tabular

except Exception as error:
    raise RuntimeError(
        f"Error durante la generación de predicciones: {error}"
    )

# Validación ---------------------------------------------------------------
if prediction_output is None:
    raise RuntimeError(
        "La función oficial de predicción no devolvió resultados."
    )

if "y_pred" not in prediction_output:
    raise RuntimeError(
        "No fue posible recuperar las predicciones del modelo."
    )

# Consolidación -------------------------------------------------------------
prediction_result = {
    "x_data": (
        None
        if model_family == "graph_neural_networks"
        else x_data
    ),

    "feature_columns": (
        None
        if model_family == "graph_neural_networks"
        else tabular_features[
            "feature_columns"
        ]
    ),

    "n_features": (
        None
        if model_family == "graph_neural_networks"
        else tabular_features[
            "n_features"
        ]
    ),

    "y_true": y_true,
    "y_pred": prediction_output[
        "y_pred"
    ],

    "inference_time": prediction_output.get(
        "inference_time",
        None
    )
} # Resultado oficial de la inferencia

# Validación ---------------------------------------------------------------
required_keys = [
    "x_data",
    "feature_columns",
    "n_features",
    "y_true",
    "y_pred",
    "inference_time"
] # Parámetros obligatorios

missing_keys = [
    key
    for key in required_keys
    if key not in prediction_result
] # Parámetros faltantes

if missing_keys:
    raise RuntimeError(
        f"El resultado oficial de la inferencia está incompleto: {missing_keys}"
    )

y_true = np.asarray(
    prediction_result[
        "y_true"
    ]
) # Valores observados

y_pred = np.asarray(
    prediction_result[
        "y_pred"
    ]
) # Valores predichos

if len(
    y_true
) != len(
    y_pred
):
    raise RuntimeError(
        "Las dimensiones de y_true y y_pred no coinciden."
    )

print(f"Observaciones evaluadas: {len(y_true):,}")

if prediction_result[
    "inference_time"
] is not None:
    print(
        f"Tiempo de inferencia: {prediction_result['inference_time']:.4f} segundos"
    )

print("-" * 80)

# BLOQUE 6. Evaluación del Modelo ------------------------------------------
## Objetivo: Evaluar el desempeño predictivo del modelo oficial utilizando
# las funciones oficiales del Benchmark Científico.
#### Entradas: - prediction_result - evaluation_function
#### Producto: - evaluation_result
#### Responde: ¿El modelo oficial fue evaluado correctamente?

# Inicialización ------------------------------------------------------------
evaluation_result = None # Resultado oficial de la evaluación

# Evaluación ----------------------------------------------------------------
try:
    evaluation_result = evaluation_function(
        y_true = prediction_result[
            "y_true"
        ],

        y_pred = prediction_result[
            "y_pred"
        ]
    ) # Evaluación oficial

except Exception as error:
    raise RuntimeError(
        f"Error durante la evaluación del modelo: {error}"
    )

# Validación ---------------------------------------------------------------
if evaluation_result is None:
    raise RuntimeError(
        "La función oficial de evaluación no devolvió resultados."
    )

required_keys = [
    "rmse",
    "mae",
    "mape",
    "r2",
    "adjusted_r2"
] # Parámetros obligatorios

missing_keys = [
    key
    for key in required_keys
    if key not in evaluation_result
] # Parámetros faltantes

if missing_keys:
    raise RuntimeError(
        f"El resultado oficial de la evaluación está incompleto: {missing_keys}"
    )

# Resumen ------------------------------------------------------------------
print(f"RMSE      : {evaluation_result['rmse']:.6f}")
print(f"MAE       : {evaluation_result['mae']:.6f}")
print(f"MAPE      : {evaluation_result['mape']:.6f}")
print(f"R²        : {evaluation_result['r2']:.6f}")
print(f"Adj. R²   : {evaluation_result['adjusted_r2']:.6f}")

# Resultado del entrenamiento ----------------------------------------------
training_result = {
    "model": trained_model,
    "training_time": None
} # Resultado del entrenamiento disponible para la evaluación

print("-" * 80)

# BLOQUE 7. Construcción del Resultado Oficial -----------------------------
## Objetivo: Consolidar el resultado oficial de la evaluación científica
# utilizando la estructura oficial del Benchmark.
#### Entradas: - model_config - training_result - prediction_result - evaluation_result
#### Producto: - evaluation_summary
#### Responde: ¿El resultado oficial fue construido correctamente?

# Inicialización -----------------------------------------------------------
evaluation_summary = None # Resultado oficial

# Construcción -------------------------------------------------------------
try:
    evaluation_summary = build_benchmark_result(
        model_config = model_config,
        training_result = training_result,
        prediction_result = prediction_result,
        evaluation_result = evaluation_result
    ) # Resultado oficial

except Exception as error:
    raise RuntimeError(
        f"Error durante la construcción del resultado oficial: {error}"
    )

# Validación ---------------------------------------------------------------
if evaluation_summary is None:
    raise RuntimeError(
        "No fue posible construir el resultado oficial."
    )

required_keys = [
    "model_code",
    "model_name",
    "family",
    "model",
    "training_time",
    "inference_time",
    "rmse",
    "mae",
    "mape",
    "r2",
    "adjusted_r2"
] # Parámetros obligatorios

missing_keys = [
    key
    for key in required_keys
    if key not in evaluation_summary
] # Parámetros faltantes

if missing_keys:
    raise RuntimeError(
        f"El resultado oficial está incompleto: {missing_keys}"
    )

# Resumen ------------------------------------------------------------------
print(f"Modelo: {evaluation_summary['model_name']}")
print(f"Familia: {evaluation_summary['family']}")
print(f"RMSE: {evaluation_summary['rmse']:.6f}")
print(f"R²: {evaluation_summary['r2']:.6f}")

print("-" * 80)

# --------------------------------------------------------------------------
## Función auxiliar: standardize_shap_output()
# Función auxiliar ---------------------------------------------------------
def standardize_shap_output(
    shap_values
):
    """
    Convierte la salida de SHAP a un arreglo NumPy homogéneo.

    Parameters
    ----------
    shap_values : object
        Resultado devuelto por SHAP.

    Returns
    -------
    numpy.ndarray
        Valores SHAP estandarizados.
    """

    if hasattr(
        shap_values,
        "values"
    ):

        shap_values = shap_values.values

    if isinstance(
        shap_values,
        list
    ):

        shap_values = shap_values[0]

    return np.asarray(
        shap_values
    )

print("-" * 80)

# BLOQUE 8. Explicabilidad Global ------------------------------------------
## Objetivo: Generar la explicabilidad global del modelo oficial mediante
# métodos de inteligencia artificial explicable (XAI), identificando las
# variables más influyentes y construyendo la interpretación científica del
# comportamiento del modelo.
## Entradas:  - trained_model - model_metadata - model_config - evaluation_data - prediction_result - evaluation_result
## Producto: - global_explainability
## Responde: ¿Cuál es la explicación científica del comportamiento del modelo oficial
# y cuáles son las variables que más influyen en sus predicciones?

# Inicialización -----------------------------------------------------------
global_explainability = {
    "model_family": model_family,
    "model_name": model_name,
    "method": None,
    "summary_metadata": None,
    "plots_dir": (EVALUATION_REPORTS_DIR / "plots"), # Directorio oficial de visualizaciones
    "batch_size": None,
    "feature_importance": None,
    "feature_ranking": None,
    "top_5_variables": None,
    "top_10_variables": None,
    "top_20_variables": None,
    "shap_values": None,
    "attention_weights": None,
    "embeddings": None,
    "scientific_summary": None,
    "plots": {
        "summary": None,
        "beeswarm": None,
        "bar": None
    },
    "execution_time": {
        "feature_importance": None,
        "feature_ranking": None,
        "shap_visualization": None,
        "scientific_summary": None,
        "export": None,
        "validation": None,
        "explainer_construction": None,
        "total": None
    },
    "status": "INITIALIZED"
} # Resultado oficial de la interpretabilidad global

global_explainability["plots_dir"].mkdir(parents = True, exist_ok = True) # Crear carpeta de visualizaciones

# Validación ---------------------------------------------------------------
required_keys = [
    "model_family",
    "model_name",
    "method",
    "summary_metadata",
    "plots_dir",
    "batch_size",
    "feature_importance",
    "feature_ranking",
    "top_5_variables",
    "top_10_variables",
    "top_20_variables",
    "shap_values",
    "attention_weights",
    "embeddings",
    "scientific_summary",
    "plots",
    "execution_time",
    "status"
] # Parámetros obligatorios

missing_keys = [
    key
    for key in required_keys
    if key not in global_explainability
] # Parámetros faltantes

if missing_keys:

    raise RuntimeError(
        f"Faltan parámetros en global_explainability: {missing_keys}"
    )

print(
    "Estructura oficial de la explicabilidad inicializada correctamente."
)

print("-" * 80)

# 8.2 Selección del Método -----------------------------------------------
## Objetivo: Seleccionar el método oficial de interpretabilidad de acuerdo
# con la familia del modelo ganador.
## Entradas: - trained_model - model_family
## Producto: - global_explainability["method"]
## Responde: ¿Cuál es el método oficial de interpretabilidad del modelo ganador?

# Métodos oficiales --------------------------------------------------------
explainability_methods = {
    "statistical": "SHAP",
    "machine_learning": "TreeSHAP",
    "deep_learning": "DeepSHAP"
} # Métodos oficiales

# Selección ---------------------------------------------------------------
if model_family == "graph_neural_networks":
    if hasattr(
        trained_model,
        "attention_weights"
    ):
        explainability_method = "Attention"

    elif hasattr(
        trained_model,
        "get_embeddings"
    ):
        explainability_method = "Embeddings"

    else:
        explainability_method = "GNNExplainer"

else:
    explainability_method = explainability_methods.get(
        model_family
    )

# Validación --------------------------------------------------------------
if explainability_method is None:
    raise RuntimeError(
        f"No existe un método oficial de interpretabilidad para '{model_family}'."
    )

# Registro ----------------------------------------------------------------
global_explainability[
    "method"
] = explainability_method # Método oficial

print(f"Método de interpretabilidad: {explainability_method}")

print("-" * 80)

# 8.3 Construcción del Explainer -------------------------------------------
## Objetivo: Construir el explicador oficial para generar la
# interpretabilidad global del modelo ganador.
#### Entradas: - trained_model - x_data - global_explainability["method"]
#### Producto: - explainer
#### Responde: ¿El explicador oficial fue construido correctamente?

# Inicio -------------------------------------------------------------------
step_start_time = time.time() # Inicio del cronómetro
print("\n8.3 Construcción del Explainer")

# Recuperación -------------------------------------------------------------
explainability_method = global_explainability[
    "method"
] # Método oficial de interpretabilidad

# Inicialización -----------------------------------------------------------
explainer = {
    "type": explainability_method,
    "object": None,
    "background": None
} # Explicador oficial

# Construcción -------------------------------------------------------------
try:
    if explainability_method == "SHAP":
        explainer["object"] = shap.Explainer(
            trained_model,
            x_data
        )
        explainer["background"] = x_data

    elif explainability_method == "TreeSHAP":
        explainer["object"] = shap.TreeExplainer(
            trained_model
        )
        explainer["background"] = x_data

    elif explainability_method == "DeepSHAP":
        trained_model.eval()
        x_tensor = torch.tensor(
            x_data.values,
            dtype = torch.float32
        )

        explainer["object"] = shap.DeepExplainer(
            trained_model,
            x_tensor
        )
        explainer["background"] = x_tensor

    elif explainability_method in [
        "Attention",
        "Embeddings",
        "GNNExplainer"
    ]:
        explainer["object"] = trained_model
        explainer["background"] = None

    else:
        raise ValueError(
            f"Método de interpretabilidad no soportado: {explainability_method}"
        )

except Exception as error:
    raise RuntimeError(
        f"Error durante la construcción del Explainer: {error}"
    )

# Validación ---------------------------------------------------------------
required_keys = [
    "type",
    "object",
    "background"
] # Parámetros obligatorios

missing_keys = [
    key
    for key in required_keys
    if key not in explainer
] # Parámetros faltantes

if missing_keys:
    raise RuntimeError(
        f"Faltan parámetros en el Explainer: {missing_keys}"
    )

if explainability_method not in [
    "Attention",
    "Embeddings",
    "GNNExplainer"
]:

    if explainer["object"] is None:
        raise RuntimeError(
            "No fue posible construir el Explainer."
        )

# Tiempo -------------------------------------------------------------------
step_elapsed_time = (
    time.time()
    - step_start_time
) # Tiempo del módulo

global_explainability[
    "execution_time"
][
    "explainer_construction"
] = step_elapsed_time # Registrar tiempo

print(f"Explainer construido correctamente en {step_elapsed_time:.2f} segundos.")

print("-" * 80)

# 8.4 Generación de la Explicabilidad --------------------------------------
## Objetivo: Generar la interpretabilidad global del modelo oficial mediante
# el método seleccionado y consolidar los resultados obtenidos.
#### Entradas: - explainer - x_data - graph_data - trained_model
#### Producto: - global_explainability
#### Responde: ¿La explicabilidad global fue generada correctamente?

# Inicio -------------------------------------------------------------------
step_start_time = time.time() # Inicio del cronómetro
print("\n8.4 Generación de la Explicabilidad")

# Configuración ------------------------------------------------------------
batch_size = EVALUATION_CONFIG.get(
    "shap_batch_size",
    512
) # Tamaño del lote

global_explainability[
    "batch_size"
] = batch_size # Registrar tamaño del lote

show_progress = EVALUATION_CONFIG.get(
    "show_progress",
    True
) # Mostrar progreso

explainer_type = explainer[
    "type"
] # Tipo de explicador

# Inicialización -----------------------------------------------------------
explainability_batches = [] # Resultados por lote

progress_bar = None # Barra de progreso

# Procesamiento ------------------------------------------------------------
try:
    if explainer_type in [
        "SHAP",
        "TreeSHAP",
        "DeepSHAP"
    ]:

        total_rows = len(
            x_data
        ) # Número de observaciones

        if total_rows == 0:
            raise RuntimeError(
                "No existen observaciones para generar la explicabilidad."
            )

        if show_progress:
            progress_bar = tqdm(
                total = int(
                    np.ceil(
                        total_rows / batch_size
                    )
                ),
                desc = "Calculando SHAP",
                unit = "batch",
                dynamic_ncols = True
            )

        for start in range(
            0,
            total_rows,
            batch_size
        ):

            end = min(
                start + batch_size,
                total_rows
            )

            batch = x_data.iloc[
                start:end
            ]

            if explainer_type == "DeepSHAP":
                batch_tensor = torch.tensor(
                    batch.values,
                    dtype = torch.float32
                )

                batch_result = explainer[
                    "object"
                ].shap_values(
                    batch_tensor
                )

            else:
                batch_result = explainer[
                    "object"
                ](
                    batch
                )

            explainability_batches.append(
                standardize_shap_output(
                    batch_result
                )
            )

            if progress_bar is not None:
                progress_bar.update(
                    1
                )

        global_explainability[
            "shap_values"
        ] = np.vstack(
            explainability_batches
        )

    elif explainer_type == "Attention":
        global_explainability[
            "attention_weights"
        ] = trained_model.attention_weights

    elif explainer_type == "Embeddings":
        with torch.no_grad():
            global_explainability[
                "embeddings"
            ] = trained_model.get_embeddings(
                graph_data
            )

    elif explainer_type == "GNNExplainer":
        raise NotImplementedError(
            "GNNExplainer aún no se encuentra implementado."
        )

    else:
        raise RuntimeError(
            f"Tipo de explicador no soportado: {explainer_type}"
        )

finally:
    if progress_bar is not None:
        progress_bar.close()

# Validación ---------------------------------------------------------------
if explainer_type in [
    "SHAP",
    "TreeSHAP",
    "DeepSHAP"
]:

    if global_explainability[
        "shap_values"
    ] is None:

        raise RuntimeError(
            "No fue posible generar los valores SHAP."
        )

elif explainer_type == "Attention":
    if global_explainability[
        "attention_weights"
    ] is None:
        raise RuntimeError(
            "No fue posible recuperar los pesos de atención."
        )

elif explainer_type == "Embeddings":
    if global_explainability[
        "embeddings"
    ] is None:

        raise RuntimeError(
            "No fue posible recuperar los embeddings."
        )

# Registro del tiempo ------------------------------------------------------
step_elapsed_time = (
    time.time()
    - step_start_time
) # Tiempo del módulo

global_explainability[
    "execution_time"
][
    "total"
] = step_elapsed_time # Tiempo de ejecución

print(f"Explicabilidad generada correctamente en {step_elapsed_time:.2f} segundos.")

print("-" * 80)

# 8.5 Construcción de la Importancia y Ranking -----------------------------
## Objetivo: Construir la importancia global de las variables y el ranking
# oficial de interpretabilidad del modelo ganador.
## Entradas: - global_explainability - x_data
## Producto: - feature_importance - feature_ranking - top_5_variables - top_10_variables - top_20_variables
## Responde: ¿Cuáles son las variables más importantes para el modelo oficial?

# Inicio -------------------------------------------------------------------
step_start_time = time.time() # Inicio del cronómetro
print("\n8.5 Construcción de la Importancia y Ranking")

# Recuperación -------------------------------------------------------------
explainer_type = explainer[
    "type"
] # Tipo de explicador

feature_names = list(
    x_data.columns
) # Variables predictoras

# Construcción de la importancia -------------------------------------------
if explainer_type in [
    "SHAP",
    "TreeSHAP",
    "DeepSHAP"
]:
    importance = np.abs(
        global_explainability[
            "shap_values"
        ]
    ).mean(
        axis = 0
    ) # Importancia media absoluta

elif explainer_type == "Attention":
    importance = np.mean(
        global_explainability[
            "attention_weights"
        ],
        axis = 0
    ) # Importancia por atención

elif explainer_type == "Embeddings":
    importance = np.linalg.norm(
        global_explainability[
            "embeddings"
        ],
        axis = 1
    ) # Magnitud de embeddings

else:
    raise RuntimeError(
        f"No existe un método de importancia para '{explainer_type}'."
    )

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
}).sort_values(
    by = "importance",
    ascending = False,
    ignore_index = True
) # Importancia global

# Construcción del ranking -------------------------------------------------
feature_ranking = feature_importance.copy()
feature_ranking[
    "rank"
] = np.arange(
    1,
    len(
        feature_ranking
    ) + 1

) # Ranking oficial

# Consolidación ------------------------------------------------------------
global_explainability[
    "feature_importance"
] = feature_importance

global_explainability[
    "feature_ranking"
] = feature_ranking

global_explainability[
    "top_5_variables"
] = feature_ranking.head(5)

global_explainability[
    "top_10_variables"
] = feature_ranking.head(10)

global_explainability[
    "top_20_variables"
] = feature_ranking.head(20)

# Validación ---------------------------------------------------------------
required_keys = [
    "feature_importance",
    "feature_ranking",
    "top_5_variables",
    "top_10_variables",
    "top_20_variables"
] # Parámetros obligatorios

missing_keys = [
    key
    for key in required_keys
    if global_explainability[
        key
    ] is None
] # Parámetros faltantes

if missing_keys:
    raise RuntimeError(
        f"No fue posible construir: {missing_keys}"
    )

# Registro del tiempo ------------------------------------------------------
step_elapsed_time = (
    time.time()
    - step_start_time
) # Tiempo del módulo

global_explainability[
    "execution_time"
][
    "feature_importance"
] = step_elapsed_time

global_explainability[
    "execution_time"
][
    "feature_ranking"
] = step_elapsed_time

print(f"Ranking construido correctamente con {len(feature_ranking)} variables.")

print("-" * 80)

# 8.6 Visualizaciones ------------------------------------------------------
## Objetivo: Generar las visualizaciones oficiales de la interpretabilidad global del modelo ganador.
#### Entradas: - explainer - global_explainability - x_data
#### Producto: - global_explainability["plots"]
#### Responde: ¿Las visualizaciones oficiales fueron generadas correctamente?

# Inicio -------------------------------------------------------------------
step_start_time = time.time() # Inicio del cronómetro

print("\n8.6 Visualizaciones")

# Recuperación -------------------------------------------------------------
explainer_type = explainer[
    "type"
] # Tipo de explicador

plots_dir = global_explainability[
    "plots_dir"
] # Directorio de salida

shap_values = global_explainability[
    "shap_values"
] # Valores SHAP

plots = global_explainability[
    "plots"
] # Registro de visualizaciones

# Generación ---------------------------------------------------------------
if explainer_type in [
    "SHAP",
    "TreeSHAP",
    "DeepSHAP"
]:

    summary_plot_file = (
        plots_dir
        / "summary_plot.png"
    ) # Summary Plot

    shap.summary_plot(
        shap_values,
        x_data,
        show = False
    )

    plt.tight_layout()

    plt.savefig(
        summary_plot_file,
        dpi = 300,
        bbox_inches = "tight"
    )

    plt.close()

    plots[
        "summary"
    ] = summary_plot_file

    beeswarm_plot_file = (
        plots_dir
        / "beeswarm_plot.png"
    ) # Beeswarm Plot

    shap.plots.beeswarm(
        shap.Explanation(
            values = shap_values,
            data = x_data.values,
            feature_names = x_data.columns
        ),
        show = False
    )

    plt.tight_layout()

    plt.savefig(
        beeswarm_plot_file,
        dpi = 300,
        bbox_inches = "tight"
    )

    plt.close()

    plots[
        "beeswarm"
    ] = beeswarm_plot_file

    bar_plot_file = (
        plots_dir
        / "bar_plot.png"
    ) # Bar Plot

    shap.plots.bar(
        shap.Explanation(
            values = shap_values,
            data = x_data.values,
            feature_names = x_data.columns
        ),
        show = False
    )

    plt.tight_layout()

    plt.savefig(
        bar_plot_file,
        dpi = 300,
        bbox_inches = "tight"
    )

    plt.close()

    plots[
        "bar"
    ] = bar_plot_file

else:

    print(
        "El método de interpretabilidad seleccionado no genera visualizaciones SHAP."
    )

# Validación ---------------------------------------------------------------
if explainer_type in [
    "SHAP",
    "TreeSHAP",
    "DeepSHAP"
]:

    for plot_name, plot_file in plots.items():

        if plot_file is None:

            raise RuntimeError(
                f"No fue posible generar '{plot_name}'."
            )

        if not plot_file.exists():

            raise RuntimeError(
                f"No existe el archivo '{plot_file}'."
            )

# Registro del tiempo ------------------------------------------------------
step_elapsed_time = (
    time.time()
    - step_start_time
) # Tiempo del módulo

global_explainability[
    "execution_time"
][
    "shap_visualization"
] = step_elapsed_time

print(
    "Visualizaciones generadas correctamente."
)

print("-" * 80)

# 8.7 Resumen Científico -----------------------------------------------
## Objetivo: Construir el resumen científico oficial de la
# interpretabilidad global del modelo ganador.
#### Entradas: - model_metadata - evaluation_result - global_explainability
#### Producto: - scientific_summary
#### Responde: ¿Cuál es la interpretación científica del modelo ganador?

# Inicio ------------------------------------------------------------------
step_start_time = time.time() # Inicio del cronómetro
print("\n8.7 Resumen Científico")

# Recuperación ------------------------------------------------------------
top_variables = global_explainability[
    "top_5_variables"
] # Variables más importantes

model_name = model_metadata[
    "model_name"
] # Nombre del modelo

rmse = evaluation_result[
    "rmse"
] # Error RMSE

r2 = evaluation_result[
    "r2"
] # Coeficiente de determinación

# Construcción ------------------------------------------------------------
scientific_summary = {
    "model": model_name,
    "rmse": rmse,
    "r2": r2,
    "top_variables": top_variables,
    "n_features": len(
        global_explainability[
            "feature_ranking"
        ]
    )
} # Resumen científico

# Consolidación -----------------------------------------------------------
global_explainability[
    "scientific_summary"
] = scientific_summary

# Validación --------------------------------------------------------------
if global_explainability[
    "scientific_summary"
] is None:
    raise RuntimeError(
        "No fue posible construir el resumen científico."
    )

# Registro del tiempo -----------------------------------------------------
step_elapsed_time = (
    time.time()
    - step_start_time
) # Tiempo del módulo

global_explainability[
    "execution_time"
][
    "scientific_summary"
] = step_elapsed_time

print("Resumen científico construido correctamente.")

print("-" * 80)

# 8.8 Exportación ----------------------------------------------------------
## Objetivo: Exportar el resultado oficial de la interpretabilidad global.
## Entradas:  - evaluation_summary - global_explainability
## Producto: - explainability_results
## Responde: ¿Los resultados oficiales fueron exportados correctamente?

# Inicio -------------------------------------------------------------------
step_start_time = time.time() # Inicio del cronómetro
print("\n8.8 Exportación")

# Recuperación -------------------------------------------------------------
output_file = (
    EVALUATION_REPORTS_DIR
    / "explainability_results.joblib"
) # Archivo oficial

# Consolidación ------------------------------------------------------------
explainability_results = {
    "evaluation_summary": evaluation_summary,
    "global_explainability": global_explainability
} # Resultado oficial

output_file.parent.mkdir(
    parents = True,
    exist_ok = True
) # Crear carpeta si no existe

# Exportación --------------------------------------------------------------
try:
    joblib.dump(
        explainability_results,
        output_file
    ) # Exportación oficial

except Exception as error:
    raise RuntimeError(
        f"Error durante la exportación: {error}"
    )

# Validación ---------------------------------------------------------------
if not output_file.exists():
    raise RuntimeError(
        "No fue posible exportar el resultado oficial."
    )

# Registro -----------------------------------------------------------------
global_explainability[
    "execution_time"
][
    "export"
] = (
    time.time()
    - step_start_time
)

print(f"Resultado oficial exportado en:\n{output_file}")

print("-" * 80)

# 8.9 Validación -----------------------------------------------------------
## Objetivo: Validar la integridad del resultado oficial de la
# interpretabilidad global.
## Entradas:  - global_explainability - evaluation_summary
## Producto: - global_explainability["status"]
## Responde: ¿El resultado oficial de la interpretabilidad es consistente?

# Inicio -------------------------------------------------------------------
step_start_time = time.time() # Inicio del cronómetro
print("\n8.9 Validación")

# Validación ---------------------------------------------------------------
required_keys = [
    "method",
    "feature_importance",
    "feature_ranking",
    "scientific_summary",
    "plots",
    "execution_time"
] # Parámetros obligatorios

missing_keys = [
    key
    for key in required_keys
    if global_explainability.get(
        key
    ) is None
] # Parámetros faltantes

if missing_keys:
    raise RuntimeError(
        f"La interpretabilidad está incompleta: {missing_keys}"
    )

global_explainability[
    "status"
] = "COMPLETED" # Estado final

# Registro del tiempo ------------------------------------------------------
step_elapsed_time = (
    time.time()
    - step_start_time
) # Tiempo del módulo

global_explainability[
    "execution_time"
][
    "validation"
] = step_elapsed_time
print("Validación final completada correctamente.")

print("-" * 80)


# Auditoría científica de la importancia de variables -----------------------

import numpy as np
import pandas as pd

feature_importance = (
    global_explainability["feature_importance"]
    .copy()
)

feature_importance = feature_importance.sort_values(
    "importance",
    ascending = False
).reset_index(drop = True)

feature_importance["importance_pct"] = (
    feature_importance["importance"]
    / feature_importance["importance"].sum()
    * 100
)

feature_importance["importance_acumulada"] = (
    feature_importance["importance_pct"].cumsum()
)

print("\n")
print("=" * 80)
print("AUDITORÍA CIENTÍFICA DE LA IMPORTANCIA DE VARIABLES")
print("=" * 80)

print(f"\nNúmero de variables: {len(feature_importance)}")

print("\nTop 20 variables\n")
print(
    feature_importance.head(20).to_string(
        index = False,
        float_format = "{:.4f}".format
    )
)

print("\n")
print("-" * 80)

top1 = feature_importance.iloc[0]["importance_pct"]
top3 = feature_importance.head(3)["importance_pct"].sum()
top5 = feature_importance.head(5)["importance_pct"].sum()
top10 = feature_importance.head(10)["importance_pct"].sum()

print(f"Importancia Top 1  : {top1:.2f}%")
print(f"Importancia Top 3  : {top3:.2f}%")
print(f"Importancia Top 5  : {top5:.2f}%")
print(f"Importancia Top 10 : {top10:.2f}%")

print("\n")
print("-" * 80)

n_mayor_1 = (feature_importance["importance_pct"] > 1).sum()
n_mayor_5 = (feature_importance["importance_pct"] > 5).sum()
n_mayor_10 = (feature_importance["importance_pct"] > 10).sum()
n_menor_01 = (feature_importance["importance_pct"] < 0.1).sum()

print(f"Variables >10% : {n_mayor_10}")
print(f"Variables >5%  : {n_mayor_5}")
print(f"Variables >1%  : {n_mayor_1}")
print(f"Variables <0.1%: {n_menor_01}")

print("\n")
print("-" * 80)

def gini(x):

    x = np.asarray(x, dtype=float)

    if np.any(x < 0):
        raise ValueError("Todos los valores deben ser positivos.")

    if np.all(x == 0):
        return 0

    x = np.sort(x)

    n = len(x)

    return (
        (
            2 * np.sum(
                (np.arange(1, n + 1) * x)
            )
            /
            (n * np.sum(x))
        )
        - (n + 1) / n
    )

print(f"Índice de concentración (Gini): {coef_gini:.4f}")

print("\n")
print("-" * 80)

if top1 > 50:

    print("ALERTA:")
    print("Una sola variable explica más del 50 de la importancia.")
    print("Se recomienda revisar posible dependencia del objetivo.")

elif top5 > 90:

    print("ALERTA:")
    print("Las cinco primeras variables concentran más del 90% de la importancia.")

else:

    print("Distribución de importancia aparentemente balanceada.")

print("\n")
print("-" * 80)

# Correlación con la variable objetivo --------------------------------------

target = DATASET_CONFIG["target_variable"]

print("\n")
print("=" * 80)
print("CORRELACIÓN CON LA VARIABLE OBJETIVO")
print("=" * 80)

correlations = (
    dataset
    .corr(numeric_only = True)[target]
    .sort_values(
        ascending = False
    )
)

print(
    correlations.to_string(
        float_format = "{:.4f}".format
    )
)

print("\n")
print("=" * 80)

print("\nVariables con correlación absoluta mayor a 0.90\n")

print(
    correlations[
        correlations.abs() > 0.90
    ]
)