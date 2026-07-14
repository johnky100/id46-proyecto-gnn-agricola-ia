# 03_train_model.py

# BLOQUE 1. Importaciones --------------------------------------------------
## Objetivo: Importar las librerías, modelos y utilidades necesarias para reentrenar el modelo ganador del Benchmark Científico.
# Funciones del sistema
import json # Lectura de configuraciones
import joblib # Serialización de objetos de Python
import warnings # Control de advertencias

# Librerías científicas
import torch # Modelos Deep Learning y GNN
import pandas as pd # Lectura del dataset científico
#import arrow # Lectura de archivos Parquet

# Configuración del proyecto
from src.python.config.paths import (
    DATASET_FILE,
    GRAPH_DATA_FILE,
    BEST_MODEL_CONFIG_FILE,
    BENCHMARK_RESULTS_FILE,
    BEST_MODEL_JOBLIB_FILE,
    BEST_MODEL_TORCH_FILE,
    BEST_MODEL_METADATA_FILE
) # Productos oficiales del proyecto

from pathlib import Path # Gestión de rutas

from datetime import datetime # Fecha y hora

from src.python.config.config_project import (
    DATASET_CONFIG
)

from src.python.models.statistical import (
    STATISTICAL_CONFIG,
    train_linear_regression
)

from src.python.models.machine_learning import (
    MACHINE_LEARNING_CONFIG,
    train_machine_learning_model
)

from src.python.models.deep_learning import (
    DEEP_LEARNING_CONFIG,
    train_mlp
)

from src.python.models.graph_neural_networks import (
    GNN_CONFIG,
    build_gnn_model,
    build_training_components,
    train_gnn
)

warnings.filterwarnings(
    "ignore"
) # Ocultar advertencias no críticas

print("-" * 80)

# BLOQUE 2. Configuración --------------------------------------------------
## Objetivo: Definir la configuración oficial utilizada durante el entrenamiento
# definitivo del modelo ganador del Benchmark Científico.
### Producto: - TRAINING_CONFIG
### Responde: ¿El entrenamiento final dispone de una configuración oficial, reproducible y consistente con el protocolo científico?

# Configuración oficial del entrenamiento ----------------------------------
TRAINING_CONFIG = {
    "training_name": "final_training",
    "training_version": "1.0",
    "use_full_dataset": True,
    "save_trained_model": True,
    "save_training_summary": True,
    "save_training_metrics": True,
    "save_training_metadata": True,
    "overwrite_existing_model": True,
    "random_state": 42,
    "verbose": True
} # Configuración oficial del entrenamiento

# Validación ---------------------------------------------------------------
required_keys = [
    "training_name",
    "training_version",
    "use_full_dataset",
    "save_trained_model",
    "save_training_summary",
    "save_training_metrics",
    "save_training_metadata",
    "overwrite_existing_model",
    "random_state",
    "verbose"
] # Parámetros obligatorios

missing_keys = [
    key
    for key in required_keys
    if key not in TRAINING_CONFIG
] # Parámetros faltantes

if missing_keys:
    raise ValueError(
        f"Faltan parámetros en TRAINING_CONFIG: {missing_keys}"
    )

print("-" * 80)

# BLOQUE 3. Carga del Modelo Ganador ---------------------------------------
## Objetivo: Recuperar el modelo ganador del Benchmark Científico y reconstruir
# automáticamente su configuración oficial para el entrenamiento definitivo.
#### Entradas: - BEST_MODEL_CONFIG_FILE - BENCHMARK_RESULTS_FILE
#### Producto: - best_model_config - benchmark_results - best_model_result
#### Responde: ¿La información del modelo ganador fue recuperada correctamente para iniciar el entrenamiento definitivo?

# Carga de la configuración del modelo ganador -----------------------------
try:
    with open(
        BEST_MODEL_CONFIG_FILE,
        "r",
        encoding = "utf-8"
    ) as file:

        best_model_config = json.load(
            file
        ) # Configuración mínima del modelo ganador

except Exception as error:
    raise RuntimeError(
        f"Error al cargar la configuración del modelo ganador: {error}"
    )

# Validación de la configuración mínima ------------------------------------
required_config_keys = [
    "model_code",
    "model_name",
    "family"
] # Parámetros obligatorios

missing_config_keys = [
    key
    for key in required_config_keys
    if key not in best_model_config
] # Parámetros faltantes

if missing_config_keys:
    raise ValueError(
        f"Faltan parámetros en best_model_config: {missing_config_keys}"
    )

# Reconstrucción de la configuración oficial -------------------------------
try:
    model_family = (
        best_model_config["family"]
    ) # Familia del modelo

    model_name = (
        best_model_config["model_name"]
    ) # Nombre oficial del modelo

    if model_family == "statistical":

        best_model_config = (
            STATISTICAL_CONFIG
        ) # Configuración oficial

    elif model_family == "machine_learning":
        best_model_config = (
            MACHINE_LEARNING_CONFIG[
                model_name
            ]
        ) # Configuración oficial

    elif model_family == "deep_learning":
        best_model_config = (
            DEEP_LEARNING_CONFIG[
                model_name
            ]
        ) # Configuración oficial

    elif model_family == "graph_neural_networks":
        best_model_config = (
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
        f"Error al reconstruir la configuración oficial del modelo: {error}"
    )

# Carga de los resultados del Benchmark -----------------------------------
try:
    benchmark_results = joblib.load(
        BENCHMARK_RESULTS_FILE
    ) # Resultados oficiales del Benchmark

except Exception as error:
    raise RuntimeError(
        f"Error al cargar los resultados del Benchmark: {error}"
    )

# Validación de los resultados ---------------------------------------------
if not benchmark_results:
    raise ValueError(
        "Los resultados del Benchmark están vacíos."
    )

# Recuperación del resultado oficial del modelo ganador --------------------
try:
    best_model_result = next(
        result
        for result in benchmark_results
        if result["model_code"]
        ==
        best_model_config["model_code"]
    ) # Resultado oficial del modelo ganador

except StopIteration:
    raise ValueError(
        "No fue posible localizar el modelo ganador dentro de los resultados del Benchmark."
    )

# Validación del resultado recuperado --------------------------------------
required_result_keys = [
    "model_code",
    "model_name",
    "family",
    "model"
] # Parámetros obligatorios

missing_result_keys = [
    key
    for key in required_result_keys
    if key not in best_model_result
] # Parámetros faltantes

if missing_result_keys:
    raise ValueError(
        f"El resultado del modelo ganador está incompleto: {missing_result_keys}"
    )

print("-" * 80)

# BLOQUE 4. Carga de Datos -------------------------------------------------
## Objetivo: Cargar el dataset científico completo y el GraphData oficial
# utilizados durante el entrenamiento definitivo del modelo ganador.
### Entradas: - DATASET_FILE - GRAPH_DATA_FILE
### Producto: - training_data
### Responde: ¿Los datos oficiales del proyecto fueron cargados correctamente para iniciar el entrenamiento definitivo?

# Validación de archivos ---------------------------------------------------
required_files = [
    DATASET_FILE,
    GRAPH_DATA_FILE
] # Archivos obligatorios

missing_files = [
    file
    for file in required_files
    if not file.exists()
] # Archivos faltantes

if missing_files:
    raise FileNotFoundError(
        f"No fue posible localizar los archivos oficiales: {missing_files}"
    )

# Carga del dataset científico ---------------------------------------------
try:
    dataset = pd.read_parquet(
        DATASET_FILE
    ) # Dataset científico oficial

except Exception as error:
    raise RuntimeError(
        f"Error al cargar el dataset científico: {error}"
    )

# Carga del GraphData ------------------------------------------------------
try:
    graph_data = torch.load(
        GRAPH_DATA_FILE,
        weights_only = False
    ) # Grafo oficial

except Exception as error:
    raise RuntimeError(
        f"Error al cargar el GraphData: {error}"
    )

# Consolidación de los datos -----------------------------------------------
training_data = {
    "dataset": dataset,
    "graph_data": graph_data
} # Datos oficiales del entrenamiento

# Validación del dataset ---------------------------------------------------
if training_data["dataset"].empty:
    raise ValueError(
        "El dataset científico está vacío."
    )

# Validación del GraphData -------------------------------------------------
if training_data["graph_data"] is None:
    raise ValueError(
        "El GraphData no fue cargado correctamente."
    )

if not hasattr(
    training_data["graph_data"],
    "num_node_features"
):
    raise ValueError(
        "El GraphData no corresponde a un objeto válido de PyTorch Geometric."
    )

print("-" * 80)

# BLOQUE 5. Preparación del Entrenamiento ----------------------------------
## Objetivo: Preparar las entradas generales requeridas para el entrenamiento
# definitivo del modelo ganador, de acuerdo con la familia identificada
# durante el Benchmark Científico.
#### Entradas: - best_model_config - training_data
#### Producto: - training_inputs
#### Responde: ¿Las entradas generales del entrenamiento fueron preparadas correctamente para el modelo ganador?

# Recuperación de la configuración -----------------------------------------
model_config = (
    best_model_config
) # Configuración oficial del modelo ganador

model_family = (
    model_config["family"]
) # Familia del modelo ganador

# Validación de la familia -------------------------------------------------
supported_families = [
    "statistical",
    "machine_learning",
    "deep_learning",
    "graph_neural_networks"
] # Familias soportadas

if model_family not in supported_families:
    raise ValueError(
        f"Familia de modelo no soportada: {model_family}"
    )

# Construcción de las entradas generales -----------------------------------
training_inputs = {
    "family": model_family,
    "model_config": best_model_config,
    "dataset": training_data["dataset"],
    "target_variable": DATASET_CONFIG["target_variable"],
    "excluded_columns": DATASET_CONFIG["excluded_columns"]
} # Entradas generales del entrenamiento

# Incorporación de los datos ------------------------------------------------
if model_family == "graph_neural_networks":
    training_inputs["graph_data"] = (
        training_data["graph_data"]
    ) # Grafo oficial

else:
    training_inputs["dataset"] = (
        training_data["dataset"]
    ) # Dataset científico

# Validación de la estructura ----------------------------------------------
required_keys = [
    "family",
    "model_config"
] # Parámetros obligatorios

missing_keys = [
    key
    for key in required_keys
    if key not in training_inputs
] # Parámetros faltantes

if missing_keys:
    raise ValueError(
        f"Faltan parámetros en training_inputs: {missing_keys}"
    )

# Validación de los datos --------------------------------------------------
if model_family == "graph_neural_networks":
    if training_inputs["graph_data"] is None:
        raise ValueError(
            "El GraphData oficial no fue cargado correctamente."
        )

else:
    if training_inputs["dataset"].empty:
        raise ValueError(
            "El dataset científico está vacío."
        )

# print(training_inputs.keys())
# print(training_inputs["model_config"])

print("-" * 80)

# BLOQUE 6. Preparación de las Entradas de Entrenamiento -------------------
## Objetivo: Construir las entradas requeridas para el entrenamiento definitivo
# del modelo ganador a partir de los datos oficiales del proyecto.
#### Entradas: - training_inputs
#### Producto: - training_features
#### Responde: ¿Las entradas del entrenamiento fueron preparadas correctamente para el modelo ganador?

# Recuperación de información ----------------------------------------------
model_family = (
    training_inputs["family"]
) # Familia del modelo ganador

model_config = (
    training_inputs["model_config"]
) # Configuración oficial del modelo

# Preparación para modelos tabulares ---------------------------------------
if model_family in [
    "statistical",
    "machine_learning",
    "deep_learning"
]:

    dataset = (
        training_inputs["dataset"]
    ) # Dataset científico

    target_variable = (
        training_inputs["target_variable"]
    ) # Variable objetivo

    excluded_columns = (
        training_inputs["excluded_columns"]
    ) # Columnas excluidas

    feature_columns = (
        dataset.drop(
            columns = excluded_columns + [target_variable],
            errors = "ignore"
        )
        .select_dtypes(
            include = [
                "number",
                "bool"
            ]
        )
        .columns.tolist()
    ) # Variables predictoras numéricas

    x_train = dataset[
        feature_columns
    ] # Variables de entrada

    y_train = dataset[
        target_variable
    ] # Variable objetivo

    training_features = {
        "family": model_family,
        "model_config": model_config,
        "feature_columns": feature_columns,
        "x_train": x_train,
        "y_train": y_train
    } # Entradas oficiales del entrenamiento

# Preparación para Graph Neural Networks -----------------------------------
elif model_family == "graph_neural_networks":
    training_features = {
        "family": model_family,
        "model_config": model_config,
        "graph_data": training_inputs["graph_data"]
    } # Entradas oficiales del entrenamiento

else:
    raise ValueError(
        f"Familia de modelo no soportada: {model_family}"
    )

# Validación ---------------------------------------------------------------
required_keys = [
    "family",
    "model_config"
] # Parámetros obligatorios

missing_keys = [
    key
    for key in required_keys
    if key not in training_features
] # Parámetros faltantes

if missing_keys:
    raise ValueError(
        f"Faltan parámetros en training_features: {missing_keys}"
    )

# Validación de modelos tabulares ------------------------------------------
if model_family in [
    "statistical",
    "machine_learning",
    "deep_learning"
]:

    if training_features["x_train"].empty:
        raise ValueError(
            "La matriz de entrenamiento X está vacía."
        )

    if training_features["y_train"].empty:
        raise ValueError(
            "La variable objetivo y está vacía."
        )

# Validación de modelos GNN ------------------------------------------------
else:
    if training_features["graph_data"] is None:
        raise ValueError(
            "El GraphData oficial no fue preparado correctamente."
        )

print("-" * 80)

# BLOQUE 7. Entrenamiento del Modelo ---------------------------------------
## Objetivo: Ejecutar el entrenamiento definitivo del modelo ganador utilizando
# las entradas oficiales preparadas durante el proceso de entrenamiento.
#### Entradas: - training_features
#### Producto: - training_result
#### Responde: ¿El modelo ganador fue entrenado correctamente utilizando los datos oficiales del proyecto?

# Recuperación de información ----------------------------------------------
model_family = (
    training_features["family"]
) # Familia del modelo ganador

model_config = (
    training_features["model_config"]
) # Configuración oficial del modelo

required_config_keys = [
    "model_code",
    "model_name",
    "family"
] # Parámetros obligatorios

missing_config_keys = [
    key
    for key in required_config_keys
    if key not in model_config
] # Parámetros faltantes

if missing_config_keys:
    raise ValueError(
        f"La configuración oficial del modelo está incompleta: {missing_config_keys}"
    )

# Entrenamiento ------------------------------------------------------------
if model_family == "statistical":

    training_result = train_linear_regression(
        x_train = training_features["x_train"],
        y_train = training_features["y_train"]
    ) # Entrenar modelo estadístico

elif model_family == "machine_learning":

    training_result = train_machine_learning_model(
        model_config = model_config,
        x_train = training_features["x_train"],
        y_train = training_features["y_train"]
    ) # Entrenar modelo de Machine Learning

elif model_family == "deep_learning":

    training_result = train_mlp(
        model_config = model_config,
        x_train = training_features["x_train"],
        y_train = training_features["y_train"]
    ) # Entrenar modelo Deep Learning

elif model_family == "graph_neural_networks":

    model = build_gnn_model(
        model_config = model_config,
        input_channels = (
            training_features["graph_data"].num_node_features
        ),
        output_channels = 1
    ) # Construcción del modelo

    training_components = build_training_components(
        model = model,
        model_config = model_config
    ) # Componentes de entrenamiento

    training_result = train_gnn(
        model = model,
        graph_data = training_features["graph_data"],
        criterion = training_components["criterion"],
        optimizer = training_components["optimizer"],
        model_config = model_config
    ) # Entrenar arquitectura GNN

else:

    raise ValueError(
        f"Familia de modelo no soportada: {model_family}"
    )

# Validación ---------------------------------------------------------------
if training_result is None:

    raise RuntimeError(
        "El entrenamiento definitivo no produjo resultados."
    )

required_keys = [
    "model"
] # Parámetros obligatorios

missing_keys = [
    key
    for key in required_keys
    if key not in training_result
] # Parámetros faltantes

if missing_keys:

    raise ValueError(
        f"El resultado del entrenamiento está incompleto: {missing_keys}"
    )

print("-" * 80)

# BLOQUE 8. Exportación del Modelo Oficial ---------------------------------
## Objetivo: Exportar el modelo oficial entrenado y los metadatos generados
# durante el entrenamiento definitivo para garantizar la reproducibilidad
# del proyecto.
#### Entradas: - training_inputs - training_result
#### Producto: - export_result
#### Responde: ¿El modelo oficial fue exportado correctamente para las siguientes etapas del proyecto?

# Recuperación de información ----------------------------------------------
best_model = (
    training_result["model"]
) # Modelo oficial entrenado

model_config = (
    training_inputs["model_config"]
) # Configuración oficial del modelo

model_family = (
    training_inputs["family"]
) # Familia del modelo ganador

# Selección del formato de exportación -------------------------------------
if model_family in [
    "statistical",
    "machine_learning"
]:
    model_file = (
        BEST_MODEL_JOBLIB_FILE
    ) # Modelo Scikit-Learn

    export_format = (
        "joblib"
    ) # Formato de exportación

elif model_family in [
    "deep_learning",
    "graph_neural_networks"
]:
    model_file = (
        BEST_MODEL_TORCH_FILE
    ) # Modelo PyTorch

    export_format = (
        "torch"
    ) # Formato de exportación

else:
    raise ValueError(
        f"Familia de modelo no soportada: {model_family}"
    )

# Exportación del modelo ---------------------------------------------------
try:
    if export_format == "joblib":

        joblib.dump(
            best_model,
            model_file
        ) # Exportar modelo Scikit-Learn

    else:
        torch.save(
            best_model,
            model_file
        ) # Exportar modelo PyTorch

except Exception as error:
    raise RuntimeError(
        f"Error al exportar el modelo oficial: {error}"
    )

# Construcción de los metadatos --------------------------------------------
training_metadata = {
    "model_code": model_config["model_code"],
    "model_name": model_config["model_name"],
    "family": model_family,
    "training_name": TRAINING_CONFIG["training_name"],
    "training_version": TRAINING_CONFIG["training_version"],
    "training_time": training_result.get(
        "training_time",
        None
    ),
    "export_format": export_format
} # Metadatos oficiales del entrenamiento

# Exportación de los metadatos ---------------------------------------------
try:
    with open(
        BEST_MODEL_METADATA_FILE,
        "w",
        encoding = "utf-8"
    ) as file:
        json.dump(
            training_metadata,
            file,
            indent = 4,
            ensure_ascii = False
        ) # Exportar metadatos

except Exception as error:
    raise RuntimeError(
        f"Error al exportar los metadatos del entrenamiento: {error}"
    )

# Construcción del resultado oficial ---------------------------------------
export_result = {
    "status": "SUCCESS",
    "model_code": model_config["model_code"],
    "model_name": model_config["model_name"],
    "family": model_family,
    "model_file": str(
        model_file
    ),
    "metadata_file": str(
        BEST_MODEL_METADATA_FILE
    ),
    "export_format": export_format
} # Resultado oficial de la exportación

print("-" * 80)

# BLOQUE 9. Validación Final -----------------------------------------------
## Objetivo: Verificar la integridad del entrenamiento definitivo y de todos
# los productos oficiales generados durante el proceso.
#### Entradas: - training_inputs - training_result - export_result
#### Producto: - validation_result
#### Responde: ¿El entrenamiento definitivo fue ejecutado correctamente y todos los
# productos oficiales fueron generados de forma íntegra?

# Recuperación de información ----------------------------------------------
model_config = (
    training_inputs["model_config"]
) # Configuración oficial del modelo

model_family = (
    training_inputs["family"]
) # Familia del modelo ganador

# Validación del entrenamiento ---------------------------------------------
if training_result is None:
    raise RuntimeError(
        "El entrenamiento definitivo no generó resultados."
    )

required_training_keys = [
    "model"
] # Parámetros obligatorios

missing_training_keys = [
    key
    for key in required_training_keys
    if key not in training_result
] # Parámetros faltantes

if missing_training_keys:
    raise ValueError(
        f"El resultado del entrenamiento está incompleto: {missing_training_keys}"
    )

# Validación de la exportación ---------------------------------------------
if export_result is None:
    raise RuntimeError(
        "La exportación del modelo no produjo resultados."
    )

required_export_keys = [
    "status",
    "model_file",
    "metadata_file",
    "export_format"
] # Parámetros obligatorios

missing_export_keys = [
    key
    for key in required_export_keys
    if key not in export_result
] # Parámetros faltantes

if missing_export_keys:
    raise ValueError(
        f"El resultado de la exportación está incompleto: {missing_export_keys}"
    )

# Validación de los archivos oficiales -------------------------------------
model_file = Path(
    export_result["model_file"]
) # Archivo del modelo

metadata_file = Path(
    export_result["metadata_file"]
) # Archivo de metadatos

if not model_file.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo exportado: {model_file.name}"
    )

if not metadata_file.exists():
    raise FileNotFoundError(
        f"No se encontraron los metadatos del entrenamiento: {metadata_file.name}"
    )

# Validación de los metadatos ----------------------------------------------
try:
    with open(
        metadata_file,
        "r",
        encoding = "utf-8"
    ) as file:
        metadata = json.load(
            file
        ) # Metadatos oficiales

except Exception as error:
    raise RuntimeError(
        f"Error al leer los metadatos del entrenamiento: {error}"
    )

required_metadata_keys = [
    "model_code",
    "model_name",
    "family",
    "training_name",
    "training_version",
    "export_format"
] # Parámetros obligatorios

missing_metadata_keys = [
    key
    for key in required_metadata_keys
    if key not in metadata
] # Parámetros faltantes

if missing_metadata_keys:
    raise ValueError(
        f"Los metadatos del entrenamiento están incompletos: {missing_metadata_keys}"
    )

# Construcción del resultado oficial ---------------------------------------
validation_result = {
    "status": "SUCCESS",
    "training_name": metadata["training_name"],
    "training_version": metadata["training_version"],
    "model_code": model_config["model_code"],
    "model_name": model_config["model_name"],
    "family": model_family,
    "model_file": str(
        model_file
    ),
    "metadata_file": str(
        metadata_file
    ),
    "validation_date": datetime.now().isoformat()
} # Resultado oficial de la validación

print("-" * 80)