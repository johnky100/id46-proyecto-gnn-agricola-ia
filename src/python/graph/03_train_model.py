# 04_train_model.py

# BLOQUE 1. IMPORTACIONES
# Objetivo: Importar exclusivamente las dependencias necesarias para entrenar, validar y persistir
# el Modelo Oficial GraphSAGE a partir de los productos científicos previamente generados por el Benchmark.

# Librerías estándar
import copy
import json
from pathlib import Path

# Librerías científicas
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F # Importar funciones funcionales de PyTorch
from torch_geometric.nn import SAGEConv # Importar capa convolucional GraphSAGE
import torch.nn as nn

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# Configuración Oficial del Proyecto
from src.python.config.config_project import (
    PROJECT_SEED,
    OFFICIAL_MODEL_CODE,
    OFFICIAL_MODEL_NAME,
    OFFICIAL_MODEL_FAMILY,
) # Cargar identidad y semilla del Modelo Oficial

# Rutas Oficiales del Proyecto
from src.python.config.paths import (
    OUTPUTS_DIR,
    BENCHMARK_DIR,
    TRAINING_DIR,
    TRAINING_CHECKPOINT_FILE,
    TRAINING_METADATA_FILE,
    TRAINING_SUMMARY_FILE,
    TRAINING_HISTORY_FILE,
    TRAINING_METRICS_FILE,
    TRAINING_MANIFEST_FILE,
    TRAINING_CONTRACT_FILE,
    TRAINING_LOG_FILE,
    TRAINING_CURVES_FILE,
    OFFICIAL_MODEL_CONFIG_FILE,
    OFFICIAL_MODEL_TORCH_FILE,
    OFFICIAL_MODEL_JOBLIB_FILE,
    BENCHMARK_EXPERIMENT_FILE,
    validate_project_structure,
) # Cargar rutas oficiales de entrenamiento y productos científicos

# BLOQUE 2. CONFIGURACIÓN Y RUTAS
# Objetivo: Definir la configuración mínima y las rutas oficiales utilizadas durante el entrenamiento,
# validación y persistencia del Modelo Oficial.

# Configuración del Modelo Oficial
OFFICIAL_MODEL_CONFIG = {
    "model_code": OFFICIAL_MODEL_CODE,
    "model_name": OFFICIAL_MODEL_NAME,
    "family": OFFICIAL_MODEL_FAMILY,
} # Construir identidad oficial del modelo

# Rutas de entrada del entrenamiento
INPUT_BENCHMARK_EXPERIMENT = BENCHMARK_EXPERIMENT_FILE
INPUT_OFFICIAL_MODEL_CONFIG = OFFICIAL_MODEL_CONFIG_FILE

# Rutas de salida del entrenamiento
OUTPUT_CHECKPOINT = TRAINING_CHECKPOINT_FILE
OUTPUT_METADATA = TRAINING_METADATA_FILE
OUTPUT_SUMMARY = TRAINING_SUMMARY_FILE
OUTPUT_HISTORY = TRAINING_HISTORY_FILE
OUTPUT_METRICS = TRAINING_METRICS_FILE
OUTPUT_MANIFEST = TRAINING_MANIFEST_FILE
OUTPUT_CONTRACT = TRAINING_CONTRACT_FILE
OUTPUT_LOG = TRAINING_LOG_FILE
OUTPUT_CURVES = TRAINING_CURVES_FILE
OUTPUT_MODEL_TORCH = OFFICIAL_MODEL_TORCH_FILE
OUTPUT_MODEL_JOBLIB = OFFICIAL_MODEL_JOBLIB_FILE

# Preparación del directorio oficial de entrenamiento
TRAINING_DIR.mkdir(
    parents=True,
    exist_ok=True
) # Garantizar directorio de entrenamiento

# BLOQUE 3. CARGA DE BENCHMARKDATA
# Objetivo: Recuperar el BenchmarkData generado y validado por el Benchmark Científico.
# Producto: BENCHMARK_DATA disponible para el entrenamiento del Modelo Oficial.

print("\n3. CARGA DE BENCHMARKDATA")

# Validar existencia del experimento científico
if not INPUT_BENCHMARK_EXPERIMENT.exists():
    raise FileNotFoundError(
        "No existe el experimento científico del Benchmark: "
        f"{INPUT_BENCHMARK_EXPERIMENT}"
    ) # Validar disponibilidad del producto de entrada

# Validar tamaño físico del experimento
if INPUT_BENCHMARK_EXPERIMENT.stat().st_size <= 0:
    raise RuntimeError(
        "El archivo del experimento científico del Benchmark está vacío."
    ) # Validar contenido físico del producto

# Recuperar experimento científico completo
benchmark_experiment_loaded = joblib.load(
    INPUT_BENCHMARK_EXPERIMENT
) # Cargar experimento científico persistido

# Validar estructura principal del experimento
if not isinstance(
    benchmark_experiment_loaded,
    dict
):
    raise TypeError(
        "El experimento científico recuperado debe ser un diccionario."
    ) # Validar estructura del producto

# Validar estado científico del experimento
benchmark_status = benchmark_experiment_loaded.get(
    "status"
) # Recuperar estado científico

if benchmark_status != "VALIDATED":
    raise RuntimeError(
        "El experimento científico del Benchmark no presenta estado VALIDATED. "
        f"Estado encontrado: {benchmark_status}"
    ) # Validar estado científico

# Validar presencia de BenchmarkData
if "benchmark_data" not in benchmark_experiment_loaded:
    raise RuntimeError(
        "El experimento científico no contiene la clave 'benchmark_data'."
    ) # Validar disponibilidad de BenchmarkData

# Recuperar BenchmarkData
BENCHMARK_DATA = benchmark_experiment_loaded[
    "benchmark_data"
] # Recuperar producto científico oficial

# Recuperar Modelo Oficial seleccionado por el Benchmark
if "official_model" not in benchmark_experiment_loaded:
    raise RuntimeError(
        "El experimento científico no contiene la clave 'official_model'."
    ) # Validar disponibilidad del Modelo Oficial

official = benchmark_experiment_loaded[
    "official_model"
] # Recuperar Modelo Oficial del Benchmark

if not isinstance(
    official,
    dict
):
    raise TypeError(
        "official debe ser un diccionario."
    ) # Validar estructura del Modelo Oficial

required_official_fields = [
    "model_code",
    "model_name",
    "family",
    "model_config",
    "ranking_position",
] # Definir contrato mínimo del Modelo Oficial

missing_official_fields = [
    field
    for field in required_official_fields
    if field not in official
] # Detectar campos faltantes

if missing_official_fields:
    raise RuntimeError(
        "El Modelo Oficial está incompleto. "
        f"Faltan: {missing_official_fields}"
    ) # Validar contrato del Modelo Oficial

if official[
    "model_code"
] != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El código del Modelo Oficial no coincide con OFFICIAL_MODEL_CODE."
    ) # Validar código oficial

if official[
    "model_name"
].strip().lower() != OFFICIAL_MODEL_NAME.strip().lower():
    raise RuntimeError(
        "El nombre del Modelo Oficial no coincide con OFFICIAL_MODEL_NAME."
    ) # Validar nombre oficial

if official[
    "family"
].strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise RuntimeError(
        "La familia del Modelo Oficial no coincide con OFFICIAL_MODEL_FAMILY."
    ) # Validar familia oficial

official_model_config_export = official[
    "model_config"
].copy() # Recuperar configuración exportada por el Benchmark

# Validar estructura de BenchmarkData
if not isinstance(
    BENCHMARK_DATA,
    dict
):
    raise TypeError(
        "BENCHMARK_DATA debe ser un diccionario."
    ) # Validar estructura de BenchmarkData

# Validar que BenchmarkData no esté vacío
if not BENCHMARK_DATA:
    raise RuntimeError(
        "BENCHMARK_DATA está vacío."
    ) # Validar contenido de BenchmarkData

# Definir contrato mínimo de BenchmarkData
required_benchmark_data_keys = [
    "graphs",
    "partitions",
    "train_index",
    "validation_index",
    "test_index",
    "x_train",
    "y_train",
    "x_validation",
    "y_validation",
    "x_test",
    "y_test",
    "scaler",
] # Definir productos obligatorios de BenchmarkData

# Identificar productos faltantes
missing_benchmark_data_keys = [
    key
    for key in required_benchmark_data_keys
    if key not in BENCHMARK_DATA
] # Detectar elementos ausentes

# Validar contrato de BenchmarkData
if missing_benchmark_data_keys:
    raise RuntimeError(
        "BenchmarkData está incompleto. "
        f"Faltan: {missing_benchmark_data_keys}"
    ) # Validar integridad estructural

# Registrar claves disponibles de BenchmarkData
benchmark_data_keys = sorted(
    BENCHMARK_DATA.keys()
) # Registrar estructura recuperada

# Mostrar resultado de la carga
print(f"Estado Benchmark            : {benchmark_status}") # Mostrar estado científico
print(f"Claves BenchmarkData        : {len(benchmark_data_keys)}") # Mostrar cantidad de productos
print("BenchmarkData               : CARGADO Y VALIDADO") # Confirmar carga del producto

# BLOQUE 4. CARGA Y VALIDACIÓN DE LA CONFIGURACIÓN DEL MODELO OFICIAL
# Objetivo: Recuperar y validar la configuración oficial de GraphSAGE generada por el Benchmark.
# Producto: Configuración oficial validada para construir posteriormente el Modelo Oficial.

print("\n4. CARGA Y VALIDACIÓN DE LA CONFIGURACIÓN DEL MODELO OFICIAL")

# Validar existencia del archivo de configuración oficial
if not OFFICIAL_MODEL_CONFIG_FILE.exists():
    raise FileNotFoundError(
        "No existe la configuración oficial del Modelo Oficial: "
        f"{OFFICIAL_MODEL_CONFIG_FILE}"
    ) # Validar disponibilidad de la configuración oficial

# Validar tamaño físico del archivo
if OFFICIAL_MODEL_CONFIG_FILE.stat().st_size <= 0:
    raise RuntimeError(
        "El archivo de configuración oficial está vacío."
    ) # Validar contenido físico de la configuración

# Cargar configuración oficial
with OFFICIAL_MODEL_CONFIG_FILE.open(
    "r",
    encoding="utf-8"
) as file:
    official_model_config = json.load(
        file
    ) # Recuperar configuración oficial del Benchmark

# Validar estructura principal
if not isinstance(
    official_model_config,
    dict
):
    raise TypeError(
        "official_model_config debe ser un diccionario."
    ) # Validar estructura principal

# Definir campos obligatorios de la configuración oficial
required_model_config_fields = [
    "model_code",
    "model_name",
    "family",
    "hidden_channels",
    "dropout",
    "learning_rate",
    "weight_decay",
    "epochs",
] # Definir contrato mínimo de configuración

# Identificar campos faltantes
missing_model_config_fields = [
    field
    for field in required_model_config_fields
    if field not in official_model_config
] # Detectar campos ausentes

# Validar contrato completo
if missing_model_config_fields:
    raise RuntimeError(
        "La configuración oficial del Modelo Oficial está incompleta. "
        f"Campos faltantes: {missing_model_config_fields}"
    ) # Validar integridad del contrato

# Validar código del Modelo Oficial
if official_model_config[
    "model_code"
] != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El código de la configuración oficial no coincide con OFFICIAL_MODEL_CODE."
    ) # Validar identidad mediante código

# Validar nombre del Modelo Oficial
if official_model_config[
    "model_name"
].strip().lower() != OFFICIAL_MODEL_NAME.strip().lower():
    raise RuntimeError(
        "El nombre de la configuración oficial no coincide con OFFICIAL_MODEL_NAME."
    ) # Validar identidad mediante nombre

# Validar familia del Modelo Oficial
if official_model_config[
    "family"
].strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise RuntimeError(
        "La familia de la configuración oficial no coincide con OFFICIAL_MODEL_FAMILY."
    ) # Validar familia oficial

# Recuperar dimensión oculta
hidden_channels = official_model_config[
    "hidden_channels"
] # Recuperar dimensión interna de GraphSAGE

# Recuperar Dropout
dropout = official_model_config[
    "dropout"
] # Recuperar regularización Dropout

# Recuperar tasa de aprendizaje
learning_rate = official_model_config[
    "learning_rate"
] # Recuperar tasa de aprendizaje

# Recuperar regularización L2
weight_decay = official_model_config[
    "weight_decay"
] # Recuperar regularización del optimizador

# Recuperar número de épocas
epochs = official_model_config[
    "epochs"
] # Recuperar duración oficial del entrenamiento

# Validar dimensión oculta
if not isinstance(
    hidden_channels,
    (int, np.integer)
):
    raise TypeError(
        "hidden_channels debe ser un entero."
    ) # Validar tipo de dimensión oculta

# Normalizar dimensión oculta
hidden_channels = int(
    hidden_channels
) # Garantizar tipo entero

# Validar rango de dimensión oculta
if hidden_channels <= 0:
    raise ValueError(
        "hidden_channels debe ser mayor que cero."
    ) # Validar rango de dimensión oculta

# Validar Dropout
if not isinstance(
    dropout,
    (int, float, np.integer, np.floating)
):
    raise TypeError(
        "dropout debe ser un valor numérico."
    ) # Validar tipo de Dropout

# Normalizar Dropout
dropout = float(
    dropout
) # Garantizar tipo numérico

# Validar rango de Dropout
if not 0.0 <= dropout < 1.0:
    raise ValueError(
        "dropout debe encontrarse en el intervalo [0, 1)."
    ) # Validar rango de Dropout

# Validar Learning Rate
if not isinstance(
    learning_rate,
    (int, float, np.integer, np.floating)
):
    raise TypeError(
        "learning_rate debe ser un valor numérico."
    ) # Validar tipo de Learning Rate

# Normalizar Learning Rate
learning_rate = float(
    learning_rate
) # Garantizar tipo numérico

# Validar rango de Learning Rate
if learning_rate <= 0.0:
    raise ValueError(
        "learning_rate debe ser mayor que cero."
    ) # Validar rango de Learning Rate

# Validar Weight Decay
if not isinstance(
    weight_decay,
    (int, float, np.integer, np.floating)
):
    raise TypeError(
        "weight_decay debe ser un valor numérico."
    ) # Validar tipo de Weight Decay

# Normalizar Weight Decay
weight_decay = float(
    weight_decay
) # Garantizar tipo numérico

# Validar rango de Weight Decay
if weight_decay < 0.0:
    raise ValueError(
        "weight_decay no puede ser negativo."
    ) # Validar rango de Weight Decay

# Validar número de épocas
if not isinstance(
    epochs,
    (int, np.integer)
):
    raise TypeError(
        "epochs debe ser un entero."
    ) # Validar tipo de épocas

# Normalizar número de épocas
epochs = int(
    epochs
) # Garantizar tipo entero

# Validar rango de épocas
if epochs <= 0:
    raise ValueError(
        "epochs debe ser mayor que cero."
    ) # Validar rango de épocas

# Construir identidad oficial validada
official_model_identity = {
    "model_code": official_model_config["model_code"],
    "model_name": official_model_config["model_name"],
    "family": official_model_config["family"],
} # Registrar identidad oficial del modelo

# Construir configuración arquitectónica validada
official_architecture_config = {
    "hidden_channels": hidden_channels,
    "dropout": dropout,
} # Registrar configuración arquitectónica

# Construir configuración de entrenamiento validada
official_training_config = {
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "epochs": epochs,
} # Registrar configuración de entrenamiento

# Mostrar identidad oficial
print(f"Modelo Oficial             : {official_model_config['model_name']}") # Mostrar nombre oficial
print(f"Código Oficial             : {official_model_config['model_code']}") # Mostrar código oficial
print(f"Familia                    : {official_model_config['family']}") # Mostrar familia oficial

# Mostrar configuración arquitectónica
print(f"Dimensión oculta           : {hidden_channels}") # Mostrar dimensión interna
print(f"Dropout                    : {dropout:.6f}") # Mostrar Dropout

# Mostrar configuración de entrenamiento
print(f"Learning Rate              : {learning_rate:.8f}") # Mostrar tasa de aprendizaje
print(f"Weight Decay               : {weight_decay:.8f}") # Mostrar regularización
print(f"Épocas                     : {epochs}") # Mostrar número de épocas

# Confirmar validación
print("Configuración oficial      : VALIDADA") # Confirmar configuración oficial
print("Arquitectura               : VALIDADA") # Confirmar configuración arquitectónica
print("Entrenamiento              : VALIDADO") # Confirmar configuración de entrenamiento

# BLOQUE 5. RECUPERACIÓN DE GRAPHDATA Y PARTICIONES
# Objetivo: Recuperar los productos científicos necesarios para el entrenamiento.

print("\n5. RECUPERACIÓN DE GRAPHDATA Y PARTICIONES")

graphs = BENCHMARK_DATA[
    "graphs"
] # Recuperar colección oficial de GraphData

train_index = BENCHMARK_DATA[
    "train_index"
] # Recuperar partición de entrenamiento

validation_index = BENCHMARK_DATA[
    "validation_index"
] # Recuperar partición de validación

test_index = BENCHMARK_DATA[
    "test_index"
] # Recuperar partición de prueba

scaler = BENCHMARK_DATA[
    "scaler"
] # Recuperar escalador oficial

x_train = BENCHMARK_DATA[
    "x_train"
] # Recuperar variables predictoras de entrenamiento

y_train = BENCHMARK_DATA[
    "y_train"
] # Recuperar variable objetivo de entrenamiento

x_validation = BENCHMARK_DATA[
    "x_validation"
] # Recuperar variables predictoras de validación

y_validation = BENCHMARK_DATA[
    "y_validation"
] # Recuperar variable objetivo de validación

x_test = BENCHMARK_DATA[
    "x_test"
] # Recuperar variables predictoras de prueba

y_test = BENCHMARK_DATA[
    "y_test"
] # Recuperar variable objetivo de prueba

n_graphs = len(
    graphs
) # Determinar cantidad total de GraphData

if n_graphs == 0:
    raise RuntimeError(
        "No existen GraphData disponibles para el entrenamiento."
    ) # Validar colección GraphData

print(f"GraphData recuperados       : {n_graphs}") # Mostrar cantidad de grafos
print(f"GraphData entrenamiento     : {len(train_index)}") # Mostrar entrenamiento
print(f"GraphData validación        : {len(validation_index)}") # Mostrar validación
print(f"GraphData prueba            : {len(test_index)}") # Mostrar prueba
print(f"Scaler oficial              : {type(scaler).__name__}") # Mostrar escalador
print("GraphData y particiones     : RECUPERADOS") # Confirmar recuperación

# BLOQUE 6. RECUPERACIÓN DE GRAPHDATA Y PARTICIONES
# Objetivo: Recuperar desde BENCHMARK_DATA los GraphData, particiones y productos
# experimentales que serán utilizados durante el entrenamiento del Modelo Oficial.
# Producto: Objetos de entrenamiento disponibles para los bloques posteriores.

print("\n6. RECUPERACIÓN DE GRAPHDATA Y PARTICIONES")

# 6.1 RECUPERACIÓN DE GRAPHDATA
# Recuperar colección oficial de GraphData
graphs = BENCHMARK_DATA[
    "graphs"
] # Recuperar GraphData desde BenchmarkData

# Registrar número total de GraphData
n_graphs = len(
    graphs
) # Registrar cantidad de GraphData

# 6.2 RECUPERACIÓN DE PARTICIONES
# Recuperar estructura agrupada de particiones
partitions = BENCHMARK_DATA[
    "partitions"
] # Recuperar particiones oficiales

# Recuperar índice de entrenamiento
train_index = BENCHMARK_DATA[
    "train_index"
] # Recuperar partición de entrenamiento

# Recuperar índice de validación
validation_index = BENCHMARK_DATA[
    "validation_index"
] # Recuperar partición de validación

# Recuperar índice de prueba
test_index = BENCHMARK_DATA[
    "test_index"
] # Recuperar partición de prueba

# 6.3 RECUPERACIÓN DEL ESCALADOR
# Recuperar escalador oficial
scaler = BENCHMARK_DATA[
    "scaler"
] # Recuperar escalador generado por el Benchmark

# 6.4 RECUPERACIÓN DE MATRICES EXPERIMENTALES
# Recuperar variables predictoras de entrenamiento
x_train = BENCHMARK_DATA[
    "x_train"
] # Recuperar matriz de entrenamiento

# Recuperar variable objetivo de entrenamiento
y_train = BENCHMARK_DATA[
    "y_train"
] # Recuperar objetivo de entrenamiento

# Recuperar variables predictoras de validación
x_validation = BENCHMARK_DATA[
    "x_validation"
] # Recuperar matriz de validación

# Recuperar variable objetivo de validación
y_validation = BENCHMARK_DATA[
    "y_validation"
] # Recuperar objetivo de validación

# Recuperar variables predictoras de prueba
x_test = BENCHMARK_DATA[
    "x_test"
] # Recuperar matriz de prueba

# Recuperar variable objetivo de prueba
y_test = BENCHMARK_DATA[
    "y_test"
] # Recuperar objetivo de prueba

# 6.5 DETERMINACIÓN DE DIMENSIONES DEL MODELO
# Recuperar primer GraphData
first_graph = graphs[
    0
] # Recuperar GraphData de referencia

# Determinar dimensión de entrada
in_channels = int(
    first_graph.x.shape[1]
) # Obtener número real de variables predictoras

# Definir dimensión de salida
out_channels = 1 # Definir una salida para la variable objetivo

# Construir contrato dimensional de GraphSAGE
graphsage_dimensions = {
    "in_channels": in_channels,
    "hidden_channels": hidden_channels,
    "out_channels": out_channels,
} # Registrar dimensiones oficiales del modelo

# 6.6 RESUMEN DE RECUPERACIÓN
print(f"GraphData recuperados      : {n_graphs}") # Mostrar cantidad de GraphData
print(f"GraphData entrenamiento    : {len(train_index)}") # Mostrar entrenamiento
print(f"GraphData validación       : {len(validation_index)}") # Mostrar validación
print(f"GraphData prueba           : {len(test_index)}") # Mostrar prueba
print(f"Variables de entrada       : {in_channels}") # Mostrar variables predictoras
print(f"Variables de salida        : {out_channels}") # Mostrar variable objetivo
print(f"Scaler oficial             : {type(scaler).__name__}") # Mostrar tipo de scaler
print("GraphData y particiones     : RECUPERADOS") # Confirmar recuperación
print("Dimensiones GraphSAGE       : DETERMINADAS") # Confirmar dimensiones

# BLOQUE 7. DETERMINACIÓN DE DIMENSIONES
# Objetivo: Determinar y validar las dimensiones de entrada, ocultas y salida
# requeridas por el Modelo Oficial GraphSAGE.
# Producto: Contrato dimensional validado para la construcción del modelo.

print("\n7. DETERMINACIÓN DE DIMENSIONES")

# 7.1 VALIDACIÓN DEL GRAPHDATA DE REFERENCIA
# Validar disponibilidad de GraphData
if not isinstance(
    graphs,
    list
) or len(graphs) == 0:
    raise RuntimeError(
        "No existen GraphData disponibles para determinar las dimensiones del modelo."
    ) # Validar disponibilidad de GraphData

# Recuperar GraphData de referencia
reference_graph = graphs[
    0
] # Seleccionar primer GraphData como referencia dimensional

# Validar presencia de Node Features
if not hasattr(
    reference_graph,
    "x"
):
    raise RuntimeError(
        "El GraphData de referencia no contiene la matriz de características 'x'."
    ) # Validar Node Features

# Validar dimensionalidad de Node Features
if reference_graph.x.ndim != 2:
    raise ValueError(
        "La matriz GraphData.x debe tener dos dimensiones."
    ) # Validar estructura de entrada

# 7.2 DETERMINACIÓN DE LA DIMENSIÓN DE ENTRADA
# Obtener número de nodos
n_nodes = int(
    reference_graph.x.shape[0]
) # Determinar número de nodos del GraphData

# Obtener número de variables predictoras
in_channels = int(
    reference_graph.x.shape[1]
) # Determinar dimensión real de entrada

# Validar dimensión de entrada
if in_channels <= 0:
    raise ValueError(
        "in_channels debe ser mayor que cero."
    ) # Validar dimensión de entrada

# 7.3 VALIDACIÓN DE CONSISTENCIA ENTRE GRAPHDATA
# Validar que todos los GraphData tengan la misma dimensión de entrada
for graph_index, graph in enumerate(
    graphs
):
    # Validar presencia de Node Features
    if not hasattr(
        graph,
        "x"
    ):
        raise RuntimeError(
            f"GraphData {graph_index} no contiene 'x'."
        ) # Validar Node Features

    # Validar dimensionalidad
    if graph.x.ndim != 2:
        raise ValueError(
            f"GraphData {graph_index}: x debe ser bidimensional."
        ) # Validar dimensionalidad

    # Validar número de variables predictoras
    graph_in_channels = int(
        graph.x.shape[1]
    ) # Obtener dimensión del GraphData

    if graph_in_channels != in_channels:
        raise ValueError(
            f"GraphData {graph_index} presenta {graph_in_channels} variables "
            f"predictoras y se esperaban {in_channels}."
        ) # Validar consistencia dimensional

# 7.4 DETERMINACIÓN DE LA DIMENSIÓN OCULTA
# Recuperar dimensión oculta desde la configuración oficial
hidden_channels = int(
    official_model_config[
        "hidden_channels"
    ]
) # Determinar dimensión interna de GraphSAGE

# Validar dimensión oculta
if hidden_channels <= 0:
    raise ValueError(
        "hidden_channels debe ser mayor que cero."
    ) # Validar dimensión oculta

# 7.5 DETERMINACIÓN DE LA DIMENSIÓN DE SALIDA
# Definir una salida para la variable objetivo
out_channels = 1 # Determinar dimensión de salida

# Validar dimensión de salida
if out_channels != 1:
    raise ValueError(
        "out_channels debe ser igual a 1 para el Modelo Oficial."
    ) # Validar dimensión de salida

# 7.6 CONSTRUIR CONTRATO DIMENSIONAL
# Registrar dimensiones oficiales de GraphSAGE
graphsage_dimensions = {
    "in_channels": in_channels,
    "hidden_channels": hidden_channels,
    "out_channels": out_channels,
} # Construir contrato dimensional

# Validar contrato dimensional
if not all(
    isinstance(
        value,
        int
    ) and value > 0
    for value in graphsage_dimensions.values()
):
    raise RuntimeError(
        "El contrato dimensional de GraphSAGE contiene dimensiones inválidas."
    ) # Validar contrato dimensional

# 7.7 RESUMEN DE DIMENSIONES
print(f"Nodos de referencia          : {n_nodes}") # Mostrar número de nodos
print(f"Variables de entrada        : {in_channels}") # Mostrar dimensión de entrada
print(f"Dimensión oculta             : {hidden_channels}") # Mostrar dimensión oculta
print(f"Dimensión de salida          : {out_channels}") # Mostrar dimensión de salida
print("Consistencia GraphData       : VALIDADA") # Confirmar consistencia dimensional
print("Contrato dimensional         : VALIDADO") # Confirmar contrato del modelo

# BLOQUE 8. CONSTRUCCIÓN DE OFFICIALGRAPHSAGE
# Objetivo: Construir el Modelo Oficial GraphSAGE utilizando el contrato dimensional
# y la configuración oficial previamente validados.
# Producto: official_graphsage listo para configurar el entrenamiento.

print("\n8. CONSTRUCCIÓN DE OFFICIALGRAPHSAGE")

# 8.1 DEFINICIÓN DE LA ARQUITECTURA GRAPHsage
class OfficialGraphSAGE(
    torch.nn.Module
):
    """Modelo Oficial GraphSAGE del proyecto."""
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout,
    ):
        super().__init__()

        # Primera capa de convolución GraphSAGE
        self.sage1 = SAGEConv(
            in_channels,
            hidden_channels,
        ) # Construir primera capa SAGEConv

        # Segunda capa de convolución GraphSAGE
        self.sage2 = SAGEConv(
            hidden_channels,
            hidden_channels,
        ) # Construir segunda capa SAGEConv

        # Capa de regularización Dropout
        self.dropout = torch.nn.Dropout(
            p=dropout,
        ) # Construir regularización Dropout

        # Capa lineal de salida
        self.output_layer = torch.nn.Linear(
            hidden_channels,
            out_channels,
        ) # Construir capa de salida


    def forward(
        self,
        x,
        edge_index,
    ):
        """Ejecutar propagación hacia adelante del Modelo Oficial."""
        # Primera propagación GraphSAGE
        x = self.sage1(
            x,
            edge_index,
        ) # Aplicar primera capa GraphSAGE

        # Primera activación no lineal
        x = F.relu(
            x,
        ) # Aplicar función de activación ReLU

        # Primera regularización
        x = self.dropout(
            x,
        ) # Aplicar Dropout

        # Segunda propagación GraphSAGE
        x = self.sage2(
            x,
            edge_index,
        ) # Aplicar segunda capa GraphSAGE

        # Segunda activación no lineal
        x = F.relu(
            x,
        ) # Aplicar función de activación ReLU

        # Segunda regularización
        x = self.dropout(
            x,
        ) # Aplicar Dropout

        # Proyección hacia la variable objetivo
        x = self.output_layer(
            x,
        ) # Generar predicción final

        # Retornar predicción
        return x

# 8.2 CONSTRUIR MODELO OFICIAL
official_graphsage = OfficialGraphSAGE(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    dropout=dropout,
) # Construir instancia oficial de GraphSAGE

# 8.3 VALIDAR TIPO DEL MODELO
if not isinstance(
    official_graphsage,
    torch.nn.Module,
):
    raise TypeError(
        "official_graphsage debe ser una instancia de torch.nn.Module."
    ) # Validar tipo del modelo

# 8.4 VALIDAR ARQUITECTURA DEL MODELO
if not hasattr(
    official_graphsage,
    "sage1",
):
    raise RuntimeError(
        "OfficialGraphSAGE no contiene la primera capa SAGEConv."
    ) # Validar primera capa

if not hasattr(
    official_graphsage,
    "sage2",
):
    raise RuntimeError(
        "OfficialGraphSAGE no contiene la segunda capa SAGEConv."
    ) # Validar segunda capa

if not hasattr(
    official_graphsage,
    "dropout",
):
    raise RuntimeError(
        "OfficialGraphSAGE no contiene la capa Dropout."
    ) # Validar regularización

if not hasattr(
    official_graphsage,
    "output_layer",
):
    raise RuntimeError(
        "OfficialGraphSAGE no contiene la capa de salida."
    ) # Validar capa de salida

# 8.5 VALIDAR DIMENSIONES DE LAS CAPAS
if official_graphsage.sage1.in_channels != in_channels:
    raise RuntimeError(
        "La dimensión de entrada de la primera SAGEConv "
        "no coincide con in_channels."
    ) # Validar dimensión de entrada

if official_graphsage.sage1.out_channels != hidden_channels:
    raise RuntimeError(
        "La dimensión de salida de la primera SAGEConv "
        "no coincide con hidden_channels."
    ) # Validar primera transformación

if official_graphsage.sage2.in_channels != hidden_channels:
    raise RuntimeError(
        "La dimensión de entrada de la segunda SAGEConv "
        "no coincide con hidden_channels."
    ) # Validar segunda entrada

if official_graphsage.sage2.out_channels != hidden_channels:
    raise RuntimeError(
        "La dimensión de salida de la segunda SAGEConv "
        "no coincide con hidden_channels."
    ) # Validar segunda transformación

if official_graphsage.output_layer.in_features != hidden_channels:
    raise RuntimeError(
        "La dimensión de entrada de la capa de salida "
        "no coincide con hidden_channels."
    ) # Validar entrada de salida

if official_graphsage.output_layer.out_features != out_channels:
    raise RuntimeError(
        "La dimensión de salida de la capa lineal "
        "no coincide con out_channels."
    ) # Validar dimensión final

# 8.6 VALIDAR CONFIGURACIÓN DE DROPOUT
if not np.isclose(
    official_graphsage.dropout.p,
    dropout,
):
    raise RuntimeError(
        "La tasa Dropout del modelo no coincide con "
        "la configuración oficial."
    ) # Validar Dropout oficial

# 8.7 CONTAR PARÁMETROS ENTRENABLES
trainable_parameters = sum(
    parameter.numel()
    for parameter in official_graphsage.parameters()
    if parameter.requires_grad
) # Calcular número de parámetros entrenables

if trainable_parameters <= 0:
    raise RuntimeError(
        "OfficialGraphSAGE no contiene parámetros entrenables."
    ) # Validar parámetros del modelo

# 8.8 REGISTRAR CONTRATO DEL MODELO
official_model_contract = {
    "model_code": official_model_identity["model_code"],
    "model_name": official_model_identity["model_name"],
    "family": official_model_identity["family"],
    "in_channels": int(in_channels),
    "hidden_channels": int(hidden_channels),
    "out_channels": int(out_channels),
    "dropout": float(dropout),
    "trainable_parameters": int(trainable_parameters),
} # Registrar contrato científico del Modelo Oficial

# 8.9 RESUMEN DEL MODELO
print(f"Modelo                     : {official_model_contract['model_name']}") # Mostrar modelo oficial
print(f"Código                     : {official_model_contract['model_code']}") # Mostrar código del modelo
print(f"Familia                    : {official_model_contract['family']}") # Mostrar familia
print(f"Entrada                    : {official_model_contract['in_channels']}") # Mostrar dimensión de entrada
print(f"Oculta                     : {official_model_contract['hidden_channels']}") # Mostrar dimensión oculta
print(f"Salida                     : {official_model_contract['out_channels']}") # Mostrar dimensión de salida
print(f"Dropout                    : {official_model_contract['dropout']:.6f}") # Mostrar Dropout
print(f"Parámetros entrenables     : {official_model_contract['trainable_parameters']:,}") # Mostrar parámetros
print("Arquitectura GraphSAGE      : VALIDADA") # Confirmar arquitectura
print("OfficialGraphSAGE           : CONSTRUIDO") # Confirmar construcción

# BLOQUE 9. VALIDACIÓN ESTRUCTURAL DEL MODELO
# Objetivo: Validar que OfficialGraphSAGE cumple la arquitectura, dimensiones
# y configuración oficial definidas para el Modelo Oficial.
# Producto: Modelo estructuralmente validado antes del entrenamiento.

print("\n9. VALIDACIÓN ESTRUCTURAL DEL MODELO")

# 9.1 VALIDACIÓN DE EXISTENCIA DEL MODELO
# Validar existencia de OfficialGraphSAGE
if official_graphsage is None:
    raise RuntimeError(
        "OfficialGraphSAGE no ha sido construido."
    ) # Validar disponibilidad del modelo

# Validar tipo del modelo
if not isinstance(
    official_graphsage,
    torch.nn.Module
):
    raise TypeError(
        "OfficialGraphSAGE debe ser una instancia de torch.nn.Module."
    ) # Validar tipo del modelo

# 9.2 VALIDACIÓN DE CAPAS PRINCIPALES
# Validar primera capa GraphSAGE
if not isinstance(
    official_graphsage.sage1,
    SAGEConv
):
    raise TypeError(
        "La primera capa del modelo debe ser SAGEConv."
    ) # Validar primera capa

# Validar segunda capa GraphSAGE
if not isinstance(
    official_graphsage.sage2,
    SAGEConv
):
    raise TypeError(
        "La segunda capa del modelo debe ser SAGEConv."
    ) # Validar segunda capa

# Validar capa Dropout
if not isinstance(
    official_graphsage.dropout,
    torch.nn.Dropout
):
    raise TypeError(
        "La capa de regularización debe ser torch.nn.Dropout."
    ) # Validar capa Dropout

# Validar capa lineal de salida
if not isinstance(
    official_graphsage.output_layer,
    torch.nn.Linear
):
    raise TypeError(
        "La capa de salida debe ser torch.nn.Linear."
    ) # Validar capa de salida

# 9.3 VALIDACIÓN DE DIMENSIONES DE ENTRADA
# Validar dimensión de entrada de la primera SAGEConv
if official_graphsage.sage1.in_channels != in_channels:
    raise RuntimeError(
        "La dimensión de entrada de sage1 no coincide con in_channels."
    ) # Validar dimensión de entrada

# Validar dimensión de salida de la primera SAGEConv
if official_graphsage.sage1.out_channels != hidden_channels:
    raise RuntimeError(
        "La dimensión de salida de sage1 no coincide con hidden_channels."
    ) # Validar primera transformación

# 9.4 VALIDACIÓN DE DIMENSIONES DE LA SEGUNDA CAPA
# Validar dimensión de entrada de la segunda SAGEConv
if official_graphsage.sage2.in_channels != hidden_channels:
    raise RuntimeError(
        "La dimensión de entrada de sage2 no coincide con hidden_channels."
    ) # Validar segunda entrada

# Validar dimensión de salida de la segunda SAGEConv
if official_graphsage.sage2.out_channels != hidden_channels:
    raise RuntimeError(
        "La dimensión de salida de sage2 no coincide con hidden_channels."
    ) # Validar segunda transformación

# 9.5 VALIDACIÓN DE LA CAPA DE SALIDA
# Validar dimensión de entrada de la capa lineal
if official_graphsage.output_layer.in_features != hidden_channels:
    raise RuntimeError(
        "La dimensión de entrada de output_layer no coincide con hidden_channels."
    ) # Validar entrada de la capa de salida

# Validar dimensión de salida
if official_graphsage.output_layer.out_features != out_channels:
    raise RuntimeError(
        "La dimensión de salida de output_layer no coincide con out_channels."
    ) # Validar salida del modelo

# 9.6 VALIDACIÓN DE DROPOUT
# Validar tasa Dropout
if not np.isclose(
    official_graphsage.dropout.p,
    dropout,
):
    raise RuntimeError(
        "La tasa Dropout del modelo no coincide con la configuración oficial."
    ) # Validar hiperparámetro Dropout

# 9.7 VALIDACIÓN DE PARÁMETROS ENTRENABLES
# Recuperar parámetros entrenables
trainable_parameters_list = [
    parameter
    for parameter in official_graphsage.parameters()
    if parameter.requires_grad
] # Recuperar parámetros entrenables

# Validar existencia de parámetros entrenables
if not trainable_parameters_list:
    raise RuntimeError(
        "OfficialGraphSAGE no contiene parámetros entrenables."
    ) # Validar parámetros

# Calcular cantidad total de parámetros entrenables
trainable_parameters = sum(
    parameter.numel()
    for parameter in trainable_parameters_list
) # Calcular parámetros entrenables

# Validar cantidad de parámetros
if trainable_parameters <= 0:
    raise RuntimeError(
        "El número de parámetros entrenables debe ser mayor que cero."
    ) # Validar cantidad de parámetros

# 9.8 VALIDACIÓN DE ESTADO DE LOS PARÁMETROS
# Validar que todos los parámetros sean tensores finitos
for parameter_name, parameter in official_graphsage.named_parameters():
    # Validar tipo tensorial
    if not isinstance(
        parameter,
        torch.Tensor
    ):
        raise TypeError(
            f"El parámetro '{parameter_name}' no es un tensor de PyTorch."
        ) # Validar tipo de parámetro

    # Validar valores iniciales
    if not torch.isfinite(
        parameter
    ).all():
        raise RuntimeError(
            f"El parámetro '{parameter_name}' contiene NaN o infinitos."
        ) # Validar integridad numérica

# 9.9 VALIDACIÓN DEL ESTADO DEL MODELO
# Establecer modo entrenamiento para validar el comportamiento esperado
official_graphsage.train() # Establecer modo entrenamiento

# Validar modo entrenamiento
if not official_graphsage.training:
    raise RuntimeError(
        "OfficialGraphSAGE no pudo establecerse en modo entrenamiento."
    ) # Validar estado del modelo

# 9.10 VALIDACIÓN DEL CONTRATO DEL MODELO
# Actualizar contrato estructural
official_model_contract.update(
    {
        "validated": True,
        "architecture": "GraphSAGE",
        "layers": 2,
        "output_layer": "Linear",
        "trainable_parameters": int(trainable_parameters),
    }
) # Registrar validación estructural

# 9.11 RESUMEN DE VALIDACIÓN
print(f"Modelo                     : {official_model_contract['model_name']}") # Mostrar modelo
print(f"Primera capa               : SAGEConv {in_channels}->{hidden_channels}") # Mostrar primera transformación
print(f"Segunda capa               : SAGEConv {hidden_channels}->{hidden_channels}") # Mostrar segunda transformación
print(f"Capa de salida             : Linear {hidden_channels}->{out_channels}") # Mostrar salida
print(f"Dropout                    : {official_graphsage.dropout.p:.6f}") # Mostrar Dropout
print(f"Parámetros entrenables     : {trainable_parameters:,}") # Mostrar parámetros
print("Integridad de parámetros    : VALIDADA") # Confirmar integridad numérica
print("Arquitectura estructural    : VALIDADA") # Confirmar arquitectura
print("Estado del modelo           : TRAIN") # Confirmar modo de entrenamiento
print("OfficialGraphSAGE           : ESTRUCTURALMENTE VALIDADO") # Confirmar validación final

# BLOQUE 10. CONFIGURACIÓN DE OPTIMIZER Y CRITERION
# Objetivo: Configurar la función de pérdida y el optimizador utilizando
# los hiperparámetros oficiales definidos por el Benchmark.
# Producto: criterion y optimizer listos para el entrenamiento de GraphSAGE.

print("\n10. CONFIGURACIÓN DE OPTIMIZER Y CRITERION")

# 10.1 VALIDACIÓN DEL MODELO
# Validar existencia del Modelo Oficial
if official_graphsage is None:
    raise RuntimeError(
        "OfficialGraphSAGE no está disponible."
    ) # Validar disponibilidad del modelo

# Validar que el modelo contenga parámetros entrenables
model_parameters = [
    parameter
    for parameter in official_graphsage.parameters()
    if parameter.requires_grad
] # Recuperar parámetros entrenables

if len(model_parameters) == 0:
    raise RuntimeError(
        "OfficialGraphSAGE no contiene parámetros entrenables."
    ) # Validar parámetros del modelo

# 10.2 CONFIGURACIÓN DE LA FUNCIÓN DE PÉRDIDA
# Construir función de pérdida oficial
criterion = torch.nn.MSELoss() # Configurar error cuadrático medio

# Validar función de pérdida
if not isinstance(
    criterion,
    torch.nn.MSELoss
):
    raise TypeError(
        "criterion debe ser una instancia de torch.nn.MSELoss."
    ) # Validar función de pérdida

# 10.3 CONFIGURACIÓN DEL OPTIMIZADOR
# Construir optimizador Adam
optimizer = torch.optim.Adam(
    official_graphsage.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
) # Configurar optimizador Adam con hiperparámetros oficiales

# Validar optimizador
if not isinstance(
    optimizer,
    torch.optim.Adam
):
    raise TypeError(
        "optimizer debe ser una instancia de torch.optim.Adam."
    ) # Validar optimizador

# 10.4 VALIDAR LEARNING RATE
# Recuperar tasa de aprendizaje configurada
optimizer_learning_rate = float(
    optimizer.param_groups[0]["lr"]
) # Recuperar Learning Rate efectivo

# Validar Learning Rate
if not np.isclose(
    optimizer_learning_rate,
    learning_rate,
):
    raise RuntimeError(
        "El Learning Rate del optimizador no coincide con "
        "la configuración oficial."
    ) # Validar Learning Rate

# 10.5 VALIDAR WEIGHT DECAY
# Recuperar regularización efectiva
optimizer_weight_decay = float(
    optimizer.param_groups[0]["weight_decay"]
) # Recuperar Weight Decay efectivo

# Validar Weight Decay
if not np.isclose(
    optimizer_weight_decay,
    weight_decay,
):
    raise RuntimeError(
        "El Weight Decay del optimizador no coincide con "
        "la configuración oficial."
    ) # Validar Weight Decay

# 10.6 REGISTRAR CONTRATO DE ENTRENAMIENTO
# Construir contrato del optimizador
training_optimization_contract = {
    "optimizer": "Adam",
    "criterion": "MSELoss",
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "epochs": epochs,
} # Registrar configuración oficial del entrenamiento

# 10.7 RESUMEN DE CONFIGURACIÓN
print(f"Optimizer                  : {training_optimization_contract['optimizer']}") # Mostrar optimizador
print(f"Criterion                  : {training_optimization_contract['criterion']}") # Mostrar función de pérdida
print(f"Learning Rate              : {training_optimization_contract['learning_rate']:.8f}") # Mostrar Learning Rate
print(f"Weight Decay               : {training_optimization_contract['weight_decay']:.8f}") # Mostrar regularización
print(f"Épocas                     : {training_optimization_contract['epochs']}") # Mostrar épocas
print("Optimizer                   : VALIDADO") # Confirmar optimizador
print("Criterion                   : VALIDADO") # Confirmar función de pérdida
print("Configuración entrenamiento : VALIDADA") # Confirmar configuración

# BLOQUE 11. PREPARACIÓN DEL ENTRENAMIENTO
# Objetivo: Preparar el entorno de ejecución para el entrenamiento oficial de GraphSAGE.
# Producto: Modelo, dispositivo, índices e historial preparados para iniciar el entrenamiento.

print("\n11. PREPARACIÓN DEL ENTRENAMIENTO")

# 11.1 VALIDACIÓN DEL MODELO
# Validar disponibilidad del Modelo Oficial
if not isinstance(
    official_graphsage,
    torch.nn.Module
):
    raise TypeError(
        "official_graphsage debe ser una instancia de torch.nn.Module."
    ) # Validar modelo

# Validar existencia de parámetros entrenables
trainable_parameters = [
    parameter
    for parameter in official_graphsage.parameters()
    if parameter.requires_grad
] # Recuperar parámetros entrenables

if len(trainable_parameters) == 0:
    raise RuntimeError(
        "OfficialGraphSAGE no contiene parámetros entrenables."
    ) # Validar parámetros

# 11.2 NORMALIZACIÓN DE LOS ÍNDICES DE PARTICIÓN
# Normalizar índices de entrenamiento
train_indices = [
    int(index)
    for index in train_index
] # Preparar índices de entrenamiento

# Normalizar índices de validación
validation_indices = [
    int(index)
    for index in validation_index
] # Preparar índices de validación

# Normalizar índices de prueba
test_indices = [
    int(index)
    for index in test_index
] # Preparar índices de prueba

# 11.3 VALIDACIÓN DE LAS PARTICIONES
# Validar partición de entrenamiento
if len(train_indices) == 0:
    raise ValueError(
        "train_indices está vacío."
    ) # Validar entrenamiento

# Validar partición de validación
if len(validation_indices) == 0:
    raise ValueError(
        "validation_indices está vacío."
    ) # Validar validación

# Validar partición de prueba
if len(test_indices) == 0:
    raise ValueError(
        "test_indices está vacío."
    ) # Validar prueba

# Validar independencia entre entrenamiento y validación
if set(train_indices) & set(validation_indices):
    raise RuntimeError(
        "Existe solapamiento entre entrenamiento y validación."
    ) # Validar independencia

# Validar independencia entre entrenamiento y prueba
if set(train_indices) & set(test_indices):
    raise RuntimeError(
        "Existe solapamiento entre entrenamiento y prueba."
    ) # Validar independencia

# Validar independencia entre validación y prueba
if set(validation_indices) & set(test_indices):
    raise RuntimeError(
        "Existe solapamiento entre validación y prueba."
    ) # Validar independencia

# 11.4 VALIDACIÓN DEL RANGO DE LOS ÍNDICES
# Definir número total de GraphData
n_graphs = len(
    graphs
) # Registrar cantidad total de grafos

# Validar índices de entrenamiento
if any(
    index < 0 or index >= n_graphs
    for index in train_indices
):
    raise IndexError(
        "train_indices contiene índices fuera del rango de GraphData."
    ) # Validar rango de entrenamiento

# Validar índices de validación
if any(
    index < 0 or index >= n_graphs
    for index in validation_indices
):
    raise IndexError(
        "validation_indices contiene índices fuera del rango de GraphData."
    ) # Validar rango de validación

# Validar índices de prueba
if any(
    index < 0 or index >= n_graphs
    for index in test_indices
):
    raise IndexError(
        "test_indices contiene índices fuera del rango de GraphData."
    ) # Validar rango de prueba

# 11.5 DETERMINACIÓN DEL DISPOSITIVO
# Seleccionar GPU cuando esté disponible
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
) # Determinar dispositivo de ejecución

# Transferir Modelo Oficial al dispositivo
official_graphsage = official_graphsage.to(
    device
) # Transferir modelo al dispositivo

# 11.6 VALIDACIÓN DEL DISPOSITIVO DEL MODELO
# Recuperar dispositivo efectivo del modelo
model_device = next(
    official_graphsage.parameters()
).device # Obtener dispositivo efectivo

# Validar correspondencia entre dispositivo solicitado y modelo
if model_device != device:
    raise RuntimeError(
        "El dispositivo efectivo del modelo no coincide con el dispositivo de entrenamiento."
    ) # Validar dispositivo

# 11.7 PREPARACIÓN DEL HISTORIAL DE ENTRENAMIENTO
# Inicializar historial científico
training_history = {
    "epoch": [],
    "train_loss": [],
    "validation_loss": [],
} # Inicializar historial

# 11.8 INICIALIZACIÓN DEL MEJOR RESULTADO
# Inicializar mejor pérdida de validación
best_validation_loss = float(
    "inf"
) # Inicializar criterio de selección

# Inicializar mejor época
best_epoch = None # Preparar registro de mejor época

# Inicializar mejor estado del modelo
best_model_state = None # Preparar almacenamiento del mejor modelo

# 11.9 REGISTRO DEL ESTADO INICIAL
# Registrar contrato del entorno de entrenamiento
training_environment = {
    "device": str(device),
    "n_graphs": int(n_graphs),
    "train_graphs": int(len(train_indices)),
    "validation_graphs": int(len(validation_indices)),
    "test_graphs": int(len(test_indices)),
    "epochs": int(epochs),
    "learning_rate": float(learning_rate),
    "weight_decay": float(weight_decay),
    "criterion": criterion.__class__.__name__,
    "optimizer": optimizer.__class__.__name__,
} # Registrar entorno oficial de entrenamiento

# 11.10 RESUMEN DE PREPARACIÓN
print(f"Modelo Oficial             : {OFFICIAL_MODEL_NAME}") # Mostrar modelo
print(f"Código Oficial             : {OFFICIAL_MODEL_CODE}") # Mostrar código
print(f"GraphData totales          : {n_graphs}") # Mostrar cantidad de grafos
print(f"Entrenamiento              : {len(train_indices)} GraphData") # Mostrar entrenamiento
print(f"Validación                 : {len(validation_indices)} GraphData") # Mostrar validación
print(f"Prueba                     : {len(test_indices)} GraphData") # Mostrar prueba
print(f"Learning Rate              : {learning_rate:.8f}") # Mostrar Learning Rate
print(f"Weight Decay               : {weight_decay:.8f}") # Mostrar Weight Decay
print(f"Épocas                     : {epochs}") # Mostrar épocas
print(f"Criterion                  : {criterion.__class__.__name__}") # Mostrar función de pérdida
print(f"Optimizer                  : {optimizer.__class__.__name__}") # Mostrar optimizador
print(f"Dispositivo                : {device}") # Mostrar dispositivo
print("Separación de particiones   : VALIDADA") # Confirmar independencia
print("Rango de índices            : VALIDADO") # Confirmar índices
print("Entorno de entrenamiento    : PREPARADO") # Confirmar preparación

# BLOQUE 12. VALIDACIÓN FINAL DEL ENTORNO DE ENTRENAMIENTO
# Objetivo: Ejecutar una validación final de todos los componentes antes de iniciar el entrenamiento.
# Producto: Entorno de entrenamiento completamente validado y autorizado para ejecutar GraphSAGE.

print("\n12. VALIDACIÓN FINAL DEL ENTORNO DE ENTRENAMIENTO")

# 12.1 VALIDAR MODELO OFICIAL
if not isinstance(
    official_graphsage,
    torch.nn.Module
):
    raise TypeError(
        "official_graphsage debe ser una instancia de torch.nn.Module."
    ) # Validar modelo

# Validar parámetros entrenables
trainable_parameters = [
    parameter
    for parameter in official_graphsage.parameters()
    if parameter.requires_grad
] # Recuperar parámetros entrenables

if len(trainable_parameters) == 0:
    raise RuntimeError(
        "El Modelo Oficial no contiene parámetros entrenables."
    ) # Validar capacidad de aprendizaje

# 12.2 VALIDAR OPTIMIZADOR
if not isinstance(
    optimizer,
    torch.optim.Optimizer
):
    raise TypeError(
        "optimizer debe ser una instancia de torch.optim.Optimizer."
    ) # Validar optimizador

# Validar parámetros registrados en el optimizador
optimizer_parameters = [
    parameter
    for parameter_group in optimizer.param_groups
    for parameter in parameter_group["params"]
] # Recuperar parámetros del optimizador

if len(optimizer_parameters) == 0:
    raise RuntimeError(
        "El optimizador no contiene parámetros."
    ) # Validar parámetros del optimizador

# 12.3 VALIDAR FUNCIÓN DE PÉRDIDA
if not isinstance(
    criterion,
    torch.nn.modules.loss._Loss
):
    raise TypeError(
        "criterion debe ser una función de pérdida de PyTorch."
    ) # Validar función de pérdida

# 12.4 VALIDAR COINCIDENCIA ENTRE MODELO Y OPTIMIZADOR
model_parameter_ids = {
    id(parameter)
    for parameter in official_graphsage.parameters()
    if parameter.requires_grad
} # Identificar parámetros entrenables del modelo

optimizer_parameter_ids = {
    id(parameter)
    for parameter in optimizer_parameters
} # Identificar parámetros registrados en optimizer

if not model_parameter_ids.issubset(
    optimizer_parameter_ids
):
    raise RuntimeError(
        "El optimizador no contiene todos los parámetros entrenables "
        "del Modelo Oficial."
    ) # Validar conexión modelo optimizador

# 12.5 VALIDAR GRAPH DATA DE ENTRENAMIENTO
for graph_index in train_indices:
    graph = graphs[
        graph_index
    ] # Recuperar GraphData de entrenamiento

    if not isinstance(
        graph,
        Data
    ):
        raise TypeError(
            f"El GraphData {graph_index} no es una instancia de Data."
        ) # Validar tipo GraphData

    if graph.x.ndim != 2:
        raise ValueError(
            f"El GraphData {graph_index}.x debe ser bidimensional."
        ) # Validar características

    if graph.edge_index.ndim != 2:
        raise ValueError(
            f"El GraphData {graph_index}.edge_index debe ser bidimensional."
        ) # Validar topología

    if graph.edge_index.shape[0] != 2:
        raise ValueError(
            f"El GraphData {graph_index}.edge_index debe tener "
            "shape [2, num_edges]."
        ) # Validar estructura topológica

    if graph.x.shape[1] != in_channels:
        raise RuntimeError(
            f"El GraphData {graph_index} presenta "
            f"{graph.x.shape[1]} variables y el modelo requiere "
            f"{in_channels}."
        ) # Validar dimensión de entrada

    if graph.x.shape[0] != n_nodes:
        raise RuntimeError(
            f"El GraphData {graph_index} presenta "
            f"{graph.x.shape[0]} nodos y se esperaban "
            f"{n_nodes}."
        ) # Validar cantidad de nodos

    if not torch.isfinite(
        graph.x
    ).all():
        raise ValueError(
            f"El GraphData {graph_index}.x contiene valores no finitos."
        ) # Validar estabilidad numérica

    if not torch.isfinite(
        graph.y
    ).all():
        raise ValueError(
            f"El GraphData {graph_index}.y contiene valores no finitos."
        ) # Validar variable objetivo

# 12.6 VALIDAR GRAPH DATA DE VALIDACIÓN
for graph_index in validation_indices:
    graph = graphs[
        graph_index
    ] # Recuperar GraphData de validación

    if graph.x.shape[1] != in_channels:
        raise RuntimeError(
            f"El GraphData de validación {graph_index} no coincide "
            "con la dimensión de entrada del modelo."
        ) # Validar dimensión

    if graph.x.shape[0] != n_nodes:
        raise RuntimeError(
            f"El GraphData de validación {graph_index} no coincide "
            "con la cantidad oficial de nodos."
        ) # Validar nodos

    if not torch.isfinite(
        graph.x
    ).all():
        raise ValueError(
            f"El GraphData de validación {graph_index}.x contiene "
            "valores no finitos."
        ) # Validar estabilidad numérica

    if not torch.isfinite(
        graph.y
    ).all():
        raise ValueError(
            f"El GraphData de validación {graph_index}.y contiene "
            "valores no finitos."
        ) # Validar objetivo

# 12.7 VALIDAR CONFIGURACIÓN DEL ENTRENAMIENTO
if epochs <= 0:
    raise ValueError(
        "epochs debe ser mayor que cero."
    ) # Validar épocas

if learning_rate <= 0:
    raise ValueError(
        "learning_rate debe ser mayor que cero."
    ) # Validar Learning Rate

if weight_decay < 0:
    raise ValueError(
        "weight_decay no puede ser negativo."
    ) # Validar Weight Decay

# 12.8 VALIDAR DISPOSITIVO
model_device = next(
    official_graphsage.parameters()
).device # Recuperar dispositivo del modelo

if model_device != device:
    raise RuntimeError(
        "El dispositivo del Modelo Oficial no coincide "
        "con el dispositivo configurado."
    ) # Validar dispositivo

# 12.9 VALIDAR HISTORIAL
if not isinstance(
    training_history,
    dict
):
    raise TypeError(
        "training_history debe ser un diccionario."
    ) # Validar historial

required_history_fields = [
    "epoch",
    "train_loss",
    "validation_loss",
] # Definir estructura del historial

missing_history_fields = [
    field
    for field in required_history_fields
    if field not in training_history
] # Detectar campos faltantes

if missing_history_fields:
    raise RuntimeError(
        "El historial de entrenamiento está incompleto: "
        f"{missing_history_fields}"
    ) # Validar historial

# 12.10 VALIDACIÓN FINAL
print(f"Modelo Oficial             : {OFFICIAL_MODEL_NAME}") # Mostrar modelo
print(f"Parámetros entrenables     : {sum(parameter.numel() for parameter in trainable_parameters):,}") # Mostrar parámetros
print(f"GraphData entrenamiento    : {len(train_indices)}") # Mostrar entrenamiento
print(f"GraphData validación       : {len(validation_indices)}") # Mostrar validación
print(f"GraphData prueba           : {len(test_indices)}") # Mostrar prueba
print(f"Dimensión de entrada       : {in_channels}") # Mostrar variables
print(f"Nodos por GraphData        : {n_nodes}") # Mostrar nodos
print(f"Learning Rate              : {learning_rate:.8f}") # Mostrar Learning Rate
print(f"Weight Decay               : {weight_decay:.8f}") # Mostrar Weight Decay
print(f"Épocas                     : {epochs}") # Mostrar épocas
print(f"Dispositivo                : {device}") # Mostrar dispositivo
print(f"Criterion                  : {criterion.__class__.__name__}") # Mostrar función de pérdida
print(f"Optimizer                  : {optimizer.__class__.__name__}") # Mostrar optimizador
print("Modelo                      : VALIDADO") # Confirmar modelo
print("GraphData                   : VALIDADO") # Confirmar datos
print("Particiones                 : VALIDADAS") # Confirmar particiones
print("Dimensiones                 : VALIDADAS") # Confirmar dimensiones
print("Optimizer                   : VALIDADO") # Confirmar optimizador
print("Criterion                   : VALIDADO") # Confirmar función de pérdida
print("Dispositivo                 : VALIDADO") # Confirmar dispositivo
print("ENTORNO DE ENTRENAMIENTO    : VALIDADO") # Autorizar entrenamiento

# BLOQUE 13. RESTAURACIÓN DEL MEJOR ESTADO
# Objetivo: Restaurar en OfficialGraphSAGE los pesos correspondientes a la mejor
# época según la pérdida de validación registrada durante el entrenamiento.
# Producto: Modelo Oficial con el mejor estado validado del entrenamiento.

print("\n13. RESTAURACIÓN DEL MEJOR ESTADO")

# 13.1 VALIDAR EXISTENCIA DEL MEJOR ESTADO
# Validar que exista un mejor estado registrado
if best_model_state is None:
    raise RuntimeError(
        "No existe best_model_state para restaurar."
    ) # Validar disponibilidad del mejor estado

# Validar que el mejor estado sea un diccionario
if not isinstance(
    best_model_state,
    dict
):
    raise TypeError(
        "best_model_state debe ser un diccionario de parámetros."
    ) # Validar estructura del mejor estado

# Validar que el mejor estado no esté vacío
if len(best_model_state) == 0:
    raise RuntimeError(
        "best_model_state está vacío."
    ) # Validar contenido del mejor estado

# 13.2 VALIDAR MEJOR ÉPOCA
# Validar existencia de la mejor época
if best_epoch is None:
    raise RuntimeError(
        "best_epoch no está definido."
    ) # Validar registro de la mejor época

# Validar tipo de la mejor época
if not isinstance(
    best_epoch,
    (int, np.integer)
):
    raise TypeError(
        "best_epoch debe ser un entero."
    ) # Validar tipo de época

# Normalizar mejor época
best_epoch = int(
    best_epoch
) # Garantizar tipo entero

# Validar rango de la mejor época
if best_epoch <= 0:
    raise ValueError(
        "best_epoch debe ser mayor que cero."
    ) # Validar rango de época

if best_epoch > epochs:
    raise ValueError(
        "best_epoch no puede ser mayor que el número total de épocas."
    ) # Validar consistencia temporal

# 13.3 VALIDAR MEJOR PÉRDIDA DE VALIDACIÓN
# Validar existencia de la mejor pérdida
if not np.isfinite(
    best_validation_loss
):
    raise RuntimeError(
        "best_validation_loss no contiene un valor finito."
    ) # Validar mejor pérdida

# Validar que la pérdida sea no negativa
if best_validation_loss < 0.0:
    raise ValueError(
        "best_validation_loss no puede ser negativa."
    ) # Validar rango de pérdida

# 13.4 VALIDAR CORRESPONDENCIA CON EL MODELO
# Recuperar estado actual del modelo
current_model_state = official_graphsage.state_dict() # Recuperar estado actual
# Validar correspondencia de parámetros
if set(
    current_model_state.keys()
) != set(
    best_model_state.keys()
):
    raise RuntimeError(
        "best_model_state no coincide estructuralmente con "
        "el estado actual de OfficialGraphSAGE."
    ) # Validar estructura de parámetros

# 13.5 VALIDAR DIMENSIONES DE LOS PESOS
# Validar dimensión de cada parámetro
for parameter_name, current_parameter in current_model_state.items():
    # Recuperar parámetro del mejor estado
    best_parameter = best_model_state[
        parameter_name
    ] # Recuperar peso correspondiente

    # Validar tipo tensorial
    if not isinstance(
        best_parameter,
        torch.Tensor
    ):
        raise TypeError(
            f"El parámetro '{parameter_name}' del mejor estado "
            "no es un tensor."
        ) # Validar tipo del parámetro

    # Validar dimensiones
    if current_parameter.shape != best_parameter.shape:
        raise RuntimeError(
            f"El parámetro '{parameter_name}' presenta dimensiones "
            "incompatibles con OfficialGraphSAGE."
        ) # Validar dimensiones

# 13.6 VALIDAR INTEGRIDAD NUMÉRICA DEL MEJOR ESTADO
# Validar valores finitos
for parameter_name, best_parameter in best_model_state.items():

    if not torch.isfinite(
        best_parameter
    ).all():
        raise RuntimeError(
            f"El parámetro '{parameter_name}' del mejor estado "
            "contiene NaN o infinitos."
        ) # Validar integridad numérica

# 13.7 RESTAURAR MEJOR ESTADO
# Cargar pesos correspondientes a la mejor época
restore_result = official_graphsage.load_state_dict(
    best_model_state,
    strict=True,
) # Restaurar estado óptimo del modelo

# Validar resultado de la restauración
if restore_result.missing_keys:
    raise RuntimeError(
        "Existen parámetros faltantes durante la restauración: "
        f"{restore_result.missing_keys}"
    ) # Validar parámetros faltantes

if restore_result.unexpected_keys:
    raise RuntimeError(
        "Existen parámetros inesperados durante la restauración: "
        f"{restore_result.unexpected_keys}"
    ) # Validar parámetros inesperados

# 13.8 VALIDAR ESTADO RESTAURADO
# Recuperar estado posterior a la restauración
restored_model_state = official_graphsage.state_dict() # Recuperar estado restaurado

# Comparar cada parámetro restaurado
for parameter_name, best_parameter in best_model_state.items():

    restored_parameter = restored_model_state[
        parameter_name
    ] # Recuperar parámetro restaurado

    if not torch.equal(
        restored_parameter,
        best_parameter
    ):
        raise RuntimeError(
            f"El parámetro '{parameter_name}' no coincide con "
            "best_model_state después de la restauración."
        ) # Validar restauración exacta

# 13.9 CONSERVAR EL MODELO EN MODO EVALUACIÓN
# Establecer modo evaluación
official_graphsage.eval() # Preparar modelo para evaluación posterior

# Validar modo evaluación
if official_graphsage.training:
    raise RuntimeError(
        "OfficialGraphSAGE no pudo establecerse en modo evaluación."
    ) # Validar estado de evaluación

# 13.10 REGISTRAR RESULTADO DE RESTAURACIÓN
# Registrar información del mejor modelo
best_model_summary = {
    "best_epoch": int(best_epoch),
    "best_validation_loss": float(best_validation_loss),
    "restored": True,
    "evaluation_mode": True,
} # Registrar estado restaurado

# 13.11 RESUMEN DE RESTAURACIÓN
print(f"Mejor época               : {best_epoch}") # Mostrar mejor época
print(f"Mejor Validation Loss     : {best_validation_loss:.10f}") # Mostrar mejor pérdida
print(f"Parámetros restaurados    : {len(best_model_state)}") # Mostrar cantidad de parámetros
print("Coincidencia de parámetros : VALIDADA") # Confirmar correspondencia
print("Integridad numérica        : VALIDADA") # Confirmar integridad
print("Restauración               : COMPLETADA") # Confirmar restauración
print("Modo evaluación            : ACTIVADO") # Confirmar modo evaluación
print("MEJOR ESTADO DEL MODELO    : RESTAURADO") # Confirmar resultado final

# BLOQUE 14. MÉTRICAS DEL ENTRENAMIENTO
# Objetivo: Calcular las métricas predictivas del Modelo Oficial restaurado sobre
# las particiones de entrenamiento y validación.
# Producto: Métricas oficiales de entrenamiento y validación del Modelo Oficial.

print("\n14. MÉTRICAS DEL ENTRENAMIENTO")

# 14.1 VALIDACIÓN DEL MODELO RESTAURADO
# Validar disponibilidad del Modelo Oficial
if official_graphsage is None:
    raise RuntimeError(
        "OfficialGraphSAGE no está disponible."
    ) # Validar disponibilidad del modelo

# Validar que el modelo se encuentre en modo evaluación
if official_graphsage.training:
    raise RuntimeError(
        "OfficialGraphSAGE debe encontrarse en modo evaluación."
    ) # Validar estado de evaluación

# 14.2 FUNCIÓN PARA CALCULAR MÉTRICAS
def calculate_regression_metrics(
    y_true,
    y_pred,
):
    """Calcular métricas de regresión para la evaluación del modelo."""

    # Convertir valores reales a NumPy
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ) # Normalizar valores observados

    # Convertir predicciones a NumPy
    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ) # Normalizar predicciones

    # Validar dimensiones
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true y y_pred deben tener la misma dimensión."
        ) # Validar correspondencia

    # Validar valores finitos
    if not np.isfinite(
        y_true
    ).all():
        raise ValueError(
            "y_true contiene valores NaN o infinitos."
        ) # Validar valores observados

    if not np.isfinite(
        y_pred
    ).all():
        raise ValueError(
            "y_pred contiene valores NaN o infinitos."
        ) # Validar predicciones

    # Calcular error
    errors = y_true - y_pred # Calcular errores de predicción

    # Calcular RMSE
    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    ) # Calcular raíz del error cuadrático medio

    # Calcular MAE
    mae = float(
        np.mean(
            np.abs(
                errors
            )
        )
    ) # Calcular error absoluto medio

    # Identificar valores distintos de cero para MAPE
    non_zero_mask = np.abs(
        y_true
    ) > np.finfo(
        np.float64
    ).eps # Identificar denominadores válidos

    # Validar existencia de observaciones válidas para MAPE
    if not np.any(
        non_zero_mask
    ):
        mape = float(
            "nan"
        ) # Registrar MAPE no definido
    else:
        mape = float(
            np.mean(
                np.abs(
                    errors[non_zero_mask]
                    / y_true[non_zero_mask]
                )
            )
            * 100.0
        ) # Calcular MAPE excluyendo valores reales cero

    # Calcular R2
    y_true_mean = np.mean(
        y_true
    ) # Calcular media observada

    total_sum_squares = np.sum(
        (
            y_true
            - y_true_mean
        ) ** 2
    ) # Calcular variabilidad total

    residual_sum_squares = np.sum(
        errors ** 2
    ) # Calcular variabilidad residual

    # Validar denominador de R2
    if total_sum_squares <= np.finfo(
        np.float64
    ).eps:
        r2 = float(
            "nan"
        ) # Registrar R2 no definido
    else:
        r2 = float(
            1.0
            - (
                residual_sum_squares
                / total_sum_squares
            )
        ) # Calcular coeficiente de determinación

    # Construir resultado de métricas
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
    } # Retornar métricas

# 14.3 FUNCIÓN DE PREDICCIÓN SOBRE UNA PARTICIÓN
def predict_graph_partition(
    model,
    graphs,
    graph_indices,
    device,
):
    """Generar predicciones para una partición de GraphData."""
    true_values = [] # Inicializar valores observados
    predicted_values = [] # Inicializar predicciones
    # Evaluar cada GraphData de la partición
    for graph_index in graph_indices:

        # Recuperar GraphData
        graph = graphs[
            graph_index
        ] # Recuperar grafo

        # Transferir GraphData al dispositivo
        graph = graph.to(
            device
        ) # Transferir datos

        # Ejecutar predicción sin gradientes
        with torch.no_grad():

            predictions = model(
                graph.x,
                graph.edge_index,
            ) # Generar predicciones

        # Aplanar predicciones
        predictions = predictions.reshape(
            -1
        ) # Normalizar estructura de predicciones

        # Aplanar valores observados
        targets = graph.y.reshape(
            -1
        ) # Normalizar estructura del objetivo

        # Validar correspondencia
        if predictions.shape != targets.shape:
            raise RuntimeError(
                f"GraphData {graph_index}: las predicciones y "
                "el objetivo presentan dimensiones diferentes."
            ) # Validar correspondencia

        # Recuperar valores observados
        true_values.append(
            targets.detach().cpu().numpy()
        ) # Acumular valores observados

        # Recuperar predicciones
        predicted_values.append(
            predictions.detach().cpu().numpy()
        ) # Acumular predicciones

    # Concatenar valores observados
    y_true = np.concatenate(
        true_values
    ) # Construir vector observado

    # Concatenar predicciones
    y_pred = np.concatenate(
        predicted_values
    ) # Construir vector predicho

    # Retornar resultados
    return y_true, y_pred

# 14.4 PREDICCIONES DE ENTRENAMIENTO
# Generar predicciones sobre entrenamiento
y_train_true, y_train_pred = predict_graph_partition(
    model=official_graphsage,
    graphs=graphs,
    graph_indices=train_indices,
    device=device,
) # Generar predicciones de entrenamiento

# 14.5 PREDICCIONES DE VALIDACIÓN
# Generar predicciones sobre validación
y_validation_true, y_validation_pred = predict_graph_partition(
    model=official_graphsage,
    graphs=graphs,
    graph_indices=validation_indices,
    device=device,
) # Generar predicciones de validación

# 14.6 CALCULAR MÉTRICAS DE ENTRENAMIENTO
# Calcular métricas sobre entrenamiento
train_metrics = calculate_regression_metrics(
    y_true=y_train_true,
    y_pred=y_train_pred,
) # Calcular métricas de entrenamiento

# 14.7 CALCULAR MÉTRICAS DE VALIDACIÓN
# Calcular métricas sobre validación
validation_metrics = calculate_regression_metrics(
    y_true=y_validation_true,
    y_pred=y_validation_pred,
) # Calcular métricas de validación

# 14.8 CALCULAR PÉRDIDAS DIRECTAMENTE
# Calcular MSE de entrenamiento
train_mse = float(
    np.mean(
        (
            y_train_true
            - y_train_pred
        ) ** 2
    )
) # Calcular pérdida MSE de entrenamiento

# Calcular MSE de validación
validation_mse = float(
    np.mean(
        (
            y_validation_true
            - y_validation_pred
        ) ** 2
    )
) # Calcular pérdida MSE de validación

# 14.9 REGISTRAR MÉTRICAS
# Construir registro de métricas del entrenamiento
training_metrics = {
    "train": {
        "loss": train_mse,
        "RMSE": train_metrics["RMSE"],
        "MAE": train_metrics["MAE"],
        "MAPE": train_metrics["MAPE"],
        "R2": train_metrics["R2"],
    },
    "validation": {
        "loss": validation_mse,
        "RMSE": validation_metrics["RMSE"],
        "MAE": validation_metrics["MAE"],
        "MAPE": validation_metrics["MAPE"],
        "R2": validation_metrics["R2"],
    },
    "best_epoch": int(
        best_epoch
    ),
    "best_validation_loss": float(
        best_validation_loss
    ),
} # Registrar métricas oficiales

# 14.10 VALIDACIÓN NUMÉRICA DE LAS MÉTRICAS
# Validar métricas de entrenamiento
for metric_name, metric_value in train_metrics.items():
    if metric_name != "MAPE" and not np.isfinite(
        metric_value
    ):
        raise RuntimeError(
            f"La métrica de entrenamiento {metric_name} no es finita."
        ) # Validar métrica de entrenamiento

# Validar métricas de validación
for metric_name, metric_value in validation_metrics.items():

    if metric_name != "MAPE" and not np.isfinite(
        metric_value
    ):
        raise RuntimeError(
            f"La métrica de validación {metric_name} no es finita."
        ) # Validar métrica de validación

# 14.11 RESUMEN DE MÉTRICAS
print("MÉTRICAS DE ENTRENAMIENTO") # Encabezado de entrenamiento
print(f"Loss                       : {training_metrics['train']['loss']:.10f}") # Mostrar pérdida
print(f"RMSE                       : {training_metrics['train']['RMSE']:.10f}") # Mostrar RMSE
print(f"MAE                        : {training_metrics['train']['MAE']:.10f}") # Mostrar MAE
print(f"MAPE                       : {training_metrics['train']['MAPE']:.10f}") # Mostrar MAPE
print(f"R2                         : {training_metrics['train']['R2']:.10f}") # Mostrar R2
print("MÉTRICAS DE VALIDACIÓN") # Encabezado de validación
print(f"Loss                       : {training_metrics['validation']['loss']:.10f}") # Mostrar pérdida
print(f"RMSE                       : {training_metrics['validation']['RMSE']:.10f}") # Mostrar RMSE
print(f"MAE                        : {training_metrics['validation']['MAE']:.10f}") # Mostrar MAE
print(f"MAPE                       : {training_metrics['validation']['MAPE']:.10f}") # Mostrar MAPE
print(f"R2                         : {training_metrics['validation']['R2']:.10f}") # Mostrar R2
print(f"Mejor época                : {training_metrics['best_epoch']}") # Mostrar mejor época
print(f"Mejor Validation Loss      : {training_metrics['best_validation_loss']:.10f}") # Mostrar mejor pérdida
print("Métricas de entrenamiento   : CALCULADAS") # Confirmar métricas
print("Métricas de validación      : CALCULADAS") # Confirmar métricas
print("Evaluación del modelo       : COMPLETADA") # Confirmar evaluación

# BLOQUE 15. PERSISTENCIA DEL MODELO
# Objetivo: Persistir el Modelo Oficial GraphSAGE restaurado en su mejor estado
# junto con la configuración y metadatos necesarios para su trazabilidad.
# Producto: Modelo Oficial persistido y disponible para evaluación, forecasting y GeoAI.

print("\n15. PERSISTENCIA DEL MODELO")

# 15.1 RECUPERACIÓN DE LAS ENTRADAS OFICIALES
print("\n15.1 RECUPERACIÓN DE LAS ENTRADAS OFICIALES")

if not isinstance(
    BENCHMARK_DATA,
    dict
):
    raise TypeError(
        "BENCHMARK_DATA debe ser un diccionario."
    ) # Validar estructura de datos

if not isinstance(
    official,
    dict
):
    raise TypeError(
        "official debe ser un diccionario."
    ) # Validar Modelo Oficial

required_benchmark_data_fields = [
    "graphs",
    "train_index",
    "validation_index",
    "test_index",
    "scaler",
] # Definir entradas obligatorias

missing_benchmark_data_fields = [
    field
    for field in required_benchmark_data_fields
    if field not in BENCHMARK_DATA
] # Identificar entradas faltantes

if missing_benchmark_data_fields:
    raise RuntimeError(
        "BENCHMARK_DATA está incompleto: "
        f"{missing_benchmark_data_fields}"
    ) # Validar contrato de datos

required_official_fields = [
    "model_code",
    "model_name",
    "family",
    "model_config",
    "ranking_position",
] # Definir campos obligatorios del Modelo Oficial

missing_official_fields = [
    field
    for field in required_official_fields
    if field not in official
] # Identificar campos faltantes

if missing_official_fields:
    raise RuntimeError(
        "El Modelo Oficial está incompleto: "
        f"{missing_official_fields}"
    ) # Validar contrato oficial

if official["model_code"] != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El código del Modelo Oficial no coincide "
        "con OFFICIAL_MODEL_CODE."
    ) # Validar código oficial

if official["model_name"].strip().lower() != OFFICIAL_MODEL_NAME.strip().lower():
    raise RuntimeError(
        "El nombre del Modelo Oficial no coincide "
        "con OFFICIAL_MODEL_NAME."
    ) # Validar nombre oficial

if official["family"].strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise RuntimeError(
        "La familia del Modelo Oficial no coincide "
        "con OFFICIAL_MODEL_FAMILY."
    ) # Validar familia oficial

if official["model_code"] != "GNN02":
    raise RuntimeError(
        "El entrenamiento oficial requiere el modelo GNN02."
    ) # Validar identidad GraphSAGE

if official["model_name"].strip().lower() != "graphsage":
    raise RuntimeError(
        "El entrenamiento oficial requiere GraphSAGE."
    ) # Validar arquitectura oficial

if official["family"].strip().lower() != "graph_neural_networks":
    raise RuntimeError(
        "El Modelo Oficial debe pertenecer a Graph Neural Networks."
    ) # Validar familia oficial

graphs = BENCHMARK_DATA[
    "graphs"
] # Recuperar GraphData

train_index = BENCHMARK_DATA[
    "train_index"
] # Recuperar índices de entrenamiento

validation_index = BENCHMARK_DATA[
    "validation_index"
] # Recuperar índices de validación

test_index = BENCHMARK_DATA[
    "test_index"
] # Recuperar índices de prueba

scaler = BENCHMARK_DATA[
    "scaler"
] # Recuperar escalador

official_model_config = official[
    "model_config"
].copy() # Recuperar configuración oficial

print(f"GraphData                  : {len(graphs)}") # Mostrar cantidad de grafos
print(f"Entrenamiento              : {len(train_index)}") # Mostrar partición de entrenamiento
print(f"Validación                 : {len(validation_index)}") # Mostrar partición de validación
print(f"Prueba                     : {len(test_index)}") # Mostrar partición de prueba
print(f"Modelo Oficial             : {official['model_name']}") # Mostrar modelo
print(f"Código Oficial             : {official['model_code']}") # Mostrar código
print(f"Familia Oficial            : {official['family']}") # Mostrar familia
print(f"Posición Ranking           : {official['ranking_position']}") # Mostrar posición
print("Entradas oficiales          : RECUPERADAS") # Confirmar recuperación

# 15.2 VALIDACIÓN DE LAS PARTICIONES OFICIALES
# Objetivo: Validar la integridad, separación y rango de las particiones oficiales.
# Producto: Particiones de entrenamiento, validación y prueba científicamente validadas.

print("\n15.2 VALIDACIÓN DE LAS PARTICIONES OFICIALES")

# 15.2.1 Validar tipo de las particiones
if not isinstance(
    train_index,
    np.ndarray
):
    raise TypeError(
        "train_index debe ser un arreglo NumPy."
    ) # Validar tipo de entrenamiento

if not isinstance(
    validation_index,
    np.ndarray
):
    raise TypeError(
        "validation_index debe ser un arreglo NumPy."
    ) # Validar tipo de validación

if not isinstance(
    test_index,
    np.ndarray
):
    raise TypeError(
        "test_index debe ser un arreglo NumPy."
    ) # Validar tipo de prueba

# 15.2.2 Validar dimensionalidad de las particiones
if train_index.ndim != 1:
    raise ValueError(
        "train_index debe ser un vector unidimensional."
    ) # Validar dimensionalidad de entrenamiento

if validation_index.ndim != 1:
    raise ValueError(
        "validation_index debe ser un vector unidimensional."
    ) # Validar dimensionalidad de validación

if test_index.ndim != 1:
    raise ValueError(
        "test_index debe ser un vector unidimensional."
    ) # Validar dimensionalidad de prueba

# 15.2.3 Validar que las particiones no estén vacías
if len(train_index) == 0:
    raise RuntimeError(
        "train_index está vacío."
    ) # Validar contenido de entrenamiento

if len(validation_index) == 0:
    raise RuntimeError(
        "validation_index está vacío."
    ) # Validar contenido de validación

if len(test_index) == 0:
    raise RuntimeError(
        "test_index está vacío."
    ) # Validar contenido de prueba

# 15.2.4 Validar tipo entero de los índices
if not np.issubdtype(
    train_index.dtype,
    np.integer
):
    raise TypeError(
        "train_index debe contener índices enteros."
    ) # Validar tipo de índices de entrenamiento

if not np.issubdtype(
    validation_index.dtype,
    np.integer
):
    raise TypeError(
        "validation_index debe contener índices enteros."
    ) # Validar tipo de índices de validación

if not np.issubdtype(
    test_index.dtype,
    np.integer
):
    raise TypeError(
        "test_index debe contener índices enteros."
    ) # Validar tipo de índices de prueba

# 15.2.5 Determinar cantidad total de GraphData
n_graphs = len(
    graphs
) # Determinar cantidad total de GraphData

if n_graphs <= 0:
    raise RuntimeError(
        "No existen GraphData disponibles para validar las particiones."
    ) # Validar cantidad de GraphData

# 15.2.6 Validar rango de los índices
for partition_name, partition_index in {
    "train_index": train_index,
    "validation_index": validation_index,
    "test_index": test_index,
}.items():

    if np.any(
        partition_index < 0
    ):
        raise ValueError(
            f"{partition_name} contiene índices negativos."
        ) # Validar límite inferior

    if np.any(
        partition_index >= n_graphs
    ):
        raise ValueError(
            f"{partition_name} contiene índices fuera del rango válido "
            f"0:{n_graphs - 1}."
        ) # Validar límite superior

# 15.2.7 Validar ausencia de duplicados internos
if len(
    np.unique(
        train_index
    )
) != len(
    train_index
):
    raise RuntimeError(
        "train_index contiene índices duplicados."
    ) # Validar duplicados de entrenamiento

if len(
    np.unique(
        validation_index
    )
) != len(
    validation_index
):
    raise RuntimeError(
        "validation_index contiene índices duplicados."
    ) # Validar duplicados de validación

if len(
    np.unique(
        test_index
    )
) != len(
    test_index
):
    raise RuntimeError(
        "test_index contiene índices duplicados."
    ) # Validar duplicados de prueba

# 15.2.8 Construir conjuntos de índices
train_set = set(
    train_index.tolist()
) # Construir conjunto de entrenamiento

validation_set = set(
    validation_index.tolist()
) # Construir conjunto de validación

test_set = set(
    test_index.tolist()
) # Construir conjunto de prueba

# 15.2.9 Validar ausencia de solapamiento
train_validation_overlap = train_set.intersection(
    validation_set
) # Detectar solapamiento entrenamiento-validación

train_test_overlap = train_set.intersection(
    test_set
) # Detectar solapamiento entrenamiento-prueba

validation_test_overlap = validation_set.intersection(
    test_set
) # Detectar solapamiento validación-prueba

if train_validation_overlap:
    raise RuntimeError(
        "Existe solapamiento entre entrenamiento y validación: "
        f"{sorted(train_validation_overlap)}"
    ) # Validar separación entrenamiento-validación

if train_test_overlap:
    raise RuntimeError(
        "Existe solapamiento entre entrenamiento y prueba: "
        f"{sorted(train_test_overlap)}"
    ) # Validar separación entrenamiento-prueba

if validation_test_overlap:
    raise RuntimeError(
        "Existe solapamiento entre validación y prueba: "
        f"{sorted(validation_test_overlap)}"
    ) # Validar separación validación-prueba

# 15.2.10 Validar cobertura temporal
partition_union = (
    train_set
    | validation_set
    | test_set
) # Construir unión de las particiones

if len(
    partition_union
) != n_graphs:
    missing_graph_indices = sorted(
        set(
            range(
                n_graphs
            )
        )
        - partition_union
    ) # Identificar GraphData no asignados

    raise RuntimeError(
        "Existen GraphData sin asignar a ninguna partición: "
        f"{missing_graph_indices}"
    ) # Validar cobertura completa

# 15.2.11 Validar suma de tamaños
total_partition_records = (
    len(train_index)
    + len(validation_index)
    + len(test_index)
) # Calcular cantidad total de elementos particionados

if total_partition_records != n_graphs:
    raise RuntimeError(
        "La suma de las particiones no coincide con el número total de GraphData."
    ) # Validar cobertura mediante cardinalidad

# 15.2.12 Construir índices normalizados
train_indices = [
    int(index)
    for index in train_index
] # Normalizar índices de entrenamiento

validation_indices = [
    int(index)
    for index in validation_index
] # Normalizar índices de validación

test_indices = [
    int(index)
    for index in test_index
] # Normalizar índices de prueba

# 15.2.13 Registrar resumen de particiones
partition_validation_summary = {
    "n_graphs": int(n_graphs),
    "train_graphs": int(len(train_indices)),
    "validation_graphs": int(len(validation_indices)),
    "test_graphs": int(len(test_indices)),
    "train_validation_overlap": 0,
    "train_test_overlap": 0,
    "validation_test_overlap": 0,
    "complete_coverage": True,
    "validated": True,
} # Registrar resultado de validación

# 15.2.14 Mostrar resultado
print(f"GraphData totales         : {n_graphs}") # Mostrar total de GraphData
print(f"Entrenamiento             : {len(train_indices)}") # Mostrar entrenamiento
print(f"Validación                : {len(validation_indices)}") # Mostrar validación
print(f"Prueba                    : {len(test_indices)}") # Mostrar prueba
print("Duplicados internos        : NO") # Confirmar ausencia de duplicados
print("Solapamiento entre grupos  : NO") # Confirmar separación
print("Cobertura de GraphData     : COMPLETA") # Confirmar cobertura
print("Particiones oficiales      : VALIDADAS") # Confirmar validación

# 15.3 AUDITORÍA DE LA CONFIGURACIÓN DE GRAPHSAGE
# Objetivo: Auditar la configuración oficial de GraphSAGE proveniente del Benchmark.
# Producto: Configuración GraphSAGE validada para construir y entrenar el Modelo Oficial.

print("\n15.3 AUDITORÍA DE LA CONFIGURACIÓN DE GRAPHSAGE")

# 15.3.1 Validar estructura de la configuración
if not isinstance(
    official_model_config,
    dict
):
    raise TypeError(
        "official_model_config debe ser un diccionario."
    ) # Validar estructura de configuración

# 15.3.2 Definir campos obligatorios
required_graphsage_config_fields = [
    "hidden_channels",
    "dropout",
    "learning_rate",
    "weight_decay",
    "epochs",
] # Definir contrato mínimo de GraphSAGE

# 15.3.3 Identificar campos faltantes
missing_graphsage_config_fields = [
    field
    for field in required_graphsage_config_fields
    if field not in official_model_config
] # Identificar campos ausentes

# 15.3.4 Validar contrato de configuración
if missing_graphsage_config_fields:
    raise RuntimeError(
        "La configuración oficial de GraphSAGE está incompleta. "
        f"Campos faltantes: {missing_graphsage_config_fields}"
    ) # Validar integridad de configuración

# 15.3.5 Recuperar hiperparámetros
hidden_channels = official_model_config[
    "hidden_channels"
] # Recuperar dimensión oculta

dropout = official_model_config[
    "dropout"
] # Recuperar Dropout

learning_rate = official_model_config[
    "learning_rate"
] # Recuperar tasa de aprendizaje

weight_decay = official_model_config[
    "weight_decay"
] # Recuperar regularización L2

epochs = official_model_config[
    "epochs"
] # Recuperar número de épocas

# 15.3.6 Validar dimensión oculta
if not isinstance(
    hidden_channels,
    (int, np.integer)
):
    raise TypeError(
        "hidden_channels debe ser un entero."
    ) # Validar tipo de dimensión oculta

hidden_channels = int(
    hidden_channels
) # Normalizar dimensión oculta

if hidden_channels <= 0:
    raise ValueError(
        "hidden_channels debe ser mayor que cero."
    ) # Validar rango de dimensión oculta

# 15.3.7 Validar Dropout
if not isinstance(
    dropout,
    (int, float, np.integer, np.floating)
):
    raise TypeError(
        "dropout debe ser un valor numérico."
    ) # Validar tipo de Dropout

dropout = float(
    dropout
) # Normalizar Dropout

if not 0.0 <= dropout < 1.0:
    raise ValueError(
        "dropout debe encontrarse en el intervalo [0, 1)."
    ) # Validar rango de Dropout

# 15.3.8 Validar Learning Rate
if not isinstance(
    learning_rate,
    (int, float, np.integer, np.floating)
):
    raise TypeError(
        "learning_rate debe ser un valor numérico."
    ) # Validar tipo de Learning Rate

learning_rate = float(
    learning_rate
) # Normalizar Learning Rate

if learning_rate <= 0.0:
    raise ValueError(
        "learning_rate debe ser mayor que cero."
    ) # Validar rango de Learning Rate

# 15.3.9 Validar Weight Decay
if not isinstance(
    weight_decay,
    (int, float, np.integer, np.floating)
):
    raise TypeError(
        "weight_decay debe ser un valor numérico."
    ) # Validar tipo de Weight Decay

weight_decay = float(
    weight_decay
) # Normalizar Weight Decay

if weight_decay < 0.0:
    raise ValueError(
        "weight_decay no puede ser negativo."
    ) # Validar rango de Weight Decay

# 15.3.10 Validar número de épocas
if not isinstance(
    epochs,
    (int, np.integer)
):
    raise TypeError(
        "epochs debe ser un entero."
    ) # Validar tipo de épocas

epochs = int(
    epochs
) # Normalizar número de épocas

if epochs <= 0:
    raise ValueError(
        "epochs debe ser mayor que cero."
    ) # Validar rango de épocas

# 15.3.11 Validar dimensiones del modelo
if "in_channels" in official_model_config:
    configured_in_channels = int(
        official_model_config[
            "in_channels"
        ]
    ) # Recuperar dimensión de entrada configurada

    if configured_in_channels <= 0:
        raise ValueError(
            "in_channels configurado debe ser mayor que cero."
        ) # Validar dimensión de entrada

else:
    configured_in_channels = None # Registrar dimensión de entrada pendiente

# 15.3.12 Validar dimensión de salida
if "out_channels" in official_model_config:
    configured_out_channels = int(
        official_model_config[
            "out_channels"
        ]
    ) # Recuperar dimensión de salida configurada

    if configured_out_channels <= 0:
        raise ValueError(
            "out_channels configurado debe ser mayor que cero."
        ) # Validar dimensión de salida

else:
    configured_out_channels = 1 # Definir salida escalar para regresión

# 15.3.13 Construir configuración auditada
graphsage_config_audit = {
    "model_code": OFFICIAL_MODEL_CODE,
    "model_name": OFFICIAL_MODEL_NAME,
    "family": OFFICIAL_MODEL_FAMILY,
    "hidden_channels": hidden_channels,
    "dropout": dropout,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "epochs": epochs,
    "configured_in_channels": configured_in_channels,
    "configured_out_channels": configured_out_channels,
} # Registrar auditoría de configuración

# 15.3.14 Mostrar configuración auditada
print(f"Modelo Oficial             : {OFFICIAL_MODEL_NAME}") # Mostrar modelo
print(f"Código Oficial             : {OFFICIAL_MODEL_CODE}") # Mostrar código
print(f"Familia                    : {OFFICIAL_MODEL_FAMILY}") # Mostrar familia
print(f"Hidden Channels            : {hidden_channels}") # Mostrar dimensión oculta
print(f"Dropout                    : {dropout:.6f}") # Mostrar Dropout
print(f"Learning Rate              : {learning_rate:.8f}") # Mostrar Learning Rate
print(f"Weight Decay               : {weight_decay:.8f}") # Mostrar regularización
print(f"Épocas                     : {epochs}") # Mostrar épocas
print(f"In Channels configurado    : {configured_in_channels}") # Mostrar dimensión configurada
print(f"Out Channels configurado   : {configured_out_channels}") # Mostrar salida configurada
print("Configuración GraphSAGE     : AUDITADA") # Confirmar auditoría

# 15.4 AUDITORÍA DE GRAPHDATA
# Objetivo: Auditar la integridad estructural, dimensional y numérica de los GraphData oficiales.
# Producto: GraphData validados y compatibles con el entrenamiento de GraphSAGE.

print("\n15.4 AUDITORÍA DE GRAPHDATA")

# 15.4.1 Validar colección GraphData
if not isinstance(
    graphs,
    list
):
    raise TypeError(
        "graphs debe ser una lista."
    ) # Validar colección GraphData

if len(graphs) == 0:
    raise RuntimeError(
        "La colección GraphData está vacía."
    ) # Validar contenido

# 15.4.2 Recuperar GraphData de referencia
reference_graph = graphs[
    0
] # Recuperar GraphData de referencia

if not isinstance(
    reference_graph,
    Data
):
    raise TypeError(
        "El GraphData de referencia no es una instancia de Data."
    ) # Validar tipo GraphData

# 15.4.3 Validar presencia de atributos obligatorios
required_graph_attributes = [
    "x",
    "edge_index",
    "y",
] # Definir atributos obligatorios

missing_reference_attributes = [
    attribute
    for attribute in required_graph_attributes
    if not hasattr(
        reference_graph,
        attribute
    )
] # Identificar atributos faltantes

if missing_reference_attributes:
    raise RuntimeError(
        "El GraphData de referencia está incompleto. "
        f"Faltan: {missing_reference_attributes}"
    ) # Validar estructura mínima

# 15.4.4 Determinar dimensiones oficiales
if reference_graph.x.ndim != 2:
    raise ValueError(
        "reference_graph.x debe ser una matriz bidimensional."
    ) # Validar dimensionalidad de características

in_channels = int(
    reference_graph.x.shape[1]
) # Determinar número de variables predictoras

n_nodes = int(
    reference_graph.x.shape[0]
) # Determinar número de nodos

if in_channels <= 0:
    raise ValueError(
        "in_channels debe ser mayor que cero."
    ) # Validar dimensión de entrada

if n_nodes <= 0:
    raise ValueError(
        "n_nodes debe ser mayor que cero."
    ) # Validar cantidad de nodos

# 15.4.5 Determinar dimensión del objetivo
if reference_graph.y.ndim == 0:
    raise ValueError(
        "reference_graph.y no puede ser un escalar."
    ) # Validar objetivo

if reference_graph.y.shape[0] != n_nodes:
    raise ValueError(
        "El número de objetivos no coincide con el número de nodos."
    ) # Validar correspondencia objetivo-nodos

out_channels = 1 # Definir salida escalar para regresión

# 15.4.6 Validar topología de referencia
if reference_graph.edge_index.ndim != 2:
    raise ValueError(
        "reference_graph.edge_index debe ser bidimensional."
    ) # Validar dimensionalidad topológica

if reference_graph.edge_index.shape[0] != 2:
    raise ValueError(
        "reference_graph.edge_index debe tener forma [2, num_edges]."
    ) # Validar contrato de PyTorch Geometric

n_edges_reference = int(
    reference_graph.edge_index.shape[1]
) # Determinar cantidad de aristas

# 15.4.7 Validar todos los GraphData
for graph_index, graph in enumerate(
    graphs
):
    if not isinstance(
        graph,
        Data
    ):
        raise TypeError(
            f"GraphData {graph_index} no es una instancia de Data."
        ) # Validar tipo GraphData

    for attribute in required_graph_attributes:

        if not hasattr(
            graph,
            attribute
        ):
            raise RuntimeError(
                f"GraphData {graph_index} no contiene '{attribute}'."
            ) # Validar atributo obligatorio

    if graph.x.ndim != 2:
        raise ValueError(
            f"GraphData {graph_index}: x debe ser bidimensional."
        ) # Validar características

    if graph.x.shape[1] != in_channels:
        raise ValueError(
            f"GraphData {graph_index}: presenta "
            f"{graph.x.shape[1]} variables y se esperaban "
            f"{in_channels}."
        ) # Validar dimensión de entrada

    if graph.x.shape[0] != n_nodes:
        raise ValueError(
            f"GraphData {graph_index}: presenta "
            f"{graph.x.shape[0]} nodos y se esperaban "
            f"{n_nodes}."
        ) # Validar cantidad de nodos

    if graph.y.ndim == 0:
        raise ValueError(
            f"GraphData {graph_index}: y no puede ser escalar."
        ) # Validar objetivo

    if graph.y.shape[0] != n_nodes:
        raise ValueError(
            f"GraphData {graph_index}: y no coincide con el número de nodos."
        ) # Validar correspondencia objetivo

    if graph.edge_index.ndim != 2:
        raise ValueError(
            f"GraphData {graph_index}: edge_index debe ser bidimensional."
        ) # Validar topología

    if graph.edge_index.shape[0] != 2:
        raise ValueError(
            f"GraphData {graph_index}: edge_index debe tener "
            "forma [2, num_edges]."
        ) # Validar estructura topológica

    if graph.edge_index.numel() > 0:
        if torch.any(
            graph.edge_index < 0
        ):
            raise ValueError(
                f"GraphData {graph_index}: edge_index contiene índices negativos."
            ) # Validar límite inferior de nodos

        if torch.any(
            graph.edge_index >= n_nodes
        ):
            raise ValueError(
                f"GraphData {graph_index}: edge_index contiene "
                "índices fuera del rango de nodos."
            ) # Validar límite superior de nodos

    if not torch.isfinite(
        graph.x
    ).all():
        raise ValueError(
            f"GraphData {graph_index}: x contiene NaN o infinitos."
        ) # Validar integridad numérica de características

    if not torch.isfinite(
        graph.y
    ).all():
        raise ValueError(
            f"GraphData {graph_index}: y contiene NaN o infinitos."
        ) # Validar integridad numérica del objetivo

    if not torch.isfinite(
        graph.edge_index.to(
            torch.float32
        )
    ).all():
        raise ValueError(
            f"GraphData {graph_index}: edge_index contiene valores no finitos."
        ) # Validar integridad numérica de la topología

# 15.4.8 Validar compatibilidad con configuración GraphSAGE
if configured_in_channels is not None:
    if configured_in_channels != in_channels:
        raise RuntimeError(
            "La dimensión de entrada configurada para GraphSAGE "
            f"({configured_in_channels}) no coincide con GraphData "
            f"({in_channels})."
        ) # Validar compatibilidad dimensional

if configured_out_channels != out_channels:
    raise RuntimeError(
        "La dimensión de salida configurada para GraphSAGE "
        f"({configured_out_channels}) no coincide con la salida esperada "
        f"({out_channels})."
    ) # Validar compatibilidad de salida

# 15.4.9 Registrar auditoría de GraphData
graph_data_audit = {
    "n_graphs": int(len(graphs)),
    "n_nodes": int(n_nodes),
    "n_edges_reference": int(n_edges_reference),
    "in_channels": int(in_channels),
    "out_channels": int(out_channels),
    "finite_features": True,
    "finite_targets": True,
    "valid_edge_indices": True,
    "consistent_node_count": True,
    "consistent_feature_count": True,
    "validated": True,
} # Registrar auditoría de GraphData

# 15.4.10 Mostrar resultado
print(f"GraphData auditados         : {len(graphs)}") # Mostrar cantidad de GraphData
print(f"Nodos por GraphData         : {n_nodes}") # Mostrar cantidad de nodos
print(f"Variables predictoras       : {in_channels}") # Mostrar variables
print(f"Variables objetivo          : {out_channels}") # Mostrar salida
print(f"Aristas GraphData referencia: {n_edges_reference}") # Mostrar aristas
print("Características              : VÁLIDAS") # Confirmar características
print("Variables objetivo           : VÁLIDAS") # Confirmar objetivos
print("Topología                    : VÁLIDA") # Confirmar topología
print("Integridad numérica          : VÁLIDA") # Confirmar valores
print("Compatibilidad GraphSAGE     : VALIDADA") # Confirmar compatibilidad
print("Auditoría GraphData          : APROBADA") # Confirmar auditoría

# 15.5 DIMENSIONES OFICIALES
# Objetivo: Consolidar y validar las dimensiones oficiales que utilizará GraphSAGE.
# Producto: Contrato dimensional oficial del Modelo GraphSAGE.

print("\n15.5 DIMENSIONES OFICIALES")

# 15.5.1 Validar dimensión de entrada
if not isinstance(
    in_channels,
    (int, np.integer)
):
    raise TypeError(
        "in_channels debe ser un entero."
    ) # Validar tipo de entrada

in_channels = int(
    in_channels
) # Normalizar dimensión de entrada

if in_channels <= 0:
    raise ValueError(
        "in_channels debe ser mayor que cero."
    ) # Validar rango de entrada

# 15.5.2 Validar dimensión oculta
if not isinstance(
    hidden_channels,
    (int, np.integer)
):
    raise TypeError(
        "hidden_channels debe ser un entero."
    ) # Validar tipo de dimensión oculta

hidden_channels = int(
    hidden_channels
) # Normalizar dimensión oculta

if hidden_channels <= 0:
    raise ValueError(
        "hidden_channels debe ser mayor que cero."
    ) # Validar rango de dimensión oculta

# 15.5.3 Validar dimensión de salida
if not isinstance(
    out_channels,
    (int, np.integer)
):
    raise TypeError(
        "out_channels debe ser un entero."
    ) # Validar tipo de salida

out_channels = int(
    out_channels
) # Normalizar dimensión de salida

if out_channels <= 0:
    raise ValueError(
        "out_channels debe ser mayor que cero."
    ) # Validar rango de salida

# 15.5.4 Validar cantidad oficial de nodos
if not isinstance(
    n_nodes,
    (int, np.integer)
):
    raise TypeError(
        "n_nodes debe ser un entero."
    ) # Validar tipo de nodos

n_nodes = int(
    n_nodes
) # Normalizar cantidad de nodos

if n_nodes <= 0:
    raise ValueError(
        "n_nodes debe ser mayor que cero."
    ) # Validar rango de nodos

# 15.5.5 Validar correspondencia con GraphData
if reference_graph.x.shape[1] != in_channels:
    raise RuntimeError(
        "in_channels no coincide con la dimensión de GraphData.x."
    ) # Validar correspondencia de entrada

if reference_graph.x.shape[0] != n_nodes:
    raise RuntimeError(
        "n_nodes no coincide con el número de nodos del GraphData de referencia."
    ) # Validar correspondencia de nodos

# 15.5.6 Validar correspondencia con configuración oficial
if configured_in_channels is not None:
    if configured_in_channels != in_channels:
        raise RuntimeError(
            "La dimensión de entrada configurada por el Benchmark "
            f"({configured_in_channels}) no coincide con la dimensión "
            f"real de GraphData ({in_channels})."
        ) # Validar contrato de entrada

if configured_out_channels != out_channels:
    raise RuntimeError(
        "La dimensión de salida configurada por el Benchmark "
        f"({configured_out_channels}) no coincide con la dimensión "
        f"oficial ({out_channels})."
    ) # Validar contrato de salida

# 15.5.7 Construir contrato dimensional oficial
official_dimensions = {
    "in_channels": int(in_channels),
    "hidden_channels": int(hidden_channels),
    "out_channels": int(out_channels),
    "n_nodes": int(n_nodes),
} # Construir contrato dimensional

# 15.5.8 Validar contrato dimensional
required_dimension_fields = [
    "in_channels",
    "hidden_channels",
    "out_channels",
    "n_nodes",
] # Definir campos dimensionales obligatorios

missing_dimension_fields = [
    field
    for field in required_dimension_fields
    if field not in official_dimensions
] # Identificar dimensiones faltantes

if missing_dimension_fields:
    raise RuntimeError(
        "El contrato dimensional oficial está incompleto: "
        f"{missing_dimension_fields}"
    ) # Validar contrato dimensional

# 15.5.9 Mostrar dimensiones oficiales
print(f"Nodos                     : {n_nodes}") # Mostrar cantidad de nodos
print(f"Variables de entrada      : {in_channels}") # Mostrar dimensión de entrada
print(f"Dimensión oculta          : {hidden_channels}") # Mostrar dimensión oculta
print(f"Variables de salida       : {out_channels}") # Mostrar dimensión de salida
print("Dimensiones GraphSAGE      : VALIDADAS") # Confirmar dimensiones
print("Contrato dimensional       : APROBADO") # Confirmar contrato

# 15.6 CONSTRUCCIÓN DE GRAPHSAGE
# Objetivo: Construir el Modelo Oficial GraphSAGE utilizando exclusivamente la configuración y dimensiones auditadas.
# Producto: OfficialGraphSAGE construido y preparado para la validación estructural.

print("\n15.6 CONSTRUCCIÓN DE GRAPHSAGE")

# 15.6.1 Validar dimensiones oficiales
if not isinstance(
    official_dimensions,
    dict
):
    raise TypeError(
        "official_dimensions debe ser un diccionario."
    ) # Validar contrato dimensional

required_dimension_fields = [
    "in_channels",
    "hidden_channels",
    "out_channels",
    "n_nodes",
] # Definir dimensiones obligatorias

missing_dimension_fields = [
    field
    for field in required_dimension_fields
    if field not in official_dimensions
] # Identificar dimensiones faltantes

if missing_dimension_fields:
    raise RuntimeError(
        "El contrato dimensional está incompleto: "
        f"{missing_dimension_fields}"
    ) # Validar dimensiones

# 15.6.2 Recuperar dimensiones oficiales
in_channels = int(
    official_dimensions[
        "in_channels"
    ]
) # Recuperar dimensión de entrada

hidden_channels = int(
    official_dimensions[
        "hidden_channels"
    ]
) # Recuperar dimensión oculta

out_channels = int(
    official_dimensions[
        "out_channels"
    ]
) # Recuperar dimensión de salida

n_nodes = int(
    official_dimensions[
        "n_nodes"
    ]
) # Recuperar cantidad de nodos

# 15.6.3 Validar dimensiones
if in_channels <= 0:
    raise ValueError(
        "in_channels debe ser mayor que cero."
    ) # Validar dimensión de entrada

if hidden_channels <= 0:
    raise ValueError(
        "hidden_channels debe ser mayor que cero."
    ) # Validar dimensión oculta

if out_channels <= 0:
    raise ValueError(
        "out_channels debe ser mayor que cero."
    ) # Validar dimensión de salida

if n_nodes <= 0:
    raise ValueError(
        "n_nodes debe ser mayor que cero."
    ) # Validar cantidad de nodos

# 15.6.4 Construir Modelo Oficial GraphSAGE
official_graphsage = OfficialGraphSAGE(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    dropout=dropout,
) # Construir arquitectura oficial GraphSAGE

# 15.6.5 Transferir modelo al dispositivo
official_graphsage = official_graphsage.to(
    device
) # Transferir Modelo Oficial al dispositivo de ejecución

# 15.6.6 Validar dispositivo del modelo
model_device = next(
    official_graphsage.parameters()
).device # Recuperar dispositivo del modelo

if model_device != device:
    raise RuntimeError(
        "OfficialGraphSAGE no fue transferido correctamente al dispositivo configurado."
    ) # Validar dispositivo

# 15.6.7 Construir registro de arquitectura
official_graphsage_architecture = {
    "model_code": OFFICIAL_MODEL_CODE,
    "model_name": OFFICIAL_MODEL_NAME,
    "family": OFFICIAL_MODEL_FAMILY,
    "in_channels": int(in_channels),
    "hidden_channels": int(hidden_channels),
    "out_channels": int(out_channels),
    "dropout": float(dropout),
    "n_nodes": int(n_nodes),
    "device": str(device),
} # Registrar arquitectura oficial

# 15.6.8 Calcular parámetros
total_parameters = int(
    sum(
        parameter.numel()
        for parameter in official_graphsage.parameters()
    )
) # Calcular parámetros totales

trainable_parameters = int(
    sum(
        parameter.numel()
        for parameter in official_graphsage.parameters()
        if parameter.requires_grad
    )
) # Calcular parámetros entrenables

if total_parameters <= 0:
    raise RuntimeError(
        "GraphSAGE no contiene parámetros."
    ) # Validar parámetros del modelo

if trainable_parameters <= 0:
    raise RuntimeError(
        "GraphSAGE no contiene parámetros entrenables."
    ) # Validar parámetros entrenables

# 15.6.9 Mostrar resultado
print(f"Modelo Oficial            : {OFFICIAL_MODEL_NAME}") # Mostrar modelo
print(f"Código Oficial            : {OFFICIAL_MODEL_CODE}") # Mostrar código
print(f"Familia                   : {OFFICIAL_MODEL_FAMILY}") # Mostrar familia
print(f"Variables de entrada      : {in_channels}") # Mostrar dimensión de entrada
print(f"Dimensión oculta          : {hidden_channels}") # Mostrar dimensión oculta
print(f"Variables de salida       : {out_channels}") # Mostrar dimensión de salida
print(f"Dropout                   : {dropout:.6f}") # Mostrar Dropout
print(f"Nodos                     : {n_nodes}") # Mostrar cantidad de nodos
print(f"Parámetros totales        : {total_parameters:,}") # Mostrar parámetros totales
print(f"Parámetros entrenables    : {trainable_parameters:,}") # Mostrar parámetros entrenables
print(f"Dispositivo               : {model_device}") # Mostrar dispositivo
print("GraphSAGE                  : CONSTRUIDO") # Confirmar construcción

# 15.7 VALIDACIÓN ESTRUCTURAL DE GRAPHSAGE
# Objetivo: Validar la arquitectura, dimensiones, parámetros y ejecución forward del Modelo Oficial.
# Producto: OfficialGraphSAGE estructuralmente validado y listo para configurar el entrenamiento.

print("\n15.7 VALIDACIÓN ESTRUCTURAL DE GRAPHSAGE")

# 15.7.1 Validar instancia del modelo
if not isinstance(
    official_graphsage,
    OfficialGraphSAGE
):
    raise TypeError(
        "official_graphsage debe ser una instancia de OfficialGraphSAGE."
    ) # Validar arquitectura oficial

# 15.7.2 Validar dispositivo
model_device = next(
    official_graphsage.parameters()
).device # Recuperar dispositivo del modelo

if model_device != device:
    raise RuntimeError(
        "El dispositivo del modelo no coincide con el dispositivo configurado."
    ) # Validar dispositivo

# 15.7.3 Validar capas GraphSAGE
if not isinstance(
    official_graphsage.sage1,
    SAGEConv
):
    raise TypeError(
        "sage1 debe ser una instancia de SAGEConv."
    ) # Validar primera capa GraphSAGE

if not isinstance(
    official_graphsage.sage2,
    SAGEConv
):
    raise TypeError(
        "sage2 debe ser una instancia de SAGEConv."
    ) # Validar segunda capa GraphSAGE

# 15.7.4 Validar capa de salida
if not isinstance(
    official_graphsage.output_layer,
    torch.nn.Linear
):
    raise TypeError(
        "output_layer debe ser una instancia de torch.nn.Linear."
    ) # Validar capa de salida

# 15.7.5 Validar Dropout
if not isinstance(
    official_graphsage.dropout,
    torch.nn.Dropout
):
    raise TypeError(
        "dropout debe ser una instancia de torch.nn.Dropout."
    ) # Validar capa Dropout

if not np.isclose(
    float(
        official_graphsage.dropout.p
    ),
    dropout,
):
    raise RuntimeError(
        "El Dropout del modelo no coincide con la configuración oficial."
    ) # Validar Dropout

# 15.7.6 Validar primera capa GraphSAGE
if official_graphsage.sage1.in_channels != in_channels:
    raise RuntimeError(
        "La primera capa GraphSAGE presenta una dimensión de entrada "
        "incompatible con in_channels."
    ) # Validar entrada de primera capa

if official_graphsage.sage1.out_channels != hidden_channels:
    raise RuntimeError(
        "La primera capa GraphSAGE presenta una dimensión de salida "
        "incompatible con hidden_channels."
    ) # Validar salida de primera capa

# 15.7.7 Validar segunda capa GraphSAGE
if official_graphsage.sage2.in_channels != hidden_channels:
    raise RuntimeError(
        "La segunda capa GraphSAGE presenta una dimensión de entrada "
        "incompatible con hidden_channels."
    ) # Validar entrada de segunda capa

if official_graphsage.sage2.out_channels != hidden_channels:
    raise RuntimeError(
        "La segunda capa GraphSAGE presenta una dimensión de salida "
        "incompatible con hidden_channels."
    ) # Validar salida de segunda capa

# 15.7.8 Validar capa de salida
if official_graphsage.output_layer.in_features != hidden_channels:
    raise RuntimeError(
        "La capa de salida presenta una dimensión de entrada incompatible "
        "con hidden_channels."
    ) # Validar entrada de salida

if official_graphsage.output_layer.out_features != out_channels:
    raise RuntimeError(
        "La capa de salida presenta una dimensión de salida incompatible "
        "con out_channels."
    ) # Validar salida

# 15.7.9 Validar parámetros entrenables
model_parameters = list(
    official_graphsage.parameters()
) # Recuperar parámetros del modelo

if len(
    model_parameters
) == 0:
    raise RuntimeError(
        "OfficialGraphSAGE no contiene parámetros."
    ) # Validar existencia de parámetros

trainable_model_parameters = [
    parameter
    for parameter in model_parameters
    if parameter.requires_grad
] # Recuperar parámetros entrenables

if len(
    trainable_model_parameters
) == 0:
    raise RuntimeError(
        "OfficialGraphSAGE no contiene parámetros entrenables."
    ) # Validar capacidad de aprendizaje

# 15.7.10 Validar integridad numérica de parámetros
for parameter_name, parameter in official_graphsage.named_parameters():
    if not torch.isfinite(
        parameter
    ).all():
        raise RuntimeError(
            f"El parámetro '{parameter_name}' contiene NaN o infinitos."
        ) # Validar integridad numérica

# 15.7.11 Validar GraphData de prueba estructural
if not isinstance(
    reference_graph,
    Data
):
    raise TypeError(
        "reference_graph debe ser una instancia de Data."
    ) # Validar GraphData de referencia

reference_x = reference_graph.x.to(
    model_device
) # Transferir características al dispositivo

reference_edge_index = reference_graph.edge_index.to(
    model_device
) # Transferir topología al dispositivo

if reference_x.shape[1] != in_channels:
    raise RuntimeError(
        "El GraphData de referencia no coincide con in_channels."
    ) # Validar entrada del GraphData

# 15.7.12 Ejecutar forward de prueba
official_graphsage.eval() # Establecer modo evaluación
with torch.no_grad():
    reference_prediction = official_graphsage(
        reference_x,
        reference_edge_index,
    ) # Ejecutar forward de prueba

# 15.7.13 Validar salida del forward
if not isinstance(
    reference_prediction,
    torch.Tensor
):
    raise TypeError(
        "La salida del Modelo Oficial debe ser un tensor."
    ) # Validar tipo de salida

if reference_prediction.ndim != 2:
    raise RuntimeError(
        "La salida de GraphSAGE debe ser bidimensional."
    ) # Validar dimensionalidad de salida

if reference_prediction.shape[0] != n_nodes:
    raise RuntimeError(
        "La salida de GraphSAGE no coincide con el número de nodos."
    ) # Validar cantidad de predicciones

if reference_prediction.shape[1] != out_channels:
    raise RuntimeError(
        "La salida de GraphSAGE no coincide con out_channels."
    ) # Validar dimensión de salida

if not torch.isfinite(
    reference_prediction
).all():
    raise RuntimeError(
        "El forward de GraphSAGE produjo valores NaN o infinitos."
    ) # Validar estabilidad numérica

# 15.7.14 Registrar validación estructural
graphsage_structural_validation = {
    "model_type": official_graphsage.__class__.__name__,
    "sage_layers": 2,
    "in_channels": int(in_channels),
    "hidden_channels": int(hidden_channels),
    "out_channels": int(out_channels),
    "dropout": float(dropout),
    "n_nodes": int(n_nodes),
    "total_parameters": int(total_parameters),
    "trainable_parameters": int(trainable_parameters),
    "forward_validated": True,
    "finite_parameters": True,
    "device": str(model_device),
    "validated": True,
} # Registrar validación estructural

# 15.7.15 Mostrar resultado
print(f"Modelo                     : {official_graphsage.__class__.__name__}") # Mostrar clase
print(f"Capas SAGEConv             : 2") # Mostrar capas GraphSAGE
print(f"Entrada                    : {in_channels}") # Mostrar entrada
print(f"Oculta                     : {hidden_channels}") # Mostrar dimensión oculta
print(f"Salida                     : {out_channels}") # Mostrar salida
print(f"Dropout                    : {dropout:.6f}") # Mostrar Dropout
print(f"Parámetros totales         : {total_parameters:,}") # Mostrar parámetros totales
print(f"Parámetros entrenables     : {trainable_parameters:,}") # Mostrar parámetros entrenables
print(f"Salida forward             : {tuple(reference_prediction.shape)}") # Mostrar forma de salida
print(f"Dispositivo                : {model_device}") # Mostrar dispositivo
print("Arquitectura SAGEConv       : VÁLIDA") # Confirmar capas
print("Parámetros                  : VÁLIDOS") # Confirmar parámetros
print("Forward                     : VÁLIDO") # Confirmar ejecución
print("Validación estructural      : APROBADA") # Confirmar validación

# 15.8 PREPARACIÓN DEL ENTRENAMIENTO
# Objetivo: Configurar el criterio de pérdida, optimizador y estructuras de seguimiento del entrenamiento.
# Producto: Entorno de entrenamiento GraphSAGE preparado para ejecutar el entrenamiento oficial.

print("\n15.8 PREPARACIÓN DEL ENTRENAMIENTO")

# 15.8.1 Validar disponibilidad del Modelo Oficial
if not isinstance(
    official_graphsage,
    OfficialGraphSAGE
):
    raise TypeError(
        "official_graphsage debe ser una instancia de OfficialGraphSAGE."
    ) # Validar disponibilidad del modelo

# 15.8.2 Validar parámetros entrenables
trainable_parameters_list = [
    parameter
    for parameter in official_graphsage.parameters()
    if parameter.requires_grad
] # Recuperar parámetros entrenables

if len(
    trainable_parameters_list
) == 0:
    raise RuntimeError(
        "OfficialGraphSAGE no contiene parámetros entrenables."
    ) # Validar capacidad de aprendizaje

# 15.8.3 Configurar función de pérdida
criterion = nn.MSELoss() # Configurar error cuadrático medio

# 15.8.4 Validar función de pérdida
if not isinstance(
    criterion,
    nn.modules.loss._Loss
):
    raise TypeError(
        "criterion debe ser una función de pérdida de PyTorch."
    ) # Validar criterio de pérdida

# 15.8.5 Configurar optimizador
optimizer = torch.optim.Adam(
    official_graphsage.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
) # Configurar optimizador Adam

# 15.8.6 Validar optimizador
if not isinstance(
    optimizer,
    torch.optim.Optimizer
):
    raise TypeError(
        "optimizer debe ser una instancia de torch.optim.Optimizer."
    ) # Validar optimizador

optimizer_parameter_count = sum(
    len(
        parameter_group["params"]
    )
    for parameter_group in optimizer.param_groups
) # Contar parámetros registrados

if optimizer_parameter_count == 0:
    raise RuntimeError(
        "El optimizador no contiene parámetros."
    ) # Validar parámetros del optimizador

# 15.8.7 Validar parámetros del optimizador
model_parameter_ids = {
    id(parameter)
    for parameter in trainable_parameters_list
} # Identificar parámetros entrenables

optimizer_parameter_ids = {
    id(parameter)
    for parameter_group in optimizer.param_groups
    for parameter in parameter_group["params"]
} # Identificar parámetros del optimizador

if not model_parameter_ids.issubset(
    optimizer_parameter_ids
):
    raise RuntimeError(
        "El optimizador no contiene todos los parámetros entrenables de GraphSAGE."
    ) # Validar conexión modelo-optimizador

# 15.8.8 Inicializar historial del entrenamiento
training_history = {
    "epoch": [],
    "train_loss": [],
    "validation_loss": [],
} # Inicializar historial científico

# 15.8.9 Inicializar mejor pérdida
best_validation_loss = float(
    "inf"
) # Inicializar mejor pérdida de validación

# 15.8.10 Inicializar mejor época
best_epoch = None # Inicializar mejor época

# 15.8.11 Inicializar mejor estado
best_model_state = None # Inicializar estado del mejor modelo

# 15.8.12 Inicializar contador de épocas
completed_epochs = 0 # Inicializar contador de épocas completadas

# 15.8.13 Construir configuración del entrenamiento
training_configuration = {
    "optimizer": optimizer.__class__.__name__,
    "criterion": criterion.__class__.__name__,
    "learning_rate": float(learning_rate),
    "weight_decay": float(weight_decay),
    "epochs": int(epochs),
    "model_code": OFFICIAL_MODEL_CODE,
    "model_name": OFFICIAL_MODEL_NAME,
    "model_family": OFFICIAL_MODEL_FAMILY,
    "in_channels": int(in_channels),
    "hidden_channels": int(hidden_channels),
    "out_channels": int(out_channels),
    "dropout": float(dropout),
} # Registrar configuración del entrenamiento

# 15.8.14 Validar configuración del entrenamiento
required_training_configuration_fields = [
    "optimizer",
    "criterion",
    "learning_rate",
    "weight_decay",
    "epochs",
    "model_code",
    "model_name",
    "model_family",
    "in_channels",
    "hidden_channels",
    "out_channels",
    "dropout",
] # Definir contrato de entrenamiento

missing_training_configuration_fields = [
    field
    for field in required_training_configuration_fields
    if field not in training_configuration
] # Identificar campos faltantes

if missing_training_configuration_fields:
    raise RuntimeError(
        "La configuración del entrenamiento está incompleta: "
        f"{missing_training_configuration_fields}"
    ) # Validar configuración

# 15.8.15 Mostrar configuración
print(f"Modelo Oficial             : {OFFICIAL_MODEL_NAME}") # Mostrar modelo
print(f"Optimizer                  : {optimizer.__class__.__name__}") # Mostrar optimizador
print(f"Criterion                  : {criterion.__class__.__name__}") # Mostrar función de pérdida
print(f"Learning Rate              : {learning_rate:.8f}") # Mostrar Learning Rate
print(f"Weight Decay               : {weight_decay:.8f}") # Mostrar regularización
print(f"Épocas                     : {epochs}") # Mostrar épocas
print(f"Parámetros entrenables     : {len(trainable_parameters_list):,}") # Mostrar cantidad de tensores entrenables
print("Historial                   : INICIALIZADO") # Confirmar historial
print("Mejor estado                : INICIALIZADO") # Confirmar mejor estado
print("Preparación entrenamiento   : APROBADA") # Confirmar preparación

# 15.9 ENTRENAMIENTO OFICIAL
# Objetivo: Entrenar GraphSAGE sobre los GraphData de entrenamiento y seleccionar el mejor estado mediante la pérdida de validación.
# Producto: Modelo GraphSAGE entrenado, historial científico y mejor estado del modelo.

print("\n15.9 ENTRENAMIENTO OFICIAL")

# 15.9.1 Validar disponibilidad de las entradas
if not isinstance(
    train_indices,
    list
):
    raise TypeError(
        "train_indices debe ser una lista."
    ) # Validar índices de entrenamiento

if not isinstance(
    validation_indices,
    list
):
    raise TypeError(
        "validation_indices debe ser una lista."
    ) # Validar índices de validación

if len(
    train_indices
) == 0:
    raise RuntimeError(
        "train_indices está vacío."
    ) # Validar entrenamiento

if len(
    validation_indices
) == 0:
    raise RuntimeError(
        "validation_indices está vacío."
    ) # Validar validación

# 15.9.2 Validar modelo y optimizador
if not isinstance(
    official_graphsage,
    OfficialGraphSAGE
):
    raise TypeError(
        "official_graphsage debe ser una instancia de OfficialGraphSAGE."
    ) # Validar modelo

if not isinstance(
    optimizer,
    torch.optim.Optimizer
):
    raise TypeError(
        "optimizer debe ser una instancia de torch.optim.Optimizer."
    ) # Validar optimizador

if not isinstance(
    criterion,
    nn.modules.loss._Loss
):
    raise TypeError(
        "criterion debe ser una función de pérdida válida."
    ) # Validar criterio

# 15.9.3 Reinicializar estado de entrenamiento
training_history = {
    "epoch": [],
    "train_loss": [],
    "validation_loss": [],
} # Inicializar historial científico

best_validation_loss = float(
    "inf"
) # Inicializar mejor pérdida de validación

best_epoch = None # Inicializar mejor época
best_model_state = None # Inicializar mejor estado del modelo
completed_epochs = 0 # Inicializar contador de épocas

# 15.9.4 Ejecutar entrenamiento por épocas
for epoch in range(
    1,
    epochs + 1
):
    official_graphsage.train() # Activar modo entrenamiento
    epoch_train_losses = [] # Inicializar pérdidas de entrenamiento de la época
    for graph_index in train_indices:
        graph = graphs[
            graph_index
        ] # Recuperar GraphData de entrenamiento

        graph_x = graph.x.to(
            device
        ) # Transferir características al dispositivo

        graph_edge_index = graph.edge_index.to(
            device
        ) # Transferir topología al dispositivo

        graph_y = graph.y.to(
            device
        ) # Transferir objetivo al dispositivo

        optimizer.zero_grad() # Reiniciar gradientes
        prediction = official_graphsage(
            graph_x,
            graph_edge_index,
        ) # Ejecutar forward

        if prediction.ndim == 1:
            prediction = prediction.view(
                -1,
                1
            ) # Normalizar dimensión de predicción

        target = graph_y
        if target.ndim == 1:
            target = target.view(
                -1,
                1
            ) # Normalizar dimensión del objetivo

        if prediction.shape != target.shape:
            raise RuntimeError(
                f"Epoch {epoch}, GraphData {graph_index}: "
                f"predicción {tuple(prediction.shape)} incompatible con "
                f"objetivo {tuple(target.shape)}."
            ) # Validar compatibilidad de salida

        if not torch.isfinite(
            prediction
        ).all():
            raise RuntimeError(
                f"Epoch {epoch}, GraphData {graph_index}: "
                "la predicción contiene NaN o infinitos."
            ) # Validar predicción

        loss = criterion(
            prediction,
            target,
        ) # Calcular pérdida de entrenamiento

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"Epoch {epoch}, GraphData {graph_index}: "
                "la pérdida contiene NaN o infinitos."
            ) # Validar pérdida

        loss.backward() # Calcular gradientes
        optimizer.step() # Actualizar parámetros
        epoch_train_losses.append(
            float(
                loss.detach().cpu().item()
            )
        ) # Registrar pérdida del GraphData

    # 15.9.5 Calcular pérdida media de entrenamiento
    if len(
        epoch_train_losses
    ) == 0:
        raise RuntimeError(
            f"Epoch {epoch}: no se generaron pérdidas de entrenamiento."
        ) # Validar resultados de la época

    mean_train_loss = float(
        np.mean(
            epoch_train_losses
        )
    ) # Calcular pérdida media de entrenamiento

    # 15.9.6 Ejecutar validación
    official_graphsage.eval() # Activar modo evaluación
    epoch_validation_losses = [] # Inicializar pérdidas de validación
    with torch.no_grad():
        for graph_index in validation_indices:
            graph = graphs[
                graph_index
            ] # Recuperar GraphData de validación

            graph_x = graph.x.to(
                device
            ) # Transferir características al dispositivo

            graph_edge_index = graph.edge_index.to(
                device
            ) # Transferir topología al dispositivo

            graph_y = graph.y.to(
                device
            ) # Transferir objetivo al dispositivo

            prediction = official_graphsage(
                graph_x,
                graph_edge_index,
            ) # Ejecutar predicción de validación

            if prediction.ndim == 1:
                prediction = prediction.view(
                    -1,
                    1
                ) # Normalizar predicción

            target = graph_y
            if target.ndim == 1:
                target = target.view(
                    -1,
                    1
                ) # Normalizar objetivo

            if prediction.shape != target.shape:
                raise RuntimeError(
                    f"Epoch {epoch}, GraphData {graph_index}: "
                    f"predicción {tuple(prediction.shape)} incompatible con "
                    f"objetivo {tuple(target.shape)}."
                ) # Validar salida de validación
            validation_loss = criterion(
                prediction,
                target,
            ) # Calcular pérdida de validación

            if not torch.isfinite(
                validation_loss
            ):
                raise RuntimeError(
                    f"Epoch {epoch}, GraphData {graph_index}: "
                    "la pérdida de validación contiene NaN o infinitos."
                ) # Validar pérdida de validación
            epoch_validation_losses.append(
                float(
                    validation_loss.detach().cpu().item()
                )
            ) # Registrar pérdida de validación

    # 15.9.7 Calcular pérdida media de validación
    if len(
        epoch_validation_losses
    ) == 0:
        raise RuntimeError(
            f"Epoch {epoch}: no se generaron pérdidas de validación."
        ) # Validar resultados de validación

    mean_validation_loss = float(
        np.mean(
            epoch_validation_losses
        )
    ) # Calcular pérdida media de validación

    # 15.9.8 Registrar historial
    training_history[
        "epoch"
    ].append(
        epoch
    ) # Registrar época

    training_history[
        "train_loss"
    ].append(
        mean_train_loss
    ) # Registrar pérdida de entrenamiento

    training_history[
        "validation_loss"
    ].append(
        mean_validation_loss
    ) # Registrar pérdida de validación

    # 15.9.9 Actualizar mejor estado
    if mean_validation_loss < best_validation_loss:
        best_validation_loss = mean_validation_loss # Actualizar mejor pérdida
        best_epoch = epoch # Registrar mejor época
        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in official_graphsage.state_dict().items()
        } # Guardar copia independiente del mejor estado

    # 15.9.10 Actualizar contador
    completed_epochs = epoch # Registrar época completada

    # 15.9.11 Mostrar progreso
    print(f"Epoch {epoch:04d}/{epochs:04d} | Train Loss: {mean_train_loss:.8f} | Validation Loss: {mean_validation_loss:.8f}") # Mostrar progreso

# 15.9.12 Validar finalización
if completed_epochs != epochs:
    raise RuntimeError(
        "El entrenamiento no completó el número oficial de épocas."
    ) # Validar finalización

# 15.9.13 Validar existencia del mejor estado
if best_model_state is None:
    raise RuntimeError(
        "No fue posible identificar un mejor estado del Modelo Oficial."
    ) # Validar selección del mejor modelo

if best_epoch is None:
    raise RuntimeError(
        "No fue posible determinar la mejor época."
    ) # Validar mejor época

# 15.9.14 Validar historial
if len(
    training_history["epoch"]
) != epochs:
    raise RuntimeError(
        "El historial no contiene todas las épocas."
    ) # Validar historial de épocas

if len(
    training_history["train_loss"]
) != epochs:
    raise RuntimeError(
        "El historial de entrenamiento está incompleto."
    ) # Validar historial de entrenamiento

if len(
    training_history["validation_loss"]
) != epochs:
    raise RuntimeError(
        "El historial de validación está incompleto."
    ) # Validar historial de validación

# 15.9.15 Registrar resultado del entrenamiento
official_training_result = {
    "completed_epochs": int(completed_epochs),
    "best_epoch": int(best_epoch),
    "best_validation_loss": float(best_validation_loss),
    "training_graphs": int(len(train_indices)),
    "validation_graphs": int(len(validation_indices)),
    "test_graphs": int(len(test_index)),
    "test_used_during_training": False,
    "training_completed": True,
} # Registrar resultado científico del entrenamiento

# 15.9.16 Mostrar resultado final
print(f"Épocas completadas         : {completed_epochs}") # Mostrar épocas completadas
print(f"Mejor época                : {best_epoch}") # Mostrar mejor época
print(f"Mejor Validation Loss      : {best_validation_loss:.8f}") # Mostrar mejor pérdida
print(f"GraphData entrenamiento    : {len(train_indices)}") # Mostrar entrenamiento
print(f"GraphData validación       : {len(validation_indices)}") # Mostrar validación
print(f"GraphData prueba           : {len(test_index)}") # Mostrar prueba
print("Prueba durante entrenamiento: NO") # Confirmar aislamiento del test
print("Entrenamiento GraphSAGE     : COMPLETADO") # Confirmar entrenamiento

# 15.10 VALIDACIÓN DEL ENTRENAMIENTO
# Objetivo: Validar la integridad y coherencia del entrenamiento oficial de GraphSAGE.
# Producto: Resultado de entrenamiento validado antes de restaurar el mejor estado.

print("\n15.10 VALIDACIÓN DEL ENTRENAMIENTO")

# 15.10.1 Validar finalización del entrenamiento
if not isinstance(
    official_training_result,
    dict
):
    raise TypeError(
        "official_training_result debe ser un diccionario."
    ) # Validar resultado del entrenamiento

if not official_training_result.get(
    "training_completed",
    False
):
    raise RuntimeError(
        "El entrenamiento oficial no fue marcado como completado."
    ) # Validar finalización

# 15.10.2 Validar número de épocas
if completed_epochs != epochs:
    raise RuntimeError(
        f"El entrenamiento completó {completed_epochs} épocas de {epochs}."
    ) # Validar número de épocas

# 15.10.3 Validar existencia del historial
if not isinstance(
    training_history,
    dict
):
    raise TypeError(
        "training_history debe ser un diccionario."
    ) # Validar estructura del historial

required_history_fields = [
    "epoch",
    "train_loss",
    "validation_loss",
] # Definir campos obligatorios del historial

missing_history_fields = [
    field
    for field in required_history_fields
    if field not in training_history
] # Identificar campos faltantes

if missing_history_fields:
    raise RuntimeError(
        "training_history está incompleto: "
        f"{missing_history_fields}"
    ) # Validar estructura del historial

# 15.10.4 Validar longitud del historial
history_length = len(
    training_history["epoch"]
) # Determinar longitud del historial

if history_length != epochs:
    raise RuntimeError(
        f"El historial contiene {history_length} épocas y se esperaban {epochs}."
    ) # Validar cantidad de épocas

if len(
    training_history["train_loss"]
) != epochs:
    raise RuntimeError(
        "train_loss no contiene todas las épocas."
    ) # Validar pérdidas de entrenamiento

if len(
    training_history["validation_loss"]
) != epochs:
    raise RuntimeError(
        "validation_loss no contiene todas las épocas."
    ) # Validar pérdidas de validación

# 15.10.5 Convertir pérdidas a valores numéricos
train_losses = np.asarray(
    training_history["train_loss"],
    dtype=float,
) # Convertir pérdidas de entrenamiento

validation_losses = np.asarray(
    training_history["validation_loss"],
    dtype=float,
) # Convertir pérdidas de validación

epoch_values = np.asarray(
    training_history["epoch"],
    dtype=int,
) # Convertir épocas

# 15.10.6 Validar finitud de las pérdidas
if not np.isfinite(
    train_losses
).all():
    raise RuntimeError(
        "El historial de entrenamiento contiene NaN o infinitos."
    ) # Validar estabilidad del entrenamiento

if not np.isfinite(
    validation_losses
).all():
    raise RuntimeError(
        "El historial de validación contiene NaN o infinitos."
    ) # Validar estabilidad de validación

# 15.10.7 Validar secuencia de épocas
expected_epoch_values = np.arange(
    1,
    epochs + 1,
    dtype=int,
) # Construir secuencia esperada

if not np.array_equal(
    epoch_values,
    expected_epoch_values,
):
    raise RuntimeError(
        "La secuencia de épocas del historial no es continua."
    ) # Validar continuidad temporal del entrenamiento

# 15.10.8 Validar pérdidas no negativas
if np.any(
    train_losses < 0.0
):
    raise RuntimeError(
        "El historial de entrenamiento contiene pérdidas negativas."
    ) # Validar rango de pérdida

if np.any(
    validation_losses < 0.0
):
    raise RuntimeError(
        "El historial de validación contiene pérdidas negativas."
    ) # Validar rango de pérdida

# 15.10.9 Determinar mejor época desde el historial
calculated_best_index = int(
    np.argmin(
        validation_losses
    )
) # Determinar posición de menor pérdida

calculated_best_epoch = int(
    epoch_values[
        calculated_best_index
    ]
) # Determinar mejor época

calculated_best_validation_loss = float(
    validation_losses[
        calculated_best_index
    ]
) # Determinar mejor pérdida

# 15.10.10 Validar mejor época registrada
if best_epoch != calculated_best_epoch:
    raise RuntimeError(
        f"best_epoch={best_epoch} no coincide con la mejor época calculada "
        f"({calculated_best_epoch})."
    ) # Validar selección de mejor época

# 15.10.11 Validar mejor pérdida registrada
if not np.isclose(
    best_validation_loss,
    calculated_best_validation_loss,
    rtol=1e-7,
    atol=1e-12,
):
    raise RuntimeError(
        "best_validation_loss no coincide con la pérdida mínima registrada."
    ) # Validar mejor pérdida

# 15.10.12 Validar mejor estado del modelo
if best_model_state is None:
    raise RuntimeError(
        "best_model_state no existe después del entrenamiento."
    ) # Validar estado del mejor modelo

if not isinstance(
    best_model_state,
    dict
):
    raise TypeError(
        "best_model_state debe ser un diccionario."
    ) # Validar estructura del estado

# 15.10.13 Validar correspondencia de parámetros
current_state_keys = set(
    official_graphsage.state_dict().keys()
) # Obtener claves actuales

best_state_keys = set(
    best_model_state.keys()
) # Obtener claves del mejor estado

if current_state_keys != best_state_keys:
    raise RuntimeError(
        "best_model_state no contiene las mismas claves que el modelo actual."
    ) # Validar estructura de parámetros

# 15.10.14 Validar integridad numérica del mejor estado
for parameter_name, parameter_value in best_model_state.items():
    if not isinstance(
        parameter_value,
        torch.Tensor
    ):
        raise TypeError(
            f"El parámetro '{parameter_name}' del mejor estado no es un tensor."
        ) # Validar tipo de parámetro

    if not torch.isfinite(
        parameter_value
    ).all():
        raise RuntimeError(
            f"El parámetro '{parameter_name}' del mejor estado contiene NaN o infinitos."
        ) # Validar integridad numérica

# 15.10.15 Validar correspondencia del mejor estado
for parameter_name, current_parameter in official_graphsage.state_dict().items():

    best_parameter = best_model_state[
        parameter_name
    ] # Recuperar parámetro del mejor estado

    if current_parameter.shape != best_parameter.shape:
        raise RuntimeError(
            f"El parámetro '{parameter_name}' presenta dimensiones incompatibles."
        ) # Validar dimensiones del parámetro

# 15.10.16 Construir resumen de validación
training_validation_result = {
    "training_completed": True,
    "completed_epochs": int(completed_epochs),
    "history_length": int(history_length),
    "best_epoch": int(best_epoch),
    "best_validation_loss": float(best_validation_loss),
    "best_state_available": True,
    "best_state_valid": True,
    "history_finite": True,
    "epoch_sequence_valid": True,
    "validated": True,
} # Registrar validación del entrenamiento

# 15.10.17 Mostrar resultado
print(f"Épocas esperadas             : {epochs}") # Mostrar épocas esperadas
print(f"Épocas completadas           : {completed_epochs}") # Mostrar épocas completadas
print(f"Historial registrado         : {history_length}") # Mostrar historial
print(f"Mejor época                  : {best_epoch}") # Mostrar mejor época
print(f"Mejor Validation Loss        : {best_validation_loss:.8f}") # Mostrar mejor pérdida
print("Pérdidas numéricamente válidas: SÍ") # Confirmar integridad de pérdidas
print("Secuencia de épocas           : VÁLIDA") # Confirmar secuencia
print("Mejor estado                  : VÁLIDO") # Confirmar estado
print("Validación del entrenamiento  : APROBADA") # Confirmar validación

# 15.11 EVALUACIÓN DE VALIDACIÓN
# Objetivo: Restaurar el mejor estado de GraphSAGE y evaluar su desempeño sobre los GraphData de validación.
# Producto: Métricas oficiales de validación correspondientes al mejor estado del modelo.

print("\n15.11 EVALUACIÓN DE VALIDACIÓN")

# 15.11.1 Validar disponibilidad del mejor estado
if best_model_state is None:
    raise RuntimeError(
        "No existe best_model_state para realizar la evaluación de validación."
    ) # Validar disponibilidad del mejor modelo

# 15.11.2 Restaurar el mejor estado
official_graphsage.load_state_dict(
    best_model_state
) # Restaurar parámetros correspondientes a la mejor época

official_graphsage = official_graphsage.to(
    device
) # Garantizar ubicación correcta del modelo

# 15.11.3 Activar modo evaluación
official_graphsage.eval() # Desactivar comportamiento estocástico del entrenamiento

# 15.11.4 Inicializar contenedores de evaluación
validation_predictions = [] # Inicializar predicciones de validación
validation_targets = [] # Inicializar valores observados de validación
validation_losses = [] # Inicializar pérdidas de validación

# 15.11.5 Evaluar GraphData de validación
with torch.no_grad():
    for graph_index in validation_indices:
        graph = graphs[
            graph_index
        ] # Recuperar GraphData de validación

        graph_x = graph.x.to(
            device
        ) # Transferir características

        graph_edge_index = graph.edge_index.to(
            device
        ) # Transferir topología

        graph_y = graph.y.to(
            device
        ) # Transferir variable objetivo

        prediction = official_graphsage(
            graph_x,
            graph_edge_index,
        ) # Generar predicciones

        if prediction.ndim == 1:
            prediction = prediction.view(
                -1,
                1
            ) # Normalizar dimensión de predicción

        target = graph_y
        if target.ndim == 1:
            target = target.view(
                -1,
                1
            ) # Normalizar dimensión del objetivo

        if prediction.shape != target.shape:
            raise RuntimeError(
                f"GraphData {graph_index}: predicción "
                f"{tuple(prediction.shape)} incompatible con objetivo "
                f"{tuple(target.shape)}."
            ) # Validar dimensiones

        loss = criterion(
            prediction,
            target,
        ) # Calcular pérdida de validación

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"GraphData {graph_index}: la pérdida contiene NaN o infinitos."
            ) # Validar estabilidad numérica

        if not torch.isfinite(
            prediction
        ).all():
            raise RuntimeError(
                f"GraphData {graph_index}: las predicciones contienen NaN o infinitos."
            ) # Validar predicciones

        validation_predictions.append(
            prediction.detach().cpu().numpy().reshape(-1)
        ) # Almacenar predicciones

        validation_targets.append(
            target.detach().cpu().numpy().reshape(-1)
        ) # Almacenar observaciones

        validation_losses.append(
            float(
                loss.detach().cpu().item()
            )
        ) # Almacenar pérdida

# 15.11.6 Validar resultados de evaluación
if len(
    validation_predictions
) == 0:
    raise RuntimeError(
        "No se generaron predicciones sobre el conjunto de validación."
    ) # Validar predicciones

if len(
    validation_targets
) == 0:
    raise RuntimeError(
        "No se generaron valores objetivo sobre el conjunto de validación."
    ) # Validar objetivos

if len(
    validation_predictions
) != len(
    validation_targets
):
    raise RuntimeError(
        "El número de predicciones y objetivos de validación no coincide."
    ) # Validar correspondencia

# 15.11.7 Consolidar predicciones y objetivos
validation_predictions_array = np.concatenate(
    validation_predictions
) # Consolidar predicciones

validation_targets_array = np.concatenate(
    validation_targets
) # Consolidar objetivos

# 15.11.8 Validar integridad numérica
if not np.isfinite(
    validation_predictions_array
).all():
    raise RuntimeError(
        "Las predicciones de validación contienen NaN o infinitos."
    ) # Validar predicciones

if not np.isfinite(
    validation_targets_array
).all():
    raise RuntimeError(
        "Los objetivos de validación contienen NaN o infinitos."
    ) # Validar objetivos

# 15.11.9 Calcular métricas de validación
validation_error = (
    validation_predictions_array
    - validation_targets_array
) # Calcular errores de predicción

validation_mse = float(
    np.mean(
        validation_error ** 2
    )
) # Calcular MSE

validation_rmse = float(
    np.sqrt(
        validation_mse
    )
) # Calcular RMSE

validation_mae = float(
    np.mean(
        np.abs(
            validation_error
        )
    )
) # Calcular MAE

# 15.11.10 Calcular MAPE con protección numérica
mape_denominator = np.abs(
    validation_targets_array
) # Obtener denominador del MAPE

mape_mask = (
    mape_denominator
    > np.finfo(
        float
    ).eps
) # Identificar valores válidos para MAPE

if np.any(
    mape_mask
):
    validation_mape = float(
        np.mean(
            np.abs(
                validation_error[
                    mape_mask
                ]
                / validation_targets_array[
                    mape_mask
                ]
            )
        )
    ) # Calcular MAPE sobre valores válidos
else:
    validation_mape = float(
        "nan"
    ) # Registrar MAPE no definido

# 15.11.11 Calcular R2
target_mean = float(
    np.mean(
        validation_targets_array
    )
) # Calcular media del objetivo

ss_res = float(
    np.sum(
        validation_error ** 2
    )
) # Calcular suma residual

ss_tot = float(
    np.sum(
        (
            validation_targets_array
            - target_mean
        ) ** 2
    )
) # Calcular suma total

if ss_tot > np.finfo(
    float
).eps:
    validation_r2 = float(
        1.0
        - (
            ss_res
            / ss_tot
        )
    ) # Calcular R2
else:
    validation_r2 = float(
        "nan"
    ) # Registrar R2 no definido

# 15.11.12 Calcular pérdida media por GraphData
validation_loss_mean = float(
    np.mean(
        validation_losses
    )
) # Calcular pérdida media de validación

# 15.11.13 Comparar con mejor pérdida registrada
if not np.isclose(
    validation_loss_mean,
    best_validation_loss,
    rtol=1e-5,
    atol=1e-8,
):
    raise RuntimeError(
        "La pérdida de validación final no coincide con "
        "best_validation_loss registrado."
    ) # Validar correspondencia del mejor estado

# 15.11.14 Construir métricas oficiales de validación
validation_metrics = {
    "loss": float(validation_loss_mean),
    "mse": float(validation_mse),
    "rmse": float(validation_rmse),
    "mae": float(validation_mae),
    "mape": float(validation_mape),
    "r2": float(validation_r2),
    "n_graphs": int(len(validation_indices)),
    "n_observations": int(len(validation_targets_array)),
    "best_epoch": int(best_epoch),
    "best_validation_loss": float(best_validation_loss),
    "evaluated_on_test": False,
} # Registrar métricas de validación

# 15.11.15 Construir resultado de evaluación
validation_evaluation_result = {
    "metrics": validation_metrics,
    "predictions": validation_predictions_array,
    "targets": validation_targets_array,
    "validated": True,
} # Registrar resultado de evaluación

# 15.11.16 Mostrar resultados
print(f"GraphData evaluados         : {len(validation_indices)}") # Mostrar GraphData evaluados
print(f"Observaciones evaluadas     : {len(validation_targets_array):,}") # Mostrar observaciones
print(f"Mejor época                 : {best_epoch}") # Mostrar mejor época
print(f"Validation Loss             : {validation_loss_mean:.8f}") # Mostrar pérdida
print(f"RMSE                        : {validation_rmse:.8f}") # Mostrar RMSE
print(f"MAE                         : {validation_mae:.8f}") # Mostrar MAE
print(f"MAPE                        : {validation_mape:.8f}") # Mostrar MAPE
print(f"R2                          : {validation_r2:.8f}") # Mostrar R2
print("Conjunto de prueba           : NO UTILIZADO") # Confirmar aislamiento
print("Evaluación validación        : APROBADA") # Confirmar evaluación

# 15.12 EVALUACIÓN DE PRUEBA
# Objetivo: Evaluar el Modelo Oficial GraphSAGE sobre el conjunto de prueba reservado.
# Producto: Métricas finales de prueba del Modelo Oficial.

print("\n15.12 EVALUACIÓN DE PRUEBA")

# 15.12.1 Validar disponibilidad del mejor estado
if best_model_state is None:
    raise RuntimeError(
        "No existe best_model_state para realizar la evaluación de prueba."
    ) # Validar disponibilidad del mejor modelo


# 15.12.2 Restaurar el mejor estado
official_graphsage.load_state_dict(
    best_model_state
) # Restaurar parámetros correspondientes a la mejor época

official_graphsage = official_graphsage.to(
    device
) # Garantizar ubicación correcta del modelo

# 15.12.3 Activar modo evaluación
official_graphsage.eval() # Activar modo evaluación

# 15.12.4 Inicializar contenedores de prueba
test_predictions = [] # Inicializar predicciones de prueba
test_targets = [] # Inicializar objetivos de prueba
test_losses = [] # Inicializar pérdidas de prueba

# 15.12.5 Evaluar GraphData de prueba
with torch.no_grad():
    for graph_index in test_indices:
        graph = graphs[
            graph_index
        ] # Recuperar GraphData de prueba

        graph_x = graph.x.to(
            device
        ) # Transferir características

        graph_edge_index = graph.edge_index.to(
            device
        ) # Transferir topología

        graph_y = graph.y.to(
            device
        ) # Transferir objetivo

        prediction = official_graphsage(
            graph_x,
            graph_edge_index,
        ) # Generar predicciones de prueba

        if prediction.ndim == 1:
            prediction = prediction.view(
                -1,
                1
            ) # Normalizar dimensión de predicción

        target = graph_y

        if target.ndim == 1:
            target = target.view(
                -1,
                1
            ) # Normalizar dimensión del objetivo

        if prediction.shape != target.shape:
            raise RuntimeError(
                f"GraphData {graph_index}: predicción "
                f"{tuple(prediction.shape)} incompatible con objetivo "
                f"{tuple(target.shape)}."
            ) # Validar dimensiones

        loss = criterion(
            prediction,
            target,
        ) # Calcular pérdida de prueba

        if not torch.isfinite(
            prediction
        ).all():
            raise RuntimeError(
                f"GraphData {graph_index}: las predicciones contienen NaN o infinitos."
            ) # Validar predicciones

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"GraphData {graph_index}: la pérdida contiene NaN o infinitos."
            ) # Validar pérdida

        test_predictions.append(
            prediction.detach().cpu().numpy().reshape(-1)
        ) # Almacenar predicciones

        test_targets.append(
            target.detach().cpu().numpy().reshape(-1)
        ) # Almacenar objetivos

        test_losses.append(
            float(
                loss.detach().cpu().item()
            )
        ) # Almacenar pérdida

# 15.12.6 Validar resultados
if len(
    test_predictions
) == 0:
    raise RuntimeError(
        "No se generaron predicciones sobre el conjunto de prueba."
    ) # Validar predicciones

if len(
    test_targets
) == 0:
    raise RuntimeError(
        "No se generaron objetivos sobre el conjunto de prueba."
    ) # Validar objetivos

if len(
    test_predictions
) != len(
    test_targets
):
    raise RuntimeError(
        "El número de predicciones y objetivos de prueba no coincide."
    ) # Validar correspondencia

# 15.12.7 Consolidar resultados
test_predictions_array = np.concatenate(
    test_predictions
) # Consolidar predicciones

test_targets_array = np.concatenate(
    test_targets
) # Consolidar objetivos

# 15.12.8 Validar integridad numérica
if not np.isfinite(
    test_predictions_array
).all():
    raise RuntimeError(
        "Las predicciones de prueba contienen NaN o infinitos."
    ) # Validar predicciones

if not np.isfinite(
    test_targets_array
).all():
    raise RuntimeError(
        "Los objetivos de prueba contienen NaN o infinitos."
    ) # Validar objetivos

# 15.12.9 Calcular errores
test_error = (
    test_predictions_array
    - test_targets_array
) # Calcular errores de predicción

# 15.12.10 Calcular MSE
test_mse = float(
    np.mean(
        test_error ** 2
    )
) # Calcular MSE

# 15.12.11 Calcular RMSE
test_rmse = float(
    np.sqrt(
        test_mse
    )
) # Calcular RMSE

# 15.12.12 Calcular MAE
test_mae = float(
    np.mean(
        np.abs(
            test_error
        )
    )
) # Calcular MAE

# 15.12.13 Calcular MAPE con protección numérica
test_mape_denominator = np.abs(
    test_targets_array
) # Obtener denominador del MAPE

test_mape_mask = (
    test_mape_denominator
    > np.finfo(
        float
    ).eps
) # Identificar valores válidos

if np.any(
    test_mape_mask
):

    test_mape = float(
        np.mean(
            np.abs(
                test_error[
                    test_mape_mask
                ]
                / test_targets_array[
                    test_mape_mask
                ]
            )
        )
    ) # Calcular MAPE

else:
    test_mape = float(
        "nan"
    ) # Registrar MAPE no definido

# 15.12.14 Calcular R2
test_target_mean = float(
    np.mean(
        test_targets_array
    )
) # Calcular media del objetivo

test_ss_res = float(
    np.sum(
        test_error ** 2
    )
) # Calcular suma residual

test_ss_tot = float(
    np.sum(
        (
            test_targets_array
            - test_target_mean
        ) ** 2
    )
) # Calcular suma total

if test_ss_tot > np.finfo(
    float
).eps:

    test_r2 = float(
        1.0
        - (
            test_ss_res
            / test_ss_tot
        )
    ) # Calcular R2

else:
    test_r2 = float(
        "nan"
    ) # Registrar R2 no definido

# 15.12.15 Calcular pérdida media
test_loss_mean = float(
    np.mean(
        test_losses
    )
) # Calcular pérdida media de prueba

# 15.12.16 Construir métricas finales de prueba
test_metrics = {
    "loss": float(test_loss_mean),
    "mse": float(test_mse),
    "rmse": float(test_rmse),
    "mae": float(test_mae),
    "mape": float(test_mape),
    "r2": float(test_r2),
    "n_graphs": int(len(test_indices)),
    "n_observations": int(len(test_targets_array)),
    "best_epoch": int(best_epoch),
    "test_used_during_training": False,
} # Registrar métricas de prueba

# 15.12.17 Construir resultado de evaluación
test_evaluation_result = {
    "metrics": test_metrics,
    "predictions": test_predictions_array,
    "targets": test_targets_array,
    "validated": True,
} # Registrar resultado de evaluación

# 15.12.18 Registrar separación científica del test
test_evaluation_result[
    "selection_source"
] = "validation" # Registrar que la selección provino de validación

test_evaluation_result[
    "test_reserved"
] = True # Confirmar reserva del conjunto de prueba

test_evaluation_result[
    "evaluated_after_training"
] = True # Confirmar evaluación posterior al entrenamiento

# 15.12.19 Mostrar resultados

print(f"GraphData evaluados         : {len(test_indices)}") # Mostrar GraphData evaluados
print(f"Observaciones evaluadas     : {len(test_targets_array):,}") # Mostrar observaciones
print(f"Mejor época utilizada       : {best_epoch}") # Mostrar mejor época
print(f"Test Loss                   : {test_loss_mean:.8f}") # Mostrar pérdida
print(f"RMSE                        : {test_rmse:.8f}") # Mostrar RMSE
print(f"MAE                         : {test_mae:.8f}") # Mostrar MAE
print(f"MAPE                        : {test_mape:.8f}") # Mostrar MAPE
print(f"R2                          : {test_r2:.8f}") # Mostrar R2
print("Test durante entrenamiento   : NO") # Confirmar aislamiento
print("Selección del modelo         : VALIDACIÓN") # Confirmar criterio de selección
print("Evaluación de prueba         : APROBADA") # Confirmar evaluación

# 15.13 MÉTRICAS OFICIALES
# Objetivo: Consolidar las métricas oficiales obtenidas sobre validación y prueba.
# Producto: official_model_metrics disponible para trazabilidad, comparación y persistencia.

print("\n15.13 MÉTRICAS OFICIALES")

# 15.13.1 Validar disponibilidad de métricas de validación
if not isinstance(
    validation_metrics,
    dict
):
    raise TypeError(
        "validation_metrics debe ser un diccionario."
    ) # Validar métricas de validación

# 15.13.2 Validar disponibilidad de métricas de prueba
if not isinstance(
    test_metrics,
    dict
):
    raise TypeError(
        "test_metrics debe ser un diccionario."
    ) # Validar métricas de prueba

# 15.13.3 Definir métricas obligatorias
required_metric_fields = [
    "loss",
    "mse",
    "rmse",
    "mae",
    "mape",
    "r2",
] # Definir contrato de métricas

# 15.13.4 Validar métricas de validación
missing_validation_metrics = [
    field
    for field in required_metric_fields
    if field not in validation_metrics
] # Identificar métricas de validación faltantes

if missing_validation_metrics:
    raise RuntimeError(
        "Las métricas de validación están incompletas: "
        f"{missing_validation_metrics}"
    ) # Validar métricas de validación

# 15.13.5 Validar métricas de prueba
missing_test_metrics = [
    field
    for field in required_metric_fields
    if field not in test_metrics
] # Identificar métricas de prueba faltantes

if missing_test_metrics:
    raise RuntimeError(
        "Las métricas de prueba están incompletas: "
        f"{missing_test_metrics}"
    ) # Validar métricas de prueba

# 15.13.6 Validar finitud de métricas fundamentales
for metric_name in [
    "loss",
    "mse",
    "rmse",
    "mae",
]:

    validation_value = float(
        validation_metrics[
            metric_name
        ]
    ) # Recuperar métrica de validación

    test_value = float(
        test_metrics[
            metric_name
        ]
    ) # Recuperar métrica de prueba

    if not np.isfinite(
        validation_value
    ):
        raise RuntimeError(
            f"La métrica de validación '{metric_name}' no es finita."
        ) # Validar métrica de validación

    if not np.isfinite(
        test_value
    ):
        raise RuntimeError(
            f"La métrica de prueba '{metric_name}' no es finita."
        ) # Validar métrica de prueba

# 15.13.7 Validar MAPE cuando esté definido
validation_mape = float(
    validation_metrics[
        "mape"
    ]
) # Recuperar MAPE de validación

test_mape = float(
    test_metrics[
        "mape"
    ]
) # Recuperar MAPE de prueba

if not np.isnan(
    validation_mape
):
    if not np.isfinite(
        validation_mape
    ):
        raise RuntimeError(
            "El MAPE de validación no es válido."
        ) # Validar MAPE de validación

if not np.isnan(
    test_mape
):
    if not np.isfinite(
        test_mape
    ):
        raise RuntimeError(
            "El MAPE de prueba no es válido."
        ) # Validar MAPE de prueba

# 15.13.8 Validar R2 cuando esté definido

validation_r2 = float(
    validation_metrics[
        "r2"
    ]
) # Recuperar R2 de validación

test_r2 = float(
    test_metrics[
        "r2"
    ]
) # Recuperar R2 de prueba

if not np.isnan(
    validation_r2
):
    if not np.isfinite(
        validation_r2
    ):
        raise RuntimeError(
            "El R2 de validación no es válido."
        ) # Validar R2 de validación

if not np.isnan(
    test_r2
):
    if not np.isfinite(
        test_r2
    ):
        raise RuntimeError(
            "El R2 de prueba no es válido."
        ) # Validar R2 de prueba

# 15.13.9 Validar coherencia de RMSE
validation_rmse_expected = float(
    np.sqrt(
        validation_metrics[
            "mse"
        ]
    )
) # Calcular RMSE esperado de validación

test_rmse_expected = float(
    np.sqrt(
        test_metrics[
            "mse"
        ]
    )
) # Calcular RMSE esperado de prueba

if not np.isclose(
    validation_metrics["rmse"],
    validation_rmse_expected,
    rtol=1e-7,
    atol=1e-10,
):
    raise RuntimeError(
        "El RMSE de validación no coincide con la raíz cuadrada del MSE."
    ) # Validar coherencia RMSE validación

if not np.isclose(
    test_metrics["rmse"],
    test_rmse_expected,
    rtol=1e-7,
    atol=1e-10,
):
    raise RuntimeError(
        "El RMSE de prueba no coincide con la raíz cuadrada del MSE."
    ) # Validar coherencia RMSE prueba

# 15.13.10 Construir métricas oficiales
official_model_metrics = {
    "validation": {
        "loss": float(validation_metrics["loss"]),
        "mse": float(validation_metrics["mse"]),
        "rmse": float(validation_metrics["rmse"]),
        "mae": float(validation_metrics["mae"]),
        "mape": float(validation_metrics["mape"]),
        "r2": float(validation_metrics["r2"]),
        "n_graphs": int(validation_metrics["n_graphs"]),
        "n_observations": int(validation_metrics["n_observations"]),
    },
    "test": {
        "loss": float(test_metrics["loss"]),
        "mse": float(test_metrics["mse"]),
        "rmse": float(test_metrics["rmse"]),
        "mae": float(test_metrics["mae"]),
        "mape": float(test_metrics["mape"]),
        "r2": float(test_metrics["r2"]),
        "n_graphs": int(test_metrics["n_graphs"]),
        "n_observations": int(test_metrics["n_observations"]),
    },
    "selection": {
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "selection_source": "validation",
    },
    "training": {
        "completed_epochs": int(completed_epochs),
        "training_graphs": int(len(train_indices)),
        "validation_graphs": int(len(validation_indices)),
        "test_graphs": int(len(test_indices)),
        "test_used_during_training": False,
    },
} # Consolidar métricas oficiales

# 15.13.11 Validar estructura de métricas oficiales
if not isinstance(
    official_model_metrics,
    dict
):
    raise TypeError(
        "official_model_metrics debe ser un diccionario."
    ) # Validar estructura

for section_name in [
    "validation",
    "test",
    "selection",
    "training",
]:
    if section_name not in official_model_metrics:
        raise RuntimeError(
            f"Falta la sección '{section_name}' en official_model_metrics."
        ) # Validar secciones

# 15.13.12 Registrar identidad del Modelo Oficial
official_model_metrics[
    "model"
] = {
    "model_code": OFFICIAL_MODEL_CODE,
    "model_name": OFFICIAL_MODEL_NAME,
    "family": OFFICIAL_MODEL_FAMILY,
} # Registrar identidad del modelo

# 15.13.13 Mostrar métricas oficiales
print(f"Modelo Oficial             : {OFFICIAL_MODEL_NAME}") # Mostrar modelo
print(f"Mejor época                : {best_epoch}") # Mostrar mejor época
print(f"Validation Loss            : {validation_metrics['loss']:.8f}") # Mostrar pérdida de validación
print(f"Validation RMSE            : {validation_metrics['rmse']:.8f}") # Mostrar RMSE de validación
print(f"Validation MAE             : {validation_metrics['mae']:.8f}") # Mostrar MAE de validación
print(f"Validation MAPE            : {validation_metrics['mape']:.8f}") # Mostrar MAPE de validación
print(f"Validation R2              : {validation_metrics['r2']:.8f}") # Mostrar R2 de validación
print(f"Test Loss                  : {test_metrics['loss']:.8f}") # Mostrar pérdida de prueba
print(f"Test RMSE                  : {test_metrics['rmse']:.8f}") # Mostrar RMSE de prueba
print(f"Test MAE                   : {test_metrics['mae']:.8f}") # Mostrar MAE de prueba
print(f"Test MAPE                  : {test_metrics['mape']:.8f}") # Mostrar MAPE de prueba
print(f"Test R2                    : {test_metrics['r2']:.8f}") # Mostrar R2 de prueba
print("Métricas oficiales          : CONSOLIDADAS") # Confirmar consolidación

# 15.14 COMPARACIÓN CON EL BENCHMARK
# Objetivo: Comparar las métricas obtenidas por el entrenamiento oficial de GraphSAGE con las métricas registradas por el Benchmark.
# Producto: Comparación cuantitativa y trazable entre Benchmark y entrenamiento oficial.

print("\n15.14 COMPARACIÓN CON EL BENCHMARK")

# 15.14.1 Validar disponibilidad del resultado del Benchmark
if not isinstance(
    official,
    dict
):
    raise TypeError(
        "official debe ser un diccionario."
    ) # Validar Modelo Oficial del Benchmark

# 15.14.2 Validar disponibilidad de métricas del Benchmark
if "metrics" not in official:
    raise RuntimeError(
        "El Modelo Oficial del Benchmark no contiene 'metrics'."
    ) # Validar métricas registradas

benchmark_metrics = official[
    "metrics"
] # Recuperar métricas del Benchmark

# 15.14.3 Validar estructura de métricas
if not isinstance(
    benchmark_metrics,
    dict
):
    raise TypeError(
        "Las métricas del Benchmark deben ser un diccionario."
    ) # Validar estructura de métricas

# 15.14.4 Definir métricas comparables
comparison_metric_names = [
    "rmse",
    "mae",
    "mape",
    "r2",
] # Definir métricas de comparación

# 15.14.5 Identificar métricas faltantes
missing_benchmark_metrics = [
    metric
    for metric in comparison_metric_names
    if metric not in benchmark_metrics
] # Identificar métricas ausentes

if missing_benchmark_metrics:
    raise RuntimeError(
        "El Benchmark no contiene todas las métricas requeridas: "
        f"{missing_benchmark_metrics}"
    ) # Validar disponibilidad

# 15.14.6 Validar métricas oficiales disponibles
if "test" not in official_model_metrics:
    raise RuntimeError(
        "official_model_metrics no contiene métricas de prueba."
    ) # Validar métricas oficiales

official_test_metrics = official_model_metrics[
    "test"
] # Recuperar métricas finales del Modelo Oficial

# 15.14.7 Construir comparación
benchmark_comparison = {} # Inicializar comparación
for metric_name in comparison_metric_names:
    benchmark_value = float(
        benchmark_metrics[
            metric_name
        ]
    ) # Recuperar métrica del Benchmark

    official_value = float(
        official_test_metrics[
            metric_name
        ]
    ) # Recuperar métrica oficial

    if not np.isfinite(
        benchmark_value
    ):
        raise RuntimeError(
            f"La métrica '{metric_name}' del Benchmark no es finita."
        ) # Validar métrica Benchmark

    if not np.isfinite(
        official_value
    ):
        raise RuntimeError(
            f"La métrica '{metric_name}' del Modelo Oficial no es finita."
        ) # Validar métrica oficial

    absolute_difference = float(
        official_value
        - benchmark_value
    ) # Calcular diferencia absoluta con signo

    absolute_error = float(
        abs(
            absolute_difference
        )
    ) # Calcular diferencia absoluta

    if abs(
        benchmark_value
    ) > np.finfo(
        float
    ).eps:
        relative_difference = float(
            absolute_difference
            / abs(
                benchmark_value
            )
        ) # Calcular diferencia relativa

    else:
        relative_difference = float(
            "nan"
        ) # Registrar diferencia relativa no definida

    benchmark_comparison[
        metric_name
    ] = {
        "benchmark": benchmark_value,
        "official": official_value,
        "difference": absolute_difference,
        "absolute_difference": absolute_error,
        "relative_difference": relative_difference,
    } # Registrar comparación de métrica

# 15.14.8 Calcular RMSE relativo al Benchmark
benchmark_rmse = benchmark_comparison[
    "rmse"
][
    "benchmark"
] # Recuperar RMSE Benchmark

official_rmse = benchmark_comparison[
    "rmse"
][
    "official"
] # Recuperar RMSE oficial

if official_rmse < benchmark_rmse:
    rmse_comparison_status = "MEJOR" # Clasificar mejora

elif official_rmse > benchmark_rmse:
    rmse_comparison_status = "PEOR" # Clasificar deterioro

else:
    rmse_comparison_status = "IGUAL" # Clasificar igualdad

# 15.14.9 Calcular MAE relativo al Benchmark
benchmark_mae = benchmark_comparison[
    "mae"
][
    "benchmark"
] # Recuperar MAE Benchmark

official_mae = benchmark_comparison[
    "mae"
][
    "official"
] # Recuperar MAE oficial

if official_mae < benchmark_mae:
    mae_comparison_status = "MEJOR" # Clasificar mejora

elif official_mae > benchmark_mae:
    mae_comparison_status = "PEOR" # Clasificar deterioro

else:
    mae_comparison_status = "IGUAL" # Clasificar igualdad

# 15.14.10 Calcular MAPE relativo al Benchmark
benchmark_mape = benchmark_comparison[
    "mape"
][
    "benchmark"
] # Recuperar MAPE Benchmark

official_mape = benchmark_comparison[
    "mape"
][
    "official"
] # Recuperar MAPE oficial

if official_mape < benchmark_mape:
    mape_comparison_status = "MEJOR" # Clasificar mejora

elif official_mape > benchmark_mape:
    mape_comparison_status = "PEOR" # Clasificar deterioro

else:
    mape_comparison_status = "IGUAL" # Clasificar igualdad

# 15.14.11 Calcular R2 relativo al Benchmark
benchmark_r2 = benchmark_comparison[
    "r2"
][
    "benchmark"
] # Recuperar R2 Benchmark

official_r2 = benchmark_comparison[
    "r2"
][
    "official"
] # Recuperar R2 oficial

if official_r2 > benchmark_r2:
    r2_comparison_status = "MEJOR" # Clasificar mejora

elif official_r2 < benchmark_r2:
    r2_comparison_status = "PEOR" # Clasificar deterioro

else:
    r2_comparison_status = "IGUAL" # Clasificar igualdad

# 15.14.12 Determinar coincidencia exacta
exact_metric_match = all(
    np.isclose(
        benchmark_comparison[
            metric_name
        ][
            "benchmark"
        ],
        benchmark_comparison[
            metric_name
        ][
            "official"
        ],
        rtol=1e-7,
        atol=1e-10,
    )
    for metric_name in comparison_metric_names
) # Determinar coincidencia numérica

# 15.14.13 Registrar resultado de comparación
benchmark_comparison_result = {
    "model_code": OFFICIAL_MODEL_CODE,
    "model_name": OFFICIAL_MODEL_NAME,
    "benchmark_metrics": {
        metric_name: benchmark_comparison[
            metric_name
        ][
            "benchmark"
        ]
        for metric_name in comparison_metric_names
    },
    "official_test_metrics": {
        metric_name: benchmark_comparison[
            metric_name
        ][
            "official"
        ]
        for metric_name in comparison_metric_names
    },
    "comparison": benchmark_comparison,
    "exact_match": bool(exact_metric_match),
    "rmse_status": rmse_comparison_status,
    "mae_status": mae_comparison_status,
    "mape_status": mape_comparison_status,
    "r2_status": r2_comparison_status,
    "validated": True,
} # Registrar comparación con Benchmark

# 15.14.14 Mostrar comparación
print(f"Modelo comparado            : {OFFICIAL_MODEL_NAME}") # Mostrar modelo
print(f"Benchmark RMSE              : {benchmark_rmse:.8f}") # Mostrar RMSE Benchmark
print(f"Oficial RMSE                : {official_rmse:.8f}") # Mostrar RMSE oficial
print(f"Comparación RMSE            : {rmse_comparison_status}") # Mostrar estado RMSE
print(f"Benchmark MAE               : {benchmark_mae:.8f}") # Mostrar MAE Benchmark
print(f"Oficial MAE                 : {official_mae:.8f}") # Mostrar MAE oficial
print(f"Comparación MAE             : {mae_comparison_status}") # Mostrar estado MAE
print(f"Benchmark MAPE              : {benchmark_mape:.8f}") # Mostrar MAPE Benchmark
print(f"Oficial MAPE                : {official_mape:.8f}") # Mostrar MAPE oficial
print(f"Comparación MAPE            : {mape_comparison_status}") # Mostrar estado MAPE
print(f"Benchmark R2                : {benchmark_r2:.8f}") # Mostrar R2 Benchmark
print(f"Oficial R2                  : {official_r2:.8f}") # Mostrar R2 oficial
print(f"Comparación R2              : {r2_comparison_status}") # Mostrar estado R2
print(f"Coincidencia exacta         : {'SÍ' if exact_metric_match else 'NO'}") # Mostrar coincidencia
print("Comparación Benchmark        : COMPLETADA") # Confirmar comparación

# 15.15 PERSISTENCIA
# Objetivo: Persistir el Modelo Oficial GraphSAGE, sus métricas, configuración y trazabilidad científica.
# Producto: Artefacto oficial del modelo entrenado y validado.

print("\n15.15 PERSISTENCIA")

# 15.15.1 Validar disponibilidad del mejor estado
if best_model_state is None:
    raise RuntimeError(
        "No existe best_model_state para persistir el Modelo Oficial."
    ) # Validar disponibilidad del mejor estado

# 15.15.2 Restaurar el mejor estado antes de persistir
official_graphsage.load_state_dict(
    best_model_state
) # Restaurar estado correspondiente a la mejor época

official_graphsage = official_graphsage.to(
    device
) # Garantizar dispositivo del modelo

# 15.15.3 Establecer modo evaluación
official_graphsage.eval() # Establecer estado final de evaluación

# 15.15.4 VALIDAR DIRECTORIO DE SALIDA
OUTPUT_MODEL_TORCH.parent.mkdir(
    parents=True,
    exist_ok=True
) # Garantizar directorio del Modelo Oficial

# 15.15.5 DEFINIR ARCHIVOS OFICIALES DE PERSISTENCIA
official_model_file = OUTPUT_MODEL_TORCH # Definir archivo del Modelo Oficial
official_metadata_file = OUTPUT_MODEL_JOBLIB # Definir archivo de metadatos

# 15.15.6 Construir estado del modelo
official_model_state = {
    key: value.detach().cpu().clone()
    for key, value in official_graphsage.state_dict().items()
} # Construir estado persistible del modelo

# 15.15.7 Validar estado persistible
if not isinstance(
    official_model_state,
    dict
):
    raise TypeError(
        "official_model_state debe ser un diccionario."
    ) # Validar estructura del estado

if len(
    official_model_state
) == 0:
    raise RuntimeError(
        "official_model_state está vacío."
    ) # Validar contenido del estado

# 15.15.8 Validar integridad numérica del estado
for parameter_name, parameter_value in official_model_state.items():
    if not torch.isfinite(
        parameter_value
    ).all():
        raise RuntimeError(
            f"El parámetro '{parameter_name}' contiene NaN o infinitos."
        ) # Validar integridad numérica

# 15.15.9 Construir metadatos oficiales
official_model_metadata = {
    "model": {
        "model_code": OFFICIAL_MODEL_CODE,
        "model_name": OFFICIAL_MODEL_NAME,
        "family": OFFICIAL_MODEL_FAMILY,
    },
    "architecture": official_graphsage_architecture,
    "dimensions": official_dimensions,
    "configuration": official_model_config,
    "training_configuration": training_configuration,
    "training_result": official_training_result,
    "training_validation": training_validation_result,
    "metrics": official_model_metrics,
    "benchmark_comparison": benchmark_comparison_result,
    "best_epoch": int(best_epoch),
    "best_validation_loss": float(best_validation_loss),
    "test_used_during_training": False,
} # Construir metadatos científicos

# 15.15.10 Construir paquete persistible del Modelo Oficial
official_model_package = {
    "model_state_dict": official_model_state,
    "model_code": OFFICIAL_MODEL_CODE,
    "model_name": OFFICIAL_MODEL_NAME,
    "family": OFFICIAL_MODEL_FAMILY,
    "architecture": official_graphsage_architecture,
    "dimensions": official_dimensions,
    "configuration": official_model_config,
    "training_configuration": training_configuration,
    "training_result": official_training_result,
    "metrics": official_model_metrics,
    "benchmark_comparison": benchmark_comparison_result,
    "best_epoch": int(best_epoch),
    "best_validation_loss": float(best_validation_loss),
    "test_used_during_training": False,
} # Construir paquete oficial del modelo

# 15.15.11 Persistir estado del Modelo Oficial
torch.save(
    official_model_package,
    official_model_file,
) # Guardar Modelo Oficial

# 15.15.12 Persistir metadatos científicos
joblib.dump(
    official_model_metadata,
    official_metadata_file,
) # Guardar metadatos científicos

# 15.15.13 Validar existencia física del modelo
if not official_model_file.exists():
    raise RuntimeError(
        "El archivo del Modelo Oficial no fue creado."
    ) # Validar persistencia física

if official_model_file.stat().st_size <= 0:
    raise RuntimeError(
        "El archivo del Modelo Oficial está vacío."
    ) # Validar tamaño físico

# 15.15.14 Validar existencia física de metadatos
if not official_metadata_file.exists():
    raise RuntimeError(
        "El archivo de metadatos oficiales no fue creado."
    ) # Validar persistencia de metadatos

if official_metadata_file.stat().st_size <= 0:
    raise RuntimeError(
        "El archivo de metadatos oficiales está vacío."
    ) # Validar tamaño físico de metadatos

# 15.15.15 Registrar rutas oficiales
official_persistence = {
    "model_file": str(
        official_model_file
    ),
    "metadata_file": str(
        official_metadata_file
    ),
    "model_size_bytes": int(
        official_model_file.stat().st_size
    ),
    "metadata_size_bytes": int(
        official_metadata_file.stat().st_size
    ),
    "best_epoch": int(best_epoch),
    "validated": True,
} # Registrar persistencia oficial

# 15.15.16 Mostrar resultado
print(f"Modelo Oficial             : {OFFICIAL_MODEL_NAME}") # Mostrar modelo
print(f"Mejor época                : {best_epoch}") # Mostrar mejor época
print(f"Archivo modelo             : {official_model_file}") # Mostrar archivo del modelo
print(f"Archivo metadatos          : {official_metadata_file}") # Mostrar archivo de metadatos
print(f"Tamaño modelo              : {official_persistence['model_size_bytes']:,} bytes") # Mostrar tamaño
print(f"Tamaño metadatos           : {official_persistence['metadata_size_bytes']:,} bytes") # Mostrar tamaño
print("Estado persistido           : MEJOR ÉPOCA") # Confirmar estado persistido
print("Persistencia Modelo Oficial : APROBADA") # Confirmar persistencia

# 15.16 VALIDACIÓN FINAL
# Objetivo: Validar integralmente el Modelo Oficial GraphSAGE, sus métricas, persistencia y trazabilidad.
# Producto: Entrenamiento Oficial completamente validado y certificado para uso posterior.

print("\n15.16 VALIDACIÓN FINAL")

# 15.16.1 Validar identidad del Modelo Oficial
if OFFICIAL_MODEL_CODE != official_model_package[
    "model_code"
]:
    raise RuntimeError(
        "El código del Modelo Oficial persistido no coincide con OFFICIAL_MODEL_CODE."
    ) # Validar identidad del modelo

if OFFICIAL_MODEL_NAME.strip().lower() != official_model_package[
    "model_name"
].strip().lower():
    raise RuntimeError(
        "El nombre del Modelo Oficial persistido no coincide con OFFICIAL_MODEL_NAME."
    ) # Validar nombre del modelo

if OFFICIAL_MODEL_FAMILY.strip().lower() != official_model_package[
    "family"
].strip().lower():
    raise RuntimeError(
        "La familia del Modelo Oficial persistido no coincide con OFFICIAL_MODEL_FAMILY."
    ) # Validar familia del modelo

# 15.16.2 Validar estado del entrenamiento
if not official_training_result.get(
    "training_completed",
    False
):
    raise RuntimeError(
        "El entrenamiento oficial no está marcado como completado."
    ) # Validar entrenamiento

if completed_epochs != epochs:
    raise RuntimeError(
        f"El entrenamiento terminó en {completed_epochs} épocas y se esperaban {epochs}."
    ) # Validar épocas

# 15.16.3 Validar mejor estado
if best_model_state is None:
    raise RuntimeError(
        "No existe el mejor estado del Modelo Oficial."
    ) # Validar mejor estado

if best_epoch is None:
    raise RuntimeError(
        "No existe la mejor época del Modelo Oficial."
    ) # Validar mejor época

if best_epoch < 1 or best_epoch > epochs:
    raise RuntimeError(
        "best_epoch se encuentra fuera del rango válido de entrenamiento."
    ) # Validar rango de mejor época

# 15.16.4 Validar dimensiones oficiales
if official_dimensions[
    "in_channels"
] != in_channels:
    raise RuntimeError(
        "La dimensión de entrada persistida no coincide con la dimensión utilizada."
    ) # Validar entrada

if official_dimensions[
    "hidden_channels"
] != hidden_channels:
    raise RuntimeError(
        "La dimensión oculta persistida no coincide con la dimensión utilizada."
    ) # Validar dimensión oculta

if official_dimensions[
    "out_channels"
] != out_channels:
    raise RuntimeError(
        "La dimensión de salida persistida no coincide con la dimensión utilizada."
    ) # Validar salida

# 15.16.5 Validar métricas oficiales
if not isinstance(
    official_model_metrics,
    dict
):
    raise TypeError(
        "official_model_metrics debe ser un diccionario."
    ) # Validar métricas

if "validation" not in official_model_metrics:
    raise RuntimeError(
        "Faltan las métricas oficiales de validación."
    ) # Validar validación

if "test" not in official_model_metrics:
    raise RuntimeError(
        "Faltan las métricas oficiales de prueba."
    ) # Validar prueba

# 15.16.6 Validar métricas finales
final_metric_fields = [
    "rmse",
    "mae",
    "mape",
    "r2",
] # Definir métricas finales

for dataset_name in [
    "validation",
    "test",
]:
    dataset_metrics = official_model_metrics[
        dataset_name
    ] # Recuperar métricas

    for metric_name in final_metric_fields:
        if metric_name not in dataset_metrics:
            raise RuntimeError(
                f"Falta la métrica '{metric_name}' en {dataset_name}."
            ) # Validar presencia de métrica
        metric_value = float(
            dataset_metrics[
                metric_name
            ]
        ) # Recuperar valor de métrica

        if np.isnan(
            metric_value
        ):
            if metric_name not in [
                "mape",
                "r2",
            ]:
                raise RuntimeError(
                    f"La métrica '{metric_name}' de {dataset_name} es NaN."
                ) # Validar métricas fundamentales

        elif not np.isfinite(
            metric_value
        ):
            raise RuntimeError(
                f"La métrica '{metric_name}' de {dataset_name} no es finita."
            ) # Validar métrica

# 15.16.7 Validar aislamiento del conjunto de prueba
if official_training_result.get(
    "test_used_during_training",
    True
):
    raise RuntimeError(
        "El conjunto de prueba aparece marcado como utilizado durante el entrenamiento."
    ) # Validar aislamiento científico

if test_metrics.get(
    "test_used_during_training",
    True
):
    raise RuntimeError(
        "Las métricas de prueba indican que el test fue utilizado durante el entrenamiento."
    ) # Validar aislamiento de prueba

# 15.16.8 Validar fuente de selección
if official_model_metrics[
    "selection"
][
    "selection_source"
] != "validation":
    raise RuntimeError(
        "La selección del Modelo Oficial no proviene exclusivamente de validación."
    ) # Validar protocolo de selección

# 15.16.9 Validar comparación con Benchmark
if not isinstance(
    benchmark_comparison_result,
    dict
):
    raise TypeError(
        "benchmark_comparison_result debe ser un diccionario."
    ) # Validar comparación

if benchmark_comparison_result.get(
    "model_code"
) != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "La comparación con Benchmark corresponde a otro Modelo Oficial."
    ) # Validar identidad de comparación

if not benchmark_comparison_result.get(
    "validated",
    False
):
    raise RuntimeError(
        "La comparación con Benchmark no fue validada."
    ) # Validar comparación

# 15.16.10 Validar persistencia del modelo
if not official_model_file.exists():
    raise RuntimeError(
        "El archivo persistido del Modelo Oficial no existe."
    ) # Validar archivo del modelo

if official_model_file.stat().st_size <= 0:
    raise RuntimeError(
        "El archivo persistido del Modelo Oficial está vacío."
    ) # Validar contenido del modelo

# 15.16.11 Validar persistencia de metadatos
if not official_metadata_file.exists():
    raise RuntimeError(
        "El archivo de metadatos del Modelo Oficial no existe."
    ) # Validar archivo de metadatos

if official_metadata_file.stat().st_size <= 0:
    raise RuntimeError(
        "El archivo de metadatos del Modelo Oficial está vacío."
    ) # Validar contenido de metadatos

# 15.16.12 Validar estado persistido
if not isinstance(
    official_model_state,
    dict
):
    raise TypeError(
        "official_model_state debe ser un diccionario."
    ) # Validar estado persistido

if set(
    official_model_state.keys()
) != set(
    official_graphsage.state_dict().keys()
):
    raise RuntimeError(
        "El estado persistido no coincide estructuralmente con GraphSAGE."
    ) # Validar estructura del estado

# 15.16.13 Validar integridad de parámetros persistidos
for parameter_name, parameter_value in official_model_state.items():
    if not torch.isfinite(
        parameter_value
    ).all():
        raise RuntimeError(
            f"El parámetro persistido '{parameter_name}' contiene NaN o infinitos."
        ) # Validar parámetros persistidos

# 15.16.14 Validar auditorías previas
if not graph_data_audit.get(
    "validated",
    False
):
    raise RuntimeError(
        "La auditoría de GraphData no está aprobada."
    ) # Validar GraphData

if not graphsage_structural_validation.get(
    "validated",
    False
):
    raise RuntimeError(
        "La validación estructural de GraphSAGE no está aprobada."
    ) # Validar arquitectura

if not training_validation_result.get(
    "validated",
    False
):
    raise RuntimeError(
        "La validación del entrenamiento no está aprobada."
    ) # Validar entrenamiento

# 15.16.15 Construir certificado final
final_validation = {
    "status": "VALIDATED",
    "model_code": OFFICIAL_MODEL_CODE,
    "model_name": OFFICIAL_MODEL_NAME,
    "family": OFFICIAL_MODEL_FAMILY,
    "best_epoch": int(best_epoch),
    "best_validation_loss": float(best_validation_loss),
    "training_completed": True,
    "graph_data_validated": True,
    "architecture_validated": True,
    "training_validated": True,
    "validation_evaluated": True,
    "test_evaluated": True,
    "benchmark_compared": True,
    "model_persisted": True,
    "metadata_persisted": True,
    "test_used_during_training": False,
} # Construir certificado final

# 15.16.16 Mostrar resumen final
print(f"Modelo Oficial             : {OFFICIAL_MODEL_NAME}") # Mostrar modelo
print(f"Código Oficial             : {OFFICIAL_MODEL_CODE}") # Mostrar código
print(f"Familia Oficial            : {OFFICIAL_MODEL_FAMILY}") # Mostrar familia
print(f"Épocas completadas         : {completed_epochs}") # Mostrar épocas
print(f"Mejor época                : {best_epoch}") # Mostrar mejor época
print(f"Validation RMSE            : {validation_metrics['rmse']:.8f}") # Mostrar RMSE validación
print(f"Test RMSE                  : {test_metrics['rmse']:.8f}") # Mostrar RMSE prueba
print(f"Validation R2              : {validation_metrics['r2']:.8f}") # Mostrar R2 validación
print(f"Test R2                    : {test_metrics['r2']:.8f}") # Mostrar R2 prueba
print(f"Modelo persistido          : {official_model_file}") # Mostrar modelo persistido
print(f"Metadatos persistidos      : {official_metadata_file}") # Mostrar metadatos
print("GraphData                   : VALIDADO") # Confirmar GraphData
print("Arquitectura GraphSAGE      : VALIDADA") # Confirmar arquitectura
print("Entrenamiento               : VALIDADO") # Confirmar entrenamiento
print("Evaluación validación       : VALIDADA") # Confirmar validación
print("Evaluación prueba           : VALIDADA") # Confirmar prueba
print("Comparación Benchmark       : VALIDADA") # Confirmar comparación
print("Persistencia                : VALIDADA") # Confirmar persistencia
print("ESTADO FINAL                : VALIDATED") # Confirmar certificado final

# BLOQUE 16. VALIDACIÓN FINAL
# Objetivo: Verificar la integridad completa del Modelo Oficial entrenado,
# restaurado y persistido, incluyendo archivos, arquitectura, pesos y metadatos.
# Producto: Modelo Oficial científicamente validado y listo para las etapas posteriores.

print("\n16. VALIDACIÓN FINAL")

# 16.1 VALIDAR ESTADO FINAL DEL MODELO
# Validar disponibilidad del Modelo Oficial
if official_graphsage is None:
    raise RuntimeError(
        "OfficialGraphSAGE no está disponible."
    ) # Validar disponibilidad del modelo

# Validar tipo del modelo
if not isinstance(
    official_graphsage,
    torch.nn.Module
):
    raise TypeError(
        "OfficialGraphSAGE debe ser una instancia de torch.nn.Module."
    ) # Validar tipo del modelo

# Validar modo evaluación
if official_graphsage.training:
    raise RuntimeError(
        "OfficialGraphSAGE debe encontrarse en modo evaluación."
    ) # Validar estado final del modelo

# 16.2 VALIDAR ARCHIVO DEL MODELO
# Validar existencia del archivo persistido
if not official_model_file.exists():
    raise FileNotFoundError(
        "No existe el archivo persistido del Modelo Oficial: "
        f"{official_model_file}"
    ) # Validar archivo del modelo

# Validar tamaño físico del archivo
if official_model_file.stat().st_size <= 0:
    raise RuntimeError(
        "El archivo persistido del Modelo Oficial está vacío."
    ) # Validar contenido físico

# 16.3 VALIDAR ARCHIVO DE METADATOS
# Validar existencia de los metadatos
if not official_metadata_file.exists():
    raise FileNotFoundError(
        "No existe el archivo de metadatos del Modelo Oficial: "
        f"{official_metadata_file}"
    ) # Validar archivo de metadatos

# Validar tamaño físico de los metadatos
if official_metadata_file.stat().st_size <= 0:
    raise RuntimeError(
        "El archivo de metadatos del Modelo Oficial está vacío."
    ) # Validar contenido físico

# 16.4 RECUPERAR MODELO PERSISTIDO
# Cargar paquete persistido
persisted_model_package = torch.load(
    official_model_file,
    map_location="cpu",
) # Recuperar modelo persistido

# Validar estructura del paquete
if not isinstance(
    persisted_model_package,
    dict
):
    raise TypeError(
        "El paquete persistido del Modelo Oficial debe ser un diccionario."
    ) # Validar estructura persistida

# Validar presencia del estado del modelo
if "model_state_dict" not in persisted_model_package:
    raise RuntimeError(
        "El paquete persistido no contiene 'model_state_dict'."
    ) # Validar estado persistido

# Recuperar estado persistido
persisted_state_dict = persisted_model_package[
    "model_state_dict"
] # Recuperar pesos persistidos

# 16.5 VALIDAR IDENTIDAD DEL MODELO PERSISTIDO
# Validar código oficial
if persisted_model_package.get(
    "model_code"
) != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El código del modelo persistido no coincide con OFFICIAL_MODEL_CODE."
    ) # Validar identidad del modelo

# Validar nombre oficial
if persisted_model_package.get(
    "model_name"
) != OFFICIAL_MODEL_NAME:
    raise RuntimeError(
        "El nombre del modelo persistido no coincide con OFFICIAL_MODEL_NAME."
    ) # Validar identidad del modelo

# Validar familia oficial
if persisted_model_package.get(
    "family"
) != OFFICIAL_MODEL_FAMILY:
    raise RuntimeError(
        "La familia del modelo persistido no coincide con OFFICIAL_MODEL_FAMILY."
    ) # Validar familia del modelo

# 16.6 VALIDAR ARQUITECTURA PERSISTIDA
# Recuperar configuración arquitectónica persistida
persisted_architecture = persisted_model_package.get(
    "architecture"
) # Recuperar arquitectura persistida

# Validar arquitectura
if not isinstance(
    persisted_architecture,
    dict
):
    raise TypeError(
        "La arquitectura persistida debe ser un diccionario."
    ) # Validar arquitectura

# Validar dimensión de entrada
if persisted_architecture.get(
    "in_channels"
) != in_channels:
    raise RuntimeError(
        "in_channels persistido no coincide con el modelo oficial."
    ) # Validar dimensión de entrada

# Validar dimensión oculta
if persisted_architecture.get(
    "hidden_channels"
) != hidden_channels:
    raise RuntimeError(
        "hidden_channels persistido no coincide con el modelo oficial."
    ) # Validar dimensión oculta

# Validar dimensión de salida
if persisted_architecture.get(
    "out_channels"
) != out_channels:
    raise RuntimeError(
        "out_channels persistido no coincide con el modelo oficial."
    ) # Validar dimensión de salida

# Validar Dropout
if not np.isclose(
    float(
        persisted_architecture.get(
            "dropout"
        )
    ),
    dropout,
):
    raise RuntimeError(
        "El Dropout persistido no coincide con la configuración oficial."
    ) # Validar Dropout

# 16.7 VALIDAR ESTADO PERSISTIDO
# Recuperar estado actual del modelo
final_model_state = official_graphsage.state_dict() # Recuperar estado final

# Validar nombres de parámetros
if set(
    final_model_state.keys()
) != set(
    persisted_state_dict.keys()
):
    raise RuntimeError(
        "Los parámetros del modelo persistido no coinciden "
        "con los parámetros del Modelo Oficial."
    ) # Validar estructura de parámetros

# Validar parámetros individualmente
for parameter_name, parameter in final_model_state.items():
    # Recuperar parámetro persistido
    persisted_parameter = persisted_state_dict[
        parameter_name
    ] # Recuperar peso persistido

    # Validar dimensiones
    if parameter.shape != persisted_parameter.shape:
        raise RuntimeError(
            f"El parámetro '{parameter_name}' presenta dimensiones incompatibles."
        ) # Validar dimensiones

    # Validar valores finitos
    if not torch.isfinite(
        persisted_parameter
    ).all():
        raise RuntimeError(
            f"El parámetro '{parameter_name}' persistido contiene NaN o infinitos."
        ) # Validar integridad numérica

    # Validar igualdad de pesos
    if not torch.equal(
        parameter.detach().cpu(),
        persisted_parameter.detach().cpu(),
    ):
        raise RuntimeError(
            f"El parámetro '{parameter_name}' no coincide con el estado persistido."
        ) # Validar correspondencia exacta

# 16.8 RECUPERAR Y VALIDAR METADATOS
# Cargar metadatos JSON
with official_metadata_file.open(
    "r",
    encoding="utf-8",
) as file:
    persisted_metadata = json.load(
        file
    ) # Recuperar metadatos persistidos

# Validar estructura
if not isinstance(
    persisted_metadata,
    dict
):
    raise TypeError(
        "Los metadatos persistidos deben ser un diccionario."
    ) # Validar estructura de metadatos

# Validar estado del modelo
if persisted_metadata.get(
    "status"
) != "TRAINED":
    raise RuntimeError(
        "Los metadatos no presentan estado TRAINED."
    ) # Validar estado científico

# Validar código
if persisted_metadata.get(
    "model_code"
) != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El código de los metadatos no coincide con el Modelo Oficial."
    ) # Validar código

# Validar nombre
if persisted_metadata.get(
    "model_name"
) != OFFICIAL_MODEL_NAME:
    raise RuntimeError(
        "El nombre de los metadatos no coincide con el Modelo Oficial."
    ) # Validar nombre

# 16.9 VALIDAR RESULTADOS DEL ENTRENAMIENTO
# Recuperar configuración de entrenamiento persistida
persisted_training = persisted_model_package.get(
    "training"
) # Recuperar resultados de entrenamiento

# Validar estructura
if not isinstance(
    persisted_training,
    dict
):
    raise TypeError(
        "La configuración de entrenamiento persistida debe ser un diccionario."
    ) # Validar configuración

# Validar mejor época
if persisted_training.get(
    "best_epoch"
) != best_epoch:
    raise RuntimeError(
        "La mejor época persistida no coincide con best_epoch."
    ) # Validar mejor época

# Validar mejor pérdida
if not np.isclose(
    float(
        persisted_training.get(
            "best_validation_loss"
        )
    ),
    float(
        best_validation_loss
    ),
):
    raise RuntimeError(
        "La mejor Validation Loss persistida no coincide con "
        "best_validation_loss."
    ) # Validar mejor pérdida

# 16.10 VALIDAR MÉTRICAS PERSISTIDAS
# Recuperar métricas persistidas
persisted_metrics = persisted_model_package.get(
    "metrics"
) # Recuperar métricas

# Validar presencia de métricas
if not isinstance(
    persisted_metrics,
    dict
):
    raise TypeError(
        "Las métricas persistidas deben ser un diccionario."
    ) # Validar métricas

# Validar partición de entrenamiento
if "train" not in persisted_metrics:
    raise RuntimeError(
        "Las métricas persistidas no contienen resultados de entrenamiento."
    ) # Validar métricas de entrenamiento

# Validar partición de validación
if "validation" not in persisted_metrics:
    raise RuntimeError(
        "Las métricas persistidas no contienen resultados de validación."
    ) # Validar métricas de validación

# 16.11 CONSTRUIR CERTIFICADO FINAL
# Construir certificado científico final
final_model_validation = {
    "status": "VALIDATED",
    "model_code": OFFICIAL_MODEL_CODE,
    "model_name": OFFICIAL_MODEL_NAME,
    "model_family": OFFICIAL_MODEL_FAMILY,
    "architecture_validated": True,
    "weights_validated": True,
    "metadata_validated": True,
    "persistence_validated": True,
    "best_epoch": int(best_epoch),
    "best_validation_loss": float(best_validation_loss),
    "train_metrics_validated": True,
    "validation_metrics_validated": True,
} # Construir certificado final

# 16.12 RESUMEN FINAL
print(f"Modelo Oficial            : {OFFICIAL_MODEL_NAME}") # Mostrar modelo
print(f"Código Oficial            : {OFFICIAL_MODEL_CODE}") # Mostrar código
print(f"Familia                   : {OFFICIAL_MODEL_FAMILY}") # Mostrar familia
print(f"Mejor época               : {best_epoch}") # Mostrar mejor época
print(f"Best Validation Loss      : {best_validation_loss:.10f}") # Mostrar mejor pérdida
print("Arquitectura               : VALIDADA") # Confirmar arquitectura
print("Pesos                      : VALIDADOS") # Confirmar pesos
print("Metadatos                  : VALIDADOS") # Confirmar metadatos
print("Persistencia               : VALIDADA") # Confirmar persistencia
print("Métricas                   : VALIDADAS") # Confirmar métricas
print("Integridad científica      : VALIDADA") # Confirmar integridad
print("MODELO OFICIAL             : VALIDADO") # Confirmar resultado final



