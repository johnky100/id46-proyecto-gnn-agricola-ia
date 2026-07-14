# 02_benchmark.py

# BLOQUE 1. Importaciones -------------------------------------------------
## Objetivo: Importar las dependencias necesarias para ejecutar el Benchmark
# Científico de modelos predictivos utilizando el GraphData construido
# durante la etapa de construcción del grafo.
## Pregunta científica: ¿Qué dependencias requiere el protocolo experimental para comparar
# objetivamente diferentes modelos predictivos?

# Librerías estándar
import time # Medición del tiempo de ejecución
import warnings # Control de advertencias

# Librerías científicas
import numpy as np # Operaciones numéricas
import pandas as pd # Manipulación de datos
import torch # PyTorch
import os # Operaciones sobre archivos
import json # Exportación JSON
import joblib # Persistencia de objetos Python
import pandas as pd # Manipulación tabular

# PyTorch Geometric
from torch_geometric.data import Data # Objeto GraphData

# Configuración oficial del proyecto
from src.python.config.config_project import (
    SEED,
    BENCHMARK_CONFIG,
    MODEL_CANDIDATES,
    MODEL_CODES,
    BENCHMARK_METRICS,
    MODEL_SELECTION,
    BENCHMARK_REPRODUCIBILITY
) # Configuración oficial del benchmark

# Rutas oficiales
from src.python.config.paths import (
    GRAPH_DATA_FILE,
    BENCHMARK_RESULTS_FILE,
    BENCHMARK_SUMMARY_FILE,
    BENCHMARK_METRICS_FILE,
    BENCHMARK_RANKING_FILE,
    BEST_MODEL_CONFIG_FILE,
    validate_project_structure
) # Rutas oficiales del proyecto

from src.python.models.statistical import (
    run_linear_regression
)

from src.python.models.machine_learning import (
    run_machine_learning_model
)

from src.python.models.deep_learning import (
    run_mlp
)

from src.python.models.graph_neural_networks import (
    GNN_CONFIG,
    run_gnn
) # Configuración y ejecución de modelos GNN

print("-" * 80)

# BLOQUE 2. Configuración del Script --------------------------------------
## Objetivo: Configurar el entorno de ejecución y verificar las condiciones
# necesarias para ejecutar el Benchmark Científico de modelos
# predictivos de forma reproducible.
## Pregunta científica: ¿El entorno de ejecución cumple las condiciones necesarias para
# realizar un benchmark reproducible y consistente?

# Ejecución 2.1. Configuración del entorno --------------------------------
## Objetivo:
# Configurar el entorno de ejecución del benchmark.
warnings.filterwarnings(
    "ignore"
) # Ocultar advertencias
print(
    "Entorno de ejecución configurado correctamente."
)

# Ejecución 2.2. Validación de la estructura del proyecto ------------------
## Objetivo:
# Verificar la existencia de los directorios oficiales del proyecto.
validate_project_structure(
    verbose = True
) # Validar estructura del proyecto

# Ejecución 2.3. Configuración de la reproducibilidad ----------------------
## Objetivo:
# Configurar las semillas para garantizar la reproducibilidad del
# benchmark.
torch.manual_seed(
    SEED
) # Configurar semilla de PyTorch

np.random.seed(
    SEED
) # Configurar semilla de NumPy

if BENCHMARK_REPRODUCIBILITY["deterministic"]:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            SEED
        ) # Configurar semilla de la GPU

        torch.cuda.manual_seed_all(
            SEED
        ) # Configurar todas las GPU

        torch.backends.cudnn.deterministic = True # Forzar reproducibilidad
        torch.backends.cudnn.benchmark = False # Desactivar benchmark automático

# Ejecución 2.4. Selección del dispositivo --------------------------------
## Objetivo:
# Seleccionar automáticamente el dispositivo de procesamiento.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
) # Dispositivo de procesamiento

print("-" * 80)

# BLOQUE 3. Carga del GraphData -------------------------------------------
## Objetivo: Cargar el objeto GraphData construido durante la etapa de construcción
# del grafo para utilizarlo como entrada del Benchmark Científico.
## Pregunta científica: ¿El GraphData fue cargado correctamente y está disponible para la
# comparación de modelos predictivos?

# Ejecución 3.1. Verificación del archivo ---------------------------------
## Objetivo:
# Verificar la existencia del archivo oficial del GraphData.
if not GRAPH_DATA_FILE.exists():
    raise FileNotFoundError(
        f"No existe el archivo: {GRAPH_DATA_FILE}"
    )

print(
    "Archivo GraphData localizado correctamente."
)

# Ejecución 3.2. Carga del GraphData --------------------------------------
## Objetivo:
# Cargar el objeto GraphData desde el archivo oficial.
graph_data = torch.load(
    GRAPH_DATA_FILE,
    weights_only = False
) # GraphData oficial

print(
    "GraphData cargado correctamente."
)

# Ejecución 3.3. Verificación del tipo de objeto ---------------------------
## Objetivo:
# Verificar que el objeto cargado corresponde a un GraphData.
if not isinstance(
    graph_data,
    Data
):
    raise TypeError(
        "El objeto cargado no corresponde a un GraphData."
    )

print(
    "Tipo de objeto verificado correctamente."
)

# Ejecución 3.4. Registro de información ----------------------------------
## Objetivo:
# Registrar las principales dimensiones del GraphData cargado.
print(
    f"Número de nodos      : {graph_data.num_nodes:,}"
)

print(
    f"Número de aristas    : {graph_data.num_edges:,}"
)

print(
    f"Node Features        : {graph_data.num_node_features}"
)

print(
    "GraphData preparado para el Benchmark.\n"
)

# Ejecución 3.5. Confirmación del bloque ----------------------------------
print(
    "Carga del GraphData finalizada correctamente."
)

print("-" * 80)

# BLOQUE 4. Validación del GraphData --------------------------------------
## Objetivo: # Verificar la integridad, consistencia y completitud del GraphData
# antes de iniciar el Benchmark Científico.
## Pregunta científica:# ¿El GraphData cumple los requisitos necesarios para comparar
# objetivamente los modelos candidatos?

# Ejecución 4.1. Validar Node Features ------------------------------------
## Objetivo: Verificar que el GraphData contiene la matriz oficial de Node Features.
if graph_data.x is None:
    raise ValueError(
        "El GraphData no contiene Node Features."
    )
print(
    "Node Features verificadas."
)

# Ejecución 4.2. Validar Variable Objetivo --------------------------------
## Objetivo: Verificar que el GraphData contiene la variable objetivo.
if graph_data.y is None:
    raise ValueError(
        "El GraphData no contiene la variable objetivo."
    )
print(
    "Variable objetivo verificada."
)

# Ejecución 4.3. Validar edge_index ---------------------------------------
## Objetivo: Verificar la estructura oficial de conectividad del grafo.
if graph_data.edge_index is None:
    raise ValueError(
        "El GraphData no contiene edge_index."
    )
print(
    "edge_index verificado."
)

# Ejecución 4.4. Validar edge_weight --------------------------------------
## Objetivo: Verificar que las aristas poseen pesos asociados.
if graph_data.edge_weight is None:
    raise ValueError(
        "El GraphData no contiene edge_weight."
    )
print(
    "edge_weight verificado."
)

# Ejecución 4.5. Validar dimensiones --------------------------------------
## Objetivo: Verificar la consistencia entre nodos y variables.
if graph_data.x.shape[0] != graph_data.num_nodes:
    raise ValueError(
        "El número de Node Features no coincide con el número de nodos."
    )

if graph_data.y.shape[0] != graph_data.num_nodes:
    raise ValueError(
        "El número de observaciones de la variable objetivo no coincide con el número de nodos."
    )
print(
    "Dimensiones verificadas."
)

# Ejecución 4.6. Registrar dimensiones ------------------------------------
## Objetivo: Registrar las principales dimensiones del GraphData.
print(
    f"Número de nodos      : {graph_data.num_nodes:,}"
)
print(
    f"Número de aristas    : {graph_data.num_edges:,}"
)
print(
    f"Node Features        : {graph_data.num_node_features}"
)
print(
    f"Observaciones objetivo : {graph_data.y.shape[0]:,}"    
)

print(f"Dimensión de y          : {graph_data.y.dim()}")

print(f"Forma de y              : {tuple(graph_data.y.shape)}")

print(f"Observaciones objetivo  : {graph_data.y.numel():,}")

# Ejecución 4.7. Confirmación del bloque ----------------------------------
print(
    "\nGraphData validado correctamente."
)
print("-" * 80)

# BLOQUE 5. Modelos Candidatos ---------------------------------------------
## Objetivo: Validar la configuración oficial de los modelos candidatos que participarán en el Benchmark Científico.
### Producto: - MODEL_CANDIDATES validado
### Responde: ¿La configuración oficial de los modelos candidatos es consistente y suficiente para ejecutar el Benchmark Científico?

# Validación de las familias de modelos ------------------------------------
if not MODEL_CANDIDATES:
    raise ValueError(
        "No existen familias de modelos registradas."
    )

# Validación de los modelos candidatos -------------------------------------
total_models = sum(
    len(models)
    for models in MODEL_CANDIDATES.values()
) # Número total de modelos

if total_models == 0:
    raise ValueError(
        "No existen modelos candidatos para ejecutar el Benchmark."
    )

# Validación de familias vacías --------------------------------------------
empty_families = [
    family
    for family, models in MODEL_CANDIDATES.items()
    if len(models) == 0
] # Familias sin modelos

if empty_families:
    raise ValueError(
        f"Familias sin modelos registrados: {empty_families}"
    )

print("-" * 80)

# BLOQUE 6. Preparación del Benchmark --------------------------------------
## Objetivo: Preparar los conjuntos oficiales de datos que serán utilizados por todas
# las familias de modelos durante el Benchmark Científico.
### Entradas: - graph_data - BENCHMARK_CONFIG
### Producto: - benchmark_data
### Responde: ¿Los datos fueron preparados correctamente para ejecutar el Benchmark Científico de manera homogénea?

# Número total de nodos ----------------------------------------------------
num_nodes = graph_data.num_nodes # Total de nodos
indices = np.arange(
    num_nodes
) # Índices oficiales

# Mezclar nodos ------------------------------------------------------------
if BENCHMARK_CONFIG["shuffle"]:
    np.random.shuffle(
        indices
    ) # Mezclar índices

# Tamaño de las particiones ------------------------------------------------
train_size = int(
    num_nodes * BENCHMARK_CONFIG["train_size"]
) # Entrenamiento

validation_size = int(
    num_nodes * BENCHMARK_CONFIG["validation_size"]
) # Validación

# Índices ------------------------------------------------------------------
train_index = indices[
    :train_size
] # Índices entrenamiento

validation_index = indices[
    train_size:
    train_size + validation_size
] # Índices validación

test_index = indices[
    train_size + validation_size:
] # Índices prueba

# Construcción de máscaras -------------------------------------------------
train_mask = torch.zeros(
    num_nodes,
    dtype = torch.bool
)

validation_mask = torch.zeros(
    num_nodes,
    dtype = torch.bool
)

test_mask = torch.zeros(
    num_nodes,
    dtype = torch.bool
)

train_mask[
    train_index
] = True

validation_mask[
    validation_index
] = True

test_mask[
    test_index
] = True

# Registrar máscaras en el grafo -------------------------------------------
graph_data.train_mask = train_mask
graph_data.validation_mask = validation_mask
graph_data.test_mask = test_mask

# Preparación para modelos clásicos ----------------------------------------
x = graph_data.x.cpu().numpy() # Variables predictoras
y = graph_data.y.cpu().numpy().ravel() # Variable objetivo

x_train = x[
    train_mask.cpu().numpy()
] # Entrenamiento

y_train = y[
    train_mask.cpu().numpy()
] # Objetivo entrenamiento

x_validation = x[
    validation_mask.cpu().numpy()
] # Validación

y_validation = y[
    validation_mask.cpu().numpy()
] # Objetivo validación

x_test = x[
    test_mask.cpu().numpy()
] # Prueba

y_test = y[
    test_mask.cpu().numpy()
] # Objetivo prueba

# Consolidación ------------------------------------------------------------
benchmark_data = {
    "graph_data": graph_data,
    "x_train": x_train,
    "y_train": y_train,
    "x_validation": x_validation,
    "y_validation": y_validation,
    "x_test": x_test,
    "y_test": y_test,
    "train_index": train_index,
    "validation_index": validation_index,
    "test_index": test_index
} # Datos oficiales del Benchmark

# Validación ---------------------------------------------------------------
if (
    len(train_index)
    + len(validation_index)
    + len(test_index)
) != num_nodes:

    raise ValueError(
        "La partición oficial del Benchmark es inconsistente."
    )

print("-" * 80)

# BLOQUE 7. Benchmark Estadístico ------------------------------------------
## Objetivo: Ejecutar el Benchmark Científico para los modelos estadísticos utilizando el protocolo experimental oficial.
### Entradas: - MODEL_CANDIDATES - benchmark_data
### Producto: - statistical_results
### Responde: ¿Cuál es el desempeño de los modelos estadísticos bajo el protocolo oficial del Benchmark Científico?
# Inicialización -----------------------------------------------------------
statistical_results = [
    run_linear_regression(
        x_train = benchmark_data["x_train"],
        y_train = benchmark_data["y_train"],
        x_test = benchmark_data["x_test"],
        y_test = benchmark_data["y_test"]
    )
]

if not statistical_results:
    raise ValueError(
        "No fue posible ejecutar el Benchmark de modelos estadísticos."
    )

print("-" * 80)

# BLOQUE 8. Benchmark Machine Learning -------------------------------------
machine_learning_results = [] # Resultados oficiales
for model_name in MODEL_CANDIDATES["machine_learning"]:
    result = run_machine_learning_model(
        model_name = model_name,
        x_train = benchmark_data["x_train"],
        y_train = benchmark_data["y_train"],
        x_test = benchmark_data["x_test"],
        y_test = benchmark_data["y_test"]
    )

    machine_learning_results.append(
        result
    )

# Validación ---------------------------------------------------------------
if not machine_learning_results:
    raise ValueError(
        "No fue posible ejecutar el Benchmark de Machine Learning."
    )

print("-" * 80)

# BLOQUE 9. Benchmark Deep Learning ----------------------------------------
## Objetivo: Ejecutar el Benchmark Científico para los modelos de Deep Learning
# utilizando el protocolo experimental oficial.
### Entradas: - MODEL_CANDIDATES - benchmark_data
### Producto: - deep_learning_results
### Responde: ¿Cuál es el desempeño de los modelos de Deep Learning bajo el
# protocolo oficial del Benchmark Científico?

# Inicialización -----------------------------------------------------------
deep_learning_results = [] # Resultados de Deep Learning

# Ejecución del Benchmark --------------------------------------------------
for model_name in MODEL_CANDIDATES["deep_learning"]:
    result = run_mlp(
        x_train = benchmark_data["x_train"],
        y_train = benchmark_data["y_train"],
        x_test = benchmark_data["x_test"],
        y_test = benchmark_data["y_test"]
    ) # Ejecutar modelo

    deep_learning_results.append(
        result
    ) # Registrar resultado

# Validación ---------------------------------------------------------------
if not deep_learning_results:
    raise ValueError(
        "No fue posible ejecutar el Benchmark de Deep Learning."
    )

print("-" * 80)

# BLOQUE 10. Benchmark GNN -----------------------------------------------
## Objetivo: Ejecutar el Benchmark Científico para todas las arquitecturas Graph Neural Networks.
### Entradas: - MODEL_CANDIDATES - GNN_CONFIG - benchmark_data
### Producto: - gnn_results
### Responde: ¿Cuál es el desempeño de las arquitecturas GNN bajo el protocolo oficial del Benchmark Científico?

# Inicialización ----------------------------------------------------------
gnn_results = [] # Resultados oficiales de las GNN

# Ejecución del Benchmark -------------------------------------------------
for model_name in MODEL_CANDIDATES["graph_neural_networks"]:
    model_config = GNN_CONFIG[
        model_name
    ] # Configuración oficial del modelo

    result = run_gnn(
        model_config = model_config,
        graph_data = benchmark_data["graph_data"]
    ) # Ejecutar arquitectura

    gnn_results.append(
        result
    ) # Registrar resultado

# Validación --------------------------------------------------------------
if not gnn_results:
    raise ValueError(
        "No fue posible ejecutar el Benchmark de modelos GNN."
    )

print("-" * 80)

# BLOQUE 11. Consolidación de Resultados -----------------------------------
## Objetivo: Consolidar los resultados obtenidos por todas las familias de modelos
# evaluadas durante el Benchmark Científico.
### Entradas: - statistical_results - machine_learning_results - deep_learning_results - gnn_results
### Producto: - benchmark_results
### Responde: ¿Los resultados de todas las familias de modelos fueron consolidados
# correctamente para su análisis comparativo?

# Consolidación ------------------------------------------------------------
benchmark_results = (
    statistical_results
    + machine_learning_results
    + deep_learning_results
    + gnn_results
) # Resultados oficiales del Benchmark

# Validación ---------------------------------------------------------------
if not benchmark_results:
    raise ValueError(
        "No existen resultados para consolidar el Benchmark."
    )

# Verificación del número de modelos ---------------------------------------
expected_models = sum(
    len(models)
    for models in MODEL_CANDIDATES.values()
) # Número esperado de modelos

if len(benchmark_results) != expected_models:
    raise ValueError(
        "El número de resultados no coincide con los modelos "
        "registrados en el Benchmark."
    )

print("-" * 80)

# BLOQUE 12. Ranking del Benchmark -----------------------------------------
## Objetivo: Ordenar los modelos evaluados de acuerdo con la métrica oficial del Benchmark Científico.
### Entradas: - benchmark_results
### Producto: - benchmark_ranking
### Responde: ¿Cuál es el orden de desempeño de los modelos evaluados bajo el protocolo oficial del Benchmark Científico?

# Métrica oficial del Benchmark --------------------------------------------
ranking_metric = BENCHMARK_CONFIG[
    "ranking_metric"
] # Métrica principal de comparación

# Construcción del ranking -------------------------------------------------
benchmark_ranking = sorted(
    benchmark_results,
    key = lambda result:
        result[ranking_metric]
) # Ranking oficial del Benchmark

# Validación ---------------------------------------------------------------
if not benchmark_ranking:
    raise ValueError(
        "No fue posible construir el ranking del Benchmark."
    )

# Verificación del ranking -------------------------------------------------
if len(benchmark_ranking) != len(benchmark_results):
    raise ValueError(
        "El ranking del Benchmark es inconsistente."
    )

print("-" * 80)

# BLOQUE 13. Selección del Modelo Ganador ----------------------------------
## Objetivo: Seleccionar el modelo con el mejor desempeño de acuerdo con el ranking
# oficial del Benchmark Científico.
### Entradas: - benchmark_ranking
### Producto: - best_model_result - best_model_config
### Responde: ¿Cuál es el modelo con el mejor desempeño predictivo bajo el protocolo oficial del Benchmark Científico?

# Validación del ranking ---------------------------------------------------
if not benchmark_ranking:
    raise ValueError(
        "El Benchmark no contiene modelos para seleccionar."
    )

# Selección del modelo ganador ---------------------------------------------
best_model_result = (
    benchmark_ranking[0]
) # Mejor modelo del Benchmark

# Recuperación de la configuración oficial ---------------------------------
if "model_config" in best_model_result:
    best_model_config = (
        best_model_result["model_config"]
    ) # Configuración almacenada durante el Benchmark

else:
    best_model_config = {
        "model_code": best_model_result["model_code"],
        "model_name": best_model_result["model_name"],
        "family": best_model_result["family"]
    } # Configuración mínima reconstruida

# Validación ---------------------------------------------------------------
required_keys = [
    "model_code",
    "model_name",
    "family"
] # Parámetros obligatorios

missing_keys = [
    key
    for key in required_keys
    if key not in best_model_config
] # Parámetros faltantes

if missing_keys:
    raise ValueError(
        f"Configuración incompleta del modelo ganador: {missing_keys}"
    )

# Verificación -------------------------------------------------------------
print("\nModelo ganador del Benchmark Científico\n")
print(f"Código : {best_model_config['model_code']}")
print(f"Modelo : {best_model_config['model_name']}")
print(f"Familia: {best_model_config['family']}")

print("-" * 80)

# BLOQUE 14. Exportación de Resultados -------------------------------------
## Objetivo: Exportar los productos oficiales generados durante el Benchmark
# Científico para garantizar la reproducibilidad del experimento.
### Entradas: - benchmark_results - benchmark_ranking - best_model_config
### Producto: - benchmark_results.joblib - benchmark_metrics.parquet - benchmark_summary.csv
# - benchmark_ranking.csv - best_mdel_config.json
### Responde: ¿Los productos oficiales del Benchmark fueron exportados correctamente?
### Responde: ¿Los productos oficiales del Benchmark fueron exportados correctamente?
# Justificación del formato de almacenamiento ------------------------------
# El resultado completo del Benchmark Científico se almacena en formato
# Joblib para preservar íntegramente los objetos generados durante la
# experimentación, incluyendo modelos entrenados, estructuras de datos,
# arreglos NumPy y demás atributos asociados.
#
# Los formatos tabulares como Parquet o CSV están diseñados para datos
# estructurados y no permiten serializar de forma segura objetos complejos
# de Python, como modelos de Scikit-Learn, modelos de PyTorch, tensores o
# instancias personalizadas.
#
# Por esta razón, el proyecto utiliza:
# - Joblib para almacenar los objetos completos del Benchmark.
# - Parquet para tablas de métricas y resultados tabulares.
# - CSV para reportes resumidos y productos de consulta.
# - JSON para configuraciones y metadatos del modelo ganador.
#
# Esta estrategia garantiza la reproducibilidad del Benchmark Científico,
# facilita la recuperación exacta de los modelos entrenados y sigue las
# prácticas recomendadas para proyectos de Machine Learning en producción.

# Resultado completo del Benchmark ----------------------------------------
joblib.dump(
    benchmark_results,
    BENCHMARK_RESULTS_FILE
) # Exportar resultado completo del Benchmark

# Construcción de tablas serializables ------------------------------------
benchmark_metrics = pd.DataFrame([
    {
        key: value
        for key, value in result.items()
        if key != "model"
    }
    for result in benchmark_results
]) # Tabla oficial de métricas

benchmark_ranking_df = pd.DataFrame([
    {
        key: value
        for key, value in result.items()
        if key != "model"
    }
    for result in benchmark_ranking
]) # Ranking oficial

benchmark_summary = (
    benchmark_metrics[
        [
            "model_code",
            "model_name",
            "family",
            "rmse",
            "mae",
            "mape",
            "r2",
            "adjusted_r2",
            "training_time",
            "inference_time"
        ]
    ]
) # Resumen ejecutivo

# Exportación de métricas --------------------------------------------------
benchmark_metrics.to_parquet(
    BENCHMARK_METRICS_FILE,
    index = False
) # Exportar métricas

# Exportación del ranking --------------------------------------------------
benchmark_ranking_df.to_csv(
    BENCHMARK_RANKING_FILE,
    index = False
) # Exportar ranking

# Exportación del resumen --------------------------------------------------
benchmark_summary.to_csv(
    BENCHMARK_SUMMARY_FILE,
    index = False
) # Exportar resumen ejecutivo

# Exportación de la configuración del modelo ganador -----------------------
with open(
    BEST_MODEL_CONFIG_FILE,
    "w",
    encoding = "utf-8"
) as file:

    json.dump(
        best_model_config,
        file,
        indent = 4,
        ensure_ascii = False
    ) # Exportar configuración

# Validación ---------------------------------------------------------------
exported_files = [
    BENCHMARK_RESULTS_FILE,
    BENCHMARK_METRICS_FILE,
    BENCHMARK_RANKING_FILE,
    BENCHMARK_SUMMARY_FILE,
    BEST_MODEL_CONFIG_FILE
] # Archivos oficiales

missing_files = [
    file
    for file in exported_files
    if not file.exists()
] # Archivos faltantes

if missing_files:
    raise FileNotFoundError(
        f"No fue posible exportar todos los productos del Benchmark: {missing_files}"
    )

print("\nProductos oficiales del Benchmark exportados correctamente.")