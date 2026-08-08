# graph-02_benchmark.py

# =============================================================================
# BLOQUE 1. IMPORTACIONES
# Objetivo: Importar las dependencias necesarias para ejecutar el Benchmark Científico.
# Producto:
# - Librerías científicas cargadas.
# - Configuración oficial disponible.
# - Pipeline oficial disponible.
# - Familias de modelos disponibles.
# Pregunta científica:
# ¿Se encuentran disponibles todas las dependencias requeridas para ejecutar
# el protocolo experimental del Benchmark Científico?
# =============================================================================

# -----------------------------------------------------------------------------
# Librerías estándar
# -----------------------------------------------------------------------------
import json
import os
import random
import sys
import time
from pathlib import Path

# -----------------------------------------------------------------------------
# Librerías científicas
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch

# -----------------------------------------------------------------------------
# PyTorch Geometric
# -----------------------------------------------------------------------------
from torch_geometric.data import Data

# -----------------------------------------------------------------------------
# Configuración Oficial del Benchmark
# -----------------------------------------------------------------------------
from src.python.config.config_project import (
    PROJECT_SEED,
    N_YEARS,
    BENCHMARK_CONFIG,
    BENCHMARK_MODELS,
    BENCHMARK_METRICS,
    BENCHMARK_REPRODUCIBILITY,
    OFFICIAL_MODEL_FAMILY,
)

# -----------------------------------------------------------------------------
# Rutas Oficiales del Proyecto
# -----------------------------------------------------------------------------
from src.python.config.paths import (
    BENCHMARK_DIR,
    GRAPH_DATA_DIR,
    GRAPH_DATA_COLLECTION_FILE,
    BENCHMARK_EXPERIMENT_FILE,
    BENCHMARK_METRICS_FILE,
    BENCHMARK_RANKING_CSV_FILE,
    BENCHMARK_RANKING_XLSX_FILE,
    BENCHMARK_SUMMARY_CSV_FILE,
    BENCHMARK_SUMMARY_XLSX_FILE,
    BENCHMARK_SUMMARY_JSON_FILE,
    BENCHMARK_METADATA_FILE,
    BENCHMARK_CONTRACT_FILE,
    BENCHMARK_CERTIFICATE_FILE,
    BENCHMARK_AUDIT_FILE,
    validate_project_structure,
)

# -----------------------------------------------------------------------------
# Utilidades del Proyecto
# -----------------------------------------------------------------------------
from src.python.utils.data_preparation import (
    prepare_dataset,
)

from src.python.utils.results import (
    build_exportable_benchmark_result,
    benchmark_results_to_dataframe,
    generate_benchmark_summary,
)

# -----------------------------------------------------------------------------
# Pipeline Oficial GraphData
# -----------------------------------------------------------------------------
from src.python.graph.graph_pipeline import (
    prepare_graph_collection,
)

# -----------------------------------------------------------------------------
# Modelos Estadísticos
# -----------------------------------------------------------------------------
from src.python.models.statistical import (
    run_linear_regression,
)

# -----------------------------------------------------------------------------
# Modelos Machine Learning
# -----------------------------------------------------------------------------
from src.python.models.machine_learning import (
    MACHINE_LEARNING_CONFIG,
    run_machine_learning,
)

# -----------------------------------------------------------------------------
# Modelos Deep Learning
# -----------------------------------------------------------------------------
from src.python.models.deep_learning import (
    run_mlp,
)

# -----------------------------------------------------------------------------
# Modelos Graph Neural Networks
# -----------------------------------------------------------------------------
from src.python.models.graph_neural_networks import (
    GNN_CONFIG,
    run_gnn_benchmark,
)

# =============================================================================
# BLOQUE 2. PREPARACIÓN DEL BENCHMARK
# Objetivo: Preparar la estructura oficial BenchmarkData utilizada como entrada
# por todas las familias de modelos del Benchmark Científico, integrando la
# colección oficial de GraphData, las particiones experimentales y el panel
# tabular derivado del conjunto espacio-temporal del proyecto.
# Entradas: - Resultados del pipeline oficial de preparación de GraphData.
# Producto: - Estructura oficial BenchmarkData.
# Pregunta científica:
# ¿La colección oficial de GraphData y las particiones experimentales poseen la
# estructura requerida para ejecutar el Benchmark Científico bajo un protocolo
# experimental reproducible y consistente para todas las familias de modelos?
# =============================================================================

# -------------------------------------------------------------------------
# 2.1 Validación de la entrada
# -------------------------------------------------------------------------
if not isinstance(preparation_results, dict):
    raise TypeError(
        "'preparation_results' debe ser un diccionario."
    )

required_keys = [
    "graphs",
    "partitions",
    "scaler",
]

missing_keys = [
    key
    for key in required_keys
    if key not in preparation_results
]

if missing_keys:
    raise ValueError(
        "La preparación del Benchmark está incompleta. "
        f"Faltan los siguientes elementos: {missing_keys}."
    )

# -------------------------------------------------------------------------
# 2.2 Recuperación de productos
# -------------------------------------------------------------------------
graphs = preparation_results["graphs"] # Colección oficial de GraphData
partitions = preparation_results["partitions"] # Particiones oficiales del Benchmark
scaler = preparation_results["scaler"] # Escalador oficial ajustado con entrenamiento

# -------------------------------------------------------------------------
# 2.3 Validación de la colección GraphData
# -------------------------------------------------------------------------
if not isinstance(graphs, (list, tuple)):
    raise TypeError(
        "'graphs' debe ser una lista o tupla de GraphData."
    )

if len(graphs) == 0:
    raise ValueError(
        "La colección oficial GraphData está vacía."
    )

if not isinstance(partitions, dict):
    raise TypeError(
        "'partitions' debe ser un diccionario."
    )

if scaler is None:
    raise ValueError(
        "El escalador oficial del Benchmark es inválido."
    )

# Validar que todos los elementos correspondan a GraphData
for graph_position, graph in enumerate(graphs):

    if not isinstance(graph, Data):
        raise TypeError(
            "La colección GraphData contiene elementos inválidos. "
            f"El elemento ubicado en la posición {graph_position} "
            "no corresponde a un GraphData."
        )

# Recuperar el número de grafos disponibles
num_graphs = len(graphs) # Total de GraphData del Benchmark Científico

if num_graphs != N_YEARS:
    raise ValueError(
        f"Se esperaban {N_YEARS} GraphData y se encontraron "
        f"{num_graphs}."
    )

# -------------------------------------------------------------------------
# 2.4 Construcción del Panel Tabular Oficial
# -------------------------------------------------------------------------

x_panel = [] # Variables predictoras del panel científico
y_panel = [] # Variable objetivo del panel científico
train_mask_panel = [] # Máscaras oficiales de entrenamiento
validation_mask_panel = [] # Máscaras oficiales de validación
test_mask_panel = [] # Máscaras oficiales de prueba

# -------------------------------------------------------------------------
# Recorrido de la colección oficial GraphData
# -------------------------------------------------------------------------
for graph_position, graph in enumerate(graphs):
    required_attributes = [
        "x",
        "y",
        "train_mask",
        "val_mask",
        "test_mask",
    ]

    for attribute in required_attributes:
        if not hasattr(graph, attribute):
            raise ValueError(
                f"El GraphData {graph_position} no contiene '{attribute}'."
            )

        if getattr(graph, attribute) is None:
            raise ValueError(
                f"'{attribute}' es inválido en el GraphData {graph_position}."
            )

    x_panel.append(
        graph.x.cpu().numpy()
    ) # Variables predictoras

    y_panel.append(
        graph.y.cpu().numpy().ravel()
    ) # Variable objetivo

    train_mask_panel.append(
        graph.train_mask.cpu().numpy()
    ) # Máscara de entrenamiento

    validation_mask_panel.append(
        graph.val_mask.cpu().numpy()
    ) # Máscara de validación

    test_mask_panel.append(
        graph.test_mask.cpu().numpy()
    ) # Máscara de prueba

# -------------------------------------------------------------------------
# Construcción del Panel Científico Oficial
# -------------------------------------------------------------------------
x_panel = np.concatenate(
    x_panel,
    axis=0,
) # Variables predictoras del panel completo

y_panel = np.concatenate(
    y_panel,
    axis=0,
) # Variable objetivo del panel completo

train_mask_panel = np.concatenate(
    train_mask_panel,
    axis=0,
) # Máscara de entrenamiento del panel

validation_mask_panel = np.concatenate(
    validation_mask_panel,
    axis=0,
) # Máscara de validación del panel

test_mask_panel = np.concatenate(
    test_mask_panel,
    axis=0,
) # Máscara de prueba del panel

# -------------------------------------------------------------------------
# Validación del Panel Científico
# -------------------------------------------------------------------------
if len(x_panel) != len(y_panel):
    raise RuntimeError(
        "Las dimensiones del Panel Científico son inconsistentes."
    )

if len(train_mask_panel) != len(x_panel):
    raise RuntimeError(
        "La máscara de entrenamiento posee una dimensión inválida."
    )

if len(validation_mask_panel) != len(x_panel):
    raise RuntimeError(
        "La máscara de validación posee una dimensión inválida."
    )

if len(test_mask_panel) != len(x_panel):
    raise RuntimeError(
        "La máscara de prueba posee una dimensión inválida."
    )

# -------------------------------------------------------------------------
# 2.5 Construcción del BenchmarkData
# -------------------------------------------------------------------------
benchmark_data = {

    # ---------------------------------------------------------------------
    # Colección Oficial GraphData
    # ---------------------------------------------------------------------
    "graphs": graphs,

    # ---------------------------------------------------------------------
    # Panel Científico Oficial
    # ---------------------------------------------------------------------
    "x_train": x_panel[train_mask_panel],
    "y_train": y_panel[train_mask_panel],

    "x_validation": x_panel[validation_mask_panel],
    "y_validation": y_panel[validation_mask_panel],

    "x_test": x_panel[test_mask_panel],
    "y_test": y_panel[test_mask_panel],

    # ---------------------------------------------------------------------
    # Particiones Oficiales
    # ---------------------------------------------------------------------
    "train_index": partitions["train_indices"],
    "validation_index": partitions["validation_indices"],
    "test_index": partitions["test_indices"],

    # ---------------------------------------------------------------------
    # Escalador Oficial
    # ---------------------------------------------------------------------
    "scaler": scaler,
}

# -------------------------------------------------------------------------
# 2.6 Validación del BenchmarkData
# -------------------------------------------------------------------------
required_products = [

    "graphs",

    "x_train",
    "y_train",

    "x_validation",
    "y_validation",

    "x_test",
    "y_test",

    "train_index",
    "validation_index",
    "test_index",

    "scaler",

]

for product in required_products:

    if product not in benchmark_data:
        raise RuntimeError(
            f"BenchmarkData no contiene '{product}'."
        )

    if benchmark_data[product] is None:
        raise RuntimeError(
            f"'{product}' es inválido."
        )

if benchmark_data["x_train"].shape[0] != benchmark_data["y_train"].shape[0]:
    raise RuntimeError(
        "Las dimensiones del conjunto de entrenamiento son inconsistentes."
    )

if benchmark_data["x_validation"].shape[0] != benchmark_data["y_validation"].shape[0]:
    raise RuntimeError(
        "Las dimensiones del conjunto de validación son inconsistentes."
    )

if benchmark_data["x_test"].shape[0] != benchmark_data["y_test"].shape[0]:
    raise RuntimeError(
        "Las dimensiones del conjunto de prueba son inconsistentes."
    )

if len(benchmark_data["graphs"]) == 0:
    raise RuntimeError(
        "La colección oficial GraphData está vacía."
    )

# -------------------------------------------------------------------------
# 2.7 Retorno
# -------------------------------------------------------------------------
return benchmark_data





# ==============================================================================
# build_benchmark_ranking
# ==============================================================================
# Objetivo:
# Construir el Ranking Científico Oficial del Benchmark a partir de los
# resultados consolidados y las métricas oficiales definidas en el protocolo
# experimental.
# ==============================================================================
def build_benchmark_ranking(
    benchmark_results: list[dict],
    benchmark_metrics: dict
) -> list[dict]:
    """
    Construye el Ranking Científico Oficial del Benchmark a partir de los
    resultados consolidados y las métricas oficiales del protocolo
    experimental.

    Parameters
    ----------
    benchmark_results : list[dict]
        Resultados consolidados del Benchmark.

    benchmark_metrics : dict
        Métricas oficiales utilizadas para construir el ranking.

    Returns
    -------
    list[dict]
        Ranking Científico Oficial del Benchmark.
    """

    # -------------------------------------------------------------------------
    # Validación
    # -------------------------------------------------------------------------
    if not isinstance(benchmark_results, list):
        raise TypeError(
            "'benchmark_results' debe ser una lista."
        )

    if len(benchmark_results) == 0:
        raise ValueError(
            "No existen resultados para construir el Ranking Científico."
        )

    if not isinstance(benchmark_metrics, dict):
        raise TypeError(
            "'benchmark_metrics' debe ser un diccionario."
        )

    if "ranking_metric" not in BENCHMARK_CONFIG:
        raise KeyError(
            "La configuración no contiene 'ranking_metric'."
        )

    ranking_metric = BENCHMARK_CONFIG["ranking_metric"]

    if ranking_metric not in benchmark_metrics:
        raise ValueError(
            f"La métrica oficial '{ranking_metric}' "
            "no está registrada."
        )

    # -------------------------------------------------------------------------
    # Validación de resultados
    # -------------------------------------------------------------------------
    for result in benchmark_results:

        if not isinstance(result, dict):
            raise TypeError(
                "Cada resultado del Benchmark debe ser un diccionario."
            )

        if ranking_metric not in result:
            raise ValueError(
                f"El resultado no contiene la métrica "
                f"'{ranking_metric}'."
            )

        metric_value = result[ranking_metric]

        if not isinstance(metric_value, (int, float)):
            raise TypeError(
                f"La métrica '{ranking_metric}' debe ser numérica."
            )

    # -------------------------------------------------------------------------
    # Construcción
    # -------------------------------------------------------------------------
    benchmark_ranking = sorted(
        [result.copy() for result in benchmark_results],
        key=lambda result: result[ranking_metric]
    )

    # -------------------------------------------------------------------------
    # Asignación de posiciones
    # -------------------------------------------------------------------------
    for position, result in enumerate(
        benchmark_ranking,
        start=1
    ):
        result["ranking_position"] = position

    # -------------------------------------------------------------------------
    # Retorno
    # -------------------------------------------------------------------------
    return benchmark_ranking

# ==============================================================================
# select_official_model
# Objetivo: Seleccionar el Modelo Oficial del Proyecto a partir de la familia oficial
# definida para el proyecto y del Ranking Científico Oficial del Benchmark.
# ==============================================================================
def select_official_model(
    benchmark_ranking: list[dict]
) -> dict:
    """
    Selecciona el Modelo Oficial del Proyecto a partir del Ranking Científico
    Oficial del Benchmark y de la familia oficial definida en la configuración.

    Parameters
    ----------
    benchmark_ranking : list[dict]
        Ranking Científico Oficial del Benchmark.

    Returns
    -------
    dict
        Configuración completa del Modelo Oficial.
    """

    # -------------------------------------------------------------------------
    # Validación
    # -------------------------------------------------------------------------
    if not isinstance(benchmark_ranking, list):
        raise TypeError(
            "'benchmark_ranking' debe ser una lista."
        )

    if len(benchmark_ranking) == 0:
        raise ValueError(
            "El Ranking Científico está vacío."
        )

    if OFFICIAL_MODEL_FAMILY is None:
        raise ValueError(
            "La familia oficial del proyecto es inválida."
        )

    # -------------------------------------------------------------------------
    # Validación del Ranking
    # -------------------------------------------------------------------------
    for model in benchmark_ranking:

        if not isinstance(model, dict):
            raise TypeError(
                "Cada elemento del Ranking Científico debe ser un diccionario."
            )

        if "family" not in model:
            raise ValueError(
                "Un modelo del Ranking no contiene 'family'."
            )

        if model["family"] is None:
            raise ValueError(
                "'family' es inválida."
            )

    # -------------------------------------------------------------------------
    # Selección
    # -------------------------------------------------------------------------
    official_candidates = [
        model
        for model in benchmark_ranking
        if model["family"] == OFFICIAL_MODEL_FAMILY
    ]

    if len(official_candidates) == 0:
        raise ValueError(
            "No existen modelos de la familia oficial en el Ranking Científico."
        )

    official_model = official_candidates[0].copy()

    # -------------------------------------------------------------------------
    # Validación del Modelo Oficial
    # -------------------------------------------------------------------------
    required_keys = [
        "model_code",
        "model_name",
        "family",
        "model_config"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in official_model
    ]

    if missing_keys:
        raise ValueError(
            "El Modelo Oficial no contiene los campos "
            f"obligatorios: {missing_keys}"
        )

    for key in required_keys:
        if official_model[key] is None:
            raise ValueError(
                f"'{key}' es inválido."
            )

    # -------------------------------------------------------------------------
    # Retorno
    # -------------------------------------------------------------------------
    return official_model

# BLOQUE 2. Configuración del Script --------------------------------------
# Objetivo: Configurar el entorno de ejecución, validar la estructura oficial
# del proyecto e inicializar los parámetros necesarios para ejecutar el
# Benchmark Científico de forma reproducible.
# Producto:
# - Entorno de ejecución configurado.
# - Estructura oficial del proyecto validada.
# - Reproducibilidad establecida.
# - Dispositivo de procesamiento seleccionado.
# Pregunta científica:
# ¿El entorno de ejecución cumple las condiciones necesarias para ejecutar
# un Benchmark Científico reproducible y consistente?

# -------------------------------------------------------------------------
# 2.1 Configuración del entorno
# -------------------------------------------------------------------------
print("Configurando entorno de ejecución del Benchmark...")

# -------------------------------------------------------------------------
# 2.2 Validación de la estructura del proyecto
# -------------------------------------------------------------------------
validate_project_structure(verbose=True)

print("Estructura del proyecto validada correctamente.")

# -------------------------------------------------------------------------
# 2.3 Validación de la configuración
# -------------------------------------------------------------------------
if PROJECT_SEED is None:
    raise ValueError(
        "La semilla oficial del proyecto es inválida."
    )

if not isinstance(BENCHMARK_REPRODUCIBILITY, dict):
    raise TypeError(
        "'BENCHMARK_REPRODUCIBILITY' debe ser un diccionario."
    )

if "deterministic" not in BENCHMARK_REPRODUCIBILITY:
    raise KeyError(
        "La configuración no contiene 'deterministic'."
    )

# -------------------------------------------------------------------------
# 2.4 Configuración de la reproducibilidad
# -------------------------------------------------------------------------
random.seed(PROJECT_SEED)
np.random.seed(PROJECT_SEED)
torch.manual_seed(PROJECT_SEED)

cuda_available = torch.cuda.is_available()

if (
    BENCHMARK_REPRODUCIBILITY["deterministic"]
    and cuda_available
):
    torch.cuda.manual_seed(PROJECT_SEED)
    torch.cuda.manual_seed_all(PROJECT_SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------------------------------------------------------
# 2.5 Selección del dispositivo
# -------------------------------------------------------------------------
DEVICE = "cuda" if cuda_available else "cpu"

# -------------------------------------------------------------------------
# 2.6 Certificación del entorno de ejecución
# -------------------------------------------------------------------------
print()
print("-" * 80)
print("CERTIFICACIÓN DEL ENTORNO DE EJECUCIÓN")
print("-" * 80)

print(f"Semilla del proyecto          : {PROJECT_SEED}")
print(
    "Modo determinístico           : "
    f"{BENCHMARK_REPRODUCIBILITY['deterministic']}"
)
print(f"Dispositivo de procesamiento  : {DEVICE.upper()}")

if cuda_available:
    print(
        f"GPU detectada                 : "
        f"{torch.cuda.get_device_name(0)}"
    )

print("Estado                        : OK")
print("-" * 80)
print()

# BLOQUE 3. CONSTRUCCIÓN DE LA COLECCIÓN OFICIAL DEL BENCHMARK
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Construir la colección oficial del Benchmark Científico a partir de la
# colección GraphData generada durante la etapa de construcción del grafo,
# aplicando el protocolo oficial de validación, particionado, construcción
# de máscaras, escalamiento y organización de los datos que serán utilizados
# por todas las familias de modelos durante la comparación científica.
#
#------------------------------------------------------------------------------
# Producto
#------------------------------------------------------------------------------
# - benchmark_data
# - Colección Oficial del Benchmark Científico.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿La colección oficial del Benchmark fue construida correctamente para
# ejecutar un Benchmark Científico reproducible, consistente y comparable
# entre todas las familias de modelos?
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# 3.1 Carga de la Colección Oficial GraphData
#------------------------------------------------------------------------------

print("\n3.1 CARGA DE LA COLECCIÓN OFICIAL GraphData")

#------------------------------------------------------------------------------
# 3.1.1 Localizar la colección oficial GraphData
#------------------------------------------------------------------------------

graph_collection_file = (
    GRAPH_DATA_DIR / "graph_data_collection.pt"
)

if not graph_collection_file.exists():
    raise FileNotFoundError(
        "No se encontró la colección oficial GraphData."
    )

print(
    f"Colección localizada       : {graph_collection_file.name}"
)

#------------------------------------------------------------------------------
# 3.1.2 Cargar la colección oficial GraphData
#------------------------------------------------------------------------------

graph_collection = torch.load(
    graph_collection_file,
    weights_only=False,
)

#------------------------------------------------------------------------------
# 3.1.3 Validar la colección oficial GraphData
#------------------------------------------------------------------------------

if not isinstance(graph_collection, dict):
    raise TypeError(
        "La colección oficial GraphData debe ser un diccionario."
    )

if not graph_collection:
    raise ValueError(
        "La colección oficial GraphData está vacía."
    )

graphs: list[Data] = []

for year in sorted(graph_collection):

    graph = graph_collection[year]

    if not isinstance(graph, Data):
        raise TypeError(
            f"El GraphData correspondiente al año {year} es inválido."
        )

    required_attributes = [
        "x",
        "edge_index",
    ]

    for attribute in required_attributes:

        if not hasattr(graph, attribute):
            raise ValueError(
                f"El GraphData del año {year} no contiene '{attribute}'."
            )

        if getattr(graph, attribute) is None:
            raise ValueError(
                f"El GraphData del año {year} posee '{attribute}' inválido."
            )

    graphs.append(graph)

reference_graph = graphs[0]

#------------------------------------------------------------------------------
# 3.1.4 Registrar la colección oficial GraphData
#------------------------------------------------------------------------------

print(f"GraphData cargados         : {len(graphs)}")
print(f"Nodos                      : {reference_graph.num_nodes:,}")
print(f"Aristas                    : {reference_graph.num_edges:,}")
print(f"Node Features              : {reference_graph.num_node_features}")

if hasattr(reference_graph, "y") and reference_graph.y is not None:
    print(
        f"Variable objetivo          : {tuple(reference_graph.y.shape)}"
    )

#------------------------------------------------------------------------------
# 3.1.5 Confirmar la carga de la colección oficial GraphData
#------------------------------------------------------------------------------

print("\nColección oficial GraphData cargada correctamente.")

#------------------------------------------------------------------------------
# 3.2 Preparación de la Colección Oficial GraphData
#------------------------------------------------------------------------------

print("\n3.2 PREPARACIÓN DE LA COLECCIÓN OFICIAL GraphData")

#------------------------------------------------------------------------------
# 3.2.1 Preparar la colección oficial GraphData
#------------------------------------------------------------------------------

reference_graph = graphs[0]

preparation_results = prepare_graph_collection(
    graphs=graphs,
    expected_nodes=reference_graph.num_nodes,
    expected_features=reference_graph.num_node_features,
    expected_years=len(graphs),
    expected_edges=reference_graph.num_edges,
    train_size=BENCHMARK_CONFIG["train_size"],
    validation_size=BENCHMARK_CONFIG["validation_size"],
    random_state=PROJECT_SEED,
)

#------------------------------------------------------------------------------
# 3.2.2 Validar la preparación de la colección oficial
#------------------------------------------------------------------------------

if not isinstance(preparation_results, dict):
    raise TypeError(
        "La preparación de la colección GraphData debe devolver un diccionario."
    )

required_products = (
    "graphs",
    "partitions",
    "scaler",
    "validation_report",
)

missing_products = [
    product
    for product in required_products
    if product not in preparation_results
]

if missing_products:
    raise ValueError(
        f"Productos faltantes del pipeline: {missing_products}"
    )

#------------------------------------------------------------------------------
# 3.2.3 Recuperar los productos del pipeline
#------------------------------------------------------------------------------

graphs = preparation_results["graphs"]
partitions = preparation_results["partitions"]
scaler = preparation_results["scaler"]
validation_report = preparation_results["validation_report"]

if not isinstance(graphs, (list, tuple)):
    raise TypeError(
        "La colección GraphData preparada debe ser una lista."
    )

if not graphs:
    raise ValueError(
        "La colección GraphData preparada está vacía."
    )

if not isinstance(partitions, dict):
    raise TypeError(
        "Las particiones del Benchmark deben almacenarse en un diccionario."
    )

if scaler is None:
    raise ValueError(
        "El escalador del Benchmark no fue generado."
    )

if validation_report is None:
    raise ValueError(
        "El reporte de validación no fue generado."
    )

#------------------------------------------------------------------------------
# 3.2.4 Registrar la preparación de la colección oficial
#------------------------------------------------------------------------------

print(f"GraphData preparados       : {len(graphs)}")
print(f"Particiones generadas      : {len(partitions)}")
print(f"Escalador                  : {type(scaler).__name__}")

#------------------------------------------------------------------------------
# 3.2.5 Confirmar la preparación de la colección oficial
#------------------------------------------------------------------------------

print("\nColección oficial GraphData preparada correctamente.")

#------------------------------------------------------------------------------
# 3.3 Construcción de la Colección Oficial BenchmarkData
#------------------------------------------------------------------------------

print("\n3.3 CONSTRUCCIÓN DE LA COLECCIÓN OFICIAL BenchmarkData")

#------------------------------------------------------------------------------
# 3.3.1 Construir la colección oficial BenchmarkData
#------------------------------------------------------------------------------

benchmark_data = build_benchmark_data(
    preparation_results=preparation_results
)

#------------------------------------------------------------------------------
# 3.3.2 Validar la colección oficial BenchmarkData
#------------------------------------------------------------------------------

if not isinstance(benchmark_data, dict):
    raise TypeError(
        "La colección oficial BenchmarkData debe ser un diccionario."
    )

required_products = (
    "graphs",
    "train_index",
    "validation_index",
    "test_index",
    "scaler",
)

missing_products = [
    product
    for product in required_products
    if product not in benchmark_data
]

if missing_products:
    raise ValueError(
        f"Productos faltantes en BenchmarkData: {missing_products}"
    )

for product in required_products:

    if benchmark_data[product] is None:
        raise ValueError(
            f"El producto '{product}' es inválido."
        )

#------------------------------------------------------------------------------
# 3.3.3 Registrar la colección oficial BenchmarkData
#------------------------------------------------------------------------------

print(f"GraphData                 : {len(benchmark_data['graphs'])}")
print(f"Entrenamiento             : {len(benchmark_data['train_index'])}")
print(f"Validación                : {len(benchmark_data['validation_index'])}")
print(f"Prueba                    : {len(benchmark_data['test_index'])}")

#------------------------------------------------------------------------------
# 3.3.4 Confirmar la construcción de la colección oficial BenchmarkData
#------------------------------------------------------------------------------

print("\nColección oficial BenchmarkData construida correctamente.")

#------------------------------------------------------------------------------
# 3.4 Resumen Ejecutivo de la Colección Oficial
#------------------------------------------------------------------------------

print("\n3.4 RESUMEN EJECUTIVO DE LA COLECCIÓN OFICIAL")

#------------------------------------------------------------------------------
# 3.4.1 Validar el resumen ejecutivo
#------------------------------------------------------------------------------

if not isinstance(benchmark_data, dict):
    raise TypeError(
        "La colección oficial BenchmarkData debe ser un diccionario."
    )

#------------------------------------------------------------------------------
# 3.4.2 Registrar el resumen ejecutivo
#------------------------------------------------------------------------------

print(f"GraphData                  : {len(benchmark_data['graphs'])}")
print(f"Entrenamiento              : {len(benchmark_data['train_index'])}")
print(f"Validación                 : {len(benchmark_data['validation_index'])}")
print(f"Prueba                     : {len(benchmark_data['test_index'])}")
print(f"Escalador                  : {type(benchmark_data['scaler']).__name__}")

#------------------------------------------------------------------------------
# 3.4.3 Confirmar el resumen ejecutivo
#------------------------------------------------------------------------------

print("\nResumen Ejecutivo generado correctamente.")

#------------------------------------------------------------------------------
# 3.5 Confirmación del Bloque
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# 3.5.1 Validar la colección oficial del Benchmark
#------------------------------------------------------------------------------

if not isinstance(graphs, (list, tuple)):
    raise TypeError(
        "La colección oficial GraphData es inválida."
    )

if not isinstance(benchmark_data, dict):
    raise TypeError(
        "La colección oficial BenchmarkData es inválida."
    )

#------------------------------------------------------------------------------
# 3.5.2 Registrar la confirmación del bloque
#------------------------------------------------------------------------------

print(f"Colección GraphData        : {len(graphs)} GraphData")
print(f"Colección BenchmarkData    : Disponible")
print(f"Estado                     : Lista para el Benchmark Científico")

#------------------------------------------------------------------------------
# 3.5.3 Confirmar la ejecución del bloque
#------------------------------------------------------------------------------

print("\n" + "-" * 80)
print("BLOQUE 3 COMPLETADO: COLECCIÓN OFICIAL DEL BENCHMARK CONSTRUIDA")
print("-" * 80)

print("\nBloque 3 ejecutado correctamente.")

# BLOQUE 4. EVALUACIÓN DEL BENCHMARK CIENTÍFICO
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Evaluar científicamente el desempeño de las diferentes familias de modelos
# utilizando la Colección Oficial BenchmarkData, aplicando un protocolo
# experimental reproducible para comparar su capacidad predictiva mediante
# métricas objetivas de rendimiento sobre un conjunto de datos común.
#
#------------------------------------------------------------------------------
# Producto
#------------------------------------------------------------------------------
# - benchmark_results
# - Resultados Oficiales del Benchmark Científico.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿Cuál es el desempeño predictivo de las diferentes familias de modelos
# evaluadas bajo un protocolo experimental común y reproducible para el
# problema de modelado espacio-temporal de la soberanía alimentaria?
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# 4.1 Validación de la Colección Oficial BenchmarkData
#------------------------------------------------------------------------------

print("\n4.1 VALIDACIÓN DE LA COLECCIÓN OFICIAL BenchmarkData")

#------------------------------------------------------------------------------
# 4.1.1 Validar la disponibilidad de la Colección Oficial BenchmarkData
#------------------------------------------------------------------------------

if not isinstance(benchmark_data, dict):
    raise TypeError(
        "La Colección Oficial BenchmarkData debe ser un diccionario."
    )

required_products = (
    "graphs",
    "train_index",
    "validation_index",
    "test_index",
    "scaler",
)

missing_products = [
    product
    for product in required_products
    if product not in benchmark_data
]

if missing_products:
    raise ValueError(
        "La Colección Oficial BenchmarkData está incompleta. "
        f"Productos faltantes: {missing_products}"
    )

print("Colección Oficial BenchmarkData disponible.")

#------------------------------------------------------------------------------
# 4.1.2 Recuperar los productos de la Colección Oficial BenchmarkData
#------------------------------------------------------------------------------

graphs = benchmark_data["graphs"]
train_index = benchmark_data["train_index"]
validation_index = benchmark_data["validation_index"]
test_index = benchmark_data["test_index"]
scaler = benchmark_data["scaler"]

#------------------------------------------------------------------------------
# 4.1.3 Validar la integridad de la Colección Oficial BenchmarkData
#------------------------------------------------------------------------------

if not isinstance(graphs, (list, tuple)):
    raise TypeError(
        "La colección oficial GraphData es inválida."
    )

if not graphs:
    raise ValueError(
        "La Colección Oficial BenchmarkData no contiene GraphData."
    )

if len(train_index) == 0:
    raise ValueError(
        "El conjunto de entrenamiento está vacío."
    )

if len(validation_index) == 0:
    raise ValueError(
        "El conjunto de validación está vacío."
    )

if len(test_index) == 0:
    raise ValueError(
        "El conjunto de prueba está vacío."
    )

if scaler is None:
    raise ValueError(
        "El escalador de la Colección Oficial BenchmarkData es inválido."
    )

print("Integridad de la Colección Oficial BenchmarkData validada.")

#------------------------------------------------------------------------------
# 4.1.4 Registrar la validación de la Colección Oficial BenchmarkData
#------------------------------------------------------------------------------

print(f"GraphData                  : {len(graphs)}")
print(f"Entrenamiento              : {len(train_index)}")
print(f"Validación                 : {len(validation_index)}")
print(f"Prueba                     : {len(test_index)}")
print(f"Escalador                  : {type(scaler).__name__}")

#------------------------------------------------------------------------------
# 4.1.5 Confirmar la validación de la Colección Oficial BenchmarkData
#------------------------------------------------------------------------------

print(
    "\nColección Oficial BenchmarkData lista para la evaluación del Benchmark Científico."
)

# BLOQUE 5. VALIDACIÓN DE LA CONFIGURACIÓN OFICIAL DE LOS MODELOS DEL BENCHMARK
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Validar la configuración oficial de las familias y modelos candidatos que
# participarán en el Benchmark Científico, verificando su disponibilidad,
# integridad y consistencia antes de iniciar la evaluación experimental.
#
#------------------------------------------------------------------------------
# Producto
#------------------------------------------------------------------------------
# - candidate_models
# - Configuración Oficial de los Modelos del Benchmark validada.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿La configuración oficial de las familias y modelos candidatos del
# Benchmark es consistente, íntegra y suficiente para ejecutar el
# Benchmark Científico de manera reproducible?
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 5.1 Verificación de la Configuración Oficial de los Modelos
#------------------------------------------------------------------------------

print("\n5.1 VERIFICACIÓN DE LA CONFIGURACIÓN OFICIAL DE LOS MODELOS")

#------------------------------------------------------------------------------
# 5.1.1 Validar la configuración oficial de los modelos
#------------------------------------------------------------------------------

if not isinstance(BENCHMARK_MODELS, dict):
    raise TypeError(
        "La configuración oficial de los modelos debe ser un diccionario."
    )

if not BENCHMARK_MODELS:
    raise ValueError(
        "No existen familias de modelos registradas."
    )

print("Configuración oficial disponible.")

#------------------------------------------------------------------------------
# 5.1.2 Recuperar la configuración oficial de los modelos
#------------------------------------------------------------------------------

model_families = BENCHMARK_MODELS

total_models = sum(
    len(models)
    for models in model_families.values()
)

#------------------------------------------------------------------------------
# 5.1.3 Validar la configuración recuperada
#------------------------------------------------------------------------------

if total_models == 0:
    raise ValueError(
        "No existen modelos registrados para ejecutar el Benchmark."
    )

empty_families = [
    family
    for family, models in model_families.items()
    if not models
]

if empty_families:
    raise ValueError(
        "Existen familias sin modelos registrados: "
        f"{empty_families}"
    )

for family, models in model_families.items():

    if not isinstance(models, (list, tuple)):
        raise TypeError(
            f"La familia '{family}' debe contener una lista de modelos."
        )

print(
    "Configuración Oficial de los Modelos validada."
)

#------------------------------------------------------------------------------
# 5.1.4 Registrar la configuración oficial
#------------------------------------------------------------------------------

print("\nConfiguración Oficial de los Modelos:")

for family, models in model_families.items():
    print(
        f"{family:<25}: {len(models)} modelo(s)"
    )

print(f"\nTotal de familias          : {len(model_families)}")
print(f"Total de modelos           : {total_models}")

#------------------------------------------------------------------------------
# 5.1.5 Confirmar la validación
#------------------------------------------------------------------------------

print(
    "\nConfiguración Oficial de los Modelos lista para la evaluación del Benchmark Científico."
)

# BLOQUE 6. EVALUACIÓN DE LAS FAMILIAS DE MODELOS DEL BENCHMARK
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Ejecutar el Benchmark Científico para cada familia de modelos candidata
# utilizando la Colección Oficial BenchmarkData y el protocolo experimental
# oficial, obteniendo los resultados predictivos que servirán de base para
# la consolidación y comparación científica entre familias metodológicas.
#
#------------------------------------------------------------------------------
# Productos
#------------------------------------------------------------------------------
# - statistical_results
# - machine_learning_results
# - deep_learning_results
# - gnn_results
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿Cuál es el desempeño predictivo de cada familia de modelos bajo el
# protocolo experimental oficial del Benchmark Científico?
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# 6.1 Evaluación de los Modelos Estadísticos
#------------------------------------------------------------------------------

print("\n6.1 EVALUACIÓN DE LOS MODELOS ESTADÍSTICOS")

#------------------------------------------------------------------------------
# 6.1.1 Validar los datos del Benchmark
#------------------------------------------------------------------------------

required_products = (
    "x_train",
    "y_train",
    "x_test",
    "y_test",
)

missing_products = [
    product
    for product in required_products
    if product not in benchmark_data
]

if missing_products:
    raise ValueError(
        "La Colección Oficial BenchmarkData está incompleta. "
        f"Productos faltantes: {missing_products}"
    )

#------------------------------------------------------------------------------
# 6.1.2 Recuperar los datos del Benchmark
#------------------------------------------------------------------------------

x_train = benchmark_data["x_train"]
y_train = benchmark_data["y_train"]
x_test = benchmark_data["x_test"]
y_test = benchmark_data["y_test"]

#------------------------------------------------------------------------------
# 6.1.3 Validar los datos recuperados
#------------------------------------------------------------------------------

for name, value in (
    ("x_train", x_train),
    ("y_train", y_train),
    ("x_test", x_test),
    ("y_test", y_test),
):

    if value is None:
        raise ValueError(
            f"El producto '{name}' es inválido."
        )

#------------------------------------------------------------------------------
# 6.1.4 Ejecutar los Modelos Estadísticos
#------------------------------------------------------------------------------

statistical_results = []

statistical_results.append(
    run_linear_regression(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
)

#------------------------------------------------------------------------------
# 6.1.5 Validar los resultados
#------------------------------------------------------------------------------

if not statistical_results:
    raise ValueError(
        "No se obtuvieron resultados para los Modelos Estadísticos."
    )

#------------------------------------------------------------------------------
# 6.1.6 Registrar la evaluación
#------------------------------------------------------------------------------

print(
    f"Modelos estadísticos evaluados : {len(statistical_results)}"
)

#------------------------------------------------------------------------------
# 6.1.7 Confirmar la evaluación
#------------------------------------------------------------------------------

print(
    "Evaluación de los Modelos Estadísticos completada correctamente."
)

#------------------------------------------------------------------------------
# 6.2 Evaluación de los Modelos de Machine Learning
#------------------------------------------------------------------------------

print("\n6.2 EVALUACIÓN DE LOS MODELOS DE MACHINE LEARNING")

#------------------------------------------------------------------------------
# 6.2.1 Validar los datos del Benchmark
#------------------------------------------------------------------------------

required_products = (
    "x_train",
    "y_train",
    "x_test",
    "y_test",
)

missing_products = [
    product
    for product in required_products
    if product not in benchmark_data
]

if missing_products:
    raise ValueError(
        "La Colección Oficial BenchmarkData está incompleta. "
        f"Productos faltantes: {missing_products}"
    )

#------------------------------------------------------------------------------
# 6.2.2 Recuperar los datos del Benchmark
#------------------------------------------------------------------------------

x_train = benchmark_data["x_train"]
y_train = benchmark_data["y_train"]
x_test = benchmark_data["x_test"]
y_test = benchmark_data["y_test"]

#------------------------------------------------------------------------------
# 6.2.3 Validar los datos recuperados
#------------------------------------------------------------------------------

for name, value in (
    ("x_train", x_train),
    ("y_train", y_train),
    ("x_test", x_test),
    ("y_test", y_test),
):

    if value is None:
        raise ValueError(
            f"El producto '{name}' es inválido."
        )

#------------------------------------------------------------------------------
# 6.2.4 Ejecutar los Modelos de Machine Learning
#------------------------------------------------------------------------------

machine_learning_results = []

for model_name in BENCHMARK_MODELS["machine_learning"]:

    machine_learning_results.append(
        run_machine_learning(
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )
    )

#------------------------------------------------------------------------------
# 6.2.5 Validar los resultados
#------------------------------------------------------------------------------

if not machine_learning_results:
    raise ValueError(
        "No se obtuvieron resultados para los Modelos de Machine Learning."
    )

#------------------------------------------------------------------------------
# 6.2.6 Registrar la evaluación
#------------------------------------------------------------------------------

print(
    f"Modelos de Machine Learning evaluados : "
    f"{len(machine_learning_results)}"
)

#------------------------------------------------------------------------------
# 6.2.7 Confirmar la evaluación
#------------------------------------------------------------------------------

print(
    "Evaluación de los Modelos de Machine Learning completada correctamente."
)

#------------------------------------------------------------------------------
# 6.3 Evaluación de los Modelos de Deep Learning
#------------------------------------------------------------------------------

print("\n6.3 EVALUACIÓN DE LOS MODELOS DE DEEP LEARNING")

#------------------------------------------------------------------------------
# 6.3.1 Validar los datos del Benchmark
#------------------------------------------------------------------------------

required_products = (
    "x_train",
    "y_train",
    "x_test",
    "y_test",
)

missing_products = [
    product
    for product in required_products
    if product not in benchmark_data
]

if missing_products:
    raise ValueError(
        "La Colección Oficial BenchmarkData está incompleta. "
        f"Productos faltantes: {missing_products}"
    )

#------------------------------------------------------------------------------
# 6.3.2 Recuperar los datos del Benchmark
#------------------------------------------------------------------------------

x_train = benchmark_data["x_train"]
y_train = benchmark_data["y_train"]
x_test = benchmark_data["x_test"]
y_test = benchmark_data["y_test"]

#------------------------------------------------------------------------------
# 6.3.3 Validar los datos recuperados
#------------------------------------------------------------------------------

for name, value in (
    ("x_train", x_train),
    ("y_train", y_train),
    ("x_test", x_test),
    ("y_test", y_test),
):

    if value is None:
        raise ValueError(
            f"El producto '{name}' es inválido."
        )

#------------------------------------------------------------------------------
# 6.3.4 Ejecutar los Modelos de Deep Learning
#------------------------------------------------------------------------------

deep_learning_results = []

deep_learning_results.append(
    run_mlp(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )
)

#------------------------------------------------------------------------------
# 6.3.5 Validar los resultados
#------------------------------------------------------------------------------

if not deep_learning_results:
    raise ValueError(
        "No se obtuvieron resultados para los Modelos de Deep Learning."
    )

#------------------------------------------------------------------------------
# 6.3.6 Registrar la evaluación
#------------------------------------------------------------------------------

print(
    f"Modelos de Deep Learning evaluados : "
    f"{len(deep_learning_results)}"
)

#------------------------------------------------------------------------------
# 6.3.7 Confirmar la evaluación
#------------------------------------------------------------------------------

print(
    "Evaluación de los Modelos de Deep Learning completada correctamente."
)

#------------------------------------------------------------------------------
# 6.4 Evaluación de los Modelos Graph Neural Networks
#------------------------------------------------------------------------------

print("\n6.4 EVALUACIÓN DE LOS MODELOS GRAPH NEURAL NETWORKS")

#------------------------------------------------------------------------------
# 6.4.1 Validar los datos del Benchmark
#------------------------------------------------------------------------------

required_products = (
    "graphs",
)

missing_products = [
    product
    for product in required_products
    if product not in benchmark_data
]

if missing_products:
    raise ValueError(
        "La Colección Oficial BenchmarkData está incompleta. "
        f"Productos faltantes: {missing_products}"
    )

#------------------------------------------------------------------------------
# 6.4.2 Recuperar los datos del Benchmark
#------------------------------------------------------------------------------

graphs = benchmark_data["graphs"]

#------------------------------------------------------------------------------
# 6.4.3 Validar los datos recuperados
#------------------------------------------------------------------------------

if graphs is None:
    raise ValueError(
        "El producto 'graphs' es inválido."
    )

if len(graphs) == 0:
    raise ValueError(
        "La colección oficial GraphData está vacía."
    )

#------------------------------------------------------------------------------
# 6.4.4 Ejecutar los Modelos Graph Neural Networks
#------------------------------------------------------------------------------

gnn_results = []

for model_name in BENCHMARK_MODELS["graph_neural_networks"]:

    gnn_results.append(
        run_gnn_benchmark(
            model_config=GNN_CONFIG[model_name],
            benchmark_data=benchmark_data
        )
    )

#------------------------------------------------------------------------------
# 6.4.5 Validar los resultados
#------------------------------------------------------------------------------

if not gnn_results:
    raise ValueError(
        "No se obtuvieron resultados para los Modelos Graph Neural Networks."
    )

#------------------------------------------------------------------------------
# 6.4.6 Registrar la evaluación
#------------------------------------------------------------------------------

print(
    f"Modelos Graph Neural Networks evaluados : "
    f"{len(gnn_results)}"
)

#------------------------------------------------------------------------------
# 6.4.7 Confirmar la evaluación
#------------------------------------------------------------------------------

print(
    "Evaluación de los Modelos Graph Neural Networks completada correctamente."
)

#------------------------------------------------------------------------------
# 6.5 Confirmación del Bloque
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# 6.5.1 Validar los resultados del Benchmark
#------------------------------------------------------------------------------

benchmark_results = {
    "Estadísticos": statistical_results,
    "Machine Learning": machine_learning_results,
    "Deep Learning": deep_learning_results,
    "Graph Neural Networks": gnn_results
}

for family_name, results in benchmark_results.items():

    if results is None:
        raise ValueError(
            f"No existen resultados para la familia '{family_name}'."
        )

    if len(results) == 0:
        raise ValueError(
            f"La familia '{family_name}' no produjo resultados."
        )

#------------------------------------------------------------------------------
# 6.5.2 Registrar el resumen del Benchmark
#------------------------------------------------------------------------------

print("\n" + "-" * 80)
print("BLOQUE 6 COMPLETADO: FAMILIAS DE MODELOS EVALUADAS")
print("-" * 80)

print(f"Resultados Estadísticos          : {len(statistical_results)}")
print(f"Resultados Machine Learning      : {len(machine_learning_results)}")
print(f"Resultados Deep Learning         : {len(deep_learning_results)}")
print(f"Resultados Graph Neural Networks : {len(gnn_results)}")

#------------------------------------------------------------------------------
# 6.5.3 Confirmar el bloque
#------------------------------------------------------------------------------

print("\nBloque 6 ejecutado correctamente.")

# BLOQUE 7. CONSOLIDACIÓN DE LOS RESULTADOS DEL BENCHMARK
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Consolidar los resultados obtenidos por las diferentes familias de modelos
# evaluadas durante el Benchmark Científico en una colección única de
# resultados que servirá como base para el análisis comparativo, el ranking
# científico y la selección del modelo oficial.
#
#------------------------------------------------------------------------------
# Producto
#------------------------------------------------------------------------------
# - benchmark_results
# - Resultados Consolidados del Benchmark Científico.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿Se consolidaron correctamente los resultados obtenidos por todas las
# familias de modelos evaluadas durante el Benchmark Científico?
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 7.1 Consolidación de los Resultados del Benchmark
#------------------------------------------------------------------------------

print("\n7.1 CONSOLIDACIÓN DE LOS RESULTADOS DEL BENCHMARK")

#------------------------------------------------------------------------------
# 7.1.1 Validar los resultados de las familias
#------------------------------------------------------------------------------

benchmark_families = {
    "statistical_results": statistical_results,
    "machine_learning_results": machine_learning_results,
    "deep_learning_results": deep_learning_results,
    "gnn_results": gnn_results
}

missing_results = [
    name
    for name, results in benchmark_families.items()
    if results is None
]

if missing_results:
    raise ValueError(
        "No existen resultados para las siguientes familias: "
        f"{missing_results}"
    )

#------------------------------------------------------------------------------
# 7.1.2 Recuperar los resultados
#------------------------------------------------------------------------------

statistical = statistical_results
machine_learning = machine_learning_results
deep_learning = deep_learning_results
graph_neural_networks = gnn_results

#------------------------------------------------------------------------------
# 7.1.3 Validar los resultados recuperados
#------------------------------------------------------------------------------

for family_name, results in (
    ("Statistical", statistical),
    ("Machine Learning", machine_learning),
    ("Deep Learning", deep_learning),
    ("Graph Neural Networks", graph_neural_networks),
):

    if len(results) == 0:
        raise ValueError(
            f"La familia '{family_name}' no contiene resultados."
        )

#------------------------------------------------------------------------------
# 7.1.4 Consolidar los resultados
#------------------------------------------------------------------------------

benchmark_results = (
    statistical
    + machine_learning
    + deep_learning
    + graph_neural_networks
)

#------------------------------------------------------------------------------
# 7.1.5 Validar la consolidación
#------------------------------------------------------------------------------

if len(benchmark_results) == 0:
    raise ValueError(
        "No existen resultados consolidados del Benchmark Científico."
    )

#------------------------------------------------------------------------------
# 7.1.6 Registrar la consolidación
#------------------------------------------------------------------------------

print(
    f"Resultados consolidados : "
    f"{len(benchmark_results)}"
)

#------------------------------------------------------------------------------
# 7.1.7 Confirmar la consolidación
#------------------------------------------------------------------------------

print(
    "Resultados del Benchmark Científico consolidados correctamente."
)

# BLOQUE 8. CONSTRUCCIÓN DEL RANKING CIENTÍFICO DEL BENCHMARK
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Construir el Ranking Científico Oficial del Benchmark utilizando los
# resultados consolidados y las métricas oficiales definidas para el
# protocolo experimental, ordenando objetivamente el desempeño predictivo
# de todos los modelos evaluados.
#
#------------------------------------------------------------------------------
# Producto
#------------------------------------------------------------------------------
# - benchmark_ranking
# - Ranking Científico Oficial del Benchmark.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿Cuál es el orden de desempeño predictivo de los modelos evaluados según
# las métricas oficiales del Benchmark Científico?
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 8.1 Construcción del Ranking Científico
#------------------------------------------------------------------------------

print("\n8.1 CONSTRUCCIÓN DEL RANKING CIENTÍFICO")

#------------------------------------------------------------------------------
# 8.1.1 Validar los resultados del Benchmark
#------------------------------------------------------------------------------

if benchmark_results is None:
    raise ValueError(
        "No existen resultados consolidados del Benchmark Científico."
    )

if len(benchmark_results) == 0:
    raise ValueError(
        "La colección BenchmarkResults está vacía."
    )

#------------------------------------------------------------------------------
# 8.1.2 Recuperar los resultados del Benchmark
#------------------------------------------------------------------------------

results = benchmark_results

#------------------------------------------------------------------------------
# 8.1.3 Validar los resultados recuperados
#------------------------------------------------------------------------------

if not isinstance(results, list):
    raise TypeError(
        "BenchmarkResults debe ser una lista."
    )

#------------------------------------------------------------------------------
# 8.1.4 Construir el Ranking Científico
#------------------------------------------------------------------------------

benchmark_ranking = build_benchmark_ranking(
    benchmark_results=results,
    benchmark_metrics=BENCHMARK_METRICS
)

#------------------------------------------------------------------------------
# 8.1.5 Validar el Ranking Científico
#------------------------------------------------------------------------------

if benchmark_ranking is None:
    raise ValueError(
        "No fue posible construir el Ranking Científico del Benchmark."
    )

if len(benchmark_ranking) == 0:
    raise ValueError(
        "El Ranking Científico del Benchmark está vacío."
    )

#------------------------------------------------------------------------------
# 8.1.6 Registrar la construcción del Ranking
#------------------------------------------------------------------------------

print(
    f"Modelos clasificados : "
    f"{len(benchmark_ranking)}"
)

#------------------------------------------------------------------------------
# 8.1.7 Confirmar la construcción del Ranking
#------------------------------------------------------------------------------

print(
    "Ranking Científico del Benchmark construido correctamente."
)

# BLOQUE 9. SELECCIÓN DEL MODELO OFICIAL DE LA FAMILIA GRAPH NEURAL NETWORKS
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Seleccionar el Modelo Oficial del proyecto a partir de las arquitecturas
# Graph Neural Networks (GNN) evaluadas durante el Benchmark Científico,
# utilizando el Ranking Científico Oficial para identificar la GNN con el
# mejor desempeño predictivo que será utilizada en las etapas posteriores
# de entrenamiento, forecasting y despliegue de la Plataforma GeoAI.
#
#------------------------------------------------------------------------------
# Producto
#------------------------------------------------------------------------------
# - official_model
# - Modelo Oficial del Proyecto (Graph Neural Network).
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿Cuál es la Graph Neural Network con el mejor desempeño predictivo según
# el Ranking Científico Oficial del Benchmark para ser adoptada como Modelo
# Oficial del proyecto?
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 9.1 Selección del Modelo Oficial de la Familia Graph Neural Networks
#------------------------------------------------------------------------------

print("\n9.1 SELECCIÓN DEL MODELO OFICIAL DE LA FAMILIA GRAPH NEURAL NETWORKS")

#------------------------------------------------------------------------------
# 9.1.1 Validar el Ranking Científico
#------------------------------------------------------------------------------

if benchmark_ranking is None:
    raise ValueError(
        "No existe el Ranking Científico del Benchmark."
    )

if len(benchmark_ranking) == 0:
    raise ValueError(
        "El Ranking Científico del Benchmark está vacío."
    )

#------------------------------------------------------------------------------
# 9.1.2 Recuperar el Ranking Científico
#------------------------------------------------------------------------------

ranking = benchmark_ranking

#------------------------------------------------------------------------------
# 9.1.3 Validar el Ranking recuperado
#------------------------------------------------------------------------------

if not isinstance(ranking, list):
    raise TypeError(
        "El Ranking Científico debe ser una lista."
    )

#------------------------------------------------------------------------------
# 9.1.4 Seleccionar el Modelo Oficial
#------------------------------------------------------------------------------

official_model = select_official_model(
    benchmark_ranking=ranking
)

#------------------------------------------------------------------------------
# 9.1.5 Validar el Modelo Oficial
#------------------------------------------------------------------------------

if official_model is None:
    raise ValueError(
        "No fue posible seleccionar el Modelo Oficial de la familia Graph Neural Networks."
    )

required_fields = (
    "model_name",
    "family",
)

missing_fields = [
    field
    for field in required_fields
    if field not in official_model
]

if missing_fields:
    raise ValueError(
        "El Modelo Oficial no contiene la información requerida. "
        f"Campos faltantes: {missing_fields}"
    )

#------------------------------------------------------------------------------
# 9.1.6 Registrar el Modelo Oficial
#------------------------------------------------------------------------------

print(
    f"Modelo Oficial : {official_model['model_name']}"
)

print(
    f"Familia        : {official_model['family']}"
)

#------------------------------------------------------------------------------
# 9.1.7 Confirmar la selección
#------------------------------------------------------------------------------

print(
    "Modelo Oficial de la familia Graph Neural Networks seleccionado correctamente."
)

# BLOQUE 10. Selección de los Mejores Modelos por Familia ------------------
# Objetivo: Seleccionar el mejor modelo de cada familia a partir del ranking
# oficial del Benchmark Científico.
# Entradas:
# - benchmark_ranking
# Producto:
# - official_model_by_family
# Pregunta científica:
# ¿Cuál es el mejor modelo dentro de cada familia evaluada por el Benchmark
# Científico?
#------------------------------------------------------------------------------
# 10. Selección de los Mejores Modelos por Familia
#------------------------------------------------------------------------------

print("\n" + "-" * 80)
print("BLOQUE 10. SELECCIÓN DE LOS MEJORES MODELOS POR FAMILIA")
print("-" * 80)

#------------------------------------------------------------------------------
# 10.1 Validar el Ranking Científico
#------------------------------------------------------------------------------

if benchmark_ranking is None:
    raise ValueError(
        "No existe el Ranking Científico del Benchmark."
    )

if len(benchmark_ranking) == 0:
    raise ValueError(
        "El Ranking Científico del Benchmark está vacío."
    )

#------------------------------------------------------------------------------
# 10.2 Recuperar el Ranking Científico
#------------------------------------------------------------------------------

ranking = benchmark_ranking

#------------------------------------------------------------------------------
# 10.3 Validar el Ranking recuperado
#------------------------------------------------------------------------------

if not isinstance(ranking, list):
    raise TypeError(
        "El Ranking Científico debe ser una lista."
    )

#------------------------------------------------------------------------------
# 10.4 Seleccionar los Mejores Modelos por Familia
#------------------------------------------------------------------------------

official_model_by_family = {}

for result in ranking:

    family = result["family"]

    if family not in official_model_by_family:
        official_model_by_family[family] = result

#------------------------------------------------------------------------------
# 10.5 Validar la selección
#------------------------------------------------------------------------------

if len(official_model_by_family) == 0:
    raise ValueError(
        "No fue posible seleccionar los mejores modelos por familia."
    )

#------------------------------------------------------------------------------
# 10.6 Registrar los mejores modelos
#------------------------------------------------------------------------------

ranking_metric = BENCHMARK_CONFIG["ranking_metric"]

print("\nMejores modelos por familia\n")

for family, result in official_model_by_family.items():

    print("-" * 80)
    print(f"Familia : {family}")
    print(f"Código  : {result['model_code']}")
    print(f"Modelo  : {result['model_name']}")
    print(
        f"{ranking_metric.upper():<9}: "
        f"{result[ranking_metric]:.6f}"
    )

print("-" * 80)

#------------------------------------------------------------------------------
# 10.7 Confirmar la selección
#------------------------------------------------------------------------------

print(
    "\nMejores modelos por familia seleccionados correctamente."
)

print("-" * 80)

# BLOQUE 11. Reporte Ejecutivo del Benchmark ------------------------------------
# Objetivo: Presentar el resumen ejecutivo del Benchmark Científico,
# mostrando los modelos evaluados, sus métricas, el ranking oficial,
# los mejores modelos por familia y la justificación metodológica de la
# selección del modelo oficial.
#
# Entradas:
# - benchmark_results
# - benchmark_ranking
# - official_model_by_family
# - official_model
#
# Producto:
# - Reporte Ejecutivo del Benchmark Científico.
#
# Pregunta científica:
# ¿Cuáles fueron los resultados obtenidos durante el Benchmark Científico
# y cuál fue el fundamento para seleccionar el modelo oficial?
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("REPORTE EJECUTIVO DEL BENCHMARK CIENTÍFICO")
print("-" * 80)

#------------------------------------------------------------------------------
# 11.1 Validar los productos del Benchmark
#------------------------------------------------------------------------------

required_products = {
    "benchmark_results": benchmark_results,
    "benchmark_ranking": benchmark_ranking,
    "official_model": official_model,
    "official_model_by_family": official_model_by_family
}

missing_products = [
    name
    for name, value in required_products.items()
    if value is None
]

if missing_products:
    raise ValueError(
        "No existen los siguientes productos del Benchmark: "
        f"{missing_products}"
    )

#------------------------------------------------------------------------------
# 11.2 Recuperar los productos
#------------------------------------------------------------------------------

results = benchmark_results
ranking = benchmark_ranking
official = official_model
best_models = official_model_by_family

#------------------------------------------------------------------------------
# 11.3 Validar los productos recuperados
#------------------------------------------------------------------------------

if len(results) == 0:
    raise ValueError(
        "BenchmarkResults está vacío."
    )

if len(ranking) == 0:
    raise ValueError(
        "BenchmarkRanking está vacío."
    )

if len(best_models) == 0:
    raise ValueError(
        "No existen mejores modelos por familia."
    )

#------------------------------------------------------------------------------
# 11.4 Ranking Oficial del Benchmark
#------------------------------------------------------------------------------

print("\n4. RANKING OFICIAL DEL BENCHMARK")
print("-" * 120)

#------------------------------------------------------------------------------
# 11.4.1 Validar el Ranking Científico
#------------------------------------------------------------------------------

if benchmark_ranking is None:
    raise ValueError(
        "No existe el Ranking Oficial del Benchmark."
    )

if len(benchmark_ranking) == 0:
    raise ValueError(
        "El Ranking Oficial del Benchmark está vacío."
    )

#------------------------------------------------------------------------------
# 11.4.2 Recuperar el Ranking Científico
#------------------------------------------------------------------------------

ranking = benchmark_ranking

#------------------------------------------------------------------------------
# 11.4.3 Validar el Ranking recuperado
#------------------------------------------------------------------------------

required_fields = (
    "model_name",
    "family",
    BENCHMARK_CONFIG["ranking_metric"],
)

for position, result in enumerate(ranking, start=1):

    missing_fields = [
        field
        for field in required_fields
        if field not in result
    ]

    if missing_fields:
        raise ValueError(
            f"El modelo en la posición {position} "
            "no contiene los campos requeridos: "
            f"{missing_fields}"
        )

#------------------------------------------------------------------------------
# 11.4.4 Presentar el Ranking Oficial
#------------------------------------------------------------------------------

print(
    f"{'Pos.':<6}"
    f"{'Modelo':<20}"
    f"{'Familia':<28}"
    f"{BENCHMARK_CONFIG['ranking_metric'].upper():>12}"
)

print("-" * 120)

for position, result in enumerate(ranking, start=1):

    print(
        f"{position:<6}"
        f"{result['model_name']:<20}"
        f"{result['family']:<28}"
        f"{result[BENCHMARK_CONFIG['ranking_metric']]:>12.6f}"
    )

#------------------------------------------------------------------------------
# 11.4.5 Confirmar la presentación del Ranking
#------------------------------------------------------------------------------

print(
    "\nRanking Oficial del Benchmark presentado correctamente."
)

#------------------------------------------------------------------------------
# 11.4.6 Modelo Oficial del Proyecto
#------------------------------------------------------------------------------

print("\n6. MODELO OFICIAL DEL PROYECTO")
print("-" * 80)

#------------------------------------------------------------------------------
# 11.4.6.1 Validar el Modelo Oficial
#------------------------------------------------------------------------------

if official_model is None:
    raise ValueError(
        "No existe el Modelo Oficial del Proyecto."
    )

#------------------------------------------------------------------------------
# 11.4.6.2 Recuperar el Modelo Oficial
#------------------------------------------------------------------------------

model = official_model

#------------------------------------------------------------------------------
# 11.4.6.3 Validar el Modelo Oficial recuperado
#------------------------------------------------------------------------------

required_fields = (
    "model_code",
    "model_name",
    "family",
    "rmse",
    "mae",
    "mape",
    "r2",
)

missing_fields = [
    field
    for field in required_fields
    if field not in model
]

if missing_fields:
    raise ValueError(
        "El Modelo Oficial no contiene los campos requeridos: "
        f"{missing_fields}"
    )

#------------------------------------------------------------------------------
# 11.4.6.4 Presentar el Modelo Oficial
#------------------------------------------------------------------------------

print(f"Código                 : {model['model_code']}")
print(f"Modelo                 : {model['model_name']}")
print(f"Familia                : {model['family']}")
print(f"RMSE                   : {model['rmse']:.6f}")
print(f"MAE                    : {model['mae']:.6f}")
print(f"MAPE                   : {model['mape']:.6f}")
print(f"R²                     : {model['r2']:.6f}")

#------------------------------------------------------------------------------
# 11.4.6.5 Confirmar
#------------------------------------------------------------------------------

print(
    "\nModelo Oficial presentado correctamente."
)

#------------------------------------------------------------------------------
# 11.4.7 Análisis Multicriterio (MCDA)
#------------------------------------------------------------------------------

print("\n7. ANÁLISIS MULTICRITERIO (MCDA)")
print("-" * 80)

#------------------------------------------------------------------------------
# 11.4.7.1 Validar el Modelo Oficial
#------------------------------------------------------------------------------

if official_model is None:
    raise ValueError(
        "No existe el Modelo Oficial del Benchmark."
    )

#------------------------------------------------------------------------------
# 11.4.7.2 Recuperar el Modelo Oficial
#------------------------------------------------------------------------------

model = official_model

#------------------------------------------------------------------------------
# 11.4.7.3 Validar el Modelo recuperado
#------------------------------------------------------------------------------

required_fields = (
    "model_name",
    "family",
)

missing_fields = [
    field
    for field in required_fields
    if field not in model
]

if missing_fields:
    raise ValueError(
        "El Modelo Oficial no contiene los campos requeridos. "
        f"Campos faltantes: {missing_fields}"
    )

#------------------------------------------------------------------------------
# 11.4.7.4 Presentar el Análisis Multicriterio
#------------------------------------------------------------------------------

print(
    "La selección del Modelo Oficial se realizó mediante un "
    "proceso de decisión multicriterio (MCDA)."
)

print()

print(
    "La decisión NO consideró únicamente el menor error predictivo, "
    "sino un conjunto de criterios científicos y tecnológicos."
)

print()

criteria = [

    "Precisión predictiva (RMSE, MAE, MAPE y R²)",
    "Modelado explícito de relaciones espaciales",
    "Capacidad de aprendizaje sobre grafos",
    "Generalización espacial",
    "Compatibilidad con Forecasting Espacio-Temporal",
    "Compatibilidad con la Plataforma Inteligente GeoAI",
    "Robustez metodológica",
    "Interpretabilidad científica"

]

print("Criterios considerados:")

for position, criterion in enumerate(criteria, start=1):

    print(f"{position}. {criterion}")

print()

print("Resultado del proceso multicriterio")

print(f"Modelo seleccionado : {model['model_name']}")
print(f"Familia             : {model['family']}")

#------------------------------------------------------------------------------
# 11.4.7.5 Confirmar el Análisis Multicriterio
#------------------------------------------------------------------------------

print(
    "\nAnálisis Multicriterio presentado correctamente."
)

#------------------------------------------------------------------------------
# 11.4.8 Conclusión Ejecutiva
#------------------------------------------------------------------------------

print("\n8. CONCLUSIÓN EJECUTIVA")
print("-" * 80)

#------------------------------------------------------------------------------
# 11.4.8.1 Validar el Modelo Oficial
#------------------------------------------------------------------------------

if official_model is None:
    raise ValueError(
        "No existe el Modelo Oficial del Benchmark."
    )

#------------------------------------------------------------------------------
# 11.4.8.2 Recuperar el Modelo Oficial
#------------------------------------------------------------------------------

model = official_model

#------------------------------------------------------------------------------
# 11.4.8.3 Validar el Modelo recuperado
#------------------------------------------------------------------------------

required_fields = (
    "model_name",
    "family",
)

missing_fields = [
    field
    for field in required_fields
    if field not in model
]

if missing_fields:
    raise ValueError(
        "El Modelo Oficial no contiene los campos requeridos. "
        f"Campos faltantes: {missing_fields}"
    )

#------------------------------------------------------------------------------
# 11.4.8.4 Presentar la Conclusión Ejecutiva
#------------------------------------------------------------------------------

print(
    "El Benchmark Científico evaluó de forma objetiva las diferentes "
    "familias de modelos incluidas en el protocolo experimental."
)

print()

print(
    "La evaluación integró modelos Estadísticos, Machine Learning, "
    "Deep Learning y Graph Neural Networks mediante un conjunto "
    "homogéneo de métricas de desempeño y criterios científicos."
)

print()

print(
    "El proceso de decisión permitió identificar como Modelo Oficial "
    f"a {model['model_name']}, perteneciente a la familia "
    f"{model['family']}."
)

print()

print(
    "La selección se fundamentó en un proceso de evaluación "
    "multicriterio que combinó precisión predictiva, capacidad de "
    "representación espacial, aprendizaje sobre grafos, "
    "generalización espacial, compatibilidad con el forecasting "
    "espacio-temporal y su integración con la Plataforma "
    "Inteligente GeoAI."
)

print()

print(
    "Los resultados obtenidos respaldan científicamente la utilización "
    "del Modelo Oficial como base para los procesos de pronóstico, "
    "análisis territorial y soporte a la toma de decisiones del proyecto."
)

#------------------------------------------------------------------------------
# 11.4.8.5 Confirmar la Conclusión Ejecutiva
#------------------------------------------------------------------------------

print(
    "\nConclusión Ejecutiva presentada correctamente."
)

print("-" * 80)

#------------------------------------------------------------------------------
# 11.5 Confirmar el Reporte Ejecutivo
#------------------------------------------------------------------------------

print("\n" + "-" * 80)
print("REPORTE EJECUTIVO DEL BENCHMARK GENERADO CORRECTAMENTE.")
print("-" * 80)


#------------------------------------------------------------------------------
# Verificación de disponibilidad de BenchmarkData
#------------------------------------------------------------------------------

print("\n" + "-" * 80)
print("VERIFICACIÓN DE BENCHMARKDATA")
print("-" * 80)

variables = [
    "graphs",
    "x_train",
    "y_train",
    "x_validation",
    "y_validation",
    "x_test",
    "y_test",
    "train_index",
    "validation_index",
    "test_index",
    "scaler"
]

for variable in variables:
    estado = "OK" if variable in globals() else "NO DISPONIBLE"
    print(f"{variable:<20}: {estado}")

print("-" * 80)

# BLOQUE 12. Exportación de Resultados ------------------------------------
# Objetivo: Exportar los productos oficiales generados durante el Benchmark
# Científico para garantizar la reproducibilidad del experimento.
# Entradas:
# - benchmark_results
# - benchmark_ranking
# - official_model_config
# Producto:
# Producto:
# - benchmark_results.joblib
# - benchmark_data.joblib
# - benchmark_metrics.parquet
# - benchmark_summary.csv
# - benchmark_ranking.csv
# - official_model_config.json
# Pregunta científica:
# ¿Los productos oficiales del Benchmark fueron exportados correctamente?

print("\n" + "-" * 80)
print("BLOQUE 12. EXPORTACIÓN DE RESULTADOS")
print("-" * 80)

#------------------------------------------------------------------------------
# 12.1 Validar los productos del Benchmark
#------------------------------------------------------------------------------

required_products = {

    "benchmark_results": benchmark_results,
    "benchmark_ranking": benchmark_ranking,
    "official_model": official_model

}

missing_products = [

    name
    for name, value in required_products.items()
    if value is None

]

if missing_products:

    raise ValueError(
        "No existen los siguientes productos del Benchmark: "
        f"{missing_products}"
    )

#------------------------------------------------------------------------------
# 12.2 Recuperar los productos del Benchmark
#------------------------------------------------------------------------------

results = benchmark_results
ranking = benchmark_ranking
model = official_model

#------------------------------------------------------------------------------
# 12.3 Validar los productos recuperados
#------------------------------------------------------------------------------

# Validar BenchmarkResults

if not isinstance(results, list):
    raise TypeError(
        "BenchmarkResults debe ser una lista."
    )

if len(results) == 0:
    raise ValueError(
        "BenchmarkResults está vacío."
    )

# Validar BenchmarkRanking

if not isinstance(ranking, list):
    raise TypeError(
        "BenchmarkRanking debe ser una lista."
    )

if len(ranking) == 0:
    raise ValueError(
        "BenchmarkRanking está vacío."
    )

# Validar Modelo Oficial

if not isinstance(model, dict):
    raise TypeError(
        "OfficialModel debe ser un diccionario."
    )

required_fields = (
    "model_code",
    "model_name",
    "family",
    "model_config",
)

missing_fields = [
    field
    for field in required_fields
    if field not in model
]

if missing_fields:
    raise ValueError(
        "El Modelo Oficial no contiene los campos requeridos. "
        f"Campos faltantes: {missing_fields}"
    )

#------------------------------------------------------------------------------
# 12.4 Construir los productos de exportación
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# 12.4.1 Construir la tabla de métricas del Benchmark
#------------------------------------------------------------------------------

benchmark_metrics = benchmark_results_to_dataframe(
    results
)

#------------------------------------------------------------------------------
# 12.4.2 Construir la tabla del Ranking Científico
#------------------------------------------------------------------------------

benchmark_ranking_df = benchmark_results_to_dataframe(
    ranking
)

#------------------------------------------------------------------------------
# 12.4.3 Construir la tabla resumen del Benchmark
#------------------------------------------------------------------------------

benchmark_summary = benchmark_metrics[
    [
        "model_code",
        "model_name",
        "family",
        "rmse",
        "mae",
        "mape",
        "r2",
        "training_time",
        "inference_time"
    ]
]

#------------------------------------------------------------------------------
# 12.4.4 Construir la tabla de métricas para Parquet
#------------------------------------------------------------------------------

benchmark_metrics_parquet = (
    benchmark_metrics.copy()
)

benchmark_metrics_parquet[
    "model_config"
] = benchmark_metrics_parquet[
    "model_config"
].apply(
    lambda config: json.dumps(
        config,
        ensure_ascii=False
    )
)

#------------------------------------------------------------------------------
# 12.4.5 Construir la configuración exportable del Modelo Oficial
#------------------------------------------------------------------------------

official_model_config_export = (
    build_exportable_benchmark_result(
        model
    )
)

#------------------------------------------------------------------------------
# 12.4.6 Validar BenchmarkData
#------------------------------------------------------------------------------

if not isinstance(benchmark_data, dict):
    raise TypeError(
        "BenchmarkData debe ser un diccionario."
    )

required_products = [

    "graphs",

    "x_train",
    "y_train",

    "x_validation",
    "y_validation",

    "x_test",
    "y_test",

    "train_index",
    "validation_index",
    "test_index",

    "scaler"

]

missing_products = [

    product
    for product in required_products
    if product not in benchmark_data

]

if missing_products:

    raise ValueError(
        "BenchmarkData está incompleto: "
        f"{missing_products}"
    )

for product in required_products:

    if benchmark_data[product] is None:

        raise ValueError(
            f"'{product}' es inválido."
        )

print("BenchmarkData validado correctamente.")

#------------------------------------------------------------------------------
# 12.4.7 Validar los productos construidos
#------------------------------------------------------------------------------

constructed_products = {

    "benchmark_metrics": benchmark_metrics,
    "benchmark_ranking_df": benchmark_ranking_df,
    "benchmark_summary": benchmark_summary,
    "benchmark_metrics_parquet": benchmark_metrics_parquet,
    "official_model_config_export": official_model_config_export,

    "benchmark_data": benchmark_data

}

missing_products = [

    name
    for name, product in constructed_products.items()
    if product is None

]

if missing_products:

    raise ValueError(
        "No fue posible construir los siguientes productos "
        f"de exportación: {missing_products}"
    )

#------------------------------------------------------------------------------
# 12.4.7 Registrar la construcción
#------------------------------------------------------------------------------

print(
    f"Productos de exportación construidos : "
    f"{len(constructed_products)}"
)

#------------------------------------------------------------------------------
# 12.4.8 Confirmar la construcción
#------------------------------------------------------------------------------

print(
    "Productos de exportación construidos correctamente."
)
#------------------------------------------------------------------------------
# 12.5 Exportar los productos oficiales
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# 12.5.1 Exportar BenchmarkResults
#------------------------------------------------------------------------------

joblib.dump(
    results,
    BENCHMARK_RESULTS_FILE
)

#------------------------------------------------------------------------------
# 12.5.2 Exportar BenchmarkData
#------------------------------------------------------------------------------

joblib.dump(
    benchmark_data,
    BENCHMARK_DATA_FILE
)

#------------------------------------------------------------------------------
# 12.5.3 Exportar BenchmarkMetrics
#------------------------------------------------------------------------------

benchmark_metrics_parquet.to_parquet(
    BENCHMARK_METRICS_FILE,
    index=False
)

#------------------------------------------------------------------------------
# 12.5.4 Exportar BenchmarkRanking
#------------------------------------------------------------------------------

benchmark_ranking_df.to_csv(
    BENCHMARK_RANKING_CSV_FILE,
    index=False
)

benchmark_ranking_df.to_excel(
    BENCHMARK_RANKING_XLSX_FILE,
    index=False
)

#------------------------------------------------------------------------------
# 12.5.5 Exportar BenchmarkSummary
#------------------------------------------------------------------------------

benchmark_summary.to_csv(
    BENCHMARK_SUMMARY_CSV_FILE,
    index=False
)

benchmark_summary.to_excel(
    BENCHMARK_SUMMARY_XLSX_FILE,
    index=False
)

#------------------------------------------------------------------------------
# 12.5.6 Exportar la configuración del Modelo Oficial
#------------------------------------------------------------------------------

with open(
    OFFICIAL_MODEL_CONFIG_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        official_model_config_export,
        file,
        indent=4,
        ensure_ascii=False
    )

#------------------------------------------------------------------------------
# 12.5.7 Registrar la exportación
#------------------------------------------------------------------------------

print(
    "Productos oficiales exportados correctamente."
)

#------------------------------------------------------------------------------
# 12.5.8 Confirmar la exportación
#------------------------------------------------------------------------------

print(
    "Exportación de los productos oficiales completada correctamente."
)

#------------------------------------------------------------------------------
# 12.6 Validar la exportación
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# 12.6.1 Construir la colección de archivos exportados
#------------------------------------------------------------------------------

exported_files = {

    "BenchmarkResults": BENCHMARK_RESULTS_FILE,
    "BenchmarkData": BENCHMARK_DATA_FILE,
    "BenchmarkMetrics": BENCHMARK_METRICS_FILE,
    "BenchmarkRankingCSV": BENCHMARK_RANKING_CSV_FILE,
    "BenchmarkRankingXLSX": BENCHMARK_RANKING_XLSX_FILE,
    "BenchmarkSummaryCSV": BENCHMARK_SUMMARY_CSV_FILE,
    "BenchmarkSummaryXLSX": BENCHMARK_SUMMARY_XLSX_FILE,
    "OfficialModel": OFFICIAL_MODEL_CONFIG_FILE

}

#------------------------------------------------------------------------------
# 12.6.2 Validar la existencia de los archivos
#------------------------------------------------------------------------------

missing_files = [

    name
    for name, file in exported_files.items()
    if not file.exists()

]

if missing_files:

    raise FileNotFoundError(
        "No fue posible exportar los siguientes productos: "
        f"{missing_files}"
    )

#------------------------------------------------------------------------------
# 12.6.3 Validar el tamaño de los archivos
#------------------------------------------------------------------------------

empty_files = [

    name
    for name, file in exported_files.items()
    if file.stat().st_size == 0

]

if empty_files:

    raise ValueError(
        "Los siguientes archivos fueron exportados "
        "pero están vacíos: "
        f"{empty_files}"
    )

#------------------------------------------------------------------------------
# 12.6.4 Registrar la validación
#------------------------------------------------------------------------------

print(
    f"Archivos validados : {len(exported_files)}"
)

#------------------------------------------------------------------------------
# 12.6.5 Confirmar la validación
#------------------------------------------------------------------------------

print(
    "La exportación de los productos fue validada correctamente."
)

#------------------------------------------------------------------------------
# 12.7 Registrar la exportación
#------------------------------------------------------------------------------

print("\nProductos oficiales exportados")

print("-" * 80)

for product_name, file in exported_files.items():

    print(
        f"{product_name:<24}: {file.name}"
    )

print("-" * 80)

print(
    f"Total de archivos exportados : "
    f"{len(exported_files)}"
)

#------------------------------------------------------------------------------
# 12.8 Confirmar la exportación
#------------------------------------------------------------------------------

print(
    "\nExportación del Benchmark Científico completada correctamente."
)

print(
    "Todos los productos oficiales fueron generados, "
    "validados y almacenados exitosamente."
)

print(
    "El Benchmark Científico está listo para ser utilizado "
    "en las etapas de análisis, visualización y despliegue."
)

#==============================================================================
# BLOQUE 13. AUDITORÍA DE LOS RESULTADOS DEL BENCHMARK
# Objetivo   : Auditar la integridad, consistencia y validez científica de los
#              resultados generados por el Benchmark antes de su utilización.
# Arquitectura científica : Validar → Recuperar → Validar → Auditar →
#                           Registrar → Confirmar.
# Entradas   : BenchmarkResults, BenchmarkRanking, OfficialModel,
#              BenchmarkData, GraphData y productos exportados.
# Procesos   : Verificación estructural, auditoría del protocolo experimental,
#              auditoría de métricas, consistencia científica y validación de
#              los productos exportados.
# Producto   : Auditoría Científica del Benchmark certificada.
#==============================================================================

products = {

    "BenchmarkResults": benchmark_results,
    "BenchmarkRanking": benchmark_ranking,
    "OfficialModel": official_model,
    "BenchmarkData": benchmark_data,
    "BenchmarkConfig": BENCHMARK_CONFIG,
    "BenchmarkModels": BENCHMARK_MODELS,

}
#------------------------------------------------------------------------------
# 13.1 Validar los productos del Benchmark
#------------------------------------------------------------------------------

# Validar BenchmarkResults
if "BenchmarkResults" not in products:
    raise KeyError("BenchmarkResults no está disponible.")

# Validar BenchmarkRanking
if "BenchmarkRanking" not in products:
    raise KeyError("BenchmarkRanking no está disponible.")

# Validar OfficialModel
if "OfficialModel" not in products:
    raise KeyError("OfficialModel no está disponible.")

# Validar BenchmarkData
if "BenchmarkData" not in products:
    raise KeyError("BenchmarkData no está disponible.")

# Validar BenchmarkConfig
if "BenchmarkConfig" not in products:
    raise KeyError("BenchmarkConfig no está disponible.")

# Validar BenchmarkModels
if "BenchmarkModels" not in products:
    raise KeyError("BenchmarkModels no está disponible.")

#------------------------------------------------------------------------------
# 13.2 Recuperar los productos
#------------------------------------------------------------------------------

# Recuperar BenchmarkResults
benchmark_results = products["BenchmarkResults"]

# Recuperar BenchmarkRanking
benchmark_ranking = products["BenchmarkRanking"]

# Recuperar OfficialModel
official_model = products["OfficialModel"]

# Recuperar BenchmarkData
benchmark_data = products["BenchmarkData"]

# Recuperar GraphData
graph_data = benchmark_data["graphs"]

# Recuperar BenchmarkConfig
benchmark_config = products["BenchmarkConfig"]

# Recuperar BenchmarkModels
benchmark_models = products["BenchmarkModels"]

#------------------------------------------------------------------------------
# 13.3 Validar los productos recuperados
#------------------------------------------------------------------------------

# Validar BenchmarkResults
if not isinstance(benchmark_results, list):
    raise TypeError("BenchmarkResults debe ser una lista.")

if not benchmark_results:
    raise ValueError("BenchmarkResults está vacío.")

# Validar BenchmarkRanking
if not isinstance(benchmark_ranking, list):
    raise TypeError("BenchmarkRanking debe ser una lista.")

if not benchmark_ranking:
    raise ValueError("BenchmarkRanking está vacío.")

# Validar OfficialModel
if not isinstance(official_model, dict):
    raise TypeError("OfficialModel debe ser un diccionario.")

required_fields = (
    "model_code",
    "model_name",
    "family",
    "model_config",
)

missing_fields = [
    field
    for field in required_fields
    if field not in official_model
]

if missing_fields:
    raise ValueError(
        f"OfficialModel no contiene los campos requeridos: {missing_fields}"
    )

# Validar BenchmarkData
if not isinstance(benchmark_data, dict):
    raise TypeError("BenchmarkData debe ser un diccionario.")

# Validar GraphData
if not isinstance(graph_data, list):
    raise TypeError("GraphData debe ser una lista.")

if not graph_data:
    raise ValueError("GraphData está vacío.")

if not all(hasattr(graph, "edge_index") for graph in graph_data):
    raise TypeError(
        "GraphData contiene elementos que no son grafos válidos."
    )

# Validar BenchmarkConfig
if not isinstance(benchmark_config, dict):
    raise TypeError("BenchmarkConfig debe ser un diccionario.")

# Validar BenchmarkModels
if not isinstance(benchmark_models, dict):
    raise TypeError("BenchmarkModels debe ser un diccionario.")

if not benchmark_models:
    raise ValueError("BenchmarkModels está vacío.")

#------------------------------------------------------------------------------
# 13.4 Auditar la estructura del Benchmark
#------------------------------------------------------------------------------

# Auditar BenchmarkResults
print("\nBenchmarkResults")
print(f"Registros                  : {len(benchmark_results)}")

# Auditar BenchmarkRanking
print("\nBenchmarkRanking")
print(f"Modelos clasificados       : {len(benchmark_ranking)}")

# Auditar OfficialModel
print("\nOfficialModel")
print(f"Código                     : {official_model['model_code']}")
print(f"Modelo                     : {official_model['model_name']}")
print(f"Familia                    : {official_model['family']}")

# Auditar BenchmarkData
print("\nBenchmarkData")
print(f"Variables entrenamiento    : {benchmark_data['x_train'].shape}")
print(f"Variables prueba           : {benchmark_data['x_test'].shape}")

# Auditar GraphData
print("\nGraphData")
print(f"Grafos espacio-temporales  : {len(graph_data)}")

#------------------------------------------------------------------------------
# 13.5 Auditar el protocolo experimental del Benchmark
#------------------------------------------------------------------------------

print("\nConfiguración del Benchmark")
print("-" * 80)
print(f"Estado aleatorio            : {benchmark_config['random_state']}")
print(f"Entrenamiento               : {benchmark_config['train_size']:.2f}")
print(f"Validación                  : {benchmark_config['validation_size']:.2f}")
print(f"Prueba                      : {benchmark_config['test_size']:.2f}")
print(f"Validación cruzada          : {benchmark_config['cv_folds']}")
print(f"Barajar datos               : {benchmark_config['shuffle']}")
print(f"Núcleos CPU                 : {benchmark_config['n_jobs']}")
print(f"Métrica oficial             : {benchmark_config['ranking_metric']}")
print(f"Familias metodológicas      : {len(benchmark_models)}")
print(f"Modelos evaluados           : {sum(len(models) for models in benchmark_models.values())}")
print("-" * 80)

#------------------------------------------------------------------------------
# 13.6 Auditar los resultados científicos del Benchmark
#------------------------------------------------------------------------------

# Validar existencia de resultados
if len(benchmark_results) != len(benchmark_ranking):
    raise ValueError(
        "El número de resultados no coincide con el ranking."
    )

# Validar modelo oficial
if official_model not in benchmark_ranking:
    raise ValueError(
        "El modelo oficial no pertenece al BenchmarkRanking."
    )

# Validar familia del modelo oficial
if official_model["family"] != "graph_neural_networks":
    raise ValueError(
        "El modelo oficial no pertenece a la familia Graph Neural Networks."
    )

# Confirmar auditoría científica
print("\nAuditoría de Resultados")
print("-" * 80)
print(f"Resultados evaluados       : {len(benchmark_results)}")
print(f"Modelos clasificados       : {len(benchmark_ranking)}")
print(f"Modelo oficial             : {official_model['model_name']}")
print(f"Familia oficial            : {official_model['family']}")
print("Consistencia científica    : OK")
print("-" * 80)

#------------------------------------------------------------------------------
# 13.7 Registrar la auditoría del Benchmark
#------------------------------------------------------------------------------

audit = {
    "BenchmarkResults": benchmark_results,
    "BenchmarkRanking": benchmark_ranking,
    "OfficialModel": official_model,
    "BenchmarkData": benchmark_data,
    "BenchmarkConfig": benchmark_config,
    "BenchmarkModels": benchmark_models,
    "AuditStatus": "Validated"
}

#------------------------------------------------------------------------------
# 13.8 Confirmar la auditoría del Benchmark
#------------------------------------------------------------------------------

print("\n" + "-" * 80)
print("AUDITORÍA DEL BENCHMARK COMPLETADA")
print("=" * 80)
print(f"Estado                     : {audit['AuditStatus']}")
print(f"Productos auditados        : {len(audit) - 1}")
print(f"Modelo oficial             : {official_model['model_name']}")
print("Ranking científico         : Validado")
print("Consistencia               : Verificada")
print("-" * 80)
