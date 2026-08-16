# 03_benchmark.py

# BLOQUE 1. IMPORTACIONES
# Objetivo: Importar las dependencias necesarias para ejecutar el Benchmark Científico.
# Producto: - Librerías científicas cargadas. - Configuración oficial disponible. - Pipeline oficial disponible.
# - Familias de modelos disponibles.
# Pregunta científica: ¿Se encuentran disponibles todas las dependencias requeridas para ejecutar el protocolo 
# experimental del Benchmark Científico?

# Librerías estándar
import json
import os
import random
import sys
import time
from pathlib import Path
import datetime

# Librerías científicas
import numpy as np
import pandas as pd
import torch
import joblib # Serialización de productos científicos del Benchmark
from sklearn.preprocessing import StandardScaler
import datetime as dt # Importar el módulo datetime con alias no ambiguo
from torch_geometric.nn import SAGEConv # Importar capa convolucional GraphSAGE
import copy # Importar herramientas para copiar objetos

# PyTorch Geometric
from torch_geometric.data import Data

# Configuración Oficial del Benchmark
from src.python.config.config_project import (
    PROJECT_SEED,
    BENCHMARK_CONFIG,
    BENCHMARK_MODELS,
    BENCHMARK_MODEL_CODES,
    BENCHMARK_METRICS,
    BENCHMARK_REPRODUCIBILITY,
    OFFICIAL_MODEL_CODE,
    OFFICIAL_MODEL_NAME,
    OFFICIAL_MODEL_FAMILY,
) # Cargar configuración oficial actualizada

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
    GRAPH_DATA_COLLECTION_FILE,
    BENCHMARK_EXPERIMENT_FILE,
    BENCHMARK_METRICS_FILE,
    BENCHMARK_RANKING_CSV_FILE,
    BENCHMARK_RANKING_XLSX_FILE,
    BENCHMARK_SUMMARY_CSV_FILE,
    BENCHMARK_SUMMARY_XLSX_FILE,
    validate_project_structure,
)

# Utilidades del Proyecto
from src.python.utils.data_preparation import (
    prepare_dataset,
)

from src.python.utils.results import (
    build_exportable_benchmark_result,
    benchmark_results_to_dataframe,
    generate_benchmark_summary,
    validate_model_result,
)

# Pipeline Oficial GraphData
from src.python.graph.graph_pipeline import (
    prepare_graph_collection,
)

# Modelos Estadísticos
from src.python.models.statistical import (
    run_linear_regression,
    STATISTICAL_CONFIG,
)

# Modelos Machine Learning
from src.python.models.machine_learning import (
    MACHINE_LEARNING_CONFIG,
    run_machine_learning,
)

# Modelos Deep Learning
from src.python.models.deep_learning import (
    run_mlp,
    DEEP_LEARNING_CONFIG,
)

# Modelos Graph Neural Networks
from src.python.models.graph_neural_networks import (
    GNN_CONFIG,
    run_gnn_benchmark,
)

# BLOQUE 2. PREPARACIÓN DEL BENCHMARK
# Objetivo: Preparar la estructura oficial BenchmarkData utilizada como entrada por todas las familias de modelos
# del Benchmark Científico, integrando la colección oficial de GraphData, las particiones experimentales y el panel
# tabular derivado del conjunto espacio-temporal del proyecto.
# Entradas: - Resultados del pipeline oficial de preparación de GraphData.
# Producto: - Estructura oficial BenchmarkData.
# Pregunta científica: ¿La colección oficial de GraphData y las particiones experimentales poseen la estructura requerida 
# para ejecutar el Benchmark Científico bajo un protocolo experimental reproducible y consistente para todas las familias de modelos?

# Construcción de BenchmarkData
def build_benchmark_data(
    preparation_results: dict
) -> dict:
    """
    Construye la estructura oficial BenchmarkData a partir de los
    productos generados por prepare_graph_collection.

    Parameters
    ----------
    preparation_results : dict
        Productos oficiales generados durante la preparación de GraphData.

    Returns
    -------
    dict
        Estructura BenchmarkData utilizada por todas las familias
        de modelos del Benchmark Científico.
    """

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
            f"La preparación del Benchmark está incompleta. "
            f"Faltan: {missing_keys}."
        )

    graphs = preparation_results["graphs"] # Recuperar colección oficial GraphData
    partitions = preparation_results["partitions"] # Recuperar particiones temporales oficiales
    scaler = preparation_results["scaler"] # Recuperar escalador oficial

    if not isinstance(graphs, (list, tuple)):
        raise TypeError(
            "'graphs' debe ser una lista o tupla."
        )

    if len(graphs) == 0:
        raise ValueError(
            "La colección oficial GraphData está vacía."
        )

    if not isinstance(partitions, dict):
        raise TypeError(
            "'partitions' debe ser un diccionario."
        )

    required_partitions = [
        "train_index",
        "validation_index",
        "test_index",
    ] # Definir particiones temporales obligatorias

    missing_partitions = [
        key
        for key in required_partitions
        if key not in partitions
    ]

    if missing_partitions:
        raise ValueError(
            "Las particiones del Benchmark están incompletas. "
            f"Faltan: {missing_partitions}."
        )

    train_index = np.asarray(
        partitions["train_index"],
        dtype=np.int64
    ) # Convertir índices de entrenamiento a NumPy

    validation_index = np.asarray(
        partitions["validation_index"],
        dtype=np.int64
    ) # Convertir índices de validación a NumPy

    test_index = np.asarray(
        partitions["test_index"],
        dtype=np.int64
    ) # Convertir índices de prueba a NumPy

    if train_index.ndim != 1:
        raise ValueError(
            "'train_index' debe ser un vector unidimensional."
        )

    if validation_index.ndim != 1:
        raise ValueError(
            "'validation_index' debe ser un vector unidimensional."
        )

    if test_index.ndim != 1:
        raise ValueError(
            "'test_index' debe ser un vector unidimensional."
        )

    if len(train_index) == 0:
        raise ValueError(
            "La partición de entrenamiento está vacía."
        )

    if len(validation_index) == 0:
        raise ValueError(
            "La partición de validación está vacía."
        )

    if len(test_index) == 0:
        raise ValueError(
            "La partición de prueba está vacía."
        )

    for partition_name, partition_index in {
        "train_index": train_index,
        "validation_index": validation_index,
        "test_index": test_index,
    }.items():

        if not np.issubdtype(
            partition_index.dtype,
            np.integer
        ):
            raise TypeError(
                f"'{partition_name}' debe contener únicamente "
                "índices enteros."
            )

        if np.any(
            partition_index < 0
        ) or np.any(
            partition_index >= len(graphs)
        ):
            raise IndexError(
                f"'{partition_name}' contiene índices fuera de rango."
            )

    all_indices = set(
        range(len(graphs))
    ) # Definir índices temporales esperados

    train_set = set(
        train_index.tolist()
    ) # Índices de entrenamiento

    validation_set = set(
        validation_index.tolist()
    ) # Índices de validación

    test_set = set(
        test_index.tolist()
    ) # Índices de prueba

    if len(train_index) != len(train_set):
        raise ValueError(
            "La partición de entrenamiento contiene índices duplicados."
        )

    if len(validation_index) != len(validation_set):
        raise ValueError(
            "La partición de validación contiene índices duplicados."
        )

    if len(test_index) != len(test_set):
        raise ValueError(
            "La partición de prueba contiene índices duplicados."
        )

    if train_set | validation_set | test_set != all_indices:
        raise ValueError(
            "Las particiones no cubren exactamente "
            "la colección oficial GraphData."
        )

    if train_set & validation_set:
        raise ValueError(
            "Existe solapamiento entre entrenamiento y validación."
        )

    if train_set & test_set:
        raise ValueError(
            "Existe solapamiento entre entrenamiento y prueba."
        )

    if validation_set & test_set:
        raise ValueError(
            "Existe solapamiento entre validación y prueba."
        )

    if scaler is None:
        raise ValueError(
            "El escalador oficial del Benchmark es inválido."
        )

    for graph_position, graph in enumerate(graphs):
        if not isinstance(graph, Data):
            raise TypeError(
                f"La colección GraphData contiene un elemento inválido "
                f"en la posición {graph_position}."
            )

        required_attributes = [
            "x",
            "y",
        ] # Definir atributos obligatorios de GraphData

        missing_attributes = [
            attribute
            for attribute in required_attributes
            if not hasattr(graph, attribute)
        ]

        if missing_attributes:
            raise ValueError(
                f"El GraphData {graph_position} no contiene: "
                f"{missing_attributes}."
            )

        if graph.x is None:
            raise ValueError(
                f"El GraphData {graph_position} contiene 'x' inválido."
            )

        if graph.y is None:
            raise ValueError(
                f"El GraphData {graph_position} contiene 'y' inválido."
            )

        if graph.x.ndim != 2:
            raise ValueError(
                f"El GraphData {graph_position} presenta 'x' "
                "con una dimensión inválida."
            )

        if graph.y.numel() != graph.x.shape[0]:
            raise ValueError(
                f"El GraphData {graph_position} presenta dimensiones "
                "incompatibles entre 'x' e 'y'."
            )

    reference_nodes = graphs[0].x.shape[0] # Número de nodos de referencia
    for graph_position, graph in enumerate(graphs):
        if graph.x.shape[0] != reference_nodes:
            raise ValueError(
                f"El GraphData {graph_position} presenta un número "
                "de nodos diferente."
            )

    train_graphs = [
        graphs[index]
        for index in train_index
    ] # GraphData temporales de entrenamiento

    validation_graphs = [
        graphs[index]
        for index in validation_index
    ] # GraphData temporales de validación

    test_graphs = [
        graphs[index]
        for index in test_index
    ] # GraphData temporales de prueba

    x_train = np.concatenate(
        [
            graph.x.detach().cpu().numpy()
            for graph in train_graphs
        ],
        axis=0
    ) # Variables predictoras de entrenamiento

    y_train = np.concatenate(
        [
            graph.y.detach().cpu().numpy().reshape(-1)
            for graph in train_graphs
        ],
        axis=0
    ) # Variable objetivo de entrenamiento

    x_validation = np.concatenate(
        [
            graph.x.detach().cpu().numpy()
            for graph in validation_graphs
        ],
        axis=0
    ) # Variables predictoras de validación

    y_validation = np.concatenate(
        [
            graph.y.detach().cpu().numpy().reshape(-1)
            for graph in validation_graphs
        ],
        axis=0
    ) # Variable objetivo de validación

    x_test = np.concatenate(
        [
            graph.x.detach().cpu().numpy()
            for graph in test_graphs
        ],
        axis=0
    ) # Variables predictoras de prueba

    y_test = np.concatenate(
        [
            graph.y.detach().cpu().numpy().reshape(-1)
            for graph in test_graphs
        ],
        axis=0
    ) # Variable objetivo de prueba

    expected_train_rows = (
        len(train_index) * reference_nodes
    ) # Observaciones esperadas de entrenamiento

    expected_validation_rows = (
        len(validation_index) * reference_nodes
    ) # Observaciones esperadas de validación

    expected_test_rows = (
        len(test_index) * reference_nodes
    ) # Observaciones esperadas de prueba

    if x_train.shape[0] != expected_train_rows:
        raise RuntimeError(
            "El número de observaciones de entrenamiento "
            "no coincide con la partición temporal."
        )

    if x_validation.shape[0] != expected_validation_rows:
        raise RuntimeError(
            "El número de observaciones de validación "
            "no coincide con la partición temporal."
        )

    if x_test.shape[0] != expected_test_rows:
        raise RuntimeError(
            "El número de observaciones de prueba "
            "no coincide con la partición temporal."
        )

    if x_train.shape[0] != y_train.shape[0]:
        raise RuntimeError(
            "X e y de entrenamiento no tienen el mismo "
            "número de observaciones."
        )

    if x_validation.shape[0] != y_validation.shape[0]:
        raise RuntimeError(
            "X e y de validación no tienen el mismo "
            "número de observaciones."
        )

    if x_test.shape[0] != y_test.shape[0]:
        raise RuntimeError(
            "X e y de prueba no tienen el mismo "
            "número de observaciones."
        )

    benchmark_data = {
        "graphs": list(graphs),
        "partitions": {
            "train_index": train_index,
            "validation_index": validation_index,
            "test_index": test_index,
        },
        "train_index": train_index,
        "validation_index": validation_index,
        "test_index": test_index,
        "x_train": x_train,
        "y_train": y_train,
        "x_validation": x_validation,
        "y_validation": y_validation,
        "x_test": x_test,
        "y_test": y_test,
        "scaler": scaler,
    } # Construir producto oficial BenchmarkData

    required_products = [
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
    ] # Definir contrato oficial de BenchmarkData

    missing_products = [
        product
        for product in required_products
        if product not in benchmark_data
    ]

    if missing_products:
        raise RuntimeError(
            f"BenchmarkData está incompleto: {missing_products}"
        )
    return benchmark_data

# Definición benchmark_ranking
def build_benchmark_ranking(
    benchmark_results: list[dict],
    benchmark_metrics: list | tuple | set | dict
) -> list[dict]:
    """
    Construye el Ranking Científico Oficial del Benchmark.
    La métrica utilizada para ordenar los modelos se obtiene
    desde BENCHMARK_CONFIG['ranking_metric'] y su dirección
    desde BENCHMARK_CONFIG['metric_directions'].

    Parameters
    ----------
    benchmark_results : list[dict]
        Resultados consolidados de los modelos evaluados.

    benchmark_metrics : list | tuple | set | dict
        Colección oficial de métricas del Benchmark.

    Returns
    -------
    list[dict]
        Ranking Científico ordenado y numerado.
    """

    if not isinstance(
        benchmark_results,
        list
    ):
        raise TypeError(
            "'benchmark_results' debe ser una lista."
        ) # Validar estructura de resultados

    if len(benchmark_results) == 0:
        raise ValueError(
            "No existen resultados para construir el Ranking Científico."
        ) # Verificar existencia de resultados

    if not isinstance(
        benchmark_metrics,
        (list, tuple, set, dict)
    ):
        raise TypeError(
            "'benchmark_metrics' debe ser una lista, "
            "tupla, conjunto o diccionario."
        ) # Validar colección de métricas

    if isinstance(
        benchmark_metrics,
        dict
    ):
        available_metrics = set(
            benchmark_metrics.keys()
        ) # Recuperar métricas desde diccionario

    else:
        available_metrics = set(
            benchmark_metrics
        ) # Recuperar métricas desde lista, tupla o conjunto

    ranking_metric = BENCHMARK_CONFIG.get(
        "ranking_metric"
    ) # Recuperar métrica oficial de ranking

    if ranking_metric is None:
        raise KeyError(
            "La configuración no contiene 'ranking_metric'."
        ) # Validar configuración del ranking

    if ranking_metric not in available_metrics:
        raise ValueError(
            f"La métrica oficial '{ranking_metric}' "
            "no está registrada en BENCHMARK_METRICS."
        ) # Validar métrica oficial

    metric_directions = BENCHMARK_CONFIG.get(
        "metric_directions",
        {}
    ) # Recuperar direcciones oficiales

    if not isinstance(
        metric_directions,
        dict
    ):
        raise TypeError(
            "'metric_directions' debe ser un diccionario."
        ) # Validar estructura de direcciones

    if ranking_metric not in metric_directions:
        raise KeyError(
            f"No existe dirección de optimización "
            f"para '{ranking_metric}'."
        ) # Validar dirección de ranking

    direction = metric_directions[
        ranking_metric
    ] # Recuperar dirección oficial

    if direction not in {
        "min",
        "max"
    }:
        raise ValueError(
            f"Dirección de optimización inválida "
            f"para '{ranking_metric}': {direction}"
        ) # Validar dirección

    candidate_identities = set()
    for candidate in CANDIDATE_MODELS["models"]:
        if not isinstance(
            candidate,
            dict
        ):
            raise TypeError(
                "Cada modelo candidato debe ser un diccionario."
            ) # Validar estructura del candidato

        required_candidate_keys = [
            "model_code",
            "model_name",
            "family",
        ] # Definir identidad obligatoria del candidato

        missing_candidate_keys = [
            key
            for key in required_candidate_keys
            if key not in candidate
        ] # Identificar campos faltantes

        if missing_candidate_keys:
            raise ValueError(
                f"El modelo candidato está incompleto: "
                f"{missing_candidate_keys}"
            ) # Validar identidad del candidato

        candidate_code = candidate[
            "model_code"
        ].strip().lower() # Normalizar código candidato

        candidate_name = candidate[
            "model_name"
        ].strip().lower() # Normalizar nombre candidato

        candidate_family = candidate[
            "family"
        ].strip().lower() # Normalizar familia candidata

        candidate_identities.add(
            (
                candidate_code,
                candidate_name,
                candidate_family,
            )
        ) # Registrar identidad oficial

    validated_results = []
    for result in benchmark_results:
        if not isinstance(
            result,
            dict
        ):
            raise TypeError(
                "Cada resultado del Benchmark debe ser un diccionario."
            ) # Validar resultado individual

        required_identity_keys = [
            "model_code",
            "model_name",
            "family",
        ] # Definir identidad mínima del resultado

        missing_identity_keys = [
            key
            for key in required_identity_keys
            if key not in result
        ] # Identificar identidad incompleta

        if missing_identity_keys:
            raise ValueError(
                f"El resultado del modelo "
                f"'{result.get('model_name', 'desconocido')}' "
                f"no contiene la identidad completa: "
                f"{missing_identity_keys}"
            ) # Validar identidad completa

        model_code = result[
            "model_code"
        ].strip().lower() # Recuperar código del resultado

        model_name = result[
            "model_name"
        ].strip().lower() # Recuperar nombre del resultado

        model_family = result[
            "family"
        ].strip().lower() # Recuperar familia del resultado

        result_identity = (
            model_code,
            model_name,
            model_family,
        ) # Construir identidad completa del resultado

        if result_identity not in candidate_identities:
            raise ValueError(
                f"El resultado del modelo "
                f"'{result['model_name']}' "
                f"no coincide exactamente con "
                "ningún modelo oficial de CANDIDATE_MODELS."
            ) # Validar correspondencia exacta con catálogo

        if ranking_metric not in result:
            raise ValueError(
                f"El resultado del modelo "
                f"'{result['model_name']}' "
                f"no contiene la métrica "
                f"'{ranking_metric}'."
            ) # Validar presencia de métrica

        metric_value = result[
            ranking_metric
        ] # Recuperar valor de ranking

        if not isinstance(
            metric_value,
            (int, float, np.integer, np.floating)
        ):
            raise TypeError(
                f"La métrica '{ranking_metric}' "
                f"del modelo '{result['model_name']}' "
                "debe ser numérica."
            ) # Validar tipo de métrica

        if not np.isfinite(
            float(metric_value)
        ):
            raise ValueError(
                f"La métrica '{ranking_metric}' "
                f"del modelo '{result['model_name']}' "
                "contiene un valor no finito."
            ) # Validar valor numérico

        validated_results.append(
            result.copy()
        ) # Registrar resultado validado

    executed_identities = [
        (
            result["model_code"].strip().lower(),
            result["model_name"].strip().lower(),
            result["family"].strip().lower(),
        )
        for result in validated_results
    ] # Construir identidades ejecutadas

    if len(executed_identities) != len(
        set(executed_identities)
    ):
        raise ValueError(
            "Existen resultados duplicados para uno o más modelos."
        ) # Validar unicidad de resultados

    executed_identity_set = set(
        executed_identities
    ) # Construir conjunto de modelos ejecutados

    missing_candidates = (
        candidate_identities
        - executed_identity_set
    ) # Identificar modelos candidatos sin resultado

    if missing_candidates:
        raise RuntimeError(
            "Existen modelos candidatos que no produjeron "
            f"resultados: {sorted(missing_candidates)}"
        ) # Validar cobertura completa

    unexpected_results = (
        executed_identity_set
        - candidate_identities
    ) # Identificar resultados no oficiales

    if unexpected_results:
        raise RuntimeError(
            "Existen resultados que no pertenecen al "
            f"catálogo oficial: {sorted(unexpected_results)}"
        ) # Validar ausencia de resultados externos

    benchmark_ranking = sorted(
        validated_results,
        key=lambda result: float(
            result[ranking_metric]
        ),
        reverse=(
            direction == "max"
        )
    ) # Ordenar modelos según métrica oficial

    for position, result in enumerate(
        benchmark_ranking,
        start=1
    ):
        result[
            "ranking_position"
        ] = position # Asignar posición científica
    return benchmark_ranking

# select_official_model ----------------------
def select_official_model(
    benchmark_ranking: list[dict]
) -> dict:
    """
    Selecciona el Modelo Oficial del Proyecto.
    La selección se realiza exclusivamente dentro de la familia
    definida mediante OFFICIAL_MODEL_FAMILY.
    El Benchmark determina el mejor modelo de dicha familia
    utilizando el Ranking Científico.
    """

    if not isinstance(benchmark_ranking, list):
        raise TypeError(
            "'benchmark_ranking' debe ser una lista."
        )

    if len(benchmark_ranking) == 0:
        raise ValueError(
            "El Ranking Científico está vacío."
        )

    if not isinstance(OFFICIAL_MODEL_FAMILY, str):
        raise TypeError(
            "OFFICIAL_MODEL_FAMILY debe ser una cadena."
        )

    if not OFFICIAL_MODEL_FAMILY:
        raise ValueError(
            "OFFICIAL_MODEL_FAMILY está vacío."
        )

    for model in benchmark_ranking:
        if not isinstance(model, dict):
            raise TypeError(
                "Cada elemento del Ranking Científico "
                "debe ser un diccionario."
            )

        required_keys = [
            "model_code",
            "model_name",
            "family",
            "model_config",
            "ranking_position",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in model
        ]

        if missing_keys:
            raise ValueError(
                "Un modelo del Ranking no contiene los campos "
                f"obligatorios: {missing_keys}."
            )

        if not isinstance(model["ranking_position"], (int, np.integer)):
            raise TypeError("'ranking_position' debe ser un entero.")

        if model["family"] is None:
            raise ValueError(
                "'family' es inválida."
            )

        if model["model_config"] is None:
            raise ValueError(
                "'model_config' es inválida."
            )

    official_candidates = [
        model
        for model in benchmark_ranking
        if model["family"] == OFFICIAL_MODEL_FAMILY
    ]

    if not official_candidates:
        raise ValueError(
            "No existen modelos de la familia oficial "
            "en el Ranking Científico."
        )

    official_model = official_candidates[0].copy()
    official_model["selection_basis"] = {
        "official_family": OFFICIAL_MODEL_FAMILY,
        "ranking_position": official_model["ranking_position"],
        "ranking_metric": BENCHMARK_CONFIG.get(
            "ranking_metric"
        ),
    }
    return official_model

# BLOQUE 3. CONFIGURACIÓN DEL SCRIPT
# Objetivo: Configurar el entorno de ejecución, validar la estructura oficial
# del proyecto e inicializar los parámetros necesarios para ejecutar el
# Benchmark Científico de forma reproducible.
# Producto: - Entorno de ejecución configurado. - Estructura oficial del proyecto validada.
# - Reproducibilidad establecida. - Dispositivo de procesamiento seleccionado.
# Pregunta científica: ¿El entorno de ejecución cumple las condiciones necesarias para ejecutar
# un Benchmark Científico reproducible y consistente?

# 3.1 Configuración del entorno
print("Configurando entorno de ejecución del Benchmark...")

# 3.2 Validación de la estructura del proyecto
validate_project_structure(verbose=True) # Validar estructura oficial del proyecto
print("Estructura del proyecto validada correctamente.")

# 3.3 Validación de la configuración científica
if not isinstance(PROJECT_SEED, (int, np.integer)):
    raise TypeError("La semilla oficial del proyecto debe ser un entero.")

if PROJECT_SEED < 0:
    raise ValueError("La semilla oficial del proyecto no puede ser negativa.")

if not isinstance(BENCHMARK_REPRODUCIBILITY, dict):
    raise TypeError("'BENCHMARK_REPRODUCIBILITY' debe ser un diccionario.")

if "deterministic" not in BENCHMARK_REPRODUCIBILITY:
    raise KeyError("La configuración no contiene 'deterministic'.")

deterministic = BENCHMARK_REPRODUCIBILITY["deterministic"] # Recuperar configuración determinística

if not isinstance(deterministic, bool):
    raise TypeError("'deterministic' debe ser un valor booleano.")

# 3.4 Configuración de la reproducibilidad
random.seed(PROJECT_SEED) # Configurar semilla oficial para Python
np.random.seed(PROJECT_SEED) # Configurar semilla oficial para NumPy
torch.manual_seed(PROJECT_SEED) # Configurar semilla oficial para PyTorch

if deterministic:
    torch.use_deterministic_algorithms(True) # Activar algoritmos determinísticos

cuda_available = torch.cuda.is_available() # Detectar disponibilidad de GPU
if cuda_available:
    torch.cuda.manual_seed_all(PROJECT_SEED) # Configurar semilla en todas las GPU
    torch.backends.cudnn.deterministic = deterministic # Configurar operaciones determinísticas de cuDNN
    torch.backends.cudnn.benchmark = False # Desactivar selección dinámica de algoritmos

# 3.5 Selección del dispositivo
DEVICE = "cuda" if cuda_available else "cpu" # Seleccionar dispositivo de procesamiento

# 3.6 Certificación del entorno de ejecución
print("-" * 80)
print("CERTIFICACIÓN DEL ENTORNO DE EJECUCIÓN")
print("-" * 80)
print(f"Semilla del proyecto         : {PROJECT_SEED}")
print(f"Modo determinístico          : {deterministic}")
print(f"Dispositivo de procesamiento : {DEVICE.upper()}")

if cuda_available:
    print(f"GPU detectada                : {torch.cuda.get_device_name(0)}")

print("Estado                       : OK")
print("-" * 80)
print("Bloque 3. Configuración del entorno completada correctamente.")

# BLOQUE 4. CONSTRUCCIÓN DE LA COLECCIÓN OFICIAL DEL BENCHMARK
# Objetivo: Construir la colección oficial BenchmarkData a partir de los GraphData generados por el módulo de construcción del grafo.
# Producto: BENCHMARK_DATA.
# Pregunta científica: ¿La colección oficial BenchmarkData fue construida correctamente para ejecutar un Benchmark Científico reproducible y comparable?

# 4.1 Carga de la Colección Oficial GraphData
print("4.1 CARGA DE LA COLECCIÓN OFICIAL GraphData")
graph_collection_file = GRAPH_DATA_COLLECTION_FILE # Utilizar archivo oficial de la colección GraphData
if not graph_collection_file.exists():
    raise FileNotFoundError(
        f"No se encontró la colección oficial GraphData: {graph_collection_file}"
    )

graph_collection = torch.load(
    graph_collection_file,
    weights_only=False,
) # Recuperar colección oficial GraphData

if not isinstance(graph_collection, (dict, list, tuple)):
    raise TypeError(
        "La colección oficial GraphData debe ser un diccionario, lista o tupla."
    )

if len(graph_collection) == 0:
    raise ValueError(
        "La colección oficial GraphData está vacía."
    )

graphs = []
if isinstance(graph_collection, dict):
    ordered_graphs = [
        graph_collection[key]
        for key in sorted(graph_collection)
    ] # Recuperar GraphData ordenados según las claves de la colección

else:
    ordered_graphs = list(graph_collection) # Convertir la colección secuencial a una lista

for index, graph in enumerate(ordered_graphs):
    if not isinstance(graph, Data):
        raise TypeError(
            f"El elemento {index} de la colección no es un GraphData válido."
        )

    required_attributes = [
        "x",
        "y",
        "edge_index",
    ] # Definir atributos estructurales obligatorios

    missing_attributes = [
        attribute
        for attribute in required_attributes
        if not hasattr(graph, attribute)
    ]

    if missing_attributes:
        raise ValueError(
            f"El GraphData {index} no contiene: {missing_attributes}."
        )

    for attribute in required_attributes:

        if getattr(graph, attribute) is None:
            raise ValueError(
                f"El GraphData {index} posee '{attribute}' inválido."
            )

    if graph.x.ndim != 2:
        raise ValueError(
            f"El GraphData {index} presenta 'x' con una dimensión inválida."
        )

    if graph.y.numel() != graph.x.shape[0]:
        raise ValueError(
            f"El GraphData {index} presenta dimensiones incompatibles entre 'x' e 'y'."
        )

    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError(
            f"El GraphData {index} presenta un 'edge_index' inválido."
        )

    if graph.edge_index.numel() > 0:

        if graph.edge_index.min().item() < 0:
            raise ValueError(
                f"El GraphData {index} contiene índices de aristas negativos."
            )

        if graph.edge_index.max().item() >= graph.num_nodes:
            raise ValueError(
                f"El GraphData {index} contiene índices de aristas "
                "fuera del rango de nodos."
            )

    graphs.append(graph)

if not graphs:
    raise ValueError(
        "No se recuperaron GraphData válidos."
    )

reference_graph = graphs[0] # Utilizar el primer GraphData como referencia estructural

print(f"Colección localizada       : {graph_collection_file.name}") # Mostrar archivo utilizado
print(f"Tipo de colección         : {type(graph_collection).__name__}") # Mostrar tipo de colección
print(f"GraphData cargados        : {len(graphs)}") # Mostrar cantidad de GraphData
print(f"Nodos                     : {reference_graph.num_nodes}") # Mostrar cantidad de nodos
print(f"Aristas                   : {reference_graph.num_edges}") # Mostrar cantidad de aristas
print(f"Node Features             : {reference_graph.num_node_features}") # Mostrar número de variables por nodo
print(f"Variable objetivo         : {tuple(reference_graph.y.shape)}") # Mostrar dimensión de la variable objetivo
print("Colección oficial GraphData cargada correctamente.") # Confirmar carga y validación

# 4.2 Validación estructural de la colección GraphData
print("\n4.2 VALIDACIÓN ESTRUCTURAL DE LA COLECCIÓN GraphData")
expected_nodes = reference_graph.num_nodes # Definir número esperado de nodos
expected_features = reference_graph.num_node_features # Definir número esperado de variables

if expected_nodes <= 0:
    raise ValueError(
        "El número esperado de nodos debe ser mayor que cero."
    )

if expected_features <= 0:
    raise ValueError(
        "El número esperado de variables debe ser mayor que cero."
    )

edge_counts = []
for position, graph in enumerate(graphs):
    if graph.num_nodes != expected_nodes:
        raise ValueError(
            f"El GraphData {position} presenta un número de nodos "
            f"diferente: {graph.num_nodes} frente a {expected_nodes}."
        )

    if graph.num_node_features != expected_features:
        raise ValueError(
            f"El GraphData {position} presenta un número de variables "
            f"diferente: {graph.num_node_features} frente a {expected_features}."
        )

    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError(
            f"El GraphData {position} presenta un 'edge_index' inválido."
        )

    if graph.edge_index.numel() > 0:

        if graph.edge_index.min().item() < 0:
            raise ValueError(
                f"El GraphData {position} contiene índices de aristas negativos."
            )

        if graph.edge_index.max().item() >= graph.num_nodes:
            raise ValueError(
                f"El GraphData {position} contiene índices de aristas "
                "fuera del rango de nodos."
            )
    edge_counts.append(graph.num_edges)

print(f"GraphData validados        : {len(graphs)}") # Mostrar cantidad de GraphData
print(f"Nodos consistentes         : {expected_nodes}") # Mostrar consistencia de nodos
print(f"Variables consistentes     : {expected_features}") # Mostrar consistencia de variables
print(f"Aristas mínimas            : {min(edge_counts)}") # Mostrar mínimo de aristas
print(f"Aristas máximas            : {max(edge_counts)}") # Mostrar máximo de aristas
print(f"Aristas promedio           : {np.mean(edge_counts):.2f}") # Mostrar promedio de aristas
print("Estructura nodal            : VALIDADA") # Confirmar consistencia de nodos
print("Estructura de variables     : VALIDADA") # Confirmar consistencia de variables
print("Integridad de aristas       : VALIDADA") # Confirmar validez topológica
print("Validación estructural      : VALIDADA") # Confirmar validación completa

# 4.3 Construcción de particiones temporales
print("\n4.3 CONSTRUCCIÓN DE LAS PARTICIONES DEL BENCHMARK")
required_split_keys = [
    "train_size",
    "validation_size",
    "test_size",
] # Definir proporciones temporales obligatorias

missing_split_keys = [
    key
    for key in required_split_keys
    if key not in BENCHMARK_CONFIG
]

if missing_split_keys:
    raise KeyError(
        f"BENCHMARK_CONFIG no contiene: {missing_split_keys}."
    )

train_size = BENCHMARK_CONFIG["train_size"] # Recuperar proporción de entrenamiento
validation_size = BENCHMARK_CONFIG["validation_size"] # Recuperar proporción de validación
test_size = BENCHMARK_CONFIG["test_size"] # Recuperar proporción de prueba

split_sizes = [
    train_size,
    validation_size,
    test_size,
] # Agrupar proporciones para validación

if not all(
    isinstance(value, (int, float, np.integer, np.floating))
    for value in split_sizes
):
    raise TypeError(
        "Las proporciones del Benchmark deben ser numéricas."
    )

if not all(
    0 < value < 1
    for value in split_sizes
):
    raise ValueError(
        "Las proporciones del Benchmark deben estar entre 0 y 1."
    )

if not np.isclose(
    sum(split_sizes),
    1.0
):
    raise ValueError(
        "Las proporciones del Benchmark deben sumar 1.0."
    )

n_graphs = len(graphs) # Obtener número total de GraphData

if n_graphs < 3:
    raise ValueError(
        "Se requieren al menos tres GraphData para construir "
        "particiones de entrenamiento, validación y prueba."
    )

train_end = int(
    n_graphs * train_size
) # Calcular límite temporal de entrenamiento

validation_end = train_end + int(
    n_graphs * validation_size
) # Calcular límite temporal de validación

if train_end <= 0:
    raise ValueError(
        "La partición de entrenamiento quedó vacía."
    )

if validation_end <= train_end:
    raise ValueError(
        "La partición de validación quedó vacía."
    )

if validation_end >= n_graphs:
    raise ValueError(
        "Las proporciones configuradas dejan vacía "
        "la partición de prueba."
    )

train_index = list(
    range(
        0,
        train_end,
    )
) # Índices temporales de entrenamiento

validation_index = list(
    range(
        train_end,
        validation_end,
    )
) # Índices temporales de validación

test_index = list(
    range(
        validation_end,
        n_graphs,
    )
) # Índices temporales de prueba

partitions = {
    "train_index": train_index,
    "validation_index": validation_index,
    "test_index": test_index,
} # Construir particiones temporales oficiales

partition_total = (
    len(train_index)
    + len(validation_index)
    + len(test_index)
) # Calcular total de GraphData particionados

if partition_total != n_graphs:
    raise RuntimeError(
        "La cantidad de GraphData particionados no coincide "
        "con la colección original."
    )

partition_indices = (
    train_index
    + validation_index
    + test_index
) # Concatenar índices para verificar cobertura temporal

if len(partition_indices) != len(
    set(partition_indices)
):
    raise RuntimeError(
        "Existen índices GraphData duplicados entre las particiones."
    )

if sorted(partition_indices) != list(
    range(n_graphs)
):
    raise RuntimeError(
        "Las particiones temporales no cubren correctamente "
        "todos los GraphData."
    )

print(f"GraphData total             : {n_graphs}") # Mostrar total de GraphData
print(f"Entrenamiento               : {len(train_index)}") # Mostrar GraphData de entrenamiento
print(f"Validación                  : {len(validation_index)}") # Mostrar GraphData de validación
print(f"Prueba                      : {len(test_index)}") # Mostrar GraphData de prueba
print("Orden temporal              : PRESERVADO") # Confirmar partición cronológica
print("Cobertura de particiones    : VALIDADA") # Confirmar cobertura completa

# 4.4 Construcción del escalador oficial
print("\n4.4 CONSTRUCCIÓN DEL ESCALADOR DEL BENCHMARK")

scaler = StandardScaler() # Crear escalador oficial
train_features = []
for index in train_index:
    graph = graphs[index]
    if graph.x is None:
        raise ValueError(
            f"El GraphData {index} no contiene 'x'."
        )

    if graph.x.ndim != 2:
        raise ValueError(
            f"El GraphData {index} presenta 'x' con una dimensión inválida."
        )

    if graph.x.shape[1] != expected_features:
        raise ValueError(
            f"El GraphData {index} presenta {graph.x.shape[1]} variables "
            f"en lugar de las {expected_features} esperadas."
        )

    train_features.append(
        graph.x.detach().cpu().numpy()
    ) # Recuperar variables del conjunto de entrenamiento

if not train_features:
    raise RuntimeError(
        "No existen variables disponibles para ajustar el escalador."
    )

train_features = np.concatenate(
    train_features,
    axis=0
) # Consolidar variables exclusivamente de entrenamiento

if train_features.shape[1] != expected_features:
    raise RuntimeError(
        "La matriz de entrenamiento presenta un número de variables "
        "incompatible con GraphData."
    )

scaler.fit(
    train_features
) # Ajustar escalador exclusivamente con entrenamiento

if scaler.n_features_in_ != expected_features:
    raise RuntimeError(
        "El escalador presenta un número de variables incompatible "
        "con GraphData."
    )

if not np.all(
    np.isfinite(train_features)
):
    raise ValueError(
        "Las variables de entrenamiento contienen valores no finitos."
    )

print(f"Escalador                   : {type(scaler).__name__}") # Mostrar tipo de escalador
print(f"Observaciones utilizadas    : {train_features.shape[0]}") # Mostrar observaciones utilizadas
print(f"Variables ajustadas         : {scaler.n_features_in_}") # Mostrar variables ajustadas
print("Fuente del ajuste           : TRAIN") # Confirmar fuente del ajuste
print("Escalador                   : AJUSTADO CON TRAIN") # Confirmar ajuste sin fuga de información

# 4.5 Construcción de BenchmarkData
print("\n4.5 CONSTRUCCIÓN DE LA COLECCIÓN OFICIAL BenchmarkData")
preparation_results = {
    "graphs": graphs,
    "partitions": partitions,
    "scaler": scaler,
} # Construir productos oficiales para BenchmarkData

BENCHMARK_DATA = build_benchmark_data(
    preparation_results
) # Construir BenchmarkData mediante el contrato oficial del Bloque 2

if not isinstance(
    BENCHMARK_DATA,
    dict
):
    raise TypeError(
        "BENCHMARK_DATA debe ser un diccionario."
    )

required_products = [
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
] # Definir contrato oficial de BenchmarkData

missing_products = [
    product
    for product in required_products
    if product not in BENCHMARK_DATA
]

if missing_products:
    raise RuntimeError(
        f"BenchmarkData está incompleto: {missing_products}"
    )

# Validar colección GraphData
if not isinstance(
    BENCHMARK_DATA["graphs"],
    list
):
    raise TypeError(
        "BENCHMARK_DATA['graphs'] debe ser una lista."
    )

if len(BENCHMARK_DATA["graphs"]) != len(graphs):
    raise RuntimeError(
        "La cantidad de GraphData de BenchmarkData no coincide "
        "con la colección original."
    )

# Validar estructura de particiones
if not isinstance(
    BENCHMARK_DATA["partitions"],
    dict
):
    raise TypeError(
        "BENCHMARK_DATA['partitions'] debe ser un diccionario."
    )

required_partition_keys = [
    "train_index",
    "validation_index",
    "test_index",
] # Definir particiones oficiales obligatorias

missing_partition_keys = [
    key
    for key in required_partition_keys
    if key not in BENCHMARK_DATA["partitions"]
]

if missing_partition_keys:
    raise RuntimeError(
        f"BENCHMARK_DATA['partitions'] está incompleto: "
        f"{missing_partition_keys}"
    )

# Validar índices directos
for partition_name in [
    "train_index",
    "validation_index",
    "test_index",
]:
    partition_index = BENCHMARK_DATA[partition_name] # Recuperar índice oficial

    if not isinstance(
        partition_index,
        np.ndarray
    ):
        raise TypeError(
            f"'{partition_name}' debe ser un arreglo NumPy."
        )

    if partition_index.ndim != 1:
        raise ValueError(
            f"'{partition_name}' debe ser un vector unidimensional."
        )

    if len(partition_index) == 0:
        raise RuntimeError(
            f"BenchmarkData no contiene datos en '{partition_name}'."
        )

    if not np.issubdtype(
        partition_index.dtype,
        np.integer
    ):
        raise TypeError(
            f"'{partition_name}' debe contener índices enteros."
        )

# Validar índices contenidos en partitions
for partition_name in [
    "train_index",
    "validation_index",
    "test_index",
]:
    partition_index = BENCHMARK_DATA["partitions"][partition_name] # Recuperar índice de la estructura de particiones

    if not isinstance(
        partition_index,
        np.ndarray
    ):
        raise TypeError(
            f"partitions['{partition_name}'] debe ser un arreglo NumPy."
        )

    if partition_index.ndim != 1:
        raise ValueError(
            f"partitions['{partition_name}'] debe ser un vector unidimensional."
        )

    if not np.issubdtype(
        partition_index.dtype,
        np.integer
    ):
        raise TypeError(
            f"partitions['{partition_name}'] debe contener índices enteros."
        )

    if len(partition_index) == 0:
        raise RuntimeError(
            f"partitions['{partition_name}'] está vacío."
        )

# Validar coherencia entre índices directos y partitions
for partition_name in [
    "train_index",
    "validation_index",
    "test_index",
]:
    direct_index = BENCHMARK_DATA[partition_name] # Recuperar índice directo
    partition_index = BENCHMARK_DATA["partitions"][partition_name] # Recuperar índice de partitions

    if not np.array_equal(
        direct_index,
        partition_index
    ):
        raise RuntimeError(
            f"El índice directo '{partition_name}' "
            "no coincide con el índice almacenado en 'partitions'."
        )

# Validar escalador oficial
if BENCHMARK_DATA["scaler"] is not scaler:
    raise RuntimeError(
        "El escalador almacenado en BenchmarkData no coincide "
        "con el escalador oficial ajustado en TRAIN."
    )

# Validar matrices experimentales
for split_name in [
    "x_train",
    "x_validation",
    "x_test",
    "y_train",
    "y_validation",
    "y_test",
]:
    if BENCHMARK_DATA[split_name] is None:
        raise RuntimeError(
            f"BenchmarkData contiene '{split_name}' con valor None."
        )

# Validar dimensiones train
if BENCHMARK_DATA["x_train"].shape[0] != BENCHMARK_DATA["y_train"].shape[0]:
    raise RuntimeError(
        "Las dimensiones de x_train e y_train no son compatibles."
    )

# Validar dimensiones validation
if BENCHMARK_DATA["x_validation"].shape[0] != BENCHMARK_DATA["y_validation"].shape[0]:
    raise RuntimeError(
        "Las dimensiones de x_validation e y_validation no son compatibles."
    )

# Validar dimensiones test
if BENCHMARK_DATA["x_test"].shape[0] != BENCHMARK_DATA["y_test"].shape[0]:
    raise RuntimeError(
        "Las dimensiones de x_test e y_test no son compatibles."
    )

# Verificación final de tipos
print(type(BENCHMARK_DATA["train_index"])) # Verificar índice de entrenamiento
print(type(BENCHMARK_DATA["validation_index"])) # Verificar índice de validación
print(type(BENCHMARK_DATA["test_index"])) # Verificar índice de prueba

print(type(BENCHMARK_DATA["partitions"]["train_index"])) # Verificar partición de entrenamiento
print(type(BENCHMARK_DATA["partitions"]["validation_index"])) # Verificar partición de validación
print(type(BENCHMARK_DATA["partitions"]["test_index"])) # Verificar partición de prueba

print(f"GraphData                   : {len(BENCHMARK_DATA['graphs'])}") # Mostrar GraphData incorporados
print(f"Entrenamiento               : {len(BENCHMARK_DATA['train_index'])}") # Mostrar partición de entrenamiento
print(f"Validación                  : {len(BENCHMARK_DATA['validation_index'])}") # Mostrar partición de validación
print(f"Prueba                      : {len(BENCHMARK_DATA['test_index'])}") # Mostrar partición de prueba
print(f"Variables de entrenamiento : {BENCHMARK_DATA['x_train'].shape}") # Mostrar dimensiones de entrenamiento
print(f"Variables de validación    : {BENCHMARK_DATA['x_validation'].shape}") # Mostrar dimensiones de validación
print(f"Variables de prueba        : {BENCHMARK_DATA['x_test'].shape}") # Mostrar dimensiones de prueba
print("Contrato BenchmarkData      : VALIDADO") # Confirmar contrato
print("Integridad de particiones   : VALIDADA") # Confirmar particiones
print("Escalador oficial           : CONSERVADO") # Confirmar escalador
print("BenchmarkData               : CONSTRUIDO CORRECTAMENTE") # Confirmar construcción

# 4.6 Persistencia de BenchmarkData
print("\n4.6 PERSISTENCIA DE BenchmarkData")

BENCHMARK_DIR.mkdir(
    parents=True,
    exist_ok=True
) # Crear directorio oficial del Benchmark

BENCHMARK_DATA_FILE = BENCHMARK_DIR / "benchmark_data.pkl" # Definir archivo oficial de BenchmarkData

joblib.dump(
    BENCHMARK_DATA,
    BENCHMARK_DATA_FILE
) # Persistir BenchmarkData

if not BENCHMARK_DATA_FILE.exists():
    raise RuntimeError(
        "No fue posible persistir BenchmarkData."
    )

if BENCHMARK_DATA_FILE.stat().st_size <= 0:
    raise RuntimeError(
        "El archivo BenchmarkData está vacío."
    )

BENCHMARK_DATA_LOADED = joblib.load(
    BENCHMARK_DATA_FILE
) # Recuperar BenchmarkData persistido para validar la serialización

if not isinstance(
    BENCHMARK_DATA_LOADED,
    dict
):
    raise RuntimeError(
        "El BenchmarkData persistido no es un diccionario válido."
    )

required_persisted_keys = [
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
] # Definir contrato mínimo del producto persistido

missing_persisted_keys = [
    key
    for key in required_persisted_keys
    if key not in BENCHMARK_DATA_LOADED
]

if missing_persisted_keys:
    raise RuntimeError(
        "El BenchmarkData persistido está incompleto: "
        f"{missing_persisted_keys}"
    )

if len(BENCHMARK_DATA_LOADED["graphs"]) != len(
    BENCHMARK_DATA["graphs"]
):
    raise RuntimeError(
        "La cantidad de GraphData persistidos no coincide "
        "con el producto original."
    )

if len(BENCHMARK_DATA_LOADED["train_index"]) != len(
    BENCHMARK_DATA["train_index"]
):
    raise RuntimeError(
        "La partición de entrenamiento persistida no coincide "
        "con el producto original."
    )

if len(BENCHMARK_DATA_LOADED["validation_index"]) != len(
    BENCHMARK_DATA["validation_index"]
):
    raise RuntimeError(
        "La partición de validación persistida no coincide "
        "con el producto original."
    )

if len(BENCHMARK_DATA_LOADED["test_index"]) != len(
    BENCHMARK_DATA["test_index"]
):
    raise RuntimeError(
        "La partición de prueba persistida no coincide "
        "con el producto original."
    )

print(f"BenchmarkData exportado    : {BENCHMARK_DATA_FILE.name}") # Mostrar archivo persistido
print(f"Tamaño del archivo         : {BENCHMARK_DATA_FILE.stat().st_size} bytes") # Mostrar tamaño
print("Persistencia               : VALIDADA") # Confirmar persistencia
print("Recuperación               : VALIDADA") # Confirmar lectura posterior

# BLOQUE 5. EVALUACIÓN DEL BENCHMARK CIENTÍFICO
# Objetivo: Validar científicamente el desempeño de las diferentes familias de modelos utilizando la 
# Colección Oficial BenchmarkData bajo un protocolo experimental común, reproducible y comparable.
# Producto: - BENCHMARK_INPUTS - Entradas oficiales para la evaluación del Benchmark Científico.
# Pregunta científica: ¿La Colección Oficial BenchmarkData y la configuración experimental cumplen las
# condiciones necesarias para ejecutar una evaluación científica común y reproducible?

print("\n" + "-" * 80)
print("BLOQUE 5. EVALUACIÓN DEL BENCHMARK CIENTÍFICO")
print("-" * 80)

# 5.1 Validación de la Colección Oficial BenchmarkData
print("5.1 VALIDACIÓN DE LA COLECCIÓN OFICIAL BenchmarkData")

if not isinstance(BENCHMARK_DATA, dict):
    raise TypeError("La Colección Oficial BenchmarkData debe ser un diccionario.")

required_products = [
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
] # Definir contrato completo de BenchmarkData

missing_products = [
    product
    for product in required_products
    if product not in BENCHMARK_DATA
]

if missing_products:
    raise ValueError(f"La Colección Oficial BenchmarkData está incompleta: {missing_products}")

# 5.1.1 Recuperar productos oficiales
graphs = BENCHMARK_DATA["graphs"] # Recuperar GraphData oficiales
partitions = BENCHMARK_DATA["partitions"] # Recuperar particiones oficiales
train_index = BENCHMARK_DATA["train_index"] # Recuperar índices de entrenamiento
validation_index = BENCHMARK_DATA["validation_index"] # Recuperar índices de validación
test_index = BENCHMARK_DATA["test_index"] # Recuperar índices de prueba
x_train = BENCHMARK_DATA["x_train"] # Recuperar variables de entrenamiento
y_train = BENCHMARK_DATA["y_train"] # Recuperar objetivos de entrenamiento
x_validation = BENCHMARK_DATA["x_validation"] # Recuperar variables de validación
y_validation = BENCHMARK_DATA["y_validation"] # Recuperar objetivos de validación
x_test = BENCHMARK_DATA["x_test"] # Recuperar variables de prueba
y_test = BENCHMARK_DATA["y_test"] # Recuperar objetivos de prueba
scaler = BENCHMARK_DATA["scaler"] # Recuperar escalador oficial

# 5.1.2 Validar estructura de GraphData
if not isinstance(graphs, (list, tuple)):
    raise TypeError("La colección oficial GraphData debe ser una lista o tupla.")

if len(graphs) == 0:
    raise ValueError("La Colección Oficial BenchmarkData no contiene GraphData.")

# 5.1.3 Validar particiones
if not isinstance(partitions, dict):
    raise TypeError("Las particiones oficiales deben almacenarse en un diccionario.")

required_partitions = [
    "train_index",
    "validation_index",
    "test_index",
]

missing_partitions = [
    key
    for key in required_partitions
    if key not in partitions
]

if missing_partitions:
    raise ValueError(f"Las particiones oficiales están incompletas: {missing_partitions}")

if len(train_index) == 0:
    raise ValueError("El conjunto de entrenamiento está vacío.")

if len(validation_index) == 0:
    raise ValueError("El conjunto de validación está vacío.")

if len(test_index) == 0:
    raise ValueError("El conjunto de prueba está vacío.")

# 5.1.4 Verificar coherencia entre índices directos y particiones
if not np.array_equal(
    train_index,
    np.asarray(partitions["train_index"])
):
    raise ValueError(
        "Los índices de entrenamiento de BenchmarkData son inconsistentes."
    )

if not np.array_equal(
    validation_index,
    np.asarray(partitions["validation_index"])
):
    raise ValueError(
        "Los índices de validación de BenchmarkData son inconsistentes."
    )

if not np.array_equal(
    test_index,
    np.asarray(partitions["test_index"])
):
    raise ValueError(
        "Los índices de prueba de BenchmarkData son inconsistentes."
    )

# 5.1.5 Validar cobertura temporal
all_indices = set(train_index) | set(validation_index) | set(test_index) # Consolidar índices temporales
expected_indices = set(range(len(graphs))) # Definir índices temporales esperados

if all_indices != expected_indices:
    raise ValueError("Las particiones no cubren exactamente la colección oficial GraphData.")

if set(train_index) & set(validation_index):
    raise ValueError("Existe solapamiento entre entrenamiento y validación.")

if set(train_index) & set(test_index):
    raise ValueError("Existe solapamiento entre entrenamiento y prueba.")

if set(validation_index) & set(test_index):
    raise ValueError("Existe solapamiento entre validación y prueba.")

# 5.1.6 Validar matrices experimentales
for x_name, y_name in [
    ("x_train", "y_train"),
    ("x_validation", "y_validation"),
    ("x_test", "y_test"),
]:
    x_data = BENCHMARK_DATA[x_name]
    y_data = BENCHMARK_DATA[y_name]

    if not isinstance(x_data, np.ndarray):
        raise TypeError(f"'{x_name}' debe ser un arreglo NumPy.")

    if not isinstance(y_data, np.ndarray):
        raise TypeError(f"'{y_name}' debe ser un arreglo NumPy.")

    if x_data.ndim != 2:
        raise ValueError(f"'{x_name}' debe ser una matriz bidimensional.")

    if y_data.ndim != 1:
        raise ValueError(f"'{y_name}' debe ser un vector unidimensional.")

    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError(f"'{x_name}' y '{y_name}' presentan diferente número de observaciones.")

# 5.1.7 Validar escalador
if scaler is None:
    raise ValueError("El escalador oficial de BenchmarkData es inválido.")

if not hasattr(scaler, "n_features_in_"):
    raise ValueError("El escalador oficial no contiene información sobre las variables ajustadas.")

if scaler.n_features_in_ != x_train.shape[1]:
    raise ValueError("El escalador oficial presenta una dimensión incompatible con las variables predictoras.")

print(f"GraphData                  : {len(graphs)}")
print(f"Entrenamiento              : {len(train_index)}")
print(f"Validación                 : {len(validation_index)}")
print(f"Prueba                     : {len(test_index)}")
print(f"Variables predictoras      : {x_train.shape[1]}")
print(f"Observaciones train        : {x_train.shape[0]}")
print(f"Observaciones validation   : {x_validation.shape[0]}")
print(f"Observaciones test         : {x_test.shape[0]}")
print(f"Escalador                  : {type(scaler).__name__}")
print("Particiones                : SIN SOLAPAMIENTO")
print("Cobertura temporal         : COMPLETA")
print("Integridad BenchmarkData   : VALIDADA")

print("Colección Oficial BenchmarkData lista para la evaluación científica.")

# 5.2 Validación de la configuración del Benchmark
print("\n5.2 VALIDACIÓN DE LA CONFIGURACIÓN DEL BENCHMARK")

if not isinstance(BENCHMARK_CONFIG, dict):
    raise TypeError(
        "'BENCHMARK_CONFIG' debe ser un diccionario."
    )

required_config = [
    "train_size",
    "validation_size",
    "test_size",
    "ranking_metric",
    "metric_directions",
    "random_state",
] # Definir parámetros científicos obligatorios

missing_config = [
    key
    for key in required_config
    if key not in BENCHMARK_CONFIG
]

if missing_config:
    raise ValueError(
        f"BENCHMARK_CONFIG está incompleto: {missing_config}"
    )

if BENCHMARK_CONFIG["random_state"] != PROJECT_SEED:
    raise ValueError(
        "La semilla de BENCHMARK_CONFIG no coincide con PROJECT_SEED."
    )

ranking_metric = BENCHMARK_CONFIG["ranking_metric"] # Recuperar métrica principal oficial
metric_directions = BENCHMARK_CONFIG["metric_directions"] # Recuperar direcciones oficiales

if not isinstance(
    ranking_metric,
    str
):
    raise TypeError(
        "'ranking_metric' debe ser una cadena."
    )

if not ranking_metric.strip():
    raise ValueError(
        "'ranking_metric' no puede estar vacío."
    )

if ranking_metric not in BENCHMARK_METRICS:
    raise ValueError(
        f"La métrica principal '{ranking_metric}' "
        "no pertenece a BENCHMARK_METRICS."
    )

if not isinstance(
    metric_directions,
    dict
):
    raise TypeError(
        "'metric_directions' debe ser un diccionario."
    )

missing_metric_directions = [
    metric
    for metric in BENCHMARK_METRICS
    if metric not in metric_directions
]

if missing_metric_directions:
    raise ValueError(
        "Faltan direcciones de optimización para las métricas oficiales: "
        f"{missing_metric_directions}"
    )

invalid_metric_directions = [
    metric
    for metric in BENCHMARK_METRICS
    if metric_directions[metric] not in {"min", "max"}
]

if invalid_metric_directions:
    raise ValueError(
        "Existen direcciones de optimización inválidas para: "
        f"{invalid_metric_directions}"
    )

ranking_direction = metric_directions[ranking_metric] # Recuperar dirección de la métrica principal

if ranking_direction not in {"min", "max"}:
    raise ValueError(
        f"La dirección de la métrica '{ranking_metric}' "
        "debe ser 'min' o 'max'."
    )

print(f"Métrica de ranking        : {ranking_metric}") # Mostrar métrica principal
print(f"Dirección de ranking      : {ranking_direction}") # Mostrar dirección de optimización
print(f"Métricas oficiales        : {', '.join(BENCHMARK_METRICS)}") # Mostrar métricas oficiales
print(f"Semilla Benchmark         : {BENCHMARK_CONFIG['random_state']}") # Mostrar semilla oficial
print("Direcciones métricas      : VALIDADAS") # Confirmar direcciones
print("Configuración Benchmark   : VALIDADA") # Confirmar configuración

# 5.3 Confirmación del conjunto experimental
print("5.3 CONFIRMACIÓN DEL CONJUNTO EXPERIMENTAL")

BENCHMARK_INPUTS = {
    "graphs": graphs,
    "partitions": partitions,
    "train_index": train_index,
    "validation_index": validation_index,
    "test_index": test_index,
    "x_train": x_train,
    "y_train": y_train,
    "x_validation": x_validation,
    "y_validation": y_validation,
    "x_test": x_test,
    "y_test": y_test,
    "scaler": scaler,
    "config": BENCHMARK_CONFIG,
    "device": DEVICE,
} # Construir entradas oficiales para la evaluación

required_inputs = [
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
    "config",
    "device",
] # Definir contrato de entradas experimentales

missing_inputs = [
    key
    for key in required_inputs
    if key not in BENCHMARK_INPUTS
]

if missing_inputs:
    raise RuntimeError(f"BENCHMARK_INPUTS está incompleto: {missing_inputs}")

print(f"GraphData                  : {len(BENCHMARK_INPUTS['graphs'])}")
print(f"Dispositivo                : {BENCHMARK_INPUTS['device'].upper()}")
print(f"Métrica oficial            : {BENCHMARK_INPUTS['config']['ranking_metric']}")
print(f"Variables predictoras      : {BENCHMARK_INPUTS['x_train'].shape[1]}")
print("Conjunto experimental       : LISTO")
print("Estado                      : VALIDADO")

print("-" * 80)
print("BLOQUE 5. PREPARACIÓN DEL BENCHMARK COMPLETADA")
print("-" * 80)

# BLOQUE 6. VALIDACIÓN DE LA CONFIGURACIÓN OFICIAL DE LOS MODELOS DEL BENCHMARK
# Objetivo: Validar la configuración oficial de las familias y modelos candidatos que participarán en 
# el Benchmark Científico, verificando su disponibilidad, integridad y consistencia antes de iniciar la evaluación experimental.
# Producto: - CANDIDATE_MODELS - Configuración Oficial de los Modelos del Benchmark validada.
# Pregunta científica: ¿La configuración oficial de las familias y modelos candidatos del Benchmark
# es consistente, íntegra y suficiente para ejecutar el Benchmark Científico?

print("\n" + "-" * 80)
print("BLOQUE 6. VALIDACIÓN DE LOS MODELOS DEL BENCHMARK")
print("-" * 80)

# 6.1 Verificación de la configuración oficial de los modelos
print("6.1 VERIFICACIÓN DE LA CONFIGURACIÓN OFICIAL DE LOS MODELOS")

if not isinstance(BENCHMARK_MODELS, dict):
    raise TypeError("La configuración oficial de los modelos debe ser un diccionario.")

if len(BENCHMARK_MODELS) == 0:
    raise ValueError("No existen familias de modelos registradas.")

normalized_families = [
    family.strip().lower()
    for family in BENCHMARK_MODELS
    if isinstance(family, str)
]

if len(normalized_families) != len(BENCHMARK_MODELS):
    raise TypeError("Todas las familias de modelos deben ser cadenas.")

if len(normalized_families) != len(set(normalized_families)):
    raise ValueError("Existen familias duplicadas por diferencia de mayúsculas o espacios.")

print("Configuración oficial disponible.")

# 6.1.1 Validar familias
for family, models in BENCHMARK_MODELS.items():
    if not isinstance(family, str):
        raise TypeError("Los nombres de familia deben ser cadenas.")

    if not family.strip():
        raise ValueError("Existe una familia con nombre vacío.")

    if not isinstance(models, (list, tuple)):
        raise TypeError(f"La familia '{family}' debe contener una lista o tupla de modelos.")

    if len(models) == 0:
        raise ValueError(f"La familia '{family}' no contiene modelos.")

# 6.1.2 Validar modelos
for family, models in BENCHMARK_MODELS.items():
    for model in models:
        if not isinstance(model, str):
            raise TypeError(f"El modelo registrado en la familia '{family}' debe ser una cadena.")

        if not model.strip():
            raise ValueError(f"La familia '{family}' contiene un nombre de modelo vacío.")

# 6.1.3 Validar duplicados dentro de cada familia
for family, models in BENCHMARK_MODELS.items():
    normalized_models = [
        model.strip().lower()
        for model in models
    ]

    duplicates = sorted({
        model
        for model in normalized_models
        if normalized_models.count(model) > 1
    })

    if duplicates:
        raise ValueError(f"La familia '{family}' contiene modelos duplicados: {duplicates}")

# 6.1.4 Validar duplicados entre familias
model_locations = {}

for family, models in BENCHMARK_MODELS.items():
    for model in models:
        normalized_model = model.strip().lower()
        model_locations.setdefault(normalized_model, []).append(family)

cross_family_duplicates = {
    model: families
    for model, families in model_locations.items()
    if len(families) > 1
}

if cross_family_duplicates:
    raise ValueError(f"Existen modelos registrados en más de una familia: {cross_family_duplicates}")

# 6.2 Construcción de CANDIDATE_MODELS
print("6.2 CONSTRUCCIÓN DE LA COLECCIÓN DE MODELOS CANDIDATOS")

print("\nRECONSTRUCCIÓN DE candidate_models")
candidate_models = []
for family, models in BENCHMARK_MODELS.items():
    for model_name in models:
        matching_codes = [
            model_code
            for model_code, configured_model_name
            in BENCHMARK_MODEL_CODES.items()
            if configured_model_name.strip().lower()
            == model_name.strip().lower()
        ] # Buscar código oficial asociado al modelo

        if len(matching_codes) != 1:
            raise RuntimeError(
                f"No existe una correspondencia única entre "
                f"'{model_name}' y BENCHMARK_MODEL_CODES."
            ) # Validar correspondencia única

        model_code = matching_codes[0] # Recuperar código oficial

        candidate_models.append(
            {
                "model_code": model_code,
                "model_name": model_name,
                "family": family,
            }
        ) # Construir identidad completa del candidato

print(
    f"Modelos construidos : {len(candidate_models)}"
) # Mostrar cantidad

for candidate in candidate_models:
    print(
        f"{candidate['model_code']} | "
        f"{candidate['model_name']} | "
        f"{candidate['family']}"
    ) # Mostrar identidad completa

# 6.3 Validación de CANDIDATE_MODELS
print("\n6.3 VALIDACIÓN DE LOS MODELOS CANDIDATOS")

if not isinstance(
    candidate_models,
    list
):
    raise TypeError(
        "'candidate_models' debe ser una lista."
    )

if len(candidate_models) == 0:
    raise ValueError(
        "La colección de modelos candidatos está vacía."
    )

configured_pairs = {
    (
        family.strip().lower(),
        model_name.strip().lower(),
    )
    for family, models in BENCHMARK_MODELS.items()
    for model_name in models
} # Construir identificadores oficiales desde BENCHMARK_MODELS

candidate_pairs = []

for candidate in candidate_models:

    required_keys = [
        "family",
        "model_name",
    ] # Definir contrato mínimo del candidato

    missing_keys = [
        key
        for key in required_keys
        if key not in candidate
    ]

    if missing_keys:
        raise RuntimeError(
            f"Modelo candidato incompleto: {missing_keys}"
        )

    if not isinstance(
        candidate["family"],
        str
    ) or not candidate["family"].strip():
        raise ValueError(
            "Un modelo candidato no tiene una familia válida."
        )

    if not isinstance(
        candidate["model_name"],
        str
    ) or not candidate["model_name"].strip():
        raise ValueError(
            "Un modelo candidato no tiene un nombre válido."
        )

    candidate_pair = (
        candidate["family"].strip().lower(),
        candidate["model_name"].strip().lower(),
    ) # Construir identificador normalizado del candidato

    candidate_pairs.append(
        candidate_pair
    )

    if candidate_pair not in configured_pairs:
        raise ValueError(
            f"El modelo candidato '{candidate['model_name']}' "
            f"de la familia '{candidate['family']}' "
            "no está registrado en BENCHMARK_MODELS."
        )

if len(candidate_pairs) != len(
    configured_pairs
):
    raise RuntimeError(
        "La cantidad de modelos candidatos no coincide "
        "con la cantidad de modelos configurados."
    )

if set(candidate_pairs) != configured_pairs:
    raise RuntimeError(
        "Los modelos candidatos no coinciden exactamente "
        "con BENCHMARK_MODELS."
    )

if len(candidate_pairs) != len(
    set(candidate_pairs)
):
    raise RuntimeError(
        "Existen modelos candidatos duplicados."
    )

print(f"Modelos configurados       : {len(configured_pairs)}") # Mostrar modelos del catálogo
print(f"Modelos candidatos         : {len(candidate_models)}") # Mostrar modelos candidatos
print("Correspondencia catálogo   : VALIDADA") # Confirmar correspondencia
print("Duplicados                 : NO DETECTADOS") # Confirmar unicidad
print("CANDIDATE_MODELS           : VALIDADO") # Confirmar producto

# 6.4 Validación del Modelo Oficial
print("\n6.4 VALIDACIÓN DEL MODELO OFICIAL")

if not isinstance(
    OFFICIAL_MODEL_CODE,
    str
):
    raise TypeError(
        "OFFICIAL_MODEL_CODE debe ser una cadena."
    )

if not OFFICIAL_MODEL_CODE.strip():
    raise ValueError(
        "OFFICIAL_MODEL_CODE está vacío."
    )

if not isinstance(
    OFFICIAL_MODEL_NAME,
    str
):
    raise TypeError(
        "OFFICIAL_MODEL_NAME debe ser una cadena."
    )

if not OFFICIAL_MODEL_NAME.strip():
    raise ValueError(
        "OFFICIAL_MODEL_NAME está vacío."
    )

if not isinstance(
    OFFICIAL_MODEL_FAMILY,
    str
):
    raise TypeError(
        "OFFICIAL_MODEL_FAMILY debe ser una cadena."
    )

if not OFFICIAL_MODEL_FAMILY.strip():
    raise ValueError(
        "OFFICIAL_MODEL_FAMILY está vacía."
    )

official_model_candidates = [
    candidate
    for candidate in candidate_models
    if (
        candidate["model_name"].strip().lower()
        == OFFICIAL_MODEL_NAME.strip().lower()
        and
        candidate["family"].strip().lower()
        == OFFICIAL_MODEL_FAMILY.strip().lower()
    )
] # Buscar el Modelo Oficial por nombre y familia

if len(official_model_candidates) == 0:
    raise ValueError(
        f"El Modelo Oficial '{OFFICIAL_MODEL_NAME}' "
        f"no está registrado en la familia oficial "
        f"'{OFFICIAL_MODEL_FAMILY}'."
    )

if len(official_model_candidates) > 1:
    raise ValueError(
        f"El Modelo Oficial '{OFFICIAL_MODEL_NAME}' "
        "aparece más de una vez en la familia oficial."
    )

official_candidate = official_model_candidates[0] # Recuperar candidato oficial
candidate_model_name = official_candidate["model_name"].strip().lower() # Recuperar nombre oficial del candidato
matching_official_codes = [
    model_code
    for model_code, configured_model_name in BENCHMARK_MODEL_CODES.items()
    if configured_model_name.strip().lower() == candidate_model_name
] # Buscar código oficial asociado al nombre

if len(matching_official_codes) != 1:
    raise ValueError(
        f"No existe una correspondencia única entre el Modelo Oficial "
        f"'{official_candidate['model_name']}' y BENCHMARK_MODEL_CODES."
    )

candidate_model_code = matching_official_codes[0] # Recuperar código oficial

if candidate_model_code != OFFICIAL_MODEL_CODE:
    raise ValueError(
        f"OFFICIAL_MODEL_CODE '{OFFICIAL_MODEL_CODE}' "
        f"no coincide con el código registrado "
        f"'{candidate_model_code}'."
    )

if official_candidate["model_name"].strip().lower() != OFFICIAL_MODEL_NAME.strip().lower():
    raise ValueError(
        "El nombre del Modelo Oficial no coincide con "
        "OFFICIAL_MODEL_NAME."
    )

if official_candidate["family"].strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise ValueError(
        "La familia del Modelo Oficial no coincide con "
        "OFFICIAL_MODEL_FAMILY."
    )

print(f"Código Oficial            : {OFFICIAL_MODEL_CODE}") # Mostrar código oficial
print(f"Modelo Oficial            : {OFFICIAL_MODEL_NAME}") # Mostrar modelo oficial
print(f"Familia Oficial           : {OFFICIAL_MODEL_FAMILY}") # Mostrar familia oficial
print("Código Oficial             : VALIDADO") # Confirmar código
print("Modelo Oficial             : VALIDADO") # Confirmar modelo
print("Familia Oficial            : VALIDADA") # Confirmar familia
print("Identidad Oficial          : CONSISTENTE") # Confirmar identidad completa

# 6.5 Construcción del producto oficial
CANDIDATE_MODELS = {
    "families": BENCHMARK_MODELS,
    "models": candidate_models,
    "total_families": len(BENCHMARK_MODELS),
    "total_models": len(candidate_models),
    "official_model": {
        "model_code": OFFICIAL_MODEL_CODE,
        "model_name": OFFICIAL_MODEL_NAME,
        "family": OFFICIAL_MODEL_FAMILY,
    },
} # Reconstruir producto oficial de modelos candidatos

print("\nAUDITORÍA CANDIDATE_MODELS")

print(f"Total familias : {CANDIDATE_MODELS['total_families']}") # Mostrar familias
print(f"Total modelos  : {CANDIDATE_MODELS['total_models']}") # Mostrar modelos

for candidate in CANDIDATE_MODELS["models"]:
    print(
        f"{candidate['model_code']} | "
        f"{candidate['model_name']} | "
        f"{candidate['family']}"
    ) # Mostrar catálogo oficial

print(
    "\nCANDIDATE_MODELS : RECONSTRUIDO"
) # Confirmar reconstrucción

# 6.6 Validación final del producto
print("\n6.6 VALIDACIÓN FINAL DEL PRODUCTO OFICIAL")

required_candidate_products = [
    "families",
    "models",
    "total_families",
    "total_models",
    "official_model",
] # Definir contrato obligatorio de CANDIDATE_MODELS

missing_candidate_products = [
    key
    for key in required_candidate_products
    if key not in CANDIDATE_MODELS
]

if missing_candidate_products:
    raise RuntimeError(
        "CANDIDATE_MODELS está incompleto: "
        f"{missing_candidate_products}"
    )

if not isinstance(
    CANDIDATE_MODELS["families"],
    dict
):
    raise TypeError(
        "CANDIDATE_MODELS['families'] debe ser un diccionario."
    )

if not isinstance(
    CANDIDATE_MODELS["models"],
    list
):
    raise TypeError(
        "CANDIDATE_MODELS['models'] debe ser una lista."
    )

if CANDIDATE_MODELS["total_families"] <= 0:
    raise RuntimeError(
        "CANDIDATE_MODELS no contiene familias."
    )

if CANDIDATE_MODELS["total_models"] <= 0:
    raise RuntimeError(
        "CANDIDATE_MODELS no contiene modelos."
    )

if CANDIDATE_MODELS["total_families"] != len(
    CANDIDATE_MODELS["families"]
):
    raise RuntimeError(
        "El total de familias no coincide con el catálogo oficial."
    )

if CANDIDATE_MODELS["total_models"] != len(
    CANDIDATE_MODELS["models"]
):
    raise RuntimeError(
        "El total de modelos no coincide con la colección de candidatos."
    )

official_model = CANDIDATE_MODELS[
    "official_model"
] # Recuperar identidad oficial del modelo

if not isinstance(
    official_model,
    dict
):
    raise TypeError(
        "CANDIDATE_MODELS['official_model'] debe ser un diccionario."
    )

required_official_keys = [
    "model_code",
    "model_name",
    "family",
] # Definir identidad obligatoria del Modelo Oficial

missing_official_keys = [
    key
    for key in required_official_keys
    if key not in official_model
]

if missing_official_keys:
    raise RuntimeError(
        "La identidad del Modelo Oficial está incompleta: "
        f"{missing_official_keys}"
    )

if official_model["model_code"] != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El código del Modelo Oficial no coincide con "
        "OFFICIAL_MODEL_CODE."
    )

if official_model["model_name"] != OFFICIAL_MODEL_NAME:
    raise RuntimeError(
        "El nombre del Modelo Oficial no coincide con "
        "OFFICIAL_MODEL_NAME."
    )

if official_model["family"] != OFFICIAL_MODEL_FAMILY:
    raise RuntimeError(
        "La familia del Modelo Oficial no coincide con "
        "OFFICIAL_MODEL_FAMILY."
    )

official_candidates = [
    candidate
    for candidate in CANDIDATE_MODELS["models"]
    if (
        candidate["model_name"].strip().lower()
        == OFFICIAL_MODEL_NAME.strip().lower()
        and
        candidate["family"].strip().lower()
        == OFFICIAL_MODEL_FAMILY.strip().lower()
    )
] # Buscar el Modelo Oficial por nombre y familia

if len(official_candidates) != 1:
    raise RuntimeError(
        "El Modelo Oficial debe aparecer exactamente una vez "
        "por nombre y familia en la colección de candidatos."
    )

official_candidate = official_candidates[0] # Recuperar candidato oficial
matching_official_codes = [
    model_code
    for model_code, configured_model_name in BENCHMARK_MODEL_CODES.items()
    if configured_model_name.strip().lower()
    == official_candidate["model_name"].strip().lower()
] # Buscar código oficial asociado al candidato

if len(matching_official_codes) != 1:
    raise RuntimeError(
        f"No existe una correspondencia única entre el modelo "
        f"'{official_candidate['model_name']}' y BENCHMARK_MODEL_CODES."
    )

candidate_model_code = matching_official_codes[0] # Recuperar código oficial
if candidate_model_code != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        f"El código del candidato oficial '{candidate_model_code}' "
        f"no coincide con OFFICIAL_MODEL_CODE '{OFFICIAL_MODEL_CODE}'."
    )

print(f"Total de familias          : {CANDIDATE_MODELS['total_families']}") # Mostrar familias
print(f"Total de modelos           : {CANDIDATE_MODELS['total_models']}") # Mostrar modelos
print(f"Código Modelo Oficial      : {official_model['model_code']}") # Mostrar código oficial
print(f"Modelo Oficial             : {official_model['model_name']}") # Mostrar nombre oficial
print(f"Familia Oficial            : {official_model['family']}") # Mostrar familia oficial
print("Conteos                    : VALIDADOS") # Confirmar consistencia de conteos
print("Identidad Modelo Oficial   : VALIDADA") # Confirmar identidad oficial
print("Correspondencia catálogo   : VALIDADA") # Confirmar presencia única
print("CANDIDATE_MODELS           : VALIDADO") # Confirmar producto final

# 6.7 Registro de la configuración oficial
print("6.7 CONFIGURACIÓN OFICIAL DE LOS MODELOS")

for family, models in BENCHMARK_MODELS.items():
    print(f"{family:<25}: {len(models)} modelo(s)")

print(f"Total de familias          : {CANDIDATE_MODELS['total_families']}")
print(f"Total de modelos           : {CANDIDATE_MODELS['total_models']}")
print(f"Modelo Oficial             : {CANDIDATE_MODELS['official_model']['model_name']}")
print(f"Familia Oficial            : {CANDIDATE_MODELS['official_model']['family']}")
print("Configuración de modelos   : VALIDADA")

# 6.8 Confirmación del bloque
print("-" * 80)
print("BLOQUE 6. MODELOS CANDIDATOS VALIDADOS CORRECTAMENTE")
print("-" * 80)

# BLOQUE 7. EJECUCIÓN DEL BENCHMARK CIENTÍFICO
# Objetivo: Ejecutar los modelos candidatos definidos en CANDIDATE_MODELS utilizando la Colección
# Oficial BenchmarkData y el protocolo experimental establecido para el Benchmark Científico.
# Producto: - BENCHMARK_RESULTS - Resultados individuales por modelo y familia.
# Pregunta científica: ¿Cuál es el desempeño predictivo de cada modelo candidato bajo un protocolo
# experimental común, reproducible y científicamente comparable?

print("\n" + "-" * 80)
print("BLOQUE 7. EJECUCIÓN DEL BENCHMARK CIENTÍFICO")
print("-" * 80)

# 7.1 Validación de las entradas oficiales del Benchmark
print(type(BENCHMARK_DATA["train_index"])) # Verificar índice de entrenamiento
print(type(BENCHMARK_DATA["validation_index"])) # Verificar índice de validación
print(type(BENCHMARK_DATA["test_index"])) # Verificar índice de prueba

print(type(BENCHMARK_DATA["partitions"]["train_index"])) # Verificar partición de entrenamiento
print(type(BENCHMARK_DATA["partitions"]["validation_index"])) # Verificar partición de validación
print(type(BENCHMARK_DATA["partitions"]["test_index"])) # Verificar partición de prueba

print("\n7.1 VALIDACIÓN DE LAS ENTRADAS OFICIALES")

if not isinstance(BENCHMARK_METRICS, (list, tuple)):
    raise TypeError("BENCHMARK_METRICS debe ser una lista o tupla.")

if len(BENCHMARK_METRICS) == 0:
    raise ValueError("BENCHMARK_METRICS no contiene métricas oficiales.")

required_metrics = {
    "rmse",
    "mae",
    "mape",
    "r2",
} # Definir métricas oficiales obligatorias

missing_metrics = [
    metric
    for metric in required_metrics
    if metric not in BENCHMARK_METRICS
]

if missing_metrics:
    raise ValueError(
        f"Faltan métricas oficiales requeridas: {missing_metrics}"
    )

# Validar CANDIDATE_MODELS
if not isinstance(CANDIDATE_MODELS, dict):
    raise TypeError("CANDIDATE_MODELS debe ser un diccionario.")

required_candidate_keys = [
    "families",
    "models",
    "total_families",
    "total_models",
    "official_model",
] # Definir contrato oficial de CANDIDATE_MODELS

missing_candidate_keys = [
    key
    for key in required_candidate_keys
    if key not in CANDIDATE_MODELS
]

if missing_candidate_keys:
    raise ValueError(
        f"CANDIDATE_MODELS está incompleto: {missing_candidate_keys}"
    )

if not isinstance(CANDIDATE_MODELS["families"], dict):
    raise TypeError(
        "CANDIDATE_MODELS['families'] debe ser un diccionario."
    )

if not isinstance(CANDIDATE_MODELS["models"], list):
    raise TypeError(
        "CANDIDATE_MODELS['models'] debe ser una lista."
    )

if CANDIDATE_MODELS["total_families"] <= 0:
    raise ValueError(
        "CANDIDATE_MODELS no contiene familias."
    )

if CANDIDATE_MODELS["total_models"] <= 0:
    raise ValueError(
        "CANDIDATE_MODELS no contiene modelos candidatos."
    )

if CANDIDATE_MODELS["total_families"] != len(
    CANDIDATE_MODELS["families"]
):
    raise ValueError(
        "El total de familias de CANDIDATE_MODELS "
        "no coincide con el catálogo de familias."
    )

if CANDIDATE_MODELS["total_models"] != len(
    CANDIDATE_MODELS["models"]
):
    raise ValueError(
        "El total de modelos de CANDIDATE_MODELS "
        "no coincide con la colección de modelos."
    )

# Validar Modelo Oficial
official_model = CANDIDATE_MODELS["official_model"] # Recuperar identidad oficial

if not isinstance(official_model, dict):
    raise TypeError(
        "CANDIDATE_MODELS['official_model'] debe ser un diccionario."
    )

required_official_keys = [
    "model_code",
    "model_name",
    "family",
] # Definir contrato del Modelo Oficial

missing_official_keys = [
    key
    for key in required_official_keys
    if key not in official_model
]

if missing_official_keys:
    raise ValueError(
        "El Modelo Oficial está incompleto: "
        f"{missing_official_keys}"
    )

if official_model["model_code"] != OFFICIAL_MODEL_CODE:
    raise ValueError(
        "El código del Modelo Oficial no coincide con "
        "OFFICIAL_MODEL_CODE."
    )

if official_model["model_name"] != OFFICIAL_MODEL_NAME:
    raise ValueError(
        "El nombre del Modelo Oficial no coincide con "
        "OFFICIAL_MODEL_NAME."
    )

if official_model["family"] != OFFICIAL_MODEL_FAMILY:
    raise ValueError(
        "La familia del Modelo Oficial no coincide con "
        "OFFICIAL_MODEL_FAMILY."
    )

# Validar BENCHMARK_DATA oficial
if not isinstance(BENCHMARK_DATA, dict):
    raise TypeError("BENCHMARK_DATA debe ser un diccionario.")

required_benchmark_keys = [
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
] # Definir contrato oficial de BENCHMARK_DATA

missing_benchmark_keys = [
    key
    for key in required_benchmark_keys
    if key not in BENCHMARK_DATA
]

if missing_benchmark_keys:
    raise ValueError(
        f"BENCHMARK_DATA está incompleto: {missing_benchmark_keys}"
    )

# Recuperar productos oficiales
graphs = BENCHMARK_DATA["graphs"] # Recuperar GraphData oficiales
partitions = BENCHMARK_DATA["partitions"] # Recuperar particiones oficiales
train_index = BENCHMARK_DATA["train_index"] # Recuperar índices de entrenamiento
validation_index = BENCHMARK_DATA["validation_index"] # Recuperar índices de validación
test_index = BENCHMARK_DATA["test_index"] # Recuperar índices de prueba
x_train = BENCHMARK_DATA["x_train"] # Recuperar variables de entrenamiento
y_train = BENCHMARK_DATA["y_train"] # Recuperar objetivos de entrenamiento
x_validation = BENCHMARK_DATA["x_validation"] # Recuperar variables de validación
y_validation = BENCHMARK_DATA["y_validation"] # Recuperar objetivos de validación
x_test = BENCHMARK_DATA["x_test"] # Recuperar variables de prueba
y_test = BENCHMARK_DATA["y_test"] # Recuperar objetivos de prueba
scaler = BENCHMARK_DATA["scaler"] # Recuperar escalador oficial

# Validar GraphData
if not isinstance(graphs, (list, tuple)):
    raise TypeError(
        "La colección GraphData debe ser una lista o tupla."
    )

if len(graphs) == 0:
    raise ValueError(
        "La colección GraphData está vacía."
    )

# Validar particiones
if not isinstance(partitions, dict):
    raise TypeError(
        "Las particiones oficiales deben ser un diccionario."
    )

required_partition_keys = [
    "train_index",
    "validation_index",
    "test_index",
] # Definir particiones oficiales obligatorias

missing_partition_keys = [
    key
    for key in required_partition_keys
    if key not in partitions
]

if missing_partition_keys:
    raise ValueError(
        "Las particiones oficiales están incompletas: "
        f"{missing_partition_keys}"
    )

for name, index in (
    ("train_index", train_index),
    ("validation_index", validation_index),
    ("test_index", test_index),
):
    if not isinstance(index, np.ndarray):
        raise TypeError(
            f"'{name}' debe ser un arreglo NumPy."
        )

    if index.ndim != 1:
        raise ValueError(
            f"'{name}' debe ser un vector unidimensional."
        )

    if len(index) == 0:
        raise ValueError(
            f"'{name}' está vacío."
        )

    if not np.issubdtype(index.dtype, np.integer):
        raise TypeError(
            f"'{name}' debe contener únicamente índices enteros."
        )

# Validar coherencia entre índices directos y particiones
if not isinstance(partitions["train_index"], np.ndarray):
    raise TypeError(
        "'partitions['train_index']' debe ser un arreglo NumPy."
    )

if not isinstance(partitions["validation_index"], np.ndarray):
    raise TypeError(
        "'partitions['validation_index']' debe ser un arreglo NumPy."
    )

if not isinstance(partitions["test_index"], np.ndarray):
    raise TypeError(
        "'partitions['test_index']' debe ser un arreglo NumPy."
    )

if not np.array_equal(
    train_index,
    partitions["train_index"]
):
    raise ValueError(
        "Los índices de entrenamiento son inconsistentes con 'partitions'."
    )

if not np.array_equal(
    validation_index,
    partitions["validation_index"]
):
    raise ValueError(
        "Los índices de validación son inconsistentes con 'partitions'."
    )

if not np.array_equal(
    test_index,
    partitions["test_index"]
):
    raise ValueError(
        "Los índices de prueba son inconsistentes con 'partitions'."
    )

print("Entradas oficiales disponibles.")
print(f"GraphData                  : {len(graphs)}")
print(f"Entrenamiento              : {len(train_index)}")
print(f"Validación                 : {len(validation_index)}")
print(f"Prueba                     : {len(test_index)}")
print(f"Variables predictoras      : {x_train.shape[1]}")
print(f"Observaciones train        : {x_train.shape[0]}")
print(f"Observaciones validation   : {x_validation.shape[0]}")
print(f"Observaciones test         : {x_test.shape[0]}")
print(f"Familias configuradas      : {CANDIDATE_MODELS['total_families']}")
print(f"Modelos candidatos         : {CANDIDATE_MODELS['total_models']}")
print(f"Código Modelo Oficial      : {official_model['model_code']}")
print(f"Modelo Oficial             : {official_model['model_name']}")
print(f"Familia Oficial            : {official_model['family']}")
print(f"Escalador                  : {type(scaler).__name__}")
print("Métricas oficiales         : VALIDADAS")
print("Conteos de candidatos      : VALIDADOS")
print("Modelo Oficial             : VALIDADO")
print("BENCHMARK_DATA             : VALIDADO")

# 7.2 Preparación de los resultados del Benchmark
print("\n7.2 PREPARACIÓN DE LOS RESULTADOS")
benchmark_family_results = {
    family: []
    for family in CANDIDATE_MODELS["families"]
} # Inicializar resultados según las familias oficiales

benchmark_results = [] # Inicializar colección global de resultados

# 7.3 Evaluación de modelos estadísticos
print("\n7.3 EVALUACIÓN DE MODELOS ESTADÍSTICOS")
statistical_family = "statistical" # Identificar familia oficial
statistical_models = CANDIDATE_MODELS["families"].get(
    statistical_family,
    []
) # Recuperar modelos estadísticos de la configuración oficial

if not statistical_models:
    raise ValueError(
        "No existen modelos estadísticos registrados en CANDIDATE_MODELS."
    )

if statistical_family not in benchmark_family_results:
    benchmark_family_results[statistical_family] = [] # Inicializar resultados de la familia estadística

for model_name in statistical_models:
    normalized_model_name = model_name.strip().lower().replace(
        " ",
        "_"
    ) # Normalizar nombre para comparación

    matching_model_codes = [
        model_code
        for model_code, configured_model_name in BENCHMARK_MODEL_CODES.items()
        if configured_model_name.strip().lower().replace(" ", "_") == normalized_model_name
    ] # Buscar código oficial correspondiente

    if len(matching_model_codes) != 1:
        raise RuntimeError(
            f"No existe una correspondencia única entre el modelo "
            f"'{model_name}' y BENCHMARK_MODEL_CODES."
        )

    model_code = matching_model_codes[0] # Recuperar código oficial
    if normalized_model_name != "linear_regression":
        raise ValueError(
            f"Modelo estadístico no implementado: '{model_name}'."
        )

    result = run_linear_regression(
        x_train=BENCHMARK_DATA["x_train"],
        y_train=BENCHMARK_DATA["y_train"],
        x_test=BENCHMARK_DATA["x_test"],
        y_test=BENCHMARK_DATA["y_test"],
    ) # Ejecutar modelo estadístico oficial

    if result is None:
        raise RuntimeError(
            f"El modelo '{model_name}' no produjo resultados."
        )

    if not isinstance(result, dict):
        raise TypeError(
            f"El resultado del modelo '{model_name}' debe ser un diccionario."
        )

    result = validate_model_result(
        result=result,
        model_code=model_code,
        model_name=model_name,
        family=statistical_family,
    ) # Validar contrato científico común del resultado

    if result["model_code"] != model_code:
        raise RuntimeError(
            f"El código del resultado '{result['model_code']}' "
            f"no coincide con el código esperado '{model_code}'."
        )

    if result["model_name"] != model_name:
        raise RuntimeError(
            f"El nombre del resultado '{result['model_name']}' "
            f"no coincide con el modelo ejecutado '{model_name}'."
        )

    if result["family"] != statistical_family:
        raise RuntimeError(
            f"La familia del resultado '{result['family']}' "
            f"no coincide con '{statistical_family}'."
        )

    benchmark_family_results[statistical_family].append(
        result
    ) # Registrar resultado por familia

    benchmark_results.append(
        result
    ) # Registrar resultado global

print(f"Modelos estadísticos evaluados : {len(benchmark_family_results[statistical_family])}") # Mostrar cantidad evaluada
print("Contrato estadístico           : VALIDADO") # Confirmar contrato

# 7.4 Evaluación de modelos de Machine Learning
print("\n7.4 EVALUACIÓN DE MODELOS DE MACHINE LEARNING")

machine_learning_family = "machine_learning" # Identificar familia oficial de Machine Learning
machine_learning_models = CANDIDATE_MODELS["families"].get(
    machine_learning_family,
    []
) # Recuperar modelos de Machine Learning de la configuración oficial

if not machine_learning_models:
    raise ValueError(
        "No existen modelos de Machine Learning registrados en CANDIDATE_MODELS."
    )

if machine_learning_family not in benchmark_family_results:
    benchmark_family_results[machine_learning_family] = [] # Inicializar resultados de la familia

for model_name in machine_learning_models:
    normalized_model_name = model_name.strip().lower().replace(
        " ",
        "_"
    ) # Normalizar nombre para comparación

    if model_name not in MACHINE_LEARNING_CONFIG:
        raise KeyError(
            f"No existe configuración de Machine Learning para '{model_name}'."
        )

    model_config = MACHINE_LEARNING_CONFIG[model_name] # Recuperar configuración oficial del modelo
    if not isinstance(model_config, dict):
        raise TypeError(
            f"La configuración de Machine Learning de '{model_name}' "
            "debe ser un diccionario."
        )

    required_model_config_keys = [
        "model_code",
        "model_name",
        "family",
    ] # Definir identidad obligatoria del modelo

    missing_model_config_keys = [
        key
        for key in required_model_config_keys
        if key not in model_config
    ]

    if missing_model_config_keys:
        raise ValueError(
            f"La configuración de Machine Learning de '{model_name}' "
            f"está incompleta: {missing_model_config_keys}"
        )

    configured_model_name = model_config["model_name"].strip().lower().replace(
        " ",
        "_"
    ) # Normalizar nombre de configuración

    if configured_model_name != normalized_model_name:
        raise ValueError(
            f"El nombre de la configuración "
            f"'{model_config['model_name']}' no coincide con "
            f"el modelo candidato '{model_name}'."
        )

    if model_config["family"] != machine_learning_family:
        raise ValueError(
            f"La familia configurada para '{model_name}' "
            f"'{model_config['family']}' no coincide con "
            f"'{machine_learning_family}'."
        )

    if not isinstance(model_config["model_code"], str):
        raise TypeError(
            f"El código del modelo '{model_name}' debe ser una cadena."
        )

    if not model_config["model_code"].strip():
        raise ValueError(
            f"El código del modelo '{model_name}' no puede estar vacío."
        )

    matching_model_codes = [
        model_code
        for model_code, configured_model_name in BENCHMARK_MODEL_CODES.items()
        if configured_model_name.strip().lower().replace(" ", "_")
        == normalized_model_name
    ] # Buscar código oficial correspondiente en el catálogo

    if len(matching_model_codes) != 1:
        raise RuntimeError(
            f"No existe una correspondencia única entre el modelo "
            f"'{model_name}' y BENCHMARK_MODEL_CODES."
        )

    expected_model_code = matching_model_codes[0] # Recuperar código oficial del catálogo
    if model_config["model_code"] != expected_model_code:
        raise ValueError(
            f"El código configurado para '{model_name}' "
            f"'{model_config['model_code']}' no coincide con el "
            f"código oficial del catálogo '{expected_model_code}'."
        )

    result = run_machine_learning(
        model_name=model_name,
        x_train=BENCHMARK_DATA["x_train"],
        y_train=BENCHMARK_DATA["y_train"],
        x_test=BENCHMARK_DATA["x_test"],
        y_test=BENCHMARK_DATA["y_test"],
    ) # Ejecutar modelo de Machine Learning

    if result is None:
        raise RuntimeError(
            f"El modelo '{model_name}' no produjo resultados."
        )

    if not isinstance(result, dict):
        raise TypeError(
            f"El resultado del modelo '{model_name}' "
            "debe ser un diccionario."
        )

    result = validate_model_result(
        result=result,
        model_code=model_config["model_code"],
        model_name=model_config["model_name"],
        family=machine_learning_family,
    ) # Validar contrato científico completo del resultado

    if result["model_code"] != model_config["model_code"]:
        raise RuntimeError(
            f"El código del resultado '{result['model_code']}' "
            f"no coincide con el código configurado "
            f"'{model_config['model_code']}'."
        )

    result_model_name = result["model_name"].strip().lower().replace(
        " ",
        "_"
    ) # Normalizar nombre devuelto por el modelo

    if result_model_name != normalized_model_name:
        raise RuntimeError(
            f"El nombre del resultado '{result['model_name']}' "
            f"no coincide con el modelo ejecutado '{model_name}'."
        )

    if result["family"] != machine_learning_family:
        raise RuntimeError(
            f"La familia del resultado '{result['family']}' "
            f"no coincide con '{machine_learning_family}'."
        )

    benchmark_family_results[machine_learning_family].append(
        result
    ) # Registrar resultado por familia

    benchmark_results.append(
        result
    ) # Registrar resultado global

print(f"Modelos Machine Learning evaluados : {len(benchmark_family_results[machine_learning_family])}") # Mostrar cantidad evaluada
print("Contrato Machine Learning          : VALIDADO") # Confirmar contrato
print("Códigos de modelos ML              : VALIDADOS") # Confirmar identidad

# 7.5 Evaluación de modelos de Deep Learning
print("\n7.5 EVALUACIÓN DE MODELOS DE DEEP LEARNING")

deep_learning_family = "deep_learning" # Identificar familia oficial de Deep Learning
deep_learning_models = CANDIDATE_MODELS["families"].get(
    deep_learning_family,
    []
) # Recuperar modelos de Deep Learning de la configuración oficial

if not deep_learning_models:
    raise ValueError(
        "No existen modelos de Deep Learning registrados en CANDIDATE_MODELS."
    )

if deep_learning_family not in benchmark_family_results:
    benchmark_family_results[deep_learning_family] = [] # Inicializar resultados de la familia

for model_name in deep_learning_models:

    normalized_model_name = model_name.strip().lower().replace(
        " ",
        "_"
    ) # Normalizar identificador del modelo

    matching_model_codes = [
        model_code
        for model_code, configured_model_name in BENCHMARK_MODEL_CODES.items()
        if configured_model_name.strip().lower().replace(" ", "_") == normalized_model_name
    ] # Buscar código oficial correspondiente

    if len(matching_model_codes) != 1:
        raise RuntimeError(
            f"No existe una correspondencia única entre el modelo "
            f"'{model_name}' y BENCHMARK_MODEL_CODES."
        )

    model_code = matching_model_codes[0] # Recuperar código oficial

    if normalized_model_name != "mlp":
        raise ValueError(
            f"Modelo de Deep Learning no implementado: '{model_name}'."
        )

    configured_model_name = DEEP_LEARNING_CONFIG["model_name"].strip().lower().replace(
        " ",
        "_"
    ) # Normalizar nombre de configuración

    if configured_model_name != normalized_model_name:
        raise RuntimeError(
            f"El nombre de DEEP_LEARNING_CONFIG "
            f"'{DEEP_LEARNING_CONFIG['model_name']}' "
            f"no coincide con el modelo candidato '{model_name}'."
        )

    if DEEP_LEARNING_CONFIG["model_code"] != model_code:
        raise RuntimeError(
            f"El código de DEEP_LEARNING_CONFIG "
            f"'{DEEP_LEARNING_CONFIG['model_code']}' "
            f"no coincide con el código oficial '{model_code}'."
        )

    if DEEP_LEARNING_CONFIG["family"] != deep_learning_family:
        raise RuntimeError(
            f"La familia de DEEP_LEARNING_CONFIG "
            f"'{DEEP_LEARNING_CONFIG['family']}' "
            f"no coincide con '{deep_learning_family}'."
        )

    result = run_mlp(
        x_train=BENCHMARK_DATA["x_train"],
        y_train=BENCHMARK_DATA["y_train"],
        x_test=BENCHMARK_DATA["x_test"],
        y_test=BENCHMARK_DATA["y_test"],
    ) # Ejecutar modelo MLP

    if result is None:
        raise RuntimeError(
            f"El modelo '{model_name}' no produjo resultados."
        )

    if not isinstance(result, dict):
        raise TypeError(
            f"El resultado del modelo '{model_name}' debe ser un diccionario."
        )

    result = validate_model_result(
        result=result,
        model_code=model_code,
        model_name=DEEP_LEARNING_CONFIG["model_name"],
        family=deep_learning_family,
    ) # Validar contrato científico completo del resultado

    if result["model_code"] != model_code:
        raise RuntimeError(
            f"El código del resultado '{result['model_code']}' "
            f"no coincide con el código esperado '{model_code}'."
        )

    if result["model_name"].strip().lower().replace(" ", "_") != normalized_model_name:
        raise RuntimeError(
            f"El nombre del resultado '{result['model_name']}' "
            f"no coincide con el modelo ejecutado '{model_name}'."
        )

    if result["family"] != deep_learning_family:
        raise RuntimeError(
            f"La familia del resultado '{result['family']}' "
            f"no coincide con '{deep_learning_family}'."
        )

    benchmark_family_results[deep_learning_family].append(
        result
    ) # Registrar resultado por familia

    benchmark_results.append(
        result
    ) # Registrar resultado global

print(
    f"Modelos Deep Learning evaluados : "
    f"{len(benchmark_family_results[deep_learning_family])}"
) # Mostrar cantidad evaluada

print("Contrato Deep Learning           : VALIDADO") # Confirmar contrato
print("Códigos de modelos DL            : VALIDADOS") # Confirmar identidad

# 7.6 Evaluación de modelos Graph Neural Networks
print(type(BENCHMARK_DATA["train_index"])) # Verificar tipo del índice de entrenamiento
print(type(BENCHMARK_DATA["validation_index"])) # Verificar tipo del índice de validación
print(type(BENCHMARK_DATA["test_index"])) # Verificar tipo del índice de prueba

print("\n7.6 EVALUACIÓN DE MODELOS GRAPH NEURAL NETWORKS")
gnn_family = "graph_neural_networks" # Identificar familia oficial de GNN
gnn_models = CANDIDATE_MODELS["families"].get(
    gnn_family,
    []
) # Recuperar modelos GNN de la configuración oficial

if not gnn_models:
    raise ValueError(
        "No existen modelos Graph Neural Networks configurados."
    )

if gnn_family not in benchmark_family_results:
    benchmark_family_results[gnn_family] = [] # Inicializar resultados de la familia

for model_name in gnn_models:

    normalized_model_name = model_name.strip().lower().replace(
        " ",
        "_"
    ) # Normalizar nombre del modelo candidato

    if model_name not in GNN_CONFIG:
        raise KeyError(
            f"No existe configuración GNN para '{model_name}'."
        )

    model_config = GNN_CONFIG[model_name] # Recuperar configuración específica del modelo

    if not isinstance(model_config, dict):
        raise TypeError(
            f"La configuración GNN de '{model_name}' debe ser un diccionario."
        )

    required_model_config_keys = [
        "model_code",
        "model_name",
        "family",
    ] # Definir identidad obligatoria del modelo GNN

    missing_model_config_keys = [
        key
        for key in required_model_config_keys
        if key not in model_config
    ]

    if missing_model_config_keys:
        raise ValueError(
            f"La configuración GNN de '{model_name}' está incompleta: "
            f"{missing_model_config_keys}"
        )

    configured_model_name = model_config["model_name"].strip().lower().replace(
        " ",
        "_"
    ) # Normalizar nombre de la configuración

    if configured_model_name != normalized_model_name:
        raise ValueError(
            f"El nombre de la configuración GNN "
            f"'{model_config['model_name']}' no coincide con "
            f"'{model_name}'."
        )

    if model_config["family"] != gnn_family:
        raise ValueError(
            f"La familia configurada para '{model_name}' "
            f"no coincide con '{gnn_family}'."
        )

    if not isinstance(model_config["model_code"], str):
        raise TypeError(
            f"El código del modelo GNN '{model_name}' debe ser una cadena."
        )

    if not model_config["model_code"].strip():
        raise ValueError(
            f"El código del modelo GNN '{model_name}' no puede estar vacío."
        )

    matching_model_codes = [
        model_code
        for model_code, configured_model_name in BENCHMARK_MODEL_CODES.items()
        if configured_model_name.strip().lower().replace(" ", "_")
        == normalized_model_name
    ] # Buscar código oficial correspondiente en el catálogo

    if len(matching_model_codes) != 1:
        raise RuntimeError(
            f"No existe una correspondencia única entre el modelo GNN "
            f"'{model_name}' y BENCHMARK_MODEL_CODES."
        )

    expected_model_code = matching_model_codes[0] # Recuperar código oficial del catálogo
    if model_config["model_code"] != expected_model_code:
        raise ValueError(
            f"El código configurado para '{model_name}' "
            f"'{model_config['model_code']}' no coincide con el "
            f"código oficial del catálogo '{expected_model_code}'."
        )

    result = run_gnn_benchmark(
        model_config=model_config,
        benchmark_data=BENCHMARK_DATA,
    ) # Ejecutar modelo Graph Neural Network

    if result is None:
        raise RuntimeError(
            f"El modelo GNN '{model_name}' no produjo resultados."
        )

    if not isinstance(result, dict):
        raise TypeError(
            f"El resultado del modelo GNN '{model_name}' "
            "debe ser un diccionario."
        )

    result = validate_model_result(
        result=result,
        model_code=model_config["model_code"],
        model_name=model_config["model_name"],
        family=gnn_family,
    ) # Validar contrato científico completo del resultado

    if result["model_code"] != model_config["model_code"]:
        raise RuntimeError(
            f"El código del resultado '{result['model_code']}' "
            f"no coincide con el código configurado "
            f"'{model_config['model_code']}'."
        )

    if result["model_name"].strip().lower().replace(
        " ",
        "_"
    ) != normalized_model_name:
        raise RuntimeError(
            f"El nombre del resultado '{result['model_name']}' "
            f"no coincide con el modelo ejecutado '{model_name}'."
        )

    if result["family"] != gnn_family:
        raise RuntimeError(
            f"La familia del resultado '{result['family']}' "
            f"no coincide con '{gnn_family}'."
        )

    benchmark_family_results[gnn_family].append(
        result
    ) # Registrar resultado por familia

    benchmark_results.append(
        result
    ) # Registrar resultado global

print(f"Modelos Graph Neural Networks evaluados : {len(benchmark_family_results[gnn_family])}") # Mostrar cantidad evaluada
print("Contrato GNN                             : VALIDADO") # Confirmar contrato
print("Códigos de modelos GNN                   : VALIDADOS") # Confirmar identidad

# 7.7 Validación de los resultados obtenidos
print("\n7.7 VALIDACIÓN DE LOS RESULTADOS DEL BENCHMARK")

if not benchmark_results:
    raise RuntimeError(
        "El Benchmark Científico no produjo resultados."
    ) # Verificar que existan resultados

# Validar identificación y contrato científico de cada resultado
required_result_keys = [
    "model_code",
    "model_name",
    "family",
    *BENCHMARK_METRICS,
] # Definir contrato científico completo de cada resultado

for result in benchmark_results:

    if not isinstance(result, dict):
        raise TypeError(
            "Cada resultado del Benchmark debe ser un diccionario."
        )

    missing_result_keys = [
        key
        for key in required_result_keys
        if key not in result
    ]

    if missing_result_keys:
        raise ValueError(
            f"Resultado del Benchmark incompleto: {missing_result_keys}"
        )

    if not isinstance(
        result["model_code"],
        str
    ) or not result["model_code"].strip():
        raise ValueError(
            "Un resultado del Benchmark contiene un 'model_code' inválido."
        )

    if not isinstance(
        result["model_name"],
        str
    ) or not result["model_name"].strip():
        raise ValueError(
            "Un resultado del Benchmark contiene un 'model_name' inválido."
        )

    if not isinstance(
        result["family"],
        str
    ) or not result["family"].strip():
        raise ValueError(
            "Un resultado del Benchmark contiene una 'family' inválida."
        )

    for metric in BENCHMARK_METRICS:

        if result[metric] is None:
            raise ValueError(
                f"El modelo '{result['model_name']}' contiene "
                f"la métrica '{metric}' con valor None."
            )

        if not np.isfinite(
            float(result[metric])
        ):
            raise ValueError(
                f"El modelo '{result['model_name']}' contiene "
                f"un valor inválido para '{metric}'."
            )

# Construir catálogo oficial de identidades de modelos
candidate_identities = set()

for model in CANDIDATE_MODELS["models"]:

    model_name = model["model_name"].strip().lower()
    family = model["family"].strip().lower()

    matching_codes = [
        model_code
        for model_code, configured_model_name
        in BENCHMARK_MODEL_CODES.items()
        if configured_model_name.strip().lower() == model_name
    ]

    if len(matching_codes) != 1:
        raise ValueError(
            f"No existe una correspondencia única de código "
            f"para el modelo candidato '{model['model_name']}'."
        )

    model_code = matching_codes[0].strip().lower()

    candidate_identities.add(
        (
            model_code,
            model_name,
            family,
        )
    ) # Construir identidad completa del modelo candidato

# Validar correspondencia individual completa
executed_identities = []

for result in benchmark_results:

    result_identity = (
        result["model_code"].strip().lower(),
        result["model_name"].strip().lower(),
        result["family"].strip().lower(),
    ) # Construir identidad completa del resultado

    if result_identity not in candidate_identities:
        raise ValueError(
            f"El resultado del modelo '{result['model_name']}' "
            f"no coincide con ningún modelo oficial de CANDIDATE_MODELS."
        )

    executed_identities.append(
        result_identity
    )

# Validar duplicados de modelos ejecutados
if len(executed_identities) != len(
    set(executed_identities)
):
    raise ValueError(
        "Existen resultados duplicados para uno o más modelos."
    )

# Validar que todos los candidatos hayan sido ejecutados
executed_identity_set = set(
    executed_identities
)

missing_candidates = sorted(
    candidate_identities - executed_identity_set
)

if missing_candidates:
    raise RuntimeError(
        "Existen modelos candidatos que no produjeron resultados: "
        f"{missing_candidates}"
    )

# Validar que no existan resultados adicionales
unexpected_results = sorted(
    executed_identity_set - candidate_identities
)

if unexpected_results:
    raise RuntimeError(
        "Existen resultados que no pertenecen al catálogo oficial: "
        f"{unexpected_results}"
    )

if executed_identity_set != candidate_identities:
    raise RuntimeError(
        "La colección de resultados no coincide exactamente "
        "con la colección de modelos candidatos."
    )

# Validar que cada familia configurada haya producido resultados
configured_families = {
    family.strip().lower()
    for family in CANDIDATE_MODELS["families"]
} # Recuperar familias oficiales configuradas

for family in configured_families:

    family_results = [
        result
        for result in benchmark_results
        if result["family"].strip().lower() == family
    ]

    if not family_results:
        raise RuntimeError(
            f"La familia '{family}' no produjo resultados."
        )

# Validar correspondencia entre resultados globales y resultados por familia
family_result_count = sum(
    len(results)
    for results in benchmark_family_results.values()
) # Calcular resultados agrupados por familia

if family_result_count != len(
    benchmark_results
):
    raise RuntimeError(
        "La cantidad de resultados globales no coincide "
        "con los resultados agrupados por familia."
    )

print(f"Resultados obtenidos        : {len(benchmark_results)}") # Mostrar cantidad de resultados
print(f"Familias evaluadas          : {len(configured_families)}") # Mostrar cantidad de familias
print("Identificación de modelos   : VALIDADA") # Confirmar identificación
print("Códigos de modelos          : VALIDADOS") # Confirmar códigos
print("Métricas oficiales          : VALIDADAS") # Confirmar métricas
print("Correspondencia catálogo    : VALIDADA") # Confirmar correspondencia completa con CANDIDATE_MODELS
print("Cobertura de candidatos     : COMPLETA") # Confirmar que todos los candidatos fueron ejecutados
print("Duplicados                  : NO DETECTADOS") # Confirmar ausencia de duplicados
print("Resultados por familia      : VALIDADOS") # Confirmar agrupación por familia
print("Integridad de resultados    : VALIDADA") # Confirmar integridad científica

# 7.8 Construcción del producto oficial
print("\n7.8 CONSTRUCCIÓN DEL PRODUCTO OFICIAL BENCHMARK_RESULTS")

if not isinstance(
    benchmark_results,
    list
):
    raise TypeError(
        "benchmark_results debe ser una lista."
    )

if not isinstance(
    benchmark_family_results,
    dict
):
    raise TypeError(
        "benchmark_family_results debe ser un diccionario."
    )

if len(benchmark_results) == 0:
    raise ValueError(
        "No existen resultados para construir BENCHMARK_RESULTS."
    )

family_result_count = sum(
    len(results)
    for results in benchmark_family_results.values()
) # Calcular cantidad total de resultados agrupados por familia

if family_result_count != len(benchmark_results):
    raise RuntimeError(
        "La cantidad de resultados por familia no coincide "
        "con el total de resultados."
    )

if set(benchmark_family_results) != set(
    CANDIDATE_MODELS["families"]
):
    raise RuntimeError(
        "Las familias de resultados no coinciden "
        "con las familias oficiales del Benchmark."
    )

# Recuperar identidad completa del Modelo Oficial
official_model = CANDIDATE_MODELS[
    "official_model"
] # Recuperar identidad oficial

required_official_keys = [
    "model_code",
    "model_name",
    "family",
] # Definir contrato del Modelo Oficial

missing_official_keys = [
    key
    for key in required_official_keys
    if key not in official_model
]

if missing_official_keys:
    raise RuntimeError(
        "El Modelo Oficial está incompleto: "
        f"{missing_official_keys}"
    )

if official_model["model_code"] != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El código del Modelo Oficial no coincide "
        "con OFFICIAL_MODEL_CODE."
    )

if official_model["model_name"] != OFFICIAL_MODEL_NAME:
    raise RuntimeError(
        "El nombre del Modelo Oficial no coincide "
        "con OFFICIAL_MODEL_NAME."
    )

if official_model["family"] != OFFICIAL_MODEL_FAMILY:
    raise RuntimeError(
        "La familia del Modelo Oficial no coincide "
        "con OFFICIAL_MODEL_FAMILY."
    )

# Recuperar métrica oficial del Ranking
ranking_metric = BENCHMARK_CONFIG.get(
    "ranking_metric"
) # Recuperar métrica principal del Benchmark

if not isinstance(
    ranking_metric,
    str
):
    raise TypeError(
        "La métrica de ranking debe ser una cadena."
    )

if ranking_metric not in BENCHMARK_METRICS:
    raise ValueError(
        f"La métrica de ranking '{ranking_metric}' "
        "no pertenece a BENCHMARK_METRICS."
    )

BENCHMARK_RESULTS = {
    "results": benchmark_results,
    "family_results": benchmark_family_results,
    "total_results": len(benchmark_results),
    "total_families": len(benchmark_family_results),
    "protocol": BENCHMARK_CONFIG,
    "ranking_metric": ranking_metric,
    "official_model": {
        "model_code": official_model["model_code"],
        "model_name": official_model["model_name"],
        "family": official_model["family"],
    },
} # Construir producto oficial consolidado

print(f"Resultados totales          : {BENCHMARK_RESULTS['total_results']}") # Mostrar resultados
print(f"Familias evaluadas          : {BENCHMARK_RESULTS['total_families']}") # Mostrar familias
print(f"Métrica de ranking          : {BENCHMARK_RESULTS['ranking_metric']}") # Mostrar métrica oficial
print(f"Código Modelo Oficial       : {BENCHMARK_RESULTS['official_model']['model_code']}") # Mostrar código oficial
print(f"Modelo Oficial              : {BENCHMARK_RESULTS['official_model']['model_name']}") # Mostrar modelo oficial
print(f"Familia Oficial             : {BENCHMARK_RESULTS['official_model']['family']}") # Mostrar familia oficial
print("Contrato Modelo Oficial     : VALIDADO") # Confirmar contrato
print("BENCHMARK_RESULTS           : CONSTRUIDO") # Confirmar construcción

# 7.9 Validación del producto oficial
print("\n7.9 VALIDACIÓN DEL PRODUCTO OFICIAL BENCHMARK_RESULTS")

required_benchmark_result_keys = [
    "results",
    "family_results",
    "total_results",
    "total_families",
    "protocol",
    "ranking_metric",
    "official_model",
] # Definir contrato oficial de BENCHMARK_RESULTS

missing_benchmark_result_keys = [
    key
    for key in required_benchmark_result_keys
    if key not in BENCHMARK_RESULTS
]

if missing_benchmark_result_keys:
    raise RuntimeError(
        "BENCHMARK_RESULTS está incompleto: "
        f"{missing_benchmark_result_keys}"
    )

# Validar tipos principales
if not isinstance(
    BENCHMARK_RESULTS["results"],
    list
):
    raise TypeError(
        "BENCHMARK_RESULTS['results'] debe ser una lista."
    )

if not isinstance(
    BENCHMARK_RESULTS["family_results"],
    dict
):
    raise TypeError(
        "BENCHMARK_RESULTS['family_results'] debe ser un diccionario."
    )

if not isinstance(
    BENCHMARK_RESULTS["total_results"],
    int
):
    raise TypeError(
        "BENCHMARK_RESULTS['total_results'] debe ser un entero."
    )

if not isinstance(
    BENCHMARK_RESULTS["total_families"],
    int
):
    raise TypeError(
        "BENCHMARK_RESULTS['total_families'] debe ser un entero."
    )

if not isinstance(
    BENCHMARK_RESULTS["protocol"],
    dict
):
    raise TypeError(
        "BENCHMARK_RESULTS['protocol'] debe ser un diccionario."
    )

if not isinstance(
    BENCHMARK_RESULTS["ranking_metric"],
    str
):
    raise TypeError(
        "BENCHMARK_RESULTS['ranking_metric'] debe ser una cadena."
    )

if not isinstance(
    BENCHMARK_RESULTS["official_model"],
    dict
):
    raise TypeError(
        "BENCHMARK_RESULTS['official_model'] debe ser un diccionario."
    )

# Validar correspondencia entre conteos y colecciones
if BENCHMARK_RESULTS["total_results"] != len(
    BENCHMARK_RESULTS["results"]
):
    raise RuntimeError(
        "El total de resultados no coincide "
        "con la colección de resultados."
    )

if BENCHMARK_RESULTS["total_families"] != len(
    BENCHMARK_RESULTS["family_results"]
):
    raise RuntimeError(
        "El total de familias no coincide "
        "con la colección de resultados por familia."
    )

# Validar correspondencia entre resultados globales y resultados por familia
family_result_count = sum(
    len(results)
    for results in BENCHMARK_RESULTS["family_results"].values()
) # Calcular cantidad total de resultados agrupados por familia

if family_result_count != BENCHMARK_RESULTS["total_results"]:
    raise RuntimeError(
        "Los resultados globales no coinciden "
        "con los resultados agrupados por familia."
    )

# Validar familias oficiales
official_families = {
    family.strip().lower()
    for family in CANDIDATE_MODELS["families"]
} # Recuperar familias oficiales

result_families = {
    family.strip().lower()
    for family in BENCHMARK_RESULTS["family_results"]
} # Recuperar familias presentes en los resultados

if result_families != official_families:
    raise RuntimeError(
        "Las familias de BENCHMARK_RESULTS no coinciden "
        "con las familias oficiales."
    )

# Validar métrica de ranking
protocol_ranking_metric = BENCHMARK_RESULTS[
    "protocol"
].get(
    "ranking_metric"
) # Recuperar métrica de ranking del protocolo

if protocol_ranking_metric is None:
    raise RuntimeError(
        "El protocolo de BENCHMARK_RESULTS no contiene "
        "'ranking_metric'."
    )

if protocol_ranking_metric != BENCHMARK_CONFIG["ranking_metric"]:
    raise RuntimeError(
        "La métrica de ranking del protocolo no coincide "
        "con BENCHMARK_CONFIG."
    )

if BENCHMARK_RESULTS["ranking_metric"] != BENCHMARK_CONFIG[
    "ranking_metric"
]:
    raise RuntimeError(
        "La métrica de ranking de BENCHMARK_RESULTS no coincide "
        "con BENCHMARK_CONFIG."
    )

if BENCHMARK_RESULTS["ranking_metric"] != protocol_ranking_metric:
    raise RuntimeError(
        "Las referencias a la métrica de ranking no son consistentes."
    )

if BENCHMARK_RESULTS["ranking_metric"] not in BENCHMARK_METRICS:
    raise RuntimeError(
        f"La métrica de ranking '{BENCHMARK_RESULTS['ranking_metric']}' "
        "no pertenece a BENCHMARK_METRICS."
    )

# Validar Modelo Oficial
required_official_model_keys = [
    "model_code",
    "model_name",
    "family",
] # Definir contrato completo del Modelo Oficial

missing_official_model_keys = [
    key
    for key in required_official_model_keys
    if key not in BENCHMARK_RESULTS["official_model"]
]

if missing_official_model_keys:
    raise RuntimeError(
        "El Modelo Oficial está incompleto: "
        f"{missing_official_model_keys}"
    )

official_model_result = BENCHMARK_RESULTS[
    "official_model"
] # Recuperar identidad oficial registrada

if official_model_result["model_code"] != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El código del Modelo Oficial no coincide "
        "con OFFICIAL_MODEL_CODE."
    )

if official_model_result["model_name"] != OFFICIAL_MODEL_NAME:
    raise RuntimeError(
        "El nombre del Modelo Oficial no coincide "
        "con OFFICIAL_MODEL_NAME."
    )

if official_model_result["family"] != OFFICIAL_MODEL_FAMILY:
    raise RuntimeError(
        "La familia del Modelo Oficial no coincide "
        "con OFFICIAL_MODEL_FAMILY."
    )

# Validar presencia exacta del Modelo Oficial en los resultados
official_result_matches = [
    result
    for result in BENCHMARK_RESULTS["results"]
    if (
        result.get("model_code") == OFFICIAL_MODEL_CODE
        and
        result.get("model_name") == OFFICIAL_MODEL_NAME
        and
        result.get("family") == OFFICIAL_MODEL_FAMILY
    )
] # Buscar el Modelo Oficial dentro de los resultados evaluados

if len(official_result_matches) != 1:
    raise RuntimeError(
        "El Modelo Oficial debe aparecer exactamente una vez "
        "en los resultados del Benchmark."
    )

print(f"Resultados validados        : {BENCHMARK_RESULTS['total_results']}") # Mostrar resultados
print(f"Familias validadas          : {BENCHMARK_RESULTS['total_families']}") # Mostrar familias
print(f"Métrica de ranking          : {BENCHMARK_RESULTS['ranking_metric']}") # Mostrar métrica
print(f"Código Modelo Oficial       : {official_model_result['model_code']}") # Mostrar código
print(f"Modelo Oficial              : {official_model_result['model_name']}") # Mostrar modelo
print(f"Familia Oficial             : {official_model_result['family']}") # Mostrar familia
print("Estructura                  : CORRECTA") # Confirmar estructura
print("Conteos                     : CONSISTENTES") # Confirmar conteos
print("Familias                    : CONSISTENTES") # Confirmar familias
print("Métrica de ranking         : VALIDADA") # Confirmar métrica
print("Modelo Oficial              : VALIDADO") # Confirmar modelo
print("BENCHMARK_RESULTS            : VALIDADO") # Confirmar producto

# 7.10 Resumen del Benchmark
print("\n" + "-" * 80)
print("RESUMEN DE LA EJECUCIÓN DEL BENCHMARK")
print("-" * 80)

for family, results in benchmark_family_results.items():
    print(f"{family:<28}: {len(results)} modelo(s) evaluado(s)") # Mostrar resultados por familia

print(f"Total de modelos evaluados : {BENCHMARK_RESULTS['total_results']}") # Mostrar total de resultados
print(f"Total de familias evaluadas: {BENCHMARK_RESULTS['total_families']}") # Mostrar total de familias
print(f"Métrica de ranking          : {BENCHMARK_RESULTS['ranking_metric']}") # Mostrar métrica oficial
print(f"Modelo Oficial              : {BENCHMARK_RESULTS['official_model']['model_name']}") # Mostrar Modelo Oficial
print(f"Código Modelo Oficial       : {BENCHMARK_RESULTS['official_model']['model_code']}") # Mostrar código oficial
print(f"Familia Oficial             : {BENCHMARK_RESULTS['official_model']['family']}") # Mostrar familia oficial
print("Estado                      : BENCHMARK EJECUTADO") # Confirmar ejecución
print("Producto oficial            : VALIDADO") # Confirmar BENCHMARK_RESULTS

print("-" * 80)
print("Bloque 7. Benchmark Científico ejecutado correctamente.")

# BLOQUE 8. CONSOLIDACIÓN DE LOS RESULTADOS DEL BENCHMARK
# Objetivo: Consolidar y validar los resultados generados por el Benchmark
# Científico en una estructura única y trazable para el análisis comparativo.
# Producto: - CONSOLIDATED_BENCHMARK_RESULTS - Colección consolidada de resultados del Benchmark Científico.
# Pregunta científica: ¿Los resultados generados por todas las familias de modelos fueron
# consolidados correctamente en una colección única y consistente?

print("\n" + "-" * 80)
print("BLOQUE 8. CONSOLIDACIÓN DE LOS RESULTADOS DEL BENCHMARK")
print("-" * 80)

# 8.1 Validación de los resultados oficiales del Benchmark
print("\n8.1 VALIDACIÓN DE LOS RESULTADOS OFICIALES")

if not isinstance(BENCHMARK_RESULTS, dict):
    raise TypeError("BENCHMARK_RESULTS debe ser un diccionario.")

required_benchmark_keys = [
    "results",
    "family_results",
    "total_results",
    "total_families",
    "protocol",
    "official_model",
] # Definir contrato oficial de BENCHMARK_RESULTS

missing_benchmark_keys = [
    key
    for key in required_benchmark_keys
    if key not in BENCHMARK_RESULTS
]

if missing_benchmark_keys:
    raise ValueError(f"BENCHMARK_RESULTS está incompleto: {missing_benchmark_keys}")

if not isinstance(BENCHMARK_RESULTS["results"], list):
    raise TypeError("BENCHMARK_RESULTS['results'] debe ser una lista.")

if not isinstance(BENCHMARK_RESULTS["family_results"], dict):
    raise TypeError("BENCHMARK_RESULTS['family_results'] debe ser un diccionario.")

if not isinstance(BENCHMARK_RESULTS["protocol"], dict):
    raise TypeError("BENCHMARK_RESULTS['protocol'] debe ser un diccionario.")

if not isinstance(BENCHMARK_RESULTS["official_model"], dict):
    raise TypeError("BENCHMARK_RESULTS['official_model'] debe ser un diccionario.")

if BENCHMARK_RESULTS["total_results"] != len(BENCHMARK_RESULTS["results"]):
    raise RuntimeError("El total de resultados de BENCHMARK_RESULTS no coincide con su colección.")

print("Estructura de BENCHMARK_RESULTS validada correctamente.")

# 8.2 Recuperación de los resultados por familia
print("\n8.2 RECUPERACIÓN DE LOS RESULTADOS POR FAMILIA")

family_results = BENCHMARK_RESULTS["family_results"] # Recuperar resultados oficiales por familia

official_families = set(CANDIDATE_MODELS["families"]) # Recuperar familias oficiales del Benchmark
result_families = set(family_results) # Recuperar familias presentes en los resultados

if result_families != official_families:
    missing_families = sorted(official_families - result_families)
    unexpected_families = sorted(result_families - official_families)
    raise ValueError(f"Las familias de resultados no coinciden con la configuración oficial. Faltantes: {missing_families}. No esperadas: {unexpected_families}.")

# 8.3 Validación de los resultados de cada familia
print("\n8.3 VALIDACIÓN DE LOS RESULTADOS POR FAMILIA")

required_result_keys = [
    "model_name",
    "family",
] # Definir identificación mínima de cada resultado

for family_name in official_families:
    results = family_results[family_name]

    if not isinstance(results, list):
        raise TypeError(f"Los resultados de la familia '{family_name}' deben almacenarse en una lista.")

    if len(results) == 0:
        raise ValueError(f"La familia '{family_name}' no contiene resultados.")

    for result in results:
        if not isinstance(result, dict):
            raise TypeError(f"Un resultado de la familia '{family_name}' no es un diccionario.")

        missing_result_keys = [
            key
            for key in required_result_keys
            if key not in result
        ]

        if missing_result_keys:
            raise ValueError(f"Resultado incompleto en la familia '{family_name}': {missing_result_keys}")

        if not isinstance(result["model_name"], str) or not result["model_name"].strip():
            raise ValueError(f"El resultado de la familia '{family_name}' contiene un model_name inválido.")

        if not isinstance(result["family"], str) or not result["family"].strip():
            raise ValueError(f"El resultado de la familia '{family_name}' contiene una family inválida.")

        if result["family"].strip().lower() != family_name.strip().lower():
            raise ValueError(f"El modelo '{result['model_name']}' declara una familia diferente a '{family_name}'.")

print("Resultados de todas las familias validados correctamente.")

# 8.4 Consolidación de los resultados
print("\n8.4 CONSOLIDACIÓN DE LOS RESULTADOS")

consolidated_results = [] # Inicializar colección consolidada
for family_name in official_families:
    for result in family_results[family_name]:
        consolidated_results.append(result.copy()) # Incorporar copia independiente del resultado

if not consolidated_results:
    raise RuntimeError("La colección consolidada de resultados está vacía.")

# 8.5 Validación de la consolidación
print("\n8.5 VALIDACIÓN DE LA CONSOLIDACIÓN")

expected_total_results = sum(
    len(family_results[family])
    for family in official_families
) # Calcular cantidad esperada de resultados

if len(consolidated_results) != expected_total_results:
    raise RuntimeError("La cantidad de resultados consolidados no coincide con la cantidad de resultados por familia.")

if len(consolidated_results) != BENCHMARK_RESULTS["total_results"]:
    raise RuntimeError("La cantidad de resultados consolidados no coincide con BENCHMARK_RESULTS['total_results'].")

result_pairs = [
    (
        result["family"].strip().lower(),
        result["model_name"].strip().lower(),
    )
    for result in consolidated_results
] # Construir identificadores únicos de familia y modelo

if len(result_pairs) != len(set(result_pairs)):
    raise ValueError("Existen modelos duplicados en los resultados consolidados.")

# 8.6 Construcción del producto oficial
print("\n8.6 CONSTRUCCIÓN DEL PRODUCTO OFICIAL")

CONSOLIDATED_BENCHMARK_RESULTS = {
    "results": consolidated_results,
    "family_results": family_results,
    "total_results": len(consolidated_results),
    "total_families": len(official_families),
    "protocol": BENCHMARK_RESULTS["protocol"],
    "official_model": BENCHMARK_RESULTS["official_model"],
    "status": "CONSOLIDATED",
} # Construir producto oficial consolidado

# 8.7 Validación del producto oficial
print("\n8.7 VALIDACIÓN DEL PRODUCTO OFICIAL")

required_consolidated_keys = [
    "results",
    "family_results",
    "total_results",
    "total_families",
    "protocol",
    "official_model",
    "status",
] # Definir contrato oficial del producto consolidado

missing_consolidated_keys = [
    key
    for key in required_consolidated_keys
    if key not in CONSOLIDATED_BENCHMARK_RESULTS
]

if missing_consolidated_keys:
    raise RuntimeError(f"CONSOLIDATED_BENCHMARK_RESULTS está incompleto: {missing_consolidated_keys}")

if not isinstance(CONSOLIDATED_BENCHMARK_RESULTS["results"], list):
    raise TypeError("CONSOLIDATED_BENCHMARK_RESULTS['results'] debe ser una lista.")

if not isinstance(CONSOLIDATED_BENCHMARK_RESULTS["family_results"], dict):
    raise TypeError("CONSOLIDATED_BENCHMARK_RESULTS['family_results'] debe ser un diccionario.")

if CONSOLIDATED_BENCHMARK_RESULTS["total_results"] != len(CONSOLIDATED_BENCHMARK_RESULTS["results"]):
    raise RuntimeError("El total de resultados consolidados es inconsistente.")

if CONSOLIDATED_BENCHMARK_RESULTS["total_families"] != len(official_families):
    raise RuntimeError("El total de familias consolidadas es inconsistente.")

if CONSOLIDATED_BENCHMARK_RESULTS["status"] != "CONSOLIDATED":
    raise RuntimeError("La consolidación del Benchmark no fue certificada.")

print("Producto consolidado validado correctamente.")

# 8.8 Resumen de la consolidación
print("\n" + "-" * 80)
print("RESULTADOS CONSOLIDADOS DEL BENCHMARK")
print("-" * 80)

for family_name in official_families:
    print(f"{family_name:<30}: {len(family_results[family_name])} modelo(s)") # Mostrar resultados por familia

print(f"Total de modelos consolidados : {CONSOLIDATED_BENCHMARK_RESULTS['total_results']}") # Mostrar total consolidado
print(f"Total de familias             : {CONSOLIDATED_BENCHMARK_RESULTS['total_families']}") # Mostrar total de familias
print(f"Modelo Oficial                : {CONSOLIDATED_BENCHMARK_RESULTS['official_model']}") # Mostrar modelo oficial configurado
print(f"Estado                        : {CONSOLIDATED_BENCHMARK_RESULTS['status']}") # Confirmar consolidación

print("-" * 80)
print("Bloque 8. Resultados del Benchmark consolidados correctamente.")

# BLOQUE 9. CONSTRUCCIÓN DEL RANKING CIENTÍFICO DEL BENCHMARK
# Objetivo: Construir el Ranking Científico Oficial del Benchmark utilizando los resultados consolidados
# y las métricas oficiales definidas para el protocolo experimental.
# Producto: - BENCHMARK_RANKING - Ranking Científico Oficial del Benchmark.
# Pregunta científica: ¿Cuál es el orden de desempeño predictivo de los modelos evaluados según
# las métricas oficiales del Benchmark Científico?

print("\n" + "-" * 80)
print("BLOQUE 9. CONSTRUCCIÓN DEL RANKING CIENTÍFICO DEL BENCHMARK")
print("-" * 80)

# 9.1 Validación de los resultados consolidados
print("\n9.1 VALIDACIÓN DE LOS RESULTADOS CONSOLIDADOS")

if not isinstance(CONSOLIDATED_BENCHMARK_RESULTS, dict):
    raise TypeError("CONSOLIDATED_BENCHMARK_RESULTS debe ser un diccionario.")

required_consolidated_keys = [
    "results",
    "family_results",
    "total_results",
    "total_families",
    "protocol",
    "official_model",
    "status",
] # Definir contrato obligatorio de los resultados consolidados

missing_consolidated_keys = [
    key
    for key in required_consolidated_keys
    if key not in CONSOLIDATED_BENCHMARK_RESULTS
]

if missing_consolidated_keys:
    raise ValueError(f"CONSOLIDATED_BENCHMARK_RESULTS está incompleto: {missing_consolidated_keys}")

if CONSOLIDATED_BENCHMARK_RESULTS["status"] != "CONSOLIDATED":
    raise RuntimeError("Los resultados del Benchmark no están consolidados.")

benchmark_results = CONSOLIDATED_BENCHMARK_RESULTS["results"] # Recuperar resultados consolidados

if not isinstance(benchmark_results, list):
    raise TypeError("Los resultados consolidados del Benchmark deben ser una lista.")

if len(benchmark_results) == 0:
    raise ValueError("La colección de resultados consolidados está vacía.")

if len(benchmark_results) != CONSOLIDATED_BENCHMARK_RESULTS["total_results"]:
    raise RuntimeError("La cantidad de resultados no coincide con total_results.")

print(f"Resultados disponibles     : {len(benchmark_results)}")

# 9.2 Validación de las métricas oficiales
print("\n9.2 VALIDACIÓN DE LAS MÉTRICAS OFICIALES")

if not isinstance(BENCHMARK_METRICS, (list, tuple)):
    raise TypeError("BENCHMARK_METRICS debe ser una lista o tupla.")

if len(BENCHMARK_METRICS) == 0:
    raise ValueError("No existen métricas oficiales para construir el ranking.")

required_metrics = [
    "rmse",
    "mae",
    "r2",
    "mape",
] # Definir métricas científicas obligatorias

missing_metrics = [
    metric
    for metric in required_metrics
    if metric not in BENCHMARK_METRICS
]

if missing_metrics:
    raise ValueError(f"Faltan métricas oficiales requeridas para el Benchmark: {missing_metrics}")

print(f"Métricas oficiales         : {', '.join(BENCHMARK_METRICS)}") # Mostrar métricas oficiales

# 9.3 Validación de la estructura de los resultados
print("\n9.3 VALIDACIÓN DE LA ESTRUCTURA DE LOS RESULTADOS")

required_result_keys = [
    "model_name",
    "family",
] # Definir identificación mínima de cada resultado

for result in benchmark_results:

    if not isinstance(result, dict):
        raise TypeError("Cada resultado del Benchmark debe ser un diccionario.")

    missing_result_keys = [
        key
        for key in required_result_keys
        if key not in result
    ]

    if missing_result_keys:
        raise ValueError(f"Resultado del Benchmark incompleto: {missing_result_keys}")

    missing_result_metrics = [
        metric
        for metric in BENCHMARK_METRICS
        if metric not in result
    ]

    if missing_result_metrics:
        raise ValueError(f"El modelo '{result['model_name']}' no contiene las métricas requeridas: {missing_result_metrics}")

    for metric in BENCHMARK_METRICS:
        if result[metric] is None:
            raise ValueError(f"El modelo '{result['model_name']}' contiene la métrica '{metric}' con valor None.")

        if not np.isfinite(float(result[metric])):
            raise ValueError(f"El modelo '{result['model_name']}' contiene un valor inválido para '{metric}'.")

print("Estructura de los resultados validada correctamente.")

# 9.4 Construcción del Ranking Científico
print("BENCHMARK_METRICS:", BENCHMARK_METRICS) # Mostrar métricas
print("Tipo:", type(BENCHMARK_METRICS)) # Mostrar tipo
print("Ranking metric:", BENCHMARK_CONFIG["ranking_metric"]) # Mostrar métrica de ranking
print("Dirección:", BENCHMARK_CONFIG["metric_directions"]["rmse"]) # Mostrar dirección

import inspect

print(
    inspect.signature(build_benchmark_ranking)
) # Verificar firma actual de la función

print("\nAUDITORÍA BENCHMARK_MODEL_CODES")

print(
    "Tipo:",
    type(BENCHMARK_MODEL_CODES)
) # Verificar tipo

print(
    "Cantidad:",
    len(BENCHMARK_MODEL_CODES)
) # Verificar cantidad

for model_code, model_name in BENCHMARK_MODEL_CODES.items():
    print(
        model_code,
        "|",
        model_name
    ) # Mostrar correspondencia oficial

print("\n9.4 CONSTRUCCIÓN DEL RANKING CIENTÍFICO")

BENCHMARK_RANKING = build_benchmark_ranking(
    benchmark_results=benchmark_results,
    benchmark_metrics=BENCHMARK_METRICS,
) # Construir Ranking Científico oficial

if not isinstance(BENCHMARK_RANKING, list):
    raise TypeError(
        "BENCHMARK_RANKING debe ser una lista."
    ) # Validar estructura

if len(BENCHMARK_RANKING) != len(benchmark_results):
    raise RuntimeError(
        "El Ranking no contiene todos los resultados del Benchmark."
    ) # Validar cobertura

for result in BENCHMARK_RANKING:
    print(
        f"{result['ranking_position']:02d} | "
        f"{result['model_code']} | "
        f"{result['model_name']} | "
        f"{result['family']} | "
        f"RMSE={result['rmse']:.6f}"
    ) # Mostrar ranking

print(
    "\nBENCHMARK_RANKING : CONSTRUIDO"
) # Confirmar construcción

# 9.5 Validación de la estructura del Ranking
print("\n9.5 VALIDACIÓN DE LA ESTRUCTURA DEL RANKING")

required_ranking_keys = [
    "model_name",
    "family",
    "ranking_position",
] # Definir contrato oficial del Ranking Científico

for ranking_result in BENCHMARK_RANKING:

    if not isinstance(ranking_result, dict):
        raise TypeError("Cada elemento del Ranking Científico debe ser un diccionario.")

    missing_ranking_keys = [
        key
        for key in required_ranking_keys
        if key not in ranking_result
    ]

    if missing_ranking_keys:
        raise ValueError(f"Elemento del Ranking incompleto: {missing_ranking_keys}")

    if not isinstance(ranking_result["ranking_position"], (int, np.integer)):
        raise TypeError(f"La posición del modelo '{ranking_result['model_name']}' debe ser un entero.")

    if ranking_result["ranking_position"] < 1:
        raise ValueError(f"La posición del modelo '{ranking_result['model_name']}' debe ser mayor o igual a 1.")

# 9.6 Validación de las posiciones del Ranking
print("\n9.6 VALIDACIÓN DE LAS POSICIONES DEL RANKING")

ranking_positions = [
    int(result["ranking_position"])
    for result in BENCHMARK_RANKING
] # Recuperar posiciones oficiales

expected_positions = list(
    range(
        1,
        len(BENCHMARK_RANKING) + 1,
    )
) # Construir posiciones esperadas

if sorted(ranking_positions) != expected_positions:
    raise ValueError("Las posiciones del Ranking Científico no son consecutivas.")

ranking_pairs = [
    (
        result["family"].strip().lower(),
        result["model_name"].strip().lower(),
    )
    for result in BENCHMARK_RANKING
] # Construir identificadores únicos de los modelos

if len(ranking_pairs) != len(set(ranking_pairs)):
    raise ValueError("Existen modelos duplicados en el Ranking Científico.")

print("Posiciones del Ranking      : VALIDADAS")
print("Modelos duplicados          : NO DETECTADOS")

# 9.7 Construcción del producto científico
print("\n9.7 CONSTRUCCIÓN DEL PRODUCTO CIENTÍFICO")

BENCHMARK_RANKING_RESULT = {
    "ranking": BENCHMARK_RANKING,
    "total_models": len(BENCHMARK_RANKING),
    "metrics": list(BENCHMARK_METRICS),
    "official_model": CONSOLIDATED_BENCHMARK_RESULTS["official_model"],
    "status": "VALIDATED",
} # Construir producto oficial del Ranking Científico

required_ranking_result_keys = [
    "ranking",
    "total_models",
    "metrics",
    "official_model",
    "status",
] # Definir contrato del producto final

missing_ranking_result = [
    key
    for key in required_ranking_result_keys
    if key not in BENCHMARK_RANKING_RESULT
]

if missing_ranking_result:
    raise RuntimeError(f"BENCHMARK_RANKING_RESULT está incompleto: {missing_ranking_result}")

if BENCHMARK_RANKING_RESULT["total_models"] != len(BENCHMARK_RANKING_RESULT["ranking"]):
    raise RuntimeError("El total de modelos del Ranking no coincide con la colección.")

if BENCHMARK_RANKING_RESULT["status"] != "VALIDATED":
    raise RuntimeError("El Ranking Científico no fue validado correctamente.")

# 9.8 Presentación del Ranking Científico
print("\n" + "-" * 80)
print("RANKING CIENTÍFICO DEL BENCHMARK")
print("-" * 80)

for ranking_result in sorted(
    BENCHMARK_RANKING,
    key=lambda result: result["ranking_position"],
):
    print(f"{ranking_result['ranking_position']:>3}. {ranking_result['model_name']:<20} | {ranking_result['family']}")

print("-" * 80)
print(f"Modelos clasificados        : {BENCHMARK_RANKING_RESULT['total_models']}") # Mostrar cantidad de modelos clasificados
print(f"Métrica(s) utilizada(s)     : {', '.join(BENCHMARK_RANKING_RESULT['metrics'])}") # Mostrar métricas utilizadas
print(f"Modelo Oficial              : {BENCHMARK_RANKING_RESULT['official_model']}") # Mostrar modelo oficial configurado
print(f"Estado                      : {BENCHMARK_RANKING_RESULT['status']}") # Confirmar validación del Ranking
print("-" * 80)
print("Bloque 9. Ranking Científico construido y validado correctamente.")

# BLOQUE 10. SELECCIÓN DEL MODELO OFICIAL DE LA FAMILIA GRAPH NEURAL NETWORKS
# Objetivo: Seleccionar el Modelo Oficial del proyecto a partir de las arquitecturas Graph Neural 
# Networks evaluadas durante el Benchmark Científico.
# Producto: OFFICIAL_MODEL
# Modelo Oficial del Proyecto perteneciente a la familia GNN.
# Pregunta científica: ¿Cuál es la Graph Neural Network con mejor desempeño predictivo según el
# Ranking Científico Oficial del Benchmark para ser adoptada como Modelo Oficial?

print("\n" + "-" * 80)
print("BLOQUE 10. SELECCIÓN DEL MODELO OFICIAL")
print("-" * 80)

# 10.1 Validación del Ranking Científico
print("\n10.1 VALIDACIÓN DEL RANKING CIENTÍFICO")

if not isinstance(BENCHMARK_RANKING_RESULT, dict):
    raise TypeError("BENCHMARK_RANKING_RESULT debe ser un diccionario.")

required_ranking_result_keys = [
    "ranking",
    "total_models",
    "metrics",
    "official_model",
    "status",
] # Definir contrato oficial del Ranking Científico

missing_ranking_result_keys = [
    key
    for key in required_ranking_result_keys
    if key not in BENCHMARK_RANKING_RESULT
]

if missing_ranking_result_keys:
    raise ValueError(f"BENCHMARK_RANKING_RESULT está incompleto: {missing_ranking_result_keys}")

if BENCHMARK_RANKING_RESULT["status"] != "VALIDATED":
    raise RuntimeError("El Ranking Científico no está validado.")

ranking = BENCHMARK_RANKING_RESULT["ranking"] # Recuperar Ranking Científico

if not isinstance(ranking, list):
    raise TypeError("El Ranking Científico debe ser una lista.")

if len(ranking) == 0:
    raise ValueError("El Ranking Científico está vacío.")

if len(ranking) != BENCHMARK_RANKING_RESULT["total_models"]:
    raise RuntimeError("La cantidad de modelos del Ranking no coincide con total_models.")

print(f"Modelos evaluados         : {len(ranking)}")

# 10.2 Validación de la familia oficial
print("\n10.2 VALIDACIÓN DE LA FAMILIA OFICIAL")

if not isinstance(OFFICIAL_MODEL_FAMILY, str):
    raise TypeError("OFFICIAL_MODEL_FAMILY debe ser una cadena.")

if not OFFICIAL_MODEL_FAMILY.strip():
    raise ValueError("OFFICIAL_MODEL_FAMILY está vacío.")

if OFFICIAL_MODEL_FAMILY not in CANDIDATE_MODELS["families"]:
    raise ValueError(f"La familia oficial '{OFFICIAL_MODEL_FAMILY}' no está registrada en CANDIDATE_MODELS.")

if OFFICIAL_MODEL_FAMILY != "graph_neural_networks":
    raise ValueError(f"La familia oficial configurada no corresponde a Graph Neural Networks: '{OFFICIAL_MODEL_FAMILY}'.")

print(f"Familia Oficial          : {OFFICIAL_MODEL_FAMILY}")

# 10.3 Identificación de modelos GNN
print("\n10.3 IDENTIFICACIÓN DE LOS MODELOS GRAPH NEURAL NETWORKS")

gnn_ranking = [
    result
    for result in ranking
    if result.get("family") == OFFICIAL_MODEL_FAMILY
] # Filtrar únicamente modelos de la familia oficial

if len(gnn_ranking) == 0:
    raise ValueError("No existen modelos de la familia Graph Neural Networks en el Ranking Científico.")

# Validar estructura mínima de los resultados GNN
required_gnn_fields = [
    "model_name",
    "family",
    "ranking_position",
] # Definir contrato mínimo de los resultados GNN

for result in gnn_ranking:
    missing_fields = [
        field
        for field in required_gnn_fields
        if field not in result
    ]

    if missing_fields:
        raise ValueError(f"Resultado GNN incompleto: {missing_fields}")

    if result["family"] != OFFICIAL_MODEL_FAMILY:
        raise ValueError(f"El modelo '{result['model_name']}' no pertenece a la familia oficial.")

    if not isinstance(result["ranking_position"], (int, np.integer)):
        raise TypeError(f"La posición del modelo '{result['model_name']}' debe ser un entero.")

# Ordenar por posición del Ranking Científico
gnn_ranking = sorted(
    gnn_ranking,
    key=lambda result: result["ranking_position"],
) # Ordenar GNN por posición global

print(f"Modelos GNN evaluados     : {len(gnn_ranking)}")

# 10.4 Selección de la mejor GNN
print("\n10.4 SELECCIÓN DE LA MEJOR GRAPH NEURAL NETWORK")

best_gnn = gnn_ranking[0] # Seleccionar la GNN con mejor posición global

selected_model_name = best_gnn["model_name"] # Recuperar nombre del modelo seleccionado
selected_ranking_position = int(best_gnn["ranking_position"]) # Recuperar posición global del modelo

if selected_model_name not in GNN_CONFIG:
    raise KeyError(f"No existe configuración GNN para el modelo seleccionado '{selected_model_name}'.")

selected_model_config = GNN_CONFIG[selected_model_name] # Recuperar configuración oficial de la arquitectura

if not isinstance(selected_model_config, dict):
    raise TypeError(f"La configuración GNN de '{selected_model_name}' debe ser un diccionario.")

# 10.5 Construcción del Modelo Oficial
print("\n10.5 CONSTRUCCIÓN DEL MODELO OFICIAL")

selected_model_code = selected_model_name.strip().lower().replace(" ", "_") # Generar código normalizado del modelo

OFFICIAL_MODEL = {
    "model_code": selected_model_code,
    "model_name": selected_model_name,
    "family": OFFICIAL_MODEL_FAMILY,
    "model_config": selected_model_config.copy(),
    "ranking_position": selected_ranking_position,
} # Construir producto oficial del modelo seleccionado

# 10.6 Validación del Modelo Oficial
print("\n10.6 VALIDACIÓN DEL MODELO OFICIAL")

required_official_fields = [
    "model_code",
    "model_name",
    "family",
    "model_config",
    "ranking_position",
] # Definir contrato oficial del Modelo Oficial

missing_official_fields = [
    field
    for field in required_official_fields
    if field not in OFFICIAL_MODEL
]

if missing_official_fields:
    raise RuntimeError(f"OFFICIAL_MODEL está incompleto: {missing_official_fields}")

if OFFICIAL_MODEL["family"] != OFFICIAL_MODEL_FAMILY:
    raise RuntimeError("El Modelo Oficial seleccionado no pertenece a la familia Graph Neural Networks.")

if OFFICIAL_MODEL["model_name"] != selected_model_name:
    raise RuntimeError("La identidad del Modelo Oficial no coincide con el resultado seleccionado del Ranking.")

if OFFICIAL_MODEL["ranking_position"] != selected_ranking_position:
    raise RuntimeError("La posición del Modelo Oficial no coincide con el Ranking Científico.")

if OFFICIAL_MODEL["model_name"] not in GNN_CONFIG:
    raise RuntimeError("La configuración del Modelo Oficial no existe en GNN_CONFIG.")

print("Modelo Oficial             : VALIDADO")
print(f"Nombre                    : {OFFICIAL_MODEL['model_name']}")
print(f"Familia                   : {OFFICIAL_MODEL['family']}")
print(f"Posición Ranking Global   : {OFFICIAL_MODEL['ranking_position']}")

# 10.7 Registro de la selección oficial
print("\n" + "-" * 80)
print("SELECCIÓN DEL MODELO OFICIAL")
print("-" * 80)

print(f"Modelo Oficial             : {OFFICIAL_MODEL['model_name']}") # Mostrar modelo seleccionado
print(f"Código                     : {OFFICIAL_MODEL['model_code']}") # Mostrar código oficial
print(f"Familia                    : {OFFICIAL_MODEL['family']}") # Mostrar familia oficial
print(f"Posición Ranking Global    : {OFFICIAL_MODEL['ranking_position']}") # Mostrar posición global
print(f"Modelos GNN evaluados      : {len(gnn_ranking)}") # Mostrar cantidad de GNN evaluadas
print("Criterio                   : MEJOR GNN DEL RANKING") # Mostrar criterio de selección
print("Estado                     : MODELO OFICIAL SELECCIONADO") # Confirmar selección

print("-" * 80)
print("Bloque 10. Modelo Oficial seleccionado correctamente.")

# BLOQUE 11. SELECCIÓN DE LOS MEJORES MODELOS POR FAMILIA
# Objetivo: Identificar el mejor modelo de cada familia a partir del Ranking Científico Oficial del Benchmark.
# Producto: - OFFICIAL_MODEL_BY_FAMILY - Mejor modelo identificado dentro de cada familia evaluada.
# Pregunta científica: ¿Cuál es el mejor modelo dentro de cada familia evaluada según el Ranking 
# Científico Oficial del Benchmark?

print("\n" + "-" * 80)
print("BLOQUE 11. SELECCIÓN DE LOS MEJORES MODELOS POR FAMILIA")
print("-" * 80)

# 11.1 Validación del Ranking Científico Oficial
print("\n11.1 VALIDACIÓN DEL RANKING CIENTÍFICO OFICIAL")

if not isinstance(BENCHMARK_RANKING_RESULT, dict):
    raise TypeError("BENCHMARK_RANKING_RESULT debe ser un diccionario.")

required_ranking_keys = [
    "ranking",
    "total_models",
    "metrics",
    "official_model",
    "status",
] # Definir contrato oficial del Ranking Científico

missing_ranking_keys = [
    key
    for key in required_ranking_keys
    if key not in BENCHMARK_RANKING_RESULT
]

if missing_ranking_keys:
    raise ValueError(f"BENCHMARK_RANKING_RESULT está incompleto: {missing_ranking_keys}")

if BENCHMARK_RANKING_RESULT["status"] != "VALIDATED":
    raise RuntimeError("El Ranking Científico no está validado.")

ranking = BENCHMARK_RANKING_RESULT["ranking"] # Recuperar Ranking Científico oficial

if not isinstance(ranking, list):
    raise TypeError("El Ranking Científico debe ser una lista.")

if len(ranking) == 0:
    raise ValueError("El Ranking Científico está vacío.")

if len(ranking) != BENCHMARK_RANKING_RESULT["total_models"]:
    raise RuntimeError("La cantidad de modelos del Ranking no coincide con total_models.")

print(f"Modelos evaluados         : {len(ranking)}")

# 11.2 Validación de la estructura del Ranking
print("\n11.2 VALIDACIÓN DE LA ESTRUCTURA DEL RANKING")

required_result_keys = [
    "ranking_position",
    "model_name",
    "family",
] # Definir campos mínimos del Ranking Científico

for result in ranking:

    if not isinstance(result, dict):
        raise TypeError("Cada resultado del Ranking debe ser un diccionario.")

    missing_result_keys = [
        key
        for key in required_result_keys
        if key not in result
    ]

    if missing_result_keys:
        raise ValueError(f"Resultado del Ranking incompleto: {missing_result_keys}")

    if not isinstance(result["ranking_position"], (int, np.integer)):
        raise TypeError(f"La posición del modelo '{result['model_name']}' debe ser un entero.")

    if result["ranking_position"] < 1:
        raise ValueError(f"La posición del modelo '{result['model_name']}' debe ser mayor o igual a 1.")

    if not isinstance(result["model_name"], str) or not result["model_name"].strip():
        raise ValueError("El Ranking contiene un model_name inválido.")

    if not isinstance(result["family"], str) or not result["family"].strip():
        raise ValueError("El Ranking contiene una family inválida.")

ranking_positions = [
    int(result["ranking_position"])
    for result in ranking
] # Recuperar posiciones del Ranking

expected_positions = list(
    range(
        1,
        len(ranking) + 1,
    )
) # Construir posiciones esperadas

if sorted(ranking_positions) != expected_positions:
    raise ValueError("Las posiciones del Ranking Científico no son consecutivas.")

print("Estructura del Ranking       : VALIDADA")
print("Posiciones del Ranking       : VALIDADAS")

# 11.3 Identificación de las familias
print("\n11.3 IDENTIFICACIÓN DE LAS FAMILIAS EVALUADAS")
families = sorted(
    {
        result["family"]
        for result in ranking
    }
) # Identificar familias presentes en el Ranking

if not families:
    raise ValueError("No existen familias en el Ranking Científico.")

official_families = set(CANDIDATE_MODELS["families"]) # Recuperar familias oficiales

if set(families) != official_families:
    missing_families = sorted(official_families - set(families))
    unexpected_families = sorted(set(families) - official_families)
    raise ValueError(f"Las familias del Ranking no coinciden con las oficiales. Faltantes: {missing_families}. No esperadas: {unexpected_families}.")

print(f"Familias identificadas      : {len(families)}")

for family in families:
    print(f"Familia                     : {family}") # Mostrar familia identificada

# 11.4 Selección del mejor modelo de cada familia
print("\n11.4 SELECCIÓN DEL MEJOR MODELO DE CADA FAMILIA")

OFFICIAL_MODEL_BY_FAMILY = {} # Inicializar colección oficial por familia
for family in families:
    family_results = [
        result
        for result in ranking
        if result["family"] == family
    ] # Recuperar modelos de la familia

    if not family_results:
        raise RuntimeError(f"No existen resultados para la familia '{family}'.")

    family_results = sorted(
        family_results,
        key=lambda result: result["ranking_position"],
    ) # Ordenar por posición global del Ranking

    best_model = family_results[0].copy() # Seleccionar mejor modelo de la familia
    model_code = best_model["model_name"].strip().lower().replace(" ", "_") # Generar código normalizado del modelo
    best_model["model_code"] = model_code # Incorporar código normalizado
    best_model["ranking_position"] = int(best_model["ranking_position"]) # Normalizar posición

    OFFICIAL_MODEL_BY_FAMILY[family] = best_model # Registrar mejor modelo de la familia

# 11.5 Validación de la selección por familia
print("\n11.5 VALIDACIÓN DE LOS MODELOS SELECCIONADOS")

if not isinstance(OFFICIAL_MODEL_BY_FAMILY, dict):
    raise TypeError("OFFICIAL_MODEL_BY_FAMILY debe ser un diccionario.")

if set(OFFICIAL_MODEL_BY_FAMILY) != set(families):
    raise RuntimeError("La colección de modelos seleccionados no coincide con las familias evaluadas.")

for family, result in OFFICIAL_MODEL_BY_FAMILY.items():

    if result["family"] != family:
        raise RuntimeError(f"El modelo seleccionado para '{family}' no pertenece a la familia correspondiente.")

    if not isinstance(result["model_name"], str) or not result["model_name"].strip():
        raise RuntimeError(f"La familia '{family}' no tiene un modelo válido.")

    if not isinstance(result["model_code"], str) or not result["model_code"].strip():
        raise RuntimeError(f"La familia '{family}' no tiene un código de modelo válido.")

    if not isinstance(result["ranking_position"], int):
        raise RuntimeError(f"La posición del modelo seleccionado para '{family}' no es válida.")

    family_positions = [
        item["ranking_position"]
        for item in ranking
        if item["family"] == family
    ] # Recuperar posiciones de la familia

    if result["ranking_position"] != min(family_positions):
        raise RuntimeError(f"El modelo seleccionado para '{family}' no corresponde al mejor ranking de la familia.")

print("Modelos seleccionados        : VALIDADOS")
print("Correspondencia por familia : VALIDADA")

# 11.6 Construcción del producto científico
print("\n11.6 CONSTRUCCIÓN DEL PRODUCTO CIENTÍFICO")

OFFICIAL_MODEL_BY_FAMILY_RESULT = {
    "models": OFFICIAL_MODEL_BY_FAMILY,
    "total_families": len(OFFICIAL_MODEL_BY_FAMILY),
    "ranking_source": "BENCHMARK_RANKING_RESULT",
    "status": "VALIDATED",
} # Construir producto científico oficial por familia

required_family_result_keys = [
    "models",
    "total_families",
    "ranking_source",
    "status",
] # Definir contrato del producto científico

missing_family_result_keys = [
    key
    for key in required_family_result_keys
    if key not in OFFICIAL_MODEL_BY_FAMILY_RESULT
]

if missing_family_result_keys:
    raise RuntimeError(f"OFFICIAL_MODEL_BY_FAMILY_RESULT está incompleto: {missing_family_result_keys}")

if OFFICIAL_MODEL_BY_FAMILY_RESULT["total_families"] != len(OFFICIAL_MODEL_BY_FAMILY_RESULT["models"]):
    raise RuntimeError("El total de familias no coincide con los modelos seleccionados.")

if OFFICIAL_MODEL_BY_FAMILY_RESULT["status"] != "VALIDATED":
    raise RuntimeError("La selección de modelos por familia no fue validada.")

# 11.7 Registrar los mejores modelos por familia
print("\n" + "-" * 80)
print("MEJORES MODELOS POR FAMILIA")
print("-" * 80)

for family, result in OFFICIAL_MODEL_BY_FAMILY.items():
    print(f"{family:<28}: {result['model_name']} | Posición global: {result['ranking_position']}") # Mostrar mejor modelo de cada familia

print(f"Total de familias          : {OFFICIAL_MODEL_BY_FAMILY_RESULT['total_families']}") # Mostrar total de familias
print(f"Estado                     : {OFFICIAL_MODEL_BY_FAMILY_RESULT['status']}") # Confirmar selección

print("-" * 80)
print("Bloque 11. Mejores modelos por familia seleccionados correctamente.")

# BLOQUE 12. REPORTE EJECUTIVO DEL BENCHMARK CIENTÍFICO
# Objetivo: Presentar el resumen ejecutivo del Benchmark Científico, mostrando los modelos evaluados,
# sus métricas, el ranking oficial, los mejores modelos por familia y el Modelo Oficial seleccionado.
# Producto: - BENCHMARK_EXECUTIVE_REPORT
# Pregunta científica: ¿Cuáles fueron los resultados obtenidos durante el Benchmark Científico
# y cuál fue el fundamento metodológico para seleccionar el Modelo Oficial?

print("\n" + "-" * 80)
print("BLOQUE 12. REPORTE EJECUTIVO DEL BENCHMARK CIENTÍFICO")
print("-" * 80)

# 12.1 Validación de los productos oficiales
print("\n12.1 VALIDACIÓN DE LOS PRODUCTOS OFICIALES")

required_products = {
    "benchmark_results": BENCHMARK_RESULTS,
    "benchmark_ranking_result": BENCHMARK_RANKING_RESULT,
    "official_model_by_family_result": OFFICIAL_MODEL_BY_FAMILY_RESULT,
    "official_model": OFFICIAL_MODEL,
} # Definir productos oficiales requeridos

missing_products = [
    name
    for name, value in required_products.items()
    if value is None
] # Identificar productos faltantes

if missing_products:
    raise ValueError(f"No existen los siguientes productos oficiales del Benchmark: {missing_products}")

if not isinstance(BENCHMARK_RESULTS, dict):
    raise TypeError("BENCHMARK_RESULTS debe ser un diccionario.")

if not isinstance(BENCHMARK_RANKING_RESULT, dict):
    raise TypeError("BENCHMARK_RANKING_RESULT debe ser un diccionario.")

if not isinstance(OFFICIAL_MODEL_BY_FAMILY_RESULT, dict):
    raise TypeError("OFFICIAL_MODEL_BY_FAMILY_RESULT debe ser un diccionario.")

if not isinstance(OFFICIAL_MODEL, dict):
    raise TypeError("OFFICIAL_MODEL debe ser un diccionario.")

required_benchmark_result_keys = [
    "results",
    "family_results",
    "total_results",
    "total_families",
    "protocol",
    "official_model",
] # Definir contrato de BENCHMARK_RESULTS

missing_benchmark_result_keys = [
    key
    for key in required_benchmark_result_keys
    if key not in BENCHMARK_RESULTS
] # Identificar campos faltantes

if missing_benchmark_result_keys:
    raise ValueError(f"BENCHMARK_RESULTS está incompleto: {missing_benchmark_result_keys}")

if not isinstance(BENCHMARK_RESULTS["results"], list):
    raise TypeError("BENCHMARK_RESULTS['results'] debe ser una lista.")

if len(BENCHMARK_RESULTS["results"]) == 0:
    raise ValueError("BENCHMARK_RESULTS['results'] está vacío.")

print("Productos oficiales del Benchmark disponibles y validados.")

# 12.2 Recuperación de los productos oficiales
print("\n12.2 RECUPERACIÓN DE LOS PRODUCTOS OFICIALES")

results = BENCHMARK_RESULTS["results"] # Recuperar resultados oficiales
ranking = BENCHMARK_RANKING_RESULT["ranking"] # Recuperar Ranking Científico
best_models = OFFICIAL_MODEL_BY_FAMILY_RESULT["models"] # Recuperar mejores modelos por familia
official = OFFICIAL_MODEL # Recuperar Modelo Oficial

print(f"Resultados del Benchmark : {len(results)}") # Mostrar cantidad de resultados
print(f"Modelos en el Ranking    : {len(ranking)}") # Mostrar cantidad de modelos clasificados
print(f"Familias evaluadas       : {len(best_models)}") # Mostrar cantidad de familias
print(f"Modelo Oficial           : {official['model_name']}") # Mostrar Modelo Oficial

# 12.3 Validación del Ranking Científico Oficial
print("\n12.3 RANKING CIENTÍFICO OFICIAL")

if not isinstance(ranking, list):
    raise TypeError("El Ranking Científico debe ser una lista.")

if len(ranking) == 0:
    raise ValueError("El Ranking Científico está vacío.")

if len(ranking) != len(results):
    raise ValueError("La cantidad de modelos del Ranking no coincide con la cantidad de resultados del Benchmark.")

if not isinstance(BENCHMARK_CONFIG, dict):
    raise TypeError("BENCHMARK_CONFIG debe ser un diccionario.")

ranking_metric = BENCHMARK_CONFIG.get("ranking_metric") # Recuperar métrica oficial del Ranking

if not isinstance(ranking_metric, str) or not ranking_metric.strip():
    raise ValueError("La métrica oficial del Ranking es inválida.")

metric_directions = BENCHMARK_CONFIG.get("metric_directions", {}) # Recuperar direcciones oficiales de optimización

if not isinstance(metric_directions, dict):
    raise TypeError("'metric_directions' debe ser un diccionario.")

ranking_direction = metric_directions.get(ranking_metric) # Recuperar dirección oficial de optimización

if ranking_direction not in {"min", "max"}:
    raise ValueError(f"La dirección de optimización de '{ranking_metric}' es inválida: {ranking_direction}")

if not isinstance(BENCHMARK_RANKING_RESULT, dict):
    raise TypeError("BENCHMARK_RANKING_RESULT debe ser un diccionario.")

if "metrics" not in BENCHMARK_RANKING_RESULT:
    raise ValueError("BENCHMARK_RANKING_RESULT no contiene las métricas oficiales.")

if ranking_metric not in BENCHMARK_RANKING_RESULT["metrics"]:
    raise ValueError(f"La métrica de ranking '{ranking_metric}' no está registrada en BENCHMARK_RANKING_RESULT.")

required_ranking_fields = [
    "ranking_position",
    "model_name",
    "family",
    ranking_metric,
] # Definir campos mínimos del Ranking

for position, result in enumerate(ranking, start=1):

    if not isinstance(result, dict):
        raise TypeError(f"El resultado {position} del Ranking debe ser un diccionario.")

    missing_fields = [
        field
        for field in required_ranking_fields
        if field not in result
    ] # Identificar campos faltantes

    if missing_fields:
        raise ValueError(f"El resultado {position} del Ranking está incompleto: {missing_fields}")

    if not isinstance(result["ranking_position"], (int, np.integer)):
        raise TypeError(f"La posición del resultado {position} debe ser un entero.")

    if result["ranking_position"] != position:
        raise ValueError(f"La posición del Ranking no coincide con el orden esperado en el resultado {position}.")

    if not isinstance(result["model_name"], str) or not result["model_name"].strip():
        raise ValueError(f"El modelo de la posición {position} tiene un nombre inválido.")

    if not isinstance(result["family"], str) or not result["family"].strip():
        raise ValueError(f"El modelo de la posición {position} tiene una familia inválida.")

    metric_value = result[ranking_metric] # Recuperar valor de la métrica de ranking

    if metric_value is None:
        raise ValueError(f"El modelo '{result['model_name']}' contiene la métrica '{ranking_metric}' con valor None.")

    if not np.isfinite(float(metric_value)):
        raise ValueError(f"El modelo '{result['model_name']}' contiene un valor inválido para '{ranking_metric}'.")

ranking_values = [
    float(result[ranking_metric])
    for result in ranking
] # Recuperar valores ordenados de la métrica oficial

if ranking_direction == "min":
    if ranking_values != sorted(ranking_values):
        raise ValueError(f"El Ranking Científico no está ordenado correctamente por '{ranking_metric}' en dirección ascendente.")
else:
    if ranking_values != sorted(ranking_values, reverse=True):
        raise ValueError(f"El Ranking Científico no está ordenado correctamente por '{ranking_metric}' en dirección descendente.")

print(f"Métrica de ranking         : {ranking_metric.upper()}") # Mostrar métrica principal
print(f"Dirección de optimización  : {ranking_direction.upper()}") # Mostrar dirección de optimización
print(f"Criterio de ranking        : {'Menor valor = mejor desempeño' if ranking_direction == 'min' else 'Mayor valor = mejor desempeño'}") # Mostrar criterio científico
print(f"Modelos clasificados       : {len(ranking)}") # Mostrar cantidad de modelos clasificados
print("Estructura del Ranking     : VALIDADA") # Confirmar estructura
print("Orden del Ranking          : VALIDADO") # Confirmar orden según métrica
print("Trazabilidad del criterio  : VALIDADA") # Confirmar trazabilidad del criterio

print("\nRanking Científico Oficial presentado correctamente.")

# 12.4 Mejores modelos por familia
print("\n12.4 CONSTRUCCIÓN DE LOS MEJORES MODELOS POR FAMILIA")

best_models = {} # Inicializar selección de mejores modelos por familia
for family in CANDIDATE_MODELS["families"]:
    family_results = [
        result
        for result in BENCHMARK_RANKING
        if result["family"].strip().lower()
        == family.strip().lower()
    ] # Filtrar resultados pertenecientes a la familia

    if not family_results:
        raise RuntimeError(
            f"La familia '{family}' no tiene resultados "
            "en BENCHMARK_RANKING."
        ) # Validar cobertura de la familia

    best_models[family] = min(
        family_results,
        key=lambda result: float(result["rmse"])
    ).copy() # Seleccionar mejor modelo de la familia por RMSE

print("\nAUDITORÍA DE best_models")
print(f"Cantidad de familias : {len(best_models)}") # Mostrar cantidad de familias

for family, result in best_models.items():
    print(
        f"{family} | "
        f"{result['model_code']} | "
        f"{result['model_name']} | "
        f"RMSE={result['rmse']:.6f} | "
        f"Ranking={result['ranking_position']}"
    ) # Mostrar mejor modelo por familia

# 12.5 Modelo Oficial del proyecto
print("\n12.5 MODELO OFICIAL DEL PROYECTO")

official_family = OFFICIAL_MODEL_FAMILY.strip().lower() # Recuperar familia oficial

if official_family not in best_models:
    raise RuntimeError(
        f"No existe un modelo seleccionado para la familia oficial "
        f"'{OFFICIAL_MODEL_FAMILY}'."
    ) # Verificar disponibilidad del mejor modelo GNN

official_candidate = best_models[official_family].copy() # Recuperar mejor modelo de la familia oficial

required_official_fields = [
    "model_code",
    "model_name",
    "family",
    "ranking_position",
    "rmse",
    "mae",
    "r2",
    "mape",
] # Definir campos obligatorios del Modelo Oficial

missing_official_fields = [
    field
    for field in required_official_fields
    if field not in official_candidate
] # Identificar campos faltantes

if missing_official_fields:
    raise ValueError(
        "El candidato del Modelo Oficial está incompleto: "
        f"{missing_official_fields}"
    ) # Validar integridad del candidato

if official_candidate["model_code"] != OFFICIAL_MODEL_CODE:
    raise ValueError(
        "El código seleccionado no coincide con OFFICIAL_MODEL_CODE."
    ) # Validar código oficial

if (
    official_candidate["model_name"].strip().lower()
    != OFFICIAL_MODEL_NAME.strip().lower()
):
    raise ValueError(
        "El nombre seleccionado no coincide con OFFICIAL_MODEL_NAME."
    ) # Validar nombre oficial

if (
    official_candidate["family"].strip().lower()
    != OFFICIAL_MODEL_FAMILY.strip().lower()
):
    raise ValueError(
        "La familia seleccionada no coincide con OFFICIAL_MODEL_FAMILY."
    ) # Validar familia oficial

if OFFICIAL_MODEL_NAME not in GNN_CONFIG:
    raise KeyError(
        f"No existe configuración GNN para el Modelo Oficial "
        f"'{OFFICIAL_MODEL_NAME}'."
    ) # Verificar configuración GNN

official_model_config = GNN_CONFIG[
    OFFICIAL_MODEL_NAME
].copy() # Recuperar configuración GNN oficial

if not isinstance(
    official_model_config,
    dict
):
    raise TypeError(
        "La configuración del Modelo Oficial debe ser un diccionario."
    ) # Validar estructura de configuración

required_config_fields = [
    "model_code",
    "model_name",
    "family",
] # Definir identidad mínima de la configuración

missing_config_fields = [
    field
    for field in required_config_fields
    if field not in official_model_config
] # Identificar campos faltantes

if missing_config_fields:
    raise ValueError(
        "La configuración GNN del Modelo Oficial está incompleta: "
        f"{missing_config_fields}"
    ) # Validar configuración

if official_model_config["model_code"] != OFFICIAL_MODEL_CODE:
    raise ValueError(
        "El código de GNN_CONFIG no coincide con OFFICIAL_MODEL_CODE."
    ) # Validar código interno

if (
    official_model_config["model_name"].strip().lower()
    != OFFICIAL_MODEL_NAME.strip().lower()
):
    raise ValueError(
        "El nombre de GNN_CONFIG no coincide con OFFICIAL_MODEL_NAME."
    ) # Validar nombre interno

if (
    official_model_config["family"].strip().lower()
    != OFFICIAL_MODEL_FAMILY.strip().lower()
):
    raise ValueError(
        "La familia de GNN_CONFIG no coincide con OFFICIAL_MODEL_FAMILY."
    ) # Validar familia interna

official = {
    "model_code": OFFICIAL_MODEL_CODE,
    "model_name": OFFICIAL_MODEL_NAME,
    "family": OFFICIAL_MODEL_FAMILY,
    "model_config": official_model_config,
    "ranking_position": official_candidate["ranking_position"],
    "rmse": official_candidate["rmse"],
    "mae": official_candidate["mae"],
    "r2": official_candidate["r2"],
    "mape": official_candidate["mape"],
} # Construir producto oficial completo

print(f"Código                   : {official['model_code']}") # Mostrar código oficial
print(f"Modelo                   : {official['model_name']}") # Mostrar nombre oficial
print(f"Familia                  : {official['family']}") # Mostrar familia oficial
print(f"Posición Ranking         : {official['ranking_position']}") # Mostrar posición global
print(f"RMSE                     : {official['rmse']:.6f}") # Mostrar RMSE
print(f"MAE                      : {official['mae']:.6f}") # Mostrar MAE
print(f"MAPE                     : {official['mape']:.6f}") # Mostrar MAPE
print(f"R2                       : {official['r2']:.6f}") # Mostrar R2

print("\nModelo Oficial validado y construido correctamente.") # Confirmar producto oficial

# 12.6 Fundamento metodológico de la selección
print("\n12.6 FUNDAMENTO METODOLÓGICO DE LA SELECCIÓN")

ranking_metric = BENCHMARK_CONFIG[
    "ranking_metric"
] # Recuperar métrica principal del Ranking Científico

ranking_direction = BENCHMARK_CONFIG[
    "metric_directions"
][
    ranking_metric
] # Recuperar dirección de optimización

if ranking_metric != "rmse":
    raise ValueError(
        f"La métrica principal esperada para la selección es 'rmse', "
        f"pero se encontró '{ranking_metric}'."
    ) # Validar métrica oficial

if ranking_direction != "min":
    raise ValueError(
        f"La dirección esperada para RMSE es 'min', "
        f"pero se encontró '{ranking_direction}'."
    ) # Validar dirección de optimización

if official["family"].strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise ValueError(
        "El Modelo Oficial no pertenece a la familia configurada "
        "como familia oficial."
    ) # Validar familia oficial

if official["model_code"] != OFFICIAL_MODEL_CODE:
    raise ValueError(
        "El código del Modelo Oficial no coincide con OFFICIAL_MODEL_CODE."
    ) # Validar código oficial

if official["model_name"].strip().lower() != OFFICIAL_MODEL_NAME.strip().lower():
    raise ValueError(
        "El nombre del Modelo Oficial no coincide con OFFICIAL_MODEL_NAME."
    ) # Validar nombre oficial

official_family_results = [
    result
    for result in BENCHMARK_RANKING
    if result["family"].strip().lower()
    == OFFICIAL_MODEL_FAMILY.strip().lower()
] # Recuperar modelos de la familia oficial

if not official_family_results:
    raise RuntimeError(
        "No existen resultados para la familia oficial "
        f"'{OFFICIAL_MODEL_FAMILY}'."
    ) # Validar cobertura de la familia oficial

best_official_family_result = min(
    official_family_results,
    key=lambda result: float(result[ranking_metric])
) # Identificar el mejor modelo de la familia oficial

if (
    best_official_family_result["model_code"]
    != official["model_code"]
):
    raise RuntimeError(
        "El Modelo Oficial no corresponde al mejor modelo "
        "de la familia GNN según la métrica oficial."
    ) # Validar criterio efectivo de selección

if (
    best_official_family_result["model_name"].strip().lower()
    != official["model_name"].strip().lower()
):
    raise RuntimeError(
        "El nombre del Modelo Oficial no coincide con el "
        "mejor modelo GNN identificado."
    ) # Validar identidad del modelo seleccionado

criteria = [
    "Precisión predictiva mediante RMSE, MAE, MAPE y R2",
    "Representación explícita de relaciones espaciales",
    "Capacidad de aprendizaje sobre estructuras de grafos",
    "Generalización espacial",
    "Compatibilidad con Forecasting espacio-temporal",
    "Compatibilidad con la Plataforma GeoAI",
    "Robustez metodológica",
    "Interpretabilidad científica",
] # Definir criterios científicos de justificación

print("La selección del Modelo Oficial se fundamenta en dos niveles.") # Presentar estructura metodológica

print(
    "Nivel 1: el Benchmark Científico permite comparar "
    "los 11 modelos bajo un protocolo experimental común."
) # Explicar comparación global

print(
    "Nivel 2: la selección del Modelo Oficial se restringe "
    "a la familia Graph Neural Networks, de acuerdo con "
    "el objetivo científico del proyecto."
) # Explicar restricción metodológica

print(f"Métrica principal         : {ranking_metric.upper()}") # Mostrar métrica principal
print(f"Dirección de optimización : {ranking_direction.upper()}") # Mostrar dirección de optimización
print("Criterio efectivo          : Mejor modelo GNN según RMSE.") # Mostrar criterio efectivo
print(f"Modelos GNN evaluados     : {len(official_family_results)}") # Mostrar cantidad de modelos GNN
print(f"Mejor modelo GNN          : {best_official_family_result['model_name']}") # Mostrar mejor GNN
print(f"Código                    : {best_official_family_result['model_code']}") # Mostrar código del mejor GNN
print(f"RMSE del mejor GNN        : {best_official_family_result['rmse']:.6f}") # Mostrar RMSE del mejor GNN
print(f"Posición Ranking Global   : {best_official_family_result['ranking_position']}") # Mostrar posición global
print(f"\nModelo seleccionado     : {official['model_name']}") # Mostrar Modelo Oficial
print(f"Familia                   : {official['family']}") # Mostrar familia oficial
print(f"Posición en el Ranking    : {official['ranking_position']}") # Mostrar posición global del Modelo Oficial
print(f"RMSE                      : {official['rmse']:.6f}") # Mostrar desempeño del Modelo Oficial
print("\nCriterios científicos de justificación:") # Presentar criterios científicos

for position, criterion in enumerate(
    criteria,
    start=1
):
    print(f"{position}. {criterion}") # Mostrar criterio científico

print("\nFundamento de selección : VALIDADO") # Confirmar fundamento metodológico
print("Restricción GNN           : VALIDADA") # Confirmar restricción de familia
print("Mejor GNN por RMSE        : VALIDADO") # Confirmar criterio efectivo
print("Modelo Oficial            : VALIDADO") # Confirmar selección final

# 12.7 Conclusión ejecutiva
print("\n12.7 CONCLUSIÓN EJECUTIVA")

ranking_metric = BENCHMARK_CONFIG[
    "ranking_metric"
] # Recuperar métrica principal del Benchmark

ranking_direction = BENCHMARK_CONFIG[
    "metric_directions"
][
    ranking_metric
] # Recuperar dirección de optimización

if ranking_metric != "rmse":
    raise ValueError(
        f"La métrica ejecutiva esperada es 'rmse', "
        f"pero se encontró '{ranking_metric}'."
    ) # Validar métrica principal

if ranking_direction != "min":
    raise ValueError(
        f"La dirección esperada para RMSE es 'min', "
        f"pero se encontró '{ranking_direction}'."
    ) # Validar dirección de optimización

if not benchmark_results:
    raise RuntimeError(
        "No existen resultados del Benchmark para construir la conclusión ejecutiva."
    ) # Validar resultados disponibles

if not BENCHMARK_RANKING:
    raise RuntimeError(
        "No existe Ranking Científico para construir la conclusión ejecutiva."
    ) # Validar ranking disponible

if not best_models:
    raise RuntimeError(
        "No existen mejores modelos por familia para construir la conclusión ejecutiva."
    ) # Validar selección por familia

if official is None:
    raise RuntimeError(
        "El Modelo Oficial no está disponible."
    ) # Validar Modelo Oficial

official_family = official[
    "family"
].strip().lower() # Recuperar familia del Modelo Oficial

official_family_results = [
    result
    for result in BENCHMARK_RANKING
    if result["family"].strip().lower() == official_family
] # Recuperar resultados de la familia oficial

if not official_family_results:
    raise RuntimeError(
        "No existen resultados para la familia del Modelo Oficial."
    ) # Validar resultados GNN

best_gnn = min(
    official_family_results,
    key=lambda result: float(result[ranking_metric])
) # Identificar mejor modelo GNN según RMSE

if best_gnn["model_code"] != official["model_code"]:
    raise RuntimeError(
        "El Modelo Oficial no coincide con el mejor modelo GNN "
        "según la métrica oficial."
    ) # Validar criterio efectivo

print(
    "El Benchmark Científico permitió evaluar de forma reproducible "
    "las diferentes familias de modelos incluidas en el protocolo experimental."
) # Resumir ejecución científica

print(
    "Los modelos fueron comparados bajo un protocolo común, utilizando "
    "la misma colección de datos, particiones temporales y métricas oficiales."
) # Confirmar comparabilidad experimental

print(
    f"La métrica principal utilizada para construir el Ranking Científico "
    f"fue {ranking_metric.upper()}, bajo el criterio de "
    f"{'menor valor representa mejor desempeño' if ranking_direction == 'min' else 'mayor valor representa mejor desempeño'}."
) # Explicar criterio cuantitativo

print(
    f"El Benchmark evaluó {len(BENCHMARK_RANKING)} modelos "
    f"distribuidos en {len(CANDIDATE_MODELS['families'])} familias."
) # Resumir cobertura experimental

print(
    f"El mejor modelo global según {ranking_metric.upper()} fue "
    f"{BENCHMARK_RANKING[0]['model_name']}, con un valor de "
    f"{BENCHMARK_RANKING[0][ranking_metric]:.6f}."
) # Informar mejor modelo global

print(
    f"El mejor modelo de la familia Graph Neural Networks fue "
    f"{best_gnn['model_name']}, con un RMSE de "
    f"{best_gnn['rmse']:.6f}."
) # Informar mejor modelo GNN

print(
    f"El Modelo Oficial seleccionado es {official['model_name']}, "
    f"perteneciente a la familia {official['family']}."
) # Identificar Modelo Oficial

print(
    f"La selección del Modelo Oficial se realizó exclusivamente dentro "
    f"de la familia {OFFICIAL_MODEL_FAMILY}, tomando como criterio efectivo "
    f"el mejor desempeño según el Ranking Científico."
) # Explicar restricción metodológica

print(
    f"El Modelo Oficial ocupa la posición "
    f"{official['ranking_position']} del Ranking Científico global."
) # Informar posición global

print(
    "La adopción de una arquitectura Graph Neural Network como Modelo Oficial "
    "responde al objetivo científico de representar explícitamente las relaciones "
    "espaciales entre los municipios mediante estructuras de grafos."
) # Justificar selección de GNN

print(
    "El Modelo Oficial constituye la base para las etapas posteriores "
    "de entrenamiento, evaluación, forecasting y despliegue de la Plataforma GeoAI."
) # Conectar con etapas posteriores

print("\nConclusión ejecutiva        : VALIDADA") # Confirmar conclusión

# 12.8 Construcción del producto oficial del reporte
print("\n12.8 CONSTRUCCIÓN DEL REPORTE EJECUTIVO")

BENCHMARK_EXECUTIVE_REPORT = {
    "total_results": len(benchmark_results),
    "total_models": len(BENCHMARK_RANKING),
    "total_families": len(CANDIDATE_MODELS["families"]),
    "ranking_metric": ranking_metric,
    "ranking_direction": ranking_direction,
    "ranking": BENCHMARK_RANKING,
    "best_models_by_family": best_models,
    "official_model": official,
    "official_selection_criterion": (
        "Mejor modelo de la familia Graph Neural Networks "
        "según la métrica oficial RMSE"
    ),
    "selection_criteria": criteria,
    "global_best_model": BENCHMARK_RANKING[0],
    "best_gnn_model": best_gnn,
    "status": "VALIDATED",
} # Construir producto oficial del Reporte Ejecutivo

print(f"Resultados incluidos       : {BENCHMARK_EXECUTIVE_REPORT['total_results']}") # Mostrar resultados
print(f"Modelos incluidos          : {BENCHMARK_EXECUTIVE_REPORT['total_models']}") # Mostrar modelos
print(f"Familias incluidas         : {BENCHMARK_EXECUTIVE_REPORT['total_families']}") # Mostrar familias
print("Producto ejecutivo          : CONSTRUIDO") # Confirmar construcción

# 12.9 Validación del producto oficial
print("\n12.9 VALIDACIÓN DEL REPORTE EJECUTIVO")

required_report_fields = [
    "total_results",
    "total_models",
    "total_families",
    "ranking_metric",
    "ranking_direction",
    "ranking",
    "best_models_by_family",
    "official_model",
    "official_selection_criterion",
    "selection_criteria",
    "global_best_model",
    "best_gnn_model",
    "status",
] # Definir contrato completo del reporte ejecutivo

missing_report_fields = [
    field
    for field in required_report_fields
    if field not in BENCHMARK_EXECUTIVE_REPORT
] # Identificar campos faltantes

if missing_report_fields:
    raise RuntimeError(
        "BENCHMARK_EXECUTIVE_REPORT está incompleto: "
        f"{missing_report_fields}"
    ) # Validar contrato completo

if BENCHMARK_EXECUTIVE_REPORT[
    "status"
] != "VALIDATED":
    raise RuntimeError(
        "El Reporte Ejecutivo del Benchmark no fue validado."
    ) # Validar estado oficial

if BENCHMARK_EXECUTIVE_REPORT[
    "total_results"
] != len(benchmark_results):
    raise RuntimeError(
        "El total de resultados del reporte no coincide "
        "con benchmark_results."
    ) # Validar resultados

if BENCHMARK_EXECUTIVE_REPORT[
    "total_models"
] != len(BENCHMARK_RANKING):
    raise RuntimeError(
        "El total de modelos del reporte no coincide "
        "con BENCHMARK_RANKING."
    ) # Validar ranking

if BENCHMARK_EXECUTIVE_REPORT[
    "total_families"
] != len(CANDIDATE_MODELS["families"]):
    raise RuntimeError(
        "El total de familias del reporte no coincide "
        "con CANDIDATE_MODELS."
    ) # Validar familias

if len(BENCHMARK_RANKING) != len(benchmark_results):
    raise RuntimeError(
        "El Ranking Científico no contiene todos los resultados."
    ) # Validar cobertura del ranking

expected_families = {
    family.strip().lower()
    for family in CANDIDATE_MODELS["families"]
} # Recuperar familias esperadas

observed_families = {
    result["family"].strip().lower()
    for result in best_models.values()
} # Recuperar familias seleccionadas

if expected_families != observed_families:
    raise RuntimeError(
        "best_models no contiene exactamente todas las familias oficiales."
    ) # Validar cobertura de familias

official_report_model = BENCHMARK_EXECUTIVE_REPORT[
    "official_model"
] # Recuperar Modelo Oficial del reporte

if official_report_model[
    "model_code"
] != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El código del Modelo Oficial no coincide con OFFICIAL_MODEL_CODE."
    ) # Validar código oficial

if official_report_model[
    "model_name"
].strip().lower() != OFFICIAL_MODEL_NAME.strip().lower():
    raise RuntimeError(
        "El nombre del Modelo Oficial no coincide con OFFICIAL_MODEL_NAME."
    ) # Validar nombre oficial

if official_report_model[
    "family"
].strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise RuntimeError(
        "La familia del Modelo Oficial no coincide con OFFICIAL_MODEL_FAMILY."
    ) # Validar familia oficial

report_best_gnn = BENCHMARK_EXECUTIVE_REPORT[
    "best_gnn_model"
] # Recuperar mejor modelo GNN

if report_best_gnn[
    "model_code"
] != official_report_model[
    "model_code"
]:
    raise RuntimeError(
        "El Modelo Oficial no coincide con el mejor modelo GNN del reporte."
    ) # Validar selección oficial

if float(report_best_gnn["rmse"]) != float(
    official_report_model["rmse"]
):
    raise RuntimeError(
        "El RMSE del Modelo Oficial no coincide con el RMSE "
        "del mejor modelo GNN."
    ) # Validar métrica oficial

if report_best_gnn[
    "ranking_position"
] != official_report_model[
    "ranking_position"
]:
    raise RuntimeError(
        "La posición del Modelo Oficial no coincide con "
        "la posición del mejor modelo GNN."
    ) # Validar posición global

if BENCHMARK_EXECUTIVE_REPORT[
    "ranking_metric"
] != BENCHMARK_CONFIG[
    "ranking_metric"
]:
    raise RuntimeError(
        "La métrica del reporte no coincide con BENCHMARK_CONFIG."
    ) # Validar configuración de ranking

if BENCHMARK_EXECUTIVE_REPORT[
    "ranking_direction"
] != BENCHMARK_CONFIG[
    "metric_directions"
][
    BENCHMARK_CONFIG["ranking_metric"]
]:
    raise RuntimeError(
        "La dirección del reporte no coincide con BENCHMARK_CONFIG."
    ) # Validar dirección de ranking

print("Contrato del reporte        : VALIDADO") # Confirmar contrato
print("Cobertura de resultados     : VALIDADA") # Confirmar resultados
print("Cobertura de modelos        : VALIDADA") # Confirmar modelos
print("Cobertura de familias       : VALIDADA") # Confirmar familias
print("Modelo Oficial              : VALIDADO") # Confirmar Modelo Oficial
print("Mejor modelo GNN            : VALIDADO") # Confirmar selección GNN
print("Métrica de ranking          : VALIDADA") # Confirmar métrica
print("Dirección de ranking        : VALIDADA") # Confirmar dirección
print("Estado del reporte          : VALIDADO") # Confirmar estado final

# 12.10 Confirmación final del Bloque 12
print("\n" + "-" * 80)
print("REPORTE EJECUTIVO DEL BENCHMARK CIENTÍFICO")
print("-" * 80)

print(f"Resultados obtenidos       : {BENCHMARK_EXECUTIVE_REPORT['total_results']}") # Mostrar resultados
print(f"Modelos evaluados          : {BENCHMARK_EXECUTIVE_REPORT['total_models']}") # Mostrar modelos
print(f"Familias evaluadas         : {BENCHMARK_EXECUTIVE_REPORT['total_families']}") # Mostrar familias
print(f"Métrica de ranking         : {BENCHMARK_EXECUTIVE_REPORT['ranking_metric'].upper()}") # Mostrar métrica
print(f"Dirección de ranking       : {BENCHMARK_EXECUTIVE_REPORT['ranking_direction'].upper()}") # Mostrar dirección
print(f"Mejor modelo global        : {BENCHMARK_EXECUTIVE_REPORT['global_best_model']['model_name']}") # Mostrar mejor modelo global
print(f"RMSE mejor modelo global   : {BENCHMARK_EXECUTIVE_REPORT['global_best_model']['rmse']:.6f}") # Mostrar RMSE global
print(f"Mejor modelo GNN           : {BENCHMARK_EXECUTIVE_REPORT['best_gnn_model']['model_name']}") # Mostrar mejor GNN
print(f"RMSE mejor GNN             : {BENCHMARK_EXECUTIVE_REPORT['best_gnn_model']['rmse']:.6f}") # Mostrar RMSE GNN
print(f"Modelo Oficial             : {BENCHMARK_EXECUTIVE_REPORT['official_model']['model_name']}") # Mostrar Modelo Oficial
print(f"Código Oficial             : {BENCHMARK_EXECUTIVE_REPORT['official_model']['model_code']}") # Mostrar código oficial
print(f"Familia Oficial            : {BENCHMARK_EXECUTIVE_REPORT['official_model']['family']}") # Mostrar familia oficial
print(f"Posición Ranking Global    : {BENCHMARK_EXECUTIVE_REPORT['official_model']['ranking_position']}") # Mostrar posición global
print(f"RMSE Modelo Oficial        : {BENCHMARK_EXECUTIVE_REPORT['official_model']['rmse']:.6f}") # Mostrar RMSE oficial
print("Criterio de selección       : Mejor modelo GNN según RMSE") # Mostrar criterio oficial
print("Estado                      : REPORTE VALIDADO") # Confirmar validación

print("-" * 80)

print("\nBloque 12. Reporte Ejecutivo del Benchmark generado y validado correctamente.") # Confirmar finalización del Bloque 12

# BLOQUE 13. EXPORTACIÓN OFICIAL DEL BENCHMARK
# Objetivo: Exportar los productos científicos oficiales generados durante el Benchmark Científico para
# garantizar reproducibilidad, trazabilidad y transferencia controlada de la configuración del
# Modelo Oficial hacia las etapas posteriores del proyecto.
# Entradas: - BENCHMARK_RESULTS - BENCHMARK_RANKING_RESULT - OFFICIAL_MODEL_BY_FAMILY_RESULT
# - OFFICIAL_MODEL - BENCHMARK_DATA
# Producto: - BENCHMARK_EXPORT_RESULT - official_model_config.json - benchmark_experiment.joblib
# - benchmark_metrics.parquet - benchmark_ranking.csv - benchmark_ranking.xlsx
# - benchmark_summary.csv - benchmark_summary.xlsx
# Pregunta científica: ¿Los productos oficiales del Benchmark fueron exportados correctamente y la
# configuración del Modelo Oficial quedó disponible para las etapas posteriores del proyecto?

# 13.1 Validación de los productos oficiales de entrada
print("\n13.1 VALIDACIÓN DE LOS PRODUCTOS OFICIALES DE ENTRADA")

required_export_products = {
    "benchmark_results": benchmark_results,
    "BENCHMARK_RANKING": BENCHMARK_RANKING,
    "best_models": best_models,
    "official": official,
    "BENCHMARK_DATA": BENCHMARK_DATA,
    "BENCHMARK_EXECUTIVE_REPORT": BENCHMARK_EXECUTIVE_REPORT,
} # Definir productos oficiales requeridos para exportación

missing_export_products = [
    name
    for name, value in required_export_products.items()
    if value is None
] # Identificar productos faltantes

if missing_export_products:
    raise ValueError(
        "Faltan productos oficiales requeridos para la exportación: "
        f"{missing_export_products}"
    ) # Validar disponibilidad de productos

if not isinstance(BENCHMARK_EXECUTIVE_REPORT, dict):
    raise TypeError(
        "BENCHMARK_EXECUTIVE_REPORT debe ser un diccionario."
    ) # Validar estructura del reporte ejecutivo

if BENCHMARK_EXECUTIVE_REPORT.get("status") != "VALIDATED":
    raise RuntimeError(
        "BENCHMARK_EXECUTIVE_REPORT no está validado."
    ) # Validar estado del producto ejecutivo

if not isinstance(benchmark_results, list) or not benchmark_results:
    raise ValueError(
        "benchmark_results debe ser una lista no vacía."
    ) # Validar resultados del Benchmark

if not isinstance(BENCHMARK_RANKING, list) or not BENCHMARK_RANKING:
    raise ValueError(
        "BENCHMARK_RANKING debe ser una lista no vacía."
    ) # Validar Ranking Científico

if not isinstance(best_models, dict) or not best_models:
    raise ValueError(
        "best_models debe ser un diccionario no vacío."
    ) # Validar mejores modelos por familia

if not isinstance(official, dict):
    raise TypeError(
        "official debe ser un diccionario."
    ) # Validar Modelo Oficial

if not isinstance(BENCHMARK_DATA, dict):
    raise TypeError(
        "BENCHMARK_DATA debe ser un diccionario."
    ) # Validar BenchmarkData

print(f"Resultados Benchmark       : {len(benchmark_results)}") # Mostrar resultados
print(f"Modelos Ranking            : {len(BENCHMARK_RANKING)}") # Mostrar modelos
print(f"Familias seleccionadas     : {len(best_models)}") # Mostrar familias
print(f"Modelo Oficial             : {official['model_name']}") # Mostrar Modelo Oficial

# 13.2 Validación de la identidad del Modelo Oficial
print("\n13.2 VALIDACIÓN DEL MODELO OFICIAL")

required_official_fields = [
    "model_code",
    "model_name",
    "family",
    "model_config",
    "ranking_position",
    "rmse",
    "mae",
    "r2",
    "mape",
] # Definir contrato completo del Modelo Oficial

missing_official_fields = [
    field
    for field in required_official_fields
    if field not in official
] # Identificar campos faltantes

if missing_official_fields:
    raise ValueError(
        "El Modelo Oficial está incompleto: "
        f"{missing_official_fields}"
    ) # Validar integridad del Modelo Oficial

if official["model_code"] != OFFICIAL_MODEL_CODE:
    raise ValueError(
        "El código del Modelo Oficial no coincide con "
        "OFFICIAL_MODEL_CODE."
    ) # Validar código oficial

if official["model_name"].strip().lower() != OFFICIAL_MODEL_NAME.strip().lower():
    raise ValueError(
        "El nombre del Modelo Oficial no coincide con "
        "OFFICIAL_MODEL_NAME."
    ) # Validar nombre oficial

if official["family"].strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise ValueError(
        "La familia del Modelo Oficial no coincide con "
        "OFFICIAL_MODEL_FAMILY."
    ) # Validar familia oficial

if not isinstance(official["model_config"], dict):
    raise TypeError(
        "official['model_config'] debe ser un diccionario."
    ) # Validar configuración oficial

official_family_results = [
    result
    for result in BENCHMARK_RANKING
    if result["family"].strip().lower()
    == OFFICIAL_MODEL_FAMILY.strip().lower()
] # Recuperar modelos de la familia oficial

if not official_family_results:
    raise RuntimeError(
        "No existen resultados de la familia oficial en "
        "BENCHMARK_RANKING."
    ) # Validar cobertura de la familia oficial

best_official_family_model = min(
    official_family_results,
    key=lambda result: float(result["rmse"])
) # Identificar mejor modelo GNN por RMSE

if best_official_family_model["model_code"] != official["model_code"]:
    raise RuntimeError(
        "El Modelo Oficial no corresponde al mejor modelo "
        "de la familia Graph Neural Networks según RMSE."
    ) # Validar criterio efectivo de selección

if best_official_family_model["ranking_position"] != official["ranking_position"]:
    raise RuntimeError(
        "La posición del Modelo Oficial no coincide con "
        "la posición del mejor modelo GNN."
    ) # Validar posición global

print(f"Modelo Oficial              : {official['model_name']}") # Mostrar modelo oficial
print(f"Código Oficial              : {official['model_code']}") # Mostrar código oficial
print(f"Familia Oficial             : {official['family']}") # Mostrar familia oficial
print(f"Posición Ranking            : {official['ranking_position']}") # Mostrar posición global
print(f"RMSE Oficial                : {official['rmse']:.6f}") # Mostrar desempeño oficial
print("Identidad del Modelo Oficial : VALIDADA") # Confirmar identidad
print("Selección dentro de GNN      : VALIDADA") # Confirmar criterio de selección

# 13.3 Validación de BenchmarkData
print("\n13.3 VALIDACIÓN DE BenchmarkData")

required_benchmark_data = [
    "graphs",
    "train_index",
    "validation_index",
    "test_index",
    "scaler",
] # Definir contrato de BenchmarkData

missing_benchmark_data = [
    field
    for field in required_benchmark_data
    if field not in BENCHMARK_DATA
] # Identificar campos faltantes

if missing_benchmark_data:
    raise ValueError(
        "BENCHMARK_DATA está incompleto: "
        f"{missing_benchmark_data}"
    ) # Validar estructura

graphs = BENCHMARK_DATA["graphs"] # Recuperar GraphData
train_index = BENCHMARK_DATA["train_index"] # Recuperar índices de entrenamiento
validation_index = BENCHMARK_DATA["validation_index"] # Recuperar índices de validación
test_index = BENCHMARK_DATA["test_index"] # Recuperar índices de prueba

if not isinstance(graphs, (list, tuple)):
    raise TypeError(
        "BENCHMARK_DATA['graphs'] debe ser una lista o tupla."
    ) # Validar GraphData

if len(graphs) == 0:
    raise ValueError(
        "BENCHMARK_DATA['graphs'] está vacío."
    ) # Validar existencia de GraphData

if len(train_index) == 0:
    raise ValueError(
        "El conjunto de entrenamiento está vacío."
    ) # Validar entrenamiento

if len(validation_index) == 0:
    raise ValueError(
        "El conjunto de validación está vacío."
    ) # Validar validación

if len(test_index) == 0:
    raise ValueError(
        "El conjunto de prueba está vacío."
    ) # Validar prueba

all_indices = (
    list(train_index)
    + list(validation_index)
    + list(test_index)
) # Consolidar índices temporales

if len(all_indices) != len(set(all_indices)):
    raise RuntimeError(
        "Existen índices GraphData duplicados entre "
        "entrenamiento, validación y prueba."
    ) # Validar particiones disjuntas

expected_indices = list(range(len(graphs))) # Construir índices esperados
if sorted(all_indices) != expected_indices:
    raise RuntimeError(
        "Las particiones temporales no cubren exactamente "
        "todos los GraphData."
    ) # Validar cobertura temporal

if BENCHMARK_DATA["scaler"] is None:
    raise ValueError(
        "El escalador oficial del Benchmark es inválido."
    ) # Validar escalador

print(f"GraphData                  : {len(graphs)}") # Mostrar GraphData
print(f"Entrenamiento              : {len(train_index)}") # Mostrar entrenamiento
print(f"Validación                 : {len(validation_index)}") # Mostrar validación
print(f"Prueba                     : {len(test_index)}") # Mostrar prueba
print(f"Escalador                  : {type(BENCHMARK_DATA['scaler']).__name__}") # Mostrar escalador
print("Particiones temporales      : VALIDADAS") # Confirmar particiones
print("BenchmarkData               : VALIDADO") # Confirmar BenchmarkData

# 13.4 Validación de cobertura del Ranking Científico
print("\n13.4 VALIDACIÓN DEL RANKING CIENTÍFICO")

if "ranking" not in BENCHMARK_EXECUTIVE_REPORT:
    raise RuntimeError(
        "BENCHMARK_EXECUTIVE_REPORT no contiene el Ranking Científico."
    ) # Validar disponibilidad del ranking oficial

benchmark_ranking_export = BENCHMARK_EXECUTIVE_REPORT[
    "ranking"
] # Recuperar Ranking Científico oficial

if not isinstance(
    benchmark_ranking_export,
    list
):
    raise TypeError(
        "El Ranking Científico debe ser una lista."
    ) # Validar estructura del ranking

if len(benchmark_ranking_export) == 0:
    raise RuntimeError(
        "El Ranking Científico está vacío."
    ) # Validar existencia del ranking

if len(benchmark_ranking_export) != len(
    benchmark_results
):
    raise RuntimeError(
        "El Ranking Científico no contiene todos los resultados "
        "del Benchmark."
    ) # Validar cobertura completa

expected_positions = list(
    range(
        1,
        len(benchmark_ranking_export) + 1
    )
) # Definir posiciones esperadas

observed_positions = [
    result["ranking_position"]
    for result in benchmark_ranking_export
] # Recuperar posiciones observadas

if observed_positions != expected_positions:
    raise RuntimeError(
        "Las posiciones del Ranking Científico no son consecutivas."
    ) # Validar numeración consecutiva

ranking_identities = [
    (
        result["model_code"],
        result["model_name"].strip().lower(),
        result["family"].strip().lower(),
    )
    for result in benchmark_ranking_export
] # Construir identidades del ranking

if len(ranking_identities) != len(
    set(ranking_identities)
):
    raise RuntimeError(
        "El Ranking Científico contiene modelos duplicados."
    ) # Validar unicidad de modelos

ranking_metric = BENCHMARK_EXECUTIVE_REPORT[
    "ranking_metric"
] # Recuperar métrica oficial del reporte

ranking_direction = BENCHMARK_EXECUTIVE_REPORT[
    "ranking_direction"
] # Recuperar dirección oficial del reporte

if ranking_metric != BENCHMARK_CONFIG[
    "ranking_metric"
]:
    raise RuntimeError(
        "La métrica del Ranking Científico no coincide "
        "con BENCHMARK_CONFIG."
    ) # Validar trazabilidad de la métrica

if ranking_direction != BENCHMARK_CONFIG[
    "metric_directions"
][
    ranking_metric
]:
    raise RuntimeError(
        "La dirección del Ranking Científico no coincide "
        "con BENCHMARK_CONFIG."
    ) # Validar trazabilidad de la dirección

ranking_values = [
    float(result[ranking_metric])
    for result in benchmark_ranking_export
] # Recuperar valores de la métrica oficial

if ranking_direction == "min":
    if ranking_values != sorted(
        ranking_values
    ):
        raise RuntimeError(
            f"El Ranking Científico no está ordenado correctamente "
            f"por '{ranking_metric}' en dirección ascendente."
        ) # Validar orden ascendente

elif ranking_direction == "max":
    if ranking_values != sorted(
        ranking_values,
        reverse=True
    ):
        raise RuntimeError(
            f"El Ranking Científico no está ordenado correctamente "
            f"por '{ranking_metric}' en dirección descendente."
        ) # Validar orden descendente
else:
    raise RuntimeError(
        f"Dirección de ranking inválida: {ranking_direction}"
    ) # Validar dirección

print(f"Modelos en Ranking          : {len(benchmark_ranking_export)}") # Mostrar modelos
print(f"Métrica de Ranking          : {ranking_metric.upper()}") # Mostrar métrica
print(f"Dirección                   : {ranking_direction.upper()}") # Mostrar dirección
print("Cobertura de modelos         : VALIDADA") # Confirmar cobertura
print("Numeración del ranking       : VALIDADA") # Confirmar numeración
print("Unicidad de modelos          : VALIDADA") # Confirmar unicidad
print("Orden del ranking            : VALIDADO") # Confirmar orden
print("Trazabilidad de configuración: VALIDADA") # Confirmar coherencia con BENCHMARK_CONFIG
print("Ranking Científico           : VALIDADO") # Confirmar ranking oficial

# 13.5 Validación de mejores modelos por familia
print("\n13.5 VALIDACIÓN DE MEJORES MODELOS POR FAMILIA")

if "best_models_by_family" not in BENCHMARK_EXECUTIVE_REPORT:
    raise RuntimeError(
        "BENCHMARK_EXECUTIVE_REPORT no contiene "
        "'best_models_by_family'."
    ) # Validar disponibilidad de selección por familia

best_models_export = BENCHMARK_EXECUTIVE_REPORT[
    "best_models_by_family"
] # Recuperar selección oficial por familia

if not isinstance(
    best_models_export,
    dict
):
    raise TypeError(
        "'best_models_by_family' debe ser un diccionario."
    ) # Validar estructura

configured_families = {
    family.strip().lower()
    for family in CANDIDATE_MODELS["families"]
} # Recuperar familias oficiales

observed_families = {
    family.strip().lower()
    for family in best_models_export
} # Recuperar familias seleccionadas

if configured_families != observed_families:
    raise RuntimeError(
        "La selección de mejores modelos no contiene exactamente "
        "las familias configuradas en CANDIDATE_MODELS."
    ) # Validar cobertura completa de familias

for family, result in best_models_export.items():

    if not isinstance(
        result,
        dict
    ):
        raise TypeError(
            f"El mejor modelo de la familia '{family}' "
            "debe ser un diccionario."
        ) # Validar estructura del resultado

    required_family_fields = [
        "model_code",
        "model_name",
        "family",
        "ranking_position",
        "rmse",
        "mae",
        "r2",
        "mape",
    ] # Definir contrato del mejor modelo

    missing_family_fields = [
        field
        for field in required_family_fields
        if field not in result
    ] # Identificar campos faltantes

    if missing_family_fields:
        raise ValueError(
            f"El mejor modelo de la familia '{family}' "
            f"está incompleto: {missing_family_fields}"
        ) # Validar integridad del producto

    if result["family"].strip().lower() != family.strip().lower():
        raise ValueError(
            f"El modelo seleccionado para '{family}' "
            "no pertenece a dicha familia."
        ) # Validar pertenencia familiar

    family_results = [
        ranking_result
        for ranking_result in benchmark_ranking_export
        if ranking_result["family"].strip().lower()
        == family.strip().lower()
    ] # Recuperar resultados oficiales de la familia

    if not family_results:
        raise RuntimeError(
            f"La familia '{family}' no contiene resultados "
            "en el Ranking Científico."
        ) # Validar existencia de resultados

    expected_best = min(
        family_results,
        key=lambda ranking_result: float(
            ranking_result["rmse"]
        )
    ) # Determinar mejor modelo esperado por RMSE

    if result["model_code"] != expected_best["model_code"]:
        raise RuntimeError(
            f"best_models seleccionó incorrectamente "
            f"la familia '{family}'."
        ) # Validar selección efectiva por RMSE

    if result["ranking_position"] != expected_best["ranking_position"]:
        raise RuntimeError(
            f"La posición del modelo seleccionado para "
            f"la familia '{family}' no coincide con el Ranking Científico."
        ) # Validar trazabilidad de posición

    if not np.isclose(
        float(result["rmse"]),
        float(expected_best["rmse"])
    ):
        raise RuntimeError(
            f"El RMSE del modelo seleccionado para "
            f"la familia '{family}' no coincide con el Ranking Científico."
        ) # Validar trazabilidad de métrica

print(f"Familias seleccionadas      : {len(best_models_export)}") # Mostrar familias
print("Cobertura de familias       : VALIDADA") # Confirmar cobertura
print("Selección por RMSE          : VALIDADA") # Confirmar criterio de selección
print("Trazabilidad con Ranking    : VALIDADA") # Confirmar correspondencia con ranking
print("Mejores modelos por familia : VALIDADOS") # Confirmar selección

# 13.6 Construcción de la configuración exportable del Modelo Oficial
print("\n13.6 CONSTRUCCIÓN DE LA CONFIGURACIÓN DEL MODELO OFICIAL")

required_export_fields = [
    "model_code",
    "model_name",
    "family",
    "model_config",
    "ranking_position",
    "rmse",
    "mae",
    "r2",
    "mape",
] # Definir contrato mínimo de exportación

missing_export_fields = [
    field
    for field in required_export_fields
    if field not in official
] # Identificar campos faltantes

if missing_export_fields:
    raise RuntimeError(
        "El Modelo Oficial no contiene todos los campos "
        "requeridos para exportación: "
        f"{missing_export_fields}"
    ) # Validar integridad del Modelo Oficial

if not isinstance(
    official["model_config"],
    dict
):
    raise TypeError(
        "official['model_config'] debe ser un diccionario."
    ) # Validar configuración del modelo

if official["model_code"] != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El código del Modelo Oficial no coincide con "
        "OFFICIAL_MODEL_CODE."
    ) # Validar código oficial

if official["model_name"].strip().lower() != OFFICIAL_MODEL_NAME.strip().lower():
    raise RuntimeError(
        "El nombre del Modelo Oficial no coincide con "
        "OFFICIAL_MODEL_NAME."
    ) # Validar nombre oficial

if official["family"].strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise RuntimeError(
        "La familia del Modelo Oficial no coincide con "
        "OFFICIAL_MODEL_FAMILY."
    ) # Validar familia oficial

if "official_model" not in BENCHMARK_EXECUTIVE_REPORT:
    raise RuntimeError(
        "BENCHMARK_EXECUTIVE_REPORT no contiene "
        "el Modelo Oficial."
    ) # Validar disponibilidad del producto consolidado

reported_official = BENCHMARK_EXECUTIVE_REPORT[
    "official_model"
] # Recuperar Modelo Oficial consolidado

if reported_official["model_code"] != official["model_code"]:
    raise RuntimeError(
        "El código de official no coincide con "
        "BENCHMARK_EXECUTIVE_REPORT."
    ) # Validar trazabilidad del código

if reported_official["model_name"].strip().lower() != official["model_name"].strip().lower():
    raise RuntimeError(
        "El nombre de official no coincide con "
        "BENCHMARK_EXECUTIVE_REPORT."
    ) # Validar trazabilidad del nombre

if reported_official["family"].strip().lower() != official["family"].strip().lower():
    raise RuntimeError(
        "La familia de official no coincide con "
        "BENCHMARK_EXECUTIVE_REPORT."
    ) # Validar trazabilidad de la familia

ranking_metric = BENCHMARK_EXECUTIVE_REPORT[
    "ranking_metric"
] # Recuperar métrica oficial consolidada

ranking_direction = BENCHMARK_EXECUTIVE_REPORT[
    "ranking_direction"
] # Recuperar dirección oficial consolidada

if ranking_metric != BENCHMARK_CONFIG["ranking_metric"]:
    raise RuntimeError(
        "La métrica de ranking del reporte no coincide "
        "con BENCHMARK_CONFIG."
    ) # Validar trazabilidad de la métrica

if ranking_direction != BENCHMARK_CONFIG[
    "metric_directions"
][
    ranking_metric
]:
    raise RuntimeError(
        "La dirección de ranking del reporte no coincide "
        "con BENCHMARK_CONFIG."
    ) # Validar trazabilidad de la dirección

official_model_config_export = {
    "model_code": official["model_code"],
    "model_name": official["model_name"],
    "family": official["family"],
    "model_config": official["model_config"].copy(),
    "ranking_position": int(
        official["ranking_position"]
    ),
    "rmse": float(
        official["rmse"]
    ),
    "mae": float(
        official["mae"]
    ),
    "r2": float(
        official["r2"]
    ),
    "mape": float(
        official["mape"]
    ),
    "selection_source": "benchmark_scientific",
    "selection_scope": OFFICIAL_MODEL_FAMILY,
    "ranking_metric": ranking_metric,
    "ranking_direction": ranking_direction,
    "benchmark_models_evaluated": len(
        benchmark_results
    ),
    "benchmark_families_evaluated": len(
        best_models
    ),
    "status": "OFFICIAL",
} # Construir configuración exportable oficial

required_config_export_fields = [
    "model_code",
    "model_name",
    "family",
    "model_config",
    "ranking_position",
    "rmse",
    "mae",
    "r2",
    "mape",
    "selection_source",
    "selection_scope",
    "ranking_metric",
    "ranking_direction",
    "benchmark_models_evaluated",
    "benchmark_families_evaluated",
    "status",
] # Definir contrato del producto exportable

missing_config_export_fields = [
    field
    for field in required_config_export_fields
    if field not in official_model_config_export
] # Identificar campos faltantes

if missing_config_export_fields:
    raise RuntimeError(
        "La configuración exportable del Modelo Oficial "
        "está incompleta: "
        f"{missing_config_export_fields}"
    ) # Validar producto exportable

if official_model_config_export["status"] != "OFFICIAL":
    raise RuntimeError(
        "La configuración exportable no está marcada "
        "como OFFICIAL."
    ) # Validar estado oficial

if official_model_config_export[
    "benchmark_models_evaluated"
] != len(
    benchmark_results
):
    raise RuntimeError(
        "La cantidad de modelos evaluados registrada "
        "en la configuración no coincide con benchmark_results."
    ) # Validar cantidad de modelos

if official_model_config_export[
    "benchmark_families_evaluated"
] != len(
    CANDIDATE_MODELS["families"]
):
    raise RuntimeError(
        "La cantidad de familias evaluadas registrada "
        "en la configuración no coincide con el catálogo oficial."
    ) # Validar cantidad de familias

print(f"Modelo Oficial             : {official_model_config_export['model_name']}") # Mostrar modelo
print(f"Código                     : {official_model_config_export['model_code']}") # Mostrar código
print(f"Familia                    : {official_model_config_export['family']}") # Mostrar familia
print(f"Posición Ranking           : {official_model_config_export['ranking_position']}") # Mostrar posición
print(f"RMSE                       : {official_model_config_export['rmse']:.6f}") # Mostrar RMSE
print(f"MAE                        : {official_model_config_export['mae']:.6f}") # Mostrar MAE
print(f"R2                         : {official_model_config_export['r2']:.6f}") # Mostrar R2
print(f"MAPE                       : {official_model_config_export['mape']:.6f}") # Mostrar MAPE
print(f"Modelos evaluados          : {official_model_config_export['benchmark_models_evaluated']}") # Mostrar cobertura de modelos
print(f"Familias evaluadas         : {official_model_config_export['benchmark_families_evaluated']}") # Mostrar cobertura de familias
print(f"Métrica de ranking         : {official_model_config_export['ranking_metric'].upper()}") # Mostrar métrica de ranking
print(f"Dirección de ranking       : {official_model_config_export['ranking_direction'].upper()}") # Mostrar dirección de ranking
print("Configuración Oficial       : VALIDADA") # Confirmar configuración oficial

# 13.7 Construcción de las tablas de exportación
print("\n13.7 CONSTRUCCIÓN DE LOS PRODUCTOS DE EXPORTACIÓN")

if "ranking" not in BENCHMARK_EXECUTIVE_REPORT:
    raise RuntimeError(
        "BENCHMARK_EXECUTIVE_REPORT no contiene el Ranking Científico."
    ) # Validar disponibilidad del ranking

benchmark_ranking_export = BENCHMARK_EXECUTIVE_REPORT[
    "ranking"
] # Recuperar Ranking Científico consolidado

if not isinstance(
    benchmark_ranking_export,
    list
):
    raise TypeError(
        "El Ranking Científico consolidado debe ser una lista."
    ) # Validar estructura del ranking

if not benchmark_ranking_export:
    raise RuntimeError(
        "El Ranking Científico consolidado está vacío."
    ) # Validar existencia del ranking

if "best_models_by_family" not in BENCHMARK_EXECUTIVE_REPORT:
    raise RuntimeError(
        "BENCHMARK_EXECUTIVE_REPORT no contiene "
        "'best_models_by_family'."
    ) # Validar disponibilidad de mejores modelos

best_models_export = BENCHMARK_EXECUTIVE_REPORT[
    "best_models_by_family"
] # Recuperar mejores modelos consolidados

if not isinstance(
    best_models_export,
    dict
):
    raise TypeError(
        "'best_models_by_family' debe ser un diccionario."
    ) # Validar estructura de mejores modelos

benchmark_metrics_df = benchmark_results_to_dataframe(
    benchmark_results
) # Construir tabla consolidada de métricas

if benchmark_metrics_df.empty:
    raise RuntimeError(
        "benchmark_metrics_df está vacío."
    ) # Validar tabla de métricas

benchmark_ranking_df = pd.DataFrame(
    [
        build_exportable_benchmark_result(result)
        for result in benchmark_ranking_export
    ]
) # Construir tabla exportable del Ranking Científico

if benchmark_ranking_df.empty:
    raise RuntimeError(
        "benchmark_ranking_df está vacío."
    ) # Validar tabla del ranking

benchmark_summary = pd.DataFrame(
    [
        {
            "total_results": len(benchmark_results),
            "total_models": len(benchmark_ranking_export),
            "total_families": len(best_models_export),
            "ranking_metric": BENCHMARK_EXECUTIVE_REPORT[
                "ranking_metric"
            ],
            "ranking_direction": BENCHMARK_EXECUTIVE_REPORT[
                "ranking_direction"
            ],
            "official_model_code": official["model_code"],
            "official_model_name": official["model_name"],
            "official_model_family": official["family"],
            "official_ranking_position": official[
                "ranking_position"
            ],
            "official_rmse": float(
                official["rmse"]
            ),
            "official_mae": float(
                official["mae"]
            ),
            "official_r2": float(
                official["r2"]
            ),
            "official_mape": float(
                official["mape"]
            ),
            "status": "VALIDATED",
        }
    ]
) # Construir resumen ejecutivo exportable

if benchmark_summary.empty:
    raise RuntimeError(
        "benchmark_summary está vacío."
    ) # Validar resumen

if len(benchmark_metrics_df) != len(
    benchmark_results
):
    raise RuntimeError(
        "La tabla de métricas no contiene un registro "
        "por cada resultado del Benchmark."
    ) # Validar cobertura de métricas

if len(benchmark_ranking_df) != len(
    benchmark_ranking_export
):
    raise RuntimeError(
        "La tabla del Ranking no contiene todos "
        "los modelos del Ranking Científico."
    ) # Validar cobertura del ranking

if len(benchmark_summary) != 1:
    raise RuntimeError(
        "El resumen ejecutivo debe contener exactamente una fila."
    ) # Validar unicidad del resumen

if benchmark_summary.loc[
    0,
    "official_model_code"
] != official["model_code"]:
    raise RuntimeError(
        "El código del Modelo Oficial en el resumen "
        "no coincide con official."
    ) # Validar trazabilidad del Modelo Oficial

if benchmark_summary.loc[
    0,
    "official_model_name"
].strip().lower() != official[
    "model_name"
].strip().lower():
    raise RuntimeError(
        "El nombre del Modelo Oficial en el resumen "
        "no coincide con official."
    ) # Validar trazabilidad del nombre

if benchmark_summary.loc[
    0,
    "official_model_family"
].strip().lower() != official[
    "family"
].strip().lower():
    raise RuntimeError(
        "La familia del Modelo Oficial en el resumen "
        "no coincide con official."
    ) # Validar trazabilidad de la familia

print(f"Métricas construidas       : {len(benchmark_metrics_df)} filas") # Mostrar tamaño de métricas
print(f"Ranking construido         : {len(benchmark_ranking_df)} filas") # Mostrar tamaño del ranking
print(f"Resumen construido         : {len(benchmark_summary)} fila") # Mostrar tamaño del resumen
print("Productos de exportación   : CONSTRUIDOS") # Confirmar construcción

# 13.8 Preparación de los directorios oficiales
print("\n13.8 PREPARACIÓN DE LOS DIRECTORIOS DE EXPORTACIÓN")

if BENCHMARK_DIR is None:
    raise ValueError(
        "BENCHMARK_DIR no puede ser None."
    ) # Validar existencia de la ruta oficial

BENCHMARK_DIR = Path(
    BENCHMARK_DIR
) # Normalizar la ruta como objeto Path

BENCHMARK_DIR.mkdir(
    parents=True,
    exist_ok=True
) # Crear directorio oficial del Benchmark

if not BENCHMARK_DIR.exists():
    raise RuntimeError(
        "No fue posible crear el directorio oficial del Benchmark."
    ) # Validar existencia física

if not BENCHMARK_DIR.is_dir():
    raise RuntimeError(
        "BENCHMARK_DIR existe pero no corresponde a un directorio."
    ) # Validar tipo de recurso

print(f"Directorio Benchmark      : {BENCHMARK_DIR}") # Mostrar directorio
print("Directorio físico          : VALIDADO") # Confirmar preparación del directorio

# 13.9 Exportación del experimento científico
print("\n13.9 EXPORTACIÓN DEL EXPERIMENTO CIENTÍFICO")

if "ranking" not in BENCHMARK_EXECUTIVE_REPORT:
    raise RuntimeError(
        "BENCHMARK_EXECUTIVE_REPORT no contiene el Ranking Científico."
    ) # Validar disponibilidad del ranking consolidado

if "best_models_by_family" not in BENCHMARK_EXECUTIVE_REPORT:
    raise RuntimeError(
        "BENCHMARK_EXECUTIVE_REPORT no contiene los mejores modelos por familia."
    ) # Validar disponibilidad de selección por familia

benchmark_experiment = {
    "benchmark_results": benchmark_results,
    "benchmark_ranking": BENCHMARK_EXECUTIVE_REPORT[
        "ranking"
    ],
    "best_models_by_family": BENCHMARK_EXECUTIVE_REPORT[
        "best_models_by_family"
    ],
    "official_model": BENCHMARK_EXECUTIVE_REPORT[
        "official_model"
    ],
    "benchmark_data": BENCHMARK_DATA,
    "ranking_metric": BENCHMARK_EXECUTIVE_REPORT[
        "ranking_metric"
    ],
    "ranking_direction": BENCHMARK_EXECUTIVE_REPORT[
        "ranking_direction"
    ],
    "total_results": BENCHMARK_EXECUTIVE_REPORT[
        "total_results"
    ],
    "total_models": BENCHMARK_EXECUTIVE_REPORT[
        "total_models"
    ],
    "total_families": BENCHMARK_EXECUTIVE_REPORT[
        "total_families"
    ],
    "official_selection_criterion": BENCHMARK_EXECUTIVE_REPORT[
        "official_selection_criterion"
    ],
    "selection_criteria": BENCHMARK_EXECUTIVE_REPORT[
        "selection_criteria"
    ],
    "status": "VALIDATED",
} # Construir experimento científico reproducible

if not isinstance(
    benchmark_experiment,
    dict
):
    raise TypeError(
        "benchmark_experiment debe ser un diccionario."
    ) # Validar estructura del experimento

if benchmark_experiment["total_results"] != len(
    benchmark_results
):
    raise RuntimeError(
        "El total de resultados del experimento "
        "no coincide con benchmark_results."
    ) # Validar resultados

if benchmark_experiment["total_models"] != len(
    benchmark_experiment["benchmark_ranking"]
):
    raise RuntimeError(
        "El total de modelos del experimento "
        "no coincide con el Ranking Científico."
    ) # Validar ranking

if benchmark_experiment["total_families"] != len(
    benchmark_experiment["best_models_by_family"]
):
    raise RuntimeError(
        "El total de familias del experimento "
        "no coincide con best_models_by_family."
    ) # Validar familias

if benchmark_experiment["official_model"][
    "model_code"
] != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El Modelo Oficial del experimento "
        "no coincide con OFFICIAL_MODEL_CODE."
    ) # Validar Modelo Oficial

joblib.dump(
    benchmark_experiment,
    BENCHMARK_EXPERIMENT_FILE
) # Exportar experimento científico completo

if not BENCHMARK_EXPERIMENT_FILE.exists():
    raise RuntimeError(
        "El archivo del experimento científico no fue generado."
    ) # Validar existencia física

if BENCHMARK_EXPERIMENT_FILE.stat().st_size == 0:
    raise RuntimeError(
        "El archivo del experimento científico está vacío."
    ) # Validar contenido físico

print(f"Experimento científico   : {BENCHMARK_EXPERIMENT_FILE}") # Mostrar archivo
print("Experimento               : EXPORTADO Y VALIDADO") # Confirmar exportación

# 13.10 Exportación de métricas
print("\n13.10 EXPORTACIÓN DE MÉTRICAS")

if not isinstance(
    benchmark_metrics_df,
    pd.DataFrame
):
    raise TypeError(
        "benchmark_metrics_df debe ser un DataFrame de pandas."
    ) # Validar estructura de métricas

if benchmark_metrics_df.empty:
    raise RuntimeError(
        "benchmark_metrics_df está vacío."
    ) # Validar existencia de métricas

if len(benchmark_metrics_df) != len(
    benchmark_results
):
    raise RuntimeError(
        "La tabla de métricas no contiene un registro "
        "por cada resultado del Benchmark."
    ) # Validar cobertura de métricas

benchmark_metrics_export_df = benchmark_metrics_df.copy() # Crear copia exclusiva para exportación

excluded_complex_columns = [
    "model_config",
    "loss_history",
    "prediction_result",
    "evaluation_result",
] # Definir estructuras que no pertenecen a la tabla de métricas

existing_complex_columns = [
    column
    for column in excluded_complex_columns
    if column in benchmark_metrics_export_df.columns
] # Identificar estructuras complejas presentes

if existing_complex_columns:
    benchmark_metrics_export_df = benchmark_metrics_export_df.drop(
        columns=existing_complex_columns
    ) # Eliminar estructuras complejas de la tabla de métricas

allowed_text_columns = [
    "model_code",
    "model_name",
    "family",
] # Definir columnas textuales permitidas

object_columns = [
    column
    for column in benchmark_metrics_export_df.columns
    if benchmark_metrics_export_df[column].dtype == "object"
] # Identificar columnas de tipo objeto

unexpected_object_columns = [
    column
    for column in object_columns
    if column not in allowed_text_columns
] # Identificar objetos no compatibles

if unexpected_object_columns:
    raise TypeError(
        "La tabla de métricas contiene columnas de tipo objeto "
        "no compatibles con Parquet: "
        f"{unexpected_object_columns}"
    ) # Validar tipos exportables

benchmark_metrics_export_df.to_parquet(
    BENCHMARK_METRICS_FILE,
    index=False
) # Exportar métricas en formato Parquet

if not BENCHMARK_METRICS_FILE.exists():
    raise RuntimeError(
        "El archivo de métricas Parquet no fue generado."
    ) # Verificar existencia física

if BENCHMARK_METRICS_FILE.stat().st_size == 0:
    raise RuntimeError(
        "El archivo de métricas Parquet está vacío."
    ) # Verificar contenido físico

benchmark_metrics_parquet = pd.read_parquet(
    BENCHMARK_METRICS_FILE
) # Leer nuevamente el archivo exportado

if benchmark_metrics_parquet.empty:
    raise RuntimeError(
        "El archivo Parquet de métricas fue generado pero está vacío."
    ) # Validar contenido recuperado

if len(benchmark_metrics_parquet) != len(
    benchmark_results
):
    raise RuntimeError(
        "El archivo Parquet no conserva la cantidad "
        "esperada de resultados."
    ) # Validar integridad de registros

if "model_config" in benchmark_metrics_parquet.columns:
    raise RuntimeError(
        "La tabla Parquet contiene model_config, "
        "que no debe formar parte de la tabla de métricas."
    ) # Validar exclusión de configuración compleja

if any(
    column in benchmark_metrics_parquet.columns
    for column in [
        "loss_history",
        "prediction_result",
        "evaluation_result",
    ]
):
    raise RuntimeError(
        "La tabla Parquet contiene estructuras complejas "
        "que no deben formar parte de la tabla de métricas."
    ) # Validar exclusión de estructuras complejas

print(f"Métricas Parquet          : {BENCHMARK_METRICS_FILE}") # Mostrar archivo
print(f"Registros exportados      : {len(benchmark_metrics_parquet)}") # Mostrar cantidad de registros
print(f"Columnas exportadas       : {len(benchmark_metrics_parquet.columns)}") # Mostrar cantidad de columnas
print("Estructuras complejas      : EXCLUIDAS") # Confirmar limpieza
print("Lectura Parquet            : VALIDADA") # Confirmar recuperación
print("Métricas                   : EXPORTADAS Y VALIDADAS") # Confirmar exportación

# 13.11 Exportación del Ranking
print("\n13.11 EXPORTACIÓN DEL RANKING")

if not isinstance(
    benchmark_ranking_df,
    pd.DataFrame
):
    raise TypeError(
        "benchmark_ranking_df debe ser un DataFrame de pandas."
    ) # Validar estructura del ranking

if benchmark_ranking_df.empty:
    raise RuntimeError(
        "benchmark_ranking_df está vacío."
    ) # Validar existencia del ranking

if len(benchmark_ranking_df) != len(
    BENCHMARK_RANKING
):
    raise RuntimeError(
        "benchmark_ranking_df no contiene todos los modelos "
        "del Ranking Científico."
    ) # Validar cobertura del ranking

required_ranking_columns = [
    "model_code",
    "model_name",
    "family",
    "ranking_position",
    "rmse",
    "mae",
    "r2",
    "mape",
] # Definir columnas obligatorias del ranking

missing_ranking_columns = [
    column
    for column in required_ranking_columns
    if column not in benchmark_ranking_df.columns
] # Identificar columnas faltantes

if missing_ranking_columns:
    raise RuntimeError(
        "benchmark_ranking_df está incompleto: "
        f"{missing_ranking_columns}"
    ) # Validar contrato del ranking

ranking_positions = benchmark_ranking_df[
    "ranking_position"
].tolist() # Recuperar posiciones del ranking

expected_ranking_positions = list(
    range(
        1,
        len(BENCHMARK_RANKING) + 1
    )
) # Definir posiciones esperadas

if ranking_positions != expected_ranking_positions:
    raise RuntimeError(
        "Las posiciones de benchmark_ranking_df "
        "no son consecutivas."
    ) # Validar numeración

complex_ranking_columns = [
    "model_config",
    "loss_history",
    "prediction_result",
    "evaluation_result",
] # Definir estructuras complejas no exportables

present_complex_ranking_columns = [
    column
    for column in complex_ranking_columns
    if column in benchmark_ranking_df.columns
] # Detectar estructuras complejas presentes

benchmark_ranking_export_df = benchmark_ranking_df.drop(
    columns=present_complex_ranking_columns
).copy() # Construir tabla limpia para exportación

if benchmark_ranking_export_df.empty:
    raise RuntimeError(
        "benchmark_ranking_export_df quedó vacío después "
        "de eliminar estructuras complejas."
    ) # Validar tabla exportable

for column in required_ranking_columns:
    if column not in benchmark_ranking_export_df.columns:
        raise RuntimeError(
            f"La columna obligatoria '{column}' "
            "no está disponible para exportación."
        ) # Validar columnas finales

benchmark_ranking_export_df.to_csv(
    BENCHMARK_RANKING_CSV_FILE,
    index=False
) # Exportar ranking en CSV

benchmark_ranking_export_df.to_excel(
    BENCHMARK_RANKING_XLSX_FILE,
    index=False
) # Exportar ranking en Excel

if not BENCHMARK_RANKING_CSV_FILE.exists():
    raise RuntimeError(
        "El archivo CSV del Ranking no fue generado."
    ) # Verificar CSV

if not BENCHMARK_RANKING_XLSX_FILE.exists():
    raise RuntimeError(
        "El archivo Excel del Ranking no fue generado."
    ) # Verificar Excel

if BENCHMARK_RANKING_CSV_FILE.stat().st_size == 0:
    raise RuntimeError(
        "El archivo CSV del Ranking está vacío."
    ) # Verificar contenido CSV

if BENCHMARK_RANKING_XLSX_FILE.stat().st_size == 0:
    raise RuntimeError(
        "El archivo Excel del Ranking está vacío."
    ) # Verificar contenido Excel

ranking_csv_check = pd.read_csv(
    BENCHMARK_RANKING_CSV_FILE
) # Leer nuevamente el CSV exportado

ranking_xlsx_check = pd.read_excel(
    BENCHMARK_RANKING_XLSX_FILE
) # Leer nuevamente el Excel exportado

if len(ranking_csv_check) != len(
    BENCHMARK_RANKING
):
    raise RuntimeError(
        "El CSV del Ranking no conserva la cantidad "
        "esperada de modelos."
    ) # Validar integridad del CSV

if len(ranking_xlsx_check) != len(
    BENCHMARK_RANKING
):
    raise RuntimeError(
        "El Excel del Ranking no conserva la cantidad "
        "esperada de modelos."
    ) # Validar integridad del Excel

exported_positions = ranking_csv_check[
    "ranking_position"
].tolist() # Recuperar posiciones exportadas

if exported_positions != expected_ranking_positions:
    raise RuntimeError(
        "El CSV exportado no conserva el orden "
        "del Ranking Científico."
    ) # Validar orden exportado

print(f"Ranking CSV               : {BENCHMARK_RANKING_CSV_FILE}") # Mostrar CSV
print(f"Ranking Excel             : {BENCHMARK_RANKING_XLSX_FILE}") # Mostrar Excel
print(f"Modelos exportados        : {len(benchmark_ranking_export_df)}") # Mostrar cobertura
print(f"Estructuras excluidas     : {present_complex_ranking_columns}") # Mostrar estructuras eliminadas
print("Lectura CSV                : VALIDADA") # Confirmar CSV
print("Lectura Excel              : VALIDADA") # Confirmar Excel
print("Orden del Ranking          : VALIDADO") # Confirmar orden
print("Ranking                    : EXPORTADO Y VALIDADO") # Confirmar exportación

# 13.12 Exportación del resumen
print("\n13.12 EXPORTACIÓN DEL RESUMEN")

if not isinstance(
    benchmark_summary,
    pd.DataFrame
):
    raise TypeError(
        "benchmark_summary debe ser un DataFrame de pandas."
    ) # Validar estructura del resumen

if benchmark_summary.empty:
    raise RuntimeError(
        "benchmark_summary está vacío."
    ) # Validar existencia del resumen

if len(benchmark_summary) != 1:
    raise RuntimeError(
        "benchmark_summary debe contener exactamente una fila."
    ) # Validar estructura ejecutiva

required_summary_columns = [
    "total_results",
    "total_models",
    "total_families",
    "ranking_metric",
    "ranking_direction",
    "official_model_code",
    "official_model_name",
    "official_model_family",
    "official_ranking_position",
    "official_rmse",
    "official_mae",
    "official_r2",
    "official_mape",
    "status",
] # Definir contrato del resumen

missing_summary_columns = [
    column
    for column in required_summary_columns
    if column not in benchmark_summary.columns
] # Identificar columnas faltantes

if missing_summary_columns:
    raise RuntimeError(
        "benchmark_summary está incompleto: "
        f"{missing_summary_columns}"
    ) # Validar contrato del resumen

summary_row = benchmark_summary.iloc[0] # Recuperar única fila del resumen

if int(summary_row["total_results"]) != len(
    benchmark_results
):
    raise RuntimeError(
        "El total_results del resumen no coincide "
        "con benchmark_results."
    ) # Validar resultados

if int(summary_row["total_models"]) != len(
    BENCHMARK_RANKING
):
    raise RuntimeError(
        "El total_models del resumen no coincide "
        "con BENCHMARK_RANKING."
    ) # Validar modelos

if int(summary_row["total_families"]) != len(
    CANDIDATE_MODELS["families"]
):
    raise RuntimeError(
        "El total_families del resumen no coincide "
        "con las familias configuradas."
    ) # Validar familias

if str(
    summary_row["ranking_metric"]
).strip().lower() != ranking_metric.strip().lower():
    raise RuntimeError(
        "La métrica de ranking del resumen no coincide "
        "con BENCHMARK_CONFIG."
    ) # Validar métrica

if str(
    summary_row["ranking_direction"]
).strip().lower() != ranking_direction.strip().lower():
    raise RuntimeError(
        "La dirección del ranking del resumen no coincide "
        "con BENCHMARK_CONFIG."
    ) # Validar dirección

if summary_row["official_model_code"] != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El código del Modelo Oficial del resumen "
        "no coincide con OFFICIAL_MODEL_CODE."
    ) # Validar código oficial

if str(
    summary_row["official_model_name"]
).strip().lower() != OFFICIAL_MODEL_NAME.strip().lower():
    raise RuntimeError(
        "El nombre del Modelo Oficial del resumen "
        "no coincide con OFFICIAL_MODEL_NAME."
    ) # Validar nombre oficial

if str(
    summary_row["official_model_family"]
).strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise RuntimeError(
        "La familia del Modelo Oficial del resumen "
        "no coincide con OFFICIAL_MODEL_FAMILY."
    ) # Validar familia oficial

if int(
    summary_row["official_ranking_position"]
) != int(
    official["ranking_position"]
):
    raise RuntimeError(
        "La posición del Modelo Oficial en el resumen "
        "no coincide con official."
    ) # Validar posición oficial

if not np.isclose(
    float(summary_row["official_rmse"]),
    float(official["rmse"])
):
    raise RuntimeError(
        "El RMSE del Modelo Oficial no coincide "
        "con official."
    ) # Validar RMSE oficial

if not np.isclose(
    float(summary_row["official_mae"]),
    float(official["mae"])
):
    raise RuntimeError(
        "El MAE del Modelo Oficial no coincide "
        "con official."
    ) # Validar MAE oficial

if not np.isclose(
    float(summary_row["official_r2"]),
    float(official["r2"])
):
    raise RuntimeError(
        "El R2 del Modelo Oficial no coincide "
        "con official."
    ) # Validar R2 oficial

if not np.isclose(
    float(summary_row["official_mape"]),
    float(official["mape"])
):
    raise RuntimeError(
        "El MAPE del Modelo Oficial no coincide "
        "con official."
    ) # Validar MAPE oficial

if summary_row["status"] != "VALIDATED":
    raise RuntimeError(
        "El resumen del Benchmark no está marcado "
        "como VALIDATED."
    ) # Validar estado oficial

benchmark_summary.to_csv(
    BENCHMARK_SUMMARY_CSV_FILE,
    index=False
) # Exportar resumen en CSV

benchmark_summary.to_excel(
    BENCHMARK_SUMMARY_XLSX_FILE,
    index=False
) # Exportar resumen en Excel

if not BENCHMARK_SUMMARY_CSV_FILE.exists():
    raise RuntimeError(
        "El archivo CSV del resumen no fue generado."
    ) # Verificar CSV

if not BENCHMARK_SUMMARY_XLSX_FILE.exists():
    raise RuntimeError(
        "El archivo Excel del resumen no fue generado."
    ) # Verificar Excel

if BENCHMARK_SUMMARY_CSV_FILE.stat().st_size == 0:
    raise RuntimeError(
        "El archivo CSV del resumen está vacío."
    ) # Verificar contenido CSV

if BENCHMARK_SUMMARY_XLSX_FILE.stat().st_size == 0:
    raise RuntimeError(
        "El archivo Excel del resumen está vacío."
    ) # Verificar contenido Excel

summary_csv_check = pd.read_csv(
    BENCHMARK_SUMMARY_CSV_FILE
) # Leer nuevamente el CSV exportado

summary_xlsx_check = pd.read_excel(
    BENCHMARK_SUMMARY_XLSX_FILE
) # Leer nuevamente el Excel exportado

if len(summary_csv_check) != 1:
    raise RuntimeError(
        "El CSV del resumen no conserva exactamente "
        "una fila."
    ) # Validar integridad del CSV

if len(summary_xlsx_check) != 1:
    raise RuntimeError(
        "El Excel del resumen no conserva exactamente "
        "una fila."
    ) # Validar integridad del Excel

print(f"Resumen CSV                : {BENCHMARK_SUMMARY_CSV_FILE}") # Mostrar archivo CSV
print(f"Resumen Excel              : {BENCHMARK_SUMMARY_XLSX_FILE}") # Mostrar archivo Excel
print(f"Resultados resumidos       : {int(summary_row['total_results'])}") # Mostrar resultados
print(f"Modelos resumidos          : {int(summary_row['total_models'])}") # Mostrar modelos
print(f"Familias resumidas         : {int(summary_row['total_families'])}") # Mostrar familias
print(f"Modelo Oficial             : {summary_row['official_model_name']}") # Mostrar Modelo Oficial
print(f"Posición Oficial           : {int(summary_row['official_ranking_position'])}") # Mostrar posición oficial
print("Lectura CSV                : VALIDADA") # Confirmar CSV
print("Lectura Excel              : VALIDADA") # Confirmar Excel
print("Resumen                    : EXPORTADO Y VALIDADO") # Confirmar exportación

# 13.13 Exportación de la configuración oficial
print("\n13.13 EXPORTACIÓN DE LA CONFIGURACIÓN OFICIAL")

if not isinstance(
    official_model_config_export,
    dict
):
    raise TypeError(
        "official_model_config_export debe ser un diccionario."
    ) # Validar producto exportable

required_config_fields = [
    "model_code",
    "model_name",
    "family",
    "model_config",
    "ranking_position",
    "rmse",
    "mae",
    "r2",
    "mape",
    "selection_source",
    "selection_scope",
    "ranking_metric",
    "ranking_direction",
    "benchmark_models_evaluated",
    "benchmark_families_evaluated",
    "status",
] # Definir contrato de configuración oficial

missing_config_fields = [
    field
    for field in required_config_fields
    if field not in official_model_config_export
] # Identificar campos faltantes

if missing_config_fields:
    raise RuntimeError(
        "La configuración oficial está incompleta: "
        f"{missing_config_fields}"
    ) # Validar contrato

if not isinstance(
    official_model_config_export["model_config"],
    dict
):
    raise TypeError(
        "official_model_config_export['model_config'] "
        "debe ser un diccionario."
    ) # Validar configuración interna

if official_model_config_export["model_code"] != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El código exportado no coincide con OFFICIAL_MODEL_CODE."
    ) # Validar código oficial

if official_model_config_export["model_name"].strip().lower() != OFFICIAL_MODEL_NAME.strip().lower():
    raise RuntimeError(
        "El nombre exportado no coincide con OFFICIAL_MODEL_NAME."
    ) # Validar nombre oficial

if official_model_config_export["family"].strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise RuntimeError(
        "La familia exportada no coincide con OFFICIAL_MODEL_FAMILY."
    ) # Validar familia oficial

if official_model_config_export["ranking_metric"] != ranking_metric:
    raise RuntimeError(
        "La métrica exportada no coincide con la métrica oficial."
    ) # Validar métrica

if official_model_config_export["ranking_direction"] != ranking_direction:
    raise RuntimeError(
        "La dirección exportada no coincide con la configuración oficial."
    ) # Validar dirección

if official_model_config_export["ranking_position"] != official["ranking_position"]:
    raise RuntimeError(
        "La posición exportada no coincide con la posición del Modelo Oficial."
    ) # Validar posición

if not np.isclose(
    float(official_model_config_export["rmse"]),
    float(official["rmse"])
):
    raise RuntimeError(
        "El RMSE exportado no coincide con el Modelo Oficial."
    ) # Validar RMSE

if not np.isclose(
    float(official_model_config_export["mae"]),
    float(official["mae"])
):
    raise RuntimeError(
        "El MAE exportado no coincide con el Modelo Oficial."
    ) # Validar MAE

if not np.isclose(
    float(official_model_config_export["r2"]),
    float(official["r2"])
):
    raise RuntimeError(
        "El R2 exportado no coincide con el Modelo Oficial."
    ) # Validar R2

if not np.isclose(
    float(official_model_config_export["mape"]),
    float(official["mape"])
):
    raise RuntimeError(
        "El MAPE exportado no coincide con el Modelo Oficial."
    ) # Validar MAPE

if official_model_config_export["benchmark_models_evaluated"] != len(
    benchmark_results
):
    raise RuntimeError(
        "La cantidad de modelos evaluados no coincide con benchmark_results."
    ) # Validar cobertura de modelos

if official_model_config_export["benchmark_families_evaluated"] != len(
    best_models
):
    raise RuntimeError(
        "La cantidad de familias evaluadas no coincide con best_models."
    ) # Validar cobertura de familias

if official_model_config_export["status"] != "OFFICIAL":
    raise RuntimeError(
        "La configuración oficial no tiene estado OFFICIAL."
    ) # Validar estado oficial

try:
    json.dumps(
        official_model_config_export,
        ensure_ascii=False
    )
except (
    TypeError,
    ValueError
) as error:
    raise RuntimeError(
        "La configuración oficial contiene elementos "
        "no serializables en JSON."
    ) from error # Validar serialización JSON

OFFICIAL_MODEL_CONFIG_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
) # Crear directorio oficial

with OFFICIAL_MODEL_CONFIG_FILE.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        official_model_config_export,
        file,
        indent=4,
        ensure_ascii=False
    ) # Exportar configuración oficial

if not OFFICIAL_MODEL_CONFIG_FILE.exists():
    raise RuntimeError(
        "El archivo official_model_config.json no fue generado."
    ) # Validar existencia

if OFFICIAL_MODEL_CONFIG_FILE.stat().st_size == 0:
    raise RuntimeError(
        "El archivo official_model_config.json está vacío."
    ) # Validar contenido

print(f"Archivo                    : {OFFICIAL_MODEL_CONFIG_FILE}") # Mostrar archivo
print(f"Modelo Oficial             : {official_model_config_export['model_name']}") # Mostrar modelo
print(f"Código Oficial             : {official_model_config_export['model_code']}") # Mostrar código
print(f"Familia                    : {official_model_config_export['family']}") # Mostrar familia
print(f"Posición Ranking           : {official_model_config_export['ranking_position']}") # Mostrar posición
print("Configuración oficial       : EXPORTADA Y VALIDADA") # Confirmar exportación

# 13.14 Construcción del resultado oficial de exportación
print("\n13.14 CONSTRUCCIÓN DEL RESULTADO OFICIAL DE EXPORTACIÓN")

export_files = {
    "benchmark_experiment_file": BENCHMARK_EXPERIMENT_FILE,
    "benchmark_metrics_file": BENCHMARK_METRICS_FILE,
    "benchmark_ranking_csv_file": BENCHMARK_RANKING_CSV_FILE,
    "benchmark_ranking_xlsx_file": BENCHMARK_RANKING_XLSX_FILE,
    "benchmark_summary_csv_file": BENCHMARK_SUMMARY_CSV_FILE,
    "benchmark_summary_xlsx_file": BENCHMARK_SUMMARY_XLSX_FILE,
    "official_model_config_file": OFFICIAL_MODEL_CONFIG_FILE,
} # Definir archivos oficiales de exportación

missing_export_files = [
    file_name
    for file_name, file_path in export_files.items()
    if not file_path.exists()
] # Identificar archivos faltantes

if missing_export_files:
    raise RuntimeError(
        "Existen archivos oficiales de exportación que no fueron generados: "
        f"{missing_export_files}"
    ) # Validar existencia

empty_export_files = [
    file_name
    for file_name, file_path in export_files.items()
    if file_path.stat().st_size == 0
] # Identificar archivos vacíos

if empty_export_files:
    raise RuntimeError(
        "Existen archivos oficiales de exportación vacíos: "
        f"{empty_export_files}"
    ) # Validar contenido físico

if not isinstance(
    official,
    dict
):
    raise TypeError(
        "official debe ser un diccionario."
    ) # Validar Modelo Oficial

required_official_fields = [
    "model_code",
    "model_name",
    "family",
    "ranking_position",
] # Definir campos obligatorios

missing_official_fields = [
    field
    for field in required_official_fields
    if field not in official
] # Identificar campos faltantes

if missing_official_fields:
    raise RuntimeError(
        "El Modelo Oficial está incompleto: "
        f"{missing_official_fields}"
    ) # Validar integridad

if official["model_code"] != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El código del Modelo Oficial no coincide con OFFICIAL_MODEL_CODE."
    ) # Validar código

if official["model_name"].strip().lower() != OFFICIAL_MODEL_NAME.strip().lower():
    raise RuntimeError(
        "El nombre del Modelo Oficial no coincide con OFFICIAL_MODEL_NAME."
    ) # Validar nombre

if official["family"].strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise RuntimeError(
        "La familia del Modelo Oficial no coincide con OFFICIAL_MODEL_FAMILY."
    ) # Validar familia

if len(BENCHMARK_RANKING) != len(
    benchmark_results
):
    raise RuntimeError(
        "El Ranking Científico no contiene todos los modelos evaluados."
    ) # Validar cobertura del ranking

if len(best_models) != len(
    CANDIDATE_MODELS["families"]
):
    raise RuntimeError(
        "best_models no contiene todas las familias configuradas."
    ) # Validar cobertura de familias

BENCHMARK_EXPORT_RESULT = {
    "status": "SUCCESS",
    "benchmark_experiment_file": str(
        BENCHMARK_EXPERIMENT_FILE
    ),
    "benchmark_metrics_file": str(
        BENCHMARK_METRICS_FILE
    ),
    "benchmark_ranking_csv_file": str(
        BENCHMARK_RANKING_CSV_FILE
    ),
    "benchmark_ranking_xlsx_file": str(
        BENCHMARK_RANKING_XLSX_FILE
    ),
    "benchmark_summary_csv_file": str(
        BENCHMARK_SUMMARY_CSV_FILE
    ),
    "benchmark_summary_xlsx_file": str(
        BENCHMARK_SUMMARY_XLSX_FILE
    ),
    "official_model_config_file": str(
        OFFICIAL_MODEL_CONFIG_FILE
    ),
    "official_model": official["model_name"],
    "model_code": official["model_code"],
    "family": official["family"],
    "ranking_position": official["ranking_position"],
    "ranking_metric": ranking_metric,
    "ranking_direction": ranking_direction,
    "total_results": len(benchmark_results),
    "total_models": len(BENCHMARK_RANKING),
    "total_families": len(best_models),
    "export_date": dt.datetime.now().isoformat(),
} # Construir producto oficial de exportación

required_export_fields = [
    "status",
    "benchmark_experiment_file",
    "benchmark_metrics_file",
    "benchmark_ranking_csv_file",
    "benchmark_ranking_xlsx_file",
    "benchmark_summary_csv_file",
    "benchmark_summary_xlsx_file",
    "official_model_config_file",
    "official_model",
    "model_code",
    "family",
    "ranking_position",
    "ranking_metric",
    "ranking_direction",
    "total_results",
    "total_models",
    "total_families",
    "export_date",
] # Definir contrato del resultado

missing_export_fields = [
    field
    for field in required_export_fields
    if field not in BENCHMARK_EXPORT_RESULT
] # Identificar campos faltantes

if missing_export_fields:
    raise RuntimeError(
        "BENCHMARK_EXPORT_RESULT está incompleto: "
        f"{missing_export_fields}"
    ) # Validar contrato

if BENCHMARK_EXPORT_RESULT["status"] != "SUCCESS":
    raise RuntimeError(
        "El resultado oficial de exportación no presenta estado SUCCESS."
    ) # Validar estado

if BENCHMARK_EXPORT_RESULT["model_code"] != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El código del resultado de exportación no coincide "
        "con OFFICIAL_MODEL_CODE."
    ) # Validar código

if BENCHMARK_EXPORT_RESULT["model_name"] if "model_name" in BENCHMARK_EXPORT_RESULT else False:
    pass # No utilizar un campo redundante de nombre de modelo

if BENCHMARK_EXPORT_RESULT["family"].strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise RuntimeError(
        "La familia del resultado de exportación no coincide "
        "con OFFICIAL_MODEL_FAMILY."
    ) # Validar familia

if BENCHMARK_EXPORT_RESULT["ranking_position"] != official["ranking_position"]:
    raise RuntimeError(
        "La posición del resultado de exportación no coincide "
        "con official."
    ) # Validar posición

if BENCHMARK_EXPORT_RESULT["total_results"] != len(
    benchmark_results
):
    raise RuntimeError(
        "El total de resultados exportados no coincide con benchmark_results."
    ) # Validar resultados

if BENCHMARK_EXPORT_RESULT["total_models"] != len(
    BENCHMARK_RANKING
):
    raise RuntimeError(
        "El total de modelos exportados no coincide con BENCHMARK_RANKING."
    ) # Validar modelos

if BENCHMARK_EXPORT_RESULT["total_families"] != len(
    best_models
):
    raise RuntimeError(
        "El total de familias exportadas no coincide con best_models."
    ) # Validar familias

print(f"Archivos exportados        : {len(export_files)}") # Mostrar cantidad de archivos
print(f"Modelos exportados         : {BENCHMARK_EXPORT_RESULT['total_models']}") # Mostrar modelos
print(f"Familias exportadas        : {BENCHMARK_EXPORT_RESULT['total_families']}") # Mostrar familias
print(f"Modelo Oficial             : {BENCHMARK_EXPORT_RESULT['official_model']}") # Mostrar Modelo Oficial
print(f"Código Oficial             : {BENCHMARK_EXPORT_RESULT['model_code']}") # Mostrar código
print(f"Posición Ranking           : {BENCHMARK_EXPORT_RESULT['ranking_position']}") # Mostrar posición
print(f"Métrica de Ranking         : {BENCHMARK_EXPORT_RESULT['ranking_metric'].upper()}") # Mostrar métrica
print("Archivos de exportación     : EXISTENTES Y NO VACÍOS") # Confirmar archivos
print("Resultado de exportación    : VALIDADO") # Confirmar producto oficial

# 13.15 Validación física de los archivos exportados
print("\n13.15 VALIDACIÓN FÍSICA DE LOS PRODUCTOS EXPORTADOS")

exported_files = {
    "BenchmarkExperiment": BENCHMARK_EXPERIMENT_FILE,
    "BenchmarkMetrics": BENCHMARK_METRICS_FILE,
    "BenchmarkRankingCSV": BENCHMARK_RANKING_CSV_FILE,
    "BenchmarkRankingXLSX": BENCHMARK_RANKING_XLSX_FILE,
    "BenchmarkSummaryCSV": BENCHMARK_SUMMARY_CSV_FILE,
    "BenchmarkSummaryXLSX": BENCHMARK_SUMMARY_XLSX_FILE,
    "OfficialModelConfig": OFFICIAL_MODEL_CONFIG_FILE,
} # Definir archivos oficiales esperados

expected_export_file_count = 7 # Definir cantidad oficial de productos físicos

if len(exported_files) != expected_export_file_count:
    raise RuntimeError(
        "La cantidad de productos físicos definidos no coincide "
        f"con la arquitectura oficial: {len(exported_files)}."
    ) # Validar cantidad de productos

invalid_paths = [
    name
    for name, file_path in exported_files.items()
    if not isinstance(file_path, Path)
] # Identificar rutas con tipo incorrecto

if invalid_paths:
    raise TypeError(
        "Las siguientes rutas de exportación no son objetos Path: "
        f"{invalid_paths}"
    ) # Validar tipo de rutas

missing_files = [
    name
    for name, file_path in exported_files.items()
    if not file_path.exists()
] # Identificar archivos faltantes

if missing_files:
    raise FileNotFoundError(
        "No fueron generados los siguientes archivos: "
        f"{missing_files}"
    ) # Validar existencia física

empty_files = [
    name
    for name, file_path in exported_files.items()
    if file_path.stat().st_size <= 0
] # Identificar archivos vacíos

if empty_files:
    raise RuntimeError(
        "Los siguientes archivos fueron generados pero están vacíos: "
        f"{empty_files}"
    ) # Validar contenido físico

expected_export_keys = {
    "benchmark_experiment_file",
    "benchmark_metrics_file",
    "benchmark_ranking_csv_file",
    "benchmark_ranking_xlsx_file",
    "benchmark_summary_csv_file",
    "benchmark_summary_xlsx_file",
    "official_model_config_file",
} # Definir contrato de archivos del resultado oficial

if set(export_files.keys()) != expected_export_keys:
    raise RuntimeError(
        "export_files no coincide con el contrato oficial de exportación."
    ) # Validar contrato de archivos

if len(export_files) != len(exported_files):
    raise RuntimeError(
        "La cantidad de archivos de export_files no coincide "
        "con exported_files."
    ) # Validar correspondencia

print(f"Archivos esperados        : {len(exported_files)}") # Mostrar cantidad esperada
print(f"Archivos encontrados      : {len(exported_files) - len(missing_files)}") # Mostrar cantidad encontrada
print("Existencia física          : VALIDADA") # Confirmar existencia
print("Tamaño de archivos         : VALIDADO") # Confirmar contenido
print("Cobertura de exportación   : COMPLETA") # Confirmar cobertura
print("Validación física          : APROBADA") # Confirmar validación física
# 13.16 Validación específica de official_model_config.json
print("\n13.16 VALIDACIÓN DE official_model_config.json")

if not OFFICIAL_MODEL_CONFIG_FILE.exists():
    raise FileNotFoundError(
        "official_model_config.json no existe."
    ) # Verificar existencia

if OFFICIAL_MODEL_CONFIG_FILE.stat().st_size <= 0:
    raise RuntimeError(
        "official_model_config.json está vacío."
    ) # Verificar contenido físico

with OFFICIAL_MODEL_CONFIG_FILE.open(
    "r",
    encoding="utf-8"
) as file:
    exported_official_config = json.load(
        file
    ) # Leer configuración exportada

if not isinstance(
    exported_official_config,
    dict
):
    raise TypeError(
        "official_model_config.json debe contener un diccionario JSON."
    ) # Validar estructura JSON

required_config_fields = [
    "model_code",
    "model_name",
    "family",
    "model_config",
    "ranking_position",
    "rmse",
    "mae",
    "r2",
    "mape",
    "selection_source",
    "selection_scope",
    "ranking_metric",
    "ranking_direction",
    "benchmark_models_evaluated",
    "benchmark_families_evaluated",
    "status",
] # Definir contrato completo del archivo oficial

missing_config_fields = [
    field
    for field in required_config_fields
    if field not in exported_official_config
] # Identificar campos faltantes

if missing_config_fields:
    raise ValueError(
        "official_model_config.json está incompleto: "
        f"{missing_config_fields}"
    ) # Validar contrato JSON

if not isinstance(
    exported_official_config["model_config"],
    dict
):
    raise TypeError(
        "'model_config' debe ser un diccionario."
    ) # Validar configuración interna

if exported_official_config["model_code"] != OFFICIAL_MODEL_CODE:
    raise ValueError(
        "El código exportado no corresponde al Modelo Oficial."
    ) # Validar código oficial

if exported_official_config["model_name"].strip().lower() != OFFICIAL_MODEL_NAME.strip().lower():
    raise ValueError(
        "El nombre exportado no corresponde al Modelo Oficial."
    ) # Validar nombre oficial

if exported_official_config["family"].strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise ValueError(
        "La familia exportada no corresponde al Modelo Oficial."
    ) # Validar familia oficial

if exported_official_config["ranking_position"] != official["ranking_position"]:
    raise ValueError(
        "La posición del Ranking exportada no coincide con el Modelo Oficial."
    ) # Validar posición oficial

if not np.isclose(
    float(exported_official_config["rmse"]),
    float(official["rmse"])
):
    raise ValueError(
        "El RMSE exportado no coincide con el Modelo Oficial."
    ) # Validar RMSE

if not np.isclose(
    float(exported_official_config["mae"]),
    float(official["mae"])
):
    raise ValueError(
        "El MAE exportado no coincide con el Modelo Oficial."
    ) # Validar MAE

if not np.isclose(
    float(exported_official_config["r2"]),
    float(official["r2"])
):
    raise ValueError(
        "El R2 exportado no coincide con el Modelo Oficial."
    ) # Validar R2

if not np.isclose(
    float(exported_official_config["mape"]),
    float(official["mape"])
):
    raise ValueError(
        "El MAPE exportado no coincide con el Modelo Oficial."
    ) # Validar MAPE

if exported_official_config["selection_source"] != "benchmark_scientific":
    raise ValueError(
        "La fuente de selección exportada no corresponde "
        "al Benchmark Científico."
    ) # Validar origen científico de selección

if exported_official_config["selection_scope"].strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise ValueError(
        "El ámbito de selección exportado no corresponde "
        "a la familia oficial."
    ) # Validar ámbito de selección

if exported_official_config["ranking_metric"] != ranking_metric:
    raise ValueError(
        "La métrica de ranking exportada no coincide "
        "con la configuración oficial."
    ) # Validar métrica

if exported_official_config["ranking_direction"] != ranking_direction:
    raise ValueError(
        "La dirección de ranking exportada no coincide "
        "con la configuración oficial."
    ) # Validar dirección

if exported_official_config["benchmark_models_evaluated"] != len(
    benchmark_results
):
    raise ValueError(
        "La cantidad de modelos evaluados exportada no coincide "
        "con benchmark_results."
    ) # Validar cobertura de modelos

if exported_official_config["benchmark_families_evaluated"] != len(
    best_models
):
    raise ValueError(
        "La cantidad de familias evaluadas exportada no coincide "
        "con best_models."
    ) # Validar cobertura de familias

if exported_official_config["status"] != "OFFICIAL":
    raise ValueError(
        "La configuración exportada no está marcada como OFFICIAL."
    ) # Validar estado oficial

print(f"Código Oficial             : {exported_official_config['model_code']}") # Mostrar código
print(f"Modelo Oficial             : {exported_official_config['model_name']}") # Mostrar modelo
print(f"Familia                    : {exported_official_config['family']}") # Mostrar familia
print(f"Posición Ranking           : {exported_official_config['ranking_position']}") # Mostrar posición
print(f"RMSE                       : {exported_official_config['rmse']:.6f}") # Mostrar RMSE
print(f"MAE                        : {exported_official_config['mae']:.6f}") # Mostrar MAE
print(f"R2                         : {exported_official_config['r2']:.6f}") # Mostrar R2
print(f"MAPE                       : {exported_official_config['mape']:.6f}") # Mostrar MAPE
print(f"Métrica Ranking            : {exported_official_config['ranking_metric']}") # Mostrar métrica
print(f"Dirección Ranking          : {exported_official_config['ranking_direction']}") # Mostrar dirección
print(f"Fuente Selección           : {exported_official_config['selection_source']}") # Mostrar fuente
print(f"Ámbito Selección           : {exported_official_config['selection_scope']}") # Mostrar ámbito
print(f"Modelos evaluados          : {exported_official_config['benchmark_models_evaluated']}") # Mostrar cobertura de modelos
print(f"Familias evaluadas         : {exported_official_config['benchmark_families_evaluated']}") # Mostrar cobertura de familias
print("official_model_config.json : VALIDADO") # Confirmar configuración oficial

# 13.17 Construcción de la respuesta oficial del Benchmark
print("\n13.17 PRESENTACIÓN FINAL DE LA EXPORTACIÓN")

if not isinstance(
    BENCHMARK_EXPORT_RESULT,
    dict
):
    raise TypeError(
        "BENCHMARK_EXPORT_RESULT debe ser un diccionario."
    ) # Validar producto oficial de exportación

if BENCHMARK_EXPORT_RESULT.get("status") != "SUCCESS":
    raise RuntimeError(
        "BENCHMARK_EXPORT_RESULT no presenta estado SUCCESS."
    ) # Validar estado de exportación

if BENCHMARK_EXPORT_RESULT.get("total_results") != len(
    benchmark_results
):
    raise RuntimeError(
        "El total de resultados exportado no coincide con benchmark_results."
    ) # Validar cantidad de resultados

if BENCHMARK_EXPORT_RESULT.get("total_models") != len(
    BENCHMARK_RANKING
):
    raise RuntimeError(
        "El total de modelos exportado no coincide con BENCHMARK_RANKING."
    ) # Validar cantidad de modelos

if BENCHMARK_EXPORT_RESULT.get("total_families") != len(
    best_models
):
    raise RuntimeError(
        "El total de familias exportado no coincide con best_models."
    ) # Validar cantidad de familias

if BENCHMARK_EXPORT_RESULT.get("model_code") != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El código exportado no coincide con OFFICIAL_MODEL_CODE."
    ) # Validar código oficial

if BENCHMARK_EXPORT_RESULT.get("official_model", "").strip().lower() != OFFICIAL_MODEL_NAME.strip().lower():
    raise RuntimeError(
        "El nombre exportado no coincide con OFFICIAL_MODEL_NAME."
    ) # Validar nombre oficial

if BENCHMARK_EXPORT_RESULT.get("family", "").strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise RuntimeError(
        "La familia exportada no coincide con OFFICIAL_MODEL_FAMILY."
    ) # Validar familia oficial

if BENCHMARK_EXPORT_RESULT.get("ranking_position") != official["ranking_position"]:
    raise RuntimeError(
        "La posición exportada no coincide con el Modelo Oficial."
    ) # Validar posición del ranking

for file_name, file_path in exported_files.items():
    if not file_path.exists():
        raise FileNotFoundError(
            f"El archivo oficial '{file_name}' no existe."
        ) # Validar existencia física

    if file_path.stat().st_size <= 0:
        raise RuntimeError(
            f"El archivo oficial '{file_name}' está vacío."
        ) # Validar contenido físico

print("-" * 80)
print("EXPORTACIÓN OFICIAL DEL BENCHMARK")
print("-" * 80)

print(f"Resultados Benchmark       : {len(benchmark_results)}") # Mostrar cantidad de resultados
print(f"Modelos evaluados         : {len(BENCHMARK_RANKING)}") # Mostrar cantidad de modelos
print(f"Familias evaluadas        : {len(best_models)}") # Mostrar cantidad de familias
print(f"Métrica de ranking        : {ranking_metric.upper()}") # Mostrar métrica oficial
print(f"Dirección ranking         : {ranking_direction.upper()}") # Mostrar dirección de optimización
print(f"Modelo Oficial            : {official['model_name']}") # Mostrar Modelo Oficial
print(f"Código Oficial            : {official['model_code']}") # Mostrar código oficial
print(f"Familia Oficial           : {official['family']}") # Mostrar familia oficial
print(f"Posición Ranking          : {official['ranking_position']}") # Mostrar posición global
print(f"Benchmark Experiment      : {BENCHMARK_EXPERIMENT_FILE.name}") # Mostrar archivo del experimento
print(f"Benchmark Metrics         : {BENCHMARK_METRICS_FILE.name}") # Mostrar archivo de métricas
print(f"Benchmark Ranking CSV     : {BENCHMARK_RANKING_CSV_FILE.name}") # Mostrar ranking CSV
print(f"Benchmark Ranking XLSX    : {BENCHMARK_RANKING_XLSX_FILE.name}") # Mostrar ranking Excel
print(f"Benchmark Summary CSV     : {BENCHMARK_SUMMARY_CSV_FILE.name}") # Mostrar resumen CSV
print(f"Benchmark Summary XLSX    : {BENCHMARK_SUMMARY_XLSX_FILE.name}") # Mostrar resumen Excel
print(f"Official Model Config     : {OFFICIAL_MODEL_CONFIG_FILE.name}") # Mostrar configuración oficial
print(f"Archivos exportados       : {len(exported_files)}") # Mostrar cantidad de productos físicos
print("Existencia de archivos    : VALIDADA") # Confirmar existencia física
print("Contenido de archivos     : VALIDADO") # Confirmar contenido físico
print("Modelo Oficial            : VALIDADO") # Confirmar identidad oficial
print("Resultado de exportación  : VALIDADO") # Confirmar producto oficial
print("Estado                    : EXPORTACIÓN VALIDADA") # Confirmar estado final

print("-" * 80)

print("\nBloque 13. Exportación Oficial del Benchmark completada correctamente.") # Confirmar finalización

# BLOQUE 14. AUDITORÍA CIENTÍFICA FINAL DEL BENCHMARK
# Objetivo: Auditar la integridad, consistencia y validez científica de los resultados generados por
# el Benchmark antes de transferir sus productos oficiales a las etapas posteriores del proyecto.
# Arquitectura científica: Validar -> Recuperar -> Validar -> Auditar -> Registrar -> Confirmar.
# Entradas: - BENCHMARK_RESULTS - BENCHMARK_RANKING_RESULT - OFFICIAL_MODEL - OFFICIAL_MODEL_BY_FAMILY_RESULT
# - BENCHMARK_DATA - BENCHMARK_CONFIG - BENCHMARK_MODELS - Productos exportados
# Producto: - BENCHMARK_AUDIT_RESULT
# Pregunta científica: ¿Los productos científicos del Benchmark presentan integridad, consistencia y
# trazabilidad suficientes para ser utilizados en las etapas posteriores del proyecto?

# 14.1 Construcción de la colección de productos
print("\n14.1 CONSTRUCCIÓN DE LA COLECCIÓN DE PRODUCTOS")

products = {
    "benchmark_results": benchmark_results,
    "BENCHMARK_RANKING": BENCHMARK_RANKING,
    "best_models": best_models,
    "official": official,
    "BENCHMARK_DATA": BENCHMARK_DATA,
    "BENCHMARK_CONFIG": BENCHMARK_CONFIG,
    "BENCHMARK_MODELS": BENCHMARK_MODELS,
    "BENCHMARK_EXPORT_RESULT": BENCHMARK_EXPORT_RESULT,
} # Registrar productos oficiales generados por el Benchmark

required_products = [
    "benchmark_results",
    "BENCHMARK_RANKING",
    "best_models",
    "official",
    "BENCHMARK_DATA",
    "BENCHMARK_CONFIG",
    "BENCHMARK_MODELS",
    "BENCHMARK_EXPORT_RESULT",
] # Definir productos obligatorios

missing_products = [
    name
    for name in required_products
    if name not in products
] # Identificar productos faltantes

if missing_products:
    raise KeyError(
        "Productos requeridos no disponibles: "
        f"{missing_products}"
    ) # Validar disponibilidad

null_products = [
    name
    for name in required_products
    if products[name] is None
] # Identificar productos nulos

if null_products:
    raise ValueError(
        "Productos requeridos con valor None: "
        f"{null_products}"
    ) # Validar integridad

if not isinstance(
    products["benchmark_results"],
    list
):
    raise TypeError(
        "benchmark_results debe ser una lista."
    ) # Validar resultados

if not isinstance(
    products["BENCHMARK_RANKING"],
    list
):
    raise TypeError(
        "BENCHMARK_RANKING debe ser una lista."
    ) # Validar ranking

if not isinstance(
    products["best_models"],
    dict
):
    raise TypeError(
        "best_models debe ser un diccionario."
    ) # Validar selección por familia

if not isinstance(
    products["official"],
    dict
):
    raise TypeError(
        "official debe ser un diccionario."
    ) # Validar Modelo Oficial

if not isinstance(
    products["BENCHMARK_CONFIG"],
    dict
):
    raise TypeError(
        "BENCHMARK_CONFIG debe ser un diccionario."
    ) # Validar configuración

if not isinstance(
    products["BENCHMARK_MODELS"],
    dict
):
    raise TypeError(
        "BENCHMARK_MODELS debe ser un diccionario."
    ) # Validar catálogo de modelos

if not isinstance(
    products["BENCHMARK_EXPORT_RESULT"],
    dict
):
    raise TypeError(
        "BENCHMARK_EXPORT_RESULT debe ser un diccionario."
    ) # Validar resultado de exportación

print(f"Productos registrados       : {len(products)}") # Mostrar cantidad de productos
print("Colección de productos     : CONSTRUIDA") # Confirmar construcción

# 14.2 Validación de disponibilidad de los productos
print("\n14.2 VALIDACIÓN DE DISPONIBILIDAD DE LOS PRODUCTOS")

if set(required_products) != set(products.keys()):
    raise RuntimeError(
        "La colección products no coincide exactamente "
        "con el catálogo de productos requeridos."
    ) # Validar correspondencia del catálogo

if len(products["benchmark_results"]) == 0:
    raise RuntimeError(
        "benchmark_results está vacío."
    ) # Validar resultados disponibles

if len(products["BENCHMARK_RANKING"]) == 0:
    raise RuntimeError(
        "BENCHMARK_RANKING está vacío."
    ) # Validar ranking disponible

if len(products["best_models"]) == 0:
    raise RuntimeError(
        "best_models está vacío."
    ) # Validar selección por familia

if len(products["official"]) == 0:
    raise RuntimeError(
        "official está vacío."
    ) # Validar Modelo Oficial

if len(products["BENCHMARK_CONFIG"]) == 0:
    raise RuntimeError(
        "BENCHMARK_CONFIG está vacío."
    ) # Validar configuración

if len(products["BENCHMARK_MODELS"]) == 0:
    raise RuntimeError(
        "BENCHMARK_MODELS está vacío."
    ) # Validar catálogo

if len(products["BENCHMARK_EXPORT_RESULT"]) == 0:
    raise RuntimeError(
        "BENCHMARK_EXPORT_RESULT está vacío."
    ) # Validar resultado de exportación

if products["BENCHMARK_EXPORT_RESULT"].get(
    "status"
) != "SUCCESS":
    raise RuntimeError(
        "BENCHMARK_EXPORT_RESULT no presenta estado SUCCESS."
    ) # Validar estado de exportación

if products["official"].get(
    "model_code"
) != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El Modelo Oficial registrado en products "
        "no coincide con OFFICIAL_MODEL_CODE."
    ) # Validar código oficial

if products["official"].get(
    "model_name",
    ""
).strip().lower() != OFFICIAL_MODEL_NAME.strip().lower():
    raise RuntimeError(
        "El Modelo Oficial registrado en products "
        "no coincide con OFFICIAL_MODEL_NAME."
    ) # Validar nombre oficial

if products["official"].get(
    "family",
    ""
).strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise RuntimeError(
        "La familia del Modelo Oficial registrada en products "
        "no coincide con OFFICIAL_MODEL_FAMILY."
    ) # Validar familia oficial

if len(products["BENCHMARK_RANKING"]) != len(
    products["benchmark_results"]
):
    raise RuntimeError(
        "BENCHMARK_RANKING no contiene todos los resultados "
        "de benchmark_results."
    ) # Validar cobertura del ranking

if len(products["best_models"]) != len(
    products["BENCHMARK_MODELS"]
):
    raise RuntimeError(
        "best_models no contiene todas las familias "
        "definidas en BENCHMARK_MODELS."
    ) # Validar cobertura familiar

print(f"Productos requeridos        : {len(required_products)}") # Mostrar productos requeridos
print(f"Productos disponibles       : {len(products)}") # Mostrar productos disponibles
print("Estructuras de productos    : VALIDADAS") # Confirmar estructuras
print("Modelo Oficial              : VALIDADO") # Confirmar Modelo Oficial
print("Ranking Científico          : VALIDADO") # Confirmar ranking
print("Selección por familias      : VALIDADA") # Confirmar mejores modelos
print("Exportación oficial         : VALIDADA") # Confirmar exportación
print("Disponibilidad de productos : VALIDADA") # Confirmar disponibilidad

# 14.3 Recuperación de los productos oficiales
print("\n14.3 RECUPERACIÓN DE LOS PRODUCTOS OFICIALES")

audit_benchmark_results = products[
    "benchmark_results"
] # Recuperar resultados del Benchmark

audit_benchmark_ranking = products[
    "BENCHMARK_RANKING"
] # Recuperar Ranking Científico

audit_best_models = products[
    "best_models"
] # Recuperar mejores modelos por familia

audit_official = products[
    "official"
] # Recuperar Modelo Oficial

audit_benchmark_data = products[
    "BENCHMARK_DATA"
] # Recuperar datos del Benchmark

audit_benchmark_config = products[
    "BENCHMARK_CONFIG"
] # Recuperar configuración del Benchmark

audit_benchmark_models = products[
    "BENCHMARK_MODELS"
] # Recuperar catálogo de modelos

audit_export_result = products[
    "BENCHMARK_EXPORT_RESULT"
] # Recuperar resultado de exportación

if audit_benchmark_results is not benchmark_results:
    raise RuntimeError(
        "La referencia recuperada de benchmark_results "
        "no coincide con el producto oficial."
    ) # Validar referencia de resultados

if audit_benchmark_ranking is not BENCHMARK_RANKING:
    raise RuntimeError(
        "La referencia recuperada de BENCHMARK_RANKING "
        "no coincide con el producto oficial."
    ) # Validar referencia del ranking

if audit_best_models is not best_models:
    raise RuntimeError(
        "La referencia recuperada de best_models "
        "no coincide con el producto oficial."
    ) # Validar referencia de selección

if audit_official is not official:
    raise RuntimeError(
        "La referencia recuperada de official "
        "no coincide con el producto oficial."
    ) # Validar referencia del Modelo Oficial

if audit_benchmark_config is not BENCHMARK_CONFIG:
    raise RuntimeError(
        "La referencia recuperada de BENCHMARK_CONFIG "
        "no coincide con el producto oficial."
    ) # Validar referencia de configuración

if audit_benchmark_models is not BENCHMARK_MODELS:
    raise RuntimeError(
        "La referencia recuperada de BENCHMARK_MODELS "
        "no coincide con el producto oficial."
    ) # Validar referencia del catálogo

if audit_export_result is not BENCHMARK_EXPORT_RESULT:
    raise RuntimeError(
        "La referencia recuperada de BENCHMARK_EXPORT_RESULT "
        "no coincide con el producto oficial."
    ) # Validar referencia de exportación

print(f"Resultados Benchmark       : {len(audit_benchmark_results)}") # Mostrar resultados
print(f"Ranking Científico         : {len(audit_benchmark_ranking)}") # Mostrar ranking
print(f"Mejores modelos/familia    : {len(audit_best_models)}") # Mostrar selección por familia
print(f"Modelo Oficial             : {audit_official['model_name']}") # Mostrar modelo oficial
print(f"Código Oficial             : {audit_official['model_code']}") # Mostrar código oficial
print(f"Familia Oficial            : {audit_official['family']}") # Mostrar familia oficial
print("Productos oficiales        : RECUPERADOS Y VALIDADOS") # Confirmar recuperación

# 14.4 Auditoría estructural de benchmark_results
print("\n14.4 AUDITORÍA ESTRUCTURAL DE BENCHMARK_RESULTS")

if not isinstance(
    audit_benchmark_results,
    list
):
    raise TypeError(
        "benchmark_results debe ser una lista."
    ) # Validar estructura

if len(audit_benchmark_results) == 0:
    raise ValueError(
        "benchmark_results está vacío."
    ) # Validar existencia

if len(audit_benchmark_results) != len(
    candidate_models
):
    raise RuntimeError(
        "benchmark_results no contiene exactamente "
        "los modelos candidatos evaluados."
    ) # Validar cobertura de modelos

required_result_fields = [
    "model_code",
    "model_name",
    "family",
    "rmse",
    "mae",
    "r2",
    "mape",
] # Definir campos obligatorios

result_identities = []
for position, result in enumerate(
    audit_benchmark_results,
    start=1
):
    if not isinstance(
        result,
        dict
    ):
        raise TypeError(
            f"El resultado {position} no es un diccionario."
        ) # Validar resultado individual

    missing_fields = [
        field
        for field in required_result_fields
        if field not in result
    ] # Identificar campos faltantes

    if missing_fields:
        raise ValueError(
            f"El resultado {position} está incompleto: "
            f"{missing_fields}"
        ) # Validar campos obligatorios

    model_code = str(
        result["model_code"]
    ).strip() # Recuperar código

    model_name = str(
        result["model_name"]
    ).strip().lower() # Recuperar nombre normalizado

    family = str(
        result["family"]
    ).strip().lower() # Recuperar familia normalizada

    if model_code not in BENCHMARK_MODEL_CODES:
        raise ValueError(
            f"El código '{model_code}' no pertenece "
            "a BENCHMARK_MODEL_CODES."
        ) # Validar código oficial

    expected_model_name = BENCHMARK_MODEL_CODES[
        model_code
    ].strip().lower() # Recuperar nombre esperado

    if model_name != expected_model_name:
        raise ValueError(
            f"El modelo '{model_name}' no corresponde "
            f"al código '{model_code}'."
        ) # Validar correspondencia código-nombre

    for metric in BENCHMARK_METRICS:
        if metric not in result:
            raise ValueError(
                f"El modelo '{result['model_name']}' "
                f"no contiene la métrica '{metric}'."
            ) # Validar métrica oficial

        metric_value = result[
            metric
        ] # Recuperar valor de métrica

        if not isinstance(
            metric_value,
            (int, float, np.integer, np.floating)
        ):
            raise TypeError(
                f"La métrica '{metric}' del modelo "
                f"'{result['model_name']}' no es numérica."
            ) # Validar tipo numérico

        if not np.isfinite(
            float(metric_value)
        ):
            raise ValueError(
                f"La métrica '{metric}' del modelo "
                f"'{result['model_name']}' no es finita."
            ) # Validar valor finito
    result_identities.append(
        (
            model_code,
            model_name,
            family,
        )
    ) # Registrar identidad del modelo

if len(result_identities) != len(
    set(result_identities)
):
    raise RuntimeError(
        "benchmark_results contiene modelos duplicados."
    ) # Validar unicidad de resultados

print(f"Resultados evaluados      : {len(audit_benchmark_results)}") # Mostrar resultados
print(f"Métricas oficiales        : {BENCHMARK_METRICS}") # Mostrar métricas
print("Identidad de modelos       : VALIDADA") # Confirmar identidad
print("Unicidad de resultados     : VALIDADA") # Confirmar unicidad
print("Estructura de resultados   : VALIDADA") # Confirmar estructura

# 14.5 Auditoría del Ranking Científico
print("\n14.5 AUDITORÍA DEL RANKING CIENTÍFICO")

if not isinstance(
    audit_benchmark_ranking,
    list
):
    raise TypeError(
        "BENCHMARK_RANKING debe ser una lista."
    ) # Validar estructura

if len(audit_benchmark_ranking) == 0:
    raise ValueError(
        "BENCHMARK_RANKING está vacío."
    ) # Validar existencia

if len(audit_benchmark_ranking) != len(
    audit_benchmark_results
):
    raise RuntimeError(
        "El Ranking no contiene todos los resultados "
        "del Benchmark."
    ) # Validar cobertura

ranking_metric = audit_benchmark_config[
    "ranking_metric"
] # Recuperar métrica oficial

ranking_direction = audit_benchmark_config[
    "metric_directions"
][
    ranking_metric
] # Recuperar dirección oficial

if ranking_metric not in BENCHMARK_METRICS:
    raise ValueError(
        f"La métrica '{ranking_metric}' no pertenece "
        "a BENCHMARK_METRICS."
    ) # Validar métrica

if ranking_direction not in {
    "min",
    "max"
}:
    raise ValueError(
        f"Dirección de ranking inválida: "
        f"{ranking_direction}"
    ) # Validar dirección

required_ranking_fields = [
    "model_code",
    "model_name",
    "family",
    ranking_metric,
    "ranking_position",
] # Definir contrato del Ranking

ranking_identities = []
for position, result in enumerate(
    audit_benchmark_ranking,
    start=1
):
    if not isinstance(
        result,
        dict
    ):
        raise TypeError(
            f"El registro {position} del Ranking "
            "no es un diccionario."
        ) # Validar registro

    missing_fields = [
        field
        for field in required_ranking_fields
        if field not in result
    ] # Identificar campos faltantes

    if missing_fields:
        raise ValueError(
            f"El registro {position} está incompleto: "
            f"{missing_fields}"
        ) # Validar estructura

    if result[
        "ranking_position"
    ] != position:
        raise ValueError(
            f"La posición del modelo "
            f"'{result['model_name']}' "
            f"no coincide con {position}."
        ) # Validar posición

    metric_value = result[
        ranking_metric
    ] # Recuperar métrica

    if not isinstance(
        metric_value,
        (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"La métrica '{ranking_metric}' "
            f"del modelo '{result['model_name']}' "
            "no es numérica."
        ) # Validar tipo de métrica

    if not np.isfinite(
        float(metric_value)
    ):
        raise ValueError(
            f"El modelo '{result['model_name']}' "
            f"contiene un valor inválido para "
            f"'{ranking_metric}'."
        ) # Validar valor de métrica

    ranking_identities.append(
        (
            str(result["model_code"]).strip(),
            str(result["model_name"]).strip().lower(),
            str(result["family"]).strip().lower(),
        )
    ) # Registrar identidad del ranking

if len(ranking_identities) != len(
    set(ranking_identities)
):
    raise RuntimeError(
        "BENCHMARK_RANKING contiene modelos duplicados."
    ) # Validar unicidad

result_identities = {
    (
        str(result["model_code"]).strip(),
        str(result["model_name"]).strip().lower(),
        str(result["family"]).strip().lower(),
    )
    for result in audit_benchmark_results
} # Construir identidades de los resultados

ranking_identity_set = set(
    ranking_identities
) # Construir identidades del ranking

if ranking_identity_set != result_identities:
    raise RuntimeError(
        "BENCHMARK_RANKING no contiene exactamente "
        "los mismos modelos que benchmark_results."
    ) # Validar cobertura exacta

ranking_values = [
    float(result[ranking_metric])
    for result in audit_benchmark_ranking
] # Recuperar valores

expected_values = (
    sorted(ranking_values)
    if ranking_direction == "min"
    else sorted(
        ranking_values,
        reverse=True
    )
) # Construir orden esperado

if ranking_values != expected_values:
    raise RuntimeError(
        f"El Ranking no está ordenado correctamente "
        f"por '{ranking_metric}'."
    ) # Validar orden

print(f"Modelos clasificados       : {len(audit_benchmark_ranking)}") # Mostrar cantidad
print(f"Métrica de ranking         : {ranking_metric.upper()}") # Mostrar métrica
print(f"Dirección de optimización  : {ranking_direction.upper()}") # Mostrar dirección
print("Cobertura exacta            : VALIDADA") # Confirmar cobertura
print("Posiciones                  : VALIDADAS") # Confirmar posiciones
print("Ordenamiento                : VALIDADO") # Confirmar orden
print("Unicidad                    : VALIDADA") # Confirmar unicidad
print("Ranking Científico          : VALIDADO") # Confirmar ranking

# 14.6 Auditoría del Modelo Oficial
print("\n14.6 AUDITORÍA DEL MODELO OFICIAL")

if not isinstance(
    audit_official,
    dict
):
    raise TypeError(
        "official debe ser un diccionario."
    ) # Validar estructura

required_official_fields = [
    "model_code",
    "model_name",
    "family",
    "model_config",
    "ranking_position",
    "rmse",
    "mae",
    "r2",
    "mape",
] # Definir contrato del Modelo Oficial

missing_official_fields = [
    field
    for field in required_official_fields
    if field not in audit_official
] # Identificar campos faltantes

if missing_official_fields:
    raise ValueError(
        "El Modelo Oficial está incompleto: "
        f"{missing_official_fields}"
    ) # Validar contrato

if audit_official["model_code"] != OFFICIAL_MODEL_CODE:
    raise ValueError(
        "El código del Modelo Oficial no coincide "
        "con OFFICIAL_MODEL_CODE."
    ) # Validar código oficial

if audit_official["model_name"].strip().lower() != OFFICIAL_MODEL_NAME.strip().lower():
    raise ValueError(
        "El nombre del Modelo Oficial no coincide "
        "con OFFICIAL_MODEL_NAME."
    ) # Validar nombre oficial

if audit_official["family"].strip().lower() != OFFICIAL_MODEL_FAMILY.strip().lower():
    raise ValueError(
        "La familia del Modelo Oficial no coincide "
        "con OFFICIAL_MODEL_FAMILY."
    ) # Validar familia oficial

if not isinstance(
    audit_official["model_config"],
    dict
):
    raise TypeError(
        "model_config del Modelo Oficial debe ser un diccionario."
    ) # Validar configuración

for metric in BENCHMARK_METRICS:
    if metric not in audit_official:
        raise ValueError(
            f"El Modelo Oficial no contiene la métrica '{metric}'."
        ) # Validar métricas oficiales
    metric_value = audit_official[
        metric
    ] # Recuperar métrica

    if not isinstance(
        metric_value,
        (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"La métrica '{metric}' del Modelo Oficial "
            "no es numérica."
        ) # Validar tipo de métrica

    if not np.isfinite(
        float(metric_value)
    ):
        raise ValueError(
            f"La métrica '{metric}' del Modelo Oficial "
            "no es finita."
        ) # Validar valor de métrica

if not isinstance(
    audit_official["ranking_position"],
    (int, np.integer)
):
    raise TypeError(
        "ranking_position del Modelo Oficial debe ser entero."
    ) # Validar posición

print(f"Código                       : {audit_official['model_code']}") # Mostrar código
print(f"Modelo                       : {audit_official['model_name']}") # Mostrar modelo
print(f"Familia                      : {audit_official['family']}") # Mostrar familia
print(f"Posición Ranking             : {audit_official['ranking_position']}") # Mostrar posición
print(f"RMSE                         : {float(audit_official['rmse']):.6f}") # Mostrar RMSE
print(f"MAE                          : {float(audit_official['mae']):.6f}") # Mostrar MAE
print(f"R2                           : {float(audit_official['r2']):.6f}") # Mostrar R2
print(f"MAPE                         : {float(audit_official['mape']):.6f}") # Mostrar MAPE
print("Identidad del Modelo Oficial : VALIDADA") # Confirmar identidad
print("Configuración del modelo     : VALIDADA") # Confirmar configuración

# 14.7 Verificación del Modelo Oficial dentro del Ranking
print("\n14.7 VERIFICACIÓN DEL MODELO OFICIAL EN EL RANKING")
ranking_model_codes = {
    result["model_code"]
    for result in audit_benchmark_ranking
} # Construir códigos presentes en el Ranking

if audit_official["model_code"] not in ranking_model_codes:
    raise ValueError(
        "El Modelo Oficial no aparece entre los modelos "
        "evaluados por el Benchmark."
    ) # Validar presencia

official_ranking_records = [
    result
    for result in audit_benchmark_ranking
    if result["model_code"] == audit_official["model_code"]
] # Recuperar registro oficial

if len(official_ranking_records) != 1:
    raise ValueError(
        "El Modelo Oficial debe aparecer exactamente una vez "
        "en el Ranking Científico."
    ) # Validar unicidad

official_ranking_record = official_ranking_records[
    0
] # Recuperar registro oficial

if official_ranking_record["model_name"].strip().lower() != audit_official["model_name"].strip().lower():
    raise ValueError(
        "El nombre del Modelo Oficial no coincide "
        "con el Ranking Científico."
    ) # Validar nombre

if official_ranking_record["family"].strip().lower() != audit_official["family"].strip().lower():
    raise ValueError(
        "La familia del Modelo Oficial no coincide "
        "con el Ranking Científico."
    ) # Validar familia

if official_ranking_record["ranking_position"] != audit_official["ranking_position"]:
    raise ValueError(
        "La posición del Modelo Oficial no coincide "
        "con el Ranking Científico."
    ) # Validar posición

for metric in BENCHMARK_METRICS:
    if not np.isclose(
        float(official_ranking_record[metric]),
        float(audit_official[metric])
    ):
        raise ValueError(
            f"La métrica '{metric}' del Modelo Oficial "
            "no coincide con el Ranking Científico."
        ) # Validar consistencia de métricas

gnn_ranking_records = [
    result
    for result in audit_benchmark_ranking
    if result["family"].strip().lower()
    == OFFICIAL_MODEL_FAMILY.strip().lower()
] # Recuperar modelos GNN

if not gnn_ranking_records:
    raise ValueError(
        "No existen modelos GNN en el Ranking Científico."
    ) # Validar existencia de modelos GNN

best_gnn_record = min(
    gnn_ranking_records,
    key=lambda result: result["ranking_position"]
) # Identificar mejor GNN según el Ranking validado

if official_ranking_record["model_code"] != best_gnn_record["model_code"]:
    raise ValueError(
        "El Modelo Oficial no corresponde al mejor modelo "
        "de la familia Graph Neural Networks."
    ) # Validar selección oficial

if audit_official["model_code"] != OFFICIAL_MODEL_CODE:
    raise RuntimeError(
        "El Modelo Oficial auditado no coincide "
        "con OFFICIAL_MODEL_CODE."
    ) # Validar código oficial final

print("Presencia en el Ranking    : VALIDADA") # Confirmar presencia
print(f"Posición en el Ranking    : {official_ranking_record['ranking_position']}") # Mostrar posición
print(f"Mejor GNN                 : {best_gnn_record['model_name']}") # Mostrar mejor GNN
print(f"Código Mejor GNN          : {best_gnn_record['model_code']}") # Mostrar código del mejor GNN
print("Mejor GNN del Ranking      : VALIDADA") # Confirmar selección GNN
print("Consistencia de métricas   : VALIDADA") # Confirmar métricas
print("Consistencia con Ranking   : VALIDADA") # Confirmar consistencia
print("Modelo Oficial             : VALIDADO") # Confirmar Modelo Oficial

# 14.8 Auditoría de BenchmarkData
print("\n14.8 AUDITORÍA DE BENCHMARK_DATA")

if not isinstance(
    audit_benchmark_data,
    dict
):
    raise TypeError(
        "BENCHMARK_DATA debe ser un diccionario."
    ) # Validar estructura

required_benchmark_data = [
    "graphs",
    "train_index",
    "validation_index",
    "test_index",
    "scaler",
] # Definir contrato de BenchmarkData

missing_benchmark_data = [
    field
    for field in required_benchmark_data
    if field not in audit_benchmark_data
] # Identificar campos faltantes

if missing_benchmark_data:
    raise ValueError(
        "BENCHMARK_DATA está incompleto: "
        f"{missing_benchmark_data}"
    ) # Validar contrato

graph_data = audit_benchmark_data[
    "graphs"
] # Recuperar colección GraphData

if not isinstance(
    graph_data,
    (list, tuple)
):
    raise TypeError(
        "BENCHMARK_DATA['graphs'] debe ser una lista o tupla."
    ) # Validar colección

if len(graph_data) == 0:
    raise ValueError(
        "La colección GraphData está vacía."
    ) # Validar existencia

for graph_position, graph in enumerate(
    graph_data,
    start=1
):
    if graph is None:
        raise ValueError(
            f"El GraphData en la posición {graph_position} es None."
        ) # Validar integridad de cada GraphData

partition_names = [
    "train_index",
    "validation_index",
    "test_index",
] # Definir particiones oficiales

partition_indices = {} # Inicializar colección de índices
for index_name in partition_names:
    indices = audit_benchmark_data[
        index_name
    ] # Recuperar índices

    if not isinstance(
        indices,
        (list, tuple, np.ndarray, pd.Index)
    ):
        raise TypeError(
            f"{index_name} debe ser una colección de índices."
        ) # Validar estructura de índices

    if len(indices) == 0:
        raise ValueError(
            f"{index_name} está vacío."
        ) # Validar partición

    partition_indices[index_name] = set(
        int(index)
        for index in indices
    ) # Normalizar índices

if (
    partition_indices["train_index"]
    & partition_indices["validation_index"]
):
    raise RuntimeError(
        "Existe solapamiento entre train_index y validation_index."
    ) # Validar independencia de particiones

if (
    partition_indices["train_index"]
    & partition_indices["test_index"]
):
    raise RuntimeError(
        "Existe solapamiento entre train_index y test_index."
    ) # Validar independencia de particiones

if (
    partition_indices["validation_index"]
    & partition_indices["test_index"]
):
    raise RuntimeError(
        "Existe solapamiento entre validation_index y test_index."
    ) # Validar independencia de particiones

if audit_benchmark_data["scaler"] is None:
    raise ValueError(
        "El escalador del Benchmark es inválido."
    ) # Validar escalador

print(f"GraphData                  : {len(graph_data)}") # Mostrar cantidad de grafos
print(f"Entrenamiento              : {len(partition_indices['train_index'])}") # Mostrar entrenamiento
print(f"Validación                 : {len(partition_indices['validation_index'])}") # Mostrar validación
print(f"Prueba                     : {len(partition_indices['test_index'])}") # Mostrar prueba
print(f"Escalador                  : {type(audit_benchmark_data['scaler']).__name__}") # Mostrar escalador
print("Solapamiento de particiones : NO DETECTADO") # Confirmar independencia
print("BenchmarkData               : VALIDADO") # Confirmar validación

# 14.9 Auditoría del protocolo experimental
print("\n14.9 AUDITORÍA DEL PROTOCOLO EXPERIMENTAL")

if not isinstance(
    audit_benchmark_config,
    dict
):
    raise TypeError(
        "BENCHMARK_CONFIG debe ser un diccionario."
    ) # Validar estructura

required_config_fields = [
    "random_state",
    "train_size",
    "validation_size",
    "test_size",
    "ranking_metric",
    "metric_directions",
] # Definir contrato

missing_config_fields = [
    field
    for field in required_config_fields
    if field not in audit_benchmark_config
] # Identificar campos faltantes

if missing_config_fields:
    raise ValueError(
        "BENCHMARK_CONFIG está incompleto: "
        f"{missing_config_fields}"
    ) # Validar configuración

ranking_metric = audit_benchmark_config[
    "ranking_metric"
] # Recuperar métrica oficial

metric_directions = audit_benchmark_config[
    "metric_directions"
] # Recuperar direcciones oficiales

if ranking_metric not in BENCHMARK_METRICS:
    raise ValueError(
        f"La métrica '{ranking_metric}' no pertenece "
        "a BENCHMARK_METRICS."
    ) # Validar métrica

if not isinstance(
    metric_directions,
    dict
):
    raise TypeError(
        "metric_directions debe ser un diccionario."
    ) # Validar direcciones

missing_metric_directions = [
    metric
    for metric in BENCHMARK_METRICS
    if metric not in metric_directions
] # Identificar direcciones faltantes

if missing_metric_directions:
    raise ValueError(
        "Faltan direcciones para las métricas: "
        f"{missing_metric_directions}"
    ) # Validar cobertura de direcciones

invalid_directions = {
    metric: direction
    for metric, direction in metric_directions.items()
    if direction not in {
        "min",
        "max",
    }
} # Identificar direcciones inválidas

if invalid_directions:
    raise ValueError(
        f"Direcciones de métricas inválidas: "
        f"{invalid_directions}"
    ) # Validar valores permitidos

ranking_direction = metric_directions[
    ranking_metric
] # Recuperar dirección de optimización

if audit_benchmark_config[
    "random_state"
] != PROJECT_SEED:
    raise ValueError(
        "La semilla del Benchmark no coincide con PROJECT_SEED."
    ) # Validar reproducibilidad

partition_names = [
    "train_size",
    "validation_size",
    "test_size",
] # Definir proporciones oficiales

partition_values = {
    name: float(
        audit_benchmark_config[name]
    )
    for name in partition_names
} # Convertir proporciones a valores numéricos

invalid_partition_values = {
    name: value
    for name, value in partition_values.items()
    if not 0.0 < value < 1.0
} # Identificar proporciones inválidas

if invalid_partition_values:
    raise ValueError(
        "Las proporciones de partición deben estar "
        "entre 0 y 1: "
        f"{invalid_partition_values}"
    ) # Validar rango de particiones

partition_total = sum(
    partition_values.values()
) # Calcular proporción total

if not np.isclose(
    partition_total,
    1.0
):
    raise ValueError(
        f"Las proporciones de las particiones no suman 1. "
        f"Total obtenido: {partition_total}"
    ) # Validar suma de particiones

ranking_records_metric = audit_benchmark_ranking[
    0
][ranking_metric] # Recuperar métrica utilizada por el Ranking

if ranking_metric != audit_benchmark_config[
    "ranking_metric"
]:
    raise RuntimeError(
        "La métrica del Ranking Científico no coincide "
        "con BENCHMARK_CONFIG."
    ) # Validar coherencia de métrica

ranking_values = [
    float(
        result[ranking_metric]
    )
    for result in audit_benchmark_ranking
] # Recuperar valores del Ranking

if ranking_direction == "min":
    expected_ranking_values = sorted(
        ranking_values
    ) # Construir orden ascendente esperado
else:
    expected_ranking_values = sorted(
        ranking_values,
        reverse=True
    ) # Construir orden descendente esperado

if ranking_values != expected_ranking_values:
    raise RuntimeError(
        "El Ranking Científico no respeta la dirección "
        "definida en BENCHMARK_CONFIG."
    ) # Validar coherencia del protocolo

print(f"Semilla                   : {audit_benchmark_config['random_state']}") # Mostrar semilla
print(f"Entrenamiento             : {partition_values['train_size']:.2f}") # Mostrar entrenamiento
print(f"Validación                : {partition_values['validation_size']:.2f}") # Mostrar validación
print(f"Prueba                    : {partition_values['test_size']:.2f}") # Mostrar prueba
print(f"Métrica de ranking        : {ranking_metric}") # Mostrar métrica
print(f"Dirección de optimización : {ranking_direction}") # Mostrar dirección
print("Reproducibilidad           : VALIDADA") # Confirmar semilla
print("Particiones                : VALIDADAS") # Confirmar particiones
print("Coherencia con Ranking     : VALIDADA") # Confirmar correspondencia
print("Protocolo experimental     : VALIDADO") # Confirmar protocolo

# 14.10 Auditoría de familias y modelos evaluados
print("\n14.10 AUDITORÍA DE FAMILIAS Y MODELOS")

if not isinstance(
    audit_benchmark_models,
    dict
):
    raise TypeError(
        "BENCHMARK_MODELS debe ser un diccionario."
    ) # Validar estructura

if not audit_benchmark_models:
    raise ValueError(
        "BENCHMARK_MODELS está vacío."
    ) # Validar existencia

configured_families = {
    family.strip().lower()
    for family in audit_benchmark_models
} # Recuperar familias configuradas

official_families = {
    family.strip().lower()
    for family in CANDIDATE_MODELS["families"]
} # Recuperar familias oficiales

if configured_families != official_families:
    missing_families = (
        official_families
        - configured_families
    ) # Identificar familias faltantes

    unexpected_families = (
        configured_families
        - official_families
    ) # Identificar familias inesperadas

    raise RuntimeError(
        "Inconsistencia entre las familias configuradas y "
        "las familias oficiales. "
        f"Faltantes: {sorted(missing_families)}. "
        f"Inesperadas: {sorted(unexpected_families)}."
    ) # Validar cobertura familiar

configured_model_codes = [] # Inicializar códigos configurados
configured_model_identities = [] # Inicializar identidades configuradas

for family, models in audit_benchmark_models.items():
    normalized_family = family.strip().lower() # Normalizar familia
    if not isinstance(
        models,
        (list, tuple)
    ):
        raise TypeError(
            f"La familia '{family}' debe contener una lista o tupla."
        ) # Validar estructura familiar

    if not models:
        raise ValueError(
            f"La familia '{family}' no contiene modelos."
        ) # Validar existencia de modelos

    for model_name in models:
        normalized_model_name = (
            model_name.strip().lower()
        ) # Normalizar nombre
        matching_codes = [
            model_code
            for model_code, configured_name
            in BENCHMARK_MODEL_CODES.items()
            if configured_name.strip().lower()
            == normalized_model_name
        ] # Buscar código oficial

        if len(matching_codes) != 1:
            raise ValueError(
                f"No existe una correspondencia única entre "
                f"'{model_name}' y BENCHMARK_MODEL_CODES."
            ) # Validar correspondencia del código
        model_code = matching_codes[0] # Recuperar código oficial
        configured_model_codes.append(
            model_code.strip().lower()
        ) # Registrar código

        configured_model_identities.append(
            (
                normalized_family,
                model_code.strip().lower(),
                normalized_model_name,
            )
        ) # Registrar identidad completa

if len(configured_model_codes) != len(
    set(configured_model_codes)
):
    raise RuntimeError(
        "BENCHMARK_MODELS contiene modelos duplicados."
    ) # Validar unicidad de modelos configurados

result_model_codes = [
    result["model_code"].strip().lower()
    for result in audit_benchmark_results
] # Recuperar códigos ejecutados

if len(result_model_codes) != len(
    set(result_model_codes)
):
    raise RuntimeError(
        "Los resultados del Benchmark contienen modelos duplicados."
    ) # Validar unicidad de resultados

if len(configured_model_codes) != len(
    result_model_codes
):
    raise RuntimeError(
        "La cantidad de modelos configurados no coincide "
        "con la cantidad de resultados del Benchmark."
    ) # Validar cobertura cuantitativa

if set(configured_model_codes) != set(
    result_model_codes
):
    missing_results = (
        set(configured_model_codes)
        - set(result_model_codes)
    ) # Identificar modelos faltantes

    unexpected_results = (
        set(result_model_codes)
        - set(configured_model_codes)
    ) # Identificar modelos inesperados

    raise RuntimeError(
        "Inconsistencia entre modelos configurados y ejecutados. "
        f"Faltantes: {sorted(missing_results)}. "
        f"Inesperados: {sorted(unexpected_results)}."
    ) # Validar correspondencia de modelos

result_model_identities = [
    (
        result["family"].strip().lower(),
        result["model_code"].strip().lower(),
        result["model_name"].strip().lower(),
    )
    for result in audit_benchmark_results
] # Construir identidades ejecutadas

if len(result_model_identities) != len(
    set(result_model_identities)
):
    raise RuntimeError(
        "Los resultados del Benchmark contienen "
        "identidades de modelos duplicadas."
    ) # Validar unicidad de identidad

if set(configured_model_identities) != set(
    result_model_identities
):
    missing_identities = (
        set(configured_model_identities)
        - set(result_model_identities)
    ) # Identificar identidades faltantes

    unexpected_identities = (
        set(result_model_identities)
        - set(configured_model_identities)
    ) # Identificar identidades inesperadas

    raise RuntimeError(
        "La identidad de los modelos configurados no coincide "
        "con la identidad de los resultados ejecutados. "
        f"Faltantes: {sorted(missing_identities)}. "
        f"Inesperadas: {sorted(unexpected_identities)}."
    ) # Validar identidad completa

candidate_model_identities = {
    (
        candidate["family"].strip().lower(),
        candidate["model_code"].strip().lower(),
        candidate["model_name"].strip().lower(),
    )
    for candidate in candidate_models
} # Recuperar identidades oficiales de candidatos

if set(configured_model_identities) != (
    candidate_model_identities
):
    missing_candidates = (
        candidate_model_identities
        - set(configured_model_identities)
    ) # Identificar candidatos faltantes

    unexpected_candidates = (
        set(configured_model_identities)
        - candidate_model_identities
    ) # Identificar candidatos inesperados

    raise RuntimeError(
        "BENCHMARK_MODELS no coincide con candidate_models. "
        f"Faltantes: {sorted(missing_candidates)}. "
        f"Inesperados: {sorted(unexpected_candidates)}."
    ) # Validar catálogo científico

for result in audit_benchmark_results:
    result_family = result[
        "family"
    ].strip().lower() # Recuperar familia ejecutada

    result_model_name = result[
        "model_name"
    ].strip().lower() # Recuperar nombre ejecutado

    result_model_code = result[
        "model_code"
    ].strip().lower() # Recuperar código ejecutado

    matching_configured_family = [
        family
        for family, models
        in audit_benchmark_models.items()
        if result_model_name in {
            model.strip().lower()
            for model in models
        }
    ] # Recuperar familia configurada del modelo

    if len(matching_configured_family) != 1:
        raise RuntimeError(
            f"No existe una familia configurada única para "
            f"el modelo '{result_model_name}'."
        ) # Validar asignación familiar
    configured_family = (
        matching_configured_family[0].strip().lower()
    ) # Recuperar familia configurada

    if result_family != configured_family:
        raise RuntimeError(
            f"El modelo '{result_model_name}' "
            f"({result_model_code}) fue ejecutado como "
            f"'{result_family}', pero está configurado en "
            f"'{configured_family}'."
        ) # Validar familia del resultado

print(f"Familias evaluadas       : {len(configured_families)}") # Mostrar familias
print(f"Modelos configurados     : {len(configured_model_codes)}") # Mostrar modelos configurados
print(f"Modelos ejecutados       : {len(result_model_codes)}") # Mostrar modelos ejecutados
print("Familias configuradas     : CONSISTENTES") # Confirmar familias
print("Identidad de modelos      : CONSISTENTE") # Confirmar identidad
print("Catálogo vs candidatos    : CONSISTENTES") # Confirmar catálogo
print("Catálogo vs resultados    : CONSISTENTES") # Confirmar ejecución
print("Duplicados                : NO DETECTADOS") # Confirmar unicidad
print("Configuración de modelos  : VALIDADA") # Confirmar auditoría