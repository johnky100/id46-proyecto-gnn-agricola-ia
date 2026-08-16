# 02_graph_analysis.py

# TARGET_VARIABLE: Variable objetivo original del Dataset Científico. ANALYSIS_TARGET_COLUMN: Variable objetivo extraída del GraphData.
# ANALYSIS_RAW_TARGET_COLUMN: Copia de la variable objetivo original para comparación. Actualmente coincide
# numéricamente con ANALYSIS_TARGET_COLUMN porque la variable objetivo no recibe transformación durante build_features.

# BLOQUE 1. CONFIGURACIÓN Y ENTORNO
# Objetivo: Preparar el entorno de ejecución para el análisis científico de los GraphData.
# Arquitectura científica. Entradas: Librerías científicas y configuración centralizada del proyecto.
# Proceso: Inicialización del entorno, dependencias y parámetros oficiales de análisis.
# Producto: Entorno científico preparado para ejecutar el análisis de los GraphData.

# 1.1 Importación de dependencias
from pathlib import Path
import json
import logging
import math

import numpy as np
import pandas as pd
import torch

from src.python.config.config_project import (
    PANEL_YEARS,
    PANEL_ID_COLUMN,
    TIME_COLUMN,
    N_YEARS,
    N_MUNICIPALITIES,
    MUNICIPALITY_ID_COLUMN,
    MUNICIPALITY_NAME_COLUMN,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    GEOMETRY_COLUMN,
    DEPARTMENT_ID_COLUMN,
    DEPARTMENT_NAME_COLUMN,
    N_PANEL_OBSERVATIONS,
    NODE_ID_COLUMN,
    NODE_INDEX_COLUMN,
    TARGET_VARIABLE,
    RAW_TARGET_VARIABLE,
    FEATURE_COLUMNS,
    HIGH_SIMILARITY_THRESHOLD,
    LOW_SIMILARITY_THRESHOLD,
    SIMILARITY_TOPOLOGICAL_ENABLED,
    SIMILARITY_STANDARDIZATION,
    SIMILARITY_PRIMARY_METRIC,
    SIMILARITY_DISTANCE_METRIC,
    SIMILARITY_FEATURE_GROUPS,
    STANDARDIZATION_TOLERANCE,
    SIMILARITY_FEATURE_COLUMNS,
    SIMILARITY_TEMPORAL_ENABLED,
    SIMILARITY_NEIGHBOR_OVERLAP_ENABLED,
    SIMILARITY_TOP_K,
    COMPARISON_DIFFERENCE_METHODS,
    COMPARISON_INCLUDE_TARGET,
    COMPARISON_TARGET_COLUMN,
    COMPARISON_INCLUDE_RAW_TARGET,
    COMPARISON_INCLUDE_SPATIAL_DISTANCE,
    COMPARISON_INCLUDE_SAME_DEPARTMENT,
    STABILITY_THRESHOLD_STABLE,
    STABILITY_THRESHOLD_MODERATE,
    STABILITY_THRESHOLD_VERY_STABLE,
    SIMILARITY_LEVEL_1,
    SIMILARITY_LEVEL_2,
    SIMILARITY_LEVEL_3,
    SIMILARITY_LEVEL_4,
    TOPOLOGY_DEGREE_WEIGHT,
    TOPOLOGY_NEIGHBOR_OVERLAP_WEIGHT,
)

from src.python.config.paths import (
    GRAPH_FILES_DIR,
    GRAPH_DATA_COLLECTION_FILE,
    NODE_CATALOG_FILE,
    ANALYSIS_DIR,
    OUTPUTS_DIR,
    SPATIAL_OUTPUT_PATH,
) # Importar rutas oficiales del proyecto

print(
    "SPATIAL_OUTPUT_PATH:",
    SPATIAL_OUTPUT_PATH
) # Verificar ruta oficial de salida espacial

# 1.2 Configuración del logging
logger = logging.getLogger("graph_analysis")

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    ) # Configurar sistema centralizado de registro

# 1.3 Configuración de rutas de análisis
ANALYSIS_DIR = Path(ANALYSIS_DIR)
ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True
) # Crear carpeta principal para los resultados del análisis

# 1.4 Rutas de salida de las tablas analíticas
PROFILE_OUTPUT_PATH = (
    ANALYSIS_DIR / "municipality_profile.parquet"
) # Perfil municipal municipio-año

COMPARISON_OUTPUT_PATH = (
    ANALYSIS_DIR / "municipality_comparison.parquet"
) # Comparación entre pares de municipios

SIMILARITY_OUTPUT_PATH = (
    ANALYSIS_DIR / "municipality_similarity.parquet"
) # Similitud entre pares de municipios

SIMILARITY_TOP10_OUTPUT_PATH = (
    ANALYSIS_DIR / "municipality_similarity_top10.parquet"
) # Ranking de los municipios más similares

TEMPORAL_SIMILARITY_OUTPUT_PATH = (
    ANALYSIS_DIR / "municipality_temporal_similarity.parquet"
) # Evolución temporal de la similitud

TOPOLOGY_SIMILARITY_OUTPUT_PATH = (
    ANALYSIS_DIR / "municipality_topology_similarity.parquet"
) # Similitud estructural del grafo

SUMMARY_OUTPUT_PATH = (
    ANALYSIS_DIR / "graph_analysis_summary.json"
) # Resumen de la ejecución

# 1.5 Configuración de archivos GraphData
GRAPH_FILE_TEMPLATE = "graph_{year}.pt" # Plantilla de archivos GraphData anuales

NODE_FEATURE_FILE_TEMPLATE = (
    "node_features_{year}.parquet"
) # Plantilla de archivos de características por año

EDGE_INDEX_FILE_TEMPLATE = (
    "edge_index_{year}.parquet"
) # Plantilla de archivos de aristas por año

# 1.6 Configuración numérica del análisis
EPSILON = np.finfo(np.float64).eps # Constante numérica para evitar divisiones por cero
FLOAT_DTYPE = np.float64 # Tipo numérico utilizado para cálculos estadísticos
TORCH_DEVICE = torch.device("cpu") # Dispositivo inicial para el análisis científico

# BLOQUE 2. INICIALIZACIÓN DE LA CONFIGURACIÓN CIENTÍFICA
# Objetivo: Inicializar la configuración oficial que será utilizada durante el análisis científico de los GraphData.
# Arquitectura científica
# Entradas: Parámetros oficiales importados desde src.python.config.config_project.
# Proceso: Organización y preparación de la configuración temporal, estructural, predictora y comparativa.
# Producto: Configuración científica inicializada y disponible para los bloques posteriores del análisis.

# 2.1 Configuración del Panel Científico
ANALYSIS_YEARS = PANEL_YEARS # Años oficiales utilizados durante el análisis
ANALYSIS_N_YEARS = N_YEARS # Número oficial de años del análisis
ANALYSIS_N_MUNICIPALITIES = N_MUNICIPALITIES # Número oficial de municipios
ANALYSIS_N_OBSERVATIONS = N_PANEL_OBSERVATIONS # Número teórico de observaciones municipio-año

# 2.2 Configuración de Identificación Municipal
MUNICIPALITY_KEY = MUNICIPALITY_ID_COLUMN # Columna oficial de identificación municipal
TIME_KEY = TIME_COLUMN # Columna oficial de identificación temporal
PANEL_KEY = PANEL_ID_COLUMN # Identificador único municipio-año
NODE_KEY = NODE_ID_COLUMN # Identificador oficial del nodo
NODE_INDEX_KEY = NODE_INDEX_COLUMN # Índice interno utilizado por PyTorch Geometric
MUNICIPALITY_NAME_KEY = MUNICIPALITY_NAME_COLUMN # Columna oficial del nombre municipal
DEPARTMENT_KEY = DEPARTMENT_ID_COLUMN # Columna oficial de identificación departamental
DEPARTMENT_NAME_KEY = DEPARTMENT_NAME_COLUMN # Columna oficial del nombre del departamento

# 2.3 Configuración Espacial
SPATIAL_COLUMNS = [
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN
] # Coordenadas oficiales utilizadas para el análisis espacial

SPATIAL_GEOMETRY_COLUMN = GEOMETRY_COLUMN # Columna oficial de geometría

# 2.4 Configuración de Variables Predictoras
ANALYSIS_FEATURE_COLUMNS = FEATURE_COLUMNS # Variables predictoras oficiales utilizadas en el análisis
ANALYSIS_FEATURE_GROUPS = SIMILARITY_FEATURE_GROUPS # Agrupación científica de las variables predictoras
ANALYSIS_N_FEATURES = len(ANALYSIS_FEATURE_COLUMNS) # Número total de variables predictoras utilizadas

# 2.5 Configuración de la Variable Objetivo
ANALYSIS_TARGET_COLUMN = TARGET_VARIABLE # Variable objetivo oficial del análisis
ANALYSIS_RAW_TARGET_COLUMN = RAW_TARGET_VARIABLE # Variable objetivo en escala original

# 2.6 Configuración de Similitud Municipal
STANDARDIZATION_METHOD = SIMILARITY_STANDARDIZATION # Método oficial de estandarización
PRIMARY_SIMILARITY_METRIC = SIMILARITY_PRIMARY_METRIC # Métrica principal de similitud
DISTANCE_METRIC = SIMILARITY_DISTANCE_METRIC # Métrica complementaria de distancia
TOP_K = SIMILARITY_TOP_K # Número de municipios similares que serán conservados

# 2.7 Configuración de Análisis Temporal y Topológico
ENABLE_TEMPORAL_ANALYSIS = SIMILARITY_TEMPORAL_ENABLED # Activar análisis temporal
ENABLE_TOPOLOGICAL_ANALYSIS = SIMILARITY_TOPOLOGICAL_ENABLED # Activar análisis topológico
ENABLE_NEIGHBOR_OVERLAP = SIMILARITY_NEIGHBOR_OVERLAP_ENABLED # Activar cálculo de overlap entre vecinos

# 2.8 Configuración de Comparación Municipal
DIFFERENCE_METHODS = COMPARISON_DIFFERENCE_METHODS # Métodos oficiales para calcular diferencias
TARGET_COMPARISON_COLUMN = COMPARISON_TARGET_COLUMN # Variable utilizada para comparar resultados
INCLUDE_TARGET_COMPARISON = COMPARISON_INCLUDE_TARGET # Incluir comparación de la variable objetivo
INCLUDE_RAW_TARGET_COMPARISON = COMPARISON_INCLUDE_RAW_TARGET # Incluir comparación de la variable objetivo en escala original
INCLUDE_SPATIAL_DISTANCE = COMPARISON_INCLUDE_SPATIAL_DISTANCE # Incluir distancia geográfica
INCLUDE_SAME_DEPARTMENT = COMPARISON_INCLUDE_SAME_DEPARTMENT # Identificar pertenencia al mismo departamento

# 2.9 Registro de la Configuración
logger.info(
    "Panel científico: %s-%s | %s municipios | %s variables predictoras",
    ANALYSIS_YEARS[0],
    ANALYSIS_YEARS[-1],
    ANALYSIS_N_MUNICIPALITIES,
    ANALYSIS_N_FEATURES
) # Registrar la configuración principal del análisis

# BLOQUE 3. DESCUBRIMIENTO Y CARGA DE GRAPHDATA
# Objetivo: Localizar y cargar los GraphData oficiales correspondientes al periodo científico del proyecto.
# Arquitectura científica
# Entradas: Archivos GraphData anuales generados por 01_build_graph.py.
# Proceso: Descubrimiento de archivos, correspondencia con los años oficiales y carga de los GraphData.
# Producto: Colección de GraphData anual disponible para los análisis posteriores.

# 3.1 Carga de los GraphData oficiales
graph_files = {} # Diccionario para almacenar las rutas de los GraphData por año
graph_data_by_year = {} # Diccionario para almacenar los GraphData cargados por año

for current_year in ANALYSIS_YEARS:
    graph_file = GRAPH_FILES_DIR / f"graph_{current_year}.pt" # Ruta oficial del GraphData anual
    graph_files[current_year] = graph_file # Registrar la ruta del GraphData correspondiente al año
    if not graph_file.exists():
        raise FileNotFoundError(
            f"No se encontró el GraphData oficial del año {current_year}: {graph_file}"
        )

    graph_data_by_year[current_year] = torch.load(
        graph_file,
        map_location="cpu",
        weights_only=False
    ) # Cargar GraphData anual en memoria

logger.info(
    "GraphData descubiertos: %s de %s años científicos.",
    len(graph_data_by_year),
    ANALYSIS_N_YEARS
) # Registrar el número de GraphData cargados

# 3.2 Carga del Catálogo Oficial de Nodos
node_catalog = pd.read_parquet(
    NODE_CATALOG_FILE
) # Cargar el Catálogo Oficial de Nodos

if node_catalog.empty:
    raise ValueError(
        "El Catálogo Oficial de Nodos está vacío."
    )

logger.info(
    "Catálogo Oficial de Nodos cargado: %s registros.",
    len(node_catalog)
) # Registrar cantidad de nodos cargados

# BLOQUE 4. VALIDACIÓN DE ENTRADA
# Objetivo: Validar la integridad estructural y dimensional de los GraphData antes de iniciar el análisis municipal.
# Arquitectura científica
# Entradas: GraphData cargados en graph_data_by_year y configuración científica oficial.
# Proceso: Validación temporal, estructural, dimensional, identificadora y de valores de los GraphData.
# Producto: Colección de GraphData validada y lista para el análisis científico.

# 4.1 Validación de la Colección Temporal
loaded_years = sorted(graph_data_by_year.keys()) # Obtener los años de los GraphData cargados
expected_years = sorted(ANALYSIS_YEARS) # Obtener los años oficiales del análisis
if loaded_years != expected_years:
    raise ValueError(
        f"Los años cargados no coinciden con los años oficiales. "
        f"Esperados: {expected_years}. Encontrados: {loaded_years}."
    )

# 4.2 Validación del Número de GraphData
if len(graph_data_by_year) != ANALYSIS_N_YEARS:
    raise ValueError(
        f"El número de GraphData no coincide con la configuración. "
        f"Esperados: {ANALYSIS_N_YEARS}. Encontrados: {len(graph_data_by_year)}."
    )

# 4.3 Validación de la Estructura GraphData
required_graph_attributes = [
    "x",
    "edge_index",
    "y",
    "num_nodes",
    "current_year",
    "graph_type",
    "builder",
    "builder_version",
] # Atributos estructurales y de trazabilidad actualmente requeridos

for current_year, graph_data in graph_data_by_year.items():
    for attribute in required_graph_attributes:
        if not hasattr(graph_data, attribute):
            raise ValueError(
                f"El GraphData del año {current_year} no contiene "
                f"el atributo requerido '{attribute}'."
            ) # Validar presencia de atributos obligatorios

    if graph_data.current_year != current_year:
        raise ValueError(
            f"El GraphData cargado para el año {current_year} "
            f"declara internamente el año {graph_data.current_year}."
        ) # Validar correspondencia entre archivo y año interno

    if graph_data.graph_type != "official_graphdata":
        raise ValueError(
            f"El GraphData del año {current_year} tiene "
            f"graph_type='{graph_data.graph_type}'; se esperaba "
            f"'official_graphdata'."
        ) # Validar tipo oficial del GraphData

    if graph_data.builder != "build_graphdata":
        raise ValueError(
            f"El GraphData del año {current_year} fue generado por "
            f"'{graph_data.builder}'; se esperaba 'build_graphdata'."
        ) # Validar constructor oficial del GraphData

    if not isinstance(graph_data.builder_version, str):
        raise ValueError(
            f"El atributo builder_version del GraphData del año "
            f"{current_year} debe ser una cadena de texto."
        ) # Validar formato de la versión del constructor

# 4.4 Validación de Node Features
for current_year, graph_data in graph_data_by_year.items():
    if graph_data.x is None:
        raise ValueError(
            f"El GraphData del año {current_year} no contiene Node Features."
        )
    if graph_data.x.ndim != 2:
        raise ValueError(
            f"Las Node Features del año {current_year} deben ser bidimensionales. "
            f"Dimensiones encontradas: {tuple(graph_data.x.shape)}."
        )
    if graph_data.x.shape[0] != ANALYSIS_N_MUNICIPALITIES:
        raise ValueError(
            f"El GraphData del año {current_year} contiene "
            f"{graph_data.x.shape[0]} nodos; se esperaban "
            f"{ANALYSIS_N_MUNICIPALITIES}."
        )
    if graph_data.x.shape[1] != ANALYSIS_N_FEATURES:
        raise ValueError(
            f"El GraphData del año {current_year} contiene "
            f"{graph_data.x.shape[1]} variables predictoras; se esperaban "
            f"{ANALYSIS_N_FEATURES}."
        )

# 4.5 Validación de Edge Index
for current_year, graph_data in graph_data_by_year.items():
    edge_index = graph_data.edge_index # Obtener estructura de aristas
    if edge_index is None:
        raise ValueError(
            f"El GraphData del año {current_year} no contiene edge_index."
        )

    if edge_index.ndim != 2:
        raise ValueError(
            f"El edge_index del año {current_year} debe ser bidimensional. "
            f"Dimensiones encontradas: {tuple(edge_index.shape)}."
        )

    if edge_index.shape[0] != 2:
        raise ValueError(
            f"El edge_index del año {current_year} debe tener dos filas. "
            f"Dimensiones encontradas: {tuple(edge_index.shape)}."
        )

    if edge_index.dtype != torch.long:
        raise ValueError(
            f"El edge_index del año {current_year} debe utilizar "
            f"el tipo torch.long. Tipo encontrado: {edge_index.dtype}."
        )

    if edge_index.numel() > 0:

        if edge_index.min().item() < 0:
            raise ValueError(
                f"El edge_index del año {current_year} contiene "
                f"índices negativos."
            )

        if edge_index.max().item() >= graph_data.num_nodes:
            raise ValueError(
                f"El edge_index del año {current_year} contiene "
                f"índices fuera del rango de los nodos."
            )

    n_edges = edge_index.shape[1] # Obtener número de aristas del GraphData
    if hasattr(graph_data, "edge_attr") and graph_data.edge_attr is not None:
        edge_attr = graph_data.edge_attr # Obtener atributos de las aristas

        if edge_attr.ndim != 2:
            raise ValueError(
                f"El edge_attr del año {current_year} debe ser "
                f"bidimensional. Dimensiones encontradas: "
                f"{tuple(edge_attr.shape)}."
            )

        if edge_attr.shape[0] != n_edges:
            raise ValueError(
                f"El edge_attr del año {current_year} contiene "
                f"{edge_attr.shape[0]} registros, pero edge_index "
                f"contiene {n_edges} aristas."
            )

        if not torch.is_floating_point(edge_attr):
            raise ValueError(
                f"El edge_attr del año {current_year} debe contener "
                f"valores numéricos de punto flotante."
            )

        if not torch.isfinite(edge_attr).all():
            raise ValueError(
                f"El edge_attr del año {current_year} contiene "
                f"NaN o valores infinitos."
            )

# 4.6 Validación de la Variable Objetivo
for current_year, graph_data in graph_data_by_year.items():
    if graph_data.y is None:
        raise ValueError(
            f"El GraphData del año {current_year} no contiene "
            f"la variable objetivo."
        )

    if graph_data.y.ndim != 2:
        raise ValueError(
            f"La variable objetivo del año {current_year} debe ser "
            f"bidimensional. Dimensiones encontradas: "
            f"{tuple(graph_data.y.shape)}."
        )

    if graph_data.y.shape[0] != ANALYSIS_N_MUNICIPALITIES:
        raise ValueError(
            f"La variable objetivo del año {current_year} contiene "
            f"{graph_data.y.shape[0]} observaciones; se esperaban "
            f"{ANALYSIS_N_MUNICIPALITIES}."
        )

    if graph_data.y.shape[1] != 1:
        raise ValueError(
            f"La variable objetivo del año {current_year} debe contener "
            f"una única variable objetivo. Dimensiones encontradas: "
            f"{tuple(graph_data.y.shape)}."
        )

# 4.7 Validación de Valores Numéricos
for current_year, graph_data in graph_data_by_year.items():
    if not torch.is_floating_point(graph_data.x):
        raise ValueError(
            f"Las Node Features del año {current_year} "
            f"no tienen un tipo numérico de punto flotante."
        )

    if not torch.is_floating_point(graph_data.y):
        raise ValueError(
            f"La variable objetivo del año {current_year} "
            f"no tiene un tipo numérico de punto flotante."
        )

    if not torch.isfinite(graph_data.x).all():
        raise ValueError(
            f"Las Node Features del año {current_year} contienen "
            f"NaN o valores infinitos."
        )

    if not torch.isfinite(graph_data.y).all():
        raise ValueError(
            f"La variable objetivo del año {current_year} contiene "
            f"NaN o valores infinitos."
        )

# 4.8 Validación de la Estructura de Nodos
for current_year, graph_data in graph_data_by_year.items():
    if graph_data.num_nodes != ANALYSIS_N_MUNICIPALITIES:
        raise ValueError(
            f"El GraphData del año {current_year} contiene "
            f"{graph_data.num_nodes} nodos; se esperaban "
            f"{ANALYSIS_N_MUNICIPALITIES}."
        ) # Validar cantidad oficial de nodos

    if graph_data.x.shape[0] != graph_data.num_nodes:
        raise ValueError(
            f"La matriz x del año {current_year} no coincide "
            f"con el número de nodos del GraphData."
        ) # Validar correspondencia entre x y nodos

    if graph_data.y.shape[0] != graph_data.num_nodes:
        raise ValueError(
            f"La variable objetivo del año {current_year} no coincide "
            f"con el número de nodos del GraphData."
        ) # Validar correspondencia entre y y nodos

# 4.9 Validación de la Correspondencia Estructural entre Años
# 4.9.1 Validación de la Identidad Estructural del Catálogo Oficial
if NODE_INDEX_KEY not in node_catalog.columns:
    raise ValueError(
        f"El Catálogo Oficial de Nodos no contiene la columna "
        f"'{NODE_INDEX_KEY}'."
    ) # Validar existencia del índice oficial de nodos

if len(node_catalog) != ANALYSIS_N_MUNICIPALITIES:
    raise ValueError(
        f"El Catálogo Oficial de Nodos contiene "
        f"{len(node_catalog)} nodos; se esperaban "
        f"{ANALYSIS_N_MUNICIPALITIES}."
    ) # Validar cantidad oficial de nodos

catalog_node_index = node_catalog[
    NODE_INDEX_KEY
] # Obtener los índices oficiales de los nodos

if catalog_node_index.isnull().any():
    raise ValueError(
        "El Catálogo Oficial de Nodos contiene índices node_idx faltantes."
    ) # Validar ausencia de índices faltantes

if not pd.api.types.is_integer_dtype(
    catalog_node_index
):
    raise ValueError(
        f"La columna '{NODE_INDEX_KEY}' del Catálogo Oficial de Nodos "
        f"debe contener índices enteros."
    ) # Validar tipo entero del índice nodal

if catalog_node_index.duplicated().any():
    raise ValueError(
        "El Catálogo Oficial de Nodos contiene índices node_idx duplicados."
    ) # Validar unicidad del índice nodal

expected_node_index = np.arange(
    ANALYSIS_N_MUNICIPALITIES
) # Construir la secuencia oficial esperada de índices

if not np.array_equal(
    np.sort(catalog_node_index.to_numpy()),
    expected_node_index
):
    raise ValueError(
        "El Catálogo Oficial de Nodos no contiene la secuencia "
        "completa de índices node_idx desde 0 hasta N-1."
    ) # Validar integridad de la numeración nodal

# 4.9.2 Validación del GraphData de Referencia
reference_year = ANALYSIS_YEARS[0] # Definir el año de referencia estructural
reference_graph = graph_data_by_year[
    reference_year
] # Obtener el GraphData del año de referencia

reference_num_nodes = reference_graph.num_nodes # Obtener número de nodos del año de referencia
if reference_num_nodes != ANALYSIS_N_MUNICIPALITIES:
    raise ValueError(
        f"El GraphData del año {reference_year} contiene "
        f"{reference_num_nodes} nodos; se esperaban "
        f"{ANALYSIS_N_MUNICIPALITIES}."
    ) # Validar cantidad oficial de nodos

if reference_num_nodes != len(node_catalog):
    raise ValueError(
        f"El GraphData del año {reference_year} contiene "
        f"{reference_num_nodes} nodos, mientras que el Catálogo "
        f"Oficial de Nodos contiene {len(node_catalog)}."
    ) # Validar correspondencia entre GraphData y catálogo

# 4.9.3 Validación de la Consistencia Nodal entre Años
for current_year in ANALYSIS_YEARS[1:]:
    current_graph = graph_data_by_year[
        current_year
    ] # Obtener GraphData del año actual

    current_num_nodes = current_graph.num_nodes # Obtener número de nodos del año actual
    if current_num_nodes != reference_num_nodes:
        raise ValueError(
            f"La estructura nodal del año {current_year} "
            f"no coincide con el año {reference_year}. "
            f"Referencia: {reference_num_nodes}. "
            f"Actual: {current_num_nodes}."
        ) # Validar estabilidad del número de nodos entre años

    if current_num_nodes != ANALYSIS_N_MUNICIPALITIES:
        raise ValueError(
            f"El GraphData del año {current_year} contiene "
            f"{current_num_nodes} nodos; se esperaban "
            f"{ANALYSIS_N_MUNICIPALITIES}."
        ) # Validar cantidad oficial de nodos

    if current_num_nodes != len(node_catalog):
        raise ValueError(
            f"El GraphData del año {current_year} contiene "
            f"{current_num_nodes} nodos, mientras que el Catálogo "
            f"Oficial de Nodos contiene {len(node_catalog)}."
        ) # Validar correspondencia con el catálogo oficial

# 4.9.4 Confirmación de la Correspondencia Estructural
logger.info(
    "Correspondencia estructural validada: %s nodos consistentes "
    "entre %s y %s.",
    ANALYSIS_N_MUNICIPALITIES,
    ANALYSIS_YEARS[0],
    ANALYSIS_YEARS[-1]
) # Registrar consistencia estructural del panel nodal

# 4.10 Confirmación de la Validación
logger.info(
    "Validación de entrada de GraphData completada correctamente."
) # Registrar finalización de la validación

logger.info(
    "GraphData validados: %s.",
    len(graph_data_by_year)
) # Registrar número de GraphData validados

logger.info(
    "Periodo científico validado: %s-%s.",
    ANALYSIS_YEARS[0],
    ANALYSIS_YEARS[-1]
) # Registrar periodo temporal validado

logger.info(
    "Municipios validados: %s.",
    ANALYSIS_N_MUNICIPALITIES
) # Registrar número oficial de municipios

logger.info(
    "Variables predictoras validadas: %s.",
    ANALYSIS_N_FEATURES
) # Registrar número de variables predictoras

logger.info(
    "Índice nodal oficial: %s.",
    NODE_INDEX_KEY
) # Registrar índice territorial definido en el Catálogo Oficial de Nodos

logger.info(
    "Variable objetivo: %s.",
    ANALYSIS_TARGET_COLUMN
) # Registrar variable objetivo oficial

logger.info(
    "Estructura validada: x, edge_index, y, num_nodes y metadatos "
    "de trazabilidad."
) # Registrar componentes estructurales y de trazabilidad validados

logger.info(
    "Consistencia nodal validada entre el Catálogo Oficial de Nodos "
    "y los GraphData del periodo científico."
) # Registrar consistencia estructural entre catálogo y GraphData

logger.info(
    "Estado de entrada: VALIDADO."
) # Confirmar que los GraphData superaron las validaciones de entrada

# BLOQUE 5. CONSTRUCCIÓN DEL PERFIL MUNICIPAL
# Objetivo: Construir el perfil científico municipio-año a partir de los GraphData validados.
# Arquitectura científica
# Entradas: GraphData validados, variables predictoras, variable objetivo y catálogo oficial de nodos.
# Proceso: Extracción de características y objetivos desde GraphData y recuperación de la identidad municipal mediante NODE_CATALOG_FILE.
# Producto: municipality_profile.parquet con el perfil científico de cada municipio para cada año.

# 5.1 Validación del Catálogo Oficial de Nodos
required_node_catalog_columns = [
    NODE_INDEX_COLUMN,
    NODE_ID_COLUMN,
    MUNICIPALITY_ID_COLUMN,
    MUNICIPALITY_NAME_COLUMN
] # Definir columnas mínimas requeridas del catálogo oficial

missing_node_catalog_columns = [
    column
    for column in required_node_catalog_columns
    if column not in node_catalog.columns
] # Identificar columnas faltantes del catálogo

if missing_node_catalog_columns:
    raise ValueError(
        f"El catálogo oficial de nodos no contiene las columnas requeridas: "
        f"{missing_node_catalog_columns}"
    ) # Detener la ejecución si faltan columnas esenciales

if node_catalog.empty:
    raise ValueError(
        "El catálogo oficial de nodos está vacío."
    ) # Validar que exista información de nodos

if node_catalog[NODE_INDEX_COLUMN].duplicated().any():
    raise ValueError(
        f"El catálogo oficial contiene valores duplicados en '{NODE_INDEX_COLUMN}'."
    ) # Validar unicidad del índice interno

if node_catalog[NODE_ID_COLUMN].duplicated().any():
    raise ValueError(
        f"El catálogo oficial contiene valores duplicados en '{NODE_ID_COLUMN}'."
    ) # Validar unicidad de la identidad científica del nodo

if node_catalog[MUNICIPALITY_ID_COLUMN].duplicated().any():
    raise ValueError(
        f"El catálogo oficial contiene valores duplicados en '{MUNICIPALITY_ID_COLUMN}'."
    ) # Validar unicidad de la identidad municipal

if node_catalog[
    [
        NODE_INDEX_COLUMN,
        NODE_ID_COLUMN,
        MUNICIPALITY_ID_COLUMN,
        MUNICIPALITY_NAME_COLUMN
    ]
].isna().any().any():
    raise ValueError(
        "El catálogo oficial contiene valores nulos en las columnas de identidad."
    ) # Validar integridad de la identidad de los nodos

expected_node_indices = np.arange(
    ANALYSIS_N_MUNICIPALITIES,
    dtype=np.int64
) # Definir secuencia oficial esperada de índices internos

catalog_node_indices = np.sort(
    node_catalog[NODE_INDEX_COLUMN].to_numpy(dtype=np.int64)
) # Obtener índices internos ordenados del catálogo

if not np.array_equal(
    catalog_node_indices,
    expected_node_indices
):
    raise ValueError(
        f"El catálogo oficial no contiene la secuencia completa de índices "
        f"internos esperada entre 0 y {ANALYSIS_N_MUNICIPALITIES - 1}."
    ) # Validar correspondencia completa con los nodos esperados

if len(node_catalog) != ANALYSIS_N_MUNICIPALITIES:
    raise ValueError(
        f"El catálogo oficial contiene {len(node_catalog):,} nodos; "
        f"se esperaban {ANALYSIS_N_MUNICIPALITIES:,}."
    ) # Validar cantidad oficial de municipios

# 5.2 Preparación de la Identidad Oficial de los Nodos
node_identity = node_catalog[
    [
        NODE_INDEX_COLUMN,
        NODE_ID_COLUMN,
        MUNICIPALITY_ID_COLUMN,
        MUNICIPALITY_NAME_COLUMN
    ]
].copy() # Crear tabla oficial de correspondencia entre node_idx, node_id e identidad municipal

node_identity = node_identity.sort_values(
    by=NODE_INDEX_COLUMN
).reset_index(drop=True) # Ordenar la identidad según el índice interno del grafo

# 5.3 Inicialización del Perfil Municipal
municipality_profiles = [] # Lista para almacenar los perfiles municipales de cada año

# 5.4 Extracción de Información por Año
for current_year in ANALYSIS_YEARS:
    graph_data = graph_data_by_year[
        current_year
    ] # Obtener el GraphData validado del año actual

    feature_values = (
        graph_data.x
        .detach()
        .cpu()
        .numpy()
    ) # Convertir Node Features a matriz NumPy

    target_values = (
        graph_data.y
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    ) # Convertir variable objetivo oficial a vector NumPy

    if feature_values.ndim != 2:
        raise ValueError(
            f"Las Node Features del GraphData {current_year} "
            "deben ser una matriz bidimensional."
        ) # Validar estructura matricial de las características

    if feature_values.shape[0] != ANALYSIS_N_MUNICIPALITIES:
        raise ValueError(
            f"El GraphData {current_year} contiene "
            f"{feature_values.shape[0]:,} nodos; "
            f"se esperaban {ANALYSIS_N_MUNICIPALITIES:,}."
        ) # Validar cantidad de nodos

    if feature_values.shape[1] != ANALYSIS_N_FEATURES:
        raise ValueError(
            f"El GraphData {current_year} contiene "
            f"{feature_values.shape[1]} variables predictoras; "
            f"se esperaban {ANALYSIS_N_FEATURES}."
        ) # Validar cantidad de variables predictoras

    if len(target_values) != ANALYSIS_N_MUNICIPALITIES:
        raise ValueError(
            f"El GraphData {current_year} contiene "
            f"{len(target_values):,} valores objetivo; "
            f"se esperaban {ANALYSIS_N_MUNICIPALITIES:,}."
        ) # Validar cantidad de valores objetivo

    if not np.isfinite(feature_values).all():
        raise ValueError(
            f"Las variables predictoras del GraphData {current_year} "
            "contienen valores no finitos."
        ) # Validar valores numéricos de las características

    if not np.isfinite(target_values).all():
        raise ValueError(
            f"La variable objetivo del GraphData {current_year} "
            "contiene valores no finitos."
        ) # Validar valores numéricos del objetivo

    year_profile = pd.DataFrame(
        feature_values,
        columns=ANALYSIS_FEATURE_COLUMNS
    ) # Construir tabla de variables predictoras del año

    year_profile[NODE_INDEX_COLUMN] = np.arange(
        ANALYSIS_N_MUNICIPALITIES,
        dtype=np.int64
    ) # Recuperar el orden nodal establecido previamente mediante node_idx

    year_profile[TIME_KEY] = current_year # Incorporar año oficial del GraphData
    year_profile[ANALYSIS_TARGET_COLUMN] = target_values # Incorporar variable objetivo oficial

    # 5.5 Recuperación de la Identidad Municipal desde el Catálogo
    year_profile = year_profile.merge(
        node_identity,
        on=NODE_INDEX_COLUMN,
        how="left",
        validate="one_to_one"
    ) # Recuperar node_id e identidad municipal mediante node_idx

    if len(year_profile) != ANALYSIS_N_MUNICIPALITIES:
        raise ValueError(
            f"La recuperación de identidad del año {current_year} "
            f"produjo {len(year_profile):,} registros; "
            f"se esperaban {ANALYSIS_N_MUNICIPALITIES:,}."
        ) # Validar conservación del número de municipios después del merge

    identity_columns = [
        NODE_ID_COLUMN,
        MUNICIPALITY_ID_COLUMN,
        MUNICIPALITY_NAME_COLUMN
    ] # Definir columnas esenciales de identidad recuperadas

    if year_profile[identity_columns].isna().any().any():
        raise ValueError(
            f"No fue posible recuperar completamente la identidad municipal "
            f"para todos los nodos del año {current_year}."
        ) # Validar recuperación completa de identidad

    if year_profile[NODE_ID_COLUMN].duplicated().any():
        raise ValueError(
            f"Se detectaron identificadores de nodo duplicados "
            f"en el perfil del año {current_year}."
        ) # Validar unicidad de node_id

    if year_profile[MUNICIPALITY_ID_COLUMN].duplicated().any():
        raise ValueError(
            f"Se detectaron identificadores municipales duplicados "
            f"en el perfil del año {current_year}."
        ) # Validar unicidad municipal
        
    # 5.6 Incorporación de la Variable Objetivo para Comparación
    if COMPARISON_INCLUDE_RAW_TARGET:
        raise ValueError(
            f"La variable objetivo original "
            f"'{ANALYSIS_RAW_TARGET_COLUMN}' no está disponible "
            "en el GraphData oficial."
        ) # Evitar utilizar una variable externa no contenida en GraphData

    municipality_profiles.append(
        year_profile
    ) # Agregar perfil anual a la colección

# 5.7 Consolidación del Perfil Municipal
municipality_profile = pd.concat(
    municipality_profiles,
    ignore_index=True
) # Consolidar todos los perfiles anuales en una única tabla

# 5.8 Ordenamiento del Perfil Municipal
municipality_profile = municipality_profile.sort_values(
    by=[TIME_KEY, NODE_INDEX_COLUMN]
).reset_index(drop=True) # Ordenar por año e índice interno del grafo

# 5.9 Validación Básica del Perfil Municipal
expected_profile_rows = (
    ANALYSIS_N_MUNICIPALITIES
    * ANALYSIS_N_YEARS
) # Calcular número esperado de registros municipio-año

if len(municipality_profile) != expected_profile_rows:
    raise ValueError(
        f"El perfil municipal contiene {len(municipality_profile):,} registros; "
        f"se esperaban {expected_profile_rows:,}."
    ) # Validar cobertura total del panel

# 5.10 Validación de Cobertura Temporal y Municipal
for current_year in ANALYSIS_YEARS:
    year_profile = municipality_profile[
        municipality_profile[TIME_KEY] == current_year
    ] # Seleccionar el perfil del año actual

    if len(year_profile) != ANALYSIS_N_MUNICIPALITIES:
        raise ValueError(
            f"El año {current_year} contiene "
            f"{len(year_profile):,} municipios; "
            f"se esperaban {ANALYSIS_N_MUNICIPALITIES:,}."
        ) # Validar cobertura municipal anual

    if year_profile[NODE_INDEX_COLUMN].duplicated().any():
        raise ValueError(
            f"El año {current_year} contiene índices internos duplicados."
        ) # Validar unicidad de node_idx

    if year_profile[NODE_ID_COLUMN].duplicated().any():
        raise ValueError(
            f"El año {current_year} contiene identificadores de nodo duplicados."
        ) # Validar unicidad de node_id

    if year_profile[MUNICIPALITY_ID_COLUMN].duplicated().any():
        raise ValueError(
            f"El año {current_year} contiene identificadores municipales duplicados."
        ) # Validar unicidad municipal

# 5.11 Validación de Duplicados Municipio-Año
duplicate_profile_rows = municipality_profile.duplicated(
    subset=[MUNICIPALITY_ID_COLUMN, TIME_KEY]
).sum() # Contar duplicados municipio-año

if duplicate_profile_rows > 0:
    raise ValueError(
        f"El perfil municipal contiene "
        f"{duplicate_profile_rows:,} registros municipio-año duplicados."
    ) # Detener ejecución ante duplicados municipio-año

# 5.12 Validación de Identidad Científica
identity_columns = [
    NODE_ID_COLUMN,
    NODE_INDEX_COLUMN
] # Definir identificadores científicos obligatorios

if municipality_profile[
    identity_columns
].isna().any().any():
    raise ValueError(
        "El perfil municipal contiene identificadores de nodo faltantes."
    ) # Validar integridad de la identidad científica

duplicate_node_pairs = (
    municipality_profile[
        [
            TIME_KEY,
            NODE_ID_COLUMN,
            NODE_INDEX_COLUMN
        ]
    ]
    .duplicated()
    .sum()
) # Contar duplicados dentro de la unidad municipio-año

if duplicate_node_pairs > 0:
    raise ValueError(
        "El perfil municipal contiene registros duplicados "
        "para la misma combinación de año, node_id y node_idx."
    ) # Validar unicidad de la unidad analítica municipio-año

node_mapping = (
    municipality_profile[
        [
            NODE_ID_COLUMN,
            NODE_INDEX_COLUMN
        ]
    ]
    .drop_duplicates()
) # Construir correspondencia única entre node_id y node_idx

if node_mapping[
    NODE_ID_COLUMN
].duplicated().any():
    raise ValueError(
        "Un mismo node_id está asociado con múltiples node_idx."
    ) # Validar unicidad de node_id respecto al índice nodal

if node_mapping[
    NODE_INDEX_COLUMN
].duplicated().any():
    raise ValueError(
        "Un mismo node_idx está asociado con múltiples node_id."
    ) # Validar unicidad de node_idx respecto a la identidad científica

# 5.13 Validación de Valores Numéricos
profile_numeric_columns = [
    column
    for column in ANALYSIS_FEATURE_COLUMNS + [
        ANALYSIS_TARGET_COLUMN,
        ANALYSIS_RAW_TARGET_COLUMN
    ]
    if column in municipality_profile.columns
] # Identificar variables numéricas del perfil municipal

if not np.isfinite(
    municipality_profile[
        profile_numeric_columns
    ].to_numpy(dtype=np.float64)
).all():
    raise ValueError(
        "El perfil municipal contiene valores numéricos no finitos."
    ) # Validar integridad numérica del perfil

# 5.14 Validación Final de Cobertura del Perfil
if municipality_profile[MUNICIPALITY_ID_COLUMN].nunique() != ANALYSIS_N_MUNICIPALITIES:
    raise ValueError(
        f"El perfil municipal contiene "
        f"{municipality_profile[MUNICIPALITY_ID_COLUMN].nunique():,} municipios únicos; "
        f"se esperaban {ANALYSIS_N_MUNICIPALITIES:,}."
    ) # Validar cantidad final de municipios

if municipality_profile[TIME_KEY].nunique() != ANALYSIS_N_YEARS:
    raise ValueError(
        f"El perfil municipal contiene "
        f"{municipality_profile[TIME_KEY].nunique():,} años; "
        f"se esperaban {ANALYSIS_N_YEARS:,}."
    ) # Validar cantidad final de años

# 5.15 Exportación del Perfil Municipal
municipality_profile.to_parquet(
    PROFILE_OUTPUT_PATH,
    index=False
) # Exportar perfil científico municipio-año

# 5.16 Confirmación del Perfil Municipal
logger.info(
    "Perfil municipal generado: %s registros, %s municipios, %s años y %s variables predictoras.",
    len(municipality_profile),
    municipality_profile[MUNICIPALITY_ID_COLUMN].nunique(),
    municipality_profile[TIME_KEY].nunique(),
    len(ANALYSIS_FEATURE_COLUMNS)
) # Registrar resumen del perfil municipal

# BLOQUE 6. ESTANDARIZACIÓN DE VARIABLES
# Objetivo: Estandarizar exclusivamente las variables predictoras utilizadas para el análisis de similitud municipal.
# Arquitectura científica
# Entradas: municipality_profile y configuración oficial de variables predictoras y estandarización.
# Proceso: Validación, selección y transformación Z Score de las variables predictoras.
# Producto: Matriz estandarizada de variables predictoras con identidad municipal y temporal preservada.

# 6.1 Validación de las Variables Predictoras
if not ANALYSIS_FEATURE_COLUMNS:
    raise ValueError(
        "No existen variables predictoras configuradas para el análisis."
    ) # Validar existencia de variables predictoras

missing_feature_columns = [
    column
    for column in ANALYSIS_FEATURE_COLUMNS
    if column not in municipality_profile.columns
] # Identificar variables predictoras ausentes del perfil municipal

if missing_feature_columns:
    raise ValueError(
        "Las siguientes variables predictoras no existen en "
        f"municipality_profile: {missing_feature_columns}"
    ) # Detener ejecución si faltan variables predictoras

if len(ANALYSIS_FEATURE_COLUMNS) != ANALYSIS_N_FEATURES:
    raise ValueError(
        f"La configuración indica {ANALYSIS_N_FEATURES} variables predictoras, "
        f"pero se encontraron {len(ANALYSIS_FEATURE_COLUMNS)}."
    ) # Validar consistencia entre configuración y variables predictoras

# 6.2 Validación de la Identidad Analítica
required_similarity_identity = [
    NODE_ID_COLUMN,
    NODE_INDEX_COLUMN,
    MUNICIPALITY_ID_COLUMN,
    TIME_KEY
] # Definir identificadores requeridos para trazabilidad

missing_similarity_identity = [
    column
    for column in required_similarity_identity
    if column not in municipality_profile.columns
] # Identificar identificadores ausentes del perfil municipal

if missing_similarity_identity:
    raise ValueError(
        "Faltan columnas de identificación requeridas en "
        f"municipality_profile: {missing_similarity_identity}"
    ) # Detener ejecución si falta información de identidad

if municipality_profile[
    required_similarity_identity
].isna().any().any():
    raise ValueError(
        "El perfil municipal contiene valores faltantes en los "
        "identificadores requeridos para el análisis de similitud."
    ) # Validar integridad de la identidad analítica

if municipality_profile[
    [NODE_ID_COLUMN, TIME_KEY]
].duplicated().any():
    raise ValueError(
        "El perfil municipal contiene identificadores node_id-año duplicados."
    ) # Validar unicidad de la unidad analítica municipio-año

# 6.3 Selección de Variables Predictoras
similarity_feature_matrix = municipality_profile[
    ANALYSIS_FEATURE_COLUMNS
].copy() # Seleccionar exclusivamente las variables predictoras oficiales

# 6.4 Validación de Valores para Estandarización
if similarity_feature_matrix.isna().any().any():
    missing_columns = similarity_feature_matrix.columns[
        similarity_feature_matrix.isna().any()
    ].tolist() # Identificar variables con valores faltantes

    raise ValueError(
        f"Las variables predictoras contienen valores faltantes: "
        f"{missing_columns}"
    ) # Detener ejecución ante valores faltantes

feature_array = similarity_feature_matrix.to_numpy(
    dtype=np.float64
) # Convertir las variables predictoras a representación numérica

if not np.isfinite(feature_array).all():
    raise ValueError(
        "Las variables predictoras contienen valores infinitos o no finitos."
    ) # Validar integridad numérica de las variables predictoras

# 6.5 Validación del Método de Estandarización
if STANDARDIZATION_METHOD != "zscore":
    raise ValueError(
        f"Método de estandarización no soportado: "
        f"{STANDARDIZATION_METHOD}"
    ) # Validar método estadístico configurado

# 6.6 Cálculo de Parámetros de Estandarización
feature_means = similarity_feature_matrix.mean(
    axis=0
) # Calcular la media global del panel para cada variable predictora

feature_stds = similarity_feature_matrix.std(
    axis=0,
    ddof=0
) # Calcular la desviación estándar poblacional global del panel

constant_features = feature_stds[
    feature_stds <= EPSILON
].index.tolist() # Identificar variables sin variabilidad suficiente

if constant_features:
    raise ValueError(
        "Las siguientes variables no presentan variabilidad suficiente "
        f"para la estandarización: {constant_features}"
    ) # Detener ejecución ante variables constantes

# 6.7 Estandarización Z Score
standardized_feature_matrix = (
    similarity_feature_matrix - feature_means
) / feature_stds # Estandarizar las variables predictoras mediante Z Score

# 6.8 Validación de la Estandarización
standardized_array = standardized_feature_matrix.to_numpy(
    dtype=np.float64
) # Convertir la matriz estandarizada a representación numérica

if not np.isfinite(standardized_array).all():
    raise ValueError(
        "La matriz estandarizada contiene valores infinitos o no finitos."
    ) # Validar integridad numérica posterior a la estandarización

standardized_means = standardized_feature_matrix.mean(
    axis=0
) # Calcular las medias de las variables estandarizadas

standardized_stds = standardized_feature_matrix.std(
    axis=0,
    ddof=0
) # Calcular las desviaciones estándar de las variables estandarizadas

if not np.allclose(
    standardized_means.to_numpy(dtype=np.float64),
    0.0,
    atol=STANDARDIZATION_TOLERANCE
):
    raise ValueError(
        "La matriz estandarizada no presenta medias suficientemente "
        "próximas a cero."
    ) # Validar propiedad de media cero del Z Score

if not np.allclose(
    standardized_stds.to_numpy(dtype=np.float64),
    1.0,
    atol=STANDARDIZATION_TOLERANCE
):
    raise ValueError(
        "La matriz estandarizada no presenta desviaciones estándar "
        "suficientemente próximas a uno."
    ) # Validar propiedad de desviación estándar unitaria del Z Score

# 6.9 Conservación de la Identidad Municipio-Año
similarity_metadata = municipality_profile[
    [
        NODE_ID_COLUMN,
        NODE_INDEX_COLUMN,
        MUNICIPALITY_ID_COLUMN,
        TIME_KEY
    ]
].copy() # Conservar identidad científica, índice nodal, municipio y año

if len(similarity_metadata) != len(standardized_feature_matrix):
    raise ValueError(
        "La cantidad de identificadores municipio-año no coincide "
        "con la matriz de variables estandarizadas."
    ) # Validar correspondencia de registros

# 6.10 Construcción de la Matriz Analítica
standardized_features = pd.concat(
    [
        similarity_metadata.reset_index(drop=True),
        standardized_feature_matrix.reset_index(drop=True)
    ],
    axis=1
) # Construir matriz estandarizada conservando correspondencia fila a fila

# 6.11 Validación de la Matriz Analítica
if len(standardized_features) != len(municipality_profile):
    raise ValueError(
        "La matriz estandarizada no conserva todos los registros "
        "del perfil municipal."
    ) # Validar conservación completa de los registros

if standardized_features[
    [
        NODE_ID_COLUMN,
        NODE_INDEX_COLUMN,
        MUNICIPALITY_ID_COLUMN,
        TIME_KEY
    ]
].isna().any().any():
    raise ValueError(
        "La matriz estandarizada contiene identificadores "
        "o años faltantes."
    ) # Validar integridad de la identificación analítica

expected_standardized_columns = (
    4 + ANALYSIS_N_FEATURES
) # Calcular número esperado de columnas

if standardized_features.shape[1] != expected_standardized_columns:
    raise ValueError(
        f"La matriz estandarizada contiene "
        f"{standardized_features.shape[1]} columnas; "
        f"se esperaban {expected_standardized_columns}."
    ) # Validar estructura de la matriz analítica

if standardized_features[
    [NODE_ID_COLUMN, TIME_KEY]
].duplicated().any():
    raise ValueError(
        "La matriz estandarizada contiene identificadores "
        "node_id-año duplicados."
    ) # Validar unicidad de la unidad analítica

# 6.12 Validación de Correspondencia entre Identidad e Índice
node_identity_check = (
    standardized_features[
        [
            NODE_ID_COLUMN,
            NODE_INDEX_COLUMN
        ]
    ]
    .drop_duplicates()
) # Obtener relaciones únicas entre identidad e índice nodal

if node_identity_check[
    NODE_ID_COLUMN
].duplicated().any():
    raise ValueError(
        "Un mismo node_id está asociado con múltiples valores de node_idx."
    ) # Validar correspondencia uno a uno entre identidad e índice

if node_identity_check[
    NODE_INDEX_COLUMN
].duplicated().any():
    raise ValueError(
        "Un mismo node_idx está asociado con múltiples valores de node_id."
    ) # Validar correspondencia uno a uno entre índice e identidad

if len(node_identity_check) != ANALYSIS_N_MUNICIPALITIES:
    raise ValueError(
        "La correspondencia entre node_id y node_idx no contiene "
        f"los {ANALYSIS_N_MUNICIPALITIES:,} nodos esperados."
    ) # Validar cobertura completa de la identidad nodal

# 6.13 Construcción de los Parámetros de Estandarización
standardization_parameters = pd.DataFrame({
    "variable": ANALYSIS_FEATURE_COLUMNS,
    "media": feature_means.to_numpy(dtype=np.float64),
    "desviacion_estandar": feature_stds.to_numpy(dtype=np.float64)
}) # Construir tabla de parámetros utilizados en la estandarización

# 6.14 Validación de Parámetros de Estandarización
if len(standardization_parameters) != ANALYSIS_N_FEATURES:
    raise ValueError(
        "La tabla de parámetros de estandarización no contiene "
        "el número esperado de variables."
    ) # Validar cantidad de parámetros registrados

if not np.isfinite(
    standardization_parameters[
        [
            "media",
            "desviacion_estandar"
        ]
    ].to_numpy(dtype=np.float64)
).all():
    raise ValueError(
        "Los parámetros de estandarización contienen valores no finitos."
    ) # Validar integridad numérica de los parámetros

# 6.15 Exportación de Parámetros de Estandarización
standardization_parameters.to_parquet(
    ANALYSIS_DIR / "standardization_parameters.parquet",
    index=False
) # Exportar parámetros para garantizar trazabilidad y reproducibilidad

# 6.16 Registro de la Estandarización
logger.info(
    "Estandarización completada: %s registros, %s variables predictoras y método %s.",
    len(standardized_features),
    ANALYSIS_N_FEATURES,
    STANDARDIZATION_METHOD
) # Registrar resultado de la estandarización

# BLOQUE 7. COMPARACIÓN ENTRE MUNICIPIOS
# Objetivo: Construir comparaciones por pares entre municipios dentro de cada año del panel científico.
# Arquitectura científica
# Entradas: municipality_profile, standardized_features y configuración oficial de comparación.
# Proceso: Generación de pares municipales, cálculo de diferencias absolutas y estandarizadas y comparación de log_rendimiento.
# Producto: municipality_comparison.parquet con las diferencias entre municipios para cada año.

# 7.1 Inicialización de las Comparaciones Municipales
municipality_comparisons = [] # Lista para almacenar las comparaciones municipales por año

# 7.2 Procesamiento de Cada Año
for current_year in ANALYSIS_YEARS:
    year_profile = municipality_profile[
        municipality_profile[TIME_KEY] == current_year
    ].copy() # Seleccionar los municipios correspondientes al año actual

    year_standardized = standardized_features[
        standardized_features[TIME_KEY] == current_year
    ].copy() # Seleccionar las variables estandarizadas correspondientes al año actual

    n_year_municipalities = len(year_profile) # Obtener número de municipios disponibles en el año

    if n_year_municipalities != ANALYSIS_N_MUNICIPALITIES:
        raise ValueError(
            f"El año {current_year} contiene "
            f"{n_year_municipalities:,} municipios; "
            f"se esperaban {ANALYSIS_N_MUNICIPALITIES:,}."
        ) # Validar cobertura municipal anual

    if len(year_standardized) != ANALYSIS_N_MUNICIPALITIES:
        raise ValueError(
            f"La matriz estandarizada del año {current_year} contiene "
            f"{len(year_standardized):,} municipios; "
            f"se esperaban {ANALYSIS_N_MUNICIPALITIES:,}."
        ) # Validar cobertura estandarizada anual

    # 7.3 Ordenamiento por Índice Nodal Oficial
    year_profile = year_profile.sort_values(
        by=NODE_INDEX_COLUMN
    ).reset_index(drop=True) # Ordenar el perfil anual según node_idx

    year_standardized = year_standardized.sort_values(
        by=NODE_INDEX_COLUMN
    ).reset_index(drop=True) # Ordenar la matriz estandarizada según node_idx

    # 7.4 Validación de Correspondencia entre Perfil y Matriz Estandarizada
    if not year_profile[
        NODE_INDEX_COLUMN
    ].equals(
        year_standardized[NODE_INDEX_COLUMN]
    ):
        raise ValueError(
            f"Los índices node_idx no coinciden entre "
            f"municipality_profile y standardized_features "
            f"para el año {current_year}."
        ) # Validar correspondencia del espacio nodal

    if not year_profile[
        NODE_ID_COLUMN
    ].equals(
        year_standardized[NODE_ID_COLUMN]
    ):
        raise ValueError(
            f"Los identificadores node_id no coinciden entre "
            f"municipality_profile y standardized_features "
            f"para el año {current_year}."
        ) # Validar correspondencia de identidad científica

    if not year_profile[
        TIME_KEY
    ].equals(
        year_standardized[TIME_KEY]
    ):
        raise ValueError(
            f"Los años no coinciden entre municipality_profile "
            f"y standardized_features para el año {current_year}."
        ) # Validar correspondencia temporal

    # 7.5 Generación de Pares Municipales
    municipality_indices_a, municipality_indices_b = np.triu_indices(
        n_year_municipalities,
        k=1
    ) # Generar pares únicos sin autocombinaciones

    pair_data = pd.DataFrame({
        "node_idx_a": year_profile.loc[
            municipality_indices_a,
            NODE_INDEX_COLUMN
        ].to_numpy(),

        "node_idx_b": year_profile.loc[
            municipality_indices_b,
            NODE_INDEX_COLUMN
        ].to_numpy(),

        "node_id_a": year_profile.loc[
            municipality_indices_a,
            NODE_ID_COLUMN
        ].to_numpy(),

        "node_id_b": year_profile.loc[
            municipality_indices_b,
            NODE_ID_COLUMN
        ].to_numpy(),

        "municipality_id_a": year_profile.loc[
            municipality_indices_a,
            MUNICIPALITY_ID_COLUMN
        ].to_numpy(),

        "municipality_id_b": year_profile.loc[
            municipality_indices_b,
            MUNICIPALITY_ID_COLUMN
        ].to_numpy(),

        TIME_KEY: current_year
    }) # Construir tabla de pares con trazabilidad territorial completa

   # 7.6 Cálculo de Diferencias Absolutas
    if "absolute" in DIFFERENCE_METHODS:
        features_a = year_profile.loc[
            municipality_indices_a,
            ANALYSIS_FEATURE_COLUMNS
        ].to_numpy(
            dtype=np.float64
        ) # Extraer características originales del municipio A

        features_b = year_profile.loc[
            municipality_indices_b,
            ANALYSIS_FEATURE_COLUMNS
        ].to_numpy(
            dtype=np.float64
        ) # Extraer características originales del municipio B

        absolute_differences = np.abs(
            features_a - features_b
        ) # Calcular diferencias absolutas entre municipios

        pair_data["difference_absolute_mean"] = (
            absolute_differences.mean(axis=1)
        ) # Registrar diferencia absoluta media entre las variables predictoras

    # 7.7 Cálculo de Diferencias Estandarizadas
    if "standardized" in DIFFERENCE_METHODS:

        standardized_a = year_standardized.loc[
            municipality_indices_a,
            ANALYSIS_FEATURE_COLUMNS
        ].to_numpy(
            dtype=np.float64
        ) # Extraer características estandarizadas del municipio A

        standardized_b = year_standardized.loc[
            municipality_indices_b,
            ANALYSIS_FEATURE_COLUMNS
        ].to_numpy(
            dtype=np.float64
        ) # Extraer características estandarizadas del municipio B

        standardized_differences = np.abs(
            standardized_a - standardized_b
        ) # Calcular diferencias absolutas en escala estandarizada

        pair_data["difference_standardized_mean"] = (
            standardized_differences.mean(axis=1)
        ) # Registrar diferencia estandarizada media entre las variables predictoras

    # 7.8 Comparación de la Variable Objetivo
    if COMPARISON_INCLUDE_TARGET:
        target_a = year_profile.loc[
            municipality_indices_a,
            ANALYSIS_TARGET_COLUMN
        ].to_numpy(
            dtype=np.float64
        ) # Extraer log_rendimiento del municipio A

        target_b = year_profile.loc[
            municipality_indices_b,
            ANALYSIS_TARGET_COLUMN
        ].to_numpy(
            dtype=np.float64
        ) # Extraer log_rendimiento del municipio B

        pair_data["difference_target"] = np.abs(
            target_a - target_b
        ) # Calcular diferencia absoluta de log_rendimiento

    # 7.9 Validación de Objetivo en Escala Original
    if COMPARISON_INCLUDE_RAW_TARGET:
        raise ValueError(
            "COMPARISON_INCLUDE_RAW_TARGET está activado, "
            "pero la variable objetivo original no está disponible "
            "en el GraphData oficial. No se permite reconstruirla "
            "a partir de graph_data.y."
        ) # Evitar sustituir rendimiento_promedio por log_rendimiento

    # 7.10 Validación de las Comparaciones del Año
    if (
        pair_data["node_idx_a"] >= pair_data["node_idx_b"]
    ).any():
        raise ValueError(
            f"Los pares municipales del año {current_year} "
            "no respetan el orden nodal estrictamente creciente."
        ) # Validar orden único de los pares

    if (
        pair_data["node_idx_a"] == pair_data["node_idx_b"]
    ).any():
        raise ValueError(
            f"Se detectaron autocombinaciones en el año {current_year}."
        ) # Validar ausencia de comparaciones de un municipio consigo mismo

    if pair_data[
        ["node_idx_a", "node_idx_b"]
    ].duplicated().any():
        raise ValueError(
            f"Se detectaron pares nodales duplicados "
            f"en el año {current_year}."
        ) # Validar unicidad estructural de los pares

    expected_year_pairs = (
        ANALYSIS_N_MUNICIPALITIES
        * (ANALYSIS_N_MUNICIPALITIES - 1)
        // 2
    ) # Calcular número esperado de pares del año

    if len(pair_data) != expected_year_pairs:
        raise ValueError(
            f"El año {current_year} contiene "
            f"{len(pair_data):,} pares; "
            f"se esperaban {expected_year_pairs:,}."
        ) # Validar cantidad de pares del año

    municipality_comparisons.append(
        pair_data
    ) # Agregar comparaciones del año actual

# 7.11 Consolidación de las Comparaciones
if not municipality_comparisons:
    raise ValueError(
        "No se generaron comparaciones municipales."
    ) # Validar generación de resultados

municipality_comparison = pd.concat(
    municipality_comparisons,
    ignore_index=True
) # Consolidar las comparaciones de todos los años

# 7.12 Ordenamiento de las Comparaciones
municipality_comparison = municipality_comparison.sort_values(
    by=[
        TIME_KEY,
        "node_idx_a",
        "node_idx_b"
    ]
).reset_index(drop=True) # Ordenar las comparaciones por año y espacio nodal

# 7.13 Validación de la Cantidad de Pares
expected_pairs_per_year = (
    ANALYSIS_N_MUNICIPALITIES
    * (ANALYSIS_N_MUNICIPALITIES - 1)
    // 2
) # Calcular número esperado de pares únicos por año

expected_total_pairs = (
    expected_pairs_per_year
    * ANALYSIS_N_YEARS
) # Calcular número esperado de pares para todo el panel

if len(municipality_comparison) != expected_total_pairs:
    raise ValueError(
        f"La tabla de comparación contiene "
        f"{len(municipality_comparison):,} pares; "
        f"se esperaban {expected_total_pairs:,}."
    ) # Validar cantidad total de pares

# 7.14 Validación de Cobertura Temporal
if municipality_comparison[TIME_KEY].nunique() != ANALYSIS_N_YEARS:
    raise ValueError(
        f"La tabla de comparación contiene "
        f"{municipality_comparison[TIME_KEY].nunique():,} años; "
        f"se esperaban {ANALYSIS_N_YEARS:,}."
    ) # Validar cobertura temporal completa

# 7.15 Validación de Identidad de los Pares
pair_identity_columns = [
    TIME_KEY,
    "node_idx_a",
    "node_idx_b"
] # Definir clave única de cada comparación

if municipality_comparison[
    pair_identity_columns
].duplicated().any():
    raise ValueError(
        "La tabla de comparación contiene pares "
        "nodales duplicados dentro del panel."
    ) # Validar unicidad de cada comparación

# 7.16 Validación de Identidades Territoriales
if municipality_comparison[
    [
        "node_idx_a",
        "node_idx_b",
        "node_id_a",
        "node_id_b",
        "municipality_id_a",
        "municipality_id_b"
    ]
].isna().any().any():
    raise ValueError(
        "La tabla de comparación contiene identificadores "
        "territoriales faltantes."
    ) # Validar integridad de la identidad territorial

# 7.17 Validación de Valores Numéricos
comparison_numeric_columns = [
    column
    for column in [
        "difference_absolute_mean",
        "difference_standardized_mean",
        "difference_target",
        "difference_raw_target"
    ]
    if column in municipality_comparison.columns
] # Identificar métricas numéricas disponibles

if comparison_numeric_columns:

    comparison_numeric_values = municipality_comparison[
        comparison_numeric_columns
    ].to_numpy(
        dtype=np.float64
    ) # Convertir métricas de comparación a matriz numérica

    if not np.isfinite(
        comparison_numeric_values
    ).all():
        raise ValueError(
            "La tabla de comparación contiene valores "
            "numéricos no finitos."
        ) # Validar integridad numérica

    if (
        comparison_numeric_values < 0
    ).any():
        raise ValueError(
            "La tabla de comparación contiene diferencias negativas."
        ) # Validar propiedad no negativa de las diferencias absolutas

# 7.18 Exportación de las Comparaciones
municipality_comparison.to_parquet(
    COMPARISON_OUTPUT_PATH,
    index=False
) # Exportar tabla de comparación municipal

# 7.19 Registro del Resultado
logger.info(
    "Comparación municipal generada: %s pares municipio-año | %s pares por año.",
    len(municipality_comparison),
    expected_pairs_per_year
) # Registrar número de comparaciones generadas

# BLOQUE 8. CÁLCULO DE SIMILITUD POR DOMINIO
# Objetivo: Calcular la similitud coseno entre municipios para cada dominio científico,
# conservando la identidad nodal mediante node_idx y node_id.
# Entradas: Variables predictoras estandarizadas, pares municipales y agrupaciones científicas oficiales.
# Proceso: Cálculo de similitud coseno independiente para cada dominio científico y cada par municipal.
# Producto: Similitudes climática, agrícola, de riego, ambiental, cobertura y tenencia para cada par municipio-año.

# 8.1 Inicialización de la Similitud por Dominio
domain_similarity_results = [] # Lista para almacenar las similitudes por dominio y año
domain_zero_norm_counts = {} # Diccionario para registrar casos con norma cero por dominio

# 8.2 Validación de los Dominios Científicos
if not ANALYSIS_FEATURE_GROUPS:
    raise ValueError(
        "No existen agrupaciones científicas configuradas para "
        "el cálculo de similitud por dominio."
    ) # Validar existencia de agrupaciones científicas

for domain_name, domain_columns in ANALYSIS_FEATURE_GROUPS.items():
    if not domain_columns:
        raise ValueError(
            f"El dominio '{domain_name}' no contiene variables configuradas."
        ) # Validar que cada dominio contenga variables

    if len(domain_columns) != len(set(domain_columns)):
        raise ValueError(
            f"El dominio '{domain_name}' contiene variables "
            "predictoras duplicadas."
        ) # Validar unicidad de variables dentro del dominio

    missing_domain_columns = [
        column
        for column in domain_columns
        if column not in ANALYSIS_FEATURE_COLUMNS
    ] # Identificar variables que no pertenecen al conjunto oficial

    if missing_domain_columns:
        raise ValueError(
            f"El dominio '{domain_name}' contiene variables no incluidas "
            f"en ANALYSIS_FEATURE_COLUMNS: {missing_domain_columns}"
        ) # Validar pertenencia al conjunto oficial de predictores

# 8.3 Validación de Cobertura de los Dominios
domain_columns_flat = [
    column
    for domain_columns in ANALYSIS_FEATURE_GROUPS.values()
    for column in domain_columns
] # Construir listado consolidado de variables utilizadas por los dominios

if len(domain_columns_flat) != len(set(domain_columns_flat)):
    raise ValueError(
        "Una o más variables predictoras aparecen en más de un "
        "dominio científico."
    ) # Validar que cada variable pertenezca a un único dominio

missing_grouped_features = [
    column
    for column in ANALYSIS_FEATURE_COLUMNS
    if column not in domain_columns_flat
] # Identificar predictores que no pertenecen a ningún dominio

if missing_grouped_features:
    raise ValueError(
        f"Existen variables predictoras oficiales que no pertenecen "
        f"a ningún dominio científico: {missing_grouped_features}"
    ) # Validar cobertura completa de los predictores

if len(domain_columns_flat) != ANALYSIS_N_FEATURES:
    raise ValueError(
        f"Los dominios contienen {len(domain_columns_flat)} variables; "
        f"se esperaban {ANALYSIS_N_FEATURES}."
    ) # Validar consistencia entre dominios y número oficial de predictores

# 8.4 Procesamiento de Cada Año
for current_year in ANALYSIS_YEARS:
    year_standardized = standardized_features[
        standardized_features[TIME_KEY] == current_year
    ].sort_values(
        by=NODE_INDEX_COLUMN
    ).reset_index(drop=True) # Seleccionar y ordenar perfiles según node_idx

    year_comparisons = municipality_comparison[
        municipality_comparison[TIME_KEY] == current_year
    ].copy() # Seleccionar pares municipales del año actual

    if len(year_standardized) != ANALYSIS_N_MUNICIPALITIES:
        raise ValueError(
            f"El año {current_year} contiene "
            f"{len(year_standardized):,} perfiles estandarizados; "
            f"se esperaban {ANALYSIS_N_MUNICIPALITIES:,}."
        ) # Validar cobertura municipal estandarizada

    expected_pairs_per_year = (
        ANALYSIS_N_MUNICIPALITIES
        * (ANALYSIS_N_MUNICIPALITIES - 1)
        // 2
    ) # Calcular número esperado de pares para el año

    if len(year_comparisons) != expected_pairs_per_year:
        raise ValueError(
            f"El año {current_year} contiene "
            f"{len(year_comparisons):,} pares; "
            f"se esperaban {expected_pairs_per_year:,}."
        ) # Validar cobertura completa de pares

    # 8.5 Validación de la Identidad Nodal
    required_identity_columns = [
        NODE_INDEX_COLUMN,
        NODE_ID_COLUMN,
        MUNICIPALITY_ID_COLUMN,
        TIME_KEY
    ] # Definir identidad mínima requerida

    missing_identity_columns = [
        column
        for column in required_identity_columns
        if column not in year_standardized.columns
    ] # Identificar columnas de identidad ausentes

    if missing_identity_columns:
        raise ValueError(
            f"Faltan columnas de identidad en los perfiles estandarizados "
            f"del año {current_year}: {missing_identity_columns}"
        ) # Validar disponibilidad de identidad nodal

    if year_standardized[
        required_identity_columns
    ].isna().any().any():
        raise ValueError(
            f"Los perfiles estandarizados del año {current_year} "
            "contienen identificadores faltantes."
        ) # Validar integridad de identidad

    if year_standardized[
        NODE_INDEX_COLUMN
    ].duplicated().any():
        raise ValueError(
            f"Los perfiles estandarizados del año {current_year} "
            "contienen node_idx duplicados."
        ) # Validar unicidad del índice nodal

    if year_standardized[
        NODE_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            f"Los perfiles estandarizados del año {current_year} "
            "contienen node_id duplicados."
        ) # Validar unicidad de la identidad científica

    # 8.6 Validación del Espacio Nodal de los Pares
    expected_node_idx_a = np.arange(
        ANALYSIS_N_MUNICIPALITIES - 1,
        dtype=np.int64
    ) # Definir espacio esperado del extremo A

    expected_node_idx_b = np.arange(
        1,
        ANALYSIS_N_MUNICIPALITIES,
        dtype=np.int64
    ) # Definir espacio esperado del extremo B

    observed_node_idx_a = np.sort(
        year_comparisons["node_idx_a"].unique()
    ) # Obtener índices nodales únicos observados en el extremo A

    observed_node_idx_b = np.sort(
        year_comparisons["node_idx_b"].unique()
    ) # Obtener índices nodales únicos observados en el extremo B

    if not np.array_equal(
        observed_node_idx_a,
        expected_node_idx_a
    ):
        raise ValueError(
            f"Los node_idx_a del año {current_year} "
            "no cubren correctamente el espacio nodal esperado "
            f"del extremo A: 0 ... {ANALYSIS_N_MUNICIPALITIES - 2}."
        ) # Validar espacio nodal del extremo A

    if not np.array_equal(
        observed_node_idx_b,
        expected_node_idx_b
    ):
        raise ValueError(
            f"Los node_idx_b del año {current_year} "
            "no cubren correctamente el espacio nodal esperado "
            f"del extremo B: 1 ... {ANALYSIS_N_MUNICIPALITIES - 1}."
        ) # Validar espacio nodal del extremo B

    # 8.7 Validación de Correspondencia node_idx y node_id
    expected_node_ids_a = year_standardized.loc[
        year_comparisons["node_idx_a"].to_numpy(dtype=np.int64),
        NODE_ID_COLUMN
    ].to_numpy() # Recuperar node_id correspondiente a node_idx_a

    expected_node_ids_b = year_standardized.loc[
        year_comparisons["node_idx_b"].to_numpy(dtype=np.int64),
        NODE_ID_COLUMN
    ].to_numpy() # Recuperar node_id correspondiente a node_idx_b

    if not np.array_equal(
        expected_node_ids_a,
        year_comparisons["node_id_a"].to_numpy()
    ):
        raise ValueError(
            f"La correspondencia entre node_idx_a y node_id_a "
            f"no coincide para el año {current_year}."
        ) # Validar identidad del municipio A

    if not np.array_equal(
        expected_node_ids_b,
        year_comparisons["node_id_b"].to_numpy()
    ):
        raise ValueError(
            f"La correspondencia entre node_idx_b y node_id_b "
            f"no coincide para el año {current_year}."
        ) # Validar identidad del municipio B

    # 8.8 Construcción de la Estructura Base de Resultados
    domain_results = pd.DataFrame({
        TIME_KEY: current_year,
        "node_idx_a": year_comparisons[
            "node_idx_a"
        ].to_numpy(),
        "node_idx_b": year_comparisons[
            "node_idx_b"
        ].to_numpy(),
        "node_id_a": year_comparisons[
            "node_id_a"
        ].to_numpy(),
        "node_id_b": year_comparisons[
            "node_id_b"
        ].to_numpy(),
        "municipality_id_a": year_comparisons[
            "municipality_id_a"
        ].to_numpy(),
        "municipality_id_b": year_comparisons[
            "municipality_id_b"
        ].to_numpy()
    }) # Construir estructura base con trazabilidad nodal y territorial

    # 8.9 Cálculo de Similitud por Dominio
    for domain_name, domain_columns in ANALYSIS_FEATURE_GROUPS.items():
        missing_columns = [
            column
            for column in domain_columns
            if column not in year_standardized.columns
        ] # Identificar variables faltantes del dominio

        if missing_columns:
            raise ValueError(
                f"Faltan variables del dominio '{domain_name}' "
                f"en el año {current_year}: {missing_columns}"
            ) # Validar disponibilidad completa del dominio

        domain_a = year_standardized.loc[
            year_comparisons["node_idx_a"].to_numpy(dtype=np.int64),
            domain_columns
        ].to_numpy(
            dtype=np.float64
        ) # Extraer características estandarizadas del municipio A

        domain_b = year_standardized.loc[
            year_comparisons["node_idx_b"].to_numpy(dtype=np.int64),
            domain_columns
        ].to_numpy(
            dtype=np.float64
        ) # Extraer características estandarizadas del municipio B

        if not np.isfinite(domain_a).all():
            raise ValueError(
                f"El dominio '{domain_name}' contiene valores no finitos "
                f"para el municipio A en el año {current_year}."
            ) # Validar valores numéricos del dominio A

        if not np.isfinite(domain_b).all():
            raise ValueError(
                f"El dominio '{domain_name}' contiene valores no finitos "
                f"para el municipio B en el año {current_year}."
            ) # Validar valores numéricos del dominio B

        dot_product = np.sum(
            domain_a * domain_b,
            axis=1
        ) # Calcular producto punto entre los municipios

        norm_a = np.linalg.norm(
            domain_a,
            axis=1
        ) # Calcular norma del municipio A

        norm_b = np.linalg.norm(
            domain_b,
            axis=1
        ) # Calcular norma del municipio B

        denominator = (
            norm_a * norm_b
        ) # Calcular denominador de la similitud coseno

        zero_norm_mask = (
            denominator <= EPSILON
        ) # Identificar pares con norma cero o prácticamente cero

        zero_norm_count = int(
            zero_norm_mask.sum()
        ) # Contabilizar pares con similitud coseno indefinida

        domain_zero_norm_counts[
            f"{current_year}_{domain_name}"
        ] = zero_norm_count # Registrar cantidad de casos indefinidos

        if zero_norm_count > 0:
            raise ValueError(
                f"El dominio '{domain_name}' del año {current_year} "
                f"contiene {zero_norm_count:,} pares con norma cero "
                "o prácticamente cero; la similitud coseno está indefinida."
            ) # Detener ejecución ante vectores de norma cero

        domain_similarity = np.divide(
            dot_product,
            denominator
        ) # Calcular similitud coseno

        similarity_column = (
            f"similaridad_{domain_name}"
        ) # Definir nombre de la similitud del dominio

        domain_results[
            similarity_column
        ] = domain_similarity # Registrar similitud del dominio

    # 8.10 Validación de los Resultados del Año
    domain_similarity_columns_year = [
        f"similaridad_{domain_name}"
        for domain_name in ANALYSIS_FEATURE_GROUPS
    ] # Identificar columnas de similitud generadas para el año

    similarity_values = domain_results[
        domain_similarity_columns_year
    ].to_numpy(
        dtype=np.float64
    ) # Obtener matriz de similitudes del año

    if not np.isfinite(
        similarity_values
    ).all():
        raise ValueError(
            f"Los resultados de similitud del año {current_year} "
            "contienen valores no finitos."
        ) # Validar integridad numérica de las similitudes

    if (
        similarity_values.min() < -1.0000001
        or similarity_values.max() > 1.0000001
    ):
        raise ValueError(
            f"Los resultados de similitud del año {current_year} "
            "contienen valores fuera del rango matemático "
            "de la similitud coseno."
        ) # Validar rango matemático de la similitud coseno

    domain_similarity_results.append(
        domain_results
    ) # Agregar resultados del año actual

# 8.11 Validación de la Colección de Resultados
if len(domain_similarity_results) != ANALYSIS_N_YEARS:
    raise ValueError(
        f"Se generaron {len(domain_similarity_results):,} resultados anuales "
        f"de similitud por dominio; "
        f"se esperaban {ANALYSIS_N_YEARS:,}."
    ) # Validar cobertura completa del periodo analítico

# 8.12 Consolidación de Similitudes por Dominio
municipality_domain_similarity = pd.concat(
    domain_similarity_results,
    ignore_index=True
) # Consolidar similitudes de todos los años

# 8.13 Validación de las Columnas de Similitud
domain_similarity_columns = [
    f"similaridad_{domain_name}"
    for domain_name in ANALYSIS_FEATURE_GROUPS
] # Identificar columnas oficiales de similitud

missing_similarity_columns = [
    column
    for column in domain_similarity_columns
    if column not in municipality_domain_similarity.columns
] # Identificar columnas de similitud ausentes

if missing_similarity_columns:
    raise ValueError(
        f"Faltan columnas de similitud por dominio: "
        f"{missing_similarity_columns}"
    ) # Validar estructura completa de resultados

# 8.14 Validación de las Similitudes
for similarity_column in domain_similarity_columns:
    similarity_values = municipality_domain_similarity[
        similarity_column
    ].to_numpy(
        dtype=np.float64
    ) # Extraer valores de similitud

    if not np.isfinite(
        similarity_values
    ).all():
        raise ValueError(
            f"La columna '{similarity_column}' contiene "
            "valores no finitos."
        ) # Validar valores numéricos

    if (
        similarity_values.min() < -1.0000001
        or similarity_values.max() > 1.0000001
    ):
        raise ValueError(
            f"La columna '{similarity_column}' contiene valores "
            "fuera del rango esperado de similitud coseno."
        ) # Validar rango matemático de similitud

# 8.15 Validación de Cantidad de Resultados
expected_total_pairs = (
    ANALYSIS_N_MUNICIPALITIES
    * (ANALYSIS_N_MUNICIPALITIES - 1)
    // 2
    * ANALYSIS_N_YEARS
) # Calcular cantidad esperada de resultados municipio-año

if len(municipality_domain_similarity) != expected_total_pairs:
    raise ValueError(
        f"La tabla de similitud por dominio contiene "
        f"{len(municipality_domain_similarity):,} registros; "
        f"se esperaban {expected_total_pairs:,}."
    ) # Validar cantidad total de pares

# 8.16 Validación de Cobertura Temporal
if municipality_domain_similarity[
    TIME_KEY
].nunique() != ANALYSIS_N_YEARS:
    raise ValueError(
        f"La tabla de similitud por dominio contiene "
        f"{municipality_domain_similarity[TIME_KEY].nunique():,} años; "
        f"se esperaban {ANALYSIS_N_YEARS:,}."
    ) # Validar cobertura temporal completa

# 8.17 Validación de Identidad de los Pares
required_pair_identity = [
    TIME_KEY,
    "node_idx_a",
    "node_idx_b",
    "node_id_a",
    "node_id_b",
    "municipality_id_a",
    "municipality_id_b"
] # Definir identidad completa de cada par

if municipality_domain_similarity[
    required_pair_identity
].isna().any().any():
    raise ValueError(
        "La tabla de similitud por dominio contiene "
        "identificadores nodales o territoriales faltantes."
    ) # Validar integridad de la identidad de los pares

# 8.18 Validación de Duplicados
duplicate_domain_results = municipality_domain_similarity.duplicated(
    subset=[
        TIME_KEY,
        "node_idx_a",
        "node_idx_b"
    ]
).sum() # Contar pares nodales duplicados

if duplicate_domain_results > 0:
    raise ValueError(
        f"La tabla de similitud por dominio contiene "
        f"{duplicate_domain_results:,} pares nodales duplicados."
    ) # Detener ejecución ante duplicados

# 8.19 Ordenamiento de los Resultados
municipality_domain_similarity = (
    municipality_domain_similarity
    .sort_values(
        by=[
            TIME_KEY,
            "node_idx_a",
            "node_idx_b"
        ]
    )
    .reset_index(drop=True)
) # Ordenar similitudes por año y espacio nodal

# 8.20 Registro de Casos con Norma Cero
zero_norm_summary = pd.DataFrame([
    {
        TIME_KEY: int(key.split("_", 1)[0]),
        "domain": key.split("_", 1)[1],
        "zero_norm_pairs": count
    }
    for key, count in domain_zero_norm_counts.items()
]) # Construir resumen de casos con norma cero

if not zero_norm_summary.empty:
    if (
        zero_norm_summary["zero_norm_pairs"] > 0
    ).any():
        raise ValueError(
            "Se detectaron casos con norma cero "
            "en el cálculo de similitud."
        ) # Confirmar ausencia de casos indefinidos

# 8.21 Exportación de la Similitud por Dominio
municipality_domain_similarity.to_parquet(
    SIMILARITY_OUTPUT_PATH,
    index=False
) # Exportar similitudes municipales por dominio

# 8.22 Registro de Resultados
logger.info(
    "Similitud por dominio calculada para %s pares municipio-año "
    "y %s dominios científicos.",
    len(municipality_domain_similarity),
    len(ANALYSIS_FEATURE_GROUPS)
) # Registrar resultado del análisis por dominio

# BLOQUE 9. CÁLCULO DE SIMILITUD GLOBAL
# Objetivo: Calcular la similitud global entre municipios utilizando el conjunto completo de variables predictoras.
# Arquitectura científica
# Entradas: Variables predictoras estandarizadas, pares municipales y configuración oficial de similitud.
# Proceso: Cálculo de similitud coseno sobre el conjunto completo de variables predictoras y distancia euclídea complementaria.
# Producto: Similitud global y distancia euclídea para cada par municipio-año.

# 9.1 Validación de la Métrica Principal
if PRIMARY_SIMILARITY_METRIC != "cosine":
    raise ValueError(
        f"Métrica de similitud no soportada: {PRIMARY_SIMILARITY_METRIC}"
    ) # Validar métrica principal de similitud

# 9.2 Validación de la Métrica de Distancia
if DISTANCE_METRIC != "euclidean":
    raise ValueError(
        f"Métrica de distancia no soportada: {DISTANCE_METRIC}"
    ) # Validar métrica complementaria de distancia

# 9.3 Validación de las Variables Predictoras
if len(ANALYSIS_FEATURE_COLUMNS) != ANALYSIS_N_FEATURES:
    raise ValueError(
        f"La configuración contiene {len(ANALYSIS_FEATURE_COLUMNS)} "
        f"variables predictoras; se esperaban {ANALYSIS_N_FEATURES}."
    ) # Validar consistencia de la configuración de predictores

missing_global_features = [
    column
    for column in ANALYSIS_FEATURE_COLUMNS
    if column not in standardized_features.columns
] # Identificar variables predictoras ausentes

if missing_global_features:
    raise ValueError(
        f"Faltan variables predictoras en standardized_features: "
        f"{missing_global_features}"
    ) # Detener ejecución si faltan predictores

# 9.4 Validación de la Identidad Analítica
required_identity_columns = [
    NODE_ID_COLUMN,
    NODE_INDEX_COLUMN,
    TIME_KEY
] # Definir columnas necesarias para garantizar trazabilidad

missing_identity_columns = [
    column
    for column in required_identity_columns
    if column not in standardized_features.columns
] # Identificar columnas de identidad ausentes

if missing_identity_columns:
    raise ValueError(
        f"Faltan columnas de identidad en standardized_features: "
        f"{missing_identity_columns}"
    ) # Validar disponibilidad de identidad nodal

# 9.5 Inicialización de los Resultados
global_similarity_results = [] # Lista para almacenar la similitud global de cada año
global_zero_norm_counts = {} # Diccionario para registrar casos con norma cero por año

# 9.6 Procesamiento de Cada Año
for current_year in ANALYSIS_YEARS:
    year_standardized = (
        standardized_features[
            standardized_features[TIME_KEY] == current_year
        ]
        .sort_values(NODE_INDEX_COLUMN)
        .reset_index(drop=True)
    ) # Seleccionar y ordenar perfiles según node_idx

    year_comparisons = municipality_comparison[
        municipality_comparison[TIME_KEY] == current_year
    ].copy() # Seleccionar pares municipales del año actual

    if len(year_standardized) != ANALYSIS_N_MUNICIPALITIES:
        raise ValueError(
            f"El año {current_year} contiene "
            f"{len(year_standardized):,} perfiles estandarizados; "
            f"se esperaban {ANALYSIS_N_MUNICIPALITIES:,}."
        ) # Validar cobertura municipal del año

    expected_pairs_per_year = (
        ANALYSIS_N_MUNICIPALITIES
        * (ANALYSIS_N_MUNICIPALITIES - 1)
        // 2
    ) # Calcular número esperado de pares para el año

    if len(year_comparisons) != expected_pairs_per_year:
        raise ValueError(
            f"El año {current_year} contiene "
            f"{len(year_comparisons):,} pares; "
            f"se esperaban {expected_pairs_per_year:,}."
        ) # Validar cobertura completa de pares

    # 9.7 Validación de Identidad del Perfil Anual
    required_year_identity = [
        NODE_INDEX_COLUMN,
        NODE_ID_COLUMN,
        TIME_KEY
    ] # Definir identidad requerida para el perfil anual

    if year_standardized[
        required_year_identity
    ].isna().any().any():
        raise ValueError(
            f"Los perfiles estandarizados del año {current_year} "
            "contienen identificadores faltantes."
        ) # Validar integridad de la identidad nodal

    if year_standardized[
        NODE_INDEX_COLUMN
    ].duplicated().any():
        raise ValueError(
            f"Los perfiles estandarizados del año {current_year} "
            "contienen node_idx duplicados."
        ) # Validar unicidad del índice nodal

    if year_standardized[
        NODE_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            f"Los perfiles estandarizados del año {current_year} "
            "contienen node_id duplicados."
        ) # Validar unicidad de la identidad científica

    # 9.8 Validación de la Estructura de los Pares Nodales
    required_pair_columns = [
        "node_idx_a",
        "node_idx_b",
        "node_id_a",
        "node_id_b"
    ] # Definir estructura nodal requerida por los pares

    missing_pair_columns = [
        column
        for column in required_pair_columns
        if column not in year_comparisons.columns
    ] # Identificar columnas nodales ausentes

    if missing_pair_columns:
        raise ValueError(
            f"Faltan columnas nodales en municipality_comparison "
            f"para el año {current_year}: {missing_pair_columns}"
        ) # Validar estructura nodal de los pares

    if year_comparisons[
        ["node_idx_a", "node_idx_b"]
    ].isna().any().any():
        raise ValueError(
            f"Los pares municipales del año {current_year} "
            "contienen índices nodales faltantes."
        ) # Validar integridad de los índices nodales

    node_idx_a = year_comparisons[
        "node_idx_a"
    ].to_numpy(dtype=np.int64) # Extraer índice nodal del municipio A

    node_idx_b = year_comparisons[
        "node_idx_b"
    ].to_numpy(dtype=np.int64) # Extraer índice nodal del municipio B

    if (
        (node_idx_a < 0).any()
        or (node_idx_b < 0).any()
        or (node_idx_a >= ANALYSIS_N_MUNICIPALITIES).any()
        or (node_idx_b >= ANALYSIS_N_MUNICIPALITIES).any()
    ):
        raise ValueError(
            f"Los pares municipales del año {current_year} "
            "contienen node_idx fuera del espacio nodal oficial "
            f"0 ... {ANALYSIS_N_MUNICIPALITIES - 1}."
        ) # Validar límites del espacio nodal

    if (node_idx_a >= node_idx_b).any():
        raise ValueError(
            f"Los pares municipales del año {current_year} "
            "no cumplen la condición node_idx_a < node_idx_b."
        ) # Validar representación única y ausencia de autocombinaciones

    if year_comparisons[
        ["node_idx_a", "node_idx_b"]
    ].duplicated().any():
        raise ValueError(
            f"Los pares node_idx_a-node_idx_b del año {current_year} "
            "contienen duplicados."
        ) # Validar unicidad de los pares nodales

    # 9.9 Validación de Correspondencia entre node_idx y node_id
    expected_node_ids_a = year_standardized.loc[
        node_idx_a,
        NODE_ID_COLUMN
    ].to_numpy() # Recuperar node_id correspondiente a node_idx_a

    expected_node_ids_b = year_standardized.loc[
        node_idx_b,
        NODE_ID_COLUMN
    ].to_numpy() # Recuperar node_id correspondiente a node_idx_b

    if not np.array_equal(
        expected_node_ids_a,
        year_comparisons["node_id_a"].to_numpy()
    ):
        raise ValueError(
            f"La correspondencia entre node_idx_a y node_id_a "
            f"no coincide para el año {current_year}."
        ) # Validar identidad del municipio A

    if not np.array_equal(
        expected_node_ids_b,
        year_comparisons["node_id_b"].to_numpy()
    ):
        raise ValueError(
            f"La correspondencia entre node_idx_b y node_id_b "
            f"no coincide para el año {current_year}."
        ) # Validar identidad del municipio B

    # 9.10 Extracción de las Variables Predictoras
    features_a = year_standardized.loc[
        node_idx_a,
        ANALYSIS_FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float64
    ) # Extraer variables predictoras estandarizadas del municipio A

    features_b = year_standardized.loc[
        node_idx_b,
        ANALYSIS_FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float64
    ) # Extraer variables predictoras estandarizadas del municipio B

    # 9.11 Validación de Valores
    if not np.isfinite(features_a).all():
        raise ValueError(
            f"Las variables predictoras del municipio A "
            f"contienen valores no finitos en el año {current_year}."
        ) # Validar integridad numérica del municipio A

    if not np.isfinite(features_b).all():
        raise ValueError(
            f"Las variables predictoras del municipio B "
            f"contienen valores no finitos en el año {current_year}."
        ) # Validar integridad numérica del municipio B

    # 9.12 Cálculo de Similitud Coseno Global
    dot_product = np.sum(
        features_a * features_b,
        axis=1
    ) # Calcular producto punto entre los vectores municipales

    norm_a = np.linalg.norm(
        features_a,
        axis=1
    ) # Calcular norma del vector del municipio A

    norm_b = np.linalg.norm(
        features_b,
        axis=1
    ) # Calcular norma del vector del municipio B

    denominator = (
        norm_a * norm_b
    ) # Calcular denominador de la similitud coseno

    zero_norm_mask = (
        denominator <= EPSILON
    ) # Identificar pares con norma cero o prácticamente cero

    zero_norm_count = int(
        zero_norm_mask.sum()
    ) # Contabilizar pares con similitud coseno indefinida

    global_zero_norm_counts[
        current_year
    ] = zero_norm_count # Registrar cantidad de casos indefinidos

    if zero_norm_count > 0:
        raise ValueError(
            f"El año {current_year} contiene "
            f"{zero_norm_count:,} pares con norma cero o prácticamente cero; "
            "la similitud coseno global está indefinida."
        ) # Detener ejecución ante normas inválidas

    global_similarity = np.divide(
        dot_product,
        denominator
    ) # Calcular similitud coseno global

    # 9.13 Validación de la Similitud Coseno
    if not np.isfinite(global_similarity).all():
        raise ValueError(
            f"La similitud coseno global contiene valores no finitos "
            f"en el año {current_year}."
        ) # Validar integridad numérica de la similitud

    if (
        global_similarity.min() < -1.0000001
        or global_similarity.max() > 1.0000001
    ):
        raise ValueError(
            f"La similitud coseno global del año {current_year} "
            "contiene valores fuera del intervalo matemático [-1, 1]."
        ) # Validar rango matemático de la similitud coseno

    # 9.14 Cálculo de Distancia Euclídea
    euclidean_distance = np.linalg.norm(
        features_a - features_b,
        axis=1
    ) # Calcular distancia euclídea en el espacio estandarizado

    # 9.15 Validación de la Distancia Euclídea
    if not np.isfinite(euclidean_distance).all():
        raise ValueError(
            f"La distancia euclídea contiene valores no finitos "
            f"en el año {current_year}."
        ) # Validar integridad numérica de la distancia

    if (euclidean_distance < 0).any():
        raise ValueError(
            f"La distancia euclídea contiene valores negativos "
            f"en el año {current_year}."
        ) # Validar propiedad no negativa de la distancia

    # 9.16 Construcción de Resultados Globales
    year_global_similarity = pd.DataFrame({
        TIME_KEY: current_year,
        "node_idx_a": node_idx_a,
        "node_idx_b": node_idx_b,
        "node_id_a": year_comparisons["node_id_a"].to_numpy(),
        "node_id_b": year_comparisons["node_id_b"].to_numpy(),
        "similaridad_global": global_similarity,
        "distancia_euclidiana": euclidean_distance
    }) # Construir resultados globales con trazabilidad nodal

    global_similarity_results.append(
        year_global_similarity
    ) # Agregar resultados del año actual

# 9.17 Validación de la Colección de Resultados
if not global_similarity_results:
    raise ValueError(
        "No se generaron resultados de similitud global."
    ) # Validar existencia de resultados

# 9.18 Consolidación de la Similitud Global
municipality_global_similarity = pd.concat(
    global_similarity_results,
    ignore_index=True
) # Consolidar similitudes globales de todos los años

# 9.19 Validación de Identidad de los Resultados
required_result_identity = [
    TIME_KEY,
    "node_idx_a",
    "node_idx_b",
    "node_id_a",
    "node_id_b"
] # Definir identidad completa de los resultados

if municipality_global_similarity[
    required_result_identity
].isna().any().any():
    raise ValueError(
        "La tabla de similitud global contiene "
        "identificadores nodales faltantes."
    ) # Validar identidad completa de los pares

# 9.20 Validación de la Similitud Global
global_similarity_values = municipality_global_similarity[
    "similaridad_global"
].to_numpy(
    dtype=np.float64
) # Extraer valores de similitud global

if not np.isfinite(global_similarity_values).all():
    raise ValueError(
        "La similitud global contiene valores no finitos."
    ) # Validar valores finitos de la similitud global

if (
    global_similarity_values.min() < -1.0000001
    or global_similarity_values.max() > 1.0000001
):
    raise ValueError(
        "La similitud global contiene valores fuera "
        "del intervalo matemático [-1, 1]."
    ) # Validar rango matemático de la similitud global

# 9.21 Validación de la Distancia Euclídea
euclidean_values = municipality_global_similarity[
    "distancia_euclidiana"
].to_numpy(
    dtype=np.float64
) # Extraer distancias euclídeas

if not np.isfinite(euclidean_values).all():
    raise ValueError(
        "La distancia euclídea contiene valores no finitos."
    ) # Validar valores finitos de la distancia

if (euclidean_values < 0).any():
    raise ValueError(
        "La distancia euclídea contiene valores negativos."
    ) # Validar propiedad no negativa de la distancia

# 9.22 Validación de Cantidad de Resultados
expected_total_pairs = (
    ANALYSIS_N_MUNICIPALITIES
    * (ANALYSIS_N_MUNICIPALITIES - 1)
    // 2
    * ANALYSIS_N_YEARS
) # Calcular cantidad esperada de resultados municipio-año

if len(municipality_global_similarity) != expected_total_pairs:
    raise ValueError(
        f"La tabla de similitud global contiene "
        f"{len(municipality_global_similarity):,} registros; "
        f"se esperaban {expected_total_pairs:,}."
    ) # Validar cantidad total de pares

# 9.23 Validación de Cobertura Temporal
if municipality_global_similarity[
    TIME_KEY
].nunique() != ANALYSIS_N_YEARS:
    raise ValueError(
        f"La tabla de similitud global contiene "
        f"{municipality_global_similarity[TIME_KEY].nunique():,} años; "
        f"se esperaban {ANALYSIS_N_YEARS:,}."
    ) # Validar cobertura temporal completa

# 9.24 Validación de Duplicados
duplicate_global_results = municipality_global_similarity.duplicated(
    subset=[
        TIME_KEY,
        "node_idx_a",
        "node_idx_b"
    ]
).sum() # Contar pares municipio-año duplicados

if duplicate_global_results > 0:
    raise ValueError(
        f"La tabla de similitud global contiene "
        f"{duplicate_global_results:,} pares municipio-año duplicados."
    ) # Detener ejecución ante duplicados

# 9.25 Validación de Orden y Autocombinaciones
if (
    municipality_global_similarity["node_idx_a"]
    >= municipality_global_similarity["node_idx_b"]
).any():
    raise ValueError(
        "La tabla de similitud global contiene "
        "pares nodales mal ordenados o autocombinaciones."
    ) # Validar representación única y ausencia de autocombinaciones

# 9.26 Ordenamiento de los Resultados
municipality_global_similarity = (
    municipality_global_similarity
    .sort_values(
        by=[
            TIME_KEY,
            "node_idx_a",
            "node_idx_b"
        ]
    )
    .reset_index(drop=True)
) # Ordenar resultados por año y espacio nodal

# 9.27 Exportación de la Similitud Global
municipality_global_similarity.to_parquet(
    GLOBAL_SIMILARITY_OUTPUT_PATH,
    index=False
) # Exportar similitud global y distancia euclídea

# 9.28 Registro de Resultados
logger.info(
    "Similitud global calculada: %s pares municipio-año mediante %s.",
    len(municipality_global_similarity),
    PRIMARY_SIMILARITY_METRIC
) # Registrar resultado del análisis de similitud global

# BLOQUE 10. ANÁLISIS TOPOLÓGICO
# Objetivo: Caracterizar la similitud estructural entre municipios utilizando
# la posición de cada municipio dentro del grafo y la estructura de sus vecinos.
# Arquitectura científica
# Entradas: GraphData validados, edge_index anual, Catálogo Oficial de Nodos y todos los pares municipales 
# generados en municipality_comparison.
# Proceso: Construcción de vecindarios a partir de edge_index y cálculo de grado, vecinos compartidos, overlap
# y similitud topológica para todos los pares municipio-año.
# Producto: municipality_topology_similarity.parquet con indicadores topológicos para todos los pares municipio-año.

# 10.1 Validación de la Configuración Topológica
if ENABLE_TOPOLOGICAL_ANALYSIS is not True:
    logger.info(
        "El análisis topológico está desactivado en la configuración."
    ) # Registrar que el análisis topológico no será ejecutado

else:
    if not 0 <= TOPOLOGY_NEIGHBOR_OVERLAP_WEIGHT <= 1:
        raise ValueError(
            f"El peso del overlap de vecinos debe estar entre 0 y 1: "
            f"{TOPOLOGY_NEIGHBOR_OVERLAP_WEIGHT}"
        ) # Validar peso del overlap de vecinos

    if not 0 <= TOPOLOGY_DEGREE_WEIGHT <= 1:
        raise ValueError(
            f"El peso de similitud de grado debe estar entre 0 y 1: "
            f"{TOPOLOGY_DEGREE_WEIGHT}"
        ) # Validar peso de similitud de grado

    topology_weight_sum = (
        TOPOLOGY_NEIGHBOR_OVERLAP_WEIGHT
        + TOPOLOGY_DEGREE_WEIGHT
    ) # Calcular suma de los pesos topológicos

    if not np.isclose(
        topology_weight_sum,
        1.0,
        atol=1e-10
    ):
        raise ValueError(
            f"Los pesos topológicos deben sumar 1.0; "
            f"suma actual: {topology_weight_sum}"
        ) # Validar suma de pesos topológicos

    topology_similarity_results = [] # Inicializar resultados topológicos por año

    # 10.2 Procesamiento Topológico por Año
    for current_year in ANALYSIS_YEARS:
        graph_data = graph_data_by_year[
            current_year
        ] # Obtener GraphData validado del año actual

        year_profile = municipality_profile[
            municipality_profile[TIME_KEY] == current_year
        ].copy() # Obtener perfil municipal del año actual

        year_comparisons = municipality_comparison[
            municipality_comparison[TIME_KEY] == current_year
        ].copy() # Obtener todos los pares municipales del año actual

        # 10.3 Validación del GraphData y edge_index
        if not hasattr(graph_data, "edge_index"):
            raise ValueError(
                f"El GraphData del año {current_year} no contiene edge_index."
            ) # Validar existencia de la estructura topológica

        if graph_data.num_nodes != ANALYSIS_N_MUNICIPALITIES:
            raise ValueError(
                f"El GraphData del año {current_year} contiene "
                f"{graph_data.num_nodes:,} nodos; "
                f"se esperaban {ANALYSIS_N_MUNICIPALITIES:,}."
            ) # Validar cantidad oficial de nodos

        edge_index = (
            graph_data.edge_index
            .detach()
            .cpu()
            .numpy()
        ) # Obtener edge_index utilizando exclusivamente índices nodales

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"El edge_index del año {current_year} debe tener "
                f"estructura [2, E]. "
                f"Dimensiones encontradas: {tuple(edge_index.shape)}."
            ) # Validar estructura matricial de edge_index

        if edge_index.shape[1] == 0:
            raise ValueError(
                f"El GraphData del año {current_year} no contiene aristas."
            ) # Validar existencia de conexiones topológicas

        if not np.issubdtype(
            edge_index.dtype,
            np.integer
        ):
            raise ValueError(
                f"El edge_index del año {current_year} debe contener "
                "índices enteros."
            ) # Validar tipo de los índices topológicos

        if edge_index.min() < 0:
            raise ValueError(
                f"El edge_index del año {current_year} contiene "
                "índices negativos."
            ) # Validar límite inferior del espacio nodal

        if edge_index.max() >= ANALYSIS_N_MUNICIPALITIES:
            raise ValueError(
                f"El edge_index del año {current_year} contiene "
                "índices fuera del espacio nodal oficial "
                f"0 ... {ANALYSIS_N_MUNICIPALITIES - 1}."
            ) # Validar límite superior del espacio nodal

        # 10.4 Validación de la Identidad Municipal
        required_identity_columns = [
            NODE_INDEX_COLUMN,
            NODE_ID_COLUMN,
            MUNICIPALITY_ID_COLUMN,
            MUNICIPALITY_NAME_COLUMN
        ] # Definir columnas oficiales de identidad territorial

        missing_profile_identity = [
            column
            for column in required_identity_columns
            if column not in year_profile.columns
        ] # Identificar columnas de identidad ausentes del perfil

        if missing_profile_identity:
            raise ValueError(
                f"El perfil municipal del año {current_year} no contiene "
                f"las columnas de identidad requeridas: "
                f"{missing_profile_identity}."
            ) # Validar identidad completa del perfil

        missing_catalog_identity = [
            column
            for column in required_identity_columns
            if column not in node_catalog.columns
        ] # Identificar columnas de identidad ausentes del catálogo

        if missing_catalog_identity:
            raise ValueError(
                f"El Catálogo Oficial de Nodos no contiene "
                f"las columnas de identidad requeridas: "
                f"{missing_catalog_identity}."
            ) # Validar identidad completa del catálogo

        profile_identity = year_profile[
            required_identity_columns
        ].copy() # Extraer identidad territorial del perfil anual

        catalog_identity = node_catalog[
            required_identity_columns
        ].copy() # Extraer identidad territorial del catálogo oficial

        if len(profile_identity) != ANALYSIS_N_MUNICIPALITIES:
            raise ValueError(
                f"El perfil municipal del año {current_year} contiene "
                f"{len(profile_identity):,} registros; "
                f"se esperaban {ANALYSIS_N_MUNICIPALITIES:,}."
            ) # Validar cobertura municipal del perfil

        if len(catalog_identity) != ANALYSIS_N_MUNICIPALITIES:
            raise ValueError(
                f"El Catálogo Oficial de Nodos contiene "
                f"{len(catalog_identity):,} registros; "
                f"se esperaban {ANALYSIS_N_MUNICIPALITIES:,}."
            ) # Validar cobertura del catálogo

        profile_identity = (
            profile_identity
            .sort_values(NODE_INDEX_COLUMN)
            .reset_index(drop=True)
        ) # Ordenar identidad del perfil según node_idx

        catalog_identity = (
            catalog_identity
            .sort_values(NODE_INDEX_COLUMN)
            .reset_index(drop=True)
        ) # Ordenar identidad del catálogo según node_idx

        expected_node_indices = np.arange(
            ANALYSIS_N_MUNICIPALITIES,
            dtype=np.int64
        ) # Construir secuencia oficial de node_idx

        profile_node_indices = profile_identity[
            NODE_INDEX_COLUMN
        ].to_numpy(
            dtype=np.int64
        ) # Obtener índices nodales del perfil

        catalog_node_indices = catalog_identity[
            NODE_INDEX_COLUMN
        ].to_numpy(
            dtype=np.int64
        ) # Obtener índices nodales del catálogo

        if not np.array_equal(
            profile_node_indices,
            expected_node_indices
        ):
            raise ValueError(
                f"El perfil municipal del año {current_year} no contiene "
                "la secuencia nodal oficial completa."
            ) # Validar espacio nodal completo del perfil

        if not np.array_equal(
            catalog_node_indices,
            expected_node_indices
        ):
            raise ValueError(
                "El Catálogo Oficial de Nodos no contiene "
                "la secuencia nodal oficial completa."
            ) # Validar espacio nodal completo del catálogo

        if not np.array_equal(
            profile_identity[NODE_ID_COLUMN].to_numpy(),
            catalog_identity[NODE_ID_COLUMN].to_numpy()
        ):
            raise ValueError(
                f"La correspondencia node_idx-node_id del perfil municipal "
                f"del año {current_year} no coincide con el Catálogo Oficial."
            ) # Validar identidad científica mediante node_idx

        if not np.array_equal(
            profile_identity[MUNICIPALITY_ID_COLUMN].to_numpy(),
            catalog_identity[MUNICIPALITY_ID_COLUMN].to_numpy()
        ):
            raise ValueError(
                f"La correspondencia node_idx-municipality_id del perfil "
                f"del año {current_year} no coincide con el Catálogo Oficial."
            ) # Validar identidad municipal mediante node_idx

        # 10.5 Validación de los Pares Analíticos
        required_pair_columns = [
            TIME_KEY,
            "node_idx_a",
            "node_idx_b",
            "node_id_a",
            "node_id_b"
        ] # Definir estructura mínima de los pares analíticos

        missing_pair_columns = [
            column
            for column in required_pair_columns
            if column not in year_comparisons.columns
        ] # Identificar columnas faltantes de los pares

        if missing_pair_columns:
            raise ValueError(
                f"Los pares municipales del año {current_year} "
                f"no contienen las columnas requeridas: "
                f"{missing_pair_columns}."
            ) # Validar estructura de los pares

        expected_pairs_per_year = (
            ANALYSIS_N_MUNICIPALITIES
            * (ANALYSIS_N_MUNICIPALITIES - 1)
            // 2
        ) # Calcular todos los pares municipales posibles

        if len(year_comparisons) != expected_pairs_per_year:
            raise ValueError(
                f"El año {current_year} contiene "
                f"{len(year_comparisons):,} pares municipales; "
                f"se esperaban {expected_pairs_per_year:,}."
            ) # Validar universo completo de pares

        if year_comparisons[
            ["node_idx_a", "node_idx_b"]
        ].isna().any().any():
            raise ValueError(
                f"Los pares municipales del año {current_year} "
                "contienen índices nodales faltantes."
            ) # Validar integridad de los pares

        node_idx_a = year_comparisons[
            "node_idx_a"
        ].to_numpy(
            dtype=np.int64
        ) # Extraer índices del extremo A

        node_idx_b = year_comparisons[
            "node_idx_b"
        ].to_numpy(
            dtype=np.int64
        ) # Extraer índices del extremo B

        if (
            (node_idx_a < 0).any()
            or (node_idx_b < 0).any()
            or (node_idx_a >= ANALYSIS_N_MUNICIPALITIES).any()
            or (node_idx_b >= ANALYSIS_N_MUNICIPALITIES).any()
        ):
            raise ValueError(
                f"Los pares municipales del año {current_year} "
                "contienen node_idx fuera del espacio nodal oficial."
            ) # Validar espacio nodal de los pares

        if (node_idx_a >= node_idx_b).any():
            raise ValueError(
                f"Los pares municipales del año {current_year} "
                "no cumplen la condición node_idx_a < node_idx_b."
            ) # Validar representación única de pares

        if year_comparisons[
            ["node_idx_a", "node_idx_b"]
        ].duplicated().any():
            raise ValueError(
                f"Los pares municipales del año {current_year} "
                "contienen pares node_idx_a-node_idx_b duplicados."
            ) # Validar unicidad de los pares

        # 10.6 Validación de Correspondencia entre node_idx y node_id
        expected_node_ids_a = catalog_identity.loc[
            node_idx_a,
            NODE_ID_COLUMN
        ].to_numpy() # Recuperar node_id oficial del nodo A

        expected_node_ids_b = catalog_identity.loc[
            node_idx_b,
            NODE_ID_COLUMN
        ].to_numpy() # Recuperar node_id oficial del nodo B

        if not np.array_equal(
            expected_node_ids_a,
            year_comparisons["node_id_a"].to_numpy()
        ):
            raise ValueError(
                f"La correspondencia node_idx_a-node_id_a "
                f"no coincide en el año {current_year}."
            ) # Validar identidad del extremo A

        if not np.array_equal(
            expected_node_ids_b,
            year_comparisons["node_id_b"].to_numpy()
        ):
            raise ValueError(
                f"La correspondencia node_idx_b-node_id_b "
                f"no coincide en el año {current_year}."
            ) # Validar identidad del extremo B

        # 10.7 Construcción de los Vecindarios desde edge_index
        adjacency_neighbors = [
            set()
            for _ in range(
                ANALYSIS_N_MUNICIPALITIES
            )
        ] # Inicializar vecindario de cada nodo

        for source_node, target_node in edge_index.T:
            source_node = int(
                source_node
            ) # Convertir nodo origen a entero

            target_node = int(
                target_node
            ) # Convertir nodo destino a entero

            if source_node == target_node:
                raise ValueError(
                    f"El edge_index del año {current_year} "
                    f"contiene una autoarista en el nodo {source_node}."
                ) # Validar ausencia de autoaristas

            adjacency_neighbors[
                source_node
            ].add(
                target_node
            ) # Registrar vecino del origen

            adjacency_neighbors[
                target_node
            ].add(
                source_node
            ) # Registrar vecino del destino

        logger.info(
            "Vecindarios construidos para %s a partir de %s aristas.",
            current_year,
            edge_index.shape[1]
        ) # Registrar construcción de la estructura de vecindarios

        # 10.8 Cálculo del Grado para Todos los Pares Analíticos
        degree_a = np.array(
            [
                len(
                    adjacency_neighbors[node_index]
                )
                for node_index in node_idx_a
            ],
            dtype=np.float64
        ) # Obtener grado topológico del nodo A

        degree_b = np.array(
            [
                len(
                    adjacency_neighbors[node_index]
                )
                for node_index in node_idx_b
            ],
            dtype=np.float64
        ) # Obtener grado topológico del nodo B

        # 10.9 Cálculo de Vecinos Compartidos
        shared_neighbors = np.array(
            [
                len(
                    adjacency_neighbors[node_a]
                    &
                    adjacency_neighbors[node_b]
                )
                for node_a, node_b in zip(
                    node_idx_a,
                    node_idx_b
                )
            ],
            dtype=np.float64
        ) # Calcular número de vecinos compartidos para todos los pares

        # 10.10 Cálculo del Overlap de Vecinos
        if SIMILARITY_NEIGHBOR_OVERLAP_ENABLED:
            if TOPOLOGY_NEIGHBOR_OVERLAP_WEIGHT <= 0:
                raise ValueError(
                    "El overlap de vecinos está habilitado, "
                    "pero su peso debe ser mayor que cero."
                ) # Validar coherencia entre activación y peso

            neighbor_union = np.array(
                [
                    len(
                        adjacency_neighbors[node_a]
                        |
                        adjacency_neighbors[node_b]
                    )
                    for node_a, node_b in zip(
                        node_idx_a,
                        node_idx_b
                    )
                ],
                dtype=np.float64
            ) # Calcular tamaño de la unión de vecinos

            both_zero_union = (
                neighbor_union == 0
            ) # Identificar pares cuyos dos vecindarios son vacíos

            neighbor_overlap = np.divide(
                shared_neighbors,
                neighbor_union,
                out=np.ones_like(shared_neighbors),
                where=neighbor_union > 0
            ) # Calcular Jaccard y asignar 1 a dos vecindarios vacíos

        else:

            if not np.isclose(
                TOPOLOGY_NEIGHBOR_OVERLAP_WEIGHT,
                0.0,
                atol=1e-10
            ):
                raise ValueError(
                    "El overlap de vecinos está deshabilitado, "
                    "pero su peso es diferente de cero."
                ) # Validar coherencia entre activación y peso

            neighbor_overlap = np.zeros_like(
                shared_neighbors
            ) # Mantener overlap en cero cuando está deshabilitado

        # 10.11 Cálculo de Similitud de Grado
        degree_minimum = np.minimum(
            degree_a,
            degree_b
        ) # Obtener menor grado de cada par

        degree_maximum = np.maximum(
            degree_a,
            degree_b
        ) # Obtener mayor grado de cada par

        degree_similarity = np.divide(
            degree_minimum,
            degree_maximum,
            out=np.zeros_like(degree_minimum),
            where=degree_maximum > 0
        ) # Calcular similitud relativa de grado

        both_zero_degree = (
            (degree_a == 0)
            &
            (degree_b == 0)
        ) # Identificar pares con grado cero en ambos nodos

        degree_similarity[
            both_zero_degree
        ] = 1.0 # Considerar equivalentes dos nodos aislados

        # 10.12 Cálculo de Similitud Topológica
        topology_similarity = (
            TOPOLOGY_NEIGHBOR_OVERLAP_WEIGHT
            * neighbor_overlap
            +
            TOPOLOGY_DEGREE_WEIGHT
            * degree_similarity
        ) # Combinar componentes topológicos según configuración oficial

        # 10.13 Validación Numérica de los Resultados
        topology_arrays = {
            "grado_a": degree_a,
            "grado_b": degree_b,
            "vecinos_compartidos": shared_neighbors,
            "overlap_vecinos": neighbor_overlap,
            "similaridad_grado": degree_similarity,
            "similaridad_topologica": topology_similarity
        } # Agrupar métricas topológicas para validación

        for metric_name, metric_values in topology_arrays.items():

            if not np.isfinite(metric_values).all():
                raise ValueError(
                    f"La métrica topológica '{metric_name}' "
                    f"contiene valores no finitos en el año {current_year}."
                ) # Validar valores numéricos finitos

        if (
            neighbor_overlap.min() < -1e-10
            or neighbor_overlap.max() > 1.0000001
        ):
            raise ValueError(
                f"El overlap de vecinos del año {current_year} "
                "está fuera del rango [0, 1]."
            ) # Validar rango del overlap

        if (
            degree_similarity.min() < -1e-10
            or degree_similarity.max() > 1.0000001
        ):
            raise ValueError(
                f"La similitud de grado del año {current_year} "
                "está fuera del rango [0, 1]."
            ) # Validar rango de similitud de grado

        if (
            topology_similarity.min() < -1e-10
            or topology_similarity.max() > 1.0000001
        ):
            raise ValueError(
                f"La similitud topológica del año {current_year} "
                "está fuera del rango [0, 1]."
            ) # Validar rango de similitud topológica

        # 10.14 Construcción de la Tabla Topológica
        year_topology = pd.DataFrame({
            TIME_KEY: current_year,
            "node_idx_a": node_idx_a,
            "node_idx_b": node_idx_b,
            "node_id_a": year_comparisons[
                "node_id_a"
            ].to_numpy(),
            "node_id_b": year_comparisons[
                "node_id_b"
            ].to_numpy(),
            "municipality_id_a": year_comparisons[
                "municipality_id_a"
            ].to_numpy()
            if "municipality_id_a" in year_comparisons.columns
            else catalog_identity.loc[
                node_idx_a,
                MUNICIPALITY_ID_COLUMN
            ].to_numpy(),
            "municipality_id_b": year_comparisons[
                "municipality_id_b"
            ].to_numpy()
            if "municipality_id_b" in year_comparisons.columns
            else catalog_identity.loc[
                node_idx_b,
                MUNICIPALITY_ID_COLUMN
            ].to_numpy(),
            "municipality_a": year_comparisons[
                "municipality_a"
            ].to_numpy()
            if "municipality_a" in year_comparisons.columns
            else catalog_identity.loc[
                node_idx_a,
                MUNICIPALITY_NAME_COLUMN
            ].to_numpy(),
            "municipality_b": year_comparisons[
                "municipality_b"
            ].to_numpy()
            if "municipality_b" in year_comparisons.columns
            else catalog_identity.loc[
                node_idx_b,
                MUNICIPALITY_NAME_COLUMN
            ].to_numpy(),
            "grado_a": degree_a,
            "grado_b": degree_b,
            "diferencia_grado": np.abs(
                degree_a - degree_b
            ),
            "vecinos_compartidos": shared_neighbors,
            "overlap_vecinos": neighbor_overlap,
            "similaridad_grado": degree_similarity,
            "similaridad_topologica": topology_similarity
        }) # Construir resultados topológicos para todos los pares municipales

        topology_similarity_results.append(
            year_topology
        ) # Agregar resultados topológicos del año

    # 10.15 Validación de la Colección de Resultados
    if not topology_similarity_results:
        raise ValueError(
            "No se generaron resultados del análisis topológico."
        ) # Validar existencia de resultados

    # 10.16 Consolidación del Análisis Topológico
    municipality_topology_similarity = pd.concat(
        topology_similarity_results,
        ignore_index=True
    ) # Consolidar resultados de todos los años

    # 10.17 Validación del Número de Resultados
    expected_pairs_per_year = (
        ANALYSIS_N_MUNICIPALITIES
        * (ANALYSIS_N_MUNICIPALITIES - 1)
        // 2
    ) # Calcular pares esperados por año

    expected_total_pairs = (
        expected_pairs_per_year
        * ANALYSIS_N_YEARS
    ) # Calcular pares esperados para todo el periodo

    if len(municipality_topology_similarity) != expected_total_pairs:
        raise ValueError(
            f"La tabla topológica contiene "
            f"{len(municipality_topology_similarity):,} registros; "
            f"se esperaban {expected_total_pairs:,}."
        ) # Validar cobertura completa del universo analítico

    # 10.18 Validación de Cobertura Temporal
    observed_years = set(
        municipality_topology_similarity[TIME_KEY]
        .dropna()
        .astype(int)
    ) # Obtener años presentes en los resultados

    expected_years = set(
        ANALYSIS_YEARS
    ) # Definir años científicos oficiales

    if observed_years != expected_years:
        raise ValueError(
            f"La cobertura temporal de la tabla topológica no coincide "
            f"con los años oficiales. "
            f"Esperados: {sorted(expected_years)}. "
            f"Encontrados: {sorted(observed_years)}."
        ) # Validar cobertura temporal completa

    # 10.19 Validación de Duplicados
    duplicate_topology_results = (
        municipality_topology_similarity
        .duplicated(
            subset=[
                TIME_KEY,
                "node_idx_a",
                "node_idx_b"
            ]
        )
        .sum()
    ) # Contar pares topológicos duplicados por año

    if duplicate_topology_results > 0:
        raise ValueError(
            f"La tabla topológica contiene "
            f"{duplicate_topology_results:,} pares "
            "node_idx_a-node_idx_b duplicados."
        ) # Detener ejecución ante duplicados

    # 10.20 Validación de Autocombinaciones y Orden Nodal
    if (
        municipality_topology_similarity[
            "node_idx_a"
        ]
        >=
        municipality_topology_similarity[
            "node_idx_b"
        ]
    ).any():
        raise ValueError(
            "La tabla topológica contiene pares inválidos: "
            "node_idx_a debe ser menor que node_idx_b."
        ) # Validar ausencia de autocombinaciones y representación única

    # 10.21 Validación del Espacio Nodal
    topology_node_indices = municipality_topology_similarity[
        ["node_idx_a", "node_idx_b"]
    ].to_numpy(
        dtype=np.int64
    ) # Extraer índices nodales de los resultados

    if (
        topology_node_indices.min() < 0
        or topology_node_indices.max() >= ANALYSIS_N_MUNICIPALITIES
    ):
        raise ValueError(
            "La tabla topológica contiene node_idx fuera "
            "del espacio oficial 0 ... N-1."
        ) # Validar espacio nodal de los resultados

    # 10.22 Validación de la Correspondencia Nodal
    topology_identity = municipality_topology_similarity[
        [
            "node_idx_a",
            "node_idx_b",
            "node_id_a",
            "node_id_b"
        ]
    ].copy() # Extraer identidad nodal de los resultados

    expected_ids_a = catalog_identity.loc[
        topology_identity["node_idx_a"].to_numpy(dtype=np.int64),
        NODE_ID_COLUMN
    ].to_numpy() # Recuperar node_id oficial del extremo A

    expected_ids_b = catalog_identity.loc[
        topology_identity["node_idx_b"].to_numpy(dtype=np.int64),
        NODE_ID_COLUMN
    ].to_numpy() # Recuperar node_id oficial del extremo B

    if not np.array_equal(
        expected_ids_a,
        topology_identity["node_id_a"].to_numpy()
    ):
        raise ValueError(
            "La correspondencia node_idx_a-node_id_a "
            "no es consistente en la tabla topológica."
        ) # Validar identidad del extremo A

    if not np.array_equal(
        expected_ids_b,
        topology_identity["node_id_b"].to_numpy()
    ):
        raise ValueError(
            "La correspondencia node_idx_b-node_id_b "
            "no es consistente en la tabla topológica."
        ) # Validar identidad del extremo B


    # 10.23 Validación de las Métricas Topológicas
    topology_similarity_columns = [
        "overlap_vecinos",
        "similaridad_grado",
        "similaridad_topologica"
    ] # Definir métricas de similitud topológica

    for similarity_column in topology_similarity_columns:
        similarity_values = municipality_topology_similarity[
            similarity_column
        ].to_numpy(
            dtype=np.float64
        ) # Extraer valores de la métrica

        if not np.isfinite(
            similarity_values
        ).all():
            raise ValueError(
                f"La columna '{similarity_column}' "
                "contiene valores no finitos."
            ) # Validar valores finitos

        if (
            similarity_values.min() < -1e-10
            or similarity_values.max() > 1.0000001
        ):
            raise ValueError(
                f"La columna '{similarity_column}' "
                "contiene valores fuera del rango [0, 1]."
            ) # Validar rango matemático de la similitud

    # 10.24 Validación de la Estructura Final
    required_topology_columns = [
        TIME_KEY,
        "node_idx_a",
        "node_idx_b",
        "node_id_a",
        "node_id_b",
        "municipality_id_a",
        "municipality_id_b",
        "municipality_a",
        "municipality_b",
        "grado_a",
        "grado_b",
        "diferencia_grado",
        "vecinos_compartidos",
        "overlap_vecinos",
        "similaridad_grado",
        "similaridad_topologica"
    ] # Definir estructura mínima de la tabla topológica

    missing_topology_columns = [
        column
        for column in required_topology_columns
        if column not in municipality_topology_similarity.columns
    ] # Identificar columnas faltantes

    if missing_topology_columns:
        raise ValueError(
            f"La tabla topológica no contiene las columnas requeridas: "
            f"{missing_topology_columns}."
        ) # Validar estructura final de la tabla

    if municipality_topology_similarity[
        [
            "node_idx_a",
            "node_idx_b",
            "node_id_a",
            "node_id_b"
        ]
    ].isna().any().any():
        raise ValueError(
            "La tabla topológica contiene identificadores nodales faltantes."
        ) # Validar integridad de la identidad nodal

    # 10.25 Ordenamiento de los Resultados
    municipality_topology_similarity = (
        municipality_topology_similarity
        .sort_values(
            by=[
                TIME_KEY,
                "node_idx_a",
                "node_idx_b"
            ]
        )
        .reset_index(drop=True)
    ) # Ordenar resultados mediante el espacio nodal oficial

    # 10.26 Exportación del Análisis Topológico
    municipality_topology_similarity.to_parquet(
        TOPOLOGY_SIMILARITY_OUTPUT_PATH,
        index=False
    ) # Exportar tabla de similitud topológica

    # 10.27 Registro del Resultado
    logger.info(
        "Análisis topológico completado: %s pares municipio-año.",
        len(municipality_topology_similarity)
    ) # Registrar número total de resultados

    logger.info(
        "Periodo topológico: %s-%s.",
        ANALYSIS_YEARS[0],
        ANALYSIS_YEARS[-1]
    ) # Registrar periodo analizado

    logger.info(
        "Pesos topológicos: overlap=%.3f | grado=%.3f.",
        TOPOLOGY_NEIGHBOR_OVERLAP_WEIGHT,
        TOPOLOGY_DEGREE_WEIGHT
    ) # Registrar configuración de ponderación topológica

    logger.info(
        "Tabla topológica exportada: %s.",
        TOPOLOGY_SIMILARITY_OUTPUT_PATH
    ) # Registrar ubicación del resultado exportado

# BLOQUE 11. INTEGRACIÓN DE SIMILITUD MUNICIPAL
# Objetivo: Integrar las similitudes por dominio, similitud global, diferencias municipales y estructura topológica
# en una única tabla analítica.
# Arquitectura científica
# Entradas: resultados de similitud por dominio, similitud global, comparación municipal y análisis topológico.
# Proceso: Integración mediante la clave tiempo-node_idx_a-node_idx_b y validación de correspondencia entre
# todas las fuentes analíticas.
# Producto: municipality_similarity.parquet como tabla científica consolidada de similitud municipal.

# 11.1 Preparación de las Tablas Analíticas
domain_similarity = (
    municipality_domain_similarity.copy()
) # Copiar resultados de similitud por dominio

global_similarity = (
    municipality_global_similarity.copy()
) # Copiar resultados de similitud global

comparison_results = (
    municipality_comparison.copy()
) # Copiar resultados de comparación municipal

if ENABLE_TOPOLOGICAL_ANALYSIS:
    topology_similarity = (
        municipality_topology_similarity.copy()
    ) # Copiar resultados topológicos cuando el análisis está habilitado

# 11.2 Definición de las Claves de Integración
similarity_keys = [
    TIME_KEY,
    "node_idx_a",
    "node_idx_b"
] # Definir la unidad analítica única municipio-año-par

# 11.3 Definición de las Tablas Analíticas
analysis_tables = [
    ("domain_similarity", domain_similarity),
    ("global_similarity", global_similarity),
    ("comparison_results", comparison_results)
] # Definir tablas analíticas obligatorias

if ENABLE_TOPOLOGICAL_ANALYSIS:
    analysis_tables.append(
        ("topology_similarity", topology_similarity)
    ) # Incorporar tabla topológica cuando está habilitada

# 11.4 Validación de Existencia de las Tablas
for table_name, table in analysis_tables:
    if table.empty:
        raise ValueError(
            f"La tabla '{table_name}' está vacía."
        ) # Detener integración ante tablas sin resultados

# 11.5 Validación de las Claves de Integración
for table_name, table in analysis_tables:
    missing_key_columns = [
        column
        for column in similarity_keys
        if column not in table.columns
    ] # Identificar claves estructurales ausentes

    if missing_key_columns:
        raise ValueError(
            f"La tabla '{table_name}' no contiene las claves "
            f"estructurales requeridas: {missing_key_columns}."
        ) # Detener integración ante claves incompletas

# 11.6 Validación del Tipo y Espacio de los Índices Nodales
for table_name, table in analysis_tables:
    for node_column in [
        "node_idx_a",
        "node_idx_b"
    ]:

        node_values = table[
            node_column
        ].to_numpy(
            dtype=np.int64
        ) # Obtener índices nodales

        if table[node_column].isna().any():
            raise ValueError(
                f"La tabla '{table_name}' contiene valores faltantes "
                f"en '{node_column}'."
            ) # Validar ausencia de índices faltantes

        if (
            node_values < 0
        ).any() or (
            node_values >= ANALYSIS_N_MUNICIPALITIES
        ).any():
            raise ValueError(
                f"La tabla '{table_name}' contiene valores de "
                f"'{node_column}' fuera del espacio nodal oficial "
                f"0 ... {ANALYSIS_N_MUNICIPALITIES - 1}."
            ) # Validar espacio oficial de índices nodales

# 11.7 Validación del Orden de los Pares
for table_name, table in analysis_tables:
    if (
        table["node_idx_a"]
        >=
        table["node_idx_b"]
    ).any():
        raise ValueError(
            f"La tabla '{table_name}' contiene pares inválidos. "
            "Debe cumplirse node_idx_a < node_idx_b."
        ) # Validar representación única de los pares

# 11.8 Validación de Unicidad de las Tablas
for table_name, table in analysis_tables:
    duplicate_count = (
        table
        .duplicated(
            subset=similarity_keys
        )
        .sum()
    ) # Contar unidades analíticas duplicadas

    if duplicate_count > 0:
        raise ValueError(
            f"La tabla '{table_name}' contiene "
            f"{duplicate_count:,} claves "
            "tiempo-node_idx_a-node_idx_b duplicadas."
        ) # Detener integración ante duplicados

# 11.9 Definición del Universo Analítico Esperado
expected_pairs_per_year = (
    ANALYSIS_N_MUNICIPALITIES
    * (ANALYSIS_N_MUNICIPALITIES - 1)
    // 2
) # Calcular todos los pares municipales posibles por año

expected_total_pairs = (
    expected_pairs_per_year
    * ANALYSIS_N_YEARS
) # Calcular todos los pares municipio-año del panel

# 11.10 Validación de Cobertura de Cada Fuente
for table_name, table in analysis_tables:
    if len(table) != expected_total_pairs:
        raise ValueError(
            f"La tabla '{table_name}' contiene "
            f"{len(table):,} registros; "
            f"se esperaban {expected_total_pairs:,}."
        ) # Validar cobertura completa del universo analítico

# 11.11 Construcción del Universo de Referencia
reference_pairs = set(
    map(
        tuple,
        domain_similarity[
            similarity_keys
        ].itertuples(
            index=False,
            name=None
        )
    )
) # Definir conjunto oficial de claves analíticas

if len(reference_pairs) != expected_total_pairs:
    raise ValueError(
        f"La similitud por dominio contiene "
        f"{len(reference_pairs):,} pares únicos; "
        f"se esperaban {expected_total_pairs:,}."
    ) # Validar universo único de referencia

# 11.12 Validación de Cobertura entre Fuentes
for table_name, table in analysis_tables:
    table_pairs = set(
        map(
            tuple,
            table[
                similarity_keys
            ].itertuples(
                index=False,
                name=None
            )
        )
    ) # Construir conjunto de claves de cada fuente

    if table_pairs != reference_pairs:
        missing_pairs = reference_pairs - table_pairs
        extra_pairs = table_pairs - reference_pairs

        raise ValueError(
            f"La cobertura de '{table_name}' no coincide con "
            "el universo analítico de referencia. "
            f"Pares faltantes: {len(missing_pairs):,}. "
            f"Pares adicionales: {len(extra_pairs):,}."
        ) # Validar correspondencia exacta entre fuentes

# 11.13 Integración de Similitud por Dominio y Global
municipality_similarity = domain_similarity.merge(
    global_similarity,
    on=similarity_keys,
    how="inner",
    validate="one_to_one",
    suffixes=("", "_global")
) # Integrar similitud por dominio y similitud global


if len(municipality_similarity) != expected_total_pairs:
    raise ValueError(
        f"La integración de similitud por dominio y global produjo "
        f"{len(municipality_similarity):,} registros; "
        f"se esperaban {expected_total_pairs:,}."
    ) # Validar conservación completa de los pares

# 11.14 Integración de las Diferencias Municipales
comparison_columns = [
    column
    for column in [
        "difference_absolute_mean",
        "difference_standardized_mean",
        "difference_target",
        "difference_raw_target"
    ]
    if column in comparison_results.columns
] # Seleccionar únicamente diferencias disponibles

if comparison_columns:

    municipality_similarity = municipality_similarity.merge(
        comparison_results[
            similarity_keys + comparison_columns
        ],
        on=similarity_keys,
        how="inner",
        validate="one_to_one"
    ) # Incorporar diferencias municipales

    if len(municipality_similarity) != expected_total_pairs:
        raise ValueError(
            f"La integración de diferencias municipales produjo "
            f"{len(municipality_similarity):,} registros; "
            f"se esperaban {expected_total_pairs:,}."
        ) # Validar conservación completa de los pares

# 11.15 Integración de la Información Topológica
if ENABLE_TOPOLOGICAL_ANALYSIS:
    topology_columns = [
        column
        for column in [
            "grado_a",
            "grado_b",
            "diferencia_grado",
            "vecinos_compartidos",
            "overlap_vecinos",
            "similaridad_grado",
            "similaridad_topologica"
        ]
        if column in topology_similarity.columns
    ] # Seleccionar indicadores topológicos disponibles

    if not topology_columns:
        raise ValueError(
            "El análisis topológico está habilitado, "
            "pero no existen indicadores topológicos para integrar."
        ) # Validar disponibilidad de información topológica

    municipality_similarity = municipality_similarity.merge(
        topology_similarity[
            similarity_keys + topology_columns
        ],
        on=similarity_keys,
        how="inner",
        validate="one_to_one"
    ) # Incorporar indicadores topológicos

    if len(municipality_similarity) != expected_total_pairs:
        raise ValueError(
            f"La integración topológica produjo "
            f"{len(municipality_similarity):,} registros; "
            f"se esperaban {expected_total_pairs:,}."
        ) # Validar conservación completa de los pares

# 11.16 Validación de Identidad Territorial
required_identity_columns = [
    "node_id_a",
    "node_id_b",
    "municipality_id_a",
    "municipality_id_b"
] # Definir identidad territorial requerida

missing_identity_columns = [
    column
    for column in required_identity_columns
    if column not in municipality_similarity.columns
] # Identificar columnas territoriales ausentes

if missing_identity_columns:
    raise ValueError(
        f"La tabla consolidada no contiene las columnas "
        f"de identidad requeridas: {missing_identity_columns}."
    ) # Validar disponibilidad de identidad territorial

if municipality_similarity[
    required_identity_columns
].isna().any().any():
    raise ValueError(
        "La tabla consolidada contiene identificadores "
        "territoriales faltantes."
    ) # Validar integridad de la identidad territorial

# 11.17 Validación de Correspondencia mediante node_idx
identity_pairs = [
    (
        "node_idx_a",
        "node_id_a",
        "municipality_id_a"
    ),
    (
        "node_idx_b",
        "node_id_b",
        "municipality_id_b"
    )
] # Definir correspondencias nodales esperadas

catalog_identity_sorted = (
    node_catalog[
        [
            NODE_INDEX_COLUMN,
            NODE_ID_COLUMN,
            MUNICIPALITY_ID_COLUMN
        ]
    ]
    .sort_values(NODE_INDEX_COLUMN)
    .reset_index(drop=True)
) # Preparar catálogo oficial ordenado por node_idx

expected_node_indices = np.arange(
    ANALYSIS_N_MUNICIPALITIES,
    dtype=np.int64
) # Construir espacio nodal oficial

if not np.array_equal(
    catalog_identity_sorted[
        NODE_INDEX_COLUMN
    ].to_numpy(dtype=np.int64),
    expected_node_indices
):
    raise ValueError(
        "El Catálogo Oficial de Nodos no contiene "
        "el espacio nodal completo 0 ... N-1."
    ) # Validar espacio nodal oficial

for (
    node_idx_column,
    node_id_column,
    municipality_id_column
) in identity_pairs:

    node_indices = municipality_similarity[
        node_idx_column
    ].to_numpy(
        dtype=np.int64
    ) # Obtener índices nodales de la tabla consolidada

    expected_node_ids = (
        catalog_identity_sorted
        .loc[
            node_indices,
            NODE_ID_COLUMN
        ]
        .to_numpy()
    ) # Recuperar node_id oficial mediante node_idx

    expected_municipality_ids = (
        catalog_identity_sorted
        .loc[
            node_indices,
            MUNICIPALITY_ID_COLUMN
        ]
        .to_numpy()
    ) # Recuperar municipality_id oficial mediante node_idx

    if not np.array_equal(
        expected_node_ids,
        municipality_similarity[
            node_id_column
        ].to_numpy()
    ):
        raise ValueError(
            f"La correspondencia {node_idx_column}-"
            f"{node_id_column} no es consistente."
        ) # Validar correspondencia node_idx-node_id

    if not np.array_equal(
        expected_municipality_ids,
        municipality_similarity[
            municipality_id_column
        ].to_numpy()
    ):
        raise ValueError(
            f"La correspondencia {node_idx_column}-"
            f"{municipality_id_column} no es consistente."
        ) # Validar correspondencia node_idx-municipality_id

# 11.18 Validación de Autocombinaciones
if (
    municipality_similarity["node_idx_a"]
    ==
    municipality_similarity["node_idx_b"]
).any():
    raise ValueError(
        "La tabla consolidada contiene autocombinaciones donde "
        "node_idx_a == node_idx_b."
    ) # Validar ausencia de autocombinaciones

# 11.19 Validación de Cobertura Temporal
expected_years = set(
    ANALYSIS_YEARS
) # Definir años científicos oficiales

observed_years = set(
    municipality_similarity[
        TIME_KEY
    ]
    .dropna()
    .astype(int)
) # Obtener años presentes en la tabla consolidada

if observed_years != expected_years:
    raise ValueError(
        f"La cobertura temporal no coincide con los años oficiales. "
        f"Esperados: {sorted(expected_years)}. "
        f"Encontrados: {sorted(observed_years)}."
    ) # Validar cobertura temporal completa

# 11.20 Validación Numérica General
numeric_columns = [
    column
    for column in municipality_similarity.columns
    if column not in similarity_keys
    and pd.api.types.is_numeric_dtype(
        municipality_similarity[column]
    )
] # Identificar variables numéricas

if numeric_columns:
    numeric_values = municipality_similarity[
        numeric_columns
    ].to_numpy(
        dtype=np.float64
    ) # Convertir variables numéricas a matriz

    if not np.isfinite(
        numeric_values
    ).all():
        raise ValueError(
            "La tabla consolidada contiene valores "
            "numéricos no finitos."
        ) # Validar integridad numérica

# 11.21 Validación de Similitudes Coseno
cosine_similarity_columns = [
    column
    for column in municipality_similarity.columns
    if (
        column == "similaridad_global"
        or (
            column.startswith("similaridad_")
            and column not in [
                "similaridad_grado",
                "similaridad_topologica"
            ]
        )
    )
] # Identificar similitudes basadas en distancia coseno

for similarity_column in cosine_similarity_columns:
    similarity_values = municipality_similarity[
        similarity_column
    ].to_numpy(
        dtype=np.float64
    ) # Obtener valores de similitud coseno

    if (
        similarity_values.min() < -1.0000001
        or
        similarity_values.max() > 1.0000001
    ):
        raise ValueError(
            f"La columna '{similarity_column}' contiene "
            "valores fuera del rango matemático [-1, 1]."
        ) # Validar rango de la similitud coseno

# 11.22 Validación de Similitudes Topológicas
bounded_similarity_columns = [
    column
    for column in [
        "overlap_vecinos",
        "similaridad_grado",
        "similaridad_topologica"
    ]
    if column in municipality_similarity.columns
] # Identificar métricas topológicas acotadas a [0, 1]

for similarity_column in bounded_similarity_columns:

    similarity_values = municipality_similarity[
        similarity_column
    ].to_numpy(
        dtype=np.float64
    ) # Obtener valores de similitud topológica

    if (
        similarity_values.min() < -1e-10
        or
        similarity_values.max() > 1.0000001
    ):
        raise ValueError(
            f"La columna '{similarity_column}' contiene "
            "valores fuera del rango matemático [0, 1]."
        ) # Validar rango de las métricas topológicas

# 11.23 Validación de Cantidad Final de Registros
if len(municipality_similarity) != expected_total_pairs:
    raise ValueError(
        f"La tabla consolidada contiene "
        f"{len(municipality_similarity):,} registros; "
        f"se esperaban {expected_total_pairs:,}."
    ) # Validar conservación completa del universo analítico

# 11.24 Validación de Unicidad Final
if municipality_similarity[
    similarity_keys
].duplicated().any():
    raise ValueError(
        "La tabla consolidada contiene claves "
        "tiempo-node_idx_a-node_idx_b duplicadas."
    ) # Validar unicidad final de la unidad analítica

# 11.25 Ordenamiento de la Tabla Consolidada
municipality_similarity = (
    municipality_similarity
    .sort_values(
        by=[
            TIME_KEY,
            "node_idx_a",
            "node_idx_b"
        ]
    )
    .reset_index(drop=True)
) # Ordenar tabla consolidada mediante el espacio nodal oficial

# 11.26 Exportación de la Tabla Científica
municipality_similarity.to_parquet(
    SIMILARITY_OUTPUT_PATH,
    index=False
) # Exportar tabla científica consolidada

# 11.27 Registro del Resultado
logger.info(
    "Tabla de similitud municipal generada: %s registros y %s variables analíticas.",
    len(municipality_similarity),
    len(municipality_similarity.columns)
) # Registrar resultado de la integración

logger.info(
    "Periodo consolidado: %s-%s.",
    ANALYSIS_YEARS[0],
    ANALYSIS_YEARS[-1]
) # Registrar periodo científico integrado

logger.info(
    "Claves estructurales utilizadas: %s.",
    similarity_keys
) # Registrar las claves oficiales de integración

logger.info(
    "Universo analítico validado: %s pares por año y %s pares municipio-año.",
    expected_pairs_per_year,
    expected_total_pairs
) # Registrar cobertura analítica completa

# BLOQUE 12. ANÁLISIS ESPACIAL
# Objetivo: Incorporar la proximidad geográfica y la relación administrativa a la comparación entre municipios.
# Arquitectura científica
# Entradas: Tabla de similitud municipal consolidada, Catálogo Oficial de Nodos y configuración espacial del proyecto.
# Proceso: Asociación mediante node_idx, cálculo de distancia geográfica y comparación departamental.
# Producto: Indicadores espaciales por cada par municipio-año preparados para la integración analítica.

# 12.1 Carga del Catálogo Oficial de Nodos
node_catalog = pd.read_parquet(
    NODE_CATALOG_FILE
) # Cargar el Catálogo Oficial de Nodos

if node_catalog.empty:
    raise ValueError(
        "El Catálogo Oficial de Nodos está vacío."
    ) # Validar existencia de registros en el catálogo

# 12.2 Validación de las Variables Espaciales
required_spatial_columns = [
    NODE_INDEX_COLUMN,
    NODE_ID_COLUMN,
    MUNICIPALITY_ID_COLUMN,
    MUNICIPALITY_NAME_COLUMN,
    DEPARTMENT_NAME_COLUMN,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN
] # Definir columnas requeridas para el análisis espacial

missing_spatial_columns = [
    column
    for column in required_spatial_columns
    if column not in node_catalog.columns
] # Identificar columnas espaciales, administrativas y nodales faltantes

if missing_spatial_columns:
    raise ValueError(
        f"El Catálogo Oficial de Nodos no contiene las columnas requeridas: "
        f"{missing_spatial_columns}"
    ) # Detener ejecución ante columnas faltantes

expected_node_indices = np.arange(
    ANALYSIS_N_MUNICIPALITIES,
    dtype=np.int64
) # Definir espacio nodal oficial esperado

observed_node_indices = np.sort(
    node_catalog[NODE_INDEX_COLUMN].to_numpy(dtype=np.int64)
) # Obtener espacio nodal observado en el catálogo

if not np.array_equal(
    observed_node_indices,
    expected_node_indices
):
    raise ValueError(
        "El Catálogo Oficial de Nodos no contiene exactamente "
        "el espacio nodal esperado entre 0 y "
        f"{ANALYSIS_N_MUNICIPALITIES - 1}."
    ) # Validar correspondencia completa del espacio nodal

if node_catalog[NODE_INDEX_COLUMN].duplicated().any():
    raise ValueError(
        "El Catálogo Oficial de Nodos contiene node_idx duplicados."
    ) # Validar unicidad del índice nodal

if node_catalog[NODE_ID_COLUMN].duplicated().any():
    raise ValueError(
        "El Catálogo Oficial de Nodos contiene node_id duplicados."
    ) # Validar unicidad de la identidad científica

# 12.3 Validación de Identidad y Espacio Nodal
if node_catalog[
    [
        NODE_INDEX_COLUMN,
        NODE_ID_COLUMN,
        MUNICIPALITY_ID_COLUMN
    ]
].isna().any().any():
    raise ValueError(
        "El Catálogo Oficial de Nodos contiene valores faltantes "
        "en las columnas de identidad nodal."
    ) # Validar integridad de la identidad nodal

if node_catalog[NODE_INDEX_COLUMN].duplicated().any():
    raise ValueError(
        f"El Catálogo Oficial de Nodos contiene valores duplicados "
        f"en '{NODE_INDEX_COLUMN}'."
    ) # Validar unicidad del índice nodal

if node_catalog[NODE_ID_COLUMN].duplicated().any():
    raise ValueError(
        f"El Catálogo Oficial de Nodos contiene valores duplicados "
        f"en '{NODE_ID_COLUMN}'."
    ) # Validar unicidad del identificador científico

if node_catalog[MUNICIPALITY_ID_COLUMN].duplicated().any():
    raise ValueError(
        f"El Catálogo Oficial de Nodos contiene valores duplicados "
        f"en '{MUNICIPALITY_ID_COLUMN}'."
    ) # Validar unicidad de la identidad municipal

expected_node_indices = np.arange(
    ANALYSIS_N_MUNICIPALITIES,
    dtype=np.int64
) # Construir espacio nodal oficial

catalog_node_indices = np.sort(
    node_catalog[NODE_INDEX_COLUMN].to_numpy(dtype=np.int64)
) # Obtener espacio nodal del catálogo

if not np.array_equal(
    catalog_node_indices,
    expected_node_indices
):
    raise ValueError(
        f"El Catálogo Oficial de Nodos no contiene el espacio completo "
        f"de índices 0 ... {ANALYSIS_N_MUNICIPALITIES - 1}."
    ) # Validar espacio nodal oficial

if len(node_catalog) != ANALYSIS_N_MUNICIPALITIES:
    raise ValueError(
        f"El Catálogo Oficial de Nodos contiene "
        f"{len(node_catalog):,} nodos; "
        f"se esperaban {ANALYSIS_N_MUNICIPALITIES:,}."
    ) # Validar cantidad oficial de municipios

# 12.4 Validación de Coordenadas
if node_catalog[
    [
        LATITUDE_COLUMN,
        LONGITUDE_COLUMN
    ]
].isna().any().any():
    raise ValueError(
        "El Catálogo Oficial de Nodos contiene coordenadas faltantes."
    ) # Validar disponibilidad completa de coordenadas

if not np.isfinite(
    node_catalog[
        [
            LATITUDE_COLUMN,
            LONGITUDE_COLUMN
        ]
    ].to_numpy(
        dtype=np.float64
    )
).all():
    raise ValueError(
        "El Catálogo Oficial de Nodos contiene coordenadas no finitas."
    ) # Validar integridad numérica de las coordenadas

# 12.5 Validación del Rango Geográfico
if (
    (node_catalog[LATITUDE_COLUMN] < -90)
    |
    (node_catalog[LATITUDE_COLUMN] > 90)
).any():
    raise ValueError(
        "Se encontraron latitudes fuera del rango geográfico válido."
    ) # Validar rango de latitud

if (
    (node_catalog[LONGITUDE_COLUMN] < -180)
    |
    (node_catalog[LONGITUDE_COLUMN] > 180)
).any():
    raise ValueError(
        "Se encontraron longitudes fuera del rango geográfico válido."
    ) # Validar rango de longitud

# 12.6 Construcción del Catálogo Espacial
spatial_node_catalog = node_catalog[
    required_spatial_columns
].copy() # Seleccionar información espacial, administrativa y nodal

spatial_node_catalog = (
    spatial_node_catalog
    .sort_values(
        NODE_INDEX_COLUMN
    )
    .reset_index(drop=True)
) # Ordenar el catálogo según node_idx

# 12.7 Preparación de la Información Espacial del Nodo A
spatial_a = spatial_node_catalog.rename(
    columns={
        NODE_INDEX_COLUMN: "node_idx_a",
        NODE_ID_COLUMN: "node_id_a",
        MUNICIPALITY_ID_COLUMN: "cod_municipio_a",
        MUNICIPALITY_NAME_COLUMN: "municipio_a",
        DEPARTMENT_NAME_COLUMN: "departamento_a",
        LATITUDE_COLUMN: "latitud_a",
        LONGITUDE_COLUMN: "longitud_a"
    }
) # Preparar información espacial del nodo A

# 12.8 Preparación de la Información Espacial del Nodo B
spatial_b = spatial_node_catalog.rename(
    columns={
        NODE_INDEX_COLUMN: "node_idx_b",
        NODE_ID_COLUMN: "node_id_b",
        MUNICIPALITY_ID_COLUMN: "cod_municipio_b",
        MUNICIPALITY_NAME_COLUMN: "municipio_b",
        DEPARTMENT_NAME_COLUMN: "departamento_b",
        LATITUDE_COLUMN: "latitud_b",
        LONGITUDE_COLUMN: "longitud_b"
    }
) # Preparar información espacial del nodo B

# 12.9 Preparación de los Pares Analíticos
spatial_comparison = municipality_similarity[
    [
        TIME_KEY,
        "node_idx_a",
        "node_idx_b"
    ]
].copy() # Utilizar los pares nodales consolidados del Bloque 11

if spatial_comparison.empty:
    raise ValueError(
        "La tabla municipality_similarity está vacía."
    ) # Validar disponibilidad de pares analíticos

# 12.10 Validación del Espacio Nodal de los Pares
for node_column in [
    "node_idx_a",
    "node_idx_b"
]:

    node_values = spatial_comparison[
        node_column
    ].to_numpy(
        dtype=np.int64
    ) # Obtener índices nodales de los pares

    if (
        node_values < 0
    ).any() or (
        node_values >= ANALYSIS_N_MUNICIPALITIES
    ).any():
        raise ValueError(
            f"La columna '{node_column}' contiene índices "
            f"fuera del espacio nodal oficial 0 ... "
            f"{ANALYSIS_N_MUNICIPALITIES - 1}."
        ) # Validar espacio oficial de índices

# 12.11 Validación de Autocombinaciones
if (
    spatial_comparison["node_idx_a"]
    ==
    spatial_comparison["node_idx_b"]
).any():
    raise ValueError(
        "La tabla espacial contiene autocombinaciones "
        "donde node_idx_a == node_idx_b."
    ) # Validar ausencia de autocombinaciones

# 12.12 Asociación de la Información Espacial
spatial_comparison = spatial_comparison.merge(
    spatial_a,
    on="node_idx_a",
    how="left",
    validate="many_to_one"
) # Asociar información espacial del nodo A mediante node_idx

spatial_comparison = spatial_comparison.merge(
    spatial_b,
    on="node_idx_b",
    how="left",
    validate="many_to_one"
) # Asociar información espacial del nodo B mediante node_idx

# 12.13 Validación de la Asociación Espacial
spatial_identity_columns = [
    "node_id_a",
    "node_id_b",
    "cod_municipio_a",
    "municipio_a",
    "departamento_a",
    "latitud_a",
    "longitud_a",
    "cod_municipio_b",
    "municipio_b",
    "departamento_b",
    "latitud_b",
    "longitud_b"
] # Definir información territorial cuya asociación debe estar completa

if spatial_comparison[
    spatial_identity_columns
].isna().any().any():
    missing_spatial_records = (
        spatial_comparison[
            spatial_identity_columns
        ]
        .isna()
        .any(axis=1)
        .sum()
    ) # Contabilizar pares sin información espacial completa

    raise ValueError(
        f"No fue posible asociar información espacial completa "
        f"para {missing_spatial_records:,} pares municipio-año."
    ) # Detener ejecución ante pérdida de información espacial

# 12.14 Conversión de Coordenadas a Radianes
latitude_a = np.radians(
    spatial_comparison[
        "latitud_a"
    ].to_numpy(
        dtype=np.float64
    )
) # Convertir latitudes del nodo A a radianes

longitude_a = np.radians(
    spatial_comparison[
        "longitud_a"
    ].to_numpy(
        dtype=np.float64
    )
) # Convertir longitudes del nodo A a radianes

latitude_b = np.radians(
    spatial_comparison[
        "latitud_b"
    ].to_numpy(
        dtype=np.float64
    )
) # Convertir latitudes del nodo B a radianes

longitude_b = np.radians(
    spatial_comparison[
        "longitud_b"
    ].to_numpy(
        dtype=np.float64
    )
) # Convertir longitudes del nodo B a radianes

# 12.15 Cálculo de la Distancia Geográfica
earth_radius_km = 6371.0088 # Radio medio de la Tierra utilizado para la distancia geodésica
delta_latitude = (
    latitude_b - latitude_a
) # Diferencia de latitud en radianes

delta_longitude = (
    longitude_b - longitude_a
) # Diferencia de longitud en radianes

haversine_a = (
    np.sin(
        delta_latitude / 2
    ) ** 2
    +
    np.cos(latitude_a)
    *
    np.cos(latitude_b)
    *
    np.sin(
        delta_longitude / 2
    ) ** 2
) # Calcular término Haversine

haversine_a = np.clip(
    haversine_a,
    0.0,
    1.0
) # Limitar el término para estabilidad numérica

central_angle = (
    2
    *
    np.arcsin(
        np.sqrt(
            haversine_a
        )
    )
) # Calcular ángulo central entre las coordenadas

spatial_comparison[
    "distancia_geografica_km"
] = (
    earth_radius_km
    *
    central_angle
) # Calcular distancia geodésica entre los nodos

# 12.16 Comparación Administrativa
if COMPARISON_INCLUDE_SAME_DEPARTMENT:
    spatial_comparison[
        "mismo_departamento"
    ] = (
        spatial_comparison[
            "departamento_a"
        ]
        ==
        spatial_comparison[
            "departamento_b"
        ]
    ) # Identificar si ambos municipios pertenecen al mismo departamento

# 12.17 Selección de Variables Espaciales Finales
spatial_comparison = spatial_comparison[
    [
        TIME_KEY,
        "node_idx_a",
        "node_idx_b",
        "node_id_a",
        "node_id_b",
        "cod_municipio_a",
        "municipio_a",
        "departamento_a",
        "latitud_a",
        "longitud_a",
        "cod_municipio_b",
        "municipio_b",
        "departamento_b",
        "latitud_b",
        "longitud_b",
        "distancia_geografica_km",
        "mismo_departamento"
    ]
].copy() # Seleccionar variables nodales, espaciales y administrativas finales

# 12.18 Validación de la Distancia Geográfica
distance_values = spatial_comparison[
    "distancia_geografica_km"
].to_numpy(
    dtype=np.float64
) # Obtener distancias geográficas

if not np.isfinite(
    distance_values
).all():
    raise ValueError(
        "La distancia geográfica contiene valores no finitos."
    ) # Validar integridad numérica de la distancia

if (
    distance_values < 0
).any():
    raise ValueError(
        "La distancia geográfica contiene valores negativos."
    ) # Validar rango inferior de la distancia

# 12.19 Validación de Duplicados
if spatial_comparison.duplicated(
    subset=[
        TIME_KEY,
        "node_idx_a",
        "node_idx_b"
    ]
).any():
    raise ValueError(
        "La tabla espacial contiene pares node_idx_a-node_idx_b "
        "duplicados para un mismo año."
    ) # Validar unicidad de los pares espaciales

# 12.20 Validación de Cobertura Temporal
observed_years = set(
    spatial_comparison[
        TIME_KEY
    ]
    .dropna()
    .astype(int)
) # Obtener años presentes en el análisis espacial

expected_years = set(
    ANALYSIS_YEARS
) # Definir años oficiales

if observed_years != expected_years:
    raise ValueError(
        f"La cobertura temporal no coincide con los años oficiales. "
        f"Esperados: {sorted(expected_years)}. "
        f"Encontrados: {sorted(observed_years)}."
    ) # Validar cobertura temporal completa

# 12.21 Validación de Correspondencia con municipality_similarity
spatial_pairs = set(
    map(
        tuple,
        spatial_comparison[
            [
                TIME_KEY,
                "node_idx_a",
                "node_idx_b"
            ]
        ].itertuples(
            index=False,
            name=None
        )
    )
) # Construir conjunto de pares espaciales

similarity_pairs = set(
    map(
        tuple,
        municipality_similarity[
            [
                TIME_KEY,
                "node_idx_a",
                "node_idx_b"
            ]
        ].itertuples(
            index=False,
            name=None
        )
    )
) # Construir conjunto de pares de la tabla consolidada

if spatial_pairs != similarity_pairs:
    raise ValueError(
        "La cobertura de pares node_idx_a-node_idx_b del análisis "
        "espacial no coincide con municipality_similarity."
    ) # Validar correspondencia exacta con la tabla consolidada

# 12.22 Ordenamiento de los Resultados
print(
    "Registros:",
    len(spatial_comparison)
)

print(
    "Columnas:",
    spatial_comparison.columns.tolist()
)

spatial_comparison = (
    spatial_comparison
    .sort_values(
        by=[
            TIME_KEY,
            "node_idx_a",
            "node_idx_b"
        ]
    )
    .reset_index(drop=True)
) # Ordenar resultados espaciales mediante el espacio nodal oficial

# 12.23 Exportación del Análisis Espacial
spatial_comparison.to_parquet(
    SPATIAL_OUTPUT_PATH,
    index=False
) # Exportar indicadores espaciales municipales

# 12.24 Registro del Análisis Espacial
logger.info(
    "Análisis espacial completado: %s pares municipio-año.",
    len(spatial_comparison)
) # Registrar cantidad de comparaciones espaciales

logger.info(
    "Claves espaciales utilizadas: %s.",
    [
        TIME_KEY,
        "node_idx_a",
        "node_idx_b"
    ]
) # Registrar claves estructurales oficiales del análisis espacial

print(
    "Archivo existe:",
    SPATIAL_OUTPUT_PATH.exists()
)

print(
    "Tamaño del archivo:",
    SPATIAL_OUTPUT_PATH.stat().st_size,
    "bytes"
)

spatial_export = pd.read_parquet(
    SPATIAL_OUTPUT_PATH
) # Recargar el análisis espacial exportado

print(
    "Registros exportados:",
    len(spatial_export)
)

print(
    "Primeros pares:"
)

print(
    spatial_export[
        [
            TIME_KEY,
            "node_idx_a",
            "node_idx_b"
        ]
    ].head(10)
)

# BLOQUE 13. INTEGRACIÓN FINAL Y EXPORTACIÓN DEL ANÁLISIS MUNICIPAL
# Objetivo: Construir la tabla maestra que integra la similitud municipal consolidada y la dimensión espacial.
# Arquitectura científica
# Entradas: municipality_similarity del Bloque 11 y spatial_comparison del Bloque 12.
# Proceso: Integración mediante TIME_KEY, node_idx_a y node_idx_b.
# Producto: Tabla maestra de análisis municipal municipio-año.
# Pregunta científica: ¿Qué tan similares son dos municipios considerando simultáneamente sus características
# científicas, topológicas y espaciales?

# 13.1 Preparación de la Tabla Base
municipality_analysis = municipality_similarity.copy() # Utilizar la tabla consolidada del Bloque 11 como base científica
if municipality_analysis.empty:
    raise ValueError(
        "La tabla municipality_similarity está vacía."
    ) # Validar disponibilidad de la tabla consolidada

# 13.2 Definición de las Claves de Integración
analysis_keys = [
    TIME_KEY,
    "node_idx_a",
    "node_idx_b"
] # Definir las claves estructurales oficiales del análisis

# 13.3 Validación de las Claves de Integración
missing_analysis_keys = [
    column
    for column in analysis_keys
    if column not in municipality_analysis.columns
] # Identificar claves ausentes en la tabla consolidada

if missing_analysis_keys:
    raise ValueError(
        f"municipality_similarity no contiene las claves requeridas: "
        f"{missing_analysis_keys}"
    ) # Detener ejecución ante claves incompletas

missing_spatial_keys = [
    column
    for column in analysis_keys
    if column not in spatial_comparison.columns
] # Identificar claves ausentes en la tabla espacial

if missing_spatial_keys:
    raise ValueError(
        f"spatial_comparison no contiene las claves requeridas: "
        f"{missing_spatial_keys}"
    ) # Detener ejecución ante claves espaciales incompletas

# 13.4 Validación del Espacio de Índices
for source_name, source_table in [
    ("municipality_similarity", municipality_analysis),
    ("spatial_comparison", spatial_comparison)
]:

    for node_column in [
        "node_idx_a",
        "node_idx_b"
    ]:
        node_values = source_table[
            node_column
        ].to_numpy(
            dtype=np.int64
        ) # Obtener índices nodales de la tabla

        if (
            node_values < 0
        ).any() or (
            node_values >= ANALYSIS_N_MUNICIPALITIES
        ).any():
            raise ValueError(
                f"{source_name} contiene valores fuera del espacio "
                f"nodal oficial 0 ... {ANALYSIS_N_MUNICIPALITIES - 1} "
                f"en '{node_column}'."
            ) # Validar espacio oficial de índices

# 13.5 Validación de Unicidad de la Tabla Base
if municipality_analysis.duplicated(
    subset=analysis_keys
).any():
    raise ValueError(
        "La tabla municipality_similarity contiene "
        "pares node_idx_a-node_idx_b duplicados para un mismo año."
    ) # Validar unicidad de la tabla científica consolidada

# 13.6 Validación de Unicidad de la Tabla Espacial
if spatial_comparison.duplicated(
    subset=analysis_keys
).any():
    raise ValueError(
        "La tabla spatial_comparison contiene "
        "pares node_idx_a-node_idx_b duplicados para un mismo año."
    ) # Validar unicidad de los resultados espaciales

# 13.7 Validación de Autocombinaciones
if (
    municipality_analysis["node_idx_a"]
    ==
    municipality_analysis["node_idx_b"]
).any():
    raise ValueError(
        "municipality_similarity contiene autocombinaciones "
        "donde node_idx_a == node_idx_b."
    ) # Validar ausencia de autocombinaciones

if (
    spatial_comparison["node_idx_a"]
    ==
    spatial_comparison["node_idx_b"]
).any():
    raise ValueError(
        "spatial_comparison contiene autocombinaciones "
        "donde node_idx_a == node_idx_b."
    ) # Validar ausencia de autocombinaciones espaciales

# 13.8 Validación de Correspondencia de Pares
analysis_pairs = set(
    map(
        tuple,
        municipality_analysis[
            analysis_keys
        ].itertuples(
            index=False,
            name=None
        )
    )
) # Construir conjunto de claves de la tabla consolidada

spatial_pairs = set(
    map(
        tuple,
        spatial_comparison[
            analysis_keys
        ].itertuples(
            index=False,
            name=None
        )
    )
) # Construir conjunto de claves de la tabla espacial

if analysis_pairs != spatial_pairs:
    missing_spatial_pairs = analysis_pairs - spatial_pairs
    extra_spatial_pairs = spatial_pairs - analysis_pairs

    raise ValueError(
        "La cobertura de pares del análisis espacial no coincide "
        "con municipality_similarity. "
        f"Faltantes en spatial_comparison: {len(missing_spatial_pairs):,}. "
        f"Adicionales en spatial_comparison: {len(extra_spatial_pairs):,}."
    ) # Validar correspondencia exacta entre las dos fuentes

# 13.9 Validación de la Tabla Espacial

required_spatial_columns = [
    "node_idx_a",
    "node_idx_b",
    "cod_municipio_a",
    "municipio_a",
    "departamento_a",
    "latitud_a",
    "longitud_a",
    "cod_municipio_b",
    "municipio_b",
    "departamento_b",
    "latitud_b",
    "longitud_b",
    "distancia_geografica_km",
    "mismo_departamento"
] # Definir variables espaciales y administrativas requeridas

missing_spatial_columns = [
    column
    for column in required_spatial_columns
    if column not in spatial_comparison.columns
] # Identificar variables espaciales ausentes

if missing_spatial_columns:
    raise ValueError(
        f"spatial_comparison no contiene las variables requeridas: "
        f"{missing_spatial_columns}"
    ) # Detener ejecución ante información espacial incompleta

# 13.10 Integración de la Información Espacial
spatial_merge_columns = [
    TIME_KEY
] + required_spatial_columns # Definir columnas espaciales y clave temporal

municipality_analysis = municipality_analysis.merge(
    spatial_comparison[
        spatial_merge_columns
    ],
    on=analysis_keys,
    how="inner",
    validate="one_to_one"
) # Integrar dimensión espacial mediante las claves nodales oficiales

# 13.11 Validación de la Integración Final

expected_analysis_pairs = (
    ANALYSIS_N_MUNICIPALITIES
    * (ANALYSIS_N_MUNICIPALITIES - 1)
    // 2
    * ANALYSIS_N_YEARS
) # Calcular cardinalidad científica esperada

if len(municipality_analysis) != expected_analysis_pairs:
    raise ValueError(
        f"La tabla maestra contiene "
        f"{len(municipality_analysis):,} registros; "
        f"se esperaban {expected_analysis_pairs:,}."
    ) # Validar conservación completa de los pares

if municipality_analysis.duplicated(
    subset=analysis_keys
).any():
    raise ValueError(
        "La tabla maestra contiene pares duplicados "
        "para una misma combinación año-node_idx_a-node_idx_b."
    ) # Validar unicidad de los pares finales

# 13.12 Validación de Identidad Nodal
required_identity_columns = [
    "node_idx_a",
    "node_idx_b",
    "node_id_a",
    "node_id_b",
    "cod_municipio_a",
    "cod_municipio_b"
] # Definir variables necesarias para trazabilidad territorial

missing_identity_columns = [
    column
    for column in required_identity_columns
    if column not in municipality_analysis.columns
] # Identificar columnas de identidad ausentes

if missing_identity_columns:
    raise ValueError(
        f"La tabla maestra no contiene las columnas de identidad requeridas: "
        f"{missing_identity_columns}"
    ) # Validar estructura de identidad

if municipality_analysis[
    required_identity_columns
].isna().any().any():
    raise ValueError(
        "La tabla maestra contiene información nodal o municipal faltante."
    ) # Validar integridad de la identidad territorial

# 13.13 Validación de Correspondencia Nodal
catalog_identity = node_catalog[
    [
        NODE_INDEX_COLUMN,
        NODE_ID_COLUMN,
        MUNICIPALITY_ID_COLUMN
    ]
].copy() # Seleccionar correspondencia oficial entre node_idx e identidad

if catalog_identity[NODE_INDEX_COLUMN].duplicated().any():
    raise ValueError(
        "El Catálogo Oficial de Nodos contiene node_idx duplicados."
    ) # Validar unicidad del índice nodal oficial

catalog_identity_a = catalog_identity.rename(
    columns={
        NODE_INDEX_COLUMN: "node_idx_a",
        NODE_ID_COLUMN: "node_id_oficial_a",
        MUNICIPALITY_ID_COLUMN: "cod_municipio_oficial_a"
    }
) # Preparar correspondencia oficial para el nodo A

catalog_identity_b = catalog_identity.rename(
    columns={
        NODE_INDEX_COLUMN: "node_idx_b",
        NODE_ID_COLUMN: "node_id_oficial_b",
        MUNICIPALITY_ID_COLUMN: "cod_municipio_oficial_b"
    }
) # Preparar correspondencia oficial para el nodo B

identity_validation = municipality_analysis[
    [
        "node_idx_a",
        "node_id_a",
        "cod_municipio_a",
        "node_idx_b",
        "node_id_b",
        "cod_municipio_b"
    ]
].merge(
    catalog_identity_a,
    on="node_idx_a",
    how="left",
    validate="many_to_one"
).merge(
    catalog_identity_b,
    on="node_idx_b",
    how="left",
    validate="many_to_one"
) # Recuperar identidad oficial a partir de node_idx

if identity_validation[
    [
        "node_id_oficial_a",
        "cod_municipio_oficial_a",
        "node_id_oficial_b",
        "cod_municipio_oficial_b"
    ]
].isna().any().any():
    raise ValueError(
        "Existen node_idx de la tabla maestra que no "
        "tienen correspondencia en el Catálogo Oficial de Nodos."
    ) # Validar existencia de todos los nodos en el catálogo oficial

if not (
    identity_validation["node_id_a"].to_numpy()
    ==
    identity_validation["node_id_oficial_a"].to_numpy()
).all():
    raise ValueError(
        "La correspondencia node_idx_a -> node_id_a "
        "no coincide con el Catálogo Oficial de Nodos."
    ) # Validar identidad oficial del nodo A

if not (
    identity_validation["node_id_b"].to_numpy()
    ==
    identity_validation["node_id_oficial_b"].to_numpy()
).all():
    raise ValueError(
        "La correspondencia node_idx_b -> node_id_b "
        "no coincide con el Catálogo Oficial de Nodos."
    ) # Validar identidad oficial del nodo B

if not (
    identity_validation["cod_municipio_a"].to_numpy()
    ==
    identity_validation["cod_municipio_oficial_a"].to_numpy()
).all():
    raise ValueError(
        "La correspondencia node_idx_a -> cod_municipio_a "
        "no coincide con el Catálogo Oficial de Nodos."
    ) # Validar identidad municipal oficial del nodo A

if not (
    identity_validation["cod_municipio_b"].to_numpy()
    ==
    identity_validation["cod_municipio_oficial_b"].to_numpy()
).all():
    raise ValueError(
        "La correspondencia node_idx_b -> cod_municipio_b "
        "no coincide con el Catálogo Oficial de Nodos."
    ) # Validar identidad municipal oficial del nodo B

# 13.14 Validación de Cobertura Temporal
observed_years = set(
    municipality_analysis[
        TIME_KEY
    ].dropna().astype(int)
) # Obtener años presentes en la tabla maestra

expected_years = set(
    ANALYSIS_YEARS
) # Definir años oficiales

if observed_years != expected_years:
    raise ValueError(
        f"La cobertura temporal no coincide con los años oficiales. "
        f"Esperados: {sorted(expected_years)}. "
        f"Encontrados: {sorted(observed_years)}."
    ) # Validar cobertura temporal completa

# 13.15 Validación de Información Espacial
spatial_identity_columns = [
    "node_id_a",
    "node_id_b",
    "cod_municipio_a",
    "municipio_a",
    "departamento_a",
    "latitud_a",
    "longitud_a",
    "cod_municipio_b",
    "municipio_b",
    "departamento_b",
    "latitud_b",
    "longitud_b"
] # Definir variables espaciales que no pueden faltar

if municipality_analysis[
    spatial_identity_columns
].isna().any().any():
    raise ValueError(
        "La tabla maestra contiene registros con información "
        "espacial o administrativa faltante."
    ) # Validar integridad espacial completa

if municipality_analysis[
    "distancia_geografica_km"
].isna().any():
    raise ValueError(
        "La tabla maestra contiene pares sin distancia geográfica."
    ) # Validar cobertura completa de distancia geográfica

# 13.16 Validación de la Distancia Geográfica
distance_values = municipality_analysis[
    "distancia_geografica_km"
].to_numpy(
    dtype=np.float64
) # Obtener distancias geográficas

if not np.isfinite(
    distance_values
).all():
    raise ValueError(
        "La tabla maestra contiene distancias geográficas no finitas."
    ) # Validar integridad numérica de la distancia

if (
    distance_values < 0
).any():
    raise ValueError(
        "La tabla maestra contiene distancias geográficas negativas."
    ) # Validar rango de la distancia geográfica

# 13.17 Validación de la Relación Departamental
if municipality_analysis[
    "mismo_departamento"
].isna().any():
    raise ValueError(
        "La tabla maestra contiene valores faltantes "
        "en mismo_departamento."
    ) # Validar clasificación administrativa completa

if not municipality_analysis[
    "mismo_departamento"
].isin([True, False]).all():
    raise ValueError(
        "La columna mismo_departamento contiene valores "
        "distintos de True y False."
    ) # Validar dominio lógico de la relación departamental

municipality_analysis[
    "mismo_departamento"
] = municipality_analysis[
    "mismo_departamento"
].astype(bool) # Garantizar tipo booleano para la relación departamental

# 13.18 Validación de Variables Numéricas
numeric_columns = municipality_analysis.select_dtypes(
    include=[np.number]
).columns.tolist() # Identificar variables numéricas de la tabla maestra

if numeric_columns:
    if not np.isfinite(
        municipality_analysis[
            numeric_columns
        ].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise ValueError(
            "La tabla maestra contiene valores numéricos no finitos."
        ) # Validar integridad numérica de todas las variables

# 13.19 Validación de Similitudes
similarity_columns = [
    column
    for column in municipality_analysis.columns
    if column.startswith(
        "similaridad_"
    )
] # Identificar todas las métricas de similitud integradas

if not similarity_columns:
    raise ValueError(
        "La tabla maestra no contiene ninguna variable de similitud."
    ) # Validar existencia de métricas de similitud

cosine_similarity_columns = [
    "similaridad_global"
] + [
    f"similaridad_{domain_name}"
    for domain_name in ANALYSIS_FEATURE_GROUPS
] # Definir explícitamente las similitudes basadas en coseno

missing_cosine_similarity_columns = [
    column
    for column in cosine_similarity_columns
    if column not in municipality_analysis.columns
] # Identificar similitudes coseno ausentes

if missing_cosine_similarity_columns:
    raise ValueError(
        "Faltan columnas de similitud coseno en la tabla maestra: "
        f"{missing_cosine_similarity_columns}"
    ) # Validar estructura completa de similitudes coseno

for similarity_column in cosine_similarity_columns:
    similarity_values = municipality_analysis[
        similarity_column
    ].to_numpy(
        dtype=np.float64
    ) # Obtener valores de similitud coseno

    if not np.isfinite(
        similarity_values
    ).all():
        raise ValueError(
            f"La columna '{similarity_column}' contiene "
            "valores no finitos."
        ) # Validar integridad numérica de la similitud coseno

    if (
        similarity_values.min() < -1.0000001
        or
        similarity_values.max() > 1.0000001
    ):
        raise ValueError(
            f"La columna '{similarity_column}' contiene "
            "valores fuera del intervalo matemático [-1, 1]."
        ) # Validar rango matemático de similitud coseno

# 13.20 Ordenamiento Científico
municipality_analysis = (
    municipality_analysis
    .sort_values(
        by=[
            TIME_KEY,
            "node_idx_a",
            "node_idx_b"
        ]
    )
    .reset_index(drop=True)
) # Ordenar la tabla maestra mediante el espacio nodal oficial

# 13.21 Exportación de la Tabla Maestra
municipality_analysis_output = (
    OUTPUTS_DIR
    / "municipality_analysis_complete.parquet"
) # Definir archivo maestro oficial del análisis municipal

municipality_analysis.to_parquet(
    municipality_analysis_output,
    index=False
) # Exportar tabla maestra en formato Parquet

# 13.22 Exportación de una Versión CSV
municipality_analysis_csv = (
    OUTPUTS_DIR
    / "municipality_analysis_complete.csv"
) # Definir versión interoperable de la tabla maestra

municipality_analysis.to_csv(
    municipality_analysis_csv,
    index=False,
    encoding="utf-8"
) # Exportar tabla maestra en formato CSV

# 13.23 Resumen del Producto Analítico
n_analysis_pairs = len(
    municipality_analysis
) # Calcular número total de pares municipio-año analizados

n_analysis_variables = len(
    municipality_analysis.columns
) # Calcular número total de variables de la tabla maestra

n_same_department = int(
    municipality_analysis[
        "mismo_departamento"
    ].sum()
) # Calcular número de pares pertenecientes al mismo departamento

mean_distance_km = float(
    municipality_analysis[
        "distancia_geografica_km"
    ].mean()
) # Calcular distancia geográfica promedio entre pares

# 13.24 Validación de Cardinalidad y Registro del Resultado
n_expected_pairs = (
    ANALYSIS_N_MUNICIPALITIES
    * (ANALYSIS_N_MUNICIPALITIES - 1)
    // 2
    * ANALYSIS_N_YEARS
) # Calcular cardinalidad científica esperada

if n_analysis_pairs != n_expected_pairs:
    raise ValueError(
        f"La tabla maestra contiene {n_analysis_pairs:,} registros, "
        f"pero se esperaban {n_expected_pairs:,}."
    ) # Validar cardinalidad final del producto científico

logger.info(
    "Tabla maestra municipal generada: %s pares municipio-año | "
    "%s variables | %s pares del mismo departamento | "
    "distancia media %.2f km.",
    n_analysis_pairs,
    n_analysis_variables,
    n_same_department,
    mean_distance_km
) # Registrar el resultado final del análisis municipal

logger.info(
    "Claves estructurales de integración: %s.",
    analysis_keys
) # Registrar las claves oficiales utilizadas para integrar los bloques

logger.info(
    "Estado de integración final: VALIDADO."
) # Confirmar validación completa de la tabla maestra

# BLOQUE 14. GENERACIÓN DE TABLAS ANALÍTICAS
# Objetivo: Generar tablas agregadas y rankings a partir de la tabla maestra de análisis municipal validada.
# Arquitectura científica
# Entradas: municipality_analysis_complete del Bloque 13.
# Proceso: Agregación municipal, ranking de similitud, análisis temporal, análisis por dominio y relaciones
# espacial, topológica y departamental.
# Producto: Tablas analíticas para interpretación científica, validación y visualización.
# Pregunta científica: ¿Cómo se distribuyen la similitud municipal, la estructura topológica y la relación espacial durante el periodo científico?

# 14.1 Validación de la Tabla Maestra
if municipality_analysis.empty:
    raise ValueError(
        "La tabla maestra de análisis municipal está vacía."
    ) # Validar disponibilidad de la tabla maestra

# 14.2 Definición de las Claves Analíticas
analysis_keys = [
    TIME_KEY,
    "node_idx_a",
    "node_idx_b"
] # Definir las claves estructurales oficiales del análisis

# 14.3 Validación de las Claves Analíticas
missing_analysis_keys = [
    column
    for column in analysis_keys
    if column not in municipality_analysis.columns
] # Identificar claves estructurales ausentes

if missing_analysis_keys:
    raise ValueError(
        f"La tabla maestra no contiene las claves requeridas: "
        f"{missing_analysis_keys}"
    ) # Detener ejecución ante claves incompletas

# 14.4 Validación del Espacio de Índices
for node_column in [
    "node_idx_a",
    "node_idx_b"
]:

    node_values = municipality_analysis[
        node_column
    ].to_numpy(
        dtype=np.int64
    ) # Obtener índices nodales

    if (
        node_values < 0
    ).any() or (
        node_values >= ANALYSIS_N_MUNICIPALITIES
    ).any():
        raise ValueError(
            f"La columna '{node_column}' contiene índices fuera "
            f"del espacio nodal oficial 0 ... "
            f"{ANALYSIS_N_MUNICIPALITIES - 1}."
        ) # Validar espacio oficial de índices nodales

# 14.5 Validación de Unicidad de los Pares
if municipality_analysis.duplicated(
    subset=analysis_keys
).any():
    raise ValueError(
        "La tabla maestra contiene pares node_idx_a-node_idx_b "
        "duplicados para un mismo año."
    ) # Validar unicidad de las unidades analíticas

# 14.6 Validación de Autocombinaciones
if (
    municipality_analysis["node_idx_a"]
    ==
    municipality_analysis["node_idx_b"]
).any():
    raise ValueError(
        "La tabla maestra contiene autocombinaciones donde "
        "node_idx_a == node_idx_b."
    ) # Validar ausencia de comparaciones de un nodo consigo mismo

# 14.7 Validación de Cobertura Temporal
observed_years = set(
    municipality_analysis[
        TIME_KEY
    ].dropna().astype(int)
) # Obtener años presentes en la tabla maestra

expected_years = set(
    ANALYSIS_YEARS
) # Obtener años oficiales del análisis

if observed_years != expected_years:
    raise ValueError(
        f"La tabla maestra no contiene exactamente los años oficiales. "
        f"Esperados: {sorted(expected_years)}. "
        f"Encontrados: {sorted(observed_years)}."
    ) # Validar cobertura temporal

# 14.8 Validación de la Similitud Global
if "similaridad_global" not in municipality_analysis.columns:
    raise ValueError(
        "La tabla maestra no contiene la similitud global."
    ) # Validar existencia de la métrica principal

global_similarity = municipality_analysis[
    "similaridad_global"
].to_numpy(
    dtype=np.float64
) # Obtener valores de similitud global

if not np.isfinite(
    global_similarity
).all():
    raise ValueError(
        "La similitud global contiene valores no finitos."
    ) # Validar integridad numérica de la similitud global

if (
    global_similarity.min() < -1.0000001
    or
    global_similarity.max() > 1.0000001
):
    raise ValueError(
        "La similitud global contiene valores "
        "fuera del intervalo matemático [-1, 1]."
    ) # Validar rango matemático de la similitud coseno

# 14.9 Definición de Variables de Similitud por Dominio
domain_similarity_columns = [
    f"similaridad_{domain_name}"
    for domain_name in ANALYSIS_FEATURE_GROUPS
    if f"similaridad_{domain_name}" in municipality_analysis.columns
] # Identificar similitudes disponibles por dominio científico

if not domain_similarity_columns:
    raise ValueError(
        "La tabla maestra no contiene variables de similitud por dominio."
    ) # Validar existencia de similitudes por dominio

# 14.10 Validación de Configuración del Ranking
if SIMILARITY_TOP_K < 1:
    raise ValueError(
        f"SIMILARITY_TOP_K debe ser mayor o igual a 1: "
        f"{SIMILARITY_TOP_K}"
    ) # Validar configuración mínima del ranking

if SIMILARITY_TOP_K > len(
    municipality_analysis
):
    raise ValueError(
        f"SIMILARITY_TOP_K={SIMILARITY_TOP_K:,} supera "
        f"el número disponible de pares: "
        f"{len(municipality_analysis):,}."
    ) # Validar que el ranking no exceda los pares disponibles

# 14.11 Definición de Variables para el Ranking de Pares
pair_columns = [
    TIME_KEY,
    "node_idx_a",
    "node_idx_b",
    "node_id_a",
    "node_id_b",
    "cod_municipio_a",
    "cod_municipio_b",
    "similaridad_global",
    "distancia_geografica_km"
] # Definir variables necesarias para la representación de pares

if "vecinos_compartidos" in municipality_analysis.columns:
    pair_columns.append(
        "vecinos_compartidos"
    ) # Incorporar información topológica cuando esté disponible

missing_pair_columns = [
    column
    for column in pair_columns
    if column not in municipality_analysis.columns
] # Identificar variables requeridas ausentes

if missing_pair_columns:
    raise ValueError(
        "La tabla maestra no contiene las variables requeridas "
        f"para el ranking: {missing_pair_columns}"
    ) # Validar estructura necesaria para el ranking

municipality_a_view = municipality_analysis[
    pair_columns
].copy() # Construir perspectiva del nodo A

municipality_a_view = municipality_a_view.rename(
    columns={
        "node_idx_a": "node_idx",
        "node_idx_b": "comparison_node_idx",
        "node_id_a": "node_id",
        "node_id_b": "comparison_node_id",
        "cod_municipio_a": "cod_municipio",
        "cod_municipio_b": "comparison_cod_municipio"
    }
) # Definir el nodo A como referencia analítica

municipality_b_view = municipality_analysis[
    pair_columns
].copy() # Construir perspectiva del nodo B

municipality_b_view = municipality_b_view.rename(
    columns={
        "node_idx_a": "comparison_node_idx",
        "node_idx_b": "node_idx",
        "node_id_a": "comparison_node_id",
        "node_id_b": "node_id",
        "cod_municipio_a": "comparison_cod_municipio",
        "cod_municipio_b": "cod_municipio"
    }
) # Definir el nodo B como referencia analítica

municipality_pair_view = pd.concat(
    [
        municipality_a_view,
        municipality_b_view
    ],
    ignore_index=True
) # Construir representación simétrica de las relaciones

# 14.12 Resumen de Comportamiento Municipal
municipality_summary = municipality_pair_view.groupby(
    [
        TIME_KEY,
        "node_idx",
        "node_id",
        "cod_municipio"
    ],
    as_index=False
).agg(
    **municipality_aggregation
) # Resumir comportamiento de cada nodo por año

# 14.13 Selección de los Pares con Mayor Similitud
top_similar_pairs = municipality_analysis.nlargest(
    SIMILARITY_TOP_K,
    "similaridad_global"
).copy() # Seleccionar los pares con mayor similitud global

top_similar_pairs["ranking"] = np.arange(
    1,
    len(top_similar_pairs) + 1
) # Asignar posición del ranking

# 14.14 Validación del Ranking de Similitud
if len(top_similar_pairs) != SIMILARITY_TOP_K:
    raise ValueError(
        f"El ranking contiene {len(top_similar_pairs):,} pares; "
        f"se esperaban {SIMILARITY_TOP_K:,}."
    ) # Validar cantidad de pares seleccionados

if top_similar_pairs[
    "ranking"
].duplicated().any():
    raise ValueError(
        "El ranking contiene posiciones duplicadas."
    ) # Validar unicidad de las posiciones del ranking

if not top_similar_pairs[
    "ranking"
].equals(
    pd.Series(
        np.arange(
            1,
            SIMILARITY_TOP_K + 1
        ),
        index=top_similar_pairs.index,
        name="ranking"
    )
):
    raise ValueError(
        "Las posiciones del ranking no son consecutivas "
        "desde 1 hasta SIMILARITY_TOP_K."
    ) # Validar continuidad del ranking

# 14.15 Tabla de Pares Más Diferentes
top_different_pairs = municipality_analysis.nsmallest(
    SIMILARITY_TOP_K,
    "similaridad_global"
).copy() # Seleccionar los pares con menor similitud global

top_different_pairs["ranking"] = np.arange(
    1,
    len(top_different_pairs) + 1
) # Asignar posición del ranking

if len(top_different_pairs) != SIMILARITY_TOP_K:
    raise ValueError(
        f"La tabla de pares más diferentes contiene "
        f"{len(top_different_pairs):,} registros; "
        f"se esperaban {SIMILARITY_TOP_K:,}."
    ) # Validar cantidad de pares diferentes seleccionados

if not np.array_equal(
    top_different_pairs["ranking"].to_numpy(dtype=np.int64),
    np.arange(
        1,
        SIMILARITY_TOP_K + 1,
        dtype=np.int64
    )
):
    raise ValueError(
        "El ranking de pares más diferentes no es consecutivo."
    ) # Validar continuidad del ranking

# 14.16 Tabla de Similitud Promedio por Año
temporal_similarity = municipality_analysis.groupby(
    TIME_KEY,
    as_index=False
).agg(
    similaridad_global_promedio=(
        "similaridad_global",
        "mean"
    ),
    similaridad_global_sd=(
        "similaridad_global",
        "std"
    ),
    distancia_geografica_promedio=(
        "distancia_geografica_km",
        "mean"
    ),
    distancia_geografica_sd=(
        "distancia_geografica_km",
        "std"
    )
) # Calcular evolución temporal de la similitud municipal

# 14.17 Tabla de Similitud por Dominio
domain_summary = municipality_analysis[
    [TIME_KEY] + domain_similarity_columns
].groupby(
    TIME_KEY,
    as_index=False
).mean(
    numeric_only=True
) # Calcular similitud promedio por dominio y año

# 14.18 Tabla de Relación entre Similitud y Distancia Geográfica
spatial_similarity_summary = municipality_analysis[
    [
        "similaridad_global",
        "distancia_geografica_km"
    ]
].corr(
    method="pearson"
) # Calcular correlación exploratoria entre similitud y distancia geográfica

# 14.19 Tabla de Relación entre Similitud Global y Topología
if (
    "similaridad_topologica" in municipality_analysis.columns
    and
    "overlap_vecinos" in municipality_analysis.columns
):
    topology_similarity_summary = municipality_analysis[
        [
            "similaridad_global",
            "similaridad_topologica",
            "overlap_vecinos"
        ]
    ].corr(
        method="pearson"
    ) # Calcular relación exploratoria entre similitud científica y topología
else:
    topology_similarity_summary = pd.DataFrame() # Crear tabla vacía si no existe información topológica

# 14.20 Tabla de Comparación Departamental
department_similarity_summary = municipality_analysis.groupby(
    "mismo_departamento",
    as_index=False
).agg(
    n_pares=(
        "node_idx_a",
        "count"
    ),
    similaridad_global_promedio=(
        "similaridad_global",
        "mean"
    ),
    similaridad_global_sd=(
        "similaridad_global",
        "std"
    ),
    distancia_geografica_promedio=(
        "distancia_geografica_km",
        "mean"
    )
) # Comparar similitud y distancia según pertenencia departamental

# 14.21 Validación de Resultados Analíticos
if municipality_summary.empty:
    raise ValueError(
        "La tabla de resumen municipal está vacía."
    ) # Validar resumen municipal

if temporal_similarity.empty:
    raise ValueError(
        "La tabla de análisis temporal está vacía."
    ) # Validar análisis temporal

if domain_summary.empty:
    raise ValueError(
        "La tabla de similitud por dominio está vacía."
    ) # Validar análisis por dominio

if spatial_similarity_summary.empty:
    raise ValueError(
        "La tabla de relación espacial está vacía."
    ) # Validar análisis espacial

if department_similarity_summary.empty:
    raise ValueError(
        "La tabla de comparación departamental está vacía."
    ) # Validar comparación departamental

if len(top_similar_pairs) != SIMILARITY_TOP_K:
    raise ValueError(
        f"El ranking de pares más similares contiene "
        f"{len(top_similar_pairs):,} registros; "
        f"se esperaban {SIMILARITY_TOP_K:,}."
    ) # Validar ranking de pares más similares

if len(top_different_pairs) != SIMILARITY_TOP_K:
    raise ValueError(
        f"El ranking de pares más diferentes contiene "
        f"{len(top_different_pairs):,} registros; "
        f"se esperaban {SIMILARITY_TOP_K:,}."
    ) # Validar ranking de pares más diferentes

# 14.22 Exportación del Resumen Municipal
municipality_summary.to_parquet(
    OUTPUTS_DIR / "municipality_summary.parquet",
    index=False
) # Exportar resumen municipal

# 14.23 Exportación de Pares Más Similares
top_similar_pairs.to_parquet(
    OUTPUTS_DIR / "top_similar_municipal_pairs.parquet",
    index=False
) # Exportar ranking de pares más similares

# 14.24 Exportación de Pares Más Diferentes
top_different_pairs.to_parquet(
    OUTPUTS_DIR / "top_different_municipal_pairs.parquet",
    index=False
) # Exportar ranking de pares más diferentes

# 14.25 Exportación del Análisis Temporal
temporal_similarity.to_parquet(
    OUTPUTS_DIR / "temporal_similarity_summary.parquet",
    index=False
) # Exportar evolución temporal de la similitud

# 14.26 Exportación de Similitud por Dominio
domain_summary.to_parquet(
    OUTPUTS_DIR / "domain_similarity_summary.parquet",
    index=False
) # Exportar resumen de similitud por dominio

# 14.27 Exportación de Relación Espacial

spatial_similarity_summary.to_csv(
    OUTPUTS_DIR / "spatial_similarity_correlation.csv"
) # Exportar correlación exploratoria entre similitud y distancia geográfica

# 14.28 Exportación de Relación Topológica

if not topology_similarity_summary.empty:

    topology_similarity_summary.to_csv(
        OUTPUTS_DIR / "topology_similarity_correlation.csv"
    ) # Exportar correlación exploratoria entre similitud científica y topología

# 14.29 Exportación de Comparación Departamental

department_similarity_summary.to_parquet(
    OUTPUTS_DIR / "department_similarity_summary.parquet",
    index=False
) # Exportar comparación departamental

# 14.30 Resumen de Productos Generados

n_analysis_pairs = len(
    municipality_analysis
) # Número total de pares municipio-año analizados

n_analysis_variables = len(
    municipality_analysis.columns
) # Número total de variables de la tabla maestra

n_same_department = int(
    municipality_analysis[
        "mismo_departamento"
    ].sum()
) # Número de pares pertenecientes al mismo departamento

mean_distance_km = float(
    municipality_analysis[
        "distancia_geografica_km"
    ].mean()
) # Distancia geográfica promedio entre pares analizados

n_expected_pairs = (
    ANALYSIS_N_MUNICIPALITIES
    * (ANALYSIS_N_MUNICIPALITIES - 1)
    // 2
    * ANALYSIS_N_YEARS
) # Calcular cardinalidad científica esperada

# 14.31 Validación Final y Registro del Resultado

if n_analysis_pairs != n_expected_pairs:
    raise ValueError(
        f"La tabla maestra contiene "
        f"{n_analysis_pairs:,} pares municipio-año; "
        f"se esperaban {n_expected_pairs:,}."
    ) # Validar cardinalidad completa del análisis

if n_analysis_pairs != (
    ANALYSIS_N_MUNICIPALITIES
    * (ANALYSIS_N_MUNICIPALITIES - 1)
    // 2
    * ANALYSIS_N_YEARS
):
    raise ValueError(
        "La cardinalidad final del análisis no coincide "
        "con el universo científico esperado."
    ) # Validar consistencia matemática del universo de pares

if n_analysis_variables <= 0:
    raise ValueError(
        "La tabla maestra no contiene variables analíticas."
    ) # Validar existencia de variables analíticas

logger.info(
    "Tablas analíticas generadas correctamente: "
    "resumen municipal, pares similares, pares diferentes, "
    "evolución temporal, dominios, análisis espacial "
    "y análisis departamental."
) # Registrar productos analíticos generados

logger.info(
    "Claves estructurales utilizadas: %s.",
    analysis_keys
) # Registrar claves oficiales del análisis

logger.info(
    "Resumen analítico: %s pares municipio-año | %s variables | "
    "%s pares del mismo departamento | distancia media %.2f km.",
    n_analysis_pairs,
    n_analysis_variables,
    n_same_department,
    mean_distance_km
) # Registrar indicadores generales del producto analítico

logger.info(
    "Estado del Bloque 14: VALIDADO."
) # Confirmar generación correcta de las tablas analíticas

# BLOQUE 15. ANÁLISIS DE ESTABILIDAD TEMPORAL DE LA SIMILITUD
# Objetivo: Evaluar la estabilidad, evolución y persistencia de la similitud entre los mismos pares nodales durante el período científico.
# Arquitectura científica
# Entradas: municipality_analysis y configuración temporal oficial del proyecto.
# Proceso: Seguimiento longitudinal de pares nodales, cálculo de similitud inicial, final, promedio, variabilidad y cambio temporal.
# Producto: Tablas de estabilidad temporal de las relaciones nodales.
# Pregunta científica: ¿Las relaciones de similitud entre los mismos nodos permanecen estables durante el período científico?

# 15.1 Validación de la Información Temporal
if municipality_analysis.empty:
    raise ValueError(
        "La tabla maestra de análisis municipal está vacía."
    ) # Validar disponibilidad de la tabla maestra

required_temporal_columns = [
    TIME_KEY,
    "node_idx_a",
    "node_idx_b",
    "similaridad_global"
] # Definir columnas mínimas para el análisis temporal

missing_temporal_columns = [
    column
    for column in required_temporal_columns
    if column not in municipality_analysis.columns
] # Identificar columnas temporales faltantes

if missing_temporal_columns:
    raise ValueError(
        f"La tabla maestra no contiene las columnas requeridas "
        f"para el análisis temporal: {missing_temporal_columns}."
    ) # Detener ejecución ante estructura incompleta

# 15.2 Validación de la Configuración Temporal
if ANALYSIS_N_YEARS < 2:
    raise ValueError(
        f"Se requieren al menos dos años para analizar estabilidad temporal: "
        f"{ANALYSIS_N_YEARS}."
    ) # Validar existencia de intervalos temporales

if len(ANALYSIS_YEARS) != ANALYSIS_N_YEARS:
    raise ValueError(
        f"ANALYSIS_YEARS contiene {len(ANALYSIS_YEARS):,} años; "
        f"se esperaban {ANALYSIS_N_YEARS:,}."
    ) # Validar coherencia temporal

expected_years = sorted(
    ANALYSIS_YEARS
) # Obtener años oficiales del análisis

# 15.3 Definición de las Claves del Par Nodal
temporal_pair_keys = [
    "node_idx_a",
    "node_idx_b"
] # Definir identidad estructural del par nodal

temporal_keys = [
    TIME_KEY,
    *temporal_pair_keys
] # Definir unidad analítica nodal-año

# 15.4 Validación del Espacio Nodal
for node_column in temporal_pair_keys:
    node_values = municipality_analysis[
        node_column
    ].to_numpy(
        dtype=np.int64
    ) # Obtener índices nodales de la tabla maestra

    if (
        node_values < 0
    ).any() or (
        node_values >= ANALYSIS_N_MUNICIPALITIES
    ).any():
        raise ValueError(
            f"La columna '{node_column}' contiene índices fuera "
            f"del espacio nodal oficial 0 ... "
            f"{ANALYSIS_N_MUNICIPALITIES - 1}."
        ) # Validar espacio oficial de índices

# 15.5 Validación de Autocombinaciones
if (
    municipality_analysis["node_idx_a"]
    ==
    municipality_analysis["node_idx_b"]
).any():
    raise ValueError(
        "La tabla maestra contiene autocombinaciones donde "
        "node_idx_a == node_idx_b."
    ) # Validar ausencia de autocombinaciones

# 15.6 Validación de Unicidad Nodo Año
if municipality_analysis.duplicated(
    subset=temporal_keys
).any():
    raise ValueError(
        "La tabla maestra contiene pares node_idx-año duplicados."
    ) # Validar unicidad de cada observación temporal

# 15.7 Validación de Cobertura Temporal
available_years = sorted(
    municipality_analysis[
        TIME_KEY
    ].dropna().astype(int).unique().tolist()
) # Obtener años presentes en la tabla maestra

if available_years != expected_years:
    raise ValueError(
        f"Los años disponibles {available_years} "
        f"no coinciden con los años oficiales {expected_years}."
    ) # Validar cobertura temporal exacta

# 15.8 Ordenamiento Temporal
temporal_analysis = (
    municipality_analysis
    .sort_values(
        by=temporal_pair_keys + [TIME_KEY]
    )
    .reset_index(drop=True)
) # Ordenar cada relación nodal cronológicamente

# 15.9 Resumen Temporal de Cada Par
pair_temporal_summary = (
    temporal_analysis
    .groupby(
        temporal_pair_keys,
        as_index=False
    )
    .agg(
        n_years_observed=(
            TIME_KEY,
            "nunique"
        ),
        year_first=(
            TIME_KEY,
            "min"
        ),
        year_last=(
            TIME_KEY,
            "max"
        ),
        similaridad_global_promedio=(
            "similaridad_global",
            "mean"
        ),
        similaridad_global_sd=(
            "similaridad_global",
            "std"
        ),
        similaridad_global_min=(
            "similaridad_global",
            "min"
        ),
        similaridad_global_max=(
            "similaridad_global",
            "max"
        )
    )
) # Resumir trayectoria temporal de cada par nodal

# 15.10 Validación de Cobertura Temporal por Par
incomplete_pairs = pair_temporal_summary[
    pair_temporal_summary[
        "n_years_observed"
    ] != ANALYSIS_N_YEARS
] # Identificar pares sin cobertura temporal completa

if not incomplete_pairs.empty:
    raise ValueError(
        f"Se identificaron {len(incomplete_pairs):,} pares nodales "
        "con cobertura temporal incompleta."
    ) # Exigir panel longitudinal balanceado

if not (
    pair_temporal_summary[
        "year_first"
    ] == ANALYSIS_YEARS[0]
).all():
    raise ValueError(
        "Existen pares cuyo primer año observado no coincide "
        "con el primer año oficial del análisis."
    ) # Validar inicio temporal

if not (
    pair_temporal_summary[
        "year_last"
    ] == ANALYSIS_YEARS[-1]
).all():
    raise ValueError(
        "Existen pares cuyo último año observado no coincide "
        "con el último año oficial del análisis."
    ) # Validar final temporal

# 15.11 Similitud Inicial
initial_similarity = (
    temporal_analysis[
        temporal_analysis[TIME_KEY] == ANALYSIS_YEARS[0]
    ][
        temporal_pair_keys + ["similaridad_global"]
    ]
    .rename(
        columns={
            "similaridad_global": "similaridad_inicial"
        }
    )
) # Obtener similitud del primer año

# 15.12 Similitud Final
final_similarity = (
    temporal_analysis[
        temporal_analysis[TIME_KEY] == ANALYSIS_YEARS[-1]
    ][
        temporal_pair_keys + ["similaridad_global"]
    ]
    .rename(
        columns={
            "similaridad_global": "similaridad_final"
        }
    )
) # Obtener similitud del último año

# 15.13 Integración de Similitud Inicial y Final
pair_temporal_summary = pair_temporal_summary.merge(
    initial_similarity,
    on=temporal_pair_keys,
    how="left",
    validate="one_to_one"
) # Incorporar similitud inicial

pair_temporal_summary = pair_temporal_summary.merge(
    final_similarity,
    on=temporal_pair_keys,
    how="left",
    validate="one_to_one"
) # Incorporar similitud final

# 15.14 Validación de Similitud Inicial y Final
if pair_temporal_summary[
    [
        "similaridad_inicial",
        "similaridad_final"
    ]
].isna().any().any():
    raise ValueError(
        "Existen pares nodales sin similitud inicial o final."
    ) # Validar cobertura de los extremos temporales

# 15.15 Cambio de Similitud
pair_temporal_summary[
    "cambio_similitud"
] = (
    pair_temporal_summary[
        "similaridad_final"
    ]
    -
    pair_temporal_summary[
        "similaridad_inicial"
    ]
) # Calcular cambio neto entre inicio y final

pair_temporal_summary[
    "cambio_similitud_absoluto"
] = (
    pair_temporal_summary[
        "cambio_similitud"
    ].abs()
) # Calcular magnitud absoluta del cambio

# 15.16 Tasa Media de Cambio Temporal
n_temporal_intervals = (
    ANALYSIS_N_YEARS - 1
) # Calcular número de intervalos temporales

pair_temporal_summary[
    "cambio_anual_promedio"
] = (
    pair_temporal_summary[
        "cambio_similitud"
    ]
    /
    n_temporal_intervals
) # Calcular cambio medio por intervalo temporal

# 15.17 Variabilidad Temporal
pair_temporal_summary[
    "variabilidad_temporal"
] = (
    pair_temporal_summary[
        "similaridad_global_sd"
    ]
) # Utilizar desviación estándar temporal como medida de variabilidad

# 15.18 Validación de Umbrales de Estabilidad
if not (
    0
    <
    STABILITY_THRESHOLD_VERY_STABLE
    <
    STABILITY_THRESHOLD_STABLE
    <
    STABILITY_THRESHOLD_MODERATE
):
    raise ValueError(
        "Los umbrales de estabilidad temporal deben cumplir "
        "0 < VERY_STABLE < STABLE < MODERATE."
    ) # Validar orden de los umbrales científicos

# 15.19 Clasificación de Estabilidad Temporal
pair_temporal_summary[
    "estabilidad_temporal"
] = pd.cut(
    pair_temporal_summary[
        "cambio_similitud_absoluto"
    ],
    bins=[
        -np.inf,
        STABILITY_THRESHOLD_VERY_STABLE,
        STABILITY_THRESHOLD_STABLE,
        STABILITY_THRESHOLD_MODERATE,
        np.inf
    ],
    labels=[
        "muy_estable",
        "estable",
        "moderadamente_inestable",
        "inestable"
    ],
    include_lowest=True
) # Clasificar estabilidad según magnitud del cambio entre extremos

# 15.20 Clasificación de la Dirección del Cambio
TEMPORAL_CHANGE_TOLERANCE = 1e-10 # Definir tolerancia numérica para cambios prácticamente nulos
pair_temporal_summary[
    "direccion_cambio"
] = np.select(
    [
        pair_temporal_summary[
            "cambio_similitud"
        ] > TEMPORAL_CHANGE_TOLERANCE,
        pair_temporal_summary[
            "cambio_similitud"
        ] < -TEMPORAL_CHANGE_TOLERANCE
    ],
    [
        "aumento",
        "disminucion"
    ],
    default="sin_cambio"
) # Clasificar dirección de la evolución temporal

# 15.21 Validación de Valores Temporales
temporal_numeric_columns = [
    "similaridad_global_promedio",
    "similaridad_global_sd",
    "similaridad_global_min",
    "similaridad_global_max",
    "similaridad_inicial",
    "similaridad_final",
    "cambio_similitud",
    "cambio_similitud_absoluto",
    "cambio_anual_promedio",
    "variabilidad_temporal"
] # Definir indicadores numéricos temporales

missing_temporal_columns = [
    column
    for column in temporal_numeric_columns
    if column not in pair_temporal_summary.columns
] # Identificar indicadores temporales ausentes

if missing_temporal_columns:
    raise ValueError(
        "Faltan indicadores temporales requeridos: "
        f"{missing_temporal_columns}"
    ) # Validar estructura completa del análisis temporal

temporal_numeric_values = pair_temporal_summary[
    temporal_numeric_columns
].to_numpy(
    dtype=np.float64
) # Convertir indicadores temporales a matriz numérica

if not np.isfinite(
    temporal_numeric_values
).all():
    raise ValueError(
        "El análisis de estabilidad temporal contiene "
        "valores NaN, infinitos o no finitos."
    ) # Validar integridad numérica completa

# 15.22 Validación del Rango de Similitud
for similarity_column in [
    "similaridad_global_promedio",
    "similaridad_global_min",
    "similaridad_global_max",
    "similaridad_inicial",
    "similaridad_final"
]:

    similarity_values = pair_temporal_summary[
        similarity_column
    ].to_numpy(
        dtype=np.float64
    ) # Obtener valores de similitud

    if (
        similarity_values.min() < -1.0000001
        or
        similarity_values.max() > 1.0000001
    ):
        raise ValueError(
            f"La columna '{similarity_column}' contiene valores "
            "fuera del rango esperado de similitud."
        ) # Validar rango matemático

# 15.23 Tabla de Evolución Temporal por Año
temporal_year_summary = (
    temporal_analysis
    .groupby(
        TIME_KEY,
        as_index=False
    )
    .agg(
        similaridad_global_promedio=(
            "similaridad_global",
            "mean"
        ),
        similaridad_global_sd=(
            "similaridad_global",
            "std"
        ),
        similaridad_global_min=(
            "similaridad_global",
            "min"
        ),
        similaridad_global_max=(
            "similaridad_global",
            "max"
        ),
        n_pares=(
            "similaridad_global",
            "count"
        )
    )
) # Resumir evolución global de la similitud por año

# 15.24 Validación de Cantidad de Pares por Año
pair_counts_by_year = (
    temporal_analysis
    .groupby(TIME_KEY)
    .size()
) # Obtener cantidad efectiva de pares por año

expected_pairs_per_year = (
    ANALYSIS_N_MUNICIPALITIES
    * (ANALYSIS_N_MUNICIPALITIES - 1)
    // 2
) # Calcular cantidad científica esperada de pares por año

if pair_counts_by_year.nunique() != 1:
    raise ValueError(
        "La cantidad de pares nodales no es constante entre los años."
    ) # Validar balance estructural temporal

if not (
    pair_counts_by_year == expected_pairs_per_year
).all():
    raise ValueError(
        "La cantidad de pares nodales por año no coincide con "
        f"el valor esperado de {expected_pairs_per_year:,}."
    ) # Validar cardinalidad exacta de pares por año

expected_years_sorted = sorted(
    expected_years
) # Ordenar años oficiales para comparación determinista

observed_years_sorted = sorted(
    pair_counts_by_year.index.astype(int).tolist()
) # Obtener años observados y ordenarlos

if observed_years_sorted != expected_years_sorted:
    raise ValueError(
        "La cobertura temporal de los pares no coincide "
        "con los años oficiales. "
        f"Esperados: {expected_years_sorted}. "
        f"Encontrados: {observed_years_sorted}."
    ) # Validar cobertura temporal completa

# 15.25 Validación del Umbral de Alta Similitud
if not (
    0 < HIGH_SIMILARITY_THRESHOLD <= 1
):
    raise ValueError(
        f"HIGH_SIMILARITY_THRESHOLD debe estar entre 0 y 1: "
        f"{HIGH_SIMILARITY_THRESHOLD}"
    ) # Validar umbral oficial de alta similitud

# 15.26 Persistencia de Relaciones de Alta Similitud
high_similarity_pairs = (
    temporal_analysis[
        temporal_analysis[
            "similaridad_global"
        ] >= HIGH_SIMILARITY_THRESHOLD
    ]
    .groupby(
        temporal_pair_keys,
        as_index=False
    )
    .agg(
        n_years_high_similarity=(
            "similaridad_global",
            "count"
        )
    )
) # Contar años con similitud superior al umbral

# 15.27 Proporción de Persistencia
high_similarity_pairs[
    "proporcion_persistencia"
] = (
    high_similarity_pairs[
        "n_years_high_similarity"
    ]
    /
    ANALYSIS_N_YEARS
) # Calcular proporción temporal de persistencia

# 15.28 Identificación de Relaciones Persistentes
persistent_pairs = high_similarity_pairs[
    high_similarity_pairs[
        "n_years_high_similarity"
    ] == ANALYSIS_N_YEARS
].copy() # Identificar pares altamente similares durante todo el período

# 15.29 Validación de Persistencia

persistent_proportions = persistent_pairs[
    "proporcion_persistencia"
].to_numpy(
    dtype=np.float64
) # Obtener proporciones de persistencia

if not np.isfinite(
    persistent_proportions
).all():
    raise ValueError(
        "La tabla de relaciones persistentes contiene "
        "valores no finitos en proporcion_persistencia."
    ) # Validar integridad numérica de la persistencia

if not np.isclose(
    persistent_proportions,
    1.0,
    atol=1e-12
).all():
    raise ValueError(
        "Se detectaron relaciones persistentes con proporción "
        "de persistencia distinta de 1."
    ) # Validar consistencia de las relaciones persistentes

# 15.30 Ordenamiento de Resultados
pair_temporal_summary = (
    pair_temporal_summary
    .sort_values(
        by=temporal_pair_keys
    )
    .reset_index(drop=True)
) # Ordenar estabilidad temporal por par nodal

temporal_year_summary = (
    temporal_year_summary
    .sort_values(
        by=TIME_KEY
    )
    .reset_index(drop=True)
) # Ordenar evolución temporal cronológicamente

high_similarity_pairs = (
    high_similarity_pairs
    .sort_values(
        by=[
            "proporcion_persistencia",
            "n_years_high_similarity",
            "node_idx_a",
            "node_idx_b"
        ],
        ascending=[
            False,
            False,
            True,
            True
        ]
    )
    .reset_index(drop=True)
) # Ordenar persistencia de mayor a menor

persistent_pairs = (
    persistent_pairs
    .sort_values(
        by=temporal_pair_keys
    )
    .reset_index(drop=True)
) # Ordenar relaciones persistentes

# 15.31 Exportación del Resumen Temporal
pair_temporal_summary.to_parquet(
    OUTPUTS_DIR / "municipality_temporal_stability.parquet",
    index=False
) # Exportar estabilidad temporal por par nodal

# 15.32 Exportación de la Evolución Anual
temporal_year_summary.to_parquet(
    OUTPUTS_DIR / "temporal_similarity_evolution.parquet",
    index=False
) # Exportar evolución temporal agregada

# 15.33 Exportación de Persistencia
high_similarity_pairs.to_parquet(
    OUTPUTS_DIR / "municipality_similarity_persistence.parquet",
    index=False
) # Exportar persistencia de relaciones altamente similares

# 15.34 Exportación de Relaciones Persistentes
persistent_pairs.to_parquet(
    OUTPUTS_DIR / "persistent_similar_municipal_pairs.parquet",
    index=False
) # Exportar pares altamente similares durante todo el período

# 15.35 Validación Final y Registro del Resultado

expected_total_temporal_pairs = (
    ANALYSIS_N_MUNICIPALITIES
    * (ANALYSIS_N_MUNICIPALITIES - 1)
    // 2
    * ANALYSIS_N_YEARS
) # Calcular cardinalidad longitudinal científica esperada

if len(temporal_analysis) != expected_total_temporal_pairs:
    raise ValueError(
        f"El análisis temporal contiene "
        f"{len(temporal_analysis):,} registros; "
        f"se esperaban {expected_total_temporal_pairs:,}."
    ) # Validar cardinalidad longitudinal completa

expected_unique_pairs = (
    ANALYSIS_N_MUNICIPALITIES
    * (ANALYSIS_N_MUNICIPALITIES - 1)
    // 2
) # Calcular cantidad esperada de pares nodales únicos

if len(pair_temporal_summary) != expected_unique_pairs:
    raise ValueError(
        f"El resumen temporal contiene "
        f"{len(pair_temporal_summary):,} pares; "
        f"se esperaban {expected_unique_pairs:,}."
    ) # Validar cobertura completa de pares longitudinales

if len(persistent_pairs) > expected_unique_pairs:
    raise ValueError(
        "La cantidad de relaciones persistentes supera "
        "la cantidad total de pares nodales disponibles."
    ) # Validar límite superior de relaciones persistentes

logger.info(
    "Análisis temporal completado: %s pares nodales, "
    "%s pares persistentes y %s años analizados.",
    len(pair_temporal_summary),
    len(persistent_pairs),
    ANALYSIS_N_YEARS
) # Registrar resultados del análisis temporal

logger.info(
    "Claves longitudinales utilizadas: %s.",
    temporal_keys
) # Registrar estructura temporal utilizada

logger.info(
    "Cardinalidad temporal validada: %s registros.",
    expected_total_temporal_pairs
) # Registrar cardinalidad científica validada

logger.info(
    "Estado del Bloque 15: VALIDADO."
) # Confirmar finalización correcta del análisis temporal

# BLOQUE 16. ANÁLISIS DE SIMILITUD Y RENDIMIENTO AGRÍCOLA
# Objetivo: Evaluar exploratoriamente la asociación entre la similitud
# de los perfiles municipales y la diferencia de la variable objetivo.
# Arquitectura científica
# Entradas: municipality_analysis y configuración científica oficial.
# Proceso: Evaluación exploratoria de asociaciones entre similitud global, diferencias de la variable objetivo y dominios científicos.
# Producto: Tablas analíticas sobre la asociación entre similitud nodal y comportamiento de la variable objetivo. 
# Nota metodológica: Las correlaciones de Pearson y Spearman se utilizan exclusivamente como medidas exploratorias de 
# asociación entre similitud y diferencia del objetivo. Las observaciones municipio-año-par presentan dependencia
# espacial y temporal, por lo que estos coeficientes no deben interpretarse como pruebas inferenciales bajo independencia ni como evidencia causal.

# 16.1 Validación de Variables Requeridas
required_target_columns = [
    TIME_KEY,
    "node_idx_a",
    "node_idx_b",
    "similaridad_global",
    "difference_target"
] # Definir variables estructurales y analíticas requeridas

missing_target_columns = [
    column
    for column in required_target_columns
    if column not in municipality_analysis.columns
] # Identificar variables requeridas ausentes

if missing_target_columns:
    raise ValueError(
        "La tabla maestra no contiene las variables requeridas: "
        f"{missing_target_columns}"
    ) # Detener el análisis si faltan variables esenciales

# 16.2 Validación del Espacio Nodal
for node_column in [
    "node_idx_a",
    "node_idx_b"
]:

    node_values = municipality_analysis[
        node_column
    ].to_numpy(
        dtype=np.int64
    ) # Obtener índices nodales de la tabla maestra

    if (
        node_values < 0
    ).any() or (
        node_values >= ANALYSIS_N_MUNICIPALITIES
    ).any():
        raise ValueError(
            f"La columna '{node_column}' contiene índices fuera "
            f"del espacio nodal oficial 0 ... "
            f"{ANALYSIS_N_MUNICIPALITIES - 1}."
        ) # Validar espacio oficial de índices

# 16.3 Validación del Orden de los Pares
if not (
    municipality_analysis["node_idx_a"]
    <
    municipality_analysis["node_idx_b"]
).all():
    raise ValueError(
        "La tabla maestra contiene pares que no cumplen "
        "node_idx_a < node_idx_b."
    ) # Validar representación canónica de los pares

# 16.4 Validación de Autocombinaciones
if (
    municipality_analysis["node_idx_a"]
    ==
    municipality_analysis["node_idx_b"]
).any():
    raise ValueError(
        "La tabla maestra contiene autocombinaciones donde "
        "node_idx_a == node_idx_b."
    ) # Validar ausencia de comparaciones de un nodo consigo mismo

# 16.5 Validación de Unicidad Nodo Año Par
analysis_keys = [
    TIME_KEY,
    "node_idx_a",
    "node_idx_b"
] # Definir clave estructural de cada observación analítica

if municipality_analysis.duplicated(
    subset=analysis_keys
).any():
    raise ValueError(
        "La tabla maestra contiene observaciones "
        "nodo-año-par duplicadas."
    ) # Validar unicidad de las observaciones

# 16.6 Validación de Cardinalidad
expected_pairs_per_year = (
    ANALYSIS_N_MUNICIPALITIES
    * (ANALYSIS_N_MUNICIPALITIES - 1)
    // 2
) # Calcular cantidad esperada de pares por año

expected_total_pairs = (
    expected_pairs_per_year
    * ANALYSIS_N_YEARS
) # Calcular cantidad esperada de pares municipio-año

if len(municipality_analysis) != expected_total_pairs:
    raise ValueError(
        f"municipality_analysis contiene "
        f"{len(municipality_analysis):,} registros; "
        f"se esperaban {expected_total_pairs:,}."
    ) # Validar cardinalidad completa del universo analítico

# 16.7 Construcción de la Tabla de Relación
base_target_columns = [
    TIME_KEY,
    "node_idx_a",
    "node_idx_b",
    "similaridad_global",
    "difference_target"
] # Definir variables estructurales y objetivo

domain_target_columns = [
    column
    for column in domain_similarity_columns
    if column in municipality_analysis.columns
] # Seleccionar similitudes por dominio disponibles

similarity_target_analysis = municipality_analysis[
    base_target_columns + domain_target_columns
].copy() # Construir tabla analítica de similitud y variable objetivo

# 16.8 Validación de Valores Analíticos
analysis_numeric_columns = [
    "similaridad_global",
    "difference_target"
] + domain_target_columns # Definir variables numéricas del análisis

analysis_numeric_values = similarity_target_analysis[
    analysis_numeric_columns
].to_numpy(
    dtype=np.float64
) # Convertir variables analíticas a matriz numérica

if not np.isfinite(
    analysis_numeric_values
).all():
    raise ValueError(
        "Las variables utilizadas en el análisis contienen "
        "valores NaN, infinitos o no finitos."
    ) # Validar integridad numérica completa

# 16.9 Validación del Rango de Similitud
similarity_values = similarity_target_analysis[
    "similaridad_global"
].to_numpy(
    dtype=np.float64
) # Obtener valores de similitud global

if (
    similarity_values.min() < -1.0000001
    or
    similarity_values.max() > 1.0000001
):
    raise ValueError(
        "La similitud global contiene valores fuera "
        "del intervalo matemático [-1, 1]."
    ) # Validar rango matemático de la similitud coseno

# 16.10 Correlación Global de Pearson
global_target_correlation_pearson = (
    similarity_target_analysis[
        [
            "similaridad_global",
            "difference_target"
        ]
    ]
    .corr(
        method="pearson"
    )
    .iloc[0, 1]
) # Estimar asociación lineal exploratoria global

# 16.11 Correlación Global de Spearman
global_target_correlation_spearman = (
    similarity_target_analysis[
        [
            "similaridad_global",
            "difference_target"
        ]
    ]
    .corr(
        method="spearman"
    )
    .iloc[0, 1]
) # Estimar asociación monotónica exploratoria global

# 16.12 Validación de Correlaciones Globales
global_correlations = np.array(
    [
        global_target_correlation_pearson,
        global_target_correlation_spearman
    ],
    dtype=np.float64
) # Construir vector de correlaciones globales

if not np.isfinite(
    global_correlations
).all():
    raise ValueError(
        "Las correlaciones globales contienen valores no finitos."
    ) # Validar integridad numérica de las correlaciones

if (
    global_correlations.min() < -1.0000001
    or
    global_correlations.max() > 1.0000001
):
    raise ValueError(
        "Las correlaciones globales están fuera "
        "del intervalo matemático [-1, 1]."
    ) # Validar rango matemático de las correlaciones

# 16.13 Correlación por Dominio
domain_target_correlations = []
for domain_column in domain_target_columns:
    domain_data = similarity_target_analysis[
        [
            domain_column,
            "difference_target"
        ]
    ].copy() # Seleccionar variables del dominio

    if not np.isfinite(
        domain_data.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            f"El dominio '{domain_column}' contiene "
            "valores no finitos."
        ) # Validar integridad numérica del dominio

    if len(domain_data) < 2:
        raise ValueError(
            f"El dominio '{domain_column}' no contiene "
            "suficientes observaciones."
        ) # Validar tamaño mínimo de observaciones

    pearson_value = domain_data.corr(
        method="pearson"
    ).iloc[0, 1] # Estimar asociación lineal del dominio

    spearman_value = domain_data.corr(
        method="spearman"
    ).iloc[0, 1] # Estimar asociación monotónica del dominio

    if not np.isfinite(
        [
            pearson_value,
            spearman_value
        ]
    ).all():
        raise ValueError(
            f"Las correlaciones del dominio '{domain_column}' "
            "no son finitas."
        ) # Validar resultados de correlación

    domain_target_correlations.append({
        "dominio": domain_column,
        "correlacion_pearson": pearson_value,
        "correlacion_spearman": spearman_value,
        "n_observaciones": len(domain_data)
    }) # Registrar asociaciones del dominio

domain_target_correlation = pd.DataFrame(
    domain_target_correlations
) # Construir tabla de correlaciones por dominio

# 16.14 Validación de Correlaciones por Dominio
if not domain_target_correlation.empty:
    domain_correlation_values = domain_target_correlation[
        [
            "correlacion_pearson",
            "correlacion_spearman"
        ]
    ].to_numpy(
        dtype=np.float64
    ) # Extraer correlaciones por dominio

    if not np.isfinite(
        domain_correlation_values
    ).all():
        raise ValueError(
            "Las correlaciones por dominio contienen "
            "valores no finitos."
        ) # Validar correlaciones por dominio

    if (
        domain_correlation_values.min() < -1.0000001
        or
        domain_correlation_values.max() > 1.0000001
    ):
        raise ValueError(
            "Las correlaciones por dominio están fuera "
            "del intervalo matemático [-1, 1]."
        ) # Validar rango matemático de las correlaciones

# 16.15 Validación de Umbrales de Similitud
if not (
    -1
    <= SIMILARITY_LEVEL_1
    < SIMILARITY_LEVEL_2
    < SIMILARITY_LEVEL_3
    < SIMILARITY_LEVEL_4
    <= 1
):
    raise ValueError(
        "Los umbrales de clasificación de similitud "
        "deben encontrarse en el intervalo [-1, 1] "
        "y mantener un orden estrictamente creciente."
    ) # Validar configuración de categorías para similitud coseno

# 16.16 Agrupación por Nivel de Similitud
similarity_target_analysis[
    "nivel_similitud"
] = pd.cut(
    similarity_target_analysis[
        "similaridad_global"
    ],
    bins=[
        -np.inf,
        SIMILARITY_LEVEL_1,
        SIMILARITY_LEVEL_2,
        SIMILARITY_LEVEL_3,
        SIMILARITY_LEVEL_4,
        np.inf
    ],
    labels=[
        "muy_baja",
        "baja",
        "moderada",
        "alta",
        "muy_alta"
    ],
    include_lowest=True
) # Clasificar pares según similitud global

# 16.17 Resumen de la Variable Objetivo por Nivel de Similitud
similarity_target_summary = (
    similarity_target_analysis
    .groupby(
        "nivel_similitud",
        observed=False,
        as_index=False
    )
    .agg(
        n_pares=(
            "difference_target",
            "count"
        ),
        diferencia_objetivo_promedio=(
            "difference_target",
            "mean"
        ),
        diferencia_objetivo_sd=(
            "difference_target",
            "std"
        ),
        diferencia_objetivo_mediana=(
            "difference_target",
            "median"
        ),
        diferencia_objetivo_min=(
            "difference_target",
            "min"
        ),
        diferencia_objetivo_max=(
            "difference_target",
            "max"
        )
    )
) # Resumir diferencia de la variable objetivo según similitud

# 16.18 Cuantiles de Diferencia de la Variable Objetivo
target_difference_q25 = similarity_target_analysis[
    "difference_target"
].quantile(
    0.25
) # Calcular percentil 25 de diferencia del objetivo

target_difference_q75 = similarity_target_analysis[
    "difference_target"
].quantile(
    0.75
) # Calcular percentil 75 de diferencia del objetivo

# 16.19 Alta Similitud y Baja Diferencia del Objetivo
high_similarity_low_target_difference = (
    similarity_target_analysis[
        (
            similarity_target_analysis[
                "similaridad_global"
            ] >= HIGH_SIMILARITY_THRESHOLD
        )
        &
        (
            similarity_target_analysis[
                "difference_target"
            ] <= target_difference_q25
        )
    ]
    .copy()
) # Identificar pares muy similares con baja diferencia del objetivo

# 16.20 Alta Similitud y Alta Diferencia del Objetivo
high_similarity_high_target_difference = (
    similarity_target_analysis[
        (
            similarity_target_analysis[
                "similaridad_global"
            ] >= HIGH_SIMILARITY_THRESHOLD
        )
        &
        (
            similarity_target_analysis[
                "difference_target"
            ] >= target_difference_q75
        )
    ]
    .copy()
) # Identificar pares muy similares con alta diferencia del objetivo

# 16.21 Baja Similitud y Baja Diferencia del Objetivo
low_similarity_low_target_difference = (
    similarity_target_analysis[
        (
            similarity_target_analysis[
                "similaridad_global"
            ] <= LOW_SIMILARITY_THRESHOLD
        )
        &
        (
            similarity_target_analysis[
                "difference_target"
            ] <= target_difference_q25
        )
    ]
    .copy()
) # Identificar pares diferentes con baja diferencia del objetivo

# 16.22 Baja Similitud y Alta Diferencia del Objetivo
low_similarity_high_target_difference = (
    similarity_target_analysis[
        (
            similarity_target_analysis[
                "similaridad_global"
            ] <= LOW_SIMILARITY_THRESHOLD
        )
        &
        (
            similarity_target_analysis[
                "difference_target"
            ] >= target_difference_q75
        )
    ]
    .copy()
) # Identificar pares diferentes con alta diferencia del objetivo

# 16.23 Análisis Temporal de la Relación
temporal_target_analysis = (
    similarity_target_analysis
    .groupby(
        TIME_KEY,
        as_index=False
    )
    .agg(
        similaridad_global_promedio=(
            "similaridad_global",
            "mean"
        ),
        diferencia_objetivo_promedio=(
            "difference_target",
            "mean"
        ),
        similaridad_global_sd=(
            "similaridad_global",
            "std"
        ),
        diferencia_objetivo_sd=(
            "difference_target",
            "std"
        ),
        n_pares=(
            "difference_target",
            "count"
        )
    )
) # Resumir relación entre similitud y objetivo por año

# 16.24 Validación de Cobertura Temporal
available_years = sorted(
    similarity_target_analysis[
        TIME_KEY
    ].unique().tolist()
) # Obtener años disponibles

expected_years = sorted(
    ANALYSIS_YEARS
) # Obtener años oficiales

if available_years != expected_years:
    raise ValueError(
        f"Los años disponibles {available_years} "
        f"no coinciden con los años oficiales {expected_years}."
    ) # Validar cobertura temporal exacta

expected_pairs_per_year = (
    ANALYSIS_N_MUNICIPALITIES
    * (ANALYSIS_N_MUNICIPALITIES - 1)
    // 2
) # Calcular cantidad esperada de pares por año

if not (
    temporal_target_analysis["n_pares"]
    ==
    expected_pairs_per_year
).all():
    raise ValueError(
        "La cantidad de pares por año del análisis "
        "similitud objetivo no coincide con la cardinalidad esperada."
    ) # Validar cobertura completa de pares por año.

# 16.25 Correlación Temporal
temporal_correlations = []
for current_year in ANALYSIS_YEARS:
    year_data = similarity_target_analysis[
        similarity_target_analysis[
            TIME_KEY
        ] == current_year
    ].copy() # Seleccionar observaciones del año actual

    if len(year_data) != expected_pairs_per_year:
        raise ValueError(
            f"El año {current_year} contiene "
            f"{len(year_data):,} observaciones; "
            f"se esperaban {expected_pairs_per_year:,}."
        ) # Validar cardinalidad anual

    pearson_value = year_data[
        [
            "similaridad_global",
            "difference_target"
        ]
    ].corr(
        method="pearson"
    ).iloc[0, 1] # Estimar asociación lineal anual

    spearman_value = year_data[
        [
            "similaridad_global",
            "difference_target"
        ]
    ].corr(
        method="spearman"
    ).iloc[0, 1] # Estimar asociación monotónica anual

    if not np.isfinite(
        [
            pearson_value,
            spearman_value
        ]
    ).all():
        raise ValueError(
            f"Las correlaciones del año {current_year} "
            "no son finitas."
        ) # Validar correlaciones anuales

    temporal_correlations.append({
        TIME_KEY: current_year,
        "correlacion_pearson": pearson_value,
        "correlacion_spearman": spearman_value,
        "n_pares": len(year_data)
    }) # Registrar correlaciones del año

temporal_target_correlation = pd.DataFrame(
    temporal_correlations
) # Construir tabla de correlaciones temporales

# 16.26 Validación de Correlaciones Temporales
temporal_correlation_values = (
    temporal_target_correlation[
        [
            "correlacion_pearson",
            "correlacion_spearman"
        ]
    ]
    .to_numpy(
        dtype=np.float64
    )
) # Extraer correlaciones temporales

if not np.isfinite(
    temporal_correlation_values
).all():
    raise ValueError(
        "Las correlaciones temporales contienen "
        "valores no finitos."
    ) # Validar correlaciones anuales

if (
    temporal_correlation_values.min() < -1.0000001
    or
    temporal_correlation_values.max() > 1.0000001
):
    raise ValueError(
        "Las correlaciones temporales están fuera "
        "del intervalo matemático [-1, 1]."
    ) # Validar rango matemático de las correlaciones

# 16.27 Ordenamiento de Resultados
similarity_target_analysis = (
    similarity_target_analysis
    .sort_values(
        by=analysis_keys
    )
    .reset_index(drop=True)
) # Ordenar observaciones analíticas por año y par nodal

temporal_target_analysis = (
    temporal_target_analysis
    .sort_values(
        by=TIME_KEY
    )
    .reset_index(drop=True)
) # Ordenar resumen temporal

domain_target_correlation = (
    domain_target_correlation
    .sort_values(
        by="correlacion_pearson",
        ascending=False
    )
    .reset_index(drop=True)
) # Ordenar dominios según asociación lineal

temporal_target_correlation = (
    temporal_target_correlation
    .sort_values(
        by=TIME_KEY
    )
    .reset_index(drop=True)
) # Ordenar correlaciones cronológicamente

# 16.28 Exportación del Análisis Principal
similarity_target_analysis.to_parquet(
    OUTPUTS_DIR / "similarity_target_analysis.parquet",
    index=False
) # Exportar relación entre similitud y objetivo

# 16.29 Exportación del Resumen por Nivel de Similitud
similarity_target_summary.to_parquet(
    OUTPUTS_DIR / "similarity_target_summary.parquet",
    index=False
) # Exportar diferencias del objetivo por nivel de similitud

# 16.30 Exportación de Correlaciones por Dominio
domain_target_correlation.to_parquet(
    OUTPUTS_DIR / "domain_target_correlations.parquet",
    index=False
) # Exportar correlaciones por dominio

# 16.31 Exportación del Análisis Temporal
temporal_target_analysis.to_parquet(
    OUTPUTS_DIR / "temporal_similarity_target_analysis.parquet",
    index=False
) # Exportar evolución temporal de similitud y objetivo

# 16.32 Exportación de Correlaciones Temporales
temporal_target_correlation.to_parquet(
    OUTPUTS_DIR / "temporal_target_correlations.parquet",
    index=False
) # Exportar correlaciones anuales