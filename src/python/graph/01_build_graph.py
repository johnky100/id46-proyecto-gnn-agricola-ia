# 01_build_graph.py
# BLOQUE 1. Importaciones -------------------------------------------------
## Objetivo: Importar las librerías, configuraciones y rutas oficiales necesarias
# para construir el objeto GraphData del proyecto.
## Pregunta científica: ¿Qué dependencias requiere el proyecto para construir un GraphData
# reproducible y compatible con PyTorch Geometric?

# Librerías estándar
import os # Funciones del sistema operativo
import time # Medición del tiempo de ejecución
import warnings # Control de advertencias

# Librerías científicas
import numpy as np # Operaciones numéricas
import pandas as pd # Manipulación de datos
import torch # PyTorch
from torch_geometric.data import Data # Objeto GraphData

# Librerías geoespaciales
import geopandas as gpd # Manipulación de datos geoespaciales
from shapely import wkb

from libpysal.weights import (
    Queen,
    Rook,
    KNN
) # Métodos de construcción de vecindad espacial

# Configuración oficial del proyecto
from src.python.config.config_project import (
    SEED,
    ID_COLUMNS,
    NODE_CONFIG,
    NODE_FEATURE_CONFIG,
    NODE_FEATURE_COLUMNS,
    GRAPH_NODE_FEATURES,
    GRAPH_NODE_ATTRIBUTES,
    GRAPH_SPATIAL_CONFIG,
    GRAPH_CONFIG,
    GRAPH_VALIDATION,
    GRAPH_NEIGHBORS,
    TARGET_VARIABLE
)

# Rutas oficiales del proyecto
from src.python.config.paths import (
    DATASET_FILE,
    GRAPH_DATA_FILE,
    validate_project_structure
)

# BLOQUE 2. Configuración del Script --------------------------------------
## Objetivo: Configurar el entorno de ejecución, validar la estructura oficial del
# proyecto e inicializar los parámetros necesarios para construir el
# GraphData de forma reproducible.
## Pregunta científica: ¿El entorno de ejecución cumple las condiciones necesarias para
# construir el grafo de forma reproducible?
print("\n" + "-" * 80)
print("BLOQUE 2. CONFIGURACIÓN DEL SCRIPT")
print("-" * 80)

# Ejecución 2.1. Configuración del entorno -------------------------------
warnings.filterwarnings(
    "ignore"
) # Ocultar advertencias
print(
    "Entorno de ejecución configurado correctamente."
)

# Ejecución 2.2. Validación de la estructura del proyecto ----------------
validate_project_structure(
    verbose = True
) # Validar estructura del proyecto
print(
    "Estructura del proyecto validada correctamente."
)

# Ejecución 2.3. Configuración de la reproducibilidad --------------------
torch.manual_seed(
    SEED
) # Configurar semilla de PyTorch

if torch.cuda.is_available():
    torch.cuda.manual_seed(
        SEED
    ) # Configurar semilla de la GPU

    torch.cuda.manual_seed_all(
        SEED
    ) # Configurar todas las GPU

    torch.backends.cudnn.deterministic = True # Reproducibilidad
    torch.backends.cudnn.benchmark = False # Desactivar benchmark

print(
    f"Semilla del proyecto          : {SEED}"
)

# Ejecución 2.4. Selección del dispositivo -------------------------------
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
) # Dispositivo de procesamiento

print(
    f"Dispositivo de procesamiento  : {DEVICE.upper()}"
)

# Ejecución 2.5. Confirmación del bloque ---------------------------------
print(
    "\nConfiguración del entorno finalizada correctamente."
)

print("-" * 80)

# BLOQUE 3. Carga del Dataset Científico ----------------------------------
# Objetivo:
# Cargar el dataset científico certificado que servirá como entrada para la
# construcción del grafo espacio-temporal.
## Pregunta científica:
# ¿El dataset científico oficial está disponible y puede cargarse correctamente para iniciar el proceso de construcción del grafo?
print("\n" + "-" * 80)
print("BLOQUE 3. CARGA DEL DATASET CIENTÍFICO")
print("-" * 80)

# Ejecución 1. Verificación del archivo -----------------------------------
if not DATASET_FILE.exists():
    raise FileNotFoundError(
        f"No se encontró el Dataset Científico:\n{DATASET_FILE}"
    )
print("Dataset científico localizado correctamente.")

# Ejecución 2. Carga del dataset ------------------------------------------
dataset = pd.read_parquet(
    DATASET_FILE
) # Cargar dataset científico
print("Dataset científico cargado correctamente.")

# Ejecución 3. Registro de dimensiones ------------------------------------
n_registros, n_variables = dataset.shape
print(f"Registros : {n_registros:,}")
print(f"Variables : {n_variables:,}")

# Ejecución 4. Confirmación del bloque ------------------------------------
print("\nCarga del Dataset Científico finalizada correctamente.")
print("-" * 80)

# BLOQUE 4. Validación del Dataset Científico ------------------------------
# Objetivo:
# Verificar la integridad, consistencia y completitud del Dataset Científico
# antes de iniciar la construcción del grafo espacio-temporal.
## Pregunta científica:
# ¿El Dataset Científico cumple los requisitos necesarios para construir
# correctamente el GraphData?

print("\n" + "-" * 80)
print("BLOQUE 4. VALIDACIÓN DEL DATASET CIENTÍFICO")
print("-" * 80)

# 4.1 Dataset no vacío ------------------------
if dataset.empty:
    raise ValueError(
        "El Dataset Científico está vacío."
    )
print("Dataset no vacío.")

# 4.2 Variables obligatorias -------------------------------
required_columns = (
    ID_COLUMNS +
    NODE_FEATURE_COLUMNS +
    [TARGET_VARIABLE]
)

missing_columns = [
    column
    for column in required_columns
    if column not in dataset.columns
]

if missing_columns:
    raise ValueError(
        f"Columnas faltantes: {missing_columns}"
    )
print("Columnas obligatorias verificadas.")

# 4.3 Integridad del panel (cod_mpio + anio) ------------------------------
## Objetivo: Verificar que cada municipio tenga un único registro por año dentro
# del Dataset Científico.
## Responde: ¿Existe un único registro para cada municipio y año?
panel_duplicates = (
    dataset[
        [
            NODE_CONFIG["node_id_column"],
            NODE_CONFIG["time_column"]
        ]
    ]
    .duplicated()
    .sum()
) # Registros duplicados del panel

if panel_duplicates > 0:

    raise ValueError(
        f"Se encontraron {panel_duplicates:,} registros duplicados para la combinación municipio-año."
    )

print(
    "Integridad del panel verificada."
)

# 4.4 Valores faltantes ---------------------------------------------------
#
# Objetivo:
# Verificar que las variables críticas del Dataset Científico no contengan
# valores faltantes antes de construir el grafo.
#
# Entradas:
# - dataset
# - required_columns
#
# Producto:
# - Valores faltantes validados.
#
# Responde:
# ¿Las variables críticas contienen valores faltantes?

missing_values = (
    dataset[
        required_columns
    ]
    .isna()
    .sum()
) # Valores faltantes por variable

missing_variables = (
    missing_values[
        missing_values > 0
    ]
) # Variables con valores faltantes

if not missing_variables.empty:

    raise ValueError(
        "Se encontraron valores faltantes en las siguientes variables:\n"
        f"{missing_variables.to_string()}"
    )

print(
    "Valores faltantes verificados."
)

# 4.5 Validación de las Node Features ------------------------------------
#
# Objetivo:
# Verificar que la configuración de las Node Features sea consistente y
# que exista al menos una variable predictora para construir el GraphData.
#
# Entradas:
# - NODE_FEATURE_COLUMNS
#
# Producto:
# - Configuración de Node Features validada.
#
# Responde:
# ¿La configuración de las Node Features es válida para construir el grafo?

if len(NODE_FEATURE_COLUMNS) == 0:

    raise ValueError(
        "La lista NODE_FEATURE_COLUMNS está vacía."
    )

if len(set(NODE_FEATURE_COLUMNS)) != len(NODE_FEATURE_COLUMNS):

    raise ValueError(
        "Existen variables duplicadas en NODE_FEATURE_COLUMNS."
    )

print(
    f"Node Features configuradas : {len(NODE_FEATURE_COLUMNS)}"
)

print(
    "Configuración de Node Features verificada."
)

# 4.6 Información espacial -----------------------------------------------
#
# Objetivo:
# Verificar que el Dataset Científico contiene todas las variables
# espaciales necesarias para construir el grafo.
#
# Entradas:
# - dataset
# - GRAPH_SPATIAL_CONFIG
#
# Producto:
# - Información espacial validada.
#
# Responde:
# ¿El Dataset Científico contiene toda la información espacial requerida
# para construir la estructura del grafo?

spatial_columns = [
    "latitud",
    "longitud",
    GRAPH_SPATIAL_CONFIG["geometry_column"]
] # Variables espaciales obligatorias

missing_spatial = [
    column
    for column in spatial_columns
    if column not in dataset.columns
] # Variables espaciales faltantes

if missing_spatial:

    raise ValueError(
        "Faltan las siguientes variables espaciales: "
        f"{missing_spatial}"
    )

print(
    f"Variables espaciales : {len(spatial_columns)}"
)

print(
    "Información espacial verificada."
)

spatial_columns = [
    GRAPH_SPATIAL_CONFIG["latitude_column"],
    GRAPH_SPATIAL_CONFIG["longitude_column"],
    GRAPH_SPATIAL_CONFIG["geometry_column"]
] # Variables espaciales obligatorias

# 4.7 Confirmación ------------------------------------
print("\nDataset Científico validado correctamente.")
print("-" * 80)

# BLOQUE 5. Construcción de los Nodos -------------------------------------
## Objetivo: Construir el catálogo oficial de nodos del grafo, asignando un
# identificador interno a cada municipio para su utilización en
# PyTorch Geometric.
## Pregunta científica: ¿Cómo representar cada municipio como un nodo único dentro del
# grafo espacio-temporal?

print("\n" + "-" * 80)
print("BLOQUE 5. CONSTRUCCIÓN DE LOS NODOS")
print("-" * 80)

# Ejecución 5.1. Construcción del catálogo de nodos -----------------------
## Objetivo:
# Construir el catálogo oficial de nodos a partir del Dataset Científico.
## Producto:
# - nodes

nodes = (
    dataset[
        GRAPH_NODE_ATTRIBUTES
    ]
    .drop_duplicates()
) # Catálogo oficial de nodos

if nodes.empty:

    raise ValueError(
        "El catálogo oficial de nodos está vacío."
    )

print(
    "Catálogo de nodos construido correctamente."
)

duplicate_nodes = (
    nodes[
        NODE_CONFIG["node_id_column"]
    ]
    .duplicated()
    .sum()
) # Municipios duplicados

if duplicate_nodes > 0:

    raise ValueError(
        f"Se encontraron {duplicate_nodes:,} municipios duplicados en el catálogo de nodos."
    )

# Ejecución 5.2. Ordenamiento del catálogo --------------------------------
## Objetivo:
# Ordenar el catálogo oficial utilizando el identificador oficial del
# municipio.

nodes = (
    nodes
    .sort_values(
        by = NODE_CONFIG["node_id_column"]
    )
    .reset_index(
        drop = True
    )
) # Catálogo ordenado

print(
    "Catálogo de nodos ordenado correctamente."
)

# Ejecución 5.3. Creación del identificador interno -----------------------
## Objetivo:
# Crear el identificador interno utilizado por PyTorch Geometric.

nodes[
    NODE_CONFIG["graph_node_id"]
] = range(
    len(nodes)
) # Identificador interno del grafo

print(
    "Identificador interno creado correctamente."
)

if nodes[
    NODE_CONFIG["graph_node_id"]
].duplicated().any():

    raise ValueError(
        "Existen identificadores internos duplicados."
    )

# Ejecución 5.4. Construcción del diccionario de correspondencia ----------
#
# Objetivo:
# Construir la correspondencia entre el identificador oficial del
# nodo y el identificador interno del grafo.
#
# Producto:
# - graph_node_mapping

graph_node_mapping = dict(
    zip(
        nodes[
            NODE_CONFIG["node_id_column"]
        ],
        nodes[
            NODE_CONFIG["graph_node_id"]
        ]
    )
) # Mapeo identificador oficial -> graph_node_id

print(
    "Diccionario de correspondencia construido correctamente."
)

# Ejecución 5.5. Registro de dimensiones ---------------------------------
## Objetivo:
# Registrar el tamaño del catálogo oficial de nodos.

num_nodes = len(
    nodes
) # Número de nodos

print(
    f"Número de nodos      : {num_nodes:,}"
)

print(
    f"Variables del nodo   : {len(GRAPH_NODE_ATTRIBUTES)}"
)

# Ejecución 5.6. Confirmación del bloque ---------------------------------
print(
    "\nConstrucción de los nodos finalizada correctamente."
)

print("-" * 80)

# BLOQUE 6. Construcción de las Aristas -----------------------------------
# Objetivo: Construir la estructura de conectividad del grafo mediante las relaciones
# espaciales entre municipios para generar edge_index y edge_weight compatibles con PyTorch Geometric.
## Pregunta científica: ¿Cómo representar las relaciones espaciales entre los municipios mediante
# un conjunto de aristas que preserve la estructura del territorio?

print("\n" + "-" * 80)
print("BLOQUE 6. CONSTRUCCIÓN DE LAS ARISTAS")
print("-" * 80)

# Ejecución 6.1. Validación de la Información Espacial --------------------
#
# Objetivo:
# Verificar que el catálogo oficial de nodos contiene la información
# espacial necesaria para construir la vecindad del grafo.
#
# Entradas:
# - nodes
# - NODE_CONFIG
# - GRAPH_SPATIAL_CONFIG
#
# Producto:
# - Catálogo espacial validado.
#
# Responde:
# ¿El catálogo oficial de nodos contiene la información necesaria para
# construir la estructura espacial del grafo?

print(
    "\nValidando información espacial...\n"
)

# Configuración oficial
node_id_column = (
    NODE_CONFIG["node_id_column"]
) # Identificador oficial del nodo

graph_node_id = (
    NODE_CONFIG["graph_node_id"]
) # Identificador interno del grafo

geometry_column = (
    GRAPH_SPATIAL_CONFIG["geometry_column"]
) # Columna de geometría

# Validar columnas obligatorias
required_columns = [
    node_id_column,
    graph_node_id,
    geometry_column
] # Columnas requeridas

missing_columns = [
    column
    for column in required_columns
    if column not in nodes.columns
] # Columnas faltantes

if missing_columns:

    raise ValueError(
        f"Faltan columnas obligatorias en el catálogo de nodos: {missing_columns}"
    )

# Validar identificadores internos faltantes
missing_graph_nodes = (
    nodes[
        graph_node_id
    ]
    .isna()
    .sum()
) # graph_node_id faltantes

if missing_graph_nodes > 0:

    raise ValueError(
        f"Se encontraron {missing_graph_nodes:,} identificadores internos faltantes."
    )

# Validar geometrías faltantes
missing_geometry = (
    nodes[
        geometry_column
    ]
    .isna()
    .sum()
) # Geometrías faltantes

if missing_geometry > 0:

    raise ValueError(
        f"Se encontraron {missing_geometry:,} geometrías faltantes."
    )

print(
    f"Método espacial          : {GRAPH_SPATIAL_CONFIG['method'].upper()}"
)

print(
    f"Número de nodos          : {num_nodes:,}"
)

print(
    "Información espacial validada correctamente.\n"
)

# ---------------------------------
# Ejecución 6.2. Construcción de la Vecindad Espacial ---------------------
#
# Objetivo:
# Construir la estructura oficial de vecindad espacial del grafo a partir
# de la geometría de los municipios utilizando el método definido en la
# configuración del proyecto.
#
# Entradas:
# - nodes
# - GRAPH_SPATIAL_CONFIG
#
# Producto:
# - spatial_weights
#
# Responde:
# ¿Cuál es la estructura de vecindad espacial entre los municipios?

print(
    "\nConstruyendo vecindad espacial...\n"
)

nodes["geometry"] = nodes["geometry"].apply(wkb.loads)

nodes = gpd.GeoDataFrame(
    nodes,
    geometry="geometry",
    crs=f"EPSG:{GRAPH_SPATIAL_CONFIG['crs']}"
)

# Configuración oficial
graph_method = (
    GRAPH_SPATIAL_CONFIG["method"]
) # Método de construcción de la vecindad

# Convertir el catálogo de nodos a GeoDataFrame
nodes = gpd.GeoDataFrame(
    nodes,
    geometry = GRAPH_SPATIAL_CONFIG["geometry_column"],
    crs = GRAPH_SPATIAL_CONFIG["crs"]
) # Catálogo espacial de nodos

# Construir la estructura de vecindad
if graph_method == "queen":

    spatial_weights = Queen.from_dataframe(
        nodes
    ) # Vecindad Queen

elif graph_method == "rook":

    spatial_weights = Rook.from_dataframe(
        nodes
    ) # Vecindad Rook

elif graph_method == "knn":

    spatial_weights = KNN.from_dataframe(
        nodes,
        k = GRAPH_NEIGHBORS["k_neighbors"]
    ) # Vecindad K vecinos

else:

    raise ValueError(
        f"Método de vecindad no soportado: {graph_method}"
    )

# Registrar resultados
print(
    f"Método espacial      : {graph_method.upper()}"
)

print(
    f"Número de nodos      : {spatial_weights.n:,}"
)

print(
    "Vecindad espacial construida correctamente.\n"
)

# Ejecución 6.3. Construcción de la Lista Oficial de Aristas --------------
#
# Objetivo:
# Construir la lista oficial de aristas a partir de la estructura de
# vecindad espacial del grafo.
#
# Entradas:
# - spatial_weights
#
# Producto:
# - edge_list
#
# Responde:
# ¿Qué pares de nodos se encuentran conectados espacialmente?

print(
    "\nConstruyendo lista oficial de aristas...\n"
)

# Inicializar lista de aristas
edge_list = []

# Recorrer la estructura de vecindad
for source, neighbors in spatial_weights.neighbors.items():

    for target in neighbors:

        edge_list.append(
            (
                source,
                target
            )
        ) # Arista espacial

# Validar existencia de aristas
if len(edge_list) == 0:

    raise ValueError(
        "No se construyeron aristas espaciales."
    )

print(
    f"Número de aristas : {len(edge_list):,}"
)

print(
    "Lista oficial de aristas construida correctamente.\n"
)

# Ejecución 6.4. Construcción del edge_index ------------------------------
#
# Objetivo:
# Construir la representación matricial de las aristas del grafo en el
# formato requerido por PyTorch Geometric.
#
# Entradas:
# - edge_list
#
# Producto:
# - edge_index
#
# Responde:
# ¿Las relaciones espaciales fueron representadas correctamente mediante
# la estructura edge_index?

print(
    "\nConstruyendo edge_index...\n"
)

# Construir la matriz oficial de aristas
edge_index = torch.tensor(
    edge_list,
    dtype = torch.long
).t().contiguous() # Matriz oficial de aristas

# Validar dimensiones
if edge_index.ndim != 2:

    raise ValueError(
        "edge_index debe ser una matriz bidimensional."
    )

if edge_index.shape[0] != 2:

    raise ValueError(
        "edge_index debe tener dimensión (2, n_aristas)."
    )

# Validar existencia de aristas
if edge_index.shape[1] == 0:

    raise ValueError(
        "No se construyeron aristas para el grafo."
    )

print(
    f"Dimensiones edge_index : {tuple(edge_index.shape)}"
)

print(
    "edge_index construido correctamente.\n"
)

# Ejecución 6.5. Construcción del edge_weight -----------------------------
#
# Objetivo:
# Construir el vector oficial de pesos asociado a las aristas del grafo
# para su utilización en PyTorch Geometric.
#
# Entradas:
# - edge_index
# - GRAPH_CONFIG
#
# Producto:
# - edge_weight
#
# Responde:
# ¿Cada arista del grafo posee un peso válido para el proceso de aprendizaje?

print(
    "\nConstruyendo edge_weight...\n"
)

# Construir pesos oficiales de las aristas
edge_weight = torch.ones(
    edge_index.shape[1],
    dtype = torch.float32
) # Peso unitario para cada arista

# Validar dimensiones
if edge_weight.shape[0] != edge_index.shape[1]:

    raise ValueError(
        "La longitud de edge_weight no coincide con el número de aristas."
    )

# Validar valores faltantes
if torch.isnan(edge_weight).any():

    raise ValueError(
        "Se encontraron valores faltantes en edge_weight."
    )

print(
    f"Número de pesos : {edge_weight.shape[0]:,}"
)

print(
    "edge_weight construido correctamente.\n"
)

# Ejecución 6.6. Validación Integral del Grafo ----------------------------
#
# Objetivo:
# Verificar la consistencia estructural del grafo antes de construir el
# objeto GraphData compatible con PyTorch Geometric.
#
# Entradas:
# - edge_index
# - edge_weight
# - num_nodes
#
# Producto:
# - Grafo validado.
#
# Responde:
# ¿La estructura del grafo cumple los requisitos de PyTorch Geometric?

print(
    "\nValidando estructura del grafo...\n"
)

# Validar tipo de dato del edge_index
if edge_index.dtype != torch.long:

    raise TypeError(
        "edge_index debe tener tipo torch.long."
    )

# Validar tipo de dato del edge_weight
if edge_weight.dtype != torch.float32:

    raise TypeError(
        "edge_weight debe tener tipo torch.float32."
    )

# Validar índices negativos
if torch.any(edge_index < 0):

    raise ValueError(
        "Se encontraron índices negativos en edge_index."
    )

# Validar rango de los nodos
if torch.max(edge_index) >= num_nodes:

    raise ValueError(
        "Existen identificadores de nodos fuera del rango permitido."
    )

# Validar correspondencia entre aristas y pesos
if edge_index.shape[1] != edge_weight.shape[0]:

    raise ValueError(
        "El número de pesos no coincide con el número de aristas."
    )

# Validar valores faltantes
if torch.isnan(edge_weight).any():

    raise ValueError(
        "Se encontraron valores faltantes en edge_weight."
    )

print(
    f"Número de nodos      : {num_nodes:,}"
)

print(
    f"Número de aristas    : {edge_index.shape[1]:,}"
)

print(
    "Estructura del grafo validada correctamente.\n"
)

# Ejecución 6.7. Registro de Métricas del Grafo ---------------------------
#
# Objetivo:
# Calcular y registrar las principales métricas estructurales del grafo
# para caracterizar su topología antes de construir el GraphData.
#
# Entradas:
# - num_nodes
# - edge_index
#
# Producto:
# - num_edges
# - average_degree
# - graph_density
#
# Responde:
# ¿Cuáles son las principales características estructurales del grafo?

print(
    "\nRegistrando métricas del grafo...\n"
)

# Número de aristas
num_edges = (
    edge_index.shape[1]
) # Total de aristas

# Grado promedio
average_degree = (
    (2 * num_edges)
    / num_nodes
) # Grado promedio de los nodos

# Densidad del grafo
graph_density = (
    (2 * num_edges)
    /
    (
        num_nodes
        * (num_nodes - 1)
    )
) # Densidad del grafo

print(
    f"Número de nodos      : {num_nodes:,}"
)

print(
    f"Número de aristas    : {num_edges:,}"
)

print(
    f"Grado promedio       : {average_degree:.2f}"
)

print(
    f"Densidad del grafo   : {graph_density:.6f}"
)

print(
    "\nMétricas del grafo registradas correctamente.\n"
)

# Ejecución 6.8. Confirmación del Bloque ----------------------------------
#
# Objetivo:
# Confirmar que la construcción de la estructura del grafo finalizó
# correctamente y que los objetos requeridos para construir el GraphData
# están disponibles.
#
# Producto:
# - Bloque de construcción de aristas finalizado.
#
# Responde:
# ¿La estructura del grafo está lista para construir el GraphData?

print(
    "\nConstrucción de las aristas finalizada correctamente."
)

print(
    "Objetos generados:"
)

print(
    "  - nodes"
)

print(
    "  - spatial_weights"
)

print(
    "  - edge_list"
)

print(
    "  - edge_index"
)

print(
    "  - edge_weight"
)

print("-" * 80)

# BLOQUE 7. Construcción y Validación del GraphData ------------------------
## Objetivo:
# Integrar las características de los nodos, la estructura del grafo y la
# variable objetivo en un objeto GraphData compatible con PyTorch Geometric.
## Pregunta científica:
# ¿El objeto GraphData representa correctamente la estructura espacial del
# sistema de municipios y está listo para el entrenamiento de una
# Graph Neural Network?
## Entradas:
# - dataset
# - edge_index
# - NODE_FEATURE_COLUMNS
# - TARGET_VARIABLE
## Producto generado:
# - graph_data
## Ejecuciones:
#   7.1 Construcción de las Node Features.
#   7.2 Construcción de la Variable Objetivo.
#   7.3 Conversión a Tensores.
#   7.4 Construcción del GraphData.
#   7.5 Validación del GraphData.
#   7.6 Registro de Métricas del GraphData.
#   7.7 Confirmación del Bloque.

print("\n" + "-" * 80)
print("BLOQUE 7. CONSTRUCCIÓN Y VALIDACIÓN DEL GRAPHDATA")
print("-" * 80)

# Ejecución 7.1. Construcción de las Node Features ------------------------
#
# Objetivo:
# Construir la matriz oficial de características (Node Features)
# asociada a cada nodo del grafo utilizando el mismo orden del
# catálogo oficial de nodos.
#
# Entradas:
# - dataset
# - nodes
# - NODE_CONFIG
# - NODE_FEATURE_COLUMNS
#
# Producto:
# - x_numpy
#
# Responde:
# ¿Cada nodo del grafo posee un único conjunto de variables predictoras
# alineado con la estructura del GraphData?

print(
    "\nConstruyendo Node Features...\n"
)

# Construcción de las Node Features por municipio
node_features = (
    dataset
    .groupby(
        NODE_CONFIG["node_id_column"]
    )[NODE_FEATURE_COLUMNS]
    .mean()
    .reset_index()
) # Variables predictoras agregadas por municipio

# Alinear con el orden oficial del catálogo de nodos
node_features = (
    nodes[
        [
            NODE_CONFIG["node_id_column"],
            NODE_CONFIG["graph_node_id"]
        ]
    ]
    .merge(
        node_features,
        on = NODE_CONFIG["node_id_column"],
        how = "left"
    )
    .sort_values(
        by = NODE_CONFIG["graph_node_id"]
    )
    .reset_index(
        drop = True
    )
) # Node Features ordenadas según graph_node_id

# Validar número de nodos
if len(node_features) != num_nodes:

    raise ValueError(
        "El número de Node Features no coincide con el número de nodos."
    )

# Validar valores faltantes
missing_features = (
    node_features[
        NODE_FEATURE_COLUMNS
    ]
    .isna()
    .sum()
    .sum()
) # Total de valores faltantes

if missing_features > 0:

    raise ValueError(
        f"Se encontraron {missing_features:,} valores faltantes en las Node Features."
    )

# Construir matriz oficial
x_numpy = (
    node_features[
        NODE_FEATURE_COLUMNS
    ]
    .to_numpy(
        dtype = np.float32
    )
) # Matriz oficial de Node Features

# Validar dimensiones
if x_numpy.shape[0] != num_nodes:

    raise ValueError(
        "La matriz de Node Features posee un número incorrecto de nodos."
    )

if x_numpy.shape[1] != len(NODE_FEATURE_COLUMNS):

    raise ValueError(
        "La matriz de Node Features posee un número incorrecto de variables."
    )

print(
    f"Número de nodos            : {x_numpy.shape[0]:,}"
)

print(
    f"Número de variables        : {x_numpy.shape[1]}"
)

print(
    "Node Features construidas correctamente.\n"
)

# Ejecución 7.2. Construcción de la Variable Objetivo ---------------------
#
# Objetivo:
# Construir el vector oficial de la variable objetivo asociado a cada
# nodo del grafo utilizando el mismo orden del catálogo oficial de nodos.
#
# Entradas:
# - dataset
# - nodes
# - NODE_CONFIG
# - TARGET_VARIABLE
#
# Producto:
# - y_numpy
#
# Responde:
# ¿Cada nodo del grafo posee un único valor objetivo alineado con la
# estructura del GraphData?

print(
    "\nConstruyendo variable objetivo...\n"
)

# Construcción de la variable objetivo por municipio
target_data = (
    dataset
    .groupby(
        NODE_CONFIG["node_id_column"]
    )[[TARGET_VARIABLE]]
    .mean()
    .reset_index()
) # Variable objetivo agregada por municipio

# Alinear con el catálogo oficial de nodos
target_data = (
    nodes[
        [
            NODE_CONFIG["node_id_column"],
            NODE_CONFIG["graph_node_id"]
        ]
    ]
    .merge(
        target_data,
        on = NODE_CONFIG["node_id_column"],
        how = "left"
    )
    .sort_values(
        by = NODE_CONFIG["graph_node_id"]
    )
    .reset_index(
        drop = True
    )
) # Variable objetivo ordenada según graph_node_id

# Validar número de nodos
if len(target_data) != num_nodes:

    raise ValueError(
        "El número de observaciones de la variable objetivo no coincide con el número de nodos."
    )

# Validar valores faltantes
missing_target = (
    target_data[
        TARGET_VARIABLE
    ]
    .isna()
    .sum()
) # Valores faltantes

if missing_target > 0:

    raise ValueError(
        f"Se encontraron {missing_target:,} valores faltantes en la variable objetivo."
    )

# Construir vector oficial
y_numpy = (
    target_data[
        TARGET_VARIABLE
    ]
    .to_numpy(
        dtype = np.float32
    )
) # Vector oficial de la variable objetivo

# Validar dimensiones
if y_numpy.shape[0] != num_nodes:

    raise ValueError(
        "El vector de la variable objetivo posee un número incorrecto de nodos."
    )

print(
    f"Número de nodos      : {y_numpy.shape[0]:,}"
)

print(
    f"Variable objetivo    : {TARGET_VARIABLE}"
)

print(
    "Variable objetivo construida correctamente.\n"
)

# Ejecución 7.3. Conversión a Tensores -----------------------------------
#
# Objetivo:
# Convertir las Node Features, la variable objetivo, la estructura de
# aristas y los pesos del grafo al formato tensor requerido por
# PyTorch Geometric.
#
# Entradas:
# - x_numpy
# - y_numpy
# - edge_index
# - edge_weight
#
# Producto:
# - x
# - y
# - edge_index
# - edge_weight
#
# Responde:
# ¿La información del grafo fue convertida correctamente al formato
# tensor requerido por PyTorch Geometric?

print(
    "\nConvirtiendo estructuras a tensores...\n"
)

# Node Features
x = torch.tensor(
    x_numpy,
    dtype = torch.float32
) # Tensor de Node Features

# Variable objetivo
y = torch.tensor(
    y_numpy,
    dtype = torch.float32
) # Tensor de la variable objetivo

# Validar dimensiones
if x.shape[0] != num_nodes:

    raise ValueError(
        "El tensor x posee un número incorrecto de nodos."
    )

if y.shape[0] != num_nodes:

    raise ValueError(
        "El tensor y posee un número incorrecto de nodos."
    )

# Validar tipos de datos
if x.dtype != torch.float32:

    raise TypeError(
        "El tensor x debe tener tipo torch.float32."
    )

if y.dtype != torch.float32:

    raise TypeError(
        "El tensor y debe tener tipo torch.float32."
    )

print(
    f"Tensor x              : {tuple(x.shape)}"
)

print(
    f"Tensor y              : {tuple(y.shape)}"
)

print(
    "Conversión a tensores completada correctamente.\n"
)

# Ejecución 7.4. Construcción del GraphData -------------------------------
#
# Objetivo:
# Integrar las características de los nodos, la estructura del grafo,
# los pesos de las aristas y la variable objetivo en un objeto
# GraphData compatible con PyTorch Geometric.
#
# Entradas:
# - x
# - edge_index
# - edge_weight
# - y
#
# Producto:
# - graph_data
#
# Responde:
# ¿La información del grafo fue integrada correctamente en un objeto
# GraphData compatible con PyTorch Geometric?

print(
    "\nConstruyendo GraphData...\n"
)

# Construir GraphData
graph_data = Data(
    x = x,
    edge_index = edge_index,
    edge_attr = edge_weight,
    edge_weight = edge_weight,
    y = y
) # GraphData oficial

# Validar existencia del GraphData
if graph_data is None:

    raise ValueError(
        "No fue posible construir el objeto GraphData."
    )

# Validar componentes principales
required_attributes = [
    "x",
    "edge_index",
    "edge_attr",
    "y"
] # Atributos obligatorios

missing_attributes = [
    attribute
    for attribute in required_attributes
    if not hasattr(
        graph_data,
        attribute
    )
] # Atributos faltantes

if missing_attributes:

    raise ValueError(
        f"GraphData no contiene los atributos requeridos: {missing_attributes}"
    )

print(
    "GraphData construido correctamente.\n"
)

# Ejecución 7.5. Validación Integral del GraphData ------------------------
#
# Objetivo:
# Verificar que el objeto GraphData cumple los requisitos estructurales
# necesarios para el entrenamiento de una Graph Neural Network.
#
# Entradas:
# - graph_data
# - num_nodes
# - NODE_FEATURE_COLUMNS
#
# Producto:
# - GraphData validado.
#
# Responde:
# ¿El objeto GraphData es consistente y está listo para ser utilizado
# por PyTorch Geometric?

print(
    "\nValidando GraphData...\n"
)

# Validar número de nodos
if graph_data.num_nodes != num_nodes:

    raise ValueError(
        "El número de nodos del GraphData es inconsistente."
    )

# Validar dimensiones de las Node Features
if graph_data.x.shape[0] != num_nodes:

    raise ValueError(
        "Las Node Features poseen un número incorrecto de nodos."
    )

if graph_data.x.shape[1] != len(NODE_FEATURE_COLUMNS):

    raise ValueError(
        "El número de Node Features es inconsistente."
    )

# Validar variable objetivo
if graph_data.y.shape[0] != num_nodes:

    raise ValueError(
        "La variable objetivo posee un número incorrecto de nodos."
    )

# Validar número de aristas
if graph_data.edge_index.shape[1] != graph_data.edge_attr.shape[0]:

    raise ValueError(
        "El número de pesos no coincide con el número de aristas."
    )

# Validar valores faltantes
if torch.isnan(graph_data.x).any():

    raise ValueError(
        "Las Node Features contienen valores faltantes."
    )

if torch.isnan(graph_data.y).any():

    raise ValueError(
        "La variable objetivo contiene valores faltantes."
    )

if torch.isnan(graph_data.edge_attr).any():

    raise ValueError(
        "Los pesos de las aristas contienen valores faltantes."
    )

print(
    "GraphData validado correctamente.\n"
)

# Ejecución 7.6. Registro de Métricas del GraphData -----------------------
#
# Objetivo:
# Registrar las principales características del objeto GraphData para
# documentar su estructura antes del entrenamiento de la Graph Neural
# Network.
#
# Entradas:
# - graph_data
# - TARGET_VARIABLE
# - GRAPH_CONFIG
#
# Producto:
# - Métricas del GraphData.
#
# Responde:
# ¿Cuáles son las principales características del GraphData construido?

print(
    "\nRegistrando métricas del GraphData...\n"
)

print(
    f"Número de nodos          : {graph_data.num_nodes:,}"
)

print(
    f"Número de aristas        : {graph_data.num_edges:,}"
)

print(
    f"Node Features            : {graph_data.num_node_features}"
)

print(
    f"Variable objetivo        : {TARGET_VARIABLE}"
)

print(
    f"Grafo dirigido           : {GRAPH_CONFIG['directed']}"
)

print(
    f"Grafo ponderado          : {GRAPH_CONFIG['weighted']}"
)

print(
    "\nMétricas del GraphData registradas correctamente.\n"
)

# Ejecución 7.7. Exportación del GraphData --------------------------------
## Objetivo:
# Exportar el GraphData oficial para que pueda ser utilizado durante
# las etapas posteriores del pipeline científico.
#
# Producto:
# - graph_data.pt
#
# Responde:
# ¿El GraphData fue almacenado correctamente?

torch.save(
    graph_data,
    GRAPH_DATA_FILE
) # Exportar GraphData oficial

print(
    f"\nGraphData exportado : {GRAPH_DATA_FILE}"
)

if not GRAPH_DATA_FILE.exists():

    raise FileNotFoundError(
        "No fue posible exportar el GraphData."
    )

print(
    "GraphData exportado correctamente."
)

# Ejecución 7.8. Confirmación del Bloque ----------------------------------
#
# Objetivo:
# Confirmar la construcción exitosa del GraphData y registrar los objetos
# generados que serán utilizados durante el entrenamiento de la
# Graph Neural Network.
#
# Producto:
# - Bloque de construcción del GraphData finalizado.
#
# Responde:
# ¿El GraphData está completamente construido y listo para el entrenamiento?

print(
    "\nConstrucción del GraphData finalizada correctamente."
)

graph_objects = [
    "nodes",
    "graph_node_mapping",
    "edge_index",
    "edge_weight",
    "x",
    "y",
    "graph_data"
] # Objetos generados durante el proceso

print(
    "\nObjetos generados:"
)

for obj in graph_objects:

    print(
        f"  - {obj}"
    )

print(
    "\nEl GraphData está listo para el entrenamiento de la Graph Neural Network."
)

print("-" * 80)