# graph-01_build_graph.py

# Librerías estándar
import warnings
from pathlib import Path

# Librerías científicas
import numpy as np
import pandas as pd
import json
from datetime import datetime

import torch
from torch_geometric.data import Data

# Librerías de visualización
import matplotlib.pyplot as plt
import mapclassify

# Librerías geoespaciales
import geopandas as gpd
from shapely import wkb
from libpysal.weights import (
    KNN,
    Queen,
    Rook,
)

# Librerías para grafos
import networkx as nx

# Configuración global
warnings.filterwarnings("ignore")

# Utilidades Oficiales del Proyecto --------------------------------------
from src.python.utils.data_preparation import (
    load_dataset,
    validate_dataset,
)  # Carga y validación del Dataset Científico

# Configuracion Oficial del Proyecto -------------------------------------
from src.python.config.config_project import (
    PROJECT_SEED,
    TARGET_VARIABLE,
    FEATURE_COLUMNS,
    NODE_FEATURE_CONFIG,
    GRAPH_CONFIG,
    GRAPH_SPATIAL_CONFIG,
    GRAPH_OUTPUTS,
    NODE_ID_COLUMN,
    NODE_INDEX_COLUMN,
    GRAPH_NODE_KEY_COLUMNS,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    NODE_CATALOG_CONFIG,
    GRAPH_VISUALIZATION_CONFIG
)

from src.python.config.paths import (
    validate_project_structure,
    GRAPH_DATA_DIR,
    METADATA_DIR,
    AUDITS_DIR,
    GRAPHS_DIR,
)

# BLOQUE 2. Configuración del Script --------------------------------------
# Objetivo: Configurar el entorno de ejecución, validar la estructura oficial del proyecto e inicializar
#  los parámetros necesarios para construir los GraphData anuales de forma reproducible, siguiendo la arquitectura
# científica del proyecto.
# Arquitectura científica: - Nodo científico: Municipio + Año. # - Catálogo Oficial de Nodos: Espacio-Temporal
# - key_columns: ["cod_mpio", "anio"]. # - Un GraphData independiente por cada año del panel
# Producto: - entorno de ejecución configurado - estructura oficial del proyecto validada
# - reproducibilidad establecida - dispositivo de procesamiento seleccionado
# Pregunta científica: ¿El entorno de ejecución cumple las condiciones necesarias para construir
# de forma reproducible el Catálogo Oficial de Nodos Espacio-Temporales y los GraphData anuales?
# 2.1 Configuración del entorno -------------------------------------------
warnings.filterwarnings("ignore")  # Ocultar advertencias
print("Entorno de ejecución configurado correctamente.")

# 2.2 Validación de la estructura del proyecto ----------------------------
validate_project_structure(verbose=True)  # Validar estructura oficial del proyecto
print("Estructura del proyecto validada correctamente.")

# 2.3 Configuración de la reproducibilidad -------------------------------
torch.manual_seed(PROJECT_SEED)  # Configurar semilla de PyTorch
cuda_available = torch.cuda.is_available()  # Verificar disponibilidad de CUDA

if cuda_available:
    torch.cuda.manual_seed(PROJECT_SEED)  # Configurar semilla de la GPU
    torch.cuda.manual_seed_all(PROJECT_SEED)  # Configurar todas las GPU
    torch.backends.cudnn.deterministic = True  # Forzar reproducibilidad
    torch.backends.cudnn.benchmark = False  # Desactivar optimizaciones

print(f"Semilla del proyecto          : {PROJECT_SEED}")

# 2.4 Selección del dispositivo ------------------------------------------
DEVICE = "cuda" if cuda_available else "cpu"  # Dispositivo de procesamiento
print(f"Dispositivo de procesamiento  : {DEVICE.upper()}")

# BLOQUE 3. Carga del Dataset Científico -----------------------------------
## Objetivo: Cargar y validar el Dataset Científico oficial que servirá como base para construir el
# catálogo de nodos, las Node Features, la estructura espacial del grafo y el GraphData.
# Producto: - dataset
## Pregunta científica: ¿El Dataset Científico oficial fue cargado y validado correctamente para
# iniciar la construcción del GraphData?
print("\n" + "-" * 80)
print("BLOQUE 3. CARGA DEL DATASET CIENTÍFICO")
print("-" * 80)

# 3.1 Carga del Dataset Científico ----------------------------------------
dataset = load_dataset()  # Cargar Dataset Científico oficial

# 3.2 Validación del Dataset Científico -----------------------------------
dataset = validate_dataset(dataset)  # Validar estructura del Dataset Científico

print("Dataset Científico cargado correctamente.")
print(f"Registros : {len(dataset):,}")
print(f"Variables : {dataset.shape[1]:,}")

print(f"Variable objetivo : {TARGET_VARIABLE}")

# 3.3 Inspección del Dataset Científico -----------------------------------
print("\nColumnas del Dataset Científico:")

for i, column in enumerate(dataset.columns, start=1):
    print(f"{i:2d}. {column}")

print(f"\nRegistros          : {len(dataset):,}")
print(f"Total de columnas  : {len(dataset.columns)}")

print("-" * 80)

# 3.4 Confirmación del bloque ---------------------------------------------
print("\nCarga del Dataset Científico finalizada correctamente.")
print("-" * 80)

# BLOQUE 4. Validación para la Construcción del GraphData -------------------
## Objetivo: Verificar que el Dataset Científico contiene la información necesaria para construir
# la estructura espacial del grafo y el objeto GraphData.
# Producto: - Dataset validado para la construcción del GraphData
## Pregunta científica: ¿El Dataset Científico contiene toda la información requerida para
# construir correctamente el GraphData?
print("\n" + "-" * 80)
print("BLOQUE 4. VALIDACIÓN PARA LA CONSTRUCCIÓN DEL GRAPHDATA")
print("-" * 80)

# 4.1 Información espacial -------------------------------------------------
spatial_columns = [
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    GRAPH_SPATIAL_CONFIG["geometry_column"],
]  # Variables espaciales obligatorias

missing_spatial = [
    column
    for column in spatial_columns
    if column not in dataset.columns
]  # Variables espaciales faltantes

if missing_spatial:
    raise ValueError(
        "Faltan las siguientes variables espaciales:\n"
        + "\n".join(missing_spatial)
    )

print(f"Variables espaciales : {len(spatial_columns)}")
print("Información espacial verificada.")
print(f"Columna latitud   : {LATITUDE_COLUMN}")
print(f"Columna longitud  : {LONGITUDE_COLUMN}")
print(f"Columna geometría : {GRAPH_SPATIAL_CONFIG['geometry_column']}")

# 4.2 Validación de geometrías --------------------------------------------
missing_geometry = (
    dataset[GRAPH_SPATIAL_CONFIG["geometry_column"]]
    .isna()
    .sum()
)

if missing_geometry > 0:
    raise ValueError(
        f"Se encontraron {missing_geometry:,} geometrías faltantes."
    )

print("Geometrías verificadas.")

# 4.3 Confirmación ---------------------------------------------------------
print("\nDataset preparado para construir el GraphData.")
print("-" * 80)

# BLOQUE 5. Construcción del Catálogo Oficial de Nodos ---------------------
# Objetivo: Construir el Catálogo Oficial de Nodos Espacio-Temporales a partir del Dataset Científico,
# asignando un identificador científico único e irrepetible a cada observación (Municipio + Año) y 
# un identificador interno para su utilización durante la construcción de los GraphData anuales.
# Arquitectura científica: - Unidad de análisis: Observación espacio-temporal. - Nodo científico: Municipio + Año
# - Catálogo Oficial: Espacio-Temporal. - node_id: cod_mpio_anio. - node_index: Índice interno global
# - GraphData: Un grafo independiente por cada año
# Producto: - node_catalog. - graph_node_mapping. - num_nodes.
# Pregunta científica: ¿Cada observación espacio-temporal del Dataset Científico posee un identificador científico
# único, reproducible y trazable que permita construir de forma consistente los GraphData anuales?

print("\n" + "-" * 80)
print("BLOQUE 5. CONSTRUCCIÓN DEL CATÁLOGO DE NODOS")
print("-" * 80)

# 5.1 Construcción del Catálogo Oficial de Nodos ---------------------------
# Construir el Catálogo Oficial de Nodos Espacio-Temporales a partir del Dataset Científico, utilizando la
# clave científica definida en NODE_CATALOG_CONFIG["key_columns"].
node_catalog = (
    dataset[NODE_CATALOG_CONFIG["source_columns"]]
    .drop_duplicates(
        subset=NODE_CATALOG_CONFIG["key_columns"]
    )
    .sort_values(
        by=NODE_CATALOG_CONFIG["key_columns"]
    )
    .reset_index(drop=True)
)

# Validación de existencia del catálogo
if node_catalog.empty:
    raise ValueError(
        "El Catálogo Oficial de Nodos Espacio-Temporales está vacío."
    )

# Validación de unicidad de la clave científica
if node_catalog.duplicated(
    subset=NODE_CATALOG_CONFIG["key_columns"]
).any():
    raise ValueError(
        "Existen nodos espacio-temporales duplicados en el Catálogo Oficial."
    )

print("Catálogo Oficial de Nodos Espacio-Temporales construido correctamente.")
print(f"Nodos espacio-temporales : {len(node_catalog):,}")
print(f"Atributos del catálogo   : {len(node_catalog.columns)}")

# 5.2 Construcción del Identificador Científico ----------------------------
# Construir el identificador científico único de cada nodo espacio-temporal
# utilizando las columnas definidas en la configuración oficial del proyecto.
node_catalog[NODE_ID_COLUMN] = (
    node_catalog[NODE_CATALOG_CONFIG["key_columns"]]
    .astype(str)
    .agg("_".join, axis=1)
)

# Validación de valores nulos
if node_catalog[NODE_ID_COLUMN].isna().any():
    raise ValueError(
        "Existen identificadores científicos nulos."
    )

# Validación de unicidad
if node_catalog[NODE_ID_COLUMN].duplicated().any():
    raise ValueError(
        "Existen identificadores científicos duplicados."
    )

print("Identificador científico construido correctamente.")
print(
    f"Identificadores científicos únicos : "
    f"{node_catalog[NODE_ID_COLUMN].nunique():,}"
)

# 5.3 Ordenamiento del Catálogo Oficial -----------------------------------
# Ordenar el Catálogo Oficial de Nodos Espacio-Temporales utilizando el
# identificador científico para garantizar una numeración reproducible.
node_catalog = (
    node_catalog
    .sort_values(by=NODE_ID_COLUMN)
    .reset_index(drop=True)
)

print("Catálogo Oficial de Nodos ordenado correctamente.")

# 5.4 Creación del Identificador Interno ----------------------------------
# Asignar un identificador interno global a cada nodo del Catálogo Oficial. Este identificador será
# utilizado como referencia maestra dentro del proyecto y posteriormente será remapeado a un índice 
# local durante la construcción de cada GraphData anual.
node_catalog[NODE_INDEX_COLUMN] = range(len(node_catalog))
print("Identificador interno global construido correctamente.")

# 5.5 Construcción del Diccionario de Correspondencia ----------------------
# Construir el diccionario oficial de correspondencia entre el identificador científico (node_id)
# y el identificador interno global (node_index) del Catálogo Oficial de Nodos.
graph_node_mapping = dict(
    zip(
        node_catalog[NODE_ID_COLUMN],
        node_catalog[NODE_INDEX_COLUMN]
    )
)

# Validación de integridad del diccionario
if len(graph_node_mapping) != len(node_catalog):
    raise ValueError(
        "El diccionario de correspondencia contiene identificadores científicos duplicados."
    )

print("Diccionario de correspondencia construido correctamente.")
print(f"Correspondencias registradas : {len(graph_node_mapping):,}")

# 5.6 Registro del Catálogo Oficial ----------------------------------------
# Registrar las métricas principales del Catálogo Oficial de Nodos Espacio-Temporales.
num_nodes = len(node_catalog)

print("Resumen del Catálogo Oficial de Nodos")
print(f"Nodos espacio-temporales : {num_nodes:,}")
print(f"Municipios              : {node_catalog['cod_mpio'].nunique():,}")
print(f"Años                    : {node_catalog['anio'].nunique():,}")
print(f"Atributos del catálogo  : {len(node_catalog.columns)}")

# 5.7 Confirmación ---------------------------------------------------------
# Confirmar la integridad del Catálogo Oficial de Nodos Espacio-Temporales.
if node_catalog[NODE_ID_COLUMN].isna().any():
    raise ValueError(
        "Existen identificadores científicos nulos."
    )

# Validación de completitud del panel espacio-temporal
expected_nodes = (
    node_catalog[NODE_CATALOG_CONFIG["key_columns"][0]].nunique()
    * node_catalog[NODE_CATALOG_CONFIG["key_columns"][1]].nunique()
)

if len(node_catalog) != expected_nodes:
    raise ValueError(
        "El Catálogo Oficial no contiene todas las combinaciones de la clave científica."
    )

# BLOQUE 6. Construcción de la Estructura Espacial del GraphData ----------------------------------------
# Objetivo: Construir la estructura de conectividad espacial del GraphData correspondiente al año de
# procesamiento (current_year), generando edge_index y edge_weight compatibles con PyTorch Geometric.
# Arquitectura científica:
# - Entrada: • Dataset Científico. • Catálogo Oficial de Nodos. - Filtro: • current_year
# - Salida: • edge_index. • edge_weight.
# Producto: - spatial_nodes. - spatial_weights. - edge_list.
# Pregunta científica: ¿Cómo representar las relaciones espaciales entre los nodos del GraphData del 
# año current_year preservando la estructura territorial?
print("\n" + "-" * 80)
print("BLOQUE 6. CONSTRUCCIÓN DE LA ESTRUCTURA DEL GRAFO")
print("-" * 80)

# 6.0 Preparación de la estructura espacial permanente
geometry_column = GRAPH_SPATIAL_CONFIG["geometry_column"]
municipality_positions = (
    dataset[
        [
            "cod_mpio",
            geometry_column
        ]
    ]
    .drop_duplicates("cod_mpio")
    .copy()
)

municipality_positions[geometry_column] = (
    municipality_positions[geometry_column]
    .apply(wkb.loads)
)

municipality_positions = gpd.GeoDataFrame(
    municipality_positions,
    geometry=geometry_column,
    crs=f"EPSG:{GRAPH_SPATIAL_CONFIG['crs']}"
)

municipality_positions["centroid"] = (
    municipality_positions.geometry.centroid
)

# 6.0.1 Construcción anual de GraphData ---------------------------------------------------------
time_column = NODE_CATALOG_CONFIG["key_columns"][1]
years = sorted(
    dataset[time_column]
    .dropna()
    .unique()
)

if len(years) == 0:
    raise ValueError(
        "No existen años disponibles en el Dataset Científico."
    )

print(f"Años disponibles : {years}")
print(f"Se construirán {len(years)} GraphData.\n")

# 6.1 Construcción anual de GraphData
for current_year in years:
    print("\n" + "-" * 80)
    print(f"INICIANDO GRAPHDATA DEL AÑO {current_year}")
    print("-" * 80)

    # 6.1.1 Filtrado Anual -------------------------------------------------------
    # Objetivo: Filtrar el Dataset Científico y el Catálogo Oficial de Nodos para el
    # año actualmente procesado (current_year), generando las estructuras de trabajo que servirán como
    # entrada para la construcción del GraphData anual.
    # Producto: - dataset_year. - node_catalog_year. - num_nodes.
    # Pregunta científica: ¿Se dispone de un subconjunto consistente del Dataset Científico y del
    # Catálogo Oficial correspondiente al año de procesamiento?
    print(f"\nPreparando GraphData para el año {current_year}...\n")

    # Filtrado del Dataset Científico -----------------------------------------
    dataset_year = (
        dataset[
            dataset[time_column] == current_year
        ]
        .copy()
    )

    if dataset_year.empty:
        raise ValueError(
            f"No existen observaciones para el año {current_year}."
        )

    # Filtrado del Catálogo Oficial -------------------------------------------
    node_catalog_year = (
        node_catalog[
            node_catalog[time_column] == current_year
        ]
        .copy()
    )

    if node_catalog_year.empty:
        raise ValueError(
            f"El Catálogo Oficial no contiene nodos para el año {current_year}."
        )

    # Validación de correspondencia -------------------------------------------
    if len(dataset_year) != len(node_catalog_year):
        raise ValueError(
            "El Dataset Científico y el Catálogo Oficial no tienen el mismo número de nodos para el año en proceso."
        )

    num_nodes = len(node_catalog_year)

    print("6.1.1 Filtrado anual completado.")
    # input("Presione Enter para continuar con 6.1.2...")

    # 6.1.2 Validación del Catálogo Anual ---------------------------------------
    # Objetivo: Verificar la integridad del Catálogo Oficial de Nodos correspondiente al
    # año actualmente procesado antes de construir la estructura espacial.
    # Producto: - node_catalog_year validado.
    # Pregunta científica: ¿El Catálogo Oficial del año de procesamiento contiene todos los
    # identificadores necesarios para construir el GraphData anual?
    print("\nValidando Catálogo Anual...\n")

    required_columns = [
        NODE_ID_COLUMN,
        NODE_INDEX_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in node_catalog_year.columns
    ]

    if missing_columns:
        raise ValueError(
            "Faltan columnas obligatorias en el Catálogo Anual:\n"
            + "\n".join(missing_columns)
        )

    missing_graph_nodes = (
        node_catalog_year[NODE_INDEX_COLUMN]
        .isna()
        .sum()
    )

    if missing_graph_nodes > 0:
        raise ValueError(
            f"Se encontraron {missing_graph_nodes:,} identificadores internos faltantes."
        )

    num_nodes = len(node_catalog_year)

    print("6.1.2 Validación del Catálogo Anual completado.")
    # input("Presione Enter para continuar con 6.1.3...")

    # 6.1.3 Construcción de la Vecindad Espacial -------------------------------
    # Objetivo: Construir la estructura oficial de vecindad espacial del GraphData correspondiente al año 
    # de procesamiento (current_year) utilizando el método espacial definido en la configuración oficial del proyecto.
    # Producto: - spatial_nodes. - spatial_weights.
    # Pregunta científica: ¿Cuál es la estructura de vecindad espacial entre los nodos del
    # GraphData del año de procesamiento?
    print("\nConstruyendo vecindad espacial...\n")

    # Configuración -----------------------------------------------------------
    geometry_column = GRAPH_SPATIAL_CONFIG["geometry_column"]
    if geometry_column not in dataset_year.columns:
        raise ValueError(
            f"No existe la columna '{geometry_column}' en el Dataset Científico anual."
        )

    # Incorporación de la geometría y variables científicas ------------------
    variables_dataset = [
        column
        for column in dataset_year.columns
        if column not in NODE_CATALOG_CONFIG["key_columns"]

    ]

    spatial_nodes = (
        node_catalog_year.merge(
            dataset_year[
                NODE_CATALOG_CONFIG["key_columns"] +
                variables_dataset
            ],
            on=NODE_CATALOG_CONFIG["key_columns"],
            how="left",
            validate="one_to_one"
        )
    )

    # Validación de la integración -------------------------------------------
    if len(spatial_nodes) != num_nodes:
        raise ValueError(
            "Se perdió correspondencia entre el Catálogo Anual y las geometrías."
        )

    if spatial_nodes[geometry_column].isna().any():
        raise ValueError(
            "Existen geometrías faltantes en el Catálogo Anual."
        )

    if TARGET_VARIABLE not in spatial_nodes.columns:
        raise ValueError(
            f"La variable objetivo '{TARGET_VARIABLE}' no fue incorporada a spatial_nodes."
        )

    # Conversión de geometrías -----------------------------------------------
    spatial_nodes[geometry_column] = (
        spatial_nodes[geometry_column]
        .apply(wkb.loads)
    )

    spatial_nodes = gpd.GeoDataFrame(
        spatial_nodes,
        geometry=geometry_column,
        crs=f"EPSG:{GRAPH_SPATIAL_CONFIG['crs']}"
    )

    # Validación del GeoDataFrame --------------------------------------------
    if spatial_nodes.geometry.isnull().any():
        raise ValueError(
            "Se encontraron geometrías nulas después de la conversión WKB."
        )

    if spatial_nodes.crs is None:
        raise ValueError(
            "El GeoDataFrame no tiene un CRS definido."
        )

    # Construcción de la vecindad espacial -----------------------------------
    graph_method = GRAPH_SPATIAL_CONFIG["method"].lower()
    if graph_method == "queen":

        spatial_weights = Queen.from_dataframe(
            spatial_nodes
        )

    elif graph_method == "rook":
        spatial_weights = Rook.from_dataframe(
            spatial_nodes
        )

    elif graph_method == "knn":
        spatial_weights = KNN.from_dataframe(
            spatial_nodes,
            k=GRAPH_SPATIAL_CONFIG["k_neighbors"]
        )

    else:
        raise ValueError(
            f"Método de vecindad no soportado: {graph_method}"
        )

    # Validación --------------------------------------------------------------
    if spatial_weights.n != num_nodes:
        raise ValueError(
            "El número de nodos de la estructura espacial no coincide con el Catálogo Anual."
        )

    print("6.1.3 Construcción de la Vecindad Espacial completado.")
    # input("Presione Enter para continuar con 6.1.4...")

    # 6.1.4 Construcción de la Lista Oficial de Aristas -------------------------
    # Objetivo: Construir la lista oficial de aristas del GraphData correspondiente al
    # año de procesamiento (current_year) a partir de la estructura de vecindad espacial.
    # Producto: - edge_list
    # Pregunta científica: ¿Qué pares de nodos representan correctamente la conectividad espacial
    # del GraphData del año current_year?
    print("\nConstruyendo lista oficial de aristas...\n")

    edge_list = [
        (source, target)
        for source, neighbors in spatial_weights.neighbors.items()
        for target in neighbors
    ]

    # Eliminación de aristas duplicadas ---------------------------------------
    if GRAPH_SPATIAL_CONFIG["remove_duplicate_edges"]:
        edge_list = list(
            dict.fromkeys(edge_list)
        )

    # Validación ---------------------------------------------------------------
    if not edge_list:
        raise ValueError(
            "No se construyeron aristas espaciales."
        )

    print("6.1.4 Construcción de la Lista Oficial de Aristas completado.")
    # input("Presione Enter para continuar con 6.1.5...")

    # 6.1.5 Construcción del edge_index -----------------------------------------
    # Objetivo: Construir la representación matricial de las aristas en el formato requerido por PyTorch Geometric.
    # Producto: - edge_index
    # Pregunta científica: ¿La estructura de conectividad espacial fue representada correctamente
    # mediante la matriz edge_index?
    print("\nConstruyendo edge_index...\n")
    edge_index = (
        torch.tensor(edge_list, dtype=torch.long)
        .t()
        .contiguous()
    )  # Matriz oficial de aristas

    # Validación ---------------------------------------------------------------
    if edge_index.ndim != 2:
        raise ValueError(
            "edge_index debe ser una matriz bidimensional."
        )

    if edge_index.shape[0] != 2:
        raise ValueError(
            "edge_index debe tener dimensión (2, n_aristas)."
        )

    if edge_index.shape[1] == 0:
        raise ValueError(
            "No se construyeron aristas para el grafo."
        )

    if torch.any(edge_index < 0):
        raise ValueError(
            "Se encontraron índices negativos en edge_index."
        )

    if torch.max(edge_index) >= num_nodes:
        raise ValueError(
            "Existen identificadores de nodos fuera del rango permitido."
        )

    print("6.1.5 Construcción del edge_index completado.")
    # input("Presione Enter para continuar con 6.1.6...")

    # 6.1.6 Construcción del edge_weight ----------------------------------------
    ## Objetivo: Construir el vector oficial de pesos de las aristas cuando la configuración
    # del proyecto indique que el grafo utiliza aristas ponderadas.
    ## Producto: - edge_weight
    ## Pregunta científica: ¿Cada arista del grafo posee un peso válido para el proceso de aprendizaje?
    edge_weight = None
    if GRAPH_CONFIG["weighted"]:
        edge_weight = torch.ones(
            edge_index.shape[1],
            dtype=torch.float32
        )

        if edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "La longitud de edge_weight no coincide con el número de aristas."
            )

        print(f"Número de pesos : {edge_weight.shape[0]:,}")
        print("edge_weight construido correctamente.")

    else:
        print("El proyecto no utiliza pesos en las aristas.")

    print()

    print("6.1.6 Construcción del edge_weight completado.")
    # input("Presione Enter para continuar con 6.1.7...")

    # 6.1.7 Validación integral del grafo ---------------------------------------
    ## Objetivo: Verificar la consistencia estructural del grafo antes de construir
    # el objeto GraphData compatible con PyTorch Geometric.
    ## Producto: - edge_index validado - edge_weight validado - Grafo listo para construir el GraphData
    ## Pregunta científica: ¿La estructura del grafo cumple los requisitos para construir un
    # GraphData consistente y reproducible?
    print("\nValidando estructura del grafo...\n")

    # Validación de edge_index -----------------------------------------------
    if edge_index.dtype != torch.long:
        raise TypeError(
            "edge_index debe tener tipo torch.long."
        )

    # Validación de edge_weight ----------------------------------------------
    if GRAPH_CONFIG["weighted"]:
        if edge_weight is None:
            raise ValueError(
                "edge_weight no fue construido."
            )

        if edge_weight.dtype != torch.float32:
            raise TypeError(
                "edge_weight debe tener tipo torch.float32."
            )

        if torch.isnan(edge_weight).any():
            raise ValueError(
                "Se encontraron valores faltantes en edge_weight."
            )

        if edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "La longitud de edge_weight no coincide con edge_index."
            )

    else:
        if edge_weight is not None:
            raise ValueError(
                "Se construyó edge_weight aunque el grafo fue configurado como no ponderado."
            )

    print("6.1.7 Validación integral del grafo completado.")
    # input("Presione Enter para continuar con 6.1.8...")

    # 6.1.8 Registro de Métricas del GraphData ----------------------------------
    # Objetivo: Calcular y registrar las principales métricas estructurales del GraphData
    # correspondiente al año de procesamiento (current_year) para caracterizar
    # su topología antes de construir el objeto GraphData de PyTorch Geometric.
    # Producto: - num_edges - average_degree - graph_density
    # Pregunta científica: ¿Cuáles son las principales características estructurales del GraphData
    # del año de procesamiento?
    print(f"\nRegistrando métricas del GraphData {current_year}...\n")

    # Número de aristas -------------------------------------------------------
    num_edges = edge_index.shape[1]

    # Grado promedio ----------------------------------------------------------
    average_degree = (
        (2 * num_edges) / num_nodes
        if num_nodes > 0
        else 0.0
    )

    # Densidad del grafo ------------------------------------------------------
    if num_nodes > 1:
        graph_density = (
            (2 * num_edges)
            / (num_nodes * (num_nodes - 1))
        )

    else:
        graph_density = 0.0

    print("6.1.8 Registro de Métricas del GraphData completado.")
    # input("Presione Enter para continuar con 6.1.9...")

    # 6.1.9 Reporte Ejecutivo del GraphData -------------------------------------
    # Objetivo: Consolidar y presentar un resumen ejecutivo de las principales métricas, validaciones y
    # características estructurales del GraphData correspondiente al año de procesamiento, proporcionando
    #  una visión integral del estado del grafo antes de su construcción como objeto oficial de PyTorch Geometric.
    # Producto: - Reporte Ejecutivo del GraphData Anual
    # Pregunta científica: ¿El GraphData construido para el año de procesamiento cumple los criterios
    # estructurales, espaciales y de integridad definidos por la arquitectura científica del proyecto para
    # continuar con las etapas de entrenamiento, evaluación y forecasting?
    print("\n" + "-" * 80)
    print(f"REPORTE EJECUTIVO DEL GRAPHDATA - AÑO {current_year}")
    print("-" * 80)

    print("\n1. Filtrado del Dataset Científico")
    print(f"   Año procesado         : {current_year}")
    print(f"   Nodos del GraphData   : {num_nodes:,}")
    print("   Estado                : Correcto")

    print("\n2. Catálogo Oficial de Nodos")
    print(f"   Método espacial       : {graph_method.upper()}")
    print(f"   Número de nodos       : {num_nodes:,}")
    print("   Estado                : Validado")

    print("\n3. Vecindad Espacial")
    print(f"   Método utilizado      : {graph_method.upper()}")
    print(f"   Nodos procesados      : {spatial_weights.n:,}")
    print("   Estado                : Construida")

    print("\n4. Lista Oficial de Aristas")
    print(f"   Número de aristas     : {num_edges:,}")
    print("   Estado                : Construida")

    print("\n5. Matriz edge_index")
    print(f"   Dimensiones           : {tuple(edge_index.shape)}")
    print("   Estado                : Validada")

    print("\n6. Pesos de las Aristas")

    if GRAPH_CONFIG["weighted"]:
        print(f"   Número de pesos       : {edge_weight.shape[0]:,}")
        print("   Estado                : Construido")
    else:
        print("   Configuración         : Grafo no ponderado")

    print("\n7. Validación del Grafo")
    print(f"   Número de nodos       : {num_nodes:,}")
    print(f"   Número de aristas     : {num_edges:,}")
    print("   Estado                : Validado")

    print("\n8. Métricas Estructurales")
    print(f"   Número de nodos       : {num_nodes:,}")
    print(f"   Número de aristas     : {num_edges:,}")
    print(f"   Grado promedio        : {average_degree:.2f}")
    print(f"   Densidad del grafo    : {graph_density:.6f}")

    print("\nResultado Final")
    print(f"   GraphData anual       : {current_year}")
    print("   Estado                : Construido correctamente")

    print("-" * 80)

    # Variables del Modelo ---------------------------------------------------
    SHOW_FIGURES = GRAPH_CONFIG.get(
        "show_figures",
        True
    )

    print("6.1.9 Reporte Ejecutivo del GraphData completado.")
    # input("Presione Enter para continuar con 6.1.10...")

    # 6.1.10 Generación de Figuras Científicas ---------------------------------------------------
    # Objetivo: Generar las figuras científicas oficiales del GraphData correspondiente
    # al año de procesamiento (current_year), representando la estructura espacial, las variables
    # territoriales y la topología del grafo mediante visualizaciones reproducibles para documentación
    # técnica, auditoría, publicaciones científicas y la plataforma GeoAI.
    # Arquitectura científica:
    # Entrada: • Dataset Científico Anual (dataset_year) • Catálogo Oficial de Nodos (node_catalog_year)
    # • spatial_nodes • edge_index • edge_weight • edge_list • current_year
    # Proceso: • Configuración general de figuras • Construcción de mapas científicos
    # • Construcción del grafo espacial • Exportación de figuras • Validación de archivos exportados
    # Producto: • Figuras científicas del GraphData • Mapas temáticos • Grafo espacial
    # Pregunta científica: ¿Las características espaciales y estructurales del GraphData pueden representarse
    # mediante figuras científicas reproducibles que faciliten su interpretación, validación y comunicación?
    print("\n" + "-" * 80)
    print(f"GENERACIÓN DE FIGURAS CIENTÍFICAS - AÑO {current_year}")
    print("-" * 80)

    SHOW_FIGURES = False

    print("6.1.10 Generación de Figuras Científicas completado.")
    # input("Presione Enter para continuar con 6.10.1...")

    # 6.1.10.1 Configuración General de las Figuras Científicas ---------------------------------------------------
    # Objetivo: Configurar el entorno de trabajo para la generación, visualización y exportación de las
    # figuras científicas correspondientes al GraphData del año de procesamiento (current_year), garantizando una representación
    # reproducible, homogénea y de calidad para documentación, auditoría y publicaciones científicas.
    # Arquitectura científica:
    # Entrada: • current_year • GRAPHS_DIR • GRAPH_CONFIG • dataset_year • spatial_nodes
    # Proceso: • Configuración de parámetros gráficos • Definición del tamaño de figura • Definición de resolución
    # • Configuración de estilos • Creación del directorio de exportación • Inicialización de nombres de archivos
    # Producto: • Configuración oficial de figuras • Directorio de exportación validado
    # • Parámetros gráficos estandarizados
    # Pregunta científica: ¿Las figuras científicas del GraphData serán generadas bajo una configuración homogénea,
    # reproducible y compatible con la documentación técnica, artículos científicos y la plataforma GeoAI?
    FIGURE_STYLES = {
        "produccion_total": {
            "title": "Producción Agrícola",
            "cmap": "RdYlGn",
            "scheme": "Quantiles",
            "classes": 5,
            "legend": "Producción (toneladas)"

        },

        "rendimiento_promedio": {
            "title": "Rendimiento Agrícola",
            "cmap": "RdYlGn",
            "scheme": "Quantiles",
            "classes": 3,
            "legend": "Rendimiento"

        },

        "food_security_index": {
            "title": "Índice de Seguridad Alimentaria",
            "cmap": "RdYlGn",
            "scheme": "Quantiles",
            "classes": 5,
            "legend": "Índice"

        },

        "indice_hidrico": {
            "title": "Índice Hídrico",
            "cmap": "Blues",
            "scheme": "Quantiles",
            "classes": 5,
            "legend": "Índice"

        },

        "amenaza_sequia": {
            "title": "Amenaza por Sequía",
            "cmap": "YlOrRd",
            "scheme": "Quantiles",
            "classes": 4,
            "legend": "Amenaza"

        },

        "climate_risk": {
            "title": "Vulnerabilidad Climática",
            "cmap": "Spectral_r",
            "scheme": "Quantiles",
            "classes": 5,
            "legend": "Riesgo"
        }
    }

    print("6.1.10.1 Configuración General de las Figuras Científicas completado.")
    # input("Presione Enter para continuar con 6.1.10.2...")

    # 6.1.10.2 Figura Base del Territorio ---------------------------------------------------
    # Objetivo: Generar la representación cartográfica base del territorio correspondiente al año de 
    # procesamiento (current_year), mostrando la distribución espacial de los municipios que conforman el
    # GraphData y verificando la integridad de las geometrías oficiales utilizadas durante la construcción del grafo.
    # Arquitectura científica:
    # Entrada: • spatial_nodes • current_year • geometry
    # Proceso: • Validación de geometrías • Construcción del mapa base • Representación de municipios
    # • Configuración cartográfica • Exportación de la figura
    # Producto: • fig01_territorio_<año>.png
    # Pregunta científica: ¿La cobertura territorial utilizada para construir el GraphData representa
    # correctamente la totalidad de los municipios incluidos en el año de procesamiento?
    print("Generando mapa base del territorio...")
    fig, ax = plt.subplots(
        figsize=(12, 12)
    )

    spatial_nodes.plot(
        ax=ax,
        color="#F2F2F2",
        edgecolor="black",
        linewidth=0.20
    )

    ax.set_title(
        f"Figura Base del Territorio ({current_year})",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_axis_off()

    # Exportación
    territory_image = (
        GRAPHS_DIR /
        f"fig01_territorio_{current_year}.png"
    )

    fig.savefig(
        territory_image,
        dpi=300,
        bbox_inches="tight"
    )

    # Validación
    if not territory_image.exists():
        raise FileNotFoundError(
            f"No fue posible exportar {territory_image.name}"
        )

    print(f"Figura exportada : {territory_image.name}")

    # Auditoría Visual
    if SHOW_FIGURES:
        plt.show()
        input(
            "\nPresione ENTER para continuar..."
        )
    plt.close(fig)

    print("6.1.10.2 Figura Base del Territorio completado.")
    # input("Presione Enter para continuar con 6.1.10.3...")

    # 6.1.10.3 Figura de la Variable Objetivo ---------------------------------------------------
    # Objetivo: Generar la representación cartográfica de la variable objetivo del GraphData correspondiente
    # al año de procesamiento (current_year), mostrando su distribución espacial mediante un mapa coroplético que
    # facilite el análisis exploratorio y la interpretación territorial del fenómeno modelado.
    # Arquitectura científica:
    # Entrada: • spatial_nodes • TARGET_VARIABLE • current_year
    # Proceso: • Validación de la variable objetivo • Integración de valores al mapa
    # • Construcción del mapa coroplético • Configuración cartográfica • Exportación de la figura
    # Producto: • fig02_target_<año>.png
    # Pregunta científica: ¿Cuál es la distribución espacial de la variable objetivo en el territorio correspondiente
    # al año de procesamiento y qué patrones espaciales pueden identificarse antes del entrenamiento del modelo?
    
    # Validación
    if TARGET_VARIABLE not in spatial_nodes.columns:
        raise ValueError(f"La variable objetivo '{TARGET_VARIABLE}' no existe.")

    config = FIGURE_STYLES.get(
        TARGET_VARIABLE,
        {
            "title": TARGET_VARIABLE.replace("_", " ").title(),
            "cmap": "viridis",
            "scheme": "Quantiles",
            "classes": 5,
            "legend": TARGET_VARIABLE
        }
    )

    # Construcción del Mapa
    fig, ax = plt.subplots(figsize=(12, 12))
    spatial_nodes.plot(
        column=TARGET_VARIABLE,
        cmap=config["cmap"],
        scheme=config["scheme"],
        k=config["classes"],
        legend=True,
        edgecolor="black",
        linewidth=0.20,
        ax=ax
    )

    ax.set_title(
        f'{config["title"]}\nAño {current_year}',
        fontsize=16,
        fontweight="bold"
    )
    ax.set_axis_off()

    # Exportación
    target_image = GRAPHS_DIR / f"fig02_{TARGET_VARIABLE}_{current_year}.png"
    fig.savefig(
        target_image,
        dpi=300,
        bbox_inches="tight"
    )

    # Validación
    if not target_image.exists():
        raise FileNotFoundError(f"No fue posible exportar {target_image.name}")

    print(f"Figura exportada : {target_image.name}")

    # Resumen Estadístico
    serie = spatial_nodes[TARGET_VARIABLE]
    print("\nResumen Estadístico")
    print(f"Mínimo : {serie.min():,.4f}")
    print(f"Máximo : {serie.max():,.4f}")
    print(f"Media  : {serie.mean():,.4f}")
    print(f"Std    : {serie.std():,.4f}")

    # Auditoría Visual
    if SHOW_FIGURES:
        plt.show()
        input("\nPresione ENTER para continuar...")
    plt.close(fig)

    print("6.1.10.3 Figura de la Variable Objetivo completado.")
    # input("Presione Enter para continuar con 6.1.10.4...")

    
    # 6.1.10.4 Figuras de Variables Predictoras ---------------------------------------------------
    # Objetivo: Generar las representaciones cartográficas de las variables predictoras utilizadas en la 
    # construcción del GraphData correspondiente al año de procesamiento (current_year), permitiendo visualizar 
    # su distribución espacial, identificar patrones territoriales y verificar la calidad de las variables de entrada del modelo.
    # Arquitectura científica:
    # Entrada: • spatial_nodes • FEATURE_COLUMNS • current_year
    # Proceso: • Validación de variables predictoras • Integración de variables al mapa
    # • Construcción de mapas coropléticos • Configuración cartográfica • Exportación de figuras
    # Producto: • fig03_<variable>_<año>.png • Colección de figuras de variables predictoras
    # Pregunta científica: ¿Las variables predictoras presentan patrones espaciales coherentes,
    # consistentes y útiles para explicar la variabilidad territorial de la variable objetivo antes del entrenamiento del modelo?
    print("\n" + "-" * 80)
    print("6.1.10.4 FIGURAS DE LAS VARIABLES PREDICTORAS")
    print("-" * 80)

    print(f"Variables predictoras: {len(FEATURE_COLUMNS)}")

    # Procesamiento de Variables Predictoras
    for i, variable in enumerate(FEATURE_COLUMNS, start=1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(FEATURE_COLUMNS)}] {variable}")

        # Validación
        if variable not in spatial_nodes.columns:
            print(f"Variable inexistente: {variable}")
            continue
        serie = spatial_nodes[variable]

        if serie.isnull().all():
            print(f"Variable sin información: {variable}")
            continue

        # Configuración Cartográfica
        config = FIGURE_STYLES.get(
            variable,
            {
                "title": variable.replace("_", " ").title(),
                "cmap": "viridis",
                "scheme": "Quantiles",
                "classes": 5,
                "legend": variable
            }
        )

        # Construcción del Mapa
        fig, ax = plt.subplots(
            figsize=(12, 12)
        )

        spatial_nodes.plot(
            column=variable,
            cmap=config["cmap"],
            scheme=config["scheme"],
            k=config["classes"],
            legend=True,
            edgecolor="black",
            linewidth=0.20,
            ax=ax
        )

        ax.set_title(
            f'{config["title"]}\nAño {current_year}',
            fontsize=16,
            fontweight="bold"
        )
        ax.set_axis_off()

        # Exportación
        predictor_image = (
            GRAPHS_DIR /
            f"fig03_{variable}_{current_year}.png"
        )

        fig.savefig(
            predictor_image,
            dpi=300,
            bbox_inches="tight"
        )

        # Validación
        if predictor_image.exists():
            print(f"Figura exportada : {predictor_image.name}")

        else:
            print(f"Error exportando : {predictor_image.name}")

        # Resumen Estadístico
        print(f" Mínimo : {serie.min():,.4f}")
        print(f" Máximo : {serie.max():,.4f}")
        print(f" Media  : {serie.mean():,.4f}")
        print(f" Std    : {serie.std():,.4f}")

        # Auditoría Visual
        if SHOW_FIGURES:
            plt.show()
            input(
                "\nPresione ENTER para continuar..."
            )
        plt.close(fig)

    # Resumen Final
    print("\n" + "-" * 80)
    print("Proceso finalizado correctamente.")
    print(f"Total de variables procesadas: {len(FEATURE_COLUMNS)}")
    print("-" * 80)

    print("6.1.10.4 Figuras de Variables Predictoras completado.")
    # input("Presione Enter para continuar con 6.1.10.5...")

    # 6.1.10.5 Figura del Grafo Espacial ---------------------------------------------------
    # Objetivo: Generar la representación gráfica de la estructura espacial del GraphData, visualizando los
    #  nodos (municipios) y las aristas (relaciones espaciales) construidas mediante el criterio oficial de vecindad.
    # Arquitectura científica:
    # Entrada: • spatial_nodes • edge_index • current_year
    # Proceso: • Construcción del NetworkX Graph • Cálculo de posiciones geográficas
    # • Dibujar aristas • Dibujar nodos • Exportación
    # Producto: • fig04_graph_<año>.png
    # Pregunta científica: ¿La estructura espacial construida representa correctamente las relaciones
    # territoriales entre municipios antes del entrenamiento del modelo?
    print("\n" + "-" * 80)
    print("6.1.10.5 FIGURA DEL GRAFO ESPACIAL")
    print("-" * 80)

    # Construcción del Grafo
    G = nx.Graph()

    # Nodos
    for node in range(len(spatial_nodes)):
        G.add_node(node)

    # Aristas
    edges = edge_index.cpu().numpy().T
    G.add_edges_from(edges)
    print(f"Nodos   : {G.number_of_nodes():,}")
    print(f"Aristas : {G.number_of_edges():,}")

    # Posiciones Espaciales
    centroids = spatial_nodes.geometry.centroid
    positions = {
        idx: (geom.x, geom.y)
        for idx, geom in enumerate(centroids)
    }

    # Construcción de la Figura
    fig, ax = plt.subplots(
        figsize=(14, 14)
    )

    # Aristas
    nx.draw_networkx_edges(
        G,
        positions,
        ax=ax,
        edge_color="lightgray",
        width=0.30,
        alpha=0.50
    )

    # Nodos
    nx.draw_networkx_nodes(
        G,
        positions,
        ax=ax,
        node_size=6,
        node_color="#1565C0",
        linewidths=0,
        alpha=0.85
    )

    # Configuración
    ax.set_title(
        f"Estructura Espacial del Grafo\nAño {current_year}",
        fontsize=16,
        fontweight="bold"
    )
    ax.set_axis_off()

    # Exportación
    graph_image = (
        GRAPHS_DIR /
        f"fig04_graph_{current_year}.png"
    )

    fig.savefig(
        graph_image,
        dpi=300,
        bbox_inches="tight"
    )

    # Validación
    if not graph_image.exists():
        raise FileNotFoundError(
            f"No fue posible exportar {graph_image.name}"
        )
    print(f"Figura exportada : {graph_image.name}")

    # Auditoría Visual
    if SHOW_FIGURES:
        plt.show()
        input(
            "\nPresione ENTER para continuar..."
        )
    plt.close(fig)

    print("Figura del grafo validada correctamente.")
    print("6.1.10.5 Figura del Grafo Espacial completado.")
    # input("Presione Enter para continuar con 6.1.10.6...")

    # 6.1.10.6 Métricas del Grafo ---------------------------------------------------
    # Objetivo: Calcular las principales métricas topológicas del grafo espacial construido para el año de
    # procesamiento (current_year), permitiendo evaluar la calidad estructural del GraphData antes
    # del entrenamiento de la Red Neuronal de Grafos (Graph Neural Network).
    # Arquitectura científica:
    # Entrada: • G • current_year
    # Proceso: • Cálculo de métricas globales • Cálculo de métricas por nodo 
    # • Construcción de tablas resumen • Exportación de resultados
    # Producto: • graph_metrics_<año>.csv • node_metrics_<año>.csv
    # Pregunta científica: ¿El GraphData presenta una estructura espacial consistente que permita
    # soportar adecuadamente la propagación de información durante el entrenamiento del modelo GNN?
    print("\n" + "-" * 80)
    print("6.1.10.6 MÉTRICAS DEL GRAFO")
    print("-" * 80)

    # Métricas Globales
    graph_num_nodes = G.number_of_nodes()
    graph_num_edges = G.number_of_edges()
    density = nx.density(G)
    degree_values = np.array(
        [degree for _, degree in G.degree()]
    )

    degree_mean = degree_values.mean()
    degree_min = degree_values.min()
    degree_max = degree_values.max()
    degree_std = degree_values.std()
    connected = nx.is_connected(G)

    if connected:
        diameter = nx.diameter(G)
        average_path = nx.average_shortest_path_length(G)

    else:
        diameter = np.nan
        average_path = np.nan
    average_clustering = nx.average_clustering(G)
    transitivity = nx.transitivity(G)

    # Construcción del Resumen Global
    graph_metrics = pd.DataFrame([{
        "year": current_year,
        "nodes": graph_num_nodes,
        "edges": graph_num_edges,
        "density": density,
        "degree_mean": degree_mean,
        "degree_min": degree_min,
        "degree_max": degree_max,
        "degree_std": degree_std,
        "connected": connected,
        "diameter": diameter,
        "average_path_length": average_path,
        "average_clustering": average_clustering,
        "transitivity": transitivity
    }])

    # Exportación de Métricas Globales
    graph_metrics_file = (
        GRAPHS_DIR /
        f"graph_metrics_{current_year}.csv"
    )

    graph_metrics.to_csv(
        graph_metrics_file,
        index=False,
        encoding="utf-8-sig"
    )

    # Métricas por Nodo
    print("\nCalculando métricas por nodo...")
    degree_dict = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    try:
        eigenvector = nx.eigenvector_centrality(
            G,
            max_iter=1000
        )

    except Exception:
        eigenvector = {
            node: np.nan
            for node in G.nodes()
        }

    # Tabla por Nodo
    node_metrics = pd.DataFrame({
        "node": list(G.nodes()),
        "degree": [
            degree_dict[node]
            for node in G.nodes()
        ],

        "betweenness": [
            betweenness[node]
            for node in G.nodes()
        ],

        "closeness": [
            closeness[node]
            for node in G.nodes()
        ],

        "eigenvector": [
            eigenvector[node]
            for node in G.nodes()
        ]
    })

    # Exportación de Métricas por Nodo
    node_metrics_file = (
        GRAPHS_DIR /
        f"node_metrics_{current_year}.csv"
    )

    node_metrics.to_csv(
        node_metrics_file,
        index=False,
        encoding="utf-8-sig"
    )

    # Validación
    if not graph_metrics_file.exists():
        raise FileNotFoundError(
            f"No fue posible exportar {graph_metrics_file.name}"
        )

    if not node_metrics_file.exists():
        raise FileNotFoundError(
            f"No fue posible exportar {node_metrics_file.name}"
        )

    # Resumen Ejecutivo
    print("\nResumen del Grafo")
    print("-" * 80)
    print(f"Nodos               : {num_nodes:,}")
    print(f"Aristas             : {num_edges:,}")
    print(f"Densidad            : {density:.6f}")
    print(f"Grado promedio      : {degree_mean:.2f}")
    print(f"Grado mínimo        : {degree_min}")
    print(f"Grado máximo        : {degree_max}")
    print(f"Desv. estándar      : {degree_std:.2f}")
    print(f"Conectado           : {connected}")
    print(f"Diámetro            : {diameter}")
    print(f"Camino promedio     : {average_path}")
    print(f"Clustering          : {average_clustering:.4f}")
    print(f"Transitividad       : {transitivity:.4f}")
    print("-" * 80)
    print(f"Archivo generado    : {graph_metrics_file.name}")
    print(f"Archivo generado    : {node_metrics_file.name}")
    print("Métricas del grafo calculadas correctamente.")

    print("6.1.10.6 Métricas del Grafo completado.")
    # input("Presione Enter para continuar con 6.1.10.7...")

    # 6.1.10.7 Inventario y Validación de Archivos ---------------------------------------------------
    # Objetivo: Verificar la existencia e integridad de todos los productos científicos generados durante
    # el proceso de construcción del GraphData para el año de procesamiento (current_year), garantizando
    # la reproducibilidad y disponibilidad de los recursos para las siguientes etapas del Pipeline.
    # Arquitectura científica:
    # Entrada: • GRAPHS_DIR • TARGET_VARIABLE • FEATURE_COLUMNS • current_year
    # Proceso: • Construcción del inventario esperado • Validación de existencia de archivos
    # • Conteo de productos generados • Identificación de archivos faltantes
    # Producto: • Inventario validado • Estado de integridad del bloque
    # Pregunta científica: ¿Todos los productos científicos requeridos fueron generados correctamente y
    # se encuentran disponibles para las siguientes fases del Pipeline Científico?
    print("\n" + "-" * 80)
    print("6.1.10.7 INVENTARIO Y VALIDACIÓN DE ARCHIVOS")
    print("-" * 80)

    # Inventario Esperado
    expected_files = [
        GRAPHS_DIR / f"fig01_territorio_{current_year}.png",
        GRAPHS_DIR / f"fig02_{TARGET_VARIABLE}_{current_year}.png",
        GRAPHS_DIR / f"fig04_graph_{current_year}.png",
        GRAPHS_DIR / f"graph_metrics_{current_year}.csv",
        GRAPHS_DIR / f"node_metrics_{current_year}.csv"
    ]

    # Figuras de Variables Predictoras
    for variable in FEATURE_COLUMNS:
        expected_files.append(
            GRAPHS_DIR /
            f"fig03_{variable}_{current_year}.png"
        )

    # Validación
    validated_files = []
    missing_files = []
    for file in expected_files:
        if file.exists():
            validated_files.append(file)
            print(f"✓ {file.name}")

        else:
            missing_files.append(file)
            print(f"✗ {file.name}")

    # Resumen
    print("\n" + "-" * 80)
    print(f"Archivos esperados : {len(expected_files)}")
    print(f"Archivos validados : {len(validated_files)}")
    print(f"Archivos faltantes : {len(missing_files)}")
    print("-" * 80)

    # Validación Final
    if missing_files:
        raise FileNotFoundError(
            "La auditoría detectó archivos faltantes.\n"
            + "\n".join(
                [file.name for file in missing_files]
            )
        )

    print("Inventario validado correctamente.")
    print("6.1.10.7 Inventario y Validación de Archivos completado.")
    # input("Presione Enter para continuar con 6.1.10.8...")
 
    # 6.1.10.8 Reporte Ejecutivo ---------------------------------------------------
    # Objetivo: Consolidar los resultados obtenidos durante la generación de las figuras científicas y 
    # las métricas del GraphData para el año de procesamiento (current_year), documentando los productos
    # generados y el estado final de la auditoría científica del bloque.
    # Arquitectura científica:
    # Entrada: • current_year • G • TARGET_VARIABLE • FEATURE_COLUMNS • validated_files • expected_files
    # Proceso: • Consolidación de indicadores • Elaboración del reporte ejecutivo • Exportación del reporte
    # Producto: • scientific_report_<año>.csv
    # Pregunta científica: ¿La generación de figuras científicas del GraphData fue completada correctamente y
    # todos los productos requeridos fueron validados para continuar con las siguientes etapas del Pipeline Científico?
    print("\n" + "-" * 80)
    print("6.1.10.8 REPORTE EJECUTIVO")
    print("-" * 80)

    # Construcción del Reporte
    report = pd.DataFrame([{
        "year": current_year,
        "target_variable": TARGET_VARIABLE,
        "predictor_variables": len(FEATURE_COLUMNS),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "validated_files": len(validated_files),
        "expected_files": len(expected_files),
        "missing_files": len(expected_files) - len(validated_files),
        "status": (
            "VALIDADO"
            if len(validated_files) == len(expected_files)
            else "INCOMPLETO"
        )
    }])

    # Exportación
    report_file = (
        GRAPHS_DIR /
        f"scientific_report_{current_year}.csv"
    )

    report.to_csv(
        report_file,
        index=False,
        encoding="utf-8-sig"
    )

    # Validación
    if not report_file.exists():
        raise FileNotFoundError(
            f"No fue posible exportar {report_file.name}"
        )

    # Resumen Ejecutivo
    print("\nVariable objetivo:")
    print(TARGET_VARIABLE)

    print("\nColumnas disponibles:")
    print(sorted(spatial_nodes.columns))

    print("\n" + "=" * 80)
    print("RESUMEN EJECUTIVO DEL AÑO")
    print("=" * 80)

    print(f"Año procesado              : {current_year}")
    print(f"Variable objetivo          : {TARGET_VARIABLE}")
    print(f"Variables predictoras      : {len(FEATURE_COLUMNS)}")
    print(f"Número de nodos            : {G.number_of_nodes():,}")
    print(f"Número de aristas          : {G.number_of_edges():,}")
    print(f"Archivos esperados         : {len(expected_files)}")
    print(f"Archivos validados         : {len(validated_files)}")
    print(f"Archivos faltantes         : {len(expected_files) - len(validated_files)}")
    print(f"Estado del bloque          : {report.loc[0, 'status']}")
    print(f"Reporte generado           : {report_file.name}")
    print("=" * 80)
    print("Bloque 6.1.10 finalizado correctamente.")    

# BLOQUE 7. CONSTRUCCIÓN DEL GraphData DE PYTORCH GEOMETRIC -----------------------------------------------
# Objetivo: Construir el objeto oficial GraphData correspondiente al año de procesamiento (current_year), 
# integrando las Node Features, la variable objetivo y la estructura espacial previamente construida
# para obtener una representación compatible con PyTorch Geometric.
# Arquitectura científica:
# - Entrada: • Dataset Científico Anual (dataset_year) • Catálogo Oficial de Nodos Anual (node_catalog_year)
# • edge_index • edge_weight (cuando el grafo sea ponderado)
# - Proceso: • Construcción de Node Features • Construcción de la Variable Objetivo • Conversión a Tensores
# • Construcción del GraphData • Validación Integral • Registro de Métricas • Exportación del GraphData
# • Resumen del GraphData • Registro en la Colección Oficial
# - Salida: • graph_data • graph_data_collection
# Producto: - graph_data - graph_data_collection
# Pregunta científica:
# ¿Cómo integrar las características de los nodos, la variable objetivo
# y la estructura espacial para construir un GraphData consistente,
# reproducible y compatible con PyTorch Geometric para cada año del
# panel científico?
print("\n" + "-" * 80)
print("BLOQUE 7. CONSTRUCCIÓN DEL GraphData DE PYTORCH GEOMETRIC")
print("-" * 80)

graph_data_collection = {}
graph_audit_records = []

for current_year in years:
    # 7.1 Construcción de las Node Features -----------------------------------------------
    # Objetivo: Construir la matriz oficial de Node Features correspondiente al año de
    # procesamiento (current_year), garantizando la alineación entre el
    # Catálogo Oficial de Nodos Anual y las variables predictoras del Dataset Científico Anual.
    # Producto: - x_numpy
    # Pregunta científica: ¿Cada nodo del GraphData correspondiente al año current_year posee
    # correctamente asociadas sus variables predictoras respetando el Catálogo Oficial de Nodos?
    print("\nConstruyendo Node Features...\n")

    # Configuración -----------------------------------------------------------------------
    key_columns = NODE_CATALOG_CONFIG["key_columns"]
    excluded_columns = sorted(
        set(
            NODE_FEATURE_CONFIG["excluded_columns"]
            + key_columns
        )
    )

    feature_columns = [
        column
        for column in dataset_year.columns
        if column not in excluded_columns
    ]

    # Construcción de las Node Features ---------------------------------------------------
    node_features = (
        node_catalog_year[
            key_columns + [NODE_INDEX_COLUMN]
        ]
        .merge(
            dataset_year[
                key_columns + feature_columns
            ],
            on=key_columns,
            how="left",
            validate="one_to_one"
        )
        .sort_values(NODE_INDEX_COLUMN)
        .reset_index(drop=True)
    )

    x_numpy = (
        node_features[
            feature_columns
        ].to_numpy(
            dtype=np.float32
        )
    )

    # Validación --------------------------------------------------------------------------
    if x_numpy.shape[0] != num_nodes:
        raise ValueError(
            "El número de filas de las Node Features no coincide con el número de nodos."
        )

    if x_numpy.shape[1] != len(feature_columns):
        raise ValueError(
            "El número de variables predictoras es incorrecto."
        )

    if np.isnan(x_numpy).any():
        raise ValueError(
            "Las Node Features contienen valores faltantes."
        )

    if np.isinf(x_numpy).any():
        raise ValueError(
            "Las Node Features contienen valores infinitos."
        )

    # Auditoría adicional -----------------------------------------------------------------
    print("-" * 80)
    print("AUDITORÍA DE LAS NODE FEATURES")
    print("-" * 80)

    print(f"Dimensión de x_numpy    : {x_numpy.shape}")
    print("\nVariables predictoras utilizadas:")

    for i, variable in enumerate(feature_columns, start=1):
        print(f"{i:02d}. {variable}")

    print("-" * 80)

    # Registro oficial de las variables predictoras -------------------------------
    NODE_FEATURE_CONFIG["feature_columns"] = feature_columns.copy()
    print(
        f"\nLista oficial de variables predictoras registrada "
        f"({len(feature_columns)} variables)."
    )

    # 7.2 Construcción de la Variable Objetivo ---------------------------------------------
    # Objetivo: Construir el vector oficial de la variable objetivo correspondiente al año
    # de procesamiento (current_year), garantizando la alineación entre el
    # Catálogo Oficial de Nodos Anual y la variable objetivo del Dataset Científico Anual.
    # Producto: - y_numpy
    # Pregunta científica: ¿Cada nodo del GraphData correspondiente al año current_year posee
    # correctamente asociada su variable objetivo?
    print("\nConstruyendo Variable Objetivo...\n")

    # Configuración -----------------------------------------------------------------------
    key_columns = NODE_CATALOG_CONFIG["key_columns"]
    target_column = TARGET_VARIABLE

    # Construcción de la Variable Objetivo ------------------------------------------------
    target_data = (
        node_catalog_year[
            key_columns + [NODE_INDEX_COLUMN]
        ]
        .merge(
            dataset_year[
                key_columns + [target_column]
            ],
            on=key_columns,
            how="left",
            validate="one_to_one"
        )
        .sort_values(NODE_INDEX_COLUMN)
        .reset_index(drop=True)
    )

    y_numpy = (
        target_data[target_column]
        .to_numpy(
            dtype=np.float32
        )
    )

    # Validación --------------------------------------------------------------------------
    if y_numpy.shape[0] != num_nodes:
        raise ValueError(
            "El número de observaciones de la variable objetivo no coincide con el número de nodos."
        )

    if np.isnan(y_numpy).any():
        raise ValueError(
            "La variable objetivo contiene valores faltantes."
        )

    if np.isinf(y_numpy).any():
        raise ValueError(
            "La variable objetivo contiene valores infinitos."
        )

    # Auditoría de variabilidad ------------------------------------------------------------
    n_unique = np.unique(y_numpy).size
    unique_ratio = n_unique / len(y_numpy)

    if n_unique <= 1:
        raise ValueError(
            "La variable objetivo no presenta variabilidad."
        )

    print("La variable objetivo presenta variabilidad suficiente.")

    # 7.3 Conversión a Tensores PyTorch ----------------------------------------------
    # Objetivo: Convertir las estructuras NumPy del Dataset Científico Anual a tensores oficiales de
    # PyTorch y reutilizar la estructura espacial generada en el Bloque 6 para construir posteriormente el GraphData.
    # Producto: - x_tensor - y_tensor - edge_index (reutilizado) - edge_weight (reutilizado)
    print("\nConvirtiendo estructuras NumPy a Tensores PyTorch...\n")

    # Node Features
    x_tensor = torch.as_tensor(
        x_numpy,
        dtype=torch.float32
    )

    # Variable objetivo
    y_tensor = torch.as_tensor(
        y_numpy,
        dtype=torch.float32
    )

    # La estructura espacial ya fue construida en el Bloque 6
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("edge_index debe ser un torch.Tensor.")

    if GRAPH_CONFIG["weighted"]:

        if not isinstance(edge_weight, torch.Tensor):
            raise TypeError("edge_weight debe ser un torch.Tensor.")

    
    # 7.4 Construcción del GraphData ------------------------------------------------
    # Objetivo: Integrar las Node Features, la variable objetivo y la estructura espacial previamente construida
    #  para obtener el objeto oficial GraphData compatible con PyTorch Geometric correspondiente al año de procesamiento.
    # Producto: - graph_data
    # Pregunta científica: ¿La información tabular y la estructura espacial anual pueden integrarse
    # correctamente en un único objeto GraphData compatible con PyTorch Geometric?
    print("\nConstruyendo GraphData...\n")

    # Verificación de disponibilidad
    if "Data" not in globals():
        from torch_geometric.data import Data

    # Construcción del GraphData
    if GRAPH_CONFIG["weighted"]:
        graph_data = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=y_tensor
        )

    else:
        graph_data = Data(
            x=x_tensor,
            edge_index=edge_index,
            y=y_tensor
        )

    # 7.5 Validación Integral del GraphData -----------------------------------------
    # Objetivo: Verificar la consistencia estructural y funcional del GraphData construido, garantizando 
    # su compatibilidad con PyTorch Geometric y su utilización en las etapas posteriores de entrenamiento, evaluación y forecasting.
    # Producto: - GraphData validado
    # Pregunta científica: ¿El GraphData representa correctamente la estructura espacial anual y cumple
    # los requisitos para el entrenamiento de modelos GNN?
    print("\nValidando GraphData...\n")

    # Existencia del objeto
    if graph_data is None:
        raise ValueError("graph_data no fue construido.")

    # Node Features
    if graph_data.x is None:
        raise ValueError("El GraphData no contiene Node Features.")
    if graph_data.x.dtype != torch.float32:
        raise TypeError("graph_data.x debe ser float32.")
    if graph_data.x.shape[0] != num_nodes:
        raise ValueError("Número incorrecto de nodos en graph_data.x.")
    if graph_data.x.shape[1] != len(feature_columns):
        raise ValueError("Número incorrecto de variables predictoras.")
    if torch.isnan(graph_data.x).any():
        raise ValueError("graph_data.x contiene NaN.")
    if torch.isinf(graph_data.x).any():
        raise ValueError("graph_data.x contiene valores infinitos.")

    # Variable objetivo
    if graph_data.y is None:
        raise ValueError("El GraphData no contiene variable objetivo.")
    if graph_data.y.dtype != torch.float32:
        raise TypeError("graph_data.y debe ser float32.")
    if graph_data.y.shape[0] != num_nodes:
        raise ValueError("Número incorrecto de observaciones en graph_data.y.")
    if torch.isnan(graph_data.y).any():
        raise ValueError("graph_data.y contiene NaN.")
    if torch.isinf(graph_data.y).any():
        raise ValueError("graph_data.y contiene valores infinitos.")

    # Estructura espacial
    if graph_data.edge_index is None:
        raise ValueError("El GraphData no contiene edge_index.")
    if graph_data.edge_index.dtype != torch.long:
        raise TypeError("edge_index debe ser torch.long.")
    if graph_data.edge_index.shape[0] != 2:
        raise ValueError("edge_index debe tener dimensión (2, E).")
    if graph_data.edge_index.min().item() < 0:
        raise ValueError("edge_index contiene índices negativos.")
    if graph_data.edge_index.max().item() >= num_nodes:
        raise ValueError("edge_index contiene índices fuera del rango de nodos.")

    # Pesos de las aristas
    if GRAPH_CONFIG["weighted"]:
        if graph_data.edge_attr is None:
            raise ValueError("El GraphData no contiene edge_attr.")
        if graph_data.edge_attr.dtype != torch.float32:
            raise TypeError("edge_attr debe ser float32.")
        if graph_data.edge_attr.shape[0] != graph_data.edge_index.shape[1]:
            raise ValueError(
                "El número de pesos no coincide con el número de aristas."
            )

    # Consistencia global
    if graph_data.num_nodes != num_nodes:
        raise ValueError("Número de nodos inconsistente.")
    if graph_data.num_edges != num_edges:
        raise ValueError("Número de aristas inconsistente.")
    print("GraphData validado correctamente.\n")

    # 7.8.1 Exportación del GraphData Anual ---------------------------------------------------
    # Objetivo: Exportar el GraphData correspondiente al año procesado en formato PyTorch
    # (.pt), garantizando su persistencia para las etapas posteriores del pipeline.
    # Producto: - graph_data_<año>.pt
    print("\n" + "-" * 80)
    print("7.8.1 EXPORTACIÓN DEL GRAPHDATA ANUAL")
    print("-" * 80)

    # Definición de la ruta de exportación
    GRAPH_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    graph_file = (
        GRAPH_DATA_DIR
        / f"graph_data_{current_year}.pt"
    )

    # Exportación del GraphData
    torch.save(
        graph_data,
        graph_file
    )

    # Verificación de la exportación
    if not graph_file.exists():
        raise FileNotFoundError(
            f"No fue posible exportar el GraphData del año {current_year}."
        )

    # 7.9 Consolidación del GraphData -----------------------------------------------
    # Objetivo: Registrar oficialmente los productos científicos generados durante la
    # construcción del GraphData correspondiente al año de procesamiento,
    # consolidando la colección de grafos y el registro de auditoría que serán
    # utilizados en las etapas posteriores del Pipeline Científico.
    # Productos: - graph_data_collection - graph_audit_records
    # Pregunta científica: ¿El GraphData correspondiente al año actual quedó registrado oficialmente
    # para formar parte de la colección científica del proyecto?
    print("\n" + "=" * 80)
    print("7.9 REGISTRO OFICIAL DEL GRAPHDATA")
    print("=" * 80)

    # Registro de la colección de GraphData
    graph_data_collection[current_year] = graph_data

    # Registro de la auditoría científica
    graph_audit_records.append({
        "year": current_year,
        "nodes": graph_data.num_nodes,
        "edges": graph_data.num_edges,
        "features": graph_data.num_node_features,
        "target": TARGET_VARIABLE,
        "graph_method": graph_method,
        "weighted": GRAPH_CONFIG["weighted"],
        "x_shape": tuple(graph_data.x.shape),
        "y_shape": tuple(graph_data.y.shape),
        "edge_index_shape": tuple(graph_data.edge_index.shape),
        "edge_attr": (
            tuple(graph_data.edge_attr.shape)
            if GRAPH_CONFIG["weighted"]
            else None
        ),
        "status": "OK"
    })

    # Verificación del registro
    if current_year not in graph_data_collection:
        raise ValueError(
            "El GraphData no fue registrado correctamente."
        )

    if len(graph_audit_records) != len(graph_data_collection):
        raise ValueError(
            "La colección y la auditoría presentan un número diferente de registros."
        )

    # Resumen
    print("\n" + "-" * 80)
    print("RESUMEN EJECUTIVO DEL GRAPHDATA")
    print("-" * 80)

    print(f"Año procesado               : {current_year}")
    print(f"Variable objetivo           : {TARGET_VARIABLE}")
    print(f"Método espacial             : {graph_method.upper()}")

    print("\nESTRUCTURA DEL GRAPHDATA")
    print("-" * 80)

    print(f"Número de nodos             : {graph_data.num_nodes:,}")
    print(f"Número de aristas           : {graph_data.num_edges:,}")
    print(f"Variables predictoras       : {graph_data.num_node_features:,}")
    print(f"Grado promedio              : {average_degree:.2f}")
    print(f"Densidad del grafo          : {graph_density:.6f}")

    print("\nOBJETO GraphData")
    print("-" * 80)

    print(f"x                           : {tuple(graph_data.x.shape)}")
    print(f"y                           : {tuple(graph_data.y.shape)}")
    print(f"edge_index                  : {tuple(graph_data.edge_index.shape)}")

    if GRAPH_CONFIG["weighted"]:
        print(f"edge_attr                   : {tuple(graph_data.edge_attr.shape)}")

    print("\nESTADO DEL PIPELINE")
    print("-" * 80)

    print("Filtrado anual              : Correcto")
    print("Catálogo oficial            : Validado")
    print("Vecindad espacial           : Construida")
    print("Lista de aristas            : Construida")
    print("edge_index                  : Validado")

    if GRAPH_CONFIG["weighted"]:
        print("edge_weight                 : Construido")
    else:
        print("edge_weight                 : No aplica")

    print("Node Features               : Construidas")
    print("Variable objetivo           : Construida")
    print("GraphData                   : Construido")
    print("Auditoría científica        : Completada")

    print("\nSIGUIENTE ETAPA")
    print("-" * 80)

    print("Registro del GraphData")
    print("Exportación")
    print("Benchmark")
    print("Entrenamiento")
    print("Evaluación")
    print("Forecasting")

    print("\nEstado final                : GraphData listo para el Pipeline Científico")
    print("-" * 80)


# 7.8 Exportación --------------------------------------------------------------
# Objetivo: Exportar los productos científicos generados durante la construcción del
# GraphData, garantizando su persistencia, reproducibilidad y disponibilidad
# para las etapas posteriores del pipeline científico.
# Productos: - GraphData anual (.pt) - Colección de GraphData (.pt) - Metadatos del GraphData
# - Auditorías del proceso de construcción
# Subbloques: 7.8.1 Exportación del GraphData Anual. 7.8.2 Exportación de la Colección de GraphData
# 7.8.3 Exportación de Metadatos. 7.8.4 Exportación de Auditorías. 7.8.5 Verificación de la Exportación

# 7.8.2 Exportación de la Colección de GraphData --------------------------------------------------------------
# Objetivo: Exportar la colección completa de GraphData generada durante el procesamiento de todos los años,
# permitiendo su reutilización en las etapas de Benchmark, Entrenamiento, Evaluación y Forecasting.
# Producto: - graph_data_collection.pt
print("\n" + "-" * 80)
print("7.8.2 EXPORTACIÓN DE LA COLECCIÓN DE GRAPHDATA")
print("-" * 80)

# Definición de la ruta de exportación
collection_file = (
    GRAPH_DATA_DIR
    / "graph_data_collection.pt"
)

# Validación de la colección
if not graph_data_collection:
    raise ValueError(
        "No existen GraphData registrados para exportar."
    )

# Exportación de la colección
torch.save(
    graph_data_collection,
    collection_file
)

# Verificación de la exportación
if not collection_file.exists():
    raise FileNotFoundError(
        "No fue posible exportar la colección de GraphData."
    )

# Validación de la colección
if len(graph_data_collection) != len(years):
    raise ValueError(
        "La colección no contiene todos los años procesados."
    )

# 7.8.3 Exportación de Metadatos --------------------------------------------------------------
# Objetivo: Exportar los metadatos descriptivos de la colección GraphData, documentando su estructura,
# configuración y propiedades científicas para garantizar la reproducibilidad del pipeline.
# Producto: - graph_data_metadata.json
print("\n" + "-" * 80)
print("7.8.3 EXPORTACIÓN DE METADATOS")
print("-" * 80)

# GraphData de referencia
if not graph_data_collection:
    raise ValueError(
        "La colección GraphData está vacía."
    )

reference_graph = next(iter(graph_data_collection.values()))

# Construcción de los metadatos
metadata = {
    # Información general
    "pipeline": "GraphData",
    "created_at": datetime.now().isoformat(),

    # Cobertura temporal
    "years": [int(year) for year in sorted(graph_data_collection.keys())],
    "first_year": int(min(graph_data_collection.keys())),
    "last_year": int(max(graph_data_collection.keys())),
    "number_of_graphs": len(graph_data_collection),

    # Estructura del grafo
    "number_of_nodes": int(reference_graph.num_nodes),
    "number_of_edges": int(reference_graph.num_edges),
    "number_of_features": int(reference_graph.num_node_features),

    # Configuración científica
    "target_variable": TARGET_VARIABLE,
    "graph_method": GRAPH_SPATIAL_CONFIG["method"],
    "weighted_graph": GRAPH_CONFIG["weighted"],
    "feature_source": NODE_FEATURE_CONFIG["feature_source"],
    "feature_columns": feature_columns,
    "node_index_column": NODE_INDEX_COLUMN,

    # Tipos de datos
    "dtype_x": str(reference_graph.x.dtype),
    "dtype_y": str(reference_graph.y.dtype),
    "dtype_edge_index": str(reference_graph.edge_index.dtype),

    # Tamaño de la colección
    "collection_size": len(graph_data_collection)

}

# Definición de la ruta
METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

metadata_file = (
    METADATA_DIR
    / "graph_data_metadata.json"
)

# Exportación
with open(
    metadata_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
        ensure_ascii=False
    )

# Verificación
if not metadata_file.exists():
    raise FileNotFoundError(
        "No fue posible exportar los metadatos."
    )

# 7.8.4 Exportación de Auditorías --------------------------------------------------------------
# Objetivo: Exportar la auditoría científica de la construcción de GraphData,
# consolidando los indicadores de calidad e integridad generados para cada año procesado.
# Producto: - graph_data_audit.parquet
print("\n" + "-" * 80)
print("7.8.4 EXPORTACIÓN DE AUDITORÍAS")
print("-" * 80)

# Validación de la auditoría
if not graph_audit_records:
    raise ValueError(
        "No existen registros de auditoría para exportar."
    )

# Construcción del DataFrame
audit_df = pd.DataFrame(graph_audit_records)

# Definición de la ruta
audit_file = (
    AUDITS_DIR
    / "graph_data_audit.parquet"
)

# Exportación
audit_df.to_parquet(
    audit_file,
    index=False
)

# Verificación
if not audit_file.exists():
    raise FileNotFoundError(
        "No fue posible exportar la auditoría."
    )

# 7.8.5 Verificación de la Exportación --------------------------------------------------------------
# Objetivo: Verificar la correcta exportación de todos los productos científicos
# generados durante la construcción del GraphData.
# Verificaciones: - GraphData anual - Colección de GraphData - Metadatos - Auditoría
# Producto: - Validación integral de la exportación
print("\n" + "-" * 80)
print("7.8.5 VERIFICACIÓN DE LA EXPORTACIÓN")
print("-" * 80)

# Archivos esperados --------------------------------------------------------------
expected_files = []

# GraphData anual
for year in years:

    expected_files.append(
        GRAPH_DATA_DIR / f"graph_data_{year}.pt"
    )

# Colección
expected_files.append(
    GRAPH_DATA_DIR / "graph_data_collection.pt"
)

# Metadatos
expected_files.append(
    METADATA_DIR / "graph_data_metadata.json"
)

# Auditoría
expected_files.append(
    AUDITS_DIR / "graph_data_audit.parquet"
)

# Verificación -----------------------------------------------------------------------------
verification_results = []
for file in expected_files:
    verification_results.append({
        "file": file.name,
        "exists": file.exists(),
        "size_mb": (
            round(file.stat().st_size / (1024 ** 2), 3)
            if file.exists()
            else None
        )
    })

verification_df = pd.DataFrame(
    verification_results
)

# Validación integral -----------------------------------------------------------------------------
if not verification_df["exists"].all():
    missing_files = verification_df.loc[
        ~verification_df["exists"],
        "file"
    ].tolist()

    raise FileNotFoundError(
        "No fue posible verificar todos los archivos exportados.\n\n"
        f"Archivos faltantes:\n{missing_files}"
    )

# Reporte Ejecutivo de la Exportación -----------------------------------------------------------------------------
print("\n" + "-" * 80)
print("REPORTE EJECUTIVO DE LA EXPORTACIÓN")
print("-" * 80)

print("\n1. Información General")
print("-" * 80)
print(f"Años procesados            : {len(graph_data_collection)}")
print(f"Primer año                 : {min(graph_data_collection.keys())}")
print(f"Último año                 : {max(graph_data_collection.keys())}")

print("\n2. Productos Científicos Exportados")
print("-" * 80)
print("1. Colección de GraphData")
print(f"   Archivo                 : {collection_file.name}")
print(f"   Directorio              : {GRAPH_DATA_DIR}")

print("\n2. Metadatos del GraphData")
print(f"   Archivo                 : {metadata_file.name}")
print(f"   Directorio              : {METADATA_DIR}")

print("\n3. Registro de Auditoría Científica")
print(f"   Archivo                 : {audit_file.name}")
print(f"   Directorio              : {AUDITS_DIR}")

print("\n3. Resumen de la Colección")
print("-" * 80)
print(f"GraphData                 : {metadata['number_of_graphs']}")
print(f"Nodos                     : {metadata['number_of_nodes']:,}")
print(f"Aristas                   : {metadata['number_of_edges']:,}")
print(f"Variables predictoras     : {metadata['number_of_features']}")

print("\n4. Verificación de la Exportación")
print("-" * 80)
print(f"Productos esperados       : {len(expected_files)}")
print(f"Productos encontrados     : {verification_df['exists'].sum()}")
print(f"Estado                    : OK")

print("\n5. Estado Final del Pipeline")
print("-" * 80)
print("Colección GraphData       : Exportada")
print("Metadatos                 : Exportados")
print("Auditoría Científica      : Exportada")
print("Verificación              : Completada")
print("Pipeline Científico       : Finalizado correctamente")

print("-" * 80)