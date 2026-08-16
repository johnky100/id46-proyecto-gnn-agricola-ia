# 01_build_graph.py

# Bloques 1–5: Ingeniería de datos y construcción del grafo (independientes de GeoPandas).
# Bloque 6: Visualización cartográfica (GeoPandas como herramienta de representación).
# Bloques 7 en adelante: Modelado, evaluación, pronóstico y plataforma GeoAI utilizando los 
# # GraphData construidos.

# BLOQUE 1. IMPORTACIÓN DE DEPENDENCIAS Y CONFIGURACIÓN CIENTÍFICA
# Objetivo: Importar las dependencias oficiales del proyecto, cargar la configuración científica
# entralizada y registrar los módulos especializados responsables de la construcción del Grafo Científico.
# Arquitectura científica
# Entradas: Librerías estándar, librerías científicas, configuración oficial y builders especializados.
# Producto: Entorno científico inicializado y configuración oficial cargada.
# Pregunta científica: ¿El entorno dispone de todas las dependencias, configuraciones y builders necesarios
# para construir el GraphData de forma reproducible?

# 1.1 LIBRERÍAS ESTÁNDAR
import warnings
import random
import json
from datetime import datetime
from pathlib import Path

# 1.2 LIBRERÍAS CIENTÍFICAS
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")

# 1.3 UTILIDADES OFICIALES DEL PROYECTO
from src.python.utils.data_preparation import (
    load_dataset,
    validate_dataset,
)

# 1.4 CONFIGURACIÓN GENERAL DEL PROYECTO
from src.python.config.config_project import (
    PROJECT_SEED,
    PROJECT_NAME,
    PROJECT_VERSION,
    TARGET_VARIABLE,
    FEATURE_COLUMNS,
    MUNICIPALITY_ID_COLUMN,
    DEPARTMENT_NAME_COLUMN,
    TIME_COLUMN,
)

# 1.5 RUTAS OFICIALES DEL PROYECTO
from src.python.config.paths import (
    validate_project_structure,
    OUTPUTS_DIR,
    GRAPH_DATA_DIR,
    GRAPH_FILES_DIR,
    GRAPH_NODE_FEATURES_DIR,
    GRAPH_EDGE_INDEX_DIR,
    GRAPH_EDGE_WEIGHTS_DIR,
    GRAPH_DATA_COLLECTION_FILE,
    NODE_CATALOG_FILE,
    PANEL_YEARS_FILE,
    GRAPH_EDGE_INDEX_FILE,
    GRAPH_EDGE_FEATURES_FILE,
    GRAPH_EDGE_WEIGHTS_FILE,
    GRAPH_TOPOLOGY_FILE,
    GRAPH_SCHEMA_FILE,
    GRAPH_METADATA_FILE,
    GRAPH_MANIFEST_FILE,
    GRAPH_CONTRACT_FILE,
    GRAPH_VALIDATION_FILE,
    GRAPH_STATISTICS_FILE,
    GRAPH_AUDIT_FILE,
    GRAPH_COLLECTION_METADATA_FILE,
    GRAPH_COLLECTION_CERTIFICATE_FILE,
    GRAPH_SUMMARY_FILE,
)

# 1.6 BUILDERS OFICIALES DEL GRAFO
from src.python.graph.builders.prepare_year_dataset import (
    prepare_year_dataset,
)

from src.python.graph.builders.build_node_catalog import (
    build_node_catalog,
)

from src.python.graph.builders.build_features import (
    build_features,
)

from src.python.graph.builders.build_spatial_edges import (
    prepare_spatial_edges,
)

from src.python.graph.builders.build_dynamic_edges import (
    prepare_dynamic_edges,
)

from src.python.graph.builders.build_combined_edges import (
    prepare_combined_edges,
)

from src.python.graph.builders.build_graphdata import (
    prepare_graphdata,
)

print("-" * 80)
print("IMPORTACIÓN CORRECTA")
print(prepare_graphdata)
print("-" * 80)

# BLOQUE 2. CONFIGURACIÓN DEL ENTORNO DE EJECUCIÓN
# Objetivo: Configurar el entorno oficial del proyecto, validar la estructura del Pipeline Científico,
# garantizar la reproducibilidad y seleccionar el dispositivo oficial de procesamiento.
# Arquitectura científica
# Entradas: • Configuración oficial del proyecto • Estructura oficial de directorios
# Producto: • Entorno científico inicializado • Pipeline validado
# • Reproducibilidad garantizada • Dispositivo de procesamiento configurado
# Pregunta científica: ¿El entorno cumple las condiciones necesarias para ejecutar el Pipeline
# GraphData de forma reproducible?

# 2.1 CONFIGURACIÓN GENERAL DEL ENTORNO
print("\n" + "-" * 80)
print("BLOQUE 2. CONFIGURACIÓN DEL ENTORNO DE EJECUCIÓN")
print("-" * 80)

# Información general del proyecto
warnings.filterwarnings("ignore") # Suprimir advertencias no críticas
execution_timestamp = datetime.now() # Registrar la fecha y hora de ejecución

print(f"Proyecto                  : {PROJECT_NAME}")
print(f"Versión                   : {PROJECT_VERSION}")
print(f"Fecha de ejecución        : {execution_timestamp:%Y-%m-%d %H:%M:%S}")
print(f"Directorio de trabajo     : {Path.cwd()}")

print("-" * 80)
print("Configuración general inicializada correctamente.")
print("-" * 80)

# 2.2 VALIDACIÓN DE LA ESTRUCTURA OFICIAL DEL PROYECTO
PROJECT_STRUCTURE_VALID = validate_project_structure(
    verbose=True,
) # Validar la estructura oficial del proyecto

if not PROJECT_STRUCTURE_VALID:
    raise RuntimeError(
        "La estructura oficial del proyecto no superó la validación."
    )

print("Estructura oficial del proyecto validada correctamente.")

# 2.3 CONFIGURACIÓN DE LA REPRODUCIBILIDAD
random.seed(PROJECT_SEED) # Inicializar la semilla del generador aleatorio de Python
np.random.seed(PROJECT_SEED) # Inicializar la semilla del generador aleatorio de NumPy
torch.manual_seed(PROJECT_SEED) # Inicializar la semilla del generador aleatorio de PyTorch

CUDA_AVAILABLE = torch.cuda.is_available() # Verificar la disponibilidad de CUDA
if CUDA_AVAILABLE:
    torch.cuda.manual_seed(PROJECT_SEED) # Inicializar la semilla de la GPU actual
    torch.cuda.manual_seed_all(PROJECT_SEED) # Inicializar la semilla de todas las GPU disponibles
    torch.backends.cudnn.deterministic = True # Forzar operaciones determinísticas
    torch.backends.cudnn.benchmark = False # Desactivar optimizaciones no determinísticas

try:
    torch.use_deterministic_algorithms(True) # Activar algoritmos determinísticos

except RuntimeError as error:
    warnings.warn(
        f"No fue posible activar los algoritmos determinísticos: {error}"
    )

print("Configuración de reproducibilidad inicializada correctamente.")
print(f"Semilla científica        : {PROJECT_SEED}")
print(f"CUDA disponible           : {CUDA_AVAILABLE}")
print(f"Versión de PyTorch        : {torch.__version__}")

# 2.4 CONFIGURACIÓN DEL DISPOSITIVO DE PROCESAMIENTO
DEVICE = torch.device(
    "cuda"
    if CUDA_AVAILABLE
    else "cpu"
) # Seleccionar el dispositivo oficial de procesamiento

print("Dispositivo de procesamiento configurado correctamente.")
print(f"Dispositivo               : {DEVICE}")

if CUDA_AVAILABLE:
    print(f"GPU                   : {torch.cuda.get_device_name(0)}")
    print(f"Número de GPU         : {torch.cuda.device_count()}")

# 2.5 CONFIRMACIÓN DEL ENTORNO DE EJECUCIÓN
print("-" * 80)
print("Entorno de ejecución inicializado correctamente.")
print("-" * 80)

print(f"Proyecto                  : {PROJECT_NAME}")
print(f"Versión                   : {PROJECT_VERSION}")
print(f"Dispositivo               : {DEVICE}")
print(f"Semilla científica        : {PROJECT_SEED}")
print(f"CUDA disponible           : {CUDA_AVAILABLE}")
print(f"Directorio de trabajo     : {Path.cwd()}")

print("-" * 80)
print("Pipeline listo para la construcción del GraphData.")
print("-" * 80)

# BLOQUE 3. CARGA Y VALIDACIÓN DEL DATASET CIENTÍFICO
# Objetivo: Cargar, validar e inspeccionar el Dataset Científico oficial para
# construir el Catálogo Oficial de Nodos, la Feature Matrix y los GraphData.
# Arquitectura científica
# Entradas: • Dataset Científico oficial
# Producto: • Dataset Científico cargado • Dataset Científico validado • Resumen científico del dataset
# Pregunta científica: ¿El Dataset Científico oficial fue cargado y validado correctamente para
# iniciar la construcción reproducible del Pipeline GraphData?

print("\n" + "-" * 80)
print("BLOQUE 3. CARGA Y VALIDACIÓN DEL DATASET CIENTÍFICO")
print("-" * 80)

# 3.1 CARGA DEL DATASET CIENTÍFICO
dataset = load_dataset(
) # Cargar el Dataset Científico Oficial

if dataset is None:
    raise RuntimeError(
        "No fue posible cargar el Dataset Científico."
    )

if not isinstance(dataset, pd.DataFrame):
    raise TypeError(
        "El Dataset Científico debe ser un DataFrame de pandas."
    )

print("Dataset Científico cargado correctamente.")
print(f"Registros                 : {len(dataset):,}")
print(f"Variables                 : {dataset.shape[1]:,}")

# 3.2 VALIDACIÓN DEL DATASET CIENTÍFICO
dataset = validate_dataset(
    dataset=dataset,
) # Validar la estructura e integridad del Dataset Científico

if dataset is None:
    raise RuntimeError(
        "La validación del Dataset Científico no retornó un resultado válido."
    )

if not isinstance(dataset, pd.DataFrame):
    raise TypeError(
        "El Dataset Científico validado debe ser un DataFrame de pandas."
    )

if dataset.empty:
    raise RuntimeError(
        "El Dataset Científico se encuentra vacío."
    )

print("Dataset Científico validado correctamente.")
print(f"Registros                 : {len(dataset):,}")
print(f"Variables                 : {dataset.shape[1]:,}")
print("Estado                     : CORRECTO")

# 3.3 CARACTERIZACIÓN DEL DATASET CIENTÍFICO
n_records = dataset.shape[0] # Número total de registros
n_variables = dataset.shape[1] # Número total de variables

memory_mb = (
    dataset.memory_usage(deep=True).sum()
    / (1024 ** 2)
) # Memoria utilizada por el Dataset Científico (MB)

n_municipalities = dataset[
    MUNICIPALITY_ID_COLUMN
].nunique() # Número de municipios

start_year = dataset[
    TIME_COLUMN
].min() # Año inicial del panel

end_year = dataset[
    TIME_COLUMN
].max() # Año final del panel

n_years = dataset[
    TIME_COLUMN
].nunique() # Número de años científicos

print("Caracterización del Dataset Científico")
print(f"Proyecto                  : {PROJECT_NAME}")
print(f"Versión                   : {PROJECT_VERSION}")
print(f"Registros                 : {n_records:,}")
print(f"Variables                 : {n_variables:,}")
print(f"Memoria (MB)              : {memory_mb:.2f}")
print(f"Municipios                : {n_municipalities:,}")
print(f"Años científicos          : {n_years}")
print(f"Periodo                   : {start_year} - {end_year}")
print(f"Variable objetivo         : {TARGET_VARIABLE}")
print("Estado                     : CORRECTO")

# 3.4 INSPECCIÓN DE VARIABLES
print("Listado de Variables del Dataset Científico")
for index, column in enumerate(dataset.columns, start=1):
    dtype = dataset[column].dtype # Tipo de dato de la variable
    print(
        f"{index:>2}. "
        f"{column:<40} "
        f"{str(dtype)}"
    )

print("-" * 80)
print(f"Total de variables        : {len(dataset.columns):,}")

# 3.5 CONFIRMACIÓN DEL DATASET CIENTÍFICO
print("-" * 80)
print("Dataset Científico preparado correctamente.")
print("-" * 80)

print(f"Registros                 : {n_records:,}")
print(f"Variables                 : {n_variables:,}")
print(f"Municipios                : {n_municipalities:,}")
print(f"Años científicos          : {n_years}")
print(f"Periodo                   : {start_year} - {end_year}")
print(f"Variable objetivo         : {TARGET_VARIABLE}")

print("-" * 80)
print("Pipeline listo para la construcción del Catálogo Oficial de Nodos.")
print("-" * 80)

# BLOQUE 4. PREPARACIÓN CIENTÍFICA DEL GRAFO ESPACIO-TEMPORAL
# Objetivo: Preparar la información científica necesaria para construir los GraphData mediante la
# generación del Catálogo Oficial de Nodos y de la dimensión temporal del Panel Científico.
# Arquitectura científica
# Entradas: • Dataset Científico validado
# Producto: • Catálogo Oficial de Nodos • Dimensión temporal del Panel Científico
# Pregunta científica: ¿El Dataset Científico dispone de la información necesaria para construir
# de forma reproducible los GraphData del proyecto?

print("\n" + "-" * 80)
print("BLOQUE 4. PREPARACIÓN CIENTÍFICA DEL GRAFO ESPACIO-TEMPORAL")
print("-" * 80)

# 4.1 CONSTRUCCIÓN DEL CATÁLOGO OFICIAL DE NODOS
node_catalog = build_node_catalog(
    dataset=dataset,
) # Construir el Catálogo Oficial de Nodos

if node_catalog is None:
    raise RuntimeError(
        "No fue posible construir el Catálogo Oficial de Nodos."
    )

if node_catalog.empty:
    raise RuntimeError(
        "El Catálogo Oficial de Nodos está vacío."
    )

n_nodes = len(node_catalog) # Número total de nodos
n_departments = node_catalog[
    DEPARTMENT_NAME_COLUMN
].nunique() # Número de departamentos

n_municipalities = node_catalog[
    MUNICIPALITY_ID_COLUMN
].nunique() # Número de municipios

print("Catálogo Oficial de Nodos construido correctamente.")
print(f"Nodos                    : {n_nodes:,}")
print(f"Departamentos            : {n_departments:,}")
print(f"Municipios               : {n_municipalities:,}")
print("Estado                    : CORRECTO")

# 4.2 PREPARACIÓN DE LA DIMENSIÓN TEMPORAL
years = sorted(
    dataset[
        TIME_COLUMN
    ].dropna().unique().tolist()
) # Obtener los años del Panel Científico

if len(years) == 0:
    raise RuntimeError(
        "No se encontraron años en el Dataset Científico."
    )

expected_years = list(
    range(
        years[0],
        years[-1] + 1,
    )
) # Construir la secuencia temporal esperada

if years != expected_years:
    raise RuntimeError(
        "El Panel Científico presenta años faltantes."
    )

start_year = years[0] # Año inicial del Panel Científico
end_year = years[-1] # Año final del Panel Científico
n_years = len(years) # Número de años científicos

print("Dimensión temporal preparada correctamente.")
print(f"Años científicos         : {n_years}")
print(f"Periodo                  : {start_year} - {end_year}")
print(f"Primer año               : {start_year}")
print(f"Último año               : {end_year}")
print("Estado                    : CORRECTO")

# 4.3 REGISTRO DE LOS ARTEFACTOS CIENTÍFICOS
graph_artifacts = {
    "node_catalog": NODE_CATALOG_FILE.name,
    "panel_years": PANEL_YEARS_FILE.name,
} # Artefactos científicos base del pipeline

print("Artefactos científicos registrados correctamente.")
print(f"Catálogo Oficial de Nodos : {graph_artifacts['node_catalog']}")
print(f"Panel Científico          : {graph_artifacts['panel_years']}")
print(f"Total de artefactos       : {len(graph_artifacts)}")
print("Estado                     : CORRECTO")

# 4.4 CONFIRMACIÓN DEL BLOQUE
print("-" * 80)
print("Estructuras base del grafo preparadas correctamente.")
print("-" * 80)

print(f"Nodos oficiales           : {n_nodes:,}")
print(f"Municipios                : {n_municipalities:,}")
print(f"Departamentos             : {n_departments:,}")
print(f"Años científicos          : {n_years}")
print(f"Periodo                   : {start_year} - {end_year}")
print(f"Artefactos registrados    : {len(graph_artifacts)}")
print("Estado                     : CORRECTO")

print("-" * 80)
print("Pipeline listo para la construcción de la colección oficial de GraphData.")
print("-" * 80)

# BLOQUE 5. CONSTRUCCIÓN CIENTÍFICA DE LA COLECCIÓN OFICIAL DE GRAPHDATA
# Objetivo: Construir la colección oficial de GraphData mediante la generación de la topología espacial,
# las Node Features y los objetos GraphData para cada año del período de estudio.
# Arquitectura científica
# Entradas: • Dataset Científico validado • Catálogo Oficial de Nodos
# • Dimensión temporal del Panel Científico
# Producto: • Colección oficial de GraphData • Artefactos científicos preparados en memoria
# Pregunta científica: ¿Los GraphData del Panel Científico fueron construidos correctamente para 
# representar de forma reproducible la dinámica espacio-temporal del proyecto?

# 5.0 INICIALIZACIÓN DE LA COLECCIÓN OFICIAL DE GRAPHDATA
print("\n" + "-" * 80)
print("5.0 INICIALIZACIÓN DE LA COLECCIÓN OFICIAL DE GRAPHDATA")
print("-" * 80)

# 5.0.1 Inicialización de la Colección Oficial
graph_data_collection = [] # Colección oficial de GraphData
processed_years = [] # Años científicos procesados correctamente
graph_node_features_collection = {} # Matrices de características por año
graph_edge_index_collection = {} # Aristas combinadas por año

# 5.0.2 Inicialización de Artefactos Científicos
graph_collection_metadata = {} # Metadatos de la colección GraphData
graph_summary = {} # Resumen científico de la colección GraphData

# 5.0.3 Confirmación de la Inicialización
print("Colección oficial inicializada correctamente.")
print(f"GraphData acumulados       : {len(graph_data_collection):,}")
print("Estado                      : CORRECTO")
print("-" * 80)

# 5.1 CONSTRUCCIÓN DE LA TOPOLOGÍA ESPACIAL OFICIAL
edge_index_spatial = prepare_spatial_edges(
    node_catalog=node_catalog,
) # Construir las Aristas Espaciales Oficiales

if edge_index_spatial is None:
    raise RuntimeError(
        "No fue posible construir las Aristas Espaciales Oficiales."
    )

n_edges_spatial = edge_index_spatial.shape[1] # Número de Aristas Espaciales
if n_edges_spatial == 0:
    raise RuntimeError(
        "Las Aristas Espaciales Oficiales están vacías."
    )

graph_topology = {
    "nodes": n_nodes,
    "spatial_edges": n_edges_spatial,
    "directed": False,
    "weighted": False,
    "graph_type": "official_spatial_graph",
    "builder": "01_build_graph",
    "version": PROJECT_VERSION,
} # Información estructural del Grafo Espacial Oficial

print("Topología Espacial construida correctamente.")
print(f"Nodos oficiales            : {n_nodes:,}")
print(f"Aristas espaciales         : {n_edges_spatial:,}")

# 5.2 CONSTRUCCIÓN DE LOS GRAPHDATA OFICIALES
for current_year in years:
    print("\n" + "-" * 80)
    print(f"Año científico: {current_year}")
    print("-" * 80)

    # 5.2.1 Preparación del Dataset Científico Anual  
    dataset_year = prepare_year_dataset(
        dataset=dataset,
        current_year=current_year,
        node_catalog=node_catalog,
    ) # Preparar el Dataset Científico Anual

    if dataset_year is None:
        raise RuntimeError(
            f"No fue posible preparar el Dataset Científico del año {current_year}."
        )

    input("Presione Enter para continuar...")
  
    # 5.2.2 Construcción de la Matriz de Características  
    x, y = build_features(
        dataset_year=dataset_year,
    ) # Construir la matriz de características y la variable objetivo

    if x is None or y is None:
        raise RuntimeError(
            f"No fue posible construir la matriz de características del año {current_year}."
        )

    graph_node_features_collection[current_year] = x # Registrar la matriz de características
    if current_year not in graph_node_features_collection:
        raise RuntimeError(
            "No fue posible registrar la matriz de características."
        )
    input("Presione Enter para continuar...")
   
    # 5.2.3 Construcción de las Aristas Dinámicas   
    edge_index_dynamic = prepare_dynamic_edges(
        dataset_year=dataset_year,
    ) # Construir las Aristas Dinámicas Oficiales

    if edge_index_dynamic is None:
        raise RuntimeError(
            f"No fue posible construir las Aristas Dinámicas del año {current_year}."
        )
    input("Presione Enter para continuar...")
 
    # 5.2.4 Construcción de las Aristas Combinadas   
    edge_index_final = prepare_combined_edges(
        edge_index_spatial=edge_index_spatial,
        edge_index_dynamic=edge_index_dynamic,
        n_nodes=x.shape[0],
    ) # Construir las Aristas Combinadas Oficiales

    if edge_index_final is None:
        raise RuntimeError(
            f"No fue posible construir las Aristas Combinadas del año {current_year}."
        )

    graph_edge_index_collection[current_year] = edge_index_final # Registrar las Aristas Combinadas
    if current_year not in graph_edge_index_collection:
        raise RuntimeError(
            "No fue posible registrar las Aristas Combinadas."
        )
    input("Presione Enter para continuar...")
 
    # 5.2.5 Construcción del GraphData Oficial  
    graph_data = prepare_graphdata(
        x=x,
        y=y,
        edge_index_final=edge_index_final,
        current_year=current_year,
    ) # Construir el GraphData Oficial

    if graph_data is None:
        raise RuntimeError(
            f"No fue posible construir el GraphData del año {current_year}."
        )
    input("Presione Enter para continuar...")
 
    # 5.2.6 Incorporación del GraphData a la Colección Oficial  
    graph_data_collection.append(
        graph_data
    ) # Incorporar el GraphData Oficial

    processed_years.append(
        current_year
    ) # Registrar el año procesado correctamente

    print("GraphData construido correctamente.")
    print(f"Año                      : {current_year}")
    print(f"Nodos                    : {graph_data.num_nodes:,}")
    print(f"Variables                : {graph_data.x.shape[1]:,}")
    print(f"Aristas                  : {graph_data.edge_index.shape[1]:,}")
    print(f"GraphData acumulados     : {len(graph_data_collection):,}")

    input("Presione Enter para continuar...")

# 5.3 VALIDACIÓN DE LA COLECCIÓN OFICIAL DE GRAPHDATA
expected_graphs = len(years) # Número esperado de GraphData
generated_graphs = len(graph_data_collection) # Número de GraphData construidos
if generated_graphs != expected_graphs:
    raise RuntimeError(
        "La colección oficial de GraphData está incompleta."
    )

# Metadatos de la Colección Oficial
if len(set(processed_years)) != len(processed_years):
    raise RuntimeError(
        "Se detectaron años científicos duplicados."
    )

if not processed_years:
    raise RuntimeError(
        "No existen años científicos procesados."
    )

graph_collection_metadata = {
    "graphs": generated_graphs,
    "start_year": processed_years[0],
    "end_year": processed_years[-1],
    "years": len(processed_years),
    "nodes": n_nodes,
    "spatial_edges": n_edges_spatial,
    "graph_type": "official_graphdata",
    "builder": "01_build_graph",
    "builder_version": PROJECT_VERSION,
} # Metadatos oficiales de la colección

# Resumen Científico
graph_summary = {
    "status": "SUCCESS",
    "graphs": generated_graphs,
    "years": len(processed_years),
    "nodes": n_nodes,
    "spatial_edges": n_edges_spatial,
    "period": f"{start_year}-{end_year}",
} # Resumen científico de la colección

print("Colección oficial de GraphData construida correctamente.")
print(f"GraphData construidos     : {generated_graphs:,}")
print(f"Años científicos          : {len(processed_years):,}")
print(f"Nodos oficiales           : {n_nodes:,}")
print(f"Aristas espaciales        : {n_edges_spatial:,}")
print("Colección Oficial de GraphData preparada correctamente.")
print("Pipeline listo para la exportación de los artefactos científicos.")
print("-" * 80)

# BLOQUE 6. EXPORTACIÓN OFICIAL DE LOS ARTEFACTOS CIENTÍFICOS
# Objetivo: Exportar de forma reproducible los artefactos científicos oficiales generados durante la 
# construcción del Pipeline GraphData, garantizando su persistencia, integridad y disponibilidad para 
# las etapas de entrenamiento, evaluación, pronóstico y despliegue de la Plataforma Inteligente GeoAI.
# Arquitectura científica
# Entradas: • Colección oficial de GraphData • Artefactos científicos generados durante
# la construcción del grafo • Directorios oficiales del proyecto
# Producto: • Colección oficial de GraphData • Artefactos científicos del GraphData • Metadatos y resúmenes oficiales
# Pregunta científica: ¿Los artefactos científicos oficiales fueron exportados y validados correctamente,
# preservando la reproducibilidad, integridad y trazabilidad del Pipeline Científico para las etapas
# posteriores del proyecto?

# 6.1 PREPARACIÓN DE LA EXPORTACIÓN OFICIAL
print("\n" + "-" * 80)
print("6.1 PREPARACIÓN DE LA EXPORTACIÓN OFICIAL")
print("-" * 80)

# 6.1.1 Directorios Oficiales de Exportación
export_directories = [
    GRAPH_DATA_DIR,
    GRAPH_FILES_DIR,
    GRAPH_NODE_FEATURES_DIR,
    GRAPH_EDGE_INDEX_DIR,
    GRAPH_EDGE_WEIGHTS_DIR,
] # Directorios oficiales de exportación

# 6.1.2 Creación de los Directorios Oficiales
for directory in export_directories:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    ) # Crear el directorio oficial si no existe

    if not directory.exists():
        raise RuntimeError(
            f"No fue posible crear el directorio oficial '{directory}'."
        )

# 6.1.3 Validación de los Directorios Oficiales
for directory in export_directories:
    if not directory.is_dir():
        raise RuntimeError(
            f"'{directory}' no corresponde a un directorio válido."
        )

# 6.1.4 Confirmación de la Preparación
print("Directorios oficiales preparados correctamente.")
for directory in export_directories:
    print(f"Directorio           : {directory}")

print(f"Total de directorios     : {len(export_directories)}")
print("Estado                    : CORRECTO")

# 6.2 EXPORTACIÓN DE LOS ARTEFACTOS CIENTÍFICOS
print("\n" + "-" * 80)
print("6.2 EXPORTACIÓN DE LOS ARTEFACTOS CIENTÍFICOS")
print("-" * 80)

# 6.2.1 Exportación del Catálogo Oficial de Nodos
if node_catalog is None:
    raise RuntimeError(
        "El Catálogo Oficial de Nodos no existe."
    )
node_catalog.to_parquet(
    NODE_CATALOG_FILE,
    index=False,
) # Exportar el Catálogo Oficial de Nodos

# 6.2.2 Exportación del Panel Científico
with open(
    PANEL_YEARS_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        processed_years,
        file,
        indent=4,
    ) # Exportar los años científicos procesados

# 6.2.3 Exportación de las Node Features
GRAPH_NODE_FEATURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
) # Crear directorio de Node Features

for current_year, feature_matrix in graph_node_features_collection.items():
    feature_file = (
        GRAPH_NODE_FEATURES_DIR
        / f"node_features_{current_year}.parquet"
    ) # Archivo oficial de Node Features

    pd.DataFrame(
        feature_matrix.cpu().numpy(),
    ).to_parquet(
        feature_file,
        index=False,
    ) # Exportar las Node Features

# 6.2.4 Exportación de las Aristas Combinadas
for current_year, edge_index in graph_edge_index_collection.items():
    edge_index_file = (
        GRAPH_EDGE_INDEX_DIR
        / f"edge_index_{current_year}.parquet"
    ) # Archivo oficial del Edge Index

    pd.DataFrame(
        edge_index.cpu().numpy().T,
        columns=[
            "source",
            "target",
        ],
    ).to_parquet(
        edge_index_file,
        index=False,
    ) # Exportar las Aristas Combinadas

# 6.2.5 Exportación de la Topología Espacial
with open(
    GRAPH_TOPOLOGY_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        graph_topology,
        file,
        indent=4,
    ) # Exportar la Topología Espacial Oficial

# 6.2.6 Confirmación de la Exportación
print("Artefactos científicos exportados correctamente.")
print(f"Node Features          : {len(graph_node_features_collection):,}")
print(f"Edge Index             : {len(graph_edge_index_collection):,}")
print(f"Años científicos       : {len(processed_years):,}")
print("Estado                  : CORRECTO")

# 6.3 EXPORTACIÓN DE LA COLECCIÓN OFICIAL DE GRAPHDATA
print("\n" + "-" * 80)
print("6.3 EXPORTACIÓN DE LA COLECCIÓN OFICIAL DE GRAPHDATA")
print("-" * 80)

# 6.3.1 Validación de la Colección Oficial
if graph_data_collection is None:
    raise RuntimeError(
        "La colección oficial de GraphData no existe."
    )

if len(graph_data_collection) == 0:
    raise RuntimeError(
        "La colección oficial de GraphData está vacía."
    )

# 6.3.2 Exportación de la Colección Oficial
# Exportación de cada GraphData
for graph_data in graph_data_collection:

    graph_file = (
        GRAPH_FILES_DIR
        / f"graph_{graph_data.current_year}.pt"
    ) # Archivo oficial del GraphData

    torch.save(
        graph_data,
        graph_file,
    ) # Exportar el GraphData oficial

    if not graph_file.exists():
        raise RuntimeError(
            f"No fue posible exportar el GraphData del año {graph_data.current_year}."
        )

# Exportación de la colección oficial completa
torch.save(
    graph_data_collection,
    GRAPH_DATA_COLLECTION_FILE,
) # Exportar la colección oficial de GraphData

if not GRAPH_DATA_COLLECTION_FILE.exists():
    raise RuntimeError(
        "No fue posible exportar la colección oficial de GraphData."
    )  

# 6.3.3 Exportación de los Metadatos Oficiales
with open(
    GRAPH_COLLECTION_METADATA_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        graph_collection_metadata,
        file,
        indent=4,
    ) # Exportar los metadatos oficiales

# 6.3.4 Exportación del Resumen Científico
with open(
    GRAPH_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        graph_summary,
        file,
        indent=4,
    ) # Exportar el resumen científico

# 6.3.5 Confirmación de la Exportación
print("Colección oficial de GraphData exportada correctamente.")
print(f"GraphData               : {len(graph_data_collection):,}")
print(f"Metadatos               : {GRAPH_COLLECTION_METADATA_FILE.name}")
print(f"Resumen                 : {GRAPH_SUMMARY_FILE.name}")
print(f"Colección               : {GRAPH_DATA_COLLECTION_FILE.name}")
print("Estado                   : CORRECTO")

# 6.4 VALIDACIÓN DE LA EXPORTACIÓN
print("\n" + "-" * 80)
print("6.4 VALIDACIÓN DE LA EXPORTACIÓN")
print("-" * 80)

# 6.4.1 Validación de los Artefactos Científicos Globales
required_files = [
    NODE_CATALOG_FILE,
    PANEL_YEARS_FILE,
    GRAPH_TOPOLOGY_FILE,
    GRAPH_DATA_COLLECTION_FILE,
    GRAPH_COLLECTION_METADATA_FILE,
    GRAPH_SUMMARY_FILE,
] # Artefactos científicos globales

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"No existe el artefacto científico '{file_path.name}'."
        )

# 6.4.2 Validación de las Node Features y Edge Index por Año
for current_year in processed_years:
    feature_file = (
        GRAPH_NODE_FEATURES_DIR
        / f"node_features_{current_year}.parquet"
    ) # Archivo oficial de Node Features

    if not feature_file.exists():
        raise FileNotFoundError(
            f"No existen las Node Features del año {current_year}."
        )

    edge_index_file = (
        GRAPH_EDGE_INDEX_DIR
        / f"edge_index_{current_year}.parquet"
    ) # Archivo oficial del Edge Index

    if not edge_index_file.exists():
        raise FileNotFoundError(
            f"No existe el Edge Index del año {current_year}."
        )

# 6.4.3 Validación de la Colección Oficial
validated_collection = torch.load(
    GRAPH_DATA_COLLECTION_FILE,
    weights_only=False,
) # Cargar la colección oficial de GraphData

if not isinstance(validated_collection, list):
    raise TypeError(
        "La colección exportada no corresponde a una lista de GraphData."
    )

if len(validated_collection) != len(graph_data_collection):
    raise RuntimeError(
        "La colección oficial exportada es inconsistente."
    )

for graph_data in validated_collection:
    if (
        not hasattr(graph_data, "x")
        or not hasattr(graph_data, "edge_index")
        or not hasattr(graph_data, "y")
    ):
        raise RuntimeError(
            "Se encontró un GraphData inválido en la colección."
        )

# 6.4.4 Validación del Periodo Científico
exported_years = [
    graph.current_year
    for graph in validated_collection
] # Años científicos exportados

if exported_years != processed_years:
    raise RuntimeError(
        "El periodo científico exportado es inconsistente."
    )

# 6.4.5 Confirmación de la Validación
print("Exportación validada correctamente.")
print(f"Artefactos globales  : {len(required_files):,}")
print(f"Node Features        : {len(processed_years):,}")
print(f"Edge Index           : {len(processed_years):,}")
print(f"GraphData validados  : {len(validated_collection):,}")
print(
    f"Periodo científico     : "
    f"{exported_years[0]} - {exported_years[-1]}"
)
print("Estado                : CORRECTO")

# 6.5 RESUMEN CIENTÍFICO DE LA EXPORTACIÓN
print("\n" + "-" * 80)
print("6.5 RESUMEN CIENTÍFICO DE LA EXPORTACIÓN")
print("-" * 80)

# 6.5.1 Resumen General del Proyecto
print(f"Proyecto                  : {PROJECT_NAME}")
print(f"Versión                   : {PROJECT_VERSION}")
print(
    f"Periodo científico        : "
    f"{processed_years[0]} - {processed_years[-1]}"
)
print(f"Años procesados           : {len(processed_years):,}")
print(f"GraphData construidos     : {len(graph_data_collection):,}")
print(f"Nodos oficiales           : {n_nodes:,}")
print(f"Aristas espaciales        : {n_edges_spatial:,}")
print(f"Variables predictoras     : {len(FEATURE_COLUMNS):,}")

# 6.5.2 Artefactos Científicos Globales
generated_files = [
    NODE_CATALOG_FILE,
    PANEL_YEARS_FILE,
    GRAPH_TOPOLOGY_FILE,
    GRAPH_DATA_COLLECTION_FILE,
    GRAPH_COLLECTION_METADATA_FILE,
    GRAPH_SUMMARY_FILE,
] # Artefactos científicos globales

print("\nArtefactos científicos globales:")
for file_path in generated_files:
    print(f"- {file_path.name}")

# 6.5.3 Node Features Exportadas
print("\nGraphData exportados:")
for current_year in processed_years:
    graph_file = (
        GRAPH_FILES_DIR
        / f"graph_{current_year}.pt"
    )
    print(f"- {graph_file.name}")

# 6.5.4 Edge Index Exportados
print("\nEdge Index exportados:")
for current_year in processed_years:
    edge_index_file = (
        GRAPH_EDGE_INDEX_DIR
        / f"edge_index_{current_year}.parquet"
    ) # Archivo oficial del Edge Index
    print(f"- {edge_index_file.name}")

# 6.5.5 Confirmación de la Exportación
total_artifacts = (
    len(generated_files)
    + len(processed_years)  # GraphData
    + len(processed_years)  # Node Features
    + len(processed_years)  # Edge Index
) # Globales + Node Features + Edge Index

print("-" * 80)
print(f"Artefactos globales    : {len(generated_files):,}")
print(f"Node Features          : {len(processed_years):,}")
print(f"Edge Index             : {len(processed_years):,}")
print(f"GraphData              : {len(processed_years):,}")
print(f"Total de artefactos    : {total_artifacts:,}")
print("Estado final            : CORRECTO")
print("Pipeline GraphData finalizado correctamente.")
print("-" * 80)
print("BLOQUE 6 FINALIZADO CORRECTAMENTE")
print("-" * 80)