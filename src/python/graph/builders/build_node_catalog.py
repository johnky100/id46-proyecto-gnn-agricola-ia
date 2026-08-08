# build_node_catalog.py

# =============================================================================
# BLOQUE 1. INFORMACIÓN DEL MÓDULO
# =============================================================================
# Objetivo: Definir la identidad, versión y propósito científico del módulo responsable de construir
# el Catálogo Oficial de Nodos utilizado en la generación del GraphData del proyecto.
# Arquitectura: Este módulo construye el Catálogo Oficial de Nodos a partir del Dataset Científico, 
# asignando un identificador secuencial único a cada municipio y preservando la correspondencia entre
# los identificadores territoriales originales y los índices utilizados por Graph Neural Networks (GNN).
# Producto: Catálogo Oficial de Nodos (GeoDataFrame).
# Pregunta Científica: ¿Cómo construir un Catálogo Oficial de Nodos geoespacial,  consistente y reproducible
# para representar correctamente la estructura territorial utilizada por Graph Neural Networks?
# =============================================================================

# =============================================================================
# 1.1 MODULE_INFO
# =============================================================================
MODULE_INFO = {
    # Nombre oficial del módulo
    "name": "Node Catalog Builder",

    # Versión del software
    "version": "1.0.0",

    # Versión de la especificación científica
    "node_catalog_specification_version": "1.0",

    # Descripción del módulo
    "description": (
        "Builds the official GeoDataFrame-based Node Catalog by assigning "
        "scientific node identifiers, graph indices and valid geometries "
        "for spatial graph construction."
    ),

    # Nombre del módulo
    "module": "build_node_catalog",

    # Pipeline al que pertenece
    "pipeline": "Graph Construction",

    # Autor del proyecto
    "author": "AVANZADO-IA",
}

# =============================================================================
# BLOQUE 2. IMPORTACIONES
# =============================================================================
# Objetivo: Importar las constantes, configuraciones y dependencias necesarias para construir el 
# Catálogo Oficial de Nodos del GraphData.
# Arquitectura: - Configuración del Proyecto - Configuración del Grafo
# Producto: Dependencias oficiales requeridas para la construcción del Catálogo Oficial de Nodos.
# Pregunta Científica: ¿Qué dependencias son necesarias para construir el Catálogo Oficial de Nodos del proyecto?
# =============================================================================
import pandas as pd # Manipulación de datos tabulares
import geopandas as gpd # Construcción del Catálogo Oficial de Nodos

from shapely import wkb # Conversión de geometrías WKB a objetos Shapely

# =============================================================================
# 2.1 CONFIGURACIÓN DEL PROYECTO
# =============================================================================

from src.python.config.config_project import (
    MUNICIPALITY_ID_COLUMN,
    MUNICIPALITY_NAME_COLUMN,
    DEPARTMENT_NAME_COLUMN,
    NODE_ID_COLUMN,
    NODE_INDEX_COLUMN,
    GEOMETRY_COLUMN,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
)

# =============================================================================
# BLOQUE 3. CONFIGURACIÓN DEL BUILDER
# =============================================================================
# Objetivo: Definir la configuración utilizada durante la construcción del Catálogo Oficial de Nodos,
# estableciendo los criterios que garantizan un proceso consistente, reproducible y compatible con el GraphData.
# Arquitectura: - Configuración del Builder
# Producto: Configuración oficial para la construcción del Catálogo Oficial de Nodos.
# Pregunta Científica: ¿Qué parámetros controlan la construcción del Catálogo Oficial de
# Nodos para garantizar su consistencia y reproducibilidad?
# =============================================================================

# =============================================================================
# 3.1 BUILD_CONFIG
# =============================================================================
BUILD_CONFIG = {
    # Validar las entradas antes de iniciar la construcción del catálogo.
    "validate_inputs": True,

    # Preservar el orden oficial de los municipios durante la construcción.
    "preserve_node_order": True,

    # Eliminar registros duplicados antes de generar los identificadores.
    "remove_duplicates": True,

    # Generar identificadores secuenciales (node_id) para cada municipio.
    "generate_node_ids": True,

    # Incorporar información de trazabilidad y metadatos al catálogo generado.
    "include_metadata": True,

    # Ejecutar validaciones estrictas durante todo el proceso de construcción.
    "strict_mode": True,
}

# =============================================================================
# BLOQUE 4. FUNCIÓN build_node_catalog()
# =============================================================================
# Objetivo: Construir el Catálogo Oficial de Nodos a partir del Dataset Científico, asignando identificadores
# secuenciales únicos a cada municipio para su utilización en la construcción del GraphData.
# Arquitectura: 4.1 Validación del Dataset 4.2 Selección de Municipios 4.3 Eliminación de Duplicados
# 4.4 Generación del Node Catalog 4.5 Validación del Catálogo 4.6 Incorporación de Información de Trazabilidad
# 4.7 Retorno del Catálogo
# Producto: Catálogo Oficial de Nodos con identificadores únicos y reproducibles para todos los 
# municipios utilizados por el proyecto.
# Pregunta Científica: ¿Cómo construir un Catálogo Oficial de Nodos consistente, reproducible
# y compatible con GraphData para representar los municipios del estudio?
# =============================================================================
def build_node_catalog(
    dataset: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Construye el Catálogo Oficial de Nodos del proyecto.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset Científico oficial del proyecto.

    Returns
    -------
    gpd.GeoDataFrame
        Catálogo Oficial de Nodos con identificadores científicos,
        índices internos para PyTorch Geometric y geometrías válidas
        para la construcción del Grafo Espacial Oficial.
    """

    # =========================================================================
    # 4.1 VALIDACIÓN DEL DATASET
    # =========================================================================
    if BUILD_CONFIG["validate_inputs"]:
        # ---------------------------------------------------------------------
        # Validar existencia del Dataset Científico
        # ---------------------------------------------------------------------

        if dataset is None:
            raise ValueError(
                "The scientific dataset cannot be None."
            )

        # ---------------------------------------------------------------------
        # Validar tipo del Dataset Científico
        # ---------------------------------------------------------------------
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(
                "The scientific dataset must be a pandas DataFrame."
            )

        # ---------------------------------------------------------------------
        # Validar que el Dataset no esté vacío
        # ---------------------------------------------------------------------
        if dataset.empty:
            raise ValueError(
                "The scientific dataset is empty."
            )

        # ---------------------------------------------------------------------
        # Validar columnas obligatorias
        # ---------------------------------------------------------------------
        required_columns = [
            MUNICIPALITY_ID_COLUMN,
            MUNICIPALITY_NAME_COLUMN,
            DEPARTMENT_NAME_COLUMN,
            LATITUDE_COLUMN,
            LONGITUDE_COLUMN,
            GEOMETRY_COLUMN,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in dataset.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                f"{missing_columns}"
            )

        # ---------------------------------------------------------------------
        # Validar valores nulos
        # ---------------------------------------------------------------------
        for column in required_columns:
            if dataset[column].isna().any():
                raise ValueError(
                    f"Column '{column}' contains missing values."
                )

        # ---------------------------------------------------------------------
        # Validar coordenadas geográficas
        # ---------------------------------------------------------------------
        if (
            (dataset[LATITUDE_COLUMN] < -90)
            | (dataset[LATITUDE_COLUMN] > 90)
        ).any():
            raise ValueError(
                "Latitude values are outside the valid range (-90, 90)."
            )

        if (
            (dataset[LONGITUDE_COLUMN] < -180)
            | (dataset[LONGITUDE_COLUMN] > 180)
        ).any():
            raise ValueError(
                "Longitude values are outside the valid range (-180, 180)."
            )

        # ---------------------------------------------------------------------
        # Validar códigos de municipio vacíos
        # ---------------------------------------------------------------------
        if (
            dataset[MUNICIPALITY_ID_COLUMN]
            .astype(str)
            .str.strip()
            .eq("")
            .any()
        ):
            raise ValueError(
                "Municipality codes cannot be empty."
            )

        # ---------------------------------------------------------------------
        # Validar nombres de municipio vacíos
        # ---------------------------------------------------------------------
        if (
            dataset[MUNICIPALITY_NAME_COLUMN]
            .astype(str)
            .str.strip()
            .eq("")
            .any()
        ):
            raise ValueError(
                "Municipality names cannot be empty."
            )

        # ---------------------------------------------------------------------
        # Validar departamentos vacíos
        # ---------------------------------------------------------------------
        if (
            dataset[DEPARTMENT_NAME_COLUMN]
            .astype(str)
            .str.strip()
            .eq("")
            .any()
        ):
            raise ValueError(
                "Department names cannot be empty."
            )

    # =========================================================================
    # 4.2 SELECCIÓN DE MUNICIPIOS
    # =========================================================================
    municipality_catalog = dataset[
        [
            MUNICIPALITY_ID_COLUMN,
            MUNICIPALITY_NAME_COLUMN,
            DEPARTMENT_NAME_COLUMN,
            LATITUDE_COLUMN,
            LONGITUDE_COLUMN,
            GEOMETRY_COLUMN,
        ]
    ].copy() # Seleccionar las variables oficiales del Catálogo de Nodos

    # =========================================================================
    # 4.3 ELIMINACIÓN DE DUPLICADOS
    # =========================================================================
    if BUILD_CONFIG["remove_duplicates"]:
        municipality_catalog = municipality_catalog.drop_duplicates(
            subset=[
                MUNICIPALITY_ID_COLUMN,
            ],
            keep="first",
        ).reset_index(
            drop=True,
        )

    # =========================================================================
    # 4.4 GENERACIÓN DEL CATÁLOGO OFICIAL DE NODOS
    # =========================================================================
    if BUILD_CONFIG["generate_node_ids"]:

        # ---------------------------------------------------------------------
        # 4.4.1 Generación de Identificadores de Nodo
        # ---------------------------------------------------------------------
        municipality_catalog[NODE_ID_COLUMN] = range(
            len(municipality_catalog)
        ) # Identificador científico permanente del nodo

        municipality_catalog[NODE_INDEX_COLUMN] = range(
            len(municipality_catalog)
        ) # Índice interno utilizado por PyTorch Geometric

        # ---------------------------------------------------------------------
        # 4.4.2 Conversión de geometrías
        # ---------------------------------------------------------------------
        if municipality_catalog[GEOMETRY_COLUMN].isnull().any():
            raise ValueError(
                "Existen geometrías faltantes."
            )

        municipality_catalog[GEOMETRY_COLUMN] = municipality_catalog[
            GEOMETRY_COLUMN
        ].apply(
            wkb.loads
        ) # Convertir geometrías WKB a objetos Shapely

        # ---------------------------------------------------------------------
        # 4.4.3 Reorganización de columnas
        # ---------------------------------------------------------------------
        municipality_catalog = municipality_catalog[
            [
                NODE_ID_COLUMN,
                NODE_INDEX_COLUMN,
                MUNICIPALITY_ID_COLUMN,
                MUNICIPALITY_NAME_COLUMN,
                DEPARTMENT_NAME_COLUMN,
                LATITUDE_COLUMN,
                LONGITUDE_COLUMN,
                GEOMETRY_COLUMN,
            ]
        ].copy() # Organizar el Catálogo Oficial de Nodos

        # ---------------------------------------------------------------------
        # 4.4.4 Construcción del Catálogo Oficial de Nodos
        # ---------------------------------------------------------------------
        node_catalog = gpd.GeoDataFrame(
            municipality_catalog,
            geometry=GEOMETRY_COLUMN,
        ) # Construir el Catálogo Oficial de Nodos


    # =========================================================================
    # 4.5 VALIDACIÓN DEL CATÁLOGO OFICIAL DE NODOS
    # =========================================================================
    if BUILD_CONFIG["validate_inputs"]:

        # ---------------------------------------------------------------------
        # Validar que el Catálogo Oficial de Nodos no esté vacío
        # ---------------------------------------------------------------------
        if node_catalog.empty:
            raise ValueError(
                "El Catálogo Oficial de Nodos está vacío."
            )

        # ---------------------------------------------------------------------
        # Validar columnas obligatorias
        # ---------------------------------------------------------------------
        required_columns = [
            NODE_ID_COLUMN,
            NODE_INDEX_COLUMN,
            MUNICIPALITY_ID_COLUMN,
            MUNICIPALITY_NAME_COLUMN,
            DEPARTMENT_NAME_COLUMN,
            LATITUDE_COLUMN,
            LONGITUDE_COLUMN,
            GEOMETRY_COLUMN,
        ] # Columnas obligatorias del Catálogo Oficial de Nodos

        missing_columns = [
            column
            for column in required_columns
            if column not in node_catalog.columns
        ] # Identificar columnas faltantes

        if missing_columns:
            raise ValueError(
                "Faltan las siguientes columnas obligatorias: "
                f"{missing_columns}."
            )

        # ---------------------------------------------------------------------
        # Validar identificadores científicos
        # ---------------------------------------------------------------------
        if node_catalog[NODE_ID_COLUMN].duplicated().any():
            raise ValueError(
                "Existen identificadores científicos de nodo duplicados."
            )

        if node_catalog[NODE_ID_COLUMN].isnull().any():
            raise ValueError(
                "Existen identificadores científicos de nodo faltantes."
            )

        # ---------------------------------------------------------------------
        # Validar índices internos del grafo
        # ---------------------------------------------------------------------
        if node_catalog[NODE_INDEX_COLUMN].duplicated().any():
            raise ValueError(
                "Existen índices internos de nodo duplicados."
            )

        if node_catalog[NODE_INDEX_COLUMN].isnull().any():
            raise ValueError(
                "Existen índices internos de nodo faltantes."
            )

        expected_node_index = list(range(len(node_catalog)))

        if node_catalog[NODE_INDEX_COLUMN].tolist() != expected_node_index:
            raise ValueError(
                "Los índices internos del grafo no son secuenciales."
            )

        # ---------------------------------------------------------------------
        # Validar códigos de municipio
        # ---------------------------------------------------------------------
        if node_catalog[MUNICIPALITY_ID_COLUMN].duplicated().any():
            raise ValueError(
                "Existen códigos de municipio duplicados."
            )

        # ---------------------------------------------------------------------
        # Validar valores faltantes
        # ---------------------------------------------------------------------
        if node_catalog[required_columns].isnull().any().any():
            raise ValueError(
                "Existen valores faltantes en el Catálogo Oficial de Nodos."
            )

        # ---------------------------------------------------------------------
        # Validar geometrías
        # ---------------------------------------------------------------------
        if node_catalog[GEOMETRY_COLUMN].isnull().any():
            raise ValueError(
                "Existen geometrías faltantes."
            )

        if not node_catalog.geometry.is_valid.all():
            raise ValueError(
                "Existen geometrías inválidas."
            )

    # =========================================================================
    # 4.6 INCORPORACIÓN DE INFORMACIÓN DE TRAZABILIDAD
    # =========================================================================
    if BUILD_CONFIG["include_metadata"]:
        node_catalog.attrs["builder"] = MODULE_INFO["name"]
        node_catalog.attrs["version"] = MODULE_INFO["version"]
        node_catalog.attrs["num_nodes"] = len(node_catalog)
        node_catalog.attrs["num_departments"] = (
            node_catalog[DEPARTMENT_NAME_COLUMN].nunique()
        )

    # =========================================================================
    # 4.7 RETORNO DEL CATÁLOGO
    # =========================================================================
    return node_catalog

# =============================================================================
# BLOQUE 5. EXPORTACIONES DEL MÓDULO
# =============================================================================
# Objetivo: Definir la interfaz pública del módulo, especificando las funciones
# disponibles para la construcción del Catálogo Oficial de Nodos dentro del pipeline científico.
# Producto: Funciones públicas exportadas por el módulo.
# =============================================================================

__all__ = [
    "build_node_catalog",
]
