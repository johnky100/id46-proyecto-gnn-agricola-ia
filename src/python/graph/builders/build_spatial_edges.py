# build_spatial_edges.py

# =============================================================================
# BUILD_SPATIAL_EDGES.PY
# Objetivo: Construir el Grafo Espacial Oficial del proyecto a partir del Catálogo Oficial de Nodos,
# generando la estructura de conectividad espacial permanente utilizada por el GraphData.
# Arquitectura científica
# Entradas: Catálogo Oficial de Nodos.
# Producto: edge_index_spatial y edge_weight_spatial.
# Pregunta científica: ¿La estructura espacial representa correctamente las relaciones territoriales permanentes entre los municipios?
# =============================================================================

# =============================================================================
# 1. IMPORTACIÓN DE DEPENDENCIAS
# =============================================================================
import torch # Construcción de tensores para PyTorch Geometric
import geopandas as gpd # Operaciones espaciales sobre geometrías

from libpysal.weights import Queen # Construcción de contigüidad espacial Queen

from src.python.config.config_project import (
    NODE_INDEX_COLUMN,
    GEOMETRY_COLUMN,
)

# =============================================================================
# 2. VALIDACIÓN DEL CATÁLOGO OFICIAL DE NODOS
# =============================================================================
# Objetivo: Verificar que el Catálogo Oficial de Nodos cumpla los requisitos estructurales y espaciales
# necesarios para construir las aristas espaciales oficiales del proyecto.
# Entradas: - Catálogo Oficial de Nodos (GeoDataFrame).
# Producto: - Confirmación de la integridad estructural y espacial del Catálogo Oficial de Nodos.
# Responde:
# ¿El Catálogo Oficial de Nodos posee la estructura requerida para
# construir correctamente las aristas espaciales mediante contigüidad Queen?
# =============================================================================
def validate_node_catalog(
    node_catalog: gpd.GeoDataFrame,
) -> None:
    """
    Valida la integridad estructural del Catálogo Oficial de Nodos.

    Parameters
    ----------
    node_catalog : gpd.GeoDataFrame
        Catálogo Oficial de Nodos utilizado para construir el
        Grafo Espacial Oficial.

    Returns
    -------
    None
        La función finaliza correctamente cuando el Catálogo Oficial
        de Nodos cumple todos los requisitos estructurales.
    """

    print("-" * 80)
    print("Validación del Catálogo Oficial de Nodos")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # Validación de la estructura del Catálogo Oficial de Nodos
    # -------------------------------------------------------------------------
    if node_catalog is None:
        raise ValueError(
            "El Catálogo Oficial de Nodos no puede ser None."
        )

    if not isinstance(node_catalog, gpd.GeoDataFrame):
        raise TypeError(
            "El Catálogo Oficial de Nodos debe ser un GeoDataFrame."
        )

    if node_catalog.empty:
        raise ValueError(
            "El Catálogo Oficial de Nodos está vacío."
        )

    # -------------------------------------------------------------------------
    # Validación de columnas obligatorias
    # -------------------------------------------------------------------------
    required_columns = [
        NODE_INDEX_COLUMN,
        GEOMETRY_COLUMN,
    ] # Columnas mínimas requeridas para construir las aristas espaciales

    missing_columns = [
        column
        for column in required_columns
        if column not in node_catalog.columns
    ] # Identificar columnas faltantes

    if missing_columns:
        raise ValueError(
            "Faltan las siguientes columnas obligatorias: "
            f"{', '.join(missing_columns)}."
        )

    # -------------------------------------------------------------------------
    # Validación de identificadores
    # -------------------------------------------------------------------------
    if node_catalog[NODE_INDEX_COLUMN].duplicated().any():
        raise ValueError(
            "Existen identificadores de nodo duplicados."
        )

    if node_catalog[NODE_INDEX_COLUMN].isnull().any():
        raise ValueError(
            "Existen identificadores de nodo faltantes."
        )

    expected_node_index = list(range(len(node_catalog))) # Secuencia esperada de índices

    if node_catalog[NODE_INDEX_COLUMN].tolist() != expected_node_index:
        raise ValueError(
            "Los índices internos del grafo no son secuenciales."
        )

    # -------------------------------------------------------------------------
    # Validación de geometrías
    # -------------------------------------------------------------------------
    if node_catalog[GEOMETRY_COLUMN].isnull().any():
        raise ValueError(
            "Existen geometrías faltantes."
        )

    if not node_catalog.geometry.is_valid.all():
        raise ValueError(
            "Existen geometrías inválidas."
        )

    # -------------------------------------------------------------------------
    # Resumen científico
    # -------------------------------------------------------------------------
    print("Catálogo Oficial de Nodos validado correctamente.")
    print(f"Nodos                    : {len(node_catalog):,}")
    print("Estado                   : CORRECTO")


# =============================================================================
# 3. CONSTRUCCIÓN DE LAS ARISTAS ESPACIALES
# =============================================================================

def build_spatial_edges(
    node_catalog: gpd.GeoDataFrame,
) -> torch.Tensor:
    """
    Construye las aristas espaciales oficiales del proyecto a partir
    del Catálogo Oficial de Nodos mediante contigüidad Queen.

    Parameters
    ----------
    node_catalog : gpd.GeoDataFrame
        Catálogo Oficial de Nodos validado.

    Returns
    -------
    torch.Tensor
        Tensor edge_index_spatial con la conectividad espacial
        utilizada por Graph Neural Networks.
    """

    print("-" * 80)
    print("Construcción de las Aristas Espaciales")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # Construcción de la contigüidad espacial
    # -------------------------------------------------------------------------
    spatial_graph = Queen.from_dataframe(
        node_catalog,
        use_index=False,
    ) # Construir la contigüidad espacial Queen

    # -------------------------------------------------------------------------
    # Construcción de las aristas espaciales
    # -------------------------------------------------------------------------
    source_nodes = [] # Nodos de origen

    target_nodes = [] # Nodos de destino

    for source_node, neighbors in spatial_graph.neighbors.items():

        for target_node in neighbors:

            source_nodes.append(source_node) # Registrar nodo de origen

            target_nodes.append(target_node) # Registrar nodo de destino

    # -------------------------------------------------------------------------
    # Construcción del tensor edge_index_spatial
    # -------------------------------------------------------------------------
    edge_index_spatial = torch.tensor(
        [
            source_nodes,
            target_nodes,
        ],
        dtype=torch.long,
    ) # Construir el tensor de conectividad espacial

    # -------------------------------------------------------------------------
    # Validación de la construcción
    # -------------------------------------------------------------------------
    if edge_index_spatial.numel() == 0:
        raise ValueError(
            "No fue posible construir las aristas espaciales."
        )

    # -------------------------------------------------------------------------
    # Resumen científico
    # -------------------------------------------------------------------------
    print("Aristas espaciales construidas correctamente.")
    print(f"Nodos                    : {len(node_catalog):,}")
    print(f"Aristas                  : {edge_index_spatial.shape[1]:,}")
    print("Estado                   : CORRECTO")

    return edge_index_spatial

# =============================================================================
# 4. VALIDACIÓN DE LAS ARISTAS ESPACIALES
# =============================================================================
# Objetivo: Verificar que el tensor edge_index_spatial represente correctamente
# la conectividad espacial oficial del proyecto antes de ser utilizado por Graph Neural Networks.
# Entradas: - edge_index_spatial. - Catálogo Oficial de Nodos.
# Producto: - Confirmación de la integridad estructural de las aristas espaciales.
# Responde: ¿Las aristas espaciales representan correctamente la conectividad
# territorial permanente del Grafo Espacial Oficial?
# =============================================================================

def validate_spatial_edges(
    edge_index_spatial: torch.Tensor,
    node_catalog: gpd.GeoDataFrame,
) -> None:
    """
    Valida la integridad estructural de las aristas espaciales.
    """

    print("-" * 80)
    print("Validación de las Aristas Espaciales")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # Validación del tensor
    # -------------------------------------------------------------------------
    if edge_index_spatial is None:
        raise ValueError(
            "Las aristas espaciales no pueden ser None."
        )

    if not isinstance(edge_index_spatial, torch.Tensor):
        raise TypeError(
            "Las aristas espaciales deben ser un tensor de PyTorch."
        )

    # -------------------------------------------------------------------------
    # Validación de dimensiones
    # -------------------------------------------------------------------------
    if edge_index_spatial.ndim != 2:
        raise ValueError(
            "El tensor edge_index_spatial debe tener dos dimensiones."
        )

    if edge_index_spatial.shape[0] != 2:
        raise ValueError(
            "La primera dimensión de edge_index_spatial debe ser igual a 2."
        )

    if edge_index_spatial.shape[1] == 0:
        raise ValueError(
            "No existen aristas espaciales."
        )

    # -------------------------------------------------------------------------
    # Validación de índices
    # -------------------------------------------------------------------------
    if edge_index_spatial.min().item() < 0:
        raise ValueError(
            "Existen índices negativos en las aristas espaciales."
        )

    if edge_index_spatial.max().item() >= len(node_catalog):
        raise ValueError(
            "Existen índices de nodo fuera del rango permitido."
        )

    # -------------------------------------------------------------------------
    # Resumen científico
    # -------------------------------------------------------------------------
    print("Aristas espaciales validadas correctamente.")
    print(f"Nodos                    : {len(node_catalog):,}")
    print(f"Aristas                  : {edge_index_spatial.shape[1]:,}")
    print("Estado                   : CORRECTO")

# =============================================================================
# 5. CONSTRUCCIÓN OFICIAL DE LAS ARISTAS ESPACIALES
# =============================================================================
# Objetivo:
# Centralizar la validación y construcción de las aristas espaciales
# oficiales del proyecto para entregar una única interfaz al resto
# del pipeline científico.
#
# Entradas:
# - Catálogo Oficial de Nodos.
#
# Producto:
# - Tensor edge_index_spatial.
#
# Responde:
# ¿Las aristas espaciales oficiales fueron construidas correctamente
# para representar la conectividad territorial permanente?
# =============================================================================

def prepare_spatial_edges(
    node_catalog: gpd.GeoDataFrame,
) -> torch.Tensor:
    """
    Construye las aristas espaciales oficiales del proyecto.

    Parameters
    ----------
    node_catalog : gpd.GeoDataFrame
        Catálogo Oficial de Nodos.

    Returns
    -------
    torch.Tensor
        Tensor edge_index_spatial listo para ser utilizado por
        PyTorch Geometric.
    """

    # -------------------------------------------------------------------------
    # Validación del Catálogo Oficial de Nodos
    # -------------------------------------------------------------------------
    validate_node_catalog(
        node_catalog
    ) # Validar el Catálogo Oficial de Nodos

    # -------------------------------------------------------------------------
    # Construcción de las Aristas Espaciales
    # -------------------------------------------------------------------------
    edge_index_spatial = build_spatial_edges(
        node_catalog
    ) # Construir las aristas espaciales

    # -------------------------------------------------------------------------
    # Validación de las Aristas Espaciales
    # -------------------------------------------------------------------------
    validate_spatial_edges(
        edge_index_spatial,
        node_catalog,
    ) # Validar las aristas espaciales

    # -------------------------------------------------------------------------
    # Retorno
    # -------------------------------------------------------------------------
    return edge_index_spatial