# build_dynamic_edges.py

# =============================================================================
# BUILD_DYNAMIC_EDGES.PY
# Objetivo: Construir las Aristas Dinámicas Oficiales del proyecto a partir del
# Dataset Científico Anual, generando la conectividad basada en similitud entre
# municipios utilizada por el GraphData.
#
# Arquitectura científica
# Entradas: Dataset Científico Anual.
# Producto: edge_index_dynamic.
# Pregunta científica: ¿Las aristas dinámicas representan correctamente las
# relaciones de similitud entre los municipios para un año específico?
# =============================================================================

# =============================================================================
# 1. IMPORTACIÓN DE DEPENDENCIAS
# =============================================================================
import torch # Construcción de tensores para PyTorch Geometric
import pandas as pd # Manipulación del Dataset Científico Anual

from sklearn.neighbors import NearestNeighbors # Construcción del grafo k-NN

from sklearn.preprocessing import StandardScaler # Estandarización de variables predictoras

from src.python.config.config_project import (
    NODE_INDEX_COLUMN,
    FEATURE_COLUMNS,
    GRAPH_DYNAMIC_K,
)

# =============================================================================
# 2. VALIDACIÓN DEL DATASET CIENTÍFICO ANUAL
# =============================================================================
# Objetivo:
# Verificar la integridad estructural del Dataset Científico Anual antes
# de construir las Aristas Dinámicas Oficiales del proyecto.
#
# Entradas:
# - Dataset Científico Anual.
#
# Producto:
# - Confirmación de la integridad estructural del Dataset Científico Anual.
#
# Responde:
# ¿El Dataset Científico Anual contiene la información necesaria para
# construir correctamente las Aristas Dinámicas Oficiales?
# =============================================================================

def validate_dataset_year(
    dataset_year: pd.DataFrame,
) -> None:
    """
    Valida la integridad estructural del Dataset Científico Anual.
    """

    print("-" * 80)
    print("Validación del Dataset Científico Anual")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # Validación del Dataset Científico Anual
    # -------------------------------------------------------------------------
    if dataset_year is None:
        raise ValueError(
            "El Dataset Científico Anual no puede ser None."
        )

    if not isinstance(dataset_year, pd.DataFrame):
        raise TypeError(
            "El Dataset Científico Anual debe ser un DataFrame."
        )

    if dataset_year.empty:
        raise ValueError(
            "El Dataset Científico Anual está vacío."
        )

    # -------------------------------------------------------------------------
    # Validación de columnas obligatorias
    # -------------------------------------------------------------------------
    required_columns = [
        NODE_INDEX_COLUMN,
        *FEATURE_COLUMNS,
    ] # Columnas mínimas requeridas para construir el grafo dinámico

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset_year.columns
    ] # Identificar columnas faltantes

    if missing_columns:
        raise ValueError(
            "Faltan las siguientes columnas obligatorias: "
            f"{', '.join(missing_columns)}."
        )

    # -------------------------------------------------------------------------
    # Validación del índice interno del grafo
    # -------------------------------------------------------------------------
    if dataset_year[NODE_INDEX_COLUMN].isnull().any():
        raise ValueError(
            "Existen índices internos del grafo faltantes."
        )

    if dataset_year[NODE_INDEX_COLUMN].duplicated().any():
        raise ValueError(
            "Existen índices internos del grafo duplicados."
        )

    expected_node_index = list(range(len(dataset_year))) # Secuencia esperada

    if dataset_year[NODE_INDEX_COLUMN].tolist() != expected_node_index:
        raise ValueError(
            "Los índices internos del grafo no son secuenciales."
        )

    # -------------------------------------------------------------------------
    # Validación de variables predictoras
    # -------------------------------------------------------------------------
    if dataset_year[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError(
            "Existen valores faltantes en las variables predictoras."
        )

    # -------------------------------------------------------------------------
    # Resumen científico
    # -------------------------------------------------------------------------
    print("Dataset Científico Anual validado correctamente.")
    print(f"Nodos                    : {len(dataset_year):,}")
    print(f"Variables predictoras    : {len(FEATURE_COLUMNS):,}")
    print("Estado                   : CORRECTO")

# =============================================================================
# 3. CONSTRUCCIÓN DE LAS ARISTAS DINÁMICAS
# =============================================================================
def build_dynamic_edges(
    dataset_year: pd.DataFrame,
    n_neighbors: int = GRAPH_DYNAMIC_K,
) -> torch.Tensor:
    """
    Construye las Aristas Dinámicas Oficiales del proyecto mediante
    un grafo k-Nearest Neighbors construido sobre las variables
    predictoras oficiales del Dataset Científico Anual.

    Parameters
    ----------
    dataset_year : pd.DataFrame
        Dataset Científico Anual validado.

    n_neighbors : int, default=GRAPH_DYNAMIC_K
        Número de vecinos considerados para construir el grafo dinámico.

    Returns
    -------
    torch.Tensor
        Tensor edge_index_dynamic utilizado por PyTorch Geometric.
    """

    print("-" * 80)
    print("Construcción de las Aristas Dinámicas")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # Construcción de la matriz de variables predictoras
    # -------------------------------------------------------------------------
    x_features = dataset_year[
        FEATURE_COLUMNS
    ].copy() # Variables predictoras oficiales

    # -------------------------------------------------------------------------
    # Estandarización de las variables predictoras
    # -------------------------------------------------------------------------
    scaler = StandardScaler() # Inicializar el estandarizador

    x_scaled = scaler.fit_transform(
        x_features
    ) # Estandarizar las variables predictoras

    if n_neighbors >= len(dataset_year):
        raise ValueError(
            "El número de vecinos debe ser menor que el número de nodos."
        ) # Validar el número de vecinos

    # -------------------------------------------------------------------------
    # Construcción del grafo k-Nearest Neighbors
    # -------------------------------------------------------------------------
    knn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric="euclidean",
    ) # Inicializar el modelo k-NN

    knn.fit(
        x_scaled
    ) # Ajustar el modelo k-NN

    neighbors = knn.kneighbors(
        return_distance=False
    ) # Obtener los vecinos más cercanos

    # -------------------------------------------------------------------------
    # Construcción de las aristas dinámicas
    # -------------------------------------------------------------------------
    source_nodes = [] # Nodos de origen

    target_nodes = [] # Nodos de destino

    node_index = dataset_year[
        NODE_INDEX_COLUMN
    ].to_numpy() # Índices internos del grafo

    for source_position, neighbor_positions in enumerate(neighbors):

        source_node = node_index[source_position]

        for target_position in neighbor_positions[1:]:

            target_node = node_index[target_position]

            source_nodes.append(source_node) # Registrar nodo de origen

            target_nodes.append(target_node) # Registrar nodo de destino

    # -------------------------------------------------------------------------
    # Construcción del tensor edge_index_dynamic
    # -------------------------------------------------------------------------
    edge_index_dynamic = torch.tensor(
        [
            source_nodes,
            target_nodes,
        ],
        dtype=torch.long,
    ) # Construir el tensor de conectividad dinámica

    # -------------------------------------------------------------------------
    # Validación de la construcción
    # -------------------------------------------------------------------------
    if edge_index_dynamic.numel() == 0:
        raise ValueError(
            "No fue posible construir las Aristas Dinámicas."
        )

    # -------------------------------------------------------------------------
    # Resumen científico
    # -------------------------------------------------------------------------
    print("Aristas Dinámicas construidas correctamente.")
    print(f"Nodos                    : {len(dataset_year):,}")
    print(f"Aristas                  : {edge_index_dynamic.shape[1]:,}")
    print(f"Vecinos por nodo         : {n_neighbors}")
    print("Estado                   : CORRECTO")

    return edge_index_dynamic

# =============================================================================
# 4. VALIDACIÓN DE LAS ARISTAS DINÁMICAS
# =============================================================================
# Objetivo:
# Verificar la integridad estructural de las Aristas Dinámicas Oficiales
# antes de su utilización en la construcción del GraphData.
#
# Entradas:
# - edge_index_dynamic.
# - Dataset Científico Anual.
#
# Producto:
# - Confirmación de la integridad estructural de las Aristas Dinámicas.
#
# Responde:
# ¿Las Aristas Dinámicas representan correctamente las relaciones de
# similitud entre los municipios del Dataset Científico Anual?
# =============================================================================

def validate_dynamic_edges(
    edge_index_dynamic: torch.Tensor,
    dataset_year: pd.DataFrame,
) -> None:
    """
    Valida la integridad estructural de las Aristas Dinámicas Oficiales.
    """

    print("-" * 80)
    print("Validación de las Aristas Dinámicas")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # Validación del tensor
    # -------------------------------------------------------------------------
    if edge_index_dynamic is None:
        raise ValueError(
            "Las Aristas Dinámicas no pueden ser None."
        )

    if not isinstance(edge_index_dynamic, torch.Tensor):
        raise TypeError(
            "Las Aristas Dinámicas deben ser un tensor de PyTorch."
        )

    # -------------------------------------------------------------------------
    # Validación de dimensiones
    # -------------------------------------------------------------------------
    if edge_index_dynamic.ndim != 2:
        raise ValueError(
            "El tensor edge_index_dynamic debe tener dos dimensiones."
        )

    if edge_index_dynamic.shape[0] != 2:
        raise ValueError(
            "La primera dimensión de edge_index_dynamic debe ser igual a 2."
        )

    if edge_index_dynamic.shape[1] == 0:
        raise ValueError(
            "No existen Aristas Dinámicas."
        )

    # -------------------------------------------------------------------------
    # Validación del número de aristas
    # -------------------------------------------------------------------------
    expected_min_edges = len(dataset_year) # Número mínimo esperado de aristas

    if edge_index_dynamic.shape[1] < expected_min_edges:
        raise ValueError(
            "El número de Aristas Dinámicas es inferior al número de nodos."
        )

    # -------------------------------------------------------------------------
    # Validación de índices
    # -------------------------------------------------------------------------
    if edge_index_dynamic.min().item() < 0:
        raise ValueError(
            "Existen índices negativos en las Aristas Dinámicas."
        )

    if edge_index_dynamic.max().item() >= len(dataset_year):
        raise ValueError(
            "Existen índices de nodo fuera del rango permitido."
        )

    # -------------------------------------------------------------------------
    # Resumen científico
    # -------------------------------------------------------------------------
    print("Aristas Dinámicas validadas correctamente.")
    print(f"Nodos                    : {len(dataset_year):,}")
    print(f"Aristas                  : {edge_index_dynamic.shape[1]:,}")
    print("Estado                   : CORRECTO")

# =============================================================================
# 5. CONSTRUCCIÓN OFICIAL DE LAS ARISTAS DINÁMICAS
# =============================================================================
# Objetivo:
# Centralizar la validación y construcción de las Aristas Dinámicas
# Oficiales del proyecto para entregar una única interfaz al resto
# del pipeline científico.
#
# Entradas:
# - Dataset Científico Anual.
#
# Producto:
# - Tensor edge_index_dynamic.
#
# Responde:
# ¿Las Aristas Dinámicas Oficiales fueron construidas correctamente
# para representar las relaciones de similitud entre los municipios?
# =============================================================================

def prepare_dynamic_edges(
    dataset_year: pd.DataFrame,
    n_neighbors: int = GRAPH_DYNAMIC_K,
) -> torch.Tensor:
    """
    Construye las Aristas Dinámicas Oficiales del proyecto.

    Parameters
    ----------
    dataset_year : pd.DataFrame
        Dataset Científico Anual.

    n_neighbors : int, default=GRAPH_DYNAMIC_K
        Número de vecinos utilizados para construir el grafo dinámico.

    Returns
    -------
    torch.Tensor
        Tensor edge_index_dynamic listo para ser utilizado por
        PyTorch Geometric.
    """

    # -------------------------------------------------------------------------
    # Validación del Dataset Científico Anual
    # -------------------------------------------------------------------------
    validate_dataset_year(
        dataset_year
    ) # Validar el Dataset Científico Anual

    # -------------------------------------------------------------------------
    # Construcción de las Aristas Dinámicas
    # -------------------------------------------------------------------------
    edge_index_dynamic = build_dynamic_edges(
        dataset_year=dataset_year,
        n_neighbors=n_neighbors,
    ) # Construir las Aristas Dinámicas

    # -------------------------------------------------------------------------
    # Validación de las Aristas Dinámicas
    # -------------------------------------------------------------------------
    validate_dynamic_edges(
        edge_index_dynamic=edge_index_dynamic,
        dataset_year=dataset_year,
    ) # Validar las Aristas Dinámicas

    # -------------------------------------------------------------------------
    # Retorno
    # -------------------------------------------------------------------------
    return edge_index_dynamic