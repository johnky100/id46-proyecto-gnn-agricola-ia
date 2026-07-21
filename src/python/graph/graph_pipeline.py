# graph_pipeline_py

# BLOQUE 0. Importación de dependencias
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Importar las librerías, clases y funciones requeridas para la validación,
# preparación y transformación de la colección GraphData utilizada por el
# pipeline GeoAI.
#
#------------------------------------------------------------------------------
# Dependencias
#------------------------------------------------------------------------------
# • NumPy
#       Operaciones numéricas y manipulación de arreglos.
#
# • PyTorch
#       Construcción y transformación de tensores.
#
# • Scikit-learn
#       Particionamiento del conjunto de datos y escalado de variables.
#
# • PyTorch Geometric
#       Representación de grafos mediante objetos GraphData.
#
#------------------------------------------------------------------------------
# Producto
#------------------------------------------------------------------------------
# Espacio de trabajo configurado con todas las dependencias necesarias para
# ejecutar los bloques del módulo graph_pipeline.py.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿El entorno de ejecución dispone de todas las dependencias necesarias para
# garantizar el funcionamiento reproducible del pipeline GeoAI?
#------------------------------------------------------------------------------
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Data

# BLOQUE 1. validate_graph_collection
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Verificar que la colección oficial de objetos GraphData cumple los criterios
# estructurales, dimensionales y científicos requeridos antes de ser utilizada
# por cualquier etapa del pipeline GeoAI.
#
# Este bloque constituye el primer control de calidad del proyecto y garantiza
# la integridad de los datos para las etapas de Benchmark, Entrenamiento,
# Evaluación, Forecasting y Plataforma GeoAI.
#------------------------------------------------------------------------------
# Producto. validation_report : dict
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿La colección oficial GraphData representa correctamente el sistema
# espacio-temporal y cumple los criterios metodológicos necesarios para ser
# utilizada durante el Benchmark, Entrenamiento, Evaluación, Forecasting y
# la Plataforma GeoAI?
#------------------------------------------------------------------------------

def validate_graph_collection(
    graphs: list,
    expected_nodes: int,
    expected_features: int,
    expected_years: int,
    expected_edges: int | None = None,
) -> dict:
    """
    Valida la colección oficial de objetos GraphData.
    """

    validation_report = {
        "collection_valid": True,
        "graphs": len(graphs),
        "nodes": expected_nodes,
        "edges": expected_edges,
        "features": expected_features,
        "warnings": [],
        "errors": [],
    }

    #----------------------------------------------------------------------
    # Validar la colección GraphData
    #----------------------------------------------------------------------

    _validate_graph_collection(graphs)

    #----------------------------------------------------------------------
    # Validar cada GraphData
    #----------------------------------------------------------------------

    for graph_id, graph in enumerate(graphs):

        _validate_graph_structure(
            graph=graph,
            graph_id=graph_id,
        )

        _validate_graph_dimensions(
            graph=graph,
            graph_id=graph_id,
            expected_nodes=expected_nodes,
            expected_features=expected_features,
            expected_edges=expected_edges,
        )

        _validate_graph_integrity(
            graph=graph,
            graph_id=graph_id,
        )

    #----------------------------------------------------------------------
    # Validar número esperado de grafos
    #----------------------------------------------------------------------

    if len(graphs) != expected_years:

        raise ValueError(
            f"Expected {expected_years} graphs but found {len(graphs)}."
        )

    return validation_report

# BLOQUE 2. _validate_graph_collection
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Verificar que la colección GraphData exista, tenga el tipo de dato esperado
# y contenga al menos un objeto GraphData antes de iniciar las validaciones
# individuales de cada grafo.
#
# Este bloque constituye la primera validación estructural del pipeline y evita
# errores derivados de colecciones vacías, tipos incorrectos o entradas
# inválidas.
#
#------------------------------------------------------------------------------
# Entradas
#------------------------------------------------------------------------------
# • graphs : list[torch_geometric.data.Data]
#       Colección oficial de grafos espacio-temporales.
#
#------------------------------------------------------------------------------
# Producto
#------------------------------------------------------------------------------
# None
#
# Si la colección cumple todos los criterios, la ejecución continúa.
# En caso contrario se genera una excepción.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿La colección GraphData posee una estructura válida para iniciar el proceso
# de validación científica del pipeline GeoAI?
#------------------------------------------------------------------------------


def _validate_graph_collection(graphs: list) -> None:
    """
    Valida la estructura básica de la colección GraphData.
    """

    if graphs is None:
        raise ValueError("The GraphData collection is None.")

    if not isinstance(graphs, list):
        raise TypeError("graphs must be a list.")

    if len(graphs) == 0:
        raise ValueError("The GraphData collection is empty.")

    for index, graph in enumerate(graphs):

        if not isinstance(graph, Data):
            raise TypeError(
                f"Element {index} is not a torch_geometric.data.Data object."
            )
        

# BLOQUE 3. _validate_graph_structure
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Verificar que cada objeto GraphData contenga los atributos estructurales
# mínimos requeridos por el pipeline GeoAI antes de validar sus dimensiones
# e integridad científica.
#
#------------------------------------------------------------------------------
# Entradas
#------------------------------------------------------------------------------
# • graph : torch_geometric.data.Data
#       Grafo espacio-temporal correspondiente a un año.
#
# • graph_id : int
#       Identificador del grafo dentro de la colección.
#
#------------------------------------------------------------------------------
# Producto
#------------------------------------------------------------------------------
# None
#
# Si la estructura del grafo es válida, la ejecución continúa.
# En caso contrario se genera una excepción.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿El objeto GraphData contiene todos los componentes estructurales
# necesarios para representar correctamente la red espacio-temporal?
#------------------------------------------------------------------------------

def _validate_graph_structure(
    graph: Data,
    graph_id: int,
) -> None:
    """
    Valida la estructura básica de un objeto GraphData.
    """

    required_attributes = [
        "x",
        "edge_index",
        "y",
    ]

    for attribute in required_attributes:

        if not hasattr(graph, attribute):

            raise AttributeError(
                f"Graph {graph_id} does not contain the required attribute "
                f"'{attribute}'."
            )

        if getattr(graph, attribute) is None:

            raise ValueError(
                f"Graph {graph_id} contains a null '{attribute}' attribute."
            )

    if graph.num_nodes is None:

        raise ValueError(
            f"Graph {graph_id} has an undefined number of nodes."
        )

    if graph.num_edges is None:

        raise ValueError(
            f"Graph {graph_id} has an undefined number of edges."
        )

# BLOQUE 4. _validate_graph_dimensions
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Verificar que las dimensiones estructurales de cada objeto GraphData sean
# consistentes con la configuración oficial del proyecto GeoAI.
#
#------------------------------------------------------------------------------
# Entradas
#------------------------------------------------------------------------------
# • graph : torch_geometric.data.Data
#       Grafo espacio-temporal correspondiente a un año.
#
# • graph_id : int
#       Identificador del grafo dentro de la colección.
#
# • expected_nodes : int
#       Número esperado de nodos.
#
# • expected_features : int
#       Número esperado de variables por nodo.
#
# • expected_edges : int | None
#       Número esperado de aristas.
#------------------------------------------------------------------------------
# Producto
#------------------------------------------------------------------------------
# None
#
# Si las dimensiones son correctas, la ejecución continúa.
# En caso contrario se genera una excepción.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿Las dimensiones del objeto GraphData son consistentes con la estructura
# oficial definida para el modelo espacio-temporal?
#------------------------------------------------------------------------------

def _validate_graph_dimensions(
    graph: Data,
    graph_id: int,
    expected_nodes: int,
    expected_features: int,
    expected_edges: int | None = None,
) -> None:
    """
    Valida las dimensiones estructurales de un GraphData.
    """

    #----------------------------------------------------------------------
    # Número de nodos
    #----------------------------------------------------------------------

    if graph.num_nodes != expected_nodes:

        raise ValueError(
            f"Graph {graph_id}: expected {expected_nodes} nodes "
            f"but found {graph.num_nodes}."
        )

    #----------------------------------------------------------------------
    # Número de variables
    #----------------------------------------------------------------------

    if graph.num_node_features != expected_features:

        raise ValueError(
            f"Graph {graph_id}: expected {expected_features} node features "
            f"but found {graph.num_node_features}."
        )

    #----------------------------------------------------------------------
    # Variable objetivo
    #----------------------------------------------------------------------

    if graph.y.size(0) != expected_nodes:

        raise ValueError(
            f"Graph {graph_id}: target dimension "
            f"({graph.y.size(0)}) does not match the number of nodes "
            f"({expected_nodes})."
        )

    #----------------------------------------------------------------------
    # Número de aristas
    #----------------------------------------------------------------------

    if expected_edges is not None:

        if graph.num_edges != expected_edges:

            raise ValueError(
                f"Graph {graph_id}: expected {expected_edges} edges "
                f"but found {graph.num_edges}."
            )

# BLOQUE 5. _validate_graph_integrity
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Verificar la integridad científica de cada objeto GraphData mediante la
# validación de sus atributos, garantizando que la información utilizada por
# el pipeline GeoAI sea consistente y apta para el entrenamiento de modelos
# espacio-temporales.
#
#------------------------------------------------------------------------------
# Entradas
#------------------------------------------------------------------------------
# • graph : torch_geometric.data.Data
#       Grafo espacio-temporal correspondiente a un año.
#
# • graph_id : int
#       Identificador del grafo dentro de la colección.
#------------------------------------------------------------------------------
# Producto
#------------------------------------------------------------------------------
# None
#
# Si el grafo cumple los criterios de integridad científica, la ejecución
# continúa. En caso contrario se genera una excepción.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿Los datos contenidos en el objeto GraphData poseen la calidad necesaria
# para garantizar un entrenamiento reproducible y confiable del modelo?
#------------------------------------------------------------------------------

def _validate_graph_integrity(
    graph: Data,
    graph_id: int,
) -> None:
    """
    Valida la integridad científica de un GraphData.
    """

    #----------------------------------------------------------------------
    # Variables predictoras
    #----------------------------------------------------------------------

    if torch.isnan(graph.x).any():

        raise ValueError(
            f"Graph {graph_id} contains NaN values in node features."
        )

    if torch.isinf(graph.x).any():

        raise ValueError(
            f"Graph {graph_id} contains infinite values in node features."
        )

    #----------------------------------------------------------------------
    # Variable objetivo
    #----------------------------------------------------------------------

    if torch.isnan(graph.y).any():

        raise ValueError(
            f"Graph {graph_id} contains NaN values in target variable."
        )

    if torch.isinf(graph.y).any():

        raise ValueError(
            f"Graph {graph_id} contains infinite values in target variable."
        )

    #----------------------------------------------------------------------
    # Índices de las aristas
    #----------------------------------------------------------------------

    if torch.isnan(graph.edge_index.float()).any():

        raise ValueError(
            f"Graph {graph_id} contains NaN values in edge_index."
        )

    if torch.isinf(graph.edge_index.float()).any():

        raise ValueError(
            f"Graph {graph_id} contains infinite values in edge_index."
        )

    #----------------------------------------------------------------------
    # Máscaras de entrenamiento (si existen)
    #----------------------------------------------------------------------

    for mask_name in ["train_mask", "val_mask", "test_mask"]:

        if hasattr(graph, mask_name):

            mask = getattr(graph, mask_name)

            if mask.size(0) != graph.num_nodes:

                raise ValueError(
                    f"Graph {graph_id}: '{mask_name}' has an invalid size."
                )

# BLOQUE 6. build_graph_partitions
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Construir las particiones oficiales de entrenamiento, validación y prueba
# para todos los nodos de la colección GraphData, garantizando que el mismo
# esquema de partición sea utilizado por todas las etapas del pipeline GeoAI.
#
#------------------------------------------------------------------------------
# Entradas
#------------------------------------------------------------------------------
# • graphs : list[torch_geometric.data.Data]
#       Colección oficial de grafos espacio-temporales.
#
# • train_size : float
#       Proporción destinada al conjunto de entrenamiento.
#
# • validation_size : float
#       Proporción destinada al conjunto de validación.
#
# • random_state : int
#       Semilla para garantizar la reproducibilidad.
#
#------------------------------------------------------------------------------
# Producto
#------------------------------------------------------------------------------
# partitions : dict
#
# Diccionario con los índices oficiales de entrenamiento, validación y prueba.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿La partición de los nodos garantiza un proceso de entrenamiento,
# validación y evaluación reproducible para toda la colección GraphData?
#------------------------------------------------------------------------------

def build_graph_partitions(
    graphs: list[Data],
    train_size: float = 0.70,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    Construye las particiones oficiales de entrenamiento,
    validación y prueba.
    """

    if len(graphs) == 0:
        raise ValueError("The GraphData collection is empty.")

    n_nodes = graphs[0].num_nodes

    node_indices = np.arange(n_nodes)

    train_indices, temp_indices = train_test_split(
        node_indices,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
    )

    validation_ratio = validation_size / (1.0 - train_size)

    validation_indices, test_indices = train_test_split(
        temp_indices,
        train_size=validation_ratio,
        random_state=random_state,
        shuffle=True,
    )

    partitions = {
        "train_indices": np.sort(train_indices),
        "validation_indices": np.sort(validation_indices),
        "test_indices": np.sort(test_indices),
    }

    return partitions

# BLOQUE 7. build_graph_masks
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Construir las máscaras oficiales de entrenamiento, validación y prueba para
# cada objeto GraphData de la colección, utilizando las particiones definidas
# previamente.
#
#------------------------------------------------------------------------------
# Entradas
#------------------------------------------------------------------------------
# • graphs : list[torch_geometric.data.Data]
#       Colección oficial de grafos espacio-temporales.
#
# • partitions : dict
#       Diccionario con los índices oficiales de entrenamiento, validación
#       y prueba.
#------------------------------------------------------------------------------
# Producto
#------------------------------------------------------------------------------
# graphs : list[torch_geometric.data.Data]
#
# Colección GraphData con las máscaras train_mask, val_mask y test_mask
# incorporadas en cada grafo.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿Todos los grafos utilizan exactamente la misma partición de nodos para
# garantizar un proceso de entrenamiento reproducible?
#------------------------------------------------------------------------------


def build_graph_masks(
    graphs: list,
    partitions: dict,
) -> list:
    """
    Construye las máscaras oficiales para cada GraphData.
    """

    train_indices = partitions["train_indices"]
    validation_indices = partitions["validation_indices"]
    test_indices = partitions["test_indices"]

    for graph in graphs:

        n_nodes = graph.num_nodes

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[train_indices] = True
        val_mask[validation_indices] = True
        test_mask[test_indices] = True

        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask

    return graphs

# BLOQUE 8. fit_graph_scaler
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Ajustar el escalador oficial utilizando únicamente los nodos pertenecientes
# al conjunto de entrenamiento, evitando la fuga de información hacia los
# conjuntos de validación y prueba.
#
#------------------------------------------------------------------------------
# Entradas
#------------------------------------------------------------------------------
# • graphs : list[torch_geometric.data.Data]
#       Colección oficial de grafos espacio-temporales.
#
#------------------------------------------------------------------------------
# Producto
#------------------------------------------------------------------------------
# scaler : sklearn.preprocessing.StandardScaler
#
# Escalador oficial del proyecto ajustado únicamente con los datos de
# entrenamiento.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿El escalador fue ajustado exclusivamente con información del conjunto de
# entrenamiento para garantizar una evaluación objetiva del modelo?
#------------------------------------------------------------------------------

def fit_graph_scaler(
    graphs: list,
) -> StandardScaler:
    """
    Ajusta el StandardScaler utilizando únicamente los nodos de entrenamiento.
    """

    train_features = []

    for graph in graphs:

        if not hasattr(graph, "train_mask"):

            raise AttributeError(
                "GraphData does not contain 'train_mask'."
            )

        train_features.append(
            graph.x[graph.train_mask].cpu().numpy()
        )

    train_features = np.vstack(train_features)

    scaler = StandardScaler()

    scaler.fit(train_features)

    return scaler

# BLOQUE 9. scale_graph_collection
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Aplicar el escalado oficial a las variables predictoras de toda la colección
# GraphData utilizando el StandardScaler ajustado previamente con los datos
# del conjunto de entrenamiento.
#
#------------------------------------------------------------------------------
# Entradas
#------------------------------------------------------------------------------
# • graphs : list[torch_geometric.data.Data]
#       Colección oficial de grafos espacio-temporales.
#
# • scaler : sklearn.preprocessing.StandardScaler
#       Escalador oficial previamente ajustado.
#------------------------------------------------------------------------------
# Producto
#------------------------------------------------------------------------------
# graphs : list[torch_geometric.data.Data]
#
# Colección GraphData con las variables predictoras escaladas.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿Las variables predictoras fueron transformadas utilizando exclusivamente
# el escalador ajustado con los datos de entrenamiento, garantizando la
# consistencia metodológica del pipeline?
#------------------------------------------------------------------------------

def scale_graph_collection(
    graphs: list,
    scaler: StandardScaler,
) -> list:
    """
    Aplica el escalado oficial a toda la colección GraphData.
    """

    for graph in graphs:

        graph.x = torch.tensor(
            scaler.transform(
                graph.x.cpu().numpy()
            ),
            dtype=torch.float32,
            device=graph.x.device,
        )

    return graphs

# BLOQUE 10. prepare_graph_collection
#------------------------------------------------------------------------------
# Objetivo
#------------------------------------------------------------------------------
# Preparar la colección oficial GraphData para su utilización en las etapas
# de Benchmark, Entrenamiento, Evaluación, Forecasting y Plataforma GeoAI,
# integrando las validaciones, la construcción de particiones, la generación
# de máscaras y el escalado de las variables predictoras.
#
#------------------------------------------------------------------------------
# Entradas
#------------------------------------------------------------------------------
# • graphs : list[torch_geometric.data.Data]
#       Colección oficial de grafos espacio-temporales.
#
# • expected_nodes : int
#       Número esperado de nodos.
#
# • expected_features : int
#       Número esperado de variables predictoras.
#
# • expected_years : int
#       Número esperado de grafos.
#
# • expected_edges : int | None
#       Número esperado de aristas.
#
# • train_size : float
#       Proporción del conjunto de entrenamiento.
#
# • validation_size : float
#       Proporción del conjunto de validación.
#
# • random_state : int
#       Semilla para garantizar la reproducibilidad.
#------------------------------------------------------------------------------
# Producto
#------------------------------------------------------------------------------
# preparation_results : dict
#
# Diccionario con la colección GraphData preparada, el escalador oficial,
# las particiones utilizadas y el reporte de validación.
#
#------------------------------------------------------------------------------
# Pregunta científica
#------------------------------------------------------------------------------
# ¿La colección GraphData se encuentra completamente preparada y validada
# para garantizar un proceso reproducible de entrenamiento, evaluación y
# predicción dentro del pipeline GeoAI?
#------------------------------------------------------------------------------

def prepare_graph_collection(
    graphs: list,
    expected_nodes: int,
    expected_features: int,
    expected_years: int,
    expected_edges: int | None = None,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    Prepara la colección oficial GraphData.
    """

    validation_report = validate_graph_collection(
        graphs=graphs,
        expected_nodes=expected_nodes,
        expected_features=expected_features,
        expected_years=expected_years,
        expected_edges=expected_edges,
    )

    partitions = build_graph_partitions(
        graphs=graphs,
        train_size=train_size,
        validation_size=validation_size,
        random_state=random_state,
    )

    graphs = build_graph_masks(
        graphs=graphs,
        partitions=partitions,
    )

    scaler = fit_graph_scaler(
        graphs=graphs,
    )

    graphs = scale_graph_collection(
        graphs=graphs,
        scaler=scaler,
    )

    preparation_results = {
        "graphs": graphs,
        "scaler": scaler,
        "partitions": partitions,
        "validation_report": validation_report,
    }

    return preparation_results