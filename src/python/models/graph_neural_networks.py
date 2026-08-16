# graph_neural_networks.py

# BLOQUE 1. Importaciones --------------------------------------------------
# Objetivo: Importar las librerías necesarias para construir, entrenar, evaluar y exportar las arquitecturas
# GNNs utilizadas durante el Benchmark Científico.
# Producto: - Librerías cargadas correctamente.
# Responde: ¿Las dependencias necesarias para implementar las arquitecturas GNN fueron importadas correctamente?

# Funciones del sistema
import time  # Medición del tiempo de entrenamiento
from typing import Any

# Librerías científicas
import numpy as np  # Operaciones numéricas

# PyTorch
import torch  # Tensor principal
import torch.nn as nn  # Capas neuronales
import torch.nn.functional as F  # Funciones de activación

# Optimizadores
from torch.optim import Adam  # Optimizador Adam

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    TAGConv
)  # Capas espaciales GNN

# Métricas oficiales
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)  # Métricas oficiales del Benchmark

# Utilidades del proyecto
from src.python.utils.results import (
    build_benchmark_result
)  # Construcción del resultado oficial del Benchmark

# Configuración oficial
from src.python.config.config_project import (
    BENCHMARK_MODEL_CODES,
    PROJECT_SEED
)  # Configuración oficial del Benchmark

# Reproducibilidad ---------------------------------------------------------
torch.manual_seed(
    PROJECT_SEED
)  # Semilla para PyTorch

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        PROJECT_SEED
    )  # Semilla para GPU

# BLOQUE 2. Configuración --------------------------------------------------
# Objetivo: Definir la configuración oficial de las arquitecturas GNNs utilizadas durante el Benchmark Científico.
# Producto: - GNN_CONFIG
# Responde: ¿Las arquitecturas GNNs disponen de una configuración oficial, reproducible y consistente con el Benchmark Científico?

# Configuración oficial de las arquitecturas -------------------------------
GNN_CONFIG = {
    "gcn": {
        # Identificación
        "model_code": "GNN01",
        "model_name": "gcn",
        "family": "graph_neural_networks",

        # Hiperparámetros
        "hidden_channels": 64,
        "dropout": 0.30,
        "learning_rate": 0.001,
        "weight_decay": 5e-4,
        "epochs": 300
    },

    "graphsage": {
        # Identificación
        "model_code": "GNN02", # Código oficial del modelo
        "model_name": "graphsage",
        "family": "graph_neural_networks",

        # Hiperparámetros
        "hidden_channels": 64,
        "dropout": 0.30,
        "learning_rate": 0.001,
        "weight_decay": 5e-4,
        "epochs": 300
    },

    "gat": {
        # Identificación
        "model_code": "GNN03", # Código oficial del modelo
        "model_name": "gat",
        "family": "graph_neural_networks",

        # Hiperparámetros
        "hidden_channels": 64,
        "heads": 4,
        "dropout": 0.30,
        "learning_rate": 0.001,
        "weight_decay": 5e-4,
        "epochs": 300
    },

    "gin": {
        # Identificación
        "model_code": "GNN04", # Código oficial del modelo
        "model_name": "gin",
        "family": "graph_neural_networks",

        # Hiperparámetros
        "hidden_channels": 64,
        "dropout": 0.30,
        "learning_rate": 0.001,
        "weight_decay": 5e-4,
        "epochs": 300
    },

    "tagcn": {
        # Identificación
        "model_code": "GNN05", # Código oficial del modelo
        "model_name": "tagcn",
        "family": "graph_neural_networks",

        # Hiperparámetros
        "hidden_channels": 64,
        "K": 3,
        "dropout": 0.30,
        "learning_rate": 0.001,
        "weight_decay": 5e-4,
        "epochs": 300
    }
}  # Configuración oficial de las arquitecturas GNN

# BLOQUE 3. Construcción de Arquitecturas GNN -------------------------------
# Objetivo: Definir las arquitecturas oficiales GNNs (GNN) utilizadas durante el Benchmark Científico.
# Producto: - GCNModel - GraphSAGEModel - GATModel - GINModel - TAGCNModel
# Responde: ¿Las arquitecturas oficiales GNNs fueron construidas de forma modular, reproducible y consistente con el Benchmark Científico?

# BLOQUE 3.1. Clase GCNModel -----------------------------------------------
class GCNModel(
    nn.Module
):
    """
    Arquitectura oficial Graph Convolutional Network (GCN).

    Parameters
    ----------
    input_channels : int
        Número de variables de entrada.

    hidden_channels : int
        Número de neuronas de la capa oculta.

    output_channels : int
        Número de variables de salida.

    dropout : float
        Probabilidad de desactivación de neuronas.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        dropout: float
    ) -> None:

        super().__init__()
        self.conv1 = GCNConv(
            input_channels,
            hidden_channels
        )  # Primera capa convolucional

        self.conv2 = GCNConv(
            hidden_channels,
            output_channels
        )  # Segunda capa convolucional
        self.dropout = dropout  # Probabilidad de Dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
        # edge_weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Propagación hacia adelante del modelo.

        Parameters
        ----------
        x : torch.Tensor
            Matriz de características de los nodos.

        edge_index : torch.Tensor
            Índices de las aristas del grafo.

        Returns
        -------
        torch.Tensor
            Predicciones del modelo.
        """

        x = self.conv1(
            x,
            edge_index
        )  # Primera convolución

        x = F.relu(
            x
        )  # Función de activación

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )  # Regularización

        x = self.conv2(
            x,
            edge_index
        )  # Segunda convolución
        return x  # Predicciones del modelo
    
# BLOQUE 3.2. Clase GraphSAGEModel -----------------------------------------
class GraphSAGEModel(nn.Module):
    """
    Arquitectura oficial GraphSAGE.

    Parameters
    ----------
    input_channels : int
        Número de variables de entrada.

    hidden_channels : int
        Número de neuronas de la capa oculta.

    output_channels : int
        Número de variables de salida.

    dropout : float
        Probabilidad de desactivación de neuronas.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        dropout: float
    ) -> None:

        super().__init__()

        self.conv1 = SAGEConv(
            input_channels,
            hidden_channels
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            output_channels
        )
        self.dropout = dropout

        # Auditoría temporal
        self.debug_forward = True

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Propagación hacia adelante del modelo.
        """

        # Auditoría temporal (solo la primera ejecución)
        if self.debug_forward:

            print("\n" + "-" * 80)
            print("FORWARD GraphSAGE")
            print("-" * 80)

            print(f"x.shape        : {tuple(x.shape)}")
            print(f"edge_index     : {tuple(edge_index.shape)}")

            print(
                f"Entrada x      : "
                f"min={x.min().item():.6f} "
                f"max={x.max().item():.6f}"
            )

        # Primera capa GraphSAGE
        x = self.conv1(
            x,
            edge_index
        )

        if self.debug_forward:
            print(f"conv1.shape    : {tuple(x.shape)}")
            print(
                f"Después conv1  : "
                f"min={x.min().item():.6f} "
                f"max={x.max().item():.6f}"
            )

        # Función de activación
        x = F.relu(x)
        if self.debug_forward:
            print(
                f"Después ReLU   : "
                f"min={x.min().item():.6f} "
                f"max={x.max().item():.6f}"
            )

        # Dropout
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        # Segunda capa GraphSAGE
        x = self.conv2(
            x,
            edge_index
        )

        if self.debug_forward:
            print(f"conv2.shape    : {tuple(x.shape)}")
            print(
                f"Después conv2  : "
                f"min={x.min().item():.6f} "
                f"max={x.max().item():.6f}"
                f"mean={x.mean().item():.6f}"
            )
            print("-" * 80)
            # Solo imprimir la primera pasada
            self.debug_forward = False

        return x

# BLOQUE 3.3. Clase GATModel -----------------------------------------------
class GATModel(
    nn.Module
):
    """
    Arquitectura oficial Graph Attention Network (GAT).

    Parameters
    ----------
    input_channels : int
        Número de variables de entrada.

    hidden_channels : int
        Número de neuronas de la capa oculta.

    output_channels : int
        Número de variables de salida.

    heads : int
        Número de mecanismos de atención.

    dropout : float
        Probabilidad de desactivación de neuronas.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        heads: int,
        dropout: float
    ) -> None:

        super().__init__()

        self.conv1 = GATConv(
            input_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout
        )  # Primera capa GAT

        self.conv2 = GATConv(
            hidden_channels * heads,
            output_channels,
            heads=1,
            concat=False,
            dropout=dropout
        )  # Segunda capa GAT
        self.dropout = dropout  # Probabilidad de Dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
        # edge_weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Propagación hacia adelante del modelo.

        Parameters
        ----------
        x : torch.Tensor
            Matriz de características de los nodos.

        edge_index : torch.Tensor
            Índices de las aristas del grafo.

        Returns
        -------
        torch.Tensor
            Predicciones del modelo.
        """

        x = self.conv1(
            x,
            edge_index
        )  # Primera capa GAT

        x = F.elu(
            x
        )  # Función de activación

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )  # Regularización

        x = self.conv2(
            x,
            edge_index
        )  # Segunda capa GAT

        return x  # Predicciones del modelo

# BLOQUE 3.4. Clase GINModel -----------------------------------------------
class GINModel(
    nn.Module
):
    """
    Arquitectura oficial Graph Isomorphism Network (GIN).
    Parameters
    ----------
    input_channels : int
        Número de variables de entrada.

    hidden_channels : int
        Número de neuronas de la capa oculta.

    output_channels : int
        Número de variables de salida.

    dropout : float
        Probabilidad de desactivación de neuronas.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        dropout: float
    ) -> None:

        super().__init__()
        mlp1 = nn.Sequential(
            nn.Linear(
                input_channels,
                hidden_channels
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_channels,
                hidden_channels
            )
        )  # MLP de la primera capa

        mlp2 = nn.Sequential(
            nn.Linear(
                hidden_channels,
                hidden_channels
            ),

            nn.ReLU(),
            nn.Linear(
                hidden_channels,
                output_channels
            )
        )  # MLP de la segunda capa

        self.conv1 = GINConv(
            mlp1
        )  # Primera capa GIN

        self.conv2 = GINConv(
            mlp2
        )  # Segunda capa GIN
        self.dropout = dropout  # Probabilidad de Dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
        # edge_weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Propagación hacia adelante del modelo.
        Parameters
        ----------
        x : torch.Tensor
            Matriz de características de los nodos.

        edge_index : torch.Tensor
            Índices de las aristas del grafo.

        Returns
        -------
        torch.Tensor
            Predicciones del modelo.
        """
        x = self.conv1(
            x,
            edge_index
        )  # Primera capa GIN

        x = F.relu(
            x
        )  # Función de activación

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )  # Regularización

        x = self.conv2(
            x,
            edge_index
        )  # Segunda capa GIN

        return x  # Predicciones del modelo
    
# BLOQUE 3.5. Clase TAGCNModel ---------------------------------------------
class TAGCNModel(
    nn.Module
):
    """
    Arquitectura oficial Topology Adaptive Graph Convolutional Network
    (TAGCN).

    Parameters
    ----------
    input_channels : int
        Número de variables de entrada.

    hidden_channels : int
        Número de neuronas de la capa oculta.

    output_channels : int
        Número de variables de salida.

    K : int
        Número de saltos considerados durante la convolución.

    dropout : float
        Probabilidad de desactivación de neuronas.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        K: int,
        dropout: float
    ) -> None:

        super().__init__()

        self.conv1 = TAGConv(
            input_channels,
            hidden_channels,
            K=K
        )  # Primera capa TAGCN

        self.conv2 = TAGConv(
            hidden_channels,
            output_channels,
            K=K
        )  # Segunda capa TAGCN

        self.dropout = dropout  # Probabilidad de Dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
        # edge_weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Propagación hacia adelante del modelo.

        Parameters
        ----------
        x : torch.Tensor
            Matriz de características de los nodos.

        edge_index : torch.Tensor
            Índices de las aristas del grafo.

        Returns
        -------
        torch.Tensor
            Predicciones del modelo.
        """

        x = self.conv1(
            x,
            edge_index
        )  # Primera convolución

        x = F.relu(
            x
        )  # Función de activación

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )  # Regularización

        x = self.conv2(
            x,
            edge_index
        )  # Segunda convolución

        return x  # Predicciones del modelo

# BLOQUE 4. Construcción del Modelo ----------------------------------------
# Objetivo: Construir la arquitectura GNN seleccionada utilizando la configuración oficial del Benchmark Científico.
# Entradas: - model_config - input_channels - output_channels
# Producto: - model
# Responde: ¿La arquitectura GNN fue construida correctamente?

def build_gnn_model(
    model_config: dict,
    input_channels: int,
    output_channels: int
) -> nn.Module:
    """
    Construye la arquitectura GNN especificada en la
    configuración oficial del Benchmark Científico.

    Parameters
    ----------
    model_config : dict
        Configuración oficial del modelo.

    input_channels : int
        Número de variables de entrada.

    output_channels : int
        Número de variables de salida.

    Returns
    -------
    nn.Module
        Arquitectura GNN construida.
    """

    if model_config is None:
        raise ValueError(
            "La configuración del modelo no puede ser nula."
        )

    if not isinstance(model_config, dict):
        raise TypeError(
            "model_config debe ser un diccionario."
        )

    if input_channels <= 0:
        raise ValueError(
            "input_channels debe ser mayor que cero."
        )

    if output_channels <= 0:
        raise ValueError(
            "output_channels debe ser mayor que cero."
        )

    required_products = [
        "model_name",
        "hidden_channels",
        "dropout",
    ]

    missing_products = [
        product
        for product in required_products
        if product not in model_config
    ]

    if missing_products:
        raise ValueError(
            "model_config está incompleto: "
            f"{missing_products}"
        )

    for product in required_products:

        if model_config[product] is None:
            raise ValueError(
                f"'{product}' es inválido."
            )

    try:
        model_name = str(
            model_config["model_name"]
        ).lower()
        hidden_channels = model_config["hidden_channels"]
        dropout = model_config["dropout"]

        print("\n" + "=" * 80)
        print("BUILD GNN MODEL")
        print("=" * 80)
        print(f"model_name      : {model_name}")
        print(f"input_channels  : {input_channels}")
        print(f"hidden_channels : {hidden_channels}")
        print(f"output_channels : {output_channels}")

        if model_name == "gcn":
            model = GCNModel(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                dropout=dropout
            )

        elif model_name == "graphsage":
            model = GraphSAGEModel(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                dropout=dropout
            )

        elif model_name == "gat":
            if "heads" not in model_config:
                raise ValueError(
                    "La configuración GAT debe contener 'heads'."
                )

            if model_config["heads"] is None:
                raise ValueError(
                    "'heads' es inválido."
                )

            model = GATModel(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                heads=model_config["heads"],
                dropout=dropout
            )

        elif model_name == "gin":
            model = GINModel(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                dropout=dropout
            )

        elif model_name == "tagcn":
            if "K" not in model_config:
                raise ValueError(
                    "La configuración TAGCN debe contener 'K'."
                )

            if model_config["K"] is None:
                raise ValueError(
                    "'K' es inválido."
                )

            model = TAGCNModel(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                K=model_config["K"],
                dropout=dropout
            )

        else:
            raise ValueError(
                f"Modelo GNN no soportado: {model_name}"
            )

        if not isinstance(model, nn.Module):
            raise TypeError(
                "El modelo construido no corresponde a nn.Module."
            )
        return model

    except Exception as error:
        raise RuntimeError(
            f"No fue posible construir la arquitectura GNN "
            f"'{model_config.get('model_name', 'desconocido')}'."
        ) from error

# BLOQUE 5. Función de Pérdida y Optimizador -------------------------------
# Objetivo: Construir la función de pérdida y el optimizador oficial utilizados durante el entrenamiento de 
# las arquitecturas GNNs.
# Entradas: - model - model_config
# Producto: - criterion - optimizer
# Responde: ¿La función de pérdida y el optimizador fueron configurados correctamente para el entrenamiento del modelo GNN?

def build_training_components(
    model: nn.Module,
    model_config: dict
) -> dict:
    """
    Construye la función de pérdida y el optimizador oficial para una
    arquitectura GNN.

    Parameters
    ----------
    model : nn.Module
        Modelo GNN.

    model_config : dict
        Configuración oficial del modelo.

    Returns
    -------
    dict
        Componentes oficiales del entrenamiento.
    """

    if model is None:
        raise ValueError(
            "El modelo no puede ser nulo."
        )

    if not isinstance(model, nn.Module):
        raise TypeError(
            "El modelo debe ser una instancia de nn.Module."
        )

    if model_config is None:
        raise ValueError(
            "La configuración del modelo no puede ser nula."
        )

    if not isinstance(model_config, dict):
        raise TypeError(
            "model_config debe ser un diccionario."
        )

    required_products = [
        "learning_rate",
        "weight_decay",
    ]

    missing_products = [
        product
        for product in required_products
        if product not in model_config
    ]

    if missing_products:
        raise ValueError(
            "model_config está incompleto: "
            f"{missing_products}"
        )

    for product in required_products:

        if model_config[product] is None:
            raise ValueError(
                f"'{product}' es inválido."
            )

    try:
        criterion = nn.MSELoss()  # Función de pérdida oficial
        optimizer = Adam(
            model.parameters(),
            lr=model_config["learning_rate"],
            weight_decay=model_config["weight_decay"]
        )  # Optimizador oficial

        training_components = {
            "criterion": criterion,
            "optimizer": optimizer
        }

        required_products = [
            "criterion",
            "optimizer"
        ]

        missing_products = [
            product
            for product in required_products
            if product not in training_components
        ]

        if missing_products:
            raise RuntimeError(
                "TrainingComponents está incompleto: "
                f"{missing_products}"
            )

        if not isinstance(
            training_components["criterion"],
            nn.Module
        ):
            raise TypeError(
                "criterion debe ser una instancia de nn.Module."
            )

        if not isinstance(
            training_components["optimizer"],
            torch.optim.Optimizer
        ):
            raise TypeError(
                "optimizer debe ser una instancia de Optimizer."
            )
        return training_components

    except Exception as error:
        raise RuntimeError(
            "No fue posible construir los componentes oficiales "
            "del entrenamiento."
        ) from error
        
# BLOQUE 6. Entrenamiento --------------------------------------------------
# Objetivo: Entrenar la arquitectura GNN utilizando la colección oficial de grafos 
# espacio-temporales del Benchmark Científico.
# Entradas: - model - graphs - criterion - optimizer - model_config
# Producto: - trained_model - training_time - loss
# Responde: ¿La arquitectura GNN fue entrenada correctamente sobre la colección oficial de grafos?
def train_gnn(
    model: nn.Module,
    graphs: list,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: dict,
    train_index: np.ndarray | None = None,
    validation_index: np.ndarray | None = None
) -> dict:
    """
    Entrena una arquitectura GNN utilizando la colección de GraphData
    proporcionada por el protocolo oficial del Benchmark.

    Si train_index es None, se asume que graphs contiene exclusivamente
    los GraphData temporales destinados al entrenamiento y se utilizan
    todos los nodos de cada GraphData.
    """

    if model is None:
        raise ValueError(
            "El modelo no puede ser nulo."
        )

    if not isinstance(
        model,
        nn.Module
    ):
        raise TypeError(
            "El modelo debe ser una instancia de nn.Module."
        )

    if graphs is None or len(graphs) == 0:
        raise ValueError(
            "La colección de GraphData está vacía."
        )

    if criterion is None:
        raise ValueError(
            "La función de pérdida no puede ser nula."
        )

    if optimizer is None:
        raise ValueError(
            "El optimizador no puede ser nulo."
        )

    if not isinstance(
        optimizer,
        torch.optim.Optimizer
    ):
        raise TypeError(
            "optimizer debe ser una instancia de Optimizer."
        )

    if model_config is None:
        raise ValueError(
            "La configuración del modelo no puede ser nula."
        )

    if train_index is not None:
        if not isinstance(
            train_index,
            np.ndarray
        ):
            raise TypeError(
                "train_index debe ser un arreglo NumPy."
            )

        if train_index.ndim != 1:
            raise ValueError(
                "train_index debe ser un vector unidimensional."
            )

        if len(train_index) == 0:
            raise ValueError(
                "train_index está vacío."
            )

    if validation_index is not None:
        if not isinstance(
            validation_index,
            np.ndarray
        ):
            raise TypeError(
                "validation_index debe ser un arreglo NumPy."
            )

        if validation_index.ndim != 1:
            raise ValueError(
                "validation_index debe ser un vector unidimensional."
            )

        if len(validation_index) == 0:
            raise ValueError(
                "validation_index está vacío."
            )

    model.train()

    training_start = time.time()
    epoch_loss = None
    loss_history = []

    try:

        for epoch in range(
            model_config["epochs"]
        ):

            accumulated_loss = 0.0

            for graph_index, graph in enumerate(graphs):

                optimizer.zero_grad()

                predictions = model(
                    graph.x,
                    graph.edge_index
                ) # Generar predicciones para todos los nodos

                predictions = predictions.squeeze(-1)

                if graph.y is None:
                    raise ValueError(
                        "El GraphData no contiene la variable objetivo."
                    )

                if not isinstance(
                    graph.y,
                    torch.Tensor
                ):
                    raise TypeError(
                        "graph.y debe ser un tensor de PyTorch."
                    )

                target = graph.y.squeeze(-1) # Normalizar dimensión del objetivo

                if train_index is None:

                    prediction_target = predictions
                    target_train = target # Utilizar todos los nodos del GraphData de entrenamiento

                else:

                    if np.any(train_index < 0):
                        raise ValueError(
                            "train_index contiene índices negativos."
                        )

                    if train_index.max() >= graph.num_nodes:
                        raise ValueError(
                            "train_index contiene índices fuera del rango del grafo."
                        )

                    prediction_target = predictions[
                        train_index
                    ]

                    target_train = target[
                        train_index
                    ] # Utilizar únicamente los nodos indicados

                prediction_target = prediction_target.reshape(-1)
                target_train = target_train.reshape(-1)

                if prediction_target.shape != target_train.shape:
                    raise ValueError(
                        "prediction_target y target poseen dimensiones incompatibles: "
                        f"{prediction_target.shape} != "
                        f"{target_train.shape}"
                    )

                loss = criterion(
                    prediction_target,
                    target_train
                ) # Calcular pérdida

                if epoch == 0 and graph_index == 0:

                    print("\n" + "=" * 80)
                    print("AUDITORÍA DEL ENTRENAMIENTO GNN")
                    print("=" * 80)

                    print(
                        f"Cantidad de grafos : "
                        f"{len(graphs)}"
                    )

                    print(
                        f"Nodos              : "
                        f"{graph.num_nodes}"
                    )

                    print(
                        f"Variables          : "
                        f"{graph.num_node_features}"
                    )

                    if train_index is None:
                        print(
                            "Protocolo          : "
                            "GraphData temporales de entrenamiento"
                        )
                    else:
                        print(
                            "Protocolo          : "
                            "Índice de nodos"
                        )

                    print(
                        f"Feature mínimo     : "
                        f"{graph.x.min().item()}"
                    )

                    print(
                        f"Feature máximo     : "
                        f"{graph.x.max().item()}"
                    )

                    print(
                        f"Predictions        : "
                        f"{tuple(prediction_target.shape)}"
                    )

                    print(
                        f"Target             : "
                        f"{tuple(target_train.shape)}"
                    )

                    print(
                        f"Target mínimo      : "
                        f"{target_train.min().item()}"
                    )

                    print(
                        f"Target máximo      : "
                        f"{target_train.max().item()}"
                    )

                    print(
                        f"Target promedio    : "
                        f"{target_train.mean().item()}"
                    )

                    print(
                        f"Pred mínimo        : "
                        f"{prediction_target.min().item()}"
                    )

                    print(
                        f"Pred máximo        : "
                        f"{prediction_target.max().item()}"
                    )

                    print(
                        f"Pred promedio      : "
                        f"{prediction_target.mean().item()}"
                    )

                    print(
                        f"Loss inicial       : "
                        f"{loss.item()}"
                    )

                loss.backward()

                optimizer.step()

                accumulated_loss += loss.item()

            epoch_loss = (
                accumulated_loss
                / len(graphs)
            )

            loss_history.append(
                epoch_loss
            )

    except Exception as error:

        raise RuntimeError(
            "Error durante el entrenamiento del modelo GNN."
        ) from error

    training_result = {
        "model": model,
        "training_time": time.time() - training_start,
        "loss": epoch_loss,
        "loss_history": loss_history
    }

    required_products = [
        "model",
        "training_time",
        "loss",
        "loss_history"
    ]

    missing_products = [
        product
        for product in required_products
        if product not in training_result
    ]

    if missing_products:
        raise RuntimeError(
            "TrainingResult está incompleto: "
            f"{missing_products}"
        )

    return training_result

# BLOQUE 7. Predicción -----------------------------------------------------
# Objetivo: Generar las predicciones utilizando la colección oficial de grafos espacio-temporales del Benchmark Científico.
# Entradas: - model - graphs
# Producto: - y_pred - y_true - inference_time
# Responde: ¿La arquitectura GNN genera correctamente las predicciones sobre la colección oficial de grafos?

def predict_gnn(
    model: nn.Module,
    graphs: list,
    y_true: torch.Tensor | None = None
) -> dict:
    """
    Genera las predicciones utilizando la colección oficial de GraphData.

    Parameters
    ----------
    model : nn.Module
        Modelo GNN entrenado.

    graphs : list
        Colección oficial de GraphData.

    y_true : torch.Tensor | None, optional
        Valores observados utilizados durante la evaluación.
        Para inferencia o forecasting puede ser None.

    Returns
    -------
    dict
        Predicciones oficiales, valores observados (si existen)
        y tiempo oficial de inferencia.
    """

    if model is None:
        raise ValueError(
            "El modelo no puede ser nulo."
        )

    if not isinstance(model, nn.Module):
        raise TypeError(
            "El modelo debe ser una instancia de nn.Module."
        )

    if graphs is None:
        raise ValueError(
            "La colección de GraphData no puede ser nula."
        )

    if not isinstance(graphs, list):
        raise TypeError(
            "graphs debe ser una lista."
        )

    if len(graphs) == 0:
        raise ValueError(
            "La colección de GraphData está vacía."
        )

    for graph in graphs:

        if not isinstance(graph, Data):
            raise TypeError(
                "Todos los elementos de graphs deben ser GraphData."
            )

        if graph.x is None:
            raise ValueError(
                "El GraphData no contiene variables de entrada."
            )

        if graph.edge_index is None:
            raise ValueError(
                "El GraphData no contiene edge_index."
            )

    if (
        y_true is not None
        and not isinstance(y_true, torch.Tensor)
    ):
        y_true = torch.as_tensor(
            y_true,
            dtype=torch.float32
        )

    model.eval()

    inference_start = time.time()

    try:
        prediction_list = []
        with torch.no_grad():
            for graph in graphs:

                outputs = model(
                    graph.x,
                    graph.edge_index
                )

                prediction_list.append(
                    outputs.squeeze(-1).cpu()
                )

        y_pred = torch.cat(
            prediction_list,
            dim=0
        )

        prediction_result = {
            "y_pred": y_pred,
            "inference_time": (
                time.time() - inference_start
            )
        }

        if y_true is not None:
            prediction_result["y_true"] = y_true.cpu()

        required_products = [
            "y_pred",
            "inference_time"
        ]

        missing_products = [
            product
            for product in required_products
            if product not in prediction_result
        ]

        if missing_products:
            raise RuntimeError(
                "PredictionResult está incompleto: "
                f"{missing_products}"
            )

        if not isinstance(
            prediction_result["y_pred"],
            torch.Tensor
        ):
            raise TypeError(
                "y_pred debe ser un tensor de PyTorch."
            )

        if not isinstance(
            prediction_result["inference_time"],
            (int, float)
        ):
            raise TypeError(
                "inference_time debe ser numérico."
            )

        if (
            "y_true" in prediction_result
            and not isinstance(
                prediction_result["y_true"],
                torch.Tensor
            )
        ):
            raise TypeError(
                "y_true debe ser un tensor de PyTorch."
            )
        return prediction_result

    except Exception as error:
        raise RuntimeError(
            "Error durante la inferencia del modelo GNN."
        ) from error
    
# BLOQUE 8. Evaluación -----------------------------------------------------
# Objetivo: Calcular las métricas oficiales de desempeño predictivo para la arquitectura GNN
# utilizando el conjunto de prueba.
# Entradas: - y_true - y_pred
# Producto: - evaluation_result
# Responde: ¿Cuál es el desempeño predictivo de la arquitectura GNN sobre el conjunto de prueba?

def evaluate_gnn(
    y_true: Any,
    y_pred: Any
) -> dict:
    """
    Calcula las métricas oficiales de evaluación para una arquitectura
    GNN utilizando el pipeline oficial del proyecto.

    Parameters
    ----------
    y_true : Any
        Valores observados.

    y_pred : Any
        Valores predichos.

    Returns
    -------
    dict
        Métricas oficiales de evaluación.
    """

    if y_true is None:
        raise ValueError(
            "y_true no puede ser nulo."
        )

    if y_pred is None:
        raise ValueError(
            "y_pred no puede ser nulo."
        )

    try:
        if isinstance(
            y_true,
            torch.Tensor
        ):
            y_true = (
                y_true.detach()
                .cpu()
                .numpy()
            )

        if isinstance(
            y_pred,
            torch.Tensor
        ):
            y_pred = (
                y_pred.detach()
                .cpu()
                .numpy()
            )

        if not isinstance(y_true, np.ndarray):
            y_true = np.asarray(y_true)

        if not isinstance(y_pred, np.ndarray):
            y_pred = np.asarray(y_pred)

        if len(y_true) == 0:
            raise ValueError(
                "y_true está vacío."
            )

        if len(y_pred) == 0:
            raise ValueError(
                "y_pred está vacío."
            )

        if len(y_true) != len(y_pred):
            raise ValueError(
                "y_true y y_pred deben tener la misma longitud."
            )

        if np.isnan(y_true).any():
            raise ValueError(
                "y_true contiene valores NaN."
            )

        if np.isnan(y_pred).any():
            raise ValueError(
                "y_pred contiene valores NaN."
            )

        if np.isinf(y_true).any():
            raise ValueError(
                "y_true contiene valores infinitos."
            )

        if np.isinf(y_pred).any():
            raise ValueError(
                "y_pred contiene valores infinitos."
            )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        mape = mean_absolute_percentage_error(
            y_true,
            y_pred
        )

        r2 = r2_score(
            y_true,
            y_pred
        )

        evaluation_result = {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2
        }

        required_products = [
            "rmse",
            "mae",
            "mape",
            "r2"
        ]

        missing_products = [
            product
            for product in required_products
            if product not in evaluation_result
        ]

        if missing_products:
            raise RuntimeError(
                "EvaluationResult está incompleto: "
                f"{missing_products}"
            )

        for product in required_products:
            if evaluation_result[product] is None:
                raise ValueError(
                    f"'{product}' es inválido."
                )

            if not isinstance(
                evaluation_result[product],
                (float, np.floating)
            ):
                raise TypeError(
                    f"'{product}' debe ser un valor numérico."
                )
        return evaluation_result

    except Exception as error:
        raise RuntimeError(
            "Error durante la evaluación del modelo GNN."
        ) from error

# BLOQUE 9. Construcción del Resultado Oficial -----------------------------
# Objetivo: Construir la estructura oficial de resultados de la arquitectura GNN compatible con el Benchmark Científico.
# Entradas: - model_config - prediction_result - evaluation_result - training_result
# Producto: - benchmark_result
# Responde: ¿Los resultados de la arquitectura GNN fueron consolidados correctamente para el Benchmark Científico?
def build_gnn_results(
    model_config: dict,
    prediction_result: dict,
    evaluation_result: dict,
    training_result: dict
) -> dict:
    """
    Construye el resultado oficial de una arquitectura GNN
    dentro del Benchmark Científico.

    Integra los productos de entrenamiento, predicción y evaluación
    bajo una única estructura científica estandarizada.
    """

    if model_config is None:
        raise ValueError(
            "model_config no puede ser nulo."
        )

    if not isinstance(
        model_config,
        dict
    ):
        raise TypeError(
            "model_config debe ser un diccionario."
        )

    if prediction_result is None:
        raise ValueError(
            "prediction_result no puede ser nulo."
        )

    if not isinstance(
        prediction_result,
        dict
    ):
        raise TypeError(
            "prediction_result debe ser un diccionario."
        )

    if evaluation_result is None:
        raise ValueError(
            "evaluation_result no puede ser nulo."
        )

    if not isinstance(
        evaluation_result,
        dict
    ):
        raise TypeError(
            "evaluation_result debe ser un diccionario."
        )

    if training_result is None:
        raise ValueError(
            "training_result no puede ser nulo."
        )

    if not isinstance(
        training_result,
        dict
    ):
        raise TypeError(
            "training_result debe ser un diccionario."
        )

    required_model_config_keys = [
        "model_code",
        "model_name",
        "family",
    ]

    missing_model_config_keys = [
        key
        for key in required_model_config_keys
        if key not in model_config
    ]

    if missing_model_config_keys:
        raise ValueError(
            "model_config está incompleto: "
            f"{missing_model_config_keys}"
        )

    required_prediction_keys = [
        "y_pred",
        "inference_time",
    ]

    missing_prediction_keys = [
        key
        for key in required_prediction_keys
        if key not in prediction_result
    ]

    if missing_prediction_keys:
        raise ValueError(
            "prediction_result está incompleto: "
            f"{missing_prediction_keys}"
        )

    required_evaluation_keys = [
        "rmse",
        "mae",
        "mape",
        "r2",
    ]

    missing_evaluation_keys = [
        key
        for key in required_evaluation_keys
        if key not in evaluation_result
    ]

    if missing_evaluation_keys:
        raise ValueError(
            "evaluation_result está incompleto: "
            f"{missing_evaluation_keys}"
        )

    required_training_keys = [
        "model",
        "loss",
        "loss_history",
        "training_time",
    ]

    missing_training_keys = [
        key
        for key in required_training_keys
        if key not in training_result
    ]

    if missing_training_keys:
        raise ValueError(
            "training_result está incompleto: "
            f"{missing_training_keys}"
        )

    y_pred = prediction_result["y_pred"]

    if y_pred is None:
        raise ValueError(
            "prediction_result['y_pred'] no puede ser nulo."
        )

    if not isinstance(
        y_pred,
        np.ndarray
    ):
        y_pred = np.asarray(
            y_pred
        )

    if y_pred.ndim != 1:
        y_pred = y_pred.reshape(-1)

    if len(y_pred) == 0:
        raise ValueError(
            "prediction_result['y_pred'] está vacío."
        )

    if not np.all(
        np.isfinite(y_pred)
    ):
        raise ValueError(
            "prediction_result['y_pred'] contiene valores no finitos."
        )

    if not np.isfinite(
        float(prediction_result["inference_time"])
    ):
        raise ValueError(
            "inference_time contiene un valor inválido."
        )

    for metric in required_evaluation_keys:

        metric_value = evaluation_result[metric]

        if metric_value is None:
            raise ValueError(
                f"La métrica '{metric}' no puede ser None."
            )

        if not np.isfinite(
            float(metric_value)
        ):
            raise ValueError(
                f"La métrica '{metric}' contiene un valor no finito."
            )

    benchmark_result = {
        "model": training_result["model"],
        "model_code": model_config["model_code"],
        "model_name": model_config["model_name"],
        "family": model_config["family"],
        "rmse": float(evaluation_result["rmse"]),
        "mae": float(evaluation_result["mae"]),
        "mape": float(evaluation_result["mape"]),
        "r2": float(evaluation_result["r2"]),
        "training_time": float(
            training_result["training_time"]
        ),
        "inference_time": float(
            prediction_result["inference_time"]
        ),
        "loss": float(
            training_result["loss"]
        ),
        "loss_history": training_result["loss_history"],
        "prediction_result": prediction_result,
        "evaluation_result": evaluation_result,
        "model_config": model_config,
    } # Construir producto científico oficial del Benchmark

    required_result_keys = [
        "model",
        "model_code",
        "model_name",
        "family",
        "rmse",
        "mae",
        "mape",
        "r2",
        "training_time",
        "inference_time",
        "loss",
        "loss_history",
        "prediction_result",
        "evaluation_result",
        "model_config",
    ]

    missing_result_keys = [
        key
        for key in required_result_keys
        if key not in benchmark_result
    ]

    if missing_result_keys:
        raise RuntimeError(
            "BenchmarkResult está incompleto: "
            f"{missing_result_keys}"
        )

    return benchmark_result
    
# BLOQUE 10. Ejecución del Modelo ------------------------------------------
# Objetivo: Ejecutar el flujo completo de una arquitectura GNN utilizando la colección oficial de GraphData del proyecto.
# Entradas: - model_config - graphs 
# Producto: - gnn_result
# Responde: ¿La arquitectura GNN fue ejecutada correctamente sobre la colección oficial de GraphData?

def run_gnn(
    model_config: dict,
    graphs: list
) -> dict:
    """
    Ejecuta el flujo oficial de entrenamiento de una arquitectura
    GNN.
    """

    if model_config is None:
        raise ValueError(
            "La configuración del modelo no puede ser nula."
        )

    if graphs is None or len(graphs) == 0:
        raise ValueError(
            "La colección de GraphData está vacía."
        )

    try:
        input_channels = graphs[0].num_node_features

        model = build_gnn_model(
            model_config=model_config,
            input_channels=input_channels,
            output_channels=1
        )

        training_components = build_training_components(
            model=model,
            model_config=model_config
        )

        training_result = train_gnn(
            model=model,
            graphs=graphs,
            criterion=training_components["criterion"],
            optimizer=training_components["optimizer"],
            model_config=model_config
        )

        run_result = {
            "model": training_result["model"],
            "loss": training_result["loss"],
            "loss_history": training_result["loss_history"],
            "training_time": training_result["training_time"],
            "model_config": model_config

        }

        required_products = [
            "model",
            "loss",
            "loss_history",
            "training_time",
            "model_config"
        ]

        missing_products = [
            product
            for product in required_products
            if product not in run_result
        ]

        if missing_products:
            raise RuntimeError(
                "RunResult está incompleto: "
                f"{missing_products}"
            )
        return run_result

    except Exception as error:
        raise RuntimeError(
            "Error durante la ejecución del flujo oficial del modelo GNN."
        ) from error

# BLOQUE 11. Entrenamiento del Modelo Oficial -------------------------------
# Objetivo: Ejecutar el pipeline oficial de entrenamiento de una arquitectura GNN preservando todos los productos
# científicos generados durante el entrenamiento, la predicción y la evaluación del Modelo Oficial.
# Entradas: - model_config - graphs
# Producto: - model_config - training_result - prediction_result - evaluation_result
# Responde: ¿El pipeline oficial del Modelo Oficial genera correctamente todos los productos científicos requeridos
# para la exportación, validación e inferencia del modelo?
def run_gnn_training(
    model_config: dict,
    graphs: list,
    train_index: np.ndarray,
    validation_index: np.ndarray
) -> dict:
    """
    Ejecuta el pipeline oficial de entrenamiento y validación de una GNN.

    Parameters
    ----------
    model_config : dict
        Configuración oficial del modelo.

    graphs : list
        Colección oficial de GraphData.

    train_index : np.ndarray
        Índices oficiales utilizados para el entrenamiento.

    validation_index : np.ndarray
        Índices oficiales utilizados para la validación.

    Returns
    -------
    dict
        Resultado oficial del entrenamiento y validación del modelo GNN.
    """

    # Validación de entradas
    if model_config is None:
        raise ValueError(
            "La configuración del modelo no puede ser nula."
        )

    if not isinstance(model_config, dict):
        raise TypeError(
            "model_config debe ser un diccionario."
        )

    if graphs is None or len(graphs) == 0:
        raise ValueError(
            "La colección de GraphData está vacía."
        )

    if not isinstance(graphs, (list, tuple)):
        raise TypeError(
            "graphs debe ser una lista o tupla de GraphData."
        )

    if train_index is None:
        raise ValueError(
            "train_index no puede ser nulo."
        )

    if validation_index is None:
        raise ValueError(
            "validation_index no puede ser nulo."
        )

    if not isinstance(train_index, np.ndarray):
        raise TypeError(
            "train_index debe ser un arreglo NumPy."
        )

    if not isinstance(validation_index, np.ndarray):
        raise TypeError(
            "validation_index debe ser un arreglo NumPy."
        )

    if train_index.ndim != 1:
        raise ValueError(
            "train_index debe ser un vector unidimensional."
        )

    if validation_index.ndim != 1:
        raise ValueError(
            "validation_index debe ser un vector unidimensional."
        )

    if len(train_index) == 0:
        raise ValueError(
            "train_index está vacío."
        )

    if len(validation_index) == 0:
        raise ValueError(
            "validation_index está vacío."
        )

    if not np.issubdtype(
        train_index.dtype,
        np.integer
    ):
        raise TypeError(
            "train_index debe contener índices enteros."
        )

    if not np.issubdtype(
        validation_index.dtype,
        np.integer
    ):
        raise TypeError(
            "validation_index debe contener índices enteros."
        )

    if np.any(train_index < 0) or np.any(
        train_index >= len(graphs)
    ):
        raise IndexError(
            "train_index contiene índices fuera del rango de GraphData."
        )

    if np.any(validation_index < 0) or np.any(
        validation_index >= len(graphs)
    ):
        raise IndexError(
            "validation_index contiene índices fuera del rango de GraphData."
        )

    if set(train_index.tolist()) & set(
        validation_index.tolist()
    ):
        raise ValueError(
            "Existe solapamiento entre train_index y validation_index."
        )

    try:

        # Construcción del modelo
        input_channels = graphs[0].num_node_features # Recuperar número de variables de entrada

        model = build_gnn_model(
            model_config=model_config,
            input_channels=input_channels,
            output_channels=1
        ) # Construir arquitectura GNN oficial

        # Componentes oficiales del entrenamiento
        training_components = build_training_components(
            model=model,
            model_config=model_config
        ) # Construir criterio y optimizador oficiales

        # Entrenamiento
        training_result = train_gnn(
            model=model,
            graphs=graphs,
            train_index=train_index,
            validation_index=validation_index,
            criterion=training_components["criterion"],
            optimizer=training_components["optimizer"],
            model_config=model_config
        ) # Entrenar modelo utilizando exclusivamente train_index

        # Selección de grafos de validación
        validation_graphs = [
            graphs[int(index)]
            for index in validation_index
        ] # Seleccionar únicamente los GraphData de validación

        if len(validation_graphs) == 0:
            raise RuntimeError(
                "No existen GraphData para el conjunto de validación."
            )

        # Predicción sobre validación
        prediction_result = predict_gnn(
            model=training_result["model"],
            graphs=validation_graphs
        ) # Generar predicciones exclusivamente sobre validación

        if not isinstance(prediction_result, dict):
            raise TypeError(
                "prediction_result debe ser un diccionario."
            )

        if "y_pred" not in prediction_result:
            raise RuntimeError(
                "prediction_result no contiene 'y_pred'."
            )

        if "y_true" not in prediction_result:
            raise RuntimeError(
                "prediction_result no contiene 'y_true'."
            )

        y_pred = np.asarray(
            prediction_result["y_pred"]
        ).reshape(-1) # Normalizar predicciones de validación

        y_true = np.asarray(
            prediction_result["y_true"]
        ).reshape(-1) # Normalizar valores reales de validación

        if y_pred.shape != y_true.shape:
            raise RuntimeError(
                f"Las dimensiones de validación no coinciden: "
                f"y_pred={y_pred.shape}, "
                f"y_true={y_true.shape}."
            )

        # Evaluación sobre validación
        evaluation_result = evaluate_gnn(
            y_true=y_true,
            y_pred=y_pred
        ) # Evaluar exclusivamente el conjunto de validación

        # Resultado oficial del entrenamiento
        training_output = {
            "model": training_result["model"],
            "loss": training_result["loss"],
            "loss_history": training_result["loss_history"],
            "training_time": training_result["training_time"],
            "model_config": model_config,
            "prediction_result": prediction_result,
            "evaluation_result": evaluation_result,
        } # Construir producto oficial de entrenamiento y validación

        # Validación del producto
        required_products = [
            "model",
            "loss",
            "loss_history",
            "training_time",
            "model_config",
            "prediction_result",
            "evaluation_result",
        ] # Definir productos obligatorios

        missing_products = [
            product
            for product in required_products
            if product not in training_output
        ]

        if missing_products:
            raise RuntimeError(
                "El resultado oficial del entrenamiento está incompleto: "
                f"{missing_products}"
            )

        return training_output

    except Exception as error:
        raise RuntimeError(
            "Error durante la ejecución del entrenamiento oficial GNN."
        ) from error

# BLOQUE 12. Entrenamiento Oficial del Benchmark ----------------------------
# Objetivo: Ejecutar el protocolo oficial de entrenamiento, predicción y evaluación de una arquitectura GNN 
# utilizando la estructura oficial BenchmarkData del Benchmark Científico.
# Entradas: - model_config - benchmark_data
# Producto: - benchmark_result
# Responde: ¿La arquitectura GNN fue ejecutada correctamente bajo el protocolo oficial del Benchmark Científico?

def run_gnn_benchmark(
    model_config: dict,
    benchmark_data: dict
) -> dict:
    """
    Ejecuta el protocolo oficial del Benchmark Científico para una
    arquitectura GNN utilizando la colección oficial BenchmarkData.

    Parameters
    ----------
    model_config : dict
        Configuración oficial del modelo.

    benchmark_data : dict
        Colección oficial BenchmarkData.

    Returns
    -------
    dict
        Resultado oficial del Benchmark Científico.
    """

    if model_config is None:
        raise ValueError(
            "La configuración del modelo no puede ser nula."
        )

    if benchmark_data is None:
        raise ValueError(
            "BenchmarkData no puede ser nulo."
        )

    if not isinstance(
        benchmark_data,
        dict
    ):
        raise TypeError(
            "BenchmarkData debe ser un diccionario."
        )

    required_products = [
        "graphs",
        "train_index",
        "validation_index",
        "test_index",
        "x_train",
        "y_train",
        "x_validation",
        "y_validation",
        "x_test",
        "y_test",
    ] # Definir productos obligatorios de BenchmarkData

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

    graphs = benchmark_data["graphs"] # Recuperar GraphData oficiales
    train_index = np.asarray(
        benchmark_data["train_index"],
        dtype=int
    ) # Recuperar índices temporales de entrenamiento
    validation_index = np.asarray(
        benchmark_data["validation_index"],
        dtype=int
    ) # Recuperar índices temporales de validación
    test_index = np.asarray(
        benchmark_data["test_index"],
        dtype=int
    ) # Recuperar índices temporales de prueba

    y_test = np.asarray(
        benchmark_data["y_test"]
    ).reshape(-1) # Recuperar objetivo real de prueba

    if len(graphs) == 0:
        raise ValueError(
            "BenchmarkData no contiene GraphData."
        )

    if train_index.ndim != 1:
        raise ValueError(
            "train_index debe ser un vector unidimensional."
        )

    if validation_index.ndim != 1:
        raise ValueError(
            "validation_index debe ser un vector unidimensional."
        )

    if test_index.ndim != 1:
        raise ValueError(
            "test_index debe ser un vector unidimensional."
        )

    if len(train_index) == 0:
        raise ValueError(
            "train_index está vacío."
        )

    if len(validation_index) == 0:
        raise ValueError(
            "validation_index está vacío."
        )

    if len(test_index) == 0:
        raise ValueError(
            "test_index está vacío."
        )

    if np.any(train_index < 0) or np.any(train_index >= len(graphs)):
        raise IndexError(
            "train_index contiene índices temporales fuera de rango."
        )

    if np.any(validation_index < 0) or np.any(validation_index >= len(graphs)):
        raise IndexError(
            "validation_index contiene índices temporales fuera de rango."
        )

    if np.any(test_index < 0) or np.any(test_index >= len(graphs)):
        raise IndexError(
            "test_index contiene índices temporales fuera de rango."
        )

    train_graphs = [
        graphs[index]
        for index in train_index
    ] # Recuperar GraphData temporales de entrenamiento

    validation_graphs = [
        graphs[index]
        for index in validation_index
    ] # Recuperar GraphData temporales de validación

    test_graphs = [
        graphs[index]
        for index in test_index
    ] # Recuperar GraphData temporales de prueba

    if len(train_graphs) == 0:
        raise RuntimeError(
            "No existen GraphData para entrenamiento."
        )

    if len(validation_graphs) == 0:
        raise RuntimeError(
            "No existen GraphData para validación."
        )

    if len(test_graphs) == 0:
        raise RuntimeError(
            "No existen GraphData para prueba."
        )

    print("\n" + "=" * 80)
    print("PROTOCOLO TEMPORAL GNN")
    print("=" * 80)

    print(f"GraphData totales       : {len(graphs)}")
    print(f"GraphData entrenamiento : {len(train_graphs)}")
    print(f"GraphData validación    : {len(validation_graphs)}")
    print(f"GraphData prueba        : {len(test_graphs)}")
    print(f"Índices train           : {train_index}")
    print(f"Índices validation      : {validation_index}")
    print(f"Índices test            : {test_index}")
    print(f"Nodos por GraphData     : {graphs[0].num_nodes}")
    print(f"Variables predictoras   : {graphs[0].num_node_features}")

    input_channels = graphs[0].num_node_features # Determinar dimensión de entrada

    model = build_gnn_model(
        model_config=model_config,
        input_channels=input_channels,
        output_channels=1
    ) # Construir modelo GNN

    training_components = build_training_components(
        model=model,
        model_config=model_config
    ) # Construir componentes oficiales de entrenamiento

    training_result = train_gnn(
        model=model,
        graphs=train_graphs,
        criterion=training_components["criterion"],
        optimizer=training_components["optimizer"],
        model_config=model_config,
        train_index=None,
        validation_index=None
    ) # Entrenar exclusivamente con GraphData temporales de entrenamiento

    prediction_result = predict_gnn(
        model=training_result["model"],
        graphs=test_graphs
    ) # Generar predicciones exclusivamente sobre GraphData de prueba

    y_pred = np.asarray(
        prediction_result["y_pred"]
    ).reshape(-1) # Convertir predicciones de prueba a vector

    y_test = np.asarray(
        [
            graph.y.detach().cpu().numpy().reshape(-1)
            for graph in test_graphs
        ]
    ).reshape(-1) # Construir objetivo real de prueba desde los mismos GraphData

    print("\n" + "=" * 80)
    print("AUDITORÍA DE PREDICCIONES GNN")
    print("=" * 80)

    print(f"y_pred shape             : {y_pred.shape}")
    print(f"y_test shape             : {y_test.shape}")
    print(f"Predicciones             : {len(y_pred)}")
    print(f"Objetivos                : {len(y_test)}")

    if len(y_pred) != len(y_test):
        raise RuntimeError(
            "Las predicciones y los objetivos de prueba "
            "no tienen la misma cantidad de observaciones."
        )

    evaluation_result = evaluate_gnn(
        y_true=y_test,
        y_pred=y_pred
    ) # Evaluar exclusivamente sobre el conjunto de prueba

    benchmark_result = build_gnn_results(
        model_config=model_config,
        prediction_result={
            "y_pred": y_pred,
            "y_true": y_test,
            "inference_time": prediction_result["inference_time"],
        },
        evaluation_result=evaluation_result,
        training_result=training_result,
    ) # Construir resultado oficial del Benchmark

    required_result = [
        "model",
        "loss",
        "loss_history",
        "training_time",
        "model_config",
        "prediction_result",
        "evaluation_result",
    ] # Definir contrato del resultado

    missing_result = [
        product
        for product in required_result
        if product not in benchmark_result
    ]

    if missing_result:
        raise RuntimeError(
            "BenchmarkResult está incompleto: "
            f"{missing_result}"
        )

    print("\n" + "=" * 80)
    print("RESULTADO GNN")
    print("=" * 80)

    print(
        f"Modelo                   : "
        f"{model_config['model_name']}"
    )

    print(
        f"RMSE                     : "
        f"{evaluation_result['rmse']}"
    )

    print(
        f"MAE                      : "
        f"{evaluation_result['mae']}"
    )

    print(
        f"MAPE                     : "
        f"{evaluation_result['mape']}"
    )

    print(
        f"R2                       : "
        f"{evaluation_result['r2']}"
    )

    print("Protocolo GNN            : VALIDADO")
    print("Predicciones de prueba   : ALINEADAS")
    print("Objetivos de prueba      : ALINEADOS")

    return benchmark_result

# BLOQUE 13. Inferencia ----------------------------------------------------
# Objetivo: Generar predicciones utilizando una arquitectura GNN previamente entrenada sobre un nuevo grafo de entrada.
# Entradas: - model - graph_data
# Producto: - predictions - inference_time
# Responde: ¿La arquitectura GNN genera correctamente predicciones sobre un nuevo conjunto de datos?

def predict_new_graph(
    model: nn.Module,
    graph_data: Any
) -> dict:
    """
    Genera predicciones sobre un nuevo grafo.
    Parameters
    ----------
    model : nn.Module
        Modelo GNN entrenado.

    graph_data : Any
        Nuevo grafo sobre el cual se realizará la inferencia.

    Returns
    -------
    dict
        Predicciones y tiempo oficial de inferencia.
    """

    if model is None:
        raise ValueError(
            "El modelo no puede ser nulo."
        )

    if graph_data is None:
        raise ValueError(
            "graph_data no puede ser nulo."
        )
    model.eval()  # Modo evaluación
    inference_start = time.time()  # Inicio de la inferencia

    try:
        with torch.no_grad():
            outputs = model(
                graph_data.x,
                graph_data.edge_index
            )  # Salida del modelo

    except Exception as error:
        raise RuntimeError(
            "Error durante la inferencia sobre el nuevo grafo."
        ) from error

    inference_time = (
        time.time() - inference_start
    )  # Tiempo oficial de inferencia

    predictions = (
        outputs
        .detach()
        .cpu()
        .numpy()
        .ravel()
    )  # Conversión a NumPy

    return {
        "predictions": predictions,
        "inference_time": inference_time
    }  # Resultado oficial de la inferencia

# BLOQUE 14. Exportación ---------------------------------------------------
# Objetivo: Exportar la arquitectura GNN entrenada junto con su configuración oficial para garantizar la 
# reproducibilidad del Benchmark Científico.
# Entradas: - model - model_config - output_path
# Producto: - output_path
# Responde: ¿La arquitectura GNN fue exportada correctamente?

def export_gnn_model(
    model: nn.Module,
    model_config: dict,
    output_path: str
) -> str:
    """
    Exporta una arquitectura GNN entrenada.
    Parameters
    ----------
    model : nn.Module
        Modelo GNN entrenado.

    model_config : dict
        Configuración oficial del modelo.

    output_path : str
        Ruta donde será almacenado el modelo.

    Returns
    -------
    str
        Ruta del modelo exportado.
    """

    if model is None:
        raise ValueError(
            "El modelo no puede ser nulo."
        )

    if model_config is None:
        raise ValueError(
            "La configuración del modelo no puede ser nula."
        )

    if not output_path:
        raise ValueError(
            "La ruta de salida no puede estar vacía."
        )

    try:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_config
            },
            output_path
        )  # Exportación oficial del modelo

    except Exception as error:
        raise RuntimeError(
            "No fue posible exportar el modelo GNN."
        ) from error
    return output_path  # Ruta oficial del modelo exportado