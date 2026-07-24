# models-graph_neural_networks.py

# BLOQUE 1. Importaciones --------------------------------------------------
## Objetivo: Importar las librerías necesarias para construir, entrenar,
## evaluar y exportar las arquitecturas Graph Neural Networks utilizadas
## durante el Benchmark Científico.
##
## Producto:
## - Librerías cargadas correctamente.
##
## Responde:
## ¿Las dependencias necesarias para implementar las arquitecturas GNN
## fueron importadas correctamente?

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
## Objetivo: Definir la configuración oficial de las arquitecturas Graph
## Neural Networks utilizadas durante el Benchmark Científico.
##
## Producto:
## - GNN_CONFIG
##
## Responde:
## ¿Las arquitecturas Graph Neural Networks disponen de una configuración
## oficial, reproducible y consistente con el Benchmark Científico?

# Configuración oficial de las arquitecturas -------------------------------
GNN_CONFIG = {

    "gcn": {

        # Identificación
        "model_code": BENCHMARK_MODEL_CODES["GNN01"],
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
        "model_code": BENCHMARK_MODEL_CODES["GNN02"],
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
        "model_code": BENCHMARK_MODEL_CODES["GNN03"],
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
        "model_code": BENCHMARK_MODEL_CODES["GNN04"],
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
        "model_code": BENCHMARK_MODEL_CODES["GNN05"],
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
## Objetivo: Definir las arquitecturas oficiales Graph Neural Networks (GNN)
## utilizadas durante el Benchmark Científico.
##
## Producto:
## - GCNModel
## - GraphSAGEModel
## - GATModel
## - GINModel
## - TAGCNModel
##
## Responde:
## ¿Las arquitecturas oficiales Graph Neural Networks fueron construidas de
## forma modular, reproducible y consistente con el Benchmark Científico?

# BLOQUE 3.1. Clase GCNModel -----------------------------------------------
## Objetivo: Definir la arquitectura oficial del modelo Graph Convolutional
## Network (GCN) utilizada durante el Benchmark Científico.
##
## Entradas:
## - input_channels
## - hidden_channels
## - output_channels
## - dropout
##
## Producto:
## - GCNModel
##
## Responde:
## ¿La arquitectura GCN fue construida correctamente para el proceso de
## entrenamiento del Benchmark Científico?

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
## Objetivo: Definir la arquitectura oficial del modelo GraphSAGE utilizada
## durante el Benchmark Científico.
##
## Entradas:
## - input_channels
## - hidden_channels
## - output_channels
## - dropout
##
## Producto:
## - GraphSAGEModel
##
## Responde:
## ¿La arquitectura GraphSAGE fue construida correctamente para el proceso
## de entrenamiento del Benchmark Científico?

class GraphSAGEModel(
    nn.Module
):
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
        )  # Primera capa GraphSAGE

        self.conv2 = SAGEConv(
            hidden_channels,
            output_channels
        )  # Segunda capa GraphSAGE

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

# BLOQUE 3.3. Clase GATModel -----------------------------------------------
## Objetivo: Definir la arquitectura oficial del modelo Graph Attention
## Network (GAT) utilizada durante el Benchmark Científico.
##
## Entradas:
## - input_channels
## - hidden_channels
## - output_channels
## - heads
## - dropout
##
## Producto:
## - GATModel
##
## Responde:
## ¿La arquitectura GAT fue construida correctamente para el proceso de
## entrenamiento del Benchmark Científico?

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
## Objetivo: Definir la arquitectura oficial del modelo Graph Isomorphism
## Network (GIN) utilizada durante el Benchmark Científico.
##
## Entradas:
## - input_channels
## - hidden_channels
## - output_channels
## - dropout
##
## Producto:
## - GINModel
##
## Responde:
## ¿La arquitectura GIN fue construida correctamente para el proceso de
## entrenamiento del Benchmark Científico?

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
## Objetivo: Definir la arquitectura oficial del modelo Topology Adaptive
## Graph Convolutional Network (TAGCN) utilizada durante el Benchmark
## Científico.
##
## Entradas:
## - input_channels
## - hidden_channels
## - output_channels
## - K
## - dropout
##
## Producto:
## - TAGCNModel
##
## Responde:
## ¿La arquitectura TAGCN fue construida correctamente para el proceso de
## entrenamiento del Benchmark Científico?

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
## Objetivo: Construir la arquitectura Graph Neural Network seleccionada
## utilizando la configuración oficial del Benchmark Científico.
##
## Entradas:
## - model_config
## - input_channels
## - output_channels
##
## Producto:
## - model
##
## Responde:
## ¿La arquitectura Graph Neural Network fue construida correctamente?

def build_gnn_model(
    model_config: dict,
    input_channels: int,
    output_channels: int
) -> nn.Module:
    """
    Construye la arquitectura Graph Neural Network especificada en la
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
        Arquitectura Graph Neural Network construida.
    """

    if model_config is None:
        raise ValueError(
            "La configuración del modelo no puede ser nula."
        )

    if input_channels <= 0:
        raise ValueError(
            "input_channels debe ser mayor que cero."
        )

    if output_channels <= 0:
        raise ValueError(
            "output_channels debe ser mayor que cero."
        )

    try:

        model_name = model_config["model_name"]
        hidden_channels = model_config["hidden_channels"]
        dropout = model_config["dropout"]

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

        return model

    except Exception as error:

        raise RuntimeError(
            f"No fue posible construir la arquitectura GNN '{model_config.get('model_name', 'desconocido')}'."
        ) from error
    
# BLOQUE 5. Función de Pérdida y Optimizador -------------------------------
## Objetivo: Construir la función de pérdida y el optimizador oficial
## utilizados durante el entrenamiento de las arquitecturas Graph Neural
## Networks.
##
## Entradas:
## - model
## - model_config
##
## Producto:
## - criterion
## - optimizer
##
## Responde:
## ¿La función de pérdida y el optimizador fueron configurados
## correctamente para el entrenamiento del modelo GNN?

def build_training_components(
    model: nn.Module,
    model_config: dict
) -> dict:
    """
    Construye la función de pérdida y el optimizador oficial para una
    arquitectura Graph Neural Network.

    Parameters
    ----------
    model : nn.Module
        Modelo Graph Neural Network.

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

    if model_config is None:
        raise ValueError(
            "La configuración del modelo no puede ser nula."
        )

    try:

        criterion = nn.MSELoss()  # Función de pérdida oficial

        optimizer = Adam(
            model.parameters(),
            lr=model_config["learning_rate"],
            weight_decay=model_config["weight_decay"]
        )  # Optimizador oficial

        return {
            "criterion": criterion,
            "optimizer": optimizer
        }

    except Exception as error:

        raise RuntimeError(
            "No fue posible construir los componentes oficiales del entrenamiento."
        ) from error
    
# BLOQUE 6. Entrenamiento --------------------------------------------------
## Objetivo: Entrenar la arquitectura Graph Neural Network utilizando la
## colección oficial de grafos espacio-temporales del Benchmark Científico.
##
## Entradas:
## - model
## - graphs
## - criterion
## - optimizer
## - model_config
##
## Producto:
## - trained_model
## - training_time
## - loss
##
## Responde:
## ¿La arquitectura Graph Neural Network fue entrenada correctamente sobre
## la colección oficial de grafos?

def train_gnn(
    model: nn.Module,
    graphs: list,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: dict
) -> dict:
    """
    Entrena una arquitectura Graph Neural Network utilizando la colección
    oficial de grafos del Benchmark Científico.

    Parameters
    ----------
    model : nn.Module
        Modelo Graph Neural Network.

    graphs : list
        Colección oficial de GraphData.

    criterion : nn.Module
        Función de pérdida.

    optimizer : torch.optim.Optimizer
        Optimizador.

    model_config : dict
        Configuración oficial.

    Returns
    -------
    dict
        Modelo entrenado, tiempo y pérdida final.
    """

    if model is None:
        raise ValueError("El modelo no puede ser nulo.")

    if graphs is None or len(graphs) == 0:
        raise ValueError("La colección de GraphData está vacía.")

    if criterion is None:
        raise ValueError("La función de pérdida no puede ser nula.")

    if optimizer is None:
        raise ValueError("El optimizador no puede ser nulo.")

    if model_config is None:
        raise ValueError("La configuración del modelo no puede ser nula.")

    model.train()
    training_start = time.time()
    epoch_loss = None
    loss_history = []

    try:

        for epoch in range(model_config["epochs"]):

            accumulated_loss = 0.0

            for graph_index, graph in enumerate(graphs):

                optimizer.zero_grad()

                predictions = model(
                    graph.x,
                    graph.edge_index
                )

                if epoch == 0 and graph_index == 0:

                    print("\n" + "=" * 80)
                    print("AUDITORÍA DEL ENTRENAMIENTO GNN")
                    print("=" * 80)
                    print(f"Cantidad de grafos : {len(graphs)}")
                    print(f"Nodos              : {graph.num_nodes}")
                    print(f"Variables          : {graph.num_node_features}")
                    print(f"Predictions        : {tuple(predictions.shape)}")
                    print(f"Target             : {tuple(graph.y.shape)}")

                loss = criterion(
                    predictions.squeeze(-1),
                    graph.y
                )

                loss.backward()
                optimizer.step()

                accumulated_loss += loss.item()

            epoch_loss = accumulated_loss / len(graphs)
            loss_history.append(epoch_loss)

    except Exception as error:

        raise RuntimeError(
            "Error durante el entrenamiento del modelo GNN."
        ) from error

    return {

        "model": model,
        "training_time": time.time() - training_start,
        "loss": epoch_loss,
        "loss_history": loss_history

    }

# BLOQUE 7. Predicción -----------------------------------------------------
## Objetivo: Generar las predicciones utilizando la colección oficial de
## grafos espacio-temporales del Benchmark Científico.
##
## Entradas:
## - model
## - graphs
##
## Producto:
## - y_pred
## - y_true
## - inference_time
##
## Responde:
## ¿La arquitectura Graph Neural Network genera correctamente las
## predicciones sobre la colección oficial de grafos?

def predict_gnn(
    model: nn.Module,
    graphs: list
) -> dict:
    """
    Genera las predicciones utilizando la colección oficial de GraphData.

    Parameters
    ----------
    model : nn.Module
        Modelo Graph Neural Network entrenado.

    graphs : list
        Colección oficial de GraphData.

    Returns
    -------
    dict
        Predicciones, valores observados y tiempo oficial de inferencia.
    """

    if model is None:
        raise ValueError("El modelo no puede ser nulo.")

    if graphs is None or len(graphs) == 0:
        raise ValueError("La colección de GraphData está vacía.")

    model.eval()
    inference_start = time.time()

    try:

        prediction_list = []
        target_list = []

        with torch.no_grad():

            for graph in graphs:

                outputs = model(
                    graph.x,
                    graph.edge_index
                )

                prediction_list.append(
                    outputs.squeeze(-1).cpu()
                )

                target_list.append(
                    graph.y.cpu()
                )

        y_pred = torch.cat(
            prediction_list,
            dim=0
        )

        y_true = torch.cat(
            target_list,
            dim=0
        )

    except Exception as error:

        raise RuntimeError(
            "Error durante la inferencia del modelo GNN."
        ) from error

    return {

        "y_pred": y_pred,

        "y_true": y_true,

        "inference_time": (
            time.time() - inference_start
        )

    }

# BLOQUE 8. Evaluación -----------------------------------------------------
## Objetivo: Calcular las métricas oficiales de desempeño predictivo para la
## arquitectura Graph Neural Network utilizando el conjunto de prueba.
##
## Entradas:
## - y_true
## - y_pred
##
## Producto:
## - evaluation_result
##
## Responde:
## ¿Cuál es el desempeño predictivo de la arquitectura Graph Neural Network
## sobre el conjunto de prueba?

def evaluate_gnn(
    y_true: Any,
    y_pred: Any
) -> dict:
    """
    Calcula las métricas oficiales del Benchmark para una arquitectura
    Graph Neural Network.

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

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )  # Error cuadrático medio

        mae = mean_absolute_error(
            y_true,
            y_pred
        )  # Error absoluto medio

        mape = mean_absolute_percentage_error(
            y_true,
            y_pred
        )  # Error porcentual absoluto medio

        r2 = r2_score(
            y_true,
            y_pred
        )  # Coeficiente de determinación

    except Exception as error:

        raise RuntimeError(
            "Error durante la evaluación del modelo GNN."
        ) from error

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2
    }  # Resultado oficial de la evaluación

# BLOQUE 9. Construcción del Resultado Oficial -----------------------------
## Objetivo: Construir la estructura oficial de resultados de la arquitectura
## Graph Neural Network compatible con el Benchmark Científico.
##
## Entradas:
## - model_config
## - prediction_result
## - evaluation_result
## - training_result
##
## Producto:
## - benchmark_result
##
## Responde:
## ¿Los resultados de la arquitectura Graph Neural Network fueron
## consolidados correctamente para el Benchmark Científico?

def build_gnn_results(
    model_config: dict,
    prediction_result: dict,
    evaluation_result: dict,
    training_result: dict | None = None
) -> dict:
    """
    Construye el resultado oficial del Benchmark para una arquitectura
    Graph Neural Network.

    Parameters
    ----------
    model_config : dict
        Configuración oficial del modelo.

    prediction_result : dict
        Resultado de la predicción.

    evaluation_result : dict
        Resultado de la evaluación.

    training_result : dict, optional
        Resultado del entrenamiento.

    Returns
    -------
    dict
        Resultado oficial del Benchmark.
    """

    if model_config is None:
        raise ValueError(
            "La configuración del modelo no puede ser nula."
        )

    if prediction_result is None:
        raise ValueError(
            "prediction_result no puede ser nulo."
        )

    if evaluation_result is None:
        raise ValueError(
            "evaluation_result no puede ser nulo."
        )

    try:

        return build_benchmark_result(

            model_config=model_config,

            prediction_result=prediction_result,

            evaluation_result=evaluation_result,

            training_result=training_result

        )  # Resultado oficial del Benchmark

    except Exception as error:

        raise RuntimeError(
            "No fue posible construir el resultado oficial del Benchmark."
        ) from error
    
# BLOQUE 10. Ejecución del Modelo ------------------------------------------
## Objetivo: Ejecutar el flujo completo de una arquitectura Graph Neural
## Network utilizando la colección oficial de grafos del Benchmark
## Científico.
##
## Entradas:
## - model_config
## - graphs
##
## Producto:
## - benchmark_result
##
## Responde:
## ¿La arquitectura Graph Neural Network fue ejecutada correctamente sobre
## la colección oficial de grafos?

def run_gnn(
    model_config: dict,
    graphs: list
) -> dict:
    """
    Ejecuta el flujo completo de una arquitectura Graph Neural Network.

    Parameters
    ----------
    model_config : dict
        Configuración oficial del modelo.

    graphs : list
        Colección oficial de GraphData.

    Returns
    -------
    dict
        Resultado oficial del Benchmark.
    """

    if model_config is None:
        raise ValueError("La configuración del modelo no puede ser nula.")

    if graphs is None or len(graphs) == 0:
        raise ValueError("La colección de GraphData está vacía.")

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

        prediction_result = predict_gnn(
            model=training_result["model"],
            graphs=graphs
        )

        evaluation_result = evaluate_gnn(
            y_true=prediction_result["y_true"],
            y_pred=prediction_result["y_pred"]
        )

        return build_gnn_results(
            model_config=model_config,
            prediction_result=prediction_result,
            evaluation_result=evaluation_result,
            training_result=training_result
        )

    except Exception as error:

        raise RuntimeError(
            "Error durante la ejecución del flujo oficial del modelo GNN."
        ) from error

# BLOQUE 11. Inferencia ----------------------------------------------------
## Objetivo: Generar predicciones utilizando una arquitectura Graph Neural
## Network previamente entrenada sobre un nuevo grafo de entrada.
##
## Entradas:
## - model
## - graph_data
##
## Producto:
## - predictions
## - inference_time
##
## Responde:
## ¿La arquitectura Graph Neural Network genera correctamente predicciones
## sobre un nuevo conjunto de datos?

def predict_new_graph(
    model: nn.Module,
    graph_data: Any
) -> dict:
    """
    Genera predicciones sobre un nuevo grafo.

    Parameters
    ----------
    model : nn.Module
        Modelo Graph Neural Network entrenado.

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

# BLOQUE 12. Exportación ---------------------------------------------------
## Objetivo: Exportar la arquitectura Graph Neural Network entrenada junto
## con su configuración oficial para garantizar la reproducibilidad del
## Benchmark Científico.
##
## Entradas:
## - model
## - model_config
## - output_path
##
## Producto:
## - output_path
##
## Responde:
## ¿La arquitectura Graph Neural Network fue exportada correctamente?

def export_gnn_model(
    model: nn.Module,
    model_config: dict,
    output_path: str
) -> str:
    """
    Exporta una arquitectura Graph Neural Network entrenada.

    Parameters
    ----------
    model : nn.Module
        Modelo Graph Neural Network entrenado.

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