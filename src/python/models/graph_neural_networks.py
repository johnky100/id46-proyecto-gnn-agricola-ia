# graph_neural_networks.py

# BLOQUE 1. Importaciones --------------------------------------------------
## Objetivo: Importar las librerías necesarias para construir, entrenar, evaluar y
# exportar los modelos Graph Neural Networks utilizados durante el Benchmark Científico.
### Producto: - Librerías cargadas correctamente.
### Responde: ¿Las dependencias necesarias para implementar las arquitecturas GNN fueron importadas correctamente?

# Funciones del sistema
import time # Medición del tiempo de entrenamiento
import warnings # Control de advertencias

# Librerías científicas
import numpy as np # Operaciones numéricas

# PyTorch
import torch # Tensor principal
import torch.nn as nn # Capas neuronales
import torch.nn.functional as F # Funciones de activación

# Optimizadores
from torch.optim import Adam # Optimizador Adam

# PyTorch Geometric
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    TAGConv
) # Capas espaciales GNN

# Métricas oficiales
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
) # Métricas del Benchmark

# Utilidades del proyecto
from src.python.utils.results import (
    build_benchmark_result
) # Construcción del resultado oficial

# Configuración oficial
from src.python.config.config_project import (
    MODEL_CODES,
    SEED
) # Configuración del Benchmark

warnings.filterwarnings(
    "ignore"
) # Ocultar advertencias no críticas

# Reproducibilidad
torch.manual_seed(
    SEED
) # Semilla para PyTorch

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    ) # Semilla para GPU

# BLOQUE 2. Configuración --------------------------------------------------
## Objetivo: Definir la configuración oficial de las arquitecturas Graph Neural
# Networks utilizadas durante el Benchmark Científico.
### Producto: - GNN_CONFIG
### Responde: ¿Las arquitecturas GNN disponen de una configuración oficial, reproducible y consistente con el Benchmark Científico?

# Configuración oficial de las arquitecturas -------------------------------
GNN_CONFIG = {
    "gcn": {
        "model_code": MODEL_CODES["GNN01"],
        "model_name": "gcn",
        "family": "graph_neural_networks",
        "hidden_channels": 64,
        "dropout": 0.30,
        "learning_rate": 0.001,
        "weight_decay": 5e-4,
        "epochs": 300
    },

    "graphsage": {
        "model_code": MODEL_CODES["GNN02"],
        "model_name": "graphsage",
        "family": "graph_neural_networks",
        "hidden_channels": 64,
        "dropout": 0.30,
        "learning_rate": 0.001,
        "weight_decay": 5e-4,
        "epochs": 300
    },

    "gat": {
        "model_code": MODEL_CODES["GNN03"],
        "model_name": "gat",
        "family": "graph_neural_networks",
        "hidden_channels": 64,
        "heads": 4,
        "dropout": 0.30,
        "learning_rate": 0.001,
        "weight_decay": 5e-4,
        "epochs": 300
    },

    "gin": {
        "model_code": MODEL_CODES["GNN04"],
        "model_name": "gin",
        "family": "graph_neural_networks",
        "hidden_channels": 64,
        "dropout": 0.30,
        "learning_rate": 0.001,
        "weight_decay": 5e-4,
        "epochs": 300
    },

    "tagcn": {
        "model_code": MODEL_CODES["GNN05"],
        "model_name": "tagcn",
        "family": "graph_neural_networks",
        "hidden_channels": 64,
        "K": 3,
        "dropout": 0.30,
        "learning_rate": 0.001,
        "weight_decay": 5e-4,
        "epochs": 300
    }
} # Configuración oficial de las arquitecturas GNN

# BLOQUE 3. Construcción de Arquitecturas GNN

# BLOQUE 3.1. Clase GCNModel -----------------------------------------------
## Objetivo: Definir la arquitectura oficial del modelo Graph Convolutional Network
# (GCN) utilizada durante el Benchmark Científico.
### Entradas: - input_channels - hidden_channels - output_channels - dropout
### Producto: - GCNModel
### Responde: ¿La arquitectura GCN fue construida correctamente para el proceso de entrenamiento del Benchmark Científico?
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
        input_channels,
        hidden_channels,
        output_channels,
        dropout
    ):

        super().__init__()

        self.conv1 = GCNConv(
            input_channels,
            hidden_channels
        ) # Primera capa convolucional

        self.conv2 = GCNConv(
            hidden_channels,
            output_channels
        ) # Segunda capa convolucional

        self.dropout = dropout # Probabilidad de Dropout

    def forward(
        self,
        x,
        edge_index
    ):
        """
        Propagación hacia adelante del modelo.

        Parameters
        ----------
        x : Tensor
            Matriz de características de los nodos.

        edge_index : Tensor
            Índices de las aristas del grafo.

        Returns
        -------
        Tensor
            Predicciones del modelo.
        """

        x = self.conv1(
            x,
            edge_index
        ) # Primera convolución

        x = F.relu(
            x
        ) # Función de activación

        x = F.dropout(
            x,
            p = self.dropout,
            training = self.training
        ) # Regularización

        x = self.conv2(
            x,
            edge_index
        ) # Segunda convolución

        return x
    
# BLOQUE 3.2. Clase GraphSAGEModel -----------------------------------------
## Objetivo: Definir la arquitectura oficial del modelo GraphSAGE utilizada durante el Benchmark Científico.
### Entradas: - input_channels - hidden_channels - output_channels - dropout
### Producto: - GraphSAGEModel
### Responde: ¿La arquitectura GraphSAGE fue construida correctamente para el proceso
# de entrenamiento del Benchmark Científico?
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
        input_channels,
        hidden_channels,
        output_channels,
        dropout
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            input_channels,
            hidden_channels
        ) # Primera capa GraphSAGE

        self.conv2 = SAGEConv(
            hidden_channels,
            output_channels
        ) # Segunda capa GraphSAGE

        self.dropout = dropout # Probabilidad de Dropout

    def forward(
        self,
        x,
        edge_index
    ):
        """
        Propagación hacia adelante del modelo.

        Parameters
        ----------
        x : Tensor
            Matriz de características de los nodos.

        edge_index : Tensor
            Índices de las aristas del grafo.

        Returns
        -------
        Tensor
            Predicciones del modelo.
        """

        x = self.conv1(
            x,
            edge_index
        ) # Primera convolución

        x = F.relu(
            x
        ) # Función de activación

        x = F.dropout(
            x,
            p = self.dropout,
            training = self.training
        ) # Regularización

        x = self.conv2(
            x,
            edge_index
        ) # Segunda convolución

        return x

# BLOQUE 3.3. Clase GATModel -----------------------------------------------
## Objetivo: Definir la arquitectura oficial del modelo Graph Attention Network
# (GAT) utilizada durante el Benchmark Científico.
### Entradas: - input_channels - hidden_channels - output_channels - heads - dropout
### Producto: - GATModel
### Responde: ¿La arquitectura GAT fue construida correctamente para el proceso de entrenamiento del Benchmark Científico?
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
        Número de cabezas de atención.

    dropout : float
        Probabilidad de desactivación de neuronas.
    """

    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        heads,
        dropout
    ):

        super().__init__()

        self.conv1 = GATConv(
            input_channels,
            hidden_channels,
            heads = heads,
            dropout = dropout
        ) # Primera capa GAT

        self.conv2 = GATConv(
            hidden_channels * heads,
            output_channels,
            heads = 1,
            concat = False,
            dropout = dropout
        ) # Segunda capa GAT

        self.dropout = dropout # Probabilidad de Dropout

    def forward(
        self,
        x,
        edge_index
    ):
        """
        Propagación hacia adelante del modelo.

        Parameters
        ----------
        x : Tensor
            Matriz de características de los nodos.

        edge_index : Tensor
            Índices de las aristas del grafo.

        Returns
        -------
        Tensor
            Predicciones del modelo.
        """

        x = self.conv1(
            x,
            edge_index
        ) # Primera capa de atención

        x = F.elu(
            x
        ) # Función de activación recomendada para GAT

        x = F.dropout(
            x,
            p = self.dropout,
            training = self.training
        ) # Regularización

        x = self.conv2(
            x,
            edge_index
        ) # Segunda capa de atención

        return x

# BLOQUE 3.4. Clase GINModel -----------------------------------------------
## Objetivo: Definir la arquitectura oficial del modelo Graph Isomorphism Network
# (GIN) utilizada durante el Benchmark Científico.
### Entradas: - input_channels - hidden_channels - output_channels - dropout
### Producto: - GINModel
### Responde: ¿La arquitectura GIN fue construida correctamente para el proceso de
# entrenamiento del Benchmark Científico?

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
        input_channels,
        hidden_channels,
        output_channels,
        dropout
    ):

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

        ) # MLP de la primera capa

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

        ) # MLP de la segunda capa

        self.conv1 = GINConv(
            mlp1
        ) # Primera capa GIN

        self.conv2 = GINConv(
            mlp2
        ) # Segunda capa GIN

        self.dropout = dropout # Probabilidad de Dropout

    def forward(
        self,
        x,
        edge_index
    ):
        """
        Propagación hacia adelante del modelo.

        Parameters
        ----------
        x : Tensor
            Matriz de características de los nodos.

        edge_index : Tensor
            Índices de las aristas del grafo.

        Returns
        -------
        Tensor
            Predicciones del modelo.
        """

        x = self.conv1(
            x,
            edge_index
        ) # Primera capa GIN

        x = F.relu(
            x
        ) # Función de activación

        x = F.dropout(
            x,
            p = self.dropout,
            training = self.training
        ) # Regularización

        x = self.conv2(
            x,
            edge_index
        ) # Segunda capa GIN

        return x


# BLOQUE 3.5. Clase TAGCNModel ---------------------------------------------
## Objetivo: Definir la arquitectura oficial del modelo Topology Adaptive Graph
# Convolutional Network (TAGCN) utilizada durante el Benchmark Científico.
### Entradas: - input_channels - hidden_channels - output_channels - K - dropout
### Producto: - TAGCNModel
### Responde: ¿La arquitectura TAGCN fue construida correctamente para el proceso de
# entrenamiento del Benchmark Científico?

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
        input_channels,
        hidden_channels,
        output_channels,
        K,
        dropout
    ):

        super().__init__()

        self.conv1 = TAGConv(
            input_channels,
            hidden_channels,
            K = K
        ) # Primera capa TAGCN

        self.conv2 = TAGConv(
            hidden_channels,
            output_channels,
            K = K
        ) # Segunda capa TAGCN

        self.dropout = dropout # Probabilidad de Dropout

    def forward(
        self,
        x,
        edge_index
    ):
        """
        Propagación hacia adelante del modelo.

        Parameters
        ----------
        x : Tensor
            Matriz de características de los nodos.

        edge_index : Tensor
            Índices de las aristas del grafo.

        Returns
        -------
        Tensor
            Predicciones del modelo.
        """

        x = self.conv1(
            x,
            edge_index
        ) # Primera convolución

        x = F.relu(
            x
        ) # Función de activación

        x = F.dropout(
            x,
            p = self.dropout,
            training = self.training
        ) # Regularización

        x = self.conv2(
            x,
            edge_index
        ) # Segunda convolución

        return x

# BLOQUE 4. Construcción del Modelo ----------------------------------------
## Objetivo: Construir la arquitectura Graph Neural Network seleccionada utilizando
# la configuración oficial del Benchmark Científico.
### Entradas: - model_config - input_channels - output_channels
### Producto: - model
### Responde: ¿La arquitectura Graph Neural Network fue construida correctamente?

def build_gnn_model(
    model_config,
    input_channels,
    output_channels
):
    """
    Construye la arquitectura GNN especificada en la configuración oficial.

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
        Modelo Graph Neural Network.
    """

    model_name = model_config["model_name"] # Nombre del modelo
    hidden_channels = model_config["hidden_channels"] # Capas ocultas
    dropout = model_config["dropout"] # Dropout

    try:
        if model_name == "gcn":
            model = GCNModel(
                input_channels = input_channels,
                hidden_channels = hidden_channels,
                output_channels = output_channels,
                dropout = dropout
            )

        elif model_name == "graphsage":
            model = GraphSAGEModel(
                input_channels = input_channels,
                hidden_channels = hidden_channels,
                output_channels = output_channels,
                dropout = dropout
            )

        elif model_name == "gat":
            model = GATModel(
                input_channels = input_channels,
                hidden_channels = hidden_channels,
                output_channels = output_channels,
                heads = model_config["heads"],
                dropout = dropout
            )

        elif model_name == "gin":
            model = GINModel(
                input_channels = input_channels,
                hidden_channels = hidden_channels,
                output_channels = output_channels,
                dropout = dropout
            )

        elif model_name == "tagcn":
            model = TAGCNModel(
                input_channels = input_channels,
                hidden_channels = hidden_channels,
                output_channels = output_channels,
                K = model_config["K"],
                dropout = dropout

            )

        else:
            raise ValueError(
                f"Modelo GNN no soportado: {model_name}"
            )

    except Exception as error:
        raise RuntimeError(
            f"Error al construir la arquitectura GNN: {error}"
        )

    return model

# BLOQUE 5. Función de Pérdida y Optimizador -------------------------------
## Objetivo: Construir la función de pérdida y el optimizador oficial utilizados
# durante el entrenamiento de las arquitecturas Graph Neural Networks.
### Entradas: - model - model_config
### Producto: - criterion - optimizer
### Responde: ¿La función de pérdida y el optimizador fueron configurados
# correctamente para el entrenamiento del modelo GNN?

def build_training_components(
    model,
    model_config
):
    """
    Construye la función de pérdida y el optimizador oficial para una GNN.

    Parameters
    ----------
    model : nn.Module
        Modelo Graph Neural Network.

    model_config : dict
        Configuración oficial del modelo.

    Returns
    -------
    dict
        Función de pérdida y optimizador.
    """

    try:
        criterion = nn.MSELoss() # Función de pérdida oficial
        optimizer = Adam(
            model.parameters(),
            lr = model_config["learning_rate"],
            weight_decay = model_config["weight_decay"]
        ) # Optimizador Adam

    except Exception as error:
        raise RuntimeError(
            f"Error al construir los componentes de entrenamiento: {error}"
        )

    return {
        "criterion": criterion,
        "optimizer": optimizer
    }

# BLOQUE 6. Entrenamiento --------------------------------------------------
## Objetivo: Entrenar la arquitectura Graph Neural Network utilizando el conjunto de
# entrenamiento definido por el Benchmark Científico.
### Entradas: - model - graph_data - criterion - optimizer - model_config
### Producto: - trained_model - training_time
### Responde: ¿La arquitectura Graph Neural Network fue entrenada correctamente?

def train_gnn(
    model,
    graph_data,
    criterion,
    optimizer,
    model_config
):
    """
    Entrena una arquitectura Graph Neural Network.

    Parameters
    ----------
    model : nn.Module
        Modelo Graph Neural Network.

    graph_data : Data
        Grafo oficial del Benchmark.

    criterion : nn.Module
        Función de pérdida.

    optimizer : torch.optim.Optimizer
        Optimizador del modelo.

    model_config : dict
        Configuración oficial del modelo.

    Returns
    -------
    dict
        Modelo entrenado y tiempo oficial de entrenamiento.
    """

    model.train() # Modo entrenamiento
    training_start = time.time() # Inicio del entrenamiento
    try:
        for epoch in range(
            model_config["epochs"]
        ):

            optimizer.zero_grad() # Reiniciar gradientes
            predictions = model(
                graph_data.x,
                graph_data.edge_index
            ) # Forward

            loss = criterion(
                predictions[graph_data.train_mask],
                graph_data.y[graph_data.train_mask]
            ) # Función de pérdida

            loss.backward() # Backpropagation
            optimizer.step() # Actualización de pesos

    except Exception as error:
        raise RuntimeError(
            f"Error durante el entrenamiento: {error}"
        )

    training_time = (
        time.time() - training_start
    ) # Tiempo oficial de entrenamiento

    return {
        "model": model,
        "training_time": training_time,
        "loss": loss.item()
    }

# BLOQUE 7. Predicción -----------------------------------------------------
## Objetivo: Generar las predicciones de la arquitectura Graph Neural Network sobre
# el conjunto de prueba definido por el Benchmark Científico.
### Entradas: - model - graph_data
### Producto: - y_pred - y_true - inference_time
### Responde: ¿La arquitectura Graph Neural Network genera correctamente las predicciones sobre el conjunto de prueba?

def predict_gnn(
    model,
    graph_data
):
    """
    Genera las predicciones utilizando una Graph Neural Network.

    Parameters
    ----------
    model : nn.Module
        Modelo Graph Neural Network entrenado.

    graph_data : Data
        Grafo oficial del Benchmark.

    Returns
    -------
    dict
        Predicciones, valores observados y tiempo oficial de inferencia.
    """

    model.eval() # Modo evaluación
    inference_start = time.time() # Inicio de la inferencia
    try:
        with torch.no_grad():
            predictions = model(
                graph_data.x,
                graph_data.edge_index
            ) # Predicciones sobre todos los nodos

        y_pred = predictions[
            graph_data.test_mask
        ] # Predicciones del conjunto de prueba

        y_true = graph_data.y[
            graph_data.test_mask
        ] # Valores observados

    except Exception as error:
        raise RuntimeError(
            f"Error durante la inferencia: {error}"
        )

    inference_time = (
        time.time() - inference_start
    ) # Tiempo oficial de inferencia

    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "inference_time": inference_time
    }

# BLOQUE 8. Evaluación -----------------------------------------------------
## Objetivo: Calcular las métricas oficiales de desempeño predictivo para la
# arquitectura Graph Neural Network utilizando el conjunto de prueba.
### Entradas: - y_true - y_pred
### Producto: - evaluation_result
### Responde: ¿Cuál es el desempeño predictivo de la arquitectura Graph Neural Network sobre el conjunto de prueba?

def evaluate_gnn(
    y_true,
    y_pred
):
    """
    Calcula las métricas oficiales del Benchmark para una arquitectura
    Graph Neural Network.

    Parameters
    ----------
    y_true : ndarray
        Valores observados.

    y_pred : ndarray
        Valores predichos.

    Returns
    -------
    dict
        Métricas oficiales de evaluación.
    """

    try:
        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        ) # Error cuadrático medio

        mae = mean_absolute_error(
            y_true,
            y_pred
        ) # Error absoluto medio

        mape = mean_absolute_percentage_error(
            y_true,
            y_pred
        ) # Error porcentual absoluto medio

        r2 = r2_score(
            y_true,
            y_pred
        ) # Coeficiente de determinación
        adjusted_r2 = np.nan # Se calculará posteriormente

    except Exception as error:
        raise RuntimeError(
            f"Error durante la evaluación del modelo GNN: {error}"
        )

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "adjusted_r2": adjusted_r2
    }

# BLOQUE 9. Construcción del Resultado Oficial -----------------------------
## Objetivo: Construir la estructura oficial de resultados de la arquitectura Graph
# Neural Network compatible con el Benchmark Científico.
### Entradas: - model_config - training_result - prediction_result - evaluation_result
### Producto: - benchmark_result
### Responde: ¿Los resultados de la arquitectura Graph Neural Network fueron
# consolidados correctamente para el Benchmark Científico?

def build_gnn_results(
    model_config,
    prediction_result,
    evaluation_result,
    training_result = None
):
    """
    Construye el resultado oficial del Benchmark para una arquitectura
    Graph Neural Network.

    Parameters
    ----------
    model_config : dict
        Configuración oficial del modelo.

    training_result : dict
        Resultado del entrenamiento.

    prediction_result : dict
        Resultado de la predicción.

    evaluation_result : dict
        Resultado de la evaluación.

    Returns
    -------
    dict
        Resultado oficial del Benchmark.
    """

    benchmark_result = build_benchmark_result(
        model_config = model_config,
        prediction_result = prediction_result,
        evaluation_result = evaluation_result,
        training_result = training_result

    ) # Resultado oficial del Benchmark

    return benchmark_result

# BLOQUE 10. Ejecución del Modelo ------------------------------------------
## Objetivo: Ejecutar de forma secuencial el flujo completo de una arquitectura Graph Neural Network, 
# incluyendo construcción del modelo, configuración del entrenamiento, entrenamiento, predicción,
# evaluación y construcción del resultado oficial.
### Entradas: - model_config - graph_data
### Producto: - benchmark_result
### Responde:
# ¿La arquitectura Graph Neural Network fue ejecutada correctamente bajo
# el protocolo oficial del Benchmark Científico?

def run_gnn(
    model_config,
    graph_data
):
    """
    Ejecuta el flujo completo de una arquitectura Graph Neural Network.

    Parameters
    ----------
    model_config : dict
        Configuración oficial del modelo.

    graph_data : Data
        Grafo oficial del Benchmark.

    Returns
    -------
    dict
        Resultado oficial del Benchmark.
    """

    # Construcción del modelo ----------------------------------------------
    model = build_gnn_model(
        model_config = model_config,
        input_channels = graph_data.num_node_features,
        output_channels = 1
    ) # Arquitectura GNN

    # Componentes de entrenamiento -----------------------------------------
    training_components = build_training_components(
        model = model,
        model_config = model_config
    )

    # Entrenamiento --------------------------------------------------------
    training_result = train_gnn(
        model = model,
        graph_data = graph_data,
        criterion = training_components["criterion"],
        optimizer = training_components["optimizer"],
        model_config = model_config
    )

    # Predicción -----------------------------------------------------------
    prediction_result = predict_gnn(
        model = training_result["model"],
        graph_data = graph_data
    )

    # Evaluación -----------------------------------------------------------
    evaluation_result = evaluate_gnn(
        y_true = prediction_result["y_true"],
        y_pred = prediction_result["y_pred"]
    )

    # Resultado oficial ----------------------------------------------------
    return build_gnn_results(
        model_config = model_config,
        prediction_result = prediction_result,
        evaluation_result = evaluation_result,
        training_result = training_result
    )


# BLOQUE 11. Inferencia ----------------------------------------------------
## Objetivo:
# Generar predicciones utilizando una arquitectura Graph Neural Network
# previamente entrenada sobre un nuevo grafo de entrada.
##
# Entradas:
# - model
# - graph_data
##
# Producto:
# - predictions
# - inference_time
##
# Responde:
# ¿La arquitectura Graph Neural Network genera correctamente
# predicciones sobre un nuevo conjunto de datos?

def predict_new_graph(
    model,
    graph_data
):
    """
    Genera predicciones sobre un nuevo grafo.

    Parameters
    ----------
    model : nn.Module
        Modelo Graph Neural Network entrenado.

    graph_data : Data
        Nuevo grafo sobre el cual se realizará la inferencia.

    Returns
    -------
    dict
        Predicciones y tiempo oficial de inferencia.
    """

    model.eval() # Modo evaluación
    inference_start = time.time() # Inicio de la inferencia
    try:
        with torch.no_grad():
            predictions = model(
                graph_data.x,
                graph_data.edge_index
            ) # Predicciones del modelo

    except Exception as error:
        raise RuntimeError(
            f"Error durante la inferencia: {error}"
        )

    inference_time = (
        time.time() - inference_start
    ) # Tiempo oficial de inferencia

    predictions = (
        predictions
        .detach()
        .cpu()
        .numpy()
        .ravel()
    ) # Conversión a NumPy

    return {
        "predictions": predictions,
        "inference_time": inference_time
    }

# BLOQUE 12. Exportación ---------------------------------------------------
## Objetivo:
# Exportar la arquitectura Graph Neural Network entrenada junto con su
# configuración oficial para garantizar la reproducibilidad del
# Benchmark Científico.
##
# Entradas:
# - model
# - model_config
# - output_path
##
# Producto:
# - Archivo .pt
##
# Responde:
# ¿La arquitectura Graph Neural Network fue exportada correctamente?

def export_gnn_model(
    model,
    model_config,
    output_path
):
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

    try:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_config
            },

            output_path
        ) # Exportación del modelo

    except Exception as error:
        raise RuntimeError(
            f"Error al exportar el modelo: {error}"
        )

    return output_path

