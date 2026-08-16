# build_graphdata.py

# BUILD_GRAPHDATA.PY
# Objetivo: Construir el GraphData Oficial del proyecto a partir de la Feature Matrix (X), la Variable 
# Objetivo (y) y las Aristas Combinadas Oficiales, garantizando una representación
# consistente, reproducible y compatible con PyTorch Geometric.
# Arquitectura Científica: 1. Importación de Dependencias 2. Validación de las Entradas
# 3. Construcción del GraphData 4. Validación del GraphData 5. Construcción Oficial del GraphData
# Entradas: - Feature Matrix (X) - Variable Objetivo (y) - Aristas Combinadas Oficiales
# Producto: - GraphData Oficial compatible con PyTorch Geometric.
# Pregunta Científica: ¿La integración de la Feature Matrix, la Variable Objetivo y las Aristas Combinadas
# representa correctamente el Grafo Científico utilizado por los modelos Graph Neural Networks del proyecto?

# BLOQUE 1. IMPORTACIÓN DE DEPENDENCIAS

# Objetivo: Importar las librerías científicas, la configuración oficial del proyecto y los metadatos
# requeridos para la construcción, validación y certificación del GraphData Oficial.
# Componentes: 1.1 Librerías Científicas 1.2 Configuración Oficial del Proyecto 1.3 Metadatos del Builder
# Producto: Dependencias oficiales inicializadas correctamente. 
# Responde: ¿El entorno dispone de todas las dependencias necesarias para construir el GraphData Oficial del proyecto?

import torch # Construcción y validación de tensores para PyTorch
from torch_geometric.data import Data # Objeto GraphData de PyTorch Geometric

from src.python.config.config_project import (
    TORCH_FLOAT_DTYPE,
    TORCH_INT_DTYPE,
)


# METADATOS DEL MÓDULO

MODULE_INFO = {
    "builder": "build_graphdata",
    "version": "2.0.0",
}

# BLOQUE 2. VALIDACIÓN DE LAS ENTRADAS

# Objetivo: Verificar la integridad estructural de la Feature Matrix (X), la Variable Objetivo (y)
# y las Aristas Combinadas Oficiales antes de construir el GraphData Oficial del proyecto.
# Componentes: 2.1 Validación de la Feature Matrix 2.2 Validación de la Variable Objetivo
# 2.3 Validación de las Aristas Combinadas 2.4 Resumen Científico
# Entradas: - Feature Matrix (X) - Variable Objetivo (y) - Aristas Combinadas Oficiales
# Producto: Confirmación de la integridad estructural de las entradas utilizadas para construir
# el GraphData Oficial.
# Responde: ¿Las entradas poseen la estructura necesaria para construir correctamente el GraphData 
# Oficial del proyecto?

def validate_graphdata_inputs(
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index_final: torch.Tensor,
) -> None:
    """
    Valida la integridad estructural de las entradas del GraphData.
    """

    print("-" * 80)
    print("Validación de las Entradas del GraphData")
    print("-" * 80)

    # 2.1 Validación de la Feature Matrix
    # -------------------------------------------------------------------------
    if x is None:
        raise ValueError(
            "La matriz de características no puede ser None."
        )

    if not isinstance(x, torch.Tensor):
        raise TypeError(
            "La matriz de características debe ser un tensor de PyTorch."
        )

    if x.dtype != TORCH_FLOAT_DTYPE:
        raise TypeError(
            "La matriz de características posee un tipo de dato inválido."
        )

    if x.ndim != 2:
        raise ValueError(
            "La matriz de características debe tener dos dimensiones."
        )

    if x.shape[0] == 0:
        raise ValueError(
            "La matriz de características no contiene nodos."
        )

    if x.shape[1] == 0:
        raise ValueError(
            "La matriz de características no contiene variables."
        )

    if torch.isnan(x).any():
        raise ValueError(
            "La matriz de características contiene valores NaN."
        )

    if torch.isinf(x).any():
        raise ValueError(
            "La matriz de características contiene valores Inf."
        )

    # 2.2 Validación de la Variable Objetivo
    # -------------------------------------------------------------------------
    if y is None:
        raise ValueError(
            "La variable objetivo no puede ser None."
        )

    if not isinstance(y, torch.Tensor):
        raise TypeError(
            "La variable objetivo debe ser un tensor de PyTorch."
        )

    if y.dtype != TORCH_FLOAT_DTYPE:
        raise TypeError(
            "La variable objetivo posee un tipo de dato inválido."
        )

    if y.ndim != 2:
        raise ValueError(
            "La variable objetivo debe tener dos dimensiones."
        )

    if y.shape[1] != 1:
        raise ValueError(
            "La variable objetivo debe contener una única columna."
        )

    if y.shape[0] != x.shape[0]:
        raise ValueError(
            "La variable objetivo no coincide con el número de nodos."
        )

    if torch.isnan(y).any():
        raise ValueError(
            "La variable objetivo contiene valores NaN."
        )

    if torch.isinf(y).any():
        raise ValueError(
            "La variable objetivo contiene valores Inf."
        )

    # 2.3 Validación de las Aristas Combinadas
    # -------------------------------------------------------------------------
    if edge_index_final is None:
        raise ValueError(
            "Las Aristas Combinadas no pueden ser None."
        )

    if not isinstance(edge_index_final, torch.Tensor):
        raise TypeError(
            "Las Aristas Combinadas deben ser un tensor de PyTorch."
        )

    if edge_index_final.dtype != TORCH_INT_DTYPE:
        raise TypeError(
            "Las Aristas Combinadas poseen un tipo de dato inválido."
        )

    if edge_index_final.ndim != 2:
        raise ValueError(
            "Las Aristas Combinadas deben tener dos dimensiones."
        )

    if edge_index_final.shape[0] != 2:
        raise ValueError(
            "Las Aristas Combinadas deben tener dos filas."
        )

    if edge_index_final.shape[1] == 0:
        raise ValueError(
            "Las Aristas Combinadas no contienen aristas."
        )

    if edge_index_final.min().item() < 0:
        raise ValueError(
            "Las Aristas Combinadas contienen índices negativos."
        )

    if edge_index_final.max().item() >= x.shape[0]:
        raise ValueError(
            "Las Aristas Combinadas contienen índices fuera del rango permitido."
        )

    n_unique_edges = torch.unique(
        edge_index_final.T,
        dim=0,
    ).shape[0] # Número de aristas únicas

    if n_unique_edges != edge_index_final.shape[1]:
        raise ValueError(
            "Las Aristas Combinadas contienen duplicados."
        )

    # 2.4 Resumen Científico
    # -------------------------------------------------------------------------
    print("Entradas del GraphData validadas correctamente.")
    print(f"Nodos                    : {x.shape[0]:,}")
    print(f"Variables                : {x.shape[1]:,}")
    print(f"Aristas                  : {edge_index_final.shape[1]:,}")
    print("Estado                   : CORRECTO")

# 3. CONSTRUCCIÓN DEL GRAPHDATA

# Objetivo: Construir el GraphData Oficial del proyecto integrando la Feature Matrix (X), la Variable
# Objetivo (y) y las Aristas Combinadas Oficiales en un objeto compatible con PyTorch Geometric.
# Componentes: 3.1 Construcción del GraphData 3.2 Validación de la Construcción 3.3 Resumen Científico
# Entradas: - Feature Matrix (X) - Variable Objetivo (y) - Aristas Combinadas Oficiales
# Producto: GraphData Oficial compatible con PyTorch Geometric.
# Responde: ¿La integración de la Feature Matrix, la Variable Objetivo y las Aristas Combinadas representa
# correctamente el GraphData Oficial del proyecto?

def build_graphdata(
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index_final: torch.Tensor,
) -> Data:
    """
    Construye el GraphData Oficial del proyecto.

    Parameters
    ----------
    x : torch.Tensor
        Matriz oficial de características.

    y : torch.Tensor
        Variable objetivo oficial.

    edge_index_final : torch.Tensor
        Aristas Combinadas Oficiales.

    Returns
    -------
    Data
        Objeto GraphData compatible con PyTorch Geometric.
    """

    print("-" * 80)
    print("Construcción del GraphData")
    print("-" * 80)

    # 3.1 Construcción del GraphData
    # -------------------------------------------------------------------------
    graph_data = Data(

        # Matriz oficial de características
        x=x,

        # Variable objetivo oficial
        y=y,

        # Aristas Combinadas Oficiales
        edge_index=edge_index_final,

        # Número oficial de nodos
        num_nodes=x.shape[0],
    ) # Construir el GraphData Oficial

    # 3.2 Validación de la construcción
    # -------------------------------------------------------------------------
    if graph_data.num_nodes == 0:
        raise ValueError(
            "No fue posible construir el GraphData."
        )

    if graph_data.x.shape[0] != graph_data.num_nodes:
        raise ValueError(
            "La Feature Matrix es inconsistente con el número de nodos."
        )

    if graph_data.edge_index.shape[1] == 0:
        raise ValueError(
            "El GraphData no contiene Aristas Combinadas."
        )

    # 3.3 Resumen científico
    # -------------------------------------------------------------------------
    print("GraphData construido correctamente.")
    print(f"Nodos                    : {graph_data.num_nodes:,}")
    print(f"Variables                : {graph_data.x.shape[1]:,}")
    print(f"Aristas                  : {graph_data.edge_index.shape[1]:,}")
    print("Estado                   : CORRECTO")

    return graph_data

# BLOQUE 4. VALIDACIÓN DEL GRAPHDATA

# # Objetivo: Verificar la integridad estructural del GraphData Oficial, garantizando su consistencia,
# completitud y compatibilidad con PyTorch Geometric antes de ser utilizado por los modelos Graph Neural
# Networks del proyecto.
# Componentes: 4.1 Validación del Objeto GraphData 4.2 Validación de la Feature Matrix
# 4.3 Validación de la Variable Objetivo 4.4 Validación de las Aristas Combinadas
# 4.5 Validación de la Coherencia Estructural 4.6 Resumen Científico
# Entradas: - GraphData Oficial
# Producto: Confirmación de la integridad estructural del GraphData Oficial.
# Responde:  ¿El GraphData Oficial posee una estructura consistente, completa y compatible con PyTorch 
# Geometric para el entrenamiento de Graph Neural Networks?

def validate_graphdata(
    graph_data: Data,
) -> None:
    """
    Valida la integridad estructural del GraphData Oficial.
    """

    print("-" * 80)
    print("Validación del GraphData")
    print("-" * 80)

    # 4.1 Validación del objeto
    # -------------------------------------------------------------------------
    if graph_data is None:
        raise ValueError(
            "El GraphData no puede ser None."
        )

    if not isinstance(graph_data, Data):
        raise TypeError(
            "El GraphData debe ser un objeto Data de PyTorch Geometric."
        )

    # 4.2 Validación de la matriz de características
    # -------------------------------------------------------------------------
    if graph_data.x is None:
        raise ValueError(
            "El GraphData no contiene la matriz de características."
        )

    if graph_data.x.dtype != TORCH_FLOAT_DTYPE:
        raise TypeError(
            "La matriz de características posee un tipo de dato inválido."
        )

    if graph_data.x.ndim != 2:
        raise ValueError(
            "La matriz de características debe tener dos dimensiones."
        )

    # 4.3 Validación de la variable objetivo
    # -------------------------------------------------------------------------
    if graph_data.y is None:
        raise ValueError(
            "El GraphData no contiene la variable objetivo."
        )

    if graph_data.y.dtype != TORCH_FLOAT_DTYPE:
        raise TypeError(
            "La variable objetivo posee un tipo de dato inválido."
        )

    if graph_data.y.ndim != 2:
        raise ValueError(
            "La variable objetivo debe tener dos dimensiones."
        )

    if graph_data.y.shape[1] != 1:
        raise ValueError(
            "La variable objetivo debe contener una única columna."
        )

    # 4.4 Validación de las Aristas Combinadas
    # -------------------------------------------------------------------------
    if graph_data.edge_index is None:
        raise ValueError(
            "El GraphData no contiene las Aristas Combinadas."
        )

    if graph_data.edge_index.dtype != TORCH_INT_DTYPE:
        raise TypeError(
            "Las Aristas Combinadas poseen un tipo de dato inválido."
        )

    if graph_data.edge_index.ndim != 2:
        raise ValueError(
            "Las Aristas Combinadas deben tener dos dimensiones."
        )

    if graph_data.edge_index.shape[0] != 2:
        raise ValueError(
            "Las Aristas Combinadas deben tener dos filas."
        )

    if graph_data.edge_index.shape[1] == 0:
        raise ValueError(
            "El GraphData no contiene Aristas Combinadas."
        )

    # 4.5 Validación de coherencia estructural
    # -------------------------------------------------------------------------
    if graph_data.num_nodes != graph_data.x.shape[0]:
        raise ValueError(
            "El número de nodos no coincide con la matriz de características."
        )

    if graph_data.y.shape[0] != graph_data.num_nodes:
        raise ValueError(
            "La variable objetivo no coincide con el número de nodos."
        )

    if graph_data.x.shape[0] != graph_data.y.shape[0]:
        raise ValueError(
            "La Feature Matrix y la Variable Objetivo poseen un número diferente de nodos."
        )

    if graph_data.edge_index.max().item() >= graph_data.num_nodes:
        raise ValueError(
            "Las Aristas Combinadas contienen índices fuera del rango permitido."
        )

    if graph_data.edge_index.min().item() < 0:
        raise ValueError(
            "Las Aristas Combinadas contienen índices negativos."
        )

    if torch.isnan(graph_data.x).any():
        raise ValueError(
            "La matriz de características contiene valores NaN."
        )

    if torch.isinf(graph_data.x).any():
        raise ValueError(
            "La matriz de características contiene valores Inf."
        )

    if torch.isnan(graph_data.y).any():
        raise ValueError(
            "La variable objetivo contiene valores NaN."
        )

    if torch.isinf(graph_data.y).any():
        raise ValueError(
            "La variable objetivo contiene valores Inf."
        )

    n_unique_edges = torch.unique(
        graph_data.edge_index.T,
        dim=0,
    ).shape[0] # Número de aristas únicas

    if n_unique_edges != graph_data.edge_index.shape[1]:
        raise ValueError(
            "El GraphData contiene Aristas Combinadas duplicadas."
        )

    # 4.6 Resumen científico
    # -------------------------------------------------------------------------
    print("GraphData validado correctamente.")
    print(f"Nodos                    : {graph_data.num_nodes:,}")
    print(f"Variables                : {graph_data.x.shape[1]:,}")
    print(f"Aristas                  : {graph_data.edge_index.shape[1]:,}")
    print("Estado                   : CORRECTO")

# BLOQUE 5. CONSTRUCCIÓN OFICIAL DEL GRAPHDATA

# Objetivo: Orquestar la validación, construcción y verificación del GraphData Oficial, proporcionando
# una única interfaz pública para el resto del pipeline científico del proyecto.
# Componentes: 5.1 Validación de las Entradas 5.2 Construcción del GraphData 5.3 Validación del GraphData
# 5.4 Incorporación de Metadatos 5.5 Retorno del GraphData Oficial
# Entradas: - Feature Matrix (X) - Variable Objetivo (y) - Aristas Combinadas Oficiales - Año Científico
# Producto: GraphData Oficial completamente validado y preparado para ser utilizado por los modelos Graph
# Neural Networks del proyecto.
# Responde: ¿El GraphData Oficial fue construido, validado y certificado correctamente para representar 
# el Grafo Científico del proyecto?

def prepare_graphdata(
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index_final: torch.Tensor,
    current_year: int,
) -> Data:
    """
    Construye el GraphData Oficial del proyecto.

    Parameters
    ----------
    x : torch.Tensor
        Matriz oficial de características.

    y : torch.Tensor
        Variable objetivo oficial.

    edge_index_final : torch.Tensor
        Aristas Combinadas Oficiales.

    current_year : int
        Año representado por el GraphData.

    Returns
    -------
    Data
        Objeto GraphData Oficial compatible con PyTorch Geometric.
    """

    # 5.1 Validación de las entradas
    # -------------------------------------------------------------------------
    validate_graphdata_inputs(
        x=x,
        y=y,
        edge_index_final=edge_index_final,
    ) # Validar las entradas del GraphData

    # 5.2 Construcción del GraphData
    # -------------------------------------------------------------------------
    graph_data = build_graphdata(
        x=x,
        y=y,
        edge_index_final=edge_index_final,
    ) # Construir el GraphData Oficial

    # 5.3 Validación del GraphData
    # -------------------------------------------------------------------------
    validate_graphdata(
        graph_data=graph_data,
    ) # Validar el GraphData Oficial

    # 5.4 Incorporación de Metadatos del GraphData
    # -------------------------------------------------------------------------
    if not isinstance(current_year, int):
        raise TypeError(
            "El año científico debe ser un entero."
        )

    if current_year <= 0:
        raise ValueError(
            "El año científico es inválido."
        )

    graph_data.current_year = current_year # Año representado por el GraphData

    if graph_data.current_year != current_year:
        raise RuntimeError(
            "No fue posible registrar el año científico del GraphData."
        )

    graph_data.builder = MODULE_INFO["builder"] # Constructor oficial
    graph_data.builder_version = MODULE_INFO["version"] # Versión del constructor
    graph_data.graph_type = "official_graphdata" # Tipo oficial del objeto

    # 5.5 Retorno del GraphData Oficial
    # -------------------------------------------------------------------------
    return graph_data

# def validate_graphdata_inputs() valida las entradas
# build_graphdata() → únicamente construye el objeto Data.
# validate_graphdata() → valida el objeto GraphData.
# prepare_graphdata() → orquesta todo el proceso y es la interfaz pública del builder.