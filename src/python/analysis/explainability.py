# analysis-explainability.py

# BLOQUE 1. Importaciones, Configuración y Utilidades --------------------------
## Objetivo: Importar las librerías científicas, herramientas de
# Inteligencia Artificial Explicable (XAI), componentes de
# PyTorch Geometric, utilidades del sistema y configuraciones
# oficiales necesarias para ejecutar el proceso oficial de
# explicabilidad del modelo oficial GraphSAGE.
#### Responde:
# ¿El módulo dispone de todas las dependencias necesarias para
# ejecutar de forma reproducible el proceso oficial de
# explicabilidad del modelo GraphSAGE?

"""
analysis/explainability.py

Módulo oficial de Explicabilidad (XAI) del Proyecto GeoAI.

interpretar el comportamiento del modelo oficial GraphSAGE
del proyecto GeoAI.

Responsabilidades
-----------------
- Inicializar el contexto de explicabilidad.
- Seleccionar automáticamente el método XAI compatible con GraphSAGE.
- Construir el Explainer.
- Generar la explicabilidad global.
- Calcular la importancia de variables.
- Construir rankings de importancia.
- Generar visualizaciones científicas.
- Elaborar el resumen científico.
- Exportar los resultados oficiales.
- Validar la consistencia de la explicabilidad.
- Auditar la importancia de variables.
- Analizar la relación entre las variables y la variable objetivo.
"""

from __future__ import annotations

# Librerías estándar
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any
from scipy.stats import spearmanr

# Librerías científicas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Explainable AI
import shap

# Deep Learning
import torch
import torch.nn.functional as F

from src.python.config.config_project import (
    PROJECT_SEED
)

from torch_geometric.explain import (
    Explainer,
    GNNExplainer
)

from torch_geometric.explain.config import (
    ExplanationType,
    ModelConfig,
    ModelMode,
    ModelTaskLevel
)

print("-" * 80)
print("Bloque 1. Importaciones.")

# BLOQUE 2. Inicialización del Contexto de Explicabilidad ----------------------
## Objetivo: Construir la estructura oficial que almacenará toda la información
# generada durante el proceso de explicabilidad del modelo oficial GraphSAGE,
# incluyendo el método XAI, el explainer, los resultados, las visualizaciones,
# las auditorías, las exportaciones y los tiempos de ejecución.
#### Producto:
# - global_explainability
#### Responde:
# ¿Existe una estructura oficial, reproducible y consistente para almacenar
# todos los resultados del proceso de explicabilidad del modelo oficial?

def initialize_global_explainability(
    model_metadata: dict,
    reports_dir: Path
) -> dict:
    """
    Inicializa la estructura oficial utilizada durante el proceso de
    Explicabilidad Global (XAI) del modelo oficial GraphSAGE.

    Parameters
    ----------
    model_metadata : dict
        Metadatos oficiales del modelo.

    reports_dir : Path
        Directorio oficial donde se almacenarán los resultados.

    Returns
    -------
    dict
        Contexto oficial de la explicabilidad.
    """

    # Validación -----------------------------------------------------------
    if not isinstance(model_metadata, dict):
        raise TypeError(
            "model_metadata debe ser un diccionario."
        )

    if not isinstance(reports_dir, Path):
        raise TypeError(
            "reports_dir debe ser un objeto Path."
        )

    required_metadata = [
        "model_code",
        "family",
        "model_name"
    ]

    missing_metadata = [
        key
        for key in required_metadata
        if key not in model_metadata
    ]

    if missing_metadata:
        raise RuntimeError(
            "Faltan metadatos oficiales del modelo: "
            f"{missing_metadata}"
        )

    # Construcción ---------------------------------------------------------
    explainability_dir = (
        reports_dir /
        "explainability"
    )

    explainability_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    global_explainability = {

        # Información del modelo
        "model_code": model_metadata["model_code"],
        "model_family": model_metadata["family"],
        "model_name": model_metadata["model_name"],

        # Configuración
        "configuration": {
            "xai_library": None,
            "xai_version": None,
            "correlation_method": "spearman",
            "random_state": PROJECT_SEED
        },

        # Información de ejecución
        "created_at": datetime.now().isoformat(),

        # Método de explicabilidad
        "method": None,
        "explainer": None,

        # Resultados
        "feature_importance": None,
        "feature_ranking": None,
        "top_5_variables": None,
        "top_10_variables": None,
        "top_20_variables": None,
        "scientific_summary": None,
        "target_correlation": None,

        # Visualizaciones
        "plots_dir": explainability_dir,
        "plots": {},

        # Auditorías
        "importance_audit": None,

        # Exportaciones
        "exported_files": {},

        # Tiempos de ejecución
        "execution_time": {
            "explainer_construction": 0.0,
            "explainability": 0.0,
            "feature_importance": 0.0,
            "feature_ranking": 0.0,
            "plots": 0.0,
            "scientific_summary": 0.0,
            "export": 0.0,
            "validation": 0.0,
            "importance_audit": 0.0,
            "target_correlation": 0.0,
            "total": 0.0
        },

        # Estado
        "status": "INITIALIZED"
    }

    # Validación -----------------------------------------------------------
    required_keys = [
        "model_code",
        "model_family",
        "model_name",
        "configuration",
        "created_at",
        "method",
        "explainer",
        "feature_importance",
        "feature_ranking",
        "top_5_variables",
        "top_10_variables",
        "top_20_variables",
        "scientific_summary",
        "target_correlation",
        "plots_dir",
        "plots",
        "importance_audit",
        "exported_files",
        "execution_time",
        "status"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in global_explainability
    ]

    if missing_keys:
        raise RuntimeError(
            "La estructura oficial de la explicabilidad "
            f"está incompleta: {missing_keys}"
        )

    if not isinstance(global_explainability["configuration"], dict):
        raise RuntimeError(
            "configuration debe ser un diccionario."
        )

    if not isinstance(global_explainability["execution_time"], dict):
        raise RuntimeError(
            "execution_time debe ser un diccionario."
        )

    if not isinstance(global_explainability["plots"], dict):
        raise RuntimeError(
            "plots debe ser un diccionario."
        )

    if not isinstance(global_explainability["exported_files"], dict):
        raise RuntimeError(
            "exported_files debe ser un diccionario."
        )

    # Retorno --------------------------------------------------------------
    return global_explainability

print("-" * 80)
print("Bloque 2. Inicialización del Contexto de Explicabilidad Finalizado.")

# BLOQUE 3. Selección del Método de Explicabilidad ----------------------------
## Objetivo: Seleccionar automáticamente el método oficial de explicabilidad
# compatible con el modelo oficial GraphSAGE, garantizando la compatibilidad
# entre el modelo y la técnica de interpretación.
#### Producto:
# - method
#### Responde:
# ¿Cuál es el método oficial de explicabilidad utilizado para interpretar
# el comportamiento del modelo oficial GraphSAGE?
def select_explainability_method(
    global_explainability: dict
) -> str:
    """
    Selecciona el método oficial de explicabilidad (XAI)
    utilizado para interpretar el modelo oficial GraphSAGE.

    Parameters
    ----------
    global_explainability : dict
        Contexto oficial de la explicabilidad.

    Returns
    -------
    str
        Método oficial de explicabilidad.
    """

    # Validación -------------------------------------------------------------
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    required_keys = [
        "model_family",
        "model_name",
        "configuration"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in global_explainability
    ]

    if missing_keys:
        raise RuntimeError(
            "La estructura de explicabilidad está incompleta: "
            f"{missing_keys}"
        )

    if not isinstance(global_explainability["configuration"], dict):
        raise RuntimeError(
            "configuration debe ser un diccionario."
        )

    # Recuperación -----------------------------------------------------------
    model_family = global_explainability["model_family"]
    model_name = global_explainability["model_name"].lower()

    # Validación -------------------------------------------------------------
    if model_family != "graph_neural_networks":
        raise RuntimeError(
            "El módulo de explicabilidad únicamente admite "
            "Graph Neural Networks."
        )

    if model_name != "graphsage":
        raise RuntimeError(
            "El modelo oficial del proyecto es GraphSAGE."
        )

    # Construcción -----------------------------------------------------------
    method = "GNNExplainer"

    global_explainability["configuration"]["xai_library"] = (
        "torch_geometric.explain"
    )

    global_explainability["configuration"]["xai_version"] = (
        torch.__version__
    )

    global_explainability["method"] = method
    global_explainability["status"] = "METHOD_SELECTED"

    # Retorno ----------------------------------------------------------------
    return method

# BLOQUE 4. Construcción del Explainer ---------------------------------------
# Objetivo:
# Construir y registrar el objeto oficial de explicabilidad (Explainer)
# compatible con la arquitectura GraphSAGE, estableciendo la configuración
# científica que será utilizada posteriormente para interpretar las
# predicciones del modelo mediante GNNExplainer.
#
# Entradas:
# - global_explainability
# - trained_model
#
# Productos:
# - Explainer configurado
# - global_explainability actualizado
#
# Responde:
# ¿Se configuró correctamente el objeto oficial de explicabilidad que
# permitirá interpretar las predicciones del modelo GraphSAGE durante
# la etapa de evaluación?
def build_explainer(
    global_explainability: dict,
    trained_model
):
    """
    Construye y registra el objeto oficial de explicabilidad para el
    modelo GraphSAGE.

    Parameters
    ----------
    global_explainability : dict
        Contexto oficial de la explicabilidad.

    trained_model :
        Modelo GraphSAGE previamente entrenado.

    Returns
    -------
    dict
        Contexto oficial de explicabilidad actualizado.
    """

    # Validación -------------------------------------------------------------
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    required_keys = [
        "configuration",
        "method"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in global_explainability
    ]

    if missing_keys:
        raise RuntimeError(
            "La estructura de explicabilidad está incompleta: "
            f"{missing_keys}"
        )

    if trained_model is None:
        raise RuntimeError(
            "trained_model no puede ser None."
        )

    if not isinstance(global_explainability["configuration"], dict):
        raise RuntimeError(
            "configuration debe ser un diccionario."
        )

    # Construcción -----------------------------------------------------------
    trained_model.eval()

    explainer = Explainer(
        model=trained_model,
        algorithm=GNNExplainer(
            epochs=200
        ),
        explanation_type=ExplanationType.model,
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(
            mode=ModelMode.regression,
            task_level=ModelTaskLevel.node,
            return_type="raw"
        )
    )

    global_explainability["configuration"].update({
        "algorithm": "GNNExplainer",
        "epochs": 200,
        "explanation_type": "model",
        "node_mask_type": "attributes",
        "edge_mask_type": "object",
        "task_level": "node",
        "mode": "regression",
        "return_type": "raw"
    })

    global_explainability["explainer"] = explainer
    global_explainability["explainer_ready"] = True
    global_explainability["status"] = "EXPLAINER_BUILT"

    # Retorno ----------------------------------------------------------------
    return global_explainability

# BLOQUE 5. Recorrido de los GraphData ----------------------------------------
#
# Objetivo:
# Identificar todos los nodos de los grafos oficiales que serán
# explicados mediante GNNExplainer.#
# Entradas:
# - global_explainability
# - graphs
#
# Producto:
# - global_explainability actualizado
#
# Responde:
# ¿Se identificaron correctamente los nodos que serán explicados?
def build_explanation_targets(
    global_explainability: dict,
    graphs: list
) -> dict:
    """
    Construye la lista oficial de nodos que serán utilizados durante el
    proceso de explicabilidad.

    Parameters
    ----------
    global_explainability : dict
        Contexto oficial de la explicabilidad.

    graphs : list
        Colección oficial de GraphData.

    Returns
    -------
    dict
        Contexto oficial de explicabilidad actualizado.
    """

    # Validación -------------------------------------------------------------
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    if not isinstance(graphs, list):
        raise TypeError(
            "graphs debe ser una lista."
        )

    if len(graphs) == 0:
        raise RuntimeError(
            "La colección de grafos está vacía."
        )

    # Construcción -----------------------------------------------------------
    explanation_targets = []

    for graph_index, graph in enumerate(graphs):

        if not hasattr(graph, "num_nodes"):
            raise RuntimeError(
                "El grafo no contiene el atributo num_nodes."
            )

        for node_index in range(graph.num_nodes):

            explanation_targets.append({

                "graph_index": graph_index,
                "node_index": node_index

            })

    global_explainability["explanation_targets"] = (
        explanation_targets
    )

    global_explainability["explained_graphs"] = (
        len(graphs)
    )

    global_explainability["explained_nodes"] = (
        len(explanation_targets)
    )

    global_explainability["status"] = (
        "TARGETS_IDENTIFIED"
    )

    # Retorno ----------------------------------------------------------------
    return global_explainability

# BLOQUE 5.1. Inicialización del Contexto de Explicabilidad -------------------
#
# Objetivo:
# Inicializar el contexto oficial de explicabilidad antes de generar las
# explicaciones individuales del modelo GraphSAGE.
#
# Entradas:
# - global_explainability
#
# Producto:
# - global_explainability inicializado
#
# Responde:
# ¿Está preparado el contexto oficial para almacenar las explicaciones
# individuales y los resultados agregados?
def initialize_explainability_context(
    global_explainability: dict
) -> dict:
    """
    Inicializa el contexto oficial utilizado durante la generación de
    explicaciones del modelo GraphSAGE.

    Parameters
    ----------
    global_explainability : dict
        Contexto oficial de la explicabilidad.

    Returns
    -------
    dict
        Contexto oficial de explicabilidad inicializado.
    """

    # Validación -------------------------------------------------------------
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    required_keys = [
        "method",
        "explainer",
        "status"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in global_explainability
    ]

    if missing_keys:
        raise RuntimeError(
            "La estructura de explicabilidad está incompleta: "
            f"{missing_keys}"
        )

    # Construcción -----------------------------------------------------------
    global_explainability["node_explanations"] = []

    global_explainability["feature_importance"] = None

    global_explainability["feature_importance_std"] = None

    global_explainability["feature_ranking"] = None

    global_explainability["explained_nodes"] = 0

    global_explainability["explained_graphs"] = 0

    global_explainability["status"] = (
        "EXPLAINABILITY_INITIALIZED"
    )

    # Retorno ----------------------------------------------------------------
    return global_explainability

# BLOQUE 5.2. Generación de las Explicaciones ---------------------------------
#
# Objetivo:
# Generar las explicaciones individuales del modelo oficial GraphSAGE
# utilizando el Explainer configurado para todos los nodos
# seleccionados para el análisis.
#
# Entradas:
# - global_explainability
# - graphs
#
# Producto:
# - global_explainability actualizado
#
# Responde:
# ¿Se generaron correctamente las explicaciones individuales del modelo?

def generate_node_explanations(
    global_explainability: dict,
    graphs: list
) -> dict:
    """
    Genera las explicaciones individuales del modelo GraphSAGE para cada
    nodo identificado durante la etapa de construcción de objetivos.

    Parameters
    ----------
    global_explainability : dict
        Contexto oficial de la explicabilidad.

    graphs : list
        Colección oficial de GraphData.

    Returns
    -------
    dict
        Contexto oficial de explicabilidad actualizado.
    """

    # Validación -------------------------------------------------------------
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    if not isinstance(graphs, list):
        raise TypeError(
            "graphs debe ser una lista."
        )

    required_keys = [
        "explainer",
        "explanation_targets"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in global_explainability
    ]

    if missing_keys:
        raise RuntimeError(
            "La estructura de explicabilidad está incompleta: "
            f"{missing_keys}"
        )

    explainer = global_explainability["explainer"]

    if explainer is None:
        raise RuntimeError(
            "El Explainer no ha sido inicializado."
        )

    # Construcción -----------------------------------------------------------
    node_explanations = []

    for target in global_explainability["explanation_targets"]:

        graph = graphs[target["graph_index"]]

        explanation = explainer(
            x=graph.x,
            edge_index=graph.edge_index,
            index=target["node_index"]
        )

        node_explanations.append({
            "graph_index": target["graph_index"],
            "node_index": target["node_index"],
            "explanation": explanation
        })

    global_explainability["node_explanations"] = (
        node_explanations
    )

    global_explainability["explained_nodes"] = (
        len(node_explanations)
    )

    global_explainability["status"] = (
        "EXPLANATIONS_GENERATED"
    )

    # Retorno ----------------------------------------------------------------
    return global_explainability

# BLOQUE 5.3. Validación y Extracción de Máscaras ------------------------------
#
# Objetivo:
# Validar las explicaciones generadas por el Explainer oficial y extraer las
# máscaras de atributos y conexiones cuando estén disponibles.
#
# Entradas:
# - global_explainability
#
# Producto:
# - global_explainability actualizado
#
# Responde:
# ¿Las explicaciones son válidas y contienen las máscaras necesarias?
def extract_explanation_masks(
    global_explainability: dict
) -> dict:
    """
    Valida las explicaciones individuales y extrae las máscaras generadas
    por PyTorch Geometric.

    Parameters
    ----------
    global_explainability : dict
        Contexto oficial de la explicabilidad.

    Returns
    -------
    dict
        Contexto oficial de explicabilidad actualizado.
    """

    # Validación -------------------------------------------------------------
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    required_keys = [
        "node_explanations"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in global_explainability
    ]

    if missing_keys:
        raise RuntimeError(
            "La estructura de explicabilidad está incompleta: "
            f"{missing_keys}"
        )

    # Construcción -----------------------------------------------------------
    validated_explanations = []

    for item in global_explainability["node_explanations"]:

        explanation = item["explanation"]

        explanation.validate()

        node_mask = getattr(
            explanation,
            "node_mask",
            None
        )

        edge_mask = getattr(
            explanation,
            "edge_mask",
            None
        )

        validated_explanations.append({

            "graph_index": item["graph_index"],
            "node_index": item["node_index"],
            "node_mask": node_mask,
            "edge_mask": edge_mask

        })

    global_explainability["validated_explanations"] = (
        validated_explanations
    )

    global_explainability["status"] = (
        "MASKS_EXTRACTED"
    )

    # Retorno ----------------------------------------------------------------
    return global_explainability

# BLOQUE 5.4. Agregación de la Importancia de Variables ------------------------
#
# Objetivo:
# Calcular la importancia global de las variables a partir de las máscaras
# generadas por GNNExplainer para cada nodo explicado.
#
# Entradas:
# - global_explainability
#
# Producto:
# - global_explainability actualizado
#
# Responde:
# ¿Cuál es la importancia global de cada variable del modelo?
def aggregate_feature_importance(
    global_explainability: dict
) -> dict:
    """
    Calcula la importancia global de las variables a partir de las
    explicaciones individuales generadas por GNNExplainer.

    Parameters
    ----------
    global_explainability : dict
        Contexto oficial de la explicabilidad.

    Returns
    -------
    dict
        Contexto oficial de explicabilidad actualizado.
    """

    # Validación -------------------------------------------------------------
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    required_keys = [
        "validated_explanations"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in global_explainability
    ]

    if missing_keys:
        raise RuntimeError(
            "La estructura de explicabilidad está incompleta: "
            f"{missing_keys}"
        )

    # Construcción -----------------------------------------------------------
    feature_masks = []

    for item in global_explainability["validated_explanations"]:

        node_mask = item["node_mask"]

        if node_mask is not None:

            print("-" * 80)
            print(
                "Nodo:",
                item["node_index"]
            )

            print(
                "Shape:",
                tuple(node_mask.shape)
            )

            print(
                "ndim:",
                node_mask.ndim
            )
            print("-" * 80)

        if node_mask is None:
            continue

        if node_mask.ndim == 1:

            feature_masks.append(
                node_mask.cpu().numpy()
            )

        elif node_mask.ndim == 2:

            feature_masks.append(
                node_mask.mean(
                    dim=0
                ).cpu().numpy()
            )

        else:

            raise RuntimeError(
                "Dimensión no soportada para node_mask: "
                f"{tuple(node_mask.shape)}"
            )

    if len(feature_masks) == 0:
        raise RuntimeError(
            "No fue posible calcular la importancia de "
            "variables porque no existen máscaras válidas."
        )

    feature_masks = np.asarray(
        feature_masks
    )

    print("-" * 80)
    print("DEBUG aggregate_feature_importance")
    print("feature_masks.shape:", feature_masks.shape)
    print("-" * 80)

    feature_importance = feature_masks.mean(
        axis=0
    )

    feature_importance_std = feature_masks.std(
        axis=0
    )

    print("-" * 80)
    print("feature_importance.shape:", feature_importance.shape)
    print("feature_importance_std.shape:", feature_importance_std.shape)
    print("-" * 80)

    feature_ranking = np.argsort(
        feature_importance
    )[::-1]

    global_explainability["feature_importance"] = (
        feature_importance
    )

    global_explainability["feature_importance_std"] = (
        feature_importance_std
    )

    global_explainability["feature_ranking"] = (
        feature_ranking.tolist()
    )

    global_explainability["status"] = (
        "FEATURE_IMPORTANCE_COMPUTED"
    )

    # Retorno ----------------------------------------------------------------
    return global_explainability

# BLOQUE 5.5. Consolidación del Contexto de Explicabilidad ---------------------
#
# Objetivo:
# Consolidar los resultados del proceso de explicabilidad y actualizar el
# contexto oficial del proyecto.
#
# Entradas:
# - global_explainability
#
# Producto:
# - global_explainability actualizado
#
# Responde:
# ¿El contexto de explicabilidad quedó listo para ser utilizado?
def finalize_explainability(
    global_explainability: dict
) -> dict:
    """
    Consolida el contexto oficial de explicabilidad y registra la
    información final del proceso.

    Parameters
    ----------
    global_explainability : dict
        Contexto oficial de la explicabilidad.

    Returns
    -------
    dict
        Contexto oficial de explicabilidad actualizado.
    """

    # Validación -------------------------------------------------------------
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    required_keys = [
        "method",
        "explainer",
        "feature_importance",
        "feature_ranking"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in global_explainability
    ]

    if missing_keys:
        raise RuntimeError(
            "La estructura de explicabilidad está incompleta: "
            f"{missing_keys}"
        )

    # Construcción -----------------------------------------------------------
    global_explainability["status"] = (
        "EXPLAINABILITY_COMPLETED"
    )

    # Retorno ----------------------------------------------------------------
    return global_explainability

# BLOQUE 6. Construcción del Ranking de Variables ------------------------------
#
# Objetivo:
# Construir el ranking oficial de importancia de variables a partir de los
# resultados de explicabilidad del modelo oficial GraphSAGE, identificando
# las variables con mayor influencia sobre las predicciones.
#
# Entradas:
# - global_explainability
# - feature_names
#
# Producto:
# - global_explainability actualizado
#
# Responde:
# ¿Cuáles son las variables con mayor influencia sobre el comportamiento del
# modelo oficial?
def build_feature_ranking(
    global_explainability: dict,
    feature_names: list
) -> dict:
    """
    Construye el ranking oficial de importancia de variables.

    Parameters
    ----------
    global_explainability : dict
        Contexto oficial de la explicabilidad.

    feature_names : list
        Lista oficial de nombres de las variables del modelo.

    Returns
    -------
    dict
        Contexto oficial actualizado.
    """

    # Validación -------------------------------------------------------------
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    required_keys = [
        "feature_importance",
        "feature_importance_std"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in global_explainability
    ]

    if missing_keys:
        raise RuntimeError(
            "La estructura de explicabilidad está incompleta: "
            f"{missing_keys}"
        )

    if not isinstance(feature_names, list):
        raise TypeError(
            "feature_names debe ser una lista."
        )

    if len(feature_names) == 0:
        raise RuntimeError(
            "feature_names está vacío."
        )

    # Recuperación -----------------------------------------------------------
    feature_importance = (
        global_explainability["feature_importance"]
    )

    feature_importance_std = (
        global_explainability["feature_importance_std"]
    )

    print("-" * 80)
    print("DEBUG build_feature_ranking")

    print("feature_names:", len(feature_names))
    print("feature_importance:", len(feature_importance))
    print("feature_importance_std:", len(feature_importance_std))

    print("type(feature_importance):", type(feature_importance))

    if hasattr(feature_importance, "shape"):
        print("shape(feature_importance):", feature_importance.shape)

    if hasattr(feature_importance_std, "shape"):
        print("shape(feature_importance_std):", feature_importance_std.shape)

    print("-" * 80)

    if feature_importance is None:
        raise RuntimeError(
            "No existe información de importancia de variables."
        )

    if feature_importance_std is None:
        raise RuntimeError(
            "No existe la desviación estándar de la importancia."
        )

    print("-" * 80)
    print("Número de feature_names:", len(feature_names))
    print("Número de feature_importance:", len(feature_importance))
    print("Número de feature_importance_std:", len(feature_importance_std))
    print("-" * 80)

    if len(feature_names) != len(feature_importance):
        raise RuntimeError(
            "El número de variables no coincide con la "
            "importancia calculada."
        )

    # Construcción -----------------------------------------------------------
    feature_ranking = pd.DataFrame({

        "variable": feature_names,

        "importance": feature_importance,

        "std": feature_importance_std

    })

    feature_ranking = feature_ranking.sort_values(

        by="importance",

        ascending=False,

        ignore_index=True

    )

    if feature_ranking.empty:
        raise RuntimeError(
            "No fue posible construir el ranking oficial."
        )

    global_explainability["feature_ranking"] = (
        feature_ranking
    )

    global_explainability["top_5_variables"] = (
        feature_ranking.head(5).copy()
    )

    global_explainability["top_10_variables"] = (
        feature_ranking.head(10).copy()
    )

    global_explainability["top_20_variables"] = (
        feature_ranking.head(20).copy()
    )

    global_explainability["n_features"] = (
        len(feature_ranking)
    )

    global_explainability["status"] = (
        "FEATURE_RANKING_BUILT"
    )

    # Retorno ----------------------------------------------------------------
    return global_explainability

# BLOQUE 7. Generación de Visualizaciones de Explicabilidad --------------------
#
# Objetivo:
# Generar las visualizaciones oficiales del proceso de explicabilidad del
# modelo GraphSAGE, facilitando la interpretación científica de la
# importancia de las variables.
#
# Entradas:
# - global_explainability
#
# Producto:
# - global_explainability actualizado
#
# Responde:
# ¿Se generaron correctamente las visualizaciones oficiales del modelo?
def generate_explainability_plots(
    global_explainability: dict
) -> dict:
    """
    Genera las visualizaciones oficiales de la explicabilidad.

    Parameters
    ----------
    global_explainability : dict
        Contexto oficial de la explicabilidad.

    Returns
    -------
    dict
        Contexto oficial actualizado.
    """

    # Validación -------------------------------------------------------------
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    required_keys = [
        "feature_ranking",
        "plots_dir"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in global_explainability
    ]

    if missing_keys:
        raise RuntimeError(
            "La estructura oficial está incompleta: "
            f"{missing_keys}"
        )

    # Recuperación -----------------------------------------------------------
    feature_ranking = global_explainability[
        "feature_ranking"
    ]

    plots_dir = global_explainability[
        "plots_dir"
    ]

    if not isinstance(plots_dir, Path):
        raise RuntimeError(
            "plots_dir debe ser un objeto Path."
        )

    if feature_ranking.empty:
        raise RuntimeError(
            "El ranking de variables está vacío."
        )

    if "std" not in feature_ranking.columns:
        raise RuntimeError(
            "No existe la columna 'std' en el ranking."
        )

    # Construcción -----------------------------------------------------------
    plots = {}

    top20 = feature_ranking.head(20)

    plt.figure(figsize=(12, 8))

    plt.barh(
        top20["variable"],
        top20["importance"],
        xerr=top20["std"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Importancia")

    plt.ylabel("Variable")

    plt.title(
        "Importancia Global de Variables"
    )

    plt.tight_layout()

    ranking_path = (
        plots_dir /
        "feature_importance_ranking.png"
    )

    plt.savefig(
        ranking_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    plots["feature_importance"] = ranking_path

    top10 = feature_ranking.head(10)

    plt.figure(figsize=(10, 6))

    plt.bar(
        top10["variable"],
        top10["importance"],
        yerr=top10["std"]
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.ylabel(
        "Importancia"
    )

    plt.title(
        "Top 10 Variables Más Importantes"
    )

    plt.tight_layout()

    top10_path = (
        plots_dir /
        "top10_variables.png"
    )

    plt.savefig(
        top10_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    plots["top10"] = top10_path

    plt.figure(figsize=(12, 6))

    plt.errorbar(
        x=top20["variable"],
        y=top20["importance"],
        yerr=top20["std"],
        fmt="o"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.ylabel(
        "Importancia"
    )

    plt.title(
        "Importancia de Variables con Incertidumbre"
    )

    plt.tight_layout()

    uncertainty_path = (
        plots_dir /
        "feature_importance_uncertainty.png"
    )

    plt.savefig(
        uncertainty_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    plots["uncertainty"] = uncertainty_path

    if not plots:
        raise RuntimeError(
            "No fue posible generar las visualizaciones."
        )

    global_explainability["plots"] = plots

    global_explainability["status"] = (
        "PLOTS_GENERATED"
    )

    # Retorno ----------------------------------------------------------------
    return global_explainability

# BLOQUE 8. Construcción del Resumen Científico -------------------------------
#
# Objetivo:
# Elaborar el resumen científico oficial del proceso de explicabilidad,
# sintetizando las principales variables influyentes y proporcionando una
# interpretación reproducible del comportamiento del modelo oficial GraphSAGE.
#
# Entradas:
# - global_explainability
#
# Producto:
# - global_explainability actualizado
#
# Responde:
# ¿Cuál es la interpretación científica oficial del modelo?
def build_scientific_summary(
    global_explainability: dict
) -> dict:
    """
    Construye el resumen científico oficial de la explicabilidad.

    Parameters
    ----------
    global_explainability : dict
        Contexto oficial de la explicabilidad.

    Returns
    -------
    dict
        Contexto oficial actualizado.
    """

    # Validación -------------------------------------------------------------
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    required_keys = [
        "model_code",
        "model_name",
        "model_family",
        "method",
        "feature_ranking"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in global_explainability
    ]

    if missing_keys:
        raise RuntimeError(
            "La estructura oficial de explicabilidad está incompleta: "
            f"{missing_keys}"
        )

    # Recuperación -----------------------------------------------------------
    model_code = global_explainability[
        "model_code"
    ]

    model_name = global_explainability[
        "model_name"
    ]

    model_family = global_explainability[
        "model_family"
    ]

    method = global_explainability[
        "method"
    ]

    feature_ranking = global_explainability[
        "feature_ranking"
    ]

    if feature_ranking is None:
        raise RuntimeError(
            "No existe ranking de variables."
        )

    if feature_ranking.empty:
        raise RuntimeError(
            "El ranking de variables está vacío."
        )

    # Construcción -----------------------------------------------------------
    top_variables = feature_ranking.head(5)

    summary = []

    summary.append(
        "RESUMEN CIENTÍFICO DE EXPLICABILIDAD"
    )

    summary.append("=" * 70)

    summary.append("")

    summary.append(
        f"Código del modelo : {model_code}"
    )

    summary.append(
        f"Modelo oficial    : {model_name}"
    )

    summary.append(
        f"Familia           : {model_family}"
    )

    summary.append(
        f"Método XAI        : {method}"
    )

    summary.append("")

    summary.append(
        "Top 5 variables con mayor importancia:"
    )

    summary.append("")

    for index, row in top_variables.iterrows():

        summary.append(

            f"{index + 1}. "
            f"{row['variable']} "
            f"(Importancia = {row['importance']:.6f}, "
            f"Std = {row['std']:.6f})"

        )

    summary.append("")

    summary.append(
        "Interpretación científica:"
    )

    summary.append(

        f"El modelo oficial {model_name}, perteneciente a la familia "
        f"{model_family}, fue interpretado mediante {method}. "
        "Las variables con mayor importancia corresponden a aquellas que "
        "presentan la mayor contribución relativa durante el proceso "
        "predictivo del modelo sobre el conjunto de evaluación. "
        "La importancia fue obtenida mediante técnicas de Inteligencia "
        "Artificial Explicable (XAI) y representa una medida de influencia "
        "sobre las predicciones del modelo. Estos resultados no deben "
        "interpretarse como evidencia de relaciones causales entre las "
        "variables y el fenómeno estudiado."

    )

    scientific_summary = "\n".join(summary)

    if not scientific_summary.strip():
        raise RuntimeError(
            "No fue posible construir el resumen científico."
        )

    global_explainability[
        "scientific_summary"
    ] = scientific_summary

    global_explainability[
        "status"
    ] = "SCIENTIFIC_SUMMARY_BUILT"

    # Retorno ----------------------------------------------------------------
    return global_explainability

# BLOQUE 9. Exportación de los Resultados de Explicabilidad -------------------
#
# Objetivo:
# Exportar los resultados oficiales del proceso de explicabilidad,
# garantizando la trazabilidad, reproducibilidad y disponibilidad de la
# información para informes científicos, auditorías y procesos posteriores.
#
# Entradas:
# - global_explainability
#
# Producto:
# - global_explainability actualizado
#
# Responde:
# ¿Fueron exportados correctamente los resultados oficiales del proceso de
# explicabilidad?
def export_explainability_results(
    global_explainability: dict
) -> dict:
    """
    Exporta los resultados oficiales de la explicabilidad.

    Parameters
    ----------
    global_explainability : dict
        Contexto oficial de la explicabilidad.

    Returns
    -------
    dict
        Contexto oficial actualizado.
    """

    # Validación -------------------------------------------------------------
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    required_keys = [
        "model_code",
        "model_name",
        "model_family",
        "method",
        "configuration",
        "created_at",
        "plots_dir",
        "plots",
        "feature_ranking",
        "top_5_variables",
        "top_10_variables",
        "top_20_variables",
        "scientific_summary",
        "execution_time"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in global_explainability
    ]

    if missing_keys:
        raise RuntimeError(
            "La estructura oficial está incompleta: "
            f"{missing_keys}"
        )

    # Recuperación -----------------------------------------------------------
    export_dir = global_explainability["plots_dir"]

    feature_ranking = global_explainability[
        "feature_ranking"
    ]

    scientific_summary = global_explainability[
        "scientific_summary"
    ]

    plots = global_explainability[
        "plots"
    ]

    if not isinstance(export_dir, Path):
        raise RuntimeError(
            "plots_dir debe ser un objeto Path."
        )

    if feature_ranking.empty:
        raise RuntimeError(
            "El ranking de variables está vacío."
        )

    if not scientific_summary.strip():
        raise RuntimeError(
            "El resumen científico está vacío."
        )

    # Construcción -----------------------------------------------------------
    export_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    exported_files = {}

    csv_file = (
        export_dir /
        "feature_ranking.csv"
    )

    feature_ranking.to_csv(
        csv_file,
        index=False,
        encoding="utf-8"
    )

    exported_files[
        "feature_ranking_csv"
    ] = csv_file

    excel_file = (
        export_dir /
        "feature_ranking.xlsx"
    )

    feature_ranking.to_excel(
        excel_file,
        index=False
    )

    exported_files[
        "feature_ranking_excel"
    ] = excel_file

    summary_file = (
        export_dir /
        "scientific_summary.txt"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            scientific_summary
        )

    exported_files[
        "scientific_summary"
    ] = summary_file

    json_file = (
        export_dir /
        "explainability_results.json"
    )

    json_content = {

        "model_code":
            global_explainability["model_code"],

        "model_name":
            global_explainability["model_name"],

        "model_family":
            global_explainability["model_family"],

        "method":
            global_explainability["method"],

        "configuration":
            global_explainability["configuration"],

        "created_at":
            global_explainability["created_at"],

        "execution_time":
            global_explainability["execution_time"],

        "feature_ranking":
            feature_ranking.to_dict(
                orient="records"
            ),

        "top_5_variables":
            global_explainability[
                "top_5_variables"
            ].to_dict(
                orient="records"
            ),

        "top_10_variables":
            global_explainability[
                "top_10_variables"
            ].to_dict(
                orient="records"
            ),

        "top_20_variables":
            global_explainability[
                "top_20_variables"
            ].to_dict(
                orient="records"
            ),

        "plots": {

            key: str(value)

            for key, value in plots.items()

        },

        "scientific_summary":
            scientific_summary

    }

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_content,
            file,
            indent=4,
            ensure_ascii=False
        )

    exported_files[
        "explainability_json"
    ] = json_file

    # Validación -------------------------------------------------------------
    for exported_file in exported_files.values():

        if not Path(exported_file).exists():

            raise RuntimeError(
                f"No fue posible exportar {exported_file}."
            )

    global_explainability[
        "exported_files"
    ] = exported_files

    global_explainability[
        "n_exported_files"
    ] = len(exported_files)

    global_explainability[
        "exported_at"
    ] = datetime.now().isoformat()

    global_explainability[
        "status"
    ] = "RESULTS_EXPORTED"

    # Retorno ----------------------------------------------------------------
    return global_explainability

# BLOQUE 10. Validación de la Explicabilidad ----------------------------------
#
# Objetivo:
# Verificar que el proceso oficial de explicabilidad haya sido ejecutado
# correctamente y que todos los resultados científicos requeridos estén
# disponibles antes de finalizar la evaluación.
#
# Entradas:
# - global_explainability
#
# Producto:
# - validation_status
#
# Responde:
# ¿El proceso oficial de explicabilidad fue completado correctamente?
def validate_global_explainability(
    global_explainability: dict
) -> bool:
    """
    Valida la estructura oficial de la explicabilidad.

    Parameters
    ----------
    global_explainability : dict
        Contexto oficial de la explicabilidad.

    Returns
    -------
    bool
        True si la validación fue exitosa.
    """

    # Validación -------------------------------------------------------------
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    required_keys = [
        "model_code",
        "model_name",
        "model_family",
        "configuration",
        "created_at",
        "method",
        "explainer",
        "feature_importance",
        "feature_importance_std",
        "feature_ranking",
        "n_features",
        "top_5_variables",
        "top_10_variables",
        "top_20_variables",
        "scientific_summary",
        "plots",
        "execution_time",
        "exported_files",
        "status"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in global_explainability
    ]

    if missing_keys:
        raise RuntimeError(
            "La estructura oficial de explicabilidad está incompleta: "
            f"{missing_keys}"
        )

    # Recuperación -----------------------------------------------------------
    feature_ranking = global_explainability[
        "feature_ranking"
    ]

    scientific_summary = global_explainability[
        "scientific_summary"
    ]

    plots = global_explainability[
        "plots"
    ]

    exported_files = global_explainability[
        "exported_files"
    ]

    # Validación -------------------------------------------------------------
    if global_explainability["status"] != "RESULTS_EXPORTED":
        raise RuntimeError(
            "La exportación oficial no fue completada."
        )

    if feature_ranking.empty:
        raise RuntimeError(
            "El ranking de variables está vacío."
        )

    for key in [
        "top_5_variables",
        "top_10_variables",
        "top_20_variables"
    ]:

        if global_explainability[key].empty:
            raise RuntimeError(
                f"{key} está vacío."
            )

    if not scientific_summary.strip():
        raise RuntimeError(
            "El resumen científico está vacío."
        )

    if not isinstance(plots, dict):
        raise TypeError(
            "plots debe ser un diccionario."
        )

    if not plots:
        raise RuntimeError(
            "No existen figuras generadas."
        )

    for plot_name, plot_file in plots.items():

        if not Path(plot_file).exists():
            raise RuntimeError(
                f"La figura '{plot_name}' no existe: "
                f"{plot_file}"
            )

    if not isinstance(exported_files, dict):
        raise TypeError(
            "exported_files debe ser un diccionario."
        )

    if not exported_files:
        raise RuntimeError(
            "No existen archivos exportados."
        )

    for file_name, exported_file in exported_files.items():

        if not Path(exported_file).exists():
            raise RuntimeError(
                f"El archivo '{file_name}' no existe: "
                f"{exported_file}"
            )

    global_explainability[
        "status"
    ] = "VALIDATED"

    # Retorno ----------------------------------------------------------------
    return True

# BLOQUE 11. Auditoría de la Importancia de Variables -------------------------
#
# Objetivo:
# Auditar la importancia de variables calculada durante el proceso oficial
# de explicabilidad, verificando su consistencia estadística y almacenando
# los resultados dentro del contexto global.
#
# Entradas:
# - global_explainability
#
# Producto:
# - global_explainability actualizado
#
# Responde:
# ¿La importancia de variables cumple los criterios mínimos de calidad?
def audit_feature_importance(
    global_explainability: dict
) -> dict:
    """
    Audita la importancia de variables calculada durante la
    explicabilidad del modelo oficial.

    Parameters
    ----------
    global_explainability : dict
        Contexto oficial de la explicabilidad.

    Returns
    -------
    dict
        Contexto actualizado con la auditoría.
    """

    # Validación -------------------------------------------------------------
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    required_keys = [
        "feature_ranking",
        "feature_importance",
        "feature_importance_std"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in global_explainability
    ]

    if missing_keys:
        raise RuntimeError(
            "La estructura oficial está incompleta: "
            f"{missing_keys}"
        )

    # Recuperación -----------------------------------------------------------
    feature_ranking = global_explainability[
        "feature_ranking"
    ]

    # Validación -------------------------------------------------------------
    if feature_ranking.empty:
        raise RuntimeError(
            "El ranking de variables está vacío."
        )

    required_columns = [
        "importance",
        "std"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in feature_ranking.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Faltan columnas en feature_ranking: "
            f"{missing_columns}"
        )

    # Construcción -----------------------------------------------------------
    importance = feature_ranking["importance"]

    uncertainty = feature_ranking["std"]

    top10 = importance.head(
        min(10, len(importance))
    )

    audit = {

        "n_variables":
            int(len(feature_ranking)),

        "importance_sum":
            float(importance.sum()),

        "importance_mean":
            float(importance.mean()),

        "importance_std":
            float(importance.std()),

        "importance_max":
            float(importance.max()),

        "importance_min":
            float(importance.min()),

        "mean_uncertainty":
            float(uncertainty.mean()),

        "max_uncertainty":
            float(uncertainty.max()),

        "positive_variables":
            int((importance > 0).sum()),

        "zero_variables":
            int((importance == 0).sum()),

        "top10_cumulative_importance":
            float(top10.sum())

    }

    audit["passed"] = bool(

        audit["importance_sum"] > 0

        and

        audit["positive_variables"] > 0

    )

    audit["quality"] = (

        "APPROVED"

        if audit["passed"]

        else

        "FAILED"

    )

    global_explainability[
        "importance_audit"
    ] = audit

    global_explainability[
        "status"
    ] = "IMPORTANCE_AUDITED"

    # Retorno ----------------------------------------------------------------
    return global_explainability

# BLOQUE 12. Análisis de Consistencia entre Explicabilidad y Correlación -------
#
# Objetivo:
# Comparar la importancia de variables obtenida mediante el método oficial
# de explicabilidad con la correlación de Spearman respecto a la variable
# objetivo, proporcionando una validación estadística complementaria de los
# resultados del modelo GraphSAGE.
#
# Entradas:
# - global_explainability
# - feature_data
# - target_data
#
# Producto:
# - global_explainability actualizado
#
# Responde:
# ¿Existe consistencia entre la importancia estimada por el modelo y la
# asociación estadística con la variable objetivo?
def analyze_target_correlation(
    global_explainability: dict,
    feature_data: pd.DataFrame,
    target_data
) -> dict:
    """
    Analiza la consistencia entre la importancia de variables obtenida
    mediante explicabilidad y la correlación de Spearman con la variable
    objetivo.

    Parameters
    ----------
    global_explainability : dict
        Contexto oficial de la explicabilidad.

    feature_data : pandas.DataFrame
        Variables predictoras utilizadas durante la evaluación.

    target_data : array-like
        Variable objetivo.

    Returns
    -------
    dict
        Contexto oficial actualizado.
    """

    # Validación -------------------------------------------------------------
    if not isinstance(global_explainability, dict):
        raise TypeError(
            "global_explainability debe ser un diccionario."
        )

    if not isinstance(feature_data, pd.DataFrame):
        raise TypeError(
            "feature_data debe ser un DataFrame."
        )

    if target_data is None:
        raise RuntimeError(
            "target_data no puede ser None."
        )

    required_keys = [
        "feature_ranking"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in global_explainability
    ]

    if missing_keys:
        raise RuntimeError(
            "La estructura oficial está incompleta: "
            f"{missing_keys}"
        )

    # Recuperación -----------------------------------------------------------
    feature_ranking = global_explainability[
        "feature_ranking"
    ]

    # Validación -------------------------------------------------------------
    if feature_ranking.empty:
        raise RuntimeError(
            "El ranking de variables está vacío."
        )

    # Construcción -----------------------------------------------------------
    correlations = []

    target_series = pd.Series(target_data)

    for variable in feature_data.columns:

        rho = feature_data[variable].corr(
            target_series,
            method="spearman"
        )

        if pd.isna(rho):
            rho = 0.0

        correlations.append({

            "variable": variable,

            "spearman": float(rho),

            "abs_spearman": float(abs(rho))

        })

    correlation_df = pd.DataFrame(
        correlations
    )

    comparison = feature_ranking.merge(

        correlation_df,

        on="variable",

        how="left"

    )

    comparison["importance_rank"] = (

        comparison["importance"]

        .rank(
            ascending=False,
            method="dense"
        )

        .astype(int)

    )

    comparison["correlation_rank"] = (

        comparison["abs_spearman"]

        .rank(
            ascending=False,
            method="dense"
        )

        .astype(int)

    )

    comparison["rank_difference"] = (

        comparison["importance_rank"]

        - comparison["correlation_rank"]

    ).abs()

    comparison["consistency"] = np.where(

        comparison["rank_difference"] <= 3,

        "HIGH",

        np.where(

            comparison["rank_difference"] <= 10,

            "MEDIUM",

            "LOW"

        )

    )

    rho_global, p_value = spearmanr(

        comparison["importance"],

        comparison["abs_spearman"]

    )

    if pd.isna(rho_global):
        rho_global = 0.0

    if pd.isna(p_value):
        p_value = 1.0

    consistency_summary = {

        "high_consistency":
            int((comparison["consistency"] == "HIGH").sum()),

        "medium_consistency":
            int((comparison["consistency"] == "MEDIUM").sum()),

        "low_consistency":
            int((comparison["consistency"] == "LOW").sum()),

        "mean_rank_difference":
            float(comparison["rank_difference"].mean()),

        "global_spearman":
            float(rho_global),

        "global_pvalue":
            float(p_value)

    }

    comparison = comparison.sort_values(

        by="importance",

        ascending=False,

        ignore_index=True

    )

    global_explainability[
        "target_correlation"
    ] = comparison

    global_explainability[
        "target_correlation_summary"
    ] = consistency_summary

    global_explainability[
        "status"
    ] = "TARGET_CORRELATION_ANALYZED"

    # Retorno ----------------------------------------------------------------
    return global_explainability