# build_features.py

# =============================================================================
# BLOQUE 1. INFORMACIÓN DEL MÓDULO
# =============================================================================
# Objetivo: Definir la identidad, versión y propósito científico del módulo responsable de construir la 
# matriz oficial de características (feature_matrix) utilizada por el GraphData del proyecto.
# Arquitectura: Este módulo construye la feature_matrix oficial a partir del Dataset Científico, garantizando
# una representación consistente, reproducible y compatible con Graph Neural Networks (GNN).
# Producto: Matriz oficial de características (feature_matrix) del GraphData.
# Pregunta Científica: ¿Cómo construir una matriz oficial de características consistente, reproducible y 
# compatible con Graph Neural Networks a partir Dataset Científico?
# =============================================================================

# =============================================================================
# 1.1 MODULE_INFO
# =============================================================================
MODULE_INFO = {
    # Nombre oficial del módulo
    "name": "Feature Builder",

    # Versión del software
    "version": "1.0.0",

    # Versión de la especificación científica
    "feature_specification_version": "1.0",

    # Descripción del módulo
    "description": (
        "Builds the official feature matrix (X) and target vector (y) from the yearly scientific dataset. "
    ),

    # Nombre del módulo
    "module": "build_features",

    # Pipeline al que pertenece
    "pipeline": "Graph Construction",

    # Autor del proyecto
    "author": "AVANZADO-IA",
}

# =============================================================================
# BLOQUE 2. IMPORTACIONES
# =============================================================================
# Objetivo: Importar las librerías y configuraciones necesarias para construir la matriz oficial de
# características (feature_matrix) del GraphData.
# Arquitectura: Este bloque organiza las dependencias del módulo en categorías funcionales, incluyendo
# librerías científicas de Python, configuraciones oficiales del proyecto y parámetros del grafo,
# garantizando una estructura modular, legible y mantenible.
# Producto: Dependencias necesarias para construir la feature_matrix.
# Pregunta Científica: ¿Qué dependencias son necesarias para construir la matriz oficial
# de características del GraphData?
# =============================================================================

# =============================================================================
# 2.1 LIBRERÍAS EXTERNAS
# =============================================================================
import pandas as pd
import torch

# =============================================================================
# 2.2 CONFIGURACIÓN DEL PROYECTO
# =============================================================================
from src.python.config.config_project import (
    FEATURE_COLUMNS,
    MUNICIPALITY_ID_COLUMN,
    TARGET_VARIABLE,
    TIME_COLUMN,
    TORCH_FLOAT_DTYPE,
)

# =============================================================================
# BLOQUE 3. CONFIGURACIÓN DEL BUILDER
# =============================================================================
# Objetivo: Definir la configuración general utilizada durante la construcción de la matriz oficial de 
# características (feature_matrix).
# Arquitectura: Este bloque establece las opciones que controlan el proceso de construcción de la
# feature_matrix, incluyendo la validación de las entradas, la transformación de las variables y el manejo 
# estricto de errores para garantizar la integridad del producto construido.
# Producto: Configuración oficial del proceso de construcción de la feature_matrix.
# Pregunta Científica: ¿Qué criterios de construcción deben aplicarse para garantizar una feature_matrix
# consistente, reproducible y compatible con Graph Neural Networks?
# =============================================================================

# =============================================================================
# 3.1 BUILD_CONFIG
# =============================================================================
BUILD_CONFIG = {
    # Validar las entradas antes de construir la feature_matrix
    "validate_inputs": True,

    # Normalizar las variables oficiales
    "normalize_features": True,

    # Ejecutar validaciones estrictas durante la construcción
    "strict_mode": True,
}

# =============================================================================
# BLOQUE 4. FUNCIÓN build_features()
# =============================================================================
# Objetivo: Construir la matriz oficial de características (feature_matrix) y la variable objetivo (target_vector)
# a partir del Dataset Científico Anual, garantizando una representación consistente,
# reproducible y compatible con Graph Neural Networks (GNN).
# Arquitectura: 4.1 Validación del Dataset Científico 4.2 Selección Oficial de Variables
# 4.3 Transformación de Features 4.4 Construcción de la Feature Matrix
# 4.5 Construcción de la Variable Objetivo 4.6 Validación Feature Matrix
# 4.7 Validación Target 4.8 Retorno
# Producto: Feature Matrix (X) Target Vector (y).
# Pregunta Científica: ¿Cómo construir de forma consistente la matriz de características (X) y la variable
# objetivo (y) a partir del Dataset Científico Anual para su utilización en Graph Neural Networks?
# =============================================================================

def build_features(
    dataset_year: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor]:

    # =========================================================================
    # 4.1 Validación del Dataset Científico
    # =========================================================================
    if BUILD_CONFIG["validate_inputs"]:

        if not isinstance(dataset_year, pd.DataFrame):
            raise TypeError(
                "The Scientific Dataset must be a pandas DataFrame."
            )

        if dataset_year.empty:
            raise ValueError(
                "The Scientific Dataset is empty."
            )

        required_columns = [
            MUNICIPALITY_ID_COLUMN,
            TIME_COLUMN,
            TARGET_VARIABLE,
            *FEATURE_COLUMNS,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in dataset_year.columns
        ]

        if missing_columns:
            raise ValueError(
                "The Scientific Dataset is missing the following "
                f"required columns: {missing_columns}"
            )

        missing_values = (
            dataset_year[required_columns]
            .isnull()
            .sum()
        )

        if missing_values.any():
            raise ValueError(
                "The Scientific Dataset contains missing values "
                f"in required columns: "
                f"{missing_values[missing_values > 0].to_dict()}"
            )

        duplicate_rows = dataset_year.duplicated(
            subset=[
                MUNICIPALITY_ID_COLUMN,
                TIME_COLUMN,
            ]
        )

        if duplicate_rows.any():
            raise ValueError(
                "The Scientific Dataset contains duplicated "
                "municipality-year records."
            )
    
    # =========================================================================
    # 4.2 Selección Oficial de Variables
    # =========================================================================
    selected_columns = [
        MUNICIPALITY_ID_COLUMN,
        TIME_COLUMN,
        *FEATURE_COLUMNS,
    ]

    feature_data = dataset_year.loc[
        :,
        selected_columns,
    ].copy()

    if feature_data.empty:
        raise ValueError(
            "The official feature selection produced an empty dataset."
        )

    # =========================================================================
    # 4.3 Transformación de Features
    # =========================================================================
    if BUILD_CONFIG["normalize_features"]:
        feature_columns = FEATURE_COLUMNS.copy()

        if feature_columns:
            feature_min = feature_data[feature_columns].min()
            feature_max = feature_data[feature_columns].max()

            feature_range = feature_max - feature_min
            feature_range = feature_range.replace(0, 1)

            feature_data.loc[:, feature_columns] = (
                feature_data[feature_columns] - feature_min
            ) / feature_range

    # =========================================================================
    # 4.4 Construcción de la Feature Matrix
    # =========================================================================
    feature_matrix = torch.tensor(
        feature_data.loc[:, FEATURE_COLUMNS].to_numpy(),
        dtype=TORCH_FLOAT_DTYPE,
    )  

    # =========================================================================
    # 4.5 Construcción de la Variable Objetivo
    # =========================================================================
    target_vector = torch.tensor(
        dataset_year[TARGET_VARIABLE].to_numpy(),
        dtype = TORCH_FLOAT_DTYPE,
    ).view(-1, 1)

    # =========================================================================
    # 4.6 Validación de la Feature Matrix
    # =========================================================================
    if BUILD_CONFIG["validate_inputs"]:

        if feature_matrix.numel() == 0:
            raise ValueError(
                "The feature_matrix is empty."
            )

        if feature_matrix.dtype != TORCH_FLOAT_DTYPE:
            raise TypeError(
                "The feature_matrix has an invalid data type."
            )

        if feature_matrix.ndim != 2:
            raise ValueError(
                "The feature_matrix must be a two-dimensional tensor."
            )

        if torch.isnan(feature_matrix).any():
            raise ValueError(
                "The feature_matrix contains NaN values."
            )

        if torch.isinf(feature_matrix).any():
            raise ValueError(
                "The feature_matrix contains infinite values."
            )


    # =========================================================================
    # 4.7 Validación de la Variable Objetivo
    # =========================================================================
    if BUILD_CONFIG["validate_inputs"]:

        if target_vector.numel() == 0:
            raise ValueError(
                "The target_vector is empty."
            )

        if target_vector.dtype != TORCH_FLOAT_DTYPE:
            raise TypeError(
                "The target_vector has an invalid data type."
            )

        if target_vector.ndim != 2:
            raise ValueError(
                "The target_vector must be a two-dimensional tensor."
            )

        if target_vector.shape[1] != 1:
            raise ValueError(
                "The target_vector must contain a single column."
            )

        if target_vector.shape[0] != feature_matrix.shape[0]:
            raise ValueError(
                "The target_vector does not match the number of nodes."
            )

        if torch.isnan(target_vector).any():
            raise ValueError(
                "The target_vector contains NaN values."
            )

        if torch.isinf(target_vector).any():
            raise ValueError(
                "The target_vector contains infinite values."
            )

    # =========================================================================
    # 4.8 Retorno de la Feature Matrix y Variable Objetivo
    # =========================================================================
    return feature_matrix, target_vector

# =============================================================================
# BLOQUE 5. EXPORTACIONES
# =============================================================================
# Objetivo: Definir la interfaz pública del módulo, indicando los elementos disponibles para ser utilizados
# por otros componentes del proyecto.
# Arquitectura: Este bloque expone únicamente la función oficial encargada de construir la matriz de 
# características (feature_matrix), preservando la encapsulación de la implementación interna.
# Producto: Interfaz pública del módulo build_features.
# Pregunta Científica: ¿Qué elemento oficial debe exponerse para Construir la Feature Matrix y la Variable 
# Objetivo. dentro del pipeline de Graph Construction?
# =============================================================================

# =============================================================================
# EXPORTACIONES
# =============================================================================
__all__ = [
    "build_features",
]