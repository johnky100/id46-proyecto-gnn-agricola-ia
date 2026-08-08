# data_preparation.py

# BLOQUE 1. Importaciones -----------------------------------------------------
## Objetivo: Importar las dependencias necesarias para la carga, validación y
## preparación del Dataset Científico utilizado por el pipeline del proyecto.

import warnings # Emisión de advertencias durante la validación
import pandas as pd # Manipulación del Dataset Científico

from config.config_project import (
    N_PANEL_OBSERVATIONS,
    PANEL_ID_COLUMN,
    STRUCTURAL_COLUMNS,
    TARGET_VARIABLE,
    FEATURE_COLUMNS
)

from config.paths import DATASET_FILE # Ruta oficial del Dataset Científico

# BLOQUE 2. Carga del Dataset Científico --------------------------------------
## Objetivo: Cargar el Dataset Científico oficial del proyecto y verificar la
## existencia del archivo antes de su lectura.

def load_dataset() -> pd.DataFrame:
    """
    Carga el Dataset Científico oficial del proyecto.

    Returns
    -------
    pd.DataFrame
        Dataset Científico cargado desde el archivo oficial.
    """

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el Dataset Científico:\n{DATASET_FILE}"
        ) # Validar existencia del Dataset Científico

    return pd.read_parquet(DATASET_FILE) # Cargar Dataset Científico

# BLOQUE 3. Validación del Dataset Científico -------------------------------
## Objetivo: Validar la estructura mínima del Dataset Científico antes de su
## utilización en las siguientes fases del pipeline.

def validate_dataset(
    dataset: pd.DataFrame
) -> pd.DataFrame:
    """
    Valida la estructura mínima del Dataset Científico.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset Científico a validar.

    Returns
    -------
    pd.DataFrame
        Dataset validado.
    """

    if dataset.empty:
        raise ValueError(
            "El Dataset Científico está vacío."
        )  # Validar observaciones

    if dataset.columns.empty:
        raise ValueError(
            "El Dataset Científico no contiene variables."
        )  # Validar variables

    required_columns = (
        STRUCTURAL_COLUMNS
        + [TARGET_VARIABLE]
    )  # Columnas obligatorias del Dataset Científico

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]  # Identificar columnas faltantes

    if missing_columns:

        missing = ", ".join(missing_columns)

        raise ValueError(
            f"Faltan las siguientes columnas obligatorias: {missing}."
        )  # Validar estructura del Dataset Científico

    if len(dataset) != N_PANEL_OBSERVATIONS:

        warnings.warn(
            (
                f"Se esperaban {N_PANEL_OBSERVATIONS:,} observaciones y "
                f"se encontraron {len(dataset):,}."
            ),
            UserWarning
        )  # Informar diferencias en el tamaño del panel

    if dataset[PANEL_ID_COLUMN].duplicated().any():
        raise ValueError(
            "Se encontraron registros duplicados del panel municipio-año."
        )  # Validar unicidad del panel

    if dataset[required_columns].isnull().any().any():
        raise ValueError(
            "Existen valores faltantes en las columnas obligatorias."
        )  # Validar valores faltantes

    if dataset[TARGET_VARIABLE].isnull().all():
        raise ValueError(
            f"La variable objetivo '{TARGET_VARIABLE}' no contiene datos."
        )  # Validar variable objetivo

    return dataset

# BLOQUE 4. Preparación del Dataset Científico ------------------------------
## Objetivo: Centralizar la carga y validación del Dataset Científico para
## entregar una única interfaz de acceso al resto del pipeline.

def prepare_dataset() -> pd.DataFrame:
    """
    Carga y valida el Dataset Científico oficial del proyecto.

    Returns
    -------
    pd.DataFrame
        Dataset Científico preparado y listo para ser utilizado
        por las siguientes fases del pipeline.
    """

    dataset = load_dataset()  # Cargar Dataset Científico

    dataset = validate_dataset(
        dataset
    )  # Validar Dataset Científico

    return dataset

# BLOQUE 5. Preparación de Variables Tabulares ------------------------------
## Objetivo: Construir las variables predictoras y la variable objetivo
## utilizadas por los modelos tabulares del Benchmark Científico.
#### Entradas:
#### Producto:
#### Responde: ¿Las variables tabulares fueron preparadas correctamente?

def prepare_tabular_features(
    dataset: pd.DataFrame,
    target_variable: str,
    excluded_columns: list
) -> dict:
    """
    Construye las variables predictoras y la variable objetivo para
    los modelos tabulares del proyecto.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset Científico validado.

    target_variable : str
        Variable objetivo del proyecto.

    excluded_columns : list
        Columnas que no deben utilizarse como variables predictoras.

    Returns
    -------
    dict
        Diccionario con las variables preparadas para la inferencia.
    """

    # Validación ------------------------------------------------------------
    if dataset is None:
        raise ValueError(
            "El Dataset Científico no puede ser None."
        )

    if dataset.empty:
        raise ValueError(
            "El Dataset Científico está vacío."
        )

    if target_variable not in dataset.columns:
        raise ValueError(
            f"La variable objetivo '{target_variable}' no existe."
        )

    # Construcción ----------------------------------------------------------
    feature_columns = FEATURE_COLUMNS # Variables predictoras oficiales del Dataset Científico

    if len(feature_columns) == 0:
        raise ValueError(
            "No existen variables predictoras disponibles."
        )

    x_data = dataset[
        feature_columns
    ].copy() # Variables predictoras oficiales

    y_true = dataset[
        target_variable
    ].copy() # Variable objetivo

    # Construcción del resultado -------------------------------------------
    tabular_features = {

        "x_data": x_data,

        "y_true": y_true,

        "feature_columns": feature_columns,

        "n_features": len(feature_columns)

    }

    # Validación ------------------------------------------------------------
    required_keys = [

        "x_data",

        "y_true",

        "feature_columns",

        "n_features"

    ]

    missing_keys = [

        key
        for key in required_keys
        if key not in tabular_features

    ]

    if missing_keys:

        raise RuntimeError(
            f"La preparación tabular está incompleta: {missing_keys}"
        )

    # Retorno ---------------------------------------------------------------
    return tabular_features