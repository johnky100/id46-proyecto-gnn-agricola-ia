# prepare_year_dataset.py

# PREPARE_YEAR_DATASET.PY
# Objetivo: Preparar el Dataset Científico correspondiente a un año específico,
# validando su integridad espacial y organizándolo para la construcción del
# GraphData oficial.
# Arquitectura científica
# Entradas: Dataset Científico completo, año de procesamiento y Catálogo
# Oficial de Nodos.
# Producto: Dataset Científico Anual validado, ordenado e indexado mediante node_idx.
# Pregunta científica: ¿El Dataset Científico Anual posee la estructura
# requerida para construir el GraphData oficial?

# 1. IMPORTACIÓN DE DEPENDENCIAS

import pandas as pd
import geopandas as gpd

from src.python.config.config_project import (
    TIME_COLUMN,
    MUNICIPALITY_ID_COLUMN,
    GRAPH_NODE_KEY_COLUMNS,
    NODE_INDEX_COLUMN,
)

# 2. BUILDER OFICIAL DEL DATASET CIENTÍFICO ANUAL

def prepare_year_dataset(
    dataset: pd.DataFrame,
    current_year: int,
    node_catalog: gpd.GeoDataFrame,
) -> pd.DataFrame:

    """
    Prepara el Dataset Científico correspondiente a un año específico,
    validando su integridad espacial e incorporando el índice interno
    del grafo utilizado por PyTorch Geometric.
    """

    print("-" * 80)
    print(f"Preparación del Dataset Científico Anual - {current_year}")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # Filtrado del Dataset Científico
    # -------------------------------------------------------------------------
    dataset_year = (
        dataset.loc[
            dataset[TIME_COLUMN] == current_year
        ]
        .copy()
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Validación del Dataset Anual
    # -------------------------------------------------------------------------
    if dataset_year.empty:
        raise ValueError(
            f"No existen registros para el año {current_year}."
        )

    # -------------------------------------------------------------------------
    # Validación de cobertura espacial
    # -------------------------------------------------------------------------
    num_dataset_nodes = dataset_year[
        MUNICIPALITY_ID_COLUMN
    ].nunique()

    if num_dataset_nodes != node_catalog[
        MUNICIPALITY_ID_COLUMN
    ].nunique():
        raise ValueError(
            "La cobertura espacial del Dataset Científico Anual no coincide "
            "con el Catálogo Oficial de Nodos."
        )

    # -------------------------------------------------------------------------
    # Validación de duplicados científicos
    # -------------------------------------------------------------------------
    duplicated_records = dataset_year.duplicated(
        subset=GRAPH_NODE_KEY_COLUMNS
    ).sum()

    if duplicated_records > 0:
        raise ValueError(
            f"Se encontraron {duplicated_records:,} registros "
            "duplicados para la clave científica."
        )

    # -------------------------------------------------------------------------
    # Incorporación del índice interno del grafo
    # -------------------------------------------------------------------------
    dataset_year = (
        dataset_year
        .merge(
            node_catalog[
                [
                    MUNICIPALITY_ID_COLUMN,
                    NODE_INDEX_COLUMN,
                ]
            ],
            on=MUNICIPALITY_ID_COLUMN,
            how="left",
            validate="many_to_one",
        )
    ) # Incorporar el índice interno del grafo

    if dataset_year[NODE_INDEX_COLUMN].isnull().any():
        raise ValueError(
            "No fue posible asignar el índice interno del grafo a todos los municipios."
        )

    if dataset_year[NODE_INDEX_COLUMN].duplicated().any():
        raise ValueError(
            "Se encontraron índices internos del grafo duplicados."
        )

    expected_node_index = list(range(len(node_catalog))) # Secuencia esperada de índices

    if sorted(dataset_year[NODE_INDEX_COLUMN].tolist()) != expected_node_index:
        raise ValueError(
            "El Dataset Científico Anual no contiene todos los índices internos del grafo."
        )

    # -------------------------------------------------------------------------
    # Ordenamiento científico
    # -------------------------------------------------------------------------
    dataset_year = (
        dataset_year
        .sort_values(NODE_INDEX_COLUMN)
        .reset_index(drop=True)
    ) # Ordenar según el índice interno del grafo

    # -------------------------------------------------------------------------
    # Resumen científico
    # -------------------------------------------------------------------------
    print("Dataset Científico Anual preparado correctamente.")
    print(f"Año                      : {current_year}")
    print(f"Registros                : {len(dataset_year):,}")
    print(f"Municipios               : {num_dataset_nodes:,}")
    print(f"Variables                : {dataset_year.shape[1]:,}")
    print("Estado                    : CORRECTO")
    print(f"Índice interno del grafo : {NODE_INDEX_COLUMN}")

    return dataset_year


