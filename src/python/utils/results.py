# results.py

import numpy as np
from src.python.config.config_project import (
    BENCHMARK_METRICS,
    BENCHMARK_CONFIG,
)

# BLOQUE 1. Construcción del Resultado del Benchmark
## Objetivo: Construir la estructura oficial de resultados utilizada por el
## Benchmark Científico del proyecto.
def build_benchmark_result(
    model_config: dict,
    prediction_result: dict,
    evaluation_result: dict,
    training_result: dict | None = None
) -> dict:
    """
    Construye la estructura oficial de resultados utilizada por el
    Benchmark Científico.
    """

    if not isinstance(model_config, dict):
        raise TypeError("'model_config' debe ser un diccionario.")

    if not isinstance(prediction_result, dict):
        raise TypeError("'prediction_result' debe ser un diccionario.")

    if not isinstance(evaluation_result, dict):
        raise TypeError("'evaluation_result' debe ser un diccionario.")

    if training_result is None:
        training_result = {} # Utilizar configuración de entrenamiento vacía

    elif not isinstance(training_result, dict):
        raise TypeError("'training_result' debe ser un diccionario.")

    required_model_keys = [
        "model_code",
        "model_name",
        "family",
    ] # Definir identificación oficial obligatoria

    missing_model_keys = [
        key
        for key in required_model_keys
        if key not in model_config
    ]

    if missing_model_keys:
        missing = ", ".join(missing_model_keys)
        raise ValueError(
            f"Faltan las siguientes claves en 'model_config': {missing}."
        )

    if not isinstance(model_config["model_code"], str):
        raise TypeError("'model_code' debe ser una cadena.")

    if not model_config["model_code"].strip():
        raise ValueError("'model_code' no puede estar vacío.")

    if not isinstance(model_config["model_name"], str):
        raise TypeError("'model_name' debe ser una cadena.")

    if not model_config["model_name"].strip():
        raise ValueError("'model_name' no puede estar vacío.")

    if not isinstance(model_config["family"], str):
        raise TypeError("'family' debe ser una cadena.")

    if not model_config["family"].strip():
        raise ValueError("'family' no puede estar vacía.")

    required_evaluation_keys = list(
        BENCHMARK_METRICS
    ) # Definir métricas científicas obligatorias

    missing_evaluation_keys = [
        key
        for key in required_evaluation_keys
        if key not in evaluation_result
    ]

    if missing_evaluation_keys:
        missing = ", ".join(missing_evaluation_keys)
        raise ValueError(
            f"Faltan las siguientes métricas de evaluación: {missing}."
        )

    benchmark_result = {
        "model_code": model_config["model_code"],
        "model_name": model_config["model_name"],
        "family": model_config["family"],
        "model_config": model_config,
        "model": training_result.get("model"),
        "training_time": training_result.get("training_time"),
        "loss": training_result.get("loss"),
        "inference_time": prediction_result.get("inference_time"),
        "y_pred": prediction_result.get("y_pred"),
        "y_true": prediction_result.get("y_true"),
    } # Construir estructura oficial del resultado

    for metric in BENCHMARK_METRICS:
        benchmark_result[metric] = evaluation_result[metric] # Incorporar métrica oficial

    benchmark_result = validate_model_result(
        result=benchmark_result,
        model_code=model_config["model_code"],
        model_name=model_config["model_name"],
        family=model_config["family"],
    ) # Validar identificación y métricas oficiales

    return benchmark_result

# BLOQUE 2. Exportación del Resultado del Benchmark -------------------------
## Objetivo: Exportar el resultado oficial del Benchmark Científico en formato
## JSON para su almacenamiento y reutilización por las siguientes fases del
## pipeline.
import json
from pathlib import Path

def save_benchmark_result(
    benchmark_result: dict,
    output_file: Path
) -> None:
    """
    Guarda el resultado oficial del Benchmark Científico en formato JSON.

    Parameters
    ----------
    benchmark_result : dict
        Resultado oficial del Benchmark.

    output_file : Path
        Ruta del archivo JSON de salida.

    Returns
    -------
    None
    """

    if not isinstance(benchmark_result, dict):
        raise TypeError(
            "'benchmark_result' debe ser un diccionario."
        )

    if not isinstance(output_file, Path):
        raise TypeError(
            "'output_file' debe ser un objeto Path."
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )  # Crear directorio si no existe

    with output_file.open(
        mode="w",
        encoding="utf-8"
    ) as file:

        json.dump(
            benchmark_result,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )  # Exportar resultado del Benchmark

# BLOQUE 2A. Construcción del Resultado Exportable
def build_exportable_benchmark_result(
    benchmark_result: dict
) -> dict:
    """
    Construye una versión serializable del resultado oficial
    del Benchmark Científico.
    """

    if not isinstance(benchmark_result, dict):
        raise TypeError("'benchmark_result' debe ser un diccionario.")

    export_result = benchmark_result.copy() # Crear copia independiente del resultado

    export_result.pop("model", None) # Eliminar objeto del modelo entrenado
    export_result.pop("y_pred", None) # Eliminar predicciones
    export_result.pop("y_true", None) # Eliminar valores observados

    if "model_config" in export_result:

        if export_result["model_config"] is None:
            export_result["model_config"] = {} # Normalizar configuración ausente

        if not isinstance(export_result["model_config"], dict):
            raise TypeError("'model_config' debe ser un diccionario.")

        model_config = export_result["model_config"].copy() # Copiar configuración del modelo

        estimator = model_config.get("estimator") # Recuperar estimador

        if estimator is not None:
            model_config["estimator"] = getattr(
                estimator,
                "__name__",
                str(estimator)
            ) # Convertir estimador a representación serializable

        model_config.pop("model_code", None) # Eliminar información duplicada
        model_config.pop("model_name", None) # Eliminar información duplicada
        model_config.pop("family", None) # Eliminar información duplicada

        export_result["model_config"] = model_config # Guardar configuración normalizada

    return export_result

# BLOQUE 2B. VALIDACIÓN DEL RESULTADO DEL BENCHMARK
def validate_model_result(
    result: dict,
    model_code: str,
    model_name: str,
    family: str,
) -> dict:
    """Validar el contrato científico común de un resultado del Benchmark."""

    if not isinstance(result, dict):
        raise TypeError(
            f"El resultado del modelo '{model_name}' debe ser un diccionario."
        )

    if not isinstance(model_code, str):
        raise TypeError(
            "El código oficial del modelo debe ser una cadena."
        )

    if not model_code.strip():
        raise ValueError(
            "El código oficial del modelo no puede estar vacío."
        )

    if not isinstance(model_name, str):
        raise TypeError(
            "El nombre oficial del modelo debe ser una cadena."
        )

    if not model_name.strip():
        raise ValueError(
            "El nombre oficial del modelo no puede estar vacío."
        )

    if not isinstance(family, str):
        raise TypeError(
            "La familia oficial del modelo debe ser una cadena."
        )

    if not family.strip():
        raise ValueError(
            "La familia oficial del modelo no puede estar vacía."
        )

    required_keys = [
        "model_code",
        "model_name",
        "family",
        *BENCHMARK_METRICS,
    ] # Definir contrato científico común del resultado

    missing_keys = [
        key
        for key in required_keys
        if key not in result
    ]

    if missing_keys:
        raise ValueError(
            f"El resultado del modelo '{model_name}' está incompleto: "
            f"{missing_keys}"
        )

    if not isinstance(result["model_code"], str):
        raise TypeError(
            f"El 'model_code' del modelo '{model_name}' debe ser una cadena."
        )

    if not result["model_code"].strip():
        raise ValueError(
            f"El 'model_code' del modelo '{model_name}' está vacío."
        )

    if result["model_code"] != model_code:
        raise ValueError(
            f"El código del resultado '{result['model_code']}' "
            f"no coincide con el código esperado '{model_code}'."
        )

    if result["model_name"] != model_name:
        raise ValueError(
            f"El nombre del resultado '{result['model_name']}' "
            f"no coincide con el modelo esperado '{model_name}'."
        )

    if result["family"] != family:
        raise ValueError(
            f"La familia del resultado '{result['family']}' "
            f"no coincide con la familia esperada '{family}'."
        )

    for metric in BENCHMARK_METRICS:

        value = result[metric]

        if value is None:
            raise ValueError(
                f"El modelo '{model_name}' contiene "
                f"la métrica '{metric}' con valor None."
            )

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"La métrica '{metric}' del modelo '{model_name}' "
                "debe ser numérica."
            ) from error

        if not np.isfinite(numeric_value):
            raise ValueError(
                f"El modelo '{model_name}' contiene "
                f"un valor inválido para '{metric}'."
            )

    return result

# BLOQUE 3. Conversión de Resultados del Benchmark --------------------------
## Objetivo: Convertir los resultados oficiales del Benchmark Científico en un
## DataFrame homogéneo para facilitar su análisis, comparación y exportación.
import pandas as pd

def benchmark_results_to_dataframe(
    benchmark_results: list[dict]
) -> pd.DataFrame:
    """
    Convierte una colección de resultados oficiales del Benchmark
    Científico en un DataFrame.
    """

    if not isinstance(benchmark_results, list):
        raise TypeError(
            "'benchmark_results' debe ser una lista."
        )

    if not benchmark_results:
        raise ValueError(
            "La lista de resultados del Benchmark está vacía."
        )

    for result in benchmark_results:

        if not isinstance(result, dict):
            raise TypeError(
                "Cada resultado del Benchmark debe ser un diccionario."
            )

    benchmark_dataframe = pd.DataFrame(
        [
            build_exportable_benchmark_result(result)
            for result in benchmark_results
        ]
    )

    return benchmark_dataframe

# BLOQUE 4. Resumen del Benchmark Científico
## Objetivo: Generar un resumen homogéneo del Benchmark Científico para su
## utilización en reportes, dashboards y la Plataforma GeoAI.

def generate_benchmark_summary(
    benchmark_dataframe: pd.DataFrame
) -> dict:
    """
    Genera un resumen oficial del Benchmark Científico.

    Parameters
    ----------
    benchmark_dataframe : pd.DataFrame
        DataFrame oficial del Benchmark.

    Returns
    -------
    dict
        Resumen oficial del Benchmark.
    """

    if not isinstance(benchmark_dataframe, pd.DataFrame):
        raise TypeError(
            "'benchmark_dataframe' debe ser un DataFrame."
        )

    if benchmark_dataframe.empty:
        raise ValueError(
            "El DataFrame del Benchmark está vacío."
        )

    ranking_metric = BENCHMARK_CONFIG.get("ranking_metric") # Recuperar métrica oficial de ranking

    if ranking_metric is None:
        raise KeyError(
            "BENCHMARK_CONFIG no contiene 'ranking_metric'."
        )

    if ranking_metric not in benchmark_dataframe.columns:
        raise ValueError(
            f"La métrica oficial '{ranking_metric}' no existe en el DataFrame."
        )

    metric_directions = BENCHMARK_CONFIG.get(
        "metric_directions",
        {}
    ) # Recuperar direcciones oficiales de optimización

    direction = metric_directions.get(
        ranking_metric,
        "min"
    ) # Recuperar dirección de optimización

    if direction not in {"min", "max"}:
        raise ValueError(
            f"Dirección de optimización inválida para '{ranking_metric}': {direction}"
        )

    if direction == "min":
        best_index = benchmark_dataframe[ranking_metric].idxmin() # Identificar mejor resultado mediante minimización
    else:
        best_index = benchmark_dataframe[ranking_metric].idxmax() # Identificar mejor resultado mediante maximización

    best_model = benchmark_dataframe.loc[best_index] # Recuperar mejor modelo según el protocolo oficial

    benchmark_summary = {
        "n_models": len(benchmark_dataframe),
        "ranking_metric": ranking_metric,
        "ranking_direction": direction,
        "best_model_code": best_model["model_code"],
        "best_model_name": best_model["model_name"],
        "best_family": best_model["family"],
        "best_rmse": best_model["rmse"],
        "best_mae": best_model["mae"],
        "best_mape": best_model["mape"],
        "best_r2": best_model["r2"],
    }

    return benchmark_summary

# BLOQUE 5. Visualización del Benchmark Científico --------------------------
## Objetivo: Generar las visualizaciones oficiales del Benchmark Científico
## para facilitar la comparación entre modelos.
import matplotlib.pyplot as plt

def plot_benchmark_metric(
    benchmark_dataframe: pd.DataFrame,
    metric: str = "rmse"
) -> plt.Figure:
    """
    Genera un gráfico de barras para comparar una métrica del Benchmark.

    Parameters
    ----------
    benchmark_dataframe : pd.DataFrame
        DataFrame oficial del Benchmark.

    metric : str, default="rmse"
        Métrica a visualizar.

    Returns
    -------
    matplotlib.figure.Figure
        Figura del gráfico generado.
    """

    if not isinstance(benchmark_dataframe, pd.DataFrame):
        raise TypeError(
            "'benchmark_dataframe' debe ser un DataFrame."
        )

    if benchmark_dataframe.empty:
        raise ValueError(
            "El DataFrame del Benchmark está vacío."
        )

    if metric not in benchmark_dataframe.columns:
        raise ValueError(
            f"La métrica '{metric}' no existe."
        )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.bar(
        benchmark_dataframe["model_name"],
        benchmark_dataframe[metric]
    )

    axis.set_title(
        f"Benchmark Científico - {metric.upper()}"
    )

    axis.set_xlabel(
        "Modelo"
    )

    axis.set_ylabel(
        metric.upper()
    )

    axis.tick_params(
        axis="x",
        rotation=45
    )

    figure.tight_layout()

    return figure