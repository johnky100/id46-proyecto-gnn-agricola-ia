# utils-results.py

# BLOQUE 1. Construcción del Resultado del Benchmark -------------------------
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

    # Validar tipos de entrada --------------------------------------------

    if not isinstance(model_config, dict):
        raise TypeError(
            "'model_config' debe ser un diccionario."
        )

    if not isinstance(prediction_result, dict):
        raise TypeError(
            "'prediction_result' debe ser un diccionario."
        )

    if not isinstance(evaluation_result, dict):
        raise TypeError(
            "'evaluation_result' debe ser un diccionario."
        )

    if training_result is None:
        training_result = {}  # Resultado vacío

    elif not isinstance(training_result, dict):
        raise TypeError(
            "'training_result' debe ser un diccionario."
        )

    # Validar configuración del modelo ------------------------------------

    required_model_keys = [
        "model_code",
        "model_name",
        "family"
    ]

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

    # Validar métricas de evaluación --------------------------------------

    required_evaluation_keys = [
        "rmse",
        "mae",
        "mape",
        "r2"
    ]

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

    # Construir resultado oficial del Benchmark ---------------------------

    benchmark_result = {

        # Identificación ---------------------------------------------------
        "model_code": model_config["model_code"],
        "model_name": model_config["model_name"],
        "family": model_config["family"],

        # Configuración ----------------------------------------------------
        "model_config": model_config,

        # Modelo -----------------------------------------------------------
        "model": training_result.get("model"),

        # Entrenamiento ----------------------------------------------------
        "training_time": training_result.get("training_time"),
        "loss": training_result.get("loss"),

        # Inferencia -------------------------------------------------------
        "inference_time": prediction_result.get("inference_time"),
        "y_pred": prediction_result.get("y_pred"),
        "y_true": prediction_result.get("y_true"),

        # Evaluación -------------------------------------------------------
        "rmse": evaluation_result["rmse"],
        "mae": evaluation_result["mae"],
        "mape": evaluation_result["mape"],
        "r2": evaluation_result["r2"]

        # Metadatos --------------------------------------------------------
        # Espacio reservado para futuras versiones del Benchmark
        # (timestamp, seed, fold, device, framework, etc.)

    }

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

# BLOQUE 2A. Construcción del Resultado Exportable ----------------------------
## Objetivo: Generar una versión serializable del Benchmark para exportación
## en formatos CSV, Parquet y JSON.

def build_exportable_benchmark_result(
    benchmark_result: dict
) -> dict:
    """
    Construye una versión completamente serializable del resultado
    oficial del Benchmark Científico.

    Parameters
    ----------
    benchmark_result : dict
        Resultado oficial del Benchmark.

    Returns
    -------
    dict
        Resultado exportable.
    """

    if not isinstance(benchmark_result, dict):
        raise TypeError(
            "'benchmark_result' debe ser un diccionario."
        )

    export_result = benchmark_result.copy()

    # -------------------------------------------------------
    # Modelo entrenado
    # -------------------------------------------------------

    export_result.pop(
        "model",
        None
    )

    # -------------------------------------------------------
    # Predicciones
    # -------------------------------------------------------

    export_result.pop(
        "y_pred",
        None
    )

    export_result.pop(
        "y_true",
        None
    )

    # -------------------------------------------------------
    # Configuración
    # -------------------------------------------------------

    if "model_config" in export_result:

        model_config = export_result[
            "model_config"
        ].copy()

        estimator = model_config.get(
            "estimator"
        )

        if estimator is not None:

            model_config["estimator"] = getattr(
                estimator,
                "__name__",
                str(estimator)
    )

        export_result[
            "model_config"
        ] = model_config

    return export_result

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

# BLOQUE 4. Resumen del Benchmark Científico -------------------------------
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

    best_model = benchmark_dataframe.loc[
        benchmark_dataframe["rmse"].idxmin()
    ]

    benchmark_summary = {

        # Información general ---------------------------------------------
        "n_models": len(benchmark_dataframe),

        # Mejor modelo ----------------------------------------------------
        "best_model_code": best_model["model_code"],
        "best_model_name": best_model["model_name"],
        "best_family": best_model["family"],

        # Métricas --------------------------------------------------------
        "best_rmse": best_model["rmse"],
        "best_mae": best_model["mae"],
        "best_mape": best_model["mape"],
        "best_r2": best_model["r2"]

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