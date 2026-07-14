# results.py

def build_benchmark_result(
    model_config,
    prediction_result,
    evaluation_result,
    training_result = None
):
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

    # Inicialización --------------------------------------------------------
    if training_result is None:

        training_result = {}

    # Construcción ----------------------------------------------------------
    benchmark_result = {

        "model_code": model_config["model_code"],

        "model_name": model_config["model_name"],

        "family": model_config["family"],

        "model": training_result.get(
            "model"
        ),

        "training_time": training_result.get(
            "training_time"
        ),

        "inference_time": prediction_result[
            "inference_time"
        ],

        "rmse": evaluation_result[
            "rmse"
        ],

        "mae": evaluation_result[
            "mae"
        ],

        "mape": evaluation_result[
            "mape"
        ],

        "r2": evaluation_result[
            "r2"
        ],

        "adjusted_r2": evaluation_result[
            "adjusted_r2"
        ]

    } # Resultado oficial del Benchmark

    return benchmark_result