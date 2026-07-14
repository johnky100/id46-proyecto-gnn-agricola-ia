# paths.py

# BLOQUE 1. Directorio raíz -----------------------------------------------
from pathlib import Path # Manejo de rutas
ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[3] # 0 config. 1 Python. 2 src. 3 avanzado-ia
) # Directorio raíz del proyecto

# BLOQUE 2. Directorios principales ---------------------------------------
DATA_DIR = (
    ROOT_DIR
    / "data"
) # Carpeta principal de datos

CONFIG_DIR = (
    ROOT_DIR
    / "src"
    / "config"
) # Configuración del proyecto

RAW_DIR = (
    DATA_DIR
    / "raw"
) # Datos originales

INTERIM_DIR = (
    DATA_DIR
    / "interim"
) # Productos temporales del ETL

PROCESSED_DIR = (
    DATA_DIR
    / "processed"
) # Productos oficiales

REPORTS_DIR = (
    ROOT_DIR
    / "reports"
) # Reportes generales del proyecto

LOGS_DIR = (
    ROOT_DIR
    / "logs"
) # Registros

DOCS_DIR = (
    ROOT_DIR
    / "docs"
) # Documentación

# BLOQUE 3. Ingeniería de Datos (R) ---------------------------------------
R_DIR = (
    PROCESSED_DIR
    / "r"
) # Productos del pipeline en R

R_DATASET_DIR = (
    R_DIR
    / "dataset"
) # Dataset científico

R_BENCHMARK_DIR = (
    R_DIR
    / "benchmark"
) # Benchmark de selección de datasets

R_REPORTS_DIR = (
    R_DIR
    / "reports"
) # Reportes del pipeline en R

R_LOGS_DIR = (
    R_DIR
    / "logs"
) # Registros del pipeline

# BLOQUE 4. Modelado GNN (Python) -----------------------------------------
PYTHON_DIR = (
    PROCESSED_DIR
    / "python"
) # Productos del pipeline GNN

GRAPH_DIR = (
    PYTHON_DIR
    / "graph"
) # Construcción del GraphData

BENCHMARK_DIR = (
    PYTHON_DIR
    / "benchmark"
) # Benchmark científico

MODELS_DIR = (
    PYTHON_DIR
    / "models"
) # Modelos entrenados

CHECKPOINTS_DIR = (
    MODELS_DIR
    / "checkpoints"
) # Checkpoints

EMBEDDINGS_DIR = (
    MODELS_DIR
    / "embeddings"
) # Embeddings

EVALUATION_DIR = (
    PYTHON_DIR
    / "evaluation"
) # Evaluación

EVALUATION_PLOTS_DIR = (
    EVALUATION_DIR
    / "plots"
) # Figuras de evaluación

FORECAST_DIR = (
    PYTHON_DIR
    / "forecast"
) # Forecasting

FORECAST_MAPS_DIR = (
    FORECAST_DIR
    / "maps"
) # Mapas de pronóstico

GEOAI_DIR = (
    PYTHON_DIR
    / "geoai"
) # Plataforma GeoAI

# BLOQUE 5. Archivos oficiales --------------------------------------------
# Etapa 0. Dataset Científico ---------------------------------------------
DATASET_FILE = (
    R_DATASET_DIR
    / "dataset_gnn_certificado.parquet"
) # Dataset científico oficial

# Etapa 1. Construcción del Grafo -----------------------------------------
GRAPH_DATA_FILE = (
    GRAPH_DIR
    / "graph_data.pt"
) # Grafo oficial

# Etapa 2. Benchmark ------------------------------------------------------
BENCHMARK_RESULTS_FILE = (
    BENCHMARK_DIR
    / "benchmark_results.joblib"
) # Resultados completos del benchmark

BENCHMARK_SUMMARY_FILE = (
    BENCHMARK_DIR
    / "benchmark_summary.csv"
) # Resumen ejecutivo del benchmark

BENCHMARK_METRICS_FILE = (
    BENCHMARK_DIR
    / "benchmark_metrics.parquet"
) # Métricas de todos los modelos

BENCHMARK_RANKING_FILE = (
    BENCHMARK_DIR
    / "benchmark_ranking.csv"
) # Ranking oficial de modelos

BEST_MODEL_CONFIG_FILE = (
    BENCHMARK_DIR
    / "best_model_config.json"
) # Configuración del modelo ganador

# Etapa 3. Entrenamiento --------------------------------------------------
BEST_MODEL_JOBLIB_FILE = (
    MODELS_DIR
    / "best_model.joblib"
) # Modelo oficial para las familias Statistical y Machine Learning

BEST_MODEL_TORCH_FILE = (
    MODELS_DIR
    / "best_model.pt"
) # Modelo oficial para las familias Deep Learning y Graph Neural Networks

BEST_MODEL_METADATA_FILE = (
    MODELS_DIR
    / "best_model_metadata.json"
) # Metadatos del entrenamiento definitivo

# Etapa 4. Evaluación ------------------------------------------------------
EVALUATION_RESULTS_FILE = (
    EVALUATION_DIR
    / "evaluation_results.parquet"
) # Métricas oficiales de evaluación

EVALUATION_SUMMARY_FILE = (
    EVALUATION_DIR
    / "evaluation_summary.joblib"
) # Resultado completo de la evaluación científica

VALIDATION_RESULTS_FILE = (
    EVALUATION_DIR
    / "validation_results.parquet"
) # Validación científica del modelo

FEATURE_IMPORTANCE_FILE = (
    EVALUATION_DIR
    / "feature_importance.parquet"
) # Importancia global de las variables

RESIDUALS_FILE = (
    EVALUATION_DIR
    / "residuals.parquet"
) # Residuos del modelo

SHAP_VALUES_FILE = (
    EVALUATION_DIR
    / "shap_values.joblib"
) # Valores SHAP del modelo

GNN_EMBEDDINGS_FILE = (
    EVALUATION_DIR
    / "gnn_embeddings.pt"
) # Embeddings generados por la GNN

EVALUATION_REPORT_FILE = (
    EVALUATION_DIR
    / "evaluation_report.json"
) # Reporte ejecutivo de la evaluación científica

# Etapa 5. Forecasting ----------------------------------------------------
FORECAST_RESULTS_FILE = (
    FORECAST_DIR
    / "forecast_panel.parquet"
) # Pronósticos oficiales

SCENARIO_RESULTS_FILE = (
    FORECAST_DIR
    / "scenario_results.parquet"
) # Resultados oficiales de los escenarios

FORECAST_REPORT_FILE = (
    FORECAST_DIR
    / "forecast_report.pdf"
) # Reporte técnico del forecasting

# Productos oficiales del proyecto ---------------------------------------
PROJECT_FILES = {
    "dataset_scientific": DATASET_FILE,
    "graph_data": GRAPH_DATA_FILE,

    # Benchmark
    "benchmark_results": BENCHMARK_RESULTS_FILE,
    "benchmark_summary": BENCHMARK_SUMMARY_FILE,
    "benchmark_metrics": BENCHMARK_METRICS_FILE,
    "benchmark_ranking": BENCHMARK_RANKING_FILE,
    "best_model_config": BEST_MODEL_CONFIG_FILE,

    # Entrenamiento
    "trained_model_joblib": BEST_MODEL_JOBLIB_FILE,
    "trained_model_torch": BEST_MODEL_TORCH_FILE,
    "trained_model_metadata": BEST_MODEL_METADATA_FILE,

    # Evaluación
    "evaluation_results": EVALUATION_RESULTS_FILE,
    "evaluation_summary": EVALUATION_SUMMARY_FILE,
    "validation_results": VALIDATION_RESULTS_FILE,
    "feature_importance": FEATURE_IMPORTANCE_FILE,
    "residuals": RESIDUALS_FILE,
    "shap_values": SHAP_VALUES_FILE,
    "gnn_embeddings": GNN_EMBEDDINGS_FILE,
    "evaluation_report": EVALUATION_REPORT_FILE,

    # Forecasting
    "forecast": FORECAST_RESULTS_FILE

} # Productos oficiales del proyecto

# Etapa 6 → GeoAI

# BLOQUE 6. Plataforma GeoAI ----------------------------------------------
GEOAI_REPORTS_DIR = (
    GEOAI_DIR
    / "reports"
) # Reportes de la plataforma

GEOAI_DATA_DIR = (
    GEOAI_DIR
    / "data"
) # Datos consumidos por la plataforma

GEOAI_LOGS_DIR = (
    GEOAI_DIR
    / "logs"
) # Registros de la plataforma

GEOAI_DASHBOARD_DIR = (
    GEOAI_DIR
    / "dashboard"
) # Dashboard

GEOAI_API_DIR = (
    GEOAI_DIR
    / "api"
) # API

# BLOQUE 7. Directorios del proyecto --------------------------------------

PROJECT_DIRECTORIES: dict[str, Path] = {
    # Directorios generales
    "project_reports": REPORTS_DIR,
    "logs": LOGS_DIR,
    "docs": DOCS_DIR,

    # Ingeniería de Datos (R)
    "r_dataset": R_DATASET_DIR,
    "r_benchmark": R_BENCHMARK_DIR,
    "r_reports": R_REPORTS_DIR,
    "r_logs": R_LOGS_DIR,

    # Modelado GNN
    "graph": GRAPH_DIR,
    "benchmark": BENCHMARK_DIR,
    "models": MODELS_DIR,
    "checkpoints": CHECKPOINTS_DIR,
    "embeddings": EMBEDDINGS_DIR,
    "evaluation": EVALUATION_DIR,
    "evaluation_plots": EVALUATION_PLOTS_DIR,
    "forecast": FORECAST_DIR,
    "forecast_maps": FORECAST_MAPS_DIR,

    # Plataforma GeoAI
    "geoai": GEOAI_DIR,
    "geoai_reports": GEOAI_REPORTS_DIR,
    "geoai_logs": GEOAI_LOGS_DIR,
    "geoai_dashboard": GEOAI_DASHBOARD_DIR,
    "geoai_api": GEOAI_API_DIR

} # Directorios oficiales del proyecto

# BLOQUE 8. Validación de directorios --------------------------------------
def validate_directory(directory: Path) -> str:
    """
    Verifica la existencia de un directorio.

    Si el directorio no existe, lo crea automáticamente.

    Parameters
    ----------
    directory : Path
        Directorio a validar.

    Returns
    -------
    str
        Estado del directorio:

        - EXISTE
        - CREADO
    """

    if directory.exists() and directory.is_dir():
        return "EXISTE"

    directory.mkdir(
        parents = True,
        exist_ok = True
    ) # Crear directorio

    return "CREADO"


def validate_project_structure(
    verbose: bool = True
) -> dict[str, dict]:
    """
    Valida la estructura oficial del proyecto.

    Recorre todos los directorios definidos en
    PROJECT_DIRECTORIES, verifica su existencia y crea
    aquellos que no existan.

    Parameters
    ----------
    verbose : bool, default=True
        Indica si se imprime el reporte de validación
        en la consola.

    Returns
    -------
    dict[str, dict]
        Estado de todos los directorios del proyecto.
    """

    directory_status: dict[str, dict] = {}

    if verbose:

        print(
            "\nValidación de directorios del proyecto\n"
        )

    for name, directory in PROJECT_DIRECTORIES.items():

        status = validate_directory(
            directory
        )

        directory_status[name] = {
            "status": status,
            "path": directory
        }

        if verbose:

            print(
                f"{name:<20}"
                f"{status:<10}"
                f"{directory}"
            )

    return directory_status

# BLOQUE 9. Reportes de evaluación -----------------------------------------
EVALUATION_REPORTS_DIR = (
    REPORTS_DIR
    / "evaluation"
) # Reportes oficiales de evaluación