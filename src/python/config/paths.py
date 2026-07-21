# config-paths.py

# BLOQUE 1. Directorio raíz ------------------------------------------------
## Objetivo: Definir el directorio raíz oficial del proyecto que servirá como
## punto de partida para construir toda la arquitectura de directorios.
from pathlib import Path # Manejo de rutas

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[3] # 0=config, 1=Python, 2=src, 3=raíz del proyecto
) # Directorio raíz del proyecto

# BLOQUE 2. Directorios del Proyecto ---------------------------------------
## Objetivo:
## Definir la estructura oficial de directorios utilizada por el proyecto,
## incluyendo los directorios generales, los productos del pipeline y los
## recursos de Ingeniería de Datos.

# -----------------------------------------------------------------------------
# 2.1. Directorios Generales
# -----------------------------------------------------------------------------

DATA_DIR = (
    ROOT_DIR
    / "data"
)

CONFIG_DIR = (
    ROOT_DIR
    / "src"
    / "config"
)

RAW_DIR = (
    DATA_DIR
    / "raw"
)

INTERIM_DIR = (
    DATA_DIR
    / "interim"
)

PROCESSED_DIR = (
    DATA_DIR
    / "processed"
)

REPORTS_DIR = (
    ROOT_DIR
    / "reports"
)

LOGS_DIR = (
    ROOT_DIR
    / "logs"
)

DOCS_DIR = (
    ROOT_DIR
    / "docs"
)

# -----------------------------------------------------------------------------
# 2.2. Directorios Oficiales del Pipeline
# -----------------------------------------------------------------------------

OUTPUTS_DIR = (
    ROOT_DIR
    / "src"
    / "python"
    / "outputs"
)

GRAPH_DATA_DIR = (
    OUTPUTS_DIR
    / "graph_data"
)

GRAPH_DATA_COLLECTION_FILE = (
    GRAPH_DATA_DIR
    / "graph_data_collection.pt"
)

METADATA_DIR = (
    OUTPUTS_DIR
    / "metadata"
)

AUDITS_DIR = (
    OUTPUTS_DIR
    / "audits"
)

BENCHMARK_DIR = (
    OUTPUTS_DIR
    / "benchmark"
)

TRAINING_DIR = (
    OUTPUTS_DIR
    / "training"
)

EVALUATION_DIR = (
    OUTPUTS_DIR
    / "evaluation"
)

# Reportes de Explicabilidad (XAI) ------------------------------------------
EVALUATION_REPORTS_DIR = (
    EVALUATION_DIR
    / "reports"
)

FORECASTING_DIR = (
    OUTPUTS_DIR
    / "forecasting"
)

MODELS_DIR = (
    OUTPUTS_DIR
    / "models"
)

FIGURES_DIR = (
    OUTPUTS_DIR
    / "figures"
)


# -----------------------------------------------------------------------------
# Visualizaciones del Grafo
# -----------------------------------------------------------------------------

GRAPHS_DIR = (
    FIGURES_DIR
    / "graphs"
)

# BLOQUE 3. Ingeniería de Datos (R) ---------------------------------------
## Objetivo: Definir los directorios oficiales utilizados por el pipeline de
## Ingeniería de Datos en R para el almacenamiento de productos científicos.
R_DIR = (
    PROCESSED_DIR
    / "r"
) # Productos del pipeline en R

R_MASTER_DIR = (
    R_DIR
    / "master"
) # Datasets oficiales del proyecto

R_BENCHMARK_DIR = (
    R_DIR
    / "benchmark"
) # Benchmark científico

R_REPORTS_DIR = (
    R_DIR
    / "reports"
) # Reportes del pipeline en R

R_LOGS_DIR = (
    R_DIR
    / "logs"
) # Registros del pipeline

# BLOQUE 4. Plataforma GeoAI ---------------------------------------
## Objetivo:
## Definir la estructura oficial de almacenamiento utilizada por la Plataforma
## GeoAI para la gestión de datos, reportes, servicios y visualizaciones
## generadas por el pipeline científico.

# -----------------------------------------------------------------------------
# 4.1. Directorio principal de la Plataforma GeoAI
# -----------------------------------------------------------------------------

GEOAI_DIR = (
    OUTPUTS_DIR
    / "geoai"
)

# -----------------------------------------------------------------------------
# 4.2. Reportes
# -----------------------------------------------------------------------------

GEOAI_REPORTS_DIR = (
    GEOAI_DIR
    / "reports"
)

# -----------------------------------------------------------------------------
# 4.3. Datos
# -----------------------------------------------------------------------------

GEOAI_DATA_DIR = (
    GEOAI_DIR
    / "data"
)

# -----------------------------------------------------------------------------
# 4.4. Registros
# -----------------------------------------------------------------------------

GEOAI_LOGS_DIR = (
    GEOAI_DIR
    / "logs"
)

# -----------------------------------------------------------------------------
# 4.5. Dashboard
# -----------------------------------------------------------------------------

GEOAI_DASHBOARD_DIR = (
    GEOAI_DIR
    / "dashboard"
)

# -----------------------------------------------------------------------------
# 4.6. API
# -----------------------------------------------------------------------------

GEOAI_API_DIR = (
    GEOAI_DIR
    / "api"
)

# -----------------------------------------------------------------------------
# 4.7. Agentes de IA
# -----------------------------------------------------------------------------

GEOAI_AGENTS_DIR = (
    GEOAI_DIR
    / "agents"
)

# -----------------------------------------------------------------------------
# 4.8. Caché
# -----------------------------------------------------------------------------

GEOAI_CACHE_DIR = (
    GEOAI_DIR
    / "cache"
)

# -----------------------------------------------------------------------------
# 4.9. Configuración
# -----------------------------------------------------------------------------

GEOAI_CONFIG_DIR = (
    GEOAI_DIR
    / "config"
)


# BLOQUE 5. Archivos oficiales --------------------------------------------
## Objetivo: Definir las rutas oficiales de los productos científicos
## generados durante cada fase del pipeline y centralizar su acceso.

# -----------------------------------------------------------------------------
# 5.1. Dataset Científico
# -----------------------------------------------------------------------------

DATASET_FILE = (
    R_MASTER_DIR
    / "dataset_gnn_certificado.parquet"
)

# -----------------------------------------------------------------------------
# 5.2. Construcción del GraphData
# -----------------------------------------------------------------------------

GRAPH_DATA_FILE = (
    GRAPH_DATA_DIR
    / "graph_data.pt"
)

NODE_CATALOG_FILE = (
    GRAPH_DATA_DIR
    / "node_catalog.parquet"
)

GRAPH_VALIDATION_FILE = (
    GRAPH_DATA_DIR
    / "graph_validation.json"
)

GRAPH_STATISTICS_FILE = (
    GRAPH_DATA_DIR
    / "graph_statistics.json"
)

GRAPH_METADATA_FILE = (
    METADATA_DIR
    / "graph_metadata.json"
)

GRAPH_AUDIT_FILE = (
    AUDITS_DIR
    / "graph_audit.csv"
)

# -----------------------------------------------------------------------------
# 5.3. Benchmark Científico
# -----------------------------------------------------------------------------

BENCHMARK_RESULTS_FILE = (
    BENCHMARK_DIR
    / "benchmark_results.joblib"
)

BENCHMARK_SUMMARY_FILE = (
    BENCHMARK_DIR
    / "benchmark_summary.csv"
)

BENCHMARK_METRICS_FILE = (
    BENCHMARK_DIR
    / "benchmark_metrics.parquet"
)

BENCHMARK_RANKING_FILE = (
    BENCHMARK_DIR
    / "benchmark_ranking.csv"
)

BEST_MODEL_CONFIG_FILE = (
    BENCHMARK_DIR
    / "best_model_config.json"
)

# -----------------------------------------------------------------------------
# 5.4. Entrenamiento
# -----------------------------------------------------------------------------

BEST_MODEL_JOBLIB_FILE = (
    MODELS_DIR
    / "best_model.joblib"
)

BEST_MODEL_TORCH_FILE = (
    MODELS_DIR
    / "best_model.pt"
)

BEST_MODEL_METADATA_FILE = (
    MODELS_DIR
    / "best_model_metadata.json"
)

TRAINING_SUMMARY_FILE = (
    TRAINING_DIR
    / "training_summary.json"
)

TRAINING_LOG_FILE = (
    TRAINING_DIR
    / "training_log.json"
)

# -----------------------------------------------------------------------------
# 5.5. Evaluación
# -----------------------------------------------------------------------------

EVALUATION_RESULTS_FILE = (
    EVALUATION_DIR
    / "evaluation_results.parquet"
)

EVALUATION_SUMMARY_FILE = (
    EVALUATION_DIR
    / "evaluation_summary.joblib"
)

VALIDATION_RESULTS_FILE = (
    EVALUATION_DIR
    / "validation_results.parquet"
)

FEATURE_IMPORTANCE_FILE = (
    EVALUATION_DIR
    / "feature_importance.parquet"
)

RESIDUALS_FILE = (
    EVALUATION_DIR
    / "residuals.parquet"
)

SHAP_VALUES_FILE = (
    EVALUATION_DIR
    / "shap_values.joblib"
)

ATTENTION_WEIGHTS_FILE = (
    EVALUATION_DIR
    / "attention_weights.pt"
)

GNN_EMBEDDINGS_FILE = (
    EVALUATION_DIR
    / "gnn_embeddings.pt"
)

EVALUATION_REPORT_FILE = (
    EVALUATION_DIR
    / "evaluation_report.json"
)

# -----------------------------------------------------------------------------
# 5.6. Forecasting
# -----------------------------------------------------------------------------

FORECAST_RESULTS_FILE = (
    FORECASTING_DIR
    / "forecast_panel.parquet"
)

SCENARIO_RESULTS_FILE = (
    FORECASTING_DIR
    / "scenario_results.parquet"
)

FORECAST_REPORT_FILE = (
    FORECASTING_DIR
    / "forecast_report.pdf"
)

# -----------------------------------------------------------------------------
# 5.7. Catálogo Oficial de Productos
# -----------------------------------------------------------------------------

PROJECT_FILES = {

    # Dataset Científico
    "dataset_scientific": DATASET_FILE,

    # GraphData
    "graph_data": GRAPH_DATA_FILE,
    "node_catalog": NODE_CATALOG_FILE,
    "graph_validation": GRAPH_VALIDATION_FILE,
    "graph_statistics": GRAPH_STATISTICS_FILE,
    "graph_metadata": GRAPH_METADATA_FILE,
    "graph_audit": GRAPH_AUDIT_FILE,

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
    "training_summary": TRAINING_SUMMARY_FILE,
    "training_log": TRAINING_LOG_FILE,

    # Evaluación
    "evaluation_results": EVALUATION_RESULTS_FILE,
    "evaluation_summary": EVALUATION_SUMMARY_FILE,
    "validation_results": VALIDATION_RESULTS_FILE,
    "feature_importance": FEATURE_IMPORTANCE_FILE,
    "residuals": RESIDUALS_FILE,
    "shap_values": SHAP_VALUES_FILE,
    "attention_weights": ATTENTION_WEIGHTS_FILE,
    "gnn_embeddings": GNN_EMBEDDINGS_FILE,
    "evaluation_report": EVALUATION_REPORT_FILE,

    # Forecasting
    "forecast_results": FORECAST_RESULTS_FILE,
    "scenario_results": SCENARIO_RESULTS_FILE,
    "forecast_report": FORECAST_REPORT_FILE

}

# =============================================================================
# BLOQUE 6. Catálogo Oficial de Directorios --------------------------------------------
# =============================================================================
## Objetivo:
## Centralizar el acceso a todos los directorios oficiales del proyecto
## mediante un único catálogo reutilizable por todos los módulos del pipeline.
# =============================================================================

PROJECT_DIRECTORIES: dict[str, Path] = {

    # -------------------------------------------------------------------------
    # 6.1. Directorios generales
    # -------------------------------------------------------------------------

    "data": DATA_DIR,
    "config": CONFIG_DIR,
    "raw": RAW_DIR,
    "interim": INTERIM_DIR,
    "processed": PROCESSED_DIR,
    "reports": REPORTS_DIR,
    "logs": LOGS_DIR,
    "docs": DOCS_DIR,

    # -------------------------------------------------------------------------
    # 6.2. Ingeniería de Datos (R)
    # -------------------------------------------------------------------------

    "r": R_DIR,
    "r_master": R_MASTER_DIR,
    "r_benchmark": R_BENCHMARK_DIR,
    "r_reports": R_REPORTS_DIR,
    "r_logs": R_LOGS_DIR,

    # -------------------------------------------------------------------------
    # 6.3. Pipeline Científico (Python)
    # -------------------------------------------------------------------------

    "outputs": OUTPUTS_DIR,

    "graph_data": GRAPH_DATA_DIR,
    "metadata": METADATA_DIR,
    "audits": AUDITS_DIR,

    "benchmark": BENCHMARK_DIR,
    "training": TRAINING_DIR,
    "evaluation": EVALUATION_DIR,
    "evaluation_reports": EVALUATION_REPORTS_DIR,
    "forecasting": FORECASTING_DIR,
    "models": MODELS_DIR,
    "figures": FIGURES_DIR,

    # -------------------------------------------------------------------------
    # 6.4. Plataforma GeoAI
    # -------------------------------------------------------------------------

    "geoai": GEOAI_DIR,
    "geoai_reports": GEOAI_REPORTS_DIR,
    "geoai_data": GEOAI_DATA_DIR,
    "geoai_logs": GEOAI_LOGS_DIR,
    "geoai_dashboard": GEOAI_DASHBOARD_DIR,
    "geoai_api": GEOAI_API_DIR,
    "geoai_agents": GEOAI_AGENTS_DIR,
    "geoai_cache": GEOAI_CACHE_DIR,
    "geoai_config": GEOAI_CONFIG_DIR

}

# =============================================================================
# BLOQUE 7. Validación de la estructura del proyecto
# =============================================================================

from pathlib import Path


def validate_directory(
    directory: Path,
    verbose: bool = True,
) -> None:
    """
    Crea el directorio si no existe.
    """
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if verbose:
        print(f"✓ {directory}")


def validate_project_structure(
    verbose: bool = True,
) -> None:
    """
    Verifica y crea la estructura oficial del proyecto.
    """

    for directory in PROJECT_DIRECTORIES.values():
        validate_directory(
            directory=directory,
            verbose=verbose,
        )