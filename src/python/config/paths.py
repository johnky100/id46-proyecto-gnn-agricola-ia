# paths.py

# BLOQUE 1. Directorio raíz ---------------------------------------------------
## Objetivo: Definir el directorio raíz oficial del proyecto que servirá como
## punto de partida para construir toda la arquitectura de directorios.
from pathlib import Path # Manejo de rutas

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[3] # 0=config, 1=Python, 2=src, 3=raíz del proyecto
) # Directorio raíz del proyecto

PROJECT_NAME = ROOT_DIR.name # Nombre del directorio raíz del proyecto

# -----------------------------------------------------------------------------
# BLOQUE 2. Directorios del Proyecto
# -----------------------------------------------------------------------------
# Objetivo:
# Definir la estructura oficial de directorios utilizada por la plataforma
# GeoAI, incluyendo la infraestructura del proyecto, los datos científicos,
# los productos del pipeline y los recursos compartidos.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 2.1 Infraestructura del Proyecto
# -----------------------------------------------------------------------------

DATA_DIR = (
    ROOT_DIR
    / "data"
) # Directorio principal de datos

CONFIG_DIR = (
    ROOT_DIR
    / "src"
    / "Python"
    / "config"
) # Configuración oficial del proyecto

REPORTS_DIR = (
    ROOT_DIR
    / "reports"
) # Reportes científicos

LOGS_DIR = (
    ROOT_DIR
    / "logs"
) # Registros de ejecución

DOCS_DIR = (
    ROOT_DIR
    / "docs"
) # Documentación técnica y científica

CACHE_DIR = (
    ROOT_DIR
    / "cache"
) # Caché del proyecto

TEMP_DIR = (
    ROOT_DIR
    / "temp"
) # Archivos temporales

ARTIFACTS_DIR = (
    ROOT_DIR
    / "artifacts"
) # Artefactos científicos compartidos

TESTS_DIR = (
    ROOT_DIR
    / "tests"
) # Pruebas del proyecto

# -----------------------------------------------------------------------------
# 2.2 Datos del Proyecto
# -----------------------------------------------------------------------------

RAW_DIR = (
    DATA_DIR
    / "raw"
) # Datos originales

INTERIM_DIR = (
    DATA_DIR
    / "interim"
) # Datos intermedios

PROCESSED_DIR = (
    DATA_DIR
    / "processed"
) # Datos procesados

# -----------------------------------------------------------------------------
# 2.3 Pipeline Científico
# -----------------------------------------------------------------------------

OUTPUTS_DIR = (
    ROOT_DIR
    / "src"
    / "Python"
    / "outputs"
) # Productos oficiales del pipeline

# -----------------------------------------------------------------------------
# 2.3.1 Recursos Compartidos
# -----------------------------------------------------------------------------

METADATA_DIR = (
    OUTPUTS_DIR
    / "metadata"
) # Metadatos científicos

CONTRACTS_DIR = (
    OUTPUTS_DIR
    / "contracts"
) # Contratos científicos

CERTIFICATES_DIR = (
    OUTPUTS_DIR
    / "certificates"
) # Certificados científicos

AUDITS_DIR = (
    OUTPUTS_DIR
    / "audits"
) # Auditorías oficiales

MODELS_DIR = (
    OUTPUTS_DIR
    / "models"
) # Modelos entrenados

# -----------------------------------------------------------------------------
# 2.3.2 Módulos Científicos
# -----------------------------------------------------------------------------

GRAPH_DATA_DIR = (
    OUTPUTS_DIR
    / "graph_data"
) # Módulo 01_build_graph

BENCHMARK_DIR = (
    OUTPUTS_DIR
    / "benchmark"
) # Módulo 02_benchmark

TRAINING_DIR = (
    OUTPUTS_DIR
    / "training"
) # Módulo 03_train_model

EVALUATION_DIR = (
    OUTPUTS_DIR
    / "evaluation"
) # Módulo 04_evaluation

FORECASTING_DIR = (
    OUTPUTS_DIR
    / "forecasting"
) # Módulo 05_forecasting

# -----------------------------------------------------------------------------
# BLOQUE 3. Ingeniería de Datos (R)
# -----------------------------------------------------------------------------
# Objetivo:
# Definir los directorios oficiales utilizados por el pipeline de Ingeniería
# de Datos en R para la construcción, validación y certificación del Dataset
# Científico.
# -----------------------------------------------------------------------------

R_DIR = (
    PROCESSED_DIR
    / "r"
) # Productos del pipeline de Ingeniería de Datos en R

R_MASTER_DIR = (
    R_DIR
    / "master"
) # Datasets científicos oficiales generados por el pipeline

R_REPORTS_DIR = (
    R_DIR
    / "reports"
) # Reportes científicos del pipeline

R_METADATA_DIR = (
    R_DIR
    / "metadata"
) # Metadatos del Dataset Científico

R_CONTRACTS_DIR = (
    R_DIR
    / "contracts"
) # Contratos del Dataset Científico

R_CERTIFICATES_DIR = (
    R_DIR
    / "certificates"
) # Certificados del Dataset Científico

R_AUDITS_DIR = (
    R_DIR
    / "audits"
) # Auditorías del pipeline de Ingeniería de Datos

R_LOGS_DIR = (
    R_DIR
    / "logs"
) # Registros de ejecución del pipeline

# -----------------------------------------------------------------------------
# BLOQUE 4. Plataforma GeoAI
# -----------------------------------------------------------------------------
# Objetivo:
# Definir la estructura oficial de almacenamiento utilizada por la Plataforma
# GeoAI para la gestión de productos científicos, servicios inteligentes,
# visualizaciones y componentes de despliegue del sistema.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 4.1 Directorio Principal de la Plataforma GeoAI
# -----------------------------------------------------------------------------

GEOAI_DIR = (
    OUTPUTS_DIR
    / "geoai"
) # Directorio principal de la Plataforma GeoAI

# -----------------------------------------------------------------------------
# 4.2 Artefactos Científicos
# -----------------------------------------------------------------------------

GEOAI_DATA_DIR = (
    GEOAI_DIR
    / "data"
) # Datos utilizados por la Plataforma GeoAI

GEOAI_REPORTS_DIR = (
    GEOAI_DIR
    / "reports"
) # Reportes científicos generados por la plataforma

# -----------------------------------------------------------------------------
# 4.3 Servicios de la Plataforma
# -----------------------------------------------------------------------------

GEOAI_DASHBOARD_DIR = (
    GEOAI_DIR
    / "dashboard"
) # Dashboard científico interactivo

GEOAI_API_DIR = (
    GEOAI_DIR
    / "api"
) # API oficial de la plataforma

GEOAI_AGENTS_DIR = (
    GEOAI_DIR
    / "agents"
) # Agentes inteligentes GeoAI

GEOAI_EXPORT_DIR = (
    GEOAI_DIR
    / "export"
) # Exportaciones de la Plataforma GeoAI

# -----------------------------------------------------------------------------
# 4.4 Configuración
# -----------------------------------------------------------------------------

GEOAI_CONFIG_DIR = (
    GEOAI_DIR
    / "config"
) # Configuración oficial de la plataforma

# -----------------------------------------------------------------------------
# 4.5 Metadatos
# -----------------------------------------------------------------------------

GEOAI_METADATA_DIR = (
    GEOAI_DIR
    / "metadata"
) # Metadatos de la Plataforma GeoAI

# -----------------------------------------------------------------------------
# 4.6 Contratos
# -----------------------------------------------------------------------------

GEOAI_CONTRACTS_DIR = (
    GEOAI_DIR
    / "contracts"
) # Contratos oficiales de la Plataforma GeoAI

# -----------------------------------------------------------------------------
# 4.7 Certificación
# -----------------------------------------------------------------------------

GEOAI_CERTIFICATES_DIR = (
    GEOAI_DIR
    / "certificates"
) # Certificados científicos de la Plataforma GeoAI

# -----------------------------------------------------------------------------
# 4.8 Auditoría
# -----------------------------------------------------------------------------

GEOAI_AUDITS_DIR = (
    GEOAI_DIR
    / "audits"
) # Auditorías de la Plataforma GeoAI

# -----------------------------------------------------------------------------
# 4.9 Registros
# -----------------------------------------------------------------------------

GEOAI_LOGS_DIR = (
    GEOAI_DIR
    / "logs"
) # Registros de ejecución de la plataforma

# -----------------------------------------------------------------------------
# 4.10 Caché
# -----------------------------------------------------------------------------

GEOAI_CACHE_DIR = (
    GEOAI_DIR
    / "cache"
) # Caché utilizada por la plataforma

# -----------------------------------------------------------------------------
# BLOQUE 5. Archivos Oficiales del Proyecto
# -----------------------------------------------------------------------------
# Objetivo: Centralizar las rutas oficiales de todos los artefactos científicos, metadatos, contratos, manifiestos,
# certificaciones y auditorías generados por el pipeline.
#
# Organización: 5.1 Dataset Científico. 5.2 GraphData. 5.2.1 Artefactos científicos. 5.2.2 Reportes de validación.
# 5.2.3 Reportes estadísticos. 5.2.4 Certificación. 5.2.5 Auditoría.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 5.1 Dataset Científico Oficial
# -----------------------------------------------------------------------------

DATASET_DIR = (
    PROCESSED_DIR
    / "dataset"
) # Directorio del Dataset Científico

DATASET_FILE = (
    DATASET_DIR
    / "dataset_cientifico_certificado.parquet"
) # Dataset Científico Oficial

DATASET_CONTRACT_FILE = (
    R_MASTER_DIR
    / "dataset_contract.json"
) # Contrato oficial del Dataset Científico

DATASET_VALIDATION_FILE = (
    R_MASTER_DIR
    / "dataset_validation.json"
) # Validación oficial del Dataset Científico

DATASET_STATISTICS_FILE = (
    R_MASTER_DIR
    / "dataset_statistics.json"
) # Estadísticas descriptivas oficiales del Dataset Científico

DATASET_CERTIFICATE_FILE = (
    CERTIFICATES_DIR
    / "dataset"
    / "dataset_certificate.json"
) # Certificado científico del Dataset Científico

DATASET_CERTIFICATE_SIGNATURE_FILE = (
    CERTIFICATES_DIR
    / "dataset"
    / "dataset_signature.json"
) # Firma digital del certificado científico

DATASET_AUDIT_FILE = (
    AUDITS_DIR
    / "dataset"
    / "dataset_audit.csv"
) # Auditoría oficial del proceso de construcción del Dataset Científico

DATASET_AUDIT_LOG_FILE = (
    AUDITS_DIR
    / "dataset"
    / "dataset_audit.json"
) # Registro detallado de auditoría del Dataset Científico

# -----------------------------------------------------------------------------
# 5.2 GraphData
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 5.2.1 Artefactos Científicos del GraphData
# -----------------------------------------------------------------------------

GRAPH_FILES_DIR = (
    GRAPH_DATA_DIR
    / "graphs"
) # Directorio oficial de los GraphData por año

GRAPH_NODE_FEATURES_DIR = (
    GRAPH_DATA_DIR
    / "node_features"
) # Directorio oficial de las Node Features

GRAPH_EDGE_INDEX_DIR = (
    GRAPH_DATA_DIR
    / "edge_index"
) # Directorio oficial de los Edge Index

GRAPH_EDGE_WEIGHTS_DIR = (
    GRAPH_DATA_DIR
    / "edge_weights"
) # Directorio oficial de los pesos de las aristas

GRAPH_DATA_COLLECTION_FILE = (
    GRAPH_DATA_DIR
    / "graph_data_collection.pt"
) # Colección oficial de GraphData

NODE_CATALOG_FILE = (
    GRAPH_DATA_DIR
    / "node_catalog.parquet"
) # Catálogo Maestro de Nodos

GRAPH_EDGE_INDEX_FILE = (
    GRAPH_DATA_DIR
    / "edge_index.parquet"
) # Índices oficiales de las aristas

GRAPH_EDGE_FEATURES_FILE = (
    GRAPH_DATA_DIR
    / "edge_features.parquet"
) # Atributos oficiales de las aristas (edge_attr)

GRAPH_EDGE_WEIGHTS_FILE = (
    GRAPH_DATA_DIR
    / "edge_weights.parquet"
) # Pesos oficiales de las aristas

GRAPH_TOPOLOGY_FILE = (
    GRAPH_DATA_DIR
    / "graph_topology.json"
) # Información estructural del grafo

GRAPH_SCHEMA_FILE = (
    GRAPH_DATA_DIR
    / "graph_schema.json"
) # Esquema oficial del GraphData

PANEL_YEARS_FILE = (
    GRAPH_DATA_DIR
    / "panel_years.json"
) # Años oficiales del panel científico

GRAPH_COLLECTION_METADATA_FILE = (
    GRAPH_DATA_DIR
    / "graph_collection_metadata.json"
) # Metadatos oficiales de la colección

GRAPH_COLLECTION_CERTIFICATE_FILE = (
    GRAPH_DATA_DIR
    / "graph_collection_certificate.json"
) # Certificado científico oficial de la colección

GRAPH_SUMMARY_FILE = (
    GRAPH_DATA_DIR
    / "graph_summary.json"
) # Resumen científico oficial de la colección GraphData

# -----------------------------------------------------------------------------
# 5.2.2 Metadatos
# -----------------------------------------------------------------------------

GRAPH_METADATA_FILE = (
    GRAPH_DATA_DIR
    / "graph_metadata.json"
) # Metadatos científicos del GraphData

# -----------------------------------------------------------------------------
# 5.2.3 Manifiestos
# -----------------------------------------------------------------------------

DATASET_MANIFEST_FILE = (
    R_MASTER_DIR
    / "dataset_manifest.json"
) # Manifiesto oficial del Dataset Científico

GRAPH_MANIFEST_FILE = (
    GRAPH_DATA_DIR
    / "graph_manifest.json"
) # Manifiesto oficial del GraphData

# -----------------------------------------------------------------------------
# 5.2.4 Contrato del GraphData
# -----------------------------------------------------------------------------

GRAPH_CONTRACT_FILE = (
    GRAPH_DATA_DIR
    / "graph_contract.json"
) # Contrato oficial del GraphData

# -----------------------------------------------------------------------------
# 5.2.5 Reportes de Validación del GraphData
# -----------------------------------------------------------------------------

GRAPH_VALIDATION_DIR = (
    REPORTS_DIR
    / "graph_validation"
) # Reportes oficiales de validación del GraphData

GRAPH_VALIDATION_FILE = (
    GRAPH_VALIDATION_DIR
    / "graph_validation.json"
) # Validación estructural del GraphData

GRAPH_TOPOLOGY_VALIDATION_FILE = (
    GRAPH_VALIDATION_DIR
    / "graph_topology_validation.json"
) # Validación topológica del grafo

# -----------------------------------------------------------------------------
# 5.2.6 Reportes Estadísticos del GraphData
# -----------------------------------------------------------------------------

GRAPH_STATISTICS_DIR = (
    REPORTS_DIR
    / "graph_statistics"
) # Reportes estadísticos del grafo

GRAPH_STATISTICS_FILE = (
    GRAPH_STATISTICS_DIR
    / "graph_statistics.json"
) # Resumen estadístico del grafo

GRAPH_DEGREE_DISTRIBUTION_FILE = (
    GRAPH_STATISTICS_DIR
    / "degree_distribution.csv"
) # Distribución de grados

GRAPH_NODE_STATISTICS_FILE = (
    GRAPH_STATISTICS_DIR
    / "node_statistics.csv"
) # Estadísticas por nodo

GRAPH_COMPONENT_STATISTICS_FILE = (
    GRAPH_STATISTICS_DIR
    / "component_statistics.json"
) # Componentes conectados

GRAPH_TOPOLOGY_SUMMARY_FILE = (
    GRAPH_STATISTICS_DIR
    / "topology_summary.json"
) # Resumen topológico

GRAPH_CENTRALITY_STATISTICS_FILE = (
    GRAPH_STATISTICS_DIR
    / "centrality_statistics.csv"
) # Estadísticas de centralidad

GRAPH_CLUSTERING_STATISTICS_FILE = (
    GRAPH_STATISTICS_DIR
    / "clustering_statistics.json"
) # Estadísticas de clustering

GRAPH_TRIADS_STATISTICS_FILE = (
    GRAPH_STATISTICS_DIR
    / "triads_statistics.json"
) # Estadísticas de tríadas

GRAPH_PATH_STATISTICS_FILE = (
    GRAPH_STATISTICS_DIR
    / "path_statistics.json"
) # Estadísticas de caminos mínimos, diámetro y radio

# -----------------------------------------------------------------------------
# 5.2.7 Certificación del GraphData
# -----------------------------------------------------------------------------

GRAPH_CERTIFICATES_DIR = (
    CERTIFICATES_DIR
    / "graph_data"
) # Certificados científicos del GraphData

GRAPH_CERTIFICATE_FILE = (
    GRAPH_CERTIFICATES_DIR
    / "graph_certificate.json"
) # Certificado científico del GraphData

GRAPH_CERTIFICATE_SIGNATURE_FILE = (
    GRAPH_CERTIFICATES_DIR
    / "graph_signature.json"
) # Firma digital del certificado científico

# -----------------------------------------------------------------------------
# 5.2.8 Auditoría del GraphData
# -----------------------------------------------------------------------------

GRAPH_AUDITS_DIR = (
    AUDITS_DIR
    / "graph_data"
) # Auditorías del proceso de construcción del GraphData

GRAPH_AUDIT_FILE = (
    GRAPH_AUDITS_DIR
    / "graph_audit.csv"
) # Auditoría completa del GraphData

GRAPH_AUDIT_LOG_FILE = (
    GRAPH_AUDITS_DIR
    / "graph_audit.json"
) # Registro detallado de auditoría del GraphData

# -----------------------------------------------------------------------------
# 5.3 Benchmark Científico
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 5.3.1 Artefactos Científicos del Benchmark
# -----------------------------------------------------------------------------

BENCHMARK_EXPERIMENT_FILE = (
    BENCHMARK_DIR
    / "benchmark_experiment.joblib"
) # Experimento científico completo del Benchmark

BENCHMARK_MODELS_FILE = (
    BENCHMARK_DIR
    / "benchmark_models.csv"
) # Catálogo oficial de modelos evaluados

# -----------------------------------------------------------------------------
# 5.3.2 Ranking Científico
# -----------------------------------------------------------------------------

BENCHMARK_RANKING_CSV_FILE = (
    BENCHMARK_DIR
    / "benchmark_ranking.csv"
) # Ranking científico de modelos

BENCHMARK_RANKING_XLSX_FILE = (
    BENCHMARK_DIR
    / "benchmark_ranking.xlsx"
) # Ranking científico en formato Excel

# -----------------------------------------------------------------------------
# 5.3.3 Métricas Científicas
# -----------------------------------------------------------------------------

BENCHMARK_METRICS_FILE = (
    BENCHMARK_DIR
    / "benchmark_metrics.parquet"
) # Métricas oficiales del Benchmark

BENCHMARK_METRICS_CSV_FILE = (
    BENCHMARK_DIR
    / "benchmark_metrics.csv"
) # Exportación de métricas en formato CSV

BENCHMARK_MULTICRITERIA_FILE = (
    BENCHMARK_DIR
    / "benchmark_multicriteria.csv"
) # Resultados del análisis multicriterio

# -----------------------------------------------------------------------------
# 5.3.4 Resumen Científico
# -----------------------------------------------------------------------------

BENCHMARK_SUMMARY_CSV_FILE = (
    BENCHMARK_DIR
    / "benchmark_summary.csv"
) # Resumen científico en formato CSV

BENCHMARK_SUMMARY_XLSX_FILE = (
    BENCHMARK_DIR
    / "benchmark_summary.xlsx"
) # Resumen científico en formato Excel

BENCHMARK_SUMMARY_JSON_FILE = (
    BENCHMARK_DIR
    / "benchmark_summary.json"
) # Resumen científico en formato JSON

# -----------------------------------------------------------------------------
# 5.3.5 Metadatos del Benchmark
# -----------------------------------------------------------------------------

BENCHMARK_METADATA_FILE = (
    BENCHMARK_DIR
    / "benchmark_metadata.json"
) # Metadatos científicos del Benchmark

# -----------------------------------------------------------------------------
# 5.3.6 Contrato del Benchmark
# -----------------------------------------------------------------------------

BENCHMARK_CONTRACT_FILE = (
    BENCHMARK_DIR
    / "benchmark_contract.json"
) # Contrato oficial del Benchmark

# -----------------------------------------------------------------------------
# 5.3.7 Certificación del Benchmark
# -----------------------------------------------------------------------------

BENCHMARK_CERTIFICATES_DIR = (
    CERTIFICATES_DIR
    / "benchmark"
) # Certificados científicos del Benchmark

BENCHMARK_CERTIFICATE_FILE = (
    BENCHMARK_CERTIFICATES_DIR
    / "benchmark_certificate.json"
) # Certificado científico del Benchmark

BENCHMARK_CERTIFICATE_SIGNATURE_FILE = (
    BENCHMARK_CERTIFICATES_DIR
    / "benchmark_signature.json"
) # Firma digital del certificado científico

BENCHMARK_MANIFEST_FILE = (
    BENCHMARK_DIR
    / "benchmark_manifest.json"
) # Manifiesto oficial del Benchmark

# -----------------------------------------------------------------------------
# 5.3.8 Auditoría del Benchmark
# -----------------------------------------------------------------------------

BENCHMARK_AUDITS_DIR = (
    AUDITS_DIR
    / "benchmark"
) # Auditorías del Benchmark

BENCHMARK_AUDIT_FILE = (
    BENCHMARK_AUDITS_DIR
    / "benchmark_audit.csv"
) # Auditoría completa del Benchmark

BENCHMARK_AUDIT_LOG_FILE = (
    BENCHMARK_AUDITS_DIR
    / "benchmark_audit.json"
) # Registro detallado de auditoría del Benchmark

# -----------------------------------------------------------------------------
# 5.4 Entrenamiento del Modelo Oficial
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 5.4.1 Artefactos Científicos
# -----------------------------------------------------------------------------

OFFICIAL_MODEL_TORCH_FILE = (
    MODELS_DIR
    / "official_model.pt"
) # Modelo oficial entrenado en formato PyTorch

OFFICIAL_MODEL_JOBLIB_FILE = (
    MODELS_DIR
    / "official_model.joblib"
) # Modelo oficial serializado para inferencia

TRAINING_HISTORY_FILE = (
    TRAINING_DIR
    / "training_history.parquet"
) # Historial completo del entrenamiento

TRAINING_METRICS_FILE = (
    TRAINING_DIR
    / "training_metrics.parquet"
) # Métricas obtenidas durante el entrenamiento

TRAINING_CHECKPOINT_FILE = (
    TRAINING_DIR
    / "training_checkpoint.pt"
) # Punto de restauración del entrenamiento

# -----------------------------------------------------------------------------
# 5.4.2 Metadatos
# -----------------------------------------------------------------------------

TRAINING_METADATA_FILE = (
    TRAINING_DIR
    / "training_metadata.json"
) # Metadatos científicos del entrenamiento

# -----------------------------------------------------------------------------
# 5.4.3 Manifest
# -----------------------------------------------------------------------------

TRAINING_MANIFEST_FILE = (
    TRAINING_DIR
    / "training_manifest.json"
) # Manifiesto oficial del entrenamiento

# -----------------------------------------------------------------------------
# 5.4.4 Contrato
# -----------------------------------------------------------------------------

TRAINING_CONTRACT_FILE = (
    TRAINING_DIR
    / "training_contract.json"
) # Contrato oficial del entrenamiento

# -----------------------------------------------------------------------------
# 5.4.5 Reportes
# -----------------------------------------------------------------------------

TRAINING_SUMMARY_FILE = (
    TRAINING_DIR
    / "training_summary.json"
) # Resumen oficial del entrenamiento

TRAINING_LOG_FILE = (
    TRAINING_DIR
    / "training_log.json"
) # Registro detallado del entrenamiento

TRAINING_CURVES_FILE = (
    TRAINING_DIR
    / "training_curves.parquet"
) # Curvas de aprendizaje del entrenamiento

# -----------------------------------------------------------------------------
# 5.4.6 Certificación
# -----------------------------------------------------------------------------

TRAINING_CERTIFICATES_DIR = (
    CERTIFICATES_DIR
    / "training"
) # Certificados científicos del entrenamiento

TRAINING_CERTIFICATE_FILE = (
    TRAINING_CERTIFICATES_DIR
    / "training_certificate.json"
) # Certificado científico del entrenamiento

TRAINING_CERTIFICATE_SIGNATURE_FILE = (
    TRAINING_CERTIFICATES_DIR
    / "training_signature.json"
) # Firma digital del certificado científico

# -----------------------------------------------------------------------------
# 5.4.7 Auditoría
# -----------------------------------------------------------------------------

TRAINING_AUDITS_DIR = (
    AUDITS_DIR
    / "training"
) # Auditorías del entrenamiento

TRAINING_AUDIT_FILE = (
    TRAINING_AUDITS_DIR
    / "training_audit.csv"
) # Auditoría completa del entrenamiento

TRAINING_AUDIT_LOG_FILE = (
    TRAINING_AUDITS_DIR
    / "training_audit.json"
) # Registro detallado de auditoría del entrenamiento

# -----------------------------------------------------------------------------
# 5.5 Evaluación del Modelo
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 5.5.1 Artefactos Científicos
# -----------------------------------------------------------------------------

EVALUATION_RESULTS_FILE = (
    EVALUATION_DIR
    / "evaluation_results.parquet"
) # Resultados oficiales de la evaluación

VALIDATION_RESULTS_FILE = (
    EVALUATION_DIR
    / "validation_results.parquet"
) # Resultados oficiales de validación

PREDICTIONS_FILE = (
    EVALUATION_DIR
    / "predictions.parquet"
) # Valores observados, predichos y errores de predicción

FEATURE_IMPORTANCE_FILE = (
    EVALUATION_DIR
    / "feature_importance.parquet"
) # Importancia de variables

RESIDUALS_FILE = (
    EVALUATION_DIR
    / "residuals.parquet"
) # Residuos del modelo

SHAP_VALUES_FILE = (
    EVALUATION_DIR
    / "shap_values.joblib"
) # Valores SHAP para interpretabilidad

GNN_EMBEDDINGS_FILE = (
    EVALUATION_DIR
    / "gnn_embeddings.pt"
) # Embeddings generados por la GNN

# -----------------------------------------------------------------------------
# 5.5.2 Reportes
# -----------------------------------------------------------------------------

EVALUATION_REPORT_FILE = (
    EVALUATION_DIR
    / "evaluation_report.json"
) # Reporte científico de evaluación

EVALUATION_SUMMARY_FILE = (
    EVALUATION_DIR
    / "evaluation_summary.json"
) # Resumen ejecutivo de la evaluación

METRICS_BY_YEAR_FILE = (
    EVALUATION_DIR
    / "metrics_by_year.parquet"
) # Métricas por año del panel

# -----------------------------------------------------------------------------
# 5.5.3 Metadatos
# -----------------------------------------------------------------------------

EVALUATION_METADATA_FILE = (
    EVALUATION_DIR
    / "evaluation_metadata.json"
) # Metadatos científicos de la evaluación

# -----------------------------------------------------------------------------
# 5.5.4 Manifest
# -----------------------------------------------------------------------------

EVALUATION_MANIFEST_FILE = (
    EVALUATION_DIR
    / "evaluation_manifest.json"
) # Manifiesto oficial de la evaluación

# -----------------------------------------------------------------------------
# 5.5.5 Contrato
# -----------------------------------------------------------------------------

EVALUATION_CONTRACT_FILE = (
    EVALUATION_DIR
    / "evaluation_contract.json"
) # Contrato oficial de la evaluación

# -----------------------------------------------------------------------------
# 5.5.6 Certificación
# -----------------------------------------------------------------------------

EVALUATION_CERTIFICATES_DIR = (
    CERTIFICATES_DIR
    / "evaluation"
) # Certificados científicos de la evaluación

EVALUATION_CERTIFICATE_FILE = (
    EVALUATION_CERTIFICATES_DIR
    / "evaluation_certificate.json"
) # Certificado científico de la evaluación

EVALUATION_CERTIFICATE_SIGNATURE_FILE = (
    EVALUATION_CERTIFICATES_DIR
    / "evaluation_signature.json"
) # Firma digital del certificado científico

# -----------------------------------------------------------------------------
# 5.5.7 Auditoría
# -----------------------------------------------------------------------------

EVALUATION_AUDITS_DIR = (
    AUDITS_DIR
    / "evaluation"
) # Auditorías del proceso de evaluación

EVALUATION_AUDIT_FILE = (
    EVALUATION_AUDITS_DIR
    / "evaluation_audit.csv"
) # Auditoría completa de la evaluación

EVALUATION_AUDIT_LOG_FILE = (
    EVALUATION_AUDITS_DIR
    / "evaluation_audit.json"
) # Registro detallado de auditoría de la evaluación

# -----------------------------------------------------------------------------
# 5.6 Forecasting
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 5.6.1 Artefactos Científicos
# -----------------------------------------------------------------------------

FORECAST_RESULTS_FILE = (
    FORECASTING_DIR
    / "forecast_results.parquet"
) # Predicciones oficiales generadas por el modelo

SCENARIO_RESULTS_FILE = (
    FORECASTING_DIR
    / "scenario_results.parquet"
) # Resultados de escenarios prospectivos

FORECAST_PANEL_FILE = (
    FORECASTING_DIR
    / "forecast_panel.parquet"
) # Panel científico con los valores pronosticados

FORECAST_MAP_FILE = (
    FORECASTING_DIR
    / "forecast_map.parquet"
) # Resultados espaciales utilizados por la Plataforma GeoAI

FORECAST_UNCERTAINTY_FILE = (
    FORECASTING_DIR
    / "forecast_uncertainty.parquet"
) # Intervalos de confianza, cuantiles e incertidumbre de las predicciones

# -----------------------------------------------------------------------------
# 5.6.2 Reportes
# -----------------------------------------------------------------------------

FORECAST_REPORT_FILE = (
    FORECASTING_DIR
    / "forecast_report.pdf"
) # Reporte científico del Forecasting

FORECAST_SUMMARY_FILE = (
    FORECASTING_DIR
    / "forecast_summary.json"
) # Resumen ejecutivo del Forecasting

FORECAST_SCENARIOS_FILE = (
    FORECASTING_DIR
    / "forecast_scenarios.json"
) # Descripción de los escenarios evaluados

# -----------------------------------------------------------------------------
# 5.6.3 Metadatos
# -----------------------------------------------------------------------------

FORECAST_METADATA_FILE = (
    FORECASTING_DIR
    / "forecast_metadata.json"
) # Metadatos científicos del Forecasting

# -----------------------------------------------------------------------------
# 5.6.4 Manifest
# -----------------------------------------------------------------------------

FORECAST_MANIFEST_FILE = (
    FORECASTING_DIR
    / "forecast_manifest.json"
) # Manifiesto oficial del Forecasting

# -----------------------------------------------------------------------------
# 5.6.5 Contrato
# -----------------------------------------------------------------------------

FORECAST_CONTRACT_FILE = (
    FORECASTING_DIR
    / "forecast_contract.json"
) # Contrato oficial del Forecasting

# -----------------------------------------------------------------------------
# 5.6.6 Certificación
# -----------------------------------------------------------------------------

FORECAST_CERTIFICATES_DIR = (
    CERTIFICATES_DIR
    / "forecasting"
) # Certificados científicos del Forecasting

FORECAST_CERTIFICATE_FILE = (
    FORECAST_CERTIFICATES_DIR
    / "forecast_certificate.json"
) # Certificado científico del Forecasting

FORECAST_CERTIFICATE_SIGNATURE_FILE = (
    FORECAST_CERTIFICATES_DIR
    / "forecast_signature.json"
) # Firma digital del certificado científico

# -----------------------------------------------------------------------------
# 5.6.7 Auditoría
# -----------------------------------------------------------------------------

FORECAST_AUDITS_DIR = (
    AUDITS_DIR
    / "forecasting"
) # Auditorías del proceso de Forecasting

FORECAST_AUDIT_FILE = (
    FORECAST_AUDITS_DIR
    / "forecast_audit.csv"
) # Auditoría completa del Forecasting

FORECAST_AUDIT_LOG_FILE = (
    FORECAST_AUDITS_DIR
    / "forecast_audit.json"
) # Registro detallado de auditoría del Forecasting

# -----------------------------------------------------------------------------
# BLOQUE 6. Catálogo Oficial de Directorios
# -----------------------------------------------------------------------------
# Objetivo:
# Centralizar el acceso a todos los directorios oficiales del proyecto mediante
# una arquitectura jerárquica organizada por dominios funcionales, facilitando
# la reutilización, el mantenimiento y la escalabilidad del pipeline científico.
#
# Organización:
#   6.1 Directorios Generales
#   6.2 Ingeniería de Datos (R)
#   6.3 Pipeline Científico (Python)
#   6.4 Figuras Científicas
#   6.5 Plataforma GeoAI
# -----------------------------------------------------------------------------

PROJECT_DIRECTORIES = {

    # -------------------------------------------------------------------------
    # 6.1 Directorios Generales
    # -------------------------------------------------------------------------

    "general": {
        "root": ROOT_DIR,
        "data": DATA_DIR,
        "config": CONFIG_DIR,
        "raw": RAW_DIR,
        "interim": INTERIM_DIR,
        "processed": PROCESSED_DIR,
        "reports": REPORTS_DIR,
        "logs": LOGS_DIR,
        "docs": DOCS_DIR
    },

    # -------------------------------------------------------------------------
    # 6.2 Ingeniería de Datos (R)
    # -------------------------------------------------------------------------

    "r_pipeline": {
        "root": R_DIR,
        "master": R_MASTER_DIR,
        "reports": R_REPORTS_DIR,
        "logs": R_LOGS_DIR,
        "metadata": R_METADATA_DIR,
        "contracts": R_CONTRACTS_DIR,
        "certificates": R_CERTIFICATES_DIR,
        "audits": R_AUDITS_DIR
    },

    # -------------------------------------------------------------------------
    # 6.3 Pipeline Científico (Python)
    # -------------------------------------------------------------------------

    "python_pipeline": {
        "root": OUTPUTS_DIR,
        "graph_data": GRAPH_DATA_DIR,
        "benchmark": BENCHMARK_DIR,
        "training": TRAINING_DIR,
        "evaluation": EVALUATION_DIR,
        "forecasting": FORECASTING_DIR,
        "models": MODELS_DIR,
        "metadata": METADATA_DIR,
        "contracts": CONTRACTS_DIR,
        "certificates": CERTIFICATES_DIR,
        "audits": AUDITS_DIR
    },

    # -------------------------------------------------------------------------
    # 6.5 Plataforma GeoAI
    # -------------------------------------------------------------------------

    "geoai": {
        "root": GEOAI_DIR,
        "reports": GEOAI_REPORTS_DIR,
        "data": GEOAI_DATA_DIR,
        "logs": GEOAI_LOGS_DIR,
        "dashboard": GEOAI_DASHBOARD_DIR,
        "api": GEOAI_API_DIR,
        "agents": GEOAI_AGENTS_DIR,
        "export": GEOAI_EXPORT_DIR,
        "cache": GEOAI_CACHE_DIR,
        "config": GEOAI_CONFIG_DIR
    }
}

# -----------------------------------------------------------------------------
# BLOQUE 7. Validación de la Estructura del Proyecto
# -----------------------------------------------------------------------------
# Objetivo:
# Crear y validar automáticamente toda la estructura oficial de directorios del
# proyecto definida en PROJECT_DIRECTORIES.
# -----------------------------------------------------------------------------

def get_project_directories() -> list[Path]:
    """
    Devuelve una lista plana con todos los directorios oficiales del proyecto.
    """
    directories = []
    for group in PROJECT_DIRECTORIES.values():
        directories.extend(group.values())
    return directories


def validate_directory(
    directory: Path,
    verbose: bool = True,
) -> None:
    """
    Crea un directorio oficial del proyecto si no existe.
    """

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if verbose:
        print(f"Directorio verificado: {directory}")


def validate_project_structure(
    verbose: bool = True,
) -> bool:
    """
    Crea y valida la estructura oficial de directorios del proyecto.
    """

    try:
        for directory in get_project_directories():

            validate_directory(
                directory=directory,
                verbose=verbose,
            )

        if verbose:
            print("Estructura oficial del proyecto validada correctamente.")

        return True

    except Exception as error:

        if verbose:
            print(f"Error al validar la estructura del proyecto: {error}")

        return False

# -----------------------------------------------------------------------------
# BLOQUE 8. Funciones Auxiliares
# -----------------------------------------------------------------------------
# Objetivo:
# Proporcionar funciones auxiliares para acceder de forma segura y uniforme a
# los directorios oficiales definidos en PROJECT_DIRECTORIES, desacoplando el
# resto del proyecto de la estructura interna del catálogo.
# -----------------------------------------------------------------------------

def get_directory_group(
    group: str,
) -> dict[str, Path]:
    """
    Devuelve un grupo de directorios del catálogo oficial.
    """
    return PROJECT_DIRECTORIES[group]

def get_directory(
    group: str,
    name: str,
) -> Path:
    """
    Devuelve un directorio específico perteneciente a un grupo.
    """
    return PROJECT_DIRECTORIES[group][name]

def get_project_directory() -> Path:
    """
    Devuelve el directorio raíz del proyecto.
    """
    return get_directory(
        "general",
        "root",
    )

def get_module_directory(
    module: str,
) -> Path:
    """
    Devuelve el directorio raíz de un módulo del pipeline científico.
    """
    return get_directory(
        "python_pipeline",
        module,
    )

def get_shared_directory(
    name: str,
) -> Path:
    """
    Devuelve un directorio compartido del pipeline científico.
    """
    return get_directory(
        "python_pipeline",
        name,
    )
