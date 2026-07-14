# config_project.py

# BLOQUE 1. Configuración General ---------------------------------------------
from pathlib import Path # Manejo de rutas

# Información general del proyecto
PROJECT_NAME = "proyecto-gnn-agricola" # Nombre oficial del proyecto
PROJECT_VERSION = "3.0.0" # Versión del proyecto
PROJECT_DESCRIPTION = "Modelado espacio-temporal de la soberanía alimentaria mediante Graph Neural Networks" # Descripción del proyecto

# Configuración de reproducibilidad
SEED = 5477976 # Semilla global del proyecto

# Configuración temporal del panel
ANIO_INICIO = 2006 # Año inicial del panel
ANIO_FIN = 2018 # Año final del panel
N_ANIOS = ANIO_FIN - ANIO_INICIO + 1 # Número de años del panel
PANEL_YEARS = list(range(ANIO_INICIO, ANIO_FIN + 1)) # Años oficiales del panel

# Configuración territorial
N_MUNICIPIOS = 1121 # Número oficial de municipios
N_OBS_PANEL = N_MUNICIPIOS * N_ANIOS # Número esperado de observaciones

# Configuración espacial
CRS_EPSG = 4686 # Sistema de referencia oficial MAGNA-SIRGAS
GEOMETRY_COLUMN = "geometry" # Nombre de la columna de geometría

# BLOQUE 2. Dataset Científico -----------------------------------------------
# Información del dataset
DATASET_NAME = "dataset_gnn_certificado.parquet" # Nombre oficial del dataset
DATASET_VERSION = "1.0.0" # Versión del dataset
DATASET_FORMAT = "parquet" # Formato oficial
PANEL_ID_COLUMN = "panel_id" # Identificador del panel
NODE_ID_COLUMN = "cod_mpio" # Identificador del nodo
TIME_COLUMN = "anio" # Variable temporal
TARGET_VARIABLE = "log_rendimiento" # Variable objetivo

# Variables identificadoras
ID_COLUMNS = [
    PANEL_ID_COLUMN,
    NODE_ID_COLUMN,
    "municipio",
    "cod_depto",
    "departamento"
] # Variables identificadoras

# Variables espaciales
SPATIAL_COLUMNS = [
    "latitud",
    "longitud"
] # Variables espaciales

# Columnas estructurales
STRUCTURAL_COLUMNS = (
    ID_COLUMNS
    + [TIME_COLUMN]
    + SPATIAL_COLUMNS
) # Estructura oficial del dataset

# Variables excluidas del modelado
EXCLUDED_COLUMNS = (
    ID_COLUMNS
    + [TIME_COLUMN]
    + [TARGET_VARIABLE]
) # Variables que no forman parte de las Node Features

# Variables predictoras
NODE_FEATURE_COLUMNS = [
    column
    for column in STRUCTURAL_COLUMNS
    if column not in EXCLUDED_COLUMNS
] # Variables predictoras del modelo GNN

GRAPH_NODE_FEATURES = NODE_FEATURE_COLUMNS # Variables oficiales utilizadas como Node Features del grafo

# BLOQUE 3. Construcción del GraphData ------------------------------------
## Objetivo: Definir la configuración científica utilizada para construir el GraphData oficial del proyecto.
## Contenido: 3.1 Configuración de Nodos 3.2 Configuración de Node Features 3.3 Configuración Espacial del Grafo
# 3.4 Configuración General del GraphData 3.5 Configuración de Validación 3.6 Productos Oficiales
# 3.7 Configuración de Vecindad Espacial

# 3.1 Configuración de Nodos ----------------------------------------------
NODE_CONFIG = {
    "node_id_column": NODE_ID_COLUMN,
    "graph_node_id": "node_id",
    "time_column": TIME_COLUMN,
    "target_column": TARGET_VARIABLE
} # Configuración oficial de los nodos

# 3.2 Configuración Espacial del Grafo ------------------------------------
GRAPH_SPATIAL_CONFIG = {
    "method": "queen", # Método de construcción de la vecindad espacial
    "crs": CRS_EPSG, # Sistema de referencia espacial
    "geometry_column": GEOMETRY_COLUMN, # Columna de geometría
    "latitude_column": "latitud", # Columna de latitud
    "longitude_column": "longitud", # Columna de longitud
    "require_geometry": True, # Requerir geometría para construir el grafo
    "validate_geometry": True, # Validar geometrías antes del procesamiento
    "allow_multipart": True, # Permitir geometrías multipartes
    "remove_self_loops": True, # Eliminar autoconexiones
    "remove_duplicate_edges": True # Eliminar aristas duplicadas
} # Configuración espacial del grafo

# 3.3 Configuración GRAPH_NODE_ATTRIBUTES ------------------------------------
GRAPH_NODE_ATTRIBUTES = [
    NODE_CONFIG["node_id_column"],
    "municipio",
    "cod_depto",
    "departamento",
    "latitud",
    "longitud",
    GRAPH_SPATIAL_CONFIG["geometry_column"]
] # Variables utilizadas para construir el catálogo oficial de nodos

# 3.4 Configuración de Node Features --------------------------------------
NODE_FEATURE_CONFIG = {
    "id_column": NODE_ID_COLUMN,
    "time_column": TIME_COLUMN,
    "target_column": TARGET_VARIABLE,
    "feature_columns": NODE_FEATURE_COLUMNS,
    "normalize": True
} # Configuración de las características de los nodos

# Variables predictoras del modelo GNN
NODE_FEATURE_COLUMNS = [
    column
    for column in STRUCTURAL_COLUMNS
    if column not in (
        ID_COLUMNS
        + [TIME_COLUMN]
        + [TARGET_VARIABLE]
    )
] # Variables utilizadas como Node Features

# 3.5 Configuración General del GraphData ---------------------------------
GRAPH_CONFIG = {
    "graph_type": "spatial", # Tipo de grafo
    "directed": False, # Grafo dirigido
    "weighted": False, # Grafo ponderado
    "normalize_features": True, # Normalizar Node Features
    "normalize_edge_weights": False, # Normalizar pesos de las aristas
    "include_edge_weights": False, # Incluir edge_weight en GraphData
    "edge_index_dtype": "int64" # Tipo de dato para edge_index
} # Configuración general del GraphData 

# 3.6 Configuración de Validación -----------------------------------------
GRAPH_VALIDATION = {
    "check_duplicates": True,
    "check_missing_nodes": True,
    "check_connectivity": True,
    "check_edge_weights": True,
    "check_features": True,
    "check_target": True
} # Validaciones oficiales del GraphData

# 3.6 Productos Oficiales -------------------------------------------------
GRAPH_OUTPUT = "graph_data.pt" # Producto oficial del script 01_build_graph.py

# 3.7 Configuración de Vecindad Espacial ----------------------------------
GRAPH_NEIGHBORS = {
    "method": GRAPH_SPATIAL_CONFIG["method"],
    "k_neighbors": None,
    "distance_threshold": None,
    "use_cache": True
} # Configuración para la construcción de la vecindad espacial

# BLOQUE Configuración del Benchmark ----------------------------------------------
BENCHMARK_CONFIG = {
    "random_state": SEED,
    "train_size": 0.70,
    "validation_size": 0.15,
    "test_size": 0.15,
    "cv_folds": 5,
    "shuffle": True,
    "n_jobs": -1,
    "ranking_metric": "rmse",
    "save_predictions": True,
    "save_metrics": True,
    "save_models": False
} # Configuración general del benchmark

# Configuración oficial del dataset ----------------------------------------
DATASET_CONFIG = {
    "target_variable": "log_rendimiento",
    "excluded_columns": [
        "panel_id",
        "cod_mpio",
        "municipio",
        "cod_depto",
        "departamento",
        "geometry"
    ]
} # Configuración oficial del dataset

# Modelos candidatos del benchmark -----------------------------------------
MODEL_CANDIDATES = {

    "statistical": [
        "linear_regression"
    ], # Modelos estadísticos

    "machine_learning": [
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost"
    ], # Modelos de Machine Learning

    "deep_learning": [
        "mlp"
    ], # Redes neuronales tradicionales

    "graph_neural_networks": [
        "gcn",
        "graphsage",
        "gat",
        "gin",
        "tagcn"
    ] # Graph Neural Networks

} # Modelos candidatos del benchmark

# Identificadores oficiales de los modelos
MODEL_CODES = {

    "ST01": "linear_regression",

    "ML01": "random_forest",
    "ML02": "xgboost",
    "ML03": "lightgbm",
    "ML04": "catboost",

    "DL01": "mlp",

    "GNN01": "gcn",
    "GNN02": "graphsage",
    "GNN03": "gat",
    "GNN04": "gin",
    "GNN05": "tagcn"    
} # Identificadores oficiales

# Métricas oficiales del benchmark
BENCHMARK_METRICS = {
    "prediction": [
        "rmse",
        "mae",
        "mape",
        "r2",
        "adjusted_r2"
    ],

    "computational": [
        "training_time",
        "inference_time",
        "memory_usage",
        "num_parameters"
    ],

    "robustness": [
        "cross_validation_mean",
        "cross_validation_std"
    ]

} # Métricas oficiales del benchmark

# Criterios para seleccionar el mejor modelo
MODEL_SELECTION = {
    "primary_metric": "rmse",
    "secondary_metric": "mae",
    "criterion": "min"
} # Configuración de selección del modelo

# Configuración de reproducibilidad
BENCHMARK_REPRODUCIBILITY = {
    "random_seeds": [SEED], # Semillas utilizadas durante el benchmark
    "deterministic": True, # Forzar operaciones determinísticas para garantizar la reproducibilidad
    "save_predictions": True, # Guardar predicciones de cada modelo
    "save_models": False, # No almacenar todos los modelos entrenados
    "save_results": True # Guardar resultados consolidados del benchmark
} # Configuración de reproducibilidad

# BLOQUE 5. Entrenamiento del Modelo ------------------------------------------
# Configuración general del entrenamiento
TRAIN_CONFIG = {
    "preferred_device": "cuda", # Dispositivo preferido (cuda o cpu)
    "epochs": 300, # Número máximo de épocas
    "batch_size": 64, # Tamaño del lote
    "learning_rate": 0.001, # Tasa de aprendizaje
    "weight_decay": 0.0001, # Regularización L2
    "optimizer": "Adam", # Optimizador
    "loss_function": "MSELoss", # Función de pérdida
    "scheduler": "ReduceLROnPlateau", # Scheduler de la tasa de aprendizaje
    "gradient_clipping": True, # Activar Gradient Clipping
    "clip_value": 1.0, # Valor máximo del gradiente
    "mixed_precision": False, # Entrenamiento de precisión mixta (AMP)
    "num_workers": 4, # Procesos para cargar datos
    "pin_memory": True, # Optimizar transferencia CPU-GPU
    "persistent_workers": False, # No Mantener workers activos
    "random_state": SEED # Semilla del entrenamiento
} # Configuración general del entrenamiento

# Configuración de Early Stopping
EARLY_STOPPING_CONFIG = {
    "enabled": True, # Activar Early Stopping
    "monitor": "validation_loss", # Métrica monitoreada
    "patience": 30, # Número máximo de épocas sin mejora
    "min_delta": 0.0001, # Mejora mínima considerada
    "restore_best_weights": True # Restaurar automáticamente el mejor modelo
} # Configuración de Early Stopping

# Configuración de checkpoints
CHECKPOINT_CONFIG = {
    "enabled": True, # Activar guardado de checkpoints
    "save_best_model": True, # Guardar únicamente el mejor modelo
    "save_last_model": False, # Guardar el último modelo entrenado
    "save_training_history": True, # Guardar historial del entrenamiento
    "save_embeddings": True # Guardar embeddings finales
} # Configuración de checkpoints

# Archivos oficiales generados
TRAIN_OUTPUTS = {
    "best_model": "best_model.pt", # Modelo oficial entrenado
    "checkpoint": "checkpoint.pt", # Punto de restauración
    "history": "training_history.parquet", # Historial del entrenamiento
    "embeddings": "embeddings.parquet" # Embeddings finales
} # Productos oficiales del entrenamiento

# BLOQUE 6. Evaluación y Explicabilidad ---------------------------------------
# Configuración general de la evaluación
EVALUATION_CONFIG = {
    "evaluate_train": True, # Evaluar conjunto de entrenamiento
    "evaluate_validation": True, # Evaluar conjunto de validación
    "evaluate_test": True, # Evaluar conjunto de prueba
    "save_predictions": True, # Guardar predicciones
    "save_residuals": True, # Guardar residuos
    "save_metrics": True # Guardar métricas
} # Configuración general de la evaluación

# Métricas oficiales de evaluación --------------------------------------------
EVALUATION_METRICS = {

    "prediction": [
        "rmse",
        "mae",
        "mape",
        "r2",
        "adjusted_r2"
    ], # Métricas de desempeño predictivo

    "computational": [
        "training_time",
        "inference_time",
        "memory_usage",
        "num_parameters"
    ], # Métricas de desempeño computacional

    "robustness": [
        "cross_validation_mean",
        "cross_validation_std",
        "seed_variability"
    ], # Métricas de robustez

    "spatial": [
        "moran_i_residuals",
        "spatial_bias"
    ], # Métricas espaciales

    "graph": [
        "graph_density",
        "average_degree",
        "connected_components",
        "isolated_nodes"
    ] # Métricas estructurales del grafo

} # Métricas oficiales de evaluación

# Configuración de explicabilidad
EXPLAINABILITY_CONFIG = {
    "feature_importance": True, # Calcular importancia de variables
    "shap": True, # Calcular valores SHAP
    "attention_weights": "auto", # Exportar pesos de atención si el modelo los soporta
    "embeddings": True, # Exportar embeddings finales
    "uncertainty": False # Calcular incertidumbre predictiva
} # Configuración de explicabilidad

# Configuración de visualizaciones
PLOT_CONFIG = {
    "prediction_plot": True, # Observado vs predicho
    "residual_plot": True, # Distribución de residuos
    "loss_curve": True, # Curva de pérdida
    "learning_curve": True, # Curva de aprendizaje
    "feature_importance_plot": True, # Importancia de variables
    "shap_summary_plot": True, # Resumen SHAP
    "attention_heatmap": True # Mapa de calor de atención
} # Figuras generadas automáticamente

# Archivos oficiales generados
EVALUATION_OUTPUTS = {
    "metrics": "evaluation_results.parquet", # Métricas completas
    "metrics_summary": "metrics_summary.csv", # Resumen de métricas
    "predictions": "predictions.parquet", # Predicciones del modelo
    "residuals": "prediction_residuals.parquet", # Residuos del modelo
    "feature_importance": "feature_importance.parquet", # Importancia de variables
    "shap": "shap_values.parquet", # Valores SHAP
    "attention": "attention_weights.parquet", # Pesos de atención
    "report": "evaluation_report.pdf" # Reporte técnico
} # Productos oficiales de evaluación

# BLOQUE 7. Forecasting y Escenarios -------------------------------------------------------
FORECAST_HORIZON = 5 # Horizonte de pronóstico
FORECAST_START_YEAR = ANIO_FIN + 1 # Primer año proyectado
FORECAST_END_YEAR = FORECAST_START_YEAR + FORECAST_HORIZON - 1 # Último año proyectado

# Configuración general del forecasting
FORECAST_CONFIG = {
    "forecast_horizon": FORECAST_HORIZON, # Horizonte del pronóstico
    "forecast_frequency": "annual", # Frecuencia temporal
    "forecast_start_year": FORECAST_START_YEAR, # Primer año proyectado
    "forecast_end_year": FORECAST_END_YEAR, # Último año proyectado
    "historical_years": N_ANIOS, # Número de años históricos utilizados
    "target_variable": TARGET_VARIABLE, # Variable objetivo
    "random_state": SEED # Semilla para reproducibilidad
}

# Configuración de escenarios
SCENARIO_CONFIG = {
    "baseline": True, # Escenario base
    "optimistic": True, # Escenario optimista
    "pessimistic": True # Escenario pesimista
} # Escenarios de simulación

# Configuración de predicción
PREDICTION_CONFIG = {
    "recursive_forecast": True, # Utilizar predicción recursiva
    "save_node_predictions": True, # Guardar predicciones por municipio
    "save_graph_predictions": True, # Guardar resultados del grafo
    "calculate_uncertainty": False # Calcular incertidumbre predictiva
} # Configuración de la predicción

# Archivos oficiales generados
FORECAST_OUTPUTS = {
    "forecast": "forecast_panel.parquet", # Pronósticos del panel
    "scenarios": "scenario_results.parquet", # Resultados de escenarios
    "maps": "forecast_maps.gpkg", # Mapas de pronóstico
    "report": "forecast_report.pdf" # Reporte técnico
} # Productos oficiales del forecasting

# BLOQUE 8. Plataforma GeoAI -----------------------------------------------
# Configuración del Dashboard
DASHBOARD_CONFIG = {
    "enabled": True, # Habilitar Dashboard GeoAI
    "interactive_map": True, # Mostrar mapa interactivo
    "display_predictions": True, # Mostrar predicciones
    "display_scenarios": True, # Mostrar escenarios
    "display_explainability": True, # Mostrar resultados de explicabilidad
    "display_metrics": True, # Mostrar métricas del modelo
    "display_graph": True # Mostrar visualización del grafo
} # Configuración del Dashboard

# Configuración de reportes automáticos
REPORT_CONFIG = {
    "generate_pdf": True, # Generar reporte en PDF
    "generate_excel": True, # Exportar resultados a Excel
    "generate_html": True, # Generar reporte HTML
    "include_maps": True, # Incluir mapas
    "include_figures": True # Incluir gráficos
} # Configuración de reportes

# Configuración de la API
API_CONFIG = {
    "enabled": True, # Habilitar API
    "host": "0.0.0.0", # Dirección del servidor
    "port": 8000, # Puerto de la API
    "docs": True # Habilitar documentación automática
} # Configuración de la API

# Configuración de agentes inteligentes
AI_AGENT_CONFIG = {
    "data_analysis": True, # Agente de análisis de datos
    "forecast_analysis": True, # Agente de análisis del forecasting
    "graph_analysis": True, # Agente de análisis del grafo
    "recommendations": True, # Agente de recomendaciones
    "report_generation": True # Agente generador de reportes
} # Configuración de agentes IA

# Archivos oficiales generados
GEOAI_OUTPUTS = {
    "dashboard": "geoai_dashboard", # Aplicación del Dashboard
    "api": "geoai_api", # Servicio de la API
    "reports": "reports/", # Reportes generados
    "logs": "logs/", # Registros de ejecución
    "recommendations": "recommendations.parquet" # Recomendaciones generadas por IA
} # Productos oficiales de la Plataforma GeoAI

PROJECT_STRUCTURE = {
    "dataset": DATASET_NAME,
    "graph": GRAPH_OUTPUT,
    "model": TRAIN_OUTPUTS["best_model"],
    "evaluation": EVALUATION_OUTPUTS["metrics"],
    "forecast": FORECAST_OUTPUTS["forecast"]
} # Productos oficiales del pipeline

# BLOQUE 9. Estándares del Proyecto -------------------------------------------
# Formatos oficiales
FILE_FORMATS = {
    "dataset": "parquet", # Formato oficial de los datasets
    "graph": "pt", # Formato oficial del GraphData
    "report": "pdf", # Formato oficial de los reportes
    "table": "csv", # Formato de tablas resumen
    "figure": "png" # Formato de las figuras
} # Formatos oficiales del proyecto

# Precisión numérica
NUMERIC_CONFIG = {
    "float_dtype": "float32", # Precisión para variables continuas
    "int_dtype": "int64", # Precisión para variables enteras
    "round_digits": 4 # Número de decimales en reportes
} # Configuración numérica

# Configuración temporal
DATE_CONFIG = {
    "date_format": "%Y-%m-%d", # Formato oficial de fechas
    "datetime_format": "%Y-%m-%d %H:%M:%S", # Formato oficial de fecha y hora
    "timezone": "America/Bogota" # Zona horaria del proyecto
} # Configuración temporal

# Configuración regional
LOCALE_CONFIG = {
    "country": "Colombia", # País del estudio
    "language": "es", # Idioma principal
    "encoding": "utf-8" # Codificación de caracteres
} # Configuración regional

# Convenciones de nombres
NAMING_CONVENTIONS = {
    "model_prefix": "gnn", # Prefijo de los modelos
    "forecast_prefix": "forecast", # Prefijo de los pronósticos
    "report_prefix": "report", # Prefijo de los reportes
    "figure_prefix": "figure" # Prefijo de las figuras
} # Convenciones de nombres
