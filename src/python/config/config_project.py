# config_project.py

# IMPORTACIONES
# Objetivo:
# Importar las dependencias oficiales necesarias para inicializar la configuración
# centralizada del proyecto.

from src.python.config.paths import validate_project_structure
import torch

# BLOQUE 0. Inicialización del Proyecto
# Objetivo: Validar la estructura oficial del proyecto antes de ejecutar cualquier módulo del Pipeline Científico.
validate_project_structure(verbose=False)

# BLOQUE 1. Proyecto
# Objetivo: Definir la identidad oficial del proyecto y los metadatos científicos reutilizados por todos los módulos del Pipeline.
# ¿Qué es el proyecto?

# 1.1 Identidad del Proyecto
PROJECT_NAME = "proyecto-gnn-agricola"
PROJECT_ID = "GEOAI-GNN-001"
PROJECT_VERSION = "3.0.0"
PROJECT_DESCRIPTION = (
    "Modelado espacio-temporal de la soberanía alimentaria "
    "mediante Graph Neural Networks."
)

# 1.2 Autores del Proyecto

PROJECT_AUTHOR_ID = "ID_46" # Identificador oficial del proyecto
PROJECT_AUTHORS = [
    {"id": PROJECT_AUTHOR_ID, "name": "John Jairo Prado Piñeres", "role": "Investigador Principal"},
    {"id": PROJECT_AUTHOR_ID, "name": "Adriana María Redondo Alvarado", "role": "Investigadora"},
    {"id": PROJECT_AUTHOR_ID, "name": "Gloria Patricia Redondo Alvarado", "role": "Investigadora"},
    {"id": PROJECT_AUTHOR_ID, "name": "María José Redondo Alvarado", "role": "Investigadora"}
] # Autores oficiales del proyecto

# 1.3 Componentes Oficiales de la Plataforma GeoAI
# Plataforma GeoAI
PLATFORM_NAME = "GeoAI Platform" # Nombre oficial de la Plataforma GeoAI
PLATFORM_TYPE = "GeoAI Platform" # Tipo oficial de la Plataforma GeoAI
PLATFORM_DESCRIPTION = (
    "Plataforma GeoAI para la integración, ejecución y gestión del Pipeline Científico."
) # Descripción oficial de la Plataforma GeoAI

# Dashboard
DASHBOARD_NAME = "GeoAI Dashboard" # Nombre oficial del Dashboard
DASHBOARD_TYPE = "Scientific Dashboard" # Tipo oficial del Dashboard
DASHBOARD_DESCRIPTION = (
    "Dashboard interactivo para la exploración y análisis de resultados de la Plataforma GeoAI."
) # Descripción oficial del Dashboard

# API
API_NAME = "GeoAI API" # Nombre oficial de la API
API_TYPE = "REST API" # Tipo oficial de la API
API_DESCRIPTION = (
    "API para el acceso a los servicios de la Plataforma GeoAI."
) # Descripción oficial de la API

# Agente Inteligente
AI_AGENT_NAME = "GeoAI Agent" # Nombre oficial del Agente Inteligente
AI_AGENT_TYPE = "Intelligent Agent" # Tipo oficial del Agente Inteligente
AI_AGENT_DESCRIPTION = (
    "Agente Inteligente para el análisis e interpretación de resultados de la Plataforma GeoAI."
) # Descripción oficial del Agente Inteligente

# Sistema de Reportes
REPORT_SERVICE_NAME = "GeoAI Reports" # Nombre oficial del sistema de reportes
REPORT_SERVICE_TYPE = "Reporting Service" # Tipo oficial del sistema de reportes
REPORT_SERVICE_DESCRIPTION = (
    "Sistema de generación de reportes científicos de la Plataforma GeoAI."
) # Descripción oficial del sistema de reportes

# Sistema de Exportaciones
EXPORT_SERVICE_NAME = "GeoAI Export Service" # Nombre oficial del sistema de exportaciones
EXPORT_SERVICE_TYPE = "Export Service" # Tipo oficial del sistema de exportaciones
EXPORT_SERVICE_DESCRIPTION = (
    "Sistema de exportación de productos científicos de la Plataforma GeoAI."
) # Descripción oficial del sistema de exportaciones

# 1.4 Configuración Oficial de los Servicios
DASHBOARD_HOST = "0.0.0.0" # Dirección oficial del Dashboard
DASHBOARD_PORT = 8050 # Puerto oficial del Dashboard

API_HOST = "0.0.0.0" # Dirección oficial de la API
API_PORT = 8000 # Puerto oficial de la API
API_VERSION = "v1" # Versión oficial de la API

# 1.5 Configuración de Reproducibilidad
PROJECT_SEED = 5477976 # Semilla oficial para garantizar la reproducibilidad

## BLOQUE 2. Panel Científico
# Objetivo: Definir la configuración oficial del Panel Científico, incluyendo su cobertura territorial, temporal,
# espacial, la identidad del Dataset Científico y las variables estructurales utilizadas por todos los módulos del Pipeline.

# 2.1 Cobertura del Proyecto
COUNTRY = "Colombia" # País de estudio
LANGUAGE = "Español" # Idioma oficial del proyecto
LICENSE = "MIT" # Licencia oficial del proyecto

# 2.2 Cobertura Temporal
START_YEAR = 2006 # Primer año del panel científico
END_YEAR = 2018 # Último año del panel científico
N_YEARS = END_YEAR - START_YEAR + 1 # Número total de años del panel
PANEL_YEARS = list(range(START_YEAR, END_YEAR + 1)) # Años oficiales del panel científico

# 2.3 Cobertura Territorial
N_MUNICIPALITIES = 1121 # Número oficial de municipios
N_PANEL_OBSERVATIONS = (
    N_MUNICIPALITIES * N_YEARS
) # Número total de observaciones del panel

# 2.4 Configuración Espacial
DEFAULT_CRS_EPSG = 4686 # Sistema de referencia espacial oficial (MAGNA-SIRGAS)
GEOMETRY_COLUMN = "geometry" # Columna oficial de geometría
LATITUDE_COLUMN = "latitud" # Columna oficial de latitud
LONGITUDE_COLUMN = "longitud" # Columna oficial de longitud

# 2.5 Identidad del Dataset Científico
DATASET_NAME = "dataset_cientifico_certificado.parquet" # Nombre oficial del Dataset Científico
DATASET_VERSION = "1.0.0" # Versión oficial del Dataset Científico
DATASET_FORMAT = "parquet" # Formato oficial del Dataset Científico

PANEL_ID_COLUMN = "id_panel" # Identificador único del registro municipio-año
MUNICIPALITY_ID_COLUMN = "cod_municipio" # Código DIVIPOLA del municipio
TIME_COLUMN = "anio" # Columna temporal del panel científico

MUNICIPALITY_NAME_COLUMN = "municipio" # Nombre oficial del municipio
DEPARTMENT_ID_COLUMN = "cod_departamento" # Código DIVIPOLA del departamento
DEPARTMENT_NAME_COLUMN = "departamento" # Nombre oficial del departamento

# 2.6 Variable Objetivo
TARGET_VARIABLE = "log_rendimiento" # Variable objetivo utilizada durante el modelado
RAW_TARGET_VARIABLE = "rendimiento_promedio" # Variable objetivo en escala original

# 2.7 Variables Estructurales
ID_COLUMNS = [
    PANEL_ID_COLUMN,
    MUNICIPALITY_ID_COLUMN,
    MUNICIPALITY_NAME_COLUMN,
    DEPARTMENT_ID_COLUMN,
    DEPARTMENT_NAME_COLUMN
] # Variables de identificación del panel científico

SPATIAL_COLUMNS = [
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN
] # Variables de localización geográfica

STRUCTURAL_COLUMNS = (
    ID_COLUMNS
    + [TIME_COLUMN]
    + SPATIAL_COLUMNS
) # Variables estructurales del Dataset Científico

NON_FEATURE_COLUMNS = (
    STRUCTURAL_COLUMNS
    + [TARGET_VARIABLE]
) # Variables que no forman parte de las Node Features

EXCLUDED_COLUMNS = (
    NON_FEATURE_COLUMNS
    + [
        GEOMETRY_COLUMN,
        RAW_TARGET_VARIABLE
    ]
) # Variables excluidas del entrenamiento del modelo

# 2.8 Variables Climáticas
CLIMATE_COLUMNS = [
    "precip_total_anual", # Precipitación acumulada anual (mm)
    "desviacion_precip_mensual", # Desviación estándar mensual de la precipitación
    "coeficiente_variacion_precip", # Coeficiente de variación de la precipitación
    "precip_minima_mensual", # Precipitación mínima mensual (mm)
    "temp_punto_rocio_era5", # Temperatura del punto de rocío ERA5
    "evaporacion_era5", # Evaporación acumulada ERA5
    "indice_area_foliar_vegetacion_alta", # Índice de área foliar de vegetación alta (ERA5)
    "evaporacion_potencial_era5", # Evaporación potencial ERA5
    "escorrentia_total", # Escorrentía total
    "escorrentia_superficial", # Escorrentía superficial
    "radiacion_termica_descendente", # Radiación térmica descendente
    "temperatura_aire_era5", # Temperatura del aire ERA5
    "precipitacion_total_era5", # Precipitación total ERA5
    "componente_u_viento", # Componente U del viento
    "componente_v_viento" # Componente V del viento
] # Variables climáticas oficiales del Dataset Científico

# 2.9 Variables Agrícolas
AGRICULTURE_COLUMNS = [
    "tasa_promedio_cosecha", # Tasa promedio de cosecha
    "area_sembrada_total", # Área total sembrada (ha)
    "produccion_total", # Producción agrícola total (toneladas)
    "numero_cultivos", # Número de cultivos registrados
    "numero_grupos_cultivo", # Número de grupos de cultivos
    "porc_cultivos_permanentes", # Proporción de cultivos permanentes
    "log_produccion_total", # Logaritmo de la producción total
    "log_area_total_sembrada" # Logaritmo del área total sembrada
] # Variables agrícolas oficiales del Dataset Científico

# 2.10 Variables de Riego
IRRIGATION_COLUMNS = [
    "area_total_irrigable" # Área total con potencial de riego (ha)
] # Variables de riego oficiales del Dataset Científico

# 2.11 Índices Ambientales
ENVIRONMENTAL_INDEX_COLUMNS = [
    "puntaje_promedio_potencial", # Puntaje promedio de potencial agroambiental
    "indice_hidrico", # Índice de disponibilidad hídrica
    "indice_sostenibilidad" # Índice de sostenibilidad territorial
] # Índices ambientales oficiales del Dataset Científico

# 2.12 Cobertura del Suelo
LAND_COVER_COLUMNS = [
    "area_total_municipio", # Área total del municipio (ha)
    "porc_uso_agropecuario", # Proporción de uso agropecuario
    "porc_cobertura_natural", # Proporción de cobertura natural
    "porc_uso_no_agropecuario", # Proporción de uso no agropecuario
    "porc_otros_usos" # Proporción de otros usos del suelo
] # Variables de cobertura del suelo oficiales del Dataset Científico

# 2.13 Tenencia de la Tierra
LAND_TENURE_COLUMNS = [
    "porc_tierra_propia", # Porcentaje de tierra en propiedad
    "porc_tierra_arrendada", # Porcentaje de tierra arrendada
    "porc_tierra_colectiva", # Porcentaje de tierra de propiedad colectiva
    "porc_tierra_mixta" # Porcentaje de tierra con tenencia mixta
] # Variables de tenencia de la tierra oficiales del Dataset Científico

# 2.14 Variables Predictoras
FEATURE_COLUMNS = (
    CLIMATE_COLUMNS
    + AGRICULTURE_COLUMNS
    + IRRIGATION_COLUMNS
    + ENVIRONMENTAL_INDEX_COLUMNS
    + LAND_COVER_COLUMNS
    + LAND_TENURE_COLUMNS
) # Variables predictoras oficiales del Dataset Científico

# 2.15 Clasificación Oficial
VARIABLE_GROUPS = {
    "identification": ID_COLUMNS,
    "temporal": [TIME_COLUMN],
    "spatial": SPATIAL_COLUMNS,
    "geometry": [GEOMETRY_COLUMN],
    "target": [TARGET_VARIABLE],
    "climate": CLIMATE_COLUMNS,
    "agriculture": AGRICULTURE_COLUMNS,
    "irrigation": IRRIGATION_COLUMNS,
    "environmental_indices": ENVIRONMENTAL_INDEX_COLUMNS,
    "land_cover": LAND_COVER_COLUMNS,
    "land_tenure": LAND_TENURE_COLUMNS,
    "features": FEATURE_COLUMNS,
    "structural": STRUCTURAL_COLUMNS,
    "excluded": EXCLUDED_COLUMNS
} # Clasificación oficial de variables del Dataset Científico

# 2.16 Configuración del Análisis Comparativo Municipal
SIMILARITY_FEATURE_GROUPS = {
    "climate": CLIMATE_COLUMNS,
    "agriculture": AGRICULTURE_COLUMNS,
    "irrigation": IRRIGATION_COLUMNS,
    "environmental_indices": ENVIRONMENTAL_INDEX_COLUMNS,
    "land_cover": LAND_COVER_COLUMNS,
    "land_tenure": LAND_TENURE_COLUMNS
} # Grupos de variables utilizados para comparar municipios

SIMILARITY_FEATURE_COLUMNS = [
    column
    for group_columns in SIMILARITY_FEATURE_GROUPS.values()
    for column in group_columns
] # Construir variables predictoras oficiales utilizadas en similitud

SIMILARITY_STANDARDIZATION = "zscore" # Método oficial de estandarización para similitud
STANDARDIZATION_TOLERANCE = 1e-6 # Tolerancia numérica para validar propiedades del Z Score
SIMILARITY_PRIMARY_METRIC = "cosine" # Métrica principal de similitud
SIMILARITY_DISTANCE_METRIC = "euclidean" # Métrica de distancia complementaria

HIGH_SIMILARITY_THRESHOLD = 0.90 # Umbral operativo de alta similitud
LOW_SIMILARITY_THRESHOLD = 0.50 # Umbral operativo de baja similitud

SIMILARITY_TOPOLOGICAL_ENABLED = True # Activar análisis topológico
SIMILARITY_TEMPORAL_ENABLED = True # Activar análisis temporal
SIMILARITY_NEIGHBOR_OVERLAP_ENABLED = True # Activar overlap de vecinos

SIMILARITY_TOP_K = 10 # Número de municipios similares conservados en el ranking

TOPOLOGY_NEIGHBOR_OVERLAP_WEIGHT = 0.5 # Peso del overlap de vecinos
TOPOLOGY_DEGREE_WEIGHT = 0.5 # Peso de la similitud de grado

STABILITY_THRESHOLD_VERY_STABLE = 0.05 # Límite superior de estabilidad muy alta
STABILITY_THRESHOLD_STABLE = 0.15 # Límite superior de estabilidad
STABILITY_THRESHOLD_MODERATE = 0.30 # Límite superior de estabilidad moderada

SIMILARITY_LEVEL_1 = 0.25 # Límite superior de similitud muy baja
SIMILARITY_LEVEL_2 = 0.50 # Límite superior de similitud baja
SIMILARITY_LEVEL_3 = 0.75 # Límite superior de similitud moderada
SIMILARITY_LEVEL_4 = 0.90 # Límite superior de similitud alta

# 2.17 Configuración de Diferencias entre Municipios
COMPARISON_DIFFERENCE_METHODS = [
    "absolute",
    "standardized"
] # Métodos oficiales para medir diferencias entre municipios

COMPARISON_INCLUDE_TARGET = True # Incluir la variable objetivo en el análisis comparativo
COMPARISON_TARGET_COLUMN = TARGET_VARIABLE # Variable objetivo utilizada para comparar resultados
COMPARISON_INCLUDE_RAW_TARGET = False # No incorporar variables que no estén disponibles en GraphData
COMPARISON_INCLUDE_SPATIAL_DISTANCE = True # Calcular distancia geográfica entre municipios
COMPARISON_INCLUDE_SAME_DEPARTMENT = True # Identificar municipios pertenecientes al mismo departamento

# BLOQUE 3. GraphData
# Objetivo: Definir la configuración oficial del GraphData, incluyendo la identidad de los nodos, las variables científicas
# y el método espacial oficial utilizado para su construcción.

# 3.1 Identidad del Nodo
NODE_ID_COLUMN = "node_id" # Identificador científico único del nodo
NODE_INDEX_COLUMN = "node_idx" # Índice interno utilizado por PyTorch Geometric

GRAPH_NODE_KEY_COLUMNS = [
    MUNICIPALITY_ID_COLUMN,
    TIME_COLUMN
] # Variables que identifican unívocamente un nodo del GraphData

# 3.2 Variables del GraphData
GRAPH_VARIABLE_GROUPS = {
    "node_identification": [
        PANEL_ID_COLUMN,
        MUNICIPALITY_ID_COLUMN,
        TIME_COLUMN
    ],

    "node_metadata": [
        MUNICIPALITY_NAME_COLUMN,
        DEPARTMENT_ID_COLUMN,
        DEPARTMENT_NAME_COLUMN,
        LATITUDE_COLUMN,
        LONGITUDE_COLUMN,
        GEOMETRY_COLUMN
    ],

    "node_features": FEATURE_COLUMNS,
    "target": [
        TARGET_VARIABLE
    ],

    "excluded": EXCLUDED_COLUMNS
} # Clasificación oficial de variables del GraphData

# 3.3 Configuración Espacial del Grafo
GRAPH_SPATIAL_METHOD = "queen" # Método oficial para construir la conectividad espacial
GRAPH_DYNAMIC_K = 10

# 3.4 Configuración de PyTorch
TORCH_FLOAT_DTYPE = torch.float32 # Tipo de dato para variables continuas
TORCH_INT_DTYPE = torch.int64 # Tipo de dato para índices enteros
TORCH_LONG_DTYPE = torch.long # Alias para índices de PyTorch
TORCH_BOOL_DTYPE = torch.bool # Tipo de dato booleano

# 3.5 Edge Weight
EDGE_WEIGHT_METHOD = "hybrid" # Método oficial para calcular el peso de las aristas

# BLOQUE 4. Benchmark Científico
# Objetivo: Definir la configuración oficial del Benchmark Científico, incluyendo los experimentos, los modelos evaluados, las métricas, los criterios de selección y los parámetros de reproducibilidad utilizados durante la evaluación.

# 4.1 Configuración científica del Benchmark
BENCHMARK_CONFIG = {
    "train_size": 0.70, # Proporción oficial para entrenamiento
    "validation_size": 0.15, # Proporción oficial para validación
    "test_size": 0.15, # Proporción oficial para prueba
    "ranking_metric": "rmse", # Métrica oficial utilizada para ordenar los modelos
    "metric_directions": {
        "rmse": "min", # Menor RMSE representa mejor desempeño
        "mae": "min", # Menor MAE representa mejor desempeño
        "mape": "min", # Menor MAPE representa mejor desempeño
        "r2": "max" # Mayor R2 representa mejor desempeño
    },
    "selection_criterion": "min", # Criterio oficial de selección del modelo
    "random_state": PROJECT_SEED # Semilla oficial del Benchmark
} # Configuración científica y experimental del Benchmark

# 4.2 Experimentos
BENCHMARK_EXPERIMENTS = {
    "EXP01": {
        "name": "Variables Climáticas",
        "features": VARIABLE_GROUPS["climate"],
        "description": "Evaluación utilizando únicamente variables climáticas."
    },
    "EXP02": {
        "name": "Variables Agrícolas",
        "features": VARIABLE_GROUPS["agriculture"],
        "description": "Evaluación utilizando únicamente variables agrícolas."
    },
    "EXP03": {
        "name": "Variables Climáticas + Agrícolas",
        "features": (
            VARIABLE_GROUPS["climate"]
            + VARIABLE_GROUPS["agriculture"]
        ),
        "description": "Evaluación conjunta de variables climáticas y agrícolas."
    }
} # Experimentos oficiales del Benchmark Científico

# 4.3 Modelos
BENCHMARK_MODELS = {
    "statistical": [
        "linear_regression",
    ], # Nombre canónico de los modelos estadísticos

    "machine_learning": [
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
    ], # Nombres canónicos de los modelos de Machine Learning

    "deep_learning": [
        "mlp",
    ], # Nombre canónico de los modelos de Deep Learning

    "graph_neural_networks": [
        "gcn",
        "graphsage",
        "gat",
        "gin",
        "tagcn",
    ], # Nombres canónicos de los modelos GNN
} # Modelos oficiales del Benchmark Científico

BENCHMARK_MODEL_CODES = {
    "STAT01": "linear_regression",
    "ML01": "random_forest",
    "ML02": "xgboost",
    "ML03": "lightgbm",
    "ML04": "catboost",
    "DL01": "mlp",
    "GNN01": "gcn",
    "GNN02": "graphsage",
    "GNN03": "gat",
    "GNN04": "gin",
    "GNN05": "tagcn",
} # Códigos oficiales de los modelos del Benchmark Científico

OFFICIAL_MODEL_CODE = "GNN02" # Código oficial de GraphSAGE
OFFICIAL_MODEL_NAME = "graphsage" # Nombre canónico oficial de GraphSAGE
OFFICIAL_MODEL_FAMILY = "graph_neural_networks" # Familia oficial de GraphSAGE

# 4.4 Métricas del Benchmark
BENCHMARK_METRICS = [
    "rmse",
    "mae",
    "r2",
    "mape"
] # Métricas oficiales del Benchmark Científico

# 4.5 Reproducibilidad
BENCHMARK_REPRODUCIBILITY = {
    "deterministic": True, # Ejecutar operaciones determinísticas cuando sea posible
    "runs": 1, # Número de ejecuciones independientes del Benchmark
    "random_state": PROJECT_SEED # Semilla oficial de reproducibilidad
} # Configuración oficial de reproducibilidad del Benchmark

# BLOQUE 5. Entrenamiento del Modelo
# Objetivo: Definir la configuración oficial para el entrenamiento reproducible del modelo seleccionado, incluyendo los
# hiperparámetros, los criterios de optimización y los mecanismos de regularización utilizados durante el proceso de aprendizaje.

# 5.1 Configuración General
TRAIN_EPOCHS = 300 # Número máximo de épocas de entrenamiento
TRAIN_BATCH_SIZE = 64 # Tamaño oficial del lote

# 5.2 Optimización
OPTIMIZER = "Adam" # Optimizador oficial del entrenamiento
LOSS_FUNCTION = "MSELoss" # Función de pérdida para problemas de regresión
LEARNING_RATE = 0.001 # Tasa de aprendizaje inicial
WEIGHT_DECAY = 0.0001 # Factor de regularización L2

# 5.3 Regularización
EARLY_STOPPING = True # Activar Early Stopping durante el entrenamiento
EARLY_STOPPING_PATIENCE = 30 # Número máximo de épocas sin mejora
EARLY_STOPPING_MIN_DELTA = 0.0001 # Mejora mínima para considerar progreso

# BLOQUE 6. Evaluación del Modelo
# Objetivo: Definir la configuración oficial para la evaluación científica del modelo, incluyendo las métricas utilizadas
# para evaluar el desempeño predictivo, espacial y estructural del proyecto.

# 6.1 Métricas de Evaluación
EVALUATION_METRICS = {
    "prediction": [
        "rmse",
        "mae",
        "r2",
        "mape"
    ],
    "robustness": [
        "seed_variability"
    ],
    "spatial": [
        "moran_i_residuals",
        "spatial_bias"
    ],
    "graph": [
        "graph_density",
        "average_degree",
        "connected_components",
        "isolated_nodes"
    ]
} # Métricas oficiales para la evaluación del modelo

# 6.2 Explicabilidad
EXPLAINABILITY_METHODS = [
    "feature_importance", # Importancia de las variables predictoras
    "shap", # Valores SHAP para interpretación del modelo
    "embedding_analysis" # Análisis de los embeddings aprendidos
] # Métodos oficiales de explicabilidad del modelo

# BLOQUE 7. Forecasting y Escenarios
# Objetivo: Definir la configuración oficial para la generación de pronósticos y la simulación de escenarios futuros del proyecto.

# 7.1 Configuración General
FORECAST_START_YEAR = 2019 # Primer año del horizonte de pronóstico
FORECAST_END_YEAR = 2035 # Último año del horizonte de pronóstico
FORECAST_HORIZON = FORECAST_END_YEAR - FORECAST_START_YEAR + 1 # Horizonte oficial de pronóstico (años)

# 7.2 Escenarios
FORECAST_SCENARIOS = [
    "baseline",
    "optimistic",
    "pessimistic"
] # Escenarios oficiales de pronóstico

# BLOQUE 8. Plataforma GeoAI
# Objetivo: Definir los componentes oficiales que conforman la Plataforma GeoAI para el análisis, visualización y apoyo a la toma de decisiones.

# 8.1 Componentes de la Plataforma GeoAI
GEOAI_COMPONENTS = [
    "dashboard",
    "api",
    "agents",
    "reports",
    "export"
] # Componentes oficiales de la Plataforma GeoAI

# 8.2 Servicios Inteligentes
GEOAI_SERVICES = [
    "data_analysis", # Análisis del Dataset Científico
    "graph_analysis", # Análisis del GraphData
    "forecast_analysis", # Análisis de los pronósticos
    "climate_analysis", # Análisis de variables climáticas
    "recommendations" # Generación de recomendaciones para la toma de decisiones
] # Servicios inteligentes oficiales de la Plataforma GeoAI
