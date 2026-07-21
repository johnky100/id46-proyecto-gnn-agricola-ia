# config-config_project.py

# IMPORTACIONES
# ------------------------------------------------------------------------------
# Objetivo:
# Importar las bibliotecas estándar y de terceros necesarias para la
# configuración, validación y ejecución del proyecto.
# ------------------------------------------------------------------------------
from src.python.config.paths import (
    validate_project_structure
)

# BLOQUE 0. Inicialización del Proyecto 
# ------------------------------------------------------------------------------
# Objetivo:
# Validar la estructura oficial del proyecto y crear automáticamente todos los
# directorios requeridos por el pipeline científico antes de ejecutar cualquier
# proceso de lectura, escritura, entrenamiento o exportación.
# ------------------------------------------------------------------------------
validate_project_structure(
    verbose=False
)

# BLOQUE 1. Configuración General
# ------------------------------------------------------------------------------
# Objetivo:
# Definir los metadatos oficiales del proyecto, la configuración de
# reproducibilidad, el alcance temporal y territorial del panel científico,
# así como los parámetros espaciales utilizados por todos los módulos.
# ------------------------------------------------------------------------------
# 1.1 Información general del proyecto
PROJECT_NAME = "proyecto-gnn-agricola"
PROJECT_ID = "GEOAI-GNN-001"
PROJECT_VERSION = "3.0.0"
PROJECT_DESCRIPTION = (
    "Modelado espacio-temporal de la soberanía alimentaria "
    "mediante Graph Neural Networks."
)

PROJECT_AUTHORS = [
    {"id": "ID_46", "name": "John Jairo Prado Piñeres", "role": "Investigador Principal"},
    {"id": "ID_47", "name": "Adriana María Redondo Alvarado", "role": "Investigadora"},
    {"id": "ID_48", "name": "Gloria Patricia Redondo Alvarado", "role": "Investigadora"},
    {"id": "ID_49", "name": "María José Redondo Alvarado", "role": "Investigadora"},
]

COUNTRY = "Colombia"
LANGUAGE = "Español"
LICENSE = "MIT"

# 1.2 Configuración de reproducibilidad
PROJECT_SEED = 5477976

# 1.3 Configuración temporal del panel
ANIO_INICIO = 2006
ANIO_FIN = 2018
N_ANIOS = ANIO_FIN - ANIO_INICIO + 1
PANEL_YEARS = list(range(ANIO_INICIO, ANIO_FIN + 1))

# 1.4 Configuración territorial
N_MUNICIPIOS = 1121
N_OBS_PANEL = N_MUNICIPIOS * N_ANIOS

# 1.5 Configuración espacial
DEFAULT_CRS_EPSG = 4686 # MAGNA-SIRGAS
GEOMETRY_COLUMN = "geometry"

# BLOQUE 2. Dataset Científico
# ------------------------------------------------------------------------------
# Objetivo:
# Definir la estructura oficial del Dataset Científico, incluyendo la identidad
# del panel, las variables estructurales, la variable objetivo, los dominios
# científicos, las variables predictoras y la clasificación oficial utilizada
# por todos los módulos del pipeline.
# ------------------------------------------------------------------------------
# 2.1 Información general del Dataset Científico ------------------------------
DATASET_NAME = "dataset_gnn_certificado.parquet" # Nombre oficial del Dataset Científico
DATASET_VERSION = "1.0.0" # Versión oficial del Dataset Científico
DATASET_FORMAT = "parquet" # Formato oficial del Dataset Científico

# Identidad oficial del panel científico
PANEL_ID_COLUMN = "panel_id" # Identificador único del registro municipio-año
MUNICIPALITY_ID_COLUMN = "cod_mpio" # Código DIVIPOLA del municipio
TIME_COLUMN = "anio" # Variable temporal del panel
TARGET_VARIABLE = "log_rendimiento" # Variable objetivo oficial

# Variables de identificación del panel
ID_COLUMNS = [
    PANEL_ID_COLUMN,
    MUNICIPALITY_ID_COLUMN,
    "municipio",
    "cod_depto",
    "departamento"
] # Variables de identificación y trazabilidad del panel científico

# 2.2 Variables estructurales del Dataset Científico --------------------------
LATITUDE_COLUMN = "latitud" # Columna oficial de latitud
LONGITUDE_COLUMN = "longitud" # Columna oficial de longitud

SPATIAL_COLUMNS = [
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN
] # Variables de localización geográfica

# Columnas estructurales del Dataset Científico
STRUCTURAL_COLUMNS = (
    ID_COLUMNS
    + [TIME_COLUMN]
    + SPATIAL_COLUMNS
) # Columnas necesarias para identificar y estructurar el panel

# 2.3 Variables excluidas del modelado ----------------------------------------
# Variable objetivo original antes de la transformación
RAW_TARGET_VARIABLE = "rendimiento_promedio"

# Variables que no forman parte de las Node Features
NON_FEATURE_COLUMNS = (
    STRUCTURAL_COLUMNS
    + [TARGET_VARIABLE]
) # Variables excluidas de las Node Features

# Variables excluidas del entrenamiento
EXCLUDED_COLUMNS = (
    NON_FEATURE_COLUMNS
    + [
        GEOMETRY_COLUMN,
        RAW_TARGET_VARIABLE
    ]
) # Variables excluidas del entrenamiento del modelo

# 2.4. Dominios científicos del Dataset Científico ----------------------------
# Variables climáticas
CLIMATE_COLUMNS = [
    "precip_total_anual",      # Precipitación acumulada anual (mm)
    "precip_sd_mensual",       # Desviación estándar mensual de la precipitación
    "precip_cv",               # Coeficiente de variación de la precipitación
    "precip_min_mensual",      # Precipitación mínima mensual (mm)
    "d2m",                     # Temperatura del punto de rocío a 2 metros (°C)
    "e",                       # Tasa de evaporación (m)
    "lai_hv",                  # Índice de área foliar de vegetación alta (Leaf Area Index)
    "pev",                     # Evaporación potencial (m)
    "ro",                      # Escorrentía superficial total (m)
    "sro",                     # Escorrentía superficial (Surface Runoff)
    "strd",                    # Radiación térmica descendente en superficie (W/m²)
    "t2m",                     # Temperatura del aire a 2 metros (°C)
    "tp",                      # Precipitación total (m)
    "u10",                     # Componente zonal del viento a 10 metros (m/s)
    "v10"                      # Componente meridional del viento a 10 metros (m/s)
] # Variables climáticas

# Variables agrícolas
AGRICULTURE_COLUMNS = [
    "tasa_cosecha_promedio",       # Tasa promedio de cosecha
    "area_sembrada_total",         # Área total sembrada (ha)
    "produccion_total",            # Producción agrícola total (toneladas)
    "n_cultivos",                  # Número de cultivos registrados
    "n_grupos_cultivo",            # Número de grupos de cultivos
    "porcentaje_permanentes",      # Porcentaje de cultivos permanentes
    "log_produccion_total",        # Logaritmo de la producción total
    "log_area_sembrada_total"      # Logaritmo del área sembrada total
] # Variables agrícolas

# Variables de riego
IRRIGATION_COLUMNS = [
    "area_irrigable_total"         # Área con potencial de riego (ha)
] # Variables de riego

# Índices ambientales
ENVIRONMENTAL_INDEX_COLUMNS = [
    "potencial_score_promedio",    # Puntaje promedio de potencial agroambiental
    "indice_hidrico",              # Índice de disponibilidad hídrica
    "indice_sostenibilidad"        # Índice de sostenibilidad territorial
] # Índices ambientales

# Cobertura del suelo
LAND_COVER_COLUMNS = [
    "area_total",                  # Área total del municipio (ha)
    "pct_agro",                    # Porcentaje de cobertura agropecuaria
    "pct_natural",                 # Porcentaje de cobertura natural
    "pct_no_agro",                 # Porcentaje de superficie no agropecuaria
    "pct_otros_usos"               # Porcentaje de otros usos del suelo
] # Cobertura del suelo

# Tenencia de la tierra
LAND_TENURE_COLUMNS = [
    "pct_propia",                  # Porcentaje de tierra en propiedad
    "pct_arrendada",               # Porcentaje de tierra arrendada
    "pct_colectiva",               # Porcentaje de tierra de propiedad colectiva
    "pct_mixta"                    # Porcentaje de tierra con tenencia mixta
] # Tenencia de la tierra

# Variables predictoras oficiales del proyecto
FEATURE_COLUMNS = (
    CLIMATE_COLUMNS
    + AGRICULTURE_COLUMNS
    + IRRIGATION_COLUMNS
    + ENVIRONMENTAL_INDEX_COLUMNS
    + LAND_COVER_COLUMNS
    + LAND_TENURE_COLUMNS
) # Conjunto oficial de variables predictoras utilizadas durante el modelado científico

# 2.5. Clasificación oficial de variables ------------------------------------
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

# BLOQUE 3. Construcción del GraphData
# ------------------------------------------------------------------------------
# Objetivo:
# Definir la arquitectura oficial del GraphData, incluyendo la configuración
# de los nodos, la estructura espacial del grafo, el catálogo maestro de nodos,
# las Node Features, las reglas de validación y los productos oficiales
# generados durante su construcción.
# ------------------------------------------------------------------------------
# 3.1 Configuración de Nodos ----------------------------------------------
NODE_ID_COLUMN = "node_id" # Identificador científico único del nodo
NODE_INDEX_COLUMN = "node_idx" # Índice interno utilizado por PyTorch Geometric
GRAPH_NODE_TYPE = "municipality_year" # Tipo oficial de nodo del proyecto

GRAPH_NODE_KEY_COLUMNS = [
    MUNICIPALITY_ID_COLUMN,
    TIME_COLUMN
] # Columnas utilizadas para construir la identidad científica del nodo

NODE_CONFIG = {
    "node_type": GRAPH_NODE_TYPE,
    "node_id_column": NODE_ID_COLUMN,
    "node_index_column": NODE_INDEX_COLUMN,
    "key_columns": GRAPH_NODE_KEY_COLUMNS
} # Configuración oficial de la identidad científica y computacional del nodo

# 3.2 Configuración Espacial del Grafo ------------------------------------
GRAPH_CONNECTIVITY = "spatial_only"
GRAPH_NEIGHBOR_METHOD = "queen"
GRAPH_K_NEIGHBORS = 8

GRAPH_SPATIAL_CONFIG = {
    "method": GRAPH_NEIGHBOR_METHOD,
    "k_neighbors": GRAPH_K_NEIGHBORS,
    "graph_connectivity": GRAPH_CONNECTIVITY,
    "crs": DEFAULT_CRS_EPSG,
    "geometry_column": GEOMETRY_COLUMN,
    "latitude_column": LATITUDE_COLUMN,
    "longitude_column": LONGITUDE_COLUMN,
    "require_geometry": True,
    "validate_geometry": True,
    "allow_multipart": True,
    "remove_self_loops": True,
    "remove_duplicate_edges": True
}  # Configuración oficial de la estructura espacial del GraphData

# 3.3 Configuración del Catálogo Maestro de Nodos --------------------------
# Metadatos oficiales del nodo
NODE_METADATA_COLUMNS = [
    MUNICIPALITY_ID_COLUMN,
    "municipio",
    "cod_depto",
    "departamento",
    TIME_COLUMN,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN
]

NODE_CATALOG_CONFIG = {
    "file_name": "node_catalog.parquet",
    "node_index_column": NODE_INDEX_COLUMN,
    "node_id_column": NODE_ID_COLUMN,
    "key_columns": GRAPH_NODE_KEY_COLUMNS,
    "source_columns": NODE_METADATA_COLUMNS,

    "columns": [
        NODE_INDEX_COLUMN,
        NODE_ID_COLUMN,
        *NODE_METADATA_COLUMNS
    ],

    "include_geometry": False
} # Configuración oficial del Catálogo Maestro de Nodos

# 3.4 Configuración de Node Features --------------------------------------
NODE_FEATURE_CONFIG = {
    "feature_source": "dataset", # Las Node Features provienen del Dataset Científico
    "excluded_columns": EXCLUDED_COLUMNS, # Columnas excluidas de la matriz X
    "normalize_features": True, # Normalizar variables predictoras
    "allow_structural_columns": False, # No permitir variables estructurales
    "allow_identifier_columns": False, # No permitir identificadores
    "allow_target_column": False # No permitir la variable objetivo
} # Configuración oficial para la construcción de las Node Features

# 3.5 Configuración General del GraphData ---------------------------------
GRAPH_SCHEMA_VERSION = "2.0.0"  # Versión oficial del esquema del GraphData

GRAPH_COMPONENTS = [
    "x",
    "edge_index",
    "edge_weight",
    "y"
]  # Componentes oficiales del GraphData

GRAPH_CONFIG = {
    "graph_type": "panel_spatial_graph",      # Tipo oficial de GraphData
    "graph_node_type": GRAPH_NODE_TYPE,       # Tipo oficial de nodo
    "directed": False,                        # Grafo no dirigido
    "weighted": True,                         # El GraphData incorpora edge_weight
    "edge_connectivity": GRAPH_CONNECTIVITY,  # Tipo oficial de conectividad
    "schema_version": GRAPH_SCHEMA_VERSION,   # Versión oficial del esquema
    "components": GRAPH_COMPONENTS,           # Componentes oficiales del GraphData
    "edge_index_dtype": "int64",               # Tipo de dato del edge_index
    "show_figures": False,
    "dpi": 300,
    "figure_size": (12, 12)
}  # Configuración oficial del GraphData

# 3.6 Configuración de Validación -----------------------------------------
GRAPH_VALIDATION = {
    "panel": {
        "check_integrity": True,
        "check_size": True,
        "check_duplicates": True,
        "check_missing_nodes": True
    },

    "catalog": {
        "check_node_catalog": True,
        "check_node_id": True,
        "check_node_index": True
    },

    "graph": {
        "check_connectivity": True,
        "check_edge_index": True,
        "check_edge_weights": True,
        "check_features": True,
        "check_target": True
    },

    "architecture": {
        "check_same_year_edges": True,
        "check_identifier_leakage": True,
        "check_catalog_alignment": True
    }
} # Validaciones oficiales del GraphData

# 3.7 Configuración de Productos Oficiales -------------------------------
GRAPH_OUTPUTS = {
    "graph_data": "graph_data.pt", # GraphData oficial del proyecto
    "node_catalog": "node_catalog.parquet", # Catálogo Maestro de Nodos
    "graph_validation": "graph_validation.json", # Resultado de las validaciones
    "graph_statistics": "graph_statistics.json", # Estadísticas estructurales del grafo
    "graph_metadata": "graph_metadata.json" # Metadatos oficiales del GraphData
} # Productos oficiales generados por graph_01_build_graph.py

GRAPH_VISUALIZATION_CONFIG = {
    "show_figures": False,
    "dpi": 300,
    "figure_size": (12, 12),
    "graph_node_size": 6,
    "graph_edge_width": 0.30,
    "graph_edge_alpha": 0.50
}

# BLOQUE 4. Benchmark Científico
# ------------------------------------------------------------------------------
# Objetivo:
# Definir la arquitectura oficial del Benchmark Científico para la evaluación,
# comparación reproducible y selección del modelo con mejor desempeño, así como
# la configuración de los experimentos, métricas, criterios de selección,
# reproducibilidad y productos oficiales generados por el proceso.
# ------------------------------------------------------------------------------

# 4.1 Configuración General del Benchmark ------------------------------------
PRIMARY_METRIC = "rmse"

# Configuración general del Benchmark Científico
BENCHMARK_CONFIG = {
    "random_state": PROJECT_SEED,      # Semilla oficial del proyecto
    "train_size": 0.70,                # Proporción para entrenamiento
    "validation_size": 0.15,           # Proporción para validación
    "test_size": 0.15,                 # Proporción para prueba
    "cv_folds": 5,                     # Número de particiones para validación cruzada
    "shuffle": True,                   # Mezclar observaciones antes de particionar
    "n_jobs": -1,                      # Utilizar todos los núcleos disponibles
    "ranking_metric": PRIMARY_METRIC   # Métrica utilizada para construir el ranking
}  # Configuración oficial del Benchmark Científico

# 4.2 Configuración del Dataset para el Benchmark ----------------------------
DATASET_CONFIG = {
    "target_variable": TARGET_VARIABLE,          # Variable objetivo oficial
    "excluded_columns": EXCLUDED_COLUMNS,        # Variables excluidas del modelado
    "variable_groups": VARIABLE_GROUPS           # Clasificación oficial de variables   
}  # Configuración oficial del Dataset para el Benchmark

# 4.3 Configuración de Experimentos ------------------------------------------
BENCHMARK_EXPERIMENTS = {
    "EXP01": {
        "name": "Variables Climáticas",
        "features": VARIABLE_GROUPS["climate"],
        "description": "Evaluación utilizando únicamente variables climáticas.",
        "enabled": True
    },

    "EXP02": {
        "name": "Variables Agrícolas",
        "features": VARIABLE_GROUPS["agriculture"],
        "description": "Evaluación utilizando únicamente variables agrícolas.",
        "enabled": True
    },

    "EXP03": {
        "name": "Variables Climáticas + Agrícolas",
        "features": (
            VARIABLE_GROUPS["climate"] +
            VARIABLE_GROUPS["agriculture"]
        ),
        "description": "Evaluación conjunta de variables climáticas y agrícolas.",
        "enabled": True
    }

}  # Experimentos oficiales del Benchmark Científico

# 4.4 Modelos Oficiales del Benchmark ----------------------------------------
BENCHMARK_MODELS = {

    # Modelos Estadísticos
    "statistical": [
        "linear_regression"
    ],

    # Modelos de Machine Learning
    "machine_learning": [
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost"
    ],

    # Modelos de Deep Learning
    "deep_learning": [
        "mlp"
    ],

    # Modelos de Graph Neural Networks
    "graph_neural_networks": [
        "gcn",
        "graphsage",
        "gat",
        "gin",
        "tagcn"
    ]

}  # Modelos oficiales del Benchmark Científico

# 4.5 Identificadores Oficiales de los Modelos -------------------------------
BENCHMARK_MODEL_CODES = {
    # Modelos Estadísticos
    "ST01": "linear_regression",

    # Modelos de Machine Learning
    "ML01": "random_forest",
    "ML02": "xgboost",
    "ML03": "lightgbm",
    "ML04": "catboost",

    # Modelos de Deep Learning
    "DL01": "mlp",

    # Modelos de Graph Neural Networks
    "GNN01": "gcn",
    "GNN02": "graphsage",
    "GNN03": "gat",
    "GNN04": "gin",
    "GNN05": "tagcn"

}  # Identificadores oficiales de los modelos del Benchmark Científico

# 4.6 Métricas Oficiales del Benchmark ---------------------------------------
BENCHMARK_METRICS = {

    # Métricas de regresión
    "rmse": {
        "name": "Root Mean Squared Error",
        "maximize": False,
        "official": True
    },

    "mae": {
        "name": "Mean Absolute Error",
        "maximize": False,
        "official": True
    },

    "r2": {
        "name": "Coefficient of Determination",
        "maximize": True,
        "official": True
    },

    "mape": {
        "name": "Mean Absolute Percentage Error",
        "maximize": False,
        "official": True
    }

}  # Métricas oficiales del Benchmark Científico

# 4.7 Configuración de Selección del Modelo Oficial --------------------------
BENCHMARK_SELECTION = {
    # Métrica principal utilizada para construir el ranking oficial
    "primary_metric": PRIMARY_METRIC,

    # Métrica utilizada para resolver empates
    "secondary_metric": "mae",

    # Criterio de optimización de la métrica principal
    "criterion": "min"
}  # Configuración oficial de selección del modelo

# 4.8 Configuración de Reproducibilidad --------------------------------------
BENCHMARK_REPRODUCIBILITY = {
    # Semilla oficial del proyecto
    "random_seed": PROJECT_SEED,
    # Ejecutar operaciones determinísticas cuando la librería lo permita
    "deterministic": True,
    # Repeticiones independientes de cada experimento
    "n_runs": 1,
    # Guardar la configuración utilizada en cada ejecución
    "save_configuration": True
}  # Configuración oficial de reproducibilidad del Benchmark

# 4.9 Productos Oficiales del Benchmark --------------------------------------
BENCHMARK_OUTPUTS = {
    # Guardar predicciones generadas por cada modelo
    "save_predictions": True,
    # Guardar métricas individuales de cada experimento
    "save_metrics": True,
    # Guardar el ranking consolidado de modelos
    "save_ranking": True,
    # Guardar el modelo oficial seleccionado
    "save_best_model": True,
    # Guardar los modelos entrenados de todos los candidatos
    "save_models": False,
    # Guardar el reporte consolidado del Benchmark
    "save_results": True
}  # Productos oficiales del Benchmark Científico

# BLOQUE 5. Entrenamiento del Modelo
# ------------------------------------------------------------------------------
# Objetivo:
# Definir la configuración oficial para el entrenamiento del modelo seleccionado
# por el Benchmark Científico, incluyendo la preparación del dataset, la
# optimización, el proceso de entrenamiento, los mecanismos de regularización,
# el almacenamiento de checkpoints y la generación de los productos oficiales.
# ------------------------------------------------------------------------------
# 5.1 Configuración General del Entrenamiento
TRAIN_CONFIG = {
    # Dispositivo de ejecución
    "preferred_device": "cuda",

    # Configuración del entrenamiento
    "epochs": 300,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,

    # Optimización
    "optimizer": "Adam",
    "loss_function": "MSELoss",
    "scheduler": "ReduceLROnPlateau",

    # Estabilidad del entrenamiento
    "gradient_clipping": True,
    "clip_value": 1.0,
    "mixed_precision": False,

    # Carga de datos
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": False,

    # Reproducibilidad
    "random_seed": PROJECT_SEED
}  # Configuración general del entrenamiento

# 5.2 Configuración del Modelo Oficial ---------------------------------------
OFFICIAL_MODEL_CONFIG = {
    # Cargar automáticamente el modelo ganador del Benchmark
    "load_from_benchmark": True,
    # Código oficial del modelo seleccionado (ej. ML02, GNN03)
    "model_code": None,
    # Nombre interno del modelo (ej. xgboost, gat)
    "model_name": None,
    # Familia del modelo
    "model_family": None,
    # Utilizar los mejores hiperparámetros encontrados
    "use_best_hyperparameters": True
}  # Configuración oficial del modelo

# 5.3 Configuración del Dataset de Entrenamiento -----------------------------
TRAIN_DATASET_CONFIG = {
    # Variable objetivo del proyecto
    "target_variable": TARGET_VARIABLE,
    # Variables excluidas del entrenamiento
    "excluded_columns": EXCLUDED_COLUMNS,
    # Grupos oficiales de variables
    "variable_groups": VARIABLE_GROUPS,
    # Experimento del Benchmark utilizado para el entrenamiento definitivo
    "experiment": None,
    # Utilizar las mismas particiones definidas en el Benchmark
    "use_benchmark_split": True
}  # Configuración oficial del Dataset de Entrenamiento

# 5.4 Configuración del Proceso de Entrenamiento -----------------------------
TRAIN_PROCESS_CONFIG = {
    # Entrenar utilizando las particiones oficiales del Benchmark
    "use_benchmark_split": True,
    # Validar el modelo durante el entrenamiento
    "validate_during_training": True,
    # Evaluar el modelo al finalizar el entrenamiento
    "evaluate_after_training": True,
    # Guardar automáticamente el mejor modelo
    "save_best_model": True,
    # Registrar el historial del entrenamiento
    "log_training_history": True,
    # Mostrar el progreso del entrenamiento
    "verbose": True

}  # Configuración oficial del proceso de entrenamiento

# 5.5 Configuración de Early Stopping ----------------------------------------
EARLY_STOPPING_CONFIG = {
    "enabled": True,                     # Activar Early Stopping
    "monitor": "validation_loss",        # Métrica monitoreada
    "patience": 30,                      # Número máximo de épocas sin mejora
    "min_delta": 0.0001,                 # Mejora mínima considerada
    "restore_best_weights": True         # Restaurar automáticamente el mejor modelo
}  # Configuración oficial de Early Stopping

# 5.6 Configuración de Checkpoints -------------------------------------------
CHECKPOINT_CONFIG = {
    "enabled": True,                     # Activar guardado de checkpoints
    "save_best_model": True,             # Guardar únicamente el mejor modelo
    "save_last_model": False,            # Guardar el último modelo entrenado
    "save_optimizer_state": True,        # Guardar el estado del optimizador
    "save_scheduler_state": True         # Guardar el estado del scheduler
}  # Configuración oficial de Checkpoints

# 5.7 Productos del Entrenamiento --------------------------------------------
TRAIN_OUTPUTS = {
    "save_training_history": True,       # Guardar historial del entrenamiento
    "save_learning_curves": True,        # Guardar curvas de aprendizaje
    "save_embeddings": True,             # Guardar embeddings finales
    "save_model_summary": True,          # Guardar resumen de la arquitectura del modelo
    "save_training_report": True         # Guardar reporte consolidado del entrenamiento
}  # Configuración oficial de los productos del entrenamiento

# BLOQUE 6. Evaluación del Modelo
# ------------------------------------------------------------------------------
# Objetivo:
# Definir la configuración oficial para la evaluación científica del modelo
# seleccionado, incluyendo la validación del desempeño predictivo, el análisis
# de explicabilidad, la generación de visualizaciones y los productos oficiales
# derivados del proceso de evaluación.
# ------------------------------------------------------------------------------
# 6.1 Configuración General de la Evaluación -------------------------------
EVALUATION_CONFIG = {
    "evaluate_train": True,             # Evaluar el conjunto de entrenamiento
    "evaluate_validation": True,        # Evaluar el conjunto de validación
    "evaluate_test": True,              # Evaluar el conjunto de prueba
    "evaluate_best_model": True,        # Evaluar únicamente el mejor modelo
    "use_best_checkpoint": True,        # Utilizar el mejor checkpoint disponible
    "verbose": True                     # Mostrar el progreso de la evaluación
}  # Configuración oficial de la evaluación

# 6.2 Métricas Oficiales de la Evaluación -------------------------------
EVALUATION_METRICS = {
    "prediction": [
        "rmse",
        "mae",
        "r2",
        "mape"
    ],  # Desempeño predictivo del modelo

    "robustness": [
        "seed_variability"
    ],  # Estabilidad del modelo entre ejecuciones

    "spatial": [
        "moran_i_residuals",
        "spatial_bias"
    ],  # Comportamiento espacial de los residuos

    "graph": [
        "graph_density",
        "average_degree",
        "connected_components",
        "isolated_nodes"
    ]  # Propiedades estructurales del GraphData

}  # Configuración oficial de las métricas de evaluación

# 6.3 Configuración de Explicabilidad --------------------------------------
EXPLAINABILITY_CONFIG = {
    "feature_importance": True,       # Calcular importancia de variables
    "shap": True,                     # Calcular valores SHAP
    "attention_weights": "auto",      # Calcular pesos de atención si el modelo los soporta
    "embedding_analysis": True,       # Analizar los embeddings aprendidos
    "uncertainty": False              # Calcular incertidumbre predictiva
}  # Configuración oficial de explicabilidad

# 6.4 Configuración de Visualizaciones -------------------------------------
PLOT_CONFIG = {
    "prediction_plot": True,            # Observado vs. predicho
    "residual_plot": True,              # Distribución de residuos
    "loss_curve": True,                 # Curva de pérdida
    "learning_curve": True,             # Curva de aprendizaje
    "feature_importance_plot": True,    # Importancia de variables
    "shap_summary_plot": True,          # Resumen SHAP
    "attention_heatmap": True,          # Mapa de calor de atención
    "embedding_projection": True        # Proyección de embeddings (t-SNE/UMAP)
}  # Configuración oficial de visualizaciones

# 6.5 Configuración de Productos de la Evaluación -------------------------
EVALUATION_OUTPUTS = {
    "save_metrics": True,               # Guardar métricas de evaluación
    "save_predictions": True,           # Guardar predicciones del modelo
    "save_residuals": True,             # Guardar residuos del modelo
    "save_feature_importance": True,    # Guardar importancia de variables
    "save_shap": True,                  # Guardar valores SHAP
    "save_attention": True,             # Guardar pesos de atención cuando existan
    "save_embeddings": True,            # Guardar embeddings aprendidos
    "save_report": True                 # Generar reporte técnico de evaluación
}  # Configuración oficial de los productos de evaluación

# BLOQUE 7. Forecasting y Escenarios
# ------------------------------------------------------------------------------
# Objetivo:
# Definir la configuración oficial para la generación de pronósticos y la
# simulación de escenarios futuros, incluyendo el horizonte de predicción, la
# configuración del proceso de forecasting, los escenarios de análisis y los
# productos oficiales derivados del modelo.
# ------------------------------------------------------------------------------
# 7.1 Configuración General del Forecasting ------------------------------
FORECAST_HORIZON = 5 # Horizonte de pronóstico
FORECAST_START_YEAR = ANIO_FIN + 1 # Primer año proyectado
FORECAST_END_YEAR = FORECAST_START_YEAR + FORECAST_HORIZON - 1 # Último año proyectado

FORECAST_CONFIG = {
    "forecast_horizon": FORECAST_HORIZON,
    "forecast_frequency": "annual",
    "forecast_start_year": FORECAST_START_YEAR,
    "forecast_end_year": FORECAST_END_YEAR,
    "historical_years": N_ANIOS,
    "target_variable": TARGET_VARIABLE,
    "use_official_model": True,
    "use_best_checkpoint": True,
    "random_state": PROJECT_SEED
} # Configuración general del forecasting

# 7.2 Configuración de Escenarios ----------------------------------------
SCENARIO_CONFIG = {
    "enabled_scenarios": [
        "baseline",
        "optimistic",
        "pessimistic"
    ]
}  # Configuración oficial de escenarios

# 7.3 Configuración de la Predicción -------------------------------------
PREDICTION_CONFIG = {
    "recursive_forecast": True,       # Utilizar predicción recursiva
    "calculate_uncertainty": False,   # Calcular incertidumbre predictiva
    "clip_forecast_horizon": True,    # indicar si las predicciones deben limitarse al horizonte configurado
    "save_intermediate_predictions": False # controlar el almacenamiento de predicciones intermedias
}  # Configuración oficial de la predicción

# 7.4 Configuración de Productos del Forecasting -------------------------
FORECAST_OUTPUTS = {
    "save_node_predictions": True,   # Guardar predicciones por municipio
    "save_graph_predictions": True,  # Guardar resultados del grafo
    "save_scenarios": True,          # Guardar resultados de escenarios
    "save_maps": True,               # Generar mapas de pronóstico
    "save_report": True,              # Generar reporte técnico
    "save_dashboard_data": True
}  # Configuración oficial de productos del forecasting

# BLOQUE 8. Plataforma GeoAI
# ------------------------------------------------------------------------------
# Objetivo:
# Definir la configuración oficial de la Plataforma GeoAI, incluyendo la
# visualización geoespacial, los servicios de análisis, la generación de
# reportes, las interfaces de programación (API), los agentes inteligentes,
# los servicios de apoyo a la decisión y los productos finales del sistema.
# ------------------------------------------------------------------------------
# 8.1 Configuración del Dashboard ----------------------------------------
DASHBOARD_CONFIG = {
    "enabled": True,                  # Habilitar Dashboard GeoAI
    "interactive_map": True,          # Mostrar mapa interactivo
    "display_predictions": True,      # Mostrar predicciones
    "display_scenarios": True,        # Mostrar escenarios
    "display_explainability": True,   # Mostrar resultados de explicabilidad
    "display_metrics": True,          # Mostrar métricas del modelo
    "display_graph": True,            # Mostrar visualización del grafo
    "display_time_series": True
}  # Configuración oficial del Dashboard

# 8.2 Configuración de Reportes ------------------------------------------
REPORT_CONFIG = {
    "generate_pdf": True,      # Generar reporte en PDF
    "generate_excel": True,    # Exportar resultados a Excel
    "generate_html": True,     # Generar reporte HTML
    "include_maps": True,      # Incluir mapas
    "include_figures": True,    # Incluir gráficos
    "include_recommendations": True
}  # Configuración oficial de reportes

# 8.3 Configuración de la API --------------------------------------------
API_CONFIG = {
    "enabled": True,     # Habilitar API
    "host": "0.0.0.0",   # Dirección del servidor
    "port": 8000,        # Puerto de la API
    "version": "v1",     # Versión de la API
    "docs": True,        # Habilitar documentación automática
    "base_path": "/api"
}  # Configuración oficial de la API

# 8.4 Configuración de Agentes Inteligentes ------------------------------
AI_AGENT_CONFIG = {
    "data_analysis": True,        # Agente de análisis de datos
    "forecast_analysis": True,    # Agente de análisis del forecasting
    "graph_analysis": True,       # Agente de análisis del grafo
    "recommendations": True,      # Agente de recomendaciones
    "report_generation": True,    # Agente generador de reportes
    "climate_analysis": True
}  # Configuración oficial de agentes inteligentes

# 8.5 Configuración de Productos de la Plataforma GeoAI ------------------
GEOAI_OUTPUTS = {
    "save_recommendations": True,   # Guardar recomendaciones generadas por IA
    "generate_dashboard": True,     # Publicar Dashboard GeoAI
    "deploy_api": True,             # Publicar API
    "generate_reports": True,       # Generar reportes automáticos
    "save_logs": True,               # Guardar registros de ejecución
    "save_dashboard_state": True
}  # Configuración oficial de los productos de la Plataforma GeoAI

# BLOQUE 9. Estándares del Proyecto
# ------------------------------------------------------------------------------
# Objetivo:
# Definir los estándares oficiales del proyecto que garantizan la consistencia,
# interoperabilidad y reproducibilidad del ecosistema GeoAI, incluyendo la
# representación numérica, los formatos temporales, la configuración regional,
# las convenciones de nomenclatura y las normas transversales utilizadas por
# todos los módulos del pipeline.
# ------------------------------------------------------------------------------
# 9.1 Configuración Numérica ---------------------------------------------
NUMERIC_CONFIG = {
    "float_dtype": "float32",  # Precisión para variables continuas
    "int_dtype": "int64",      # Precisión para variables enteras
    "round_digits": 4          # Número de decimales en reportes
}  # Configuración oficial numérica

# 9.2 Configuración Temporal ---------------------------------------------
DATE_CONFIG = {
    "date_format": "%Y-%m-%d",           # Formato oficial de fechas
    "datetime_format": "%Y-%m-%d %H:%M:%S",  # Formato oficial de fecha y hora
    "timezone": "America/Bogota"          # Zona horaria del proyecto
}  # Configuración oficial temporal

# 9.3 Configuración Regional ---------------------------------------------
LOCALE_CONFIG = {
    "country": "Colombia",
    "language": "es",
    "encoding": "utf-8"
}  # Configuración oficial regional

# 9.4 Convenciones de Nombres --------------------------------------------
NAMING_CONVENTIONS = {
    "model_prefix": "gnn",          # Prefijo de los modelos
    "forecast_prefix": "forecast",  # Prefijo de los pronósticos
    "report_prefix": "report",      # Prefijo de los reportes
    "figure_prefix": "figure"       # Prefijo de las figuras
}  # Convenciones oficiales de nombres