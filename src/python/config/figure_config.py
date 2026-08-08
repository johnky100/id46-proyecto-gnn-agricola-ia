# figure_config

# =============================================================================
# BLOQUE 0. IMPORTACIONES
# =============================================================================
# Objetivo:
# Importar las constantes oficiales necesarias para definir la configuración
# gráfica utilizada por el Pipeline Científico y la Plataforma GeoAI.
# Producto:
# Dependencias oficiales requeridas por figure_config.py
# Pregunta científica:
# ¿Qué configuraciones gráficas oficiales son necesarias para garantizar una
# representación visual consistente, reproducible e interpretable durante el
# Pipeline Científico y la Plataforma GeoAI?
# =============================================================================

# =============================================================================
# IMPORTACIONES
# =============================================================================

from src.python.config.config_project import (
    PROJECT_SEED,
)

# ==============================================================================
# BLOQUE 1. Proyecto
# ------------------------------------------------------------------------------
# Objetivo:
# Definir la configuración gráfica oficial utilizada por todas las figuras
# generadas durante el Pipeline Científico y la Plataforma GeoAI, garantizando
# consistencia visual, calidad de publicación y reproducibilidad.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1.1 Configuración General de Figuras
# ------------------------------------------------------------------------------

FIGURE_DPI = 300 # Resolución oficial de exportación
FIGURE_FORMAT = "png" # Formato oficial de las figuras
FIGURE_SIZE = (12, 8) # Tamaño estándar de las figuras
FIGURE_STYLE = "default" # Estilo gráfico oficial
FIGURE_BBOX = "tight" # Ajuste automático de márgenes
SAVE_FIGURES = True # Guardar automáticamente las figuras
SHOW_FIGURES = False # Mostrar figuras durante la ejecución

PROJECT_FIGURE = {
    "dpi": FIGURE_DPI, # Resolución oficial de exportación
    "format": FIGURE_FORMAT, # Formato oficial de las figuras
    "figure_size": FIGURE_SIZE, # Tamaño estándar de las figuras (pulgadas)
    "bbox_inches": FIGURE_BBOX, # Ajustar automáticamente los márgenes
    "style": FIGURE_STYLE, # Estilo gráfico oficial
    "save": SAVE_FIGURES, # Guardar automáticamente las figuras
    "show": SHOW_FIGURES # No mostrar figuras durante la ejecución
} # Configuración gráfica oficial del proyecto

# ==============================================================================
# BLOQUE 2. Panel Científico
# ------------------------------------------------------------------------------
# Objetivo:
# Definir la configuración oficial de las figuras utilizadas para representar
# el Panel Científico, incluyendo el Dataset, las variables, la cobertura
# espacial y temporal, y la estructura del panel utilizada durante el Pipeline
# Científico.
# ==============================================================================

# ------------------------------------------------------------------------------
# 2.1 Configuración General
# ------------------------------------------------------------------------------

PANEL_FIGURE = {
    "figure_size": FIGURE_SIZE, # Tamaño oficial de las figuras del panel
    "dpi": FIGURE_DPI, # Resolución oficial
    "format": FIGURE_FORMAT, # Formato oficial
    "save": SAVE_FIGURES # Guardar automáticamente las figuras
} # Configuración gráfica del Panel Científico

# ------------------------------------------------------------------------------
# 2.2 Dataset Científico
# ------------------------------------------------------------------------------

DATASET_FIGURE = {
    "enabled": True, # Generar figura del Dataset Científico
    "title": "Dataset Científico", # Título oficial
    "show_dimensions": True, # Mostrar dimensiones del dataset
    "show_variables": True # Mostrar número de variables
} # Configuración de la figura del Dataset Científico

# ------------------------------------------------------------------------------
# 2.3 Variables Científicas
# ------------------------------------------------------------------------------

VARIABLES_FIGURE = {
    "enabled": True, # Generar figura de variables científicas
    "group_by_domain": True, # Agrupar variables por dominio
    "show_counts": True # Mostrar cantidad de variables por dominio
} # Configuración de la figura de variables

# ------------------------------------------------------------------------------
# 2.4 Cobertura Espacial
# ------------------------------------------------------------------------------

SPATIAL_COVERAGE_FIGURE = {
    "enabled": True, # Generar mapa de cobertura espacial
    "show_municipalities": True, # Mostrar municipios
    "show_departments": True # Mostrar departamentos
} # Configuración de la cobertura espacial

# ------------------------------------------------------------------------------
# 2.5 Cobertura Temporal
# ------------------------------------------------------------------------------

TEMPORAL_COVERAGE_FIGURE = {
    "enabled": True, # Generar figura de cobertura temporal
    "show_period": True, # Mostrar periodo del panel
    "show_timeline": True # Mostrar línea de tiempo
} # Configuración de la cobertura temporal

# ------------------------------------------------------------------------------
# 2.6 Panel Científico
# ------------------------------------------------------------------------------

PANEL_STRUCTURE_FIGURE = {
    "enabled": True, # Generar figura de la estructura del panel
    "show_entities": True, # Mostrar municipios
    "show_time_dimension": True # Mostrar dimensión temporal
} # Configuración de la figura del Panel Científico

# =============================================================================
# BLOQUE 3. GraphData
# Objetivo:
# Centralizar la configuración oficial utilizada para la representación gráfica
# del GraphData, definiendo los parámetros visuales para la visualización de
# nodos, aristas, layouts, colores, tamaños y etiquetas empleados por el
# Pipeline Científico y la Plataforma GeoAI.
# Pregunta científica:
# ¿La configuración gráfica del GraphData garantiza una representación visual
# consistente, reproducible e interpretable de la estructura espacio-temporal
# utilizada por el Pipeline Científico y la Plataforma GeoAI?
# =============================================================================

# ------------------------------------------------------------------------------
# 3.1 Configuración General
# ------------------------------------------------------------------------------

GRAPH_FIGURE = {
    "figure_size": FIGURE_SIZE, # Tamaño oficial de las figuras del GraphData (pulgadas)
    "dpi": FIGURE_DPI, # Resolución oficial de exportación
    "format": FIGURE_FORMAT, # Formato oficial de las figuras
    "bbox_inches": FIGURE_BBOX, # Ajuste automático de márgenes
    "style": FIGURE_STYLE, # Estilo gráfico oficial
    "save": SAVE_FIGURES, # Guardar automáticamente las figuras
    "show": SHOW_FIGURES # Mostrar figuras durante la ejecución
} # Configuración general de las figuras del GraphData

# ------------------------------------------------------------------------------
# 3.2 Nodos
# ------------------------------------------------------------------------------

NODE_FIGURE = {
    "shape": "o", # Forma oficial de los nodos
    "size": 30, # Tamaño oficial de los nodos
    "color": "#1f77b4", # Color oficial de los nodos
    "edge_color": "black", # Color del borde de los nodos
    "edge_width": 0.20, # Grosor del borde
    "alpha": 0.90 # Nivel de transparencia
} # Configuración gráfica oficial de los nodos

# ------------------------------------------------------------------------------
# 3.3 Aristas
# ------------------------------------------------------------------------------

EDGE_FIGURE = {
    "style": "solid", # Estilo oficial de las aristas
    "width": 0.30, # Grosor oficial de las aristas
    "color": "#b0b0b0", # Color oficial de las aristas
    "alpha": 0.40, # Nivel de transparencia
    "arrows": False # No mostrar flechas en grafos no dirigidos
} # Configuración gráfica oficial de las aristas

# ------------------------------------------------------------------------------
# 3.4 Layout
# ------------------------------------------------------------------------------

GRAPH_LAYOUT = {
    "layout": "kamada_kawai", # Layout oficial para la visualización del GraphData
    "seed": PROJECT_SEED, # Semilla para obtener un layout reproducible
    "scale": 1.0, # Escala del layout
    "center": (0.0, 0.0), # Centro de la figura
    "iterations": 100 # Número de iteraciones del algoritmo de layout
} # Configuración oficial del layout del GraphData

# ------------------------------------------------------------------------------
# 3.5 Colores
# ------------------------------------------------------------------------------

GRAPH_COLORS = {
    "background": "white", # Color de fondo de la figura
    "nodes": "#1f77b4", # Color oficial de los nodos
    "node_border": "black", # Color del borde de los nodos
    "edges": "#b0b0b0", # Color oficial de las aristas
    "labels": "black", # Color oficial de las etiquetas
    "title": "black" # Color oficial del título
} # Configuración oficial de colores del GraphData

# ------------------------------------------------------------------------------
# 3.6 Tamaños
# ------------------------------------------------------------------------------

GRAPH_SIZES = {
    "node_size": 30, # Tamaño oficial de los nodos
    "node_border_width": 0.20, # Grosor del borde de los nodos
    "edge_width": 0.30, # Grosor oficial de las aristas
    "label_size": 8, # Tamaño oficial de las etiquetas
    "title_size": 14, # Tamaño oficial del título
    "legend_size": 10 # Tamaño oficial de la leyenda
} # Configuración oficial de tamaños del GraphData

# ------------------------------------------------------------------------------
# 3.7 Etiquetas
# ------------------------------------------------------------------------------

GRAPH_LABELS = {
    "show_node_labels": False, # Mostrar etiquetas de los nodos
    "show_edge_labels": False, # Mostrar etiquetas de las aristas
    "show_title": True, # Mostrar título de la figura
    "show_legend": True, # Mostrar leyenda
    "show_axis": False, # Mostrar ejes de la figura
    "title": "GraphData", # Título oficial de la figura
    "legend_location": "best" # Ubicación automática de la leyenda
} # Configuración oficial de etiquetas del GraphData

# =============================================================================
# BLOQUE 4. Benchmark Científico
#
# Objetivo:
# Centralizar la configuración oficial utilizada para la representación gráfica
# de los resultados del Benchmark Científico, definiendo los parámetros
# visuales empleados para comparar el desempeño de los modelos evaluados durante
# el Pipeline Científico.
# Pregunta científica:
# ¿La configuración gráfica del Benchmark Científico garantiza una
# representación visual consistente, reproducible e interpretable del
# desempeño comparativo de los modelos evaluados?
# =============================================================================

# ------------------------------------------------------------------------------
# 4.1 Configuración General
# ------------------------------------------------------------------------------

BENCHMARK_FIGURE = {
    "figure_size": FIGURE_SIZE, # Tamaño oficial de las figuras del Benchmark Científico (pulgadas)
    "dpi": FIGURE_DPI, # Resolución oficial de exportación
    "format": FIGURE_FORMAT, # Formato oficial de las figuras
    "bbox_inches": FIGURE_BBOX, # Ajuste automático de márgenes
    "style": FIGURE_STYLE, # Estilo gráfico oficial
    "save": SAVE_FIGURES, # Guardar automáticamente las figuras
    "show": SHOW_FIGURES # Mostrar figuras durante la ejecución
} # Configuración general de las figuras del Benchmark Científico

# ------------------------------------------------------------------------------
# 4.2 Métricas
# ------------------------------------------------------------------------------

BENCHMARK_METRICS_FIGURE = {
    "metrics": [
        "rmse",
        "mae",
        "r2",
        "mape"
    ], # Métricas oficiales a representar

    "chart": "bar", # Tipo oficial de gráfico
    "show_values": True, # Mostrar valores sobre las barras
    "sort_values": True, # Ordenar métricas por desempeño
    "grid": True, # Mostrar cuadrícula
    "legend": False # No mostrar leyenda
} # Configuración gráfica oficial de las métricas del Benchmark Científico

# ------------------------------------------------------------------------------
# 4.3 Modelos
# ------------------------------------------------------------------------------

BENCHMARK_MODELS_FIGURE = {
    "show_model_names": True, # Mostrar nombres de los modelos
    "group_by_family": True, # Agrupar modelos por familia
    "show_family_labels": True, # Mostrar etiquetas de las familias
    "highlight_official_model": True, # Resaltar el modelo oficial
    "highlight_best_model": True, # Resaltar el modelo con mejor desempeño
    "legend": True # Mostrar leyenda
} # Configuración gráfica oficial de los modelos del Benchmark Científico

# ------------------------------------------------------------------------------
# 4.4 Ranking
# ------------------------------------------------------------------------------

BENCHMARK_RANKING_FIGURE = {
    "show_ranking": True, # Mostrar ranking oficial de los modelos
    "top_models": 10, # Número de modelos a visualizar
    "sort_descending": False, # Ordenar según la métrica oficial (menor RMSE primero)
    "highlight_official_model": True, # Resaltar el modelo oficial
    "highlight_best_model": True, # Resaltar el modelo con mejor desempeño
    "show_metric_values": True, # Mostrar el valor de la métrica utilizada para el ranking
    "show_position": True # Mostrar la posición de cada modelo
} # Configuración gráfica oficial del ranking del Benchmark Científico

# ------------------------------------------------------------------------------
# 4.5 Colores
# ------------------------------------------------------------------------------

BENCHMARK_COLORS = {
    "background": "white", # Color de fondo de las figuras
    "bars": "#1f77b4", # Color oficial de las barras
    "best_model": "#2ca02c", # Color del mejor modelo
    "official_model": "#ff7f0e", # Color del modelo oficial del proyecto
    "text": "black", # Color oficial del texto
    "grid": "#d9d9d9" # Color oficial de la cuadrícula
} # Configuración oficial de colores del Benchmark Científico

# ------------------------------------------------------------------------------
# 4.6 Tamaños
# ------------------------------------------------------------------------------

BENCHMARK_SIZES = {
    "bar_width": 0.80, # Ancho oficial de las barras
    "marker_size": 8, # Tamaño oficial de los marcadores
    "line_width": 2.0, # Grosor oficial de las líneas
    "label_size": 10, # Tamaño oficial de las etiquetas
    "title_size": 14, # Tamaño oficial del título
    "legend_size": 10 # Tamaño oficial de la leyenda
} # Configuración oficial de tamaños del Benchmark Científico

# ------------------------------------------------------------------------------
# 4.7 Etiquetas
# ------------------------------------------------------------------------------

BENCHMARK_LABELS = {
    "show_title": True, # Mostrar título de la figura
    "show_axis_labels": True, # Mostrar etiquetas de los ejes
    "show_values": True, # Mostrar valores sobre las barras o marcadores
    "show_legend": True, # Mostrar leyenda
    "show_grid": True, # Mostrar cuadrícula
    "title": "Benchmark Científico", # Título oficial de las figuras
    "x_label": "Modelos", # Etiqueta oficial del eje X
    "y_label": "Valor de la Métrica", # Etiqueta oficial del eje Y
    "legend_location": "best" # Ubicación automática de la leyenda
} # Configuración oficial de etiquetas del Benchmark Científico

# =============================================================================
# BLOQUE 5. Entrenamiento
#
# Objetivo:
# Centralizar la configuración oficial utilizada para la representación gráfica
# del proceso de entrenamiento del modelo, definiendo los parámetros visuales
# empleados para visualizar la evolución del aprendizaje durante el Pipeline
# Científico.
# Pregunta científica:
# ¿La configuración gráfica del entrenamiento garantiza una representación
# visual consistente, reproducible e interpretable del proceso de aprendizaje
# del modelo?
# =============================================================================

# ------------------------------------------------------------------------------
# 5.1 Configuración General
# ------------------------------------------------------------------------------

TRAINING_FIGURE = {
    "figure_size": FIGURE_SIZE, # Tamaño oficial de las figuras del entrenamiento (pulgadas)
    "dpi": FIGURE_DPI, # Resolución oficial de exportación
    "format": FIGURE_FORMAT, # Formato oficial de las figuras
    "bbox_inches": FIGURE_BBOX, # Ajuste automático de márgenes
    "style": FIGURE_STYLE, # Estilo gráfico oficial
    "save": SAVE_FIGURES, # Guardar automáticamente las figuras
    "show": SHOW_FIGURES # Mostrar figuras durante la ejecución
} # Configuración general de las figuras del entrenamiento

# ------------------------------------------------------------------------------
# 5.2 Curvas de Aprendizaje
# ------------------------------------------------------------------------------

LEARNING_CURVES_FIGURE = {
    "show_training_curve": True, # Mostrar la curva de entrenamiento
    "show_validation_curve": True, # Mostrar la curva de validación
    "show_best_epoch": True, # Resaltar la mejor época
    "show_grid": True, # Mostrar cuadrícula
    "show_legend": True, # Mostrar leyenda
    "line_style": "solid", # Estilo oficial de las curvas
    "marker": None # No utilizar marcadores
} # Configuración gráfica oficial de las curvas de aprendizaje

# ------------------------------------------------------------------------------
# 5.3 Métricas
# ------------------------------------------------------------------------------

TRAINING_METRICS_FIGURE = {
    "metrics": [
        "loss",
        "rmse",
        "mae"
    ], # Métricas oficiales a representar durante el entrenamiento

    "chart": "line", # Tipo oficial de gráfico
    "show_training": True, # Mostrar métricas del entrenamiento
    "show_validation": True, # Mostrar métricas de validación
    "show_best_epoch": True, # Resaltar la mejor época
    "show_grid": True, # Mostrar cuadrícula
    "show_legend": True # Mostrar leyenda
} # Configuración gráfica oficial de las métricas del entrenamiento

# ------------------------------------------------------------------------------
# 5.4 Colores
# ------------------------------------------------------------------------------

TRAINING_COLORS = {
    "background": "white", # Color de fondo de las figuras
    "training_curve": "#1f77b4", # Color oficial de la curva de entrenamiento
    "validation_curve": "#ff7f0e", # Color oficial de la curva de validación
    "best_epoch": "#2ca02c", # Color oficial para resaltar la mejor época
    "text": "black", # Color oficial del texto
    "grid": "#d9d9d9" # Color oficial de la cuadrícula
} # Configuración oficial de colores del entrenamiento

# ------------------------------------------------------------------------------
# 5.5 Tamaños
# ------------------------------------------------------------------------------

TRAINING_SIZES = {
    "line_width": 2.0, # Grosor oficial de las curvas
    "marker_size": 6, # Tamaño oficial de los marcadores
    "label_size": 10, # Tamaño oficial de las etiquetas
    "title_size": 14, # Tamaño oficial del título
    "tick_size": 10, # Tamaño oficial de las etiquetas de los ejes
    "legend_size": 10 # Tamaño oficial de la leyenda
} # Configuración oficial de tamaños del entrenamiento

# ------------------------------------------------------------------------------
# 5.6 Etiquetas
# ------------------------------------------------------------------------------

TRAINING_LABELS = {
    "show_title": True, # Mostrar título de la figura
    "show_axis_labels": True, # Mostrar etiquetas de los ejes
    "show_values": False, # No mostrar valores sobre las curvas
    "show_legend": True, # Mostrar leyenda
    "show_grid": True, # Mostrar cuadrícula
    "title": "Entrenamiento del Modelo", # Título oficial de las figuras
    "x_label": "Épocas", # Etiqueta oficial del eje X
    "y_label": "Valor de la Métrica", # Etiqueta oficial del eje Y
    "legend_location": "best" # Ubicación automática de la leyenda
} # Configuración oficial de etiquetas del entrenamiento

# =============================================================================
# BLOQUE 6. Evaluación
# Objetivo:
# Centralizar la configuración oficial utilizada para la representación gráfica
# de los resultados de la evaluación del modelo, definiendo los parámetros
# visuales empleados para visualizar el desempeño predictivo, la explicabilidad
# y los análisis derivados durante el Pipeline Científico.
# Pregunta científica:
# ¿La configuración gráfica de la evaluación garantiza una representación
# visual consistente, reproducible e interpretable del desempeño del modelo y
# de los resultados obtenidos durante el Pipeline Científico?
# =============================================================================

# ------------------------------------------------------------------------------
# 6.1 Configuración General
# ------------------------------------------------------------------------------

EVALUATION_FIGURE = {
    "figure_size": FIGURE_SIZE, # Tamaño oficial de las figuras de evaluación (pulgadas)
    "dpi": FIGURE_DPI, # Resolución oficial de exportación
    "format": FIGURE_FORMAT, # Formato oficial de las figuras
    "bbox_inches": FIGURE_BBOX, # Ajuste automático de márgenes
    "style": FIGURE_STYLE, # Estilo gráfico oficial
    "save": SAVE_FIGURES, # Guardar automáticamente las figuras
    "show": SHOW_FIGURES # Mostrar figuras durante la ejecución
} # Configuración general de las figuras de evaluación

# ------------------------------------------------------------------------------
# 6.2 Métricas
# ------------------------------------------------------------------------------

EVALUATION_METRICS_FIGURE = {
    "metrics": [
        "rmse",
        "mae",
        "r2",
        "mape"
    ], # Métricas oficiales a representar

    "chart": "bar", # Tipo oficial de gráfico
    "show_values": True, # Mostrar valores sobre las barras
    "sort_values": True, # Ordenar métricas según su desempeño
    "show_grid": True, # Mostrar cuadrícula
    "show_legend": False # No mostrar leyenda
} # Configuración gráfica oficial de las métricas de evaluación

# ------------------------------------------------------------------------------
# 6.3 Explicabilidad
# ------------------------------------------------------------------------------

EXPLAINABILITY_FIGURE = {
    "methods": [
        "feature_importance",
        "shap",
        "embedding_analysis"
    ], # Métodos oficiales de explicabilidad a representar

    "show_feature_importance": True, # Mostrar importancia de variables
    "show_shap_summary": True, # Mostrar resumen de valores SHAP
    "show_embedding_projection": True, # Mostrar proyección de embeddings
    "chart": "bar", # Tipo oficial de gráfico para importancia de variables
    "show_grid": True, # Mostrar cuadrícula
    "show_legend": False # No mostrar leyenda
} # Configuración gráfica oficial de la explicabilidad

# ------------------------------------------------------------------------------
# 6.4 Colores
# ------------------------------------------------------------------------------

EVALUATION_COLORS = {
    "background": "white", # Color de fondo de las figuras
    "metrics": "#1f77b4", # Color oficial de las métricas de evaluación
    "feature_importance": "#ff7f0e", # Color oficial de la importancia de variables
    "shap": "#2ca02c", # Color oficial de los gráficos SHAP
    "embeddings": "#9467bd", # Color oficial de la proyección de embeddings
    "text": "black", # Color oficial del texto
    "grid": "#d9d9d9" # Color oficial de la cuadrícula
} # Configuración oficial de colores de la evaluación

# ------------------------------------------------------------------------------
# 6.5 Tamaños
# ------------------------------------------------------------------------------

EVALUATION_SIZES = {
    "bar_width": 0.80, # Ancho oficial de las barras
    "marker_size": 8, # Tamaño oficial de los marcadores
    "line_width": 2.0, # Grosor oficial de las líneas
    "label_size": 10, # Tamaño oficial de las etiquetas
    "title_size": 14, # Tamaño oficial del título
    "tick_size": 10, # Tamaño oficial de las etiquetas de los ejes
    "legend_size": 10 # Tamaño oficial de la leyenda
} # Configuración oficial de tamaños de la evaluación

# ------------------------------------------------------------------------------
# 6.6 Etiquetas
# ------------------------------------------------------------------------------

EVALUATION_LABELS = {
    "show_title": True, # Mostrar título de la figura
    "show_axis_labels": True, # Mostrar etiquetas de los ejes
    "show_values": True, # Mostrar valores sobre las barras o marcadores
    "show_legend": True, # Mostrar leyenda
    "show_grid": True, # Mostrar cuadrícula
    "title": "Evaluación del Modelo", # Título oficial de las figuras
    "x_label": "Métricas", # Etiqueta oficial del eje X
    "y_label": "Valor", # Etiqueta oficial del eje Y
    "legend_location": "best" # Ubicación automática de la leyenda
} # Configuración oficial de etiquetas de la evaluación

# =============================================================================
# BLOQUE 7. Forecasting
# Objetivo:
# Centralizar la configuración oficial utilizada para la representación gráfica
# de los resultados del Forecasting, definiendo los parámetros visuales
# empleados para visualizar los pronósticos, escenarios y tendencias
# espacio-temporales generados durante el Pipeline Científico.
# Pregunta científica:
# ¿La configuración gráfica del Forecasting garantiza una representación
# visual consistente, reproducible e interpretable de los pronósticos y
# escenarios generados por el modelo?
# =============================================================================

# ------------------------------------------------------------------------------
# 7.1 Configuración General
# ------------------------------------------------------------------------------

FORECASTING_FIGURE = {
    "figure_size": FIGURE_SIZE, # Tamaño oficial de las figuras de Forecasting (pulgadas)
    "dpi": FIGURE_DPI, # Resolución oficial de exportación
    "format": FIGURE_FORMAT, # Formato oficial de las figuras
    "bbox_inches": FIGURE_BBOX, # Ajuste automático de márgenes
    "style": FIGURE_STYLE, # Estilo gráfico oficial
    "save": SAVE_FIGURES, # Guardar automáticamente las figuras
    "show": SHOW_FIGURES # Mostrar figuras durante la ejecución
} # Configuración general de las figuras de Forecasting

# ------------------------------------------------------------------------------
# 7.2 Pronósticos
# ------------------------------------------------------------------------------

FORECAST_FIGURE = {
    "show_historical": True, # Mostrar la serie histórica
    "show_forecast": True, # Mostrar la serie pronosticada
    "show_confidence_interval": False, # Mostrar intervalos de confianza
    "show_reference_line": True, # Mostrar línea de referencia entre histórico y pronóstico
    "chart": "line", # Tipo oficial de gráfico
    "show_grid": True, # Mostrar cuadrícula
    "show_legend": True # Mostrar leyenda
} # Configuración gráfica oficial de los pronósticos

# ------------------------------------------------------------------------------
# 7.3 Escenarios
# ------------------------------------------------------------------------------

SCENARIO_FIGURE = {
    "show_baseline": True, # Mostrar escenario base
    "show_optimistic": True, # Mostrar escenario optimista
    "show_pessimistic": True, # Mostrar escenario pesimista
    "chart": "line", # Tipo oficial de gráfico
    "show_grid": True, # Mostrar cuadrícula
    "show_legend": True, # Mostrar leyenda
    "highlight_reference_year": True # Resaltar el inicio del período de pronóstico
} # Configuración gráfica oficial de los escenarios

# ------------------------------------------------------------------------------
# 7.4 Colores
# ------------------------------------------------------------------------------

FORECASTING_COLORS = {
    "background": "white", # Color de fondo de las figuras
    "historical": "#1f77b4", # Color oficial de la serie histórica
    "forecast": "#ff7f0e", # Color oficial de la serie pronosticada
    "baseline": "#2ca02c", # Color oficial del escenario base
    "optimistic": "#17becf", # Color oficial del escenario optimista
    "pessimistic": "#d62728", # Color oficial del escenario pesimista
    "text": "black", # Color oficial del texto
    "grid": "#d9d9d9" # Color oficial de la cuadrícula
} # Configuración oficial de colores del Forecasting

# ------------------------------------------------------------------------------
# 7.5 Tamaños
# ------------------------------------------------------------------------------

FORECASTING_SIZES = {
    "line_width": 2.0, # Grosor oficial de las series
    "marker_size": 6, # Tamaño oficial de los marcadores
    "label_size": 10, # Tamaño oficial de las etiquetas
    "title_size": 14, # Tamaño oficial del título
    "tick_size": 10, # Tamaño oficial de las etiquetas de los ejes
    "legend_size": 10 # Tamaño oficial de la leyenda
} # Configuración oficial de tamaños del Forecasting

# ------------------------------------------------------------------------------
# 7.6 Etiquetas
# ------------------------------------------------------------------------------

FORECASTING_LABELS = {
    "show_title": True, # Mostrar título de la figura
    "show_axis_labels": True, # Mostrar etiquetas de los ejes
    "show_values": False, # No mostrar valores sobre las series
    "show_legend": True, # Mostrar leyenda
    "show_grid": True, # Mostrar cuadrícula
    "title": "Forecasting", # Título oficial de las figuras
    "x_label": "Año", # Etiqueta oficial del eje X
    "y_label": "Valor Pronosticado", # Etiqueta oficial del eje Y
    "legend_location": "best" # Ubicación automática de la leyenda
} # Configuración oficial de etiquetas del Forecasting

# =============================================================================
# BLOQUE 8. Plataforma GeoAI
# Objetivo:
# Centralizar la configuración oficial utilizada para la representación gráfica
# de los componentes de la Plataforma GeoAI, definiendo los parámetros visuales
# empleados para visualizar el Dashboard, los reportes, la API y el Agente
# Inteligente durante el Pipeline Científico.
# Pregunta científica:
# ¿La configuración gráfica de la Plataforma GeoAI garantiza una representación
# visual consistente, reproducible e interpretable de los componentes que
# integran el ecosistema GeoAI?
# =============================================================================

# ------------------------------------------------------------------------------
# 8.1 Configuración General
# ------------------------------------------------------------------------------

GEOAI_PLATFORM_FIGURE = {
    "figure_size": FIGURE_SIZE, # Tamaño oficial de las figuras de la Plataforma GeoAI (pulgadas)
    "dpi": FIGURE_DPI, # Resolución oficial de exportación
    "format": FIGURE_FORMAT, # Formato oficial de las figuras
    "bbox_inches": FIGURE_BBOX, # Ajuste automático de márgenes
    "style": FIGURE_STYLE, # Estilo gráfico oficial
    "save": SAVE_FIGURES, # Guardar automáticamente las figuras
    "show": SHOW_FIGURES # Mostrar figuras durante la ejecución
} # Configuración general de las figuras de la Plataforma GeoAI

# ------------------------------------------------------------------------------
# 8.2 Dashboard
# ------------------------------------------------------------------------------

DASHBOARD_FIGURE = {
    "show_map": True, # Mostrar mapa principal
    "show_graph": True, # Mostrar visualización del GraphData
    "show_metrics": True, # Mostrar métricas del modelo
    "show_forecasting": True, # Mostrar resultados del Forecasting
    "show_filters": True, # Mostrar filtros interactivos
    "show_legend": True, # Mostrar leyenda
    "layout": "grid" # Distribución oficial de los componentes
} # Configuración gráfica oficial del Dashboard GeoAI

# ------------------------------------------------------------------------------
# 8.3 Reportes
# ------------------------------------------------------------------------------

REPORT_FIGURE = {
    "show_cover": True, # Mostrar portada del reporte
    "show_table_of_contents": True, # Mostrar tabla de contenido
    "show_figures": True, # Incluir figuras
    "show_tables": True, # Incluir tablas
    "show_maps": True, # Incluir mapas
    "show_appendix": True, # Incluir anexos
    "page_orientation": "portrait" # Orientación oficial del reporte
} # Configuración gráfica oficial de los reportes

# ------------------------------------------------------------------------------
# 8.4 API
# ------------------------------------------------------------------------------

API_FIGURE = {
    "show_client": True, # Mostrar cliente de la API
    "show_endpoints": True, # Mostrar endpoints disponibles
    "show_requests": True, # Mostrar flujo de solicitudes
    "show_responses": True, # Mostrar flujo de respuestas
    "show_data_flow": True, # Mostrar flujo de datos
    "show_legend": True, # Mostrar leyenda
    "layout": "horizontal" # Distribución oficial de los componentes
} # Configuración gráfica oficial de la API

# ------------------------------------------------------------------------------
# 8.5 Colores
# ------------------------------------------------------------------------------

GEOAI_PLATFORM_COLORS = {
    "background": "white", # Color de fondo de las figuras
    "dashboard": "#1f77b4", # Color oficial del Dashboard GeoAI
    "reports": "#ff7f0e", # Color oficial de los reportes
    "api": "#2ca02c", # Color oficial de la API
    "agent": "#9467bd", # Color oficial del Agente Inteligente
    "text": "black", # Color oficial del texto
    "grid": "#d9d9d9" # Color oficial de la cuadrícula
} # Configuración oficial de colores de la Plataforma GeoAI

# ------------------------------------------------------------------------------
# 8.6 Tamaños
# ------------------------------------------------------------------------------

GEOAI_PLATFORM_SIZES = {
    "component_width": 2.0, # Ancho oficial de los componentes
    "component_height": 1.2, # Alto oficial de los componentes
    "icon_size": 24, # Tamaño oficial de los iconos
    "label_size": 10, # Tamaño oficial de las etiquetas
    "title_size": 14, # Tamaño oficial del título
    "legend_size": 10 # Tamaño oficial de la leyenda
} # Configuración oficial de tamaños de la Plataforma GeoAI

# ------------------------------------------------------------------------------
# 8.7 Etiquetas
# ------------------------------------------------------------------------------

GEOAI_PLATFORM_LABELS = {
    "show_title": True, # Mostrar título de la figura
    "show_component_labels": True, # Mostrar etiquetas de los componentes
    "show_legend": True, # Mostrar leyenda
    "show_grid": False, # No mostrar cuadrícula
    "title": "Plataforma GeoAI", # Título oficial de las figuras
    "legend_location": "best" # Ubicación automática de la leyenda
} # Configuración oficial de etiquetas de la Plataforma GeoAI