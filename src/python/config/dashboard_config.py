# dashboard_config.py

# =============================================================================
# DASHBOARD_CONFIG
# Objetivo:
# Importar las dependencias oficiales necesarias para definir la configuración
# funcional de la Plataforma GeoAI y de sus principales componentes,
# garantizando una administración centralizada, consistente y reproducible
# durante el Pipeline Científico y la Plataforma GeoAI.
# Pregunta científica:
# ¿Qué dependencias oficiales son necesarias para garantizar una configuración
# funcional consistente y reproducible de la Plataforma GeoAI y de sus
# principales componentes?
# =============================================================================

from src.python.config.config_project import (
    PROJECT_VERSION,
    PROJECT_LANGUAGE,
    PROJECT_TIMEZONE,

    PLATFORM_NAME,
    PLATFORM_TYPE,
    PLATFORM_DESCRIPTION,

    DASHBOARD_NAME,
    DASHBOARD_TYPE,
    DASHBOARD_DESCRIPTION,

    API_NAME,
    API_TYPE,
    API_DESCRIPTION,

    AI_AGENT_NAME,
    AI_AGENT_TYPE,
    AI_AGENT_DESCRIPTION,

    REPORT_SERVICE_NAME,
    REPORT_SERVICE_TYPE,
    REPORT_SERVICE_DESCRIPTION,

    EXPORT_SERVICE_NAME,
    EXPORT_SERVICE_TYPE,
    EXPORT_SERVICE_DESCRIPTION,

    DASHBOARD_HOST,
    DASHBOARD_PORT,

    API_HOST,
    API_PORT,
    API_VERSION,
)

# =============================================================================
# BLOQUE 1. Plataforma GeoAI
# Objetivo:
# Centralizar la configuración funcional de la Plataforma GeoAI, definiendo los
# parámetros generales que gobiernan su identidad, ejecución y comportamiento
# como núcleo integrador del Pipeline Científico.
# Pregunta científica:
# ¿La configuración funcional de la Plataforma GeoAI garantiza un
# funcionamiento integrado, consistente y reproducible de todos los módulos
# del ecosistema GeoAI?
# =============================================================================

# ------------------------------------------------------------------------------
# 1.1 Información
# ------------------------------------------------------------------------------

PLATFORM_INFO = {
    "name": PLATFORM_NAME,
    "version": PROJECT_VERSION,
    "description": PLATFORM_DESCRIPTION,
    "language": PROJECT_LANGUAGE,
    "timezone": PROJECT_TIMEZONE,
    "type": PLATFORM_TYPE
} # Información oficial de la Plataforma GeoAI

# ------------------------------------------------------------------------------
# 1.2 Configuración General
# ------------------------------------------------------------------------------

PLATFORM_CONFIG = {
    "enabled": True, # Habilitar la Plataforma GeoAI
    "mode": "production", # Modo oficial de ejecución
    "initialize_on_start": True, # Inicializar automáticamente la plataforma
    "load_modules": True, # Cargar los módulos configurados
    "validate_configuration": True, # Validar la configuración al iniciar
    "stop_on_error": True # Detener la ejecución ante errores críticos
} # Configuración general de la Plataforma GeoAI

# ------------------------------------------------------------------------------
# 1.3 Configuración Funcional
# ------------------------------------------------------------------------------

PLATFORM_FUNCTIONS = {
    "dashboard": True, # Habilitar el Dashboard GeoAI
    "api": True, # Habilitar la API
    "agent": True, # Habilitar el Agente Inteligente
    "reports": True, # Habilitar la generación de reportes
    "exports": True # Habilitar las exportaciones
} # Configuración funcional de la Plataforma GeoAI

# ------------------------------------------------------------------------------
# 1.4 Validación
# ------------------------------------------------------------------------------

PLATFORM_VALIDATION = {
    "validate_configuration": True, # Validar la configuración de la plataforma
    "validate_modules": True, # Validar los módulos habilitados
    "validate_dependencies": True, # Validar las dependencias requeridas
    "validate_paths": True, # Validar las rutas de trabajo
    "stop_on_error": True # Detener la inicialización ante errores críticos
} # Configuración de validación de la Plataforma GeoAI

# ------------------------------------------------------------------------------
# 1.5 Salidas
# ------------------------------------------------------------------------------

PLATFORM_OUTPUTS = {
    "generate_logs": True, # Generar registros de ejecución
    "generate_metadata": True, # Generar metadatos de la ejecución
    "save_configuration": True, # Guardar la configuración utilizada
    "save_execution_summary": True, # Guardar el resumen de la ejecución
    "overwrite": True # Sobrescribir archivos existentes
} # Configuración de salidas de la Plataforma GeoAI

# =============================================================================
# BLOQUE 2. Dashboard
# Objetivo:
# Centralizar la configuración funcional del Dashboard de la Plataforma GeoAI,
# definiendo los parámetros necesarios para la exploración, visualización e
# interacción con los resultados generados por el Pipeline Científico.
# Pregunta científica:
# ¿La configuración funcional del Dashboard garantiza una exploración
# consistente, reproducible e interactiva de los resultados generados por la
# Plataforma GeoAI?
# =============================================================================

# ------------------------------------------------------------------------------
# 2.1 Información
# ------------------------------------------------------------------------------

DASHBOARD_INFO = {
    "name": DASHBOARD_NAME,
    "version": PROJECT_VERSION,
    "description": DASHBOARD_DESCRIPTION,
    "language": PROJECT_LANGUAGE,
    "timezone": PROJECT_TIMEZONE,
    "type": DASHBOARD_TYPE
} # Información oficial del Dashboard

# ------------------------------------------------------------------------------
# 2.2 Configuración General
# ------------------------------------------------------------------------------

DASHBOARD_CONFIG = {
    "enabled": True,
    "auto_start": True,
    "debug_mode": False,
    "reload_on_change": False,
    "host": DASHBOARD_HOST,
    "port": DASHBOARD_PORT
} # Configuración general del Dashboard

# ------------------------------------------------------------------------------
# 2.3 Configuración Funcional
# ------------------------------------------------------------------------------

DASHBOARD_FUNCTIONS = {
    "interactive_map": True, # Habilitar el mapa interactivo
    "graphdata": True, # Habilitar la exploración del GraphData
    "benchmark": True, # Habilitar la visualización del Benchmark Científico
    "evaluation": True, # Habilitar la visualización de la evaluación del modelo
    "forecasting": True, # Habilitar la visualización del Forecasting
    "filters": True, # Habilitar los filtros interactivos
    "metrics": True # Habilitar la visualización de métricas
} # Configuración funcional del Dashboard

# ------------------------------------------------------------------------------
# 2.4 Validación
# ------------------------------------------------------------------------------

DASHBOARD_VALIDATION = {
    "validate_configuration": True, # Validar la configuración del Dashboard
    "validate_components": True, # Validar los componentes habilitados
    "validate_data_sources": True, # Validar las fuentes de datos requeridas
    "validate_server": True, # Validar la configuración del servidor
    "stop_on_error": True # Detener la inicialización ante errores críticos
} # Configuración de validación del Dashboard

# ------------------------------------------------------------------------------
# 2.5 Salidas
# ------------------------------------------------------------------------------

DASHBOARD_OUTPUTS = {
    "save_session": True, # Guardar la sesión del Dashboard
    "export_views": True, # Exportar las vistas del Dashboard
    "generate_snapshots": True, # Generar capturas del Dashboard
    "save_user_preferences": True, # Guardar las preferencias del usuario
    "overwrite": True # Sobrescribir archivos existentes
} # Configuración de salidas del Dashboard

# =============================================================================
# BLOQUE 3. API
# Objetivo:
# Centralizar la configuración funcional de la API de la Plataforma GeoAI,
# definiendo los parámetros necesarios para la comunicación entre los módulos
# del Pipeline Científico, el Dashboard y los servicios externos mediante una
# interfaz de programación consistente y reproducible.
# Pregunta científica:
# ¿La configuración funcional de la API garantiza una comunicación consistente,
# segura y reproducible entre los componentes de la Plataforma GeoAI?
# =============================================================================

# ------------------------------------------------------------------------------
# 3.1 Información
# ------------------------------------------------------------------------------

API_INFO = {
    "name": API_NAME,
    "version": PROJECT_VERSION,
    "description": API_DESCRIPTION,
    "language": PROJECT_LANGUAGE,
    "timezone": PROJECT_TIMEZONE,
    "type": API_TYPE
} # Información oficial de la API

# ------------------------------------------------------------------------------
# 3.2 Configuración General
# ------------------------------------------------------------------------------

API_CONFIG = {
    "enabled": True,
    "host": API_HOST,
    "port": API_PORT,
    "version": API_VERSION,
    "docs": True,
    "debug_mode": False
} # Configuración general de la API

# ------------------------------------------------------------------------------
# 3.3 Configuración Funcional
# ------------------------------------------------------------------------------

API_FUNCTIONS = {
    "graphdata": True, # Habilitar consulta del GraphData
    "benchmark": True, # Habilitar consulta del Benchmark Científico
    "training": True, # Habilitar consulta del entrenamiento
    "evaluation": True, # Habilitar consulta de la evaluación
    "forecasting": True, # Habilitar consulta del Forecasting
    "geoai": True, # Habilitar los servicios de la Plataforma GeoAI
    "documentation": True # Habilitar la documentación de la API
} # Configuración funcional de la API

# ------------------------------------------------------------------------------
# 3.4 Validación
# ------------------------------------------------------------------------------

API_VALIDATION = {
    "validate_configuration": True, # Validar la configuración de la API
    "validate_services": True, # Validar los servicios habilitados
    "validate_dependencies": True, # Validar las dependencias requeridas
    "validate_documentation": True, # Validar la documentación de la API
    "stop_on_error": True # Detener la inicialización ante errores críticos
} # Configuración de validación de la API

# ------------------------------------------------------------------------------
# 3.5 Salidas
# ------------------------------------------------------------------------------

API_OUTPUTS = {
    "generate_responses": True, # Generar respuestas de la API
    "generate_logs": True, # Generar registros de las solicitudes
    "generate_metadata": True, # Generar metadatos de las respuestas
    "save_requests": True, # Guardar el historial de solicitudes
    "overwrite": True # Sobrescribir archivos existentes
} # Configuración de salidas de la API

# =============================================================================
# BLOQUE 4. Agente Inteligente
# Objetivo:
# Centralizar la configuración funcional del Agente Inteligente de la
# Plataforma GeoAI, definiendo los parámetros necesarios para la interacción,
# el análisis y la generación de respuestas a partir de los resultados
# producidos por el Pipeline Científico.
# Pregunta científica:
# ¿La configuración funcional del Agente Inteligente garantiza una interacción
# consistente, reproducible y confiable con los resultados generados por la
# Plataforma GeoAI?
# =============================================================================

# ------------------------------------------------------------------------------
# 4.1 Información
# ------------------------------------------------------------------------------

AI_AGENT_INFO = {
    "name": AI_AGENT_NAME,
    "version": PROJECT_VERSION,
    "description": AI_AGENT_DESCRIPTION,
    "language": PROJECT_LANGUAGE,
    "timezone": PROJECT_TIMEZONE,
    "type": AI_AGENT_TYPE
} # Información oficial del Agente Inteligente

# ------------------------------------------------------------------------------
# 4.2 Configuración General
# ------------------------------------------------------------------------------

AI_AGENT_CONFIG = {
    "enabled": True, # Habilitar el Agente Inteligente
    "auto_start": True, # Iniciar automáticamente el agente
    "conversation_history": True, # Mantener el historial de conversación
    "context_memory": True, # Mantener el contexto de la conversación
    "debug_mode": False, # Ejecutar en modo depuración
    "stop_on_error": True # Detener la ejecución ante errores críticos
} # Configuración general del Agente Inteligente

# ------------------------------------------------------------------------------
# 4.3 Configuración Funcional
# ------------------------------------------------------------------------------

AI_AGENT_FUNCTIONS = {
    "graphdata_analysis": True, # Habilitar el análisis del GraphData
    "benchmark_analysis": True, # Habilitar el análisis del Benchmark Científico
    "evaluation_analysis": True, # Habilitar el análisis de la evaluación
    "forecasting_analysis": True, # Habilitar el análisis del Forecasting
    "generate_recommendations": True, # Habilitar la generación de recomendaciones
    "explain_results": True, # Habilitar la explicación de resultados
    "decision_support": True # Habilitar el apoyo a la toma de decisiones
} # Configuración funcional del Agente Inteligente

# ------------------------------------------------------------------------------
# 4.4 Validación
# ------------------------------------------------------------------------------

AI_AGENT_VALIDATION = {
    "validate_configuration": True, # Validar la configuración del Agente Inteligente
    "validate_functions": True, # Validar las funciones habilitadas
    "validate_dependencies": True, # Validar las dependencias requeridas
    "validate_context": True, # Validar la disponibilidad del contexto
    "stop_on_error": True # Detener la inicialización ante errores críticos
} # Configuración de validación del Agente Inteligente

# ------------------------------------------------------------------------------
# 4.5 Salidas
# ------------------------------------------------------------------------------

AI_AGENT_OUTPUTS = {
    "generate_responses": True, # Generar respuestas del Agente Inteligente
    "generate_recommendations": True, # Generar recomendaciones
    "generate_explanations": True, # Generar explicaciones de los resultados
    "save_conversation_history": True, # Guardar el historial de conversaciones
    "generate_metadata": True, # Generar metadatos de las interacciones
    "overwrite": True # Sobrescribir archivos existentes
} # Configuración de salidas del Agente Inteligente

# =============================================================================
# BLOQUE 5. Reportes
# Objetivo:
# Centralizar la configuración funcional de los reportes de la Plataforma
# GeoAI, definiendo los parámetros necesarios para la generación,
# organización y administración de los productos documentales derivados del
# Pipeline Científico.
# Pregunta científica:
# ¿La configuración funcional de los reportes garantiza una generación
# consistente, reproducible y organizada de la documentación producida por la
# Plataforma GeoAI?
# =============================================================================

# ------------------------------------------------------------------------------
# 5.1 Información
# ------------------------------------------------------------------------------

REPORT_INFO = {
    "name": REPORT_SERVICE_NAME,
    "version": PROJECT_VERSION,
    "description": REPORT_SERVICE_DESCRIPTION,
    "language": PROJECT_LANGUAGE,
    "timezone": PROJECT_TIMEZONE,
    "type": REPORT_SERVICE_TYPE
} # Información oficial del sistema de reportes

# ------------------------------------------------------------------------------
# 5.2 Configuración General
# ------------------------------------------------------------------------------

REPORT_CONFIG = {
    "enabled": True, # Habilitar el sistema de reportes
    "auto_generate": True, # Generar reportes automáticamente
    "organize_reports": True, # Organizar los reportes generados
    "include_metadata": True, # Incluir metadatos en los reportes
    "validate_before_generation": True, # Validar antes de generar los reportes
    "stop_on_error": True # Detener la generación ante errores críticos
} # Configuración general del sistema de reportes

# ------------------------------------------------------------------------------
# 5.3 Configuración Funcional
# ------------------------------------------------------------------------------

REPORT_FUNCTIONS = {
    "project": True, # Habilitar reportes del proyecto
    "panel": True, # Habilitar reportes del Panel Científico
    "graphdata": True, # Habilitar reportes del GraphData
    "benchmark": True, # Habilitar reportes del Benchmark Científico
    "training": True, # Habilitar reportes del entrenamiento
    "evaluation": True, # Habilitar reportes de la evaluación
    "forecasting": True, # Habilitar reportes del Forecasting
    "geoai": True # Habilitar reportes de la Plataforma GeoAI
} # Configuración funcional del sistema de reportes

# ------------------------------------------------------------------------------
# 5.4 Validación
# ------------------------------------------------------------------------------

REPORT_VALIDATION = {
    "validate_configuration": True, # Validar la configuración del sistema de reportes
    "validate_functions": True, # Validar los tipos de reportes habilitados
    "validate_data_sources": True, # Validar las fuentes de datos requeridas
    "validate_templates": True, # Validar las plantillas de los reportes
    "stop_on_error": True # Detener la generación ante errores críticos
} # Configuración de validación del sistema de reportes

# ------------------------------------------------------------------------------
# 5.5 Salidas
# ------------------------------------------------------------------------------

REPORT_OUTPUTS = {
    "generate_reports": True, # Generar los reportes científicos
    "generate_summary": True, # Generar el resumen ejecutivo
    "generate_metadata": True, # Generar los metadatos del reporte
    "generate_logs": True, # Generar los registros de la generación
    "overwrite": True # Sobrescribir archivos existentes
} # Configuración de salidas del sistema de reportes

# =============================================================================
# BLOQUE 6. Exportaciones
# Objetivo:
# Centralizar la configuración funcional del sistema de exportaciones de la
# Plataforma GeoAI, definiendo los parámetros necesarios para la generación,
# organización y distribución de los productos derivados del Pipeline
# Científico en formatos interoperables y reproducibles.
# Pregunta científica:
# ¿La configuración funcional del sistema de exportaciones garantiza una
# distribución consistente, reproducible e interoperable de los productos
# generados por la Plataforma GeoAI?
# =============================================================================

# ------------------------------------------------------------------------------
# 6.1 Información
# ------------------------------------------------------------------------------

EXPORT_INFO = {
    "name": EXPORT_SERVICE_NAME,
    "version": PROJECT_VERSION,
    "description": EXPORT_SERVICE_DESCRIPTION,
    "language": PROJECT_LANGUAGE,
    "timezone": PROJECT_TIMEZONE,
    "type": EXPORT_SERVICE_TYPE
} # Información oficial del sistema de exportaciones

# ------------------------------------------------------------------------------
# 6.2 Configuración General
# ------------------------------------------------------------------------------

EXPORT_CONFIG = {
    "enabled": True, # Habilitar el sistema de exportaciones
    "auto_export": True, # Exportar automáticamente los productos generados
    "organize_exports": True, # Organizar los archivos exportados
    "validate_before_export": True, # Validar antes de exportar
    "include_metadata": True, # Incluir metadatos en las exportaciones
    "stop_on_error": True # Detener la exportación ante errores críticos
} # Configuración general del sistema de exportaciones

# ------------------------------------------------------------------------------
# 6.3 Configuración Funcional
# ------------------------------------------------------------------------------

EXPORT_FUNCTIONS = {
    "graphdata": True,
    "benchmark": True,
    "training": True,
    "evaluation": True,
    "forecasting": True,
    "geoai": True,
    "formats": [
        "csv",
        "json",
        "parquet",
        "geopackage",
        "geojson",
        "pt"
    ]
}

# ------------------------------------------------------------------------------
# 6.4 Validación
# ------------------------------------------------------------------------------

EXPORT_VALIDATION = {
    "validate_configuration": True, # Validar la configuración del sistema de exportaciones
    "validate_functions": True, # Validar las funciones de exportación habilitadas
    "validate_formats": True, # Validar los formatos de exportación soportados
    "validate_output_directory": True, # Validar el directorio de exportación
    "stop_on_error": True # Detener la exportación ante errores críticos
} # Configuración de validación del sistema de exportaciones

# ------------------------------------------------------------------------------
# 6.5 Salidas
# ------------------------------------------------------------------------------

EXPORT_OUTPUTS = {
    "generate_export_files": True, # Generar los archivos de exportación
    "generate_export_summary": True, # Generar el resumen de la exportación
    "generate_metadata": True, # Generar los metadatos de la exportación
    "generate_logs": True, # Generar los registros de la exportación
    "overwrite": True # Sobrescribir archivos existentes
} # Configuración de salidas del sistema de exportaciones














