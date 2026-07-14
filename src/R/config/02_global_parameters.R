# config/02_global_parameters.R

# Pregunta: ¿Cómo garantizar la reproducibilidad del proyecto?

seed_global <- 5477976 # Semilla oficial del proyecto

set.seed(seed_global) # Fijar semilla global

# Pregunta: ¿Cuál es la ventana temporal oficial del proyecto?

anio_inicio <- 2006 # Año inicial del proyecto

anio_fin <- 2018 # Año final del proyecto

anios_pipeline <- seq(
  from = anio_inicio,
  to = anio_fin,
  by = 1
) # Secuencia oficial de años

# Pregunta: ¿Cuál es la cobertura temporal de las fuentes principales?

anio_inicio_eva <- 2006 # Inicio cobertura EVA

anio_fin_eva <- 2018 # Fin cobertura EVA

anio_inicio_chirps <- 2006 # Inicio cobertura CHIRPS

anio_fin_chirps <- 2018 # Fin cobertura CHIRPS

# Pregunta: ¿Cuál es la unidad de análisis del proyecto?

unidad_analisis <- "municipio_anio" # Unidad oficial del proyecto

granularidad_espacial <- "municipio" # Unidad espacial

granularidad_temporal <- "anio" # Unidad temporal

# Pregunta: ¿Cuáles son las llaves oficiales de integración?

variables_id <- c(
  "cod_mpio",
  "anio"
) # Llave principal municipio-año

# Pregunta: ¿Cuál es la variable objetivo del proyecto?

variable_target <- "log_rendimiento" # Variable objetivo principal

# Pregunta: ¿Qué variables tienen riesgo de leakage?

variables_prohibidas_modelo <- c(
  "rendimiento_futuro",
  "produccion_futura",
  "target_lead",
  "log_rendimiento_futuro",
  "rendimiento_lead",
  "log_rendimiento_lead"
) # Variables prohibidas para modelado

permitir_leakage_temporal <- FALSE # Control de leakage temporal

# Pregunta: ¿Cuál es el sistema de referencia espacial oficial?

crs_geografico <- "EPSG:4326" # Coordenadas geográficas WGS84

crs_proyectado <- "EPSG:9377" # Sistema oficial Colombia

# Pregunta: ¿Qué reglas de calidad se utilizarán?

umbral_na_variable <- 0.40 # Máximo porcentaje de NA permitido por variable

umbral_na_fila <- 0.50 # Máximo porcentaje de NA permitido por fila

umbral_correlacion_alta <- 0.90 # Umbral de correlación alta

# Pregunta: ¿Cómo se realizará la imputación de datos faltantes?

config_mice <- list(
  m = 5,
  maxit = 10,
  metodo_numerico = "pmm",
  metodo_binario = "logreg",
  metodo_categorico = "polyreg",
  metodo_ordinal = "polr",
  seed = seed_global
) # Configuración oficial MICE

# Pregunta: ¿Cómo se construirá el grafo?

k_vecinos_grafo <- 5 # Número inicial de vecinos

k_min_grafo <- 3 # Límite inferior para análisis de sensibilidad

k_max_grafo <- 10 # Límite superior para análisis de sensibilidad

metrica_similitud <- "cosine" # Métrica de similitud entre nodos

normalizar_features_gnn <- TRUE # Escalar variables antes del GNN

# Pregunta: ¿Cómo se exportarán los resultados?

formato_exportacion <- "parquet" # Formato principal de exportación

guardar_version_csv <- TRUE # Generar versión CSV

guardar_version_parquet <- TRUE # Generar versión Parquet

num_trees_rf <- 500 # Número de árboles Random Forest

# Pregunta: ¿Los parámetros son consistentes?

if (anio_inicio >= anio_fin) {
  stop("anio_inicio debe ser menor que anio_fin") # Validar rango temporal
}

if (umbral_na_variable < 0 || umbral_na_variable > 1) {
  stop("umbral_na_variable debe estar entre 0 y 1") # Validar porcentaje de NA
}

if (umbral_na_fila < 0 || umbral_na_fila > 1) {
  stop("umbral_na_fila debe estar entre 0 y 1") # Validar porcentaje de NA
}

if (length(variables_id) == 0) {
  stop("Debe existir al menos una variable identificadora") # Validar llaves
}

if (!variable_target %in% c("log_rendimiento")) {
  stop("Variable objetivo no reconocida") # Validar variable objetivo
}

# Pregunta: ¿Cuál es el resumen de la configuración global?

cat("\nPARÁMETROS GLOBALES CARGADOS CORRECTAMENTE\n") # Mostrar encabezado

cat(
  "Ventana temporal:",
  anio_inicio,
  "-",
  anio_fin,
  "\n"
) # Mostrar periodo oficial

cat(
  "Unidad de análisis:",
  unidad_analisis,
  "\n"
) # Mostrar unidad de análisis

cat(
  "Variable objetivo:",
  variable_target,
  "\n"
) # Mostrar variable objetivo

cat(
  "CRS geográfico:",
  crs_geografico,
  "\n"
) # Mostrar CRS geográfico

cat(
  "CRS proyectado:",
  crs_proyectado,
  "\n"
) # Mostrar CRS proyectado

cat(
  "Vecinos grafo:",
  k_vecinos_grafo,
  "\n"
) # Mostrar número de vecinos

cat(
  "Formato exportación:",
  formato_exportacion,
  "\n"
) # Mostrar formato de exportación
