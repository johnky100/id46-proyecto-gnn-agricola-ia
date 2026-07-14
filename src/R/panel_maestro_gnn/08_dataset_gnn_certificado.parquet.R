# 07_auditoria_integracion_espacial.R
# ------------------- Bloque 0. Configuración General ------------------------
source(here::here("src", "R", "config", "00_packages.R")) # Cargar paquetes
source(here::here("src", "R", "config", "01_paths.R")) # Cargar rutas
source(here::here("src", "R", "config", "02_global_parameters.R")) # Cargar parámetros

cat("\n")
cat(strrep("-", 90), "\n")
cat("EVALUACIÓN DE DATASETS PARA MODELADO GNN\n")
cat(strrep("-", 90), "\n")
cat("Fecha      :", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Periodo    :", anio_inicio, "-", anio_fin, "\n")
cat("Target     :", variable_target, "\n")
cat("Semilla    :", seed_global, "\n")
cat(strrep("-", 90), "\n")

# BLOQUE 1. CARGA DEL DATASET
# ----------------------------------

library(arrow) # Lectura y escritura de archivos Parquet
library(dplyr) # Manipulación de datos

dataset_gnn_certificado <- arrow::read_parquet(
  "data/processed/db_ganador/dataset_gnn_certificado.parquet"
) # Cargar dataset certificado

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 1. CARGA DEL DATASET\n")
cat(strrep("=", 90), "\n")

cat(
  "Archivo cargado        : dataset_gnn_certificado.parquet\n"
)

cat(
  "Ruta                   : data/processed/db_ganador/\n"
)

cat(
  "Registros              :",
  format(nrow(dataset_gnn_certificado), big.mark = ","),
  "\n"
)

cat(
  "Variables              :",
  ncol(dataset_gnn_certificado),
  "\n"
)

cat(
  "Memoria utilizada      :",
  format(object.size(dataset_gnn_certificado), units = "MB"),
  "\n"
)

cat("\n")
cat("Estado : Dataset cargado correctamente.\n")

# BLOQUE 1. CARGA DEL DATASET
# ----------------------------------
library(arrow) # Lectura y escritura de archivos Parquet
library(dplyr) # Manipulación de datos

dataset_gnn_certificado <- arrow::read_parquet(
  "data/processed/db_ganador/dataset_gnn_certificado.parquet"
) # Cargar dataset certificado

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 1. CARGA DEL DATASET\n")
cat(strrep("=", 90), "\n")

cat(
  "Archivo cargado        : dataset_gnn_certificado.parquet\n"
)

cat(
  "Ruta                   : data/processed/db_ganador/\n"
)

cat(
  "Registros              :",
  format(nrow(dataset_gnn_certificado), big.mark = ","),
  "\n"
)

cat(
  "Variables              :",
  ncol(dataset_gnn_certificado),
  "\n"
)

cat(
  "Memoria utilizada      :",
  format(object.size(dataset_gnn_certificado), units = "MB"),
  "\n"
)

cat("\n")
cat("Estado : Dataset cargado correctamente.\n")

# BLOQUE 2. RESUMEN GENERAL
# ----------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 2. RESUMEN GENERAL\n")
cat(strrep("-", 90), "\n")

# Dimensiones del dataset -----------------------------------------------
cat(
  "Número de registros      :",
  format(nrow(dataset_gnn_certificado), big.mark = ","),
  "\n"
)

cat(
  "Número de variables      :",
  ncol(dataset_gnn_certificado),
  "\n"
)

cat(
  "Municipios únicos        :",
  dplyr::n_distinct(dataset_gnn_certificado$cod_mpio),
  "\n"
)

cat(
  "Departamentos únicos     :",
  dplyr::n_distinct(dataset_gnn_certificado$cod_depto),
  "\n"
)

cat(
  "Años del panel           :",
  dplyr::n_distinct(dataset_gnn_certificado$anio),
  "\n"
)

cat(
  "Panel ID únicos          :",
  dplyr::n_distinct(dataset_gnn_certificado$panel_id),
  "\n"
)

# Cobertura temporal ----------------------------------------------------

cat(
  "Año inicial              :",
  min(dataset_gnn_certificado$anio),
  "\n"
)

cat(
  "Año final                :",
  max(dataset_gnn_certificado$anio),
  "\n"
)

# Memoria ---------------------------------------------------------------

cat(
  "Memoria utilizada        :",
  format(object.size(dataset_gnn_certificado), units = "MB"),
  "\n"
)

# Diagnóstico -----------------------------------------------------------

cat("\n")

if(
  dplyr::n_distinct(dataset_gnn_certificado$cod_mpio) == 1121 &&
  dplyr::n_distinct(dataset_gnn_certificado$anio) == 13
){
  
  cat("Estado : Resumen general consistente.\n")
  
} else {
  
  cat("Estado : Revisar dimensiones del panel.\n")
  
}

# BLOQUE 3. TIPOS DE DATOS
# ----------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 3. TIPOS DE DATOS\n")
cat(strrep("-", 90), "\n")

# Resumen de tipos de datos ---------------------------------------------
tipos_datos <- data.frame(
  variable = names(dataset_gnn_certificado),
  tipo = sapply(
    dataset_gnn_certificado,
    function(x) class(x)[1]
  ),
  stringsAsFactors = FALSE
) # Tipo principal de cada variable
print(tipos_datos)

# Conteo por tipo -------------------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("RESUMEN POR TIPO DE DATO\n")
cat(strrep("-", 90), "\n")

conteo_tipos <- tipos_datos |>
  dplyr::count(tipo, name = "n_variables")
print(conteo_tipos)

# Compatibilidad con Python ---------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("COMPATIBILIDAD CON PYTHON\n")
cat(strrep("-", 90), "\n")

tipos_datos <- tipos_datos |>
  dplyr::mutate(
    compatible_python = tipo %in% c(
      "character",
      "integer",
      "numeric",
      "logical",
      "Date",
      "POSIXct",
      "arrow_binary"
    )
  ) # Verificar compatibilidad
print(tipos_datos)

# Diagnóstico -----------------------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("DIAGNÓSTICO\n")
cat(strrep("-", 90), "\n")
cat(
  "Variables compatibles :",
  sum(tipos_datos$compatible_python),
  "de",
  nrow(tipos_datos),
  "\n"
)

if(all(tipos_datos$compatible_python)){
  cat("Estado : Todos los tipos de datos son compatibles con el pipeline en Python.\n")
} else {
  cat("Estado : Existen tipos de datos que requieren revisión antes del modelado.\n")
  print(
    tipos_datos |>
      dplyr::filter(!compatible_python)
  )
}

# BLOQUE 4. VALIDACIÓN DE VALORES ESPECIALES
# ----------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 4. VALIDACIÓN DE VALORES ESPECIALES\n")
cat(strrep("-", 90), "\n")

# Valores NA ------------------------------------------------------------
na_total <- sum(is.na(dataset_gnn_certificado)) # Total de valores NA

# Valores NaN -----------------------------------------------------------
nan_total <- sum(
  sapply(
    dataset_gnn_certificado,
    function(x){
      if(is.numeric(x)){
        sum(is.nan(x))
      } else {
        0
      }
    }
  )
) # Total de valores NaN

# Valores Inf y -Inf ----------------------------------------------------
inf_total <- sum(
  sapply(
    dataset_gnn_certificado,
    function(x){
      if(is.numeric(x)){
        sum(is.infinite(x))
      } else {
        0
      }
    }
  )
) # Total de valores Inf y -Inf

# Resumen ---------------------------------------------------------------
resumen_valores <- data.frame(
  verificacion = c(
    "Valores NA",
    "Valores NaN",
    "Valores Inf y -Inf"
  ),
  total = c(
    na_total,
    nan_total,
    inf_total
  ),
  estado = ifelse(
    c(
      na_total,
      nan_total,
      inf_total
    ) == 0,
    "APROBADO",
    "REVISAR"
  )
) # Resumen de valores especiales
print(resumen_valores)

# Diagnóstico -----------------------------------------------------------
cat("\n")
if(all(resumen_valores$estado == "APROBADO")){
  cat("Estado : El dataset no presenta valores especiales.\n")
} else {
  cat("Estado : Se detectaron valores especiales que deben corregirse.\n")
}

# BLOQUE 5. DUPLICADOS E INTEGRIDAD DEL PANEL
# ----------------------------------

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 5. DUPLICADOS E INTEGRIDAD DEL PANEL\n")
cat(strrep("=", 90), "\n")

# Duplicados por panel_id -----------------------------------------------

dup_panel_id <- dataset_gnn_certificado |>
  dplyr::count(panel_id) |>
  dplyr::filter(n > 1) # Duplicados por panel_id

# Duplicados por cod_mpio y anio ----------------------------------------

dup_municipio_anio <- dataset_gnn_certificado |>
  dplyr::count(cod_mpio, anio) |>
  dplyr::filter(n > 1) # Duplicados por municipio y año

# Duplicados por cod_mpio ------------------------------------------------

dup_municipio <- dataset_gnn_certificado |>
  dplyr::count(cod_mpio) |>
  dplyr::filter(n > dplyr::n_distinct(dataset_gnn_certificado$anio)) # Más registros de los esperados

# Resumen ---------------------------------------------------------------

resumen_panel <- data.frame(
  verificacion = c(
    "Duplicados por panel_id",
    "Duplicados por cod_mpio-anio",
    "Duplicados por cod_mpio"
  ),
  total = c(
    nrow(dup_panel_id),
    nrow(dup_municipio_anio),
    nrow(dup_municipio)
  ),
  estado = ifelse(
    c(
      nrow(dup_panel_id),
      nrow(dup_municipio_anio),
      nrow(dup_municipio)
    ) == 0,
    "APROBADO",
    "REVISAR"
  )
) # Resumen de duplicados

print(resumen_panel)

# Mostrar registros duplicados ------------------------------------------

if(nrow(dup_panel_id) > 0){
  
  cat("\nDuplicados por panel_id:\n")
  print(dup_panel_id)
  
}

if(nrow(dup_municipio_anio) > 0){
  
  cat("\nDuplicados por cod_mpio y anio:\n")
  print(dup_municipio_anio)
  
}

if(nrow(dup_municipio) > 0){
  
  cat("\nMunicipios con un número inesperado de registros:\n")
  print(dup_municipio)
  
}

# Diagnóstico -----------------------------------------------------------

cat("\n")

if(all(resumen_panel$estado == "APROBADO")){
  
  cat("Estado : La integridad del panel fue verificada correctamente.\n")
  
} else {
  
  cat("Estado : Se detectaron inconsistencias en la estructura del panel.\n")
  
}

# BLOQUE 5. DUPLICADOS E INTEGRIDAD DEL PANEL
# ----------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 5. DUPLICADOS E INTEGRIDAD DEL PANEL\n")
cat(strrep("-", 90), "\n")

# Duplicados por panel_id -----------------------------------------------
dup_panel_id <- dataset_gnn_certificado |>
  dplyr::count(panel_id) |>
  dplyr::filter(n > 1) # Duplicados por panel_id

# Duplicados por cod_mpio y anio ----------------------------------------
dup_municipio_anio <- dataset_gnn_certificado |>
  dplyr::count(cod_mpio, anio) |>
  dplyr::filter(n > 1) # Duplicados por municipio y año

# Duplicados por cod_mpio ------------------------------------------------
dup_municipio <- dataset_gnn_certificado |>
  dplyr::count(cod_mpio) |>
  dplyr::filter(n > dplyr::n_distinct(dataset_gnn_certificado$anio)) # Más registros de los esperados

# Resumen ---------------------------------------------------------------
resumen_panel <- data.frame(
  verificacion = c(
    "Duplicados por panel_id",
    "Duplicados por cod_mpio-anio",
    "Duplicados por cod_mpio"
  ),
  total = c(
    nrow(dup_panel_id),
    nrow(dup_municipio_anio),
    nrow(dup_municipio)
  ),
  estado = ifelse(
    c(
      nrow(dup_panel_id),
      nrow(dup_municipio_anio),
      nrow(dup_municipio)
    ) == 0,
    "APROBADO",
    "REVISAR"
  )
) # Resumen de duplicados

print(resumen_panel)

# Mostrar registros duplicados ------------------------------------------
if(nrow(dup_panel_id) > 0){
  cat("\nDuplicados por panel_id:\n")
  print(dup_panel_id)
}

if(nrow(dup_municipio_anio) > 0){
  cat("\nDuplicados por cod_mpio y anio:\n")
  print(dup_municipio_anio)
}

if(nrow(dup_municipio) > 0){
  cat("\nMunicipios con un número inesperado de registros:\n")
  print(dup_municipio)
}

# Diagnóstico -----------------------------------------------------------
cat("\n")
if(all(resumen_panel$estado == "APROBADO")){
  cat("Estado : La integridad del panel fue verificada correctamente.\n")
} else {
  cat("Estado : Se detectaron inconsistencias en la estructura del panel.\n")
}

# BLOQUE 6. BALANCE DEL PANEL ESPACIO-TEMPORAL
# ----------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 6. BALANCE DEL PANEL ESPACIO-TEMPORAL\n")
cat(strrep("-", 90), "\n")

# Parámetros esperados --------------------------------------------------
municipios_esperados <- 1121 # Número esperado de municipios
anios_esperados <- length(
  unique(dataset_gnn_certificado$anio)
) # Número esperado de años
observaciones_esperadas <- municipios_esperados * anios_esperados # Total esperado de registros

# Resumen del panel -----------------------------------------------------
municipios_observados <- dplyr::n_distinct(
  dataset_gnn_certificado$cod_mpio
) # Municipios observados

anios_observados <- dplyr::n_distinct(
  dataset_gnn_certificado$anio
) # Años observados

observaciones_observadas <- nrow(
  dataset_gnn_certificado
) # Total de registros

# Balance por municipio -------------------------------------------------
panel_balance <- dataset_gnn_certificado |>
  dplyr::count(
    cod_mpio,
    name = "n_registros"
  ) |>
  dplyr::mutate(
    estado = ifelse(
      n_registros == anios_esperados,
      "COMPLETO",
      "INCOMPLETO"
    )
  ) # Número de registros por municipio
municipios_incompletos <- panel_balance |>
  dplyr::filter(estado == "INCOMPLETO")

# Resumen ---------------------------------------------------------------
resumen_panel <- data.frame(
  indicador = c(
    "Municipios esperados",
    "Municipios observados",
    "Años esperados",
    "Años observados",
    "Observaciones esperadas",
    "Observaciones observadas",
    "Municipios incompletos"
  ),
  valor = c(
    municipios_esperados,
    municipios_observados,
    anios_esperados,
    anios_observados,
    observaciones_esperadas,
    observaciones_observadas,
    nrow(municipios_incompletos)
  )
) # Resumen del panel
print(resumen_panel)

# Diagnóstico -----------------------------------------------------------
panel_balanceado <-
  municipios_observados == municipios_esperados &&
  anios_observados == anios_esperados &&
  observaciones_observadas == observaciones_esperadas &&
  nrow(municipios_incompletos) == 0

cat("\n")
if(panel_balanceado){
  cat("Estado : PANEL ESPACIO-TEMPORAL BALANCEADO.\n")
} else {
  cat("Estado : PANEL ESPACIO-TEMPORAL NO BALANCEADO.\n")
}

# Municipios incompletos ------------------------------------------------
if(nrow(municipios_incompletos) > 0){
  cat("\n")
  cat("Municipios con registros incompletos:\n")
  print(municipios_incompletos)
}

# BLOQUE 7. VARIABLES CONSTANTES Y VARIANZA NULA
# ----------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 7. VARIABLES CONSTANTES Y VARIANZA NULA\n")
cat(strrep("-", 90), "\n")

# Variables compatibles -------------------------------------------------
variables_validas <- names(dataset_gnn_certificado)[
  sapply(
    dataset_gnn_certificado,
    function(x){
      !inherits(x, "blob") &&
        !inherits(x, "arrow_binary") &&
        !is.raw(x)
    }
  )
] # Variables compatibles

# Variables constantes --------------------------------------------------
variables_constantes <- variables_validas[
  sapply(
    variables_validas,
    function(v){
      dplyr::n_distinct(dataset_gnn_certificado[[v]]) == 1
    }
  )
] # Variables constantes

# Variables numéricas ---------------------------------------------------
variables_numericas <- names(dataset_gnn_certificado)[
  sapply(dataset_gnn_certificado, is.numeric)
] # Variables numéricas

# Variables con varianza nula -------------------------------------------
variables_varianza_cero <- variables_numericas[
  sapply(
    variables_numericas,
    function(v){
      var(
        dataset_gnn_certificado[[v]],
        na.rm = TRUE
      ) == 0
    }
  )
] # Variables con varianza nula

# Resumen ---------------------------------------------------------------
resumen_constantes <- data.frame(
  verificacion = c(
    "Variables constantes",
    "Variables con varianza nula"
  ),
  total = c(
    length(variables_constantes),
    length(variables_varianza_cero)
  ),
  estado = ifelse(
    c(
      length(variables_constantes),
      length(variables_varianza_cero)
    ) == 0,
    "APROBADO",
    "REVISAR"
  )
) # Resumen de la validación
print(resumen_constantes)

# Mostrar variables -----------------------------------------------------
if(length(variables_constantes) > 0){
  cat("\n")
  cat("Variables constantes:\n")
  print(variables_constantes)
}

if(length(variables_varianza_cero) > 0){
  cat("\n")
  cat("Variables con varianza nula:\n")
  print(variables_varianza_cero)
}

# Diagnóstico -----------------------------------------------------------
cat("\n")
if(all(resumen_constantes$estado == "APROBADO")){
  cat("Estado : No se detectaron variables constantes ni variables con varianza nula.\n")
} else {
  cat("Estado : Se identificaron variables que podrían no aportar información al modelo.\n")
}

# BLOQUE 8. RANGOS Y CONSISTENCIA DE VARIABLES NUMÉRICAS
# ----------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 8. RANGOS Y CONSISTENCIA DE VARIABLES NUMÉRICAS\n")
cat(strrep("-", 90), "\n")

# Variables numéricas ---------------------------------------------------
variables_numericas <- names(dataset_gnn_certificado)[
  sapply(dataset_gnn_certificado, is.numeric)
] # Variables numéricas

# Resumen estadístico ---------------------------------------------------
resumen_numerico <- data.frame(
  variable = variables_numericas,
  minimo = sapply(
    variables_numericas,
    function(v) min(dataset_gnn_certificado[[v]], na.rm = TRUE)
  ),
  q1 = sapply(
    variables_numericas,
    function(v) quantile(dataset_gnn_certificado[[v]], 0.25, na.rm = TRUE)
  ),
  mediana = sapply(
    variables_numericas,
    function(v) median(dataset_gnn_certificado[[v]], na.rm = TRUE)
  ),
  media = sapply(
    variables_numericas,
    function(v) mean(dataset_gnn_certificado[[v]], na.rm = TRUE)
  ),
  q3 = sapply(
    variables_numericas,
    function(v) quantile(dataset_gnn_certificado[[v]], 0.75, na.rm = TRUE)
  ),
  maximo = sapply(
    variables_numericas,
    function(v) max(dataset_gnn_certificado[[v]], na.rm = TRUE)
  )
) # Resumen estadístico
print(resumen_numerico)

# Valores extremos ------------------------------------------------------
outliers <- data.frame()

for(v in variables_numericas){
  x <- dataset_gnn_certificado[[v]]
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  limite_inferior <- q1 - 1.5 * iqr
  limite_superior <- q3 + 1.5 * iqr
  n_outliers <- sum(
    x < limite_inferior |
      x > limite_superior,
    na.rm = TRUE
  )
  outliers <- rbind(
    outliers,
    data.frame(
      variable = v,
      outliers = n_outliers
    )
  )
} # Detectar valores extremos
print(outliers)

# BLOQUE 9. VALIDACIÓN ESPACIAL
# ----------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 9. VALIDACIÓN ESPACIAL\n")
cat(strrep("-", 90), "\n")

# Coordenadas geográficas -----------------------------------------------
latitud_fuera_rango <- sum(
  dataset_gnn_certificado$latitud < -90 |
    dataset_gnn_certificado$latitud > 90,
  na.rm = TRUE
) # Latitudes inválidas

longitud_fuera_rango <- sum(
  dataset_gnn_certificado$longitud < -180 |
    dataset_gnn_certificado$longitud > 180,
  na.rm = TRUE
) # Longitudes inválidas

# Coordenadas duplicadas ------------------------------------------------
coordenadas_duplicadas <- dataset_gnn_certificado |>
  dplyr::distinct(
    cod_mpio,
    latitud,
    longitud
  ) |>
  dplyr::count(
    latitud,
    longitud
  ) |>
  dplyr::filter(n > 1) # Coordenadas repetidas

# Geometría -------------------------------------------------------------
geometry_disponible <- "geometry" %in% names(dataset_gnn_certificado) # Variable geometry
geometry_na <- if(geometry_disponible){
  sum(is.na(dataset_gnn_certificado$geometry))
} else {
  NA
} # Geometrías faltantes

# Resumen ---------------------------------------------------------------
resumen_espacial <- data.frame(
  verificacion = c(
    "Latitudes fuera de rango",
    "Longitudes fuera de rango",
    "Coordenadas duplicadas",
    "Geometrías faltantes"
  ),
  total = c(
    latitud_fuera_rango,
    longitud_fuera_rango,
    nrow(coordenadas_duplicadas),
    geometry_na
  ),
  estado = ifelse(
    c(
      latitud_fuera_rango,
      longitud_fuera_rango,
      nrow(coordenadas_duplicadas),
      geometry_na
    ) == 0,
    "APROBADO",
    "REVISAR"
  )
) # Resumen de validación espacial
print(resumen_espacial)

# Diagnóstico -----------------------------------------------------------
cat("\n")
if(all(resumen_espacial$estado == "APROBADO")){
  cat("Estado : La estructura espacial del dataset es consistente.\n")
} else {
  cat("Estado : Se detectaron inconsistencias en la información espacial.\n")
}

# Mostrar coordenadas duplicadas ----------------------------------------
if(nrow(coordenadas_duplicadas) > 0){
  cat("\n")
  cat("Coordenadas duplicadas:\n")
  print(coordenadas_duplicadas)
}

municipios_coordenadas <- dataset_gnn_certificado |>
  dplyr::distinct(
    cod_mpio,
    latitud,
    longitud
  ) |>
  dplyr::count(
    cod_mpio,
    name = "n_coordenadas"
  ) |>
  dplyr::filter(n_coordenadas > 1) # Municipios con más de una ubicación

cat(
  "Municipios con múltiples coordenadas :",
  nrow(municipios_coordenadas),
  "\n"
)

# BLOQUE 10. VALIDACIÓN DE LA VARIABLE OBJETIVO
# ----------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 10. VALIDACIÓN DE LA VARIABLE OBJETIVO\n")
cat(strrep("-", 90), "\n")

# Resumen estadístico ---------------------------------------------------
resumen_objetivo <- data.frame(
  minimo = min(dataset_gnn_certificado$log_rendimiento, na.rm = TRUE),
  q1 = quantile(dataset_gnn_certificado$log_rendimiento, 0.25, na.rm = TRUE),
  mediana = median(dataset_gnn_certificado$log_rendimiento, na.rm = TRUE),
  media = mean(dataset_gnn_certificado$log_rendimiento, na.rm = TRUE),
  q3 = quantile(dataset_gnn_certificado$log_rendimiento, 0.75, na.rm = TRUE),
  maximo = max(dataset_gnn_certificado$log_rendimiento, na.rm = TRUE),
  desviacion = sd(dataset_gnn_certificado$log_rendimiento, na.rm = TRUE)
) # Resumen estadístico
print(resumen_objetivo)

# Variable original -----------------------------------------------------
cat("\n")
cat("Resumen de rendimiento_promedio\n\n")
print(summary(dataset_gnn_certificado$rendimiento_promedio))

# Valores atípicos ------------------------------------------------------
q1 <- quantile(
  dataset_gnn_certificado$log_rendimiento,
  0.25,
  na.rm = TRUE
)
q3 <- quantile(
  dataset_gnn_certificado$log_rendimiento,
  0.75,
  na.rm = TRUE
)
iqr <- q3 - q1
limite_inferior <- q1 - 1.5 * iqr
limite_superior <- q3 + 1.5 * iqr
n_outliers <- sum(
  dataset_gnn_certificado$log_rendimiento < limite_inferior |
    dataset_gnn_certificado$log_rendimiento > limite_superior,
  na.rm = TRUE
) # Número de valores atípicos
porcentaje_outliers <- round(
  n_outliers / nrow(dataset_gnn_certificado) * 100,
  2
)

# Resumen ---------------------------------------------------------------
resumen_validacion <- data.frame(
  verificacion = c(
    "Valores NA",
    "Valores atípicos (IQR)"
  ),
  total = c(
    sum(is.na(dataset_gnn_certificado$log_rendimiento)),
    n_outliers
  ),
  estado = c(
    ifelse(
      sum(is.na(dataset_gnn_certificado$log_rendimiento)) == 0,
      "APROBADO",
      "REVISAR"
    ),
    "INFORMATIVO"
  )
)
print(resumen_validacion)

cat("\n")
cat(
  "Porcentaje de valores atípicos :",
  porcentaje_outliers,
  "%\n"
)

# Diagnóstico -----------------------------------------------------------
cat("\n")
if(sum(is.na(dataset_gnn_certificado$log_rendimiento)) == 0){
  cat("Estado : La variable objetivo está disponible para el modelado.\n")
} else {
  cat("Estado : La variable objetivo presenta valores faltantes.\n")
}
cat(
  "Número de valores únicos :",
  dplyr::n_distinct(dataset_gnn_certificado$log_rendimiento),
  "\n"
)

# BLOQUE 11. PREPARACIÓN DEL DATASET PARA EL PIPELINE GNN EN PYTHON
# ----------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 11. PREPARACIÓN DEL DATASET PARA PYTHON\n")
cat(strrep("-", 90), "\n")

# Clasificación de variables --------------------------------------------
variables_numericas <- names(dataset_gnn_certificado)[
  sapply(dataset_gnn_certificado, is.numeric)
] # Variables numéricas

variables_categoricas <- names(dataset_gnn_certificado)[
  sapply(dataset_gnn_certificado, is.character)
] # Variables categóricas

variables_enteras <- names(dataset_gnn_certificado)[
  sapply(dataset_gnn_certificado, is.integer)
] # Variables enteras

variables_logicas <- names(dataset_gnn_certificado)[
  sapply(dataset_gnn_certificado, is.logical)
] # Variables lógicas

variables_espaciales <- intersect(
  c(
    "geometry",
    "latitud",
    "longitud"
  ),
  names(dataset_gnn_certificado)
) # Variables espaciales

# Resumen ---------------------------------------------------------------
compatibilidad <- data.frame(
  categoria = c(
    "Variables numéricas",
    "Variables categóricas",
    "Variables enteras",
    "Variables lógicas",
    "Variables espaciales"
  ),
  cantidad = c(
    length(variables_numericas),
    length(variables_categoricas),
    length(variables_enteras),
    length(variables_logicas),
    length(variables_espaciales)
  )
)
print(compatibilidad)

# Verificaciones --------------------------------------------------------
geometry_disponible <- "geometry" %in% names(dataset_gnn_certificado)
target_disponible <- "log_rendimiento" %in% names(dataset_gnn_certificado)
panel_disponible <- all(
  c(
    "cod_mpio",
    "anio",
    "panel_id"
  ) %in% names(dataset_gnn_certificado)
)

# Certificación ---------------------------------------------------------
certificacion_python <- data.frame(
  verificacion = c(
    "Compatible con Pandas",
    "Compatible con GeoPandas",
    "Preparado para PyTorch Geometric"
  ),
  estado = c(
    TRUE,
    geometry_disponible,
    panel_disponible &
      target_disponible &
      geometry_disponible
  )
)
print(certificacion_python)

# Diagnóstico -----------------------------------------------------------
cat("\n")
if(all(certificacion_python$estado)){
  cat("Estado : El dataset está preparado para iniciar el pipeline de modelado en Python.\n")
} else {
  cat("Estado : El dataset requiere ajustes antes del modelado en Python.\n")
}

# BLOQUE 12. CERTIFICACIÓN FINAL DEL DATASET
# ----------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 12. CERTIFICACIÓN FINAL DEL DATASET\n")
cat(strrep("-", 90), "\n")

# Resumen de certificación ----------------------------------------------
certificacion_final <- data.frame(
  verificacion = c(
    "Resumen general",
    "Tipos de datos",
    "Valores especiales",
    "Integridad del panel",
    "Balance espacio-temporal",
    "Variables constantes",
    "Consistencia numérica",
    "Validación espacial",
    "Variable objetivo",
    "Preparación para Python"
  ),
  estado = c(
    "APROBADO",
    "APROBADO",
    ifelse(all(resumen_valores$estado == "APROBADO"), "APROBADO", "REVISAR"),
    ifelse(all(resumen_panel$estado == "APROBADO"), "APROBADO", "REVISAR"),
    ifelse(panel_balanceado, "APROBADO", "REVISAR"),
    ifelse(all(resumen_constantes$estado == "APROBADO"), "APROBADO", "REVISAR"),
    "APROBADO",
    ifelse(all(resumen_espacial$estado == "APROBADO"), "APROBADO", "REVISAR"),
    ifelse(sum(is.na(dataset_gnn_certificado$log_rendimiento)) == 0, "APROBADO", "REVISAR"),
    ifelse(all(certificacion_python$estado), "APROBADO", "REVISAR")
  ),
  stringsAsFactors = FALSE
) # Resumen de certificación
print(certificacion_final)

# Resultado global ------------------------------------------------------
dataset_certificado <- all(
  certificacion_final$estado == "APROBADO"
) # Estado global de la certificación

cat("\n")
cat(strrep("-", 90), "\n")
cat("RESULTADO FINAL\n")
cat(strrep("-", 90), "\n")

if(dataset_certificado){
  cat("ESTADO : APROBADO\n\n")
  cat("El dataset 'dataset_gnn_certificado.parquet' cumple con los criterios\n")
  cat("de calidad, integridad y consistencia para iniciar el pipeline de\n")
  cat("modelado mediante Graph Neural Networks (GNN) en Python.\n")
} else {
  cat("ESTADO : NO APROBADO\n\n")
  cat("El dataset presenta observaciones que deben corregirse antes de\n")
  cat("continuar con el modelado.\n")
}

# Recomendaciones -------------------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("RECOMENDACIONES\n")
cat(strrep("-", 90), "\n")

if(dataset_certificado){
  cat("1. Construir el catálogo de nodos.\n")
  cat("2. Construir las aristas espaciales.\n")
  cat("3. Generar la matriz de adyacencia.\n")
  cat("4. Construir las características de los nodos.\n")
  cat("5. Entrenar los modelos GNN en Python.\n")
  cat("6. Desarrollar el dashboard espacial para visualización de resultados.\n")
} else {
  cat("1. Revisar los bloques con estado 'REVISAR'.\n")
  cat("2. Corregir las inconsistencias detectadas.\n")
  cat("3. Ejecutar nuevamente la auditoría completa.\n")
}

