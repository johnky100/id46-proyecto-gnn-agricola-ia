# 02_build_irrigacion_municipio_anio.R
# Construcción dataset municipio-año irrigación
# ============================================================

# --Bloque 1. Definición de rutas, validación de insumos y carga de datos --
source(here::here("config", "00_packages.R"))          # Cargar paquetes
source(here::here("config", "01_paths.R"))             # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros

# PARTE 1: DEFINICIÓN DE RUTAS, VALIDACIÓN DE INSUMOS Y CARGA
# ============================================================================
# 1. DEFINICIÓN DE RUTAS -----------------------------------------------------
ruta_processed_irrigacion <- file.path(ruta_processed, "irrigacion") # Carpeta irrigación
ruta_outputs_irrigacion   <- file.path(ruta_outputs, "irrigacion")   # Carpeta auditorías
ruta_spatial_raw          <- file.path(ruta_raw, "spatial")          # Carpeta capas espaciales

archivo_irrigacion <- file.path(
  ruta_processed_irrigacion,
  "irrigacion_final.gpkg"
) # Capa irrigación limpia

archivo_municipios <- file.path(
  ruta_spatial_raw,
  "Muni.shp"
) # Shapefile municipios Colombia

dir.create(
  ruta_outputs_irrigacion,
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta auditorías

# 2. VALIDACIÓN DE INSUMOS ---------------------------------------------------
if (!file.exists(archivo_irrigacion)) {
  stop(
    paste(
      "No existe archivo irrigación:",
      archivo_irrigacion
    )
  )
} # Validar irrigación

if (!file.exists(archivo_municipios)) {
  stop(
    paste(
      "No existe shapefile municipal:",
      archivo_municipios
    )
  )
} # Validar municipios

# 3. CARGA DE DATOS ----------------------------------------------------------
cli::cli_h2("Cargando capas espaciales")

irrigacion_sf <- sf::st_read(
  archivo_irrigacion,
  quiet = TRUE
) # Leer irrigación

muni_sf <- sf::st_read(
  archivo_municipios,
  quiet = TRUE
) # Leer municipios

cat("\nCAPAS CARGADAS CORRECTAMENTE\n")
cat(
  "Polígonos irrigación:",
  nrow(irrigacion_sf),
  "\n"
)

cat("Municipios:", nrow(muni_sf), "\n")

# 4. VALIDACIÓN DE CRS -------------------------------------------------------
crs_irrigacion <- sf::st_crs(irrigacion_sf)$epsg # CRS irrigación
crs_municipios <- sf::st_crs(muni_sf)$epsg       # CRS municipios

cat("\nVALIDACIÓN DE CRS\n")
cat("EPSG irrigación:", crs_irrigacion, "\n")
cat("EPSG municipios:", crs_municipios, "\n")

if (is.na(sf::st_crs(irrigacion_sf))) {
  stop(
    "La capa de irrigación no tiene CRS definido"
  )
} # Validar CRS irrigación

if (is.na(sf::st_crs(muni_sf))) {
  stop(
    "La capa municipal no tiene CRS definido"
  )
} # Validar CRS municipios

# 5. VALIDACIÓN DE VARIABLES REQUERIDAS --------------------------------------
variables_requeridas <- c(
  "id_poligono",
  "area_ha",
  "tipo_tierra_score",
  "necesidad_hidrica_score",
  "disponibilidad_score",
  "regulacion_score",
  "ecosistemico_score",
  "socioeconomico_score",
  "potencial_score"
) # Variables requeridas para agregación

faltantes <- setdiff(
  variables_requeridas,
  names(irrigacion_sf)
) # Detectar variables faltantes

if (length(faltantes) > 0) {
  stop(
    paste(
      "Variables faltantes:",
      paste(faltantes, collapse = ", ")
    )
  )
} # Detener proceso

# 6. VALIDACIÓN DE ESTRUCTURA MUNICIPAL --------------------------------------
variables_municipio <- c(
  "MunCodigo",
  "MunNombre"
) # Variables mínimas requeridas

faltantes_municipio <- setdiff(
  variables_municipio,
  names(muni_sf)
) # Detectar variables faltantes

if (length(faltantes_municipio) > 0) {
  stop(
    paste(
      "Variables municipales faltantes:",
      paste(faltantes_municipio, collapse = ", ")
    )
  )
} # Validar estructura municipal

# 7. RESUMEN PARTE 1 ---------------------------------------------------------
cli::cli_h2("Resumen Irrigación - Parte 1")

cli::cli_bullets(c(
  "v" = "Polígonos irrigación: {.val {nrow(irrigacion_sf)}}",
  "v" = "Municipios: {.val {nrow(muni_sf)}}",
  "v" = "Variables irrigación: {.val {ncol(irrigacion_sf)}}",
  "v" = "Variables requeridas encontradas"
))

cat("\nEstado: APROBADO PARA VALIDACIÓN ESPACIAL\n")
# --------Fin Bloque 1 -------------------------------------

# -------- Bloque 2 -------------------------------------
# PARTE 2: VALIDACIÓN ESPACIAL Y TERRITORIAL
# ============================================================================
cli::cli_h2("Validación espacial y territorial")

# Configuración espacial -----------------------------------------------------
sf::sf_use_s2(FALSE) # Evitar problemas topológicos en intersecciones

# 1. HOMOLOGACIÓN DE CRS -----------------------------------------------------
if (
  sf::st_crs(irrigacion_sf)$epsg !=
  sf::st_crs(muni_sf)$epsg
) {
cat("\nTransformando municipios al CRS de irrigación...\n")
  
  muni_sf <- sf::st_transform(
    muni_sf,
    sf::st_crs(irrigacion_sf)
  ) # Transformar municipios al CRS de irrigación
}

stopifnot(
  sf::st_crs(irrigacion_sf)$epsg ==
    sf::st_crs(muni_sf)$epsg
) # Confirmar CRS homogéneo

cat("\nCRS FINAL IRRIGACIÓN:", sf::st_crs(irrigacion_sf)$epsg, "\n")
cat("CRS FINAL MUNICIPIOS:", sf::st_crs(muni_sf)$epsg, "\n")

# 2. VALIDACIÓN GEOMÉTRICA ---------------------------------------------------
geom_invalid_irrigacion <- sum(
  !sf::st_is_valid(irrigacion_sf)
) # Geometrías inválidas irrigación

geom_empty_irrigacion <- sum(
  sf::st_is_empty(irrigacion_sf)
) # Geometrías vacías irrigación

geom_invalid_municipios <- sum(
  !sf::st_is_valid(muni_sf)
) # Geometrías inválidas municipios

geom_empty_municipios <- sum(
  sf::st_is_empty(muni_sf)
) # Geometrías vacías municipios

cat("\nVALIDACIÓN GEOMÉTRICA\n")
cat("Irrigación inválidas:", geom_invalid_irrigacion, "\n")
cat("Irrigación vacías:", geom_empty_irrigacion, "\n")
cat("Municipios inválidos:",  geom_invalid_municipios, "\n")
cat("Municipios vacíos:", geom_empty_municipios, "\n")

# 3. REPARACIÓN GEOMÉTRICA ---------------------------------------------------
if (geom_invalid_irrigacion > 0) {
  cat("\nReparando geometrías de irrigación...\n")
  
  irrigacion_sf <- sf::st_make_valid(
    irrigacion_sf
  ) # Reparar geometrías irrigación
}

if (geom_invalid_municipios > 0) {
  cat(
    "\nReparando geometrías municipales...\n"
  )
  
  muni_sf <- sf::st_make_valid(
    muni_sf
  ) # Reparar geometrías municipios
}

# 4. VALIDACIÓN POSTERIOR A REPARACIÓN ---------------------------------------
geom_invalid_irrigacion_final <- sum(
  !sf::st_is_valid(irrigacion_sf)
) # Geometrías inválidas finales irrigación

geom_invalid_municipios_final <- sum(
  !sf::st_is_valid(muni_sf)
) # Geometrías inválidas finales municipios

cat("\nVALIDACIÓN POSTERIOR A REPARACIÓN\n")
cat("Irrigación inválidas finales:", geom_invalid_irrigacion_final, "\n")
cat("Municipios inválidos finales:", geom_invalid_municipios_final, "\n")

# 5. VALIDACIÓN TERRITORIAL --------------------------------------------------
municipios_sin_codigo <- sum(
  is.na(muni_sf$MunCodigo) |
    muni_sf$MunCodigo == ""
) # Municipios sin código

municipios_duplicados <- muni_sf |>
  sf::st_drop_geometry() |>
  dplyr::count(MunCodigo) |>
  dplyr::filter(n > 1)

cat("\nVALIDACIÓN TERRITORIAL\n")
cat("Municipios:", nrow(muni_sf), "\n")
cat("Municipios sin código:", municipios_sin_codigo, "\n")
cat("Códigos duplicados:", nrow(municipios_duplicados), "\n")

# 6. VALIDACIÓN DE COBERTURA ESPACIAL ----------------------------------------
bbox_irrigacion <- sf::st_bbox(
  irrigacion_sf
) # Extensión irrigación

bbox_municipios <- sf::st_bbox(
  muni_sf
) # Extensión municipios

cat("\nEXTENSIÓN ESPACIAL IRRIGACIÓN\n")
print(bbox_irrigacion)

cat("\nEXTENSIÓN ESPACIAL MUNICIPIOS\n")
print(bbox_municipios)

# 7. VALIDACIÓN DE GEOMETRÍA -------------------------------------------------
tipo_geom_irrigacion <- unique(
  as.character(
    sf::st_geometry_type(irrigacion_sf)
  )
) # Tipos geométricos irrigación

tipo_geom_municipios <- unique(
  as.character(
    sf::st_geometry_type(muni_sf)
  )
) # Tipos geométricos municipios

cat("\nTIPOS GEOMÉTRICOS\n")
cat(
  "Irrigación:",
  paste(tipo_geom_irrigacion, collapse = ", "),
  "\n"
)
cat(
  "Municipios:",
  paste(tipo_geom_municipios, collapse = ", "),
  "\n"
)

# 8. RESUMEN -----------------------------------------------------------------
cli::cli_h2("Resumen Irrigación - Parte 2")

cli::cli_bullets(c(
  "v" = paste(
    "Polígonos irrigación:",
    nrow(irrigacion_sf)
  ),
  "v" = paste(
    "Municipios:",
    nrow(muni_sf)
  ),
  "v" = paste(
    "Geometrías irrigación inválidas:",
    geom_invalid_irrigacion_final
  ),
  "v" = paste(
    "Geometrías municipios inválidas:",
    geom_invalid_municipios_final
  ),
  "v" = paste(
    "Municipios sin código:",
    municipios_sin_codigo
  ),
  "v" = paste(
    "Códigos duplicados:",
    nrow(municipios_duplicados)
  )
))
cat("\nEstado: APROBADO PARA INTERSECCIÓN ESPACIAL\n")
# -------- Fin Bloque 2 -------------------------------------

# -------- Bloque 3 -------------------------------------
# PARTE 3: INTERSECCIÓN ESPACIAL Y ASIGNACIÓN MUNICIPAL
# ============================================================================

cli::cli_h2("Asignación espacial de municipios")

# 1. SELECCIÓN DE VARIABLES MUNICIPALES --------------------------------------

municipios_join <- muni_sf |>
  dplyr::select(
    MunCodigo,
    MunNombre,
    geometry
  ) # Variables necesarias para asignación

# 2. CREACIÓN DE CENTROIDES --------------------------------------------------

irrigacion_centroides <- sf::st_centroid(
  irrigacion_sf
) # Generar centroide de cada polígono

# 3. ASIGNACIÓN MUNICIPAL ----------------------------------------------------

irrigacion_municipio_sf <- sf::st_join(
  irrigacion_centroides,
  municipios_join,
  join = sf::st_within,
  left = TRUE
) # Asignar municipio que contiene el centroide

# 4. AUDITORÍA DE RESULTADOS -------------------------------------------------

poligonos_originales <- nrow(irrigacion_sf) # Polígonos originales

poligonos_finales <- nrow(
  irrigacion_municipio_sf
) # Polígonos posteriores al join

poligonos_asignados <- sum(
  !is.na(irrigacion_municipio_sf$MunCodigo)
) # Polígonos asignados

poligonos_sin_municipio <- sum(
  is.na(irrigacion_municipio_sf$MunCodigo)
) # Polígonos sin municipio

pct_asignacion <- round(
  (poligonos_asignados / poligonos_originales) * 100,
  2
) # Porcentaje asignado

cat("\nRESULTADOS DE ASIGNACIÓN\n")

cat(
  "Polígonos originales:",
  poligonos_originales,
  "\n"
)

cat(
  "Polígonos finales:",
  poligonos_finales,
  "\n"
)

cat(
  "Polígonos asignados:",
  poligonos_asignados,
  "\n"
)

cat(
  "Polígonos sin municipio:",
  poligonos_sin_municipio,
  "\n"
)

cat(
  "Porcentaje asignado:",
  pct_asignacion,
  "%\n"
)

# 5. CONTROL DE DUPLICADOS ---------------------------------------------------

duplicados_join <- irrigacion_municipio_sf |>
  sf::st_drop_geometry() |>
  dplyr::count(id_poligono) |>
  dplyr::filter(n > 1)

cat(
  "\nPolígonos duplicados:",
  nrow(duplicados_join),
  "\n"
)

# 6. MUNICIPIOS CON IRRIGACIÓN -----------------------------------------------

municipios_con_irrigacion <- irrigacion_municipio_sf |>
  sf::st_drop_geometry() |>
  dplyr::filter(!is.na(MunCodigo)) |>
  dplyr::distinct(MunCodigo) |>
  nrow()

cat(
  "Municipios con irrigación:",
  municipios_con_irrigacion,
  "\n"
)

# 7. VALIDACIONES CRÍTICAS ---------------------------------------------------

if (poligonos_originales != poligonos_finales) {
  
  stop(
    paste(
      "Número de registros cambió:",
      poligonos_originales,
      "->",
      poligonos_finales
    )
  )
  
} # Validar integridad de registros

if (nrow(duplicados_join) > 0) {
  
  stop(
    paste(
      "Existen",
      nrow(duplicados_join),
      "polígonos duplicados"
    )
  )
  
} # Validar duplicados

# 8. AUDITORÍA DE CASOS NO ASIGNADOS -----------------------------------------

if (poligonos_sin_municipio > 0) {
  
  cat(
    "\nADVERTENCIA:",
    poligonos_sin_municipio,
    "polígonos sin municipio asignado\n"
  )
  
} # Informar casos sin asignación

# 9. RESUMEN -----------------------------------------------------------------

cli::cli_h2("Resumen Irrigación - Parte 3")

cli::cli_bullets(c(
  "v" = paste(
    "Polígonos originales:",
    poligonos_originales
  ),
  "v" = paste(
    "Polígonos finales:",
    poligonos_finales
  ),
  "v" = paste(
    "Asignados:",
    poligonos_asignados
  ),
  "v" = paste(
    "Sin municipio:",
    poligonos_sin_municipio
  ),
  "v" = paste(
    "Duplicados:",
    nrow(duplicados_join)
  ),
  "v" = paste(
    "Municipios con irrigación:",
    municipios_con_irrigacion
  )
))

cat(
  "\nEstado: APROBADO PARA AGREGACIÓN MUNICIPAL\n"
)
# -------- Fin Bloque 3 -------------------------------------

# -------- Bloque 4 -------------------------------------
# PARTE 4: AGREGACIÓN MUNICIPAL
# ============================================================================

cli::cli_h2("Agregación municipal")

# 1. ELIMINAR GEOMETRÍA ------------------------------------------------------

irrigacion_municipio_dt <- irrigacion_municipio_sf |>
  sf::st_drop_geometry() |>
  data.table::as.data.table() # Convertir a tabla

# 2. VARIABLES AUXILIARES ----------------------------------------------------

irrigacion_municipio_dt[
  ,
  alto_potencial := ifelse(
    potencial_score >= 4,
    1,
    0
  )
] # Potencial alto

irrigacion_municipio_dt[
  ,
  muy_alto_potencial := ifelse(
    potencial_score == 5,
    1,
    0
  )
] # Potencial muy alto

# 3. AGREGACIÓN MUNICIPAL ----------------------------------------------------

irrigacion_municipio <- irrigacion_municipio_dt[
  ,
  .(
    
    n_poligonos_irrigacion = .N,
    
    area_irrigable_total = sum(
      area_ha,
      na.rm = TRUE
    ),
    
    area_irrigable_promedio = mean(
      area_ha,
      na.rm = TRUE
    ),
    
    area_irrigable_max = max(
      area_ha,
      na.rm = TRUE
    ),
    
    potencial_score_promedio = mean(
      potencial_score,
      na.rm = TRUE
    ),
    
    potencial_score_max = max(
      potencial_score,
      na.rm = TRUE
    ),
    
    potencial_score_sd = sd(
      potencial_score,
      na.rm = TRUE
    ),
    
    tipo_tierra_score_promedio = mean(
      tipo_tierra_score,
      na.rm = TRUE
    ),
    
    necesidad_hidrica_score_promedio = mean(
      necesidad_hidrica_score,
      na.rm = TRUE
    ),
    
    disponibilidad_score_promedio = mean(
      disponibilidad_score,
      na.rm = TRUE
    ),
    
    regulacion_score_promedio = mean(
      regulacion_score,
      na.rm = TRUE
    ),
    
    ecosistemico_score_promedio = mean(
      ecosistemico_score,
      na.rm = TRUE
    ),
    
    socioeconomico_score_promedio = mean(
      socioeconomico_score,
      na.rm = TRUE
    ),
    
    pct_alto_potencial = mean(
      alto_potencial,
      na.rm = TRUE
    ) * 100,
    
    pct_muy_alto_potencial = mean(
      muy_alto_potencial,
      na.rm = TRUE
    ) * 100
    
  ),
  by = .(
    cod_mpio = MunCodigo,
    municipio = MunNombre
  )
]

# 4. AUDITORÍA ---------------------------------------------------------------

cat(
  "\nMunicipios agregados:",
  nrow(irrigacion_municipio),
  "\n"
)

cat(
  "Variables generadas:",
  ncol(irrigacion_municipio),
  "\n"
)

# 5. RESUMEN -----------------------------------------------------------------

cli::cli_h2("Resumen Irrigación - Parte 4")

cli::cli_bullets(c(
  "v" = paste(
    "Municipios:",
    nrow(irrigacion_municipio)
  ),
  "v" = paste(
    "Variables:",
    ncol(irrigacion_municipio)
  ),
  "v" = "Agregación completada"
))

cat(
  "\nEstado: APROBADO PARA FEATURE ENGINEERING\n"
)

# Estadísticos descriptivos de la base municipal de irrigación
summary(irrigacion_municipio)
sapply(irrigacion_municipio, function(x) sum(is.na(x)))

# Auditoría de los municipios líderes
irrigacion_municipio[
  order(-area_irrigable_total)
][1:10]

# -------- Fin Bloque 4 -------------------------------------

# -------- Bloque 5 -------------------------------------
# PARTE 5: FEATURE ENGINEERING
# ============================================================================

cli::cli_h2("Feature Engineering")

# 1. CORRECCIÓN DE NULOS -----------------------------------------------------

irrigacion_municipio[
  is.na(potencial_score_sd),
  potencial_score_sd := 0
] # Municipios con un único polígono

# 2. TRANSFORMACIONES --------------------------------------------------------

irrigacion_municipio[
  ,
  log_area_irrigable_total := log1p(
    area_irrigable_total
  )
] # Área irrigable en escala log

# 3. CONCENTRACIÓN -----------------------------------------------------------

irrigacion_municipio[
  ,
  concentracion_irrigacion :=
    area_irrigable_max /
    area_irrigable_total
] # Concentración espacial

# 4. ÍNDICE DE CALIDAD -------------------------------------------------------

irrigacion_municipio[
  ,
  indice_calidad_irrigacion :=
    (
      potencial_score_promedio +
        tipo_tierra_score_promedio +
        disponibilidad_score_promedio
    ) / 3
] # Calidad integral

# 5. ÍNDICE HÍDRICO ----------------------------------------------------------

irrigacion_municipio[
  ,
  indice_hidrico :=
    disponibilidad_score_promedio -
    necesidad_hidrica_score_promedio
] # Balance hídrico

# 6. ÍNDICE DE SOSTENIBILIDAD ------------------------------------------------

irrigacion_municipio[
  ,
  indice_sostenibilidad :=
    (
      ecosistemico_score_promedio +
        socioeconomico_score_promedio
    ) / 2
] # Sostenibilidad territorial

# 7. AUDITORÍA ---------------------------------------------------------------

cat(
  "\nMunicipios:",
  nrow(irrigacion_municipio),
  "\n"
)

cat(
  "Variables:",
  ncol(irrigacion_municipio),
  "\n"
)

# 8. VALIDACIÓN DE NULOS -----------------------------------------------------

nulos_finales <- sapply(
  irrigacion_municipio,
  function(x) sum(is.na(x))
)

cat(
  "\nTotal de nulos:",
  sum(nulos_finales),
  "\n"
)

# 9. RESUMEN -----------------------------------------------------------------

cli::cli_h2("Resumen Irrigación - Parte 5")

cli::cli_bullets(c(
  "v" = paste(
    "Municipios:",
    nrow(irrigacion_municipio)
  ),
  "v" = paste(
    "Variables finales:",
    ncol(irrigacion_municipio)
  ),
  "v" = paste(
    "Nulos:",
    sum(nulos_finales)
  )
))

# Distribuciones de variables
summary(irrigacion_municipio)

# Variables con varianza 0. 
# Varianza cero = Variable inútil para ML/GNN
var_cero <- names(
  irrigacion_municipio
)[
  sapply(
    irrigacion_municipio,
    function(x) {
      is.numeric(x) &&
        dplyr::n_distinct(x) <= 1
    }
  )
]; var_cero
cat(
  "\nEstado: APROBADO PARA VALIDACIÓN FINAL\n"
)
# -------- Fin Bloque 5 -------------------------------------

# -------- Bloque 6 -------------------------------------
# PARTE 6: VALIDACIÓN FINAL Y AUDITORÍA
# ============================================================================

cli::cli_h2("Validación final y auditoría")

# 1. VALIDACIÓN DE ESTRUCTURA -----------------------------------------------

n_registros <- nrow(
  irrigacion_municipio
) # Número de municipios

n_variables <- ncol(
  irrigacion_municipio
) # Número de variables

municipios_unicos <- data.table::uniqueN(
  irrigacion_municipio$cod_mpio
) # Municipios únicos

cat("\nVALIDACIÓN ESTRUCTURAL\n")

cat(
  "Registros:",
  n_registros,
  "\n"
)

cat(
  "Variables:",
  n_variables,
  "\n"
)

cat(
  "Municipios únicos:",
  municipios_unicos,
  "\n"
)

# 2. VALIDACIÓN DE DUPLICADOS -----------------------------------------------

duplicados <- sum(
  duplicated(
    irrigacion_municipio$cod_mpio
  )
) # Municipios duplicados

cat(
  "\nDuplicados:",
  duplicados,
  "\n"
)

# 3. VALIDACIÓN DE NULOS ----------------------------------------------------

tabla_nulos <- data.frame(
  variable = names(irrigacion_municipio),
  nulos = sapply(
    irrigacion_municipio,
    function(x) sum(is.na(x))
  )
)

tabla_nulos$porcentaje_nulos <- round(
  tabla_nulos$nulos /
    n_registros * 100,
  4
)

cat(
  "\nTotal nulos:",
  sum(tabla_nulos$nulos),
  "\n"
)

# 4. TIPOS DE DATOS ---------------------------------------------------------

tabla_tipos <- data.frame(
  variable = names(irrigacion_municipio),
  tipo = sapply(
    irrigacion_municipio,
    class
  )
)

# 5. VARIABLES NUMÉRICAS ----------------------------------------------------

variables_numericas <- names(
  irrigacion_municipio
)[
  sapply(
    irrigacion_municipio,
    is.numeric
  )
]

# 6. RESUMEN ESTADÍSTICO ----------------------------------------------------

tabla_estadisticas <- data.frame(
  variable = variables_numericas,
  
  minimo = sapply(
    irrigacion_municipio[, ..variables_numericas],
    min,
    na.rm = TRUE
  ),
  
  media = sapply(
    irrigacion_municipio[, ..variables_numericas],
    mean,
    na.rm = TRUE
  ),
  
  mediana = sapply(
    irrigacion_municipio[, ..variables_numericas],
    median,
    na.rm = TRUE
  ),
  
  maximo = sapply(
    irrigacion_municipio[, ..variables_numericas],
    max,
    na.rm = TRUE
  )
)

# 7. VARIABLES CON VARIANZA CERO --------------------------------------------

var_cero <- names(
  irrigacion_municipio
)[
  sapply(
    irrigacion_municipio,
    function(x) {
      is.numeric(x) &&
        dplyr::n_distinct(x) <= 1
    }
  )
]

# 8. MUNICIPIOS LÍDERES -----------------------------------------------------

top_municipios <- irrigacion_municipio[
  order(-area_irrigable_total)
][1:20]

# 9. VALIDACIONES CRÍTICAS --------------------------------------------------

if (duplicados > 0) {
  
  stop(
    paste(
      "Existen",
      duplicados,
      "municipios duplicados"
    )
  )
  
}

if (
  sum(tabla_nulos$nulos) > 0
) {
  
  stop(
    "Existen valores faltantes"
  )
  
}

if (
  municipios_unicos != n_registros
) {
  
  stop(
    "La llave municipal no es única"
  )
  
}

# 10. RESUMEN EJECUTIVO -----------------------------------------------------

cli::cli_h2("Resumen Irrigación - Parte 6")

cli::cli_bullets(c(
  "v" = paste(
    "Municipios:",
    n_registros
  ),
  "v" = paste(
    "Variables:",
    n_variables
  ),
  "v" = paste(
    "Duplicados:",
    duplicados
  ),
  "v" = paste(
    "Nulos:",
    sum(tabla_nulos$nulos)
  ),
  "v" = paste(
    "Varianza cero:",
    length(var_cero)
  )
))

cat(
  "\nEstado: APROBADO PARA EXPORTACIÓN\n"
)
# -------- Fin Bloque 6 -------------------------------------

# --------  Bloque 7 -------------------------------------
# PARTE 7: EXPORTACIÓN Y AUDITORÍAS
# ============================================================================

cli::cli_h2("Exportación de resultados")

# 1. DEFINICIÓN DE RUTAS -----------------------------------------------------

ruta_processed_irrigacion <- file.path(
  ruta_processed,
  "irrigacion"
) # Carpeta de salida principal

ruta_outputs_irrigacion <- file.path(
  ruta_outputs,
  "irrigacion"
) # Carpeta auditorías

dir.create(
  ruta_processed_irrigacion,
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta processed

dir.create(
  ruta_outputs_irrigacion,
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta outputs

# 2. EXPORTAR DATASET FINAL --------------------------------------------------

archivo_final <- file.path(
  ruta_processed_irrigacion,
  "irrigacion_municipio.parquet"
) # Dataset final

arrow::write_parquet(
  irrigacion_municipio,
  archivo_final
) # Exportar dataset

# 3. EXPORTAR TABLA DE NULOS -------------------------------------------------

archivo_nulos <- file.path(
  ruta_outputs_irrigacion,
  "auditoria_nulos_irrigacion.csv"
) # Auditoría de nulos

data.table::fwrite(
  tabla_nulos,
  archivo_nulos
) # Exportar auditoría

# 4. EXPORTAR TIPOS DE DATOS -------------------------------------------------

archivo_tipos <- file.path(
  ruta_outputs_irrigacion,
  "auditoria_tipos_irrigacion.csv"
) # Tipos de datos

data.table::fwrite(
  tabla_tipos,
  archivo_tipos
) # Exportar auditoría

# 5. EXPORTAR ESTADÍSTICAS ---------------------------------------------------

archivo_estadisticas <- file.path(
  ruta_outputs_irrigacion,
  "auditoria_estadisticas_irrigacion.csv"
) # Estadísticas descriptivas

data.table::fwrite(
  tabla_estadisticas,
  archivo_estadisticas
) # Exportar estadísticas

# 6. EXPORTAR TOP MUNICIPIOS -------------------------------------------------

archivo_top <- file.path(
  ruta_outputs_irrigacion,
  "top_municipios_irrigacion.csv"
) # Municipios líderes

data.table::fwrite(
  top_municipios,
  archivo_top
) # Exportar ranking

# 7. REPORTE EJECUTIVO -------------------------------------------------------

reporte_ejecutivo <- data.frame(
  indicador = c(
    "municipios",
    "variables",
    "duplicados",
    "nulos",
    "varianza_cero"
  ),
  valor = c(
    n_registros,
    n_variables,
    duplicados,
    sum(tabla_nulos$nulos),
    length(var_cero)
  )
)

archivo_reporte <- file.path(
  ruta_outputs_irrigacion,
  "reporte_ejecutivo_irrigacion.csv"
) # Resumen ejecutivo

data.table::fwrite(
  reporte_ejecutivo,
  archivo_reporte
) # Exportar reporte

# 8. VALIDACIÓN DE EXPORTACIÓN -----------------------------------------------

archivos_generados <- c(
  archivo_final,
  archivo_nulos,
  archivo_tipos,
  archivo_estadisticas,
  archivo_top,
  archivo_reporte
)

validacion_exportacion <- all(
  file.exists(archivos_generados)
)

if (!validacion_exportacion) {
  
  stop(
    "Error en la exportación de archivos"
  )
  
} # Validar exportación

# 9. RESUMEN FINAL -----------------------------------------------------------

cli::cli_h2("Resumen Final Script 02")

cli::cli_bullets(c(
  "v" = paste(
    "Municipios:",
    n_registros
  ),
  "v" = paste(
    "Variables:",
    n_variables
  ),
  "v" = paste(
    "Duplicados:",
    duplicados
  ),
  "v" = paste(
    "Nulos:",
    sum(tabla_nulos$nulos)
  ),
  "v" = paste(
    "Archivos exportados:",
    length(archivos_generados)
  )
))

cat(
  "\nSCRIPT 02 FINALIZADO CORRECTAMENTE\n"
)

cat(
  "\nDataset final:",
  archivo_final,
  "\n"
)
# --------  Fin Bloque 7 -------------------------------------
