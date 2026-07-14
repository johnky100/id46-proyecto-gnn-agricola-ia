# D:/Proyectos_IA/proyecto-gnn-agricola/cna/02_build_cna_municipio_anio.R

source(here::here("config", "00_packages.R"))          # Cargar paquetes
source(here::here("config", "01_paths.R"))              # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros

# PARTE 1: DEFINICIÓN DE RUTAS, VALIDACIÓN DE INSUMOS Y CARGA DEL CNA
# ===========================================================================

# 1. DEFINICIÓN DE RUTAS ----------------------------------------------------

ruta_processed_cna <- file.path(ruta_processed, "cna") # Carpeta CNA procesados
ruta_outputs_cna   <- file.path(ruta_outputs, "cna")   # Carpeta auditorías CNA

archivo_cna_clean <- file.path(ruta_processed_cna, "cna_clean.gpkg")             # GeoPackage CNA limpio
archivo_catalogo  <- file.path(ruta_processed_cna, "cna_catalogo_variables.csv") # Catálogo de variables

dir.create(ruta_outputs_cna,   recursive = TRUE, showWarnings = FALSE) # Crear carpeta auditorías
dir.create(ruta_processed_cna, recursive = TRUE, showWarnings = FALSE) # Crear carpeta procesados

# 2. VALIDACIÓN DE INSUMOS --------------------------------------------------

if (!file.exists(archivo_cna_clean)) stop(paste("No existe GeoPackage CNA limpio:", archivo_cna_clean)) # Validar GeoPackage
if (!file.exists(archivo_catalogo))  stop(paste("No existe catálogo CNA:", archivo_catalogo))           # Validar catálogo

# 3. CARGA DEL CATÁLOGO DE VARIABLES ----------------------------------------

catalogo_cna <- data.table::fread(archivo_catalogo) # Leer catálogo de variables

variables_excluir <- c(
  "OBJECTID", "COD_DEP", "COD_MUN", "Shape_Leng",
  "Shape_Area", "ShapeSTAre", "ShapeSTLen", "geometry"
) # Variables técnicas que no deben agregarse

variables_numericas <- catalogo_cna |>
  dplyr::filter(
    stringr::str_detect(clase, "numeric|integer|double"),
    !variable %in% variables_excluir
  ) |>
  dplyr::pull(variable) # Variables agropecuarias numéricas

cat("\nVARIABLES CNA DETECTADAS\n")
cat("Numéricas útiles:", length(variables_numericas), "\n")

# 4. CARGA DEL CNA LIMPIO ---------------------------------------------------

cna_sf  <- sf::st_read(archivo_cna_clean, quiet = TRUE) # Leer GeoPackage CNA
crs_cna <- sf::st_crs(cna_sf)                           # Extraer CRS

if (is.na(crs_cna$epsg))     stop("El CNA no tiene CRS definido")              # Validar CRS
if (crs_cna$epsg != 9377) warning(paste("CRS esperado: 9377 | encontrado:", crs_cna$epsg)) # Advertir CRS

cat("\nCNA CARGADO CORRECTAMENTE\n")
cat("Registros: ", nrow(cna_sf), "\n")
cat("Variables: ", ncol(cna_sf), "\n")
cat("CRS (EPSG):", crs_cna$epsg, "\n")

# 5. VALIDACIÓN DE ESTRUCTURA MUNICIPAL -------------------------------------
# Pregunta: ¿todos los registros tienen código DANE? ¿hay duplicados?

municipios_sin_codigo <- sum(is.na(cna_sf$COD_MUN)) # Municipios sin código DANE

if (municipios_sin_codigo > 0) {
  warning(paste("Municipios sin código DANE:", municipios_sin_codigo))
  cna_sf |>
    sf::st_drop_geometry() |>
    dplyr::filter(is.na(COD_MUN)) |>
    dplyr::select(OBJECTID, COD_DEP, DPTO, COD_MUN, MPIO) |>
    print()
} # Reportar registros sin código — se conservan con NA

n_municipios <- dplyr::n_distinct(cna_sf$COD_MUN, na.rm = TRUE) # Municipios con código válido

cat("\nVALIDACIÓN TERRITORIAL CNA\n")
cat("Registros CNA:         ", nrow(cna_sf), "\n")
cat("Municipios válidos:    ", n_municipios, "\n")
cat("Municipios sin código: ", municipios_sin_codigo, "\n")

# 6. CREAR TABLA BASE CNA ---------------------------------------------------
# Pregunta: ¿el registro con NA en COD_MUN se conserva?
# Sí — str_pad sobre NA produce NA, no falla. El registro viaja con cod_mpio = NA.

cli::cli_progress_step("Eliminando geometría y construyendo tabla base...")

cna_tabla <- cna_sf |>
  sf::st_drop_geometry() |>
  dplyr::mutate(
    cod_mpio     = stringr::str_pad(as.character(COD_MUN), width = 5, side = "left", pad = "0"), # Código DANE 5 dígitos
    municipio    = MPIO,   # Nombre municipio
    departamento = DPTO    # Nombre departamento
  )

cli::cli_progress_done()

# Chequeo único de duplicados (solo sobre registros con código válido) -------
duplicados <- cna_tabla |>
  dplyr::filter(!is.na(cod_mpio)) |>
  dplyr::count(cod_mpio) |>
  dplyr::filter(n > 1) # Buscar municipios repetidos

if (nrow(duplicados) > 0) {
  warning(paste("Municipios con más de un registro:", nrow(duplicados), "— revisar antes de continuar"))
  print(duplicados)
} # Advertir duplicados sin detener el proceso

# 7. VALIDACIÓN DE VARIABLES REQUERIDAS PARA FEATURE ENGINEERING ------------
# Incluye variables base Y las usadas en Parte 2

variables_requeridas <- c(
  "COD_MUN", "MPIO", "DPTO",          # Identificadores territoriales
  "HaNatura", "HaAgro", "HanoAgro", "HaOtro", # Usos del suelo
  "UPA", "UPNA",                       # Unidades productivas
  "PrPropia", "PrArrien", "PrColect", "PrMixta" # Tenencia de tierra
) # Variables obligatorias para features

faltantes <- setdiff(variables_requeridas, names(cna_tabla)) # Detectar ausentes

if (length(faltantes) > 0) {
  stop(paste("Variables faltantes en CNA:", paste(faltantes, collapse = ", ")))
} # Detener si faltan variables críticas

# 8. RESUMEN DE VALIDACIÓN PARTE 1 ------------------------------------------

cli::cli_h2("Resumen CNA — Parte 1")
cli::cli_bullets(c(
  "v" = "Registros CNA:             {.val {nrow(cna_tabla)}}",
  "v" = "Municipios válidos (c/cod):{.val {dplyr::n_distinct(cna_tabla$cod_mpio, na.rm = TRUE)}}",
  "i" = "Registro(s) sin código:    {.val {municipios_sin_codigo}} (se conserva con cod_mpio = NA)",
  "v" = "Departamentos:             {.val {dplyr::n_distinct(cna_tabla$departamento)}}",
  "v" = "Variables numéricas útiles:{.val {length(variables_numericas)}}"
))

cat("Estado CNA: APROBADO PARA FEATURE ENGINEERING\n")

# PARTE 2: FEATURE ENGINEERING CNA
# ===========================================================================

cli::cli_h2("Construcción de variables derivadas CNA")

cna_features <- cna_tabla |>
  dplyr::mutate(
    
    area_total = rowSums(
      as.matrix(dplyr::pick(HaNatura, HaAgro, HanoAgro, HaOtro)),
      na.rm = TRUE
    ), # Área total reportada (ha)
    
    pct_agro = dplyr::if_else(
      area_total > 0 & !is.na(HaAgro), HaAgro / area_total, NA_real_
    ), # Proporción agropecuaria
    
    pct_natural = dplyr::if_else(
      area_total > 0 & !is.na(HaNatura), HaNatura / area_total, NA_real_
    ), # Cobertura natural
    
    pct_no_agro = dplyr::if_else(
      area_total > 0 & !is.na(HanoAgro), HanoAgro / area_total, NA_real_
    ), # Área no agropecuaria
    
    pct_otros_usos = dplyr::if_else(
      area_total > 0 & !is.na(HaOtro), HaOtro / area_total, NA_real_
    ), # Otros usos del suelo
    
    pct_propia = dplyr::if_else(
      UPA > 0 & !is.na(PrPropia), PrPropia / UPA, NA_real_
    ), # Participación de propiedad propia
    
    pct_arrendada = dplyr::if_else(
      UPA > 0 & !is.na(PrArrien), PrArrien / UPA, NA_real_
    ), # Participación de arrendamiento
    
    pct_colectiva = dplyr::if_else(
      UPA > 0 & !is.na(PrColect), PrColect / UPA, NA_real_
    ), # Participación de propiedad colectiva
    
    pct_mixta = dplyr::if_else(
      UPA > 0 & !is.na(PrMixta), PrMixta / UPA, NA_real_
    ), # Participación de propiedad mixta
    
    densidad_upa = dplyr::if_else(
      area_total > 0 & !is.na(UPA), UPA / area_total, NA_real_
    ), # UPA por hectárea
    
    densidad_upna = dplyr::if_else(
      area_total > 0 & !is.na(UPNA), UPNA / area_total, NA_real_
    ), # UPNA por hectárea
    
    pct_upna = dplyr::if_else(
      (UPA + UPNA) > 0 & !is.na(UPA) & !is.na(UPNA), UPNA / (UPA + UPNA), NA_real_
    ) # Participación de unidades no agropecuarias
  )

# PARTE 3: VALIDACIÓN FINAL DE FEATURES CNA
# ===========================================================================

cat("\nVALIDANDO FEATURES CNA\n")

tiempo_inicio_total <- Sys.time()                                              # Registrar inicio
n_registros         <- nrow(cna_features)                                      # Total registros
n_municipios        <- dplyr::n_distinct(cna_features$cod_mpio, na.rm = TRUE) # Municipios con código válido
n_variables         <- ncol(cna_features)                                      # Total variables
sin_codigo_features <- sum(is.na(cna_features$cod_mpio))                      # Registros con cod_mpio = NA

tiempo_fin_total  <- Sys.time() # Registrar fin
tiempo_total_min  <- round(as.numeric(difftime(tiempo_fin_total, tiempo_inicio_total, units = "mins")), 2) # Duración

cat("\nVALIDACIÓN DE DIMENSIONES CNA\n")
cat("Registros:             ", n_registros, "\n")
cat("Municipios válidos:    ", n_municipios, "\n")
cat("Registros sin código:  ", sin_codigo_features, "\n")
cat("Variables:             ", n_variables, "\n")
cat("Tiempo total (min):    ", tiempo_total_min, "\n")

# VALIDACIÓN 1: Duplicados reales (excluyendo NA) ---------------------------

duplicados <- cna_features |>
  dplyr::filter(!is.na(cod_mpio)) |>
  dplyr::count(cod_mpio) |>
  dplyr::filter(n > 1) # Buscar duplicados reales

if (nrow(duplicados) > 0) {
  stop(paste("Existen municipios duplicados en features:", nrow(duplicados)))
} # Detener si hay duplicados en el producto final

# VALIDACIÓN 2: Variables críticas en features ------------------------------

variables_criticas <- c(
  "area_total", "pct_agro", "pct_natural", "pct_no_agro", "pct_otros_usos",
  "pct_propia", "pct_arrendada", "pct_colectiva", "pct_mixta",
  "densidad_upa", "densidad_upna", "pct_upna"
) # Variables derivadas obligatorias

faltantes_features <- setdiff(variables_criticas, names(cna_features)) # Detectar ausentes

if (length(faltantes_features) > 0) {
  stop(paste("Variables derivadas faltantes:", paste(faltantes_features, collapse = ", ")))
} # Detener si faltan features

# VALIDACIÓN 3: Rangos de proporciones [0, 1] ------------------------------

variables_pct <- c(
  "pct_agro", "pct_natural", "pct_no_agro", "pct_otros_usos",
  "pct_propia", "pct_arrendada", "pct_colectiva", "pct_mixta", "pct_upna"
) # Variables proporcionales

for (v in variables_pct) {
  fuera_rango <- any(cna_features[[v]] < 0 | cna_features[[v]] > 1, na.rm = TRUE) # Detectar valores inválidos
  if (fuera_rango) warning(paste("Valores fuera de rango [0,1] en:", v))
} # Advertir proporciones inválidas

# VALIDACIÓN 4: Área total --------------------------------------------------

if (all(is.na(cna_features$area_total)))              stop("area_total contiene únicamente valores NA") # Verificar NA total
if (all(cna_features$area_total <= 0, na.rm = TRUE))  stop("area_total no contiene valores positivos")  # Verificar positivos

# RESUMEN FINAL PARTE 3 -----------------------------------------------------

cat("\nVALIDACIÓN EXITOSA\n")
cat("Municipios válidos:    ", n_municipios, "\n")
cat("Registros sin código:  ", sin_codigo_features, "\n")
cat("Variables finales:     ", n_variables, "\n")
cat("Estado: APROBADO PARA EXPORTACIÓN\n")

# PARTE 4: EXPORTACIÓN DE FEATURES CNA
# ===========================================================================

cli::cli_h2("Exportando productos CNA")

archivo_rds     <- file.path(ruta_processed_cna, "cna_features.rds")     # Archivo RDS
archivo_parquet <- file.path(ruta_processed_cna, "cna_features.parquet") # Archivo Parquet
archivo_csv     <- file.path(ruta_processed_cna, "cna_features.csv")     # Archivo CSV

cna_features <- cna_features |> dplyr::arrange(cod_mpio) # Ordenar por código (NA al final)

saveRDS(cna_features,         archivo_rds)                   # Exportar RDS
arrow::write_parquet(cna_features, sink = archivo_parquet)   # Exportar Parquet
readr::write_csv(cna_features,     archivo_csv)              # Exportar CSV

# VALIDACIÓN DE EXPORTACIONES -----------------------------------------------

archivos_generados <- c(archivo_rds, archivo_parquet, archivo_csv) # Archivos esperados

if (!all(file.exists(archivos_generados))) {
  stop("No todos los archivos fueron exportados correctamente")
} # Verificar exportaciones

# RESUMEN FINAL -------------------------------------------------------------

cat("\nCNA FEATURES GENERADO CORRECTAMENTE\n")
cat("Registros:       ", nrow(cna_features), "\n")
cat("Municipios:      ", dplyr::n_distinct(cna_features$cod_mpio, na.rm = TRUE), "\n")
cat("Registro sin cod:", sum(is.na(cna_features$cod_mpio)), "(conservado con cod_mpio = NA)\n")
cat("Variables:       ", ncol(cna_features), "\n")
cat("Archivo RDS:     ", basename(archivo_rds), "\n")
cat("Archivo Parquet: ", basename(archivo_parquet), "\n")
cat("Archivo CSV:     ", basename(archivo_csv), "\n")
cat("Estado: APROBADO PARA INTEGRACIÓN CON EVA Y ERA5\n")
