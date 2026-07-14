# 01_build_divipola.R

# ---- Bloque 1. Configuración y rutas ----
# Configuración y rutas
# ============================================================
source(here::here("config", "00_packages.R"))
source(here::here("config", "01_paths.R"))
source(here::here("config", "02_global_parameters.R"))

# Rutas DIVIPOLA
ruta_raw_divipola <- file.path(
  ruta_raw,
  "divipola",
  "divipola_municipios.csv"
) # Archivo fuente

ruta_processed_divipola <- file.path(
  ruta_processed,
  "divipola",
  "divipola_municipio_anio.parquet"
) # Dataset final municipio-año

ruta_outputs_divipola <- file.path(
  ruta_outputs,
  "divipola"
) # Carpeta auditorías

# Crear carpetas necesarias
dir.create(
  dirname(ruta_processed_divipola),
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta processed/divipola

dir.create(
  ruta_outputs_divipola,
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta output/divipola

# Validar archivo fuente
if (!file.exists(ruta_raw_divipola)) {
  stop(
    paste(
      "No existe el archivo:",
      ruta_raw_divipola
    )
  ) # Detener ejecución
}

# Fecha de proceso
fecha_proceso <- Sys.time() # Timestamp auditoría

# Resumen inicial
cat("\n")
cat("CONSTRUCCIÓN PANEL DIVIPOLA\n")
cat("Archivo fuente :", basename(ruta_raw_divipola), "\n")
cat("Periodo        :", min(anios_pipeline), "-", max(anios_pipeline), "\n")
cat("Unidad análisis:", unidad_analisis, "\n")
cat("Fecha proceso  :", format(fecha_proceso), "\n")
# ---- Fin Bloque 1. Configuración y rutas ----

# ---- Bloque 2. Ingesta y limpieza DIVIPOLA ----
# Leer archivo fuente
divipola_raw <- readr::read_csv(
  ruta_raw_divipola,
  col_types = readr::cols(
    COD_DPTO = readr::col_character(),
    COD_MPIO = readr::col_character(),
    LATITUD  = readr::col_character(),
    LONGITUD = readr::col_character()
  ),
  show_col_types = FALSE
) # Leer preservando códigos y coordenadas

head(divipola_raw[, c("LATITUD", "LONGITUD")], 10)

# Dimensiones originales
filas_originales <- nrow(divipola_raw) # Total registros
columnas_originales <- ncol(divipola_raw) # Total variables

# Estandarizar estructura
divipola <- divipola_raw |>
  janitor::clean_names() |>
  dplyr::transmute(
    
    cod_depto = stringr::str_pad(
      gsub("\\.", "", as.character(cod_dpto)),
      width = 2,
      side = "left",
      pad = "0"
    ),
    
    departamento = stringr::str_squish(
      stringr::str_to_upper(nom_dpto)
    ),
    
    cod_mpio = stringr::str_pad(
      gsub("\\.", "", as.character(cod_mpio)),
      width = 5,
      side = "left",
      pad = "0"
    ),
    
    municipio = stringr::str_squish(
      stringr::str_to_upper(nom_mpio)
    ),
    
    latitud = as.numeric(
      gsub(",", ".", latitud)
    ),
    
    longitud = as.numeric(
      gsub(",", ".", longitud)
    )
    
  ) # Estandarizar variables

# Auditoría coordenadas
coordenadas_invalidas <- divipola |>
  dplyr::filter(
    
    is.na(latitud) |
      is.na(longitud) |
      
      latitud < -6 |
      latitud > 17 |
      
      longitud < -84 |
      longitud > -65
    
  ) # Coordenadas fuera de Colombia

nrow(coordenadas_invalidas)

# ------------------------------------
# Auditoría 1 - Integridad de códigos
codigos_mpio_duplicados <- divipola |>
  dplyr::count(cod_mpio, name = "n") |>
  dplyr::filter(n > 1)

codigos_depto_invalidos <- divipola |>
  dplyr::filter(
    stringr::str_length(cod_depto) != 2
  )

codigos_mpio_invalidos <- divipola |>
  dplyr::filter(
    stringr::str_length(cod_mpio) != 5
  )

inconsistencias_divipola <- divipola |>
  dplyr::filter(
    substr(cod_mpio, 1, 2) != cod_depto
  )

# Auditoría 2 - Coordenadas
coordenadas_invalidas <- divipola |>
  dplyr::filter(
    
    is.na(latitud) |
      is.na(longitud) |
      
      latitud < -6 |
      latitud > 17 |
      
      longitud < -84 |
      longitud > -65
    
  )

# Auditoría 3 - Cobertura
municipios_unicos <- dplyr::n_distinct(divipola$cod_mpio)

departamentos_unicos <- dplyr::n_distinct(divipola$cod_depto)

na_cod_mpio <- sum(is.na(divipola$cod_mpio))

na_cod_depto <- sum(is.na(divipola$cod_depto))

# ------------------- Validación final ------------------

if (nrow(codigos_mpio_duplicados) > 0) {
  
  stop(
    paste(
      "Se detectaron",
      nrow(codigos_mpio_duplicados),
      "municipios duplicados."
    )
  ) # Error crítico
  
}

if (nrow(codigos_depto_invalidos) > 0) {
  
  stop(
    paste(
      "Se detectaron",
      nrow(codigos_depto_invalidos),
      "códigos de departamento inválidos."
    )
  ) # Error crítico
  
}

if (nrow(codigos_mpio_invalidos) > 0) {
  
  stop(
    paste(
      "Se detectaron",
      nrow(codigos_mpio_invalidos),
      "códigos de municipio inválidos."
    )
  ) # Error crítico
  
}

if (nrow(inconsistencias_divipola) > 0) {
  
  stop(
    paste(
      "Se detectaron",
      nrow(inconsistencias_divipola),
      "inconsistencias entre cod_depto y cod_mpio."
    )
  ) # Error crítico
  
}

if (nrow(coordenadas_invalidas) > 0) {
  
  stop(
    paste(
      "Se detectaron",
      nrow(coordenadas_invalidas),
      "coordenadas inválidas."
    )
  ) # Error crítico
  
}

# Resumen final
cat("\n")
cat("BLOQUE 2 APROBADO\n")
cat("Municipios:", municipios_unicos, "\n")
cat("Departamentos:", departamentos_unicos, "\n")
cat("Duplicados:", nrow(codigos_mpio_duplicados), "\n")
cat("Coordenadas inválidas:", nrow(coordenadas_invalidas), "\n")
# ---- Fin Bloque 2. Ingesta y limpieza DIVIPOLA ----

# ---- Bloque 3. Construcción panel municipio-año ----
# Construcción panel municipio-año
# ============================================================
# Validar años del proyecto
if (length(anios_pipeline) == 0) {
  stop(
    "anios_pipeline está vacío."
  ) # Error crítico
}

# Construir panel municipio-año
panel_divipola <- divipola |>
  tidyr::crossing(
    anio = anios_pipeline
  ) |>
  dplyr::mutate(
    anio = as.integer(anio),
    panel_id = paste0(
      cod_mpio,
      "_",
      anio
    )
  ) |>
  dplyr::arrange(
    cod_mpio,
    anio
  ) # Panel ordenado

# Validación de integridad
filas_esperadas <- nrow(divipola) * length(anios_pipeline)

filas_reales <- nrow(
  panel_divipola
)

if (filas_reales != filas_esperadas) {
  stop(
    paste(
      "Panel incompleto.",
      "Esperadas:",
      filas_esperadas,
      "| Reales:",
      filas_reales
    )
  ) # Error crítico
}

# Validar panel_id único
panel_id_duplicados <- panel_divipola |>
  dplyr::count(
    panel_id
  ) |>
  dplyr::filter(
    n > 1
  )

if (nrow(panel_id_duplicados) > 0) {
  stop(
    paste(
      "Se detectaron",
      nrow(panel_id_duplicados),
      "panel_id duplicados."
    )
  ) # Error crítico
}

# Métricas
municipios_panel <- dplyr::n_distinct(
  panel_divipola$cod_mpio
)

anios_panel <- dplyr::n_distinct(
  panel_divipola$anio
)

# Resumen
cat("\n")
cat("PANEL MUNICIPIO-AÑO\n")
cat("Municipios :", municipios_panel, "\n")
cat("Años       :", anios_panel, "\n")
cat("Periodo    :", min(anios_pipeline), "-", max(anios_pipeline), "\n")
cat("Filas      :", filas_reales, "\n")
cat("Panel ID duplicados :", nrow(panel_id_duplicados), "\n")


# ------------------- Bloque 4 ------------------
# Exportación y auditoría final
# ============================================================

# Auditoría final
auditoria_divipola <- tibble::tibble(
  
  fecha_proceso = fecha_proceso,
  
  municipios = municipios_panel,
  
  departamentos = departamentos_unicos,
  
  anio_min = min(anios_pipeline),
  
  anio_max = max(anios_pipeline),
  
  n_anios = anios_panel,
  
  filas = filas_reales,
  
  panel_balanceado = filas_reales == filas_esperadas,
  
  panel_id_duplicados = nrow(panel_id_duplicados),
  
  coordenadas_invalidas = nrow(coordenadas_invalidas)
  
) # Métricas finales

# Exportar panel
if (guardar_version_parquet) {
  
  arrow::write_parquet(
    panel_divipola,
    ruta_processed_divipola
  ) # Guardar parquet
  
}

# Exportar CSV opcional
if (guardar_version_csv) {
  
  readr::write_csv(
    panel_divipola,
    fs::path_ext_set(
      ruta_processed_divipola,
      "csv"
    )
  ) # Guardar csv
  
}

# Exportar auditoría
readr::write_csv(
  auditoria_divipola,
  file.path(
    ruta_outputs_divipola,
    "auditoria_divipola.csv"
  )
) # Guardar auditoría

# Validar panel_id único
panel_id_unicos <- dplyr::n_distinct(
  panel_divipola$panel_id
) # Total panel_id únicos

cat(
  "Panel ID únicos:",
  panel_id_unicos,
  "\n"
)

cat(
  "Filas totales:",
  nrow(panel_divipola),
  "\n"
)

cat(
  "Validación panel_id:",
  panel_id_unicos == nrow(panel_divipola),
  "\n"
)

# Validar municipio-año único
municipio_anio_unicos <- panel_divipola |>
  dplyr::distinct(
    cod_mpio,
    anio
  ) |>
  nrow() # Combinaciones únicas

cat(
  "Municipio-año únicos:",
  municipio_anio_unicos,
  "\n"
)

cat(
  "Filas totales:",
  nrow(panel_divipola),
  "\n"
)

cat(
  "Validación municipio-año:",
  municipio_anio_unicos == nrow(panel_divipola),
  "\n"
)

# Resumen final
cat("\n")
cat("EXPORTACIÓN DIVIPOLA COMPLETADA\n")
cat("====================================\n")
cat("Archivo:", basename(ruta_processed_divipola), "\n")
cat("Municipios:", municipios_panel, "\n")
cat("Departamentos:", departamentos_unicos, "\n")
cat("Periodo:", min(anios_pipeline), "-", max(anios_pipeline), "\n")
cat("Filas:", filas_reales, "\n")
cat("Panel balanceado:", filas_reales == filas_esperadas, "\n")
cat("Panel ID duplicados:", nrow(panel_id_duplicados), "\n")
cat("Coordenadas inválidas:", nrow(coordenadas_invalidas), "\n")
cat("\n")
cat("BLOQUE 4 APROBADO\n")
cat("\n")
