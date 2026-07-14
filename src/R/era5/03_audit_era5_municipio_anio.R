# 03_audit_era5_municipio_anio.R
# ---------------------------------------------------
# Auditoría del panel climático municipal-mensual ERA5-Land (2006-2018)
# Insumo: era5_municipal_mensual.parquet (generado en 02_build_era5_municipal_mensual.R)
# ---------------------------------------------------

source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros globales

# 1. DEFINICIÓN DE RUTAS -----------------------------------------------------
ruta_outputs_era5 <- file.path(
  ruta_outputs,
  "era5"
) # Carpeta de auditorías ERA5

dir.create(
  ruta_outputs_era5,
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta de salida si no existe

archivo_entrada <- here::here(
  "datos",
  "procesados",
  "era5_municipal_mensual.parquet"
) # Panel municipal-mensual (parquet, más rápido y liviano)

archivo_auditoria_general    <- file.path(ruta_outputs_era5, "auditoria_era5_municipal.csv")
archivo_auditoria_variable   <- file.path(ruta_outputs_era5, "auditoria_era5_por_variable.csv")
archivo_auditoria_municipio  <- file.path(ruta_outputs_era5, "auditoria_era5_por_municipio.csv")
archivo_cobertura_temporal   <- file.path(ruta_outputs_era5, "cobertura_era5_municipal.csv")
archivo_municipios_con_na    <- file.path(ruta_outputs_era5, "municipios_con_na_era5.csv")

# 2. VALIDACIÓN DE INSUMOS ----------------------------------------------------
if (!file.exists(archivo_entrada)) {
  stop(
    paste(
      "No existe archivo de entrada:",
      archivo_entrada
    )
  )
} # Validar existencia del parquet

# 3. CARGA DEL PANEL -----------------------------------------------------------
era5_municipal_mensual <- arrow::read_parquet(
  archivo_entrada
) # Cargar versión parquet (rápida y compacta)

cat("\nINICIANDO AUDITORÍA ERA5 MUNICIPAL-MENSUAL\n")
cat("Registros leídos:", format(nrow(era5_municipal_mensual), big.mark = ","), "\n")

# 4. VALIDACIÓN DE ESTRUCTURA -----------------------------------------------------
columnas_esperadas <- c("cod_mpio", "municipio", "fecha", "anio", "mes", "variable", "valor")

columnas_faltantes <- setdiff(
  columnas_esperadas,
  names(era5_municipal_mensual)
) # Detectar columnas ausentes

if (length(columnas_faltantes) > 0) {
  stop(
    paste(
      "Columnas faltantes en el panel:",
      paste(columnas_faltantes, collapse = ", ")
    )
  )
} # Validar estructura mínima

# 5. VALIDACIÓN DE DUPLICADOS -------------------------------------------------------
duplicados <- era5_municipal_mensual |>
  dplyr::count(
    cod_mpio,
    fecha,
    variable,
    name = "n"
  ) |>
  dplyr::filter(
    n > 1
  ) # Identificar combinaciones repetidas (municipio-fecha-variable)

# 6. COBERTURA TEMPORAL ----------------------------------------------------------
cobertura_temporal <- era5_municipal_mensual |>
  dplyr::distinct(
    anio,
    mes
  ) |>
  dplyr::arrange(
    anio,
    mes
  ) |>
  dplyr::mutate(
    indice_mes = dplyr::row_number()
  ) # Tabla de meses únicos presentes

anio_min <- min(era5_municipal_mensual$anio, na.rm = TRUE) # Año inicial
anio_max <- max(era5_municipal_mensual$anio, na.rm = TRUE) # Año final

meses_esperados  <- (anio_max - anio_min + 1) * 12 # 13 años x 12 meses = 156
meses_observados <- nrow(cobertura_temporal) # Meses realmente presentes

cobertura_temporal_completa <- (meses_observados == meses_esperados) # TRUE/FALSE

if (!cobertura_temporal_completa) {
  cat("\nADVERTENCIA: cobertura temporal incompleta\n")
  cat("Esperados:", meses_esperados, "| Observados:", meses_observados, "\n")
} # Alertar si faltan meses

# 7. COBERTURA ESPACIAL -----------------------------------------------------------
municipios_unicos <- dplyr::n_distinct(
  era5_municipal_mensual$cod_mpio
) # Número de municipios presentes en el panel

# 8. VARIABLES CLIMÁTICAS ----------------------------------------------------------
variables_presentes <- sort(
  unique(
    era5_municipal_mensual$variable
  )
) # Variables únicas detectadas

n_variables <- length(variables_presentes) # Conteo de variables

# 9. VALIDACIÓN DIMENSIONAL TEÓRICA ---------------------------------------------------
registros_esperados <- municipios_unicos * meses_observados * n_variables # Filas teóricas

consistencia_dimensional <- (
  nrow(era5_municipal_mensual) == registros_esperados
) # TRUE si coincide exactamente

if (!consistencia_dimensional) {
  cat("\nADVERTENCIA: inconsistencia dimensional\n")
  cat(
    "Esperados:", format(registros_esperados, big.mark = ","),
    "| Observados:", format(nrow(era5_municipal_mensual), big.mark = ","),
    "\n"
  )
} # Alertar si las dimensiones no calzan con municipios x meses x variables

# 10. AUDITORÍA POR VARIABLE -------------------------------------------------------
auditoria_por_variable <- era5_municipal_mensual |>
  dplyr::group_by(
    variable
  ) |>
  dplyr::summarise(
    registros        = dplyr::n(),
    registros_esperados = municipios_unicos * meses_observados,
    valores_na       = sum(is.na(valor)),
    porcentaje_na    = round(mean(is.na(valor)) * 100, 4),
    valor_min        = min(valor, na.rm = TRUE),
    valor_p25        = stats::quantile(valor, 0.25, na.rm = TRUE),
    valor_mediana    = stats::median(valor, na.rm = TRUE),
    valor_promedio   = mean(valor, na.rm = TRUE),
    valor_p75        = stats::quantile(valor, 0.75, na.rm = TRUE),
    valor_max        = max(valor, na.rm = TRUE),
    .groups = "drop"
  ) |>
  dplyr::arrange(
    variable
  ) # Resumen estadístico y de completitud por variable

# 11. AUDITORÍA POR MUNICIPIO -------------------------------------------------------
auditoria_por_municipio <- era5_municipal_mensual |>
  dplyr::group_by(
    cod_mpio,
    municipio
  ) |>
  dplyr::summarise(
    registros      = dplyr::n(),
    registros_esperados = meses_observados * n_variables,
    valores_na     = sum(is.na(valor)),
    porcentaje_na  = round(mean(is.na(valor)) * 100, 4),
    .groups = "drop"
  ) |>
  dplyr::arrange(
    dplyr::desc(porcentaje_na)
  ) # Ordenar de mayor a menor faltantes

municipios_con_na <- auditoria_por_municipio |>
  dplyr::filter(
    valores_na > 0
  ) # Municipios con al menos un valor faltante

# 12. AUDITORÍA GENERAL --------------------------------------------------------------
auditoria_general <- tibble::tibble(
  fecha_proceso              = Sys.time(),
  archivo_evaluado           = basename(archivo_entrada),
  registros                  = nrow(era5_municipal_mensual),
  registros_esperados        = registros_esperados,
  consistencia_dimensional   = consistencia_dimensional,
  municipios                 = municipios_unicos,
  variables                  = n_variables,
  meses_observados           = meses_observados,
  meses_esperados            = meses_esperados,
  cobertura_temporal_completa = cobertura_temporal_completa,
  anio_min                   = anio_min,
  anio_max                   = anio_max,
  valores_na                 = sum(is.na(era5_municipal_mensual$valor)),
  porcentaje_na              = round(mean(is.na(era5_municipal_mensual$valor)) * 100, 4),
  combinaciones_duplicadas   = nrow(duplicados),
  municipios_con_na          = nrow(municipios_con_na)
) # Tabla resumen general de auditoría

# 13. EXPORTAR RESULTADOS --------------------------------------------------------------
data.table::fwrite(
  auditoria_general,
  archivo_auditoria_general
) # Exportar auditoría general

data.table::fwrite(
  auditoria_por_variable,
  archivo_auditoria_variable
) # Exportar auditoría por variable

data.table::fwrite(
  auditoria_por_municipio,
  archivo_auditoria_municipio
) # Exportar auditoría por municipio

data.table::fwrite(
  cobertura_temporal,
  archivo_cobertura_temporal
) # Exportar cobertura temporal

data.table::fwrite(
  municipios_con_na,
  archivo_municipios_con_na
) # Exportar municipios con NA

# 14. RESUMEN FINAL EN CONSOLA ----------------------------------------------------------
cat("\nAUDITORÍA ERA5 MUNICIPAL-MENSUAL FINALIZADA\n")
cat("Registros:", format(auditoria_general$registros, big.mark = ","),
    "(esperados:", format(auditoria_general$registros_esperados, big.mark = ","), ")\n")
cat("Consistencia dimensional:", auditoria_general$consistencia_dimensional, "\n")
cat("Municipios:", auditoria_general$municipios, "\n")
cat("Variables:", auditoria_general$variables, "->", paste(variables_presentes, collapse = ", "), "\n")
cat("Meses:", auditoria_general$meses_observados, "/", auditoria_general$meses_esperados,
    "(", auditoria_general$anio_min, "-", auditoria_general$anio_max, ")\n")
cat("Cobertura temporal completa:", auditoria_general$cobertura_temporal_completa, "\n")
cat("Porcentaje NA global:", auditoria_general$porcentaje_na, "%\n")
cat("Combinaciones duplicadas (mpio-fecha-variable):", auditoria_general$combinaciones_duplicadas, "\n")
cat("Municipios con al menos un NA:", auditoria_general$municipios_con_na, "\n")
cat("\nArchivos generados en:", ruta_outputs_era5, "\n")
cat("Estado ERA5: AUDITORÍA COMPLETADA\n")
