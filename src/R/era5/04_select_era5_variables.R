# 04_select_era5_variables.R
# ---------------------------------------------------
# Selección final de variables ERA5 para integración con EVA / CHIRPS
# Formato largo (cod_mpio, municipio, fecha, anio, mes, variable, valor)
# Insumo: era5_municipal_mensual.parquet (auditado en 03_audit_era5_municipio_anio.R)
# Salidas: data/processed/era5/
# ---------------------------------------------------

source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros globales

# 1. DEFINICIÓN DE RUTAS -----------------------------------------------------
ruta_processed_era5 <- file.path(
  ruta_processed,
  "era5"
) # Carpeta de datos procesados ERA5

ruta_outputs_era5 <- file.path(
  ruta_outputs,
  "era5"
) # Carpeta de auditorías ERA5

dir.create(
  ruta_processed_era5,
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta de procesados si no existe

dir.create(
  ruta_outputs_era5,
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta de salida si no existe

archivo_entrada <- here::here(
  "datos",
  "procesados",
  "era5_municipal_mensual.parquet"
) # Panel municipal-mensual auditado (entrada del 02/03)

# Salidas mensuales -----------------------------------------------------------
archivo_salida_rds     <- file.path(ruta_processed_era5, "era5_municipal_mensual_final.rds")
archivo_salida_parquet <- file.path(ruta_processed_era5, "era5_municipal_mensual_final.parquet")
archivo_salida_csv     <- file.path(ruta_processed_era5, "era5_municipal_mensual_final.csv")
archivo_resumen        <- file.path(ruta_outputs_era5, "resumen_era5_final.csv")
archivo_municipios_na  <- file.path(ruta_outputs_era5, "municipios_con_na_era5_final.csv")

# Salidas anuales ---------------------------------------------------------------
archivo_salida_anual_rds     <- file.path(ruta_processed_era5, "era5_municipal_anual.rds")
archivo_salida_anual_parquet <- file.path(ruta_processed_era5, "era5_municipal_anual.parquet")
archivo_salida_anual_csv     <- file.path(ruta_processed_era5, "era5_municipal_anual.csv")
archivo_resumen_anual        <- file.path(ruta_outputs_era5, "resumen_era5_anual.csv")
archivo_municipios_na_anual  <- file.path(ruta_outputs_era5, "municipios_con_na_era5_anual.csv")

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

cat("\nINICIANDO SELECCIÓN FINAL DE VARIABLES ERA5\n")
cat("Registros de entrada:", format(nrow(era5_municipal_mensual), big.mark = ","), "\n")

# 4. DEFINICIÓN DE VARIABLES A CONSERVAR -----------------------------------------
# Por defecto se conservan las 11 variables ERA5-Land detectadas en la auditoría.
# Ajustar este vector si se desea un subconjunto específico para el modelo.
variables_seleccionadas <- c(
  "d2m",   # Temperatura de punto de rocío a 2 m
  "e",     # Evaporación
  "lai_hv",# Índice de área foliar - vegetación alta
  "pev",   # Evaporación potencial
  "ro",    # Escorrentía total
  "sro",   # Escorrentía superficial
  "strd",  # Radiación térmica descendente
  "t2m",   # Temperatura a 2 metros
  "tp",    # Precipitación total
  "u10",   # Viento zonal a 10 m
  "v10"    # Viento meridional a 10 m
)

variables_disponibles  <- sort(unique(era5_municipal_mensual$variable)) # Variables presentes en el panel
variables_no_encontradas <- setdiff(variables_seleccionadas, variables_disponibles)

if (length(variables_no_encontradas) > 0) {
  stop(
    paste(
      "Las siguientes variables no existen en el panel:",
      paste(variables_no_encontradas, collapse = ", ")
    )
  )
} # Validar que las variables solicitadas existan

# 5. NOTA SOBRE MUNICIPIOS INSULARES (NO SE EXCLUYEN) -----------------------------
# Hallazgo de la auditoría (03): San Andrés (88001) y Providencia (88564)
# presentan 100% de valores NA en las 11 variables ERA5-Land, por ausencia
# de celdas terrestres válidas sobre territorios insulares.
# Decisión: se CONSERVAN en el panel final con NA, para estudio e imputación
# posterior (no se eliminan registros ni municipios).
municipios_insulares_codigos <- c("88001", "88564")

# 6. FILTRADO FINAL (SOLO POR VARIABLE) -------------------------------------------
era5_final <- era5_municipal_mensual |>
  dplyr::filter(
    variable %in% variables_seleccionadas
  ) |>
  dplyr::select(
    cod_mpio,
    municipio,
    fecha,
    anio,
    mes,
    variable,
    valor
  ) |>
  dplyr::arrange(
    cod_mpio,
    fecha,
    variable
  ) # Formato largo, igual estructura que CHIRPS (sin excluir municipios)

# 7. VALIDACIÓN DIMENSIONAL ------------------------------------------------------
municipios_finales <- dplyr::n_distinct(era5_final$cod_mpio) # Municipios en la salida final
meses_finales      <- dplyr::n_distinct(paste(era5_final$anio, era5_final$mes)) # Meses únicos
variables_finales  <- dplyr::n_distinct(era5_final$variable) # Variables conservadas

registros_esperados <- municipios_finales * meses_finales * variables_finales # Filas teóricas

consistencia_dimensional <- (
  nrow(era5_final) == registros_esperados
) # TRUE si las dimensiones calzan exactamente

if (!consistencia_dimensional) {
  cat("\nADVERTENCIA: inconsistencia dimensional en la salida final\n")
  cat(
    "Esperados:", format(registros_esperados, big.mark = ","),
    "| Obtenidos:", format(nrow(era5_final), big.mark = ","),
    "\n"
  )
} # Alertar si las dimensiones no calzan

# 8. AUDITORÍA DE MUNICIPIOS CON NA (TRAZABILIDAD PARA IMPUTACIÓN) ----------------
municipios_con_na <- era5_final |>
  dplyr::group_by(
    cod_mpio,
    municipio
  ) |>
  dplyr::summarise(
    registros     = dplyr::n(),
    valores_na    = sum(is.na(valor)),
    porcentaje_na = round(mean(is.na(valor)) * 100, 4),
    .groups = "drop"
  ) |>
  dplyr::filter(
    valores_na > 0
  ) |>
  dplyr::arrange(
    dplyr::desc(porcentaje_na)
  ) # Listado de municipios pendientes de imputación

# 9. RESUMEN DE CALIDAD FINAL ----------------------------------------------------
resumen_final <- tibble::tibble(
  fecha_proceso            = Sys.time(),
  registros                = nrow(era5_final),
  registros_esperados      = registros_esperados,
  consistencia_dimensional = consistencia_dimensional,
  municipios               = municipios_finales,
  variables                = variables_finales,
  meses                    = meses_finales,
  anio_min                 = min(era5_final$anio),
  anio_max                 = max(era5_final$anio),
  valores_na               = sum(is.na(era5_final$valor)),
  porcentaje_na            = round(mean(is.na(era5_final$valor)) * 100, 4),
  municipios_con_na        = nrow(municipios_con_na)
) # Tabla resumen de la salida final

# 10. EXPORTAR RESULTADOS MENSUALES -------------------------------------------------
saveRDS(
  era5_final,
  file = archivo_salida_rds
) # Exportar RDS

arrow::write_parquet(
  era5_final,
  sink = archivo_salida_parquet
) # Exportar parquet

readr::write_csv(
  era5_final,
  file = archivo_salida_csv
) # Exportar csv

data.table::fwrite(
  resumen_final,
  archivo_resumen
) # Exportar resumen de calidad

data.table::fwrite(
  municipios_con_na,
  archivo_municipios_na
) # Exportar listado de municipios con NA pendientes de imputación

# 11. VERIFICACIÓN FINAL MENSUAL --------------------------------------------------------
archivos_generados <- c(
  archivo_salida_rds,
  archivo_salida_parquet,
  archivo_salida_csv
)

cat("\nSELECCIÓN FINAL ERA5 (MENSUAL) COMPLETADA\n")
cat("Variables conservadas:", paste(variables_seleccionadas, collapse = ", "), "\n")
cat("Municipios insulares conservados con NA:", paste(municipios_insulares_codigos, collapse = ", "), "\n")
cat("Registros finales:", format(resumen_final$registros, big.mark = ","),
    "(esperados:", format(resumen_final$registros_esperados, big.mark = ","), ")\n")
cat("Consistencia dimensional:", resumen_final$consistencia_dimensional, "\n")
cat("Municipios:", resumen_final$municipios, "\n")
cat("Variables:", resumen_final$variables, "\n")
cat("Meses:", resumen_final$meses, "(", resumen_final$anio_min, "-", resumen_final$anio_max, ")\n")
cat("Porcentaje NA:", resumen_final$porcentaje_na, "% | Municipios con NA:", resumen_final$municipios_con_na, "\n")
cat("Archivos generados:\n")
print(file.exists(archivos_generados))
cat("Ruta:", ruta_processed_era5, "\n")
cat("Estado ERA5: PANEL MENSUAL FINAL LISTO (NA pendientes de estudio e imputación)\n")

# ---------------------------------------------------
# 12. AGREGACIÓN ANUAL ----------------------------------------------------------
# Construye el panel municipal-anual a partir del panel mensual final,
# promediando cada variable climática por municipio y año.
# Mantiene los NA de los municipios insulares (San Andrés/Providencia)
# para su posterior estudio e imputación.
# ---------------------------------------------------

# 12.1 Agregación: promedio anual por municipio-variable -------------------------
era5_anual <- era5_final |>
  dplyr::group_by(
    cod_mpio,
    municipio,
    anio,
    variable
  ) |>
  dplyr::summarise(
    valor       = mean(valor, na.rm = FALSE), # NA se propaga si falta algún mes
    meses_obs   = sum(!is.na(valor)),
    .groups = "drop"
  ) |>
  dplyr::select(
    cod_mpio,
    municipio,
    anio,
    variable,
    valor,
    meses_obs
  ) |>
  dplyr::arrange(
    cod_mpio,
    anio,
    variable
  ) # Formato largo anual

# 12.2 Validación dimensional anual -----------------------------------------------
municipios_anual <- dplyr::n_distinct(era5_anual$cod_mpio)
anios_anual      <- dplyr::n_distinct(era5_anual$anio)
variables_anual  <- dplyr::n_distinct(era5_anual$variable)

registros_esperados_anual <- municipios_anual * anios_anual * variables_anual

consistencia_dimensional_anual <- (
  nrow(era5_anual) == registros_esperados_anual
)

if (!consistencia_dimensional_anual) {
  cat("\nADVERTENCIA: inconsistencia dimensional en el panel anual\n")
  cat(
    "Esperados:", format(registros_esperados_anual, big.mark = ","),
    "| Obtenidos:", format(nrow(era5_anual), big.mark = ","),
    "\n"
  )
} # Alertar si las dimensiones no calzan

# 12.3 Municipios con NA en el panel anual (trazabilidad para imputación) ---------
municipios_con_na_anual <- era5_anual |>
  dplyr::group_by(
    cod_mpio,
    municipio
  ) |>
  dplyr::summarise(
    registros     = dplyr::n(),
    valores_na    = sum(is.na(valor)),
    porcentaje_na = round(mean(is.na(valor)) * 100, 4),
    .groups = "drop"
  ) |>
  dplyr::filter(
    valores_na > 0
  ) |>
  dplyr::arrange(
    dplyr::desc(porcentaje_na)
  )

# 12.4 Resumen de calidad del panel anual ------------------------------------------
resumen_anual <- tibble::tibble(
  fecha_proceso            = Sys.time(),
  registros                = nrow(era5_anual),
  registros_esperados      = registros_esperados_anual,
  consistencia_dimensional = consistencia_dimensional_anual,
  municipios               = municipios_anual,
  variables                = variables_anual,
  anios                    = anios_anual,
  anio_min                 = min(era5_anual$anio),
  anio_max                 = max(era5_anual$anio),
  valores_na               = sum(is.na(era5_anual$valor)),
  porcentaje_na            = round(mean(is.na(era5_anual$valor)) * 100, 4),
  municipios_con_na        = nrow(municipios_con_na_anual)
)

# 12.5 Exportar panel anual -----------------------------------------------------------
saveRDS(
  era5_anual,
  file = archivo_salida_anual_rds
)

arrow::write_parquet(
  era5_anual,
  sink = archivo_salida_anual_parquet
)

readr::write_csv(
  era5_anual,
  file = archivo_salida_anual_csv
)

data.table::fwrite(
  resumen_anual,
  archivo_resumen_anual
)

data.table::fwrite(
  municipios_con_na_anual,
  archivo_municipios_na_anual
)

# 12.6 Verificación final del panel anual --------------------------------------------
archivos_generados_anual <- c(
  archivo_salida_anual_rds,
  archivo_salida_anual_parquet,
  archivo_salida_anual_csv
)

cat("\nAGREGACIÓN ANUAL ERA5 COMPLETADA\n")
cat("Registros anuales:", format(resumen_anual$registros, big.mark = ","),
    "(esperados:", format(resumen_anual$registros_esperados, big.mark = ","), ")\n")
cat("Consistencia dimensional:", resumen_anual$consistencia_dimensional, "\n")
cat("Municipios:", resumen_anual$municipios, "\n")
cat("Variables:", resumen_anual$variables, "\n")
cat("Años:", resumen_anual$anios, "(", resumen_anual$anio_min, "-", resumen_anual$anio_max, ")\n")
cat("Porcentaje NA:", resumen_anual$porcentaje_na, "% | Municipios con NA:", resumen_anual$municipios_con_na, "\n")
cat("Archivos generados:\n")
print(file.exists(archivos_generados_anual))
cat("Ruta:", ruta_processed_era5, "\n")
cat("Estado ERA5: PANEL ANUAL LISTO (NA pendientes de estudio e imputación)\n")
