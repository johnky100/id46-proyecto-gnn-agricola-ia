# D:/Proyectos_IA/proyecto-gnn-agricola/cna/03_audit_cna_municipio_anio.R

source(here::here("config", "00_packages.R"))          # Cargar paquetes
source(here::here("config", "01_paths.R"))              # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros

# PARTE 1: CARGA Y VALIDACIÓN DE ENTRADA
# ===========================================================================

# 1. DEFINICIÓN DE RUTAS ----------------------------------------------------

ruta_processed_cna <- file.path(ruta_processed, "cna") # Carpeta CNA procesados
ruta_outputs_cna   <- file.path(ruta_outputs,   "cna") # Carpeta auditorías CNA

archivo_features <- file.path(ruta_processed_cna, "cna_features.rds") # Features CNA

dir.create(ruta_outputs_cna, recursive = TRUE, showWarnings = FALSE) # Crear carpeta auditorías

# 2. VALIDACIÓN DE INSUMOS --------------------------------------------------

if (!file.exists(archivo_features)) stop(paste("No existe cna_features.rds:", archivo_features)) # Validar entrada

# 3. CARGA DE FEATURES ------------------------------------------------------

cna_features <- readRDS(archivo_features) # Leer features CNA

cat("\nFEATURES CNA CARGADOS\n")
cat("Registros totales: ", nrow(cna_features), "\n")
cat("Variables:         ", ncol(cna_features), "\n")
cat("Municipios válidos:", dplyr::n_distinct(cna_features$cod_mpio, na.rm = TRUE), "\n")
cat("Registros sin cod: ", sum(is.na(cna_features$cod_mpio)), "\n")

# PARTE 2: AUDITORÍA DESCRIPTIVA DE LAS 12 VARIABLES DERIVADAS
# (Hallazgo 2 del reporte técnico)
# ===========================================================================

cli::cli_h2("Auditoría descriptiva — variables derivadas CNA")

variables_derivadas <- c(
  "area_total",
  "pct_agro", "pct_natural", "pct_no_agro", "pct_otros_usos",
  "pct_propia", "pct_arrendada", "pct_colectiva", "pct_mixta",
  "densidad_upa", "densidad_upna",
  "pct_upna"
) # 12 variables derivadas obligatorias

faltantes_vars <- setdiff(
  variables_derivadas,
  names(cna_features)
) # Variables esperadas que no existen

if (length(faltantes_vars) > 0) {
  stop(
    paste(
      "Variables derivadas faltantes:",
      paste(faltantes_vars, collapse = ", ")
    )
  )
} # Detener si faltan variables obligatorias

# Función segura para estadísticos — protege variables completamente NA ------
# Problema 1: min/max sobre all-NA produce Inf/-Inf con warning
stat_seguro <- function(x, fn, ...) {
  if (all(is.na(x))) return(NA_real_) # Proteger variable totalmente vacía
  fn(x, ...)
} # Función segura para estadísticos

# Calcular estadísticas descriptivas por variable ---------------------------
auditoria_descriptiva <- purrr::map_dfr(
  variables_derivadas,
  function(v) {
    x <- cna_features[[v]]
    tibble::tibble(
      variable    = v,
      n_total     = length(x),                                                       # Total registros
      n_validos   = sum(!is.na(x)),                                                  # Registros con valor
      n_missing   = sum(is.na(x)),                                                   # Valores faltantes
      pct_missing = round(mean(is.na(x)) * 100, 2),                                 # % faltante
      media       = round(stat_seguro(x, mean,             na.rm = TRUE), 4),       # Media
      mediana     = round(stat_seguro(x, stats::median,    na.rm = TRUE), 4),       # Mediana
      sd          = round(stat_seguro(x, stats::sd,        na.rm = TRUE), 4),       # Desviación estándar
      minimo      = round(stat_seguro(x, min,              na.rm = TRUE), 4),       # Mínimo
      p25 = round(as.numeric(stat_seguro(x, stats::quantile, probs = 0.25,na.rm = TRUE)),4), # Percentil 25
      p75 = round(as.numeric(stat_seguro(x, stats::quantile, probs = 0.75, na.rm = TRUE)), 4), # Percentil 75
      maximo = round(stat_seguro(x, max, na.rm = TRUE), 4)                          # Máximo
    )
  }
) # Tabla descriptiva completa

cat("\nESTADÍSTICAS DESCRIPTIVAS — VARIABLES DERIVADAS CNA\n")
print(auditoria_descriptiva, n = Inf, width = Inf)

# PARTE 3: VALIDACIÓN DE SUMA DE PROPORCIONES DE USO DEL SUELO ≈ 1
# (Hallazgo 3 del reporte técnico)
# ===========================================================================

cli::cli_h2("Validación: pct_agro + pct_natural + pct_no_agro + pct_otros_usos ≈ 1")

tolerancia <- 0.01 # Tolerancia metodológica ±0.01

cna_features <- cna_features |>
  dplyr::mutate(
    suma_pct_uso = pct_agro + pct_natural + pct_no_agro + pct_otros_usos, # Suma de proporciones (NA si cualquiera es NA)
    suma_valida  = dplyr::case_when(
      is.na(suma_pct_uso)                 ~ "sin_dato",     # Al menos una proporción es NA
      abs(suma_pct_uso - 1) <= tolerancia ~ "ok",           # Suma ≈ 1
      TRUE                                ~ "fuera_rango"   # Suma ≠ 1
    )
  ) # Clasificar cada registro

# Resumen de la validación --------------------------------------------------
resumen_suma <- cna_features |>
  dplyr::count(suma_valida) |>
  dplyr::mutate(pct = round(n / sum(n) * 100, 2)) # Contar por categoría

cat("\nRESUMEN VALIDACIÓN SUMA DE PROPORCIONES\n")
print(resumen_suma)

# Registros fuera de rango --------------------------------------------------
fuera_rango <- cna_features |>
  dplyr::filter(suma_valida == "fuera_rango") |>
  dplyr::select(cod_mpio, municipio, departamento, pct_agro, pct_natural, pct_no_agro, pct_otros_usos, suma_pct_uso)

if (nrow(fuera_rango) > 0) {
  warning(paste(nrow(fuera_rango), "municipio(s) con suma de proporciones fuera del rango [0.99, 1.01]"))
  cat("\nMUNICIPIOS FUERA DE RANGO\n")
  print(tibble::as_tibble(fuera_rango), n = Inf) # Forzar tibble para garantizar n = Inf
} else {
  cli::cli_alert_success("Todos los municipios con datos válidos suman proporciones ≈ 1")
} # Reportar resultado

# Problema 2: exportar municipios sin_dato para revisión --------------------
municipios_sin_dato_suma <- cna_features |>
  dplyr::filter(suma_valida == "sin_dato") |>
  dplyr::select(cod_mpio, municipio, departamento, pct_agro, pct_natural, pct_no_agro, pct_otros_usos, suma_pct_uso)

cat("\nMUNICIPIOS SIN DATO EN SUMA DE PROPORCIONES:", nrow(municipios_sin_dato_suma), "\n")
if (nrow(municipios_sin_dato_suma) > 0) print(tibble::as_tibble(municipios_sin_dato_suma), n = Inf) # Mostrar casos

# Auditoría de suma de proporciones -----------------------------------------
auditoria_suma_uso <- tibble::tibble(
  total_registros     = nrow(cna_features),                                                          # Total registros
  suma_ok             = sum(cna_features$suma_valida == "ok",          na.rm = TRUE),                # Correctos
  suma_fuera_rango    = sum(cna_features$suma_valida == "fuera_rango", na.rm = TRUE),                # Fuera de rango
  suma_sin_dato       = sum(cna_features$suma_valida == "sin_dato",    na.rm = TRUE),                # Sin dato
  pct_ok              = round(sum(cna_features$suma_valida == "ok") / nrow(cna_features) * 100, 2), # % correctos
  tolerancia_aplicada = tolerancia                                                                    # Tolerancia usada
) # Tabla resumen suma

# PARTE 4: HALLAZGO 1 — SEPARACIÓN DEL REGISTRO HUÉRFANO
# Se conservan los 1122 en la auditoría; se genera versión limpia con 1121
# ===========================================================================

cli::cli_h2("Gestión del registro sin código DANE")

registro_huerfano <- cna_features |>
  dplyr::filter(is.na(cod_mpio)) # Registro sin COD_MUN

cat("\nREGISTRO HUÉRFANO DETECTADO\n")
if (nrow(registro_huerfano) > 0) {
  print(tibble::as_tibble(registro_huerfano |> dplyr::select(where(~ !all(is.na(.)))))) # Mostrar columnas no vacías
} else {
  cat("Ningún registro huérfano encontrado\n")
} # Mostrar registro

cna_features_limpio <- cna_features |>
  dplyr::filter(!is.na(cod_mpio)) # 1121 municipios válidos

cat("\nVERSIÓN LIMPIA PARA INTEGRACIÓN\n")
cat("Registros con código DANE: ", nrow(cna_features_limpio), "\n")
cat("Registros huérfanos:       ", nrow(registro_huerfano), "(guardados en auditoría)\n")

# PARTE 5: VALIDACIÓN DE DUPLICADOS EN cod_mpio — CRÍTICA PARA JOIN
# (Validación faltante identificada en revisión técnica)
# ===========================================================================

cli::cli_h2("Validación de unicidad de cod_mpio — requisito para left_join con EVA y ERA5")

duplicados_cod <- cna_features_limpio |>
  dplyr::count(cod_mpio) |>
  dplyr::filter(n > 1) # Buscar municipios con más de un registro

if (nrow(duplicados_cod) > 0) {
  cat("\nDUPLICADOS DETECTADOS EN cod_mpio:\n")
  print(tibble::as_tibble(duplicados_cod), n = Inf)
  stop(paste(
    nrow(duplicados_cod),
    "municipio(s) con registros duplicados — el left_join con EVA/ERA5 generará filas duplicadas.",
    "Resolver antes de continuar."
  )) # Detener: duplicados rompen el ensamblaje
} else {
  cli::cli_alert_success("cod_mpio único en todos los registros — join con EVA y ERA5 es seguro")
} # Confirmar unicidad

# PARTE 6: AUDITORÍA DE COBERTURA TERRITORIAL
# ===========================================================================

cli::cli_h2("Cobertura territorial — CNA vs referencia nacional")

cobertura_departamental <- cna_features_limpio |>
  dplyr::group_by(departamento) |>
  dplyr::summarise(
    n_municipios    = dplyr::n(),                                 # Municipios por dpto
    pct_agro_media  = round(mean(pct_agro,  na.rm = TRUE), 3),   # Media pct agro
    area_total_suma = round(sum(area_total, na.rm = TRUE), 0),   # Área total dpto
    upa_suma        = round(sum(UPA,        na.rm = TRUE), 0),   # UPAs totales dpto
    .groups = "drop"
  ) |>
  dplyr::arrange(dplyr::desc(n_municipios)) # Ordenar por cantidad de municipios

cat("\nCOBERTURA POR DEPARTAMENTO (top 10)\n")
print(head(cobertura_departamental, 10))

# PARTE 7: AUDITORÍA DE MISSING VALUES EN VARIABLES DERIVADAS
# ===========================================================================

cli::cli_h2("Mapa de valores faltantes — variables derivadas")

mapa_missing <- cna_features_limpio |>
  dplyr::summarise(
    dplyr::across(
      dplyr::all_of(variables_derivadas),
      list(
        n_na   = ~ sum(is.na(.)),
        pct_na = ~ round(mean(is.na(.)) * 100, 2)
      ),
      .names = "{.col}__{.fn}"
    )
  ) |>
  tidyr::pivot_longer(
    cols      = dplyr::everything(),
    names_to  = c("variable", "stat"),
    names_sep = "__"
  ) |>
  tidyr::pivot_wider(names_from = stat, values_from = value) |>
  dplyr::arrange(dplyr::desc(pct_na)) # Ordenar por % de NA descendente

cat("\nMISSING VALUES POR VARIABLE DERIVADA (ordenado por % NA)\n")
print(mapa_missing, n = Inf)

# PARTE 8: EXPORTACIÓN DE AUDITORÍAS
# ===========================================================================

cli::cli_h2("Exportando auditorías CNA")

archivo_audit_descriptiva   <- file.path(ruta_outputs_cna, "audit_cna_descriptiva.csv")       # Estadísticas descriptivas
archivo_audit_suma_uso      <- file.path(ruta_outputs_cna, "audit_cna_suma_uso.csv")           # Validación suma proporciones
archivo_audit_sin_dato_suma <- file.path(ruta_outputs_cna, "audit_cna_sin_dato_suma.csv")      # Municipios sin dato en suma
archivo_audit_missing       <- file.path(ruta_outputs_cna, "audit_cna_missing.csv")            # Mapa de NAs
archivo_audit_cobertura     <- file.path(ruta_outputs_cna, "audit_cna_cobertura_dpto.csv")     # Cobertura departamental
archivo_huerfano            <- file.path(ruta_outputs_cna, "cna_registro_huerfano.csv")        # Registro sin código
archivo_limpio_rds          <- file.path(ruta_processed_cna, "cna_features_limpio.rds")        # RDS limpio
archivo_limpio_parquet      <- file.path(ruta_processed_cna, "cna_features_limpio.parquet")    # Parquet limpio
archivo_limpio_csv          <- file.path(ruta_processed_cna, "cna_features_limpio.csv")        # CSV limpio

data.table::fwrite(auditoria_descriptiva,    archivo_audit_descriptiva)   # Exportar descriptiva
data.table::fwrite(auditoria_suma_uso,       archivo_audit_suma_uso)      # Exportar suma uso
data.table::fwrite(municipios_sin_dato_suma, archivo_audit_sin_dato_suma) # Exportar sin_dato suma (Problema 2)
data.table::fwrite(mapa_missing,             archivo_audit_missing)       # Exportar missing
data.table::fwrite(cobertura_departamental,  archivo_audit_cobertura)     # Exportar cobertura
data.table::fwrite(registro_huerfano,        archivo_huerfano)            # Exportar huérfano

saveRDS(cna_features_limpio,              archivo_limpio_rds)             # Exportar RDS limpio
arrow::write_parquet(cna_features_limpio, sink = archivo_limpio_parquet)  # Exportar Parquet limpio
readr::write_csv(cna_features_limpio,     archivo_limpio_csv)             # Exportar CSV limpio

# Verificación de exportaciones ---------------------------------------------
archivos_generados <- c(
  archivo_audit_descriptiva, archivo_audit_suma_uso, archivo_audit_sin_dato_suma,
  archivo_audit_missing, archivo_audit_cobertura, archivo_huerfano,
  archivo_limpio_rds, archivo_limpio_parquet, archivo_limpio_csv
) # Archivos esperados

if (!all(file.exists(archivos_generados))) {
  faltantes_export <- archivos_generados[!file.exists(archivos_generados)]
  stop(paste("Archivos no generados:", paste(basename(faltantes_export), collapse = ", ")))
} # Detener si falta algún archivo

# PARTE 9: RESUMEN FINAL
# ===========================================================================

cat("\nAUDITORÍA CNA COMPLETADA\n")
cat("Registros auditados (total):   ", nrow(cna_features), "\n")
cat("Municipios válidos (con cod):  ", nrow(cna_features_limpio), "\n")
cat("Registro huérfano (sin cod):   ", nrow(registro_huerfano), "\n")
cat("Variables derivadas auditadas: ", length(variables_derivadas), "\n")
cat("Municipios suma uso ≈ 1:       ", auditoria_suma_uso$suma_ok, "\n")
cat("Municipios suma fuera rango:   ", auditoria_suma_uso$suma_fuera_rango, "\n")
cat("Municipios sin dato en suma:   ", auditoria_suma_uso$suma_sin_dato, "\n")
cat("Duplicados en cod_mpio:        ", nrow(duplicados_cod), "\n")
cat("Departamentos cubiertos:       ", dplyr::n_distinct(cna_features_limpio$departamento), "\n")
cat("Producto para integración:     cna_features_limpio (1121 municipios)\n")
cat("Registro huérfano preservado:  cna_registro_huerfano.csv\n")
cat("Estado: APROBADO EN AUDITORÍA CNA\n")
cat("Pendiente: validación de integración EVA y ERA5\n")
