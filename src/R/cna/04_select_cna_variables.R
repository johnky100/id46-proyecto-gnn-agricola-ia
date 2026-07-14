# D:/Proyectos_IA/proyecto-gnn-agricola/cna/04_select_cna_variables.R

source(here::here("config", "00_packages.R"))          # Cargar paquetes
source(here::here("config", "01_paths.R"))             # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros

# PARTE 1: DEFINICIÓN DE RUTAS Y VALIDACIÓN DE INSUMOS
# ======================================================================

ruta_processed_cna <- file.path(ruta_processed, "cna") # Carpeta CNA procesados
ruta_outputs_cna   <- file.path(ruta_outputs, "cna")   # Carpeta auditorías CNA

archivo_cna_limpio <- file.path(
  ruta_processed_cna,
  "cna_features_limpio.rds"
) # Producto aprobado del Script 03

dir.create(ruta_outputs_cna, recursive = TRUE, showWarnings = FALSE) # Crear carpeta auditorías

if (!file.exists(archivo_cna_limpio)) {
  stop(
    paste(
      "No existe:",
      basename(archivo_cna_limpio)
    )
  )
} # Validar entrada

# PARTE 2: CARGA DEL CNA AUDITADO
# ======================================================================

cna <- readRDS(archivo_cna_limpio) # Leer CNA auditado

cat("\nCNA AUDITADO CARGADO\n")
cat("Registros: ", nrow(cna), "\n")
cat("Variables: ", ncol(cna), "\n")

# PARTE 3: VALIDACIÓN DE UNICIDAD
# ======================================================================

duplicados <- cna |>
  dplyr::count(cod_mpio) |>
  dplyr::filter(n > 1) # Buscar duplicados

if (nrow(duplicados) > 0) {
  stop(
    paste(
      "Existen",
      nrow(duplicados),
      "municipios duplicados"
    )
  )
} # Detener si hay duplicados

if (any(is.na(cna$cod_mpio))) {
  stop("Existen cod_mpio faltantes")
} # Validar llave

# PARTE 4: VARIABLES A CONSERVAR
# ======================================================================

cli::cli_h2("Selección de variables CNA")

variables_conservar <- c(
  
  # Llaves territoriales
  "cod_mpio",
  "municipio",
  "departamento",
  
  # Cobertura y uso del suelo
  "area_total",
  "pct_agro",
  "pct_natural",
  "pct_no_agro",
  "pct_otros_usos",
  
  # Tenencia de la tierra
  "pct_propia",
  "pct_arrendada",
  "pct_colectiva",
  "pct_mixta"
  
) # Variables finales CNA

faltantes <- setdiff(
  variables_conservar,
  names(cna)
) # Detectar variables ausentes

if (length(faltantes) > 0) {
  stop(
    paste(
      "Variables faltantes:",
      paste(faltantes, collapse = ", ")
    )
  )
} # Validar estructura

# PARTE 5: CONSTRUCCIÓN DEL CNA FINAL
# ======================================================================

cna_modelado <- cna |>
  dplyr::select(
    dplyr::all_of(variables_conservar)
  ) # Conservar únicamente variables seleccionadas

# PARTE 6: AUDITORÍA DE REDUCCIÓN
# ======================================================================

variables_eliminadas <- setdiff(
  names(cna),
  names(cna_modelado)
) # Variables removidas

auditoria_reduccion <- tibble::tibble(
  variable_eliminada = variables_eliminadas
) # Registrar reducción

auditoria_reduccion <- tibble::tibble(
  variable_eliminada = variables_eliminadas,
  motivo = dplyr::case_when(
    variable_eliminada %in% c(
      "densidad_upna",
      "pct_upna"
    ) ~ "Redundancia perfecta con pct_no_agro",
    
    TRUE ~ "Variable no seleccionada para modelado"
  )
); auditoria_reduccion # Registrar motivo de eliminación

cat("\nRESUMEN DE REDUCCIÓN\n")
cat("\nVARIABLES FINALES CNA\n")
print(names(cna_modelado))
cat("Variables originales: ", ncol(cna), "\n")
cat("Variables finales:    ", ncol(cna_modelado), "\n")
cat("Variables eliminadas: ", length(variables_eliminadas), "\n")

# PARTE 7: EXPORTACIÓN
# ======================================================================

cli::cli_h2("Exportando CNA para integración EVA")

archivo_rds <- file.path(
  ruta_processed_cna,
  "cna_modelado.rds"
) # Producto final CNA

archivo_parquet <- file.path(
  ruta_processed_cna,
  "cna_modelado.parquet"
)

archivo_csv <- file.path(
  ruta_processed_cna,
  "cna_modelado.csv"
)

archivo_auditoria <- file.path(
  ruta_outputs_cna,
  "audit_cna_reduccion_variables.csv"
)

saveRDS(
  cna_modelado,
  archivo_rds
) # Exportar RDS

arrow::write_parquet(
  cna_modelado,
  sink = archivo_parquet
) # Exportar Parquet

readr::write_csv(
  cna_modelado,
  archivo_csv
) # Exportar CSV

data.table::fwrite(
  auditoria_reduccion,
  archivo_auditoria
) # Exportar auditoría

# PARTE 8: VALIDACIÓN DE EXPORTACIÓN
# ======================================================================

archivos_generados <- c(
  archivo_rds,
  archivo_parquet,
  archivo_csv,
  archivo_auditoria
) # Archivos esperados

if (!all(file.exists(archivos_generados))) {
  faltantes_export <- archivos_generados[
    !file.exists(archivos_generados)
  ]
  
  stop(
    paste(
      "Archivos no generados:",
      paste(
        basename(faltantes_export),
        collapse = ", "
      )
    )
  )
} # Verificar exportaciones

# PARTE 9: RESUMEN FINAL
# ======================================================================

cat("\nSELECCIÓN DE VARIABLES CNA COMPLETADA\n")
cat("Municipios:             ", nrow(cna_modelado), "\n")
cat("Variables finales:      ", ncol(cna_modelado), "\n")
cat("Variables eliminadas:   ", length(variables_eliminadas), "\n")
cat("Archivo final:          ", basename(archivo_rds), "\n")
cat("Producto: cna_modelado\n")
cat("Estado: APROBADO PARA INTEGRACIÓN CON EVA\n")
