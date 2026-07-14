# D:/Proyectos_IA/proyecto-gnn-agricola/chirps/04_select_chirps_variables.R

# 04_select_chirps_variables.R
# Ejecutar configuración global
source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros globales

# Pregunta: ¿Cuáles son las rutas de entrada y salida?
ruta_processed_chirps <- file.path(
  ruta_processed,
  "chirps"
) # Carpeta CHIRPS procesado

ruta_outputs_chirps <- file.path(
  ruta_outputs,
  "chirps"
) # Carpeta auditorías

archivo_chirps_anual <- file.path(
  ruta_processed_chirps,
  "chirps_municipal_anual.csv"
) # Panel anual

archivo_chirps_final <- file.path(
  ruta_processed_chirps,
  "chirps_variables_finales.csv"
) # Dataset final CHIRPS

archivo_resumen_seleccion <- file.path(
  ruta_outputs_chirps,
  "resumen_seleccion_chirps.csv"
) # Resumen metodológico

# Pregunta: ¿Existe el panel anual?
if (!file.exists(archivo_chirps_anual)) {
  stop(
    paste(
      "No existe archivo:",
      archivo_chirps_anual
    )
  )
} # Validar existencia

# Pregunta: ¿Puede cargarse el panel anual?
chirps_municipal_anual <- data.table::fread(
  archivo_chirps_anual
) # Leer panel anual

# Pregunta: ¿Cuántas variables climáticas existen inicialmente?
variables_originales <- setdiff(
  names(chirps_municipal_anual),
  c(
    "cod_mpio",
    "municipio",
    "anio"
  )
) # Identificar variables climáticas

# Pregunta: ¿Qué variables serán seleccionadas para integración EVA-GNN?
variables_finales <- c(
  "cod_mpio",
  "municipio",
  "anio",
  "precip_total_anual",
  "precip_sd_mensual",
  "precip_cv",
  "precip_min_mensual",
  "precip_max_mensual",
  "precip_q25",
  "precip_q75"
) # Variables seleccionadas

# Pregunta: ¿Puede construirse el dataset final?
chirps_variables_finales <- chirps_municipal_anual |>
  dplyr::select(
    dplyr::all_of(
      variables_finales
    )
  ) # Seleccionar variables finales

# Pregunta: ¿Qué variables fueron descartadas?
variables_descartadas <- setdiff(
  variables_originales,
  variables_finales
) # Identificar descartadas

# Pregunta: ¿Puede documentarse la selección realizada?
resumen_seleccion <- tibble::tribble(
  ~variable, ~decision, ~justificacion,
  "precip_total_anual", "Mantener", "Disponibilidad hídrica anual",
  "precip_sd_mensual", "Mantener", "Variabilidad intraanual",
  "precip_cv", "Mantener", "Variabilidad relativa",
  "precip_min_mensual", "Mantener", "Estrés hídrico anual",
  "precip_max_mensual", "Mantener", "Exceso hídrico anual",
  "precip_q25", "Mantener", "Condiciones relativamente secas",
  "precip_q75", "Mantener", "Condiciones relativamente húmedas",
  "precip_promedio_mensual", "Eliminar", "Redundante con precip_total_anual",
  "precip_q50", "Eliminar", "Información similar a tendencia central ya representada",
  "meses_validos", "Eliminar", "Variable de control de calidad"
) # Documentar selección

# Pregunta: ¿Pueden exportarse los productos finales?
data.table::fwrite(
  chirps_variables_finales,
  archivo_chirps_final
) # Exportar dataset final

data.table::fwrite(
  resumen_seleccion,
  archivo_resumen_seleccion
) # Exportar trazabilidad

# Pregunta: ¿Cuál fue el resultado final de la selección?
cat("\nSELECCIÓN DE VARIABLES CHIRPS FINALIZADA\n") # Mostrar encabezado

cat("\nDimensiones dataset de entrada\n") # Mostrar encabezado

cat("Filas:", nrow(chirps_municipal_anual), "\n") # Mostrar filas

cat("Columnas:", ncol(chirps_municipal_anual), "\n") # Mostrar columnas

cat( "\nCobertura\n") # Mostrar encabezado

cat(
  "Municipios:",
  dplyr::n_distinct(
    chirps_municipal_anual$cod_mpio
  ), "\n") # Mostrar municipios

cat(
  "Años:",
  dplyr::n_distinct(
    chirps_municipal_anual$anio
  ), "\n") # Mostrar años

cat(
  "Registros:",
  nrow(
    chirps_municipal_anual
  ), "\n") # Mostrar registros

cat("\nSelección de variables\n") # Mostrar encabezado

cat(
  "Variables climáticas originales:",
  length(
    variables_originales
  ),
  "\n"
) # Mostrar originales

cat(
  "Variables climáticas seleccionadas:",
  length(
    variables_finales
  ) - 3,
  "\n"
) # Mostrar seleccionadas

cat("Variables eliminadas:", length(variables_descartadas), "\n") # Mostrar eliminadas

cat("\nVariables seleccionadas:\n") # Mostrar encabezado

print(variables_finales[-c(1:3)]) # Mostrar variables finales

cat("\nVariables descartadas:\n") # Mostrar encabezado

print(variables_descartadas) # Mostrar descartadas

cat("\nArchivos generados\n") # Mostrar encabezado

cat("Dataset final:", archivo_chirps_final, "\n") # Mostrar dataset

cat("Resumen selección:", archivo_resumen_seleccion, "\n") # Mostrar resumen

cat("\nEstado dataset: APROBADO PARA INTEGRACIÓN GNN\n") # Confirmar estado
