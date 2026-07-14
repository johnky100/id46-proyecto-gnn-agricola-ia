# config/01_paths.R

# Configuración centralizada de rutas del proyecto
# ------------------- Rutas principales ------------------

rutas <- list(
  root = here::here(),
  data = file.path(here::here(), "data"),
  raw = file.path(here::here(), "data", "raw"),
  processed = file.path(here::here(), "data", "processed"),
  outputs = file.path(here::here(), "outputs"),
  reportes = file.path(here::here(), "REPORTES")
) # Definir rutas principales

# ------------------- Rutas EVA ------------------

rutas$raw_eva <- file.path(
  rutas$raw,
  "eva"
) # Carpeta EVA original

rutas$archivo_eva <- file.path(
  rutas$raw_eva,
  "Evaluaciones_Agropecuarias_Municipales_EVA_20260413.csv"
) # Archivo oficial EVA

rutas$processed_eva <- file.path(
  rutas$processed,
  "eva"
) # Carpeta EVA procesada

rutas$outputs_eva <- file.path(
  rutas$outputs,
  "eva"
) # Carpeta resultados EVA

# ------------------- Rutas Panel Maestro GNN ------------------

rutas$processed_panel_maestro_gnn <- file.path(
  rutas$processed,
  "panel_maestro_gnn"
) # Carpeta principal panel maestro gnn

# ------------------- Carpetas requeridas ------------------

rutas_crear <- c(
  rutas$data,
  rutas$raw,
  rutas$processed,
  rutas$outputs,
  rutas$reportes,
  rutas$processed_eva,
  rutas$outputs_eva,
  rutas$processed_panel_maestro_gnn
) # Estructura mínima del proyecto

# ------------------- Creación de carpetas ------------------

purrr::walk(
  rutas_crear,
  ~ dir.create(.x, recursive = TRUE, showWarnings = FALSE)
) # Crear carpetas necesarias

# ------------------- Validación de carpetas ------------------

rutas_faltantes <- rutas_crear[
  !dir.exists(rutas_crear)
] # Detectar rutas faltantes

if (length(rutas_faltantes) > 0) {
  
  stop(
    paste(
      "No fue posible crear las siguientes rutas:",
      paste(rutas_faltantes, collapse = ", ")
    )
  ) # Detener ejecución si existen errores
  
}

rutas$db_ganador <- here(
  "data",
  "processed",
  "db_ganador"
) # Carpeta del dataset ganador
# ------------------- Validación archivo EVA ------------------

if (!file.exists(rutas$archivo_eva)) {
  
  stop(
    paste(
      "No se encontró el archivo EVA:",
      rutas$archivo_eva
    )
  ) # Validar existencia del archivo EVA
  
}

# ------------------- Resumen configuración ------------------

cat(
  "\nRUTAS CONFIGURADAS CORRECTAMENTE\n"
) # Confirmar configuración

cat(
  "Total rutas configuradas:",
  length(rutas_crear),
  "\n"
) # Mostrar total de rutas

cat(
  "Archivo EVA validado correctamente\n"
) # Confirmar archivo EVA

cat(
  "Ruta EVA:",
  rutas$archivo_eva,
  "\n"
) # Mostrar ruta EVA

cat(
  "Ruta raíz proyecto:",
  rutas$root,
  "\n"
) # Mostrar ruta principal

cat(
  "Ruta panel maestro GNN:",
  rutas$processed_panel_maestro_gnn,
  "\n"
) # Mostrar ruta panel maestro gnn