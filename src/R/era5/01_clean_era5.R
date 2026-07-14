# D:/Proyectos_IA/proyecto-gnn-agricola/era5/01_clean_era5.R
# 01_clean_era5.R

source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros globales

# Pregunta: ¿Cuáles son las rutas de entrada y salida?
ruta_era5_raw <- file.path(
  ruta_raw,
  "era5"
) # Carpeta ERA5 raw

ruta_processed_era5 <- file.path(
  ruta_processed,
  "era5"
) # Carpeta ERA5 procesado

ruta_outputs_era5 <- file.path(
  ruta_outputs,
  "era5"
) # Carpeta auditorías ERA5

dir.create(
  ruta_processed_era5,
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta procesados

dir.create(
  ruta_outputs_era5,
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta auditorías

archivo_catalogo <- file.path(
  ruta_processed_era5,
  "era5_catalogo.csv"
) # Catálogo temporal ERA5

archivo_auditoria <- file.path(
  ruta_outputs_era5,
  "auditoria_era5.csv"
) # Auditoría general

archivo_cobertura <- file.path(
  ruta_outputs_era5,
  "cobertura_temporal_era5.csv"
) # Cobertura temporal

archivo_variables <- file.path(
  ruta_outputs_era5,
  "variables_detectadas_era5.csv"
) # Variables detectadas

# Pregunta: ¿Existe el archivo NetCDF?
archivos_era5 <- list.files(
  ruta_era5_raw,
  pattern = "\\.nc$",
  full.names = TRUE
) # Buscar archivos NetCDF

if (length(archivos_era5) == 0) {
  stop(
    paste(
      "No se encontró archivo NetCDF en:",
      ruta_era5_raw
    )
  )
} # Validar existencia

ruta_nc <- archivos_era5[1] # Seleccionar archivo principal

# Pregunta: ¿Puede leerse correctamente el NetCDF?
era5_stack <- terra::rast(
  ruta_nc
) # Leer NetCDF

# Pregunta: ¿Puede recuperarse la dimensión temporal?
fechas_era5 <- as.Date(
  terra::time(
    era5_stack
  )
) # Extraer fechas

if (
  length(fechas_era5) != nlyr(era5_stack)
) {
  stop(
    "La dimensión temporal no coincide con el número de capas"
  )
} # Validar tiempo

# Pregunta: ¿Cuáles son los nombres originales de las capas?
capas_originales <- names(
  era5_stack
) # Extraer nombres

# Pregunta: ¿Qué variables climáticas contiene ERA5?
variables_detectadas <- stringr::str_extract(
  capas_originales,
  "^[A-Za-z0-9_]+"
) |> unique() |> sort() # Detectar variables

variables_detectadas <- tibble::tibble(
  variable = variables_detectadas
) # Construir tabla variables

# Pregunta: ¿Puede construirse el catálogo temporal?
era5_catalogo <- tibble::tibble(
  capa = seq_len(
    nlyr(era5_stack)
  ),
  nombre_capa = capas_originales,
  fecha = fechas_era5,
  anio = lubridate::year(
    fechas_era5
  ),
  mes = lubridate::month(
    fechas_era5
  )
) |> dplyr::arrange(
  fecha
) # Construir catálogo

# Pregunta: ¿Cuál es la cobertura temporal disponible?
cobertura_temporal <- era5_catalogo |>
  dplyr::count(
    anio,
    mes,
    name = "capas"
  ) |>
  dplyr::arrange(
    anio,
    mes
  ) # Construir cobertura

# Pregunta: ¿Existen fechas duplicadas?
fechas_duplicadas <- era5_catalogo |>
  dplyr::count(
    fecha
  ) |>
  dplyr::filter(
    n > 1
  ) # Detectar duplicados

# Pregunta: ¿Puede construirse la auditoría general?
auditoria_era5 <- tibble::tibble(
  fecha_proceso = Sys.time(),
  
  archivo = basename(
    ruta_nc
  ),
  
  capas =
    nlyr(
      era5_stack
    ),
  
  filas =
    nrow(
      era5_stack
    ),
  
  columnas =
    ncol(
      era5_stack
    ),
  
  celdas =
    terra::ncell(
      era5_stack
    ),
  
  resolucion_x =
    terra::res(
      era5_stack
    )[1],
  
  resolucion_y =
    terra::res(
      era5_stack
    )[2],
  
  anio_min =
    min(
      era5_catalogo$anio,
      na.rm = TRUE
    ),
  
  anio_max =
    max(
      era5_catalogo$anio,
      na.rm = TRUE
    ),
  
  meses_unicos =
    dplyr::n_distinct(
      era5_catalogo$fecha
    ),
  
  variables =
    nrow(
      variables_detectadas
    ),
  
  fechas_duplicadas =
    nrow(
      fechas_duplicadas
    )
) # Construir auditoría

# Pregunta: ¿Pueden exportarse los resultados?
data.table::fwrite(
  era5_catalogo,
  archivo_catalogo
) # Exportar catálogo

data.table::fwrite(
  auditoria_era5,
  archivo_auditoria
) # Exportar auditoría

data.table::fwrite(
  cobertura_temporal,
  archivo_cobertura
) # Exportar cobertura

data.table::fwrite(
  variables_detectadas,
  archivo_variables
) # Exportar variables

# Pregunta: ¿Finalizó correctamente la certificación ERA5?
cat("\nCATÁLOGO ERA5 GENERADO CORRECTAMENTE\n") # Mostrar encabezado

cat(
  "Archivo:",
  basename(ruta_nc),
  "\n"
) # Mostrar archivo

cat(
  "Capas:",
  nlyr(era5_stack),
  "\n"
) # Mostrar capas

cat(
  "Cobertura:",
  min(era5_catalogo$anio),
  "-",
  max(era5_catalogo$anio),
  "\n"
) # Mostrar cobertura

cat(
  "Variables detectadas:",
  nrow(variables_detectadas),
  "\n"
) # Mostrar variables

cat(
  "Fechas duplicadas:",
  nrow(fechas_duplicadas),
  "\n"
) # Mostrar duplicados

cat(
  "Catálogo:",
  archivo_catalogo,
  "\n"
) # Mostrar salida

cat(
  "Estado ERA5: APROBADO PARA EXTRACCIÓN MUNICIPAL\n"
) # Confirmar estado
