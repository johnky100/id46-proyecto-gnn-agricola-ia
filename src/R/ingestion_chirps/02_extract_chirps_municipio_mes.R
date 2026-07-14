# D:/Proyectos_IA/proyecto-gnn-agricola/chirps/02_build_chirps_municipio_anio.R

# 02_build_chirps_municipio_anio.R
# Ejecutar configuración global
source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros globales

# Activar barra de progreso
progressr::handlers(
  progressr::handler_txtprogressbar()
) # Activar barra de progreso

# Pregunta: ¿Cuáles son las rutas de entrada y salida?
ruta_chirps_raw <- file.path(
  ruta_raw,
  "chirps"
) # Carpeta rasters CHIRPS

archivo_municipios <- file.path(
  ruta_raw,
  "spatial",
  "Muni.shp"
) # Capa municipal

ruta_processed_chirps <- file.path(
  ruta_processed,
  "chirps"
) # Carpeta procesados CHIRPS

ruta_outputs_chirps <- file.path(
  ruta_outputs,
  "chirps"
) # Carpeta auditorías CHIRPS

dir.create(
  ruta_processed_chirps,
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta procesados

dir.create(
  ruta_outputs_chirps,
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta auditorías

archivo_chirps_mensual <- file.path(
  ruta_processed_chirps,
  "chirps_municipal_mensual.csv"
) # Panel municipio-mes

archivo_auditoria_extraccion <- file.path(
  ruta_outputs_chirps,
  "auditoria_extraccion_chirps.csv"
) # Auditoría extracción

# Pregunta: ¿Existen los insumos requeridos?
if (!dir.exists(ruta_chirps_raw)) {
  stop(
    paste(
      "No existe carpeta CHIRPS:",
      ruta_chirps_raw
    )
  )
} # Validar rasters

if (!file.exists(archivo_municipios)) {
  stop(
    paste(
      "No existe capa municipal:",
      archivo_municipios
    )
  )
} # Validar municipios

# Pregunta: ¿Puede cargarse la capa municipal certificada?
municipios <- sf::st_read(
  archivo_municipios,
  quiet = TRUE
) # Leer municipios

municipios <- municipios |>
  dplyr::select(
    MunCodigo,
    MunNombre
  ) |>
  dplyr::rename(
    cod_mpio = MunCodigo,
    municipio = MunNombre
  ) |>
  dplyr::mutate(
    cod_mpio = stringr::str_pad(
      as.character(cod_mpio),
      width = 5,
      side = "left",
      pad = "0"
    )
  ) # Estandarizar municipios

# Pregunta: ¿Cuáles son los rasters oficiales del proyecto?
archivos_chirps <- list.files(
  ruta_chirps_raw,
  pattern = "^chirps-v3\\.0\\.[0-9]{4}\\.[0-9]{2}\\.tif$",
  full.names = TRUE
) # Buscar rasters

catalogo_chirps <- tibble::tibble(
  ruta = archivos_chirps,
  archivo = basename(archivos_chirps)
) |>
  dplyr::mutate(
    
    anio = as.integer(
      stringr::str_extract(
        archivo,
        "[0-9]{4}"
      )
    ),
    
    mes = as.integer(
      stringr::str_extract(
        archivo,
        "(?<=\\.)[0-9]{2}(?=\\.tif$)"
      )
    )
  ) |>
  dplyr::filter(
    anio >= 2006,
    anio <= 2018
  ) |>
  
  dplyr::arrange(
    anio,
    mes
  ) # Catálogo oficial

# Pregunta: ¿Puede extraerse la precipitación media municipal?
extraer_chirps_municipal <- function(
    ruta_raster,
    archivo,
    anio,
    mes
) {
  
  raster_chirps <- terra::rast(
    ruta_raster
  ) # Leer raster
  
  terra::NAflag(
    raster_chirps
  ) <- -9999 # Definir NoData
  
  raster_chirps[
    raster_chirps == -9999
  ] <- NA # Convertir NoData
  
  extraccion <- terra::extract(
    raster_chirps,
    terra::vect(
      municipios
    ),
    fun = mean,
    na.rm = TRUE
  ) # Extraer precipitación media
  
  tibble::tibble(
    cod_mpio = municipios$cod_mpio,
    municipio = municipios$municipio,
    anio = anio,
    mes = mes,
    precip_mm = extraccion[, 2]
  )
} # Función extracción municipal

# Pregunta: ¿Puede ejecutarse la extracción para todos los meses?
cat(
  "\nIniciando extracción municipal CHIRPS...\n"
) # Informar inicio

tiempo_inicio_extraccion <- Sys.time() # Registrar inicio

progressr::with_progress({
  
  progreso <- progressr::progressor(
    steps = nrow(catalogo_chirps)
  ) # Definir total de pasos
  
  chirps_municipal_mensual <- purrr::pmap_dfr(
    
    list(
      catalogo_chirps$ruta,
      catalogo_chirps$archivo,
      catalogo_chirps$anio,
      catalogo_chirps$mes
    ),
    
    function(
    ruta_raster,
    archivo,
    anio,
    mes
    ) {
      
      progreso(
        paste0(
          "Procesando ",
          anio,
          "-",
          sprintf("%02d", mes)
        )
      ) # Actualizar progreso
      
      extraer_chirps_municipal(
        ruta_raster = ruta_raster,
        archivo = archivo,
        anio = anio,
        mes = mes
      )
    }
  )
}) # Ejecutar extracción con barra de progreso

tiempo_fin_extraccion <- Sys.time() # Registrar fin

cat(
  "\nExtracción municipal finalizada\n"
) # Informar finalización

cat(
  "Tiempo total (minutos):",
  round(
    as.numeric(
      difftime(
        tiempo_fin_extraccion,
        tiempo_inicio_extraccion,
        units = "mins"
      )
    ),
    2
  ),
  "\n"
) # Mostrar tiempo total

# Pregunta: ¿Existen registros duplicados municipio-año-mes?
duplicados_municipio_mes <- chirps_municipal_mensual |>
  dplyr::count(
    cod_mpio,
    anio,
    mes
  ) |>
  dplyr::filter(
    n > 1
  ) # Buscar duplicados

# Pregunta: ¿Cuál es la cobertura temporal por año?
cobertura_anual <- chirps_municipal_mensual |>
  dplyr::count(
    anio,
    mes
  ) |>
  dplyr::count(
    anio,
    name = "meses_disponibles"
  ) |>
  dplyr::mutate(
    meses_esperados = 12,
    cobertura_completa =
      meses_disponibles == 12
  ) # Validar cobertura anual

# Pregunta: ¿Existen municipios sin precipitación?
municipios_sin_datos <- chirps_municipal_mensual |>
  dplyr::group_by(
    cod_mpio
  ) |>
  dplyr::summarise(
    registros_validos =
      sum(
        !is.na(precip_mm)
      ),
    .groups = "drop"
  ) |>
  dplyr::filter(
    registros_validos == 0
  ) # Municipios sin datos

# Pregunta: ¿Existen precipitaciones negativas?
precipitaciones_negativas <- chirps_municipal_mensual |>
  dplyr::filter(
    precip_mm < 0
  ) # Detectar negativos

# Pregunta: ¿Puede construirse la auditoría de extracción?
auditoria_extraccion <- tibble::tibble(
  fecha_proceso = Sys.time(),
  registros = nrow(
    chirps_municipal_mensual
  ),
  
  municipios = dplyr::n_distinct(
    chirps_municipal_mensual$cod_mpio
  ),
  
  anio_min = min(
    chirps_municipal_mensual$anio,
    na.rm = TRUE
  ),
  
  anio_max = max(
    chirps_municipal_mensual$anio,
    na.rm = TRUE
  ),
  
  meses_unicos = dplyr::n_distinct(
    paste(
      chirps_municipal_mensual$anio,
      chirps_municipal_mensual$mes
    )
  ),
  
  duplicados = nrow(
    duplicados_municipio_mes
  ),
  
  municipios_sin_datos = nrow(
    municipios_sin_datos
  ),
  
  precipitaciones_negativas = nrow(
    precipitaciones_negativas
  )
) # Construir auditoría

# Pregunta: ¿Puede exportarse el panel mensual?
data.table::fwrite(
  chirps_municipal_mensual,
  archivo_chirps_mensual
) # Exportar panel mensual

data.table::fwrite(
  auditoria_extraccion,
  archivo_auditoria_extraccion
) # Exportar auditoría

# Pregunta: ¿Finalizó correctamente la construcción del panel mensual?
cat("\nPANEL MUNICIPIO-MES CHIRPS GENERADO CORRECTAMENTE\n") # Mostrar encabezado

# Pregunta: ¿Cuál fue la cobertura temporal procesada?
cat("Cobertura temporal:", min(chirps_municipal_mensual$anio), "-", max(chirps_municipal_mensual$anio), "\n") # Mostrar cobertura

# Pregunta: ¿Cuántos municipios fueron procesados?
cat("Municipios:", dplyr::n_distinct(chirps_municipal_mensual$cod_mpio), "\n") # Mostrar municipios

# Pregunta: ¿Cuántos registros fueron generados?
cat("Registros municipio-mes:", nrow(chirps_municipal_mensual), "\n") # Mostrar registros

# Pregunta: ¿Cuántos meses fueron procesados?
cat("Meses únicos:",
  dplyr::n_distinct(paste(chirps_municipal_mensual$anio, 
    chirps_municipal_mensual$mes)), "\n") # Mostrar meses

# Pregunta: ¿Existen duplicados municipio-año-mes?
cat("Duplicados municipio-año-mes:", nrow(duplicados_municipio_mes), "\n") # Mostrar duplicados

# Pregunta: ¿Existen municipios sin información climática?
cat("Municipios sin datos:", nrow(municipios_sin_datos), "\n") # Mostrar municipios sin datos

# Pregunta: ¿Existen precipitaciones negativas?
cat("Precipitaciones negativas:", nrow(precipitaciones_negativas), "\n") # Mostrar negativos

# Pregunta: ¿Dónde fueron almacenados los productos?
cat("Archivo generado:", archivo_chirps_mensual, "\n") # Mostrar salida
cat("Auditoría generada:", archivo_auditoria_extraccion, "\n") # Mostrar auditoría

# Pregunta: ¿El panel mensual está listo para agregación anual?
cat("Estado panel: APROBADO PARA AGREGACIÓN MUNICIPIO-AÑO\n") # Confirmar estado
