# D:/Proyectos_IA/proyecto-gnn-agricola/chirps/01_clean_chirps.R

# chirps/01_clean_chirps.R
# 1. ¿Cómo cargar la configuración global del proyecto?
source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros globales

# 2. ¿Cuáles serán las rutas de entrada y salida?
ruta_chirps_raw <- file.path(ruta_raw, "chirps") # Carpeta rasters CHIRPS

ruta_processed_chirps <- file.path(ruta_processed, "chirps") # Carpeta procesados CHIRPS
ruta_outputs_chirps <- file.path(ruta_outputs, "chirps") # Carpeta auditorías CHIRPS

dir.create(ruta_processed_chirps, recursive = TRUE, showWarnings = FALSE) # Crear carpeta processed
dir.create(ruta_outputs_chirps, recursive = TRUE, showWarnings = FALSE) # Crear carpeta outputs

archivo_catalogo_chirps <- file.path(ruta_processed_chirps, "chirps_catalogo.csv") # Catálogo CHIRPS
archivo_validacion_chirps <- file.path(ruta_outputs_chirps, "auditoria_rasters_chirps.csv") # Auditoría raster
archivo_cobertura_chirps <- file.path(ruta_outputs_chirps, "cobertura_temporal_chirps.csv") # Cobertura temporal
archivo_consistencia_espacial <- file.path(ruta_outputs_chirps, "consistencia_espacial_chirps.csv") # Consistencia espacial
archivo_resumen_chirps <- file.path(ruta_outputs_chirps, "resumen_chirps.csv") # Resumen ejecutivo

# 3. ¿Existe la carpeta CHIRPS?
if (!dir.exists(ruta_chirps_raw)) {
  stop(paste("No existe la carpeta CHIRPS:", ruta_chirps_raw))
} # Validar carpeta

# 4. ¿Cuántos archivos CHIRPS fueron encontrados?
archivos_chirps <- list.files(
  ruta_chirps_raw,
  pattern = "^chirps-v3\\.0\\.[0-9]{4}\\.[0-9]{2}\\.tif$",
  full.names = TRUE
) # Buscar rasters

if (length(archivos_chirps) == 0) {
  stop("No se encontraron archivos CHIRPS")
} # Validar archivos

# 5. ¿Cuál es el catálogo oficial de rasters?
catalogo_chirps <- tibble::tibble(
  ruta = archivos_chirps,
  archivo = basename(archivos_chirps)
) |>
  dplyr::mutate(
    anio = as.integer(stringr::str_extract(archivo, "[0-9]{4}")),
    mes = as.integer(stringr::str_extract(archivo, "(?<=\\.)[0-9]{2}(?=\\.tif$)"))
  ) |>
  dplyr::arrange(anio, mes) # Construir catálogo

# 6. ¿Cuál es el periodo oficial del proyecto?
anio_inicio <- 2006 # Año inicial del proyecto
anio_fin <- 2018 # Año final del proyecto

catalogo_chirps <- catalogo_chirps |>
  dplyr::filter(
    anio >= anio_inicio,
    anio <= anio_fin
  ) # Filtrar periodo oficial

anio_min <- anio_inicio # Año inicial
anio_max <- anio_fin # Año final

archivos_esperados <- (anio_fin - anio_inicio + 1) * 12 # Archivos esperados

# 7. ¿La cantidad de rasters coincide con lo esperado?
if (nrow(catalogo_chirps) != archivos_esperados) {
  warning(
    paste(
      "Se esperaban",
      archivos_esperados,
      "rasters y se encontraron",
      nrow(catalogo_chirps)
    )
  )
} # Validar cobertura esperada

# 8. ¿La cobertura temporal está completa?
cobertura_temporal <- catalogo_chirps |>
  dplyr::count(anio, name = "meses_disponibles") |>
  dplyr::mutate(
    meses_esperados = 12,
    cobertura_completa = meses_disponibles == 12
  ) |>
  dplyr::arrange(anio) # Cobertura anual

# 9. ¿Todos los años poseen los 12 meses esperados?
anios_completos <- all(cobertura_temporal$cobertura_completa) # Validar años completos

# 10. ¿Cómo auditar cada raster CHIRPS?
auditar_raster_chirps <- function(ruta_raster, archivo, anio, mes) {
  
  raster_chirps <- terra::rast(ruta_raster) # Leer raster
  
  nodata_detectados <- as.numeric(
    terra::global(
      raster_chirps == -9999,
      "sum",
      na.rm = TRUE
    )[1,1]
  ) # Contar NoData
  
  terra::NAflag(raster_chirps) <- -9999 # Definir NoData
  
  raster_chirps[raster_chirps == -9999] <- NA # Convertir NoData a NA
  
  valores_negativos <- as.numeric(
    terra::global(
      raster_chirps < 0,
      "sum",
      na.rm = TRUE
    )[1,1]
  ) # Contar negativos reales
  
  resumen <- terra::global(
    raster_chirps,
    c("min", "max", "mean", "sd"),
    na.rm = TRUE
  ) # Resumen raster
  
  tibble::tibble(
    archivo = archivo,
    anio = anio,
    mes = mes,
    filas = nrow(raster_chirps),
    columnas = ncol(raster_chirps),
    capas = terra::nlyr(raster_chirps),
    capa_valida = terra::nlyr(raster_chirps) == 1,
    celdas = terra::ncell(raster_chirps),
    resolucion_x = terra::res(raster_chirps)[1],
    resolucion_y = terra::res(raster_chirps)[2],
    crs = terra::crs(raster_chirps),
    xmin = terra::ext(raster_chirps)[1],
    xmax = terra::ext(raster_chirps)[2],
    ymin = terra::ext(raster_chirps)[3],
    ymax = terra::ext(raster_chirps)[4],
    crs_vacio = terra::crs(raster_chirps) == "",
    extension_valida = terra::ext(raster_chirps)[1] < terra::ext(raster_chirps)[2] &
      terra::ext(raster_chirps)[3] < terra::ext(raster_chirps)[4],
    min = resumen[1,1],
    max = resumen[1,2],
    media = resumen[1,3],
    sd = resumen[1,4],
    nodata_detectados = nodata_detectados,
    valores_negativos = valores_negativos
  )
  
} # Auditar raster

# 11. ¿Todos los rasters cumplen los criterios mínimos?
validacion_rasters <- purrr::pmap_dfr(
  list(
    catalogo_chirps$ruta,
    catalogo_chirps$archivo,
    catalogo_chirps$anio,
    catalogo_chirps$mes
  ),
  auditar_raster_chirps
) # Ejecutar auditoría

# 12. ¿Existe consistencia espacial entre rasters?
resoluciones_detectadas <- validacion_rasters |>
  dplyr::distinct(resolucion_x, resolucion_y) # Resoluciones únicas

crs_detectados <- validacion_rasters |>
  dplyr::distinct(crs) # CRS únicos

extensiones_detectadas <- validacion_rasters |>
  dplyr::distinct(xmin, xmax, ymin, ymax) # Extensiones únicas

consistencia_espacial <- tibble::tibble(
  resoluciones_unicas = nrow(resoluciones_detectadas),
  crs_unicos = nrow(crs_detectados),
  extensiones_unicas = nrow(extensiones_detectadas)
) # Resumen consistencia espacial

# 13. ¿Puede certificarse la capa municipal?
archivo_municipios <- file.path(
  ruta_raw,
  "spatial",
  "Muni.shp"
) # Capa municipal

municipios <- sf::st_read(
  archivo_municipios,
  quiet = TRUE
) # Leer municipios

if (nrow(municipios) == 0) {
  stop("La capa municipal no contiene registros")
} # Validar municipios

# class(municipios)
# names(municipios)

total_municipios <- nrow(municipios) # Total municipios
municipios_con_codigo <- sum(!is.na(municipios$MunCodigo)) # Municipios con código
municipios_con_nombre <- sum(!is.na(municipios$MunNombre)) # Municipios con nombre
municipios_unicos <- dplyr::n_distinct(municipios$MunCodigo) # Códigos únicos
crs_municipios <- sf::st_crs(municipios)$epsg # EPSG municipios
crs_municipios_valido <- !is.na(crs_municipios) && crs_municipios == 4686 # Validar CRS municipios

municipios_duplicados <- municipios |>
  sf::st_drop_geometry() |>
  dplyr::count(MunCodigo) |>
  dplyr::filter(n > 1) # Buscar municipios duplicados

geometrias_invalidas <- sum(
  !sf::st_is_valid(municipios)
) # Contar geometrías inválidas

# 14. ¿Puede certificarse el conjunto CHIRPS?
class(municipios)

exists("total_municipios")
exists("municipios_unicos")
exists("geometrias_invalidas")
exists("municipios_duplicados")
exists("municipios_con_codigo")
exists("municipios_con_nombre")
exists("crs_municipios")
exists("crs_municipios_valido")

str(total_municipios)
str(municipios_unicos)
str(geometrias_invalidas)
str(municipios_duplicados)
str(municipios_con_codigo)
str(municipios_con_nombre)
str(crs_municipios)
str(crs_municipios_valido)

resumen_chirps <- tibble::tibble(
  fecha_proceso = Sys.time(),
  anio_min = anio_min,
  anio_max = anio_max,
  archivos_encontrados = nrow(catalogo_chirps),
  archivos_esperados = archivos_esperados,
  cobertura_completa = nrow(catalogo_chirps) == archivos_esperados,
  archivos_con_crs_vacio = sum(validacion_rasters$crs_vacio),
  archivos_con_extension_invalida = sum(!validacion_rasters$extension_valida),
  archivos_con_capas_invalidas = sum(!validacion_rasters$capa_valida),
  archivos_con_negativos = sum(validacion_rasters$valores_negativos > 0),
  archivos_con_nodata = sum(validacion_rasters$nodata_detectados > 0),
  municipios = total_municipios,
  municipios_unicos = municipios_unicos,
  geometrias_invalidas = geometrias_invalidas,
  municipios_duplicados = nrow(municipios_duplicados),
  municipios_con_codigo = municipios_con_codigo,
  municipios_con_nombre = municipios_con_nombre,
  crs_municipios = crs_municipios,
  crs_municipios_valido = crs_municipios_valido
) # Resumen ejecutivo

dataset_aprobado <- (
  resumen_chirps$cobertura_completa &
    anios_completos &
    resumen_chirps$archivos_con_crs_vacio == 0 &
    resumen_chirps$archivos_con_extension_invalida == 0 &
    resumen_chirps$archivos_con_capas_invalidas == 0 &
    resumen_chirps$archivos_con_negativos == 0 &
    nrow(resoluciones_detectadas) == 1 &
    nrow(crs_detectados) == 1 &
    nrow(extensiones_detectadas) == 1 &
    nrow(municipios_duplicados) == 0 &
    geometrias_invalidas == 0 &
    total_municipios == 1122 &
    crs_municipios_valido
) # Certificar conjunto CHIRPS

# 15. ¿El conjunto CHIRPS quedó certificado?
cat("\nPROCESO DE LIMPIEZA CHIRPS FINALIZADO\n") # Mostrar encabezado
cat("Periodo analizado:", anio_min, "-", anio_max, "\n") # Mostrar periodo
cat("Archivos encontrados:", nrow(catalogo_chirps), "\n") # Mostrar archivos encontrados
cat("Archivos esperados:", archivos_esperados, "\n") # Mostrar archivos esperados
cat("Cobertura temporal completa:", resumen_chirps$cobertura_completa, "\n") # Mostrar cobertura temporal
cat("Todos los años completos:", anios_completos, "\n") # Mostrar validación anual
cat("Archivos con CRS vacío:", resumen_chirps$archivos_con_crs_vacio, "\n") # Mostrar CRS vacíos
cat("Archivos con extensión inválida:", resumen_chirps$archivos_con_extension_invalida, "\n") # Mostrar extensiones inválidas
cat("Archivos con capas inválidas:", resumen_chirps$archivos_con_capas_invalidas, "\n") # Mostrar capas inválidas
cat("Archivos con NoData (-9999):", resumen_chirps$archivos_con_nodata, "\n") # Mostrar NoData
cat("Archivos con valores negativos:", resumen_chirps$archivos_con_negativos, "\n") # Mostrar negativos reales
cat("CRS únicos detectados:", nrow(crs_detectados), "\n") # Mostrar CRS únicos
cat("Resoluciones únicas detectadas:", nrow(resoluciones_detectadas), "\n") # Mostrar resoluciones únicas
cat("Extensiones únicas detectadas:", nrow(extensiones_detectadas), "\n") # Mostrar extensiones únicas
cat("Municipios:", total_municipios, "\n") # Mostrar municipios
cat("Municipios únicos:", municipios_unicos, "\n") # Mostrar municipios únicos
cat("Geometrías inválidas:", geometrias_invalidas, "\n") # Mostrar geometrías inválidas
cat("Municipios duplicados:", nrow(municipios_duplicados), "\n") # Mostrar duplicados
cat("CRS municipios:", crs_municipios, "\n") # Mostrar CRS municipios
cat("CRS municipios válido:", crs_municipios_valido, "\n") # Mostrar validación CRS municipios
cat("Dataset aprobado:", dataset_aprobado, "\n") # Mostrar certificación final

# 16. ¿Pueden exportarse los resultados?
readr::write_csv(catalogo_chirps, archivo_catalogo_chirps) # Exportar catálogo
readr::write_csv(validacion_rasters, archivo_validacion_chirps) # Exportar auditoría
readr::write_csv(cobertura_temporal, archivo_cobertura_chirps) # Exportar cobertura
readr::write_csv(consistencia_espacial, archivo_consistencia_espacial) # Exportar consistencia espacial
readr::write_csv(resumen_chirps, archivo_resumen_chirps) # Exportar resumen

