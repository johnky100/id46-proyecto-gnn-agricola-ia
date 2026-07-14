# 02_build_era5_municipal_mensual.R

# install.packages("progress")
library(progress)

source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros

# Parte 1 ------------------------------------------------
# 1. DEFINICIÓN DE RUTAS
ruta_era5_raw <- file.path(
  ruta_raw,
  "era5"
) # Carpeta ERA5

archivo_era5 <- file.path(
  ruta_era5_raw,
  "data_stream-mnth.nc"
) # NetCDF ERA5

archivo_municipios <- file.path(
  ruta_raw,
  "spatial",
  "Muni.shp"
) # Capa municipal

ruta_processed_era5 <- file.path(
  ruta_processed,
  "era5"
) # Carpeta procesados

archivo_catalogo <- file.path(
  ruta_processed_era5,
  "era5_catalogo.csv"
) # Catálogo temporal

archivo_salida <- file.path(
  ruta_processed_era5,
  "era5_municipal_mensual.csv"
) # Salida final

# 2. VALIDACIÓN DE INSUMOS

if (!file.exists(archivo_era5)) {
  stop(
    paste(
      "No existe archivo ERA5:",
      archivo_era5
    )
  )
} # Validar NetCDF

if (!file.exists(archivo_municipios)) {
  stop(
    paste(
      "No existe capa municipal:",
      archivo_municipios
    )
  )
} # Validar municipios

if (!file.exists(archivo_catalogo)) {
  stop(
    paste(
      "No existe catálogo:",
      archivo_catalogo
    )
  )
} # Validar catálogo

# 3. CARGA DEL CATÁLOGO
catalogo_era5 <- data.table::fread(
  archivo_catalogo
) # Leer catálogo

catalogo_era5 <- catalogo_era5 |>
  dplyr::filter(
    anio >= 2006,
    anio <= 2018
  ) |>
  dplyr::mutate(
    variable = stringr::str_replace(
      nombre_capa,
      "_[0-9]+$",
      ""
    )
  ) # Extraer variable

# 4. VALIDACIÓN TEMPORAL
meses_obtenidos <- dplyr::n_distinct(
  paste(
    catalogo_era5$anio,
    catalogo_era5$mes
  )
)

meses_esperados <- 13 * 12 # 2006-2018

if (meses_obtenidos != meses_esperados) {
  stop(
    paste(
      "Se esperaban",
      meses_esperados,
      "meses y se encontraron",
      meses_obtenidos
    )
  )
} # Validar periodo

# 5. CARGA DE MUNICIPIOS
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
  ) # Estandarizar código

if (
  any(
    !sf::st_is_valid(
      municipios
    )
  )
) {
  
  municipios <- sf::st_make_valid(
    municipios
  )
} # Corregir geometrías

# 6. CARGA DEL NETCDF ERA5

era5_stack <- terra::rast(
  archivo_era5
) # Leer NetCDF

capas_faltantes <- setdiff(
  catalogo_era5$nombre_capa,
  names(era5_stack)
)

if (
  length(capas_faltantes) > 0
) {
  stop(
    paste(
      "Capas faltantes:",
      length(capas_faltantes)
    )
  )
} # Validar capas

era5_stack <- era5_stack[[
  catalogo_era5$nombre_capa
]] # Mantener capas válidas

# 7. PREPARACIÓN ESPACIAL
municipios <- sf::st_transform(
  municipios,
  terra::crs(
    era5_stack
  )
) # Transformar CRS

municipios_vect <- terra::vect(
  municipios
) # Convertir a SpatVector

municipios_atributos <- municipios |>
  sf::st_drop_geometry() |>
  dplyr::mutate(
    id_zona = dplyr::row_number()
  ) |>
  dplyr::select(
    id_zona,
    cod_mpio,
    municipio
  ) # Atributos

municipios_vect$id_zona <- seq_len(
  nrow(
    municipios_vect
  )
) # Identificador

# 8. VARIABLES DISPONIBLES
variables_era5 <- sort(
  unique(
    catalogo_era5$variable
  )
)
cat("\nVARIABLES DETECTADAS\n")
print(variables_era5)

cat("\nMunicipios:", nrow(municipios), "\n")
cat("Variables:", length(variables_era5), "\n")
cat("Meses:", meses_obtenidos, "\n")
cat("Periodo:", min(catalogo_era5$anio), "-", max(catalogo_era5$anio), "\n")


# Parte 2 ------------------------------------------------

# Bloque 2.1: Función de media segura
# 9. FUNCIÓN DE MEDIA SEGURA
media_segura <- function(
    x,
    ...
) {
  x <- x[
    is.finite(x)
  ] # Mantener valores válidos
  if (
    length(x) == 0
  ) {
    return(
      NA_real_
    )
  } # Retornar NA si no hay datos
  mean(
    x,
    na.rm = TRUE
  ) # Calcular promedio
} # Media segura

# Bloque 2.2: Función de extracción
# 10. FUNCIÓN DE EXTRACCIÓN MUNICIPAL
extraer_variable <- function(
    variable_actual
) {
  cat("\n====================================================\n")
  cat("Procesando variable:", variable_actual, "\n")
  tiempo_inicio <- Sys.time() # Registrar inicio
  capas_variable <- grep(
    paste0(
      "^",
      variable_actual,
      "_"
    ),
    names(
      era5_stack
    ),
    value = TRUE
  ) # Buscar capas
  pb <- progress::progress_bar$new(
    format = paste0(
      variable_actual,
      " [:bar] :percent | Mes :current/:total | ETA: :eta"
    ),
    total = length(capas_variable),
    clear = FALSE,
    width = 80
  ) # Barra progreso
  lista_resultados <- vector(
    mode = "list",
    length = length(capas_variable)
  ) # Crear lista
  for (i in seq_along(capas_variable)) {
    pb$tick() # Actualizar barra
    capa_actual <- capas_variable[i]
    extraccion <- terra::extract(
      era5_stack[[capa_actual]],
      municipios_vect,
      fun = media_segura,
      na.rm = TRUE
    ) # Extraer una capa
    extraccion <- tibble::as_tibble(
      extraccion
    )
    names(
      extraccion
    )[2] <- "valor"
    extraccion$nombre_capa <- capa_actual
    lista_resultados[[i]] <- extraccion
  } # Fin ciclo capas
  extraccion_total <- dplyr::bind_rows(
    lista_resultados
  ) # Unir capas
  resultado <- extraccion_total |>
    dplyr::left_join(
      catalogo_era5 |>
        dplyr::select(
          nombre_capa,
          fecha,
          anio,
          mes
        ),
      by = "nombre_capa"
    ) |>
    
    dplyr::left_join(
      municipios_atributos,
      by = c(
        "ID" = "id_zona"
      )
    ) |>
    
    dplyr::mutate(
      variable = variable_actual
    ) |>
    
    dplyr::select(
      cod_mpio,
      municipio,
      fecha,
      anio,
      mes,
      variable,
      valor
    )
  
  cat(
    "\nRegistros generados:",
    format(
      nrow(resultado),
      big.mark = ","), "\n" )
  
  cat(
    "Tiempo (min):",
    round(
      as.numeric(
        difftime(
          Sys.time(),
          tiempo_inicio,
          units = "mins"
        )
      ),
      2
    ),
    "\n"
  )
  resultado
} # Función extracción

# Bloque 2.3: Prueba de certificación
# 11. PRUEBA DE CERTIFICACIÓN
prueba_tp <- extraer_variable(
  "tp"
)

dim(
  prueba_tp
)

summary(
  prueba_tp$valor
)

sum(is.na(prueba_tp$valor))

mean(is.na(prueba_tp$valor)) * 100

# Parte 3 ------------------------------------------------
# EXTRACCIÓN COMPLETA ERA5
cat("\nINICIANDO EXTRACCIÓN COMPLETA ERA5\n") # Informar inicio

tiempo_inicio <- Sys.time() # Registrar inicio

progressr::with_progress({
  progreso <- progressr::progressor(
    steps = length(
      variables_era5
    )
  ) # Definir pasos
  
  era5_municipal_mensual <- purrr::map_dfr(
    variables_era5,
    function(variable_actual) {
      progreso(
        paste(
          "Procesando",
          variable_actual
        )
      ) # Actualizar progreso
      
      extraer_variable(
        variable_actual
      )
    }
  )
}) # Ejecutar extracción

tiempo_fin <- Sys.time() # Registrar fin

# era5_municipal_mensual
saveRDS(
  era5_municipal_mensual,
  file = "era5_municipal_mensual.rds"
) # Guardar objeto comprimido

# Verificar que quedó guardado:

file.exists(
  "era5_municipal_mensual.rds"
) # Debe retornar TRUE

# Y si cierras R, lo recuperas con:

# era5_municipal_mensual <- readRDS(0
#  "era5_municipal_mensual.rds"
#) # Cargar objeto


# Resumen final ------------------------------------------------------------

tiempo_fin <- Sys.time() # Registrar fin

tiempo_total_horas <- round(
  as.numeric(
    difftime(
      tiempo_fin,
      tiempo_inicio,
      units = "hours"
    )
  ),
  2
) # Calcular duración total

cat("\nVALIDACIÓN DE DIMENSIONES\n")

cat(
  "Registros:",
  nrow(era5_municipal_mensual),
  "\n"
)

cat(
  "Municipios:",
  dplyr::n_distinct(
    era5_municipal_mensual$cod_mpio
  ),
  "\n"
)

cat(
  "Variables:",
  dplyr::n_distinct(
    era5_municipal_mensual$variable
  ),
  "\n"
)

cat(
  "Meses:",
  dplyr::n_distinct(
    paste(
      era5_municipal_mensual$anio,
      era5_municipal_mensual$mes
    )
  ),
  "\n"
)

cat(
  "Tiempo total (horas):",
  tiempo_total_horas,
  "\n"
)

# Guardar resultado definitivo ---------------------------------------------

dir.create(
  here::here(
    "datos",
    "procesados"
  ),
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta de salida

saveRDS(
  era5_municipal_mensual,
  file = here::here(
    "datos",
    "procesados",
    "era5_municipal_mensual.rds"
  )
) # Guardar objeto maestro

# versión que probablemente usarás más adelante porque 
# es mucho más rápida y compacta.

# Exportar parquet ---------------------------------------------------------

arrow::write_parquet(
  era5_municipal_mensual,
  sink = here::here(
    "datos",
    "procesados",
    "era5_municipal_mensual.parquet"
  )
) # Exportar parquet

# Exportar csv -------------------------------------------------------------

readr::write_csv(
  era5_municipal_mensual,
  file = here::here(
    "datos",
    "procesados",
    "era5_municipal_mensual.csv"
  )
) # Exportar csv

# Verificación de archivos creados

# Verificación -------------------------------------------------------------

archivos_generados <- c(
  here::here(
    "datos",
    "procesados",
    "era5_municipal_mensual.rds"
  ),
  here::here(
    "datos",
    "procesados",
    "era5_municipal_mensual.parquet"
  ),
  here::here(
    "datos",
    "procesados",
    "era5_municipal_mensual.csv"
  )
)

file.exists(
  archivos_generados
) # Verificar exportaciones

# guardaría un respaldo de la sesión ----------------
# ---------------------------------------------------
save.image(
  file = here::here(
    "backup_workspace_era5.RData"
  )
) # Backup completo
# ---------------------------------------------------

#####################################################
# ---------------------------------------------------
# Nota: Para la Parte 3 y posteriores, te recomiendo trabajar siempre desde:
# Porque abrirá mucho más rápido que el CSV y ocupará bastante menos espacio en disco.
# ---------------------------------------------------  
  era5_municipal_mensual <- arrow::read_parquet(
    here::here(
      "datos",
      "procesados",
      "era5_municipal_mensual.parquet"
    )
  ) # Cargar versión parquet
# ---------------------------------------------------
#####################################################

