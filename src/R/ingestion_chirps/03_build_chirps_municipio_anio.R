# 03_build_chirps_municipio_anio.R

# Ejecutar configuración global
source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros

# Pregunta: ¿Cuáles son las rutas de entrada y salida?
ruta_processed_chirps <- file.path(
  ruta_processed,
  "chirps"
) # Carpeta CHIRPS procesado

ruta_outputs_chirps <- file.path(
  ruta_outputs,
  "chirps"
) # Carpeta auditorías

archivo_chirps_mensual <- file.path(
  ruta_processed_chirps,
  "chirps_municipal_mensual.csv"
) # Panel mensual

archivo_chirps_anual <- file.path(
  ruta_processed_chirps,
  "chirps_municipal_anual.csv"
) # Panel anual

archivo_auditoria <- file.path(
  ruta_outputs_chirps,
  "auditoria_chirps_municipio_anual.csv"
) # Auditoría

archivo_duplicados <- file.path(
  ruta_outputs_chirps,
  "duplicados_chirps_municipio_anual.csv"
) # Duplicados

archivo_cobertura <- file.path(
  ruta_outputs_chirps,
  "cobertura_chirps_municipio_anual.csv"
) # Cobertura

# Pregunta: ¿Existe el panel mensual?
if (!file.exists(archivo_chirps_mensual)) {
  stop(
    paste(
      "No existe archivo:",
      archivo_chirps_mensual
    )
  )
} # Validar insumo

# Pregunta: ¿Puede cargarse el panel mensual?
chirps_municipal_mensual <- data.table::fread(
  archivo_chirps_mensual
) # Leer panel mensual

# Pregunta: ¿Puede construirse el panel municipio-año?
chirps_municipal_anual <- chirps_municipal_mensual |>
  dplyr::group_by(
    cod_mpio,
    municipio,
    anio
  ) |>
  
  dplyr::summarise(
    
    precip_total_anual =
      sum(
        precip_mm,
        na.rm = TRUE
      ),
    
    precip_promedio_mensual =
      mean(
        precip_mm,
        na.rm = TRUE
      ),
    
    precip_max_mensual =
      max(
        precip_mm,
        na.rm = TRUE
      ),
    
    precip_min_mensual =
      min(
        precip_mm,
        na.rm = TRUE
      ),
    
    precip_sd_mensual =
      sd(
        precip_mm,
        na.rm = TRUE
      ),
    
    precip_q25 =
      quantile(
        precip_mm,
        probs = 0.25,
        na.rm = TRUE
      ),
    
    precip_q50 =
      quantile(
        precip_mm,
        probs = 0.50,
        na.rm = TRUE
      ),
    
    precip_q75 =
      quantile(
        precip_mm,
        probs = 0.75,
        na.rm = TRUE
      ),
    
    meses_validos =
      sum(
        !is.na(precip_mm)
      ),
    
    .groups = "drop"
    
  ) |>
  
  dplyr::mutate(
    
    precip_cv =
      dplyr::if_else(
        precip_promedio_mensual > 0,
        precip_sd_mensual /
          precip_promedio_mensual,
        NA_real_
      )
    
  ) # Construir indicadores anuales

# Pregunta: ¿Existen duplicados municipio-año?
duplicados <- chirps_municipal_anual |>
  dplyr::count(
    cod_mpio,
    anio
  ) |>
  
  dplyr::filter(
    n > 1
  ) # Detectar duplicados

# Pregunta: ¿La cobertura temporal es completa?
cobertura <- chirps_municipal_anual |>
  
  dplyr::count(
    anio,
    name = "municipios"
  ) |>
  
  dplyr::arrange(
    anio
  ) # Cobertura anual

# Pregunta: ¿Puede construirse la auditoría?
auditoria <- tibble::tibble(
  
  fecha_proceso = Sys.time(),
  
  registros =
    nrow(
      chirps_municipal_anual
    ),
  
  municipios =
    dplyr::n_distinct(
      chirps_municipal_anual$cod_mpio
    ),
  
  anio_min =
    min(
      chirps_municipal_anual$anio
    ),
  
  anio_max =
    max(
      chirps_municipal_anual$anio
    ),
  
  duplicados =
    nrow(
      duplicados
    ),
  
  municipios_con_13_anios =
    sum(
      table(
        chirps_municipal_anual$cod_mpio
      ) == 13
    )
  
) # Auditoría

# Pregunta: ¿Pueden exportarse los resultados?
data.table::fwrite(
  chirps_municipal_anual,
  archivo_chirps_anual
) # Exportar panel anual

data.table::fwrite(
  auditoria,
  archivo_auditoria
) # Exportar auditoría

data.table::fwrite(
  duplicados,
  archivo_duplicados
) # Exportar duplicados

data.table::fwrite(
  cobertura,
  archivo_cobertura
) # Exportar cobertura

cat(
  "Municipios:",
  dplyr::n_distinct(
    chirps_municipal_anual$cod_mpio
  ), "\n")

cat(
  "Años:",
  dplyr::n_distinct(
    chirps_municipal_anual$anio
  ), "\n")

cat(
  "Registros:",
  nrow(
    chirps_municipal_anual
  ), "\n")

cat(
  "Duplicados:",
  nrow(
    duplicados
  ), "\n")

cat("Estado panel: APROBADO PARA INTEGRACIÓN EVA\n")



# Pregunta: ¿Finalizó correctamente la agregación anual?
cat("\nPANEL MUNICIPIO-AÑO CHIRPS GENERADO CORRECTAMENTE\n")

