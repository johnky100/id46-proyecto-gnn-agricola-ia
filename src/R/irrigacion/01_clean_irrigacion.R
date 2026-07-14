# 01_clean_irrigacion.R
# Limpieza y estandarización capa de irrigación
# ============================================================

# -------------- Bloque 1 ---------------------
# 1. Configuración y rutas
# ============================================================
source(here::here("config", "00_packages.R"))
source(here::here("config", "01_paths.R"))
source(here::here("config", "02_global_parameters.R"))

# Rutas específicas irrigación
# ============================================================
ruta_raw_irrigacion <- file.path(
  ruta_raw,
  "irrigacion",
  "Areas_potenciales_para_adecuación_de_tierras_con_fines_de_irrigacion_20260413.csv"
) # Archivo fuente

ruta_processed_irrigacion <- file.path(
  ruta_processed,
  "irrigacion",
  "irrigacion_final.parquet"
) # Dataset atributos

ruta_processed_irrigacion_gpkg <- file.path(
  ruta_processed,
  "irrigacion",
  "irrigacion_final.gpkg"
) # Dataset espacial

dir.create(
  dirname(ruta_processed_irrigacion),
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta de salida

# Validaciones previas
# ============================================================
if (!file.exists(ruta_raw_irrigacion)) {
  stop(
    paste(
      "No existe el archivo:",
      ruta_raw_irrigacion
    )
  ) # Detener ejecución
}
# ------------------- Fin Bloque 1 ------------------

# ------------------- Bloque 2 ------------------
# 2. Ingesta
# ============================================================
irrigacion_raw <- readr::read_csv(
  ruta_raw_irrigacion,
  show_col_types = FALSE,
  locale = readr::locale(
    encoding = "UTF-8"
  )
) # Leer archivo fuente

filas_originales <- nrow(
  irrigacion_raw
) # Número registros

columnas_originales <- ncol(
  irrigacion_raw
) # Número columnas

# Corrección nombres de columnas
# ============================================================
names(irrigacion_raw) <- names(irrigacion_raw) |>
  stringr::str_replace_all(
    c(
      "Necesidad Recurso HÃ­drico" = "Necesidad Recurso Hídrico",
      "RegulaciÃ³n" = "Regulación",
      "SocioeconÃ³mico" = "Socioeconómico",
      "Ãrea \\(ha\\)" = "Área (ha)"
    )
  ) # Corregir encoding

# Validación estructura
# ============================================================
columnas_esperadas <- c(
  "The geom",
  "Fisico",
  "Tipo tierras",
  "Necesidad Recurso Hídrico",
  "Disponibilidad",
  "Regulación",
  "Ecosistemico",
  "Socioeconómico",
  "Área (ha)",
  "Potencial",
  "Consecutivo"
)

columnas_faltantes <- setdiff(
  columnas_esperadas,
  names(irrigacion_raw)
) # Columnas faltantes

columnas_extras <- setdiff(
  names(irrigacion_raw),
  columnas_esperadas
) # Columnas adicionales

if (length(columnas_faltantes) > 0) {
  stop(
    paste(
      "Columnas faltantes:",
      paste(
        columnas_faltantes,
        collapse = ", "
      )
    )
  )
}

if (length(columnas_extras) > 0) {
  warning(
    paste(
      "Columnas adicionales detectadas:",
      paste(
        columnas_extras,
        collapse = ", "
      )
    )
  )
}

# Resumen ingesta
# ============================================================
cat("\nIRRIGACIÓN - INGESTA\n")
cat("Filas originales:", filas_originales, "\n")
cat("Columnas detectadas:", ncol(irrigacion_raw), "\n")
cat("Columnas faltantes:", length(columnas_faltantes), "\n")
cat("Columnas adicionales:", length(columnas_extras), "\n")
# ------------------- Fin Bloque 2 ------------------

# ------------------- Bloque 3 ------------------
# 3. Estandarización de variables
# ============================================================
irrigacion_clean <- irrigacion_raw |>
  dplyr::transmute(
    geometry = `The geom`,
    consecutivo_fuente = Consecutivo,
    
    id_poligono = paste0(
      "IRR_",
      dplyr::row_number()
    ),
    
    fisico = as.numeric(
      Fisico
    ),
    
    tipo_tierra = as.character(
      `Tipo tierras`
    ),
    
    necesidad_hidrica = as.character(
      `Necesidad Recurso Hídrico`
    ),
    
    disponibilidad = as.character(
      Disponibilidad
    ),
    
    regulacion = as.character(
      Regulación
    ),
    
    ecosistemico = as.character(
      Ecosistemico
    ),
    
    socioeconomico = as.character(
      Socioeconómico
    ),
    
    potencial = as.character(
      Potencial
    ),
    
    area_ha = as.numeric(
      `Área (ha)`
    )
  ) |>
  dplyr::mutate(
    dplyr::across(
      c(
        tipo_tierra,
        necesidad_hidrica,
        disponibilidad,
        regulacion,
        ecosistemico,
        socioeconomico,
        potencial
      ),
      ~ stringr::str_squish(
        stringr::str_to_upper(.x)
      )
    )
  ) # Estandarizar variables categóricas

# Validaciones básicas
# ============================================================
geometrias_wkt_vacias <- sum(
  is.na(irrigacion_clean$geometry) |
    irrigacion_clean$geometry == ""
)
cat("Geometrías WKT vacías:", geometrias_wkt_vacias, "\n")

# Resumen estandarización
# ============================================================
cat("\nESTANDARIZACIÓN COMPLETADA\n")
cat("Registros:", nrow(irrigacion_clean), "\n")
cat("Variables:", ncol(irrigacion_clean), "\n")
# ------------------- Fin Bloque 3 ------------------

# ------------------- Bloque 4 ------------------
# 4. Construcción de scores
# ============================================================
irrigacion_clean <- irrigacion_clean |>
  dplyr::mutate(
    tipo_tierra_score = dplyr::case_when(
      
      stringr::str_detect(
        tipo_tierra,
        "^TIPO 1"
      ) ~ 5,
      
      stringr::str_detect(
        tipo_tierra,
        "^TIPO 2"
      ) ~ 4,
      
      stringr::str_detect(
        tipo_tierra,
        "^TIPO 3"
      ) ~ 3,
      
      stringr::str_detect(
        tipo_tierra,
        "^TIPO 4"
      ) ~ 2,
      
      stringr::str_detect(
        tipo_tierra,
        "^TIPO 5"
      ) ~ 1,
      TRUE ~ NA_real_
    ), # Score aptitud física
    
    necesidad_hidrica_score = dplyr::case_when(
      necesidad_hidrica == "N1_ALTA" ~ 3,
      necesidad_hidrica == "N2_MODERADA" ~ 2,
      necesidad_hidrica == "N3_BAJA" ~ 1,
      TRUE ~ NA_real_
    ), # Score necesidad hídrica
    
    disponibilidad_score = dplyr::case_when(
      disponibilidad == "D1_ALTA" ~ 4,
      disponibilidad == "D2_MODERADA" ~ 3,
      disponibilidad == "D3_BAJA" ~ 2,
      disponibilidad == "D4_CRITICA" ~ 1,
      TRUE ~ NA_real_
    ), # Score disponibilidad hídrica
    
    regulacion_score = dplyr::case_when(
      regulacion == "R1_ALTA" ~ 3,
      regulacion == "R2_MODERADA" ~ 2,
      regulacion == "R3_BAJA" ~ 1,
      TRUE ~ NA_real_
    ), # Score regulación
    
    ecosistemico_score = dplyr::case_when(
      ecosistemico == "A1_ALTA" ~ 3,
      ecosistemico == "A2_MODERADA" ~ 2,
      ecosistemico == "A3_BAJA" ~ 1,
      TRUE ~ NA_real_
    ), # Score ecosistémico
    
    socioeconomico_score = dplyr::case_when(
      socioeconomico == "Q1" ~ 4,
      socioeconomico == "Q2" ~ 3,
      socioeconomico == "Q3" ~ 2,
      socioeconomico == "Q4" ~ 1,
      TRUE ~ NA_real_
    ), # Score socioeconómico
    
    potencial_score = as.numeric(
      stringr::str_extract(
        potencial,
        "^[1-5]"
      )
    ) # Score potencial
  )

# Validación básica scores
# ============================================================
na_scores <- irrigacion_clean |>
  dplyr::summarise(
    tipo_tierra_score = sum(is.na(tipo_tierra_score)),
    necesidad_hidrica_score = sum(is.na(necesidad_hidrica_score)),
    disponibilidad_score = sum(is.na(disponibilidad_score)),
    regulacion_score = sum(is.na(regulacion_score)),
    ecosistemico_score = sum(is.na(ecosistemico_score)),
    socioeconomico_score = sum(is.na(socioeconomico_score)),
    potencial_score = sum(is.na(potencial_score))
  )
print(na_scores)

# Resumen construcción scores
# ============================================================
cat("Variables score construidas correctamente\n")
# ------------------- Fin Bloque 4 ------------------

# ------------------- Bloque 5 ------------------
# 5. Validaciones básicas de calidad
# ============================================================

# Validación identificador único
# ============================================================
duplicados_id_original <- irrigacion_clean |>
  dplyr::count(
    id_poligono
  ) |>
  dplyr::filter(
    n > 1
  ) # Validar unicidad identificador

if (nrow(duplicados_id_original) > 0) {
  warning(
    paste(
      "Existen",
      nrow(duplicados_id_original),
      "id_poligono duplicados."
    )
  )
}

# Validación variables obligatorias
# ============================================================
na_criticos <- irrigacion_clean |>
  dplyr::filter(
    is.na(id_poligono) |
      is.na(fisico) |
      is.na(area_ha)
  ) # Variables obligatorias

if (nrow(na_criticos) > 0) {
  warning(
    paste(
      "Existen",
      nrow(na_criticos),
      "registros con NA en variables obligatorias."
    )
  )
}

# Auditoría scores
# ============================================================
auditoria_scores <- irrigacion_clean |>
  dplyr::summarise(
    tipo_tierra_score = sum(
      is.na(tipo_tierra_score)
    ),
    
    necesidad_hidrica_score = sum(
      is.na(necesidad_hidrica_score)
    ),
    
    disponibilidad_score = sum(
      is.na(disponibilidad_score)
    ),
    
    regulacion_score = sum(
      is.na(regulacion_score)
    ),
    
    ecosistemico_score = sum(
      is.na(ecosistemico_score)
    ),
    
    socioeconomico_score = sum(
      is.na(socioeconomico_score)
    ),
    
    potencial_score = sum(
      is.na(potencial_score)
    )
  ) # Auditoría scores
cat("Scores con NA:", sum(unlist(auditoria_scores)), "\n")

if (
  sum(
    unlist(auditoria_scores)
  ) > 0
) {
  
  warning(
    "Existen scores con valores NA."
  )
}

# Resumen validaciones
# ============================================================
cat("\nVALIDACIONES COMPLETADAS\n")
cat("Registros:", nrow(irrigacion_clean), "\n")
cat("Duplicados id_poligono:", nrow(duplicados_id_original), "\n")
cat("NA críticos:", nrow(na_criticos), "\n")
cat("Área mínima (ha):", min(irrigacion_clean$area_ha, na.rm = TRUE), "\n")
cat("Área máxima (ha):", max(irrigacion_clean$area_ha, na.rm = TRUE), "\n")
cat("Fisico mínimo:", min(irrigacion_clean$fisico, na.rm = TRUE), "\n")
cat("Fisico máximo:", max(irrigacion_clean$fisico, na.rm = TRUE), "\n")
# ------------------- Fin Bloque 5 ------------------

# ------------------- Bloque 6 ------------------
# 6. Conversión WKT → sf
# ============================================================
if (!"geometry" %in% names(irrigacion_clean)) {
  stop(
    "No existe la columna geometry."
  ) # Validar existencia de geometría
}

irrigacion_sf <- irrigacion_clean |>
  dplyr::mutate(
    geometry = sf::st_as_sfc(
      geometry,
      crs = crs_geografico
    )
  ) |>
  sf::st_as_sf() # Convertir WKT a objeto sf

# Validación estructura espacial
# ============================================================
tipos_geometria <- unique(
  as.character(
    sf::st_geometry_type(
      irrigacion_sf
    )
  )
) # Tipos geométricos detectados

crs_detectado <- sf::st_crs(
  irrigacion_sf
)$epsg # CRS detectado

if (is.na(crs_detectado)) {
  stop(
    "La capa de irrigación no tiene CRS definido."
  ) # Validar CRS
}

cat("\nGEOMETRÍAS\n")
cat("Tipo geometría:", paste(tipos_geometria, collapse = ", "), "\n")
cat("CRS:", crs_detectado, "\n")
cat("Registros espaciales:", nrow(irrigacion_sf), "\n")
# ------------------- Fin Bloque 6 ------------------

# ------------------- Bloque 7 ------------------
# 7. Validación y reparación geométrica
# ============================================================

# Validación inicial
# ============================================================
geometrias_na <- irrigacion_sf |>
  dplyr::filter(
    is.na(geometry)
  ) # Geometrías NA

geometrias_vacias <- irrigacion_sf |>
  dplyr::filter(
    sf::st_is_empty(geometry)
  ) # Geometrías vacías

geometrias_invalidas <- irrigacion_sf |>
  dplyr::filter(
    !sf::st_is_valid(geometry)
  ) # Geometrías inválidas

cat("Geometrías NA:", nrow(geometrias_na), "\n")
cat("Geometrías vacías:", nrow(geometrias_vacias), "\n")
cat("Geometrías inválidas:", nrow(geometrias_invalidas), "\n")

# Reparación geométrica
# ============================================================
if (nrow(geometrias_invalidas) > 0) {
  irrigacion_sf <- sf::st_make_valid(
    irrigacion_sf
  ) # Reparar geometrías inválidas
}
cat("Registros después de reparación:", nrow(irrigacion_sf), "\n")

# Validación posterior
# ============================================================
geometrias_invalidas_post <- irrigacion_sf |>
  dplyr::filter(
    !sf::st_is_valid(geometry)
  ) # Geometrías inválidas posteriores

cat(
  "Geometrías inválidas después de reparación:",
  nrow(geometrias_invalidas_post),
  "\n"
)

# Resumen geométrico
# ============================================================
estado_geometrias <- (
  nrow(geometrias_na) == 0 &&
    nrow(geometrias_vacias) == 0 &&
    nrow(geometrias_invalidas_post) == 0
) # Estado final geometrías

cat("Estado geometrías:", ifelse(estado_geometrias, "OK", "REVISAR"), "\n")
# ------------------- Fin Bloque 7 ------------------

# ------------------- Bloque 8 ------------------
# 8. Validación de áreas
# ============================================================

# Cálculo área geométrica
# ============================================================
irrigacion_sf <- irrigacion_sf |>
  dplyr::mutate(
    area_geometrica_ha = as.numeric(
      units::set_units(
        sf::st_area(
          sf::st_transform(
            geometry,
            crs_proyectado
          )
        ),
        "ha"
      )
    )
  ) # Área calculada desde geometría

# Diagnóstico escala área atributo
# ============================================================
datos_ratio <- irrigacion_sf |>
  sf::st_drop_geometry() |>
  dplyr::filter(
    area_ha >= 10,
    area_geometrica_ha >= 10
  ) # Polígonos válidos para comparación

if (nrow(datos_ratio) == 0) {
  stop(
    "No existen polígonos válidos para evaluar consistencia de áreas."
  )
}

ratio_area <- datos_ratio |>
  dplyr::summarise(
    ratio_medio = mean(
      area_ha /
        area_geometrica_ha,
      na.rm = TRUE
    ),
    
    ratio_mediano = median(
      area_ha /
        area_geometrica_ha,
      na.rm = TRUE
    )
  ) # Diagnóstico de escala

cat("Ratio medio área atributo/geometría:", round(ratio_area$ratio_medio, 2), "\n")
cat("Ratio mediano área atributo/geometría:", round(ratio_area$ratio_mediano, 2), "\n")

# Corrección escala institucional
# ============================================================
correccion_area_aplicada <- FALSE # Indicador corrección

if (
  ratio_area$ratio_mediano > 95 &&
  ratio_area$ratio_mediano < 105
) {
  
  irrigacion_sf <- irrigacion_sf |>
    dplyr::mutate(
      area_ha_fuente = area_ha,
      area_ha = area_ha / 100
    ) # Corregir escala institucional
  correccion_area_aplicada <- TRUE # Registrar corrección
  cat("Corrección aplicada: area_ha dividido por 100\n")
}

# Comparación área atributo vs área geométrica
# ============================================================
irrigacion_sf <- irrigacion_sf |>
  dplyr::mutate(
    diferencia_area_pct = dplyr::if_else(
      area_ha > 0,
      abs(
        area_geometrica_ha -
          area_ha
      ) /
        area_ha * 100,
      NA_real_
    )
  ) # Diferencia porcentual

areas_inconsistentes <- irrigacion_sf |>
  dplyr::filter(
    area_geometrica_ha >= 10,
    diferencia_area_pct > 10
  ) # Diferencias superiores al 10%

universo_comparable <- irrigacion_sf |>
  dplyr::filter(
    area_geometrica_ha >= 10
  ) # Universo comparable

porcentaje_areas_inconsistentes <- round(
  nrow(areas_inconsistentes) /
    max(
      nrow(universo_comparable),
      1
    ) * 100,
  2
) # Porcentaje inconsistente

cat("Áreas inconsistentes (>10%):", nrow(areas_inconsistentes), "\n")
cat("Áreas inconsistentes (%):", porcentaje_areas_inconsistentes, "\n")

# Resumen áreas
# ============================================================
cat("Área atributo promedio:", round(mean(irrigacion_sf$area_ha, na.rm = TRUE), 2), "\n")
cat("Área geométrica promedio:", round(mean(irrigacion_sf$area_geometrica_ha, na.rm = TRUE), 2), "\n")
cat("Corrección área aplicada:", ifelse(correccion_area_aplicada, "SI", "NO"), "\n")
# ------------------- Fin Bloque 8 ------------------

# ------------------- Bloque 9 ------------------
# 9. Exportación dataset limpio
# ============================================================
if (!inherits(irrigacion_sf, "sf")) {
  stop(
    "El objeto final no es una capa sf."
  ) # Validar objeto espacial
}

if (guardar_version_parquet) {
  arrow::write_parquet(
    sf::st_drop_geometry(
      irrigacion_sf
    ),
    ruta_processed_irrigacion
  ) # Exportar atributos
}

sf::st_write(
  irrigacion_sf,
  ruta_processed_irrigacion_gpkg,
  delete_dsn = TRUE,
  quiet = TRUE
) # Exportar capa espacial

# Validación exportación
# ============================================================
if (
  guardar_version_parquet &&
  !file.exists(
    ruta_processed_irrigacion
  )
) {
  stop(
    "No fue posible exportar el archivo parquet."
  ) # Validar parquet
}

if (
  !file.exists(
    ruta_processed_irrigacion_gpkg
  )
) {
  stop(
    "No fue posible exportar el archivo geopackage."
  ) # Validar geopackage
}

# Resumen limpieza
# ============================================================
cat("\nLIMPIEZA COMPLETADA\n")
cat("Registros finales:", nrow(irrigacion_sf), "\n")
cat("Variables finales:", ncol(irrigacion_sf), "\n")
cat("Estado geometrías:", ifelse(estado_geometrias, "OK", "REVISAR"), "\n")
cat("Corrección área aplicada:", ifelse(correccion_area_aplicada, "SI", "NO"), "\n")
cat("Archivo parquet:", ifelse(guardar_version_parquet, "EXPORTADO", "NO EXPORTADO"), "\n")
cat("Archivo geopackage: EXPORTADO\n")
cat("Exportación completada correctamente\n")
# ------------------- Fin Bloque 9 ------------------
