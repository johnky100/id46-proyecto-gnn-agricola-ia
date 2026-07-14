# ============================================================
# Bloque 1. Configuración y carga
# Objetivo: Cargar dataset limpio de precios de tierra
# Salida: precio_tierra
# ============================================================

# 02_build_precios_tierra_municipio.R

source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros

archivo_precios_tierra <- here::here(
  "data",
  "processed",
  "precios_tierra",
  "precios_tierra_limpio.parquet"
) # Ruta dataset limpio

if (!file.exists(archivo_precios_tierra)) {
  stop("ERROR: No se encontró precios_tierra_limpio.parquet")
} # Validar existencia

precio_tierra <- arrow::read_parquet(
  archivo_precios_tierra
) # Cargar dataset

cat(
  "\n================ BLOQUE 1 =================\n",
  "Registros:", format(nrow(precio_tierra), big.mark = ","), "|",
  "Variables:", ncol(precio_tierra), "|",
  "Departamentos:", dplyr::n_distinct(precio_tierra$cod_depto), "|",
  "Municipios:", dplyr::n_distinct(precio_tierra$cod_mpio), "\n",
  "Nulos:", sum(is.na(precio_tierra)), "|",
  "Duplicados:", sum(duplicated(precio_tierra)), "\n"
) # Resumen auditoría

tipos_variables <- tibble::tibble(
  variable = names(precio_tierra),
  tipo = purrr::map_chr(precio_tierra, ~ class(.x)[1])
) # Inventario de variables

print(tipos_variables) # Mostrar estructura

cat(
  "\nCarga completada correctamente.\n"
) # Confirmación

# ============================================================
# Bloque 2. Análisis exploratorio de precios
# Objetivo: Comprender la distribución económica y territorial
# Salida: tablas_eda
# ============================================================

eda_precio <- precio_tierra |>
  dplyr::summarise(
    precio_min = min(precio_ha_referencia, na.rm = TRUE),
    precio_p25 = quantile(precio_ha_referencia, 0.25, na.rm = TRUE),
    precio_mediana = median(precio_ha_referencia, na.rm = TRUE),
    precio_media = mean(precio_ha_referencia, na.rm = TRUE),
    precio_p75 = quantile(precio_ha_referencia, 0.75, na.rm = TRUE),
    precio_p95 = quantile(precio_ha_referencia, 0.95, na.rm = TRUE),
    precio_max = max(precio_ha_referencia, na.rm = TRUE)
  ) # Distribución de precios

eda_area <- precio_tierra |>
  dplyr::summarise(
    area_min = min(area_ha, na.rm = TRUE),
    area_p25 = quantile(area_ha, 0.25, na.rm = TRUE),
    area_mediana = median(area_ha, na.rm = TRUE),
    area_media = mean(area_ha, na.rm = TRUE),
    area_p75 = quantile(area_ha, 0.75, na.rm = TRUE),
    area_p95 = quantile(area_ha, 0.95, na.rm = TRUE),
    area_max = max(area_ha, na.rm = TRUE)
  ) # Distribución de áreas

eda_categoria_mt <- precio_tierra |>
  dplyr::count(
    categoria_mt,
    sort = TRUE
  ) |>
  dplyr::mutate(
    porcentaje = round(
      n / sum(n) * 100,
      2
    )
  ) # Distribución categorías

eda_municipios_area <- precio_tierra |>
  dplyr::group_by(
    cod_depto,
    departamento,
    cod_mpio,
    municipio
  ) |>
  dplyr::summarise(
    area_total_ha = sum(area_ha, na.rm = TRUE),
    .groups = "drop"
  ) |>
  dplyr::arrange(
    dplyr::desc(area_total_ha)
  ) |>
  dplyr::slice_head(n = 20) # Top municipios por área

tablas_eda <- list(
  eda_precio = eda_precio,
  eda_area = eda_area,
  eda_categoria_mt = eda_categoria_mt,
  eda_municipios_area = eda_municipios_area
) # Consolidar EDA

cat(
  "\n================ BLOQUE 2 =================\n",
  "Categorías MT:", nrow(eda_categoria_mt), "|",
  "Municipios evaluados:", dplyr::n_distinct(precio_tierra$cod_mpio),
  "\n"
) # Resumen

print(eda_precio)

print(eda_area)

print(eda_categoria_mt)

print(eda_municipios_area)

# ============================================================
# Bloque 3. Construcción de indicadores municipales
# Objetivo: Agregar información de precios a nivel municipio
# Salida: precios_tierra_municipio
# ============================================================

precios_tierra_municipio <- precio_tierra |>
  dplyr::filter(area_ha > 0) |> # Excluir áreas nulas
  dplyr::group_by(
    cod_depto,
    departamento,
    cod_mpio,
    municipio
  ) |>
  dplyr::summarise(
    
    precio_ha_promedio = mean(
      precio_ha_referencia,
      na.rm = TRUE
    ), # Precio promedio
    
    precio_ha_mediana = median(
      precio_ha_referencia,
      na.rm = TRUE
    ), # Precio mediano
    
    precio_ha_ponderado = weighted.mean(
      precio_ha_referencia,
      area_ha,
      na.rm = TRUE
    ), # Precio ponderado por área
    
    precio_ha_min = min(
      precio_ha_referencia,
      na.rm = TRUE
    ), # Precio mínimo
    
    precio_ha_max = max(
      precio_ha_referencia,
      na.rm = TRUE
    ), # Precio máximo
    
    precio_ha_sd = sd(
      precio_ha_referencia,
      na.rm = TRUE
    ), # Desviación estándar
    
    precio_ha_p25 = quantile(
      precio_ha_referencia,
      0.25,
      na.rm = TRUE
    ), # Percentil 25
    
    precio_ha_p75 = quantile(
      precio_ha_referencia,
      0.75,
      na.rm = TRUE
    ), # Percentil 75
    
    area_total_ha = sum(
      area_ha,
      na.rm = TRUE
    ), # Área total
    
    area_promedio_ha = mean(
      area_ha,
      na.rm = TRUE
    ), # Área promedio
    
    n_poligonos = dplyr::n(), # Número de polígonos
    
    .groups = "drop"
    
  ) # Agregar municipio

cat(
  "\n================ BLOQUE 3 =================\n",
  "Municipios:", nrow(precios_tierra_municipio), "|",
  "Variables:", ncol(precios_tierra_municipio), "\n"
) # Resumen

glimpse(precios_tierra_municipio)

summary(precios_tierra_municipio$precio_ha_ponderado)

summary(precios_tierra_municipio$area_total_ha)

precios_tierra_municipio |>
  dplyr::arrange(
    dplyr::desc(precio_ha_ponderado)
  ) |>
  dplyr::select(
    cod_mpio,
    municipio,
    precio_ha_ponderado,
    area_total_ha,
    n_poligonos
  ) |>
  head(10)

summary(precios_tierra_municipio$n_poligonos)
summary(precios_tierra_municipio$precio_ha_sd)

# ============================================================
# Bloque 4. Indicadores territoriales
# Objetivo: Construir métricas territoriales para GNN
# Salida: precios_tierra_features
# ============================================================

categoria_municipio <- precio_tierra |>
  dplyr::filter(area_ha > 0) |>
  dplyr::group_by(
    cod_mpio
  ) |>
  dplyr::summarise(
    
    area_condicionada = sum(
      area_ha[categoria_mt == "Condicionada"],
      na.rm = TRUE
    ), # Área condicionada
    
    area_incluida = sum(
      area_ha[categoria_mt == "Incluida"],
      na.rm = TRUE
    ), # Área incluida
    
    area_total = sum(
      area_ha,
      na.rm = TRUE
    ), # Área total
    
    .groups = "drop"
    
  ) |>
  dplyr::mutate(
    
    pct_area_condicionada =
      area_condicionada / area_total,
    
    pct_area_incluida =
      area_incluida / area_total
    
  ) |>
  dplyr::select(
    cod_mpio,
    pct_area_condicionada,
    pct_area_incluida
  )

precios_tierra_features <- precios_tierra_municipio |>
  dplyr::left_join(
    categoria_municipio,
    by = "cod_mpio"
  ) |>
  dplyr::mutate(
    
    indice_fragmentacion =
      n_poligonos / area_total_ha
    
  ) # Fragmentación territorial

cat(
  "\n================ BLOQUE 4 =================\n",
  "Municipios:", nrow(precios_tierra_features), "|",
  "Variables:", ncol(precios_tierra_features), "\n"
)

glimpse(precios_tierra_features)

summary(
  precios_tierra_features$pct_area_condicionada +
    precios_tierra_features$pct_area_incluida
)

# ============================================================
# Bloque 5. Auditoría de calidad
# Objetivo: Validar dataset municipal final
# Salida: tablas_auditoria
# ============================================================

auditoria_general <- tibble::tibble(
  registros = nrow(precios_tierra_features),
  variables = ncol(precios_tierra_features),
  departamentos = dplyr::n_distinct(precios_tierra_features$cod_depto),
  municipios = dplyr::n_distinct(precios_tierra_features$cod_mpio),
  nulos = sum(is.na(precios_tierra_features)),
  duplicados = sum(duplicated(precios_tierra_features))
) # Resumen general

auditoria_nulos <- tibble::tibble(
  variable = names(precios_tierra_features),
  nulos = sapply(precios_tierra_features, \(x) sum(is.na(x))),
  porcentaje_nulos = round(
    sapply(precios_tierra_features, \(x) mean(is.na(x))) * 100,
    4
  )
) # Auditoría de nulos

auditoria_rangos <- precios_tierra_features |>
  dplyr::summarise(
    precio_min = min(precio_ha_ponderado, na.rm = TRUE),
    precio_max = max(precio_ha_ponderado, na.rm = TRUE),
    area_min = min(area_total_ha, na.rm = TRUE),
    area_max = max(area_total_ha, na.rm = TRUE),
    fragmentacion_min = min(indice_fragmentacion, na.rm = TRUE),
    fragmentacion_max = max(indice_fragmentacion, na.rm = TRUE),
    pct_condicionada_min = min(pct_area_condicionada, na.rm = TRUE),
    pct_condicionada_max = max(pct_area_condicionada, na.rm = TRUE),
    pct_incluida_min = min(pct_area_incluida, na.rm = TRUE),
    pct_incluida_max = max(pct_area_incluida, na.rm = TRUE)
  ) # Rangos de variables

auditoria_cobertura <- precios_tierra_features |>
  dplyr::count(
    cod_depto,
    departamento,
    name = "municipios"
  ) |>
  dplyr::arrange(
    dplyr::desc(municipios)
  ) # Cobertura territorial

tablas_auditoria <- list(
  auditoria_general = auditoria_general,
  auditoria_nulos = auditoria_nulos,
  auditoria_rangos = auditoria_rangos,
  auditoria_cobertura = auditoria_cobertura
) # Consolidar auditorías

cat(
  "\n================ BLOQUE 5 =================\n",
  "Municipios:", auditoria_general$municipios, "|",
  "Departamentos:", auditoria_general$departamentos, "|",
  "Nulos:", auditoria_general$nulos, "|",
  "Duplicados:", auditoria_general$duplicados, "\n"
) # Resumen auditoría

print(auditoria_general)

print(auditoria_rangos)

print(auditoria_cobertura)

# ============================================================
# Bloque 6. Exportación
# Objetivo: Exportar dataset municipal y auditorías
# Salida: archivos finales
# ============================================================

dir.create(
  here::here("data", "processed", "precios_tierra"),
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta salida

arrow::write_parquet(
  precios_tierra_features,
  here::here(
    "data",
    "processed",
    "precios_tierra",
    "precios_tierra_features.parquet"
  )
) # Exportar dataset principal

readr::write_csv(
  auditoria_general,
  here::here(
    "data",
    "processed",
    "precios_tierra",
    "auditoria_general_precios_tierra.csv"
  )
) # Exportar auditoría general

readr::write_csv(
  auditoria_nulos,
  here::here(
    "data",
    "processed",
    "precios_tierra",
    "auditoria_nulos_precios_tierra.csv"
  )
) # Exportar auditoría nulos

readr::write_csv(
  auditoria_rangos,
  here::here(
    "data",
    "processed",
    "precios_tierra",
    "auditoria_rangos_precios_tierra.csv"
  )
) # Exportar auditoría rangos

readr::write_csv(
  auditoria_cobertura,
  here::here(
    "data",
    "processed",
    "precios_tierra",
    "auditoria_cobertura_precios_tierra.csv"
  )
) # Exportar auditoría cobertura

reporte_ejecutivo <- tibble::tibble(
  indicador = c(
    "registros",
    "variables",
    "departamentos",
    "municipios",
    "nulos",
    "duplicados",
    "precio_min",
    "precio_max"
  ),
  valor = c(
    nrow(precios_tierra_features),
    ncol(precios_tierra_features),
    dplyr::n_distinct(precios_tierra_features$cod_depto),
    dplyr::n_distinct(precios_tierra_features$cod_mpio),
    sum(is.na(precios_tierra_features)),
    sum(duplicated(precios_tierra_features)),
    min(precios_tierra_features$precio_ha_ponderado),
    max(precios_tierra_features$precio_ha_ponderado)
  )
) # Resumen ejecutivo

readr::write_csv(
  reporte_ejecutivo,
  here::here(
    "data",
    "processed",
    "precios_tierra",
    "reporte_ejecutivo_precios_tierra_municipio.csv"
  )
) # Exportar reporte ejecutivo

cat(
  "\n================ BLOQUE 6 =================\n",
  "Municipios:", nrow(precios_tierra_features), "|",
  "Variables:", ncol(precios_tierra_features), "\n",
  "Archivos generados: 6\n"
) # Resumen exportación

list.files(
  here::here("data", "processed", "precios_tierra"),
  full.names = FALSE
) # Validar archivos exportados
