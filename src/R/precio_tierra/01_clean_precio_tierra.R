# 01_clean_precio_tierra.R

# Bloque 1. Configuración y carga de datos
# Objetivo: Cargar y validar la base original de precios de tierra
# Salida: precio_tierra_raw
# ============================================================

source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros

archivo_precio_tierra <- here::here(
  "data/raw/precios_tierra",
  "Precios_comerciales_tierra_rural_agropecuaria_mercado_de_tierras_20260502.csv"
) # Ruta archivo fuente

if (!file.exists(archivo_precio_tierra)) {
  stop("ERROR: No se encontró el archivo de precios de tierra.")
} # Validar existencia

precio_tierra_raw <- data.table::fread(
  archivo_precio_tierra,
  encoding = "UTF-8"
) # Cargar dataset

cat("\n================ AUDITORÍA INICIAL =================\n") # Encabezado
cat("Registros:", format(nrow(precio_tierra_raw), big.mark = ","), "\n") # Total registros
cat("Variables:", ncol(precio_tierra_raw), "\n") # Total variables
cat("Departamentos:", dplyr::n_distinct(precio_tierra_raw$Departamento), "\n") # Total departamentos
cat("Municipios:", dplyr::n_distinct(precio_tierra_raw$`Código municipio`), "\n") # Total municipios
cat(
  "Memoria:",
  round(as.numeric(object.size(precio_tierra_raw)) / 1024^2, 2),
  "MB\n"
) # Tamaño objeto
cat("====================================================\n") # Fin auditoría

tipos_variables <- tibble::tibble(
  variable = names(precio_tierra_raw),
  tipo = purrr::map_chr(precio_tierra_raw, ~ class(.x)[1])
) # Inventario de variables

print(tipos_variables) # Mostrar tipos

# Hallazgo 1: Memoria excesiva
sapply(
  precio_tierra_raw,
  \(x) round(as.numeric(object.size(x)) / 1024^2, 2)
) |> sort(decreasing = TRUE)

# Hallazgo 2: Cobertura territorial
sort(unique(precio_tierra_raw$Departamento))
table(precio_tierra_raw$Departamento)

# Hallazgo 3: Tipos de datos
# Aquí existe mezcla de coma decimal punto miles Por tanto 
# NO debes usar as.numeric() directamente. Habrá que construir una función de limpieza robusta en el Bloque 3. 

# Hallazgo 4: Consecutivo
nrow(precio_tierra_raw)
dplyr::n_distinct(precio_tierra_raw$Consecutivo)

# Hallazgo 5: Geometría
# Debes decidir desde ahora si la conservarás. Tu objetivo final es: municipio-año y esta base: 
# no tiene año Además ya tienes: cod_mpio municipio 
# Por tanto para el proyecto GNN yo eliminaría: 
# the_geom en el Bloque 3. 
# Ganancia esperada: 1.8 GB → 30-50 MB aproximadamente. 
# Conclusión Puedes avanzar al Bloque 2 porque: 
# lectura correcta estructura estable sin nulos 
# variables identificadas 
# Pero antes de iniciar el Bloque 3 te recomiendo ejecutar estas tres validaciones: 
sort(unique(precio_tierra_raw$Departamento))
table(precio_tierra_raw$Departamento)
sapply(
  precio_tierra_raw,
  \(x) round(as.numeric(object.size(x)) / 1024^2, 2)
) |> sort(decreasing = TRUE)

cat("\nCarga completada correctamente.\n") # Confirmación

# Bloque 2. Estandarización de variables
# Objetivo: Homogeneizar nombres y estructura
# Salida: precio_tierra_std
# ============================================================

precio_tierra_std <- precio_tierra_raw |>
  dplyr::rename(
    cod_depto = `Código departamento`, # Código DANE departamento
    departamento = Departamento, # Nombre departamento
    cod_mpio = `Código municipio`, # Código DANE municipio
    municipio = Municipio, # Nombre municipio
    categoria_mt = `Categoría MT`, # Categoría mercado de tierras
    rango_precio = `Rango precios`, # Rango económico
    codigo_precio = `Código precios`, # Código rango precio
    area_ha = `Área (ha)`, # Área en hectáreas
    consecutivo = Consecutivo # Identificador interno
  ) # Estandarizar nombres

cat("\n================ BLOQUE 2 =================\n") # Encabezado

cat(
  "Variables renombradas:",
  ncol(precio_tierra_std),
  "| Registros:",
  format(nrow(precio_tierra_std), big.mark = ","),
  "\n"
) # Resumen general

cat(
  "Nombres estandarizados correctamente.\n"
) # Confirmación

glimpse(precio_tierra_std) # Verificar estructura

summary(precio_tierra_std$area_ha)
head(sort(unique(precio_tierra_std$area_ha)), 20)
tail(sort(unique(precio_tierra_std$area_ha)), 20)
sum(stringr::str_detect(precio_tierra_std$area_ha, "\\."))

precio_tierra_std |>
  dplyr::filter(stringr::str_detect(area_ha, "\\.")) |>
  dplyr::select(area_ha) |>
  dplyr::distinct() |>
  head(30)

nchar(as.character(precio_tierra_std$cod_mpio)) |> table()

precio_tierra_std |>
  dplyr::filter(
    nchar(as.character(cod_mpio)) == 4
  ) |>
  dplyr::distinct(
    cod_depto,
    cod_mpio,
    municipio
  ) |>
  head(30)

# ============================================================
# Bloque 3. Limpieza y transformación
# Objetivo: Corregir tipos de datos y estandarizar estructura
# Salida: precio_tierra_limpio
# ============================================================

precio_tierra_limpio <- precio_tierra_std |>
  dplyr::select(-the_geom) |> # Eliminar geometría
  dplyr::mutate(
    cod_depto = stringr::str_pad(as.character(cod_depto), 2, "left", "0"), # Código departamento
    cod_mpio = stringr::str_pad(as.character(cod_mpio), 5, "left", "0"), # Código municipio
    dplyr::across(
      c(departamento, municipio, categoria_mt, rango_precio, codigo_precio),
      stringr::str_squish
    ), # Limpiar textos
    area_ha = as.numeric(
      stringr::str_replace(
        stringr::str_remove_all(area_ha, "\\."),
        ",",
        "."
      )
    ) # Convertir área a numérico
  )

cat(
  "\n================ BLOQUE 3 =================\n",
  "Registros:", format(nrow(precio_tierra_limpio), big.mark = ","), "|",
  "Municipios:", dplyr::n_distinct(precio_tierra_limpio$cod_mpio), "|",
  "Nulos área:", sum(is.na(precio_tierra_limpio$area_ha)), "|",
  "Memoria:", round(as.numeric(object.size(precio_tierra_limpio)) / 1024^2, 2), "MB\n"
) # Resumen auditoría

# Verificar codigo DANE
precio_tierra_limpio |>
  dplyr::select(
    cod_depto,
    cod_mpio,
    municipio
  ) |>
  dplyr::distinct() |>
  head(20)

table(nchar(precio_tierra_limpio$cod_depto))
table(nchar(precio_tierra_limpio$cod_mpio))

summary(precio_tierra_limpio$area_ha)

quantile(
  precio_tierra_limpio$area_ha,
  probs = c(0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1),
  na.rm = TRUE
)

precio_tierra_limpio |>
  dplyr::arrange(desc(area_ha)) |>
  dplyr::select(
    cod_depto,
    departamento,
    cod_mpio,
    municipio,
    area_ha,
    categoria_mt
  ) |>
  head(20)

sum(precio_tierra_limpio$area_ha == 0)
sum(precio_tierra_limpio$area_ha < 0.001)

precio_tierra_limpio |>
  dplyr::filter(area_ha < 0.001) |>
  dplyr::count(categoria_mt, sort = TRUE)

precio_tierra_limpio |>
  dplyr::filter(area_ha < 0.001) |>
  dplyr::count(rango_precio, sort = TRUE)

# ============================================================
# Bloque 4. Auditoría de calidad
# Objetivo: Evaluar integridad del dataset
# Salida: tablas_auditoria
# ============================================================

auditoria_general <- tibble::tibble(
  registros = nrow(precio_tierra_limpio),
  variables = ncol(precio_tierra_limpio),
  departamentos = dplyr::n_distinct(precio_tierra_limpio$cod_depto),
  municipios = dplyr::n_distinct(precio_tierra_limpio$cod_mpio),
  nulos = sum(is.na(precio_tierra_limpio)),
  duplicados = sum(duplicated(precio_tierra_limpio))
) # Resumen general

auditoria_nulos <- tibble::tibble(
  variable = names(precio_tierra_limpio),
  nulos = sapply(precio_tierra_limpio, \(x) sum(is.na(x))),
  porcentaje_nulos = round(
    sapply(precio_tierra_limpio, \(x) mean(is.na(x))) * 100,
    4
  )
) # Auditoría de nulos

auditoria_tipos <- tibble::tibble(
  variable = names(precio_tierra_limpio),
  tipo = purrr::map_chr(precio_tierra_limpio, ~ class(.x)[1])
) # Tipos de datos

auditoria_area <- precio_tierra_limpio |>
  dplyr::summarise(
    area_min = min(area_ha, na.rm = TRUE),
    area_p01 = quantile(area_ha, 0.01, na.rm = TRUE),
    area_p25 = quantile(area_ha, 0.25, na.rm = TRUE),
    area_mediana = median(area_ha, na.rm = TRUE),
    area_media = mean(area_ha, na.rm = TRUE),
    area_p75 = quantile(area_ha, 0.75, na.rm = TRUE),
    area_p99 = quantile(area_ha, 0.99, na.rm = TRUE),
    area_max = max(area_ha, na.rm = TRUE),
    area_cero = sum(area_ha == 0, na.rm = TRUE),
    area_menor_001 = sum(area_ha < 0.001, na.rm = TRUE)
  ) # Distribución de áreas

tablas_auditoria <- list(
  auditoria_general = auditoria_general,
  auditoria_nulos = auditoria_nulos,
  auditoria_tipos = auditoria_tipos,
  auditoria_area = auditoria_area
) # Consolidar auditorías

cat(
  "\n================ BLOQUE 4 =================\n",
  "Registros:", auditoria_general$registros, "|",
  "Municipios:", auditoria_general$municipios, "|",
  "Nulos:", auditoria_general$nulos, "|",
  "Duplicados:", auditoria_general$duplicados, "\n"
) # Resumen auditoría

print(auditoria_area)

# ============================================================
# Bloque 5. Recodificación económica
# Objetivo: Transformar rangos de precios en variable numérica
# Salida: precio_tierra_modelado
# ============================================================

rangos_excluir <- c(
  "CA",
  "NS",
  "SI",
  "ZU",
  "No suelo",
  "Sin valor",
  "Cuerpo de Agua",
  "Zona Urbana"
) # Categorías no económicas

precio_tierra_modelado <- precio_tierra_limpio |>
  dplyr::filter(
    !codigo_precio %in% c("CA", "NS", "SI", "ZU"),
    !rango_precio %in% rangos_excluir
  ) |>
  dplyr::mutate(
    rango_precio = stringr::str_replace_all(
      rango_precio,
      "Mayor que",
      "Mayor a"
    )
  ) |>
  dplyr::mutate(
    
    limite_inferior = dplyr::case_when(
      rango_precio == "Menor a 1" ~ 0,
      rango_precio == "Mayor a 1000" ~ 1000,
      TRUE ~ as.numeric(
        stringr::str_extract(
          rango_precio,
          "\\d+"
        )
      )
    ),
    
    limite_superior = dplyr::case_when(
      rango_precio == "Menor a 1" ~ 1,
      rango_precio == "Mayor a 1000" ~ 1200,
      TRUE ~ as.numeric(
        stringr::str_extract(
          rango_precio,
          "(?<=hasta\\s)\\d+"
        )
      )
    ),
    
    precio_ha_referencia = (
      limite_inferior +
        limite_superior
    ) / 2
    
  ) # Construir precio de referencia

precio_tierra_modelado

# ¿Existen NAs en precio_ha_referencia?
sum(is.na(precio_tierra_modelado$precio_ha_referencia))

# ¿Cómo quedaron los rangos únicos?
precio_tierra_modelado |>
  dplyr::select(
    rango_precio,
    precio_ha_referencia
  ) |>
  dplyr::distinct() |>
  dplyr::arrange(precio_ha_referencia)

# Resumen de precio_ha_referencia
summary(precio_tierra_modelado$precio_ha_referencia)

precio_tierra_modelado <- precio_tierra_modelado |>
  dplyr::select(
    -limite_inferior,
    -limite_superior
  )

# ============================================================
# Bloque 6. Validación final
# Objetivo: Garantizar calidad antes de exportar
# Salida: dataset_aprobado
# ============================================================

validacion_final <- tibble::tibble(
  registros = nrow(precio_tierra_modelado),
  variables = ncol(precio_tierra_modelado),
  departamentos = dplyr::n_distinct(precio_tierra_modelado$cod_depto),
  municipios = dplyr::n_distinct(precio_tierra_modelado$cod_mpio),
  nulos = sum(is.na(precio_tierra_modelado)),
  duplicados = sum(duplicated(precio_tierra_modelado)),
  precio_no_positivo = sum(precio_tierra_modelado$precio_ha_referencia <= 0),
  cod_depto_invalidos = sum(
    nchar(precio_tierra_modelado$cod_depto) != 2
  ),
  cod_mpio_invalidos = sum(
    nchar(precio_tierra_modelado$cod_mpio) != 5
  )
) # Resumen validación

dataset_aprobado <- precio_tierra_modelado # Dataset final aprobado

cat(
  "\n================ BLOQUE 6 =================\n",
  "Registros:", validacion_final$registros, "|",
  "Municipios:", validacion_final$municipios, "|",
  "Nulos:", validacion_final$nulos, "|",
  "Duplicados:", validacion_final$duplicados, "\n",
  "Precio <= 0:", validacion_final$precio_no_positivo, "|",
  "Cod. depto inválidos:", validacion_final$cod_depto_invalidos, "|",
  "Cod. municipio inválidos:", validacion_final$cod_mpio_invalidos, "\n"
) # Resumen validación

print(validacion_final) # Mostrar resultados

# ============================================================
# Bloque 7. Exportación
# Objetivo: Exportar dataset final y auditorías
# Salida: parquet + csv
# ============================================================

dir.create(
  here::here("data", "processed", "precios_tierra"),
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta salida

arrow::write_parquet(
  dataset_aprobado,
  here::here(
    "data",
    "processed",
    "precios_tierra",
    "precios_tierra_limpio.parquet"
  )
) # Exportar dataset principal

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
  auditoria_tipos,
  here::here(
    "data",
    "processed",
    "precios_tierra",
    "auditoria_tipos_precios_tierra.csv"
  )
) # Exportar auditoría tipos

readr::write_csv(
  auditoria_area,
  here::here(
    "data",
    "processed",
    "precios_tierra",
    "auditoria_estadisticas_precios_tierra.csv"
  )
) # Exportar estadísticas descriptivas

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
    nrow(dataset_aprobado),
    ncol(dataset_aprobado),
    dplyr::n_distinct(dataset_aprobado$cod_depto),
    dplyr::n_distinct(dataset_aprobado$cod_mpio),
    sum(is.na(dataset_aprobado)),
    sum(duplicated(dataset_aprobado)),
    min(dataset_aprobado$precio_ha_referencia),
    max(dataset_aprobado$precio_ha_referencia)
  )
) # Resumen ejecutivo

readr::write_csv(
  reporte_ejecutivo,
  here::here(
    "data",
    "processed",
    "precios_tierra",
    "reporte_ejecutivo_precios_tierra.csv"
  )
) # Exportar reporte ejecutivo

# Validación
list.files(
  here::here("data", "processed", "precios_tierra"),
  full.names = FALSE
)

cat(
  "\n================ BLOQUE 7 =================\n",
  "Dataset exportado:", nrow(dataset_aprobado), "registros |",
  ncol(dataset_aprobado), "variables\n",
  "Archivos generados: 5\n",
  "Ruta:",
  here::here("data", "processed", "precios_tierra"),
  "\n"
) # Resumen exportación
