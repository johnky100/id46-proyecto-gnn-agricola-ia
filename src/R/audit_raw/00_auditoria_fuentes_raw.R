# 00_auditoria_fuentes_raw.R

cat("\n")
cat("BLOQUE 1. REGISTRO DE LAS FUENTES RAW\n")
cat("-----------------------------------------------------\n")

# Directorio raíz de datos RAW
ruta_raw <- "D:/Proyectos_IA/proyecto-gnn-agricola/data/raw"

# Registro de fuentes
fuentes_raw <- data.frame(
  fuente = c(
    "EVA",
    "ERA5",
    "CNA",
    "IRRIGACION",
    "DIVIPOLA",
    "SPATIAL"
  ),
  tipo = c(
    "CSV",
    "NetCDF",
    "Shapefile",
    "CSV",
    "CSV",
    "Shapefile"
  ),
  ruta = c(
    file.path(ruta_raw, "eva", "Evaluaciones_Agropecuarias_Municipales_EVA_20260413.csv"),
    file.path(ruta_raw, "era5", "data_stream-mnth.nc"),
    file.path(ruta_raw, "cna", "Censo_Nacional_Agropecuario_Uso_de_la_tierra.shp"),
    file.path(ruta_raw, "irrigacion", "Areas_potenciales_para_adecuación_de_tierras_con_fines_de_irrigacion_20260413.csv"),
    file.path(ruta_raw, "divipola", "divipola_municipios.csv"),
    file.path(ruta_raw, "spatial", "Muni.shp")
  ),
  stringsAsFactors = FALSE
)

print(fuentes_raw)

cat("\nNúmero de fuentes registradas :", nrow(fuentes_raw), "\n")

cat("\n")
cat("BLOQUE 2. VALIDACIÓN DE EXISTENCIA DE LAS FUENTES RAW\n")
cat("-----------------------------------------------------\n")

# Verificar existencia de los archivos
fuentes_raw$existe <- file.exists(
  fuentes_raw$ruta
)

# Obtener tamaño del archivo (MB)
fuentes_raw$tamano_mb <- ifelse(
  fuentes_raw$existe,
  round(file.info(fuentes_raw$ruta)$size / 1024^2, 2),
  NA_real_
)

# Estado de validación
fuentes_raw$estado <- ifelse(
  fuentes_raw$existe,
  "OK",
  "NO ENCONTRADO"
)

# Mostrar resultados
print(
  fuentes_raw |>
    dplyr::select(
      fuente,
      tipo,
      estado,
      tamano_mb
    )
)

cat("\n")
cat("Fuentes encontradas     :", sum(fuentes_raw$existe), "\n")
cat("Fuentes no encontradas  :", sum(!fuentes_raw$existe), "\n")

# Detener el proceso si falta alguna fuente
if (any(!fuentes_raw$existe)) {
  
  stop(
    "ERROR: Existen fuentes RAW que no fueron encontradas. Corregir antes de continuar."
  )
  
} else {
  
  cat("\n")
  cat("OK - Todas las fuentes RAW existen y están disponibles.\n")
  
}

cat("\n")
cat("BLOQUE 3. AUDITORÍA DEL DATASET EVA\n")
cat("-----------------------------------------------------\n")

# Ruta del dataset EVA
ruta_eva <- fuentes_raw$ruta[
  fuentes_raw$fuente == "EVA"
]

# Cargar dataset
eva <- read.csv(
  ruta_eva,
  header = TRUE,
  stringsAsFactors = FALSE,
  fileEncoding = "latin1"
)

cat("Filas    :", nrow(eva), "\n")
cat("Columnas :", ncol(eva), "\n\n")

# Estructura
glimpse(eva)

cat("\n")
cat("VALORES FALTANTES\n")
cat("-----------------------------------------------------\n")

print(
  data.frame(
    variable = names(eva),
    tipo = sapply(eva, \(x) class(x)[1]),
    nulos = colSums(is.na(eva)),
    pct_nulos = round(
      100 * colSums(is.na(eva)) / nrow(eva),
      2
    ),
    stringsAsFactors = FALSE
  )
)

cat("\n")
cat("RESUMEN ESTADÍSTICO\n")
cat("-----------------------------------------------------\n")

summary(eva)

cat("\n")
cat("BLOQUE 4. AUDITORÍA DEL DATASET IRRIGACIÓN\n")
cat("-----------------------------------------------------\n")

# Ruta del dataset de irrigación
ruta_irrigacion <- fuentes_raw$ruta[
  fuentes_raw$fuente == "IRRIGACION"
]

# Cargar dataset
irrigacion <- read.csv(
  ruta_irrigacion,
  header = TRUE,
  stringsAsFactors = FALSE,
  fileEncoding = "latin1"
)

cat("Filas    :", nrow(irrigacion), "\n")
cat("Columnas :", ncol(irrigacion), "\n\n")

# Estructura
glimpse(irrigacion)

cat("\n")
cat("VALORES FALTANTES\n")
cat("-----------------------------------------------------\n")

print(
  data.frame(
    variable = names(irrigacion),
    tipo = sapply(irrigacion, \(x) class(x)[1]),
    nulos = colSums(is.na(irrigacion)),
    pct_nulos = round(
      100 * colSums(is.na(irrigacion)) / nrow(irrigacion),
      2
    ),
    stringsAsFactors = FALSE
  )
)

cat("\n")
cat("RESUMEN ESTADÍSTICO\n")
cat("-----------------------------------------------------\n")

summary(irrigacion)

cat("\n")
cat("BLOQUE 5. AUDITORÍA DEL DATASET DIVIPOLA\n")
cat("-----------------------------------------------------\n")

# Ruta del dataset DIVIPOLA
ruta_divipola <- fuentes_raw$ruta[
  fuentes_raw$fuente == "DIVIPOLA"
]

# Cargar dataset
divipola <- read.csv(
  ruta_divipola,
  header = TRUE,
  stringsAsFactors = FALSE,
  fileEncoding = "latin1"
)

cat("Filas    :", nrow(divipola), "\n")
cat("Columnas :", ncol(divipola), "\n\n")

# Estructura
glimpse(divipola)

cat("\n")
cat("VALORES FALTANTES\n")
cat("-----------------------------------------------------\n")

print(
  data.frame(
    variable = names(divipola),
    tipo = sapply(divipola, \(x) class(x)[1]),
    nulos = colSums(is.na(divipola)),
    pct_nulos = round(
      100 * colSums(is.na(divipola)) / nrow(divipola),
      2
    ),
    stringsAsFactors = FALSE
  )
)

cat("\n")
cat("RESUMEN ESTADÍSTICO\n")
cat("-----------------------------------------------------\n")

summary(divipola)

cat("\n")
cat("BLOQUE 6. AUDITORÍA DEL DATASET CNA\n")
cat("-----------------------------------------------------\n")

# Ruta del dataset CNA
ruta_cna <- fuentes_raw$ruta[
  fuentes_raw$fuente == "CNA"
]

# Cargar dataset
cna <- sf::st_read(
  ruta_cna,
  quiet = TRUE
)

cat("Filas    :", nrow(cna), "\n")
cat("Columnas :", ncol(cna), "\n\n")

# Estructura
glimpse(cna)

cat("\n")
cat("VALORES FALTANTES\n")
cat("-----------------------------------------------------\n")

print(
  data.frame(
    variable = names(cna),
    tipo = sapply(cna, \(x) class(x)[1]),
    nulos = colSums(is.na(cna)),
    pct_nulos = round(
      100 * colSums(is.na(cna)) / nrow(cna),
      2
    ),
    stringsAsFactors = FALSE
  )
)

cat("\n")
cat("RESUMEN ESTADÍSTICO\n")
cat("-----------------------------------------------------\n")

summary(cna)

cat("\n")
cat("BLOQUE 7. AUDITORÍA DEL DATASET SPATIAL\n")
cat("-----------------------------------------------------\n")

# Ruta del dataset SPATIAL
ruta_spatial <- fuentes_raw$ruta[
  fuentes_raw$fuente == "SPATIAL"
]

# Cargar dataset
spatial <- sf::st_read(
  ruta_spatial,
  quiet = TRUE
)

cat("Filas    :", nrow(spatial), "\n")
cat("Columnas :", ncol(spatial), "\n\n")

# Estructura
glimpse(spatial)

cat("\n")
cat("VALORES FALTANTES\n")
cat("-----------------------------------------------------\n")

print(
  data.frame(
    variable = names(spatial),
    tipo = sapply(spatial, \(x) class(x)[1]),
    nulos = colSums(is.na(spatial)),
    pct_nulos = round(
      100 * colSums(is.na(spatial)) / nrow(spatial),
      2
    ),
    stringsAsFactors = FALSE
  )
)

cat("\n")
cat("RESUMEN ESTADÍSTICO\n")
cat("-----------------------------------------------------\n")

summary(spatial)

cat("\n")
cat("BLOQUE 8. AUDITORÍA DEL DATASET ERA5\n")
cat("-----------------------------------------------------\n")

# Ruta del dataset ERA5
ruta_era5 <- fuentes_raw$ruta[
  fuentes_raw$fuente == "ERA5"
]

# Cargar dataset
era5 <- terra::rast(ruta_era5)

cat("Capas        :", terra::nlyr(era5), "\n")
cat("Filas        :", terra::nrow(era5), "\n")
cat("Columnas     :", terra::ncol(era5), "\n")
cat("Resolución   :", paste(terra::res(era5), collapse = " x "), "\n")
cat("CRS          :", terra::crs(era5), "\n")
cat("Extensión    :", paste(terra::ext(era5)), "\n\n")

# Estructura del raster
print(era5)

cat("\n")
cat("VALORES FALTANTES\n")
cat("-----------------------------------------------------\n")

nulos_era5 <- terra::global(
  is.na(era5),
  fun = "sum",
  na.rm = FALSE
)

nombres_variables <- names(era5)

print(
  data.frame(
    variable = nombres_variables,
    nulos = as.numeric(nulos_era5[1, ]),
    stringsAsFactors = FALSE
  )
)

cat("\n")
cat("RESUMEN ESTADÍSTICO\n")
cat("-----------------------------------------------------\n")

print(
  terra::global(
    era5,
    fun = c(
      "min",
      "max",
      "mean",
      "sd"
    ),
    na.rm = TRUE
  )
)

cat("\n")
cat("BLOQUE 9. AUDITORÍA DE VARIABLES AGRÍCOLAS DEL DATASET EVA\n")
cat("-----------------------------------------------------\n")

# Variables agrícolas de interés
variables_agricolas <- c(
  "GRUPO.DE.CULTIVO",
  "SUBGRUPO.DE.CULTIVO",
  "CULTIVO",
  "DESAGREGACIÓN.REGIONAL.Y.O.SISTEMA.PRODUCTIVO",
  "ÁREA.SEMBRADA..HA.",
  "ÁREA.COSECHADA..HA.",
  "PRODUCCIÓN..T.",
  "RENDIMIENTO..T.HA.",
  "ESTADO.FÍSICO.PRODUCCIÓN",
  "NOMBRE.CIENTÍFICO",
  "CICLO.DE.CULTIVO"
)

# Conservar únicamente las variables existentes
variables_agricolas <- intersect(
  variables_agricolas,
  names(eva)
)

# Subconjunto para auditoría
eva_agricola <- eva |>
  dplyr::select(
    dplyr::all_of(variables_agricolas)
  )

cat("Variables agrícolas:", ncol(eva_agricola), "\n\n")

# Estructura
glimpse(eva_agricola)

cat("\n")
cat("VALORES FALTANTES\n")
cat("-----------------------------------------------------\n")

print(
  data.frame(
    variable = names(eva_agricola),
    tipo = sapply(eva_agricola, \(x) class(x)[1]),
    nulos = colSums(is.na(eva_agricola)),
    pct_nulos = round(
      100 * colSums(is.na(eva_agricola)) / nrow(eva_agricola),
      2
    ),
    stringsAsFactors = FALSE
  )
)

cat("\n")
cat("RESUMEN ESTADÍSTICO\n")
cat("-----------------------------------------------------\n")

summary(eva_agricola)