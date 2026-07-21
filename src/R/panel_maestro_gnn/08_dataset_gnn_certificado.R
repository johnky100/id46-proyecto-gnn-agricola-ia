# 08_dataset_gnn_certificado.R

# BLOQUE 0. Configuración del Entorno --------------------------------------
## Objetivo: Cargar las librerías, definir la semilla y configurar el entorno
# de trabajo para la certificación del Dataset GNN.
### Producto:
# - Entorno de ejecución configurado.
### Responde:
# ¿El entorno de ejecución fue configurado correctamente?

# 0.1. Cargar librerías ----------------------------------------------------
## Objetivo: Cargar las librerías necesarias.

library(arrow) # Lectura y escritura de archivos Parquet
library(dplyr) # Manipulación de datos
library(tidyr) # Transformación de datos
library(tibble) # Construcción de tablas
library(here) # Gestión de rutas del proyecto
library(sf) # Manejo de información espacial

# 0.2. Configuración -------------------------------------------------------
## Objetivo: Definir la semilla y configurar el entorno.

seed_global <- 5477976 # Semilla oficial del proyecto

set.seed(seed_global) # Inicializar semilla

options(
  scipen = 999,
  dplyr.summarise.inform = FALSE
) # Configuración global

# 0.3. Validación del proyecto ---------------------------------------------
## Objetivo: Verificar la estructura mínima del proyecto.

if (!dir.exists(here("data"))) {
  
  stop(
    "No existe la carpeta 'data' del proyecto."
  )
  
}

# 0.4. Información del entorno ---------------------------------------------
## Objetivo: Mostrar la información del entorno.

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 0. CONFIGURACIÓN DEL ENTORNO\n")
cat(strrep("=", 90), "\n")

cat(sprintf("R                  : %s\n", R.version.string))
cat(sprintf("Semilla            : %s\n", seed_global))
cat(sprintf("Proyecto           : %s\n", here()))
cat(sprintf("Directorio         : %s\n", normalizePath(getwd())))
cat(sprintf("Fecha              : %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))

cat("\nBloque 0 finalizado correctamente.\n")

# BLOQUE 1. Carga del Dataset Maestro --------------------------------------
## Objetivo: Cargar el Dataset Maestro certificado que servirá como fuente
# oficial para generar los datasets de modelado y dashboard.
### Producto:
# - dataset_gnn_maestro
### Responde:
# ¿El Dataset Maestro fue cargado correctamente y está disponible para iniciar
# la certificación final?

# 1.1. Ruta oficial ---------------------------------------------------------
## Objetivo: Definir la ruta oficial del Dataset Maestro.

ruta_dataset <- here(
  "data",
  "processed",
  "r",
  "master",
  "dataset_gnn_certificado.parquet"
) # Ruta oficial del Dataset Maestro

# 1.2. Verificar existencia -------------------------------------------------
## Objetivo: Verificar que el Dataset Maestro existe.

if (!file.exists(ruta_dataset)) {
  
  stop(
    "No existe el Dataset Maestro en la ruta especificada."
  )
  
}

cat("Dataset localizado correctamente.\n")

# 1.3. Cargar Dataset -------------------------------------------------------
## Objetivo: Cargar el Dataset Maestro.

dataset_gnn_maestro <- arrow::read_parquet(
  ruta_dataset
) # Dataset Maestro certificado

cat("Dataset cargado correctamente.\n")

# 1.4. Validación -----------------------------------------------------------
## Objetivo: Verificar la integridad del Dataset Maestro.

if (nrow(dataset_gnn_maestro) == 0) {
  
  stop(
    "El Dataset Maestro no contiene observaciones."
  )
  
}

if (ncol(dataset_gnn_maestro) == 0) {
  
  stop(
    "El Dataset Maestro no contiene variables."
  )
  
}

# 1.5. Resumen --------------------------------------------------------------
## Objetivo: Mostrar la información general del Dataset Maestro.

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 1. CARGA DEL DATASET MAESTRO\n")
cat(strrep("=", 90), "\n")

cat(sprintf("Archivo                : %s\n", basename(ruta_dataset)))
cat(sprintf("Ruta                   : %s\n", dirname(ruta_dataset)))
cat(sprintf("Observaciones          : %s\n", format(nrow(dataset_gnn_maestro), big.mark = ",")))
cat(sprintf("Variables              : %s\n", format(ncol(dataset_gnn_maestro), big.mark = ",")))
cat(sprintf("Memoria                : %s\n", format(object.size(dataset_gnn_maestro), units = "MB")))

cat("\nBloque 1 finalizado correctamente.\n")

# BLOQUE 2. Auditoría Estructural del Dataset Maestro -----------------------
## Objetivo: Verificar la estructura general del Dataset Maestro antes de
# construir los datasets derivados para modelado y visualización.
### Entradas:
# - dataset_gnn_maestro
### Producto:
# - Diagnóstico estructural.
### Responde:
# ¿El Dataset Maestro presenta una estructura consistente para continuar con
# el proceso de certificación?

# 2.1. Variables obligatorias ----------------------------------------------
## Objetivo: Verificar la existencia de las variables esenciales.

variables_obligatorias <- c(
  "cod_mpio",
  "municipio",
  "anio",
  "panel_id",
  "log_rendimiento",
  "geometry"
) # Variables obligatorias

variables_faltantes <- setdiff(
  variables_obligatorias,
  names(dataset_gnn_maestro)
) # Variables faltantes

if (length(variables_faltantes) > 0) {
  
  stop(
    paste(
      "Faltan variables obligatorias:",
      paste(variables_faltantes, collapse = ", ")
    )
  )
  
}

# 2.2. Dimensiones ---------------------------------------------------------
## Objetivo: Obtener las dimensiones generales.

n_observaciones <- nrow(dataset_gnn_maestro) # Observaciones

n_variables <- ncol(dataset_gnn_maestro) # Variables

# 2.3. Municipios ----------------------------------------------------------
## Objetivo: Contar municipios únicos.

n_municipios <- dplyr::n_distinct(
  dataset_gnn_maestro$cod_mpio
) # Municipios

# 2.4. Años ----------------------------------------------------------------
## Objetivo: Contar años disponibles.

n_anios <- dplyr::n_distinct(
  dataset_gnn_maestro$anio
) # Años

# 2.5. Panel ---------------------------------------------------------------
## Objetivo: Verificar la unicidad del panel.

panel_duplicado <- dataset_gnn_maestro |>
  dplyr::count(
    cod_mpio,
    anio
  ) |>
  dplyr::filter(
    n > 1
  ) # Duplicados municipio-año

# 2.6. Registros duplicados ------------------------------------------------
## Objetivo: Verificar registros completamente duplicados.

registros_duplicados <- sum(
  duplicated(dataset_gnn_maestro)
) # Registros duplicados

# 2.7. Variable objetivo ---------------------------------------------------
## Objetivo: Verificar disponibilidad de la variable objetivo.

target_na <- sum(
  is.na(dataset_gnn_maestro$log_rendimiento)
) # Valores NA del target

# 2.8. Geometría -----------------------------------------------------------
## Objetivo: Verificar disponibilidad de la geometría.

geometry_na <- sum(
  is.na(dataset_gnn_maestro$geometry)
) # Geometrías faltantes

# 2.9. Resultados ----------------------------------------------------------
## Objetivo: Mostrar el resumen estructural.

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 2. AUDITORÍA ESTRUCTURAL\n")
cat(strrep("=", 90), "\n")

cat(sprintf("Observaciones               : %s\n", format(n_observaciones, big.mark = ",")))
cat(sprintf("Variables                   : %s\n", format(n_variables, big.mark = ",")))
cat(sprintf("Municipios                  : %s\n", format(n_municipios, big.mark = ",")))
cat(sprintf("Años                        : %s\n", format(n_anios, big.mark = ",")))
cat(sprintf("Duplicados municipio-año    : %s\n", nrow(panel_duplicado)))
cat(sprintf("Registros duplicados        : %s\n", registros_duplicados))
cat(sprintf("NA variable objetivo        : %s\n", target_na))
cat(sprintf("NA geometría                : %s\n", geometry_na))

# 2.10. Dictamen -----------------------------------------------------------
## Objetivo: Emitir el dictamen estructural.

cat("\n")
cat(strrep("-", 90), "\n")
cat("DICTAMEN\n")
cat(strrep("-", 90), "\n")

if (
  nrow(panel_duplicado) == 0 &&
  registros_duplicados == 0 &&
  target_na == 0 &&
  geometry_na == 0
) {
  
  cat("El Dataset Maestro supera la auditoría estructural.\n")
  
} else {
  
  cat("El Dataset Maestro requiere revisión antes de continuar.\n")
  
}

cat("\nBloque 2 finalizado correctamente.\n")

# BLOQUE 3. Auditoría de Calidad del Dataset Maestro -----------------------
## Objetivo: Verificar la calidad general del Dataset Maestro antes de
# generar los datasets derivados.
### Entradas:
# - dataset_gnn_maestro
### Producto:
# - Diagnóstico de calidad.
### Responde:
# ¿El Dataset Maestro presenta una calidad adecuada para construir los
# datasets de modelado y dashboard?

# 3.1. Valores faltantes ---------------------------------------------------
## Objetivo: Cuantificar los valores faltantes.

auditoria_na <- tibble(
  
  variable = names(dataset_gnn_maestro),
  
  n_na = sapply(
    dataset_gnn_maestro,
    function(x) sum(is.na(x))
  )
  
) |>
  
  mutate(
    
    pct_na = round(
      100 * n_na / nrow(dataset_gnn_maestro),
      4
    )
    
  ) |>
  
  arrange(
    desc(n_na)
  ) # Auditoría de valores faltantes

total_na <- sum(auditoria_na$n_na) # Total de valores NA

pct_na_global <- round(
  100 * total_na /
    (nrow(dataset_gnn_maestro) * ncol(dataset_gnn_maestro)),
  4
) # Porcentaje global de NA

# 3.2. Valores NaN ---------------------------------------------------------
## Objetivo: Cuantificar valores NaN.

variables_numericas <- dataset_gnn_maestro |>
  dplyr::select(
    where(is.numeric)
  ) # Variables numéricas

nan_total <- sum(
  
  sapply(
    
    variables_numericas,
    
    function(x) sum(is.nan(x))
    
  )
  
) # Total de valores NaN

# 3.3. Valores infinitos ---------------------------------------------------
## Objetivo: Cuantificar valores infinitos.

inf_total <- sum(
  
  sapply(
    
    variables_numericas,
    
    function(x) sum(is.infinite(x))
    
  )
  
) # Total de valores infinitos

# 3.4. Registros duplicados ------------------------------------------------
## Objetivo: Detectar registros duplicados.

duplicados <- sum(
  
  duplicated(
    
    dataset_gnn_maestro |>
      
      dplyr::select(
        
        -any_of("geometry")
        
      )
    
  )
  
) # Registros duplicados

# 3.5. Variables constantes ------------------------------------------------
## Objetivo: Detectar variables sin variabilidad.

variables_constantes <- names(variables_numericas)[
  
  sapply(
    
    variables_numericas,
    
    function(x)
      dplyr::n_distinct(
        x,
        na.rm = TRUE
      ) <= 1
    
  )
  
] # Variables constantes

# 3.6. Resumen -------------------------------------------------------------
## Objetivo: Mostrar el resumen de calidad.

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 3. AUDITORÍA DE CALIDAD\n")
cat(strrep("=", 90), "\n")

cat(sprintf("Variables con NA            : %s\n", sum(auditoria_na$n_na > 0)))
cat(sprintf("Total valores NA            : %s\n", format(total_na, big.mark = ",")))
cat(sprintf("Porcentaje global NA        : %.4f%%\n", pct_na_global))
cat(sprintf("Total valores NaN           : %s\n", nan_total))
cat(sprintf("Total valores Inf           : %s\n", inf_total))
cat(sprintf("Variables constantes        : %s\n", length(variables_constantes)))
cat(sprintf("Registros duplicados        : %s\n", duplicados))

# 3.7. Variables con NA ----------------------------------------------------
## Objetivo: Mostrar las variables con valores faltantes.

if (total_na > 0) {
  
  cat("\n")
  cat(strrep("-", 90), "\n")
  cat("VARIABLES CON VALORES FALTANTES\n")
  cat(strrep("-", 90), "\n")
  
  print(
    
    auditoria_na |>
      
      dplyr::filter(
        
        n_na > 0
        
      )
    
  )
  
}

# 3.8. Dictamen ------------------------------------------------------------
## Objetivo: Emitir el dictamen de calidad.

cat("\n")
cat(strrep("-", 90), "\n")
cat("DICTAMEN\n")
cat(strrep("-", 90), "\n")

if (
  
  total_na == 0 &&
  nan_total == 0 &&
  inf_total == 0 &&
  duplicados == 0 &&
  length(variables_constantes) == 0
  
) {
  
  cat("El Dataset Maestro supera la auditoría de calidad.\n")
  
} else {
  
  cat("El Dataset Maestro requiere revisión antes de generar los datasets derivados.\n")
  
}

cat("\nBloque 3 finalizado correctamente.\n")

# BLOQUE 4. Auditoría Espacial del Dataset Maestro -------------------------
## Objetivo: Verificar la integridad de la información espacial contenida en
# el Dataset Maestro antes de generar los datasets derivados.
### Entradas:
# - dataset_gnn_maestro
### Producto:
# - Diagnóstico espacial.
### Responde:
# ¿La información espacial del Dataset Maestro es consistente para construir
# el grafo y el dashboard?

# 4.1. Variables espaciales ------------------------------------------------
## Objetivo: Verificar la existencia de las variables espaciales.

variables_espaciales <- c(
  "geometry",
  "latitud",
  "longitud"
) # Variables espaciales

variables_faltantes <- setdiff(
  variables_espaciales,
  names(dataset_gnn_maestro)
) # Variables faltantes

if (length(variables_faltantes) > 0) {
  
  stop(
    paste(
      "Faltan variables espaciales:",
      paste(variables_faltantes, collapse = ", ")
    )
  )
  
}

# 4.2. Valores faltantes ---------------------------------------------------
## Objetivo: Verificar valores faltantes.

na_geometry <- sum(is.na(dataset_gnn_maestro$geometry)) # Geometrías faltantes

na_latitud <- sum(is.na(dataset_gnn_maestro$latitud)) # Latitudes faltantes

na_longitud <- sum(is.na(dataset_gnn_maestro$longitud)) # Longitudes faltantes

# 4.3. Coordenadas válidas -------------------------------------------------
## Objetivo: Verificar que las coordenadas pertenecen al territorio colombiano.

latitud_valida <- all(
  dataset_gnn_maestro$latitud >= -4.3 &
    dataset_gnn_maestro$latitud <= 16.0,
  na.rm = TRUE
) # Latitudes válidas

longitud_valida <- all(
  dataset_gnn_maestro$longitud >= -81.9 &
    dataset_gnn_maestro$longitud <= -66.8,
  na.rm = TRUE
) # Longitudes válidas

# 4.4. Geometrías válidas --------------------------------------------------
## Objetivo: Verificar la validez topológica de las geometrías.

geometrias_validas <- all(
  sf::st_is_valid(dataset_gnn_maestro$geometry)
) # Geometrías válidas

# 4.5. Resumen -------------------------------------------------------------
## Objetivo: Mostrar el resumen espacial.

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 4. AUDITORÍA ESPACIAL\n")
cat(strrep("=", 90), "\n")

cat(sprintf("Geometrías faltantes      : %s\n", na_geometry))
cat(sprintf("Latitudes faltantes       : %s\n", na_latitud))
cat(sprintf("Longitudes faltantes      : %s\n", na_longitud))
cat(sprintf("Latitudes válidas         : %s\n", ifelse(latitud_valida, "SI", "NO")))
cat(sprintf("Longitudes válidas        : %s\n", ifelse(longitud_valida, "SI", "NO")))
cat(sprintf("Geometrías válidas        : %s\n", ifelse(geometrias_validas, "SI", "NO")))

# 4.6. Dictamen ------------------------------------------------------------
## Objetivo: Emitir el dictamen espacial.

cat("\n")
cat(strrep("-", 90), "\n")
cat("DICTAMEN\n")
cat(strrep("-", 90), "\n")

if (
  na_geometry == 0 &&
  na_latitud == 0 &&
  na_longitud == 0 &&
  latitud_valida &&
  longitud_valida &&
  geometrias_validas
) {
  
  cat("El Dataset Maestro supera la auditoría espacial.\n")
  
} else {
  
  cat("El Dataset Maestro requiere revisión de la información espacial.\n")
  
}

cat("\nBloque 4 finalizado correctamente.\n")

# BLOQUE 5. Certificación del Dataset Oficial ------------------------------
## Objetivo: Certificar el Dataset Maestro como el Dataset Oficial del
# proyecto para modelado, Graph Neural Networks y dashboard.
### Entradas:
# - dataset_gnn_maestro
### Producto:
# - dataset_gnn_certificado
### Responde:
# ¿El Dataset Oficial fue construido correctamente?

# 5.1. Dataset Oficial -----------------------------------------------------
## Objetivo: Definir el Dataset Oficial del proyecto.

dataset_gnn_certificado <- dataset_gnn_maestro # Dataset oficial

# 5.2. Validación ----------------------------------------------------------
## Objetivo: Verificar la estructura del Dataset Oficial.

stopifnot(
  nrow(dataset_gnn_certificado) ==
    nrow(dataset_gnn_maestro)
) # Validar observaciones

stopifnot(
  ncol(dataset_gnn_certificado) ==
    ncol(dataset_gnn_maestro)
) # Validar variables

# 5.3. Resumen -------------------------------------------------------------
## Objetivo: Mostrar la estructura del Dataset Oficial.

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 5. DATASET OFICIAL\n")
cat(strrep("=", 90), "\n")

cat(sprintf("Observaciones          : %s\n", format(nrow(dataset_gnn_certificado), big.mark = ",")))
cat(sprintf("Variables              : %s\n", format(ncol(dataset_gnn_certificado), big.mark = ",")))
cat(sprintf("Variable geometry      : %s\n", ifelse("geometry" %in% names(dataset_gnn_certificado), "SI", "NO")))

# 5.4. Dictamen ------------------------------------------------------------
## Objetivo: Confirmar la certificación del Dataset Oficial.

cat("\n")
cat(strrep("-", 90), "\n")
cat("DICTAMEN\n")
cat(strrep("-", 90), "\n")

cat("Dataset Oficial : Certificado correctamente.\n")

cat("\nBloque 5 finalizado correctamente.\n")

# BLOQUE 6. Certificación del Dataset Oficial ------------------------------
## Objetivo: Verificar que el Dataset Oficial cumple las especificaciones
# establecidas para el proyecto.
### Entradas:
# - dataset_gnn_maestro
# - dataset_gnn_certificado
### Producto:
# - Certificación del Dataset Oficial.
### Responde:
# ¿El Dataset Oficial cumple las especificaciones del proyecto?

# 6.1. Dimensiones ---------------------------------------------------------
## Objetivo: Verificar la consistencia estructural.

stopifnot(
  nrow(dataset_gnn_maestro) ==
    nrow(dataset_gnn_certificado)
) # Validar observaciones

stopifnot(
  ncol(dataset_gnn_maestro) ==
    ncol(dataset_gnn_certificado)
) # Validar variables

# 6.2. Variable objetivo ---------------------------------------------------
## Objetivo: Verificar la existencia de la variable objetivo.

stopifnot(
  "log_rendimiento" %in%
    names(dataset_gnn_certificado)
) # Variable objetivo

# 6.3. Geometría -----------------------------------------------------------
## Objetivo: Verificar la disponibilidad de la geometría.

stopifnot(
  "geometry" %in%
    names(dataset_gnn_certificado)
) # Variable geometry

# 6.4. Identificadores -----------------------------------------------------
## Objetivo: Verificar los identificadores principales.

variables_id <- c(
  "cod_mpio",
  "municipio",
  "anio",
  "panel_id"
) # Variables identificadoras

stopifnot(
  all(
    variables_id %in%
      names(dataset_gnn_certificado)
  )
) # Identificadores

# 6.5. Resumen -------------------------------------------------------------
## Objetivo: Mostrar el resumen de certificación.

certificacion <- tibble(
  
  Dataset = "Dataset Oficial",
  
  Observaciones = nrow(dataset_gnn_certificado),
  
  Variables = ncol(dataset_gnn_certificado),
  
  Geometry = "SI",
  
  Estado = "APROBADO"
  
) # Resumen de certificación

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 6. CERTIFICACIÓN DEL DATASET OFICIAL\n")
cat(strrep("=", 90), "\n")

print(certificacion)

# 6.6. Dictamen ------------------------------------------------------------
## Objetivo: Confirmar la certificación.

cat("\n")
cat(strrep("-", 90), "\n")
cat("DICTAMEN\n")
cat(strrep("-", 90), "\n")

cat("Dataset Oficial : APROBADO\n")

cat("\nBloque 6 finalizado correctamente.\n")

# BLOQUE 7. Exportación del Dataset Oficial -------------------------------
## Objetivo: Exportar el Dataset Oficial certificado del proyecto.
### Entradas:
# - dataset_gnn_certificado
### Producto:
# - dataset_gnn_certificado.parquet
### Responde:
# ¿El Dataset Oficial fue exportado correctamente?

# 7.1. Directorio de salida ------------------------------------------------
## Objetivo: Definir el directorio oficial de exportación.

ruta_salida <- here(
  "data",
  "processed",
  "r",
  "master"
) # Directorio oficial

dir.create(ruta_salida, recursive = TRUE, showWarnings = FALSE) # Crear directorio

# 7.2. Exportación ---------------------------------------------------------
## Objetivo: Exportar el Dataset Oficial.

arrow::write_parquet(
  dataset_gnn_certificado,
  file.path(
    ruta_salida,
    "dataset_gnn_certificado.parquet"
  )
) # Exportar Dataset Oficial

# 7.3. Verificación --------------------------------------------------------
## Objetivo: Verificar que el archivo fue creado.

archivo <- "dataset_gnn_certificado.parquet" # Archivo esperado

verificacion <- tibble(
  
  archivo = archivo,
  
  existe = file.exists(
    file.path(
      ruta_salida,
      archivo
    )
  )
  
) # Verificación

# 7.4. Tamaño del archivo --------------------------------------------------
## Objetivo: Registrar el tamaño del Dataset Oficial.

verificacion <- verificacion |>
  
  mutate(
    
    tamano_mb = round(
      
      file.info(
        
        file.path(
          ruta_salida,
          archivo
        )
        
      )$size / 1024^2,
      
      2
      
    )
    
  )

# 7.5. Resultados ----------------------------------------------------------
## Objetivo: Mostrar la exportación realizada.

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 7. EXPORTACIÓN DEL DATASET OFICIAL\n")
cat(strrep("=", 90), "\n")

print(verificacion)

# 7.6. Dictamen ------------------------------------------------------------
## Objetivo: Confirmar la exportación.

cat("\n")
cat(strrep("-", 90), "\n")
cat("DICTAMEN\n")
cat(strrep("-", 90), "\n")

if (all(verificacion$existe)) {
  
  cat("El Dataset Oficial fue exportado correctamente.\n")
  
} else {
  
  stop(
    "No fue posible exportar el Dataset Oficial."
  )
  
}

cat("\nBloque 7 finalizado correctamente.\n")

# BLOQUE 8. Informe Oficial de Certificación -------------------------------
## Objetivo: Generar el informe oficial de certificación del Dataset Oficial
# del proyecto.
### Entradas:
# - dataset_gnn_certificado
### Producto:
# - informe_certificacion.csv
# - informe_certificacion.parquet
### Responde:
# ¿El Dataset Oficial cumple los requisitos para iniciar el Benchmark
# Científico, Graph Neural Networks y el Dashboard?

# 8.1. Construcción del informe --------------------------------------------
## Objetivo: Consolidar la información de certificación.

informe_certificacion <- tibble(
  
  fecha = Sys.time(),
  
  dataset = "Dataset Oficial",
  
  observaciones = nrow(dataset_gnn_certificado),
  
  variables = ncol(dataset_gnn_certificado),
  
  geometry = ifelse(
    "geometry" %in% names(dataset_gnn_certificado),
    "SI",
    "NO"
  ),
  
  variable_objetivo = ifelse(
    "log_rendimiento" %in% names(dataset_gnn_certificado),
    "SI",
    "NO"
  ),
  
  estado = "APROBADO"
  
) # Informe oficial

# 8.2. Mostrar informe -----------------------------------------------------
## Objetivo: Presentar el resumen de certificación.

cat("\n")
cat(strrep("=", 90), "\n")
cat("INFORME OFICIAL DE CERTIFICACIÓN\n")
cat(strrep("=", 90), "\n")

print(informe_certificacion)

# 8.3. Exportación ---------------------------------------------------------
## Objetivo: Exportar el informe oficial.

arrow::write_parquet(
  informe_certificacion,
  file.path(
    ruta_salida,
    "informe_certificacion.parquet"
  )
) # Exportar informe Parquet

write.csv(
  informe_certificacion,
  file.path(
    ruta_salida,
    "informe_certificacion.csv"
  ),
  row.names = FALSE
) # Exportar informe CSV

# 8.4. Mensaje final -------------------------------------------------------
## Objetivo: Confirmar la finalización del proceso.

cat("\n")
cat(strrep("=", 90), "\n")
cat("CERTIFICACIÓN FINALIZADA CORRECTAMENTE\n")
cat(strrep("=", 90), "\n")

cat("Dataset Oficial : APROBADO\n")

cat("\nArchivos generados:\n")

cat(sprintf("%s\n", file.path(ruta_salida, "dataset_gnn_certificado.parquet")))
cat(sprintf("%s\n", file.path(ruta_salida, "informe_certificacion.parquet")))
cat(sprintf("%s\n", file.path(ruta_salida, "informe_certificacion.csv")))

cat("\nProceso finalizado correctamente.\n")

cat("\nBloque 8 finalizado correctamente.\n")

# BLOQUE 9. Certificación de Reproducibilidad ------------------------------
## Objetivo: Registrar la información necesaria para garantizar la
# reproducibilidad del Dataset Oficial del proyecto.
### Entradas:
# - dataset_gnn_certificado
### Producto:
# - metadata_certificacion.csv
# - metadata_certificacion.parquet
### Responde:
# ¿El proceso de certificación es completamente reproducible?

# 9.1. Información del entorno ---------------------------------------------
## Objetivo: Registrar la información del entorno de ejecución.

metadata <- tibble(
  
  fecha = Sys.time(),
  
  version_r = R.version.string,
  
  sistema_operativo = Sys.info()["sysname"],
  
  usuario = Sys.info()["user"],
  
  directorio = normalizePath(getwd()),
  
  semilla = seed_global
  
) # Metadata del entorno

# 9.2. Información del Dataset Oficial -------------------------------------
## Objetivo: Registrar la estructura del Dataset Oficial.

metadata_dataset <- tibble(
  
  dataset = "Dataset Oficial",
  
  archivo = "dataset_gnn_certificado.parquet",
  
  observaciones = nrow(dataset_gnn_certificado),
  
  variables = ncol(dataset_gnn_certificado),
  
  geometry = ifelse(
    "geometry" %in% names(dataset_gnn_certificado),
    "SI",
    "NO"
  ),
  
  variable_objetivo = "log_rendimiento",
  
  tamano_mb = round(
    
    file.info(
      
      file.path(
        ruta_salida,
        "dataset_gnn_certificado.parquet"
      )
      
    )$size / 1024^2,
    
    2
    
  )
  
) # Metadata del Dataset Oficial

# 9.3. Mostrar resultados --------------------------------------------------
## Objetivo: Mostrar la información registrada.

cat("\n")
cat(strrep("=", 90), "\n")
cat("CERTIFICACIÓN DE REPRODUCIBILIDAD\n")
cat(strrep("=", 90), "\n")

print(metadata)

cat("\n")

print(metadata_dataset)

# 9.4. Exportación ---------------------------------------------------------
## Objetivo: Exportar la metadata.

arrow::write_parquet(
  metadata,
  file.path(
    ruta_salida,
    "metadata_certificacion.parquet"
  )
) # Exportar metadata

write.csv(
  metadata,
  file.path(
    ruta_salida,
    "metadata_certificacion.csv"
  ),
  row.names = FALSE
) # Exportar metadata

arrow::write_parquet(
  metadata_dataset,
  file.path(
    ruta_salida,
    "metadata_dataset.parquet"
  )
) # Exportar metadata del dataset

write.csv(
  metadata_dataset,
  file.path(
    ruta_salida,
    "metadata_dataset.csv"
  ),
  row.names = FALSE
) # Exportar metadata del dataset

# 9.5. Finalización --------------------------------------------------------
## Objetivo: Confirmar la finalización del proceso.

cat("\n")
cat(strrep("=", 90), "\n")
cat("CERTIFICACIÓN COMPLETADA\n")
cat(strrep("=", 90), "\n")

cat("Dataset Oficial certificado correctamente.\n")
cat("La metadata de reproducibilidad fue registrada.\n")
cat("El Dataset Oficial constituye la fuente oficial para el Benchmark, Graph Neural Networks y Dashboard.\n")

cat("\nBloque 9 finalizado correctamente.\n")

# BLOQUE 10. Validación Final del Proceso ----------------------------------
## Objetivo: Verificar que todos los productos oficiales fueron generados
# correctamente y que el proceso de certificación finalizó sin errores.
### Entradas:
# - ruta_salida
### Producto:
# - Validación final del proceso.
### Responde:
# ¿El proceso de certificación finalizó correctamente y todos los productos
# oficiales están disponibles?

# 10.1. Archivos oficiales -------------------------------------------------
## Objetivo: Definir los productos oficiales esperados.

archivos_oficiales <- c(
  "dataset_gnn_certificado.parquet",
  "informe_certificacion.parquet",
  "informe_certificacion.csv",
  "metadata_certificacion.parquet",
  "metadata_certificacion.csv",
  "metadata_dataset.parquet",
  "metadata_dataset.csv"
) # Productos oficiales

# 10.2. Verificación -------------------------------------------------------
## Objetivo: Verificar la existencia de los productos oficiales.

validacion_final <- tibble(
  
  archivo = archivos_oficiales,
  
  existe = file.exists(
    file.path(
      ruta_salida,
      archivos_oficiales
    )
  )
  
) |>
  
  mutate(
    
    tamano_mb = round(
      
      file.info(
        file.path(
          ruta_salida,
          archivo
        )
      )$size / 1024^2,
      
      2
      
    )
    
  ) # Validación de archivos

# 10.3. Resultados ---------------------------------------------------------
## Objetivo: Mostrar los productos generados.

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 10. VALIDACIÓN FINAL DEL PROCESO\n")
cat(strrep("=", 90), "\n")

print(validacion_final)

# 10.4. Dictamen -----------------------------------------------------------
## Objetivo: Emitir el dictamen final del proceso.

cat("\n")
cat(strrep("-", 90), "\n")
cat("DICTAMEN FINAL\n")
cat(strrep("-", 90), "\n")

if (all(validacion_final$existe)) {
  
  cat("El Dataset Oficial fue certificado correctamente.\n")
  cat("Todos los productos oficiales fueron generados.\n")
  cat("El proyecto está listo para iniciar el Benchmark Científico.\n")
  
} else {
  
  stop(
    "No fue posible validar todos los productos oficiales."
  )
  
}

cat("\nBloque 10 finalizado correctamente.\n")