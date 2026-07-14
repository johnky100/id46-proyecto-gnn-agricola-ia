# 0. config/00_packages.R

# Pregunta: ¿Qué librerías necesita el proyecto GNN agrícola?

paquetes_requeridos <- c(
  "tidyverse", # Manipulación y visualización de datos
  "data.table", # Procesamiento eficiente de grandes volúmenes de datos
  "readxl", # Lectura de archivos Excel
  "writexl", # Exportación de archivos Excel
  "janitor", # Limpieza y estandarización de nombres de variables
  "lubridate", # Manejo de fechas
  "stringi", # Procesamiento avanzado de texto
  "glue", # Construcción dinámica de cadenas de texto
  "fs", # Gestión de archivos y carpetas
  "here", # Gestión reproducible de rutas
  "mice", # Imputación múltiple de datos faltantes
  "arrow", # Lectura y escritura de archivos Parquet
  "sf", # Manejo de datos espaciales vectoriales
  "terra", # Manejo de datos raster y NetCDF
  "exactextractr", # Extracción de estadísticas raster por polígonos
  "spdep", # Construcción de vecindades espaciales
  "future", # Procesamiento paralelo
  "furrr" # Paralelización de flujos purrr
) # Dependencias oficiales del proyecto

# Pregunta: ¿Qué paquetes ya están instalados en el sistema?

paquetes_instalados <- rownames(installed.packages()) # Obtener paquetes instalados

# Pregunta: ¿Qué paquetes faltan para ejecutar el proyecto?

paquetes_faltantes <- setdiff(paquetes_requeridos, paquetes_instalados) # Identificar paquetes faltantes

# Pregunta: ¿Cómo garantizar que todas las dependencias estén disponibles?

if (length(paquetes_faltantes) > 0) { # Verificar si existen paquetes faltantes
  install.packages(paquetes_faltantes, dependencies = TRUE) # Instalar dependencias faltantes
}

# Pregunta: ¿Cómo cargar todas las librerías requeridas?

invisible(lapply(paquetes_requeridos, library, character.only = TRUE)) # Cargar librerías oficiales

# Pregunta: ¿Cómo configurar el entorno de trabajo de R?

options(repos = c(CRAN = "https://cloud.r-project.org")) # Definir repositorio CRAN oficial
options(scipen = 999) # Evitar notación científica
options(stringsAsFactors = FALSE) # Evitar conversión automática a factor
options(dplyr.summarise.inform = FALSE) # Ocultar mensajes de summarise
options(tibble.width = Inf) # Mostrar todas las columnas
options(tibble.print_max = 50) # Limitar filas impresas

# Pregunta: ¿Cómo garantizar resultados reproducibles?

seed_global <- 5477976 # Semilla oficial del proyecto
set.seed(seed_global) # Fijar semilla global

# Pregunta: ¿Existen conflictos frecuentes entre librerías?

conflictos <- conflicts()[grepl("filter|lag|select|extract|intersect|union", conflicts())] # Detectar conflictos frecuentes

# Pregunta: ¿Cuál es el estado final del entorno computacional?

cat("\nCONFIGURACIÓN DEL ENTORNO COMPLETADA\n") # Mostrar encabezado
cat("Paquetes requeridos:", length(paquetes_requeridos), "\n") # Mostrar total de paquetes
cat("Paquetes instalados automáticamente:", length(paquetes_faltantes), "\n") # Mostrar paquetes instalados
cat("Semilla global:", seed_global, "\n") # Mostrar semilla utilizada
cat("Versión de R:", R.version.string, "\n") # Mostrar versión de R
cat("Sistema operativo:", Sys.info()["sysname"], "\n") # Mostrar sistema operativo
cat("Directorio de trabajo:", getwd(), "\n") # Mostrar directorio actual

# Pregunta: ¿Qué conflictos fueron detectados durante la carga de librerías?

if (length(conflictos) > 0) { # Verificar si existen conflictos
  cat("\nConflictos detectados:\n") # Mostrar encabezado de conflictos
  print(conflictos) # Mostrar conflictos encontrados
}

cat("\nPaquetes cargados correctamente\n") # Confirmar carga exitosa
