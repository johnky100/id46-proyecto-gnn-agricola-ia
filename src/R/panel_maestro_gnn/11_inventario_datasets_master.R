# 11_inventario_datasets_master.R

# BLOQUE 0. Configuración del Entorno --------------------------------------
## Objetivo: Cargar las librerías, definir la semilla y configurar el entorno
# de trabajo para la auditoría de los datasets oficiales del proyecto.
### Producto:
# - Entorno de ejecución configurado.
### Responde:
# ¿El entorno de ejecución fue configurado correctamente?

# --------------------------------------------
rm(list = ls()) # Eliminar todos los objetos
gc() # Liberar memoria
# --------------------------------------------

# 0.1. Cargar librerías ----------------------------------------------------
## Objetivo: Cargar las librerías necesarias.

library(arrow) # Lectura y escritura de archivos Parquet
library(dplyr) # Manipulación de datos
library(tidyr) # Transformación de datos
library(tibble) # Construcción de tablas
library(purrr) # Programación funcional
library(stringr) # Manipulación de texto
library(openxlsx) # Exportación a Excel
library(here) # Gestión de rutas del proyecto

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

cat("BLOQUE 0. CONFIGURACIÓN DEL ENTORNO\n")
cat(strrep("-", 50), "\n")

cat(sprintf("R                  : %s\n", R.version.string))
cat(sprintf("Semilla            : %s\n", seed_global))
cat(sprintf("Proyecto           : %s\n", here()))
cat(sprintf("Directorio         : %s\n", normalizePath(getwd())))
cat(sprintf("Fecha              : %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))

cat("\nBloque 0 finalizado correctamente.\n")

# BLOQUE 1. Carga de los Datasets ------------------------------------------
## Objetivo: Localizar, validar y cargar los datasets oficiales del proyecto.
### Producto:
# - Datasets oficiales cargados.
# - Lista datasets_master.
### Responde:
# ¿Los datasets oficiales fueron cargados correctamente?

# 1.1. Directorio ----------------------------------------------------------
## Objetivo: Definir el directorio que contiene los datasets oficiales.

ruta_master <- here::here(
  "data",
  "processed",
  "r",
  "master"
) # Directorio master

# 1.2. Carga de dataset_gnn ------------------------------------------------
## Objetivo: Cargar el dataset final para Benchmark y GNN.

dataset_gnn <- arrow::read_parquet(
  file.path(
    ruta_master,
    "dataset_gnn.parquet"
  )
) # Dataset final

# 1.3. Carga de panel_clean -----------------------------------------------
## Objetivo: Cargar el dataset posterior al proceso ETL.

panel_clean <- arrow::read_parquet(
  file.path(
    ruta_master,
    "panel_clean.parquet"
  )
) # Dataset limpio

# 1.4. Carga de panel_master ----------------------------------------------
## Objetivo: Cargar el dataset maestro del proyecto.

panel_master <- arrow::read_parquet(
  file.path(
    ruta_master,
    "panel_master.parquet"
  )
) # Dataset maestro

# 1.5. Carga de panel_modelado --------------------------------------------
## Objetivo: Cargar el dataset oficial para modelado.

panel_modelado <- arrow::read_parquet(
  file.path(
    ruta_master,
    "panel_modelado.parquet"
  )
) # Dataset de modelado

# 1.6. Registro de los datasets -------------------------------------------
## Objetivo: Consolidar los datasets oficiales en una lista.

datasets_master <- list(
  panel_master = panel_master,
  panel_clean = panel_clean,
  panel_modelado = panel_modelado,
  dataset_gnn = dataset_gnn
) # Datasets oficiales

# 1.7. Validación ----------------------------------------------------------
## Objetivo: Verificar que todos los datasets fueron cargados correctamente.

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 1. CARGA DE LOS DATASETS\n")
cat(strrep("=", 90), "\n")

cat(sprintf(
  "Directorio               : %s\n",
  ruta_master
))

cat(sprintf(
  "Datasets cargados        : %s\n",
  length(datasets_master)
))

cat("\nDatasets oficiales:\n")

cat(
  paste0(
    " - ",
    names(datasets_master),
    collapse = "\n"
  ),
  "\n"
)

if (all(purrr::map_lgl(datasets_master, ~ nrow(.x) > 0))) {
  cat("\nTodos los datasets fueron cargados correctamente.\n")
} else {
  stop(
    "Uno o más datasets no contienen observaciones."
  )
}

cat("\nBloque 1 finalizado correctamente.\n")

# BLOQUE 2. Exploración Inicial --------------------------------------------
## Objetivo: Explorar la estructura general de los datasets oficiales del
# proyecto antes de iniciar el inventario.
### Entradas:
# - datasets_master
### Producto:
# - Resumen estructural de los datasets.
### Responde:
# ¿Cuál es el estado general de los datasets oficiales?

# 2.1. Función de exploración ----------------------------------------------
## Objetivo: Mostrar la estructura general de un dataset.
resumen_glimpse <- function(datos, nombre) {
  cat("\n")
  cat(strrep("=", 90), "\n")
  cat(sprintf("DATASET: %s\n", toupper(nombre)))
  cat(strrep("=", 90), "\n")
  dplyr::glimpse(datos)
  cat("\n")
} # Resumen estructural

# 2.2. Función de auditoría de nulos ---------------------------------------
## Objetivo: Resumir la cantidad de valores faltantes por dataset.
resumen_nulos <- function(datos, nombre) {
  tibble::tibble(
    dataset = nombre,
    observaciones = nrow(datos),
    variables = ncol(datos),
    nulos = sum(is.na(datos)),
    porcentaje_nulos = round(
      100 * sum(is.na(datos)) / (nrow(datos) * ncol(datos)),
      2
    )
  )
} # Resumen de valores faltantes

# 2.3. Exploración estructural ---------------------------------------------
## Objetivo: Mostrar la estructura de cada dataset.
purrr::iwalk(
  datasets_master,
  resumen_glimpse
)

# 2.4. Auditoría de valores faltantes --------------------------------------
## Objetivo: Resumir los valores faltantes de cada dataset.

auditoria_nulos <- purrr::imap_dfr(
  datasets_master,
  resumen_nulos
) # Auditoría de nulos

# 2.5. Resultados ----------------------------------------------------------
## Objetivo: Mostrar el resumen de la exploración inicial.
cat("\n")
cat(strrep("-", 50), "\n")
cat("BLOQUE 2. EXPLORACIÓN INICIAL\n")
cat(strrep("-", 50), "\n")

print(
  auditoria_nulos
)

# 2.6. Dictamen ------------------------------------------------------------
## Objetivo: Emitir el estado general de los datasets.

cat("\n")
cat(strrep("-", 90), "\n")
cat("DICTAMEN\n")
cat(strrep("-", 90), "\n")

cat(sprintf(
  "Datasets explorados         : %s\n",
  length(datasets_master)
))

cat(sprintf(
  "Observaciones esperadas     : %s\n",
  unique(auditoria_nulos$observaciones)
))

cat(sprintf(
  "Variables (mínimo)          : %s\n",
  min(auditoria_nulos$variables)
))

cat(sprintf(
  "Variables (máximo)          : %s\n",
  max(auditoria_nulos$variables)
))

cat(sprintf(
  "Total de valores faltantes  : %s\n",
  sum(auditoria_nulos$nulos)
))

# panel_master, panel_clean, panel_modelado, dataset_gnn

# panel_master
print(data.frame(
  posicion = seq_along(names(panel_master)),
  variable = names(panel_master)
), row.names = FALSE)

# panel_clean
print(data.frame(
  posicion = seq_along(names(panel_clean)),
  variable = names(panel_clean)
), row.names = FALSE)
# names(dataset_gnn)

# panel_modelado
print(data.frame(
  posicion = seq_along(names(panel_modelado)),
  variable = names(panel_modelado)
), row.names = FALSE)
# names(dataset_gnn)


print(data.frame(
  posicion = seq_along(names(dataset_gnn)),
  variable = names(dataset_gnn)
), row.names = FALSE)
# names(dataset_gnn)

cat("\nBloque 2 finalizado correctamente.\n")



# BLOQUE 3. Inventario de Variables ----------------------------------------
## Objetivo: Inventariar las variables contenidas en los datasets oficiales
# del proyecto.
### Entradas:
# - datasets_master
### Producto:
# - inventario_variables
### Responde:
# ¿Qué variables contiene cada dataset oficial?

# 3.1. Inventario ----------------------------------------------------------
## Objetivo: Registrar todas las variables de los datasets oficiales.
inventario_variables <- purrr::imap_dfr(
  datasets_master,
  function(datos, nombre) {
    tibble(
      dataset = nombre,
      variable = names(datos),
      clase = vapply(
        datos,
        function(x) class(x)[1],
        character(1)
      ),
      tipo = vapply(
        datos,
        typeof,
        character(1)
      )
    )
  }
) # Inventario de variables

# 3.2. Resumen -------------------------------------------------------------
## Objetivo: Contabilizar el número de variables por dataset.
resumen_variables <- inventario_variables |>
  dplyr::count(
    dataset,
    name = "n_variables"
  ) # Resumen de variables

# 3.3. Variables por dataset -----------------------------------------------
## Objetivo: Mostrar el listado de variables, clase y tipo de cada dataset.

cat("\n")
cat(strrep("-", 50), "\n")
cat("BLOQUE 3. INVENTARIO DE VARIABLES\n")
cat(strrep("-", 50), "\n")

print(resumen_variables)

for (nombre in unique(inventario_variables$dataset)) {
  cat("\n")
  cat(strrep("-", 50), "\n")
  cat(sprintf("DATASET: %s\n", toupper(nombre)))
  cat(strrep("-", 50), "\n")
  print(
    inventario_variables |>
      dplyr::filter(dataset == nombre) |>
      dplyr::select(
        variable,
        clase,
        tipo),
    n = Inf
  )
}

# 3.4. Resumen general -----------------------------------------------------
## Objetivo: Resumir el inventario generado.
cat("\n")
cat(strrep("-", 50), "\n")
cat("RESUMEN\n")
cat(strrep("-", 50), "\n")
cat(sprintf("Datasets inventariados: %s\n", length(datasets_master)))
cat(sprintf("Variables inventariadas: %s\n", nrow(inventario_variables)))
cat(sprintf("Variables mínimas: %s\n", min(resumen_variables$n_variables)))
cat(sprintf("Variables máximas: %s\n", max(resumen_variables$n_variables)))

# 3.5. Dictamen ------------------------------------------------------------
## Objetivo: Verificar que el inventario fue generado correctamente.
cat("\n")
cat(strrep("-", 50), "\n")
cat("DICTAMEN\n")
cat(strrep("-", 50), "\n")

if (nrow(inventario_variables) > 0) {
  cat("El inventario de variables fue generado correctamente.\n")
} else {
  stop(
    "No fue posible generar el inventario de variables."
  )
}

cat("\nBloque 3 finalizado correctamente.\n")

# BLOQUE 4. Matriz de Trazabilidad -----------------------------------------
## Objetivo: Construir la matriz de trazabilidad de las variables a lo largo
# del pipeline oficial del proyecto.
### Entradas:
# - inventario_variables
### Producto:
# - matriz_trazabilidad
### Responde:
# ¿Cómo evolucionan las variables durante el pipeline oficial?

# 4.0. Funcion -------------------------------------------------------------
construir_matriz_trazabilidad <- function(inventario, pipeline) {
  
  inventario |>
    dplyr::mutate(
      presente = "SI"
    ) |>
    dplyr::select(
      variable,
      dataset,
      presente
    ) |>
    tidyr::pivot_wider(
      names_from = dataset,
      values_from = presente,
      values_fill = "NO"
    ) |>
    dplyr::select(
      variable,
      dplyr::all_of(pipeline)
    ) |>
    dplyr::group_by(variable) |>
    dplyr::summarise(
      panel_master = ifelse(any(panel_master == "SI"), "SI", "NO"),
      panel_clean = ifelse(any(panel_clean == "SI"), "SI", "NO"),
      panel_modelado = ifelse(any(panel_modelado == "SI"), "SI", "NO"),
      dataset_gnn = ifelse(any(dataset_gnn == "SI"), "SI", "NO"),
      .groups = "drop"
    ) |>
    dplyr::rowwise() |>
    dplyr::mutate(
      n_datasets = sum(
        c_across(
          panel_master:dataset_gnn
        ) == "SI"
      ),
      primera_aparicion = {
        estado <- c_across(
          panel_master:dataset_gnn
        )
        pipeline[which(estado == "SI")[1]]
      },
      ultima_aparicion = {
        estado <- c_across(
          panel_master:dataset_gnn
        )
        pipeline[max(which(estado == "SI"))]
      }
    ) |>
    dplyr::ungroup() |>
    dplyr::arrange(
      dplyr::desc(n_datasets),
      variable
    )
  
} # Construir matriz de trazabilidad

pipeline <- c(
  "panel_master",
  "panel_clean",
  "panel_modelado",
  "dataset_gnn"
) # Orden oficial del pipeline

# 4.1. Construcción --------------------------------------------------------
## Objetivo: Construir la matriz de trazabilidad oficial del pipeline.

matriz_trazabilidad <- construir_matriz_trazabilidad(
  inventario_variables,
  pipeline
) # Matriz de trazabilidad

cat("\n")
cat(strrep("-", 50), "\n")
cat("4.1. CONSTRUCCIÓN\n")
cat(strrep("-", 50), "\n")

print(matriz_trazabilidad, n = Inf)

cat("\n")
cat(strrep("-", 50), "\n")
cat("RESUMEN\n")
cat(strrep("-", 50), "\n")

cat(sprintf(
  "Variables inventariadas: %s\n",
  nrow(matriz_trazabilidad)
))

cat(sprintf(
  "Datasets evaluados: %s\n",
  length(pipeline)
))

# 4.2. Variables Conservadas -----------------------------------------------
## Objetivo: Identificar las variables presentes en todos los datasets
# oficiales del proyecto.

variables_conservadas <- matriz_trazabilidad |>
  dplyr::filter(
    n_datasets == length(pipeline)
  ) # Variables conservadas

cat("\n")
cat(strrep("-", 50), "\n")
cat("4.2. VARIABLES CONSERVADAS\n")
cat(strrep("-", 50), "\n")

print(
  variables_conservadas,
  n = Inf
)

cat("\n")
cat(strrep("-", 50), "\n")
cat("RESUMEN\n")
cat(strrep("-", 50), "\n")

cat(sprintf(
  "Variables conservadas: %s\n",
  nrow(variables_conservadas)
))

# 4.3. Variables Eliminadas -----------------------------------------------
## Objetivo: Identificar las variables eliminadas durante el pipeline
# oficial del proyecto.

variables_eliminadas <- matriz_trazabilidad |>
  dplyr::filter(
    dataset_gnn == "NO"
  ) # Variables eliminadas

cat("\n")
cat(strrep("-", 50), "\n")
cat("4.3. VARIABLES ELIMINADAS\n")
cat(strrep("-", 50), "\n")

print(
  variables_eliminadas,
  n = Inf
)

cat("\n")
cat(strrep("-", 50), "\n")
cat("RESUMEN\n")
cat(strrep("-", 50), "\n")

cat(sprintf(
  "Variables eliminadas: %s\n",
  nrow(variables_eliminadas)
))

# 4.4. Variables Incorporadas ---------------------------------------------
## Objetivo: Identificar las variables incorporadas durante el pipeline
# oficial del proyecto.

variables_incorporadas <- matriz_trazabilidad |>
  dplyr::filter(
    panel_master == "NO"
  ) # Variables incorporadas

cat("\n")
cat(strrep("-", 50), "\n")
cat("4.4. VARIABLES INCORPORADAS\n")
cat(strrep("-", 50), "\n")

print(
  variables_incorporadas,
  n = Inf
)

cat("\n")
cat(strrep("-", 50), "\n")
cat("RESUMEN\n")
cat(strrep("-", 50), "\n")

cat(sprintf(
  "Variables incorporadas: %s\n",
  nrow(variables_incorporadas)
))

# 4.5. Primera Aparición ---------------------------------------------------
## Objetivo: Mostrar el primer dataset donde aparece cada variable.

cat("\n")
cat(strrep("-", 50), "\n")
cat("4.5. PRIMERA APARICIÓN\n")
cat(strrep("-", 50), "\n")

print(
  matriz_trazabilidad |>
    dplyr::select(
      variable,
      primera_aparicion
    ),
  n = Inf
)

cat("\n")
cat(strrep("-", 50), "\n")
cat("RESUMEN\n")
cat(strrep("-", 50), "\n")

print(
  matriz_trazabilidad |>
    dplyr::count(
      primera_aparicion,
      name = "n_variables"
    )
)

cat(sprintf(
  "Datasets con primera aparición: %s\n",
  matriz_trazabilidad |>
    dplyr::distinct(
      primera_aparicion
    ) |>
    nrow()
))

# 4.6. Última Aparición ----------------------------------------------------
## Objetivo: Mostrar el último dataset donde permanece cada variable.

cat("\n")
cat(strrep("-", 50), "\n")
cat("4.6. ÚLTIMA APARICIÓN\n")
cat(strrep("-", 50), "\n")

print(
  matriz_trazabilidad |>
    dplyr::select(
      variable,
      ultima_aparicion
    ),
  n = Inf
)

cat("\n")
cat(strrep("-", 50), "\n")
cat("RESUMEN\n")
cat(strrep("-", 50), "\n")

print(
  matriz_trazabilidad |>
    dplyr::count(
      ultima_aparicion,
      name = "n_variables"
    )
)

cat(sprintf(
  "Datasets con última aparición: %s\n",
  matriz_trazabilidad |>
    dplyr::distinct(
      ultima_aparicion
    ) |>
    nrow()
))

# 4.7. Diccionario de Equivalencias ----------------------------------------
## Objetivo: Documentar las variables renombradas durante la construcción del
# dataset GNN.

diccionario_equivalencias <- tibble::tribble(
  ~variable_original, ~variable_gnn, ~panel_master, ~panel_clean, ~panel_modelado,
  "era5_d2m", "d2m", "SI", "SI", "SI",
  "era5_e", "e", "SI", "SI", "SI",
  "era5_lai_hv", "lai_hv", "SI", "SI", "SI",
  "era5_pev", "pev", "SI", "SI", "SI",
  "era5_ro", "ro", "SI", "SI", "SI",
  "era5_sro", "sro", "SI", "SI", "SI",
  "era5_strd", "strd", "SI", "SI", "SI",
  "era5_t2m", "t2m", "SI", "SI", "SI",
  "era5_tp", "tp", "SI", "SI", "SI",
  "era5_u10", "u10", "SI", "SI", "SI",
  "era5_v10", "v10", "SI", "SI", "SI"
) # Diccionario oficial de equivalencias

cat("\n"); cat(strrep("-", 50), "\n"); cat("4.7. DICCIONARIO DE EQUIVALENCIAS\n"); cat(strrep("-", 50), "\n")

print(diccionario_equivalencias)

cat("\n"); cat(strrep("-", 50), "\n"); cat("RESUMEN\n"); cat(strrep("-", 50), "\n")

cat(sprintf("Variables renombradas: %s\n", nrow(diccionario_equivalencias)))
cat(sprintf("Renombramiento aplicado en: panel_master, panel_clean y panel_modelado\n"))
cat(sprintf("Cambio de nombre realizado durante la construcción de dataset_gnn.\n"))

# 4.8. Equivalencias en el Pipeline ----------------------------------------
## Objetivo: Presentar las variables que fueron renombradas durante la
# construcción del dataset GNN.

cat("\n"); cat(strrep("-", 50), "\n"); cat("4.8. EQUIVALENCIAS EN EL PIPELINE\n"); cat(strrep("-", 50), "\n")

print(diccionario_equivalencias)

cat("\n"); cat(strrep("-", 50), "\n"); cat("RESUMEN\n"); cat(strrep("-", 50), "\n")

cat(sprintf("Variables renombradas: %s\n", nrow(diccionario_equivalencias)))
cat("El renombramiento aplica a: panel_master, panel_clean y panel_modelado.\n")
cat("Las variables fueron estandarizadas durante la construcción de dataset_gnn.\n")


# 4.8. Comparación ---------------------------------------------------------
## Objetivo: Comparar la matriz de trazabilidad con el diccionario de
# equivalencias de variables.

comparacion_equivalencias <- matriz_trazabilidad |>
  dplyr::left_join(
    diccionario_equivalencias,
    by = c("variable" = "variable_original")
  ) |>
  dplyr::mutate(
    variable_gnn = dplyr::coalesce(
      variable_gnn,
      variable
    ),
    renombrada = dplyr::if_else(
      variable == variable_gnn,
      "NO",
      "SI"
    )
  ) # Comparación entre trazabilidad y equivalencias

cat("\n"); cat(strrep("-", 50), "\n"); cat("4.8. COMPARACIÓN\n"); cat(strrep("-", 50), "\n")

print(comparacion_equivalencias, n = Inf)

cat("\n"); cat(strrep("-", 50), "\n"); cat("RESUMEN\n"); cat(strrep("-", 50), "\n")

cat(sprintf("Variables evaluadas: %s\n", nrow(comparacion_equivalencias)))
cat(sprintf("Variables renombradas: %s\n", sum(comparacion_equivalencias$renombrada == "SI")))
cat(sprintf("Variables sin cambio: %s\n", sum(comparacion_equivalencias$renombrada == "NO")))

# 4.9. Resumen Ejecutivo ---------------------------------------------------
## Objetivo: Resumir la trazabilidad de las variables durante el pipeline
# oficial del proyecto.

resumen_trazabilidad <- tibble::tibble(
  indicador = c(
    "Datasets evaluados",
    "Variables inventariadas",
    "Variables conservadas",
    "Variables eliminadas",
    "Variables incorporadas",
    "Variables renombradas"
  ),
  valor = c(
    length(pipeline),
    nrow(matriz_trazabilidad),
    nrow(variables_conservadas),
    nrow(variables_eliminadas),
    nrow(variables_incorporadas),
    nrow(diccionario_equivalencias)
  )
) # Resumen ejecutivo

cat("\n"); cat(strrep("-", 50), "\n"); cat("4.9. RESUMEN EJECUTIVO\n"); cat(strrep("-", 50), "\n")

print(resumen_trazabilidad)

cat("\n"); cat(strrep("-", 50), "\n"); cat("RESULTADO\n"); cat(strrep("-", 50), "\n")

cat(sprintf("Datasets evaluados: %s\n", length(pipeline)))
cat(sprintf("Variables inventariadas: %s\n", nrow(matriz_trazabilidad)))
cat(sprintf("Variables conservadas: %s\n", nrow(variables_conservadas)))
cat(sprintf("Variables eliminadas: %s\n", nrow(variables_eliminadas)))
cat(sprintf("Variables incorporadas: %s\n", nrow(variables_incorporadas)))
cat(sprintf("Variables renombradas: %s\n", nrow(diccionario_equivalencias)))

# 4.10. Dictamen -----------------------------------------------------------
## Objetivo: Validar la consistencia de la trazabilidad del pipeline oficial.

cat("\n"); cat(strrep("-", 50), "\n"); cat("4.10. DICTAMEN\n"); cat(strrep("-", 50), "\n")

if (nrow(matriz_trazabilidad) == 0) {
  stop("La matriz de trazabilidad no fue generada.")
}

if (nrow(diccionario_equivalencias) == 0) {
  stop("El diccionario de equivalencias no fue generado.")
}

if (nrow(resumen_trazabilidad) == 0) {
  stop("El resumen de trazabilidad no fue generado.")
}

cat("La matriz de trazabilidad fue generada correctamente.\n")
cat("El diccionario de equivalencias fue generado correctamente.\n")
cat("El resumen de trazabilidad fue generado correctamente.\n")
cat("La trazabilidad del pipeline quedó documentada correctamente.\n")

cat("\nBloque 4.10 finalizado correctamente.\n")


# 4.11. Exportación --------------------------------------------------------
## Objetivo: Exportar los productos oficiales de la trazabilidad del pipeline.

# 4.11.a, Vista tablas
cat("\n"); cat(strrep("-", 50), "\n"); cat("VISTA PREVIA DE LOS OBJETOS\n"); cat(strrep("-", 50), "\n")

cat("\n"); cat(strrep("-", 50), "\n"); cat("MATRIZ DE TRAZABILIDAD\n"); cat(strrep("-", 50), "\n")
print(matriz_trazabilidad, n = Inf)

cat("\n"); cat(strrep("-", 50), "\n"); cat("DICCIONARIO DE EQUIVALENCIAS\n"); cat(strrep("-", 50), "\n")
print(diccionario_equivalencias, n = Inf)

cat("\n"); cat(strrep("-", 50), "\n"); cat("RESUMEN DE TRAZABILIDAD\n"); cat(strrep("-", 50), "\n")
print(resumen_trazabilidad)


cat("\n"); cat(strrep("-", 50), "\n"); cat("4.11. EXPORTACIÓN\n"); cat(strrep("-", 50), "\n")

print(matriz_trazabilidad, n = Inf)
print(diccionario_equivalencias, n = Inf)
print(resumen_trazabilidad)

ruta_reportes <- here::here(
  "data",
  "processed",
  "r",
  "reports"
) # Directorio de reportes

dir.create(ruta_reportes, recursive = TRUE, showWarnings = FALSE) # Crear directorio

objetos_exportar <- list(
  matriz_trazabilidad = matriz_trazabilidad,
  diccionario_equivalencias = diccionario_equivalencias,
  resumen_trazabilidad = resumen_trazabilidad
) # Objetos a exportar

purrr::iwalk(
  objetos_exportar,
  function(objeto, nombre) {
    
    arrow::write_parquet(
      objeto,
      file.path(
        ruta_reportes,
        paste0(nombre, ".parquet")
      )
    ) # Exportar Parquet
    
  }
)

wb <- openxlsx::createWorkbook() # Crear libro Excel

purrr::iwalk(
  objetos_exportar,
  function(objeto, nombre) {
    
    openxlsx::addWorksheet(
      wb,
      nombre
    ) # Crear hoja
    
    openxlsx::writeData(
      wb,
      nombre,
      objeto
    ) # Escribir datos
    
  }
)

openxlsx::saveWorkbook(
  wb,
  file.path(
    ruta_reportes,
    "trazabilidad_pipeline.xlsx"
  ),
  overwrite = TRUE
) # Exportar libro Excel

cat("\n"); cat(strrep("-", 50), "\n"); cat("ARCHIVOS EXPORTADOS\n"); cat(strrep("-", 50), "\n")

cat("Parquet: matriz_trazabilidad.parquet\n")
cat("Parquet: diccionario_equivalencias.parquet\n")
cat("Parquet: resumen_trazabilidad.parquet\n")
cat("Excel: trazabilidad_pipeline.xlsx\n")
cat(sprintf("Directorio: %s\n", ruta_reportes))

cat("\nBloque 4.11 finalizado correctamente.\n")
cat("\nBloque 4 finalizado correctamente.\n")