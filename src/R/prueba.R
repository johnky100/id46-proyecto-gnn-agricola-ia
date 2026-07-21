# ------------------- Bloque 1. Librerías -------------------
library(arrow) # Lectura y escritura de archivos Parquet
library(data.table) # Lectura rápida de archivos CSV
library(tidyverse) # Ecosistema tidyverse

# ------------------- Bloque 2. Cargar Bases -------------------
chirps_modelado <- fread(
  "data/processed/chirps/chirps_variables_finales.csv"
) # Leer dataset CHIRPS

cna_modelado <- read_parquet(
  "data/processed/cna/cna_modelado.parquet"
) # Leer dataset CNA

divipola_modelado <- read_parquet(
  "data/processed/divipola/divipola_municipio_anio.parquet"
) # Leer DIVIPOLA municipal anual

divipola_bruto <- fread(
  "data/raw/divipola/divipola_municipios.csv"
) # Leer DIVIPOLA municipal anual

era5_modelado <- read_parquet(
  "data/processed/era5/era5_municipal_anual.parquet"
) # Leer variables climáticas ERA5

eva_modelado <- fread(
  "data/processed/eva/eva_gnn_final.csv"
) # Leer dataset EVA

irrigacion_modelado <- read_parquet(
  "data/processed/irrigacion/irrigacion_municipio.parquet"
) # Leer dataset de irrigación

precio_tierra_modelado <- read_parquet(
  "data/processed/precios_tierra/precios_tierra_features.parquet"
) # Leer dataset de precios de tierra

panel_master_model <- read_parquet(
  "data/processed/modelado/panel_master_model.parquet"
)

panel_train <- read_parquet(
  "data/processed/modelado/panel_train.parquet"
)

panel_train_sin_precio <- read_parquet(
  "data/processed/modelado/panel_train_sin_precio.parquet"
) 

panel_train_gnn <- read_parquet(
  "data/processed/modelado/panel_train_gnn.parquet"
) 

panel_train_gnn_complete <- read_parquet(
  "data/processed/modelado/panel_train_gnn_complete.parquet"
) 

panel_master_gnn <- read_parquet(
  "data/processed/modelado/panel_master_gnn.parquet"
)

# D:\Proyectos_IA\proyecto-gnn-agricola\data\DB_no_necesitan
panel_master_gnn_modelado <- read_parquet(
  "data/DB_no_necesitan/panel_master_gnn_modelado.parquet"
)

panel_isam_v1_unido <- read_parquet(
  "data/processed/panel_maestro_gnn/isam/panel_isam_v1.parquet"
) # Leer panel ISAM unificado

# ------------------- Bloque 3. Lista de Bases -------------------
bases <- list(
  CHIRPS = chirps_modelado,
  CNA = cna_modelado,
  DIVIPOLA = divipola_modelado,
  ERA5 = era5_modelado,
  EVA = eva_modelado,
  IRRIGACION = irrigacion_modelado,
  PRECIO_TIERRA = precio_tierra_modelado,
  PANEL_MASTER_MODEL = panel_master_model,
  PANEL_TRAIN = panel_train,
  PANEL_TRAIN_SIN_PRECIO = panel_train_sin_precio,
  PANEL_TRAIN_GNN = panel_train_gnn,
  PANEL_TRAIN_GNN_COMPLETE = panel_train_gnn_complete,
  PANEL_MASTER_GNN = panel_master_gnn,
  PANEL_MASTER_GNN_MODELADO = panel_master_gnn_modelado,
  ISAM_UNIDO = panel_isam_v1_unido
) # Lista de datasets

# ------------------- Bloque 4. Función Glimpse -------------------
mostrar_glimpse <- function(data, nombre) {
  
  cat(
    "\n",
    paste(rep("=", 80), collapse = ""),
    "\n",
    nombre,
    "\n",
    paste(rep("=", 80), collapse = ""),
    "\n",
    "Filas: ", nrow(data),
    " | Columnas: ", ncol(data),
    "\n\n",
    sep = ""
  ) # Mostrar encabezado
  
  dplyr::glimpse(data) # Mostrar estructura resumida
  
} # Función para visualizar datasets

# ------------------- Bloque 5. Ejecutar Glimpse -------------------
purrr::iwalk(
  bases,
  mostrar_glimpse
) # Ejecutar glimpse para cada dataset

# ------------------- Bloque 6. Función Auditoría de Nulos -------------------
auditar_nulos <- function(data, nombre) {
  
  resultado <- tibble::tibble(
    variable = names(data),
    nulos = sapply(data, \(x) sum(is.na(x))),
    porcentaje_nulos = round(
      sapply(data, \(x) mean(is.na(x))) * 100,
      2
    )
  ) |>
    dplyr::arrange(desc(nulos)) # Ordenar por cantidad de nulos
  
  cat(
    "\n",
    strrep("=", 80),
    "\n",
    nombre,
    "\n",
    strrep("=", 80),
    "\n",
    "Variables: ", ncol(data),
    " | Filas: ", nrow(data),
    " | Variables con NA: ", sum(resultado$nulos > 0),
    " | Total NA: ", sum(resultado$nulos),
    "\n\n",
    sep = ""
  ) # Resumen general
  
  print(resultado)
  
  invisible(resultado)
  
} # Auditoría de valores faltantes

# ------------------- Bloque 7. Ejecutar Auditoría de Nulos -------------------
nulos_chirps <- auditar_nulos(
  chirps_modelado,
  "CHIRPS"
)

nulos_cna <- auditar_nulos(
  cna_modelado,
  "CNA"
)

nulos_divipola <- auditar_nulos(
  divipola_modelado,
  "DIVIPOLA"
)

nulos_era5 <- auditar_nulos(
  era5_modelado,
  "ERA5"
)

nulos_eva <- auditar_nulos(
  eva_modelado,
  "EVA"
)

nulos_irrigacion <- auditar_nulos(
  irrigacion_modelado,
  "IRRIGACION"
)

nulos_precio_tierra <- auditar_nulos(
  precio_tierra_modelado,
  "PRECIO_TIERRA"
)

nulos_panel_master_model <- auditar_nulos(
  precio_tierra_modelado,
  "PANEL_MASTER_MODEL"
)

nulos_panel_train <- auditar_nulos(
  precio_tierra_modelado,
  "PANEL_TRAIN"
)

nulos_panel_train_sin_precio <- auditar_nulos(
  precio_tierra_modelado,
  "PANEL_TRAIN_SIN_PRECIO"
)

nulos_panel_train_gnn <- auditar_nulos(
  precio_tierra_modelado,
  "PANEL_TRAIN_GNN"
)

nulos_panel_train_gnn_complete <- auditar_nulos(
  precio_tierra_modelado,
  "PANEL_TRAIN_GNN_COMPLETE"
)

nulos_panel_master_gnn <- auditar_nulos(
  precio_tierra_modelado,
  "PANEL_MASTER_GNN"
)

nulos_panel_master_gnn_modelado <- auditar_nulos(
  precio_tierra_modelado,
  "PANEL_MASTER_GNN_MODELADON"
)

nulos_isam_unido <- auditar_nulos(
  panel_isam_v1_unido,
  "ISAM_UNIDO"
)

# ------------------- Bloque 8. Catálogo Maestro de Municipios -------------------
municipios_referencia <- divipola_modelado |>
  dplyr::mutate(
    cod_mpio = stringr::str_pad(
      as.character(cod_mpio),
      width = 5,
      side = "left",
      pad = "0"
    )
  ) |>
  dplyr::distinct(
    cod_mpio,
    municipio,
    departamento
  ) # Catálogo maestro de municipios

# ------------------- Bloque 9. Función Auditoría Municipal -------------------
auditar_municipios <- function(data, nombre) {
  
  codigos_fuente <- unique(
    stringr::str_pad(
      as.character(data$cod_mpio),
      width = 5,
      side = "left",
      pad = "0"
    )
  ) # Normalizar códigos DANE
  
  faltantes <- municipios_referencia |>
    dplyr::filter(
      !cod_mpio %in% codigos_fuente
    ) |>
    dplyr::arrange(
      cod_mpio
    ) # Municipios ausentes
  
  cat(
    "\n",
    strrep("=", 80),
    "\n",
    nombre,
    "\n",
    strrep("=", 80),
    "\n",
    "Municipios esperados: ", n_distinct(municipios_referencia$cod_mpio),
    " | Municipios encontrados: ", length(codigos_fuente),
    " | Municipios faltantes: ", nrow(faltantes),
    "\n\n",
    sep = ""
  ) # Resumen de cobertura
  
  print(faltantes)
  
  invisible(faltantes)
  
} # Auditoría de cobertura municipal

# ------------------- Bloque 10. Ejecutar Auditoría Municipal -------------------
faltantes_chirps <- auditar_municipios(
  chirps_modelado,
  "CHIRPS"
)

faltantes_cna <- auditar_municipios(
  cna_modelado,
  "CNA"
)

faltantes_divipola <- auditar_municipios(
  divipola_modelado,
  "DIVIPOLA"
)

faltantes_era5 <- auditar_municipios(
  era5_modelado,
  "ERA5"
)

faltantes_eva <- auditar_municipios(
  eva_modelado,
  "EVA"
)

faltantes_irrigacion <- auditar_municipios(
  irrigacion_modelado,
  "IRRIGACION"
)

faltantes_precio_tierra <- auditar_municipios(
  precio_tierra_modelado,
  "PRECIO_TIERRA"
)

faltantes_panel_master_model <- auditar_municipios(
  precio_tierra_modelado,
  "PANEL_MASTER_MODEL"
)

faltantes_panel_train <- auditar_municipios(
  precio_tierra_modelado,
  "PANEL_TRAIN"
)

faltantes_panel_train_sin_precio <- auditar_municipios(
  precio_tierra_modelado,
  "PANEL_TRAIN_SIN_PRECIO"
)

faltantes_panel_train_gnn <- auditar_municipios(
  precio_tierra_modelado,
  "PANEL_TRAIN_GNN"
)

faltantes_panel_train_gnn_complete <- auditar_municipios(
  precio_tierra_modelado,
  "PANEL_TRAIN_GNN_COMPLETE"
)

faltantes_panel_master_gnn <- auditar_municipios(
  precio_tierra_modelado,
  "PANEL_MASTER_GNN"
)

faltantes_panel_master_gnn_modelado <- auditar_municipios(
  precio_tierra_modelado,
  "PANEL_MASTER_GNN_MODELADO"
)

faltantes_isam_unido <- auditar_municipios(
  panel_isam_v1_unido,
  "ISAM_UNIDO"
)

# ------------------- Bloque 11. Resumen Cobertura Municipal -------------------
resumen_municipios <- tibble::tibble(
  dataset = c(
    "CHIRPS",
    "CNA",
    "DIVIPOLA",
    "ERA5",
    "EVA",
    "IRRIGACION",
    "PRECIO_TIERRA",
    "PANEL_MASTER_MODEL",
    "PANEL_TRAIN",
    "PANEL_TRAIN_SIN_PRECIO",
    "PANEL_TRAIN_GNN",
    "PANEL_TRAIN_GNN_COMPLETE",
    "PANEL_MASTER_GNN",
    "PANEL_MASTER_GNN_MODELADO",
    "ISAM_UNIDO"
  ),
  municipios = c(
    n_distinct(chirps_modelado$cod_mpio),
    n_distinct(cna_modelado$cod_mpio),
    n_distinct(divipola_modelado$cod_mpio),
    n_distinct(era5_modelado$cod_mpio),
    n_distinct(eva_modelado$cod_mpio),
    n_distinct(irrigacion_modelado$cod_mpio),
    n_distinct(precio_tierra_modelado$cod_mpio),
    
    n_distinct(panel_master_model$cod_mpio),
    n_distinct(panel_train$cod_mpio),
    n_distinct(panel_train_sin_precio$cod_mpio),
    n_distinct(panel_train_gnn$cod_mpio),
    n_distinct(panel_train_gnn_complete$cod_mpio),
    n_distinct(panel_master_gnn$cod_mpio),
    n_distinct(panel_master_gnn_modelado$cod_mpio),
    n_distinct(panel_isam_v1_unido$cod_mpio)
  )
); resumen_municipios # Resumen de municipios por fuente

panel_master_gnn |>
  count(panel_id) |>
  filter(n > 1)

n_distinct(divipola_modelado$cod_mpio)

length(unique(divipola_modelado$anio))

nrow(divipola_modelado)

panel_master_gnn |>
  count(cod_mpio) |>
  arrange(desc(n))

sort(unique(panel_master_gnn$anio))

n_distinct(panel_master_gnn$cod_mpio)

n_distinct(divipola_modelado$cod_mpio)

setdiff(
  unique(panel_master_gnn$cod_mpio),
  unique(divipola_modelado$cod_mpio)
)

setdiff(
  unique(divipola_modelado$cod_mpio),
  unique(panel_master_gnn$cod_mpio)
)

panel_master_gnn |>
  filter(
    cod_mpio %in% c(
      "25480",
      "94663"
    )
  ) |>
  distinct(
    cod_mpio,
    municipio,
    departamento
  ) |>
  arrange(cod_mpio)

divipola_modelado |>
  filter(
    cod_mpio == "25483"
  ) |>
  distinct(
    cod_mpio,
    municipio,
    departamento
  )

# Eliminar Mapiripana
panel_master_gnn <- panel_master_gnn |>
  filter(
    cod_mpio != "94663"
  ) # Eliminar unidad territorial no municipal

# Corregir Nariño
panel_master_gnn <- panel_master_gnn |>
  mutate(
    cod_mpio = if_else(
      cod_mpio == "25480",
      "25483",
      cod_mpio
    )
  ) # Homologar código DANE de Nariño

# Verificación final
n_distinct(panel_master_gnn$cod_mpio)
nrow(panel_master_gnn)

panel_master_gnn |>
  summarise(
    municipios = n_distinct(cod_mpio),
    anios = n_distinct(anio),
    registros = n(),
    paneles_unicos = n_distinct(panel_id)
  )

# ------------------- Bloque 12. Exportación Panel Master GNN Validado -------------------
dir.create(
  "data/auditorias",
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta de auditorías

arrow::write_parquet(
  panel_master_gnn,
  "data/auditorias/panel_master_gnn_validado_1121_municipios.parquet"
) # Exportar panel validado en formato Parquet

data.table::fwrite(
  panel_master_gnn,
  "data/auditorias/panel_master_gnn_validado_1121_municipios.csv"
) # Exportar panel validado en formato CSV

cat(
  "\nExportación completada correctamente.\n",
  "Archivo Parquet: panel_master_gnn_validado_1121_municipios.parquet\n",
  "Archivo CSV: panel_master_gnn_validado_1121_municipios.csv\n",
  sep = ""
) # Confirmar exportación

# ------------------- Bloque 12. Exportación Versionada -------------------
dir.create(
  "data/auditorias",
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta de auditorías

fecha_auditoria <- format(
  Sys.Date(),
  "%Y%m%d"
) # Fecha de auditoría

arrow::write_parquet(
  panel_master_gnn,
  paste0(
    "data/auditorias/panel_master_gnn_validado_",
    fecha_auditoria,
    ".parquet"
  )
) # Exportar versión Parquet

data.table::fwrite(
  panel_master_gnn,
  paste0(
    "data/auditorias/panel_master_gnn_validado_",
    fecha_auditoria,
    ".csv"
  )
) # Exportar versión CSV


# D:\Proyectos_IA\proyecto-gnn-agricola\data\auditorias
panel_master_gnn_validado_1121_municipios <- read_parquet(
  "data/auditorias/panel_master_gnn_validado_1121_municipios.parquet"
) # Leer DIVIPOLA municipal anual

# ----------------------------------------------
# ----------------------------------------------

library(arrow)
library(dplyr)
library(tibble)

# Cargar bases ---------------------------------------------------------------
panel_modelado_final <- read_parquet(
  "data/processed/panel_maestro_gnn/panel_modelado_final.parquet"
) # Panel final para modelado

panel_nacional_1121 <- read_parquet(
  "data/auditorias/panel_nacional_1121_municipios_corregido.parquet"
) # Panel nacional corregido

panel_master_gnn_validado <- read_parquet(
  "data/auditorias/panel_master_gnn_validado_20260622.parquet"
) # Panel maestro validado

panel_gnn_maestro_51 <- read_parquet(
  "data/modelado/panel_gnn_maestro_51_variables.parquet"
) # Panel con 51 variables

panel_gnn_reducido_46 <- read_parquet(
  "data/modelado/panel_gnn_reducido_46_variables.parquet"
) # Panel reducido

panel_gnn_final <- read_parquet(
  "data/modelado/panel_gnn_reducido_auto.parquet"
) # Panel final automático

dataset_ganador <- read_parquet(
  "data/DB_no_necesitan/dataset_ganador_gnn.parquet"
) # Dataset ganador

# Función de auditoría -------------------------------------------------------
auditar_dataset <- function(datos, nombre) {
  cat("\n")
  cat(strrep("=", 90), "\n")
  cat(nombre, "\n")
  cat(strrep("=", 90), "\n")
  
  cat("\nGLIMPSE\n\n")
  glimpse(datos)
  
  auditoria_nulos <- tibble(
    variable = names(datos),
    nulos = sapply(datos, \(x) sum(is.na(x))),
    porcentaje_nulos = round(
      sapply(datos, \(x) mean(is.na(x)) * 100),
      4
    )
  ) |>
    arrange(desc(nulos)) # Ordenar por cantidad de NA
  
  cat("\n")
  cat(strrep("-", 90), "\n")
  cat("RESUMEN\n")
  cat(strrep("-", 90), "\n")
  
  cat("Observaciones        :", nrow(datos), "\n")
  cat("Variables            :", ncol(datos), "\n")
  cat("Variables con NA     :", sum(auditoria_nulos$nulos > 0), "\n")
  cat("Variables completas  :", sum(auditoria_nulos$nulos == 0), "\n")
  cat("Total valores NA     :", sum(auditoria_nulos$nulos), "\n")
  
  cat(
    "Porcentaje global NA :",
    round(
      sum(auditoria_nulos$nulos) /
        (nrow(datos) * ncol(datos)) * 100,
      4
    ),
    "%\n" )
  
  if (all(c("cod_mpio", "anio") %in% names(datos))) {
    cat("\n")
    cat(strrep("-", 90), "\n")
    cat("MUNICIPIOS Y AÑOS\n")
    cat(strrep("-", 90), "\n")
    
    print(
      datos |>
        summarise(
          municipios = n_distinct(cod_mpio),
          anios = n_distinct(anio),
          observaciones = n()
        )
    )
  }
  
  cat("\n")
  cat(strrep("-", 90), "\n")
  cat("VARIABLES CON NULOS\n")
  cat(strrep("-", 90), "\n")
  
  variables_con_na <- auditoria_nulos |>
    filter(nulos > 0) # Variables con valores NA
  if (nrow(variables_con_na) == 0) {
    cat("No se encontraron valores NA.\n")
  } else {
    print(variables_con_na)
  }
  invisible(auditoria_nulos)
}

# Auditorías -----------------------------------------------------------------
auditar_dataset(
  panel_modelado_final,
  "PANEL MODELADO FINAL"
)

auditar_dataset(
  panel_nacional_1121,
  "PANEL NACIONAL 1121"
)

auditar_dataset(
  panel_master_gnn_validado,
  "PANEL MASTER GNN VALIDADO"
)

auditar_dataset(
  panel_gnn_maestro_51,
  "PANEL GNN MAESTRO 51 VARIABLES"
)

auditar_dataset(
  panel_gnn_reducido_46,
  "PANEL GNN REDUCIDO 46 VARIABLES"
)

auditar_dataset(
  panel_gnn_final,
  "PANEL GNN FINAL"
)

auditar_dataset(
  dataset_ganador,
  "DATASET GANADOR"
)