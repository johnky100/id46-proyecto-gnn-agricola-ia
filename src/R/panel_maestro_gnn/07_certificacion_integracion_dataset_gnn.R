# 07_auditoria_integracion_espacial.R

# ------------------- Bloque 0. Configuración General ------------------------
source(here::here("src", "R", "config", "00_packages.R")) # Cargar paquetes
source(here::here("src", "R", "config", "01_paths.R")) # Cargar rutas
source(here::here("src", "R", "config", "02_global_parameters.R")) # Cargar parámetros

cat("\n")
cat(strrep("-", 90), "\n")
cat("EVALUACIÓN DE DATASETS PARA MODELADO GNN\n")
cat(strrep("=", 90), "\n")
cat("Fecha      :", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Periodo    :", anio_inicio, "-", anio_fin, "\n")
cat("Target     :", variable_target, "\n")
cat("Semilla    :", seed_global, "\n")
cat(strrep("-", 90), "\n")

# BLOQUE 1. CARGA DE BASES DE DATOS
# ----------------------------------
library(arrow) # Lectura y escritura de archivos Parquet
library(dplyr) # Manipulación de datos
library(sf) # Manejo de información geoespacial

# Cargar dataset ganador GNN -------------------------------------------

dataset_ganador_gnn <- arrow::read_parquet(
  "data/processed/db_ganador/dataset_ganador_gnn.parquet"
) # Dataset maestro para modelado GNN

# Cargar catálogo espacial ---------------------------------------------

municipios_panel <- arrow::read_parquet(
  "data/processed/spatial/municipios_panel.parquet"
) # Catálogo espacial oficial de municipios

# Resumen de carga ------------------------------------------------------

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 1. CARGA DE BASES DE DATOS\n")
cat(strrep("=", 90), "\n")

cat(
  "Dataset GNN      :",
  format(nrow(dataset_ganador_gnn), big.mark = ","),
  "registros |",
  ncol(dataset_ganador_gnn),
  "variables\n"
)

cat(
  "Municipios Panel :",
  format(nrow(municipios_panel), big.mark = ","),
  "registros |",
  ncol(municipios_panel),
  "variables\n"
)

stopifnot(nrow(dataset_ganador_gnn) > 0) # Validar dataset no vacío
stopifnot(nrow(municipios_panel) == 1121) # Validar catálogo oficial
stopifnot("cod_mpio" %in% names(dataset_ganador_gnn)) # Validar llave principal
stopifnot(sum(is.na(dataset_ganador_gnn)) == 0) # Validar ausencia de NA

# BLOQUE 2. AUDITORÍA DEL DATASET GANADOR GNN
# ----------------------------------
# Resumen general -------------------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 2. AUDITORÍA DEL DATASET GANADOR GNN\n")
cat(strrep("-", 90), "\n")

cat(
  "Número de registros :",
  format(nrow(dataset_ganador_gnn), big.mark = ","),
  "\n"
)

cat(
  "Número de variables :",
  ncol(dataset_ganador_gnn),
  "\n"
)

cat(
  "Memoria utilizada   :",
  format(
    object.size(dataset_ganador_gnn),
    units = "MB"
  ),
  "\n"
)
# Subbloque 2.1
# Estructura del dataset ------------------------------------------------
glimpse(dataset_ganador_gnn)

# Subbloque 2.2
# Tipos de datos --------------------------------------------------------
tipos_variables <- data.frame(
  variable = names(dataset_ganador_gnn),
  clase = sapply(dataset_ganador_gnn, class),
  row.names = NULL
) # Tipo de dato por variable

print(tipos_variables)

# Subbloque 2.3
# Resumen estadístico ---------------------------------------------------
summary(dataset_ganador_gnn)

# Subbloque 2.4
# Variables por tipo ----------------------------------------------------
cat("\n")
cat(
  "Variables numéricas :",
  sum(sapply(dataset_ganador_gnn, is.numeric)),
  "\n"
)

cat(
  "Variables carácter :",
  sum(sapply(dataset_ganador_gnn, is.character)),
  "\n"
)

cat(
  "Variables enteras :",
  sum(sapply(dataset_ganador_gnn, is.integer)),
  "\n"
)

# Alcance de la auditoría ------------------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("ALCANCE DE LA AUDITORÍA\n")
cat(strrep("-", 90), "\n")

alcance_auditoria <- data.frame(
  componente = c(
    "Dimensiones del dataset",
    "Estructura del dataset",
    "Tipos de datos",
    "Resumen estadístico",
    "Variables obligatorias",
    "Integridad de identificadores",
    "Valores faltantes (NA)",
    "Valores NaN",
    "Valores Inf y -Inf"
  ),
  evaluado = "SI",
  stringsAsFactors = FALSE
) # Componentes evaluados en la auditoría

print(alcance_auditoria)

# Limitaciones de la auditoría -------------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("LIMITACIONES DE LA AUDITORÍA\n")
cat(strrep("-", 90), "\n")

limitaciones_auditoria <- data.frame(
  componente = c(
    "Correlación entre variables",
    "Redundancia de atributos",
    "Multicolinealidad",
    "Valores atípicos (Outliers)",
    "Distribución estadística",
    "Importancia predictiva",
    "Selección de variables",
    "Importancia espacial",
    "Importancia temporal"
  ),
  evaluado = "NO",
  etapa = c(
    "Análisis exploratorio",
    "Ingeniería de variables",
    "Ingeniería de variables",
    "Análisis exploratorio",
    "Análisis exploratorio",
    "Modelado",
    "Ingeniería de variables",
    "Modelado GNN",
    "Modelado espacio-temporal"
  ),
  stringsAsFactors = FALSE
) # Componentes pendientes de evaluación

print(limitaciones_auditoria)

# BLOQUE 3. AUDITORÍA DEL CATÁLOGO ESPACIAL
# ----------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 3. AUDITORÍA DEL CATÁLOGO ESPACIAL\n")
cat(strrep("-", 90), "\n")

# Resumen general -------------------------------------------------------
cat(
  "Número de municipios :",
  format(nrow(municipios_panel), big.mark = ","),
  "\n"
)

cat(
  "Número de variables  :",
  ncol(municipios_panel),
  "\n"
)

cat(
  "Memoria utilizada    :",
  format(object.size(municipios_panel), units = "MB"),
  "\n"
)

# Estructura del dataset ------------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("ESTRUCTURA DEL DATASET\n")
cat(strrep("-", 90), "\n")

glimpse(municipios_panel)

# Tipos de datos --------------------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("TIPOS DE DATOS\n")
cat(strrep("-", 90), "\n")

tipos_variables <- data.frame(
  variable = names(municipios_panel),
  clase = sapply(
    municipios_panel,
    function(x) class(x)[1]
  ),
  row.names = NULL
) # Tipo principal de cada variable

print(tipos_variables)

# Resumen estadístico ---------------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("RESUMEN ESTADÍSTICO\n")
cat(strrep("-", 90), "\n")

class(municipios_panel)

sapply(
  municipios_panel,
  function(x) paste(class(x), collapse = ", ")
)

summary(
  municipios_panel |>
    select(
      -geometry
    )
) # Resumen estadístico de variables no espaciales

# Tipo de la geometría --------------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("TIPO DE GEOMETRÍA\n")
cat(strrep("-", 90), "\n")

cat(
  "Clase de geometry :",
  paste(class(municipios_panel$geometry), collapse = ", "),
  "\n"
)

# Variables obligatorias ------------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("VARIABLES OBLIGATORIAS\n")
cat(strrep("-", 90), "\n")

variables_obligatorias <- c(
  "cod_mpio",
  "MunNombre",
  "geometry"
)

variables_faltantes <- setdiff(
  variables_obligatorias,
  names(municipios_panel)
)

if(length(variables_faltantes) == 0){
  cat("Resultado: Todas las variables obligatorias están presentes.\n")
} else {
  cat("Resultado: Faltan variables obligatorias.\n")
  print(variables_faltantes)
}

# Integridad de identificadores -----------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("INTEGRIDAD DE IDENTIFICADORES\n")
cat(strrep("-", 90), "\n")

cat(
  "Municipios únicos :",
  dplyr::n_distinct(municipios_panel$cod_mpio),
  "\n"
)

cat(
  "Duplicados cod_mpio :",
  sum(duplicated(municipios_panel$cod_mpio)),
  "\n"
)

# Auditoría de valores faltantes ----------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("VALORES FALTANTES (NA)\n")
cat(strrep("-", 90), "\n")

na_resumen <- data.frame(
  variable = names(municipios_panel),
  n_na = colSums(is.na(municipios_panel)),
  pct_na = round(
    colSums(is.na(municipios_panel)) /
      nrow(municipios_panel) * 100,
    3
  )
) # Resumen de NA

na_resumen <- na_resumen |>
  arrange(desc(n_na))

print(na_resumen)

cat(
  "\nVariables con NA :",
  sum(na_resumen$n_na > 0),
  "\n"
)

cat(
  "Total NA :",
  sum(na_resumen$n_na),
  "\n"
)

# Auditoría de NaN ------------------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("VALORES NaN\n")
cat(strrep("-", 90), "\n")

nan_resumen <- sapply(
  municipios_panel,
  function(x){
    if(is.numeric(x)){
      sum(is.nan(x))
    } else {
      0
    }
  }
) # Contar NaN

print(nan_resumen[nan_resumen > 0])

cat(
  "Total NaN :",
  sum(nan_resumen),
  "\n"
)

# Auditoría de Inf ------------------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("VALORES Inf Y -Inf\n")
cat(strrep("-", 90), "\n")

inf_resumen <- sapply(
  municipios_panel,
  function(x){
    if(is.numeric(x)){
      sum(is.infinite(x))
    } else {
      0
    }
  }
) # Contar Inf

print(inf_resumen[inf_resumen > 0])

cat(
  "Total Inf :",
  sum(inf_resumen),
  "\n"
)

# BLOQUE 4. COMPARACIÓN DE VARIABLES COMUNES
# ----------------------------------
cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 4. COMPARACIÓN DE VARIABLES COMUNES\n")
cat(strrep("=", 90), "\n")

# Variables de cada base ------------------------------------------------
variables_gnn <- names(dataset_ganador_gnn) # Variables del dataset GNN
variables_spatial <- names(municipios_panel) # Variables del catálogo espacial

# Variables comunes -----------------------------------------------------
variables_comunes <- intersect(
  variables_gnn,
  variables_spatial
) # Variables presentes en ambas bases

cat(
  "Variables comunes :",
  length(variables_comunes),
  "\n\n"
)
print(variables_comunes)

# Variables exclusivas del dataset GNN ----------------------------------
variables_gnn_unicas <- setdiff(
  variables_gnn,
  variables_spatial
) # Variables exclusivas del dataset GNN
cat("\n")
cat("Variables exclusivas del dataset GNN :", length(variables_gnn_unicas), "\n\n")
print(variables_gnn_unicas)

# Variables exclusivas del catálogo espacial ----------------------------
variables_spatial_unicas <- setdiff(
  variables_spatial,
  variables_gnn
) # Variables exclusivas del catálogo espacial

cat("\n")
cat("Variables exclusivas del catálogo espacial :", length(variables_spatial_unicas), "\n\n")
print(variables_spatial_unicas)

# Compatibilidad de tipos ------------------------------------------------
comparacion_tipos <- data.frame(
  variable = variables_comunes,
  tipo_gnn = sapply(
    variables_comunes,
    function(v) class(dataset_ganador_gnn[[v]])[1]
  ),
  tipo_spatial = sapply(
    variables_comunes,
    function(v) class(municipios_panel[[v]])[1]
  ),
  stringsAsFactors = FALSE
) # Comparar tipos de datos

comparacion_tipos$compatible <-
  comparacion_tipos$tipo_gnn ==
  comparacion_tipos$tipo_spatial

cat("\n")
cat(strrep("-", 90), "\n")
cat("COMPATIBILIDAD DE VARIABLES COMUNES\n")
cat(strrep("-", 90), "\n")

print(comparacion_tipos)

# BLOQUE 5. VALIDACIÓN DE INTEGRIDAD REFERENCIAL
# ----------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 5. VALIDACIÓN DE INTEGRIDAD REFERENCIAL\n")
cat(strrep("-", 90), "\n")

# Municipios únicos -----------------------------------------------------
n_gnn <- dplyr::n_distinct(dataset_ganador_gnn$cod_mpio) # Municipios únicos del dataset GNN
n_spatial <- dplyr::n_distinct(municipios_panel$cod_mpio) # Municipios únicos del catálogo espacial
cat("Municipios únicos (Dataset GNN)      :", n_gnn, "\n")
cat("Municipios únicos (Catálogo espacial):", n_spatial, "\n")

# Duplicados ------------------------------------------------------------
dup_gnn <- dataset_ganador_gnn |>
  dplyr::distinct(cod_mpio) |>
  dplyr::count(cod_mpio) |>
  dplyr::filter(n > 1) # Duplicados en dataset GNN

dup_spatial <- municipios_panel |>
  dplyr::count(cod_mpio) |>
  dplyr::filter(n > 1) # Duplicados en catálogo espacial

cat("\n")
cat("Duplicados (Dataset GNN)      :", nrow(dup_gnn), "\n")
cat("Duplicados (Catálogo espacial):", nrow(dup_spatial), "\n")

# Municipios faltantes --------------------------------------------------
cod_gnn <- unique(dataset_ganador_gnn$cod_mpio) # Códigos del dataset GNN
cod_spatial <- unique(municipios_panel$cod_mpio) # Códigos del catálogo espacial
faltan_en_spatial <- setdiff(
  cod_gnn,
  cod_spatial
) # Presentes en GNN pero ausentes en el catálogo espacial

faltan_en_gnn <- setdiff(
  cod_spatial,
  cod_gnn
) # Presentes en el catálogo espacial pero ausentes en GNN

cat("\n")
cat("Municipios faltantes en el catálogo espacial :", length(faltan_en_spatial), "\n")
cat("Municipios faltantes en el dataset GNN        :", length(faltan_en_gnn), "\n")

# Diagnóstico -----------------------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("DIAGNÓSTICO\n")
cat(strrep("-", 90), "\n")

if(
  n_gnn == n_spatial &&
  length(faltan_en_spatial) == 0 &&
  length(faltan_en_gnn) == 0 &&
  nrow(dup_spatial) == 0
){
  
  cat("RESULTADO: INTEGRIDAD REFERENCIAL APROBADA\n")
  cat("Los 1.121 municipios del dataset GNN coinciden exactamente con los 1.121 municipios del catálogo espacial.\n")
  
} else {
  
  cat("RESULTADO: SE DETECTARON INCONSISTENCIAS EN LA INTEGRIDAD REFERENCIAL\n")
  
}

# BLOQUE 6. VALIDACIÓN DE CONSISTENCIA ENTRE VARIABLES COMUNES
# ----------------------------------
cat("BLOQUE 6. VALIDACIÓN DE CONSISTENCIA ENTRE VARIABLES COMUNES\n")
cat(strrep("-", 90), "\n")

# Preparar base de comparación ------------------------------------------
comparacion_variables <- dataset_ganador_gnn |>
  dplyr::distinct(
    cod_mpio,
    municipio,
    area_total
  ) |>
  dplyr::left_join(
    municipios_panel |>
      dplyr::select(
        cod_mpio,
        MunNombre,
        MunArea
      ),
    by = "cod_mpio"
  ) # Integrar variables comunes

# Comparación de nombres -------------------------------------------------

cat("\n")
cat(strrep("-", 90), "\n")
cat("COMPARACIÓN DE NOMBRES MUNICIPALES\n")
cat(strrep("-", 90), "\n")

comparacion_variables <- comparacion_variables |>
  dplyr::mutate(
    nombres_iguales = municipio == MunNombre
  ) # Comparar nombres

cat(
  "Municipios con nombres iguales :",
  sum(comparacion_variables$nombres_iguales),
  "\n"
)

cat(
  "Municipios con nombres diferentes :",
  sum(!comparacion_variables$nombres_iguales),
  "\n"
)

if(sum(!comparacion_variables$nombres_iguales) > 0){
  
  print(
    comparacion_variables |>
      dplyr::filter(!nombres_iguales) |>
      dplyr::select(
        cod_mpio,
        municipio,
        MunNombre
      )
  )
  
}

# Comparación de áreas ---------------------------------------------------

cat("\n")
cat(strrep("-", 90), "\n")
cat("COMPARACIÓN DE ÁREAS MUNICIPALES\n")
cat(strrep("-", 90), "\n")

comparacion_variables <- comparacion_variables |>
  dplyr::mutate(
    area_total_km2 = area_total / 100,
    diferencia_km2 = area_total_km2 - MunArea,
    diferencia_abs = abs(diferencia_km2),
    razon = area_total / MunArea,
    error_pct = abs(diferencia_km2) / MunArea * 100
  ) # Calcular indicadores

cat("Resumen del error porcentual\n\n")

print(summary(comparacion_variables$error_pct))

cat("\n")
cat("Correlación entre áreas : ")

correlacion <- cor(
  comparacion_variables$area_total_km2,
  comparacion_variables$MunArea,
  use = "complete.obs"
)

cat(round(correlacion, 6), "\n")

# Municipios con mayores diferencias ------------------------------------

cat("\n")
cat(strrep("-", 90), "\n")
cat("MUNICIPIOS CON MAYORES DIFERENCIAS\n")
cat(strrep("-", 90), "\n")

print(
  
  comparacion_variables |>
    
    dplyr::arrange(
      dplyr::desc(error_pct)
    ) |>
    
    dplyr::select(
      cod_mpio,
      municipio,
      area_total_km2,
      MunArea,
      diferencia_km2,
      error_pct
    ) |>
    
    head(20)
  
)

# Diagnóstico ------------------------------------------------------------

cat("\n")
cat(strrep("-", 90), "\n")
cat("DIAGNÓSTICO DEL BLOQUE 6\n")
cat(strrep("-", 90), "\n")

cat(
  "Correlación observada :",
  round(correlacion, 6),
  "\n"
)

if(correlacion >= 0.99){
  
  cat("Resultado: Las variables representan prácticamente la misma información espacial.\n")
  
} else if(correlacion >= 0.95){
  
  cat("Resultado: Las variables presentan alta consistencia, aunque existen diferencias localizadas.\n")
  
} else {
  
  cat("Resultado: Las variables representan información diferente y requieren revisión.\n")
  
}

cat("\n")

cat("Recomendación técnica:\n")
cat("- Conservar 'area_total' para el modelado GNN.\n")
cat("- Utilizar 'MunArea' únicamente como referencia cartográfica.\n")
cat("- Incorporar 'geometry' desde municipios_panel durante la integración final.\n")


# BLOQUE 7. VALIDACIÓN DE LA INFORMACIÓN ESPACIAL
# ----------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 7. VALIDACIÓN DE LA INFORMACIÓN ESPACIAL\n")
cat(strrep("-", 90), "\n")

# Variables espaciales disponibles --------------------------------------

variables_espaciales <- c(
  "geometry",
  "MunArea",
  "SHAPE_Area",
  "SHAPE_Leng"
) # Variables espaciales esperadas

validacion_variables <- data.frame(
  variable = variables_espaciales,
  disponible = variables_espaciales %in% names(municipios_panel)
) # Verificar disponibilidad

print(validacion_variables)

# Geometría -------------------------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("GEOMETRÍA\n")
cat(strrep("-", 90), "\n")

cat(
  "Registros con geometry :",
  sum(!is.na(municipios_panel$geometry)),
  "\n"
)

cat(
  "Registros sin geometry :",
  sum(is.na(municipios_panel$geometry)),
  "\n"
)

cat(
  "Clase de geometry :",
  paste(class(municipios_panel$geometry), collapse = ", "),
  "\n"
)

# Coordenadas disponibles -----------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("COORDENADAS GEOGRÁFICAS\n")
cat(strrep("-", 90), "\n")

if(all(c("latitud", "longitud") %in% names(dataset_ganador_gnn))){
  
  cat("Latitud : Disponible\n")
  cat("Longitud: Disponible\n")
  
} else {
  
  cat("Coordenadas geográficas incompletas\n")
  
}

# Variables útiles para el dashboard ------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("VARIABLES PARA DASHBOARD ESPACIAL\n")
cat(strrep("-", 90), "\n")

dashboard_variables <- c(
  "cod_mpio",
  "municipio",
  "departamento",
  "latitud",
  "longitud",
  "geometry"
)

dashboard_estado <- data.frame(
  variable = dashboard_variables,
  disponible = dashboard_variables %in% c(
    names(dataset_ganador_gnn),
    names(municipios_panel)
  )
)

print(dashboard_estado)

# Diagnóstico -----------------------------------------------------------
cat("\n")
cat(strrep("-", 90), "\n")
cat("DIAGNÓSTICO DEL BLOQUE 7\n")
cat(strrep("-", 90), "\n")

if(
  all(validacion_variables$disponible) &&
  sum(is.na(municipios_panel$geometry)) == 0
){
  
  cat("Resultado: La información espacial está disponible para la integración.\n")
  
} else {
  
  cat("Resultado: Existen elementos espaciales que requieren revisión.\n")
  
}

cat("\n")

cat("Estado para Python:\n")
cat("- Catálogo espacial disponible.\n")
cat("- Geometría almacenada en formato binario (arrow_binary).\n")
cat("- La reconstrucción del objeto espacial se realizará durante el pipeline en Python.\n")

# BLOQUE 8. INTEGRACIÓN DEL CATÁLOGO ESPACIAL
# ----------------------------------
cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 8. INTEGRACIÓN DEL CATÁLOGO ESPACIAL\n")
cat(strrep("=", 90), "\n")

# Seleccionar variables espaciales --------------------------------------
catalogo_espacial <- municipios_panel |>
  dplyr::select(
    cod_mpio,
    geometry
  ) # Catálogo espacial para integración

# Integrar información espacial -----------------------------------------
dataset_gnn_final <- dataset_ganador_gnn |>
  dplyr::left_join(
    catalogo_espacial,
    by = "cod_mpio"
  ) # Incorporar geometría

# Validación de la integración ------------------------------------------
cat(
  "Registros del dataset final :",
  format(nrow(dataset_gnn_final), big.mark = ","),
  "\n"
)

cat(
  "Variables del dataset final :",
  ncol(dataset_gnn_final),
  "\n"
)

cat(
  "Geometrías incorporadas :",
  sum(!is.na(dataset_gnn_final$geometry)),
  "\n"
)

cat(
  "Geometrías faltantes :",
  sum(is.na(dataset_gnn_final$geometry)),
  "\n"
)

cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 9. CERTIFICACIÓN DEL DATASET INTEGRADO\n")
cat(strrep("-", 90), "\n")

# Resumen general -------------------------------------------------------
cat("Registros              :", format(nrow(dataset_gnn_final), big.mark = ","), "\n")
cat("Variables              :", ncol(dataset_gnn_final), "\n")
cat("Municipios únicos      :", dplyr::n_distinct(dataset_gnn_final$cod_mpio), "\n")
cat("Años                   :", dplyr::n_distinct(dataset_gnn_final$anio), "\n")
cat("Panel ID únicos        :", dplyr::n_distinct(dataset_gnn_final$panel_id), "\n")

# Calidad del dataset ---------------------------------------------------
na_total <- sum(is.na(dataset_gnn_final)) # Total de NA
nan_total <- sum(
  sapply(
    dataset_gnn_final,
    function(x) if(is.numeric(x)) sum(is.nan(x)) else 0
  )
) # Total de NaN

inf_total <- sum(
  sapply(
    dataset_gnn_final,
    function(x) if(is.numeric(x)) sum(is.infinite(x)) else 0
  )
) # Total de Inf

geometry_total <- sum(!is.na(dataset_gnn_final$geometry)) # Geometrías disponibles
cat("Valores NA             :", na_total, "\n")
cat("Valores NaN            :", nan_total, "\n")
cat("Valores Inf            :", inf_total, "\n")
cat("Geometrías disponibles :", geometry_total, "\n")

# Certificación ---------------------------------------------------------
certificacion <- data.frame(
  criterio = c(
    "Registros preservados",
    "Municipios únicos",
    "Valores NA",
    "Valores NaN",
    "Valores Inf",
    "Geometrías disponibles"
  ),
  estado = c(
    nrow(dataset_gnn_final) == nrow(dataset_ganador_gnn),
    dplyr::n_distinct(dataset_gnn_final$cod_mpio) == 1121,
    na_total == 0,
    nan_total == 0,
    inf_total == 0,
    geometry_total == nrow(dataset_gnn_final)
  )
) # Resultado de la certificación

print(certificacion)

cat("\n")
if(all(certificacion$estado)){
  cat("RESULTADO: DATASET CERTIFICADO PARA EL PIPELINE GNN EN PYTHON\n")
} else {
  cat("RESULTADO: EL DATASET REQUIERE CORRECCIONES ANTES DEL MODELADO\n")
}

cat("\n")
cat(strrep("-", 90), "\n")
cat("BLOQUE 10. EXPORTACIÓN DEL DATASET FINAL\n")
cat(strrep("-", 90), "\n")

# Ruta de exportación ---------------------------------------------------
ruta_salida <- "data/processed/db_ganador" # Carpeta del dataset certificado

dir.create(ruta_salida, recursive = TRUE, showWarnings = FALSE) # Crear carpeta si no existe

archivo_salida <- file.path(
  ruta_salida,
  "dataset_gnn_certificado.parquet"
) # Archivo de salida

# Exportar dataset ------------------------------------------------------

arrow::write_parquet(
  dataset_gnn_final,
  archivo_salida
) # Exportar dataset certificado

# Confirmación ----------------------------------------------------------
cat("Archivo exportado :", archivo_salida, "\n")
cat(
  "Tamaño del archivo :",
  round(file.info(archivo_salida)$size / 1024^2, 2),
  "MB\n"
)
cat("Estado : Exportación completada correctamente.\n")