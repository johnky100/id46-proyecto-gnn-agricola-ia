# 09_preparacion_dataset_gnn_agricola.R
# DATASET GNN AGRÍCOLA PARA MODELADO

# Ruta del dataset certificado
ruta_dataset_certificado <- "D:/Proyectos_IA/proyecto-gnn-agricola/data/processed/r/dataset/dataset_gnn_certificado.parquet"

# Cargar dataset certificado
dataset_gnn_certificado <- arrow::read_parquet(
  ruta_dataset_certificado
)

cat("\nDataset certificado cargado correctamente.\n")
cat("Filas    :", nrow(dataset_gnn_certificado), "\n")
cat("Columnas :", ncol(dataset_gnn_certificado), "\n")

glimpse(dataset_gnn_certificado)

cat("BLOQUE 1. AUDITORÍA INICIAL DE DATA LEAKAGE\n")
cat("============================================================\n")

# Patrones de búsqueda
patrones_leakage <- c(
  "rend",
  "yield",
  "prod",
  "cosecha",
  "area",
  "log"
)

# Variables sospechosas
variables_sospechosas <- names(dataset_gnn_certificado)[
  grepl(
    paste(patrones_leakage, collapse = "|"),
    names(dataset_gnn_certificado),
    ignore.case = TRUE
  )
]

cat("\nVariables candidatas para revisión:\n")
print(variables_sospechosas)

cat("\n")
cat("BLOQUE 2. CORRELACIÓN CON LA VARIABLE OBJETIVO\n")
cat("-----------------------------------------------------\n")

# Variables numéricas (excepto la variable objetivo)
variables_numericas <- names(dataset_gnn_certificado)[
  sapply(dataset_gnn_certificado, is.numeric)
]

variables_numericas <- setdiff(
  variables_numericas,
  "log_rendimiento"
)

# Calcular correlación con la variable objetivo
correlaciones_target <- data.frame(
  variable = variables_numericas,
  correlacion = sapply(
    variables_numericas,
    function(x) cor(
      dataset_gnn_certificado[[x]],
      dataset_gnn_certificado$log_rendimiento,
      use = "complete.obs"
    )
  ),
  stringsAsFactors = FALSE
)

# Valor absoluto de la correlación
correlaciones_target$abs_correlacion <- abs(
  correlaciones_target$correlacion
)

# Ordenar de mayor a menor
correlaciones_target <- correlaciones_target |>
  dplyr::arrange(desc(abs_correlacion))

cat("\nVariables con mayor correlación respecto a log_rendimiento:\n\n")

print(correlaciones_target)

# Variables con correlación muy alta
correlaciones_altas <- correlaciones_target |>
  dplyr::filter(abs_correlacion >= 0.90)

cat("\n")
cat("Variables con |correlación| >= 0.90 :", nrow(correlaciones_altas), "\n\n")

print(correlaciones_altas)

cat("\n")
cat("BLOQUE 3. AUDITORÍA FUNCIONAL DE DATA LEAKAGE\n")
cat("-----------------------------------------------------\n")

# Variables candidatas para revisión funcional
variables_revision <- c(
  "rendimiento_promedio",
  "produccion_total",
  "log_produccion_total",
  "area_sembrada_total",
  "log_area_sembrada_total",
  "tasa_cosecha_promedio"
)

# Conservar únicamente las variables presentes
variables_revision <- intersect(
  variables_revision,
  names(dataset_gnn_certificado)
)

# Resumen de variables candidatas
auditoria_funcional <- data.frame(
  variable = variables_revision,
  existe = variables_revision %in% names(dataset_gnn_certificado),
  correlacion = correlaciones_target$correlacion[
    match(
      variables_revision,
      correlaciones_target$variable
    )
  ],
  stringsAsFactors = FALSE
)

print(auditoria_funcional)

cat("\n")
cat("BLOQUE 4. IMPORTANCIA PREDICTIVA DE LAS VARIABLES\n")
cat("-----------------------------------------------------\n")

# Variables predictoras
variables_modelo <- setdiff(
  variables_numericas,
  "log_rendimiento"
)

# Dataset para auditoría
datos_modelo <- dataset_gnn_certificado |>
  dplyr::select(
    dplyr::all_of(c("log_rendimiento", variables_modelo))
  )

# Entrenar Random Forest de auditoría
modelo_rf <- ranger::ranger(
  formula = log_rendimiento ~ .,
  data = datos_modelo,
  importance = "permutation",
  num.trees = 300,
  seed = 123
)

# Importancia de variables
importancia_variables <- data.frame(
  variable = names(modelo_rf$variable.importance),
  importancia = unname(modelo_rf$variable.importance),
  stringsAsFactors = FALSE
) |>
  dplyr::arrange(desc(importancia))

cat("\nRanking de importancia predictiva:\n\n")
print(importancia_variables)

# Variables con mayor importancia
top_variables <- importancia_variables |>
  dplyr::slice_head(n = 10)

cat("\nTop 10 variables más importantes:\n\n")
print(top_variables)

cat("\n")
cat("BLOQUE 5. DICTAMEN TÉCNICO DE VARIABLES CON DATA LEAKAGE\n")
cat("-----------------------------------------------------\n")

dictamen_leakage <- data.frame(
  variable = c(
    "rendimiento_promedio",
    "log_rendimiento",
    "produccion_total",
    "log_produccion_total",
    "area_sembrada_total",
    "log_area_sembrada_total",
    "tasa_cosecha_promedio"
  ),
  decision = c(
    "ELIMINAR",
    "CONSERVAR (TARGET)",
    "REVISAR",
    "REVISAR",
    "CONSERVAR",
    "CONSERVAR",
    "CONSERVAR"
  ),
  justificacion = c(
    "Variable original utilizada para construir el target.",
    "Variable objetivo del modelo.",
    "Variable predictora importante; verificar disponibilidad temporal.",
    "Transformación de la producción; verificar disponibilidad temporal.",
    "No existe evidencia de Data Leakage.",
    "No existe evidencia de Data Leakage.",
    "No existe evidencia de Data Leakage."
  ),
  stringsAsFactors = FALSE
)

print(dictamen_leakage)

cat("\nResumen del dictamen:\n")

print(
  table(dictamen_leakage$decision)
)

cat("\n")
cat("BLOQUE 6. CREACIÓN DEL DATASET CERTIFICADO CLEAN\n")
cat("-----------------------------------------------------\n")

# Variables identificadas con Data Leakage
variables_data_leakage <- c(
  "rendimiento_promedio"
)

# Crear versión limpia del dataset certificado
dataset_gnn_certificado_clean <- dataset_gnn_certificado |>
  dplyr::select(
    -dplyr::any_of(variables_data_leakage)
  )

cat("Variables eliminadas :", length(variables_data_leakage), "\n")
print(variables_data_leakage)

cat("\n")
cat("Dimensiones del dataset limpio\n")
cat("Filas    :", nrow(dataset_gnn_certificado_clean), "\n")
cat("Columnas :", ncol(dataset_gnn_certificado_clean), "\n")

# Verificar que las variables fueron eliminadas
variables_restantes <- intersect(
  variables_data_leakage,
  names(dataset_gnn_certificado_clean)
)

if (length(variables_restantes) == 0) {
  cat("\nOK - Variables con Data Leakage eliminadas correctamente.\n")
} else {
  stop("ERROR: Persisten variables con Data Leakage en el dataset.")
}

cat("\n")
cat("BLOQUE 7. CORRELACIÓN CON LA VARIABLE OBJETIVO (DATASET CLEAN)\n")
cat("-----------------------------------------------------\n")

# Variables numéricas (excepto la variable objetivo)
variables_numericas <- names(dataset_gnn_certificado_clean)[
  sapply(dataset_gnn_certificado_clean, is.numeric)
]

variables_numericas <- setdiff(
  variables_numericas,
  "log_rendimiento"
)

# Calcular correlación con la variable objetivo
correlaciones_target <- data.frame(
  variable = variables_numericas,
  correlacion = sapply(
    variables_numericas,
    function(x) {
      cor(
        dataset_gnn_certificado_clean[[x]],
        dataset_gnn_certificado_clean$log_rendimiento,
        use = "complete.obs"
      )
    }
  ),
  stringsAsFactors = FALSE
)

# Valor absoluto de la correlación
correlaciones_target$abs_correlacion <- abs(
  correlaciones_target$correlacion
)

# Ordenar de mayor a menor
correlaciones_target <- correlaciones_target |>
  dplyr::arrange(desc(abs_correlacion))

cat("\nVariables ordenadas por correlación con log_rendimiento:\n\n")
print(correlaciones_target)

# Variables con alta correlación
correlaciones_altas <- correlaciones_target |>
  dplyr::filter(abs_correlacion >= 0.90)

cat("\nNúmero de variables con |correlación| >= 0.90 :", nrow(correlaciones_altas), "\n\n")
print(correlaciones_altas)

cat("\n")
cat("BLOQUE 8. VALIDACIÓN DEL MODELO TRAS LA ELIMINACIÓN DEL DATA LEAKAGE\n")
cat("-----------------------------------------------------\n")

# Variables predictoras
variables_modelo <- setdiff(
  variables_numericas,
  "log_rendimiento"
)

# Dataset para auditoría
datos_modelo <- dataset_gnn_certificado_clean |>
  dplyr::select(
    dplyr::all_of(c("log_rendimiento", variables_modelo))
  )

# Entrenar Random Forest para auditoría
modelo_rf <- ranger::ranger(
  formula = log_rendimiento ~ .,
  data = datos_modelo,
  importance = "permutation",
  num.trees = 300,
  seed = 123
)

# Importancia de variables
importancia_variables <- data.frame(
  variable = names(modelo_rf$variable.importance),
  importancia = unname(modelo_rf$variable.importance),
  stringsAsFactors = FALSE
) |>
  dplyr::arrange(desc(importancia))

cat("\nRanking de importancia de variables:\n\n")
print(importancia_variables)

# Top 10 variables
top_variables <- importancia_variables |>
  dplyr::slice_head(n = 10)

cat("\nTop 10 variables más importantes:\n\n")
print(top_variables)

# Concentración de importancia en la variable principal
importancia_total <- sum(importancia_variables$importancia)

porcentaje_principal <- 100 * max(importancia_variables$importancia) / importancia_total

cat("\n")
cat("Importancia total           :", round(importancia_total, 4), "\n")
cat("Importancia variable líder  :", round(max(importancia_variables$importancia), 4), "\n")
cat("Porcentaje explicado por la variable líder :", round(porcentaje_principal, 2), "%\n")

if (porcentaje_principal >= 40) {
  
  cat("\n")
  cat("DICTAMEN: Existe una variable dominante. Revisar posible Data Leakage o disponibilidad temporal.\n")
  
} else {
  
  cat("\n")
  cat("DICTAMEN: La importancia se distribuye entre múltiples variables.\n")
  
}

cat("\n")
cat("BLOQUE 9. AUDITORÍA DE REDUNDANCIA ENTRE VARIABLES PREDICTORAS\n")
cat("-----------------------------------------------------\n")

# Matriz de correlación entre variables numéricas
matriz_correlacion <- cor(
  dataset_gnn_certificado_clean[, variables_numericas],
  use = "complete.obs"
)

# Convertir matriz a formato largo
correlaciones_predictores <- as.data.frame(
  as.table(matriz_correlacion),
  stringsAsFactors = FALSE
)

names(correlaciones_predictores) <- c(
  "variable_1",
  "variable_2",
  "correlacion"
)

# Eliminar autocorrelaciones
correlaciones_predictores <- correlaciones_predictores |>
  dplyr::filter(variable_1 != variable_2)

# Valor absoluto
correlaciones_predictores$abs_correlacion <- abs(
  correlaciones_predictores$correlacion
)

# Conservar un solo sentido del par
correlaciones_predictores <- correlaciones_predictores |>
  dplyr::rowwise() |>
  dplyr::mutate(
    par = paste(sort(c(variable_1, variable_2)), collapse = " | ")
  ) |>
  dplyr::ungroup() |>
  dplyr::distinct(par, .keep_all = TRUE)

# Correlaciones altas
correlaciones_altas_predictores <- correlaciones_predictores |>
  dplyr::filter(abs_correlacion >= 0.90) |>
  dplyr::arrange(desc(abs_correlacion))

cat("\nNúmero de pares con |correlación| >= 0.90 :", nrow(correlaciones_altas_predictores), "\n\n")

print(
  correlaciones_altas_predictores |>
    dplyr::select(
      variable_1,
      variable_2,
      correlacion
    )
)

cat("\n")
cat("BLOQUE 10. EVALUACIÓN DE VARIABLES ALTAMENTE CORRELACIONADAS\n")
cat("-----------------------------------------------------\n")

# Unir importancia predictiva de cada variable del par
evaluacion_correlacion <- correlaciones_altas_predictores |>
  dplyr::left_join(
    importancia_variables,
    by = c("variable_1" = "variable")
  ) |>
  dplyr::rename(
    importancia_1 = importancia
  ) |>
  dplyr::left_join(
    importancia_variables,
    by = c("variable_2" = "variable")
  ) |>
  dplyr::rename(
    importancia_2 = importancia
  )

print(
  evaluacion_correlacion |>
    dplyr::select(
      variable_1,
      variable_2,
      correlacion,
      importancia_1,
      importancia_2
    )
)

cat("\n")
cat("BLOQUE 11. ESTABILIDAD NUMÉRICA DEL DATASET\n")
cat("-----------------------------------------------------\n")

# Matriz de variables numéricas (sin el target)
matriz_predictores <- dataset_gnn_certificado_clean |>
  dplyr::select(
    dplyr::all_of(
      setdiff(
        variables_numericas,
        "log_rendimiento"
      )
    )
  ) |>
  as.matrix()

# Número de condición
numero_condicion <- kappa(
  matriz_predictores,
  exact = FALSE
)

cat("Número de condición :", round(numero_condicion, 2), "\n")

if (numero_condicion < 30) {
  
  cat("DICTAMEN: Colinealidad baja.\n")
  
} else if (numero_condicion < 100) {
  
  cat("DICTAMEN: Colinealidad moderada.\n")
  
} else {
  
  cat("DICTAMEN: Colinealidad alta. Revisar predictores.\n")
  
}


glimpse(dataset_gnn_certificado_clean)
glimpse(dataset_gnn_certificado)