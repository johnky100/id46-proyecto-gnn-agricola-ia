# 05_evaluacion_datasets_gnn.R

# ------------------- Bloque 1. Configuración General ------------------------

source(here::here("src", "R", "config", "00_packages.R")) # Cargar paquetes
source(here::here("src", "R", "config", "01_paths.R")) # Cargar rutas
source(here::here("src", "R", "config", "02_global_parameters.R")) # Cargar parámetros

cat("\n")
cat(strrep("=", 90), "\n")
cat("EVALUACIÓN DE DATASETS PARA MODELADO GNN\n")
cat(strrep("=", 90), "\n")
cat("Fecha      :", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Periodo    :", anio_inicio, "-", anio_fin, "\n")
cat("Target     :", variable_target, "\n")
cat("Semilla    :", seed_global, "\n")
cat(strrep("=", 90), "\n")

library(caret)

# ------------------- Bloque 2. Carga de Datasets ----------------------------
cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 2. CARGA DE DATASETS\n")
cat(strrep("=", 90), "\n")

# Cargar datasets candidatos -------------------------------------------------
panel_maestro <- arrow::read_parquet(
  "data/modelado/panel_gnn_maestro_51_variables.parquet"
) # Dataset maestro (51 variables)

panel_manual <- arrow::read_parquet(
  "data/modelado/panel_gnn_reducido_46_variables.parquet"
) # Dataset reducido manual (46 variables)

panel_auto <- arrow::read_parquet(
  "data/modelado/panel_gnn_reducido_auto.parquet"
) # Dataset reducido automático

# Organizar datasets en una lista --------------------------------------------
datasets_gnn <- list(
  Maestro = panel_maestro,
  Reducido_Manual = panel_manual,
  Reducido_Automatico = panel_auto
) # Lista de datasets candidatos

# Confirmar carga -------------------------------------------------------------
cat("\nDatasets cargados correctamente:\n\n")

for (nombre in names(datasets_gnn)) {
  cat(
    sprintf(
      "%-22s %6d filas x %3d variables\n",
      nombre,
      nrow(datasets_gnn[[nombre]]),
      ncol(datasets_gnn[[nombre]])
    )
  )
}

# ------------------- Bloque 3. Auditoría Inicial ----------------------------
auditar_dataset <- function(datos, nombre) {
  auditoria_na <- tibble(
    variable = names(datos),
    nulos = sapply(datos, \(x) sum(is.na(x))),
    porcentaje_na = round(
      sapply(datos, \(x) mean(is.na(x)) * 100),
      4
    )
  ) |> arrange(desc(nulos)) # Auditoría de valores NA
  
  cat("\n")
  cat(strrep("=", 90), "\n")
  cat(nombre, "\n")
  cat(strrep("=", 90), "\n")
  
  cat("Observaciones        :", nrow(datos), "\n")
  cat("Variables            :", ncol(datos), "\n")
  cat("Municipios           :", n_distinct(datos$cod_mpio), "\n")
  cat("Años                :", n_distinct(datos$anio), "\n")
  cat("Periodo             :", min(datos$anio), "-", max(datos$anio), "\n")
  cat("Variables con NA    :", sum(auditoria_na$nulos > 0), "\n")
  cat("Total valores NA    :", sum(auditoria_na$nulos), "\n")
  
  cat("\nTipos de datos\n")
  print(sapply(datos, class))
  
  variables_na <- auditoria_na |>
    filter(nulos > 0) # Variables con NA
  
  if (nrow(variables_na) == 0) {
    cat("\nNo se encontraron valores NA.\n")
    
  } else {
    cat("\nVariables con valores NA\n")
    print(variables_na)
  }
  
  invisible(auditoria_na)
}

for (nombre in names(datasets_gnn)) { # Auditar todos los datasets
  auditar_dataset(datasets_gnn[[nombre]], nombre)
}

# ------------------- Bloque 4. Caracterización Estadística --------------------------
calcular_estadisticas <- function(datos) {
  datos |>
    select(where(is.numeric), -anio) |>
    summarise(
      across(
        everything(),
        list(
          media = ~mean(.x),
          sd = ~sd(.x),
          cv = ~sd(.x) / abs(mean(.x)),
          min = ~min(.x),
          q25 = ~quantile(.x, 0.25),
          mediana = ~median(.x),
          q75 = ~quantile(.x, 0.75),
          max = ~max(.x),
          asimetria = ~e1071::skewness(.x),
          curtosis = ~e1071::kurtosis(.x)
        ),
        .names = "{.col}_{.fn}"
      )
    )
}

resumen_estadistico <- lapply(datasets_gnn, calcular_estadisticas) # Calcular estadísticas

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 4. CALIDAD ESTADÍSTICA\n")
cat(strrep("=", 90), "\n")

for (nombre in names(resumen_estadistico)) {
  cat("\n", nombre, "\n")
  cat(strrep("-", 90), "\n")
  print(resumen_estadistico[[nombre]])
}


# ------------------- Bloque 5. Calidad de Variables -------------------------
evaluar_variables <- function(datos, nombre) {
  datos_num <- datos |>
    select(where(is.numeric), -anio) # Variables numéricas
  correlacion <- cor(
    datos_num,
    use = "pairwise.complete.obs"
  ) # Matriz de correlación
  
  correlacion_alta <- which(
    abs(correlacion) > umbral_correlacion_alta &
      abs(correlacion) < 1,
    arr.ind = TRUE
  ) # Pares altamente correlacionados
  
  nzv <- caret::nearZeroVar(
    datos_num,
    saveMetrics = TRUE
  ) # Near Zero Variance
  
  variables_constantes <- names(datos_num)[
    sapply(datos_num, \(x) dplyr::n_distinct(x) == 1)
  ] # Variables constantes
  
  cat("\n")
  cat(strrep("=", 90), "\n")
  cat(nombre, "\n")
  cat(strrep("=", 90), "\n")
  
  cat("Variables numéricas        :", ncol(datos_num), "\n")
  cat("Correlaciones >",
      umbral_correlacion_alta,
      ":",
      nrow(correlacion_alta) / 2,
      "\n")
  cat("Near Zero Variance         :", sum(nzv$nzv), "\n")
  cat("Variables constantes       :", length(variables_constantes), "\n")
  
  invisible(
    list(
      correlacion = correlacion,
      correlacion_alta = correlacion_alta,
      nzv = nzv,
      constantes = variables_constantes
    )
  )
}

resultado_variables <- lapply(
  names(datasets_gnn),
  \(x) evaluar_variables(datasets_gnn[[x]], x)
)

names(resultado_variables) <- names(datasets_gnn)

# ------------------- Bloque 6. Benchmark Predictivo -------------------------
evaluar_rf <- function(datos, nombre) {
  predictores <- datos |>
    select(
      where(is.numeric),
      -all_of(variable_target)
    ) # Variables predictoras
  
  respuesta <- datos[[variable_target]] # Variable objetivo
  
  control <- trainControl(
    method = "cv",
    number = 5
  ) # Validación cruzada
  
  modelo <- train(
    x = predictores,
    y = respuesta,
    method = "rf",
    trControl = control,
    ntree = num_trees_rf,
    importance = TRUE
  ) # Entrenar Random Forest
  
  metricas <- tibble(
    dataset = nombre,
    RMSE = min(modelo$results$RMSE),
    Rsquared = max(modelo$results$Rsquared),
    MAE = min(modelo$results$MAE)
  ) # Métricas del modelo
  
  importancia <- varImp(modelo)$importance |>
    rownames_to_column("variable") |>
    arrange(desc(Overall)) # Importancia de variables
  
  list(
    metricas = metricas,
    importancia = importancia,
    modelo = modelo
  )
  
}

benchmark_rf <- vector("list", length(datasets_gnn)) # Inicializar lista
names(benchmark_rf) <- names(datasets_gnn)

for (i in seq_along(datasets_gnn)) {
  nombre <- names(datasets_gnn)[i]
  cat("\n")
  cat(strrep("-", 90), "\n")
  cat("Procesando:", nombre, "(", i, "de", length(datasets_gnn), ")\n")
  cat(strrep("-", 90), "\n")
  
  tiempo <- Sys.time() # Tiempo inicial
  
  benchmark_rf[[i]] <- evaluar_rf(
    datasets_gnn[[i]],
    nombre
  )
  
  cat(
    "Finalizado en",
    round(difftime(Sys.time(), tiempo, units = "secs"), 2),
    "segundos\n"
  )
}

metricas_rf <- purrr::map_dfr(
  benchmark_rf,
  "metricas"
) |>
  arrange(RMSE) |>
  mutate(
    ranking = row_number()
  ) # Ranking por desempeño

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 6. BENCHMARK PREDICTIVO\n")
cat(strrep("=", 90), "\n")

print(metricas_rf)

# ------------------- Bloque 7. Justificación Espacial para GNN --------------
datos <- panel_maestro # Dataset de referencia

vecinos <- spdep::knearneigh(
  as.matrix(datos |> distinct(cod_mpio, longitud, latitud) |> select(longitud, latitud)),
  k = k_vecinos_grafo
) # Vecinos KNN

lista_vecinos <- spdep::knn2nb(vecinos) # Lista de vecinos

pesos <- spdep::nb2listw(
  lista_vecinos,
  style = "W"
) # Matriz de pesos espaciales

moran_global <- spdep::moran.test(
  datos |>
    group_by(cod_mpio) |>
    summarise(
      rendimiento = mean(log_rendimiento)
    ) |>
    pull(rendimiento),
  pesos
) # Índice de Moran

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 7. JUSTIFICACIÓN ESPACIAL PARA GNN\n")
cat(strrep("=", 90), "\n")

print(moran_global)

# ------------------- Bloque 8. Selección del Dataset Ganador ----------------
cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 8. SELECCIÓN DEL DATASET GANADOR\n")
cat(strrep("=", 90), "\n")

ranking_rf <- metricas_rf |>
  mutate(
    ranking_rmse = rank(RMSE, ties.method = "min"),
    ranking_mae = rank(MAE, ties.method = "min"),
    ranking_r2 = rank(-Rsquared, ties.method = "min"),
    puntaje_total = ranking_rmse + ranking_mae + ranking_r2
  ) |>
  arrange(
    puntaje_total,
    RMSE,
    MAE,
    desc(Rsquared)
  ) # Ranking multicriterio

dataset_ganador_nombre <- ranking_rf$dataset[1] # Dataset con mejor desempeño

dataset_ganador <- datasets_gnn[[dataset_ganador_nombre]] # Dataset ganador

cat("\nRanking final\n\n")

print(ranking_rf)

cat("\n")
cat(strrep("-", 90), "\n")
cat("DATASET GANADOR\n")
cat(strrep("-", 90), "\n")

cat("Nombre       :", dataset_ganador_nombre, "\n")
cat("RMSE         :", round(ranking_rf$RMSE[1], 6), "\n")
cat("MAE          :", round(ranking_rf$MAE[1], 6), "\n")
cat("R²           :", round(ranking_rf$Rsquared[1], 6), "\n")
cat("Variables    :", ncol(dataset_ganador), "\n")
cat("Registros    :", nrow(dataset_ganador), "\n")
cat("Municipios   :", n_distinct(dataset_ganador$cod_mpio), "\n")
cat("Periodo      :", min(dataset_ganador$anio), "-", max(dataset_ganador$anio), "\n")

# ------------------- Bloque 9. Exportación de Resultados --------------------
dir.create(
  rutas$db_ganador,
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta de salida

write_parquet(
  dataset_ganador,
  file.path(
    rutas$db_ganador,
    "dataset_ganador_gnn.parquet"
  )
) # Exportar dataset ganador en Parquet

write_csv(
  dataset_ganador,
  file.path(
    rutas$db_ganador,
    "dataset_ganador_gnn.csv"
  )
) # Exportar dataset ganador en CSV

write_csv(
  ranking_rf,
  file.path(
    rutas$db_ganador,
    "ranking_datasets_gnn.csv"
  )
) # Exportar ranking de datasets

saveRDS(
  benchmark_rf,
  file.path(
    rutas$db_ganador,
    "benchmark_random_forest.rds"
  )
) # Guardar benchmark completo

cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 9. EXPORTACIÓN DE RESULTADOS\n")
cat(strrep("=", 90), "\n")
cat("Dataset ganador exportado correctamente.\n")
cat("Ubicación :", rutas$db_ganador, "\n")
cat(strrep("=", 90), "\n")

# ------------------- Backup del Script --------------------------------------
dir.create(
  here("src", "R", "backup"),
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta de backup

file.copy(
  from = here(
    "src",
    "R",
    "panel_maestro_gnn",
    "06_evaluacion_datasets_gnn.R"
  ),
  to = here(
    "src",
    "R",
    "backup",
    paste0(
      "06_evaluacion_datasets_gnn_",
      format(Sys.Date(), "%Y%m%d"),
      ".R"
    )
  ),
  overwrite = TRUE
) # Crear respaldo del script

# ------------------- Bloque 10. Resumen Final -------------------------------
cat("\n")
cat(strrep("=", 90), "\n")
cat("BLOQUE 10. RESUMEN FINAL\n")
cat(strrep("=", 90), "\n")

cat("Dataset ganador     :", dataset_ganador_nombre, "\n")
cat("Observaciones       :", nrow(dataset_ganador), "\n")
cat("Variables           :", ncol(dataset_ganador), "\n")
cat("Municipios          :", n_distinct(dataset_ganador$cod_mpio), "\n")
cat("Periodo             :", min(dataset_ganador$anio), "-", max(dataset_ganador$anio), "\n")
cat("Variable objetivo   :", variable_target, "\n")
cat("RMSE                :", round(ranking_rf$RMSE[1], 6), "\n")
cat("MAE                 :", round(ranking_rf$MAE[1], 6), "\n")
cat("R²                  :", round(ranking_rf$Rsquared[1], 6), "\n")
cat("Directorio salida   :", rutas$db_ganador, "\n")
cat("Fecha finalización  :", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat(strrep("=", 90), "\n")
cat("PROCESO FINALIZADO CORRECTAMENTE\n")
cat(strrep("=", 90), "\n")

stopifnot(sum(is.na(dataset_ganador)) == 0) # Validar ausencia de NA
stopifnot(n_distinct(dataset_ganador$cod_mpio) == 1121) # Validar municipios
stopifnot(n_distinct(dataset_ganador$anio) == length(anios_pipeline)) # Validar cobertura temporal
stopifnot(variable_target %in% names(dataset_ganador)) # Validar variable objetivo