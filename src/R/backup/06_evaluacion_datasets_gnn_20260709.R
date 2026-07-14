# 05_evaluacion_datasets_gnn.R

# ------------------- Bloque 1. Configuración General ------------------
source(here::here("config_R", "00_packages.R")) # Cargar paquetes
source(here::here("config_R", "01_paths.R")) # Cargar rutas
source(here::here("config_R", "02_global_parameters.R")) # Cargar parámetros

# ------------------- Bloque 2. Carga Datasets ------------------
panel_maestro <- arrow::read_parquet(
  "data/modelado/panel_gnn_maestro_51_variables.parquet"
) # Leer dataset maestro

panel_manual <- arrow::read_parquet(
  "data/modelado/panel_gnn_reducido_46_variables.parquet"
) # Leer dataset reducido manual

panel_auto <- arrow::read_parquet(
  "data/modelado/panel_gnn_reducido_auto.parquet"
) # Leer dataset reducido automático

# ------------------- Bloque 3. Comparación estructural ------------------
comparacion_estructura <- tibble::tibble(
  dataset = c(
    "Maestro",
    "Reducido_Manual",
    "Reducido_Automatico"
  ),
  variables = c(
    ncol(panel_maestro),
    ncol(panel_manual),
    ncol(panel_auto)
  ),
  registros = c(
    nrow(panel_maestro),
    nrow(panel_manual),
    nrow(panel_auto)
  ),
  na_totales = c(
    sum(is.na(panel_maestro)),
    sum(is.na(panel_manual)),
    sum(is.na(panel_auto))
  )
) # Comparación estructural

# ------------------- Bloque 4. Calidad Estadística ------------------
# Evaluación de distribución y variabilidad
# ============================================================

calcular_metricas_estadisticas <- function(dataset){
  
  variables_numericas <- names(dataset)[
    sapply(
      dataset,
      is.numeric
    )
  ] # Identificar variables numéricas
  
  datos_numericos <- as.data.frame(
    dataset[, ..variables_numericas]
  ) # Convertir a data.frame
  
  tibble::tibble(
    variable = variables_numericas,
    media = sapply(
      datos_numericos,
      mean,
      na.rm = TRUE
    ),
    desviacion = sapply(
      datos_numericos,
      sd,
      na.rm = TRUE
    ),
    coef_variacion = sapply(
      datos_numericos,
      \(x){
        sd(
          x,
          na.rm = TRUE
        ) /
          abs(
            mean(
              x,
              na.rm = TRUE
            )
          )
      }
    ),
    asimetria = sapply(
      datos_numericos,
      moments::skewness,
      na.rm = TRUE
    ),
    curtosis = sapply(
      datos_numericos,
      moments::kurtosis,
      na.rm = TRUE
    )
  )
  
} # Función métricas estadísticas

# Aplicar a los 3 dataset
metricas_maestro <- calcular_metricas_estadisticas(
  panel_maestro
) # Métricas dataset maestro

metricas_manual <- calcular_metricas_estadisticas(
  panel_manual
) # Métricas dataset reducido manual

metricas_auto <- calcular_metricas_estadisticas(
  panel_auto
) # Métricas dataset reducido automático

# Resumen ejecutivo
resumen_calidad <- tibble::tibble(
  dataset = c(
    "Maestro",
    "Reducido_Manual",
    "Reducido_Automatico"
  ),
  cv_promedio = c(
    mean(metricas_maestro$coef_variacion, na.rm = TRUE),
    mean(metricas_manual$coef_variacion, na.rm = TRUE),
    mean(metricas_auto$coef_variacion, na.rm = TRUE)
  ),
  asimetria_promedio = c(
    mean(abs(metricas_maestro$asimetria), na.rm = TRUE),
    mean(abs(metricas_manual$asimetria), na.rm = TRUE),
    mean(abs(metricas_auto$asimetria), na.rm = TRUE)
  ),
  curtosis_promedio = c(
    mean(metricas_maestro$curtosis, na.rm = TRUE),
    mean(metricas_manual$curtosis, na.rm = TRUE),
    mean(metricas_auto$curtosis, na.rm = TRUE)
  )
) # Resumen calidad estadística

print(
  resumen_calidad
) # Mostrar resumen


# ------------------- Bloque 5. Correlación Residual ------------------
# Evaluación de redundancia entre variables
# ============================================================

calcular_correlacion_residual <- function(dataset){
  
  variables_numericas <- names(dataset)[
    sapply(
      dataset,
      is.numeric
    )
  ] # Seleccionar variables numéricas
  
  matriz_cor <- cor(
    as.data.frame(
      dataset[, ..variables_numericas]
    ),
    use = "pairwise.complete.obs"
  ) # Matriz de correlaciones
  
  matriz_abs <- abs(
    matriz_cor
  ) # Correlaciones absolutas
  
  matriz_abs[
    lower.tri(
      matriz_abs,
      diag = TRUE
    )
  ] <- NA # Eliminar diagonal e inferiores
  
  correlaciones <- as.vector(
    matriz_abs
  )
  
  correlaciones <- correlaciones[
    !is.na(correlaciones)
  ] # Vector limpio
  
  tibble::tibble(
    correlacion_promedio = mean(
      correlaciones
    ),
    correlacion_mediana = median(
      correlaciones
    ),
    correlacion_maxima = max(
      correlaciones
    ),
    pares_cor_80 = sum(
      correlaciones > 0.80
    ),
    pares_cor_90 = sum(
      correlaciones > 0.90
    ),
    pares_cor_95 = sum(
      correlaciones > 0.95
    )
  )
  
} # Función evaluación correlación residual

# Aplicar a cada dataset
resultado_cor_maestro <- calcular_correlacion_residual(
  panel_maestro
) # Dataset maestro

resultado_cor_manual <- calcular_correlacion_residual(
  panel_manual
) # Dataset reducido manual

resultado_cor_auto <- calcular_correlacion_residual(
  panel_auto
) # Dataset reducido automático

# Construir tabla comparativa:
comparacion_correlacion <- dplyr::bind_rows(
  Maestro = resultado_cor_maestro,
  Reducido_Manual = resultado_cor_manual,
  Reducido_Automatico = resultado_cor_auto,
  .id = "dataset"
)

print(comparacion_correlacion) # Comparación correlación residual

# identificar exactamente qué pares siguen siendo problemáticos
extraer_correlaciones_altas <- function(
    dataset,
    umbral = 0.90
){
  
  variables_numericas <- names(dataset)[
    sapply(
      dataset,
      is.numeric
    )
  ]
  
  matriz_cor <- cor(
    as.data.frame(
      dataset[, ..variables_numericas]
    ),
    use = "pairwise.complete.obs"
  )
  
  indice <- which(
    abs(matriz_cor) > umbral &
      abs(matriz_cor) < 1,
    arr.ind = TRUE
  )
  
  tibble::tibble(
    variable_1 = rownames(
      matriz_cor
    )[indice[,1]],
    variable_2 = colnames(
      matriz_cor
    )[indice[,2]],
    correlacion = matriz_cor[
      indice
    ]
  ) |>
    distinct()
}

# Aplicar:
cor_altas_maestro <- extraer_correlaciones_altas(
  panel_maestro
)

cor_altas_manual <- extraer_correlaciones_altas(
  panel_manual
)

cor_altas_auto <- extraer_correlaciones_altas(
  panel_auto
)

# Inspreccionar
ncol(panel_maestro)
ncol(panel_manual)
ncol(panel_auto)

setdiff(
  names(panel_manual),
  names(panel_auto)
)

# ------------------- Bloque 6. Evaluación Near Zero Variance ------------------
# Comparación de variables con baja variabilidad
# ============================================================
evaluar_nzv <- function(dataset){
  
  resultado <- caret::nearZeroVar(
    as.data.frame(dataset),
    saveMetrics = TRUE
  ) # Calcular métricas NZV
  
  resultado |>
    tibble::rownames_to_column(
      "variable"
    ) |>
    summarise(
      variables_totales = n(),
      variables_nzv = sum(nzv),
      variables_zero_var = sum(zeroVar),
      porcentaje_nzv = round(
        100 * sum(nzv) / n(),
        2
      )
    )
  
} # Función evaluación NZV

# saber exactamente cuáles variables siguen siendo problemáticas.
extraer_nzv <- function(dataset){
  
  caret::nearZeroVar(
    as.data.frame(dataset),
    saveMetrics = TRUE
  ) |>
    tibble::rownames_to_column(
      "variable"
    ) |>
    filter(
      nzv == TRUE
    ) |>
    select(
      variable,
      freqRatio,
      percentUnique,
      zeroVar,
      nzv
    )
  
} # Extraer variables NZV

# Aplicar
nzv_maestro <- extraer_nzv(
  panel_maestro
) # NZV maestro

nzv_manual <- extraer_nzv(
  panel_manual
) # NZV manual

nzv_auto <- extraer_nzv(
  panel_auto
) # NZV automático

# Mostrar
cat(
  "\nDataset Maestro\n"
) # Encabezado

print(
  nzv_maestro
) # Mostrar NZV

cat(
  "\nDataset Reducido Manual\n"
) # Encabezado

print(
  nzv_manual
) # Mostrar NZV

cat(
  "\nDataset Reducido Automático\n"
) # Encabezado

print(
  nzv_auto
) # Mostrar NZV



# ------------------- Bloque 7. Importancia Predictiva ------------------
# Benchmark Random Forest para selección de dataset
# ============================================================
evaluar_random_forest <- function(
    dataset,
    nombre_dataset
){
  
  set.seed(
    seed_global
  ) # Reproducibilidad
  
  variables_excluir <- c(
    "cod_mpio",
    "municipio",
    "panel_id",
    "departamento"
  ) # Variables identificadoras
  
  datos_modelo <- dataset |>
    dplyr::select(
      -any_of(
        variables_excluir
      )
    ) # Dataset modelado
  
  indice_train <- sample(
    seq_len(
      nrow(datos_modelo)
    ),
    size = floor(
      0.80 * nrow(datos_modelo)
    )
  ) # División entrenamiento
  
  train <- datos_modelo[
    indice_train,
  ] # Entrenamiento
  
  test <- datos_modelo[
    -indice_train,
  ] # Validación
  
  modelo <- ranger::ranger(
    log_rendimiento ~ .,
    data = train,
    num.trees = 500,
    importance = "permutation",
    seed = seed_global
  ) # Entrenar Random Forest
  
  predicciones <- predict(
    modelo,
    data = test
  )$predictions # Predicciones
  
  rmse <- sqrt(
    mean(
      (
        test$log_rendimiento -
          predicciones
      )^2
    )
  ) # RMSE
  
  mae <- mean(
    abs(
      test$log_rendimiento -
        predicciones
    )
  ) # MAE
  
  r2 <- cor(
    test$log_rendimiento,
    predicciones
  )^2 # R²
  
  importancia <- tibble::tibble(
    variable = names(
      modelo$variable.importance
    ),
    importancia = as.numeric(
      modelo$variable.importance
    )
  ) |>
    arrange(
      desc(importancia)
    ) # Ranking variables
  
  list(
    resumen = tibble::tibble(
      dataset = nombre_dataset,
      RMSE = rmse,
      MAE = mae,
      R2 = r2,
      variables = ncol(dataset)
    ),
    importancia = importancia
  )
  
} # Función benchmark RF

# Aplicar a los tres datasets
resultado_rf_maestro <- evaluar_random_forest(
  panel_maestro,
  "Maestro"
) # Dataset maestro

resultado_rf_manual <- evaluar_random_forest(
  panel_manual,
  "Reducido_Manual"
) # Dataset reducido manual

resultado_rf_auto <- evaluar_random_forest(
  panel_auto,
  "Reducido_Automatico"
) # Dataset reducido automático

# Comparación
comparacion_rf <- dplyr::bind_rows(
  resultado_rf_maestro$resumen,
  resultado_rf_manual$resumen,
  resultado_rf_auto$resumen
) |>
  arrange(
    desc(R2)
  ) # Ranking predictivo

print(comparacion_rf) # Mostrar resultados

# Top 20 variables del ganador:
resultado_rf_manual$importancia |>
head(20)

# o si gana otro dataset:
resultado_rf_maestro$importancia |>
head(20)

resultado_rf_auto$importancia |>
head(20)

# ------------------- Subbloque 7.1 Auditoría Leakage ------------------
# Evaluación sin variables derivadas del rendimiento
# ============================================================
variables_leakage <- c(
  "rendimiento_promedio",
  "log_produccion_total",
  "produccion_total",
  "area_sembrada_total",
  "area_cosechada_total"
) # Variables potencialmente asociadas al target

# Función de evaluación
evaluar_random_forest_sin_leakage <- function(
    dataset,
    nombre_dataset
){
  
  set.seed(
    seed_global
  ) # Reproducibilidad
  
  variables_excluir <- c(
    "cod_mpio",
    "municipio",
    "panel_id",
    "departamento",
    variables_leakage
  ) # Excluir identificadores y variables leakage
  
  datos_modelo <- dataset |>
    dplyr::select(
      -any_of(
        variables_excluir
      )
    ) # Dataset limpio
  
  indice_train <- sample(
    seq_len(
      nrow(datos_modelo)
    ),
    size = floor(
      0.80 * nrow(datos_modelo)
    )
  ) # División entrenamiento
  
  train <- datos_modelo[
    indice_train,
  ]
  
  test <- datos_modelo[
    -indice_train,
  ]
  
  modelo <- ranger::ranger(
    log_rendimiento ~ .,
    data = train,
    num.trees = 500,
    importance = "permutation",
    seed = seed_global
  ) # Entrenar RF
  
  predicciones <- predict(
    modelo,
    data = test
  )$predictions
  
  tibble::tibble(
    dataset = nombre_dataset,
    RMSE = sqrt(
      mean(
        (test$log_rendimiento - predicciones)^2
      )
    ),
    MAE = mean(
      abs(
        test$log_rendimiento - predicciones
      )
    ),
    R2 = cor(
      test$log_rendimiento,
      predicciones
    )^2,
    variables = ncol(datos_modelo)
  )
  
} # Función benchmark sin leakage

# Aplicar a los tres datasets
resultado_sin_leak_maestro <- evaluar_random_forest_sin_leakage(
  panel_maestro,
  "Maestro"
)

resultado_sin_leak_manual <- evaluar_random_forest_sin_leakage(
  panel_manual,
  "Reducido_Manual"
)

resultado_sin_leak_auto <- evaluar_random_forest_sin_leakage(
  panel_auto,
  "Reducido_Automatico"
)

# Comparación final
comparacion_sin_leakage <- dplyr::bind_rows(
  resultado_sin_leak_maestro,
  resultado_sin_leak_manual,
  resultado_sin_leak_auto
) |>
  arrange(
    desc(R2)
  )

print(
  comparacion_sin_leakage
) # Benchmark sin leakage



# ------------------- Bloque 8. Evaluación Espacial Comparativa ------------------
# Comparación de dependencia espacial entre datasets candidatos GNN
# ============================================================

library(sf)
library(spdep)

# ------------------- Subbloque 8.0 Parámetros Espaciales ------------------
k_auditoria <- 5 # Vecinos KNN para Moran y LISA


# ------------------- Subbloque 8.1 Función Evaluación Espacial ------------------
evaluar_espacialidad <- function(
    dataset,
    nombre_dataset,
    anio_referencia = NULL,
    k = k_auditoria
){
  
  if(
    is.null(anio_referencia)
  ){
    
    anio_referencia <- max(
      dataset$anio,
      na.rm = TRUE
    )
    
  } # Último año disponible
  
  panel_espacial <- dataset |>
    dplyr::filter(
      anio == anio_referencia
    ) |>
    dplyr::select(
      cod_mpio,
      municipio,
      latitud,
      longitud,
      log_rendimiento
    ) |>
    distinct() # Corte transversal espacial
  
  panel_sf <- sf::st_as_sf(
    panel_espacial,
    coords = c(
      "longitud",
      "latitud"
    ),
    crs = 4326
  ) # Crear objeto espacial
  
  coords <- sf::st_coordinates(
    panel_sf
  ) # Extraer coordenadas
  
  vecinos_knn <- spdep::knearneigh(
    coords,
    k = k
  ) # Construir KNN
  
  vecinos_nb <- spdep::knn2nb(
    vecinos_knn
  ) # Convertir a nb
  
  pesos_knn <- spdep::nb2listw(
    vecinos_nb,
    style = "W",
    zero.policy = TRUE
  ) # Matriz espacial normalizada
  
  moran_global <- spdep::moran.test(
    panel_sf$log_rendimiento,
    pesos_knn,
    zero.policy = TRUE
  ) # Moran global
  
  set.seed(
    seed_global
  ) # Reproducibilidad
  
  moran_perm <- spdep::moran.mc(
    panel_sf$log_rendimiento,
    pesos_knn,
    nsim = 999,
    zero.policy = TRUE
  ) # Moran por permutaciones
  
  lisa_local <- spdep::localmoran(
    panel_sf$log_rendimiento,
    pesos_knn,
    zero.policy = TRUE
  ) # LISA local
  
  panel_sf$lisa_i <- lisa_local[, 1]
  panel_sf$lisa_p <- lisa_local[, 5]
  
  media_global <- mean(
    panel_sf$log_rendimiento,
    na.rm = TRUE
  ) # Media global
  
  lag_espacial <- spdep::lag.listw(
    pesos_knn,
    panel_sf$log_rendimiento,
    zero.policy = TRUE
  ) # Lag espacial
  
  panel_sf$cluster_lisa <- dplyr::case_when(
    
    panel_sf$log_rendimiento >= media_global &
      lag_espacial >= media_global &
      panel_sf$lisa_p < 0.05 ~ "High-High",
    
    panel_sf$log_rendimiento < media_global &
      lag_espacial < media_global &
      panel_sf$lisa_p < 0.05 ~ "Low-Low",
    
    panel_sf$log_rendimiento >= media_global &
      lag_espacial < media_global &
      panel_sf$lisa_p < 0.05 ~ "High-Low",
    
    panel_sf$log_rendimiento < media_global &
      lag_espacial >= media_global &
      panel_sf$lisa_p < 0.05 ~ "Low-High",
    
    TRUE ~ "No_Significativo"
    
  ) # Clasificación LISA
  
  resumen <- tibble::tibble(
    dataset = nombre_dataset,
    anio = anio_referencia,
    municipios = nrow(panel_sf),
    k_vecinos = k,
    moran_i = as.numeric(
      moran_global$estimate[1]
    ),
    p_value_moran = moran_global$p.value,
    p_value_perm = moran_perm$p.value,
    high_high = sum(
      panel_sf$cluster_lisa == "High-High"
    ),
    low_low = sum(
      panel_sf$cluster_lisa == "Low-Low"
    ),
    high_low = sum(
      panel_sf$cluster_lisa == "High-Low"
    ),
    low_high = sum(
      panel_sf$cluster_lisa == "Low-High"
    )
  ) # Resumen ejecutivo
  
  list(
    resumen = resumen,
    lisa = panel_sf,
    moran = moran_global,
    moran_perm = moran_perm
  )
  
} # Función evaluación espacial


# ------------------- Subbloque 8.2 Evaluación Comparativa ------------------
espacial_maestro <- evaluar_espacialidad(
  panel_maestro,
  "Maestro"
) # Dataset maestro

espacial_manual <- evaluar_espacialidad(
  panel_manual,
  "Reducido_Manual"
) # Dataset reducido manual

espacial_auto <- evaluar_espacialidad(
  panel_auto,
  "Reducido_Automatico"
) # Dataset reducido automático


# ------------------- Subbloque 8.3 Comparación Moran y LISA ------------------
comparacion_espacial <- dplyr::bind_rows(
  espacial_maestro$resumen,
  espacial_manual$resumen,
  espacial_auto$resumen
) |>
  arrange(
    desc(moran_i)
  ) # Ranking espacial

print(comparacion_espacial) # Mostrar comparación

# ------------------- Subbloque 8.4 Ranking Espacial ------------------
comparacion_espacial |>
  dplyr::select(
    dataset,
    moran_i,
    p_value_moran,
    p_value_perm,
    high_high,
    low_low,
    high_low,
    low_high
  ) |>
  print()

# ------------------- Bloque 9. Evaluación Temporal Comparativa ------------------
# Justificación temporal para modelado GNN espacio-temporal
# ============================================================
# ------------------- Subbloque 9.1 Función Evaluación Temporal ------------------
evaluar_temporalidad <- function(
    dataset,
    nombre_dataset
){
  
  serie_anual <- dataset |>
    dplyr::group_by(
      anio
    ) |>
    dplyr::summarise(
      log_rendimiento = mean(
        log_rendimiento,
        na.rm = TRUE
      ),
      .groups = "drop"
    ) # Serie anual promedio nacional
  
  acf_obj <- acf(
    serie_anual$log_rendimiento,
    plot = FALSE
  ) # ACF
  
  pacf_obj <- pacf(
    serie_anual$log_rendimiento,
    plot = FALSE
  ) # PACF
  
  persistencia_lag1 <- cor(
    serie_anual$log_rendimiento[-1],
    serie_anual$log_rendimiento[-nrow(serie_anual)]
  ) # Correlación lag 1
  
  persistencia_lag2 <- cor(
    serie_anual$log_rendimiento[-c(1,2)],
    serie_anual$log_rendimiento[-c(
      nrow(serie_anual)-1,
      nrow(serie_anual)
    )]
  ) # Correlación lag 2
  
  tibble::tibble(
    dataset = nombre_dataset,
    anios = nrow(serie_anual),
    persistencia_lag1 = persistencia_lag1,
    persistencia_lag2 = persistencia_lag2,
    acf_lag1 = acf_obj$acf[2],
    acf_lag2 = acf_obj$acf[3],
    pacf_lag1 = pacf_obj$acf[1],
    pacf_lag2 = pacf_obj$acf[2]
  )
  
} # Función temporal

# ------------------- Subbloque 9.2 Evaluación Comparativa ------------------
temporal_maestro <- evaluar_temporalidad(
  panel_maestro,
  "Maestro"
) # Dataset maestro

temporal_manual <- evaluar_temporalidad(
  panel_manual,
  "Reducido_Manual"
) # Dataset reducido manual

temporal_auto <- evaluar_temporalidad(
  panel_auto,
  "Reducido_Automatico"
) # Dataset reducido automático

# ------------------- Subbloque 9.3 Comparación Temporal ------------------
comparacion_temporal <- dplyr::bind_rows(
  temporal_maestro,
  temporal_manual,
  temporal_auto
)

print(
  comparacion_temporal
) # Mostrar comparación

# ------------------- Subbloque 9.4 Serie Nacional ------------------
# ------------------- Subbloque 9.4 Serie Nacional Tabular ------------------
serie_nacional <- panel_manual |>
  dplyr::group_by(
    anio
  ) |>
  dplyr::summarise(
    log_rendimiento = mean(
      log_rendimiento,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) |>
  dplyr::arrange(
    anio
  ) |>
  dplyr::mutate(
    cambio_absoluto = log_rendimiento -
      dplyr::lag(log_rendimiento),
    cambio_porcentual = 100 * (
      log_rendimiento /
        dplyr::lag(log_rendimiento) - 1
    )
  ) # Variación anual

print(
  serie_nacional,
  n = Inf
) # Mostrar evolución temporal



serie_nacional <- panel_manual |>
  dplyr::group_by(
    anio
  ) |>
  dplyr::summarise(
    log_rendimiento = mean(
      log_rendimiento,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) # Serie promedio nacional

ggplot2::ggplot(
  serie_nacional,
  ggplot2::aes(
    x = anio,
    y = log_rendimiento
  )
) +
  ggplot2::geom_line() +
  ggplot2::geom_point() +
  ggplot2::labs(
    title = "Evolución Nacional del Log Rendimiento",
    x = "Año",
    y = "Log Rendimiento"
  )

# ------------------- Subbloque 9.5 ACF ------------------
acf_tabla <- acf(
  serie_nacional$log_rendimiento,
  plot = FALSE
)

tabla_acf <- tibble::tibble(
  lag = seq_along(acf_tabla$acf) - 1,
  acf = as.numeric(acf_tabla$acf)
)

print(
  tabla_acf,
  n = Inf
) # Mostrar autocorrelaciones

# ------------------- Subbloque 9.6 PACF Tabular ------------------
pacf_obj <- pacf(
  serie_nacional$log_rendimiento,
  plot = FALSE
) # Calcular PACF sin gráfico

tabla_pacf <- tibble::tibble(
  lag = seq_along(
    pacf_obj$acf
  ),
  pacf = as.numeric(
    pacf_obj$acf
  )
) # Construir tabla PACF

print(
  tabla_pacf,
  n = Inf
) # Mostrar PACF completa

# ------------------- Subbloque 9.6.1 PACF Significativa ------------------
limite_significancia <- 1.96 / sqrt(
  nrow(serie_nacional)
) # Aproximación intervalo 95%

tabla_pacf <- tabla_pacf |>
  dplyr::mutate(
    significativo = abs(
      pacf
    ) > limite_significancia
  ) # Marcar rezagos significativos

print(
  tabla_pacf,
  n = Inf
) # Mostrar resultados

# ------------------- Subbloque 9.6.2 Resumen PACF ------------------
tabla_pacf |>
  dplyr::filter(
    significativo
  ) |>
  dplyr::arrange(
    desc(abs(pacf))
  )

# ------------------- Subbloque 9.7 Tabla Consolidada ------------------
acf_obj <- acf(
  serie_nacional$log_rendimiento,
  plot = FALSE
)

pacf_obj <- pacf(
  serie_nacional$log_rendimiento,
  plot = FALSE
)

tabla_dependencia <- tibble::tibble(
  lag = seq_len(
    min(
      length(acf_obj$acf) - 1,
      length(pacf_obj$acf)
    )
  ),
  acf = as.numeric(
    acf_obj$acf[-1]
  )[1:min(
    length(acf_obj$acf) - 1,
    length(pacf_obj$acf)
  )],
  pacf = as.numeric(
    pacf_obj$acf
  )[1:min(
    length(acf_obj$acf) - 1,
    length(pacf_obj$acf)
  )]
)

print(
  tabla_dependencia,
  n = Inf
) # Comparar ACF y PACF

# ------------------- Bloque 9.8 Persistencia Temporal Municipal ------------------
# Evaluación de dependencia temporal municipio a municipio
# ============================================================
persistencia_municipal <- panel_manual |>
  dplyr::arrange(
    cod_mpio,
    anio
  ) |>
  dplyr::group_by(
    cod_mpio
  ) |>
  dplyr::summarise(
    observaciones = sum(
      !is.na(log_rendimiento)
    ),
    corr_lag1 = cor(
      log_rendimiento[-1],
      log_rendimiento[-length(log_rendimiento)],
      use = "complete.obs"
    ),
    .groups = "drop"
  ) # Correlación temporal por municipio

# ------------------- Subbloque 9.8.1 Resumen Ejecutivo ------------------
resumen_persistencia <- persistencia_municipal |>
  summarise(
    municipios = n(),
    persistencia_media = mean(
      corr_lag1,
      na.rm = TRUE
    ),
    persistencia_mediana = median(
      corr_lag1,
      na.rm = TRUE
    ),
    persistencia_sd = sd(
      corr_lag1,
      na.rm = TRUE
    ),
    persistencia_min = min(
      corr_lag1,
      na.rm = TRUE
    ),
    persistencia_max = max(
      corr_lag1,
      na.rm = TRUE
    )
  )

print(
  resumen_persistencia
)

# ------------------- Subbloque 9.8.2 Clasificación Persistencia ------------------
persistencia_municipal <- persistencia_municipal |>
  mutate(
    categoria = case_when(
      corr_lag1 >= 0.80 ~ "Muy_Alta",
      corr_lag1 >= 0.60 ~ "Alta",
      corr_lag1 >= 0.40 ~ "Moderada",
      corr_lag1 >= 0.20 ~ "Baja",
      TRUE ~ "Muy_Baja"
    )
  ) # Clasificar persistencia

persistencia_municipal |>
  count(
    categoria
  ) |>
  arrange(
    desc(n)
  )

# ------------------- Subbloque 9.8.3 Top Persistencia ------------------
persistencia_municipal |>
  arrange(
    desc(corr_lag1)
  ) |>
  head(20)

# ------------------- Subbloque 9.8.4 Baja Persistencia ------------------
persistencia_municipal |>
  arrange(
    corr_lag1
  ) |>
  head(20)

# ------------------- Subbloque 9.8.5 Distribución ------------------
persistencia_municipal |>
  summarise(
    p05 = quantile(
      corr_lag1,
      0.05,
      na.rm = TRUE
    ),
    p25 = quantile(
      corr_lag1,
      0.25,
      na.rm = TRUE
    ),
    p50 = quantile(
      corr_lag1,
      0.50,
      na.rm = TRUE
    ),
    p75 = quantile(
      corr_lag1,
      0.75,
      na.rm = TRUE
    ),
    p95 = quantile(
      corr_lag1,
      0.95,
      na.rm = TRUE
    )
  )


# ------------------- Bloque 10. Construcción y Auditoría de Grafos ------------------
# Construcción de grafos candidatos para modelado GNN
# ============================================================
# ------------------- Subbloque 10.0 Parámetros ------------------
  n_vecinos <- 5 # Número de vecinos KNN
# ------------------- Subbloque 10.1 Nodos Base ------------------
  nodos <- panel_manual |>
  dplyr::filter(
    anio == 2018
  ) |>
  dplyr::select(
    cod_mpio,
    municipio,
    latitud,
    longitud
  ) |>
  distinct() # Un nodo por municipio
# ------------------- Subbloque 10.2 Grafo Geográfico ------------------
  coords_geo <- as.matrix(
    nodos[, c(
      "longitud",
      "latitud"
    )]
  ) # Coordenadas geográficas
knn_geo <- FNN::get.knn(
  coords_geo,
  k = n_vecinos
) # Vecinos geográficos
edges_geo <- data.table::rbindlist(
  lapply(
    seq_len(
      nrow(coords_geo)
    ),
    function(i){
      data.table::data.table(
        origen = nodos$cod_mpio[i],
        destino = nodos$cod_mpio[
          knn_geo$nn.index[i, ]
        ]
      )
      
    }
  )
) # Lista de aristas geográficas
# ------------------- Subbloque 10.3 Grafo Climático ------------------
  variables_climaticas <- c(
    "d2m",
    "t2m",
    "tp",
    "u10",
    "v10",
    "strd",
    "pev",
    "ro",
    "sro",
    "lai_hv"
  ) # Variables climáticas
datos_clima <- panel_manual |>
  dplyr::filter(
    anio == 2018
  ) |>
  dplyr::select(
    cod_mpio,
    all_of(
      variables_climaticas
    )
  ) # Dataset climático
matriz_clima <- scale(
  datos_clima[, -1]
) # Estandarizar variables
knn_clima <- FNN::get.knn(
  matriz_clima,
  k = n_vecinos
) # Vecinos climáticos
edges_clima <- data.table::rbindlist(
  lapply(
    seq_len(
      nrow(matriz_clima)
    ),
    function(i){
      data.table::data.table(
        origen = datos_clima$cod_mpio[i],
        destino = datos_clima$cod_mpio[
          knn_clima$nn.index[i, ]
        ]
      )
      
    }
  )
) # Lista de aristas climáticas
# ------------------- Subbloque 10.4 Grafo Productivo ------------------
  variables_productivas <- c(
    "area_irrigable_total",
    "indice_hidrico",
    "n_cultivos",
    "porcentaje_permanentes",
    "potencial_score_promedio"
  ) # Variables productivas
datos_prod <- panel_manual |>
  dplyr::filter(
    anio == 2018
  ) |>
  dplyr::select(
    cod_mpio,
    all_of(
      variables_productivas
    )
  ) # Dataset productivo
matriz_prod <- scale(
  datos_prod[, -1]
) # Estandarizar variables
knn_prod <- FNN::get.knn(
  matriz_prod,
  k = n_vecinos
) # Vecinos productivos
edges_prod <- data.table::rbindlist(
  lapply(
    seq_len(
      nrow(matriz_prod)
    ),
    function(i){
      data.table::data.table(
        origen = datos_prod$cod_mpio[i],
        destino = datos_prod$cod_mpio[
          knn_prod$nn.index[i, ]
        ]
      )
      
    }
  )
) # Lista de aristas productivas
# ------------------- Subbloque 10.5 Grafo Híbrido ------------------
  edges_hibrido <- dplyr::bind_rows(
    edges_geo,
    edges_clima,
    edges_prod
  ) |>
  distinct() # Combinar relaciones
#------------------- Subbloque 10.6 Construcción Objetos Grafo ------------------
  grafo_geo <- igraph::graph_from_data_frame(
    edges_geo,
    directed = FALSE
  ) # Grafo geográfico
grafo_clima <- igraph::graph_from_data_frame(
  edges_clima,
  directed = FALSE
) # Grafo climático
grafo_prod <- igraph::graph_from_data_frame(
  edges_prod,
  directed = FALSE
) # Grafo productivo
grafo_hibrido <- igraph::graph_from_data_frame(
  edges_hibrido,
  directed = FALSE
) # Grafo híbrido
# ------------------- Subbloque 10.7 Auditoría de Grafos ------------------
auditar_grafo <- function(
    g,
    nombre
){
  
  tibble::tibble(
    grafo = nombre,
    nodos = igraph::vcount(g),
    aristas = igraph::ecount(g),
    densidad = igraph::edge_density(g),
    grado_promedio = mean(
      igraph::degree(g)
    ),
    componentes = igraph::count_components(g),
    diametro = igraph::diameter(g),
    conectado = igraph::is_connected(g)
  )
  
} # Función auditoría grafos

# ------------------- Subbloque 10.8 Comparación Grafos ------------------
  comparacion_grafos <- dplyr::bind_rows(
    auditar_grafo(
      grafo_geo,
      "Geografico"
    ),
    auditar_grafo(
      grafo_clima,
      "Climatico"
    ),
    auditar_grafo(
      grafo_prod,
      "Productivo"
    ),
    auditar_grafo(
      grafo_hibrido,
      "Hibrido"
    )
  ) |>
  dplyr::arrange(
    dplyr::desc(
      densidad
    )
  ) # Ranking de grafos
print(
  comparacion_grafos
) # Mostrar auditoría
# ------------------- Subbloque 10.9 Centralidad ------------------
  centralidad_geo <- tibble::tibble(
    cod_mpio = names(
      igraph::degree(
        grafo_geo
      )
    ),
    grado = igraph::degree(
      grafo_geo
    ),
    betweenness = igraph::betweenness(
      grafo_geo
    ),
    eigen = igraph::eigen_centrality(
      grafo_geo
    )$vector
  ) |>
  dplyr::arrange(
    dplyr::desc(
      grado
    )
  ) # Ranking centralidad
print(
  head(
    centralidad_geo,
    20
  )
) # Top 20 nodos

# ------------------- Reconstruir Comparación NZV ------------------
comparacion_nzv <- dplyr::bind_rows(
  
  Maestro = evaluar_nzv(
    panel_maestro
  ),
  
  Reducido_Manual = evaluar_nzv(
    panel_manual
  ),
  
  Reducido_Automatico = evaluar_nzv(
    panel_auto
  ),
  
  .id = "dataset"
  
) # Comparación NZV

# ------------------- Subbloque 10.10 Exportación Grafos ------------------
  dir.create(
    "data/grafos",
    recursive = TRUE,
    showWarnings = FALSE
  ) # Crear carpeta grafos
arrow::write_parquet(
  edges_geo,
  "data/grafos/edges_geo.parquet"
) # Exportar geográfico
arrow::write_parquet(
  edges_clima,
  "data/grafos/edges_clima.parquet"
) # Exportar climático
arrow::write_parquet(
  edges_prod,
  "data/grafos/edges_prod.parquet"
) # Exportar productivo
arrow::write_parquet(
  edges_hibrido,
  "data/grafos/edges_hibrido.parquet"
) # Exportar híbrido
data.table::fwrite(
  edges_geo,
  "data/grafos/edges_geo.csv"
) # Exportar geográfico CSV
data.table::fwrite(
  edges_clima,
  "data/grafos/edges_clima.csv"
) # Exportar climático CSV
data.table::fwrite(
  edges_prod,
  "data/grafos/edges_prod.csv"
) # Exportar productivo CSV
data.table::fwrite(
  edges_hibrido,
  "data/grafos/edges_hibrido.csv"
) # Exportar híbrido CSV
cat(
  "\nGrafos exportados correctamente",
  "\nGeográfico",
  "\nClimático",
  "\nProductivo",
  "\nHíbrido",
  "\n",
  sep = ""
) # Confirmar exportación

# ------------------- Bloque 11. Exportación Resultados Evaluación ------------------
dir.create(
  "data/evaluacion_datasets_gnn",
  recursive = TRUE,
  showWarnings = FALSE
) # Carpeta principal

dir.create(
  "data/evaluacion_datasets_gnn/estadistica",
  recursive = TRUE,
  showWarnings = FALSE
) # Resultados estadísticos

dir.create(
  "data/evaluacion_datasets_gnn/espacial",
  recursive = TRUE,
  showWarnings = FALSE
) # Resultados espaciales

dir.create(
  "data/evaluacion_datasets_gnn/temporal",
  recursive = TRUE,
  showWarnings = FALSE
) # Resultados temporales

dir.create(
  "data/evaluacion_datasets_gnn/grafos",
  recursive = TRUE,
  showWarnings = FALSE
) # Resultados grafos

# ------------------- Subbloque 11.1 Calidad Estadística ------------------
dataset_ganador <- panel_manual # Dataset seleccionado

arrow::write_parquet(
  dataset_ganador,
  "data/modelado/dataset_ganador_gnn.parquet"
) # Dataset definitivo


arrow::write_parquet(
  metricas_maestro,
  "data/evaluacion_datasets_gnn/estadistica/metricas_maestro.parquet"
) # Métricas maestro

arrow::write_parquet(
  metricas_manual,
  "data/evaluacion_datasets_gnn/estadistica/metricas_manual.parquet"
) # Métricas manual

arrow::write_parquet(
  metricas_auto,
  "data/evaluacion_datasets_gnn/estadistica/metricas_auto.parquet"
) # Métricas automático

readr::write_csv(
  resumen_calidad,
  "data/evaluacion_datasets_gnn/estadistica/resumen_calidad.csv"
) # Resumen calidad

arrow::write_parquet(
  resumen_calidad,
  "data/evaluacion_datasets_gnn/estadistica/resumen_calidad.parquet"
) # Resumen calidad parquet

# ------------------- Subbloque 11.2 Correlación Residual ------------------

readr::write_csv(
  comparacion_correlacion,
  "data/evaluacion_datasets_gnn/estadistica/comparacion_correlacion.csv"
) # Comparación correlación

arrow::write_parquet(
  comparacion_correlacion,
  "data/evaluacion_datasets_gnn/estadistica/comparacion_correlacion.parquet"
) # Comparación correlación parquet

arrow::write_parquet(
  cor_altas_maestro,
  "data/evaluacion_datasets_gnn/estadistica/cor_altas_maestro.parquet"
) # Correlaciones altas maestro

arrow::write_parquet(
  cor_altas_manual,
  "data/evaluacion_datasets_gnn/estadistica/cor_altas_manual.parquet"
) # Correlaciones altas manual

arrow::write_parquet(
  cor_altas_auto,
  "data/evaluacion_datasets_gnn/estadistica/cor_altas_auto.parquet"
) # Correlaciones altas automático

# ------------------- Subbloque 11.3 Near Zero Variance ------------------

readr::write_csv(
  comparacion_nzv,
  "data/evaluacion_datasets_gnn/estadistica/comparacion_nzv.csv"
) # Comparación NZV

arrow::write_parquet(
  comparacion_nzv,
  "data/evaluacion_datasets_gnn/estadistica/comparacion_nzv.parquet"
) # Comparación NZV parquet

arrow::write_parquet(
  nzv_maestro,
  "data/evaluacion_datasets_gnn/estadistica/nzv_maestro.parquet"
) # NZV maestro

arrow::write_parquet(
  nzv_manual,
  "data/evaluacion_datasets_gnn/estadistica/nzv_manual.parquet"
) # NZV manual

arrow::write_parquet(
  nzv_auto,
  "data/evaluacion_datasets_gnn/estadistica/nzv_auto.parquet"
) # NZV automático

# ------------------- Subbloque 11.4 Benchmark Predictivo ------------------

readr::write_csv(
  comparacion_rf,
  "data/evaluacion_datasets_gnn/estadistica/comparacion_rf.csv"
) # RF original

arrow::write_parquet(
  comparacion_rf,
  "data/evaluacion_datasets_gnn/estadistica/comparacion_rf.parquet"
) # RF original parquet

readr::write_csv(
  comparacion_sin_leakage,
  "data/evaluacion_datasets_gnn/estadistica/comparacion_rf_sin_leakage.csv"
) # RF sin leakage

arrow::write_parquet(
  comparacion_sin_leakage,
  "data/evaluacion_datasets_gnn/estadistica/comparacion_rf_sin_leakage.parquet"
) # RF sin leakage parquet

arrow::write_parquet(
  resultado_rf_manual$importancia,
  "data/evaluacion_datasets_gnn/estadistica/importancia_variables_manual.parquet"
) # Importancia variables

# ------------------- Subbloque 11.5 Evaluación Espacial ------------------

readr::write_csv(
  comparacion_espacial,
  "data/evaluacion_datasets_gnn/espacial/comparacion_espacial.csv"
) # Comparación espacial

arrow::write_parquet(
  comparacion_espacial,
  "data/evaluacion_datasets_gnn/espacial/comparacion_espacial.parquet"
) # Comparación espacial parquet

arrow::write_parquet(
  sf::st_drop_geometry(
    espacial_maestro$lisa
  ),
  "data/evaluacion_datasets_gnn/espacial/lisa_maestro.parquet"
) # LISA maestro

arrow::write_parquet(
  sf::st_drop_geometry(
    espacial_manual$lisa
  ),
  "data/evaluacion_datasets_gnn/espacial/lisa_manual.parquet"
) # LISA manual

arrow::write_parquet(
  sf::st_drop_geometry(
    espacial_auto$lisa
  ),
  "data/evaluacion_datasets_gnn/espacial/lisa_auto.parquet"
) # LISA automático

# ------------------- Subbloque 11.6 Evaluación Temporal ------------------

readr::write_csv(
  comparacion_temporal,
  "data/evaluacion_datasets_gnn/temporal/comparacion_temporal.csv"
) # Comparación temporal

arrow::write_parquet(
  comparacion_temporal,
  "data/evaluacion_datasets_gnn/temporal/comparacion_temporal.parquet"
) # Comparación temporal parquet

readr::write_csv(
  serie_nacional,
  "data/evaluacion_datasets_gnn/temporal/serie_nacional.csv"
) # Serie nacional

arrow::write_parquet(
  serie_nacional,
  "data/evaluacion_datasets_gnn/temporal/serie_nacional.parquet"
) # Serie nacional parquet

readr::write_csv(
  persistencia_municipal,
  "data/evaluacion_datasets_gnn/temporal/persistencia_municipal.csv"
) # Persistencia municipal

arrow::write_parquet(
  persistencia_municipal,
  "data/evaluacion_datasets_gnn/temporal/persistencia_municipal.parquet"
) # Persistencia municipal parquet

# ------------------- Subbloque 11.7 Grafos ------------------

readr::write_csv(
  comparacion_grafos,
  "data/evaluacion_datasets_gnn/grafos/comparacion_grafos.csv"
) # Comparación grafos

arrow::write_parquet(
  comparacion_grafos,
  "data/evaluacion_datasets_gnn/grafos/comparacion_grafos.parquet"
) # Comparación grafos parquet

readr::write_csv(
  centralidad_geo,
  "data/evaluacion_datasets_gnn/grafos/centralidad_geografica.csv"
) # Centralidad

arrow::write_parquet(
  centralidad_geo,
  "data/evaluacion_datasets_gnn/grafos/centralidad_geografica.parquet"
) # Centralidad parquet

saveRDS(
  grafo_geo,
  "data/evaluacion_datasets_gnn/grafos/grafo_geografico.rds"
) # Grafo geográfico

saveRDS(
  grafo_clima,
  "data/evaluacion_datasets_gnn/grafos/grafo_climatico.rds"
) # Grafo climático

saveRDS(
  grafo_prod,
  "data/evaluacion_datasets_gnn/grafos/grafo_productivo.rds"
) # Grafo productivo

saveRDS(
  grafo_hibrido,
  "data/evaluacion_datasets_gnn/grafos/grafo_hibrido.rds"
) # Grafo híbrido

cat(
  "\nExportación completa de evaluación GNN finalizada correctamente\n"
) # Confirmar exportación




