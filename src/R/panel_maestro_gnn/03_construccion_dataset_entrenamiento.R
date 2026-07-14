# 03_construccion_dataset_entrenamiento.R
# ------------------- Bloque 1. Configuración General ------------------
# Configuración inicial del dataset de entrenamiento

source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros globales

cat("Periodo:", anio_inicio, "-", anio_fin, "| Unidad:", unidad_analisis, "| Target:", variable_target, "| Seed:", seed_global, "\n") # Resumen configuración
cat("Fecha ejecución:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n") # Fecha ejecución
cat("Objetivo: Construir dataset final libre de NA para entrenamiento en Python\n") # Objetivo script
cat("Estrategia: Auditoría NA | Exclusión variables críticas | Exclusión registros sin target | Imputación | Validación | Exportación\n") # Resumen pipeline
cat("Ruta salida:", rutas$processed_panel_maestro_gnn, "\n") # Ruta exportación
cat("\nBloque 1 completado correctamente\n") # Confirmación bloque
# ------------------- Fin Bloque 1. Configuración General ------------------

# ------------------- Bloque 2. Carga Dataset Modelado ------------------
# Cargar dataset final generado en Script 02
ruta_dataset_modelado <- file.path(
  rutas$processed_panel_maestro_gnn,
  "panel_modelado_final.parquet"
) # Ruta dataset modelado

if (!file.exists(ruta_dataset_modelado)) {
  stop(
    "No se encontró panel_modelado_final.parquet"
  )
} # Validar existencia dataset

panel_modelado_final <- arrow::read_parquet(
  ruta_dataset_modelado
) # Cargar dataset modelado

cat(
  "\nDataset cargado correctamente:",
  "panel_modelado_final.parquet\n"
) # Confirmar carga

cat(
  "Dimensiones:",
  nrow(panel_modelado_final),
  "filas |",
  ncol(panel_modelado_final),
  "columnas\n"
) # Resumen dimensiones

cat(
  "Municipios:",
  dplyr::n_distinct(panel_modelado_final$cod_mpio),
  "| Años:",
  dplyr::n_distinct(panel_modelado_final$anio),
  "\n"
) # Cobertura espacio temporal

cat(
  "Target:",
  variable_target,
  "\n"
) # Variable objetivo

cat(
  "Target NA:",
  sum(is.na(panel_modelado_final[[variable_target]])),
  "\n"
) # Registros sin variable objetivo

cat(
  "\nBloque 2 completado correctamente\n"
) # Confirmación bloque

# ------------------- Fin Bloque 2. Carga Dataset Modelado ------------------

# ------------------- Bloque 3. Auditoría Inicial de Valores Faltantes ------------------
# Diagnóstico de valores faltantes previo a la construcción del dataset entrenamiento
auditoria_na <- tibble::tibble(
  variable = names(panel_modelado_final),
  nulos = sapply(
    panel_modelado_final,
    \(x) sum(is.na(x))
  ),
  porcentaje_nulos = round(
    sapply(
      panel_modelado_final,
      \(x) mean(is.na(x))
    ) * 100,
    2
  )
) |>
  arrange(desc(nulos)) # Ordenar por cantidad de nulos

cat(
  "\nVariables con NA:",
  sum(auditoria_na$nulos > 0),
  "de",
  ncol(panel_modelado_final),
  "\n"
) # Resumen variables con faltantes

cat(
  "Total valores faltantes:",
  sum(auditoria_na$nulos),
  "\n"
) # Total de NA en el dataset

cat(
  "Máximo porcentaje de NA:",
  max(auditoria_na$porcentaje_nulos),
  "%\n"
) # Mayor porcentaje de faltantes

auditoria_na |>
  filter(nulos > 0) |>
  print(n = Inf) # Mostrar variables con NA

# Clasificación de riesgo
auditoria_na_resumen <- auditoria_na |>
  mutate(
    clasificacion = case_when(
      porcentaje_nulos > 70 ~ "CRITICO",
      porcentaje_nulos > 20 ~ "ALTO",
      porcentaje_nulos > 5 ~ "MODERADO",
      porcentaje_nulos > 0 ~ "BAJO",
      TRUE ~ "SIN_NA"
    )
  ) # Clasificar nivel de riesgo

auditoria_na_resumen |>
  count(clasificacion) |>
  print() # Resumen por categoría

# Variables con cobertura crítica
variables_criticas <- auditoria_na |>
  filter(porcentaje_nulos > 70)

cat(
  "\nVariables con cobertura crítica:",
  nrow(variables_criticas),
  "\n"
) # Resumen variables críticas

variables_criticas |>
  print(n = Inf)

# Eliminar variables críticas:
variables_criticas <- c(
  "precio_ha_promedio",
  "precio_ha_mediana",
  "precio_ha_min",
  "precio_ha_max",
  "n_poligonos",
  "indice_fragmentacion"
) # Variables con cobertura crítica

# Paso 1. Crear dataset sin variables críticas
panel_sin_criticas <- panel_modelado_final |>
  select(
    -all_of(variables_criticas)
  ) # Eliminar variables críticas

# Verificar dimensiones
cat(
  "\nDimensiones:",
  nrow(panel_sin_criticas),
  "filas |",
  ncol(panel_sin_criticas),
  "columnas\n"
) # Verificar dimensiones

cat(
  "\nNA antes:",
  sum(is.na(panel_modelado_final)),
  "\nNA después:",
  sum(is.na(panel_sin_criticas)),
  "\nReducción:",
  sum(is.na(panel_modelado_final)) -
    sum(is.na(panel_sin_criticas)),
  "\n"
) # Comparar NA

panel_pre_imputacion <- panel_sin_criticas |>
  filter(
    !is.na(log_rendimiento)
  ) # Eliminar registros sin target

cat(
  "\nRegistros antes:",
  nrow(panel_sin_criticas),
  "\nRegistros después:",
  nrow(panel_pre_imputacion),
  "\nRegistros eliminados:",
  nrow(panel_sin_criticas) -
    nrow(panel_pre_imputacion),
  "\n"
) # Auditoría eliminación target
# ------------------- Fin Bloque 3. Auditoría Inicial de Valores Faltantes ------------------

# ------------------- Bloque 4. Auditoría Post-Depuración ------------------
# Diagnóstico final previo a imputación
auditoria_post <- tibble::tibble(
  variable = names(panel_pre_imputacion),
  nulos = sapply(
    panel_pre_imputacion,
    \(x) sum(is.na(x))
  ),
  porcentaje_nulos = round(
    sapply(
      panel_pre_imputacion,
      \(x) mean(is.na(x))
    ) * 100,
    2
  )
) |>
  arrange(desc(nulos)) # Ordenar por cantidad de nulos

cat(
  "\nDataset pre-imputación:",
  nrow(panel_pre_imputacion),
  "filas |",
  ncol(panel_pre_imputacion),
  "columnas\n"
) # Resumen dataset

cat(
  "Variables con NA:",
  sum(auditoria_post$nulos > 0),
  "\n"
) # Variables con faltantes

cat(
  "Total NA:",
  sum(auditoria_post$nulos),
  "\n"
) # Total faltantes

cat(
  "Máximo porcentaje NA:",
  max(auditoria_post$porcentaje_nulos),
  "%\n"
) # Máximo porcentaje faltantes

auditoria_post |>
  filter(nulos > 0) |>
  print(n = Inf) # Mostrar variables con NA


# ------------------- Fin Bloque 4. Auditoría Post-Depuración ------------------

# ------------------- Bloque 5. Imputación de Valores Faltantes ------------------
# Imputación híbrida para dataset de entrenamiento

panel_imputado <- panel_imputacion |>
  mutate(
    across(
      starts_with("era5_"),
      ~ tidyr::replace_na(
        .x,
        median(.x, na.rm = TRUE)
      )
    )
  ) |>
  mutate(
    across(
      c(
        area_total,
        pct_agro,
        pct_natural,
        pct_no_agro,
        pct_otros_usos,
        pct_propia,
        pct_arrendada,
        pct_colectiva,
        pct_mixta
      ),
      ~ tidyr::replace_na(
        .x,
        median(.x, na.rm = TRUE)
      )
    )
  ) |>
  mutate(
    across(
      c(
        n_poligonos_irrigacion,
        area_irrigable_total,
        area_irrigable_promedio,
        area_irrigable_max,
        potencial_score_promedio,
        potencial_score_max,
        potencial_score_sd,
        tipo_tierra_score_promedio,
        necesidad_hidrica_score_promedio,
        disponibilidad_score_promedio,
        regulacion_score_promedio,
        ecosistemico_score_promedio,
        socioeconomico_score_promedio,
        pct_alto_potencial,
        pct_muy_alto_potencial,
        log_area_irrigable_total,
        concentracion_irrigacion,
        indice_calidad_irrigacion,
        indice_hidrico,
        indice_sostenibilidad
      ),
      ~ tidyr::replace_na(
        .x,
        median(.x, na.rm = TRUE)
      )
    )
  ) # Imputar NA con mediana

cat(
  "\nNA antes:",
  sum(is.na(panel_imputacion)),
  "\n"
) # NA iniciales

cat(
  "NA después:",
  sum(is.na(panel_imputado)),
  "\n"
) # NA finales

stopifnot(
  sum(is.na(panel_imputado)) == 0
) # Validar ausencia total de NA

panel_entrenamiento <- dplyr::bind_cols(
  panel_pre_imputacion |>
    select(
      cod_depto,
      departamento,
      cod_mpio,
      municipio
    ),
  panel_imputado
) # Reconstruir dataset final

cat(
  "\nDimensiones finales:",
  nrow(panel_entrenamiento),
  "filas |",
  ncol(panel_entrenamiento),
  "columnas\n"
) # Resumen final

cat(
  "\nBloque 5 completado correctamente\n"
) # Confirmación bloque
# ------------------- Fin Bloque 5. Imputación de Valores Faltantes ------------------

# ------------------- Bloque 6. Evaluación del Impacto de la Imputación ------------------
# Auditoría de calidad de la imputación
# Variables que originalmente tenían NA
variables_imputadas <- names(panel_imputacion)[
  colSums(is.na(panel_imputacion)) > 0
] # Variables imputadas

# Métrica 1. Porcentaje imputado por variable
auditoria_imputacion <- tibble::tibble(
  variable = variables_imputadas,
  nulos_originales = sapply(
    panel_imputacion[variables_imputadas],
    \(x) sum(is.na(x))
  ),
  porcentaje_imputado = round(
    sapply(
      panel_imputacion[variables_imputadas],
      \(x) mean(is.na(x))
    ) * 100,
    2
  )
) |>
  arrange(desc(porcentaje_imputado)) # Ordenar por porcentaje imputado

cat(
  "\nVariables imputadas:",
  nrow(auditoria_imputacion),
  "\n"
) # Resumen variables imputadas

# Métrica 2. Cambio en media, mediana y desviación estándar
evaluacion_sesgo <- tibble::tibble(
  variable = variables_imputadas,
  media_original = sapply(
    panel_imputacion[variables_imputadas],
    \(x) mean(x, na.rm = TRUE)
  ),
  media_imputada = sapply(
    panel_imputado[variables_imputadas],
    mean
  ),
  mediana_original = sapply(
    panel_imputacion[variables_imputadas],
    \(x) median(x, na.rm = TRUE)
  ),
  mediana_imputada = sapply(
    panel_imputado[variables_imputadas],
    median
  ),
  sd_original = sapply(
    panel_imputacion[variables_imputadas],
    \(x) sd(x, na.rm = TRUE)
  ),
  sd_imputada = sapply(
    panel_imputado[variables_imputadas],
    sd
  )
) |>
  mutate(
    sesgo_media_pct = round(
      100 * (media_imputada - media_original) /
        abs(media_original),
      4
    ),
    sesgo_sd_pct = round(
      100 * (sd_imputada - sd_original) /
        abs(sd_original),
      4
    )
  ) # Calcular cambios porcentuales

# Métrica 3. Resumen global de sesgo
cat(
  "\nSesgo promedio media (%):",
  round(
    mean(abs(evaluacion_sesgo$sesgo_media_pct)),
    4
  ),
  "\n"
) # Sesgo medio absoluto

cat(
  "Sesgo promedio SD (%):",
  round(
    mean(abs(evaluacion_sesgo$sesgo_sd_pct)),
    4
  ),
  "\n"
) # Cambio promedio desviación estándar

# Métrica 4. Variables más afectadas
variables_mayor_sesgo <- evaluacion_sesgo |>
  left_join(
    auditoria_imputacion,
    by = "variable"
  ) |>
  arrange(
    desc(abs(sesgo_media_pct))
  ) # Incorporar porcentaje imputado
cat(
  "\nTop 10 variables con mayor cambio en media:\n"
) # Encabezado

print(
  variables_mayor_sesgo |>
    select(
      variable,
      porcentaje_imputado,
      sesgo_media_pct,
      sesgo_sd_pct
    ),
  n = 10
)

# Métrica 5. Prueba KS (Kolmogorov-Smirnov)
resultado_ks <- purrr::map_dfr(
  variables_imputadas,
  \(v) {
    observados <- panel_imputacion[[v]][!is.na(panel_imputacion[[v]])]
    imputados <- panel_imputado[[v]]
    
    tibble::tibble(
      variable = v,
      ks_statistic = suppressWarnings(
        ks.test(
          observados,
          imputados
        )$statistic
      ),
      p_value = suppressWarnings(
        ks.test(
          observados,
          imputados
        )$p.value
      )
    )
  }
) # Comparar distribuciones

cat(
  "\nVariables con diferencias significativas (KS p < 0.05):",
  sum(resultado_ks$p_value < 0.05),
  "de",
  nrow(resultado_ks),
  "\n"
) # Resumen KS

# Métrica 6. Resumen final
cat(
  "\nNA originales:",
  sum(is.na(panel_imputacion)),
  "\nNA finales:",
  sum(is.na(panel_imputado)),
  "\n"
) # Resumen imputación

cat(
  "\nBloque 6 completado correctamente\n"
) # Confirmación bloque

# ------------------- Fin Bloque 6. Evaluación del Impacto de la Imputación ------------------

# ------------------- Bloque 7. Exportación Dataset Entrenamiento ------------------
# Exportar dataset final libre de NA para Python
dir.create(
  file.path(
    rutas$processed_panel_maestro_gnn,
    "training"
  ),
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta entrenamiento

# Exportar dataset entrenamiento parquet
arrow::write_parquet(
  panel_entrenamiento,
  file.path(
    rutas$processed_panel_maestro_gnn,
    "training",
    "panel_entrenamiento.parquet"
  )
) # Exportar parquet

# Exportar dataset entrenamiento csv
readr::write_csv(
  panel_entrenamiento,
  file.path(
    rutas$processed_panel_maestro_gnn,
    "training",
    "panel_entrenamiento.csv"
  )
) # Exportar csv

# Exportar auditoría imputación
readr::write_csv(
  auditoria_imputacion,
  file.path(
    rutas$processed_panel_maestro_gnn,
    "training",
    "auditoria_imputacion.csv"
  )
) # Exportar auditoría imputación

# Exportar evaluación sesgo
readr::write_csv(
  evaluacion_sesgo,
  file.path(
    rutas$processed_panel_maestro_gnn,
    "training",
    "evaluacion_sesgo_imputacion.csv"
  )
) # Exportar evaluación sesgo

# Exportar prueba KS
readr::write_csv(
  resultado_ks,
  file.path(
    rutas$processed_panel_maestro_gnn,
    "training",
    "resultado_ks.csv"
  )
) # Exportar KS

# Auditoría final dataset entrenamiento
auditoria_final_entrenamiento <- tibble::tibble(
  variable = names(panel_entrenamiento),
  tipo = sapply(
    panel_entrenamiento,
    typeof
  ),
  nulos = sapply(
    panel_entrenamiento,
    \(x) sum(is.na(x))
  )
) # Auditoría final

readr::write_csv(
  auditoria_final_entrenamiento,
  file.path(
    rutas$processed_panel_maestro_gnn,
    "training",
    "auditoria_final_entrenamiento.csv"
  )
) # Exportar auditoría final

cat(
  "\nDataset exportado correctamente\n"
) # Confirmación

cat(
  "Registros:",
  nrow(panel_entrenamiento),
  "\nVariables:",
  ncol(panel_entrenamiento),
  "\nNA:",
  sum(is.na(panel_entrenamiento)),
  "\n"
) # Resumen dataset

cat(
  "\nArchivos generados:\n",
  "- panel_entrenamiento.parquet\n",
  "- panel_entrenamiento.csv\n",
  "- auditoria_imputacion.csv\n",
  "- evaluacion_sesgo_imputacion.csv\n",
  "- resultado_ks.csv\n",
  "- auditoria_final_entrenamiento.csv\n"
) # Resumen exportación
