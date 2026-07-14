# # 02_preparacion_dataset_modelado.R
# Construcción Panel Maestro GNN Agrícola
# # 02_preparacion_dataset_modelado.R

# ------------------- Bloque 1. Configuración general ------------------
# Configuración inicial del pipeline de modelado
# ============================================================

source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros globales

cat("\nPREPARACIÓN DATASET MODELADO GNN\n") # Título principal

cat(
  "Periodo:", anio_inicio, "-", anio_fin,
  "| Unidad:", unidad_analisis,
  "| Target:", variable_target,
  "| Seed:", seed_global,
  "\n"
) # Resumen configuración

# ------------------- Fin Bloque 1. Configuración general ------------------

# ------------------- Bloque 2. Carga Panel Maestro ------------------
# Cargar dataset consolidado
# ============================================================
if (
  !file.exists(
    file.path(
      rutas$processed_panel_maestro_gnn,
      "panel_master_clean.parquet"
    )
  )
) {
  stop(
    "No se encontró panel_master_clean.parquet"
  )
  
} # Validar existencia del archivo

panel_master_clean <- arrow::read_parquet(
  file.path(
    rutas$processed_panel_maestro_gnn,
    "panel_master_clean.parquet"
  )
) # Cargar panel maestro limpio

cat(
  "\nPanel maestro cargado:",
  nrow(panel_master_clean),
  "filas |",
  ncol(panel_master_clean),
  "columnas\n"
) # Resumen carga

exists("panel_master_clean")
dim(panel_master_clean)
# ------------------- Fin Bloque 2. Carga Panel Maestro ------------------

# ------------------- Bloque 3. Validación Estructural ------------------
# Validación de integridad del panel maestro
# ============================================================
panel_master_clean |>
  summarise(
    municipios = n_distinct(cod_mpio),
    anios = n_distinct(anio),
    paneles = n_distinct(panel_id)
  ) |>
  print() # Resumen universo analítico

cat(
  "\nDuplicados panel_id:",
  sum(duplicated(panel_master_clean$panel_id)),
  "\n"
) # Verificar duplicados

cat(
  "Panel ID NA:",
  sum(is.na(panel_master_clean$panel_id)),
  "\n"
) # Verificar llaves faltantes

cat(
  "Target NA:",
  sum(is.na(panel_master_clean[[variable_target]])),
  "\n"
) # Verificar variable objetivo

data.frame(
  variable = names(panel_master_clean),
  tipo = sapply(panel_master_clean, class)
) |>
  dplyr::count(tipo) |>
  print() # Resumen tipos de datos

cat(
  "\nDimensiones:",
  nrow(panel_master_clean),
  "filas |",
  ncol(panel_master_clean),
  "columnas\n"
) # Resumen dimensiones

cat(
  "Duplicados panel_id:",
  sum(duplicated(panel_master_clean$panel_id)),
  "\n"
)

sum(duplicated(panel_master_clean$panel_id))

panel_master_clean |>
  filter(is.na(log_rendimiento)) |>
  count(anio)

panel_master_clean |>
  filter(is.na(log_rendimiento)) |>
  count(cod_mpio, sort = TRUE)

panel_master_clean |>
  filter(is.na(log_rendimiento)) |>
  select(
    cod_mpio,
    anio,
    produccion_total,
    area_cosechada_total,
    rendimiento_promedio
  ) |>
  print(n = 50)

panel_master_clean |>
  filter(is.na(log_rendimiento)) |>
  summarise(
    prod_na = sum(is.na(produccion_total)),
    area_na = sum(is.na(area_cosechada_total)),
    rend_na = sum(is.na(rendimiento_promedio))
  )
# ------------------- Fin Bloque 3. Validación Estructural ------------------

# ------------------- Bloque 4. Auditoría de Valores Faltantes ------------------
# Identificación de variables con datos faltantes
# ============================================================
panel_master_clean_nulos <- tibble::tibble(
  variable = names(panel_master_clean),
  nulos = sapply(
    panel_master_clean,
    \(x) sum(is.na(x))
  ),
  porcentaje_nulos = round(
    sapply(
      panel_master_clean,
      \(x) mean(is.na(x))
    ) * 100,
    2
  )
) |>
  arrange(desc(nulos)) # Ordenar por cantidad de nulos

cat(
  "\nVariables con NA:",
  sum(panel_master_clean_nulos$nulos > 0),
  "de",
  ncol(panel_master_clean),
  "\n"
) # Resumen variables con faltantes

panel_master_clean_nulos |>
  filter(nulos > 0) |>
  print(n = Inf) # Mostrar variables con faltantes

cat(
  "\nMáximo porcentaje de NA:",
  max(panel_master_clean_nulos$porcentaje_nulos),
  "%\n"
) # Variable con mayor porcentaje de faltantes

# ¿Las variables de precio contienen señal útil o son prácticamente constantes?
panel_master_clean |>
  summarise(
    across(
      starts_with("precio_"),
      ~ n_distinct(.x, na.rm = TRUE)
    )
  ) |>
  pivot_longer(
    everything(),
    names_to = "variable",
    values_to = "valores_unicos"
  ) |>
  arrange(valores_unicos)

# ¿Los 244 municipios cubiertos pertenecen a zonas agrícolas importantes?
  panel_master_clean |>
  filter(!is.na(precio_ha_promedio)) |>
  summarise(
    municipios_precio = n_distinct(cod_mpio)
  )

# ¿Precio Tierra aporta información predictiva?
  panel_master_clean |>
    select(
      log_rendimiento,
      precio_ha_promedio,
      precio_ha_mediana,
      precio_ha_ponderado
    ) |>
    cor(
      use = "complete.obs"
    )
# ------------------- Fin Bloque 4. Auditoría de Valores Faltantes ------------------

# ------------------- Bloque 5. Auditoría de Variables para Modelado ------------------
# Evaluación de variables candidatas para entrenamiento
# ============================================================
panel_modelado <- panel_master_clean |>
  mutate(
    flag_target_disponible = if_else(
      !is.na(log_rendimiento),
      1L,
      0L
    ), # Identificar target disponible

    flag_target_faltante = if_else(
      is.na(log_rendimiento),
      1L,
      0L
    ) # Identificar target faltante

  )
cat(
  "\nRegistros con target válido:",
  sum(panel_modelado$flag_target_disponible),
  "\n"
) # Resumen target válido

cat(
  "Registros con target faltante:",
  sum(panel_modelado$flag_target_faltante),
  "\n"
) # Resumen target faltante

# ------------------- Auditoría 1. Variables constantes ------------------
variables_unicas <- tibble::tibble(
  variable = names(panel_modelado),
  valores_unicos = sapply(
    panel_modelado,
    \(x) length(unique(x))
  )
) |>
  arrange(valores_unicos) # Contar valores únicos
cat(
  "\nVariables constantes:",
  sum(variables_unicas$valores_unicos == 1),
  "\n"
) # Resumen variables constantes

variables_unicas |>
  filter(valores_unicos == 1) |>
  print(n = Inf) # Mostrar variables constantes

# ------------------- Auditoría 2. Cobertura de variables ------------------
panel_master_clean_nulos |>
  mutate(
    decision = case_when(
      porcentaje_nulos > 70 ~ "REVISAR_ELIMINACION",
      porcentaje_nulos > 20 ~ "REVISAR_IMPUTACION",
      TRUE ~ "CONSERVAR"
    )
  ) |>
  arrange(desc(porcentaje_nulos)) |>
  print(n = Inf) # Clasificación según porcentaje NA

# ------------------- Auditoría 3. Variables precio tierra ------------------
panel_modelado |>
  summarise(
    across(
      starts_with("precio_"),
      ~ n_distinct(.x, na.rm = TRUE)
    )
  ) |>
  pivot_longer(
    everything(),
    names_to = "variable",
    values_to = "valores_unicos"
  ) |>
  arrange(valores_unicos) |>
  print(n = Inf) # Variabilidad variables precio

panel_modelado |>
  filter(!is.na(precio_ha_promedio)) |>
  summarise(
    municipios_precio = n_distinct(cod_mpio)
  ) |>
  print() # Cobertura territorial precio tierra

panel_modelado |>
  select(
    log_rendimiento,
    precio_ha_promedio,
    precio_ha_mediana,
    precio_ha_ponderado
  ) |>
  cor(
    use = "complete.obs"
  ) |>
  print() # Correlación con variable objetivo

# ------------------- Auditoría 4. Posible fuga de información ------------------
panel_modelado |>
  select(
    log_rendimiento,
    rendimiento_promedio,
    produccion_total,
    log_produccion_total,
    area_sembrada_total,
    area_cosechada_total
  ) |>
  cor(
    use = "complete.obs"
  ) |>
  print() # Evaluar posibles variables con leakage

panel_modelado |>
  mutate(
    log_calculado = log1p(rendimiento_promedio)
  ) |>
  summarise(
    correlacion = cor(
      log_rendimiento,
      log_calculado,
      use = "complete.obs"
    )
  )
# ------------------- Fin Bloque 5. Auditoría de Variables para Modelado ------------------

# ------------------- Bloque 6. Selección de Variables para Modelado ------------------
# Auditoría integral de variables candidatas

# ------------------- 6.1 Inventario General ------------------
auditoria_variables <- tibble::tibble(
  variable = names(panel_modelado),
  tipo = sapply(panel_modelado, class),
  valores_unicos = sapply(
    panel_modelado,
    \(x) length(unique(x))
  ),
  nulos = sapply(
    panel_modelado,
    \(x) sum(is.na(x))
  ),
  porcentaje_nulos = round(
    sapply(
      panel_modelado,
      \(x) mean(is.na(x))
    ) * 100,
    2
  )
) # Inventario inicial

# ------------------- 6.2 Cobertura Temporal ------------------
cobertura_temporal <- sapply(
  panel_modelado,
  \(x) {
    
    panel_modelado |>
      filter(!is.na(x)) |>
      summarise(
        anios = n_distinct(anio)
      ) |>
      pull(anios)
    
  }
)

# ------------------- 6.3 Cobertura Espacial ------------------
cobertura_espacial <- sapply(
  panel_modelado,
  \(x) {
    
    panel_modelado |>
      filter(!is.na(x)) |>
      summarise(
        municipios = n_distinct(cod_mpio)
      ) |>
      pull(municipios)
    
  }
)

# ------------------- 6.4 Correlación con Target ------------------
variables_numericas <- panel_modelado |>
  select(where(is.numeric)) # Variables numéricas

cor_target <- sapply(
  variables_numericas,
  \(x) cor(
    x,
    panel_modelado[[variable_target]],
    use = "pairwise.complete.obs"
  )
)

cor_target <- tibble::tibble(
  variable = names(cor_target),
  correlacion_target = round(
    as.numeric(cor_target),
    4
  )
)

# ------------------- 6.5 Consolidar Auditoría ------------------
auditoria_variables <- auditoria_variables |>
  mutate(
    cobertura_anios = cobertura_temporal,
    cobertura_municipios = cobertura_espacial
  ) |>
  left_join(
    cor_target,
    by = "variable"
  )

# ------------------- 6.6 Detección de Leakage ------------------
auditoria_variables <- auditoria_variables |>
  mutate(
    leakage = case_when(
      variable == "rendimiento_promedio" ~ "SI",
      TRUE ~ "NO"
    )
  ) # Variable derivada del target

# ------------------- 6.7 Clasificación Inicial ------------------
auditoria_variables <- auditoria_variables |>
  mutate(
    decision_preliminar = case_when(
      variable %in% c(
        "panel_id"
      ) ~ "ELIMINAR_ID",
      
      leakage == "SI" ~ "ELIMINAR_LEAKAGE",
      porcentaje_nulos > 70 ~ "REVISAR_NA",
      valores_unicos <= 1 ~ "ELIMINAR_CONSTANTE",
      TRUE ~ "CONSERVAR"
    )
  )

# ------------------- 6.8 Resumen General ------------------
cat(
  "\nVariables evaluadas:",
  nrow(auditoria_variables),
  "\n"
) # Total variables

cat(
  "Variables conservar:",
  sum(auditoria_variables$decision_preliminar == "CONSERVAR"),
  "\n"
) # Variables conservar

cat(
  "Variables revisar:",
  sum(auditoria_variables$decision_preliminar == "REVISAR_NA"),
  "\n"
) # Variables revisar

cat(
  "Variables eliminar:",
  sum(
    auditoria_variables$decision_preliminar %in%
      c(
        "ELIMINAR_ID",
        "ELIMINAR_LEAKAGE",
        "ELIMINAR_CONSTANTE"
      )
  ),
  "\n"
) # Variables eliminar

auditoria_variables |>
  arrange(
    decision_preliminar,
    desc(abs(correlacion_target))
  ) |>
  print(n = Inf)

# ------------------- 6.9 Colinealidad ------------------
cor_matrix <- panel_modelado |>
  select(where(is.numeric)) |>
  cor(use = "pairwise.complete.obs")

cor_altas <- as.data.frame(
  which(
    abs(cor_matrix) > 0.90 &
      abs(cor_matrix) < 1,
    arr.ind = TRUE
  )
)

cor_altas$variable_1 <- rownames(cor_matrix)[cor_altas$row]
cor_altas$variable_2 <- colnames(cor_matrix)[cor_altas$col]
cor_altas$correlacion <- cor_matrix[
  cbind(
    cor_altas$row,
    cor_altas$col
  )
]

cor_altas <- cor_altas |>
  select(
    variable_1,
    variable_2,
    correlacion
  ) |>
  distinct()

cat(
  "\nPares con correlación > 0.90:",
  nrow(cor_altas),
  "\n"
) # Resumen colinealidad

cor_altas

# Prueba 1. Redundancia interna
panel_modelado |>
  select(
    starts_with("precio_")
  ) |>
  cor(
    use = "pairwise.complete.obs"
  ) |>
  round(3)

# Prueba 3. Importancia Random Forest
library(ranger)

rf_temp <- ranger(
  log_rendimiento ~ .,
  data = panel_modelado |>
    select(
      log_rendimiento,
      starts_with("precio_")
    ) |>
    na.omit(),
  importance = "permutation"
)

sort(
  rf_temp$variable.importance,
  decreasing = TRUE
)

# Prueba 4. Estabilidad espacial
panel_modelado |>
  group_by(cod_mpio) |>
  summarise(
    precio = mean(
      precio_ha_promedio,
      na.rm = TRUE
    )
  ) |>
  summarise(
    municipios_con_info = sum(!is.na(precio)),
    sd_municipal = sd(precio, na.rm = TRUE)
  )

# Correlación interna
panel_modelado |>
  select(
    starts_with("precio_")
  ) |>
  cor(use = "pairwise.complete.obs")

# Importancia RF
# (sobre variables precio)

# Correlación con target
# (ya la tenemos)

panel_modelado |>
  select(
    log_rendimiento,
    area_total_ha,
    area_promedio_ha,
    n_poligonos,
    indice_fragmentacion,
    pct_area_condicionada,
    pct_area_incluida
  ) |>
  cor(use = "pairwise.complete.obs")
# ------------------- Fin Bloque 6. Selección de Variables para Modelado ------------------

# ------------------- Bloque 7. Construcción Dataset Final ------------------
# Construcción de datasets para Python
# ============================================================
variables_eliminadas <- c(
  "panel_id",
  "rendimiento_promedio",
  "precio_ha_ponderado",
  "precio_ha_sd",
  "precio_ha_p75",
  "precio_ha_p25",
  "area_total_ha",
  "area_promedio_ha",
  "pct_area_condicionada",
  "pct_area_incluida"
) # Variables descartadas

# Dataset completo
panel_modelado_full <- panel_modelado # Base completa auditada

# Dataset depurado
panel_modelado_final <- panel_modelado |>
  dplyr::select(
    -all_of(variables_eliminadas)
  ) # Base optimizada para modelado

cat(
  "\nDataset Full:",
  nrow(panel_modelado_full),
  "filas |",
  ncol(panel_modelado_full),
  "columnas\n"
) # Resumen dataset completo

cat(
  "Dataset Final:",
  nrow(panel_modelado_final),
  "filas |",
  ncol(panel_modelado_final),
  "columnas\n"
) # Resumen dataset final

cat(
  "Variables eliminadas:",
  length(variables_eliminadas),
  "\n"
) # Total variables eliminadas

# ------------------- Auditoría Final ------------------
auditoria_variables_final <- tibble::tibble(
  variable = names(panel_modelado),
  decision = ifelse(
    names(panel_modelado) %in% variables_eliminadas,
    "ELIMINAR",
    "CONSERVAR"
  )
) # Inventario final de decisiones

# ------------------- Bloque 8. Exportación Dataset Modelado ------------------
# Exportación final para Python
# ============================================================
dir.create(
  rutas$processed_panel_maestro_gnn,
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta destino

# Exportar panel completo (Parquet)
arrow::write_parquet(
  panel_modelado_full,
  file.path(
    rutas$processed_panel_maestro_gnn,
    "panel_modelado_full.parquet"
  )
) # Exportar dataset completo

# Exportar panel completo (CSV)
readr::write_csv(
  panel_modelado_full,
  file.path(
    rutas$processed_panel_maestro_gnn,
    "panel_modelado_full.csv"
  )
) # Exportar dataset completo csv

# Exportar panel final (Parquet)
arrow::write_parquet(
  panel_modelado_final,
  file.path(
    rutas$processed_panel_maestro_gnn,
    "panel_modelado_final.parquet"
  )
) # Exportar dataset final

# Exportar panel final (CSV)
readr::write_csv(
  panel_modelado_final,
  file.path(
    rutas$processed_panel_maestro_gnn,
    "panel_modelado_final.csv"
  )
) # Exportar dataset final csv

# Exportar auditoría (Parquet)
arrow::write_parquet(
  auditoria_variables_final,
  file.path(
    rutas$processed_panel_maestro_gnn,
    "auditoria_variables_final.parquet"
  )
) # Exportar auditoría parquet

# Exportar auditoría (CSV)
readr::write_csv(
  auditoria_variables_final,
  file.path(
    rutas$processed_panel_maestro_gnn,
    "auditoria_variables_final.csv"
  )
) # Exportar auditoría csv

cat(
  "\nEXPORTACIÓN FINALIZADA\n",
  "\nArchivos generados:",
  "\n- panel_modelado_full.parquet",
  "\n- panel_modelado_full.csv",
  "\n- panel_modelado_final.parquet",
  "\n- panel_modelado_final.csv",
  "\n- auditoria_variables_final.parquet",
  "\n- auditoria_variables_final.csv",
  "\n"
) # Resumen exportación
