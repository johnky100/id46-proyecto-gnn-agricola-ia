# 05_auditoria_estadistica_panel

# ------------------- Bloque 1. Configuración General ------------------
# Configuración del entorno
# ============================================================

source(here::here("config_R", "00_packages.R")) # Cargar paquetes
source(here::here("config_R", "01_paths.R")) # Cargar rutas
source(here::here("config_R", "02_global_parameters.R")) # Cargar parámetros globales

cat("\nAUDITORÍA ESTADÍSTICA PANEL GNN | Periodo: ", anio_inicio, "-", anio_fin, " | Seed: ", seed_global, "\n", sep = "") # Resumen configuración

# ------------------- Fin Bloque 1. Configuración General ------------------

# ------------------- Bloque 2. Carga Panel Final ------------------
# Lectura y validación del dataset final
# ============================================================

panel_gnn_final <- arrow::read_parquet(
  file.path(
    "data",
    "auditorias",
    "panel_gnn_final.parquet"
  )
) # Leer panel final para auditoría estadística

cat(
  "\nPanel cargado correctamente",
  "\nFilas: ", nrow(panel_gnn_final),
  "\nColumnas: ", ncol(panel_gnn_final),
  "\n",
  sep = ""
) # Resumen carga

panel_gnn_final |>
  summarise(
    municipios = n_distinct(cod_mpio),
    anios = n_distinct(anio),
    registros = n(),
    paneles_unicos = n_distinct(panel_id),
    na_totales = sum(is.na(across(everything())))
  ) |>
  print() # Validación estructural

glimpse(
  panel_gnn_final
) # Verificar estructura general

# ------------------- Bloque 3. Resumen Estadístico General ------------------
# Estadísticos descriptivos de variables numéricas
# ============================================================
variables_numericas <- names(
  panel_gnn_final
)[
  sapply(
    panel_gnn_final,
    is.numeric
  )
] # Identificar variables numéricas

resumen_estadistico <- tibble::tibble(
  variable = variables_numericas,
  minimo = sapply(
    panel_gnn_final[variables_numericas],
    min,
    na.rm = TRUE
  ),
  maximo = sapply(
    panel_gnn_final[variables_numericas],
    max,
    na.rm = TRUE
  ),
  media = sapply(
    panel_gnn_final[variables_numericas],
    mean,
    na.rm = TRUE
  ),
  mediana = sapply(
    panel_gnn_final[variables_numericas],
    median,
    na.rm = TRUE
  ),
  desviacion = sapply(
    panel_gnn_final[variables_numericas],
    sd,
    na.rm = TRUE
  )
) |>
  mutate(
    coef_variacion = round(
      ifelse(
        media == 0,
        NA,
        desviacion / abs(media)
      ),
      4
    )
  ) |>
  arrange(
    desc(coef_variacion)
  ) # Ordenar por heterogeneidad

cat(
  "\nVariables numéricas: ",
  length(variables_numericas),
  "\n",
  sep = ""
) # Resumen auditoría

print(
  resumen_estadistico,
  n = Inf
) # Mostrar estadísticos completos

# ------------------- Fin Bloque 3. Resumen Estadístico General ------------------

# ------------------- Fin Bloque 2. Carga Panel Final ------------------

# ------------------- Bloque 2. Carga Panel Final ------------------
# Lectura y validación del dataset final
# ============================================================

panel_gnn_final <- arrow::read_parquet(
  file.path(
    "data",
    "auditorias",
    "panel_gnn_final.parquet"
  )
) # Leer panel final para auditoría estadística

cat(
  "\nPanel cargado correctamente",
  "\nFilas: ", nrow(panel_gnn_final),
  "\nColumnas: ", ncol(panel_gnn_final),
  "\n",
  sep = ""
) # Resumen carga

panel_gnn_final |>
  summarise(
    municipios = n_distinct(cod_mpio),
    anios = n_distinct(anio),
    registros = n(),
    paneles_unicos = n_distinct(panel_id),
    na_totales = sum(is.na(across(everything())))
  ) |>
  print() # Validación estructural

glimpse(
  panel_gnn_final
) # Verificar estructura general

# ------------------- Fin Bloque 2. Carga Panel Final ------------------

# ------------------- Bloque 3. Resumen Estadístico General ------------------
# Estadísticos descriptivos de variables numéricas
# ============================================================
variables_numericas <- names(
  panel_gnn_final
)[
  sapply(
    panel_gnn_final,
    is.numeric
  )
] # Identificar variables numéricas

datos_numericos <- as.data.frame(
  panel_gnn_final[, ..variables_numericas]
) # Extraer variables numéricas

resumen_estadistico <- tibble::tibble(
  variable = variables_numericas,
  minimo = sapply(
    datos_numericos,
    min,
    na.rm = TRUE
  ),
  maximo = sapply(
    datos_numericos,
    max,
    na.rm = TRUE
  ),
  media = sapply(
    datos_numericos,
    mean,
    na.rm = TRUE
  ),
  mediana = sapply(
    datos_numericos,
    median,
    na.rm = TRUE
  ),
  desviacion = sapply(
    datos_numericos,
    sd,
    na.rm = TRUE
  )
) |>
  mutate(
    coef_variacion = round(
      ifelse(
        media == 0,
        NA,
        desviacion / abs(media)
      ),
      4
    )
  ) |>
  arrange(
    desc(coef_variacion)
  ) # Ordenar por heterogeneidad

cat(
  "\nVariables numéricas: ",
  length(variables_numericas),
  "\n",
  sep = ""
) # Resumen auditoría

print(
  resumen_estadistico,
  n = Inf
) # Mostrar estadísticos completos

dir.create(
  "data/auditorias",
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta auditorías

readr::write_csv(
  resumen_estadistico,
  "data/auditorias/resumen_estadistico_panel_gnn.csv"
) # Exportar resumen estadístico

arrow::write_parquet(
  resumen_estadistico,
  "data/auditorias/resumen_estadistico_panel_gnn.parquet"
) # Exportar resumen estadístico

cat(
  "\nResumen estadístico exportado correctamente\n",
  sep = ""
) # Confirmar exportación

# ------------------- Fin Bloque 3. Resumen Estadístico General ------------------

# ------------------- Bloque 4. Variables de Baja Varianza ------------------
# Identificación de variables con poca capacidad informativa
# ============================================================

datos_numericos <- panel_gnn_final |>
  dplyr::select(
    where(is.numeric)
  ) # Seleccionar variables numéricas

resultado_nzv <- caret::nearZeroVar(
  datos_numericos,
  saveMetrics = TRUE
) |>
  tibble::rownames_to_column(
    "variable"
  ) # Calcular métricas de baja varianza

variables_baja_varianza <- resultado_nzv |>
  filter(
    nzv == TRUE
  ) # Variables candidatas a exclusión

cat(
  "\nVariables evaluadas: ",
  ncol(datos_numericos),
  "\nVariables baja varianza: ",
  nrow(variables_baja_varianza),
  "\n",
  sep = ""
) # Resumen auditoría

resultado_nzv |>
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

# Variables con menor diversidad
resultado_nzv |>
  arrange(
    percentUnique
  ) |>
  select(
    variable,
    percentUnique,
    freqRatio,
    nzv
  ) |>
  head(15)

readr::write_csv(
  resultado_nzv,
  "data/auditorias/auditoria_baja_varianza.csv"
) # Exportar auditoría

arrow::write_parquet(
  resultado_nzv,
  "data/auditorias/auditoria_baja_varianza.parquet"
) # Exportar auditoría

cat(
  "\nAuditoría de baja varianza exportada correctamente\n",
  sep = ""
) # Confirmar exportación
# ------------------- Subbloque 4.1 Exportación Auditoría Baja Varianza ------------------

# ------------------- Bloque 5. Matriz de Correlaciones ------------------
# Identificación de variables redundantes
# ============================================================
variables_excluir_cor <- c(
  "anio",
  "latitud",
  "longitud"
) # Variables no analíticas para correlación

datos_correlacion <- panel_gnn_final |>
  dplyr::select(
    where(is.numeric)
  ) |>
  dplyr::select(
    -all_of(variables_excluir_cor)
  ) # Dataset para correlación

matriz_correlacion <- cor(
  datos_correlacion,
  method = "pearson",
  use = "pairwise.complete.obs"
) # Matriz de correlaciones

round(matriz_correlacion, 3) # Mostrar matriz numérica

# ------------------- Subbloque 5.1 Exportación Matriz ------------------
readr::write_csv(
  as.data.frame(
    matriz_correlacion
  ) |>
    tibble::rownames_to_column(
      "variable"
    ),
  "data/auditorias/matriz_correlacion_panel_gnn.csv"
) # Exportar matriz CSV

arrow::write_parquet(
  as.data.frame(
    matriz_correlacion
  ) |>
    tibble::rownames_to_column(
      "variable"
    ),
  "data/auditorias/matriz_correlacion_panel_gnn.parquet"
) # Exportar matriz Parquet

cat(
  "\nMatriz de correlaciones exportada correctamente\n"
) # Confirmar exportación

#------------------- Bloque 6. Construcción Datasets de Modelado ------------------
# Dataset maestro y dataset reducido
# ============================================================
variables_redundantes <- c(
  "precip_q25",
  "precip_q75",
  "precip_max_mensual",
  "n_poligonos_irrigacion",
  "area_cosechada_total"
) # Variables altamente correlacionadas

panel_gnn_maestro <- panel_gnn_final # Dataset completo

panel_gnn_reducido <- panel_gnn_final |>
  dplyr::select(
    -all_of(
      variables_redundantes
    )
  ) # Dataset reducido para modelado

cat(
  "\nDataset Maestro",
  "\nVariables: ", ncol(panel_gnn_maestro),
  "\nRegistros: ", nrow(panel_gnn_maestro),
  "\n",
  sep = ""
) # Resumen dataset maestro

cat(
  "\nDataset Reducido",
  "\nVariables: ", ncol(panel_gnn_reducido),
  "\nRegistros: ", nrow(panel_gnn_reducido),
  "\n",
  sep = ""
) # Resumen dataset reducido

# ------------------- Subbloque 6.1 Auditoría Variables Eliminadas ------------------
auditoria_reduccion <- tibble::tibble(
  variable_eliminada = variables_redundantes
) # Variables removidas

print(
  auditoria_reduccion
) # Mostrar auditoría

# ------------------- Subbloque 6.2 Dataset Reducido Automático ------------------
# Selección automática por correlación
# ============================================================

variables_auto_eliminadas <- caret::findCorrelation(
  matriz_correlacion,
  cutoff = 0.90,
  names = TRUE
) # Detectar variables redundantes automáticamente

panel_gnn_reducido_auto <- panel_gnn_final |>
  dplyr::select(
    -all_of(
      variables_auto_eliminadas
    )
  ) # Construir dataset automático

cat(
  "\nVariables eliminadas automáticamente: ",
  length(variables_auto_eliminadas),
  "\nVariables finales: ",
  ncol(panel_gnn_reducido_auto),
  "\n",
  sep = ""
) # Resumen selección automática

print(
  variables_auto_eliminadas
) # Mostrar variables eliminadas

auditoria_reduccion_auto <- tibble::tibble(
  variable_eliminada = variables_auto_eliminadas
) # Auditoría reducción automática

comparacion_datasets <- tibble::tibble(
  dataset = c(
    "panel_gnn_maestro",
    "panel_gnn_reducido",
    "panel_gnn_reducido_auto"
  ),
  variables = c(
    ncol(panel_gnn_maestro),
    ncol(panel_gnn_reducido),
    ncol(panel_gnn_reducido_auto)
  ),
  registros = c(
    nrow(panel_gnn_maestro),
    nrow(panel_gnn_reducido),
    nrow(panel_gnn_reducido_auto)
  )
) # Comparación datasets

# ------------------- Subbloque 6.3 Exportación Datasets ------------------
# Exportación de datasets y auditorías
# ============================================================

dir.create(
  "data/modelado",
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta modelado

# Dataset Maestro (51 variables)
arrow::write_parquet(
  panel_gnn_maestro,
  "data/modelado/panel_gnn_maestro_51_variables.parquet"
) # Exportar dataset maestro Parquet

data.table::fwrite(
  panel_gnn_maestro,
  "data/modelado/panel_gnn_maestro_51_variables.csv"
) # Exportar dataset maestro CSV

# Dataset Reducido Manual (46 variables)
arrow::write_parquet(
  panel_gnn_reducido,
  "data/modelado/panel_gnn_reducido_46_variables.parquet"
) # Exportar dataset reducido manual Parquet

data.table::fwrite(
  panel_gnn_reducido,
  "data/modelado/panel_gnn_reducido_46_variables.csv"
) # Exportar dataset reducido manual CSV

# Dataset Reducido Automático
arrow::write_parquet(
  panel_gnn_reducido_auto,
  "data/modelado/panel_gnn_reducido_auto.parquet"
) # Exportar dataset reducido automático Parquet

data.table::fwrite(
  panel_gnn_reducido_auto,
  "data/modelado/panel_gnn_reducido_auto.csv"
) # Exportar dataset reducido automático CSV

# Auditoría reducción manual
readr::write_csv(
  auditoria_reduccion,
  "data/modelado/auditoria_variables_eliminadas_manual.csv"
) # Exportar auditoría reducción manual CSV

arrow::write_parquet(
  auditoria_reduccion,
  "data/modelado/auditoria_variables_eliminadas_manual.parquet"
) # Exportar auditoría reducción manual Parquet


# Auditoría reducción automática
readr::write_csv(
  auditoria_reduccion_auto,
  "data/modelado/auditoria_variables_eliminadas_auto.csv"
) # Exportar auditoría reducción automática CSV

arrow::write_parquet(
  auditoria_reduccion_auto,
  "data/modelado/auditoria_variables_eliminadas_auto.parquet"
) # Exportar auditoría reducción automática Parquet

# Comparación de datasets
readr::write_csv(
  comparacion_datasets,
  "data/modelado/comparacion_datasets_modelado.csv"
) # Exportar comparación CSV

arrow::write_parquet(
  comparacion_datasets,
  "data/modelado/comparacion_datasets_modelado.parquet"
) # Exportar comparación Parquet

cat(
  "\nEXPORTACIÓN COMPLETADA",
  "\nDataset Maestro: panel_gnn_maestro_51_variables",
  "\nDataset Reducido Manual: panel_gnn_reducido_46_variables",
  "\nDataset Reducido Automático: panel_gnn_reducido_auto",
  "\nAuditoría Manual: auditoria_variables_eliminadas_manual",
  "\nAuditoría Automática: auditoria_variables_eliminadas_auto",
  "\nComparación: comparacion_datasets_modelado",
  "\n",
  sep = ""
) # Confirmar exportación
