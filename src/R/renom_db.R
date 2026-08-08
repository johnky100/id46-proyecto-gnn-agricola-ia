# ------------------- Bloque 1. Librerías -------------------
library(arrow) # Lectura y escritura de archivos Parquet
library(data.table) # Lectura rápida de archivos CSV
library(tidyverse) # Ecosistema tidyverse
library(here)

# ------------------- Bloque 2. Cargar Bases -------------------
# Ruta: "C:/Users/john/Desktop/id46-proyecto-gnn-agricola-ia
#/avanzado-ia/data/processed/python/dataset_gnn_certificado.parquet

dataset_gnn_certificado <- read_parquet(
  here("data", "processed", "python", "dataset_gnn_certificado.parquet")
) # Leer Dataset Científico Certificado

glimpse(dataset_gnn_certificado)

dataset_gnn_certificado_renom <- dataset_gnn_certificado |>
  dplyr::rename(
    cod_municipio = cod_mpio,
    id_panel = panel_id,
    cod_departamento = cod_depto,
    desviacion_precip_mensual = precip_sd_mensual,
    coeficiente_variacion_precip = precip_cv,
    precip_minima_mensual = precip_min_mensual,
    area_total_irrigable = area_irrigable_total,
    puntaje_promedio_potencial = potencial_score_promedio,
    tasa_promedio_cosecha = tasa_cosecha_promedio,
    numero_cultivos = n_cultivos,
    numero_grupos_cultivo = n_grupos_cultivo,
    porc_cultivos_permanentes = porcentaje_permanentes,
    log_produccion_total = log_produccion_total,
    log_area_total_sembrada = log_area_sembrada_total,
    temp_punto_rocio_era5 = d2m,
    evaporacion_era5 = e,
    indice_area_foliar_vegetacion_alta = lai_hv,
    evaporacion_potencial_era5 = pev,
    escorrentia_total = ro,
    escorrentia_superficial = sro,
    radiacion_termica_descendente = strd,
    temperatura_aire_era5 = t2m,
    precipitacion_total_era5 = tp,
    componente_u_viento = u10,
    componente_v_viento = v10,
    area_total_municipio = area_total,
    porc_uso_agropecuario = pct_agro,
    porc_cobertura_natural = pct_natural,
    porc_uso_no_agropecuario = pct_no_agro,
    porc_otros_usos = pct_otros_usos,
    porc_tierra_propia = pct_propia,
    porc_tierra_arrendada = pct_arrendada,
    porc_tierra_colectiva = pct_colectiva,
    porc_tierra_mixta = pct_mixta
  ) # Renombrar variables del Dataset Científico


glimpse(dataset_gnn_certificado_renom)

# Exportar

ruta_dataset <- here("data", "processed", "dataset") # Carpeta oficial del Dataset Científico

dir.create(ruta_dataset, recursive = TRUE, showWarnings = FALSE) # Crear carpeta si no existe

write_parquet(
  dataset_gnn_certificado_renom,
  file.path(ruta_dataset, "dataset_cientifico_certificado.parquet")
) # Exportar Dataset Científico en formato Parquet

dataset_cientifico_tabular <- dataset_gnn_certificado_renom |>
  dplyr::select(-geometry) # Eliminar geometría para exportaciones tabulares

writexl::write_xlsx(
  dataset_cientifico_tabular,
  file.path(ruta_dataset, "dataset_cientifico_certificado.xlsx")
) # Exportar Dataset Científico en formato Excel

readr::write_csv(
  dataset_cientifico_tabular,
  file.path(ruta_dataset, "dataset_cientifico_certificado.csv")
) # Exportar Dataset Científico en formato CSV