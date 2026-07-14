# 01_construccion_dataset_modelado.R
# Construcción Panel Maestro GNN Agrícola
# 01_build_panel_master
# ============================================================

# ------------------- Bloque 1. Configuración general ------------------
# Configuración general
# ============================================================

source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Parámetros globales
# ------------------- Fin Bloque 1. Configuración general ------------------

# ------------------- Bloque 2. Carga de bases ------------------
# Carga de datos
# ============================================================

chirps_modelado <- fread(
  "data/processed/chirps/chirps_variables_finales.csv"
) # Leer CHIRPS

divipola_modelado <- read_parquet(
  "data/processed/divipola/divipola_municipio_anio.parquet"
) # Leer DIVIPOLA

era5_modelado <- read_parquet(
  "data/processed/era5/era5_municipal_anual.parquet"
) # Leer ERA5

eva_modelado <- fread(
  "data/processed/eva/eva_gnn_final.csv"
) # Leer EVA

cna_modelado <- read_parquet(
  "data/processed/cna/cna_modelado.parquet"
) # Leer CNA

irrigacion_modelado <- read_parquet(
  "data/processed/irrigacion/irrigacion_municipio.parquet"
) # Leer irrigación

precio_tierra_modelado <- read_parquet(
  "data/processed/precios_tierra/precios_tierra_features.parquet"
) # Leer precios tierra

cat(
  "\nBases cargadas |",
  "DIVIPOLA:", nrow(divipola_modelado),
  "| CHIRPS:", nrow(chirps_modelado),
  "| EVA:", nrow(eva_modelado),
  "| ERA5:", nrow(era5_modelado),
  "\n"
) # Resumen de carga

cat(
  "\nMunicipios únicos |",
  "DIVIPOLA:", n_distinct(divipola_modelado$cod_mpio),
  "| CHIRPS:", n_distinct(sprintf("%05d", chirps_modelado$cod_mpio)),
  "| ERA5:", n_distinct(era5_modelado$cod_mpio),
  "| EVA:", n_distinct(eva_modelado$cod_mpio),
  "\n"
) # Auditoría territorial rápida

cat(
  "\nCobertura panel-año |",
  "DIVIPOLA:", n_distinct(divipola_modelado$panel_id),
  "| CHIRPS:", n_distinct(
    paste0(
      sprintf("%05d", chirps_modelado$cod_mpio),
      "_",
      chirps_modelado$anio
    )
  ),
  "| EVA:", n_distinct(
    paste0(
      sprintf("%05d", eva_modelado$cod_mpio),
      "_",
      eva_modelado$anio
    )
  ),
  "\n"
) # Auditoría panel-año
# ------------------- Fin Bloque 2. Carga de bases ------------------

# ------------------- Bloque 3. Homologación de identificadores ------------------
# Homologación de identificadores
# ============================================================

# CHIRPS
chirps_modelado <- chirps_modelado %>%
  mutate(
    cod_mpio = sprintf("%05d", cod_mpio),
    anio = as.integer(anio),
    panel_id = paste0(cod_mpio, "_", anio)
  ) # Estandarizar llaves CHIRPS

# EVA
eva_modelado <- eva_modelado %>%
  mutate(
    cod_depto = sprintf("%02d", cod_depto),
    cod_mpio = sprintf("%05d", cod_mpio),
    anio = as.integer(anio),
    panel_id = paste0(cod_mpio, "_", anio)
  ) # Estandarizar llaves EVA

# DIVIPOLA
divipola_modelado <- divipola_modelado %>%
  mutate(
    anio = as.integer(anio)
  ) # Validar tipo de año DIVIPOLA

# ERA5
era5_modelado <- era5_modelado %>%
  mutate(
    cod_mpio = stringr::str_pad(
      as.character(cod_mpio),
      width = 5,
      side = "left",
      pad = "0"
    ),
    anio = as.integer(anio)
  ) # Estandarizar llaves ERA5

# CNA
cna_modelado <- cna_modelado %>%
  mutate(
    cod_mpio = stringr::str_pad(
      as.character(cod_mpio),
      width = 5,
      side = "left",
      pad = "0"
    )
  ) # Estandarizar llaves CNA

# Irrigación
irrigacion_modelado <- irrigacion_modelado %>%
  mutate(
    cod_mpio = stringr::str_pad(
      as.character(cod_mpio),
      width = 5,
      side = "left",
      pad = "0"
    )
  ) # Estandarizar llaves irrigación

# Precio de tierra
precio_tierra_modelado <- precio_tierra_modelado %>%
  mutate(
    cod_depto = stringr::str_pad(
      as.character(cod_depto),
      width = 2,
      side = "left",
      pad = "0"
    ),
    cod_mpio = stringr::str_pad(
      as.character(cod_mpio),
      width = 5,
      side = "left",
      pad = "0"
    )
  ) # Estandarizar llaves precio tierra

# Auditoría de registros
cat(
  "\nLlaves estandarizadas |",
  "CHIRPS:", nrow(chirps_modelado),
  "| EVA:", nrow(eva_modelado),
  "| DIVIPOLA:", nrow(divipola_modelado),
  "| ERA5:", nrow(era5_modelado),
  "| CNA:", nrow(cna_modelado),
  "| IRRIGACIÓN:", nrow(irrigacion_modelado),
  "| PRECIO:", nrow(precio_tierra_modelado),
  "\n"
) # Resumen de registros

# Auditoría de cobertura panel-año
list(
  CHIRPS = n_distinct(chirps_modelado$panel_id),
  EVA = n_distinct(eva_modelado$panel_id),
  DIVIPOLA = n_distinct(divipola_modelado$panel_id)
) # Verificar paneles únicos

# Auditoría cobertura ERA5
era5_modelado %>%
  count(variable) %>%
  arrange(variable) # Verificar registros por variable

# Auditoría calidad temporal ERA5
era5_modelado %>%
  count(meses_obs) %>%
  arrange(meses_obs) # Verificar cobertura temporal

# Auditoría tipos de llave
sapply(
  list(
    chirps = chirps_modelado$cod_mpio,
    eva = eva_modelado$cod_mpio,
    divipola = divipola_modelado$cod_mpio,
    era5 = era5_modelado$cod_mpio,
    cna = cna_modelado$cod_mpio,
    irrigacion = irrigacion_modelado$cod_mpio,
    precio = precio_tierra_modelado$cod_mpio
  ),
  class
) # Verificar tipo de cod_mpio

# Auditoría detallada ERA5
era5_modelado %>%
  count(meses_obs, variable) # Verificar distribución por variable

era5_modelado %>%
  filter(meses_obs == 0) %>%
  distinct(cod_mpio, municipio) %>%
  arrange(cod_mpio) # Identificar municipios con meses_obs = 0

era5_modelado %>%
  filter(meses_obs == 0) %>%
  count(variable) # Verificar variables afectadas

# ------------------- Fin Bloque 3. Homologación de identificadores ------------------

# ------------------- Bloque 4. Transformación ERA5 Wide ------------------
# Conversión formato largo a ancho
# ============================================================
era5_wide <- era5_modelado %>%
  pivot_wider(
    id_cols = c(
      cod_mpio,
      municipio,
      anio,
      meses_obs
    ),
    names_from = variable,
    values_from = valor,
    names_prefix = "era5_"
  ) %>%
  mutate(
    panel_id = paste0(cod_mpio, "_", anio)
  ) %>%
  select(
    panel_id,
    cod_mpio,
    municipio,
    anio,
    meses_obs,
    everything()
  ) # Convertir ERA5 a formato ancho

# Auditoría de dimensiones
cat(
  "\nERA5 Wide |",
  "Filas:", nrow(era5_wide),
  "| Columnas:", ncol(era5_wide),
  "\n"
) # Resumen ERA5 Wide

# Auditoría de paneles únicos
cat(
  "\nPaneles únicos ERA5:",
  n_distinct(era5_wide$panel_id),
  "\n"
) # Validar integridad municipio-año

# Auditoría de municipios
cat(
  "\nMunicipios ERA5 Wide:",
  n_distinct(era5_wide$cod_mpio),
  "\n"
) # Validar universo territorial

# Auditoría de duplicados
era5_wide %>%
  count(panel_id) %>%
  filter(n > 1) # Verificar duplicados

# Auditoría de calidad temporal
era5_wide %>%
  count(meses_obs) %>%
  arrange(meses_obs) # Verificar distribución temporal

# Auditoría de valores faltantes
era5_wide %>%
  summarise(
    across(
      starts_with("era5_"),
      ~ sum(is.na(.))
    )
  ) # Revisar faltantes por variable climática

# Vista rápida de variables
names(era5_wide)

# Vista estructural
glimpse(era5_wide)
glimpse(cna_modelado)
glimpse(irrigacion_modelado)
glimpse(precio_tierra_modelado)
# ------------------- Fin Bloque 4. Transformación ERA5 Wide ------------------

# ------------------- Bloque 5. Construcción Panel Maestro ------------------
# Integración de fuentes
# ============================================================
panel_master <- divipola_modelado %>%
  
  left_join(
    chirps_modelado,
    by = c(
      "panel_id",
      "cod_mpio",
      "anio"
    )
  ) %>% # Incorporar CHIRPS
  
  left_join(
    era5_wide,
    by = c(
      "panel_id",
      "cod_mpio",
      "anio"
    )
  ) %>% # Incorporar ERA5
  
  left_join(
    eva_modelado,
    by = c(
      "panel_id",
      "cod_mpio",
      "anio"
    )
  ) %>% # Incorporar EVA
  
  left_join(
    cna_modelado,
    by = "cod_mpio"
  ) %>% # Incorporar CNA
  
  left_join(
    irrigacion_modelado,
    by = "cod_mpio"
  ) %>% # Incorporar irrigación
  
  left_join(
    precio_tierra_modelado,
    by = "cod_mpio"
  ) # Incorporar precios de tierra

# Auditoría general
cat(
  "\nPanel Maestro |",
  "Filas:", nrow(panel_master),
  "| Columnas:", ncol(panel_master),
  "\n"
) # Resumen general

# Auditoría territorial
cat(
  "\nMunicipios:",
  n_distinct(panel_master$cod_mpio),
  "| Paneles:",
  n_distinct(panel_master$panel_id),
  "\n"
) # Verificar integridad

# Auditoría de duplicados
panel_master %>%
  count(panel_id) %>%
  filter(n > 1) # Verificar duplicados

# Auditoría de valores faltantes
panel_master %>%
  summarise(
    across(
      everything(),
      ~ sum(is.na(.))
    )
  ) # Resumen de NA

# Vista estructural
glimpse(panel_master)

panel_master %>%
  filter(
    is.na(area_sembrada_total)
  ) %>%
  distinct(
    cod_mpio,
    municipio.x
  ) %>%
  arrange(cod_mpio)

panel_master %>%
  filter(is.na(area_sembrada_total)) %>%
  distinct(cod_mpio) %>%
  count()

setdiff(
  unique(divipola_modelado$cod_mpio),
  unique(eva_modelado$cod_mpio)
)

n_distinct(eva_modelado$cod_mpio)

eva_modelado %>%
  filter(is.na(area_sembrada_total)) %>%
  distinct(cod_mpio) %>%
  count()

eva_modelado %>%
  summarise(
    municipios = n_distinct(cod_mpio),
    na_area = sum(is.na(area_sembrada_total))
  )

panel_master %>%
  filter(is.na(area_sembrada_total)) %>%
  count(cod_mpio, sort = TRUE)

panel_master %>%
  group_by(cod_mpio) %>%
  summarise(
    na_eva = sum(is.na(area_sembrada_total))
  ) %>%
  filter(na_eva > 0) %>%
  arrange(desc(na_eva))

panel_master %>%
  group_by(cod_mpio) %>%
  summarise(
    na_eva = sum(is.na(area_sembrada_total))
  ) %>%
  count(na_eva) %>%
  arrange(desc(na_eva))
# ------------------- Fin Bloque 5. Construcción Panel Maestro ------------------

# ------------------- Bloque 6. Limpieza estructural del panel ------------------
# Limpieza de columnas redundantes
# ============================================================

names(panel_master) # Revisar nombres actuales

# Eliminar columnas territoriales duplicadas provenientes de Precio Tierra
panel_master_clean <- panel_master %>%
  select(
    -cod_depto,
    -departamento
  ) %>% # Eliminar versiones incompletas de Precio Tierra
  rename(
    cod_depto = cod_depto.x,
    departamento = departamento.x,
    municipio = municipio.x
  ) %>% # Recuperar metadatos oficiales de DIVIPOLA
  select(
    -cod_depto.y,
    -departamento.y,
    -municipio.y,
    -municipio.x.x,
    -municipio.y.y,
    -municipio.x.x.x,
    -municipio.y.y.y
  ) # Eliminar columnas redundantes generadas por joins

# Auditoría dimensional
cat(
  "\nPanel limpio |",
  "Filas:", nrow(panel_master_clean),
  "| Columnas:", ncol(panel_master_clean),
  "\n"
) # Resumen dimensional

# Auditoría territorial
cat(
  "\nMunicipios:",
  n_distinct(panel_master_clean$cod_mpio),
  "| Paneles:",
  n_distinct(panel_master_clean$panel_id),
  "\n"
) # Validar integridad territorial

# Auditoría de duplicados
panel_master_clean %>%
  count(panel_id) %>%
  filter(n > 1) # Verificar duplicados

# Auditoría metadatos territoriales
panel_master_clean %>%
  summarise(
    na_cod_depto = sum(is.na(cod_depto)),
    na_departamento = sum(is.na(departamento)),
    na_municipio = sum(is.na(municipio))
  ) # Verificar integridad territorial

# Auditoría cobertura Precio Tierra
panel_master_clean %>%
  summarise(
    municipios_precio_tierra = n_distinct(
      cod_mpio[!is.na(precio_ha_promedio)]
    ),
    registros_precio_tierra = sum(
      !is.na(precio_ha_promedio)
    )
  ) # Verificar cobertura Precio Tierra

# Auditoría estructura final
glimpse(panel_master_clean)

panel_master_clean <- panel_master_clean %>%
  select(-meses_obs)
# ------------------- Fin Bloque 6. Limpieza estructural del panel ------------------

# ------------------- Bloque 7. Validación Final ------------------
# Validaciones antes de exportar
# ============================================================
stopifnot(
  sum(duplicated(panel_master_clean$panel_id)) == 0
) # Verificar ausencia de duplicados

stopifnot(
  nrow(panel_master_clean) > 0
) # Verificar existencia de registros

stopifnot(
  ncol(panel_master_clean) > 0
) # Verificar existencia de variables

cat(
  "\nValidación final aprobada:",
  nrow(panel_master_clean),
  "filas |",
  ncol(panel_master_clean),
  "columnas |",
  n_distinct(panel_master_clean$cod_mpio),
  "municipios\n"
) # Resumen validación

sum(is.na(panel_master_clean$panel_id))
n_distinct(panel_master_clean$panel_id)
# ------------------- Fin Bloque 7. Validación Final ------------------

# ------------------- Bloque 8. Exportación Panel Maestro ------------------
# Exportación panel consolidado
# ============================================================

arrow::write_parquet(
  panel_master_clean,
  file.path(
    rutas$processed_panel_maestro_gnn,
    "panel_master_clean.parquet"
  )
) # Exportar panel maestro limpio

cat(
  "\nPanel maestro exportado:",
  nrow(panel_master_clean),
  "filas |",
  ncol(panel_master_clean),
  "columnas\n"
) # Resumen exportación

cat(
  "Ruta archivo:",
  file.path(
    rutas$processed_panel_maestro_gnn,
    "panel_master_clean.parquet"
  ),
  "\n"
) # Mostrar ubicación del archivo

# ------------------- Fin Bloque 8. Exportación Panel Maestro ------------------
