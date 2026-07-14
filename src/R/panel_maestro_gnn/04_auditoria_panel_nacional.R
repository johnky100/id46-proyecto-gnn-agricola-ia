# 04_auditoria_panel_nacional
# ------------------- Bloque 1. Configuración general ------------------
# Configuración general
# ============================================================
source(here::here("config_R", "00_packages.R")) # Cargar paquetes
source(here::here("config_R", "01_paths.R")) # Cargar rutas
source(here::here("config_R", "02_global_parameters.R")) # Parámetros globales

cat("\nAUDITORÍA PANEL NACIONAL GNN | Periodo: ", anio_inicio, "-", anio_fin, " | Seed: ", seed_global, "\n", sep = "") # Resumen configuración

# ------------------- Fin Bloque 1. Configuración general ------------------


# ------------------- Bloque 2. Carga Panel Nacional ------------------
# Carga del panel auditado
# ============================================================
divipola_modelado <- read_parquet(
  "data/processed/divipola/divipola_municipio_anio.parquet"
) # Leer DIVIPOLA municipal anual


panel_nacional <- arrow::read_parquet(
  file.path(
    "data",
    "auditorias",
    "panel_master_gnn_validado_1121_municipios.parquet"
  )
) # Leer panel nacional validado

cat("\nPanel cargado: ", nrow(panel_nacional), " filas | ", ncol(panel_nacional), " columnas\n", sep = "") # Resumen carga

# ------------------- Fin Bloque 2. Carga Panel Nacional ------------------

# ------------------- Bloque 3. Validación Estructural ------------------
# Validar integridad del panel nacional
# ============================================================

panel_nacional |>
  summarise(
    municipios = n_distinct(cod_mpio),
    anios = n_distinct(anio),
    registros = n(),
    paneles_unicos = n_distinct(panel_id)
  ) |>
  print() # Resumen estructural

cat("\nDuplicados panel_id: ", sum(duplicated(panel_nacional$panel_id)), "\n", sep = "") # Verificar duplicados

# ------------------- Fin Bloque 3. Validación Estructural ------------------

# ------------------- Bloque 4. Inventario General de Variables ------------------
# Auditoría de variables
# ============================================================

inventario_variables <- tibble::tibble(
  variable = names(panel_nacional),
  tipo = sapply(
    panel_nacional,
    \(x) class(x)[1]
  ),
  nulos = sapply(
    panel_nacional,
    \(x) sum(is.na(x))
  ),
  porcentaje_nulos = round(
    sapply(
      panel_nacional,
      \(x) mean(is.na(x))
    ) * 100,
    2
  ),
  valores_unicos = sapply(
    panel_nacional,
    \(x) dplyr::n_distinct(
      x,
      na.rm = TRUE
    )
  )
) |>
  arrange(
    desc(porcentaje_nulos),
    desc(nulos)
  ) # Ordenar por porcentaje de faltantes

print(
  inventario_variables,
  n = Inf
) # Mostrar inventario completo

dir.create(
  "data/auditorias",
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta auditorías

readr::write_csv(
  inventario_variables,
  "data/auditorias/auditoria_variables_panel_nacional.csv"
) # Exportar inventario CSV

arrow::write_parquet(
  inventario_variables,
  "data/auditorias/auditoria_variables_panel_nacional.parquet"
) # Exportar inventario Parquet

cat(
  "\nInventario exportado correctamente: auditoria_variables_panel_nacional\n"
) # Confirmar exportación

# ------------------- Fin Bloque 4. Inventario General de Variables ------------------

# ------------------- Bloque 5. Auditoría Geográfica ------------------
# Identificación de municipios con información territorial faltante
# ============================================================

auditoria_geografia <- panel_nacional |>
  filter(
    is.na(cod_depto) |
      is.na(departamento) |
      is.na(latitud) |
      is.na(longitud)
  ) |>
  distinct(
    cod_mpio,
    municipio,
    departamento
  ) |>
  arrange(
    cod_mpio
  ) # Municipios con información geográfica incompleta

print(auditoria_geografia) # Mostrar municipios afectados

cat(
  "\nMunicipios con información geográfica incompleta: ",
  nrow(auditoria_geografia),
  "\n",
  sep = ""
) # Resumen auditoría geográfica

# ------------------- Fin Bloque 5. Auditoría Geográfica ------------------

# ------------------- Bloque 6. Recuperación Información Territorial ------------------
# Recuperar atributos geográficos desde DIVIPOLA
# ============================================================
catalogo_divipola <- divipola_modelado |>
  distinct(
    cod_mpio,
    cod_depto,
    departamento,
    latitud,
    longitud
  ) # Catálogo territorial oficial

panel_nacional_backup <- panel_nacional # Respaldo previo a corrección

panel_nacional <- panel_nacional |>
  select(
    -cod_depto,
    -departamento,
    -latitud,
    -longitud
  ) |>
  left_join(
    catalogo_divipola,
    by = "cod_mpio"
  ) # Recuperar atributos territoriales oficiales

panel_nacional |>
  summarise(
    na_cod_depto = sum(is.na(cod_depto)),
    na_departamento = sum(is.na(departamento)),
    na_latitud = sum(is.na(latitud)),
    na_longitud = sum(is.na(longitud))
  ) |>
  print() # Verificar recuperación territorial

# Auditoria Tulua
panel_nacional_backup |>
  filter(
    cod_mpio == "76834"
  ) |>
  distinct(
    cod_mpio,
    municipio,
    departamento,
    cod_depto,
    latitud,
    longitud
  )

panel_nacional |>
  summarise(
    municipios = n_distinct(cod_mpio),
    anios = n_distinct(anio),
    registros = n(),
    paneles_unicos = n_distinct(panel_id),
    na_cod_depto = sum(is.na(cod_depto)),
    na_departamento = sum(is.na(departamento)),
    na_latitud = sum(is.na(latitud)),
    na_longitud = sum(is.na(longitud))
  )

# ------------------- Fin Bloque 6. Recuperación Información Territorial ------------------

# ------------------- Bloque 7. Exportación Panel Corregido ------------------
# Exportar versión territorialmente validada
# ============================================================

arrow::write_parquet(
  panel_nacional,
  "data/auditorias/panel_nacional_1121_municipios_corregido.parquet"
) # Exportar panel corregido Parquet

data.table::fwrite(
  panel_nacional,
  "data/auditorias/panel_nacional_1121_municipios_corregido.csv"
) # Exportar panel corregido CSV

cat(
  "\nPanel nacional corregido exportado correctamente\n",
  "Municipios: 1121\n",
  "Años: 13\n",
  "Registros: 14573\n",
  sep = ""
) # Confirmar exportación

# ------------------- Fin Bloque 7. Exportación Panel Corregido ------------------

# ------------------- Bloque 8. Auditoría e Imputación ------------------
# Evaluación de faltantes y estrategia de imputación
# ============================================================
# Subbloque 8.1 Auditoría
auditoria_imputacion <- tibble::tibble(
  variable = names(panel_nacional),
  nulos = sapply(
    panel_nacional,
    \(x) sum(is.na(x))
  ),
  porcentaje_nulos = round(
    sapply(
      panel_nacional,
      \(x) mean(is.na(x))
    ) * 100,
    2
  )
) |>
  mutate(
    categoria = case_when(
      porcentaje_nulos == 0 ~ "Completa",
      porcentaje_nulos < 5 ~ "Baja",
      porcentaje_nulos < 20 ~ "Moderada",
      porcentaje_nulos < 70 ~ "Alta",
      TRUE ~ "Critica"
    )
  ) |>
  arrange(
    desc(porcentaje_nulos)
  ) # Clasificar variables según cobertura

# Subbloque 8.2 Estrategia
variables_excluir_imputacion <- auditoria_imputacion |>
  filter(
    porcentaje_nulos >= 70
  ) |>
  pull(variable) # Variables con cobertura crítica

variables_imputar <- auditoria_imputacion |>
  filter(
    porcentaje_nulos > 0,
    porcentaje_nulos < 70
  ) |>
  pull(variable) # Variables candidatas para imputación

variables_completas <- auditoria_imputacion |>
  filter(
    porcentaje_nulos == 0
  ) |>
  pull(variable) # Variables sin faltantes

# Resumen Ejecutivo
cat(
  "\nVariables completas: ",
  length(variables_completas),
  "\nVariables a imputar: ",
  length(variables_imputar),
  "\nVariables excluidas: ",
  length(variables_excluir_imputacion),
  "\n",
  sep = ""
) # Resumen estrategia de imputación

# ------------------- Subbloque 8.3 Variables por Categoría ------------------
cat("\nVariables completas:", length(variables_completas), "\n") # Contar variables completas
cat(paste(variables_completas, collapse = ", "), "\n") # Listar variables completas

cat("\nVariables a imputar:", length(variables_imputar), "\n") # Contar variables a imputar
cat(paste(variables_imputar, collapse = ", "), "\n") # Listar variables a imputar

cat("\nVariables excluidas:", length(variables_excluir_imputacion), "\n") # Contar variables excluidas
cat(paste(variables_excluir_imputacion, collapse = ", "), "\n") # Listar variables excluidas
# ------------------- Fin Subbloque 8.3 Variables por Categoría ------------------

# ------------------- Bloque 9. Construcción Dataset MissForest ------------------
# Preparación de datos para imputación
# ============================================================
variables_estructurales <- c(
  "cod_mpio",
  "municipio",
  "anio",
  "panel_id",
  "cod_depto",
  "departamento",
  "latitud",
  "longitud"
) # Variables de identificación y geografía

panel_imputacion <- panel_nacional |>
  dplyr::select(
    all_of(
      c(
        variables_estructurales,
        variables_imputar
      )
    )
  ) # Dataset para imputación

cat(
  "\nDataset para imputación construido correctamente\n",
  "Filas: ", nrow(panel_imputacion),
  "\nColumnas: ", ncol(panel_imputacion),
  "\nVariables estructurales: ", length(variables_estructurales),
  "\nVariables a imputar: ", length(variables_imputar),
  "\n",
  sep = ""
) # Resumen dataset imputación

# ------------------- Fin Bloque 9. Construcción Dataset MissForest ------------------

# ------------------- Subbloque 9.1 Auditoría NA Pre-Imputación ------------------
na_pre_imputacion <- tibble::tibble(
  variable = names(panel_imputacion),
  nulos = sapply(
    panel_imputacion,
    \(x) sum(is.na(x))
  )
) |>
  arrange(
    desc(nulos)
  ) # Ordenar por cantidad de NA

print(
  na_pre_imputacion,
  n = Inf
) # Mostrar auditoría

cat(
  "\nTotal NA: ",
  sum(na_pre_imputacion$nulos),
  "\n",
  sep = ""
) # Total de valores faltantes

# ------------------- Fin Subbloque 9.1 Auditoría NA Pre-Imputación ------------------

# ------------------- Subbloque 9.2 Variables Constantes ------------------
variables_constantes <- names(
  panel_imputacion
)[
  sapply(
    panel_imputacion,
    \(x) dplyr::n_distinct(
      x,
      na.rm = TRUE
    ) <= 1
  )
]

cat(
  "\nVariables constantes: ",
  length(variables_constantes),
  "\n",
  sep = ""
) # Contar variables constantes

print(
  variables_constantes
) # Mostrar variables constantes

# ------------------- Fin Subbloque 9.2 Variables Constantes ------------------

# ------------------- Bloque 10. Preparación MissForest ------------------
# Construir matriz para imputación
# ============================================================
variables_id <- c(
  "cod_mpio",
  "municipio",
  "anio",
  "panel_id",
  "cod_depto",
  "departamento",
  "latitud",
  "longitud"
) # Variables estructurales

datos_missforest <- panel_imputacion |>
  dplyr::select(
    -all_of(
      variables_id
    )
  ) # Matriz candidata a imputación

cat(
  "\nDataset MissForest\n",
  "Filas: ", nrow(datos_missforest),
  "\nColumnas: ", ncol(datos_missforest),
  "\nNA totales: ", sum(is.na(datos_missforest)),
  "\n",
  sep = ""
) # Resumen matriz imputación

# ------------------- Fin Bloque 10. Preparación MissForest ------------------

# ------------------- Subbloque 10.1 Validación Tipos de Datos ------------------
tibble::tibble(
  variable = names(datos_missforest),
  tipo = sapply(
    datos_missforest,
    \(x) class(x)[1]
  )
) |>
  print(
    n = Inf
  ) # Verificar tipos de datos

# ------------------- Fin Subbloque 10.1 Validación Tipos de Datos ------------------

# ------------------- Bloque 11. Imputación MissForest ------------------
# Imputación de valores faltantes
# ============================================================
set.seed(
  seed_global
) # Reproducibilidad

resultado_missforest <- missForest::missForest(
  xmis = datos_missforest,
  maxiter = 10,
  ntree = 200,
  variablewise = TRUE,
  verbose = TRUE
) # Ejecutar imputación

datos_imputados <- resultado_missforest$ximp # Dataset imputado

resultado_missforest$OOBerror
sum(is.na(datos_imputados))
# ------------------- Fin Bloque 11. Imputación MissForest ------------------

# ------------------- Bloque 12. Auditoría Post-Imputación ------------------
# Validación de plausibilidad de valores imputados
# ============================================================
auditoria_post_imputacion <- tibble::tibble(
  variable = names(datos_imputados),
  minimo = sapply(
    datos_imputados,
    min,
    na.rm = TRUE
  ),
  maximo = sapply(
    datos_imputados,
    max,
    na.rm = TRUE
  ),
  promedio = sapply(
    datos_imputados,
    mean,
    na.rm = TRUE
  ),
  desviacion = sapply(
    datos_imputados,
    sd,
    na.rm = TRUE
  )
)

print(
  auditoria_post_imputacion,
  n = Inf
) # Resumen estadístico completo

# ------------------- Fin Bloque 12. Auditoría Post-Imputación ------------------

# ------------------- Subbloque 12.1 Validación de Porcentajes ------------------
variables_porcentaje <- c(
  "porcentaje_permanentes",
  "pct_agro",
  "pct_natural",
  "pct_no_agro",
  "pct_otros_usos",
  "pct_propia",
  "pct_arrendada",
  "pct_colectiva",
  "pct_mixta"
)

auditoria_porcentajes <- datos_imputados |>
  summarise(
    across(
      all_of(variables_porcentaje),
      list(
        min = \(x) min(x, na.rm = TRUE),
        max = \(x) max(x, na.rm = TRUE)
      )
    )
  )

print(auditoria_porcentajes) # Verificar rangos porcentuales
glimpse(auditoria_porcentajes)

panel_nacional |>
  summarise(
    min_indice_hidrico = min(
      indice_hidrico,
      na.rm = TRUE
    ),
    max_indice_hidrico = max(
      indice_hidrico,
      na.rm = TRUE
    )
  )

setdiff(
  names(panel_nacional),
  c(
    variables_id,
    names(datos_imputados)
  )
)

variables_finales_modelo <- setdiff(
  names(panel_nacional),
  c(
    "precio_ha_promedio",
    "precio_ha_mediana",
    "precio_ha_ponderado",
    "indice_fragmentacion"
  )
) # Variables definitivas para modelado

length(variables_finales_modelo)
# ------------------- Fin Subbloque 12.1 Validación de Porcentajes ------------------

# ------------------- Bloque 13. Reconstrucción Panel Final ------------------
# Construcción del dataset definitivo para modelado GNN
variables_excluir <- c(
  "precio_ha_promedio",
  "precio_ha_mediana",
  "precio_ha_ponderado",
  "indice_fragmentacion"
) # Variables excluidas por cobertura crítica

panel_gnn_final <- bind_cols(
  panel_nacional |>
    select(
      all_of(variables_id),
      all_of(c(
        "precip_total_anual",
        "precip_sd_mensual",
        "precip_cv",
        "precip_min_mensual",
        "precip_max_mensual",
        "precip_q25",
        "precip_q75"
      ))
    ),
  datos_imputados
) # Reconstruir panel definitivo

cat(
  "\nMunicipios: ",
  n_distinct(panel_gnn_final$cod_mpio),
  "\nAños: ",
  n_distinct(panel_gnn_final$anio),
  "\nRegistros: ",
  nrow(panel_gnn_final),
  "\nNA: ",
  sum(is.na(panel_gnn_final)),
  "\n",
  sep = ""
) # Validación final

# ------------------- Bloque 14. Exportación Panel GNN Final ------------------
# Exportación dataset definitivo para modelado
# ============================================================
dir.create(
  "data/auditorias",
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta auditorías

arrow::write_parquet(
  panel_gnn_final,
  "data/auditorias/panel_gnn_final.parquet"
) # Exportar versión Parquet

data.table::fwrite(
  panel_gnn_final,
  "data/auditorias/panel_gnn_final.csv"
) # Exportar versión CSV

cat(
  "\nPanel GNN Final exportado correctamente",
  "\nFilas: ", nrow(panel_gnn_final),
  "\nColumnas: ", ncol(panel_gnn_final),
  "\nRuta Parquet: data/auditorias/panel_gnn_final.parquet",
  "\nRuta CSV: data/auditorias/panel_gnn_final.csv",
  "\n",
  sep = ""
) # Confirmar exportación

panel_gnn_final_verificacion <- arrow::read_parquet(
  "data/auditorias/panel_gnn_final.parquet"
) # Verificar lectura

panel_gnn_final_verificacion |>
  summarise(
    municipios = n_distinct(cod_mpio),
    anios = n_distinct(anio),
    registros = n(),
    variables = ncol(panel_gnn_final_verificacion)
  )
