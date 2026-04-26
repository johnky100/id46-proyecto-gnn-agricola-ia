# ============================================================
# SCRIPT: 04_insumos_limpio.R
# OBJETIVO:
# Limpiar la base de índice de precios de insumos agropecuarios,
# llevarla de frecuencia mensual a anual, controlar años incompletos
# y calcular variaciones anuales.
# ============================================================

source("R/00_packages.R")
source("R/06_export.R")

# ============================================================
# 0. PARAMETROS
# ============================================================

archivo_entrada <- "data_raw/Índice_de_precios_de_insumos_agrícolas_20260413.csv"

archivo_salida_csv <- "data_processed/insumos_anual_limpio.csv"
archivo_salida_xlsx <- "data_processed/insumos_anual_limpio.xlsx"
archivo_reporte_calidad <- "data_processed/reporte_calidad_insumos.csv"

# ============================================================
# 1. FUNCIONES
# ============================================================

leer_csv_robusto <- function(path, encoding = "Latin1") {
  
  lineas <- readLines(path, encoding = encoding, warn = FALSE)
  
  lineas_limpias <- lineas |>
    stringr::str_replace('^"', '') |>
    stringr::str_replace('"$', '') |>
    stringr::str_replace_all('""', '"')
  
  readr::read_csv(
    I(lineas_limpias),
    col_types = readr::cols(.default = readr::col_character()),
    locale = readr::locale(encoding = encoding),
    show_col_types = FALSE,
    trim_ws = TRUE
  )
}

convertir_numero <- function(x) {
  readr::parse_number(
    x,
    locale = readr::locale(decimal_mark = ".", grouping_mark = ",")
  )
}

calcular_variacion <- function(x) {
  (x / dplyr::lag(x) - 1) * 100
}

# ============================================================
# 2. LEER BASE ORIGINAL
# ============================================================

insumos_raw <- leer_csv_robusto(archivo_entrada)

cat("Columnas originales:\n")
print(names(insumos_raw))

# ============================================================
# 3. LIMPIEZA MENSUAL
# ============================================================

insumos_mensual_limpio <- insumos_raw %>%
  janitor::clean_names() %>%
  transmute(
    fecha = lubridate::my(fecha),
    anio = lubridate::year(fecha),
    mes = lubridate::month(fecha),
    
    indice_total = convertir_numero(indice_total),
    total_fertilizantes = convertir_numero(total_fertilizantes),
    total_plaguicidas = convertir_numero(total_plaguicidas),
    total_otros = convertir_numero(total_otros),
    
    total_simples = convertir_numero(total_simples),
    total_compuestos = convertir_numero(total_compuestos),
    total_herbicidas = convertir_numero(total_herbicidas),
    total_fungicidas = convertir_numero(total_fungicidas),
    total_insecticidas = convertir_numero(total_insecticidas),
    
    total_coadyuvantes = convertir_numero(total_coadyuvantes),
    total_reguladores = convertir_numero(total_reguladores),
    total_molusquicidas = convertir_numero(total_molusquicidas)
  ) %>%
  arrange(fecha)

# ============================================================
# 4. VALIDACION MENSUAL
# ============================================================

missing_mensual <- insumos_mensual_limpio %>%
  summarise(
    missing_fecha = sum(is.na(fecha)),
    missing_anio = sum(is.na(anio)),
    missing_mes = sum(is.na(mes)),
    missing_indice_total = sum(is.na(indice_total)),
    missing_total_fertilizantes = sum(is.na(total_fertilizantes)),
    missing_total_plaguicidas = sum(is.na(total_plaguicidas)),
    missing_total_otros = sum(is.na(total_otros))
  )

duplicados_mes <- insumos_mensual_limpio %>%
  count(fecha) %>%
  filter(n > 1)

cobertura_anual <- insumos_mensual_limpio %>%
  group_by(anio) %>%
  summarise(
    n_meses = n_distinct(mes),
    meses_disponibles = paste(sort(unique(mes)), collapse = ","),
    flag_anio_incompleto = n_meses < 12,
    .groups = "drop"
  ) %>%
  arrange(anio)

anios_incompletos <- cobertura_anual %>%
  filter(flag_anio_incompleto)

cat("Meses duplicados:", nrow(duplicados_mes), "\n")
cat("Años incompletos:", nrow(anios_incompletos), "\n")

print(missing_mensual)
print(duplicados_mes)
print(cobertura_anual)
print(anios_incompletos)

# ============================================================
# 5. AGREGACION ANUAL COMPLETA
# ============================================================

insumos_anual_todos <- insumos_mensual_limpio %>%
  group_by(anio) %>%
  summarise(
    indice_insumos_total = mean(indice_total, na.rm = TRUE),
    indice_fertilizantes = mean(total_fertilizantes, na.rm = TRUE),
    indice_plaguicidas = mean(total_plaguicidas, na.rm = TRUE),
    indice_otros = mean(total_otros, na.rm = TRUE),
    
    indice_fertilizantes_simples = mean(total_simples, na.rm = TRUE),
    indice_fertilizantes_compuestos = mean(total_compuestos, na.rm = TRUE),
    indice_herbicidas = mean(total_herbicidas, na.rm = TRUE),
    indice_fungicidas = mean(total_fungicidas, na.rm = TRUE),
    indice_insecticidas = mean(total_insecticidas, na.rm = TRUE),
    
    indice_coadyuvantes = mean(total_coadyuvantes, na.rm = TRUE),
    indice_reguladores = mean(total_reguladores, na.rm = TRUE),
    indice_molusquicidas = mean(total_molusquicidas, na.rm = TRUE),
    
    n_meses = n_distinct(mes),
    flag_anio_incompleto = n_meses < 12,
    .groups = "drop"
  ) %>%
  arrange(anio)

# ============================================================
# 6. FILTRAR SOLO AÑOS COMPLETOS
# ============================================================

insumos_anual_limpio <- insumos_anual_todos %>%
  filter(n_meses == 12) %>%
  arrange(anio)

# ============================================================
# 7. VARIACIONES ANUALES
# ============================================================

insumos_anual_limpio <- insumos_anual_limpio %>%
  mutate(
    variacion_anual_insumos_total = calcular_variacion(indice_insumos_total),
    variacion_anual_fertilizantes = calcular_variacion(indice_fertilizantes),
    variacion_anual_plaguicidas = calcular_variacion(indice_plaguicidas),
    variacion_anual_otros = calcular_variacion(indice_otros),
    
    variacion_anual_fertilizantes_simples = calcular_variacion(indice_fertilizantes_simples),
    variacion_anual_fertilizantes_compuestos = calcular_variacion(indice_fertilizantes_compuestos),
    variacion_anual_herbicidas = calcular_variacion(indice_herbicidas),
    variacion_anual_fungicidas = calcular_variacion(indice_fungicidas),
    variacion_anual_insecticidas = calcular_variacion(indice_insecticidas),
    
    variacion_anual_coadyuvantes = calcular_variacion(indice_coadyuvantes),
    variacion_anual_reguladores = calcular_variacion(indice_reguladores),
    variacion_anual_molusquicidas = calcular_variacion(indice_molusquicidas)
  )

# ============================================================
# 8. VALIDACIONES ANUALES FINALES
# ============================================================

duplicados_anio <- insumos_anual_limpio %>%
  count(anio) %>%
  filter(n > 1)

missing_anual <- insumos_anual_limpio %>%
  summarise(
    missing_anio = sum(is.na(anio)),
    missing_indice_insumos_total = sum(is.na(indice_insumos_total)),
    missing_indice_fertilizantes = sum(is.na(indice_fertilizantes)),
    missing_indice_plaguicidas = sum(is.na(indice_plaguicidas)),
    missing_indice_otros = sum(is.na(indice_otros)),
    missing_variacion_anual_insumos_total = sum(is.na(variacion_anual_insumos_total))
  )

cat("Años duplicados finales:", nrow(duplicados_anio), "\n")
print(duplicados_anio)
print(missing_anual)

# ============================================================
# 9. REPORTE DE CALIDAD
# ============================================================

reporte_calidad <- tibble::tibble(
  base = archivo_entrada,
  registros_raw = nrow(insumos_raw),
  registros_mensuales_limpios = nrow(insumos_mensual_limpio),
  anios_detectados = nrow(insumos_anual_todos),
  anios_incompletos = nrow(anios_incompletos),
  anios_finales_completos = nrow(insumos_anual_limpio),
  anio_min_final = min(insumos_anual_limpio$anio, na.rm = TRUE),
  anio_max_final = max(insumos_anual_limpio$anio, na.rm = TRUE),
  meses_duplicados = nrow(duplicados_mes),
  anios_duplicados_final = nrow(duplicados_anio),
  estado_general = ifelse(
    nrow(duplicados_mes) == 0 &
      nrow(duplicados_anio) == 0 &
      sum(is.na(insumos_mensual_limpio$fecha)) == 0 &
      sum(is.na(insumos_anual_limpio$anio)) == 0 &
      nrow(insumos_anual_limpio %>% filter(n_meses < 12)) == 0,
    "OK",
    "REVISAR"
  )
)

print(reporte_calidad)

# ============================================================
# 10. EXPORTAR RESULTADOS
# ============================================================

# 10.1 Dataset final para modelado
# Contiene solo años completos y variables listas para unir con EVA.
exportar_csv(
  insumos_anual_limpio,
  archivo_salida_csv
)

# 10.2 Reporte detallado en Excel
# Incluye datos finales, datos intermedios y evidencias de calidad.
writexl::write_xlsx(
  list(
    "01_dataset_final_modelado" = insumos_anual_limpio,
    "02_mensual_limpio" = insumos_mensual_limpio,
    "03_anual_todos_los_anios" = insumos_anual_todos,
    "04_cobertura_anual" = cobertura_anual,
    "05_anios_excluidos" = anios_incompletos,
    "06_missing_mensual" = missing_mensual,
    "07_missing_anual" = missing_anual,
    "08_duplicados_mes" = duplicados_mes,
    "09_duplicados_anio" = duplicados_anio,
    "10_reporte_calidad" = reporte_calidad
  ),
  archivo_salida_xlsx
)

# 10.3 Reporte ejecutivo de calidad
# Sirve para revisar rápidamente si el proceso quedó OK.
exportar_csv(
  reporte_calidad,
  archivo_reporte_calidad
)

# 10.4 Mensaje final en consola
cat("\nExportación finalizada correctamente.\n")
cat("Dataset final:", archivo_salida_csv, "\n")
cat("Reporte Excel:", archivo_salida_xlsx, "\n")
cat("Reporte calidad:", archivo_reporte_calidad, "\n")
cat("Años finales usados:", 
    min(insumos_anual_limpio$anio, na.rm = TRUE), 
    "-", 
    max(insumos_anual_limpio$anio, na.rm = TRUE), "\n")
cat("Años excluidos por cobertura incompleta:", 
    paste(anios_incompletos$anio, collapse = ", "), "\n")

# ============================================================
# 11. NOTA METODOLOGICA
# ============================================================

# Esta base no es municipal.
# Representa un índice nacional de precios de insumos agropecuarios.
# Fue agregada de frecuencia mensual a frecuencia anual mediante promedio.
# Se excluyeron los años incompletos para evitar sesgo en los promedios anuales.
# Al unirla con EVA municipio-año, se debe interpretar como un shock
# macrosectorial anual común a todos los municipios.
#
# Join posterior recomendado:
#
# insumos <- readr::read_csv("data_processed/insumos_anual_limpio.csv")
#
# eva_municipio_anio_insumos <- eva_municipio_anio %>%
#   left_join(insumos, by = "anio")