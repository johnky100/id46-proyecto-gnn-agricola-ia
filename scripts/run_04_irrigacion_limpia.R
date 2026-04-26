# ============================================================
# SCRIPT: run_04_irrigacion_limpia.R
# OBJETIVO:
# Limpiar la base de áreas potenciales para irrigación,
# crear variables derivadas, validar redundancia del índice oficial
# y exportar en formatos tabulares y geoespaciales.
# ============================================================

source("R/00_packages.R")
source("R/06_export.R")

# install.packages("sf")
library(sf)

# ============================================================
# 1. Leer base de irrigación
# ============================================================

irrigacion_raw <- readr::read_csv(
  "data_raw/Áreas_potenciales_para_adecuación_de_tierras_con_fines_de_irrigación_20260413.csv",
  col_types = readr::cols(.default = readr::col_character()),
  locale = readr::locale(encoding = "Latin1"),
  show_col_types = FALSE,
  trim_ws = TRUE
)

# ============================================================
# 2. Función para normalizar texto
# ============================================================

normalizar_texto <- function(x) {
  x %>%
    stringr::str_replace_all("\u0081", "A") %>%
    stringr::str_replace_all("\u0089", "E") %>%
    stringr::str_replace_all("\u008d", "I") %>%
    stringr::str_replace_all("\u0093", "O") %>%
    stringr::str_replace_all("\u009a", "U") %>%
    stringr::str_replace_all("\u0091", "N") %>%
    stringr::str_replace_all("\u009c", "N") %>%
    stringr::str_replace_all("\u00D1", "N") %>%
    stringr::str_replace_all("\u00F1", "N") %>%
    stringi::stri_trans_general("Latin-ASCII") %>%
    stringr::str_squish() %>%
    stringr::str_to_upper()
}

# ============================================================
# 3. Limpieza base
# ============================================================

irrigacion_limpia <- irrigacion_raw %>%
  janitor::clean_names() %>%
  transmute(
    geometria_wkt = the_geom,
    
    categoria_fisico = normalizar_texto(fisico),
    categoria_tipo_tierra = normalizar_texto(tipo_tierras),
    categoria_necesidad_hidrica = normalizar_texto(necesidad_recurso_ha_drico),
    categoria_disponibilidad = normalizar_texto(disponibilidad),
    categoria_regulacion = normalizar_texto(regulaci_a3n),
    categoria_ecosistemico = normalizar_texto(ecosistemico),
    categoria_socioeconomico = normalizar_texto(socioecon_a3mico),
    categoria_potencial_irrigacion = normalizar_texto(potencial),
    
    area_potencial_irrigacion_ha = readr::parse_number(
      a_rea_ha,
      locale = readr::locale(decimal_mark = ",", grouping_mark = ".")
    ),
    
    consecutivo = as.integer(consecutivo)
  ) %>%
  distinct() %>%
  mutate(
    id_irrigacion = dplyr::row_number()
  )

# ============================================================
# 4. Crear scores numéricos
# ============================================================

irrigacion_limpia <- irrigacion_limpia %>%
  mutate(
    score_fisico = readr::parse_number(categoria_fisico),
    score_necesidad_hidrica = readr::parse_number(categoria_necesidad_hidrica),
    score_disponibilidad = readr::parse_number(categoria_disponibilidad),
    score_regulacion = readr::parse_number(categoria_regulacion),
    score_ecosistemico = readr::parse_number(categoria_ecosistemico),
    score_socioeconomico = readr::parse_number(categoria_socioeconomico),
    score_potencial_irrigacion = readr::parse_number(categoria_potencial_irrigacion)
  )

# ============================================================
# 5. Descomponer potencial oficial
# ============================================================

irrigacion_limpia <- irrigacion_limpia %>%
  mutate(
    potencial_nivel = readr::parse_number(categoria_potencial_irrigacion),
    potencial_necesidad = stringr::str_extract(categoria_potencial_irrigacion, "N[0-9]"),
    potencial_disponibilidad = stringr::str_extract(categoria_potencial_irrigacion, "D[0-9]"),
    potencial_regulacion = stringr::str_extract(categoria_potencial_irrigacion, "R[0-9]"),
    potencial_ecosistemico = stringr::str_extract(categoria_potencial_irrigacion, "A[0-9]"),
    potencial_socioeconomico = stringr::str_extract(categoria_potencial_irrigacion, "Q[0-9]")
  )

# ============================================================
# 6. Transformaciones e índice propio corregido
# ============================================================

irrigacion_limpia <- irrigacion_limpia %>%
  mutate(
    log_area_potencial_irrigacion = log1p(area_potencial_irrigacion_ha),
    
    # Disponibilidad:
    # D1_ALTA = 1, D2_MODERADA = 2, D3_BAJA = 3, D4_CRITICA = 4
    # Se invierte para que mayor valor represente mejor condición.
    disponibilidad_ajustada = 5 - score_disponibilidad,
    
    # Ecosistémico:
    # A1_ALTA = 1, A2_MODERADA = 2, A3_BAJA = 3
    # Se invierte para que mayor valor represente mejor condición.
    ecosistemico_ajustado = 4 - score_ecosistemico,
    
    # Regulación:
    # R1_ALTA = 1, R2_MODERADA = 2, R3_BAJA = 3
    # No se invierte porque una regulación baja implica menor restricción.
    regulacion_ajustada = score_regulacion,
    
    indice_irrigacion_propio =
      0.25 * score_fisico +
      0.25 * disponibilidad_ajustada +
      0.20 * ecosistemico_ajustado +
      0.15 * regulacion_ajustada +
      0.15 * score_socioeconomico
  )

# ============================================================
# 7. Validar redundancia del índice oficial
# ============================================================

cor_potencial_fisico <- cor(
  irrigacion_limpia$score_potencial_irrigacion,
  irrigacion_limpia$score_fisico,
  use = "complete.obs"
)

cat("Correlación potencial oficial vs físico:", cor_potencial_fisico, "\n")

# ============================================================
# 8. Validaciones de calidad
# ============================================================

glimpse(irrigacion_limpia)

cat("Total polígonos de irrigación:", nrow(irrigacion_limpia), "\n")

missing_irrigacion <- irrigacion_limpia %>%
  summarise(
    missing_geometria = sum(is.na(geometria_wkt)),
    missing_area = sum(is.na(area_potencial_irrigacion_ha)),
    missing_score_fisico = sum(is.na(score_fisico)),
    missing_score_disponibilidad = sum(is.na(score_disponibilidad)),
    missing_indice_propio = sum(is.na(indice_irrigacion_propio))
  )

print(missing_irrigacion)

summary(irrigacion_limpia$area_potencial_irrigacion_ha)
summary(irrigacion_limpia$log_area_potencial_irrigacion)
summary(irrigacion_limpia$score_potencial_irrigacion)
summary(irrigacion_limpia$indice_irrigacion_propio)

# ============================================================
# 9. Flags de calidad
# ============================================================

irrigacion_limpia <- irrigacion_limpia %>%
  mutate(
    flag_area_cero = area_potencial_irrigacion_ha <= 0,
    flag_area_extrema = area_potencial_irrigacion_ha > 100000,
    flag_sin_geometria = is.na(geometria_wkt),
    flag_potencial_redundante = cor_potencial_fisico == 1
  )

irrigacion_limpia %>%
  summarise(
    areas_cero = sum(flag_area_cero, na.rm = TRUE),
    areas_extremas = sum(flag_area_extrema, na.rm = TRUE),
    sin_geometria = sum(flag_sin_geometria, na.rm = TRUE),
    potencial_redundante = unique(flag_potencial_redundante)
  ) %>%
  print()

# ============================================================
# 10. Clustering exploratorio
# ============================================================

set.seed(5477976)

matriz_cluster <- irrigacion_limpia %>%
  select(
    score_fisico,
    score_necesidad_hidrica,
    disponibilidad_ajustada,
    ecosistemico_ajustado,
    regulacion_ajustada,
    score_socioeconomico,
    log_area_potencial_irrigacion,
    indice_irrigacion_propio
  ) %>%
  scale()

sum(is.na(matriz_cluster))

cluster_irrigacion <- kmeans(
  matriz_cluster,
  centers = 4,
  nstart = 25
)

irrigacion_limpia <- irrigacion_limpia %>%
  mutate(
    cluster_irrigacion = cluster_irrigacion$cluster
  )

print(cluster_irrigacion$centers)
print(cluster_irrigacion$size)

# ============================================================
# 11. Exportar formatos tabulares
# ============================================================

readr::write_csv(
  irrigacion_limpia,
  "data_processed/irrigacion_limpia.csv"
)

# ============================================================
# 12. Exportar formato geoespacial
# ============================================================

irrigacion_sf <- sf::st_as_sf(
  irrigacion_limpia,
  wkt = "geometria_wkt",
  crs = 4326
)

irrigacion_sf <- sf::st_make_valid(irrigacion_sf)

sf::st_write(
  irrigacion_sf,
  "data_processed/irrigacion_limpia.gpkg",
  delete_dsn = TRUE
)

# ============================================================
# 13. Mensaje final
# ============================================================

cat("Exportación finalizada correctamente.\n")
cat("Archivos generados:\n")
cat("- data_processed/irrigacion_limpia.csv\n")
# cat("- data_processed/irrigacion_limpia.xlsx\n")
cat("- data_processed/irrigacion_limpia.gpkg\n")

# ============================================================
# CONCLUSIONES IRRIGACIÓN (VERSIÓN ANALÍTICA)
# ============================================================

# 1. Integridad estructural del dataset
# El dataset contiene 32,307 polígonos sin valores faltantes
# en variables clave (geometría, área, scores), lo que confirma
# una alta calidad estructural para análisis espacial.
# :contentReference[oaicite:0]{index=0}

# 2. Hallazgo crítico: redundancia del índice oficial
# Se identificó una correlación perfecta (1.0) entre el
# potencial de irrigación oficial y el score físico.
# Esto demuestra que el índice oficial no es multivariado,
# sino una transformación directa de una sola dimensión.

# 3. Implicación metodológica
# El índice oficial no integra variables clave como:
# - disponibilidad hídrica
# - regulación
# - factores ecosistémicos
# - condiciones socioeconómicas
# Por tanto, su capacidad explicativa es limitada.

# 4. Construcción de índice propio
# Se desarrolló un índice alternativo que integra múltiples
# dimensiones con ponderaciones diferenciadas, permitiendo
# capturar la complejidad del fenómeno de irrigación.

# 5. Distribución de área altamente sesgada
# La variable área presenta:
# - mediana: 141 ha
# - máximo: 294,207 ha
# Esto confirma una distribución heavy-tail que requiere
# transformación logarítmica para modelado.

# 6. Transformación adecuada
# La variable log_area_potencial_irrigacion estabiliza
# la varianza y reduce la influencia de valores extremos,
# mejorando la robustez de modelos posteriores.

# 7. Outliers estructurales
# Se identificaron 6 polígonos con áreas extremadamente altas.
# Estos no son errores, sino casos reales que deben ser tratados
# en el modelado mediante transformaciones o ponderaciones.

# 8. Variabilidad del índice propio
# El índice construido presenta rango continuo (~1.55–3.9),
# lo que indica capacidad de diferenciación entre territorios.

# 9. Segmentación territorial (clustering)
# El análisis de clustering identifica 4 grupos bien definidos
# y balanceados, lo que evidencia heterogeneidad espacial
# en las condiciones de irrigación.

# 10. Interpretación del clustering
# Los clusters no representan niveles de calidad directa,
# sino combinaciones de:
# - condiciones físicas
# - disponibilidad de recursos
# - restricciones ambientales
# Esto permite caracterizar territorios con perfiles distintos.

# 11. Valor analítico del dataset
# El valor no está en el índice oficial, sino en la
# descomposición del potencial en múltiples dimensiones.

# 12. Preparación para integración espacial
# La presencia de geometría completa permite:
# - cruce espacial con municipios
# - agregación territorial
# - construcción de grafos espaciales

# 13. Rol dentro del pipeline
# Este dataset aporta la dimensión física del sistema agrícola,
# complementando:
# - EVA (producción)
# - PDET (contexto institucional)

# 14. Limitación clave
# El dataset es estático (no temporal), por lo que su efecto
# en modelos espaciotemporales será estructural y no dinámico.

# 15. Conclusión principal
# El potencial de irrigación no puede ser representado por una
# sola variable física; requiere un enfoque multidimensional
# que capture interacciones entre factores ambientales,
# hídricos y socioeconómicos.

# ============================================================
# FIN CONCLUSIONES IRRIGACIÓN
# ============================================================
