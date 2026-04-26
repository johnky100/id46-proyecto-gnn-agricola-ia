# ============================================================
# SCRIPT: run_05_irrigacion_municipal.R
# OBJETIVO:
# Cruzar espacialmente la capa de irrigación con municipios
# y generar variables agregadas por municipio.
# ============================================================

source("R/00_packages.R")
source("R/06_export.R")

library(sf)
library(dplyr)
library(readr)
library(janitor)
library(stringr)
library(tibble)

# ============================================================
# 1. Cargar irrigación limpia geoespacial
# ============================================================

irrigacion_sf <- sf::st_read(
  "data_processed/irrigacion_limpia.gpkg",
  quiet = TRUE
)

# ============================================================
# 2. Cargar shapefile municipal
# ============================================================

municipios_sf <- sf::st_read(
  "C:/Users/john/Desktop/proyecto-gnn-agricola/data_raw/geografia/municipios/MGN_ANM_MPIOS.shp",
  quiet = TRUE
) %>%
  janitor::clean_names()

# ============================================================
# 3. Renombrar columnas municipales
# ============================================================

municipios_sf <- municipios_sf %>%
  rename(
    codigo_departamento = dpto_ccdgo,
    codigo_municipio = mpio_cdpmp,
    nombre_municipio = mpio_cnmbr
  ) %>%
  mutate(
    codigo_departamento = stringr::str_pad(
      as.character(codigo_departamento),
      width = 2,
      side = "left",
      pad = "0"
    ),
    codigo_municipio = stringr::str_pad(
      as.character(codigo_municipio),
      width = 5,
      side = "left",
      pad = "0"
    ),
    nombre_municipio = stringr::str_to_upper(nombre_municipio)
  )

# ============================================================
# 4. Validación inicial de municipios
# ============================================================

municipios_sf %>%
  st_drop_geometry() %>%
  select(codigo_departamento, codigo_municipio, nombre_municipio) %>%
  head(10) %>%
  print()

cat("Total municipios cargados:", nrow(municipios_sf), "\n")

# ============================================================
# 5. Validar geometrías
# ============================================================

irrigacion_sf <- sf::st_make_valid(irrigacion_sf)
municipios_sf <- sf::st_make_valid(municipios_sf)

# ============================================================
# 6. Homologar CRS
# ============================================================

municipios_sf <- sf::st_transform(
  municipios_sf,
  sf::st_crs(irrigacion_sf)
)

# ============================================================
# 7. Proyectar a CRS métrico para cálculo de áreas
# ============================================================

crs_area <- 3116

irrigacion_area <- sf::st_transform(irrigacion_sf, crs_area)
municipios_area <- sf::st_transform(municipios_sf, crs_area)

# ============================================================
# 8. Calcular área geométrica original de cada polígono
# ============================================================

irrigacion_area <- irrigacion_area %>%
  mutate(
    area_geom_original_m2 = as.numeric(sf::st_area(.))
  )

# ============================================================
# 9. Intersección espacial
# ============================================================

interseccion <- sf::st_intersection(
  irrigacion_area,
  municipios_area
)

# ============================================================
# 10. Calcular pesos de intersección
# ============================================================

# Nota:
# El warning de st_intersection es normal.
# Las variables se asumen constantes por fragmento,
# pero lo corregimos con ponderación por área.

interseccion <- interseccion %>%
  mutate(
    area_interseccion_m2 = as.numeric(sf::st_area(.)),
    peso_area = area_interseccion_m2 / area_geom_original_m2,
    area_irrigacion_ponderada_ha = area_potencial_irrigacion_ha * peso_area
  )

# ============================================================
# 11. VALIDACIÓN CRÍTICA (NO OMITIR)
# ============================================================

validacion_pesos <- interseccion %>%
  st_drop_geometry() %>%
  summarise(
    peso_min = min(peso_area, na.rm = TRUE),
    peso_max = max(peso_area, na.rm = TRUE),
    pesos_mayores_1 = sum(peso_area > 1.0001, na.rm = TRUE),
    pesos_negativos = sum(peso_area < 0, na.rm = TRUE),
    n_intersecciones = n()
  )

print(validacion_pesos)

# ============================================================
# 12. Agregación municipal con múltiples métricas
# ============================================================

irrigacion_municipal <- interseccion %>%
  st_drop_geometry() %>%
  group_by(
    codigo_departamento,
    codigo_municipio,
    nombre_municipio
  ) %>%
  summarise(
    # Área total ponderada
    area_potencial_irrigacion_ha = sum(
      area_irrigacion_ponderada_ha,
      na.rm = TRUE
    ),
    
    # Índice de irrigación: métricas centrales
    indice_irrigacion_media_pond = weighted.mean(
      indice_irrigacion_propio,
      w = area_irrigacion_ponderada_ha,
      na.rm = TRUE
    ),
    
    indice_irrigacion_media_simple = mean(
      indice_irrigacion_propio,
      na.rm = TRUE
    ),
    
    # Índice de irrigación: potencial alto y heterogeneidad
    indice_irrigacion_max = max(
      indice_irrigacion_propio,
      na.rm = TRUE
    ),
    
    indice_irrigacion_p75 = as.numeric(
      quantile(
        indice_irrigacion_propio,
        probs = 0.75,
        na.rm = TRUE
      )
    ),
    
    indice_irrigacion_sd = sd(
      indice_irrigacion_propio,
      na.rm = TRUE
    ),
    
    area_alto_potencial_ha = sum(
      area_irrigacion_ponderada_ha[indice_irrigacion_propio >= 3],
      na.rm = TRUE
    ),
    
    prop_area_alto_potencial = area_alto_potencial_ha /
      area_potencial_irrigacion_ha,
    
    # Scores ponderados por área
    score_fisico_media_pond = weighted.mean(
      score_fisico,
      w = area_irrigacion_ponderada_ha,
      na.rm = TRUE
    ),
    
    score_necesidad_hidrica_media_pond = weighted.mean(
      score_necesidad_hidrica,
      w = area_irrigacion_ponderada_ha,
      na.rm = TRUE
    ),
    
    disponibilidad_media_pond = weighted.mean(
      disponibilidad_ajustada,
      w = area_irrigacion_ponderada_ha,
      na.rm = TRUE
    ),
    
    ecosistemico_media_pond = weighted.mean(
      ecosistemico_ajustado,
      w = area_irrigacion_ponderada_ha,
      na.rm = TRUE
    ),
    
    regulacion_media_pond = weighted.mean(
      regulacion_ajustada,
      w = area_irrigacion_ponderada_ha,
      na.rm = TRUE
    ),
    
    socioeconomico_media_pond = weighted.mean(
      score_socioeconomico,
      w = area_irrigacion_ponderada_ha,
      na.rm = TRUE
    ),
    
    potencial_oficial_media_pond = weighted.mean(
      score_potencial_irrigacion,
      w = area_irrigacion_ponderada_ha,
      na.rm = TRUE
    ),
    
    # Trazabilidad espacial
    n_poligonos_irrigacion = n_distinct(id_irrigacion),
    n_fragmentos_interseccion = n(),
    
    .groups = "drop"
  ) %>%
  mutate(
    log_area_potencial_irrigacion = log1p(area_potencial_irrigacion_ha)
  )

# ============================================================
# 13. Validación de conservación de área
# ============================================================

area_original_total <- sum(
  irrigacion_sf$area_potencial_irrigacion_ha,
  na.rm = TRUE
)

area_municipal_total <- sum(
  irrigacion_municipal$area_potencial_irrigacion_ha,
  na.rm = TRUE
)

validacion_area <- tibble::tibble(
  area_original_total_ha = area_original_total,
  area_municipal_total_ha = area_municipal_total,
  diferencia_ha = area_original_total - area_municipal_total,
  porcentaje_conservado = area_municipal_total / area_original_total * 100
)

print(validacion_area)

# ============================================================
# 14. Unir resultados a geometría municipal
# ============================================================

irrigacion_municipal_sf <- municipios_area %>%
  left_join(
    irrigacion_municipal,
    by = c(
      "codigo_departamento",
      "codigo_municipio",
      "nombre_municipio"
    )
  )

# ============================================================
# 15. Exportar resultados
# ============================================================

readr::write_csv(
  irrigacion_municipal,
  "data_processed/irrigacion_municipal.csv"
)

sf::st_write(
  irrigacion_municipal_sf,
  "data_processed/irrigacion_municipal.gpkg",
  layer = "irrigacion_municipal",
  delete_layer = TRUE
)

# ============================================================
# 16. Mensaje final
# ============================================================

cat("Cruce espacial terminado correctamente.\n")
cat("Archivos generados:\n")
cat("- data_processed/irrigacion_municipal.csv\n")
cat("- data_processed/irrigacion_municipal.gpkg\n")
cat("Municipios con irrigación:", nrow(irrigacion_municipal), "\n")

# ============================================================
# CONCLUSIONES CRUCE ESPACIAL IRRIGACIÓN → MUNICIPIO
# ============================================================

# 1. Integridad del cruce espacial
# El proceso de intersección generó 37,959 fragmentos espaciales,
# lo cual es consistente con la fragmentación esperada al cruzar
# polígonos de irrigación con límites municipales.
# :contentReference[oaicite:0]{index=0}

# 2. Validez de la ponderación por área
# Los pesos calculados cumplen:
# - valores entre 0 y 1
# - ausencia de valores negativos
# - ausencia de sobreasignación (>1)
# Esto confirma consistencia geométrica en la intersección.

# 3. Conservación de área
# La diferencia entre el área total original y la agregada
# a nivel municipal es despreciable (<0.001%).
# Esto garantiza que el cruce espacial no introduce sesgos
# en la magnitud del fenómeno.

# 4. Robustez del proceso espacial
# La combinación de:
# - validación de geometrías
# - transformación a CRS métrico
# - ponderación por área
# produce un pipeline espacial confiable.

# 5. Agregación municipal multivariada
# Se construyen múltiples representaciones del potencial:
# - media ponderada (enfoque tradicional)
# - media simple (control)
# - percentil 75 (zonas de alto potencial)
# - máximo (potencial máximo local)
# - desviación estándar (heterogeneidad interna)

# 6. Limitación del promedio ponderado
# El uso exclusivo de ponderación por área puede ocultar
# zonas pequeñas con alto potencial.
# Por ello, es necesario complementar con métricas de distribución.

# 7. Importancia del percentil 75
# El índice p75 captura la presencia de zonas de alto potencial
# dentro del municipio, siendo más representativo que la media
# en contextos heterogéneos.

# 8. Heterogeneidad intra-municipal
# La inclusión de la desviación estándar permite modelar
# la variabilidad interna del territorio, un factor que
# no es capturado por métricas promedio.

# 9. Variable de proporción de alto potencial
# La proporción de área con índice ≥ 3 introduce una medida
# directa de capacidad productiva potencial del municipio.

# 10. Trazabilidad espacial
# Las variables:
# - n_poligonos_irrigacion
# - n_fragmentos_interseccion
# permiten evaluar la granularidad espacial y la complejidad
# del territorio.

# 11. Validación del modelo conceptual
# El cruce espacial confirma que el potencial de irrigación
# es un fenómeno distribuido y no uniforme dentro del municipio.

# 12. Preparación para modelado
# El resultado final genera una base a nivel municipal
# compatible con:
# - EVA (producción)
# - grafos espaciales
# - modelos GNN

# 13. Valor analítico
# El valor no está en una única métrica, sino en el conjunto
# de indicadores que capturan:
# - nivel
# - distribución
# - heterogeneidad

# 14. Limitación estructural
# El dataset sigue siendo estático en el tiempo, por lo que
# su contribución en modelos espaciotemporales será estructural.

# 15. Conclusión principal
# El potencial de irrigación a nivel municipal no puede
# representarse adecuadamente mediante promedios simples;
# requiere un enfoque multivariado que capture la estructura
# interna del territorio.

# ============================================================
# FIN CONCLUSIONES CRUCE ESPACIAL
# ============================================================