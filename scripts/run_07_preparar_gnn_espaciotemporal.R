# ============================================================
# SCRIPT: run_07_preparar_gnn_espaciotemporal.R
# OBJETIVO:
# Preparar base municipal-año balanceada y grafo espacial
# para modelo GNN espaciotemporal.
# ============================================================

source("R/00_packages.R")
source("R/06_export.R")

library(sf)
library(dplyr)
library(readr)
library(janitor)
library(spdep)
library(stringr)
library(tibble)

# ============================================================
# 1. Cargar base validada EVA + irrigación
# ============================================================

base_gnn <- readr::read_csv(
  "data_processed/base_modelo_eva_irrigacion_validacion.csv",
  show_col_types = FALSE
) %>%
  janitor::clean_names()

# ============================================================
# 2. Cargar geometría municipal
# ============================================================

municipios_sf <- sf::st_read(
  "data_processed/irrigacion_municipal.gpkg",
  layer = "irrigacion_municipal",
  quiet = TRUE
) %>%
  janitor::clean_names()

# ============================================================
# 3. Homologar códigos
# ============================================================

base_gnn <- base_gnn %>%
  mutate(
    codigo_municipio = stringr::str_pad(
      as.character(codigo_municipio),
      width = 5,
      side = "left",
      pad = "0"
    )
  )

municipios_sf <- municipios_sf %>%
  mutate(
    codigo_municipio = stringr::str_pad(
      as.character(codigo_municipio),
      width = 5,
      side = "left",
      pad = "0"
    )
  )

# ============================================================
# 4. Filtrar municipios con geometría disponible
# ============================================================

base_gnn <- base_gnn %>%
  filter(codigo_municipio %in% municipios_sf$codigo_municipio)

municipios_modelo <- municipios_sf %>%
  filter(codigo_municipio %in% unique(base_gnn$codigo_municipio)) %>%
  arrange(codigo_municipio)

base_gnn <- base_gnn %>%
  filter(codigo_municipio %in% municipios_modelo$codigo_municipio) %>%
  arrange(codigo_municipio, anio)

# ============================================================
# 5. Validar panel original
# ============================================================

validacion_panel_original <- base_gnn %>%
  count(codigo_municipio) %>%
  summarise(
    municipios = n(),
    min_anios = min(n),
    max_anios = max(n)
  )

print(validacion_panel_original)

# ============================================================
# 6. Balancear panel temporal
# ============================================================

n_anios_esperados <- base_gnn %>%
  summarise(n_anios = n_distinct(anio)) %>%
  pull(n_anios)

municipios_completos <- base_gnn %>%
  count(codigo_municipio) %>%
  filter(n == n_anios_esperados) %>%
  pull(codigo_municipio)

base_gnn_balanceada <- base_gnn %>%
  filter(codigo_municipio %in% municipios_completos) %>%
  arrange(anio, codigo_municipio)

municipios_modelo_balanceado <- municipios_modelo %>%
  filter(codigo_municipio %in% municipios_completos) %>%
  arrange(codigo_municipio)

validacion_panel_balanceado <- base_gnn_balanceada %>%
  count(codigo_municipio) %>%
  summarise(
    municipios = n(),
    min_anios = min(n),
    max_anios = max(n)
  )

print(validacion_panel_balanceado)

# ============================================================
# 7. Construir grafo espacial robusto con k vecinos cercanos
# ============================================================

coords <- municipios_modelo_balanceado %>%
  sf::st_centroid() %>%
  sf::st_coordinates()

vecinos <- spdep::knearneigh(coords, k = 5) %>%
  spdep::knn2nb()

componentes_grafo <- spdep::n.comp.nb(vecinos)

print(componentes_grafo$nc)

matriz_adyacencia <- spdep::nb2mat(
  vecinos,
  style = "B",
  zero.policy = TRUE
)

print(dim(matriz_adyacencia))

# ============================================================
# 8. Crear edge list para PyTorch Geometric
# ============================================================

edge_index <- which(matriz_adyacencia == 1, arr.ind = TRUE) %>%
  as.data.frame() %>%
  rename(
    source = row,
    target = col
  ) %>%
  mutate(
    source = source - 1,
    target = target - 1
  )

# ============================================================
# 9. Variables finales para GNN
# ============================================================

variables_gnn <- c(
  "codigo_municipio",
  "anio",
  "log_rendimiento",
  "rendimiento_promedio_t_ha",
  "z_indice_irrigacion_p75",
  "z_prop_area_alto_potencial",
  "z_socioeconomico_media_pond",
  "z_indice_diversificacion_cultivos",
  "z_indice_concentracion_cultivos_hhi",
  "z_intensidad_agricola",
  "z_log_area_potencial_irrigacion"
)

base_gnn_final <- base_gnn_balanceada %>%
  select(any_of(variables_gnn)) %>%
  filter(
    !is.na(log_rendimiento),
    !is.na(z_indice_irrigacion_p75),
    !is.na(z_socioeconomico_media_pond)
  ) %>%
  arrange(anio, codigo_municipio)

# ============================================================
# 10. Validaciones finales
# ============================================================

validacion_gnn_final <- base_gnn_final %>%
  count(codigo_municipio) %>%
  summarise(
    municipios = n(),
    min_anios = min(n),
    max_anios = max(n),
    filas = sum(n)
  )

print(validacion_gnn_final)

cat("Municipios en grafo:", nrow(municipios_modelo_balanceado), "\n")
cat("Filas esperadas:", nrow(municipios_modelo_balanceado) * n_anios_esperados, "\n")
cat("Filas reales:", nrow(base_gnn_final), "\n")
cat("Aristas:", nrow(edge_index), "\n")

# ============================================================
# 11. Exportar
# ============================================================

# Asegurar grafo no dirigido (simetrizar aristas)
edge_index_undirected <- dplyr::bind_rows(
  edge_index,
  edge_index %>% rename(source = target, target = source)
) %>%
  distinct()

# Validación rápida
cat("Aristas dirigidas:", nrow(edge_index), "\n")
cat("Aristas no dirigidas:", nrow(edge_index_undirected), "\n")

# Exportar base
readr::write_csv(
  base_gnn_final,
  "data_processed/base_gnn_espaciotemporal.csv"
)

# Exportar edge_index corregido
readr::write_csv(
  edge_index_undirected,
  "data_processed/edge_index_municipal.csv"
)

# Exportar nodos
readr::write_csv(
  municipios_modelo_balanceado %>%
    st_drop_geometry() %>%
    select(codigo_municipio, nombre_municipio),
  "data_processed/nodos_municipales.csv"
)

# Guardar matriz (opcional, útil para debug)
saveRDS(
  matriz_adyacencia,
  "data_processed/matriz_adyacencia_municipal.rds"
)

cat("Base GNN espaciotemporal generada correctamente.\n")
cat("Archivos creados:\n")
cat("data_processed/base_gnn_espaciotemporal.csv\n")
cat("data_processed/edge_index_municipal.csv\n")
cat("data_processed/nodos_municipales.csv\n")
cat("data_processed/matriz_adyacencia_municipal.rds\n")

# ============================================================
# CONCLUSIONES DEL PREPARADO GNN ESPACIOTEMPORAL
# ============================================================

# 1. Integración de datos correcta
# Se logró integrar la base EVA con los indicadores de irrigación
# a nivel municipal sin pérdidas estructurales relevantes.
# La base resultante es consistente para modelado espacial.
# (Ver validación inicial del panel) :contentReference[oaicite:0]{index=0}

# 2. Problema detectado y corregido: panel desbalanceado
# El panel original presentaba municipios con 1 hasta 13 años.
# Esto invalida modelos espaciotemporales directos.
# Se corrigió filtrando municipios con información completa.

# 3. Construcción de panel balanceado
# Se obtuvo:
# - 998 municipios
# - 13 años por municipio
# - 12974 observaciones totales
# Cumpliendo estructura requerida para GNN temporal.
# (Ver validación panel balanceado) :contentReference[oaicite:1]{index=1}

# 4. Consistencia estructural del dataset final
# Se cumple:
# filas = municipios x años
# 12974 = 998 x 13
# Lo que confirma ausencia de datos faltantes críticos.

# 5. Construcción correcta del grafo espacial
# Se utilizó k-nearest neighbors (k = 5) en lugar de contigüidad.
# Esto evita problemas de desconexión territorial.
# El grafo resultante tiene 1 sola componente conectada.

# 6. Grafo completamente conectado
# n.componentes = 1
# Esto garantiza propagación de información en GNN.
# (Condición necesaria para aprendizaje espacial)

# 7. Dimensión correcta del grafo
# matriz_adyacencia = 998 x 998
# consistente con número de nodos (municipios)

# 8. Estructura de aristas coherente
# Aristas dirigidas: 4990
# Aristas no dirigidas: 6152
# La simetrización elimina direccionalidad artificial
# y mejora estabilidad del modelo GNN.

# 9. Selección de variables basada en evidencia
# Se seleccionaron variables con capacidad explicativa demostrada:
# - irrigación (p75)
# - estructura productiva
# - factores socioeconómicos
# - intensidad agrícola
# Esto evita ruido y sobreajuste.

# 10. Preparación correcta del input para GNN
# Se construyó:
# - matriz de nodos (municipios)
# - edge_index (grafo)
# - features por nodo y año
# - variable objetivo (log_rendimiento)
# Cumpliendo formato requerido por PyTorch Geometric.

# 11. Eliminación de sesgos estructurales
# Se corrigieron:
# - desbalance temporal
# - desconexión espacial
# - direccionalidad artificial en aristas
# Lo que mejora validez del modelo final.

# 12. Resultado final del pipeline
# Se obtuvo un dataset espaciotemporal listo para modelado GNN,
# con estructura consistente, conectividad garantizada
# y variables seleccionadas bajo criterios estadísticos.

# 13. Implicación metodológica
# El proceso confirma que el modelado agrícola territorial
# requiere integrar:
# - estructura espacial (grafo)
# - dinámica temporal (años)
# - variables socioeconómicas y productivas
# más allá de indicadores físicos simples.

# 14. Estado del proyecto
# El pipeline de datos está completo.
# El siguiente paso es el entrenamiento del modelo
# GNN espaciotemporal en Python.

# ============================================================
# FIN DEL PREPARADO DE DATOS PARA GNN
# ============================================================