# C:\Users\john\Desktop\id46-proyecto-gnn-agricola-ia\avanzado-ia\src\python\outputs\analysis

library(arrow)

ruta_analysis <- "C:/Users/john/Desktop/id46-proyecto-gnn-agricola-ia/avanzado-ia/src/python/outputs/analysis"

archivos_parquet <- list.files(
  ruta_analysis,
  pattern = "\\.parquet$",
  full.names = TRUE
)

basename(archivos_parquet)

library(arrow)
library(openxlsx)

ruta_analysis <- "C:/Users/john/Desktop/id46-proyecto-gnn-agricola-ia/avanzado-ia/src/python/outputs/analysis"

archivos_parquet <- list.files(
  ruta_analysis,
  pattern = "\\.parquet$",
  full.names = TRUE
)

for (archivo in archivos_parquet) {
  
  nombre_archivo <- tools::file_path_sans_ext(basename(archivo))
  
  datos <- read_parquet(archivo) # Leer archivo Parquet
  
  ruta_xlsx <- file.path(
    ruta_analysis,
    paste0(nombre_archivo, ".xlsx")
  )
  
  write.xlsx(
    datos,
    ruta_xlsx,
    overwrite = TRUE
  ) # Exportar a Excel
  
  cat("Convertido:", nombre_archivo, ".xlsx\n")
}

# -------------------- .csv -----------------------
library(arrow)
library(readr)

ruta_analysis <- "C:/Users/john/Desktop/id46-proyecto-gnn-agricola-ia/avanzado-ia/src/python/outputs/analysis"

archivos_parquet <- list.files(
  ruta_analysis,
  pattern = "\\.parquet$",
  full.names = TRUE
)

for (archivo in archivos_parquet) {
  
  nombre_archivo <- tools::file_path_sans_ext(basename(archivo))
  
  datos <- read_parquet(archivo) # Leer archivo Parquet
  
  ruta_csv <- file.path(
    ruta_analysis,
    paste0(nombre_archivo, ".csv")
  )
  
  write_csv(
    datos,
    ruta_csv
  ) # Exportar a CSV
  
  cat("Convertido:", nombre_archivo, ".csv\n")
}

# ------------------------------------------------

library(readr)
library(dplyr)

ruta_analysis <- "C:/Users/john/Desktop/id46-proyecto-gnn-agricola-ia/avanzado-ia/src/python/outputs/analysis"

ruta_topologia <- file.path(
  ruta_analysis,
  "municipality_topology_similarity.csv"
)

topologia <- read_csv(
  ruta_topologia,
  show_col_types = FALSE
) # Leer tabla de similitud topológica

dim(topologia) # Dimensiones

names(topologia) # Variables

str(topologia) # Estructura

head(topologia) # Primeras observaciones

municipios_grado <- bind_rows(
  topologia %>%
    select(
      municipio_id = municipality_id_a,
      municipio = municipality_a,
      grado = grado_a
    ),
  topologia %>%
    select(
      municipio_id = municipality_id_b,
      municipio = municipality_b,
      grado = grado_b
    )
) %>%
  group_by(municipio_id, municipio) %>%
  summarise(
    grado_promedio = mean(grado, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(grado_promedio))

# ------------------------------------------------------

library(arrow)
library(dplyr)

ruta_analysis <- "C:/Users/john/Desktop/id46-proyecto-gnn-agricola-ia/avanzado-ia/src/python/outputs/analysis"

profile <- read_parquet(
  file.path(ruta_analysis, "municipality_profile.parquet")
)

comparison <- read_parquet(
  file.path(ruta_analysis, "municipality_comparison.parquet")
)

similarity <- read_parquet(
  file.path(ruta_analysis, "municipality_similarity.parquet")
)

spatial <- read_parquet(
  file.path(ruta_analysis, "municipality_spatial_analysis.parquet")
)

topology <- read_parquet(
  file.path(ruta_analysis, "municipality_topology_similarity.parquet")
)

standardization <- read_parquet(
  file.path(ruta_analysis, "standardization_parameters.parquet")
)

names(profile)
names(comparison)
names(similarity)
names(spatial)
names(topology)
names(standardization)

dim(profile)
dim(comparison)
dim(similarity)
dim(spatial)
dim(topology)
dim(standardization)

# 1. Auditoría del panel municipal

# | Pregunta                                             | Código R                                               |
# | ---------------------------------------------------- | ------------------------------------------------------ |
# | ¿Tenemos 1.121 municipios por año?                   | `profile %>% count(anio) %>% arrange(anio)`            |
# | ¿Tenemos exactamente 13 años?                        | `n_distinct(profile$anio)`                             |
# | ¿Tenemos exactamente 1.121 municipios?               | `n_distinct(profile$cod_municipio)`                    |
# | ¿Tenemos exactamente 14.573 registros municipio año? | `nrow(profile)`                                        |
# | ¿Existen duplicados municipio año?                   | `sum(duplicated(profile[c("cod_municipio", "anio")]))` |
# | ¿Qué años están presentes?                           | `sort(unique(profile$anio))`                           |
# | ¿Qué municipios tienen registros incompletos?        | `profile %>% count(cod_municipio) %>% filter(n != 13)` |
  

profile %>%
  count(anio) %>%
  arrange(anio) # Verificar cantidad de municipios por año

n_distinct(profile$anio) # Verificar cantidad de años

n_distinct(profile$cod_municipio) # Verificar cantidad de municipios

nrow(profile) # Verificar cantidad total de registros

sum(duplicated(profile[c("cod_municipio", "anio")])) # Detectar duplicados municipio-año

profile %>%
  count(cod_municipio) %>%
  filter(n != 13) # Detectar municipios con cobertura temporal incompleta

# 2. ¿Cómo se distribuye log_rendimiento?

profile %>%
  summarise(
    media = mean(log_rendimiento, na.rm = TRUE),
    mediana = median(log_rendimiento, na.rm = TRUE),
    desviacion_estandar = sd(log_rendimiento, na.rm = TRUE),
    minimo = min(log_rendimiento, na.rm = TRUE),
    maximo = max(log_rendimiento, na.rm = TRUE),
    q25 = quantile(log_rendimiento, 0.25, na.rm = TRUE),
    q75 = quantile(log_rendimiento, 0.75, na.rm = TRUE)
  )

# 3. ¿Qué municipios presentan mayor y menor log_rendimiento?

perfil_maximo <- profile %>%
  arrange(desc(log_rendimiento)) %>%
  select(
    anio,
    cod_municipio,
    municipio,
    log_rendimiento
  ) %>%
  slice_head(n = 20) # Identificar valores más altos

perfil_minimo <- profile %>%
  arrange(log_rendimiento) %>%
  select(
    anio,
    cod_municipio,
    municipio,
    log_rendimiento
  ) %>%
  slice_head(n = 20) # Identificar valores más bajos

perfil_maximo

perfil_minimo

# 4. ¿Cómo cambia log_rendimiento entre 2006 y 2018?
rendimiento_temporal <- profile %>%
  group_by(anio) %>%
  summarise(
    media = mean(log_rendimiento, na.rm = TRUE),
    mediana = median(log_rendimiento, na.rm = TRUE),
    sd = sd(log_rendimiento, na.rm = TRUE),
    minimo = min(log_rendimiento, na.rm = TRUE),
    maximo = max(log_rendimiento, na.rm = TRUE),
    .groups = "drop"
  )

rendimiento_temporal

# 5. ¿Qué variables presentan mayor variabilidad?
variables_cientificas <- standardization$variable

variabilidad <- profile %>%
  select(all_of(variables_cientificas)) %>%
  summarise(
    across(
      everything(),
      ~ sd(.x, na.rm = TRUE)
    )
  ) %>%
  tidyr::pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "desviacion_estandar"
  ) %>%
  arrange(desc(desviacion_estandar))

variabilidad

# 6. ¿Qué variables están más asociadas con log_rendimiento?
correlaciones <- profile %>%
  select(all_of(variables_cientificas), log_rendimiento) %>%
  cor(use = "complete.obs") %>%
  as.data.frame()

correlacion_objetivo <- correlaciones %>%
  select(log_rendimiento) %>%
  tibble::rownames_to_column("variable") %>%
  filter(variable != "log_rendimiento") %>%
  mutate(
    correlacion = log_rendimiento,
    correlacion_absoluta = abs(correlacion)
  ) %>%
  arrange(desc(correlacion_absoluta))

correlacion_objetivo

# 7. ¿Qué pares de municipios presentan mayores diferencias?
comparison %>%
  arrange(desc(difference_standardized_mean)) %>%
  slice_head(n = 20)

# 8. ¿Las diferencias de atributos están relacionadas con diferencias de rendimiento?
cor(
  comparison$difference_standardized_mean,
  comparison$difference_target,
  use = "complete.obs"
)

modelo_diferencias <- lm(
  difference_target ~ difference_standardized_mean,
  data = comparison
)

summary(modelo_diferencias)

# 9. ¿Qué municipios presentan mayor similitud global?
# ¿Cuáles son los pares de municipios más similares considerando globalmente sus características?
similarity %>%
  arrange(desc(similaridad_global)) %>%
  select(
    anio,
    municipality_id_a,
    municipality_id_b,
    similaridad_global,
    distancia_euclidiana
  ) %>%
  slice_head(n = 20)

# 10. ¿Qué municipios presentan menor similitud global?
similarity %>%
  arrange(similaridad_global) %>%
  select(
    anio,
    municipality_id_a,
    municipality_id_b,
    similaridad_global,
    distancia_euclidiana
  ) %>%
  slice_head(n = 20)

# 11. ¿Qué dimensión explica mejor la similitud entre municipios?
componentes_similitud <- c(
  "similaridad_climate",
  "similaridad_agriculture",
  "similaridad_irrigation",
  "similaridad_environmental_indices",
  "similaridad_land_cover",
  "similaridad_land_tenure"
)

correlaciones_similitud <- sapply(
  componentes_similitud,
  function(variable) {
    cor(
      similarity[[variable]],
      similarity$similaridad_global,
      use = "complete.obs"
    )
  }
)

sort(correlaciones_similitud, decreasing = TRUE)

# 12. ¿La similitud agrícola está relacionada con la similitud climática?
cor(
  similarity$similaridad_agriculture,
  similarity$similaridad_climate,
  use = "complete.obs"
)

# 13. ¿La similitud global está asociada con similitud en log_rendimiento?
similarity <- similarity %>%
  mutate(
    similaridad_target = 1 / (1 + difference_target)
  )

cor(
  similarity$similaridad_global,
  similarity$similaridad_target,
  use = "complete.obs"
)

# 14. ¿La similitud de atributos está relacionada con la similitud topológica?
cor(
  similarity$similaridad_global,
  similarity$similaridad_topologica,
  use = "complete.obs"
)

# Podemos ir más allá:
modelo_atributos_topologia <- lm(
  similaridad_topologica ~ similaridad_global,
  data = similarity
)

summary(modelo_atributos_topologia)

# 15. ¿Qué relación existe entre similitud agrícola y similitud topológica?
cor(
  similarity$similaridad_agriculture,
  similarity$similaridad_topologica,
  use = "complete.obs"
)

# 16. ¿La distancia geográfica está relacionada con la similitud global?
spatial_similarity <- spatial %>%
  select(
    anio,
    node_idx_a,
    node_idx_b,
    distancia_geografica_km,
    mismo_departamento
  ) %>%
  left_join(
    similarity %>%
      select(
        anio,
        node_idx_a,
        node_idx_b,
        similaridad_global
      ),
    by = c("anio", "node_idx_a", "node_idx_b")
  )

cor(
  spatial_similarity$distancia_geografica_km,
  spatial_similarity$similaridad_global,
  use = "complete.obs"
)

# 17. ¿La distancia geográfica está relacionada con la similitud topológica?
spatial_topology <- spatial %>%
  select(
    anio,
    node_idx_a,
    node_idx_b,
    distancia_geografica_km,
    mismo_departamento
  ) %>%
  left_join(
    topology %>%
      select(
        anio,
        node_idx_a,
        node_idx_b,
        similaridad_topologica
      ),
    by = c("anio", "node_idx_a", "node_idx_b")
  ); spatial_topology

cor(
  spatial_topology$distancia_geografica_km,
  spatial_topology$similaridad_topologica,
  use = "complete.obs"
)

# 18. ¿La topología está explicada solamente por la distancia geográfica?
modelo_topologia_espacio <- lm(
  similaridad_topologica ~ distancia_geografica_km,
  data = spatial_topology
)

summary(modelo_topologia_espacio)

# 19. ¿La similitud topológica está relacionada con la similitud de log_rendimiento?
topology_target <- topology %>%
  select(
    anio,
    node_idx_a,
    node_idx_b,
    similaridad_topologica
  ) %>%
  left_join(
    comparison %>%
      select(
        anio,
        node_idx_a,
        node_idx_b,
        difference_target
      ),
    by = c("anio", "node_idx_a", "node_idx_b")
  ); topology_target

cor(
  topology_target$similaridad_topologica,
  -topology_target$difference_target,
  use = "complete.obs"
)

# 20. ¿La similitud topológica cambia entre 2006 y 2018?
topologia_temporal <- topology %>%
group_by(anio) %>%
summarise(
  media = mean(similaridad_topologica, na.rm = TRUE),
  mediana = median(similaridad_topologica, na.rm = TRUE),
  sd = sd(similaridad_topologica, na.rm = TRUE),
  minimo = min(similaridad_topologica, na.rm = TRUE),
  maximo = max(similaridad_topologica, na.rm = TRUE),
  .groups = "drop"
); topologia_temporal