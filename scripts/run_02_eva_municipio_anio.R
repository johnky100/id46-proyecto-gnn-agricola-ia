# ============================================================
# SCRIPT: run_02_eva_municipio_anio.R
# OBJETIVO:
# Construir base EVA agregada por municipio-año.
# ============================================================

source("R/00_packages.R")
source("R/06_export.R")

# ============================================================
# 1. Leer base EVA limpia
# ============================================================

eva <- readr::read_csv(
  "data_processed/eva_variables_directas_con_na.csv",
  col_types = readr::cols(
    codigo_departamento = readr::col_integer(),
    codigo_municipio = readr::col_character(),
    anio = readr::col_integer(),
    .default = readr::col_guess()
  ),
  show_col_types = FALSE
)

# ============================================================
# 2. Definir llave municipio-año
# ============================================================

by_mpio_anio <- c(
  "codigo_departamento",
  "codigo_municipio",
  "anio"
)

# ============================================================
# 3. Producción por cultivo dentro de municipio-año
# ============================================================

eva_cultivo_municipio_anio <- eva %>%
  group_by(
    codigo_departamento,
    codigo_municipio,
    anio,
    grupo_cultivo,
    subgrupo_cultivo,
    cultivo,
    ciclo_cultivo
  ) %>%
  summarise(
    nombre_departamento = first(na.omit(nombre_departamento)),
    nombre_municipio = first(na.omit(nombre_municipio)),
    area_sembrada_cultivo_ha = sum(area_sembrada_ha, na.rm = TRUE),
    area_cosechada_cultivo_ha = sum(area_cosechada_ha, na.rm = TRUE),
    produccion_cultivo_t = sum(produccion_t, na.rm = TRUE),
    .groups = "drop"
  )

# ============================================================
# 4. Totales por municipio-año
# ============================================================

eva_totales_municipio_anio <- eva_cultivo_municipio_anio %>%
  group_by(
    codigo_departamento,
    codigo_municipio,
    anio
  ) %>%
  summarise(
    nombre_departamento = first(na.omit(nombre_departamento)),
    nombre_municipio = first(na.omit(nombre_municipio)),
    area_sembrada_total_ha = sum(area_sembrada_cultivo_ha, na.rm = TRUE),
    area_cosechada_total_ha = sum(area_cosechada_cultivo_ha, na.rm = TRUE),
    produccion_total_t = sum(produccion_cultivo_t, na.rm = TRUE),
    num_cultivos = n_distinct(cultivo),
    num_subgrupos_cultivo = n_distinct(subgrupo_cultivo),
    num_grupos_cultivo = n_distinct(grupo_cultivo),
    .groups = "drop"
  ) %>%
  mutate(
    rendimiento_promedio_t_ha = dplyr::case_when(
      area_cosechada_total_ha > 0 ~ produccion_total_t / area_cosechada_total_ha,
      TRUE ~ NA_real_
    )
  )

# ============================================================
# 5. Validar llave municipio-año
# ============================================================

duplicados_totales <- eva_totales_municipio_anio %>%
  count(codigo_departamento, codigo_municipio, anio) %>%
  filter(n > 1)

cat("Duplicados en totales municipio-año:", nrow(duplicados_totales), "\n")

# ============================================================
# 6. Proporción de producción por cultivo
# ============================================================

eva_participacion_cultivo <- eva_cultivo_municipio_anio %>%
  left_join(
    eva_totales_municipio_anio %>%
      select(
        codigo_departamento,
        codigo_municipio,
        anio,
        produccion_total_t
      ),
    by = by_mpio_anio
  ) %>%
  mutate(
    p_i = dplyr::case_when(
      produccion_total_t > 0 ~ produccion_cultivo_t / produccion_total_t,
      TRUE ~ NA_real_
    )
  )

# ============================================================
# 7. Índices de estructura productiva
# ============================================================

eva_indices_productivos <- eva_participacion_cultivo %>%
  group_by(
    codigo_departamento,
    codigo_municipio,
    anio
  ) %>%
  summarise(
    indice_diversificacion_cultivos = if (all(is.na(p_i))) {
      NA_real_
    } else {
      -sum(p_i * log(p_i), na.rm = TRUE)
    },
    participacion_cultivo_principal = if (all(is.na(p_i))) {
      NA_real_
    } else {
      max(p_i, na.rm = TRUE)
    },
    indice_concentracion_cultivos_hhi = if (all(is.na(p_i))) {
      NA_real_
    } else {
      sum(p_i^2, na.rm = TRUE)
    },
    .groups = "drop"
  )

# ============================================================
# 8. Proporción transitorios y permanentes
# ============================================================

eva_ciclo_productivo <- eva_participacion_cultivo %>%
  group_by(
    codigo_departamento,
    codigo_municipio,
    anio
  ) %>%
  summarise(
    produccion_transitorios_t = sum(
      ifelse(ciclo_cultivo == "TRANSITORIO", produccion_cultivo_t, 0),
      na.rm = TRUE
    ),
    produccion_permanentes_t = sum(
      ifelse(ciclo_cultivo == "PERMANENTE", produccion_cultivo_t, 0),
      na.rm = TRUE
    ),
    produccion_total_t = max(produccion_total_t, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    proporcion_transitorios = dplyr::case_when(
      produccion_total_t > 0 ~ produccion_transitorios_t / produccion_total_t,
      TRUE ~ NA_real_
    ),
    proporcion_permanentes = dplyr::case_when(
      produccion_total_t > 0 ~ produccion_permanentes_t / produccion_total_t,
      TRUE ~ NA_real_
    ),
    proporcion_ciclo_total = proporcion_transitorios + proporcion_permanentes
  ) %>%
  select(
    codigo_departamento,
    codigo_municipio,
    anio,
    produccion_transitorios_t,
    produccion_permanentes_t,
    proporcion_transitorios,
    proporcion_permanentes,
    proporcion_ciclo_total
  )

# ============================================================
# 9. Unir todo en base municipio-año
# ============================================================

eva_municipio_anio <- eva_totales_municipio_anio %>%
  left_join(
    eva_indices_productivos,
    by = by_mpio_anio
  ) %>%
  left_join(
    eva_ciclo_productivo,
    by = by_mpio_anio
  ) %>%
  arrange(codigo_departamento, codigo_municipio, anio)

# ============================================================
# 9.1 Flags de calidad de datos
# ============================================================

eva_municipio_anio <- eva_municipio_anio %>%
  mutate(
    # Outliers de rendimiento
    flag_rendimiento_extremo = rendimiento_promedio_t_ha > 100,
    
    # Municipios sin producción
    flag_sin_produccion = produccion_total_t == 0,
    
    # Posible inconsistencia en proporciones
    flag_proporcion_incompleta = proporcion_ciclo_total < 0.95
  )

# ============================================================
# 10. Validaciones finales
# ============================================================

glimpse(eva_municipio_anio)

cat("Filas municipio-año:", nrow(eva_municipio_anio), "\n")

duplicados_municipio_anio <- eva_municipio_anio %>%
  count(codigo_departamento, codigo_municipio, anio) %>%
  filter(n > 1)

cat("Duplicados municipio-año:", nrow(duplicados_municipio_anio), "\n")

missing_municipio_anio <- eva_municipio_anio %>%
  summarise(
    missing_codigo_departamento = sum(is.na(codigo_departamento)),
    missing_codigo_municipio = sum(is.na(codigo_municipio)),
    missing_nombre_departamento = sum(is.na(nombre_departamento)),
    missing_nombre_municipio = sum(is.na(nombre_municipio)),
    missing_anio = sum(is.na(anio)),
    missing_area_sembrada_total_ha = sum(is.na(area_sembrada_total_ha)),
    missing_area_cosechada_total_ha = sum(is.na(area_cosechada_total_ha)),
    missing_produccion_total_t = sum(is.na(produccion_total_t)),
    missing_rendimiento_promedio_t_ha = sum(is.na(rendimiento_promedio_t_ha)),
    missing_indice_diversificacion = sum(is.na(indice_diversificacion_cultivos)),
    missing_hhi = sum(is.na(indice_concentracion_cultivos_hhi)),
    missing_participacion_principal = sum(is.na(participacion_cultivo_principal)),
    missing_proporcion_transitorios = sum(is.na(proporcion_transitorios)),
    missing_proporcion_permanentes = sum(is.na(proporcion_permanentes))
  )

print(missing_municipio_anio)

summary(eva_municipio_anio$area_sembrada_total_ha)
summary(eva_municipio_anio$area_cosechada_total_ha)
summary(eva_municipio_anio$produccion_total_t)
summary(eva_municipio_anio$rendimiento_promedio_t_ha)

eva_municipio_anio %>%
  summarise(
    min_shannon = min(indice_diversificacion_cultivos, na.rm = TRUE),
    max_shannon = max(indice_diversificacion_cultivos, na.rm = TRUE),
    min_hhi = min(indice_concentracion_cultivos_hhi, na.rm = TRUE),
    max_hhi = max(indice_concentracion_cultivos_hhi, na.rm = TRUE),
    min_participacion = min(participacion_cultivo_principal, na.rm = TRUE),
    max_participacion = max(participacion_cultivo_principal, na.rm = TRUE),
    min_transitorios = min(proporcion_transitorios, na.rm = TRUE),
    max_transitorios = max(proporcion_transitorios, na.rm = TRUE),
    min_permanentes = min(proporcion_permanentes, na.rm = TRUE),
    max_permanentes = max(proporcion_permanentes, na.rm = TRUE)
  ) %>%
  print()

eva_municipio_anio %>%
  summarise(
    min_proporcion_ciclo_total = min(proporcion_ciclo_total, na.rm = TRUE),
    max_proporcion_ciclo_total = max(proporcion_ciclo_total, na.rm = TRUE)
  ) %>%
  print()

# ============================================================
# 10.1 Diagnóstico de flags
# ============================================================

eva_municipio_anio %>%
  summarise(
    total_registros = n(),
    rendimiento_extremo = sum(flag_rendimiento_extremo, na.rm = TRUE),
    municipios_sin_produccion = sum(flag_sin_produccion, na.rm = TRUE),
    proporciones_incompletas = sum(flag_proporcion_incompleta, na.rm = TRUE)
  ) %>%
  print()

# ============================================================
# 11. Exportar
# ============================================================

# CSV
exportar_csv(
  eva_municipio_anio,
  "data_processed/eva_municipio_anio.csv"
)

# Excel
writexl::write_xlsx(
  eva_municipio_anio,
  "data_processed/eva_municipio_anio.xlsx"
)

# ============================================================
# CONCLUSIONES EVA MUNICIPIO-AÑO (VERSIÓN ANALÍTICA)
# ============================================================

# 1. Consistencia estructural del dataset
# La agregación a nivel municipio-año genera una base con 14,201 observaciones
# únicas, sin duplicados en la llave (departamento, municipio, año).
# Esto garantiza integridad para análisis longitudinal y modelado.
# :contentReference[oaicite:0]{index=0}

# 2. Cobertura temporal completa pero desbalanceada
# El dataset cubre el periodo 2006–2018 sin años faltantes.
# Sin embargo, la cantidad de registros por año es heterogénea,
# lo que implica que no todos los municipios tienen información completa.
# Esto justifica el balanceo posterior para modelos espaciotemporales.

# 3. Validez de las variables productivas agregadas
# Las variables agregadas (área, producción, rendimiento) son coherentes:
# - no hay valores faltantes en producción total
# - el rendimiento solo es NA cuando el área es cero
# Esto indica consistencia lógica, no errores de datos.

# 4. Alta asimetría en producción y área
# Se observa fuerte sesgo:
# - producción máxima > 4.5 millones de toneladas
# - áreas con alta dispersión
# Esto confirma la necesidad de transformaciones logarítmicas
# para evitar dominancia de outliers en modelos predictivos.

# 5. Existencia de outliers estructurales
# Se identifican 88 casos de rendimiento extremo (>100 t/ha).
# Estos no deben eliminarse, ya que:
# - pueden corresponder a cultivos intensivos
# - contienen señal informativa relevante
# Su tratamiento debe hacerse en el modelado, no en limpieza.

# 6. Diversidad estructural del sistema agrícola
# Los índices muestran:
# - Shannon: 0 a ~3 → desde monocultivo hasta alta diversificación
# - HHI: valores cercanos a 1 → alta concentración en algunos municipios
# Esto evidencia heterogeneidad productiva territorial,
# clave para explicar diferencias en rendimiento.

# 7. Coexistencia de estrategias productivas
# Se identifican dos patrones:
# - municipios diversificados
# - municipios especializados
# Ambos pueden ser eficientes, lo que implica que
# la relación diversidad-rendimiento no es lineal.

# 8. Limitación en clasificación de cultivos
# La proporción total (transitorios + permanentes) no siempre suma 1.
# Esto indica:
# - presencia de cultivos no clasificados
# - limitación estructural del dataset original
# Debe considerarse en interpretación, no tratarse como error.

# 9. Calidad de la llave municipio-año
# La ausencia de duplicados confirma que:
# - la agregación fue correcta
# - no hay problemas de sobreconteo
# Esto es crítico para evitar sesgos en modelos.

# 10. Flags como variables informativas (no solo control)
# Se construyeron indicadores de:
# - rendimiento extremo
# - ausencia de producción
# - inconsistencias en proporciones
# Estos no son errores, sino variables explicativas potenciales.

# 11. Transformación del dataset
# El proceso convierte una base a nivel cultivo
# en una base estructural a nivel territorial.
# Esto es un cambio clave:
# - de microdatos → sistema productivo municipal

# 12. Preparación para integración con irrigación
# La estructura municipio-año permite:
# - merge directo con variables espaciales
# - alineación con grafo municipal
# Esto habilita el modelado GNN posterior.

# 13. Limitación clave identificada
# El dataset no incluye:
# - variables climáticas
# - variables económicas
# - variables institucionales
# Por lo tanto, el modelo capturará solo parte del fenómeno.

# 14. Valor analítico real del dataset
# El valor no está en las variables individuales,
# sino en la combinación:
# - producción
# - estructura productiva
# - composición de cultivos
# Esto permite modelar sistemas agrícolas complejos.

# 15. Conclusión principal
# El rendimiento agrícola no puede explicarse únicamente por volumen
# de producción o área, sino por la estructura productiva del territorio.
# Este dataset permite capturar dicha estructura.

# ============================================================
# FIN CONCLUSIONES EVA MUNICIPIO-AÑO
# ============================================================