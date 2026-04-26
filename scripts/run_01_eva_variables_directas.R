# ============================================================
# SCRIPT: run_01_eva_variables_directas.R
# OBJETIVO:
# Crear una nueva DB limpia de EVA solo con variables directas.
# ============================================================

# 1. Cargar funciones
source("R/00_packages.R")
source("R/01_io.R")
source("R/02_clean_names_eva.R")
source("R/03_clean_values_eva.R")
source("R/05_select_variables_eva.R")
source("R/06_export.R")

# 2. Leer datos EVA
eva_raw <- leer_eva(
  "data_raw/Evaluaciones_Agropecuarias_Municipales_EVA_20260413.csv"
)

# 3. Construir DB inicial
eva_variables_directas <- eva_raw %>%
  renombrar_variables_eva() %>%
  limpiar_valores_eva() %>%
  seleccionar_variables_eva()

# 4. Funciones auxiliares
normalizar_texto_eva <- function(x) {
  x %>%
    stringr::str_replace_all("\u0091", "N") %>%
    stringr::str_replace_all("\u00D1", "N") %>%
    stringr::str_replace_all("\u00F1", "N") %>%
    stringi::stri_trans_general("Latin-ASCII") %>%
    stringr::str_squish() %>%
    stringr::str_to_upper()
}

corregir_nombres_cultivo <- function(x) {
  x %>%
    stringr::str_replace_all("CAANA", "CANA") %>%
    stringr::str_replace_all("ANAME", "NAME") %>%
    stringr::str_replace_all("PIANA", "PINA") %>%
    stringr::str_replace_all("MARAANON", "MARANON") %>%
    stringr::str_replace_all("CHAMPIANON", "CHAMPINON")
}

# 5. Limpieza de códigos y textos
eva_variables_directas <- eva_variables_directas %>%
  mutate(
    codigo_municipio = stringr::str_replace_all(codigo_municipio, "\\.", ""),
    across(
      c(
        nombre_departamento,
        nombre_municipio,
        grupo_cultivo,
        subgrupo_cultivo,
        cultivo,
        ciclo_cultivo
      ),
      normalizar_texto_eva
    ),
    cultivo = corregir_nombres_cultivo(cultivo),
    subgrupo_cultivo = corregir_nombres_cultivo(subgrupo_cultivo)
  )

# 6. Diagnóstico inicial
glimpse(eva_variables_directas)

summary(eva_variables_directas$area_sembrada_ha)
summary(eva_variables_directas$area_cosechada_ha)
summary(eva_variables_directas$produccion_t)

produccion_sin_area <- eva_variables_directas %>%
  filter(produccion_t > 0 & area_cosechada_ha == 0)

area_cosechada_mayor_sembrada <- eva_variables_directas %>%
  filter(area_cosechada_ha > area_sembrada_ha)

outliers_produccion <- eva_variables_directas %>%
  filter(produccion_t > 1000000)

resumen_calidad_inicial <- tibble::tibble(
  problema = c(
    "produccion_sin_area",
    "area_cosechada_mayor_sembrada",
    "outliers_produccion_mayor_1000000"
  ),
  registros = c(
    nrow(produccion_sin_area),
    nrow(area_cosechada_mayor_sembrada),
    nrow(outliers_produccion)
  )
)

print(resumen_calidad_inicial)

outliers_produccion %>%
  count(cultivo, sort = TRUE)

# 7. Tratamiento de inconsistencias
eva_variables_directas <- eva_variables_directas %>%
  mutate(
    area_cosechada_ha = dplyr::case_when(
      produccion_t > 0 & area_cosechada_ha <= 0 ~ NA_real_,
      !is.na(area_cosechada_ha) &
        !is.na(area_sembrada_ha) &
        area_cosechada_ha > area_sembrada_ha ~ area_sembrada_ha,
      TRUE ~ area_cosechada_ha
    )
  )

# 8. Validación de inconsistencias después de corregir
inconsistencia_area_cero_final <- eva_variables_directas %>%
  filter(produccion_t > 0 & area_cosechada_ha == 0) %>%
  nrow()

cat(
  "Registros con producción > 0 y área cosechada = 0 después de corregir:",
  inconsistencia_area_cero_final,
  "\n"
)

inconsistencia_area_mayor_final <- eva_variables_directas %>%
  filter(area_cosechada_ha > area_sembrada_ha) %>%
  nrow()

cat(
  "Registros con área cosechada > área sembrada después de corregir:",
  inconsistencia_area_mayor_final,
  "\n"
)

# 9. Validación de textos agrícolas
cultivos_raros <- eva_variables_directas %>%
  count(cultivo, sort = TRUE) %>%
  filter(stringr::str_detect(cultivo, "[^A-Z ]"))

print(cultivos_raros)

eva_variables_directas %>%
  count(cultivo, sort = TRUE) %>%
  filter(stringr::str_detect(cultivo, "CANA|PINA|NAME|MARANON|CHAMPINON"))

# 10. Validaciones antes de limpieza final
duplicados_exactos <- eva_variables_directas %>%
  filter(duplicated(.))

cat("Duplicados exactos antes de limpiar:", nrow(duplicados_exactos), "\n")

missing_clave <- eva_variables_directas %>%
  summarise(
    missing_codigo_departamento = sum(is.na(codigo_departamento)),
    missing_nombre_departamento = sum(is.na(nombre_departamento)),
    missing_codigo_municipio = sum(is.na(codigo_municipio)),
    missing_nombre_municipio = sum(is.na(nombre_municipio)),
    missing_anio = sum(is.na(anio)),
    missing_grupo_cultivo = sum(is.na(grupo_cultivo)),
    missing_subgrupo_cultivo = sum(is.na(subgrupo_cultivo)),
    missing_cultivo = sum(is.na(cultivo)),
    missing_area_sembrada_ha = sum(is.na(area_sembrada_ha)),
    missing_area_cosechada_ha = sum(is.na(area_cosechada_ha)),
    missing_produccion_t = sum(is.na(produccion_t)),
    missing_ciclo_cultivo = sum(is.na(ciclo_cultivo))
  )

print(missing_clave)

rango_anios <- eva_variables_directas %>%
  summarise(
    anio_min = min(anio, na.rm = TRUE),
    anio_max = max(anio, na.rm = TRUE),
    total_anios = n_distinct(anio)
  )

print(rango_anios)

conteo_por_anio <- eva_variables_directas %>%
  count(anio, sort = FALSE)

print(conteo_por_anio)

anios_esperados <- seq(
  min(eva_variables_directas$anio, na.rm = TRUE),
  max(eva_variables_directas$anio, na.rm = TRUE)
)

anios_presentes <- sort(unique(eva_variables_directas$anio))
anios_faltantes <- setdiff(anios_esperados, anios_presentes)

cat("Años faltantes:", paste(anios_faltantes, collapse = ", "), "\n")

codigos_municipio_raros <- eva_variables_directas %>%
  mutate(longitud_codigo_municipio = stringr::str_length(codigo_municipio)) %>%
  filter(longitud_codigo_municipio != 5) %>%
  count(codigo_municipio, longitud_codigo_municipio, sort = TRUE)

print(codigos_municipio_raros)

# 11. Limpieza final
eva_variables_directas <- eva_variables_directas %>%
  distinct() %>%
  mutate(
    codigo_municipio = stringr::str_pad(
      codigo_municipio,
      width = 5,
      side = "left",
      pad = "0"
    )
  )

# 12. Validaciones finales después de limpiar
duplicados_finales <- eva_variables_directas %>%
  filter(duplicated(.))

cat("Duplicados finales:", nrow(duplicados_finales), "\n")

codigos_municipio_raros_final <- eva_variables_directas %>%
  mutate(longitud_codigo_municipio = stringr::str_length(codigo_municipio)) %>%
  filter(longitud_codigo_municipio != 5)

cat(
  "Códigos municipio con longitud distinta de 5:",
  nrow(codigos_municipio_raros_final),
  "\n"
)

missing_clave_final <- eva_variables_directas %>%
  summarise(
    missing_codigo_departamento = sum(is.na(codigo_departamento)),
    missing_nombre_departamento = sum(is.na(nombre_departamento)),
    missing_codigo_municipio = sum(is.na(codigo_municipio)),
    missing_nombre_municipio = sum(is.na(nombre_municipio)),
    missing_anio = sum(is.na(anio)),
    missing_grupo_cultivo = sum(is.na(grupo_cultivo)),
    missing_subgrupo_cultivo = sum(is.na(subgrupo_cultivo)),
    missing_cultivo = sum(is.na(cultivo)),
    missing_area_sembrada_ha = sum(is.na(area_sembrada_ha)),
    missing_area_cosechada_ha = sum(is.na(area_cosechada_ha)),
    missing_produccion_t = sum(is.na(produccion_t)),
    missing_ciclo_cultivo = sum(is.na(ciclo_cultivo))
  )

print(missing_clave_final)

eva_variables_directas %>%
  summarise(
    na_area_cosechada = sum(is.na(area_cosechada_ha)),
    na_produccion = sum(is.na(produccion_t)),
    na_area_sembrada = sum(is.na(area_sembrada_ha))
  )

# 13. Crear dos versiones
eva_con_na <- eva_variables_directas

eva_sin_na <- eva_variables_directas %>%
  tidyr::drop_na(
    area_sembrada_ha,
    area_cosechada_ha,
    produccion_t
  )

cat("Filas con NA:", nrow(eva_con_na), "\n")
cat("Filas sin NA:", nrow(eva_sin_na), "\n")
cat("Filas eliminadas:", nrow(eva_con_na) - nrow(eva_sin_na), "\n")


# 14. Exportar ambas DB
exportar_csv(
  eva_con_na,
  "data_processed/eva_variables_directas_con_na.csv"
)

exportar_csv(
  eva_sin_na,
  "data_processed/eva_variables_directas_sin_na.csv"
)

# ============================================================
# CONCLUSIONES LIMPIEZA EVA
# ============================================================

# 1. Estructura del dataset
# El dataset contiene más de 200 mil registros a nivel cultivo-municipio-año,
# lo que confirma un alto nivel de granularidad para el análisis agrícola.
# (206,068 registros iniciales) :contentReference[oaicite:0]{index=0}

# 2. Problemas de calidad detectados
# Se identificaron inconsistencias importantes:
# - 736 registros con producción positiva sin área cosechada
# - 65 registros donde área cosechada > área sembrada
# - 46 outliers extremos concentrados en caña azucarera
# Esto evidencia errores estructurales en la fuente original.

# 3. Corrección de inconsistencias
# Se aplicaron reglas de limpieza:
# - Se eliminó la inconsistencia área cosechada > área sembrada
# - Se corrigieron casos de producción sin área asignando NA
# Resultado:
# - 0 inconsistencias de área
# - solo 1 caso residual de producción sin área
# Esto indica una limpieza altamente efectiva.

# 4. Estandarización de variables categóricas
# Se normalizaron nombres de cultivos y textos:
# - eliminación de caracteres especiales
# - corrección de errores tipográficos
# - unificación en mayúsculas
# Resultado:
# - 0 cultivos con caracteres inválidos
# Mejora significativa en consistencia semántica.

# 5. Problemas en códigos municipales
# Se detectaron 147 códigos con longitud incorrecta (4 dígitos).
# Esto es un problema crítico para integración con otras bases.
# Se corrigió mediante padding a 5 dígitos.
# Resultado:
# - 0 códigos inválidos después de la corrección

# 6. Duplicados
# Se identificaron 6922 registros duplicados exactos.
# Se eliminaron completamente.
# Resultado:
# - 0 duplicados finales
# Esto evita sesgos en agregaciones posteriores.

# 7. Valores faltantes
# No se detectaron NA en variables clave estructurales.
# Solo se presentaron:
# - 632 NA en área cosechada
# - 0 NA en producción
# Esto indica que la pérdida de información es mínima.

# 8. Tratamiento de NA
# Se generaron dos versiones del dataset:
# - eva_con_na: mantiene todos los registros
# - eva_sin_na: elimina registros incompletos
# Solo se eliminaron 632 filas (~0.3%)
# Impacto estadístico despreciable.

# 9. Distribución de variables
# Las variables presentan alta asimetría:
# - producción máxima > 4.5 millones
# - áreas con alta dispersión
# Esto justifica el uso posterior de transformaciones logarítmicas.

# 10. Cobertura temporal
# El dataset cubre completamente el periodo 2006–2018 (13 años).
# No existen años faltantes.
# Esto permite construir modelos longitudinales robustos.

# 11. Consistencia temporal
# Aunque todos los años están presentes, el número de registros
# varía significativamente entre años.
# Esto implica necesidad de agregación a nivel municipal.

# 12. Calidad final del dataset
# Después de la limpieza:
# - sin duplicados
# - sin errores de área
# - sin códigos inválidos
# - sin problemas de texto
# El dataset es confiable para análisis estadístico y modelado.

# 13. Implicación metodológica
# La base EVA no es directamente utilizable sin limpieza.
# Requiere:
# - corrección de inconsistencias físicas
# - normalización semántica
# - validación estructural
# Este proceso es crítico para evitar sesgos en resultados.

# 14. Impacto en el pipeline
# La limpieza realizada asegura:
# - integridad en agregaciones por municipio-año
# - coherencia con datos espaciales
# - compatibilidad con modelos GNN

# 15. Conclusión general
# El proceso de limpieza transforma una base con errores estructurales
# en un dataset consistente, confiable y listo para modelado avanzado.

# ============================================================
# FIN CONCLUSIONES EVA
# ============================================================