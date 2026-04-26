# ============================================================
# SCRIPT: run_06_merge_eva_validacion.R
# OBJETIVO:
# Unir EVA municipal-año con indicadores de irrigación municipal
# y validar estadísticamente el índice de irrigación.
# ============================================================

source("R/00_packages.R")
source("R/06_export.R")

library(dplyr)
library(readr)
library(stringr)
library(janitor)
library(broom)
library(writexl)

# ============================================================
# 1. Cargar bases
# ============================================================

eva_municipio_anio <- readr::read_csv(
  "data_processed/eva_municipio_anio.csv",
  show_col_types = FALSE
) %>%
  janitor::clean_names()

irrigacion_municipal <- readr::read_csv(
  "data_processed/irrigacion_municipal.csv",
  show_col_types = FALSE
) %>%
  janitor::clean_names()

# ============================================================
# 2. Homologar códigos
# ============================================================

eva_municipio_anio <- eva_municipio_anio %>%
  mutate(
    codigo_municipio = str_pad(
      as.character(codigo_municipio),
      width = 5,
      side = "left",
      pad = "0"
    )
  )

irrigacion_municipal <- irrigacion_municipal %>%
  mutate(
    codigo_municipio = str_pad(
      as.character(codigo_municipio),
      width = 5,
      side = "left",
      pad = "0"
    )
  )

# ============================================================
# 3. Revisar variables disponibles
# ============================================================

names(eva_municipio_anio)
names(irrigacion_municipal)

# ============================================================
# 4. Merge EVA + irrigación
# ============================================================

base_modelo <- eva_municipio_anio %>%
  left_join(
    irrigacion_municipal,
    by = "codigo_municipio",
    suffix = c("_eva", "_irrigacion")
  )

# ============================================================
# 5. Validar calidad del merge
# ============================================================

validacion_merge <- base_modelo %>%
  summarise(
    filas = n(),
    municipios = n_distinct(codigo_municipio),
    anios = n_distinct(anio),
    filas_sin_irrigacion = sum(is.na(indice_irrigacion_media_pond)),
    porcentaje_sin_irrigacion = mean(is.na(indice_irrigacion_media_pond)) * 100
  )

print(validacion_merge)

# ============================================================
# 6. Preparar base de validación corregida
# ============================================================

base_validacion <- base_modelo %>%
  filter(
    !is.na(rendimiento_promedio_t_ha),
    !is.na(indice_irrigacion_media_pond),
    rendimiento_promedio_t_ha > 0,
    !flag_rendimiento_extremo
  ) %>%
  mutate(
    log_rendimiento = log1p(rendimiento_promedio_t_ha),
    intensidad_agricola = area_cosechada_total_ha / (area_sembrada_total_ha + 1e-6),
    log_produccion_total = log1p(produccion_total_t),
    log_area_cosechada = log1p(area_cosechada_total_ha),
    rendimiento_irrigable_proxy = rendimiento_promedio_t_ha * proporcion_permanentes,
    log_rendimiento_irrigable_proxy = log1p(rendimiento_irrigable_proxy)
  )

cat("Filas para validación:", nrow(base_validacion), "\n")
cat("Municipios para validación:", n_distinct(base_validacion$codigo_municipio), "\n")

# ============================================================
# 7. Estandarizar predictores
# ============================================================

vars_z <- c(
  "indice_irrigacion_media_pond",
  "indice_irrigacion_p75",
  "prop_area_alto_potencial",
  "indice_irrigacion_sd",
  "log_area_potencial_irrigacion",
  "disponibilidad_media_pond",
  "ecosistemico_media_pond",
  "regulacion_media_pond",
  "socioeconomico_media_pond",
  "intensidad_agricola",
  "indice_diversificacion_cultivos",
  "indice_concentracion_cultivos_hhi",
  "participacion_cultivo_principal",
  "proporcion_permanentes",
  "proporcion_transitorios"
)

base_z <- base_validacion %>%
  mutate(
    across(
      all_of(vars_z),
      ~ as.numeric(scale(.x)),
      .names = "z_{.col}"
    )
  )

# ============================================================
# 8. Correlaciones corregidas
# ============================================================

variables_irrigacion <- c(
  "potencial_oficial_media_pond",
  "score_fisico_media_pond",
  "indice_irrigacion_media_pond",
  "indice_irrigacion_media_simple",
  "indice_irrigacion_p75",
  "indice_irrigacion_max",
  "indice_irrigacion_sd",
  "prop_area_alto_potencial",
  "area_potencial_irrigacion_ha",
  "log_area_potencial_irrigacion"
)

correlaciones <- purrr::map_dfr(
  variables_irrigacion,
  function(var) {
    tibble(
      variable = var,
      cor_rendimiento_original = cor(
        base_validacion$rendimiento_promedio_t_ha,
        base_validacion[[var]],
        use = "complete.obs"
      ),
      cor_log_rendimiento = cor(
        base_validacion$log_rendimiento,
        base_validacion[[var]],
        use = "complete.obs"
      ),
      cor_log_rendimiento_irrigable_proxy = cor(
        base_validacion$log_rendimiento_irrigable_proxy,
        base_validacion[[var]],
        use = "complete.obs"
      )
    )
  }
) %>%
  arrange(desc(abs(cor_log_rendimiento)))

print(correlaciones)

# ============================================================
# 9. Modelos simples comparativos con target log
# ============================================================

m_oficial_log <- lm(
  log_rendimiento ~ potencial_oficial_media_pond,
  data = base_validacion
)

m_fisico_log <- lm(
  log_rendimiento ~ score_fisico_media_pond,
  data = base_validacion
)

m_indice_pond_log <- lm(
  log_rendimiento ~ indice_irrigacion_media_pond,
  data = base_validacion
)

m_indice_p75_log <- lm(
  log_rendimiento ~ indice_irrigacion_p75,
  data = base_validacion
)

m_prop_alto_log <- lm(
  log_rendimiento ~ prop_area_alto_potencial,
  data = base_validacion
)

m_sd_log <- lm(
  log_rendimiento ~ indice_irrigacion_sd,
  data = base_validacion
)

comparacion_modelos_simples <- tibble(
  modelo = c(
    "oficial_log",
    "fisico_log",
    "indice_media_pond_log",
    "indice_p75_log",
    "prop_area_alto_potencial_log",
    "heterogeneidad_sd_log"
  ),
  r2_ajustado = c(
    summary(m_oficial_log)$adj.r.squared,
    summary(m_fisico_log)$adj.r.squared,
    summary(m_indice_pond_log)$adj.r.squared,
    summary(m_indice_p75_log)$adj.r.squared,
    summary(m_prop_alto_log)$adj.r.squared,
    summary(m_sd_log)$adj.r.squared
  ),
  aic = c(
    AIC(m_oficial_log),
    AIC(m_fisico_log),
    AIC(m_indice_pond_log),
    AIC(m_indice_p75_log),
    AIC(m_prop_alto_log),
    AIC(m_sd_log)
  )
) %>%
  arrange(aic)

print(comparacion_modelos_simples)

# ============================================================
# 10. Modelo corregido con controles
# ============================================================

m_corregido <- lm(
  log_rendimiento ~
    z_indice_irrigacion_p75 +
    z_prop_area_alto_potencial +
    z_indice_irrigacion_sd +
    z_log_area_potencial_irrigacion +
    z_socioeconomico_media_pond +
    z_intensidad_agricola +
    z_indice_diversificacion_cultivos +
    z_indice_concentracion_cultivos_hhi +
    factor(anio),
  data = base_z
)

summary(m_corregido)

coeficientes_corregido <- broom::tidy(m_corregido)

# ============================================================
# 11. Modelo con interacciones
# ============================================================

m_interacciones <- lm(
  log_rendimiento ~
    z_indice_irrigacion_p75 * z_socioeconomico_media_pond +
    z_prop_area_alto_potencial * z_socioeconomico_media_pond +
    z_indice_irrigacion_sd +
    z_log_area_potencial_irrigacion +
    z_intensidad_agricola +
    factor(anio),
  data = base_z
)

summary(m_interacciones)

coeficientes_interacciones <- broom::tidy(m_interacciones)

# ============================================================
# 12. Modelo con target irrigable proxy
# ============================================================

m_irrigable_proxy <- lm(
  log_rendimiento_irrigable_proxy ~
    z_indice_irrigacion_p75 +
    z_prop_area_alto_potencial +
    z_indice_irrigacion_sd +
    z_log_area_potencial_irrigacion +
    z_socioeconomico_media_pond +
    z_intensidad_agricola +
    factor(anio),
  data = base_z
)

summary(m_irrigable_proxy)

coeficientes_irrigable_proxy <- broom::tidy(m_irrigable_proxy)

# ============================================================
# 13. Comparación general de modelos corregidos
# ============================================================

comparacion_modelos_corregidos <- tibble(
  modelo = c(
    "corregido_log_controles",
    "interacciones_log",
    "irrigable_proxy_log"
  ),
  r2_ajustado = c(
    summary(m_corregido)$adj.r.squared,
    summary(m_interacciones)$adj.r.squared,
    summary(m_irrigable_proxy)$adj.r.squared
  ),
  aic = c(
    AIC(m_corregido),
    AIC(m_interacciones),
    AIC(m_irrigable_proxy)
  )
) %>%
  arrange(aic)

print(comparacion_modelos_corregidos)

# ============================================================
# 14. Exportar resultados
# ============================================================

readr::write_csv(
  base_modelo,
  "data_processed/base_modelo_eva_irrigacion.csv"
)

readr::write_csv(
  base_z,
  "data_processed/base_modelo_eva_irrigacion_validacion.csv"
)

readr::write_csv(
  correlaciones,
  "data_processed/validacion_correlaciones_irrigacion_eva.csv"
)

readr::write_csv(
  comparacion_modelos_simples,
  "data_processed/validacion_modelos_simples_log.csv"
)

readr::write_csv(
  comparacion_modelos_corregidos,
  "data_processed/validacion_modelos_corregidos.csv"
)

readr::write_csv(
  coeficientes_corregido,
  "data_processed/validacion_coeficientes_modelo_corregido.csv"
)

readr::write_csv(
  coeficientes_interacciones,
  "data_processed/validacion_coeficientes_interacciones.csv"
)

readr::write_csv(
  coeficientes_irrigable_proxy,
  "data_processed/validacion_coeficientes_irrigable_proxy.csv"
)

writexl::write_xlsx(
  list(
    validacion_merge = validacion_merge,
    correlaciones = correlaciones,
    modelos_simples = comparacion_modelos_simples,
    modelos_corregidos = comparacion_modelos_corregidos,
    coeficientes_corregido = coeficientes_corregido,
    coeficientes_interacciones = coeficientes_interacciones,
    coeficientes_irrigable_proxy = coeficientes_irrigable_proxy
  ),
  "data_processed/validacion_irrigacion_eva.xlsx"
)

cat("Merge y validación estadística corregida terminados correctamente.\n")
cat("Archivo principal generado:\n")
cat("- data_processed/base_modelo_eva_irrigacion_validacion.csv\n")


# ============================================================
# CONCLUSIONES DEL ANÁLISIS
# ============================================================

# 1. El proceso de integración EVA + irrigación es consistente
# La cobertura es alta y no se identifican problemas estructurales en el merge.
# El dataset final es adecuado para modelado estadístico.

# 2. La transformación logarítmica del rendimiento fue determinante
# Redujo el sesgo por valores extremos y mejoró la estabilidad del modelo.
# Permitió aumentar la capacidad explicativa del modelo de forma significativa.

# 3. El modelo con controles es el más adecuado
# Integra variables productivas, estructurales y temporales.
# Presenta el mayor R² ajustado y mejor desempeño global.

# 4. El componente socioeconómico es el principal determinante
# Tiene el mayor efecto y significancia estadística.
# Indica que el desarrollo territorial condiciona la productividad agrícola.

# 5. La estructura productiva influye de manera importante
# Tanto la diversificación como la concentración de cultivos presentan efectos positivos.
# Esto sugiere la existencia de múltiples estrategias productivas eficientes.

# 6. La intensidad agrícola es relevante
# Un mayor uso efectivo del suelo se asocia con mayores niveles de rendimiento.

# 7. El potencial de irrigación no explica directamente el rendimiento
# Las métricas promedio del índice no presentan capacidad explicativa.
# Esto evidencia limitaciones en el enfoque tradicional basado en promedios.

# 8. La variable más consistente de irrigación es el percentil 75
# Captura la presencia de zonas de alto potencial dentro del municipio.
# Representa mejor la capacidad efectiva del territorio.

# 9. La proporción de área con alto potencial presenta relación negativa
# Una mayor disponibilidad de área potencial no implica mayor producción.
# Esto sugiere problemas de uso o aprovechamiento del recurso.

# 10. La heterogeneidad interna no es un factor determinante
# La variabilidad del índice dentro del municipio no muestra efectos significativos.

# 11. Los modelos con interacciones no mejoran el desempeño
# No aportan capacidad explicativa adicional frente al modelo base.
# Se descartan como modelo principal.

# 12. El modelo basado en proxy irrigable es intermedio
# Mejora respecto a modelos simples, pero no supera al modelo con controles.

# 13. Hallazgo principal
# El potencial de irrigación es una condición necesaria pero no suficiente
# para explicar el rendimiento agrícola.

# 14. Validación de la hipótesis
# El modelo oficial basado en variables físicas es insuficiente.
# El enfoque multidimensional ofrece una mejor representación del fenómeno.

# 15. Conclusión general
# El rendimiento agrícola es un fenómeno multicausal.
# Depende de factores estructurales, productivos y territoriales.
# La integración de estos factores permite construir modelos más robustos.