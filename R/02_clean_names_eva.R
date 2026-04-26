renombrar_variables_eva <- function(df) {
  df %>%
    janitor::clean_names() %>%
    rename(
      codigo_departamento = ci_1_2d_dep,
      nombre_departamento = departamento,
      codigo_municipio = ci_1_2d_mun,
      nombre_municipio = municipio,
      grupo_cultivo = grupo_de_cultivo,
      subgrupo_cultivo = subgrupo_de_cultivo,
      cultivo = cultivo,
      anio = ai_1_2o,
      area_sembrada_ha = i_1_2rea_sembrada_ha,
      area_cosechada_ha = i_1_2rea_cosechada_ha,
      produccion_t = produccii_1_2n_t,
      ciclo_cultivo = ciclo_de_cultivo
    )
}