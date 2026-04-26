seleccionar_variables_eva <- function(df) {
  df %>%
    select(
      codigo_departamento,
      nombre_departamento,
      codigo_municipio,
      nombre_municipio,
      anio,
      grupo_cultivo,
      subgrupo_cultivo,
      cultivo,
      area_sembrada_ha,
      area_cosechada_ha,
      produccion_t,
      ciclo_cultivo
    )
}