crear_variables_eva <- function(df) {
  df %>%
    mutate(
      rendimiento_t_ha = case_when(
        !is.na(area_cosechada_ha) &
          area_cosechada_ha > 0 &
          !is.na(produccion_t) ~ produccion_t / area_cosechada_ha,
        TRUE ~ NA_real_
      ),
      tasa_cosecha = case_when(
        !is.na(area_sembrada_ha) &
          area_sembrada_ha > 0 &
          !is.na(area_cosechada_ha) ~ area_cosechada_ha / area_sembrada_ha,
        TRUE ~ NA_real_
      ),
      brecha_area_ha = area_sembrada_ha - area_cosechada_ha
    )
}