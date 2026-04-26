limpiar_valores_eva <- function(df) {
  df %>%
    mutate(
      across(
        where(is.character),
        ~ .x %>%
          stringr::str_squish() %>%
          stringr::str_to_upper() %>%
          stringi::stri_trans_general("Latin-ASCII")
      )
    ) %>%
    mutate(
      codigo_departamento = as.integer(codigo_departamento),
      codigo_municipio = as.character(codigo_municipio),
      anio = as.character(anio),
      anio = stringr::str_replace_all(anio, "\\.", ""),
      anio = as.integer(anio),
      area_sembrada_ha = as.numeric(stringr::str_replace_all(area_sembrada_ha, "\\.", "")),
      area_cosechada_ha = as.numeric(stringr::str_replace_all(area_cosechada_ha, "\\.", "")),
      produccion_t = as.numeric(stringr::str_replace_all(produccion_t, "\\.", ""))
    )
}

