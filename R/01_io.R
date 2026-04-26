leer_eva <- function(path) {
  
  lineas <- readLines(path, encoding = "Latin1", warn = FALSE)
  
  lineas_limpias <- lineas |>
    stringr::str_replace('^"', '') |>
    stringr::str_replace('"$', '') |>
    stringr::str_replace_all('""', '"')
  
  readr::read_csv(
    I(lineas_limpias),
    col_types = readr::cols(.default = readr::col_character()),
    locale = readr::locale(encoding = "Latin1"),
    show_col_types = FALSE,
    trim_ws = TRUE
  )
}