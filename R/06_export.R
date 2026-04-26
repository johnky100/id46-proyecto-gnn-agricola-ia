exportar_csv <- function(df, path) {
  readr::write_csv(df, path)
}