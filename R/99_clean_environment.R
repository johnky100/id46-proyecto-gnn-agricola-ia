limpiar_entorno <- function() {
  rm(list = ls(envir = .GlobalEnv), envir = .GlobalEnv)
  gc()
  
  while (dev.cur() > 1) {
    dev.off()
  }
  
  cat("\014")
  message("Entorno, memoria, consola y gráficos limpiados.")
}

# Uso en Consola:
# source("R/99_clean_environment.R")
# limpiar_entorno()