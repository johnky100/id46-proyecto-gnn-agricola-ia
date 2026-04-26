packages <- c(
  "tidyverse",
  "janitor",
  "stringi",
  "readr"
)

invisible(lapply(packages, library, character.only = TRUE))

options(scipen = 999)
