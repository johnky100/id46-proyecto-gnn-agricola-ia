# D:/Proyectos_IA/proyecto-gnn-agricola/cna/01_clean_cna.R
source(here::here("config", "00_packages.R"))
source(here::here("config", "01_paths.R"))
source(here::here("config", "02_global_parameters.R"))

# --- Rutas ---
ruta_cna_raw      <- file.path(ruta_raw,       "cna")
ruta_processed_cna <- file.path(ruta_processed, "cna")
ruta_outputs_cna   <- file.path(ruta_outputs,   "cna")

dir.create(ruta_processed_cna, recursive = TRUE, showWarnings = FALSE)
dir.create(ruta_outputs_cna,   recursive = TRUE, showWarnings = FALSE)

archivo_catalogo  <- file.path(ruta_processed_cna, "cna_catalogo_variables.csv")
archivo_auditoria <- file.path(ruta_outputs_cna,   "auditoria_cna.csv")
archivo_variables <- file.path(ruta_outputs_cna,   "variables_detectadas_cna.csv")
ruta_gpkg_cna     <- file.path(ruta_processed_cna, "cna_clean.gpkg")  # nuevo

# --- Leer shapefile ---
archivo_shp <- list.files(ruta_cna_raw, pattern = "\\.shp$", full.names = TRUE)
if (length(archivo_shp) == 0) stop(paste("No se encontró shapefile en:", ruta_cna_raw))
ruta_shp <- archivo_shp[1]

cna_sf <- sf::st_read(ruta_shp, quiet = TRUE)

# --- CRS: validar y reproyectar si es necesario ---
crs_cna <- sf::st_crs(cna_sf)
if (!is.na(crs_cna$epsg) && crs_cna$epsg != 9377) {      # MAGNA-SIRGAS
  cna_sf <- sf::st_transform(cna_sf, 9377)
  crs_cna <- sf::st_crs(cna_sf)
  message("CRS reproyectado a EPSG:9377 (MAGNA-SIRGAS Colombia)")
}

# --- Geometrías inválidas ---
geometrias_invalidas <- sum(!sf::st_is_valid(cna_sf))
if (geometrias_invalidas > 0) {                            # reparar
  cna_sf <- sf::st_make_valid(cna_sf)
  message(glue::glue("{geometrias_invalidas} geometría(s) reparada(s) con st_make_valid()"))
}

# --- Catálogo de variables (bug fix) ---
variables_detectadas <- tibble::tibble(
  variable = names(cna_sf),
  clase    = purrr::map_chr(cna_sf, \(x) paste(class(x), collapse = ", "))  # fix
)

# --- Auditoría ---
auditoria_cna <- tibble::tibble(
  fecha_proceso        = Sys.time(),
  archivo              = basename(ruta_shp),
  registros            = nrow(cna_sf),
  variables            = ncol(cna_sf),
  geometrias_invalidas = geometrias_invalidas,
  crs_epsg             = crs_cna$epsg,
  bbox_xmin            = sf::st_bbox(cna_sf)[1],
  bbox_ymin            = sf::st_bbox(cna_sf)[2],
  bbox_xmax            = sf::st_bbox(cna_sf)[3],
  bbox_ymax            = sf::st_bbox(cna_sf)[4]
)

# --- Exportar ---
data.table::fwrite(variables_detectadas, archivo_variables)
data.table::fwrite(variables_detectadas, archivo_catalogo)
data.table::fwrite(auditoria_cna,        archivo_auditoria)
sf::st_write(cna_sf, ruta_gpkg_cna, delete_dsn = TRUE, quiet = TRUE)  # nuevo

# --------------------
names(cna_sf)

# --- Resumen consola ---
cat("\nCATÁLOGO CNA GENERADO CORRECTAMENTE\n")
cat("Archivo:              ", basename(ruta_shp), "\n")
cat("Registros:            ", nrow(cna_sf), "\n")
cat("Variables:            ", ncol(cna_sf), "\n")
cat("CRS (EPSG):           ", crs_cna$epsg, "\n")
cat("Geometrías inválidas: ", geometrias_invalidas, "\n")
cat("GeoPackage exportado: ", basename(ruta_gpkg_cna), "\n")
cat("Estado CNA: APROBADO PARA AGREGACIÓN MUNICIPAL\n")
