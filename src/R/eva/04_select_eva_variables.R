# D:/Proyectos_IA/proyecto-gnn-agricola/eva/04_select_eva_variables.R
# 04_select_eva_variables.R

# Ejecutar configuración global
source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros globales

# 1. ¿Puede cargarse correctamente el panel agrícola auditado?

# 1.1. ¿Cuáles son los archivos de entrada y salida?
archivo_eva_municipio_anio <- file.path(
  ruta_processed_eva,
  "eva_municipio_anio.csv"
) # Panel agrícola auditado

archivo_panel_eva_gnn <- file.path(
  ruta_processed_eva,
  "panel_eva_gnn.csv"
) # Panel EVA listo para integración GNN

# 1.2. ¿Existe el archivo de entrada?
stopifnot(
  file.exists(
    archivo_eva_municipio_anio
  )
) # Validar existencia del archivo
cat("Archivo de entrada encontrado:", archivo_eva_municipio_anio, "\n") # Confirmar existencia

# 1.3. ¿Puede cargarse correctamente el panel agrícola auditado?
eva_municipio_anio <- data.table::fread(
  archivo_eva_municipio_anio
); eva_municipio_anio # Cargar panel agrícola auditado

cat("\nVALIDACIÓN DEL PANEL AGRÍCOLA AUDITADO\n") # Encabezado
cat("Filas:", nrow(eva_municipio_anio), "\n") # Mostrar filas
cat("Columnas:", ncol(eva_municipio_anio), "\n") # Mostrar columnas
cat("Cobertura temporal:", min(eva_municipio_anio$anio, na.rm = TRUE), "-", max(eva_municipio_anio$anio, na.rm = TRUE), "\n") # Mostrar rango temporal
cat("Municipios únicos:", dplyr::n_distinct(eva_municipio_anio$cod_mpio), "\n") # Mostrar municipios
cat("Archivo destino:", archivo_panel_eva_gnn, "\n") # Mostrar archivo de salida

# Validaciones estructurales
stopifnot(
  nrow(eva_municipio_anio) > 0
) # Validar filas

stopifnot(
  ncol(eva_municipio_anio) > 0
) # Validar columnas

stopifnot(
  dplyr::n_distinct(
    eva_municipio_anio$anio
  ) > 0
) # Validar cobertura temporal

cat("Estado validación: APROBADA\n") # Confirmar validación
# ---------------------------------------------

# 2. SELECCIÓN DE VARIABLES DE IDENTIFICACIÓN
# 2.1. ¿Qué variables de identificación deben conservarse?
variables_identificacion <- c(
  "cod_depto",
  "cod_mpio",
  "anio"
) # Definir identificadores obligatorios
cat("\nVARIABLES DE IDENTIFICACIÓN\n") # Encabezado
cat("Variables seleccionadas:", paste(variables_identificacion, collapse = ", "), "\n") # Mostrar variables

# 2.2. ¿Existen todas las variables de identificación?

variables_faltantes <- setdiff(
  variables_identificacion,
  names(eva_municipio_anio)
) # Identificar variables ausentes

cat("Variables faltantes:", length(variables_faltantes), "\n") # Mostrar cantidad

if (length(variables_faltantes) > 0) {
  print(variables_faltantes) # Mostrar variables faltantes
}

stopifnot(
  length(variables_faltantes) == 0
) # Validar existencia de variables

# 2.3. ¿Presentan valores faltantes?
na_identificacion <- data.frame(
  variable = variables_identificacion,
  na_total = sapply(
    variables_identificacion,
    function(x) sum(is.na(eva_municipio_anio[[x]]))
  ),
  porcentaje_na = round(
    sapply(
      variables_identificacion,
      function(x) mean(is.na(eva_municipio_anio[[x]])) * 100
    ),
    4
  )
) # Auditar NA

na_identificacion # Mostrar auditoría

# 2.4. ¿Son únicas las llaves municipio-año?
duplicados_identificacion <- eva_municipio_anio |>
  dplyr::count(
    cod_mpio,
    anio
  ) |>
  dplyr::filter(
    n > 1
  ); duplicados_identificacion # Buscar duplicados

cat("Duplicados cod_mpio + anio:", nrow(duplicados_identificacion), "\n") # Mostrar duplicados

stopifnot(
  nrow(duplicados_identificacion) == 0
) # Certificar unicidad

cat("Estado variables identificación: APROBADAS\n") # Confirmar validación
# ---------------------------------------------------------

# 3. SELECCIÓN DE VARIABLES AGRÍCOLAS
# 3.1. ¿Qué variables agrícolas ingresarán al panel maestro?
variables_agricolas <- c(
  "area_sembrada_total",
  "area_cosechada_total",
  "produccion_total",
  "n_cultivos",
  "n_grupos_cultivo",
  "porcentaje_permanentes",
  "rendimiento_promedio",
  "tasa_cosecha_promedio",
  "log_produccion_total",
  "log_area_sembrada_total"
) # Definir variables agrícolas analíticas

cat("\nVARIABLES AGRÍCOLAS\n") # Encabezado
cat("Variables seleccionadas:", paste(variables_agricolas, collapse = ", "), "\n") # Mostrar variables

# 3.2. ¿Existen todas las variables agrícolas?
variables_agricolas_faltantes <- setdiff(
  variables_agricolas,
  names(eva_municipio_anio)
) # Identificar variables ausentes

cat("Variables faltantes:", length(variables_agricolas_faltantes), "\n") # Mostrar cantidad

if (length(variables_agricolas_faltantes) > 0) {
  print(variables_agricolas_faltantes) # Mostrar variables faltantes
}

stopifnot(
  length(variables_agricolas_faltantes) == 0
) # Validar existencia de variables

# 3.3. ¿Presentan valores faltantes?
na_agricolas <- data.frame(
  variable = variables_agricolas,
  na_total = sapply(
    variables_agricolas,
    function(x) sum(is.na(eva_municipio_anio[[x]]))
  ),
  porcentaje_na = round(
    sapply(
      variables_agricolas,
      function(x) mean(is.na(eva_municipio_anio[[x]])) * 100
    ),
    4
  )
) # Auditar NA

na_agricolas # Mostrar auditoría

# 3.4. ¿Cuál es la estructura de las variables agrícolas?
estructura_agricolas <- data.frame(
  variable = variables_agricolas,
  clase = sapply(
    eva_municipio_anio[, ..variables_agricolas],
    class
  )
) # Documentar tipos de datos

estructura_agricolas # Mostrar estructura

cat("Variables agrícolas seleccionadas:", length(variables_agricolas), "\n") # Mostrar cantidad

cat("Estado variables agrícolas: APROBADAS\n") # Confirmar validación
# ----------------------------------------------------------

# 4. CONSTRUCCIÓN DE VARIABLE OBJETIVO
# 4.1. ¿Debe construirse la variable objetivo oficial del proyecto?
eva_municipio_anio <- eva_municipio_anio |>
  dplyr::mutate(
    log_rendimiento = log1p(
      rendimiento_promedio
    )
  ) # Construir variable objetivo candidata

cat("\nVARIABLE OBJETIVO\n") # Encabezado

cat(
  "Registros con log_rendimiento:",
  sum(!is.na(eva_municipio_anio$log_rendimiento)),
  "\n"
) # Contar registros válidos

cat(
  "NA en log_rendimiento:",
  sum(is.na(eva_municipio_anio$log_rendimiento)),
  "\n"
) # Contar NA

# 4.2. ¿Cuál es la distribución de la variable objetivo?
summary(
  eva_municipio_anio$log_rendimiento
) # Resumen estadístico

# 4.3. ¿Existen valores infinitos?
cat(
  "Valores infinitos:",
  sum(
    is.infinite(
      eva_municipio_anio$log_rendimiento
    )
  ),
  "\n"
) # Detectar infinitos

stopifnot(
  sum(
    is.infinite(
      eva_municipio_anio$log_rendimiento
    )
  ) == 0
) # Validar construcción

cat(
  "Estado variable objetivo: APROBADA\n"
) # Confirmar construcción
# ----------------------------------------------------------

# 5. EXCLUSIÓN DE VARIABLES DE AUDITORÍA
# 5.1. ¿Qué variables de auditoría deben excluirse?
variables_auditoria <- c(
  "rendimiento_promedio_corregido",
  "audit_rendimiento_promedio",
  "flag_tasa_imposible",
  "flag_rendimiento_extremo"
) # Variables utilizadas únicamente durante auditoría

cat("\nVARIABLES DE AUDITORÍA\n") # Encabezado
cat(
  "Variables candidatas a exclusión:",
  paste(variables_auditoria, collapse = ", "),
  "\n"
) # Mostrar variables

# 5.2. ¿Existen todas las variables de auditoría?
variables_auditoria_faltantes <- setdiff(
  variables_auditoria,
  names(eva_municipio_anio)
) # Identificar variables ausentes

cat(
  "Variables faltantes:",
  length(variables_auditoria_faltantes),
  "\n"
) # Mostrar cantidad

if (length(variables_auditoria_faltantes) > 0) {
  print(variables_auditoria_faltantes)
} # Mostrar variables faltantes

stopifnot(
  length(variables_auditoria_faltantes) == 0
) # Validar existencia

# 5.3. ¿Cuántas variables serán excluidas?
cat(
  "Variables a excluir:",
  length(variables_auditoria),
  "\n"
) # Mostrar cantidad

# 5.4. ¿Cuál será la estructura analítica después de la exclusión?
variables_restantes <- setdiff(
  names(eva_municipio_anio),
  variables_auditoria
) # Calcular variables restantes

cat(
  "Variables restantes:",
  length(variables_restantes),
  "\n"
) # Mostrar cantidad
cat(
  "Estado variables auditoría: APROBADAS PARA EXCLUSIÓN\n"
) # Confirmar resultado
#-----------------------------------------------------------

# 6. ¿Puede construirse el panel final para GNN?
panel_eva_gnn <- eva_municipio_anio |>
  dplyr::select(
    cod_depto,
    cod_mpio,
    anio,
    area_sembrada_total,
    area_cosechada_total,
    produccion_total,
    n_cultivos,
    n_grupos_cultivo,
    porcentaje_permanentes,
    rendimiento_promedio,
    tasa_cosecha_promedio,
    log_produccion_total,
    log_area_sembrada_total,
    log_rendimiento
  ) # Construir panel final GNN

cat(
  "Variables panel GNN:",
  ncol(panel_eva_gnn),
  "\n"
) # Mostrar variables

cat(
  "Registros panel GNN:",
  nrow(panel_eva_gnn),
  "\n"
) # Mostrar registros
#-----------------------------------------------------------

# 7. ¿Existen valores faltantes en las variables seleccionadas para el panel GNN?
na_panel_gnn <- data.frame(
  variable = names(panel_eva_gnn),
  na_total = colSums(is.na(panel_eva_gnn)),
  porcentaje_na = round(
    colSums(is.na(panel_eva_gnn)) /
      nrow(panel_eva_gnn) * 100,
    4
  )
) |>
  dplyr::arrange(
    dplyr::desc(na_total)
  ); na_panel_gnn # Auditoría final de NA
# ----------------------------------------------------------

# 8. INTEGRIDAD DE LA LLAVE MUNICIPIO-AÑO
# 8.1. ¿La llave municipio-año permanece íntegra?
duplicados_panel_gnn <- panel_eva_gnn |>
  dplyr::count(
    cod_mpio,
    anio
  ) |>
  dplyr::filter(
    n > 1
  ); duplicados_panel_gnn # Buscar duplicados en la llave

cat(
  "Duplicados cod_mpio + anio:",
  nrow(duplicados_panel_gnn),
  "\n"
) # Mostrar cantidad

# 8.2. ¿Existen llaves repetidas?
stopifnot(
  nrow(duplicados_panel_gnn) == 0
) # Certificar unicidad

cat(
  "Estado llave municipio-año: APROBADA\n"
) # Confirmar validación
# ---------------------------------------------------------

# 9. ESTRUCTURA FINAL DE LA CAPA EVA-GNN
# 9.1. ¿Cuál es la dimensión final del panel?
cat("\nESTRUCTURA FINAL EVA-GNN\n") # Encabezado
cat("Filas:", nrow(panel_eva_gnn), "\n") # Mostrar registros
cat("Columnas:", ncol(panel_eva_gnn), "\n") # Mostrar variables

# 9.2. ¿Cuántos municipios contiene?
cat("Municipios únicos:", dplyr::n_distinct(panel_eva_gnn$cod_mpio), "\n") # Mostrar municipios

# 9.3. ¿Cuál es la cobertura temporal?
cat("Cobertura temporal:", min(panel_eva_gnn$anio), "-", max(panel_eva_gnn$anio), "\n") # Mostrar periodo

# 9.4. ¿Cuáles son las variables seleccionadas?
variables_panel_gnn <- names(
  panel_eva_gnn
) # Recuperar variables
cat("Variables seleccionadas:", length(variables_panel_gnn), "\n") # Mostrar cantidad
print(variables_panel_gnn) # Mostrar listado completo
cat("Estado estructura EVA-GNN: APROBADA\n") # Confirmar validación
# ----------------------------------------------------------

# 10. EXPORTACIÓN DE LA CAPA AGRÍCOLA FINAL
# 10.1. ¿Puede exportarse la capa agrícola final?
archivo_eva_gnn_final <- file.path(
  ruta_processed_eva,
  "eva_gnn_final.csv"
) # Archivo final EVA para integración GNN

data.table::fwrite(
  panel_eva_gnn,
  archivo_eva_gnn_final
) # Exportar capa agrícola final
cat("Archivo exportado:", archivo_eva_gnn_final, "\n") # Confirmar exportación

# 10.2. ¿Existe el archivo exportado?
stopifnot(
  file.exists(
    archivo_eva_gnn_final
  )
) # Validar exportación

# 10.3. ¿Puede recargarse correctamente?
eva_gnn_final <- data.table::fread(
  archivo_eva_gnn_final
) # Recargar archivo exportado
cat("Filas exportadas:", nrow(eva_gnn_final), "\n") # Mostrar filas
cat("Columnas exportadas:", ncol(eva_gnn_final), "\n") # Mostrar columnas

# 10.4. ¿Coinciden las dimensiones con el panel original?
stopifnot(
  nrow(eva_gnn_final) ==
    nrow(panel_eva_gnn)
) # Validar filas

stopifnot(
  ncol(eva_gnn_final) ==
    ncol(panel_eva_gnn)
) # Validar columnas
cat("Estado exportación: APROBADA\n") # Confirmar resultado





