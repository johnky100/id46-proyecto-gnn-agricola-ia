# audit/01_audit_eva_municipio_anio.R

# Cargar configuración global

source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros globales

# Pregunta: ¿Cómo cargar la base EVA municipio-año?

eva <- data.table::fread(
  file.path(
    ruta_processed,
    "eva_municipio_anio.csv"
  )
) # Cargar dataset EVA municipio-año

# Pregunta: ¿Cuál es la dimensión de la base?

cat("\nDIMENSIÓN DE LA BASE\n") # Mostrar encabezado
cat("Filas:", nrow(eva), "\n") # Mostrar filas
cat("Columnas:", ncol(eva), "\n") # Mostrar columnas

# Pregunta: ¿Qué variables contiene la base?

cat("\nVARIABLES DISPONIBLES\n") # Mostrar encabezado
print(names(eva)) # Mostrar nombres de variables

# Pregunta: ¿Cuál es la estructura de las variables?

cat("\nTIPOS DE DATOS\n") # Mostrar encabezado
print(sapply(eva, class)) # Mostrar tipos de datos

# Pregunta: ¿Existen valores faltantes?

na_por_variable <- colSums(is.na(eva)) # Contar NA por variable

cat("\nVALORES FALTANTES\n") # Mostrar encabezado
print(na_por_variable) # Mostrar NA por variable

cat(
  "Total valores faltantes:",
  sum(is.na(eva)),
  "\n"
) # Mostrar total de NA

# Pregunta: ¿Existen registros completamente duplicados?

duplicados_totales <- sum(duplicated(eva)) # Contar duplicados completos

cat(
  "\nDuplicados completos:",
  duplicados_totales,
  "\n"
) # Mostrar duplicados completos

# Pregunta: ¿La llave municipio-año es única?

duplicados_municipio_anio <- eva[
  duplicated(eva[, .(cod_mpio, anio)])
] # Buscar duplicados municipio-año

cat(
  "Duplicados municipio-año:",
  nrow(duplicados_municipio_anio),
  "\n"
) # Mostrar duplicados municipio-año

# Pregunta: ¿Cuáles son los registros duplicados?

duplicados_municipio_anio # Mostrar registros duplicados

# Pregunta: ¿Cuál es la cobertura espacial?

cat(
  "\nMunicipios únicos:",
  dplyr::n_distinct(eva$cod_mpio),
  "\n"
) # Mostrar municipios únicos

# Pregunta: ¿Cuál es la cobertura temporal?

cat(
  "Año mínimo:",
  min(eva$anio, na.rm = TRUE),
  "\n"
) # Mostrar año mínimo

cat(
  "Año máximo:",
  max(eva$anio, na.rm = TRUE),
  "\n"
) # Mostrar año máximo

cat("\nREGISTROS POR AÑO\n") # Mostrar encabezado
print(table(eva$anio)) # Mostrar registros por año

# Pregunta: ¿Existen años faltantes?

anios_esperados <- seq(
  min(eva$anio, na.rm = TRUE),
  max(eva$anio, na.rm = TRUE)
) # Construir años esperados

anios_faltantes <- setdiff(
  anios_esperados,
  unique(eva$anio)
) # Detectar años faltantes

cat("\nAÑOS FALTANTES\n") # Mostrar encabezado
print(anios_faltantes) # Mostrar años faltantes

# Pregunta: ¿Existen registros sin producción?

municipios_sin_produccion <- eva[
  produccion_total <= 0 |
    is.na(produccion_total)
] # Detectar registros sin producción

cat(
  "\nRegistros sin producción:",
  nrow(municipios_sin_produccion),
  "\n"
) # Mostrar cantidad

# Pregunta: ¿Existen valores negativos imposibles?

variables_productivas <- c(
  "area_sembrada_total",
  "area_cosechada_total",
  "produccion_total"
) # Variables productivas válidas

cat("\nVALORES NEGATIVOS\n") # Mostrar encabezado

for (var in variables_productivas) {
  negativos <- sum(eva[[var]] < 0, na.rm = TRUE) # Contar negativos
  cat(var, ":", negativos, "valores negativos\n") # Mostrar resultado
}

# Pregunta: ¿Cómo se comportan las variables principales?

cat("\nRESUMEN ESTADÍSTICO\n") # Mostrar encabezado

print(
  summary(
    eva[, .(
      area_sembrada_total,
      area_cosechada_total,
      produccion_total,
      log_produccion_total,
      log_area_sembrada_total
    )]
  )
) # Mostrar resumen estadístico

# Pregunta: ¿Existen valores extremos?

cat("\nPERCENTIL 99\n") # Mostrar encabezado

print(
  eva[, lapply(
    .SD,
    quantile,
    probs = 0.99,
    na.rm = TRUE
  ),
  .SDcols = c(
    "area_sembrada_total",
    "area_cosechada_total",
    "produccion_total"
  )]
) # Calcular percentil 99

# Pregunta: ¿Las transformaciones logarítmicas son válidas?

cat("\nVALIDACIÓN LOGARÍTMICA\n") # Mostrar encabezado

cat(
  "NA en log_produccion_total:",
  sum(is.na(eva$log_produccion_total)),
  "\n"
) # Validar log producción

cat(
  "NA en log_area_sembrada_total:",
  sum(is.na(eva$log_area_sembrada_total)),
  "\n"
) # Validar log área

# Pregunta: ¿Está bien calculado el rendimiento promedio?

eva[, rendimiento_recalculado := ifelse(
  area_cosechada_total > 0,
  produccion_total / area_cosechada_total,
  NA_real_
)] # Recalcular rendimiento correcto

eva[, diferencia_rendimiento := abs(
  rendimiento_promedio -
    rendimiento_recalculado
)] # Calcular diferencia

cat("\nVALIDACIÓN DE RENDIMIENTO\n") # Mostrar encabezado

summary(eva$diferencia_rendimiento) # Mostrar diferencias

# Pregunta: ¿Cuáles son los mayores errores de rendimiento?

eva[
  order(-diferencia_rendimiento)
][1:20,
  .(
    municipio,
    anio,
    produccion_total,
    area_cosechada_total,
    rendimiento_promedio,
    rendimiento_recalculado,
    diferencia_rendimiento
  )
] # Mostrar mayores inconsistencias

# Pregunta: ¿Existen registros con NA en rendimiento o tasa de cosecha?

eva[
  is.na(rendimiento_promedio) |
    is.na(tasa_cosecha_promedio)
] # Mostrar registros problemáticos

# Pregunta: ¿Cuál es el diagnóstico final de la base?

cat("\nRESUMEN EJECUTIVO\n") # Mostrar encabezado

cat("Filas:", nrow(eva), "\n") # Mostrar filas
cat("Columnas:", ncol(eva), "\n") # Mostrar columnas
cat("Municipios únicos:", dplyr::n_distinct(eva$cod_mpio), "\n") # Mostrar municipios
cat("Cobertura temporal:", min(eva$anio), "-", max(eva$anio), "\n") # Mostrar cobertura
cat("Duplicados municipio-año:", nrow(duplicados_municipio_anio), "\n") # Mostrar duplicados
cat("Variables con NA:", sum(na_por_variable > 0), "\n") # Mostrar variables con NA
cat("Total NA:", sum(is.na(eva)), "\n") # Mostrar total de NA

cat(
  "\nOBSERVACIONES CRÍTICAS:\n",
  "- Existe 1 duplicado municipio-año.\n",
  "- rendimiento_promedio está mal calculado.\n",
  "- tasa_cosecha_promedio está mal calculada.\n",
  "- Se recomienda corregir el script de construcción de eva_municipio_anio.csv antes de integrar el panel maestro.\n"
) # Mostrar diagnóstico final
