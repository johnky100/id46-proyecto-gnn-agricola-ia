# D:/Proyectos_IA/proyecto-gnn-agricola/eva/01_clean_eva.R
# sources/eva/01_clean_eva.R

# Pregunta: ¿Cómo cargar la configuración global del proyecto?
source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros

# Pregunta: ¿Cuáles serán los archivos de salida?
archivo_eva_limpia <- file.path(
  ruta_processed_eva,
  "eva_limpia.csv"
) # Archivo EVA limpio

archivo_auditoria <- file.path(
  ruta_outputs_eva,
  "auditoria_eva_limpia.csv"
) # Auditoría de limpieza


# 1. AUDITORÍA ESTRUCTURAL RAW
# Pregunta: ¿El archivo EVA original tiene la estructura esperada?
eva_raw <- data.table::fread(
  archivo_eva,
  encoding = "Latin-1"
) # Cargar archivo EVA raw

cat("\nAUDITORÍA ESTRUCTURAL RAW\n") # Encabezado

cat("Filas:", nrow(eva_raw), "\n") # Mostrar filas
cat("Columnas:", ncol(eva_raw), "\n") # Mostrar columnas

na_raw <- colSums(is.na(eva_raw)) # Calcular NA por variable

duplicados_raw <- sum(
  duplicated(eva_raw)
) # Contar duplicados exactos


# 2. CORRECCIÓN DEL ENCAPSULADO
# Pregunta: ¿Puede recuperarse la estructura original del archivo?
lineas_eva <- readLines(
  archivo_eva,
  encoding = "Latin1",
  warn = FALSE
) # Leer archivo EVA original

lineas_eva_corregidas <- lineas_eva |>
  stringr::str_remove('^"') |>
  stringr::str_remove('"$') |>
  stringr::str_replace_all('""', '"') # Corregir encapsulado

# Pregunta: ¿La estructura corregida conserva las 17 columnas esperadas?
eva_test <- readr::read_delim(
  I(lineas_eva_corregidas),
  delim = ",",
  quote = '"',
  col_names = TRUE,
  show_col_types = FALSE,
  col_types = readr::cols(
    .default = readr::col_character()
  )
) # Validar estructura corregida

cat("\nCORRECCIÓN DE ENCAPSULADO\n") # Encabezado

cat("Filas recuperadas:", nrow(eva_test), "\n") # Mostrar filas recuperadas

cat("Columnas recuperadas:", ncol(eva_test), "\n") # Mostrar columnas recuperadas

if (ncol(eva_test) != 17) {
  stop(
    paste(
      "Error en recuperación de estructura. Se encontraron",
      ncol(eva_test),
      "columnas y se esperaban 17."
    )
  ) # Validar estructura oficial EVA
}

cat("Estado encapsulado: CORREGIDO CORRECTAMENTE\n") # Confirmar corrección


# 3. LECTURA CORREGIDA
# Pregunta: ¿La estructura corregida coincide con la estructura oficial de EVA?
eva_corregida <- readr::read_delim(
  I(lineas_eva_corregidas),
  delim = ",",
  quote = '"',
  col_names = TRUE,
  show_col_types = FALSE,
  col_types = readr::cols(
    .default = readr::col_character()
  )
) # Leer EVA corregida

cat("\nLECTURA CORREGIDA\n") # Encabezado

cat("Filas recuperadas:", nrow(eva_corregida), "\n") # Mostrar filas recuperadas

cat("Columnas recuperadas:", ncol(eva_corregida), "\n") # Mostrar columnas recuperadas

cat("Columnas esperadas:", 17, "\n") # Mostrar estructura esperada

if (ncol(eva_corregida) != 17) {
  stop(
    paste(
      "Error: se esperaban 17 columnas y se encontraron",
      ncol(eva_corregida)
    )
  )
} # Validar estructura oficial EVA

cat("Estado lectura: ESTRUCTURA RECUPERADA CORRECTAMENTE\n") # Confirmar lectura exitosa


# 4. ESTANDARIZACIÓN DE NOMBRES DE VARIABLES
# Pregunta: ¿Cómo convertir los nombres originales de EVA a la nomenclatura oficial del proyecto?
if (ncol(eva_corregida) != 17) {
  stop(
    paste(
      "La estructura no contiene 17 columnas. Se encontraron",
      ncol(eva_corregida)
    )
  )
} # Validar estructura antes de renombrar

nombres_originales <- names(
  eva_corregida
) # Guardar nombres originales para auditoría

names(eva_corregida) <- c(
  "cod_depto",
  "departamento",
  "cod_mpio",
  "municipio",
  "grupo_cultivo",
  "subgrupo_cultivo",
  "cultivo",
  "sistema_productivo",
  "anio",
  "periodo",
  "area_sembrada_ha",
  "area_cosechada_ha",
  "produccion_t",
  "rendimiento_t_ha",
  "estado_fisico_produccion",
  "nombre_cientifico",
  "ciclo_cultivo"
) # Aplicar nomenclatura oficial del proyecto

cat("\nESTANDARIZACIÓN DE VARIABLES\n") # Encabezado

cat("Variables renombradas:", length(names(eva_corregida)), "\n") # Mostrar total variables

cat("Estado nombres: ESTANDARIZADOS CORRECTAMENTE\n") # Confirmar proceso

print(names(eva_corregida)) # Mostrar nombres finales


# 5. CONVERSIÓN Y VALIDACIÓN DE TIPOS DE DATOS

# Pregunta: ¿Las variables fueron limpiadas, convertidas y validadas correctamente?

eva_limpia <- eva_corregida |>
  dplyr::mutate(
    
    cod_depto = stringr::str_pad(
      gsub("\\D", "", cod_depto),
      width = 2,
      pad = "0"
    ), # Estandarizar código departamento
    
    cod_mpio = stringr::str_pad(
      gsub("\\D", "", cod_mpio),
      width = 5,
      pad = "0"
    ), # Estandarizar código municipio
    
    departamento = stringr::str_squish(
      stringr::str_to_upper(
        stringi::stri_trans_general(
          departamento,
          "Latin-ASCII"
        )
      )
    ), # Limpiar departamento
    
    municipio = stringr::str_squish(
      stringr::str_to_upper(
        stringi::stri_trans_general(
          municipio,
          "Latin-ASCII"
        )
      )
    ), # Limpiar municipio
    
    grupo_cultivo = stringr::str_squish(
      stringr::str_to_upper(
        stringi::stri_trans_general(
          grupo_cultivo,
          "Latin-ASCII"
        )
      )
    ), # Limpiar grupo cultivo
    
    subgrupo_cultivo = stringr::str_squish(
      stringr::str_to_upper(
        stringi::stri_trans_general(
          subgrupo_cultivo,
          "Latin-ASCII"
        )
      )
    ), # Limpiar subgrupo cultivo
    
    cultivo = stringr::str_squish(
      stringr::str_to_upper(
        stringi::stri_trans_general(
          cultivo,
          "Latin-ASCII"
        )
      )
    ), # Limpiar cultivo
    
    sistema_productivo = stringr::str_squish(
      stringr::str_to_upper(
        stringi::stri_trans_general(
          sistema_productivo,
          "Latin-ASCII"
        )
      )
    ), # Limpiar sistema productivo
    
    estado_fisico_produccion = stringr::str_squish(
      stringr::str_to_upper(
        stringi::stri_trans_general(
          estado_fisico_produccion,
          "Latin-ASCII"
        )
      )
    ), # Limpiar estado físico
    
    nombre_cientifico = stringr::str_squish(
      stringr::str_to_upper(
        stringi::stri_trans_general(
          nombre_cientifico,
          "Latin-ASCII"
        )
      )
    ), # Limpiar nombre científico
    
    ciclo_cultivo = stringr::str_squish(
      stringr::str_to_upper(
        stringi::stri_trans_general(
          ciclo_cultivo,
          "Latin-ASCII"
        )
      )
    ), # Limpiar ciclo cultivo
    
    anio = as.integer(
      gsub("\\.", "", anio)
    ), # Convertir año
    
    area_sembrada_ha = readr::parse_number(
      area_sembrada_ha,
      locale = readr::locale(decimal_mark = ",")
    ), # Convertir área sembrada
    
    area_cosechada_ha = readr::parse_number(
      area_cosechada_ha,
      locale = readr::locale(decimal_mark = ",")
    ), # Convertir área cosechada
    
    produccion_t = readr::parse_number(
      produccion_t,
      locale = readr::locale(decimal_mark = ",")
    ), # Convertir producción
    
    rendimiento_t_ha = readr::parse_number(
      rendimiento_t_ha,
      locale = readr::locale(decimal_mark = ",")
    ) # Convertir rendimiento
    
  ) |>
  dplyr::distinct() # Eliminar duplicados exactos

cat("\nCONVERSIÓN Y VALIDACIÓN DE TIPOS\n") # Encabezado

print(sapply(eva_limpia, class)) # Mostrar tipos finales

cat("NA en año:", sum(is.na(eva_limpia$anio)), "\n") # Validar año
cat("NA en área sembrada:", sum(is.na(eva_limpia$area_sembrada_ha)), "\n") # Validar área sembrada
cat("NA en área cosechada:", sum(is.na(eva_limpia$area_cosechada_ha)), "\n") # Validar área cosechada
cat("NA en producción:", sum(is.na(eva_limpia$produccion_t)), "\n") # Validar producción
cat("NA en rendimiento:", sum(is.na(eva_limpia$rendimiento_t_ha)), "\n") # Validar rendimiento
cat("Duplicados exactos:", sum(duplicated(eva_limpia)), "\n") # Validar duplicados

cat("Departamentos únicos:", dplyr::n_distinct(eva_limpia$cod_depto), "\n") # Validar departamentos
cat("Municipios únicos:", dplyr::n_distinct(eva_limpia$cod_mpio), "\n") # Validar municipios
cat("Año mínimo:", min(eva_limpia$anio, na.rm = TRUE), "\n") # Validar año mínimo
cat("Año máximo:", max(eva_limpia$anio, na.rm = TRUE), "\n") # Validar año máximo

cat("Estado conversión: COMPLETADA CORRECTAMENTE\n") # Confirmar proceso


# 6. AUDITORÍA DE RENDIMIENTOS FALTANTES
cat("\nAUDITORÍA DE RENDIMIENTOS FALTANTES\n") # Encabezado

# Pregunta: ¿Puede recalcularse el rendimiento a partir de producción y área cosechada?
eva_limpia <- eva_limpia |>
  dplyr::mutate(
    rendimiento_recalculado = dplyr::if_else(
      area_cosechada_ha > 0,
      produccion_t / area_cosechada_ha,
      NA_real_
    ),
    diferencia_rendimiento = abs(
      rendimiento_t_ha -
        rendimiento_recalculado
    )
  ) # Recalcular rendimiento y calcular diferencias

# Pregunta: ¿Cuántos NA existen actualmente en rendimiento_t_ha?
cat("NA en rendimiento_t_ha:", sum(is.na(eva_limpia$rendimiento_t_ha)), "\n") # Contar NA actuales

# Pregunta: ¿Los NA ya existían en el archivo original?
na_rendimiento_original <- sum(
  eva_corregida$rendimiento_t_ha == "" |
    is.na(eva_corregida$rendimiento_t_ha)
) # Contar NA originales

cat("NA o vacíos en archivo original:", na_rendimiento_original, "\n") # Mostrar NA originales

# Pregunta: ¿Se generaron nuevos NA durante la conversión?
na_generados_conversion <- sum(
  is.na(eva_limpia$rendimiento_t_ha)
) - na_rendimiento_original # Calcular diferencia

cat("NA generados durante conversión:", na_generados_conversion, "\n") # Mostrar NA generados

# Pregunta: ¿Cuántos rendimientos faltantes pueden recuperarse?
recalculables <- sum(
  is.na(eva_limpia$rendimiento_t_ha) &
    !is.na(eva_limpia$rendimiento_recalculado)
) # Contar registros recuperables

cat("Rendimientos recalculables:", recalculables, "\n") # Mostrar recuperables

# Pregunta: ¿Cuántos NA permanecerían después del recálculo?
na_residuales <- sum(
  is.na(eva_limpia$rendimiento_t_ha) &
    is.na(eva_limpia$rendimiento_recalculado)
) # Contar NA residuales

cat("NA residuales:", na_residuales, "\n") # Mostrar NA residuales

# Pregunta: ¿En qué filas aparecen los NA?
filas_na_rendimiento <- which(
  is.na(eva_limpia$rendimiento_t_ha)
) # Identificar filas con NA

cat("Total filas con NA:", length(filas_na_rendimiento), "\n") # Mostrar cantidad

head(filas_na_rendimiento, 100) # Mostrar primeras filas

# Pregunta: ¿Cómo se distribuyen los NA por año y qué porcentaje representan?
eva_limpia |>
  dplyr::group_by(anio) |>
  dplyr::summarise(
    total_registros = dplyr::n(),
    na_rendimiento = sum(is.na(rendimiento_t_ha)),
    porcentaje_na = round(
      100 * na_rendimiento / total_registros,
      2
    )
  ) |>
  dplyr::arrange(anio)

# Pregunta: ¿Cómo se distribuyen los NA por cultivo?
eva_limpia |>
  dplyr::filter(
    is.na(rendimiento_t_ha)
  ) |>
  dplyr::count(
    cultivo,
    sort = TRUE
  ) |>
  head(30)

# Pregunta: ¿Qué tan diferente es el rendimiento reportado frente al recalculado?
summary(
  eva_limpia$diferencia_rendimiento
) # Resumen diferencias

# Pregunta: ¿Cuáles son los registros con mayores inconsistencias de rendimiento?
eva_limpia |>
  dplyr::mutate(
    porcentaje_diferencia = round(
      100 * diferencia_rendimiento /
        rendimiento_t_ha,
      2
    )
  ) |>
  dplyr::arrange(
    dplyr::desc(diferencia_rendimiento)
  ) |>
  dplyr::select(
    departamento,
    municipio,
    cultivo,
    anio,
    area_cosechada_ha,
    produccion_t,
    rendimiento_t_ha,
    rendimiento_recalculado,
    diferencia_rendimiento,
    porcentaje_diferencia
  ) |>
  head(20)

# Pregunta: ¿Podemos exportar una auditoría detallada de rendimientos faltantes?
auditoria_na_rendimiento <- eva_limpia |>
  dplyr::filter(
    is.na(rendimiento_t_ha)
  ) |>
  dplyr::mutate(
    fila = dplyr::row_number(),
    rendimiento_recalculado = dplyr::if_else(
      area_cosechada_ha > 0,
      produccion_t / area_cosechada_ha,
      NA_real_
    ),
    recuperable = !is.na(rendimiento_recalculado)
  ) |>
  dplyr::select(
    fila,
    cod_depto,
    departamento,
    cod_mpio,
    municipio,
    cultivo,
    anio,
    area_sembrada_ha,
    area_cosechada_ha,
    produccion_t,
    rendimiento_t_ha,
    rendimiento_recalculado,
    recuperable
  ) # Construir auditoría completa

# Pregunta: ¿Podemos exportar una auditoría detallada de rendimientos faltantes?
auditoria_na_rendimiento <- eva_limpia |>
  dplyr::filter(
    is.na(rendimiento_t_ha)
  ) |>
  dplyr::mutate(
    fila = dplyr::row_number(),
    rendimiento_recalculado = dplyr::if_else(
      area_cosechada_ha > 0,
      produccion_t / area_cosechada_ha,
      NA_real_
    ),
    recuperable = !is.na(rendimiento_recalculado)
  ) |>
  dplyr::select(
    fila,
    cod_depto,
    departamento,
    cod_mpio,
    municipio,
    cultivo,
    anio,
    area_sembrada_ha,
    area_cosechada_ha,
    produccion_t,
    rendimiento_t_ha,
    rendimiento_recalculado,
    recuperable
  ) # Construir auditoría completa


# 7. VALIDACIÓN FINAL DE CALIDAD
# 7. VALIDACIÓN FINAL DE CALIDAD

# Pregunta: ¿Cuántos valores faltantes permanecen en la base final?

na_limpia <- colSums(
  is.na(eva_limpia)
) # Calcular NA finales

# Pregunta: ¿Existen registros duplicados exactos después de la limpieza?

duplicados_limpios <- sum(
  duplicated(eva_limpia)
) # Contar duplicados exactos

# Pregunta: ¿Existen registros duplicados según la llave natural de EVA?

llaves_duplicadas <- eva_limpia |>
  dplyr::count(
    cod_mpio,
    anio,
    cultivo,
    sistema_productivo,
    periodo
  ) |>
  dplyr::filter(
    n > 1
  ) |>
  nrow() # Contar duplicados según llave natural

# Pregunta: ¿Cuántos municipios únicos contiene la base?

municipios_unicos <- dplyr::n_distinct(
  eva_limpia$cod_mpio
) # Contar municipios únicos

# Pregunta: ¿Cuál es la cobertura temporal final?

anio_min <- min(
  eva_limpia$anio,
  na.rm = TRUE
) # Obtener año mínimo

anio_max <- max(
  eva_limpia$anio,
  na.rm = TRUE
) # Obtener año máximo

# Pregunta: ¿Persisten valores negativos físicamente imposibles?

negativos_area_sembrada <- sum(
  eva_limpia$area_sembrada_ha < 0,
  na.rm = TRUE
) # Contar áreas sembradas negativas

negativos_area_cosechada <- sum(
  eva_limpia$area_cosechada_ha < 0,
  na.rm = TRUE
) # Contar áreas cosechadas negativas

negativos_produccion <- sum(
  eva_limpia$produccion_t < 0,
  na.rm = TRUE
) # Contar producciones negativas


# 8. CERTIFICACIÓN DEL DATASET
# Pregunta: ¿La base cumple los criterios mínimos para pasar a Feature Engineering?
base_aprobada <- (
  duplicados_limpios == 0 &
    negativos_area_sembrada == 0 &
    negativos_area_cosechada == 0 &
    negativos_produccion == 0
) # Evaluar aprobación del dataset

# Pregunta: ¿Existen municipios sin identificador?
na_cod_mpio <- sum(
  is.na(eva_limpia$cod_mpio)
) # Contar municipios sin código

# Pregunta: ¿La cobertura temporal oficial se conserva?
cobertura_temporal_valida <- (
  min(eva_limpia$anio, na.rm = TRUE) == 2006 &
    max(eva_limpia$anio, na.rm = TRUE) == 2018
) # Validar cobertura temporal

# Pregunta: ¿La base cumple los criterios mínimos para pasar a Feature Engineering?
base_aprobada <- (
  duplicados_limpios == 0 &
    llaves_duplicadas == 0 &
    negativos_area_sembrada == 0 &
    negativos_area_cosechada == 0 &
    negativos_produccion == 0 &
    na_cod_mpio == 0 &
    cobertura_temporal_valida
) # Evaluar aprobación integral del dataset

# 8. CERTIFICACIÓN DEL DATASET

# 8. CERTIFICACIÓN DEL DATASET

# Pregunta: ¿Existen municipios sin identificador?

na_cod_mpio <- sum(
  is.na(eva_limpia$cod_mpio)
) # Contar municipios sin código

# Pregunta: ¿La cobertura temporal oficial se conserva?

cobertura_temporal_valida <- (
  anio_min == 2006 &
    anio_max == 2018
) # Validar cobertura temporal

# Pregunta: ¿La base cumple los criterios mínimos para continuar el pipeline?

base_aprobada <- (
  duplicados_limpios == 0 &
    llaves_duplicadas == 0 &
    negativos_area_sembrada == 0 &
    negativos_area_cosechada == 0 &
    negativos_produccion == 0 &
    na_cod_mpio == 0 &
    cobertura_temporal_valida
) # Evaluar aprobación integral del dataset

# 9. EXPORTACIÓN DE RESULTADOS
# Pregunta: ¿Puede exportarse la base limpia para las siguientes fases?
data.table::fwrite(
  eva_limpia,
  archivo_eva_limpia
) # Exportar EVA limpia

# Pregunta: ¿Puede exportarse el reporte de calidad del proceso?
auditoria <- data.frame(
  fecha_proceso = Sys.time(),
  filas_raw = nrow(eva_raw),
  columnas_raw = ncol(eva_raw),
  filas_limpias = nrow(eva_limpia),
  columnas_limpias = ncol(eva_limpia),
  municipios_unicos = municipios_unicos,
  anio_min = anio_min,
  anio_max = anio_max,
  duplicados_raw = duplicados_raw,
  duplicados_limpios = duplicados_limpios,
  negativos_area_sembrada = negativos_area_sembrada,
  negativos_area_cosechada = negativos_area_cosechada,
  negativos_produccion = negativos_produccion,
  base_aprobada = base_aprobada
) # Construir reporte final de calidad

data.table::fwrite(
  auditoria,
  archivo_auditoria
) # Exportar reporte de calidad

# 10. RESUMEN EJECUTIVO DEL PROCESO

# Pregunta: ¿Cuáles variables conservan valores faltantes después de la limpieza?

na_resumen <- data.frame(
  variable = names(eva_limpia),
  na_total = colSums(is.na(eva_limpia)),
  porcentaje_na = round(
    colSums(is.na(eva_limpia)) / nrow(eva_limpia) * 100,
    4
  )
) |>
  dplyr::filter(
    na_total > 0
  ) |>
  dplyr::arrange(
    dplyr::desc(na_total)
  ) # Construir resumen de variables con NA

# Pregunta: ¿Cuál es el tamaño final de la base después de la limpieza?

cat("\nPROCESO DE LIMPIEZA EVA FINALIZADO\n") # Mostrar encabezado final
cat("Filas finales:", nrow(eva_limpia), "\n") # Mostrar filas finales
cat("Columnas finales:", ncol(eva_limpia), "\n") # Mostrar columnas finales
cat("Municipios únicos:", municipios_unicos, "\n") # Mostrar municipios únicos
cat("Cobertura temporal:", anio_min, "-", anio_max, "\n") # Mostrar cobertura temporal

# Pregunta: ¿La estructura final cumple los criterios de calidad definidos?

cat("Duplicados exactos:", duplicados_limpios, "\n") # Mostrar duplicados exactos
cat("Duplicados llave natural:", llaves_duplicadas, "\n") # Mostrar duplicados por llave natural
cat("Municipios sin código:", na_cod_mpio, "\n") # Mostrar municipios sin identificador
cat("Cobertura temporal válida:", cobertura_temporal_valida, "\n") # Mostrar validación temporal
cat("Base aprobada:", base_aprobada, "\n") # Mostrar certificación final

# Pregunta: ¿Cuántas variables presentan valores faltantes?

cat("Variables con NA:", nrow(na_resumen), "\n") # Mostrar cantidad de variables con NA

if (nrow(na_resumen) > 0) {
  cat("\nDETALLE DE VARIABLES CON NA\n") # Mostrar encabezado de variables con NA
  
  for (i in seq_len(nrow(na_resumen))) {
    cat(
      na_resumen$variable[i],
      "- NA:",
      na_resumen$na_total[i],
      "- Porcentaje:",
      na_resumen$porcentaje_na[i],
      "%\n"
    ) # Mostrar detalle de faltantes por variable
  }
}

# Pregunta: ¿Dónde fueron almacenados los productos finales del proceso?

cat("Archivo generado:", archivo_eva_limpia, "\n") # Mostrar ruta de la base limpia
cat("Auditoría generada:", archivo_auditoria, "\n") # Mostrar ruta del reporte de auditoría

# Pregunta: ¿Qué cultivos explican los registros sin nombre científico?

cultivos_sin_nombre_cientifico <- eva_limpia |>
  dplyr::filter(
    is.na(nombre_cientifico)
  ) |>
  dplyr::count(
    cultivo,
    sort = TRUE
  ) # Identificar cultivos con ausencia de clasificación taxonómica

print(cultivos_sin_nombre_cientifico) # Mostrar cultivos con nombre científico faltante