# ============================================================
# SCRIPT: run_03_pdet_limpio.R
# OBJETIVO:
# Limpiar la base de municipios PDET y dejarla lista para unir
# con eva_municipio_anio.
# ============================================================

source("R/00_packages.R")
source("R/06_export.R")

# ============================================================
# 1. Función para leer PDET
# ============================================================

leer_pdet <- function(path) {
  
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

# ============================================================
# 2. Leer base PDET
# ============================================================

pdet_raw <- leer_pdet(
  "data_raw/Municipios_PDET_20260413.csv"
)

print(names(pdet_raw))

# ============================================================
# 3. Función para normalizar texto
# ============================================================

normalizar_texto <- function(x) {
  x %>%
    stringr::str_replace_all("\u0089", "E") %>%
    stringr::str_replace_all("\u008a", "E") %>%
    stringr::str_replace_all("\u008d", "I") %>%
    stringr::str_replace_all("\u0081", "A") %>%
    stringr::str_replace_all("\u0091", "N") %>%
    stringr::str_replace_all("\u0093", "O") %>%
    stringr::str_replace_all("\u009a", "U") %>%
    stringr::str_replace_all("\u009c", "N") %>%
    stringr::str_replace_all("\u00D1", "N") %>%
    stringr::str_replace_all("\u00F1", "N") %>%
    stringr::str_replace_all("\u00C1", "A") %>%
    stringr::str_replace_all("\u00C9", "E") %>%
    stringr::str_replace_all("\u00CD", "I") %>%
    stringr::str_replace_all("\u00D3", "O") %>%
    stringr::str_replace_all("\u00DA", "U") %>%
    stringi::stri_trans_general("Latin-ASCII") %>%
    stringr::str_squish() %>%
    stringr::str_to_upper()
}

# ============================================================
# 4. Limpiar y seleccionar variables PDET
# ============================================================

pdet_limpio <- pdet_raw %>%
  janitor::clean_names() %>%
  transmute(
    codigo_subregion_pdet = as.integer(codigo_subregion),
    nombre_subregion_pdet = normalizar_texto(nombre_subregion),
    codigo_departamento = stringr::str_pad(
      codigo_departamento,
      width = 2,
      side = "left",
      pad = "0"
    ),
    nombre_departamento_pdet = normalizar_texto(nombre_departamento),
    codigo_municipio = stringr::str_pad(
      codigo_municipio,
      width = 5,
      side = "left",
      pad = "0"
    ),
    nombre_municipio_pdet = normalizar_texto(nombre_municipio),
    es_pdet = 1
  ) %>%
  distinct()

# ============================================================
# 4.1 Correcciones específicas de texto
# ============================================================

pdet_limpio <- pdet_limpio %>%
  mutate(
    # Subregiones
    nombre_subregion_pdet = stringr::str_replace_all(nombre_subregion_pdet, "PATAIA", "PATIA"),
    nombre_subregion_pdet = stringr::str_replace_all(nombre_subregion_pdet, "CAGUAAN", "CAGUAN"),
    nombre_subregion_pdet = stringr::str_replace_all(nombre_subregion_pdet, "MARAIA", "MARIA"),
    nombre_subregion_pdet = stringr::str_replace_all(nombre_subregion_pdet, "PERIJAA", "PERIJA"),
    nombre_subregion_pdet = stringr::str_replace_all(nombre_subregion_pdet, "CHOCAO", "CHOCO"),
    nombre_subregion_pdet = stringr::str_replace_all(nombre_subregion_pdet, "PACAIFICO", "PACIFICO"),
    nombre_subregion_pdet = stringr::str_replace_all(nombre_subregion_pdet, "URABAA", "URABA"),
    nombre_subregion_pdet = stringr::str_replace_all(nombre_subregion_pdet, "BOLAIVAR", "BOLIVAR"),
    nombre_subregion_pdet = stringr::str_replace_all(nombre_subregion_pdet, "CAORDOBA", "CORDOBA"),
    nombre_subregion_pdet = stringr::str_replace_all(nombre_subregion_pdet, "CAQUETEANO", "CAQUETENO"),
    nombre_subregion_pdet = stringr::str_replace_all(nombre_subregion_pdet, "NARIANENSE", "NARINENSE"),
    nombre_subregion_pdet = stringr::str_replace_all(nombre_subregion_pdet, "ANTIOQUEANO", "ANTIOQUENO"),
    
    # Departamentos
    nombre_departamento_pdet = stringr::str_replace_all(nombre_departamento_pdet, "NARIANO", "NARINO"),
    nombre_departamento_pdet = stringr::str_replace_all(nombre_departamento_pdet, "CHOCAO", "CHOCO"),
    nombre_departamento_pdet = stringr::str_replace_all(nombre_departamento_pdet, "CAQUETAA", "CAQUETA"),
    nombre_departamento_pdet = stringr::str_replace_all(nombre_departamento_pdet, "BOLAIVAR", "BOLIVAR"),
    nombre_departamento_pdet = stringr::str_replace_all(nombre_departamento_pdet, "CAORDOBA", "CORDOBA"),
    
    # Municipios
    nombre_municipio_pdet = stringr::str_replace_all(nombre_municipio_pdet, "BELAEN DE LOS ANDAQUIES", "BELEN DE LOS ANDAQUIES"),
    nombre_municipio_pdet = stringr::str_replace_all(nombre_municipio_pdet, "CIAENAGA", "CIENAGA"),
    nombre_municipio_pdet = stringr::str_replace_all(nombre_municipio_pdet, "MAGANI", "MAGANGUE"),
    nombre_municipio_pdet = stringr::str_replace_all(nombre_municipio_pdet, "SAN JOSAE DEL FRAGUA", "SAN JOSE DEL FRAGUA"),
    nombre_municipio_pdet = stringr::str_replace_all(nombre_municipio_pdet, "SAN JOSAE DEL GUAVIARE", "SAN JOSE DEL GUAVIARE"),
    nombre_municipio_pdet = stringr::str_replace_all(nombre_municipio_pdet, "TIBAU", "TIBU"),
    nombre_municipio_pdet = stringr::str_replace_all(nombre_municipio_pdet, "TOLAU VIEJO", "TOLU VIEJO"),
    nombre_municipio_pdet = stringr::str_replace_all(nombre_municipio_pdet, "PIENDAMAO", "PIENDAMO"),
    nombre_municipio_pdet = stringr::str_replace_all(nombre_municipio_pdet, "JAMBALAO", "JAMBALO")
  )

# ============================================================
# 5. Validaciones
# ============================================================

glimpse(pdet_limpio)

cat("Total municipios PDET:", nrow(pdet_limpio), "\n")

duplicados_pdet <- pdet_limpio %>%
  count(codigo_municipio) %>%
  filter(n > 1)

cat("Municipios PDET duplicados:", nrow(duplicados_pdet), "\n")

missing_pdet <- pdet_limpio %>%
  summarise(
    missing_codigo_subregion_pdet = sum(is.na(codigo_subregion_pdet)),
    missing_nombre_subregion_pdet = sum(is.na(nombre_subregion_pdet)),
    missing_codigo_departamento = sum(is.na(codigo_departamento)),
    missing_nombre_departamento_pdet = sum(is.na(nombre_departamento_pdet)),
    missing_codigo_municipio = sum(is.na(codigo_municipio)),
    missing_nombre_municipio_pdet = sum(is.na(nombre_municipio_pdet)),
    missing_es_pdet = sum(is.na(es_pdet))
  )

print(missing_pdet)

# Validar distribución de subregiones
pdet_limpio %>%
  count(nombre_subregion_pdet, sort = TRUE) %>%
  print(n = Inf)

# Validar nombres de departamentos
pdet_limpio %>%
  count(nombre_departamento_pdet, sort = TRUE) %>%
  print(n = Inf)

# Validar nombres de municipios con caracteres raros
pdet_limpio %>%
  count(nombre_municipio_pdet, sort = TRUE) %>%
  filter(stringr::str_detect(nombre_municipio_pdet, "[^A-Z ]")) %>%
  print(n = Inf)

# ============================================================
# 6. Exportar base PDET limpia
# ============================================================

exportar_csv(
  pdet_limpio,
  "data_processed/pdet_limpio.csv"
)

writexl::write_xlsx(
  pdet_limpio,
  "data_processed/pdet_limpio.xlsx"
)

# ------------------------------------------------------------
# CONCLUSIÓN 1: CALIDAD E INTEGRIDAD DE LA BASE PDET
# ------------------------------------------------------------
# La base de municipios PDET presenta una estructura completa
# y consistente, con 170 observaciones únicas correspondientes
# a municipios priorizados. No se identifican valores faltantes
# en las variables clave ni duplicados en el código de municipio,
# lo que garantiza la integridad de la llave de unión con otras
# fuentes de datos. Esto confirma que la base es apta para ser
# utilizada como dimensión territorial en el análisis.

# ------------------------------------------------------------
# CONCLUSIÓN 2: NORMALIZACIÓN Y LIMPIEZA DE TEXTO
# ------------------------------------------------------------
# Se identificaron errores de codificación en las variables
# categóricas (especialmente nombres de subregión), los cuales
# fueron corregidos mediante un proceso de normalización y
# reemplazos específicos. Este paso es crítico, ya que evita
# inconsistencias en agregaciones, joins y análisis posteriores,
# asegurando la coherencia semántica de la información territorial.

# ------------------------------------------------------------
# CONCLUSIÓN 3: ESTRUCTURA TERRITORIAL PDET
# ------------------------------------------------------------
# La base contiene 16 subregiones PDET, con una distribución
# heterogénea de municipios. Algunas subregiones concentran
# mayor número de municipios (e.g., Alto Patía, Caguán),
# mientras que otras presentan menor representación. Esta
# estructura es fundamental para modelar dependencias espaciales
# y construir relaciones en modelos basados en grafos.

# ------------------------------------------------------------
# CONCLUSIÓN 4: VARIABLE BINARIA PDET
# ------------------------------------------------------------
# La inclusión de la variable es_pdet permite identificar de
# manera explícita los municipios pertenecientes a zonas PDET.
# Esta variable será clave en modelos predictivos para evaluar
# diferencias estructurales, brechas territoriales y efectos
# diferenciados en el rendimiento agrícola.

# ------------------------------------------------------------
# CONCLUSIÓN 5: PREPARACIÓN PARA INTEGRACIÓN MULTIFUENTE
# ------------------------------------------------------------
# La base ha sido alineada en términos de formato y codificación
# con la base EVA (códigos municipales en formato estándar de
# 5 dígitos), lo que permite una integración directa mediante
# operaciones de join. Esto habilita la construcción de un
# dataset enriquecido que combina variables productivas y
# características territoriales.

# ------------------------------------------------------------
# CONCLUSIÓN 6: VALOR PARA MODELADO ESPACIAL Y GNN
# ------------------------------------------------------------
# La información de subregión PDET constituye una dimensión
# espacial estructurada que puede ser utilizada para definir
# aristas en modelos de grafos. Esto permite capturar relaciones
# territoriales entre municipios, lo cual es esencial en enfoques
# de modelamiento espaciotemporal y redes neuronales de grafos.

# ------------------------------------------------------------
# OBSERVACIÓN CRÍTICA
# ------------------------------------------------------------
# La variable PDET es de tipo categórico/estructural y no
# representa un fenómeno medido directamente, sino una
# clasificación administrativa. Por lo tanto, su uso en modelos
# debe interpretarse como variable contextual y no causal.

# Dataset limpio
# in errores estructurales
# Sin NA
# Sin duplicados
# Compatible con EVA
# Listo para merge
# Listo para GNN

# ============================================================
# 6. Exportar base PDET limpia
# ============================================================

exportar_csv(
  pdet_limpio,
  "data_processed/pdet_limpio.csv"
)

writexl::write_xlsx(
  pdet_limpio,
  "data_processed/pdet_limpio.xlsx"
)


# ============================================================
# CONCLUSIONES PDET (VERSIÓN ANALÍTICA)
# ============================================================

# 1. Integridad estructural de la base
# La base PDET contiene 170 municipios únicos sin duplicados
# ni valores faltantes en variables clave. Esto garantiza que
# la llave codigo_municipio es completamente confiable para
# procesos de integración con otras bases.
# :contentReference[oaicite:0]{index=0}

# 2. Naturaleza del dataset
# A diferencia de EVA, esta base no es observacional sino
# categórica-administrativa. Representa una clasificación
# territorial definida por política pública, no una medición
# directa de variables económicas o productivas.

# 3. Calidad de normalización
# El proceso de limpieza logró:
# - eliminar problemas de codificación (Latin1)
# - unificar nombres (mayúsculas, sin tildes)
# - corregir errores semánticos específicos
# Resultado: coherencia total en variables categóricas.

# 4. Estructura territorial PDET
# La base contiene 16 subregiones con distribución desigual
# de municipios (entre 4 y 24). Esta heterogeneidad implica
# que los efectos PDET no son homogéneos territorialmente.

# 5. Representación geográfica
# Los municipios PDET están concentrados en zonas específicas
# del país (conflicto, ruralidad, baja infraestructura).
# Esto introduce un sesgo estructural importante que debe
# ser considerado en cualquier modelo.

# 6. Variable PDET como señal estructural
# La variable es_pdet no mide un fenómeno, sino que indica
# pertenencia a una política pública. Por tanto:
# - no es causal por sí misma
# - actúa como proxy de condiciones estructurales

# 7. Implicación para modelado
# PDET no debe interpretarse como variable explicativa directa,
# sino como variable de contexto o estratificación.
# Su uso adecuado es:
# - interacción con variables productivas
# - segmentación territorial
# - análisis de heterogeneidad

# 8. Riesgo metodológico
# Usar PDET como predictor directo puede inducir:
# - sesgo de política pública
# - correlaciones espurias
# - interpretación causal incorrecta
# Esto debe evitarse explícitamente.

# 9. Valor analítico real
# El valor de esta base no está en su complejidad,
# sino en su capacidad de:
# - diferenciar territorios estructuralmente distintos
# - introducir dimensión institucional al modelo

# 10. Preparación para integración
# La base está completamente alineada con EVA:
# - códigos municipales normalizados (5 dígitos)
# - nombres consistentes
# Esto permite integración directa sin pérdidas.

# 11. Uso en GNN
# La información PDET puede incorporarse como:
# - feature de nodo (es_pdet)
# - feature categórica (subregión)
# - o para análisis de clusters territoriales

# 12. Limitación clave
# La base no contiene:
# - intensidad de intervención PDET
# - indicadores socioeconómicos asociados
# - evolución temporal
# Por lo tanto, su capacidad explicativa es limitada.

# 13. Conclusión principal
# PDET no explica el rendimiento agrícola,
# pero sí identifica territorios con condiciones
# estructurales diferenciadas.

# 14. Rol dentro del pipeline
# Esta base agrega una dimensión institucional
# al modelo, complementando:
# - variables productivas (EVA)
# - variables físicas (irrigación)

# 15. Conclusión final
# La integración de PDET permite modelar no solo
# la producción agrícola, sino el contexto territorial
# en el que esta ocurre.

# ============================================================
# FIN CONCLUSIONES PDET
# ============================================================