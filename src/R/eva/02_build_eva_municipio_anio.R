# D:/Proyectos_IA/proyecto-gnn-agricola/eva/02_build_eva_municipio_anio.R

# 02_build_eva_municipio_anio.R
# Ejecutar configuración global
source(here::here("config", "00_packages.R")) # Cargar paquetes
source(here::here("config", "01_paths.R")) # Cargar rutas
source(here::here("config", "02_global_parameters.R")) # Cargar parámetros globales

# Pregunta: ¿Cuáles son los archivos de entrada y salida?
archivo_eva_limpia <- file.path(
  ruta_processed_eva,
  "eva_limpia.csv"
); archivo_eva_limpia # Archivo EVA limpio

archivo_eva_municipio_anio <- file.path(
  ruta_processed_eva,
  "eva_municipio_anio.csv"
); archivo_eva_municipio_anio # Archivo municipio-año

archivo_auditoria_rendimientos <- file.path(
  ruta_outputs_eva,
  "auditoria_rendimientos_extremos.csv"
); archivo_auditoria_rendimientos # Auditoría rendimientos extremos

# Pregunta: ¿Puede cargarse correctamente la base EVA limpia?
eva_limpia <- data.table::fread(
  archivo_eva_limpia
); eva_limpia # Cargar EVA limpia

# Pregunta: ¿Puede construirse una observación única por municipio-año?
eva_municipio_anio <- eva_limpia |>
  dplyr::group_by(
    cod_depto,
    departamento,
    cod_mpio,
    municipio,
    anio
  ) |>
  dplyr::summarise(
    
    area_sembrada_total = sum(
      area_sembrada_ha,
      na.rm = TRUE
    ), # Área sembrada total
    
    area_cosechada_total = sum(
      area_cosechada_ha,
      na.rm = TRUE
    ), # Área cosechada total
    
    produccion_total = sum(
      produccion_t,
      na.rm = TRUE
    ), # Producción total
    
    n_cultivos = dplyr::n_distinct(
      cultivo
    ), # Número de cultivos
    
    n_grupos_cultivo = dplyr::n_distinct(
      grupo_cultivo
    ), # Número de grupos
    
    porcentaje_permanentes = mean(
      ciclo_cultivo == "PERMANENTE",
      na.rm = TRUE
    ), # Porcentaje cultivos permanentes
    
    .groups = "drop"
    
  ) |>
  
  dplyr::mutate(
    
    rendimiento_promedio = dplyr::if_else(
      area_cosechada_total > 0,
      produccion_total / area_cosechada_total,
      NA_real_
    ), # Rendimiento reconstruido
    
    tasa_cosecha_promedio = dplyr::if_else(
      area_sembrada_total > 0,
      area_cosechada_total / area_sembrada_total,
      NA_real_
    ), # Tasa de cosecha
    
    log_produccion_total = log1p(
      produccion_total
    ), # Log producción
    
    log_area_sembrada_total = log1p(
      area_sembrada_total
    ) # Log área sembrada
    
  ); eva_municipio_anio

# Pregunta: ¿Existen rendimientos extremos?
auditoria_rendimientos_extremos <- eva_municipio_anio |>
  dplyr::filter(
    rendimiento_promedio > 100
  ) |>
  dplyr::arrange(
    dplyr::desc(
      rendimiento_promedio
    )
  ); auditoria_rendimientos_extremos # Identificar extremos

# Pregunta: ¿Existen registros duplicados municipio-año?
duplicados_municipio_anio <- eva_municipio_anio |>
  dplyr::count(
    cod_mpio,
    anio
  ) |>
  dplyr::filter(
    n > 1
  ); duplicados_municipio_anio # Buscar duplicados

# Pregunta: ¿Puede exportarse el panel final?
data.table::fwrite(
  eva_municipio_anio,
  archivo_eva_municipio_anio
); eva_municipio_anio # Exportar panel municipio-año

data.table::fwrite(
  auditoria_rendimientos_extremos,
  archivo_auditoria_rendimientos
); auditoria_rendimientos_extremos # Exportar auditoría


# Pregunta: ¿Cuál es el resultado final de la construcción del panel municipio-año?
cat("\nPANEL MUNICIPIO-AÑO GENERADO CORRECTAMENTE\n") # Mostrar encabezado final

# Pregunta: ¿Cuál fue el tamaño de la base de entrada utilizada?
cat("Filas EVA limpia:", nrow(eva_limpia), "\n") # Mostrar filas de entrada
cat("Columnas EVA limpia:", ncol(eva_limpia), "\n") # Mostrar columnas de entrada

# Pregunta: ¿Cuáles son las dimensiones y cobertura del panel construido?
cat("Registros municipio-año:", nrow(eva_municipio_anio), "\n") # Mostrar registros finales
cat("Municipios únicos:", dplyr::n_distinct(eva_municipio_anio$cod_mpio), "\n") # Mostrar municipios únicos
cat("Cobertura temporal:", min(eva_municipio_anio$anio, na.rm = TRUE), "-", max(eva_municipio_anio$anio, na.rm = TRUE), "\n") # Mostrar cobertura temporal

# Pregunta: ¿Cómo se comporta el rendimiento agrícola agregado?

cat("\nAUDITORÍA RENDIMIENTO MUNICIPAL\n") # Mostrar encabezado de auditoría

print(summary(eva_municipio_anio$rendimiento_promedio)) # Resumir rendimiento promedio

cat("Rendimientos mayores a 100:", nrow(auditoria_rendimientos_extremos), "\n") # Mostrar rendimientos extremos

# Pregunta: ¿Existen inconsistencias en las tasas de cosecha?

cat("Tasas mayores a 1:", sum(eva_municipio_anio$tasa_cosecha_promedio > 1, na.rm = TRUE), "\n") # Detectar tasas imposibles

# Pregunta: ¿La llave municipio-año conserva unicidad?

cat("Duplicados municipio-año:", nrow(duplicados_municipio_anio), "\n") # Mostrar duplicados

# Pregunta: ¿Cuál es la estructura final certificada del panel?

cat("Filas finales:", nrow(eva_municipio_anio), "\n") # Mostrar filas finales
cat("Columnas finales:", ncol(eva_municipio_anio), "\n") # Mostrar columnas finales

# Pregunta: ¿Dónde fueron almacenados los productos generados?

cat("Archivo generado:", archivo_eva_municipio_anio, "\n") # Mostrar archivo final
cat("Auditoría generada:", archivo_auditoria_rendimientos, "\n") # Mostrar archivo de auditoría


setdiff(
  names(eva_municipio_anio),
  names(eva_limpia)
) # Variables creadas durante la agregación

# Pregunta: ¿El panel está listo para continuar en el pipeline?

cat("Estado panel: APROBADO PARA AUDITORÍA FINAL\n") # Confirmar construcción
