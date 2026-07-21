# 10_auditoria_benchmark.R

# BLOQUE 1. Importaciones ---------------------------------------------------
## Objetivo: Importar las librerías necesarias para ejecutar la Auditoría
# Científica del Dataset Ganador utilizado durante el Benchmark.
### Producto:
# - Librerías cargadas correctamente.
### Responde:
# ¿Las dependencias necesarias para ejecutar la Auditoría Científica del
# Dataset Ganador fueron cargadas correctamente?

# Librerías del sistema
library(here) # Gestión de rutas del proyecto
library(fs) # Operaciones sobre archivos

# Manipulación de datos
library(dplyr) # Manipulación de datos
library(tidyr) # Transformación de datos
library(stringr) # Manipulación de texto
library(purrr) # Programación funcional
library(tibble) # Construcción de tablas

# Lectura de archivos
library(arrow) # Lectura de archivos Parquet

# Estadística
library(psych) # Estadísticos descriptivos
library(caret) # Detección de variables problemáticas
library(car) # Factor de inflación de la varianza (VIF)

# Visualización
library(ggplot2) # Gráficos científicos
library(corrplot) # Matriz de correlaciones

# Machine Learning
library(randomForest) # Importancia de variables

# Configuración
options(
  scipen = 999,
  digits = 6
) # Configuración numérica

set.seed(5477976) # Semilla oficial del proyecto

cat("\n")
cat(rep("=", 80), sep = "")
cat("\n")
cat("AUDITORÍA CIENTÍFICA DEL DATASET GANADOR\n")
cat(rep("=", 80), sep = "")
cat("\n")
cat("Bloque 1 finalizado correctamente.\n")

# BLOQUE 2. Carga del Dataset ----------------------------------------------
## Objetivo: Cargar y verificar el Dataset Ganador que será utilizado para
# la Auditoría Científica del Benchmark.
### Producto:
# - Dataset cargado correctamente.
# - Dataset preparado para auditoría.
### Responde:
# ¿El Dataset Ganador fue cargado correctamente y cumple las condiciones
# mínimas para iniciar la auditoría científica?

# 2.1. Ruta oficial del Dataset --------------------------------------------
## Objetivo: Definir la ruta oficial del Dataset Ganador.

ruta_dataset <- here(
  "data",
  "processed",
  "r",
  "dataset",
  "dataset_gnn_certificado.parquet"
) # Ruta oficial del dataset

# 2.2. Verificación de existencia ------------------------------------------
## Objetivo: Verificar que el archivo existe.

if (!file.exists(ruta_dataset)) {
  
  stop(
    "No existe el Dataset Ganador en la ruta especificada."
  )
  
}

cat("Dataset localizado correctamente.\n")

# 2.3. Carga del Dataset ----------------------------------------------------
## Objetivo: Cargar el Dataset Ganador.

dataset_gnn <- arrow::read_parquet(
  ruta_dataset
) # Dataset oficial

cat("Dataset cargado correctamente.\n")

# 2.4. Validación del Dataset ----------------------------------------------
## Objetivo: Verificar que el Dataset contiene observaciones y variables.

if (nrow(dataset_gnn) == 0) {
  
  stop(
    "El Dataset no contiene observaciones."
  )
  
}

if (ncol(dataset_gnn) == 0) {
  
  stop(
    "El Dataset no contiene variables."
  )
  
}

cat("Dimensiones verificadas correctamente.\n")

# 2.5. Validación de la variable objetivo ----------------------------------
## Objetivo: Verificar que existe la variable objetivo oficial.

if (!"log_rendimiento" %in% names(dataset_gnn)) {
  
  stop(
    "No existe la variable objetivo 'log_rendimiento'."
  )
  
}

cat("Variable objetivo encontrada correctamente.\n")

# 2.6. Preparación del Dataset para Auditoría -------------------------------
## Objetivo: Excluir variables que no participan en la auditoría estadística.

dataset_auditoria <- dataset_gnn |>
  dplyr::select(
    -any_of("geometry")
  ) # Dataset utilizado durante toda la auditoría

cat("Dataset preparado para la auditoría científica.\n")

# 2.7. Registro de información ---------------------------------------------
## Objetivo: Registrar la estructura general del Dataset.

cat("\n")
cat(rep("-", 80), sep = "")
cat("\n")
cat("INFORMACIÓN GENERAL DEL DATASET\n")
cat(rep("-", 80), sep = "")
cat("\n")

cat(sprintf(
  "Observaciones : %s\n",
  format(nrow(dataset_auditoria), big.mark = ",")
))

cat(sprintf(
  "Variables     : %s\n",
  format(ncol(dataset_auditoria), big.mark = ",")
))

cat("\nPrimeras variables:\n")

print(
  
  names(dataset_auditoria)[
    
    1:min(10, ncol(dataset_auditoria))
    
  ]
  
)

cat("\nVariable objetivo : log_rendimiento\n")

cat("\nBloque 2 finalizado correctamente.\n")

# BLOQUE 3. Auditoría General del Dataset ----------------------------------
## Objetivo: Auditar la estructura, tipos de datos, completitud y calidad
# general del Dataset Ganador antes de realizar los análisis predictivos.
### Producto:
# - Resumen estructural del dataset.
# - Auditoría de calidad de datos.
### Responde:
# ¿El Dataset Ganador presenta una estructura consistente y una calidad
# adecuada para iniciar la Auditoría Científica del Benchmark?

# 3.1. Estructura general --------------------------------------------------
## Objetivo: Mostrar la estructura general del Dataset.

cat("\n")
cat(rep("=", 80), sep = "")
cat("\n")
cat("ESTRUCTURA GENERAL DEL DATASET\n")
cat(rep("=", 80), sep = "")
cat("\n")

str(dataset_auditoria)

# 3.2. Tipos de variables --------------------------------------------------
## Objetivo: Resumir los tipos de datos presentes.

tipos_variables <- tibble(
  
  variable = names(dataset_auditoria),
  
  clase = sapply(
    dataset_auditoria,
    function(x) class(x)[1]
  ),
  
  tipo = sapply(
    dataset_auditoria,
    typeof
  )
  
)

print(tipos_variables)

# 3.3. Valores faltantes ---------------------------------------------------
## Objetivo: Cuantificar los valores faltantes por variable.

auditoria_na <- tibble(
  
  variable = names(dataset_auditoria),
  
  n_na = sapply(
    
    dataset_auditoria,
    
    function(x) sum(is.na(x))
    
  )
  
) |>
  
  mutate(
    
    pct_na = round(
      
      100 * n_na / nrow(dataset_auditoria),
      
      4
      
    )
    
  ) |>
  
  arrange(
    
    desc(n_na)
    
  )

print(auditoria_na)

# 3.4. Variables numéricas -------------------------------------------------
## Objetivo: Seleccionar únicamente las variables numéricas.

variables_numericas <- dataset_auditoria |>
  
  dplyr::select(
    
    where(is.numeric)
    
  )

# 3.5. Variables constantes ------------------------------------------------
## Objetivo: Detectar variables sin variabilidad.

variables_constantes <- names(variables_numericas)[
  
  sapply(
    
    variables_numericas,
    
    function(x){
      
      length(
        
        unique(
          
          x[!is.na(x)]
          
        )
        
      ) <= 1
      
    }
    
  )
  
]

cat("\nVariables constantes:\n")

if(length(variables_constantes) == 0){
  
  cat("No se encontraron variables constantes.\n")
  
}else{
  
  print(variables_constantes)
  
}

# 3.6. Variables con baja variabilidad -------------------------------------
## Objetivo: Detectar predictores con varianza cercana a cero.

nzv <- caret::nearZeroVar(
  
  variables_numericas,
  
  saveMetrics = TRUE
  
)

nzv <- tibble::as_tibble(
  
  nzv,
  
  rownames = "variable"
  
)

print(
  
  nzv |>
    
    dplyr::filter(
      nzv
    )
  
)

# 3.7. Registros duplicados ------------------------------------------------
## Objetivo: Verificar registros duplicados.

duplicados <- sum(
  
  duplicated(
    
    dataset_auditoria
    
  )
  
)

cat("\n")

cat(sprintf(
  
  "Registros duplicados : %s\n",
  
  format(
    
    duplicados,
    
    big.mark = ","
    
  )
  
))

# 3.8. Resumen de la auditoría ---------------------------------------------
## Objetivo: Consolidar la auditoría estructural.

cat("\n")

cat(rep("-", 80), sep = "")

cat("\n")

cat("RESUMEN DE LA AUDITORÍA\n")

cat(rep("-", 80), sep = "")

cat("\n")

cat(sprintf(
  
  "Observaciones               : %s\n",
  
  format(
    
    nrow(dataset_auditoria),
    
    big.mark = ","
    
  )
  
))

cat(sprintf(
  
  "Variables                   : %s\n",
  
  format(
    
    ncol(dataset_auditoria),
    
    big.mark = ","
    
  )
  
))

cat(sprintf(
  
  "Variables numéricas         : %s\n",
  
  ncol(variables_numericas)
  
))

cat(sprintf(
  
  "Variables con NA            : %s\n",
  
  sum(
    
    auditoria_na$n_na > 0
    
  )
  
))

cat(sprintf(
  
  "Variables constantes        : %s\n",
  
  length(
    
    variables_constantes
    
  )
  
))

cat(sprintf(
  
  "Variables Near Zero Var     : %s\n",
  
  sum(
    
    nzv$nzv
    
  )
  
))

cat(sprintf(
  
  "Registros duplicados        : %s\n",
  
  duplicados
  
))

cat("\nBloque 3 finalizado correctamente.\n")

# BLOQUE 4. Auditoría de la Variable Objetivo -------------------------------
## Objetivo: Analizar la distribución, variabilidad y calidad de la variable
# objetivo utilizada durante el Benchmark Científico.
### Entradas:
# - dataset_auditoria
### Producto:
# - Estadísticos descriptivos de la variable objetivo.
# - Diagnóstico de calidad.
### Responde:
# ¿La variable objetivo presenta una distribución y calidad adecuadas para
# el modelado predictivo?

# 4.1. Verificación de existencia ------------------------------------------
## Objetivo: Verificar que la variable objetivo existe.

if(!"log_rendimiento" %in% names(dataset_auditoria)){
  
  stop(
    "La variable objetivo 'log_rendimiento' no existe."
  )
  
}

# 4.2. Extracción ----------------------------------------------------------
## Objetivo: Extraer la variable objetivo.

target <- dataset_auditoria$log_rendimiento # Variable objetivo

# 4.3. Estadísticos descriptivos -------------------------------------------
## Objetivo: Calcular estadísticas descriptivas.

estadisticas_target <- tibble(
  
  observaciones = length(target),
  
  minimo = min(target, na.rm = TRUE),
  
  percentil_25 = quantile(target, 0.25, na.rm = TRUE),
  
  mediana = median(target, na.rm = TRUE),
  
  media = mean(target, na.rm = TRUE),
  
  percentil_75 = quantile(target, 0.75, na.rm = TRUE),
  
  maximo = max(target, na.rm = TRUE),
  
  desviacion = sd(target, na.rm = TRUE),
  
  varianza = var(target, na.rm = TRUE),
  
  coeficiente_variacion = sd(target, na.rm = TRUE) /
    abs(mean(target, na.rm = TRUE)),
  
  asimetria = e1071::skewness(target, na.rm = TRUE),
  
  curtosis = e1071::kurtosis(target, na.rm = TRUE)
  
)

cat("\n")
cat(rep("=",80),sep="")
cat("\n")
cat("ESTADÍSTICOS DE LA VARIABLE OBJETIVO\n")
cat(rep("=",80),sep="")
cat("\n")

print(estadisticas_target)

# 4.4. Calidad de la variable objetivo -------------------------------------
## Objetivo: Detectar valores especiales.

cat("\n")
cat(rep("-",80),sep="")
cat("\n")
cat("CALIDAD DE LA VARIABLE OBJETIVO\n")
cat(rep("-",80),sep="")
cat("\n")

cat(sprintf("Valores NA        : %s\n", sum(is.na(target))))
cat(sprintf("Valores NaN       : %s\n", sum(is.nan(target))))
cat(sprintf("Valores Inf       : %s\n", sum(is.infinite(target))))
cat(sprintf("Valores únicos    : %s\n", dplyr::n_distinct(target)))

# 4.5. Valores atípicos ----------------------------------------------------
## Objetivo: Detectar valores extremos mediante IQR.

Q1 <- quantile(target,0.25,na.rm=TRUE)

Q3 <- quantile(target,0.75,na.rm=TRUE)

IQR_target <- IQR(target,na.rm=TRUE)

limite_inferior <- Q1 - 1.5 * IQR_target

limite_superior <- Q3 + 1.5 * IQR_target

outliers <- sum(
  
  target < limite_inferior |
    
    target > limite_superior
  
)

cat(sprintf("Valores atípicos  : %s\n", outliers))

# 4.6. Prueba de normalidad -----------------------------------------------
## Objetivo: Evaluar la normalidad de la variable objetivo.

# install.packages("nortest") # Instalar 1 sola vez
library(nortest) # Pruebas de normalidad

if(length(target) <= 5000){
  
  prueba_normalidad <- shapiro.test(target)
  
}else{
  
  prueba_normalidad <- nortest::ad.test(target)
  
}

cat("\n")
cat(rep("-",80),sep="")
cat("\n")
cat("NORMALIDAD\n")
cat(rep("-",80),sep="")
cat("\n")

print(prueba_normalidad)

# 4.7. Histograma ----------------------------------------------------------

ggplot(
  
  dataset_auditoria,
  
  aes(x = log_rendimiento)
  
)+
  
  geom_histogram(
    
    bins = 30,
    
    fill = "steelblue",
    
    color = "black"
    
  )+
  
  theme_minimal()

# 4.8. Densidad ------------------------------------------------------------

ggplot(
  
  dataset_auditoria,
  
  aes(x = log_rendimiento)
  
)+
  
  geom_density(
    
    fill = "steelblue",
    
    alpha = 0.40
    
  )+
  
  theme_minimal()

# 4.9. QQ Plot -------------------------------------------------------------

ggplot(
  
  dataset_auditoria,
  
  aes(sample = log_rendimiento)
  
)+
  
  stat_qq()+
  
  stat_qq_line(
    
    colour = "red"
    
  )+
  
  theme_minimal()

# 4.10. Boxplot ------------------------------------------------------------

ggplot(
  
  dataset_auditoria,
  
  aes(y = log_rendimiento)
  
)+
  
  geom_boxplot(
    
    fill = "tomato"
    
  )+
  
  theme_minimal()

# 4.11. Dictamen -----------------------------------------------------------

cat("\n")
cat(rep("=",80),sep="")
cat("\n")
cat("DICTAMEN DE LA VARIABLE OBJETIVO\n")
cat(rep("=",80),sep="")
cat("\n")

cat(sprintf("Observaciones             : %s\n",format(length(target),big.mark=",")))

cat(sprintf("Media                     : %.6f\n",mean(target)))

cat(sprintf("Desviación estándar       : %.6f\n",sd(target)))

cat(sprintf("Coeficiente variación     : %.6f\n",sd(target)/mean(target)))

cat(sprintf("Valores atípicos          : %s\n",outliers))

cat(sprintf("Asimetría                 : %.6f\n",estadisticas_target$asimetria))

cat(sprintf("Curtosis                  : %.6f\n",estadisticas_target$curtosis))

cat("\nBloque 4 finalizado correctamente.\n")

# BLOQUE 5. Auditoría de las Variables Predictoras -------------------------
## Objetivo: Caracterizar las variables predictoras utilizadas durante el
# Benchmark Científico mediante estadísticas descriptivas y métricas de
# calidad para identificar posibles problemas de escala, variabilidad o
# información.
### Entradas:
# - dataset_auditoria
### Producto:
# - Tabla de auditoría de variables predictoras.
### Responde:
# ¿Las variables predictoras presentan una calidad y comportamiento
# adecuados para el modelado predictivo?

# 5.1. Selección de variables predictoras ----------------------------------
## Objetivo: Seleccionar únicamente las variables numéricas predictoras.

variables_predictoras <- dataset_auditoria |>
  dplyr::select(
    where(is.numeric),
    -log_rendimiento
  )

# 5.2. Construcción de la auditoría ----------------------------------------
## Objetivo: Calcular estadísticas descriptivas por variable.

auditoria_variables <- purrr::map_dfr(
  
  names(variables_predictoras),
  
  function(variable){
    
    x <- variables_predictoras[[variable]]
    
    tibble(
      
      variable = variable,
      
      observaciones = length(x),
      
      n_unicos = dplyr::n_distinct(x),
      
      n_na = sum(is.na(x)),
      
      pct_na = round(
        100 * mean(is.na(x)),
        4
      ),
      
      n_ceros = sum(x == 0, na.rm = TRUE),
      
      pct_ceros = round(
        100 * mean(x == 0, na.rm = TRUE),
        4
      ),
      
      minimo = min(x, na.rm = TRUE),
      
      percentil_25 = quantile(x, 0.25, na.rm = TRUE),
      
      mediana = median(x, na.rm = TRUE),
      
      media = mean(x, na.rm = TRUE),
      
      percentil_75 = quantile(x, 0.75, na.rm = TRUE),
      
      maximo = max(x, na.rm = TRUE),
      
      desviacion = sd(x, na.rm = TRUE),
      
      coeficiente_variacion = ifelse(
        
        abs(mean(x, na.rm = TRUE)) < .Machine$double.eps,
        
        NA_real_,
        
        sd(x, na.rm = TRUE) /
          abs(mean(x, na.rm = TRUE))
        
      ),
      
      asimetria = e1071::skewness(
        x,
        na.rm = TRUE
      ),
      
      curtosis = e1071::kurtosis(
        x,
        na.rm = TRUE
      )
      
    )
    
  }
  
)

# 5.3. Variables constantes ------------------------------------------------
## Objetivo: Detectar variables sin variabilidad.

auditoria_variables <- auditoria_variables |>
  
  mutate(
    
    constante = n_unicos <= 1
    
  )

# 5.4. Variables con alta proporción de ceros ------------------------------
## Objetivo: Identificar variables escasamente informativas.

auditoria_variables <- auditoria_variables |>
  
  mutate(
    
    alta_proporcion_ceros = pct_ceros >= 95
    
  )

# 5.5. Variables con alta variabilidad -------------------------------------
## Objetivo: Identificar variables con escalas muy diferentes.

auditoria_variables <- auditoria_variables |>
  
  mutate(
    
    escala_extrema = coeficiente_variacion > 10
    
  )

# 5.6. Ordenar resultados --------------------------------------------------
## Objetivo: Priorizar las variables potencialmente problemáticas.

auditoria_variables <- auditoria_variables |>
  
  arrange(
    
    desc(coeficiente_variacion)
    
  )

# 5.7. Mostrar resultados --------------------------------------------------
## Objetivo: Presentar la auditoría de las variables predictoras.

cat("\n")
cat(rep("=", 90), sep = "")
cat("\n")
cat("AUDITORÍA DE VARIABLES PREDICTORAS\n")
cat(rep("=", 90), sep = "")
cat("\n")

print(auditoria_variables)

# 5.8. Resumen ejecutivo ---------------------------------------------------
## Objetivo: Resumir la calidad de las variables predictoras.

cat("\n")
cat(rep("-", 90), sep = "")
cat("\n")
cat("RESUMEN\n")
cat(rep("-", 90), sep = "")
cat("\n")

cat(sprintf(
  "Variables evaluadas              : %s\n",
  nrow(auditoria_variables)
))

cat(sprintf(
  "Variables constantes             : %s\n",
  sum(auditoria_variables$constante)
))

cat(sprintf(
  "Variables con >95%% de ceros      : %s\n",
  sum(auditoria_variables$alta_proporcion_ceros)
))

cat(sprintf(
  "Variables con escala extrema     : %s\n",
  sum(auditoria_variables$escala_extrema, na.rm = TRUE)
))

cat("\nBloque 5 finalizado correctamente.\n")

# BLOQUE 6. Auditoría de Correlación y Poder Predictivo --------------------
## Objetivo: Evaluar la relación entre cada variable predictora y la variable
# objetivo mediante correlación y regresión lineal univariada.
### Entradas:
# - dataset_auditoria
### Producto:
# - Tabla de correlación y poder predictivo.
### Responde:
# ¿Existe alguna variable que explique por sí sola la mayor parte del
# rendimiento agrícola o que constituya un posible caso de Data Leakage?

# 6.1. Variables predictoras -----------------------------------------------
## Objetivo: Seleccionar únicamente las variables numéricas predictoras.

variables_predictoras <- dataset_auditoria |>
  dplyr::select(
    where(is.numeric),
    -log_rendimiento
  )

# 6.2. Variable objetivo ---------------------------------------------------
## Objetivo: Extraer la variable objetivo.

target <- dataset_auditoria$log_rendimiento

# 6.3. Correlación y regresión univariada ----------------------------------
## Objetivo: Calcular el poder predictivo individual de cada variable.

auditoria_predictiva <- purrr::map_dfr(
  
  names(variables_predictoras),
  
  function(variable){
    
    x <- variables_predictoras[[variable]]
    
    datos_modelo <- tibble(
      
      x = x,
      
      y = target
      
    ) |>
      
      tidyr::drop_na()
    
    if(
      
      nrow(datos_modelo) < 10 ||
      
      sd(datos_modelo$x) == 0
      
    ){
      
      return(
        
        tibble(
          
          variable = variable,
          
          correlacion = NA_real_,
          
          correlacion_abs = NA_real_,
          
          r2_univariado = NA_real_
          
        )
        
      )
      
    }
    
    correlacion <- cor(
      
      datos_modelo$x,
      
      datos_modelo$y,
      
      method = "pearson"
      
    )
    
    modelo <- lm(
      
      y ~ x,
      
      data = datos_modelo
      
    )
    
    tibble(
      
      variable = variable,
      
      correlacion = correlacion,
      
      correlacion_abs = abs(correlacion),
      
      r2_univariado = summary(modelo)$r.squared
      
    )
    
  }
  
)

# 6.4. Ordenar resultados --------------------------------------------------
## Objetivo: Priorizar las variables más explicativas.

auditoria_predictiva <- auditoria_predictiva |>
  
  arrange(
    
    desc(r2_univariado)
    
  )

# 6.5. Clasificación automática --------------------------------------------
## Objetivo: Clasificar automáticamente las variables.

auditoria_predictiva <- auditoria_predictiva |>
  
  mutate(
    
    clasificacion = case_when(
      
      r2_univariado >= 0.95 ~
        
        "Posible Data Leakage",
      
      r2_univariado >= 0.80 ~
        
        "Variable dominante",
      
      r2_univariado >= 0.50 ~
        
        "Alta capacidad predictiva",
      
      r2_univariado >= 0.20 ~
        
        "Capacidad predictiva moderada",
      
      TRUE ~
        
        "Baja capacidad predictiva"
      
    )
    
  )

# 6.6. Mostrar resultados --------------------------------------------------

cat("\n")
cat(rep("=",90),sep="")
cat("\n")
cat("CORRELACIÓN Y PODER PREDICTIVO INDIVIDUAL\n")
cat(rep("=",90),sep="")
cat("\n")

print(auditoria_predictiva)

# 6.7. Top 10 --------------------------------------------------------------

cat("\n")
cat(rep("-",90),sep="")
cat("\n")
cat("TOP 10 VARIABLES MÁS EXPLICATIVAS\n")
cat(rep("-",90),sep="")
cat("\n")

print(
  
  auditoria_predictiva |>
    
    slice_head(
      
      n = 10
      
    )
  
)

# 6.8. Variables sospechosas -----------------------------------------------

variables_sospechosas <- auditoria_predictiva |>
  
  filter(
    
    r2_univariado >= 0.95
    
  )

cat("\n")
cat(rep("-",90),sep="")
cat("\n")
cat("VARIABLES SOSPECHOSAS\n")
cat(rep("-",90),sep="")
cat("\n")

if(nrow(variables_sospechosas)==0){
  
  cat(
    "No se detectaron variables con evidencia fuerte de Data Leakage.\n"
  )
  
}else{
  
  print(
    
    variables_sospechosas
    
  )
  
}

cat("\nBloque 6 finalizado correctamente.\n")

# BLOQUE 7. Importancia de Variables mediante Random Forest ----------------
## Objetivo: Evaluar la importancia relativa de las variables predictoras
# utilizando Random Forest para identificar los predictores con mayor
# contribución al rendimiento agrícola.
### Entradas:
# - dataset_auditoria
### Producto:
# - Modelo Random Forest.
# - Ranking de importancia de variables.
### Responde:
# ¿Qué variables presentan la mayor importancia predictiva según
# Random Forest y son consistentes con el análisis de correlación?

# 7.1. Preparación de los datos --------------------------------------------
## Objetivo: Construir el conjunto de datos para Random Forest.

datos_rf <- dataset_auditoria |>
  
  dplyr::select(
    
    where(is.numeric)
    
  ) |>
  
  tidyr::drop_na()

# 7.2. Entrenamiento -------------------------------------------------------
## Objetivo: Ajustar el modelo Random Forest.

set.seed(5477976)

modelo_rf <- randomForest(
  
  log_rendimiento ~ .,
  
  data = datos_rf,
  
  ntree = 500,
  
  mtry = floor(
    
    sqrt(
      
      ncol(datos_rf) - 1
      
    )
    
  ),
  
  importance = TRUE,
  
  keep.forest = TRUE
  
) # Modelo Random Forest

# 7.3. Importancia de variables --------------------------------------------
## Objetivo: Extraer la importancia de cada predictor.

importancia_rf <- importance(
  
  modelo_rf,
  
  type = 1
  
)

importancia_rf <- tibble(
  
  variable = rownames(importancia_rf),
  
  importancia_rf = importancia_rf[,1]
  
) |>
  
  arrange(
    
    desc(importancia_rf)
    
  )

# 7.4. Error Out Of Bag ----------------------------------------------------
## Objetivo: Registrar el error OOB del modelo.

oob_mse <- tail(
  
  modelo_rf$mse,
  
  1
  
)

varianza_explicada <- tail(
  
  modelo_rf$rsq,
  
  1
  
)

# 7.5. Mostrar importancia -------------------------------------------------

cat("\n")
cat(rep("=",90),sep="")
cat("\n")
cat("IMPORTANCIA DE VARIABLES - RANDOM FOREST\n")
cat(rep("=",90),sep="")
cat("\n")

print(importancia_rf)

# 7.6. Top 20 --------------------------------------------------------------

cat("\n")
cat(rep("-",90),sep="")
cat("\n")
cat("TOP 20 VARIABLES MÁS IMPORTANTES\n")
cat(rep("-",90),sep="")
cat("\n")

print(
  
  importancia_rf |>
    
    slice_head(
      
      n = 20
      
    )
  
)

# 7.7. Diagnóstico del modelo ----------------------------------------------
## Objetivo: Mostrar el desempeño general del Random Forest.

cat("\n")
cat(rep("-",90),sep="")
cat("\n")
cat("DESEMPEÑO DEL RANDOM FOREST\n")
cat(rep("-",90),sep="")
cat("\n")

cat(sprintf(
  
  "Número de árboles              : %s\n",
  
  modelo_rf$ntree
  
))

cat(sprintf(
  
  "Variables evaluadas            : %s\n",
  
  nrow(importancia_rf)
  
))

cat(sprintf(
  
  "Error OOB (MSE)                : %.6f\n",
  
  oob_mse
  
))

cat(sprintf(
  
  "Varianza explicada (OOB R²)    : %.6f\n",
  
  varianza_explicada
  
))

# 7.8. Exportación ---------------------------------------------------------
## Objetivo: Guardar la importancia de variables.

arrow::write_parquet(
  
  importancia_rf,
  
  "importancia_random_forest.parquet"
  
)

write.csv(
  
  importancia_rf,
  
  "importancia_random_forest.csv",
  
  row.names = FALSE
  
)

# 7.9. Finalización --------------------------------------------------------
## Objetivo: Confirmar la finalización del bloque.

cat("\nBloque 7 finalizado correctamente.\n")

# BLOQUE 8. Consolidación del Poder Predictivo -----------------------------
## Objetivo: Integrar la evidencia obtenida mediante estadística descriptiva,
# correlación, regresión univariada e importancia de Random Forest para
# identificar las variables más relevantes del Dataset Ganador.
### Entradas:
# - auditoria_variables
# - auditoria_predictiva
# - importancia_rf
### Producto:
# - auditoria_consolidada
### Responde:
# ¿Qué variables presentan mayor capacidad predictiva y cuáles requieren
# revisión antes del Benchmark definitivo?

# 8.1. Consolidación -------------------------------------------------------
## Objetivo: Integrar todas las auditorías.

auditoria_consolidada <- auditoria_variables |>
  
  left_join(
    auditoria_predictiva,
    by = "variable"
  ) |>
  
  left_join(
    importancia_rf,
    by = "variable"
  )

# 8.2. Ranking -------------------------------------------------------------
## Objetivo: Construir rankings independientes.

auditoria_consolidada <- auditoria_consolidada |>
  
  mutate(
    
    ranking_correlacion = rank(
      -correlacion_abs,
      ties.method = "min"
    ),
    
    ranking_r2 = rank(
      -r2_univariado,
      ties.method = "min"
    ),
    
    ranking_rf = rank(
      -importancia_rf,
      ties.method = "min"
    )
    
  )

# 8.3. Score predictivo ----------------------------------------------------
## Objetivo: Calcular un indicador global de importancia.

auditoria_consolidada <- auditoria_consolidada |>
  
  mutate(
    
    score_predictivo = (
      
      ranking_correlacion +
        
        ranking_r2 +
        
        ranking_rf
      
    ) / 3
    
  )

# 8.4. Clasificación -------------------------------------------------------
## Objetivo: Clasificar automáticamente las variables.

auditoria_consolidada <- auditoria_consolidada |>
  
  mutate(
    
    decision = case_when(
      
      correlacion_abs >= 0.95 &
        r2_univariado >= 0.95 ~
        
        "REVISAR - Posible Data Leakage",
      
      score_predictivo <= 5 ~
        
        "Muy Alta Importancia",
      
      score_predictivo <= 10 ~
        
        "Alta Importancia",
      
      score_predictivo <= 20 ~
        
        "Importancia Moderada",
      
      TRUE ~
        
        "Baja Importancia"
      
    )
    
  )

# 8.5. Ordenar resultados --------------------------------------------------

auditoria_consolidada <- auditoria_consolidada |>
  
  arrange(
    score_predictivo
  )

# 8.6. Mostrar resultados --------------------------------------------------

cat("\n")
cat(rep("=",100), sep = "")
cat("\n")
cat("AUDITORÍA CONSOLIDADA DEL PODER PREDICTIVO\n")
cat(rep("=",100), sep = "")
cat("\n")

print(auditoria_consolidada)

# 8.7. Top 20 --------------------------------------------------------------

cat("\n")
cat(rep("-",100), sep = "")
cat("\n")
cat("TOP 20 VARIABLES\n")
cat(rep("-",100), sep = "")
cat("\n")

top20_variables <- auditoria_consolidada |>
  
  slice_head(
    n = 20
  )

print(top20_variables)

# 8.8. Variables sospechosas -----------------------------------------------

variables_leakage <- auditoria_consolidada |>
  
  filter(
    decision == "REVISAR - Posible Data Leakage"
  )

cat("\n")
cat(rep("-",100), sep = "")
cat("\n")
cat("VARIABLES QUE REQUIEREN REVISIÓN\n")
cat(rep("-",100), sep = "")
cat("\n")

if(nrow(variables_leakage) == 0){
  
  cat(
    "No se detectó evidencia fuerte de Data Leakage.\n"
  )
  
}else{
  
  print(
    
    variables_leakage
    
  )
  
}

# 8.9. Exportación ---------------------------------------------------------

arrow::write_parquet(
  auditoria_consolidada,
  "auditoria_poder_predictivo.parquet"
)

write.csv(
  auditoria_consolidada,
  "auditoria_poder_predictivo.csv",
  row.names = FALSE
)

write.csv(
  top20_variables,
  "top20_variables_predictivas.csv",
  row.names = FALSE
)

cat("\nBloque 8 finalizado correctamente.\n")

# BLOQUE 9. Auditoría de Multicolinealidad ---------------------------------
## Objetivo: Evaluar la existencia de relaciones lineales fuertes entre las
# variables predictoras mediante la matriz de correlaciones, el Factor de
# Inflación de la Varianza (VIF) y el número de condición.
### Entradas:
# - dataset_auditoria
### Producto:
# - Matriz de correlaciones.
# - Variables altamente correlacionadas.
# - VIF.
# - Número de condición.
### Responde:
# ¿Las variables predictoras presentan problemas de multicolinealidad que
# puedan afectar la interpretación del Benchmark Científico?

# 9.1. Preparación del conjunto de datos -----------------------------------
## Objetivo: Seleccionar las variables numéricas predictoras.

datos_vif <- dataset_auditoria |>
  dplyr::select(
    where(is.numeric),
    -log_rendimiento
  ) |>
  tidyr::drop_na()

# 9.2. Eliminación de variables constantes ---------------------------------
## Objetivo: Eliminar variables sin variabilidad.

variables_constantes <- names(datos_vif)[
  
  sapply(
    
    datos_vif,
    
    function(x){
      
      length(
        
        unique(
          
          x[!is.na(x)]
          
        )
        
      ) <= 1
      
    }
    
  )
  
]

if(length(variables_constantes) > 0){
  
  datos_vif <- datos_vif |>
    
    dplyr::select(
      
      -all_of(variables_constantes)
      
    )
  
}

# 9.3. Matriz de correlaciones ---------------------------------------------
## Objetivo: Calcular la correlación entre predictores.

matriz_cor <- cor(
  
  datos_vif,
  
  method = "pearson",
  
  use = "pairwise.complete.obs"
  
)

cat("\n")
cat(rep("=",100),sep="")
cat("\n")
cat("MATRIZ DE CORRELACIONES\n")
cat(rep("=",100),sep="")
cat("\n")

print(round(matriz_cor,3))

# 9.4. Variables altamente correlacionadas ---------------------------------
## Objetivo: Detectar correlaciones superiores a 0.90.

cor_alta <- which(
  
  abs(matriz_cor) > 0.90 &
    abs(matriz_cor) < 1,
  
  arr.ind = TRUE
  
)

cor_alta <- tibble(
  
  variable_1 = rownames(matriz_cor)[cor_alta[,1]],
  
  variable_2 = colnames(matriz_cor)[cor_alta[,2]],
  
  correlacion = matriz_cor[cor_alta]
  
) |>
  
  dplyr::filter(
    
    variable_1 < variable_2
    
  ) |>
  
  arrange(
    
    desc(abs(correlacion))
    
  )

cat("\n")
cat(rep("-",100),sep="")
cat("\n")
cat("VARIABLES ALTAMENTE CORRELACIONADAS\n")
cat(rep("-",100),sep="")
cat("\n")

if(nrow(cor_alta)==0){
  
  cat(
    "No se detectaron correlaciones superiores a 0.90.\n"
  )
  
}else{
  
  print(cor_alta)
  
}

# 9.5. Modelo para VIF -----------------------------------------------------
## Objetivo: Ajustar el modelo lineal para calcular el VIF.

modelo_vif <- lm(
  
  log_rendimiento ~ .,
  
  data = dataset_auditoria |>
    
    dplyr::select(
      
      where(is.numeric)
      
    ) |>
    
    tidyr::drop_na()
  
)

# 9.6. Factor de Inflación de la Varianza ----------------------------------

vif_resultado <- tibble(
  
  variable = names(
    
    car::vif(modelo_vif)
    
  ),
  
  vif = as.numeric(
    
    car::vif(modelo_vif)
    
  )
  
) |>
  
  mutate(
    
    diagnostico = case_when(
      
      vif < 5 ~
        
        "Sin evidencia",
      
      vif < 10 ~
        
        "Moderada",
      
      TRUE ~
        
        "Alta"
      
    )
    
  ) |>
  
  arrange(
    
    desc(vif)
    
  )

cat("\n")
cat(rep("-",100),sep="")
cat("\n")
cat("FACTOR DE INFLACIÓN DE LA VARIANZA\n")
cat(rep("-",100),sep="")
cat("\n")

print(vif_resultado)

# 9.7. Número de condición -------------------------------------------------

numero_condicion <- kappa(
  
  as.matrix(datos_vif)
  
)

cat("\n")
cat(rep("-",100),sep="")
cat("\n")
cat("NÚMERO DE CONDICIÓN\n")
cat(rep("-",100),sep="")
cat("\n")

cat(sprintf(
  
  "Número de condición : %.2f\n",
  
  numero_condicion
  
))

if(numero_condicion < 30){
  
  cat("Diagnóstico         : Sin evidencia importante.\n")
  
}else if(numero_condicion < 100){
  
  cat("Diagnóstico         : Multicolinealidad moderada.\n")
  
}else{
  
  cat("Diagnóstico         : Multicolinealidad severa.\n")
  
}

# 9.8. Heatmap -------------------------------------------------------------

corrplot::corrplot(
  
  matriz_cor,
  
  method = "color",
  
  type = "upper",
  
  tl.cex = 0.60,
  
  tl.col = "black",
  
  diag = FALSE
  
)


# Solo correlaciones fuertes
cor_long <- as.data.frame(
  
  as.table(matriz_cor),
  
  stringsAsFactors = FALSE
  
) |>
  
  dplyr::rename(
    
    variable_1 = Var1,
    
    variable_2 = Var2,
    
    correlacion = Freq
    
  ) |>
  
  dplyr::mutate(
    
    variable_1 = as.character(variable_1),
    
    variable_2 = as.character(variable_2)
    
  ) |>
  
  dplyr::filter(
    
    variable_1 != variable_2
    
  ) |>
  
  dplyr::rowwise() |>
  
  dplyr::mutate(
    
    menor = min(variable_1, variable_2),
    
    mayor = max(variable_1, variable_2)
    
  ) |>
  
  dplyr::ungroup() |>
  
  dplyr::distinct(
    
    menor,
    
    mayor,
    
    .keep_all = TRUE
    
  ) |>
  
  dplyr::select(
    
    -menor,
    
    -mayor
    
  ) |>
  
  dplyr::arrange(
    
    desc(abs(correlacion))
    
  )



cor_long |>
  
  dplyr::filter(
    
    abs(correlacion) >= 0.70
    
  )

# 9.9. Exportación ---------------------------------------------------------
## Objetivo: Exportar los resultados de la auditoría de multicolinealidad.

ruta_salida <- here(
  
  "reports",
  
  "auditoria_dataset"
  
) # Carpeta de resultados

dir.create(
  ruta_salida,
  recursive = TRUE,
  showWarnings = FALSE
) # Crear carpeta de resultados

arrow::write_parquet(
  
  vif_resultado,
  
  file.path(
    ruta_salida,
    "auditoria_vif.parquet"
  )
  
) # Exportar VIF en Parquet

write.csv(
  
  vif_resultado,
  
  file.path(
    ruta_salida,
    "auditoria_vif.csv"
  ),
  
  row.names = FALSE
  
) # Exportar VIF en CSV

write.csv(
  
  cor_alta,
  
  file.path(
    ruta_salida,
    "variables_altamente_correlacionadas.csv"
  ),
  
  row.names = FALSE
  
) # Exportar correlaciones altas

openxlsx::write.xlsx(
  
  round(matriz_cor, 3),
  
  file.path(
    ruta_salida,
    "matriz_correlaciones.xlsx"
  ),
  
  rowNames = TRUE
  
) # Exportar matriz de correlaciones

cat("\nResultados exportados correctamente.\n")

cat(sprintf(
  "Directorio: %s\n",
  ruta_salida
))

cat("\nBloque 9 finalizado correctamente.\n")

# BLOQUE 10. Integración de Evidencias -------------------------------------
## Objetivo: Integrar la evidencia obtenida mediante estadística descriptiva,
# correlación, regresión lineal univariada, importancia de Random Forest y
# diagnóstico de multicolinealidad para construir la auditoría oficial de
# las variables predictoras.
### Entradas:
# - auditoria_variables
# - auditoria_predictiva
# - importancia_rf
# - vif_resultado
### Producto:
# - auditoria_final
### Responde:
# ¿Qué variables presentan mayor capacidad predictiva y cuáles requieren
# revisión por posible Data Leakage o multicolinealidad?

# 10.1. Consolidación ------------------------------------------------------
## Objetivo: Integrar todas las auditorías.

auditoria_final <- auditoria_variables |>
  
  left_join(
    auditoria_predictiva,
    by = "variable"
  ) |>
  
  left_join(
    importancia_rf,
    by = "variable"
  ) |>
  
  left_join(
    vif_resultado,
    by = "variable"
  )

# 10.2. Rankings -----------------------------------------------------------
## Objetivo: Construir rankings independientes.

auditoria_final <- auditoria_final |>
  
  mutate(
    
    ranking_correlacion = rank(
      -replace_na(correlacion_abs, -Inf),
      ties.method = "min"
    ),
    
    ranking_r2 = rank(
      -replace_na(r2_univariado, -Inf),
      ties.method = "min"
    ),
    
    ranking_rf = rank(
      -replace_na(importancia_rf, -Inf),
      ties.method = "min"
    )
    
  )

# 10.3. Score predictivo ---------------------------------------------------
## Objetivo: Calcular un indicador global de importancia.

auditoria_final <- auditoria_final |>
  
  mutate(
    
    score_predictivo = (
      
      ranking_correlacion +
        
        ranking_r2 +
        
        ranking_rf
      
    ) / 3
    
  )

# 10.4. Índice de consenso -------------------------------------------------
## Objetivo: Contabilizar cuántas métricas consideran importante cada variable.

auditoria_final <- auditoria_final |>
  
  mutate(
    
    consenso =
      
      (correlacion_abs >= 0.70) +
      
      (r2_univariado >= 0.50) +
      
      (importancia_rf >= median(importancia_rf, na.rm = TRUE))
    
  )

# 10.5. Dictamen metodológico ----------------------------------------------
## Objetivo: Clasificar automáticamente cada variable.

auditoria_final <- auditoria_final |>
  
  mutate(
    
    dictamen = case_when(
      
      correlacion_abs >= 0.95 &
        r2_univariado >= 0.95 ~
        
        "REVISAR - Posible Data Leakage",
      
      vif >= 10 ~
        
        "REVISAR - Alta Multicolinealidad",
      
      score_predictivo <= 5 ~
        
        "Muy Alta Importancia",
      
      score_predictivo <= 10 ~
        
        "Alta Importancia",
      
      score_predictivo <= 20 ~
        
        "Importancia Moderada",
      
      TRUE ~
        
        "Baja Importancia"
      
    )
    
  )

# 10.6. Ordenar resultados -------------------------------------------------

auditoria_final <- auditoria_final |>
  
  arrange(
    
    score_predictivo,
    
    desc(importancia_rf)
    
  )

# 10.7. Mostrar auditoría --------------------------------------------------

cat("\n")
cat(rep("=",100),sep="")
cat("\n")
cat("AUDITORÍA INTEGRADA DEL PODER PREDICTIVO\n")
cat(rep("=",100),sep="")
cat("\n")

print(auditoria_final)

# 10.8. Variables críticas -------------------------------------------------

variables_criticas <- auditoria_final |>
  
  filter(
    
    grepl(
      "REVISAR",
      dictamen
    )
    
  )

cat("\n")
cat(rep("-",100),sep="")
cat("\n")
cat("VARIABLES QUE REQUIEREN REVISIÓN\n")
cat(rep("-",100),sep="")
cat("\n")

if(nrow(variables_criticas)==0){
  
  cat(
    "No se detectaron variables críticas.\n"
  )
  
}else{
  
  print(
    
    variables_criticas |>
      
      select(
        
        variable,
        
        correlacion,
        
        r2_univariado,
        
        importancia_rf,
        
        vif,
        
        consenso,
        
        dictamen
        
      )
    
  )
  
}

# 10.9. Top 20 -------------------------------------------------------------

top20_variables <- auditoria_final |>
  
  select(
    
    variable,
    
    correlacion,
    
    r2_univariado,
    
    importancia_rf,
    
    vif,
    
    consenso,
    
    score_predictivo,
    
    dictamen
    
  ) |>
  
  slice_head(
    n = 20
  )

cat("\n")
cat(rep("-",100),sep="")
cat("\n")
cat("TOP 20 VARIABLES\n")
cat(rep("-",100),sep="")
cat("\n")

print(top20_variables)

# 10.10. Resumen -----------------------------------------------------------

cat("\n")
cat(rep("-",100),sep="")
cat("\n")
cat("RESUMEN DE LA AUDITORÍA\n")
cat(rep("-",100),sep="")
cat("\n")

print(
  
  auditoria_final |>
    
    count(dictamen)
  
)

# 10.11. Exportación -------------------------------------------------------

arrow::write_parquet(
  
  auditoria_final,
  
  "auditoria_integrada_variables.parquet"
  
)

write.csv(
  
  auditoria_final,
  
  "auditoria_integrada_variables.csv",
  
  row.names = FALSE
  
)

write.csv(
  
  top20_variables,
  
  "top20_variables.csv",
  
  row.names = FALSE
  
)

cat("\nBloque 10 finalizado correctamente.\n")

# BLOQUE 11. Auditoría de Dependencias Lineales ----------------------------
## Objetivo: Detectar combinaciones lineales exactas entre las variables predictoras.
### Entradas: - dataset_auditoria
### Producto: - Variables linealmente dependientes
### Responde: ¿Existen variables que pueden reconstruirse exactamente a partir de otras variables?

# 11.1. Preparación --------------------------------------------------------
## Objetivo: Seleccionar las variables numéricas predictoras.

datos_dependencias <- dataset_auditoria |>
  dplyr::select(where(is.numeric), -log_rendimiento) |>
  tidyr::drop_na() # Variables predictoras

# 11.2. Dependencias lineales ----------------------------------------------
## Objetivo: Detectar combinaciones lineales exactas.

dependencias_lineales <- caret::findLinearCombos(as.matrix(datos_dependencias)) # Dependencias lineales

variables_eliminar <- if (is.null(dependencias_lineales)) character(0) else colnames(datos_dependencias)[dependencias_lineales$remove] # Variables sugeridas

# 11.3. Resultados ---------------------------------------------------------
## Objetivo: Mostrar los resultados de la auditoría.

cat("\n", strrep("=", 100), "\n", "AUDITORÍA DE DEPENDENCIAS LINEALES\n", strrep("=", 100), "\n", sep = "")

if (is.null(dependencias_lineales)) {
  
  cat("No se detectaron dependencias lineales exactas.\n")
  
} else {
  
  for (i in seq_along(dependencias_lineales$linearCombos)) {
    
    cat("\nCombinación", i, "\n")
    
    print(colnames(datos_dependencias)[dependencias_lineales$linearCombos[[i]]])
    
  }
  
}

cat("\n", strrep("-", 100), "\n", "VARIABLES SUGERIDAS PARA ELIMINAR\n", strrep("-", 100), "\n", sep = "")

if (length(variables_eliminar) == 0) {
  
  cat("No se identificaron variables redundantes.\n")
  
} else {
  
  print(tibble(variable = variables_eliminar))
  
}

# 11.4. Resumen ------------------------------------------------------------
## Objetivo: Resumir la auditoría.

cat("\n", strrep("-", 100), "\n", "RESUMEN\n", strrep("-", 100), "\n", sep = "")

cat(sprintf("Variables evaluadas          : %s\n", ncol(datos_dependencias)))
cat(sprintf("Dependencias detectadas      : %s\n", ifelse(is.null(dependencias_lineales), 0, length(dependencias_lineales$linearCombos))))
cat(sprintf("Variables sugeridas eliminar : %s\n", length(variables_eliminar)))

# 11.5. Exportación --------------------------------------------------------
## Objetivo: Exportar los resultados.

ruta_salida <- here("reports", "auditoria_dataset") # Carpeta de resultados

dir.create(ruta_salida, recursive = TRUE, showWarnings = FALSE) # Crear carpeta

write.csv(tibble(variable = variables_eliminar), file.path(ruta_salida, "variables_linealmente_dependientes.csv"), row.names = FALSE) # Exportar variables

if (!is.null(dependencias_lineales)) saveRDS(dependencias_lineales, file.path(ruta_salida, "dependencias_lineales.rds")) # Exportar dependencias

cat("\nBloque 11 finalizado correctamente.\n")

# BLOQUE 12. Auditoría de Correlaciones Altas ------------------------------
## Objetivo: Identificar variables altamente correlacionadas y sugerir
# variables redundantes para eliminar.
### Entradas: - dataset_auditoria
### Producto: - Variables altamente correlacionadas
### Responde: ¿Qué variables contienen prácticamente la misma información?

# 12.1. Preparación --------------------------------------------------------
## Objetivo: Seleccionar las variables numéricas predictoras.

datos_cor <- dataset_auditoria |>
  dplyr::select(where(is.numeric), -log_rendimiento) |>
  tidyr::drop_na() # Variables predictoras

# 12.2. Matriz de correlaciones --------------------------------------------
## Objetivo: Calcular la matriz de correlaciones de Pearson.

matriz_cor <- cor(datos_cor, method = "pearson") # Matriz de correlaciones

# 12.3. Variables sugeridas para eliminar ----------------------------------
## Objetivo: Detectar variables altamente correlacionadas.

variables_correlacion <- caret::findCorrelation(matriz_cor, cutoff = 0.90, names = TRUE) # Variables redundantes

# 12.4. Tabla de correlaciones altas ---------------------------------------
## Objetivo: Construir la tabla de correlaciones superiores a 0.90.

cor_alta <- which(abs(matriz_cor) > 0.90 & abs(matriz_cor) < 1, arr.ind = TRUE)

cor_alta <- tibble(
  variable_1 = rownames(matriz_cor)[cor_alta[, 1]],
  variable_2 = colnames(matriz_cor)[cor_alta[, 2]],
  correlacion = matriz_cor[cor_alta]
) |>
  dplyr::filter(variable_1 < variable_2) |>
  dplyr::arrange(desc(abs(correlacion))) # Correlaciones únicas

# 12.5. Resultados ---------------------------------------------------------
## Objetivo: Mostrar las correlaciones altas.

cat("\n", strrep("=", 100), "\n", "AUDITORÍA DE CORRELACIONES ALTAS\n", strrep("=", 100), "\n", sep = "")

print(cor_alta)

cat("\n", strrep("-", 100), "\n", "VARIABLES SUGERIDAS PARA ELIMINAR\n", strrep("-", 100), "\n", sep = "")

if (length(variables_correlacion) == 0) {
  
  cat("No se detectaron variables redundantes.\n")
  
} else {
  
  print(tibble(variable = variables_correlacion))
  
}

# 12.6. Heatmap ------------------------------------------------------------
## Objetivo: Visualizar la estructura de correlaciones.

corrplot::corrplot(
  matriz_cor,
  method = "color",
  order = "hclust",
  addrect = 4,
  tl.cex = 0.60,
  tl.col = "black",
  diag = FALSE
) # Heatmap

# 12.7. Resumen ------------------------------------------------------------
## Objetivo: Resumir la auditoría.

cat("\n", strrep("-", 100), "\n", "RESUMEN\n", strrep("-", 100), "\n", sep = "")

cat(sprintf("Variables evaluadas            : %s\n", ncol(datos_cor)))
cat(sprintf("Correlaciones > 0.90           : %s\n", nrow(cor_alta)))
cat(sprintf("Variables sugeridas eliminar   : %s\n", length(variables_correlacion)))

# 12.8. Exportación --------------------------------------------------------
## Objetivo: Exportar los resultados.

ruta_salida <- here("reports", "auditoria_dataset") # Carpeta de resultados

dir.create(ruta_salida, recursive = TRUE, showWarnings = FALSE) # Crear carpeta

write.csv(cor_alta, file.path(ruta_salida, "correlaciones_altas.csv"), row.names = FALSE) # Exportar correlaciones

write.csv(tibble(variable = variables_correlacion), file.path(ruta_salida, "variables_correlacion_alta.csv"), row.names = FALSE) # Exportar variables

openxlsx::write.xlsx(round(matriz_cor, 3), file.path(ruta_salida, "matriz_correlaciones.xlsx"), rowNames = TRUE) # Exportar matriz

cat("\nBloque 12 finalizado correctamente.\n")

# BLOQUE 14. Auditoría de Redundancia Predictiva ---------------------------
## Objetivo: Comparar el desempeño predictivo del dataset completo frente al
# dataset reducido después de eliminar variables redundantes.
### Entradas: - dataset_auditoria - variables_eliminar
### Producto: - Comparación de desempeño
### Responde: ¿Las variables redundantes aportan información predictiva?

# 14.1. Construcción de datasets -------------------------------------------
## Objetivo: Crear el dataset completo y el reducido.

dataset_completo <- dataset_auditoria |>
  dplyr::select(where(is.numeric)) |>
  tidyr::drop_na() # Dataset completo

dataset_reducido <- dataset_completo |>
  dplyr::select(-any_of(variables_eliminar)) # Dataset reducido

# 14.2. Ajuste de modelos --------------------------------------------------
## Objetivo: Ajustar la regresión lineal en ambos datasets.

modelo_completo <- lm(
  log_rendimiento ~ .,
  data = dataset_completo
) # Modelo completo

modelo_reducido <- lm(
  log_rendimiento ~ .,
  data = dataset_reducido
) # Modelo reducido

# 14.3. Predicciones -------------------------------------------------------
## Objetivo: Obtener las predicciones.

pred_completo <- predict(modelo_completo) # Predicciones modelo completo

pred_reducido <- predict(modelo_reducido) # Predicciones modelo reducido

# 14.4. Métricas -----------------------------------------------------------
## Objetivo: Calcular las métricas de desempeño.

metricas_modelo <- function(y, pred){
  
  tibble(
    
    RMSE = sqrt(mean((y - pred)^2)),
    
    MAE = mean(abs(y - pred)),
    
    R2 = cor(y, pred)^2
    
  )
  
} # Función de evaluación

metricas_completo <- metricas_modelo(
  dataset_completo$log_rendimiento,
  pred_completo
) |>
  mutate(dataset = "Completo")

metricas_reducido <- metricas_modelo(
  dataset_reducido$log_rendimiento,
  pred_reducido
) |>
  mutate(dataset = "Reducido")

comparacion_modelos <- bind_rows(
  metricas_completo,
  metricas_reducido
) # Comparación

# 14.5. Resultados ---------------------------------------------------------
## Objetivo: Mostrar la comparación.

cat("\n", strrep("=", 100), "\n", "COMPARACIÓN DE DESEMPEÑO\n", strrep("=", 100), "\n", sep = "")

print(comparacion_modelos)

# 14.6. Diferencias --------------------------------------------------------
## Objetivo: Calcular las diferencias entre modelos.

comparacion <- tibble(
  
  Delta_R2 = metricas_reducido$R2 - metricas_completo$R2,
  
  Delta_RMSE = metricas_reducido$RMSE - metricas_completo$RMSE,
  
  Delta_MAE = metricas_reducido$MAE - metricas_completo$MAE
  
)

cat("\n", strrep("-", 100), "\n", "DIFERENCIAS\n", strrep("-", 100), "\n", sep = "")

print(comparacion)

# 14.7. Dictamen -----------------------------------------------------------
## Objetivo: Emitir el dictamen de la auditoría.

cat("\n", strrep("-", 100), "\n", "DICTAMEN\n", strrep("-", 100), "\n", sep = "")

if(abs(comparacion$Delta_R2) < 0.01){
  
  cat("La eliminación de variables redundantes no modifica significativamente el desempeño.\n")
  cat("Se recomienda utilizar el dataset reducido.\n")
  
}else{
  
  cat("Las variables eliminadas contienen información predictiva relevante.\n")
  cat("Se recomienda revisar el criterio de eliminación.\n")
  
}

# 14.8. Exportación --------------------------------------------------------
## Objetivo: Exportar los resultados.

ruta_salida <- here("reports", "auditoria_dataset") # Carpeta de resultados

dir.create(ruta_salida, recursive = TRUE, showWarnings = FALSE) # Crear carpeta

write.csv(
  comparacion_modelos,
  file.path(ruta_salida, "comparacion_dataset_completo_vs_reducido.csv"),
  row.names = FALSE
) # Exportar comparación

write.csv(
  comparacion,
  file.path(ruta_salida, "diferencias_dataset_completo_vs_reducido.csv"),
  row.names = FALSE
) # Exportar diferencias

cat("\nBloque 14 finalizado correctamente.\n")

# BLOQUE 15. Construcción del Dataset Científico Depurado ------------------
## Objetivo: Construir el Dataset Científico Depurado eliminando únicamente
# las variables redundantes identificadas durante la auditoría.
### Entradas: - dataset_auditoria - variables_eliminar
### Producto: - Dataset depurado
### Responde: ¿Cuál es el dataset oficial recomendado para el Benchmark?

# 15.1. Construcción -------------------------------------------------------
## Objetivo: Eliminar las variables redundantes.

dataset_depurado <- dataset_auditoria |>
  dplyr::select(
    -any_of(variables_eliminar)
  ) # Dataset depurado

# 15.2. Variables conservadas ----------------------------------------------
## Objetivo: Registrar las variables finales.

variables_conservadas <- tibble(
  variable = names(dataset_depurado)
) # Variables conservadas

variables_eliminadas <- tibble(
  variable = variables_eliminar
) # Variables eliminadas

# 15.3. Resumen ------------------------------------------------------------
## Objetivo: Resumir el dataset final.

cat("\n", strrep("=", 100), "\n", "DATASET CIENTÍFICO DEPURADO\n", strrep("=", 100), "\n", sep = "")

cat(sprintf(
  "Observaciones                 : %s\n",
  format(nrow(dataset_depurado), big.mark = ",")
))

cat(sprintf(
  "Variables originales          : %s\n",
  ncol(dataset_auditoria)
))

cat(sprintf(
  "Variables eliminadas          : %s\n",
  length(variables_eliminar)
))

cat(sprintf(
  "Variables finales             : %s\n",
  ncol(dataset_depurado)
))

# 15.4. Verificación -------------------------------------------------------
## Objetivo: Verificar la consistencia del dataset.

stopifnot(
  nrow(dataset_depurado) == nrow(dataset_auditoria)
) # Validar observaciones

stopifnot(
  !"geometry" %in% names(dataset_depurado)
) # Validar geometry

stopifnot(
  "log_rendimiento" %in% names(dataset_depurado)
) # Validar variable objetivo

cat("Validaciones completadas correctamente.\n")

# 15.5. Exportación --------------------------------------------------------
## Objetivo: Exportar el dataset definitivo.

ruta_salida <- here("reports", "auditoria_dataset") # Carpeta de resultados

dir.create(ruta_salida, recursive = TRUE, showWarnings = FALSE) # Crear carpeta

arrow::write_parquet(
  dataset_depurado,
  file.path(ruta_salida, "dataset_gnn_certificado_v2.parquet")
) # Exportar Parquet

write.csv(
  variables_conservadas,
  file.path(ruta_salida, "variables_conservadas.csv"),
  row.names = FALSE
) # Exportar variables conservadas

write.csv(
  variables_eliminadas,
  file.path(ruta_salida, "variables_eliminadas.csv"),
  row.names = FALSE
) # Exportar variables eliminadas

# 15.6. Dictamen -----------------------------------------------------------
## Objetivo: Confirmar la construcción del dataset.

cat("\n", strrep("-", 100), "\n", "DICTAMEN\n", strrep("-", 100), "\n", sep = "")

cat("El Dataset Científico Depurado fue construido correctamente.\n")
cat("Se recomienda utilizar 'dataset_gnn_certificado_v2.parquet' únicamente si la auditoría demuestra que las variables eliminadas son realmente redundantes.\n")

cat("\nBloque 15 finalizado correctamente.\n")

# BLOQUE 16. Comparación del Dataset Original vs Dataset Depurado ----------
## Objetivo: Comparar el desempeño predictivo del Dataset Original y del
# Dataset Científico Depurado utilizando el mismo modelo estadístico.
### Entradas: - dataset_auditoria - dataset_depurado
### Producto: - Comparación de desempeño
### Responde: ¿La depuración del dataset mejora el desempeño predictivo?

# 16.1. Preparación --------------------------------------------------------
## Objetivo: Preparar ambos datasets.

dataset_original <- dataset_auditoria |>
  dplyr::select(where(is.numeric)) |>
  tidyr::drop_na() # Dataset original

dataset_final <- dataset_depurado |>
  dplyr::select(where(is.numeric)) |>
  tidyr::drop_na() # Dataset depurado

# 16.2. Ajuste de modelos --------------------------------------------------
## Objetivo: Ajustar la regresión lineal.

modelo_original <- lm(
  log_rendimiento ~ .,
  data = dataset_original
) # Modelo original

modelo_final <- lm(
  log_rendimiento ~ .,
  data = dataset_final
) # Modelo depurado

# 16.3. Predicciones -------------------------------------------------------
## Objetivo: Obtener las predicciones.

pred_original <- predict(modelo_original) # Predicciones originales

pred_final <- predict(modelo_final) # Predicciones depuradas

# 16.4. Métricas -----------------------------------------------------------
## Objetivo: Calcular las métricas oficiales.

evaluar_modelo <- function(y, pred){
  
  tibble(
    
    RMSE = sqrt(mean((y - pred)^2)),
    
    MAE = mean(abs(y - pred)),
    
    R2 = cor(y, pred)^2
    
  )
  
} # Función de evaluación

metricas_original <- evaluar_modelo(
  dataset_original$log_rendimiento,
  pred_original
) |>
  mutate(dataset = "Original")

metricas_final <- evaluar_modelo(
  dataset_final$log_rendimiento,
  pred_final
) |>
  mutate(dataset = "Depurado")

comparacion <- bind_rows(
  metricas_original,
  metricas_final
) # Comparación

# 16.5. Diferencias --------------------------------------------------------
## Objetivo: Calcular la variación entre datasets.

diferencias <- tibble(
  
  Delta_R2 = metricas_final$R2 - metricas_original$R2,
  
  Delta_RMSE = metricas_final$RMSE - metricas_original$RMSE,
  
  Delta_MAE = metricas_final$MAE - metricas_original$MAE
  
)

# 16.6. Resultados ---------------------------------------------------------
## Objetivo: Mostrar la comparación.

cat("\n", strrep("=", 100), "\n", "COMPARACIÓN DEL DATASET\n", strrep("=", 100), "\n", sep = "")

print(comparacion)

cat("\n", strrep("-", 100), "\n", "DIFERENCIAS\n", strrep("-", 100), "\n", sep = "")

print(diferencias)

# 16.7. Dictamen -----------------------------------------------------------
## Objetivo: Emitir el dictamen metodológico.

cat("\n", strrep("-", 100), "\n", "DICTAMEN\n", strrep("-", 100), "\n", sep = "")

if(abs(diferencias$Delta_R2) < 0.01){
  
  cat("La depuración no modifica significativamente el desempeño.\n")
  cat("Se recomienda utilizar el Dataset Científico Depurado.\n")
  
}else if(diferencias$Delta_R2 > 0){
  
  cat("El Dataset Científico Depurado mejora el desempeño predictivo.\n")
  
}else{
  
  cat("La depuración reduce el desempeño predictivo.\n")
  cat("Se recomienda revisar las variables eliminadas.\n")
  
}

# 16.8. Exportación --------------------------------------------------------
## Objetivo: Exportar los resultados.

ruta_salida <- here("reports", "auditoria_dataset") # Carpeta de resultados

dir.create(ruta_salida, recursive = TRUE, showWarnings = FALSE) # Crear carpeta

write.csv(
  comparacion,
  file.path(ruta_salida, "comparacion_dataset_original_vs_depurado.csv"),
  row.names = FALSE
) # Exportar comparación

write.csv(
  diferencias,
  file.path(ruta_salida, "diferencias_dataset_original_vs_depurado.csv"),
  row.names = FALSE
) # Exportar diferencias

cat("\nBloque 16 finalizado correctamente.\n")

