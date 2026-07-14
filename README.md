\# Arquitectura del Sistema Inteligente GeoAI



\## Descripción



Este proyecto implementa una arquitectura científica reproducible para el modelado espacio-temporal de la soberanía alimentaria mediante Inteligencia Artificial.



La metodología integra datos abiertos agrícolas, climáticos y espaciales para construir un \*\*Dataset Científico Certificado\*\*, comparar objetivamente múltiples paradigmas de Inteligencia Artificial mediante un \*\*Benchmark Científico Reproducible\*\*, seleccionar el modelo con mejor desempeño predictivo y generar escenarios prospectivos que apoyen la toma de decisiones mediante una plataforma \*\*GeoAI\*\*.



\---



\# Filosofía de la Arquitectura



La arquitectura está diseñada bajo seis principios metodológicos:



\- Reproducibilidad

\- Modularidad

\- Benchmark Científico

\- Reutilización

\- No sobreingeniería

\- Separación entre investigación científica y aplicación



Cada fase genera un producto oficial que puede reutilizarse sin repetir el procesamiento completo. La plataforma GeoAI consume exclusivamente estos productos certificados y nunca recalcula modelos científicos.



\---



\# Arquitectura General



```text

&#x20;                   DATOS ABIERTOS



&#x20;      EVA      CNA      Irrigación      DIVIPOLA



&#x20;                         +



&#x20;               ERA5        CHIRPS



&#x20;                         +



&#x20;             Cartografía Municipal



&#x20;                         │

&#x20;                         ▼



────────────────────────────────────────────────────────────



&#x20;                   PIPELINE R



&#x20;Auditoría

&#x20;     ↓

&#x20;Limpieza

&#x20;     ↓

&#x20;Homologación

&#x20;     ↓

&#x20;Transformación Espacial

&#x20;     ↓

&#x20;Integración (DIVIPOLA + Datos Espaciales)

&#x20;     ↓

&#x20;Control de Calidad

&#x20;     ↓



&#x20;Dataset Científico Certificado



────────────────────────────────────────────────────────────



&#x20;                PIPELINE PYTHON



01\_build\_graph.py

&#x20;       ↓

02\_benchmark.py

&#x20;       ↓

03\_train\_model.py

&#x20;       ↓

04\_evaluation.py

&#x20;       ↓

05\_forecasting.py

&#x20;       ↓

06\_geoai.py



────────────────────────────────────────────────────────────



&#x20;        PLATAFORMA INTELIGENTE GEOAI



&#x20;Dashboard

&#x20;API

&#x20;Agentes IA

&#x20;Reportes

&#x20;Mapas

```



La arquitectura establece una separación estricta entre el \*\*Pipeline Científico\*\* y la \*\*Plataforma GeoAI\*\*, permitiendo mantener independencia entre investigación, entrenamiento y consumo de resultados.



\---



\# Flujo Metodológico



\## Fase 0. Ingeniería de Datos



\### Objetivo



Construir un Dataset Científico Certificado mediante la integración de múltiples fuentes heterogéneas.



\### Entradas



\- EVA

\- CNA

\- Distritos de Irrigación

\- DIVIPOLA

\- ERA5

\- CHIRPS

\- Cartografía Municipal



\### Producto



```text

dataset\_gnn\_certificado.parquet

```



Esta fase concentra:



\- Auditoría

\- Limpieza

\- Normalización

\- Homologación

\- Transformación Espacial

\- Integración de datos

\- Control de Calidad



Ninguna fase posterior modifica este conjunto de datos.



\---



\## Fase 1. Construcción del Grafo



\### Script



```text

01\_build\_graph.py

```



\### Responsabilidad



Transformar el Dataset Científico en un objeto \*\*GraphData\*\* que represente las relaciones espaciales entre municipios.



\### Flujo



```text

Dataset Científico

&#x20;       ↓

Validación

&#x20;       ↓

Construcción de Nodos

&#x20;       ↓

Construcción de Aristas

&#x20;       ↓

Node Features

&#x20;       ↓

Variable Objetivo

&#x20;       ↓

Train / Validation / Test

&#x20;       ↓

GraphData

&#x20;       ↓

Validación

&#x20;       ↓

graph\_data.pt

```



\### Producto



```text

graph\_data.pt

```



Esta fase únicamente construye el grafo.



No realiza:



\- Entrenamiento

\- Benchmark

\- Evaluación

\- Forecasting



\---



\## Fase 2. Benchmark Científico



\### Script



```text

02\_benchmark.py

```



\### Responsabilidad



Comparar objetivamente diferentes paradigmas de Inteligencia Artificial bajo exactamente el mismo protocolo experimental.



\### Paradigmas evaluados



\#### Modelos Estadísticos



\- Regresión Lineal



\#### Machine Learning



\- Random Forest

\- XGBoost

\- LightGBM

\- CatBoost



\#### Deep Learning



\- MLP



\#### Graph Neural Networks



\- GCN

\- GraphSAGE

\- GAT

\- GIN

\- TAGCN



\### Flujo



```text

GraphData

&#x20;     ↓

Preparación Experimental

&#x20;     ↓

Modelos Estadísticos

&#x20;     ↓

Machine Learning

&#x20;     ↓

Deep Learning

&#x20;     ↓

Graph Neural Networks

&#x20;     ↓

Evaluación

&#x20;     ↓

Ranking

&#x20;     ↓

Modelo Ganador

```



\### Productos



```text

benchmark\_results.joblib

benchmark\_summary.csv

benchmark\_metrics.parquet

benchmark\_ranking.csv

best\_model\_config.json

```



La metodología \*\*no presupone\*\* que las Graph Neural Networks sean el modelo ganador.



Cualquier algoritmo podrá ser seleccionado siempre que obtenga el mejor desempeño bajo el protocolo experimental definido.



\---



\## Fase 3. Entrenamiento Definitivo



\### Script



```text

03\_train\_model.py

```



\### Responsabilidad



Entrenar exclusivamente el modelo ganador utilizando la configuración óptima obtenida durante el benchmark.



\### Flujo



```text

Configuración Ganadora

&#x20;       ↓

Modelo Ganador

&#x20;       ↓

Carga del Dataset

&#x20;       ↓

Carga del GraphData

&#x20;       ↓

Entrenamiento

&#x20;       ↓

Checkpoints

&#x20;       ↓

Modelo Oficial

```



\### Productos



```text

best\_model.joblib

best\_model.pt

best\_model\_metadata.json

```



Esta fase garantiza que el Modelo Oficial corresponde exactamente al seleccionado por el benchmark.



\---



\## Fase 4. Evaluación Científica



\### Script



```text

04\_evaluation.py

```



\### Responsabilidad



Evaluar el Modelo Oficial y generar resultados científicos reproducibles.



\### Flujo



```text

Modelo Oficial

&#x20;     ↓

Inferencia

&#x20;     ↓

Evaluación

&#x20;     ↓

Validación

&#x20;     ↓

Residuos

&#x20;     ↓

Feature Importance

&#x20;     ↓

SHAP

&#x20;     ↓

Embeddings

&#x20;     ↓

Visualizaciones

&#x20;     ↓

Resultados Oficiales

&#x20;     ↓

Reportes

```



\### Productos



```text

evaluation\_results.parquet

validation\_results.parquet

feature\_importance.parquet

residuals.parquet

shap\_values.joblib

gnn\_embeddings.pt

evaluation\_report.json

```



Esta fase únicamente evalúa.



No realiza:



\- Entrenamiento

\- Benchmark

\- Forecasting



\---



\## Fase 5. Forecasting Espacio-Temporal



\### Script



```text

05\_forecasting.py

```



\### Responsabilidad



Generar escenarios prospectivos utilizando exclusivamente el Modelo Oficial.



\### Flujo



```text

Modelo Oficial

&#x20;     ↓

Carga del Modelo

&#x20;     ↓

Horizonte Temporal

&#x20;     ↓

Construcción de Escenarios

&#x20;     ↓

Predicciones Futuras

&#x20;     ↓

Intervalos de Confianza

&#x20;     ↓

Panel Forecast

&#x20;     ↓

Mapas

&#x20;     ↓

Exportación

```



\### Productos



```text

forecast\_panel.parquet

forecast\_summary.json

forecast\_maps.parquet

```



El forecasting utiliza únicamente resultados previamente certificados y nunca reentrena el modelo.



\---



\## Fase 6. Plataforma Inteligente GeoAI



\### Script Principal



```text

06\_geoai.py

```



La Plataforma GeoAI constituye la capa de aplicación del proyecto.



Su función es consumir únicamente los productos oficiales generados por el Pipeline Científico.



\### Componentes



\- Dashboard

\- API

\- Agentes Inteligentes

\- Reportes

\- Mapas Interactivos



Los agentes especializados nunca entrenan ni recalculan modelos; únicamente consultan los artefactos oficiales.



\### Arquitectura Interna



```text

06\_geoai.py



agents/

│

├── orchestrator.py

├── prediction\_agent.py

├── forecast\_agent.py

├── explainability\_agent.py

├── recommendation\_agent.py

└── report\_agent.py



dashboard/

│

└── dashboard.py



api/

│

└── main.py



services/

│

├── prediction\_service.py

├── forecast\_service.py

├── report\_service.py

└── map\_service.py



views/

│

├── maps.py

├── indicators.py

├── benchmark.py

└── forecast.py

```



\---



\# Orden Oficial de Ejecución



| Paso | Componente | Producto Principal |

|------|------------|-------------------|

| 0 | Pipeline R | Dataset Científico Certificado |

| 1 | 01\_build\_graph.py | graph\_data.pt |

| 2 | 02\_benchmark.py | Ranking y Modelo Ganador |

| 3 | 03\_train\_model.py | Modelo Oficial |

| 4 | 04\_evaluation.py | Evaluación Científica |

| 5 | 05\_forecasting.py | Escenarios Prospectivos |

| 6 | 06\_geoai.py | Dashboard, API y Agentes IA |



Cada etapa depende exclusivamente de los productos generados por la etapa anterior, garantizando:



\- Reproducibilidad

\- Modularidad

\- Trazabilidad

\- Escalabilidad

\- Desacoplamiento entre componentes



\---



\# Principales Productos Generados



| Fase | Producto |

|------|----------|

| Ingeniería de Datos | dataset\_gnn\_certificado.parquet |

| Construcción del Grafo | graph\_data.pt |

| Benchmark Científico | benchmark\_ranking.csv |

| Entrenamiento Definitivo | best\_model.pt |

| Evaluación Científica | evaluation\_results.parquet |

| Forecasting | forecast\_panel.parquet |

| Plataforma GeoAI | Dashboard, API y Agentes IA |



\---



\# Valor Científico de la Arquitectura



La principal fortaleza de esta arquitectura no reside en utilizar una tecnología específica, sino en implementar un proceso científico reproducible para seleccionar objetivamente el modelo de Inteligencia Artificial más adecuado para el problema estudiado.



Al integrar datos abiertos agrícolas, información climática y relaciones espaciales dentro de un pipeline modular, la metodología permite incorporar nuevos algoritmos sin modificar la arquitectura general del proyecto.



La separación entre Ingeniería de Datos, Benchmark Científico, Entrenamiento, Evaluación, Forecasting y Plataforma GeoAI garantiza que cada componente tenga una única responsabilidad, facilitando la reproducibilidad, la auditabilidad y la reutilización de los resultados.



Esta arquitectura también permite extender la metodología hacia otros dominios donde el análisis espacio-temporal resulta fundamental, como:



\- Cambio climático

\- Recursos hídricos

\- Salud pública

\- Movilidad

\- Ordenamiento territorial

\- Gestión del riesgo

\- Agricultura de precisión



De esta forma, el proyecto constituye una plataforma científica escalable para el desarrollo de Sistemas Inteligentes GeoAI basados en datos abiertos e Inteligencia Artificial.

