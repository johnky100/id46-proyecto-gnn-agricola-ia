# AVANZADO IA

## Plataforma GeoAI para el modelado espacial y temporal mediante Graph Neural Networks

## 1. Propósito del README

Este documento orienta al jurado, evaluadores y colaboradores técnicos en la comprensión, navegación y revisión del proyecto avanzado ia.

El objetivo principal no es únicamente describir los archivos del repositorio, sino explicar la responsabilidad científica y técnica de cada componente, la relación entre las diferentes capas y el recorrido que siguen los datos desde su integración hasta su utilización dentro de la plataforma GeoAI.

El proyecto está organizado como un pipeline científico reproducible que separa la ingeniería de datos, la construcción del grafo, el análisis científico, el benchmark, el entrenamiento, la evaluación, el forecasting y la plataforma GeoAI.

La arquitectura documentada establece una separación funcional entre la ingeniería de datos desarrollada en R y la modelación y plataforma desarrolladas principalmente en Python.

## 2. Cómo debe entenderse el proyecto

El proyecto debe interpretarse como un sistema científico compuesto por varias capas.

La secuencia principal es:

```text
Fuentes de datos
    |
    v
Ingeniería de datos en R
    |
    v
Dataset científico
    |
    v
Construcción del grafo
    |
    v
Colección de GraphData
    |
    v
Análisis del grafo
    |
    v
Benchmark científico
    |
    v
Selección del modelo GNN
    |
    v
GraphSAGE
    |
    v
Entrenamiento
    |
    v
Evaluación
    |
    v
Explainability
    |
    v
Forecasting
    |
    v
GeoAI
    |
    +-------------------+
    |                   |
    v                   v
Dashboard             Agent
    |                   |
    +---------+---------+
              |
              v
             API
              |
              v
Sistema de apoyo a la decisión
```

Esta cadena corresponde a la arquitectura científica final definida para el proyecto.

## 3. Principio fundamental de arquitectura

Una característica fundamental del proyecto es la separación de responsabilidades.

R y Python no realizan la misma función.

R constituye la capa principal de ingeniería y preparación científica de los datos.

Python constituye la capa principal de procesamiento del grafo, análisis, modelación, evaluación, forecasting y plataforma GeoAI.

La frontera conceptual es:

```text
R
 |
 v
Data Engineering
 |
 v
Dataset científico
 |
 v
Python
```

Esta separación permite mantener una arquitectura modular y evitar que la preparación de los datos quede mezclada con la lógica de modelación.

## 4. Estructura general del repositorio

La estructura principal del proyecto es:

```text
avanzado-ia/
|
+-- app/
+-- catboost_info/
+-- data/
+-- docs/
+-- logs/
+-- recursos/
+-- reports/
+-- scripts/
+-- src/
|   +-- config/
|   +-- python/
|   +-- R/
|
+-- .gitignore
+-- .Rhistory
+-- .RData
+-- .RDataTmp
+-- .gitmodules
+-- desktop.ini
+-- proyecto-gnn-agricola.Rproj
+-- pyproject.toml
+-- README.md
```

La estructura completa del repositorio, incluyendo los componentes internos de Python y R, está documentada en la arquitectura suministrada.

## 5. Ruta recomendada para el jurado

Para comprender el proyecto de manera eficiente, no se recomienda comenzar revisando archivos individuales de forma aleatoria.

La ruta recomendada es:

```text
1. README.md
       |
       v
2. docs/
       |
       v
3. src/R/
       |
       v
4. src/python/graph/
       |
       v
5. src/python/models/
       |
       v
6. 02_benchmark.py
       |
       v
7. 03_train_model.py
       |
       v
8. 04_evaluation.py
       |
       v
9. explainability.py
       |
       v
10. forecasting.py
       |
       v
11. src/python/geoai/
       |
       v
12. src/python/outputs/
       |
       v
13. reports/
```

Esta ruta sigue la lógica científica del sistema y permite pasar desde los datos hasta los resultados y la aplicación.

# 6. Capa de datos

La carpeta `data/` organiza los datos según su nivel de procesamiento.

```text
data/
|
+-- raw/
+-- interim/
+-- processed/
```

## raw

Contiene los datos originales sin transformación.

Su responsabilidad es conservar la fuente inicial utilizada por el proceso de ingeniería de datos.

## interim

Contiene productos intermedios generados durante el procesamiento.

## processed

Contiene los datos procesados que sirven como insumo para las etapas posteriores.

Esta separación permite distinguir claramente entre datos originales, productos intermedios y datos preparados para análisis y modelación.

# 7. Ingeniería de datos en R

La ingeniería de datos se encuentra en:

```text
src/R/
|
+-- audit/
+-- audit_raw/
+-- backup/
+-- cna/
+-- config/
+-- divipola/
+-- era5/
+-- eva/
+-- ingestion_chirps/
+-- irrigacion/
+-- panel_maestro_gnn/
+-- precio_tierra/
+-- prueba_analisis.R
+-- prueba.R
+-- renom_db.R
```

La arquitectura identifica esta capa como responsable de la integración y preparación del panel científico.

Entre sus responsabilidades documentadas se encuentran:

```text
Integración de fuentes
Homologación
Control de calidad
Integración territorial
Integración temporal
Construcción del panel maestro
```

El resultado conceptual de esta etapa es:

```text
Fuentes de datos
       |
       v
Data Engineering
       |
       v
Panel maestro
       |
       v
Dataset científico
```

El proyecto utiliza una integración territorial asociada a DIVIPOLA y contempla fuentes relacionadas con CNA, ERA5, EVA, CHIRPS, irrigación y precio de tierra dentro de esta estructura.

# 8. Capa de configuración

La configuración centralizada se encuentra en:

```text
src/python/config/
|
+-- __init__.py
+-- config_project.py
+-- dashboard_config.py
+-- figure_config.py
+-- paths.py
```

La arquitectura identifica esta capa como transversal al sistema.

Sus componentes documentados son:

```text
config_project.py
    Parámetros generales y configuración científica

dashboard_config.py
    Configuración específica del dashboard

figure_config.py
    Configuración de figuras y visualizaciones

paths.py
    Definición centralizada de rutas
```

Conceptualmente:

```text
Configuración central
       |
       +----------------+
       |                |
       v                v
Project Config        Paths
       |
       v
Figure Config
       |
       v
Componentes Python
```

La finalidad arquitectónica es evitar que cada componente mantenga de forma independiente sus parámetros generales y rutas.

# 9. Preprocesamiento

La carpeta correspondiente es:

```text
src/python/preprocessing/
|
+-- __init__.py
```

La documentación disponible únicamente permite afirmar que esta capa se encuentra antes de la construcción del grafo.

Por tanto, no deben atribuirse responsabilidades adicionales sin revisar el código interno.

Su posición en el flujo es:

```text
Dataset científico
       |
       v
Preprocessing
       |
       v
Graph Construction
```

Esta precaución es importante para mantener la trazabilidad entre lo que está documentado y lo que realmente está implementado.

# 10. Construcción del grafo

La construcción del grafo constituye una de las capas centrales del proyecto.

Se encuentra en:

```text
src/python/graph/
|
+-- builders/
|   +-- build_combined_edges.py
|   +-- build_dynamic_edges.py
|   +-- build_features.py
|   +-- build_graphdata.py
|   +-- build_node_catalog.py
|   +-- build_spatial_edges.py
|   +-- prepare_year_dataset.py
|
+-- 01_build_graph.py
+-- 01.1_graph_analysis.py
+-- 01.2_temporal_maps.py
+-- 01.3_climate_maps.py
+-- 02_benchmark.py
+-- 03_train_model.py
+-- 04_evaluation.py
+-- audit_graph_collection.py
+-- comprimir_atlas.py
+-- forecasting.py
+-- geoai.py
+-- graph_pipeline.py
```

Los builders tienen responsabilidades diferenciadas:

```text
prepare_year_dataset.py
    Preparación del dataset correspondiente a cada año

build_node_catalog.py
    Construcción del catálogo de nodos

build_features.py
    Construcción de características de los nodos

build_spatial_edges.py
    Construcción de relaciones espaciales

build_dynamic_edges.py
    Construcción de relaciones dinámicas

build_combined_edges.py
    Construcción de aristas combinadas

build_graphdata.py
    Construcción de objetos GraphData
```

La arquitectura interna se resume en:

```text
Dataset por año
       |
       v
prepare_year_dataset.py
       |
       +----------------------+
       |                      |
       v                      v
build_node_catalog.py    build_features.py
       |                      |
       +----------+-----------+
                  |
                  v
        Construcción de aristas
                  |
        +---------+---------+
        |         |         |
        v         v         v
     Spatial   Dynamic   Combined
      Edges     Edges     Edges
        |         |         |
        +---------+---------+
                  |
                  v
       build_graphdata.py
                  |
                  v
               GraphData
```

Esta modularización permite separar la construcción de nodos, características y relaciones antes de formar el objeto GraphData.

# 11. Dimensión temporal del GraphData

El proyecto no se limita a un único grafo estático.

La arquitectura representa una colección temporal:

```text
Dataset 2006
    |
    v
GraphData 2006

Dataset 2007
    |
    v
GraphData 2007

Dataset 2008
    |
    v
GraphData 2008

...

Dataset 2018
    |
    v
GraphData 2018
```

Conceptualmente:

```text
G2006
G2007
G2008
...
G2018
```

Cada elemento representa el estado del sistema para un año determinado.

Por esta razón, la estructura incorpora explícitamente una dimensión temporal además de la dimensión espacial.

# 12. Análisis científico del grafo

Después de construir los GraphData se ejecutan componentes destinados al análisis estructural y científico.

Los principales archivos son:

```text
01.1_graph_analysis.py
01.2_temporal_maps.py
01.3_climate_maps.py
```

El flujo conceptual es:

```text
GraphData
    |
    +------------------+
    |                  |
    v                  v
Graph Analysis      Temporal Maps
    |
    v
Climate Maps
```

La documentación identifica como dimensiones de análisis:

```text
Estructura
Topología
Relaciones
Similitud
Dimensión espacial
Dimensión temporal
Dimensión climática
```

Esta capa genera productos asociados con análisis municipal, similitud espacial, similitud temporal y relaciones con la variable objetivo.

# 13. Sistema de análisis científico

Los productos persistidos incluyen, entre otros:

```text
department_similarity_summary.parquet
domain_similarity_summary.parquet
domain_target_correlations.parquet
municipality_analysis_complete.csv
municipality_analysis_complete.parquet
municipality_similarity_persistence.parquet
municipality_summary.parquet
municipality_temporal_stability.parquet
persistent_similar_municipal_pairs.parquet
similarity_target_analysis.parquet
similarity_target_summary.parquet
spatial_similarity_correlation.csv
temporal_similarity_evolution.parquet
temporal_similarity_summary.parquet
temporal_similarity_target_analysis.parquet
temporal_target_correlations.parquet
top_different_municipal_pairs.parquet
top_similar_municipal_pairs.parquet
topology_similarity_correlation.csv
```

Estos productos muestran que el sistema no se limita al entrenamiento de un modelo.

Existe una capa analítica orientada a caracterizar municipios, relaciones espaciales, relaciones temporales, similitudes y asociaciones con la variable objetivo.

# 14. Benchmark científico

El benchmark está asociado con:

```text
02_benchmark.py
```

Las familias de modelos están organizadas en:

```text
src/python/models/
|
+-- statistical.py
+-- machine_learning.py
+-- deep_learning.py
+-- graph_neural_networks.py
+-- spatio_temporal_gnn.py
```

La arquitectura contempla:

```text
Dataset
    |
    v
Benchmark
    |
    +---------------------+
    |                     |
    v                     v
Statistical           Machine Learning
    |                     |
    v                     v
Linear Regression     RF
                      XGBoost
                      LightGBM
                      CatBoost

Deep Learning
    |
    v
MLP

Graph Neural Networks
    |
    +--------------------------+
    |            |             |
    v            v             v
   GCN       GraphSAGE         GAT
                 |
                 +-------------+
                 |
                 v
              GIN
              TAGCN
```

La arquitectura también contempla explícitamente una estructura para GNN espacio temporales mediante `spatio_temporal_gnn.py`.

# 15. Interpretación correcta del benchmark

Una distinción fundamental para la evaluación del proyecto es la diferencia entre mejor modelo global y modelo oficial.

El benchmark permite realizar una comparación global y una comparación dentro de la familia GNN.

La arquitectura documentada es:

```text
Benchmark
    |
    +-----------------------+
    |                       |
    v                       v
Mejor global             Mejor GNN
    |                       |
    v                       v
Linear Regression        GraphSAGE
                            |
                            v
                     Modelo oficial
```

Por tanto, no debe afirmarse que GraphSAGE fue necesariamente el modelo con mejor desempeño global.

La documentación establece que la selección del modelo oficial ocurre dentro del contexto GNN. Esta distinción evita confundir el resultado estadístico del benchmark con la decisión arquitectónica del proyecto.

# 16. Entrenamiento

El entrenamiento está asociado con:

```text
03_train_model.py
```

Su responsabilidad arquitectónica se representa como:

```text
Modelo seleccionado
       |
       v
03_train_model.py
       |
       +----------------------+
       |          |           |
       v          v           v
Configuración  Training      Loss
       |          |           |
       +----------+-----------+
                  |
                  v
           Modelo entrenado
```

El archivo constituye la etapa posterior a la selección del modelo y anterior a la evaluación.

# 17. Evaluación científica

La evaluación está asociada con:

```text
04_evaluation.py
```

Su función arquitectónica es analizar el modelo entrenado mediante:

```text
Métricas
Diagnósticos
Validaciones
```

El flujo es:

```text
Modelo entrenado
       |
       v
04_evaluation.py
       |
       +-----------------------+
       |           |           |
       v           v           v
Métricas     Diagnósticos   Validaciones
       |           |           |
       +-----------+-----------+
                   |
                   v
          Resultados evaluación
```

Los resultados se separan entre:

```text
outputs/evaluation/
reports/evaluation/
```

La existencia de ambas rutas permite diferenciar la persistencia de resultados de la generación de reportes.

# 18. Auditoría de la colección de grafos

La arquitectura incorpora:

```text
audit_graph_collection.py
```

Su propósito documentado es establecer una capa de auditoría sobre la colección de GraphData.

Conceptualmente:

```text
Graph Collection
       |
       v
audit_graph_collection.py
       |
       v
Auditoría
       |
       +-------------------------+
       |            |            |
       v            v            v
Integridad    Consistencia   Trazabilidad
       |            |            |
       +------------+------------+
                    |
                    v
                  Audits
```

Esta capa es especialmente relevante desde la perspectiva de una tesis porque permite asociar el proceso de modelación con mecanismos de control de calidad y reproducibilidad.

# 19. Explainability

La explicabilidad está implementada en:

```text
src/python/analysis/explainability.py
```

Su posición conceptual es:

```text
Modelo
   |
   v
Explainability
   |
   v
Interpretación
   |
   v
GeoAI / Reports
```

Debe diferenciarse de la evaluación.

La evaluación responde principalmente a:

```text
Qué tan bien predice el modelo
```

La explainability responde:

```text
Por qué produce determinada predicción
```

Estas dos responsabilidades no deben mezclarse conceptualmente.

# 20. Forecasting

El proyecto ya contiene un componente destinado al forecasting:

```text
src/python/graph/forecasting.py
```

También existen directorios de resultados asociados:

```text
outputs/forecast/
outputs/forecasting/
```

Por tanto, forecasting forma parte de la arquitectura existente y no debe crearse como una arquitectura paralela.

La cadena documentada es:

```text
Modelo validado
       |
       v
forecasting.py
       |
       v
Forecasting
       |
       v
Future Predictions
```

La arquitectura conceptual contempla dimensiones temporal, espacial y de escenarios. Sin embargo, la metodología concreta implementada por `forecasting.py` no debe inferirse solamente a partir del nombre del archivo.

Para documentar con precisión el método de forecasting se debe revisar el código y el contrato:

```text
docs/forecasting_spec.md
```

Esta precaución forma parte de la política de trazabilidad científica del proyecto.

# 21. Plataforma GeoAI

La plataforma se encuentra en:

```text
src/python/geoai/
|
+-- adaptation_library/
+-- agents/
+-- api/
+-- components/
+-- dashboard/
+-- knowledge_/
+-- prompts/
+-- services/
+-- 08_api.py
+-- 09_agent.py
+-- 10_reports.py
+-- 11_export.py
+-- app.py
+-- logo_geoai.png
```

La arquitectura conceptual es:

```text
GeoAI
   |
   +-----------------------------+
   |              |              |
   v              v              v
Services      Components      Knowledge
   |              |              |
   +--------------+--------------+
                  |
       +----------+----------+
       |          |          |
       v          v          v
   Dashboard    API        Agent
       |          |          |
       +----------+----------+
                  |
                  v
         Decision Support
```

La responsabilidad de esta capa es transformar los resultados científicos previamente generados en componentes utilizables por la plataforma.

# 22. Knowledge Base, Adaptation Library y Prompts

La arquitectura diferencia tres conceptos:

```text
Knowledge
    Información que utiliza el sistema

Adaptation
    Mecanismos para adaptar el conocimiento o comportamiento

Prompts
    Instrucciones utilizadas por los componentes correspondientes
```

Estos componentes aparecen en:

```text
src/python/knowledge_base/
src/python/geoai/adaptation_library/
src/python/geoai/knowledge_/
src/python/geoai/prompts/
```

La implementación concreta de cada mecanismo debe verificarse directamente en el código antes de atribuirle funciones más específicas.

# 23. Dashboard, API y Agent

La capa de aplicación incluye:

```text
08_api.py
09_agent.py
10_reports.py
11_export.py
app.py
```

La arquitectura los integra como componentes de la plataforma GeoAI:

```text
GeoAI
   |
   +-------------+-------------+
   |             |             |
   v             v             v
Dashboard       API          Agent
   |             |             |
   +-------------+-------------+
                 |
                 v
              Services
                 |
                 v
        Scientific Results
                 |
        +--------+--------+
        |                 |
        v                 v
     Reports           Export
```

Esta separación permite distinguir la interfaz científica, los servicios de acceso, los componentes inteligentes, la generación de reportes y la exportación.

# 24. Documentación contractual

La carpeta `docs/` contiene especificaciones asociadas con componentes críticos:

```text
docs/
|
+-- api_spec.md
+-- dashboard_spec.md
+-- explainability_spec.md
+-- forecasting_spec.md
+-- graphdata_spec.md
+-- Archivos que componen el proyecto.xlsx
```

Estas especificaciones son relevantes porque permiten complementar la lectura del código con una descripción formal de las responsabilidades esperadas de los componentes.

La arquitectura identifica especialmente los contratos de GraphData, Forecasting, Explainability, API y Dashboard.

Para el jurado, esta capa representa una evidencia importante de que el sistema no depende exclusivamente de la implementación de código.

# 25. Sistema de outputs

Los resultados generados por el pipeline se organizan en:

```text
src/python/outputs/
|
+-- analysis/
+-- atlas_climatico_2006_2018/
+-- atlas_climatico_2006_2018_comprimido/
+-- atlas_temporal/
+-- audits/
+-- benchmark/
+-- certificates/
+-- contracts/
+-- evaluation/
+-- figures/
+-- forecast/
+-- geoai/
+-- graph_data/
+-- metadata/
+-- models/
+-- training/
```

La función de esta estructura es separar los productos generados por las distintas etapas.

La relación principal es:

```text
Pipeline
   |
   +----------------+----------------+
   |                |                |
   v                v                v
Analysis        Benchmark        Training
   |                |                |
   +----------------+----------------+
                    |
                    v
                Evaluation
                    |
                    v
                Forecasting
                    |
                    v
                  GeoAI
```

Además, `audits`, `certificates`, `contracts` y `metadata` forman una capa de gobernanza científica.

# 26. Gobernanza científica

La arquitectura incluye una capa transversal:

```text
Gobernanza
    |
    +-----------------------------+
    |              |              |
    v              v              v
Logs           Audits       Certificates
    |              |              |
    +--------------+--------------+
                   |
                   v
                Metadata
                   |
                   v
             Reproducibilidad
                   |
                   v
             Trazabilidad total
```

Esta capa no constituye una etapa aislada del pipeline.

Atraviesa el sistema y permite relacionar ejecución, auditoría, certificación y metadatos con los resultados científicos.

# 27. Logs y observabilidad

El proyecto contiene:

```text
logs/
|
+-- geoai.log
```

Este archivo corresponde al registro de eventos de la plataforma GeoAI.

Conceptualmente:

```text
Ejecución
    |
    v
Logs
    |
    v
geoai.log
```

Los logs forman parte de la observabilidad y trazabilidad del sistema.

# 28. Reports

El proyecto posee una carpeta de reportes en la raíz:

```text
reports/
|
+-- auditoria_dataset/
+-- evaluation/
+-- figures/
+-- graph_visualization/
+-- inventario_datasets/
```

También existe el sistema de outputs dentro de:

```text
src/python/outputs/
```

y un componente:

```text
10_reports.py
```

Por tanto, debe distinguirse entre la generación de resultados, la persistencia de outputs y la organización de reportes para consulta.

La arquitectura conceptual es:

```text
Resultados
    |
    v
Reporting
    |
    +-----------------------------+
    |              |              |
    v              v              v
Figures         Reports         Export
```

# 29. Visualización y atlas

El proyecto contempla productos de visualización y atlas:

```text
atlas_climatico_2006_2018/
atlas_climatico_2006_2018_comprimido/
atlas_temporal/
figures/
```

La arquitectura establece:

```text
Datos científicos
       |
       +------------------+
       |                  |
       v                  v
Visualización           Atlas
       |                  |
       v                  v
   figures/         Atlas temporal
                    Atlas climático
```

Estos productos permiten transformar resultados científicos en representaciones visuales y productos de consulta.

# 30. Aplicación web

La carpeta:

```text
app/
|
+-- assets/
    |
    +-- css/
    +-- js/
```

representa la capa de presentación de la aplicación.

Su relación arquitectónica es:

```text
GeoAI Backend
       |
       v
      App
       |
       +----------+
       |          |
       v          v
      CSS        JS
       |          |
       +----------+
             |
             v
       Interfaz de usuario
```

La arquitectura distingue esta capa de presentación de la lógica científica y del backend GeoAI.

# 31. Artefactos CatBoost

El repositorio contiene:

```text
catboost_info/
|
+-- learn/
+-- tmp/
+-- catboost_training.json
+-- learn_error.tsv
+-- time_left.tsv
```

Estos archivos corresponden a artefactos generados durante el entrenamiento de CatBoost.

No deben interpretarse como parte del núcleo arquitectónico de la plataforma.

Su ubicación conceptual es:

```text
Benchmark
    |
    v
CatBoost
    |
    v
catboost_info/
```

La documentación señala expresamente que estos artefactos no deben confundirse con la arquitectura científica principal.

# 32. Responsabilidades de las principales etapas

La siguiente tabla permite al jurado identificar rápidamente qué componente debe consultar.

| Etapa                 | Componente principal        | Responsabilidad                                |
| --------------------- | --------------------------- | ---------------------------------------------- |
| Datos                 | `data/`                     | Organización de datos raw, interim y processed |
| Ingeniería de datos   | `src/R/`                    | Integración y preparación del panel            |
| Configuración         | `config_project.py`         | Configuración general                          |
| Rutas                 | `paths.py`                  | Centralización de rutas                        |
| Preprocesamiento      | `preprocessing/`            | Preparación previa al grafo                    |
| Construcción          | `01_build_graph.py`         | Orquestación de construcción                   |
| Nodos                 | `build_node_catalog.py`     | Catálogo de nodos                              |
| Features              | `build_features.py`         | Características                                |
| Relaciones espaciales | `build_spatial_edges.py`    | Aristas espaciales                             |
| Relaciones dinámicas  | `build_dynamic_edges.py`    | Aristas dinámicas                              |
| Relaciones combinadas | `build_combined_edges.py`   | Aristas combinadas                             |
| GraphData             | `build_graphdata.py`        | Construcción de GraphData                      |
| Análisis              | `01.1_graph_analysis.py`    | Análisis estructural y topológico              |
| Mapas temporales      | `01.2_temporal_maps.py`     | Evolución temporal                             |
| Mapas climáticos      | `01.3_climate_maps.py`      | Información climática                          |
| Benchmark             | `02_benchmark.py`           | Comparación de modelos                         |
| Entrenamiento         | `03_train_model.py`         | Entrenamiento del modelo seleccionado          |
| Evaluación            | `04_evaluation.py`          | Evaluación científica                          |
| Auditoría             | `audit_graph_collection.py` | Auditoría de GraphData                         |
| Explainability        | `explainability.py`         | Interpretación                                 |
| Forecasting           | `forecasting.py`            | Componente de forecasting                      |
| GeoAI                 | `geoai/`                    | Plataforma GeoAI                               |
| API                   | `08_api.py`                 | Implementación de API                          |
| Agent                 | `09_agent.py`               | Implementación del agente                      |
| Reports               | `10_reports.py`             | Generación de reportes                         |
| Export                | `11_export.py`              | Exportación                                    |
| Aplicación            | `app.py`                    | Entrada de la aplicación GeoAI                 |

# 33. Flujo científico completo

El flujo científico que debe utilizarse para interpretar el proyecto es:

```text
FUENTES DE DATOS
       |
       v
INGENIERÍA DE DATOS R
       |
       v
DATASET CIENTÍFICO
       |
       v
GRAPH CONSTRUCTION
       |
       v
COLECCIÓN DE GRAPHDATA
       |
       v
GRAPH ANALYSIS
       |
       v
BENCHMARK CIENTÍFICO
       |
       v
SELECCIÓN DEL MODELO GNN
       |
       v
GRAPHSAGE
       |
       v
ENTRENAMIENTO
       |
       v
EVALUACIÓN
       |
       v
EXPLAINABILITY
       |
       v
FORECASTING
       |
       v
GEOAI
       |
       +-------------------+
       |                   |
       v                   v
   DASHBOARD             AGENT
       |                   |
       +---------+---------+
                 |
                 v
                API
                 |
                 v
SISTEMA DE APOYO A LA DECISIÓN
```

Este flujo constituye la cadena principal de la arquitectura científica final.

# 34. Cómo evaluar científicamente el proyecto

El jurado puede revisar el proyecto desde siete dimensiones principales.

## Datos

Verificar la procedencia, organización y transformación de los datos.

Ruta principal:

```text
data/
src/R/
```

## Grafo

Verificar cómo se construyen los nodos, características y relaciones.

Ruta principal:

```text
src/python/graph/
```

## Modelación

Verificar las familias de modelos y el proceso de comparación.

Ruta principal:

```text
src/python/models/
02_benchmark.py
```

## Selección

Verificar la diferencia entre mejor modelo global y mejor modelo GNN.

Resultado arquitectónico:

```text
Mejor global
    |
    v
Linear Regression

Mejor GNN
    |
    v
GraphSAGE
    |
    v
Modelo oficial
```

## Evaluación

Verificar métricas, diagnósticos y validaciones.

Ruta:

```text
04_evaluation.py
outputs/evaluation/
reports/evaluation/
```

## Interpretabilidad

Verificar la separación entre evaluación y explainability.

Ruta:

```text
analysis/explainability.py
```

## Aplicación

Verificar cómo los resultados científicos son utilizados por GeoAI.

Ruta:

```text
src/python/geoai/
app/
```

# 35. Evidencias que debe buscar el jurado

La arquitectura permite buscar evidencia en diferentes niveles.

```text
Código
   |
   v
Implementación

Outputs
   |
   v
Resultados

Reports
   |
   v
Evidencias documentales

Audits
   |
   v
Control de calidad

Certificates
   |
   v
Certificación

Contracts
   |
   v
Especificaciones

Metadata
   |
   v
Trazabilidad

Logs
   |
   v
Observabilidad
```

Esto permite que una afirmación científica no dependa únicamente de una presentación, sino que pueda relacionarse con código, resultados, reportes y mecanismos de gobernanza.

# 36. Qué no debe asumirse sin revisar el código

Para mantener rigor científico, este README no atribuye funcionalidades que no estén demostradas por la arquitectura suministrada.

En particular:

```text
La existencia de forecasting.py
no demuestra por sí sola
qué metodología de forecasting implementa.

La existencia de knowledge_/
no demuestra por sí sola
cómo se utiliza internamente el conocimiento.

La existencia de prompts/
no demuestra por sí sola
qué comportamiento específico implementan.

La existencia de preprocessing/
no permite atribuir funciones internas
que no aparecen documentadas.

La existencia de una API
no demuestra por sí sola
qué endpoints concretos existen.
```

Para estas cuestiones deben revisarse los archivos correspondientes y sus contratos documentales.

Esta política evita inventar funcionalidades y mantiene una correspondencia entre arquitectura, código y evidencia.

# 37. Principio de no duplicación

El proyecto debe evolucionar sobre la arquitectura existente.

No se deben crear módulos paralelos cuando ya existe un componente destinado a una responsabilidad.

Ejemplo:

```text
forecasting.py
+
docs/forecasting_spec.md
+
outputs/forecast/
+
outputs/forecasting/
```

Estos componentes ya forman parte de la arquitectura de forecasting.

Por tanto, la siguiente etapa debe reconstruir y validar su responsabilidad real a partir del código existente, no crear una segunda arquitectura de forecasting.

# 38. Arquitectura consolidada para la sustentación

La arquitectura completa puede resumirse así:

```text
                         AVANZADO IA
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
       DATA                   R                   PYTHON
        |                     |                     |
        |                     v                     |
        |             DATA ENGINEERING              |
        |                     |                     |
        +---------------------+                     |
                              |                     |
                              v                     |
                       DATASET CIENTÍFICO           |
                              |                     |
                              +---------------------+
                                    |
                                    v
                            GRAPH CONSTRUCTION
                                    |
                                    v
                              COLLECTION
                              OF GRAPHDATA
                                    |
                                    v
                              GRAPH ANALYSIS
                                    |
                                    v
                           SCIENTIFIC BENCHMARK
                                    |
                         +----------+----------+
                         |                     |
                         v                     v
                    BEST GLOBAL            BEST GNN
                         |                     |
                         v                     v
                  Linear Regression        GraphSAGE
                                               |
                                               v
                                         OFFICIAL MODEL
                                               |
                                               v
                                           TRAINING
                                               |
                                               v
                                          EVALUATION
                                               |
                              +----------------+----------------+
                              |                |                |
                              v                v                v
                           METRICS        DIAGNOSTICS      EXPLAINABILITY
                              |                |                |
                              +----------------+----------------+
                                               |
                                               v
                                          FORECASTING
                                               |
                                               v
                                         FUTURE RESULTS
                                               |
                                               v
                                             GEOAI
                                               |
                          +--------------------+--------------------+
                          |                    |                    |
                          v                    v                    v
                      KNOWLEDGE            SERVICES            COMPONENTS
                          |                    |                    |
                          +--------------------+--------------------+
                                               |
                          +--------------------+--------------------+
                          |                    |                    |
                          v                    v                    v
                     DASHBOARD                API                 AGENT
                          |                    |                    |
                          +--------------------+--------------------+
                                               |
                                               v
                                  DECISION SUPPORT SYSTEM
                                               |
                          +--------------------+--------------------+
                          |                    |                    |
                          v                    v                    v
                       REPORTS              EXPORT           VISUALIZATION
```

La arquitectura consolidada documentada incluye explícitamente las capas de modelos, evaluación comparativa, GraphSAGE, entrenamiento, evaluación, explainability, forecasting y GeoAI.

# 39. Capa transversal de gobernanza

Todas las etapas están acompañadas por una capa transversal:

```text
                    GOBERNANZA
                         |
       +-----------------+-----------------+
       |                 |                 |
       v                 v                 v
     LOGS              AUDITS         CERTIFICATES
       |                 |                 |
       +-----------------+-----------------+
                         |
                         v
                     METADATA
                         |
                         v
                  REPRODUCIBILIDAD
                         |
                         v
                  TRAZABILIDAD TOTAL
```

Adicionalmente se relacionan:

```text
Configuration
Logs
Audits
Metadata
Certificates
Contracts
Reports
Export
```

Esta capa constituye uno de los elementos fundamentales para demostrar que el proyecto está organizado como un sistema científico reproducible y auditable.

# 40. Guía rápida para el jurado

Si el jurado dispone de poco tiempo, la revisión recomendada es:

```text
Paso 1
README.md

Paso 2
docs/

Paso 3
src/R/

Paso 4
src/python/graph/

Paso 5
src/python/models/

Paso 6
02_benchmark.py

Paso 7
03_train_model.py

Paso 8
04_evaluation.py

Paso 9
analysis/explainability.py

Paso 10
forecasting.py

Paso 11
src/python/geoai/

Paso 12
src/python/outputs/

Paso 13
reports/

Paso 14
audits, certificates, contracts y metadata
```

La lectura debe seguir siempre la pregunta:

```text
Qué entra
    |
    v
Qué transformación ocurre
    |
    v
Qué componente la realiza
    |
    v
Qué resultado produce
    |
    v
Dónde queda almacenado
    |
    v
Cómo se valida
    |
    v
Cómo se utiliza posteriormente
```

# 41. Principio de trazabilidad científica

Cada etapa debe poder responder cinco preguntas:

```text
1. Qué recibe

2. Qué hace

3. Qué produce

4. Dónde queda el resultado

5. Cómo se verifica
```

Por ejemplo:

```text
Benchmark
    |
    +-- Recibe:
    |      Dataset y modelos
    |
    +-- Hace:
    |      Comparación científica
    |
    +-- Produce:
    |      Resultados del benchmark
    |
    +-- Persiste:
    |      outputs/benchmark/
    |
    +-- Se verifica:
           mediante resultados y reportes
```

Este patrón puede utilizarse para revisar todas las capas del sistema.

# 42. Responsabilidad del proyecto frente al jurado

El proyecto no debe presentarse únicamente como un modelo GraphSAGE.

GraphSAGE es una pieza central del sistema, pero la arquitectura es considerablemente más amplia.

El sistema completo comprende:

```text
Ingeniería de datos
+
Dataset científico
+
Construcción de grafos
+
GraphData
+
Análisis topológico y científico
+
Benchmark
+
Selección del modelo
+
GraphSAGE
+
Entrenamiento
+
Evaluación
+
Explainability
+
Forecasting
+
GeoAI
+
Dashboard
+
API
+
Agent
+
Reports
+
Export
+
Auditoría
+
Certificación
+
Metadata
+
Logs
```

Por tanto, el valor arquitectónico del proyecto está en la integración reproducible de estas capas y no exclusivamente en la arquitectura neuronal.

# 43. Criterio de interpretación final

Para interpretar correctamente el proyecto debe mantenerse la siguiente cadena:

```text
Los datos definen el dominio
        |
        v
La ingeniería de datos construye el dataset científico
        |
        v
El dataset permite construir el grafo
        |
        v
GraphData representa el estado espacial y temporal
        |
        v
Graph Analysis caracteriza las relaciones
        |
        v
Benchmark compara alternativas
        |
        v
La selección determina el modelo GNN oficial
        |
        v
GraphSAGE es entrenado
        |
        v
El modelo es evaluado
        |
        v
Explainability permite interpretar
        |
        v
Forecasting proyecta el componente correspondiente
        |
        v
GeoAI transforma los resultados en funcionalidades
        |
        v
Dashboard, API y Agent permiten su utilización
        |
        v
Reports y Export permiten comunicar y reutilizar resultados
        |
        v
Audits, Certificates, Contracts, Metadata y Logs
mantienen trazabilidad y gobernanza
```

# 44. Mensaje final para el evaluador

El repositorio debe entenderse como una arquitectura científica modular.

Cada capa posee una responsabilidad específica y los resultados de una etapa sirven como entrada para la siguiente.

La arquitectura comienza con la ingeniería de datos, continúa con la representación espacial y temporal mediante GraphData, incorpora análisis científico y comparación de modelos, selecciona GraphSAGE dentro del contexto GNN, realiza entrenamiento y evaluación, incorpora explainability y forecasting y finalmente integra los resultados dentro de una plataforma GeoAI orientada al apoyo a la decisión.

Alrededor de este flujo existe una capa transversal de configuración, auditoría, trazabilidad, metadatos, certificación, contratos, reportes, exportación y logs.

Por esta razón, la evaluación del proyecto debe considerar simultáneamente:

```text
Rigor científico
+
Arquitectura de software
+
Reproducibilidad
+
Trazabilidad
+
Control de calidad
+
Evaluación de modelos
+
Interpretabilidad
+
Capacidad de extensión
+
Aplicación GeoAI
```

El propósito del README es permitir que cualquier evaluador pueda recorrer el repositorio desde los datos hasta la aplicación, identificar la responsabilidad de cada componente y localizar las evidencias correspondientes sin depender exclusivamente de la explicación oral del autor.
