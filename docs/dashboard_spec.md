<!-- dashboard_spec.md -->

# Dashboard Specification

**Proyecto:** Plataforma Inteligente GeoAI para Modelado, Forecasting e Inteligencia Territorial

**Versión:** 1.0

**Estado:** Especificación Técnica Oficial (Technical Design Specification - TDS)

**Autor:** AVANZADO-IA

---

# 1. Introducción

El Dashboard constituye la interfaz principal entre la Plataforma GeoAI y los usuarios finales.

Su propósito es transformar los resultados científicos generados por los modelos de Graph Neural Networks en información visual, interactiva e interpretable que apoye la toma de decisiones territoriales.

---

# 2. Objetivo Científico

El Dashboard tiene como objetivo presentar, de forma integrada, el conocimiento generado por la plataforma GeoAI mediante componentes visuales que faciliten el análisis espacial, temporal y predictivo.

Debe permitir:

- Visualizar el estado actual del territorio.
- Explorar resultados del modelo GraphSAGE.
- Analizar pronósticos espacio-temporales.
- Interpretar explicaciones del modelo.
- Consultar indicadores territoriales.
- Apoyar procesos de decisión.

---

# 3. Filosofía de la Arquitectura

El Dashboard no genera conocimiento científico.

Su función consiste en consumir los productos oficiales generados por los módulos del pipeline y transformarlos en información visual.

Toda la lógica científica permanece en los módulos productores.

---

# 4. Arquitectura General

```
GraphData
      │
      ▼
Modelo GraphSAGE
      │
      ▼
Forecasting
      │
      ▼
Explainability
      │
      ▼
GeoAI
      │
      ▼
Dashboard
```

---

# 5. Fuentes Oficiales de Información

El Dashboard únicamente consume productos oficiales.

| Módulo | Producto |
|---------|----------|
| build_graphdata.py | graph_data.pt |
| 03_train_model.py | trained_model |
| 04_evaluation.py | evaluation_results |
| 04_evaluation.py | global_explainability |
| 05_forecasting.py | forecasting_results |
| 06_geoai.py | geoai_context |

---

# 6. Componentes del Dashboard

## Mapa Territorial

Visualización espacial de municipios y resultados.

---

## Panel de Indicadores

Indicadores agregados del territorio.

---

## Predicciones

Resultados del modelo oficial.

---

## Forecasting

Escenarios futuros.

---

## Explicabilidad

Importancia de variables.

Ranking global.

Explicaciones por municipio.

---

## Benchmark

Resumen de métricas del modelo oficial.

---

## Reportes

Generación y exportación de informes.

---

# 7. Flujo Metodológico

```
GeoAI Context
      │
      ▼
Carga de datos
      │
      ▼
Procesamiento visual
      │
      ▼
Construcción de componentes
      │
      ▼
Dashboard interactivo
```

---

# 8. Productos Oficiales

El Dashboard genera exclusivamente productos visuales.

Productos:

- Mapas.
- Tablas.
- Indicadores.
- Gráficos.
- Reportes.
- Exportaciones.

No modifica los datos científicos.

---

# 9. Reglas de Arquitectura

- El Dashboard nunca modifica el GraphData.
- El Dashboard nunca entrena modelos.
- El Dashboard nunca ejecuta procesos científicos.
- El Dashboard únicamente consume productos oficiales.
- Toda la lógica científica permanece desacoplada.

---

# 10. Integración con la Plataforma GeoAI

El Dashboard interactúa con:

- API.
- Agentes Inteligentes.
- Motor de Reglas.
- Sistema de Reportes.
- Exportaciones.

---

# 11. Validaciones

Antes de construir la interfaz deberán verificarse:

- Existencia del contexto GeoAI.
- Existencia del Forecasting.
- Existencia de global_explainability.
- Integridad de los indicadores.
- Disponibilidad de mapas.

---

# 12. Reproducibilidad

Toda visualización deberá poder reconstruirse a partir de los productos oficiales generados por el pipeline.

El Dashboard no almacenará resultados científicos adicionales.

---

# 13. Decisiones Oficiales del Proyecto

1. El Dashboard consume únicamente productos oficiales.
2. No modifica información científica.
3. Toda la lógica de negocio reside en GeoAI.
4. La visualización es completamente desacoplada del entrenamiento del modelo.
5. La interfaz debe ser reproducible y consistente.

---

# 14. Estado del Proyecto

Esta especificación define la arquitectura oficial del Dashboard de la Plataforma Inteligente GeoAI y servirá como referencia para el desarrollo del módulo `07_dashboard.py`.

