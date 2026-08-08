<!-- api_spec.md -->

# API Specification

**Proyecto:** Plataforma Inteligente GeoAI para Modelado, Forecasting e Inteligencia Territorial

**Versión:** 1.0

**Estado:** Especificación Técnica Oficial (Technical Design Specification - TDS)

**Autor:** AVANZADO-IA

---

# 1. Introducción

La API constituye la interfaz oficial de comunicación entre la Plataforma
GeoAI y las aplicaciones externas.

Su propósito es proporcionar acceso seguro, consistente y reproducible a
los productos científicos generados por la plataforma, permitiendo su
integración con dashboards, agentes inteligentes, aplicaciones web,
servicios móviles y sistemas institucionales.

---

# 2. Objetivo Científico

La API tiene como objetivo exponer los resultados oficiales de la
plataforma GeoAI sin alterar los procesos científicos que los generan.

Debe permitir:

- Consultar información territorial.
- Recuperar predicciones.
- Obtener resultados de forecasting.
- Consultar explicaciones del modelo.
- Acceder a indicadores territoriales.
- Integrarse con sistemas externos.

---

# 3. Filosofía de la Arquitectura

La API no ejecuta procesos científicos.

Su función consiste únicamente en exponer productos oficiales generados
por el pipeline GeoAI.

Toda la lógica científica permanece desacoplada de la capa de servicios.

---

# 4. Arquitectura General

```
Cliente
      │
      ▼
API
      │
      ▼
GeoAI Context
      │
      ▼
Productos Oficiales
```

La API actúa como una capa de acceso entre los consumidores externos y la
plataforma GeoAI.

---

# 5. Fuentes Oficiales de Información

La API únicamente podrá consultar productos oficiales.

| Módulo | Producto Oficial |
|---------|------------------|
| build_graphdata.py | graph_data.pt |
| 03_train_model.py | trained_model |
| 04_evaluation.py | evaluation_results |
| 04_evaluation.py | global_explainability |
| 05_forecasting.py | forecasting_results |
| 06_geoai.py | geoai_context |

---

# 6. Servicios Oficiales

La API deberá proporcionar acceso a los siguientes servicios.

## Información Territorial

Consulta de municipios.

Consulta de variables.

Consulta de indicadores.

---

## Predicciones

Resultados del modelo oficial.

---

## Forecasting

Escenarios futuros.

Series temporales.

---

## Explicabilidad

Importancia de variables.

Ranking global.

Explicaciones individuales.

---

## Dashboard

Servicios de apoyo para la visualización.

---

## Reportes

Generación y descarga de reportes.

---

# 7. Flujo Metodológico

```
Cliente
      │
      ▼
Solicitud
      │
      ▼
API
      │
      ▼
GeoAI Context
      │
      ▼
Respuesta
```

---

# 8. Productos Oficiales

La API únicamente distribuye productos científicos previamente
generados.

No crea nuevos productos.

Los principales recursos expuestos son:

- graph_data
- evaluation_results
- forecasting_results
- global_explainability
- geoai_context

---

# 9. Reglas de Arquitectura

La API nunca deberá:

- entrenar modelos;
- construir GraphData;
- ejecutar benchmarking;
- realizar forecasting;
- modificar resultados científicos;
- alterar explicaciones del modelo.

La API únicamente publica información.

---

# 10. Integración con la Plataforma GeoAI

La API interactúa con:

- Dashboard
- Agentes Inteligentes
- Sistema de Reportes
- Motor de Reglas
- Aplicaciones externas
- Sistemas institucionales

---

# 11. Validaciones

Antes de responder una solicitud deberán verificarse:

- existencia del contexto GeoAI;
- disponibilidad del producto solicitado;
- consistencia de la respuesta;
- integridad de los datos;
- estado del servicio.

---

# 12. Seguridad

La API deberá incorporar mecanismos de:

- autenticación;
- autorización;
- validación de solicitudes;
- registro de eventos;
- control de errores.

La implementación específica dependerá de la infraestructura adoptada.

---

# 13. Reproducibilidad

Todas las respuestas deberán derivarse exclusivamente de productos
oficiales generados por el pipeline científico.

La API no almacenará resultados alternativos.

---

# 14. Decisiones Oficiales del Proyecto

1. La API constituye la interfaz oficial de la Plataforma GeoAI.
2. No ejecuta procesos científicos.
3. No modifica resultados.
4. Expone únicamente productos oficiales.
5. Toda la lógica científica permanece desacoplada.
6. La API debe ser reproducible, consistente y escalable.

---

# 15. Estado del Proyecto

Esta especificación define la arquitectura oficial de la API de la
Plataforma Inteligente GeoAI y servirá como referencia para el desarrollo
del módulo `08_api.py`.