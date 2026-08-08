<!-- explainability_spec.md -->

# Explainability Specification

**Proyecto:** Plataforma Inteligente GeoAI para Modelado, Forecasting e Inteligencia Territorial

**Versión:** 1.0

**Estado:** Especificación Técnica Oficial (Technical Design Specification - TDS)

**Autor:** AVANZADO-IA

---

# 1. Introducción

Describe el propósito del módulo de explicabilidad dentro de la plataforma GeoAI y su papel como puente entre la Inteligencia Artificial Explicable (XAI) y la Inteligencia Territorial.

---

# 2. Objetivo Científico

Define el objetivo científico del módulo y los productos de conocimiento que debe generar para apoyar la toma de decisiones.

---

# 3. Filosofía de la Arquitectura

Explica la arquitectura en dos niveles:

- Nivel 1. Explainable Artificial Intelligence (XAI)
- Nivel 2. GeoAI

Se establece la separación entre las explicaciones matemáticas generadas por el framework y el conocimiento territorial consumido por la plataforma.

---

# 4. Arquitectura General del Módulo

Describe el flujo metodológico completo.

```
GraphData
      │
      ▼
GraphSAGE
      │
      ▼
GNNExplainer
      │
      ▼
Explanation
      │
      ▼
Extracción de conocimiento
      │
      ▼
global_explainability
      │
      ▼
GeoAI
```

---

# 5. Principio Metodológico

Define el principio científico del módulo.

La explicabilidad se realiza exclusivamente sobre los nodos pertenecientes al conjunto `test_mask`.

---

# 6. Modelo Oficial

## Modelo autorizado

- GraphSAGE

## Modelos excluidos

- GCN
- GAT
- ChebNet
- Graph Transformer

---

# 7. Método Oficial de Explicabilidad

## Método autorizado

- GNNExplainer

## Configuración oficial

Documenta la configuración del `Explainer` utilizada por el proyecto.

---

# 8. Fundamentos Técnicos

Describe las decisiones verificadas experimentalmente sobre la API oficial de PyTorch Geometric.

Incluye:

- Firma oficial del Explainer.
- Naturaleza del objeto `Explanation`.
- Formatos de `node_mask`.
- Máscaras opcionales.
- Validación mediante `Explanation.validate()`.

---

# 9. Pipeline Oficial de Explicabilidad

```
GraphData
      │
      ▼
Recuperar test_mask
      │
      ▼
Para cada nodo del test_mask
      │
      ▼
Generar Explanation
      │
      ▼
Validar Explanation
      │
      ▼
Extraer importancia de variables
      │
      ▼
Guardar explicación individual
      │
      ▼
Agregar explicaciones
      │
      ▼
global_explainability
```

---

# 10. Diseño Robusto del Algoritmo

Principios de programación defensiva.

- Validación de estructuras.
- Compatibilidad futura.
- Manejo de errores.
- Validación de máscaras.
- Detección automática de dimensionalidad.

---

# 11. Agregación Científica

Define la estrategia para consolidar las explicaciones individuales.

Productos:

- Media.
- Desviación estándar.
- Ranking global de variables.

---

# 12. Producto Oficial

## Producto técnico

```
Explanation
```

## Producto científico

```
global_explainability
```

El contexto `global_explainability` constituye la interfaz oficial entre el módulo de explicabilidad y el resto de la plataforma GeoAI.

---

# 13. Arquitectura del Contexto

Especifica la estructura oficial de `global_explainability`.

Incluye:

- metadata
- node_explanations
- feature_importance
- feature_importance_std
- explained_nodes
- status

---

# 14. Validaciones Obligatorias

Lista todas las validaciones requeridas antes de aceptar una explicación.

---

# 15. Integración con la Plataforma GeoAI

Describe cómo interactúa el módulo con:

- 04_evaluation.py
- 05_forecasting.py
- 06_geoai.py
- Dashboard
- API
- Agentes IA
- Reportes

---

# 16. Reglas de Arquitectura

Define las reglas obligatorias para futuras implementaciones.

Ejemplos:

- Nunca consumir directamente `Explanation`.
- Toda interacción debe realizarse mediante `global_explainability`.
- Todas las explicaciones pertenecen al `test_mask`.
- Validar siempre `Explanation.validate()`.

---

# 17. Reproducibilidad Científica

Documenta los elementos mínimos necesarios para reproducir completamente el proceso de explicabilidad.

---

# 18. Decisiones Oficiales del Proyecto

Registro oficial de todas las decisiones metodológicas adoptadas para el módulo.

---

# 19. Estado del Proyecto

Describe el estado actual de la especificación y el alcance de futuras versiones.
