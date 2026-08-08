<!-- graphdata_spec.md -->

# GraphData Specification

**Proyecto:** Plataforma Inteligente GeoAI para Modelado, Forecasting e Inteligencia Territorial

**Versión:** 1.0

**Estado:** Especificación Técnica Oficial (Technical Design Specification - TDS)

**Autor:** AVANZADO-IA

---

# 1. Introducción

Describe el propósito del GraphData dentro de la plataforma GeoAI y su papel como representación científica del territorio para el entrenamiento, evaluación y forecasting mediante Graph Neural Networks.

---

# 2. Objetivo Científico

Define el objetivo científico del GraphData y las capacidades que debe proporcionar al proyecto.

---

# 3. Filosofía de la Arquitectura

Explica los principios de diseño que rigen la construcción del GraphData.

Ejemplo:

- Separación entre identidad del nodo y variables dinámicas.
- Arquitectura modular.
- Reproducibilidad científica.
- Escalabilidad.
- Compatibilidad con PyTorch Geometric.

---

# 4. Arquitectura General del GraphData

Diagrama metodológico completo.

```
Dataset Científico
        │
        ▼
Catálogo Oficial de Nodos
        │
        ▼
Relaciones Espaciales
        │
        ▼
Matriz de Features
        │
        ▼
GraphData
```

---

# 5. Definición Oficial de los Nodos

## Entidad representada

## Identificador oficial

## Identificador interno

## Catálogo Oficial

## Atributos permanentes

## Restricciones

---

# 6. Definición Oficial de las Aristas

## Significado científico

## Método oficial

## Métodos soportados

## Tipo de grafo

## Pesos

## Self-Loops

## Restricciones

---

# 7. Definición Oficial de las Features

## Filosofía

## Variables estructurales

## Variables climáticas

## Variables agrícolas

## Variables ambientales

## Coberturas

## Variables excluidas

---

# 8. Definición Oficial del Target

## Variable objetivo

## Transformaciones

## Restricciones

---

# 9. Estructura Oficial del GraphData

## x

## edge_index

## edge_attr

## y

## masks

---

# 10. Pipeline Oficial de Construcción

```
Dataset Científico
        │
        ▼
build_node_catalog.py
        │
        ▼
build_edges.py
        │
        ▼
build_features.py
        │
        ▼
build_graphdata.py
        │
        ▼
graph_data.pt
```

---

# 11. Validaciones Obligatorias

Validaciones científicas y técnicas que debe cumplir el GraphData antes de ser utilizado.

---

# 12. Productos Oficiales del Pipeline

| Módulo | Producto Oficial |
|--------|------------------|
| build_node_catalog.py | node_catalog |
| build_edges.py | edge_index |
| build_features.py | feature_matrix |
| build_graphdata.py | graph_data.pt |

---

# 13. Reglas de Arquitectura

Principios obligatorios que todos los módulos deben respetar.

---

# 14. Reproducibilidad Científica

Elementos mínimos que deben documentarse para reproducir el GraphData.

---

# 15. Decisiones Oficiales del Proyecto

Registro de las decisiones metodológicas adoptadas.

---

# 16. Estado del Proyecto

Estado actual de la especificación y criterios para futuras actualizaciones.