# INFORME DE ANÁLISIS
## Reto 8 — Aplicación de técnicas de Data Mining
### El Mercado de las Especias de Dataclysm

---

**Autora:** Judit Giravent  
**Fecha:** 09/2026  
**Herramientas:** Python 3.13 · pandas · scikit-learn · matplotlib · seaborn  
**Dataset:** Wholesale Customers (UCI ML Repository)

---

## Índice

1. Resumen ejecutivo
2. Introducción y contexto
3. Paso 1: Selección del dataset
4. Paso 2: Importación y exploración
5. Paso 3: Preparación de datos
6. Paso 4: Aplicación de técnicas de Data Mining
7. Paso 5: Evaluación de modelos
8. Paso 6: Interpretación y visualización
9. Conclusiones y recomendaciones
10. Anexos

---

## 1. Resumen ejecutivo

Este proyecto aplica **técnicas avanzadas de Data Mining** sobre el dataset **Wholesale Customers** de UCI para extraer insights del comportamiento de 440 clientes mayoristas.

### Técnicas aplicadas

| # | Técnica | Algoritmo | Objetivo | Resultado |
|---|---|---|---|---|
| 1 | **Clustering** | K-means | Segmentar clientes | Silhouette = 0.3953 |
| 2 | **Clasificación** | Árbol de decisión | Predecir canal | F1 = 0.9468 |
| 3 | **Regresión** | Regresión lineal | Predecir Grocery | R² = 0.7797 |

### Hallazgos principales

1. **3 segmentos de clientes**: Fresh alto (restaurantes), Grocery alto (tiendas), pequeños.
2. **Detergents_Paper** es la variable clave para predecir el canal (79,6% importancia).
3. **El canal Retail** compra mucho más detergente que el Horeca.
4. **71,6% de clientes** son pequeños y aportan poco valor.

---

## 2. Introducción y contexto

### 2.1 Contexto narrativo

En Dataclysm, el **Mentor de Datos** enseña al navegante técnicas avanzadas para descubrir patrones ocultos en los datos comerciales.

### 2.2 Objetivo del reto

Aplicar técnicas de Data Mining (clustering, clasificación, regresión) para extraer insights valiosos que permitan a los mercaderes tomar decisiones estratégicas.

### 2.3 Metodología

Sigue los 6 pasos oficiales del reto: selección, importación, preparación, técnicas, evaluación e interpretación.

---

## 3. Paso 1: Selección del dataset

- **Nombre:** Wholesale Customers
- **Fuente:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/dataset/292/wholesale+customers
- **Registros:** 440 clientes mayoristas
- **Variables:** 8 (Channel, Region, Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen)
- **Periodo:** Datos anonimizados de clientes de un distribuidor

---

## 4. Paso 2: Importación y exploración

### 4.1 Dimensiones

- **440 filas × 8 columnas**
- **Sin valores nulos** ✅
- **Sin duplicados** ✅

### 4.2 Estadísticas descriptivas

| Variable | Media | Desv. Est. | Máximo |
|---|---|---|---|
| Fresh | 12.000,30 | 12.647,33 | 112.151 |
| Milk | 5.796,43 | 7.384,85 | 73.400 |
| Grocery | 7.950,79 | 9.503,20 | 92.780 |
| Frozen | 3.071,93 | 4.854,67 | 60.869 |
| Detergents_Paper | 2.881,49 | 4.767,85 | 40.827 |
| Delicassen | 1.524,87 | 2.820,11 | 47.943 |

**Observación:** la desviación estándar > media en todas las variables → alta dispersión y posibles outliers.

### 4.3 Variables categóricas

- **Channel:** 298 Horeca (67,7%) / 142 Retail (32,3%)
- **Region:** 316 Other (71,8%) / 77 Lisbon / 47 Oporto

### 4.4 Figuras

- `01_distribucion_variables.png` — Histogramas de las 6 variables de gasto.
- `02_matriz_correlacion.png` — Correlaciones entre variables.
- `03_boxplots_por_canal.png` — Distribución de gasto por canal.

---

## 5. Paso 3: Preparación de datos

### 5.1 Detección de outliers (IQR)

| Variable | N_outliers | %_outliers |
|---|---|---|
| Fresh | 20 | 4,55% |
| Milk | 28 | 6,36% |
| Grocery | 24 | 5,45% |
| Frozen | 43 | 9,77% |
| Detergents_Paper | 30 | 6,82% |
| Delicassen | 27 | 6,14% |

**Decisión:** winsorización al percentil 99 (cap) en lugar de eliminar.

### 5.2 Normalización

- **StandardScaler:** media=0, std=1 (para clustering).
- **MinMaxScaler:** rango 0-1 (para visualizaciones).

### 5.3 Codificación

- **Region:** One-Hot Encoding (3 columnas).
- **Channel:** etiquetado como Horeca/Retail.

### 5.4 Dataset final

- `wholesale_preparado.csv` (440, 11)
- `wholesale_minmax.csv` (440, 8)

### 5.5 Figura

- `04_normalizacion_comparativa.png` — Antes/después de normalización.

---

## 6. Paso 4: Aplicación de técnicas de Data Mining

### 6.1 Clustering (K-means)

**Selección de K:**

| K | Silhouette | Davies-Bouldin | Inercia |
|---|---|---|---|
| 2 | **0.4628** | 1.1648 | 1859.12 |
| 3 | 0.3953 | 1.2471 | 1482.06 |
| 4 | 0.3106 | 1.1670 | 1241.86 |

**K elegido:** 3 (por interpretabilidad de negocio).

**Resultados:**

| Cluster | Clientes | % | Perfil |
|---|---|---|---|
| 0 | 64 | 14,55% | Fresh alto + Frozen alto |
| 1 | 61 | 13,86% | Grocery alto + Detergents alto |
| 2 | 315 | 71,59% | Valores bajos (pequeños) |

**Figuras:** `05_seleccion_k.png`, `06_clusters_scatter.png`, `07_perfil_clusters.png`.

### 6.2 Clasificación (Árbol de decisión)

**Objetivo:** predecir `Channel` a partir de las 6 categorías de gasto.

**Resultados:**

| Métrica | Train | Test |
|---|---|---|
| Accuracy | 0.9643 | 0.9470 |
| Precision | — | 0.9468 |
| Recall | — | 0.9470 |
| F1-score | — | 0.9468 |
| CV (5-fold) | — | 0.8932 ± 0.0245 |

**Importancia de variables:**

| Variable | Importancia |
|---|---|
| **Detergents_Paper** | **79,56%** |
| Milk | 6,63% |
| Fresh | 5,61% |
| Grocery | 3,74% |
| Frozen | 2,86% |
| Delicassen | 1,60% |

**Figuras:** `08_arbol_decision.png`, `09_matriz_confusion.png`, `10_importancia_variables.png`.

### 6.3 Regresión (Regresión lineal)

**Objetivo:** predecir `Grocery` a partir de otras categorías.

**Resultados:**

| Métrica | Train | Test |
|---|---|---|
| R² | 0.9020 | **0.7797** |
| RMSE | 3.283,47 | 3.046,52 |
| MAE | 2.033,66 | 2.057,59 |
| CV R² | — | 0.8289 ± 0.0612 |

**Coeficientes:**

| Variable | Coeficiente |
|---|---|
| **Detergents_Paper** | **+1.6949** |
| Delicassen | +0.2274 |
| Milk | +0.1548 |
| Frozen | +0.0494 |
| Fresh | +0.0406 |

**Figuras:** `11_regresion_real_vs_predicho.png`, `12_analisis_residuos.png`, `13_coeficientes_regresion.png`.

---

## 7. Paso 5: Evaluación de modelos

### 7.1 Resumen comparativo

| Técnica | Métrica principal | Valor | Interpretación |
|---|---|---|---|
| **Clustering** | Silhouette | 0.3953 | Clusters moderadamente separados |
| **Clasificación** | F1-score | 0.9468 | Excelente clasificación |
| **Regresión** | R² | 0.7797 | Buena predicción |

### 7.2 Ajustes aplicados

- **Clustering:** K=3 elegido por interpretabilidad (aunque K=2 tenía mejor métrica).
- **Clasificación:** `max_depth=4` para evitar overfitting.
- **Regresión:** sin regularización (los coeficientes son interpretables).

### 7.3 Figura

- `14_comparativa_modelos.png` — Comparativa visual de los 3 modelos.

---

## 8. Paso 6: Interpretación y visualización

### 8.1 Interpretación de clusters

| Cluster | Nombre de negocio | Estrategia |
|---|---|---|
| 0 | "Restaurantes / Asadores" | Promociones de carne y congelados |
| 1 | "Tiendas de alimentación" | Cross-selling de lácteos y limpieza |
| 2 | "Clientes pequeños" | Campañas de reactivación y fidelización |

### 8.2 Interpretación de clasificación

El modelo clasifica correctamente el **94,7%** de los canales. `Detergents_Paper` es la variable clave:
- Los **Retail** (supermercados) compran mucho detergente.
- Los **Horeca** (restaurantes) compran poca cantidad.

### 8.3 Interpretación de regresión

El gasto en `Grocery` se predice muy bien por `Detergents_Paper`. Cada unidad extra de detergente se asocia con **1,69 unidades extra de Grocery**.

---

## 9. Conclusiones y recomendaciones

### 9.1 Conclusiones

1. **3 segmentos claros** de clientes mayoristas identificados.
2. **71,6% de clientes** son pequeños y aportan poco valor → oportunidad de crecimiento.
3. **Detergents_Paper** es la variable más informativa para clasificar clientes.
4. **El modelo de clasificación es muy preciso** (F1 = 0.9468).

### 9.2 Recomendaciones para Dataclysm

1. **Segmentación personalizada:** campañas diferenciadas por cluster.
2. **Foco en clientes pequeños:** representan el 71,6% pero aportan poco → oportunidad.
3. **Detección automática de canal:** usar el árbol de decisión para clasificar clientes nuevos.
4. **Predicción de gasto:** usar la regresión para estimar ventas futuras.

---

## 10. Anexos

### Anexo A: Estructura del proyecto

```
Reto8_DataMining/
├── datos/
│   ├── raw/
│   └── limpios/
├── scripts/
│   ├── 01_exploracion.py
│   ├── 02_preparacion.py
│   ├── 03_clustering.py
│   ├── 04_clasificacion.py
│   ├── 05_regresion.py
│   └── 06_evaluacion.py
├── modelos/
│   ├── modelo_clustering_kmeans.pkl
│   ├── modelo_clasificacion_tree.pkl
│   └── modelo_regresion_lineal.pkl
├── salidas/
│   ├── figuras/       (14 PNG)
│   └── resultados/    (varios CSV)
└── docs/
```

### Anexo B: Tecnologías

| Herramienta | Versión |
|---|---|
| Python | 3.13 |
| pandas | 3.0.5 |
| scikit-learn | última |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |

---

**FIN DEL INFORME**  
*Documento generado por Judit Giravent — 09/2026*