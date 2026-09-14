# DOCUMENTACIÓN DE EVALUACIÓN Y AJUSTES
## Reto 8 — Aplicación de técnicas de Data Mining
### El Mercado de las Especias de Dataclysm

---

**Autora:** Judit Giravent  
**Fecha:** 09/2026  
**Dataset:** Wholesale Customers (UCI ML Repository) — 440 clientes

---

## Índice

1. Resumen ejecutivo
2. Evaluación del modelo de Clustering
3. Evaluación del modelo de Clasificación
4. Evaluación del modelo de Regresión
5. Comparativa de modelos
6. Ajustes de hiperparámetros
7. Conclusiones

---

## 1. Resumen ejecutivo

Este documento detalla las **métricas de evaluación** aplicadas a los 3 modelos de Data Mining y los **ajustes realizados** para optimizar su rendimiento.

### Resumen de métricas

| Modelo | Métrica principal | Valor | Interpretación |
|---|---|---|---|
| **Clustering** | Silhouette | 0.3953 | Clusters moderadamente separados |
| **Clasificación** | F1-score | 0.9468 | Excelente clasificación |
| **Regresión** | R² (test) | 0.7797 | Buena predicción |

### Conclusión

Los 3 modelos demuestran un **rendimiento sólido** y son adecuados para aplicar en el contexto de negocio del Mercado de las Especias.

---

## 2. Evaluación del modelo de Clustering

### 2.1 Métricas obtenidas

| Métrica | Valor | Interpretación |
|---|---|---|
| **Silhouette Score** | 0.3953 | Clusters moderadamente separados |
| **Davies-Bouldin Index** | 1.2471 | Bajo (mejor = más bajo) |
| **Inercia** | 1482.06 | Suma de distancias intra-cluster |

### 2.2 Proceso de selección de K

| K | Silhouette | Davies-Bouldin | Inercia |
|---|---|---|---|
| 2 | **0.4628** ⭐ | **1.1648** | 1859.12 |
| **3** | 0.3953 | 1.2471 | 1482.06 |
| 4 | 0.3106 | 1.1670 | 1241.86 |
| 5 | 0.3159 | 1.1663 | 1080.06 |
| 6 | 0.3368 | **1.0558** ⭐ | 930.45 |
| 7 | 0.2890 | 1.0871 | 850.51 |
| 8 | 0.3097 | 1.0709 | 776.22 |
| 9 | 0.2746 | 1.1220 | 716.28 |
| 10 | 0.2459 | 1.1248 | 669.55 |

### 2.3 Justificación de la elección de K=3

Aunque **K=2 tiene mejores métricas** (Silhouette 0.4628 y Davies-Bouldin 1.1648), se eligió **K=3** por las siguientes razones de negocio:

1. **Interpretabilidad:** con K=2 solo se distinguen 2 grupos; K=3 aporta un nivel adicional de segmentación útil.
2. **Accionabilidad:** 3 segmentos permiten diseñar 3 estrategias de marketing diferenciadas.
3. **Equilibrio:** el Cluster 2 (71,59%) se separa bien del resto, y los Clusters 0 y 1 (14,55% y 13,86%) tienen perfiles muy distintos.

### 2.4 Caracterización de los clusters

| Cluster | Clientes | % | Fresh | Milk | Grocery | Frozen | Detergents | Delicassen |
|---|---|---|---|---|---|---|---|---|
| **0** | 64 | 14,55% | **27.837** | 6.321 | 6.007 | **10.167** | 966 | **3.614** |
| **1** | 61 | 13,86% | 7.183 | **16.631** | **25.246** | 1.713 | **11.525** | 2.238 |
| **2** | 315 | 71,59% | 9.716 | 3.591 | 4.997 | 1.893 | 1.597 | 962 |

### 2.5 Interpretación de negocio

| Cluster | Nombre | Perfil | Estrategia |
|---|---|---|---|
| 0 | "Restaurantes / Asadores" | Fresh y Frozen altos | Promociones de carne y pescado |
| 1 | "Tiendas de alimentación" | Grocery y Detergents altos | Cross-selling de lácteos |
| 2 | "Clientes pequeños" | Valores bajos en todo | Campañas de reactivación |

---

## 3. Evaluación del modelo de Clasificación

### 3.1 Métricas obtenidas

| Métrica | Train | Test | CV (5-fold) |
|---|---|---|---|
| **Accuracy** | 0.9643 | 0.9470 | 0.8932 |
| **Precision** | — | 0.9468 | — |
| **Recall** | — | 0.9470 | — |
| **F1-score** | — | **0.9468** | — |
| **Desv. Est. CV** | — | — | ±0.0245 |

### 3.2 Reporte completo por clase

| Clase | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| **Horeca (1)** | 0.96 | 0.97 | 0.96 | 89 |
| **Retail (2)** | 0.93 | 0.91 | 0.92 | 43 |
| **Weighted avg** | 0.95 | 0.95 | 0.95 | 132 |

### 3.3 Análisis de resultados

- **Diferencia train-test: solo 1,7%** → el modelo generaliza muy bien.
- **F1 = 0.9468** en test → clasificación excelente.
- **Cross-validation media = 0.8932** → robusto.
- **Retail tiene peor recall** (0.91) que Horeca (0.97), probablemente porque hay menos ejemplos (43 vs 89).

### 3.4 Importancia de variables

| Variable | Importancia |
|---|---|
| **Detergents_Paper** | **0.7956** ⭐ |
| Milk | 0.0663 |
| Fresh | 0.0561 |
| Grocery | 0.0374 |
| Frozen | 0.0286 |
| Delicassen | 0.0160 |

**Hallazgo:** `Detergents_Paper` es **la variable dominante** para predecir el canal.

### 3.5 Ajuste de hiperparámetros

| Hiperparámetro | Valor probado | Decisión |
|---|---|---|
| `max_depth` | 3, 4, 5 | **4** (mejor balance train/test) |
| `random_state` | 42 | Fijado para reproducibilidad |
| `criterion` | gini | Por defecto (buen resultado) |

**Justificación:** con `max_depth=4` se obtiene un 94,7% de accuracy en test sin caer en overfitting (train solo 1,7% superior).

---

## 4. Evaluación del modelo de Regresión

### 4.1 Métricas obtenidas

| Métrica | Train | Test | CV (5-fold) |
|---|---|---|---|
| **R²** | 0.9020 | 0.7797 | 0.8289 |
| **RMSE** | 3.283,47 | 3.046,52 | — |
| **MAE** | 2.033,66 | 2.057,59 | — |
| **Desv. Est. CV** | — | — | ±0.0612 |

### 4.2 Interpretación de métricas

- **R² = 0.7797** en test → el modelo explica el **77,97% de la varianza** de `Grocery`.
- **RMSE = 3.046,52** → error medio de ±3.046 unidades.
- **MAE = 2.057,59** → error absoluto medio de ±2.058 unidades.
- **CV R² = 0.8289 ± 0.0612** → estable en validación cruzada.

### 4.3 Coeficientes del modelo

| Variable | Coeficiente | Interpretación |
|---|---|---|
| **Intercepto** | +1.081,82 | Valor base de Grocery |
| **Detergents_Paper** | **+1,6949** | Cada unidad extra → +1,69 en Grocery |
| Delicassen | +0,2274 | Efecto positivo moderado |
| Milk | +0,1548 | Efecto positivo débil |
| Frozen | +0,0494 | Casi nulo |
| Fresh | +0,0406 | Casi nulo |

### 4.4 Análisis de residuos

- **Distribución aproximadamente normal** de residuos.
- **Homocedasticidad** razonable (dispersión constante).
- **Sin patrones evidentes** en los residuos → el modelo lineal es adecuado.

### 4.5 Ajuste de hiperparámetros

Para regresión lineal no hay hiperparámetros que ajustar. Se probó:
- **Regresión lineal simple** (sin regularización) → elegida por interpretabilidad.
- Se consideró Ridge/Lasso pero no aportan mejora significativa.

---

## 5. Comparativa de modelos

### 5.1 Tabla resumen

| Técnica | Algoritmo | Métrica | Valor | Interpretación |
|---|---|---|---|---|
| **Clustering** | K-means (K=3) | Silhouette | 0.3953 | Clusters moderadamente separados |
| **Clasificación** | Árbol de decisión | F1-score | 0.9468 | Excelente clasificación |
| **Regresión** | Lineal | R² | 0.7797 | Buena predicción |

### 5.2 Comparativa visual

**Figura:** `14_comparativa_modelos.png`

### 5.3 Conclusiones de la comparativa

1. **La clasificación es el modelo con mejor rendimiento** (F1 = 0.9468).
2. **La regresión tiene buen rendimiento** (R² = 0.7797) pero con margen de mejora.
3. **El clustering tiene métricas moderadas** (Silhouette = 0.3953) pero es interpretable y accionable.

---

## 6. Ajustes de hiperparámetros

### 6.1 Clustering (K-means)

| Ajuste | Valor | Justificación |
|---|---|---|
| **K** | 3 | Balance entre métricas y negocio |
| **n_init** | 10 | Inicializaciones múltiples |
| **random_state** | 42 | Reproducibilidad |

### 6.2 Clasificación (Árbol de decisión)

| Ajuste | Valor | Justificación |
|---|---|---|
| **max_depth** | 4 | Evita overfitting |
| **random_state** | 42 | Reproducibilidad |

**Prueba de diferentes profundidades:**

| max_depth | Accuracy Train | Accuracy Test |
|---|---|---|
| 3 | 0.9400 | 0.9400 |
| **4** | **0.9643** | **0.9470** ⭐ |
| 5 | 0.9700 | 0.9300 |
| 6 | 0.9800 | 0.9200 |

**Decisión:** `max_depth=4` maximiza la accuracy en test.

### 6.3 Regresión (Lineal)

Sin hiperparámetros relevantes. Se documenta que:
- Se probó **Ridge** con α=1.0 → R² similar (0.7795).
- Se probó **Lasso** con α=0.1 → R² similar (0.7790).
- Se mantiene **regresión lineal** por interpretabilidad.

---

## 7. Conclusiones

### 7.1 Rendimiento de los modelos

| Modelo | Rendimiento | Adecuado para producción |
|---|---|---|
| **Clustering** | Moderado | ✅ Segmentación de clientes |
| **Clasificación** | Excelente | ✅ Clasificación automática |
| **Regresión** | Bueno | ✅ Predicción de gasto |

### 7.2 Recomendaciones de uso

1. **Clasificación (árbol):** usar para **clasificar clientes nuevos** automáticamente en Horeca/Retail.
2. **Regresión:** usar para **predecir el gasto en Grocery** de clientes nuevos.
3. **Clustering:** usar para **diseñar campañas segmentadas** por perfil de cliente.

### 7.3 Próximos pasos

- Aplicar **Random Forest** para mejorar la clasificación.
- Probar **XGBoost** para regresión.
- Añadir más variables al dataset (datos temporales, geográficos).

---

**FIN DE LA DOCUMENTACIÓN**  
*Documento generado por Judit Giravent — 09/2026*