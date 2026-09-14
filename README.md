cd /c/Users/jdthg/Documents/curso4BI/Reto8_DataMining
cat > README.md << 'EOF'
# 🔍 Reto 8 — Data Mining con Python

Proyecto de **Data Mining** que aplica técnicas de clustering, clasificación y regresión al dataset **Wholesale Customers** de UCI.

## 🎯 Objetivo

Extraer insights valiosos del comportamiento de compra de 440 clientes mayoristas mediante:
- **Clustering (K-means)** → Segmentación de clientes.
- **Clasificación (Árbol de decisión)** → Predicción del canal de compra.
- **Regresión (Regresión lineal)** → Predicción del gasto total.

## 📊 Dataset

- **Fuente:** UCI Machine Learning Repository
- **Registros:** 440 clientes mayoristas
- **Variables:** 8 (Channel, Region, Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen)

## 🛠️ Tecnologías

- Python 3.13
- pandas · scikit-learn · matplotlib · seaborn

## 📁 Estructura
Reto8_DataMining/
├── datos/ (raw + limpios)
├── scripts/ (por fases)
├── notebooks/ (análisis principal)
├── modelos/ (modelos entrenados .pkl)
├── salidas/ (figuras + resultados)
└── docs/ (informes)

## 📊 Resultados de los modelos

| Técnica | Métrica | Valor | Interpretación |
|---|---|---|---|
| **Clustering** (K-means, K=3) | Silhouette | 0.3953 | 3 segmentos identificados |
| **Clasificación** (Árbol) | F1-score | 0.9468 | Excelente predicción del canal |
| **Regresión** (Lineal) | R² | 0.7797 | Explica 78% de la varianza |

## 💡 Hallazgos clave

1. **3 segmentos de clientes**: Fresh alto (restaurantes), Grocery alto (tiendas), pequeños.
2. **Detergents_Paper** es la variable dominante (79,6%) para predecir el canal.
3. **71,6% de clientes** son pequeños y aportan poco valor.
4. El modelo de clasificación puede predecir con **94,7% de precisión** si un cliente es Horeca o Retail.

## 📄 Documentación

- [Informe de análisis](docs/Informe_Analisis_Reto8_JuditGiravent.pdf)
- [Documentación de evaluación](docs/Documentacion_Evaluation_Reto8_JuditGiravent.pdf)

## 👤 Autora

**Judit Giravent** — [@jdthgp27](https://github.com/jdthgp27)

## 📜 Licencia

MIT
EOF