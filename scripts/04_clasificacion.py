# ============================================================
# RETO 8 — Fase 4: Clasificación con Árbol de Decisión
# Predicción del canal (Channel) a partir de las compras
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import pickle

BASE = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE / "datos" / "raw" / "wholesale_customers.csv"
RUTA_FIGURAS = BASE / "salidas" / "figuras"
RUTA_RESULTADOS = BASE / "salidas" / "resultados"
RUTA_MODELOS = BASE / "modelos"

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ============================================================
# 1. CARGA
# ============================================================
print("=" * 60)
print("1. CARGA DEL DATASET ORIGINAL")
print("=" * 60)

df = pd.read_csv(RUTA_DATOS)
variables_gasto = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

X = df[variables_gasto]
y = df['Channel']  # 1 = Horeca, 2 = Retail

print(f"X: {X.shape}, y: {y.shape}")
print(f"Distribución de y:\n{y.value_counts()}")

# ============================================================
# 2. DIVISIÓN TRAIN/TEST
# ============================================================
print("\n" + "=" * 60)
print("2. DIVISIÓN TRAIN/TEST")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

# ============================================================
# 3. ENTRENAR ÁRBOL DE DECISIÓN
# ============================================================
print("\n" + "=" * 60)
print("3. ENTRENAR ÁRBOL DE DECISIÓN")
print("=" * 60)

modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
modelo.fit(X_train, y_train)

# Predicciones
y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)

# ============================================================
# 4. EVALUACIÓN DEL MODELO
# ============================================================
print("\n" + "=" * 60)
print("4. EVALUACIÓN DEL MODELO")
print("=" * 60)

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')

print(f"\n--- Métricas en TRAIN ---")
print(f"Accuracy: {accuracy_train:.4f}")

print(f"\n--- Métricas en TEST ---")
print(f"Accuracy:  {accuracy_test:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\n--- Reporte completo ---")
print(classification_report(y_test, y_pred_test))

# Cross-validation
cv_scores = cross_val_score(modelo, X, y, cv=5, scoring='accuracy')
print(f"\n--- Cross-Validation (5-fold) ---")
print(f"Media: {cv_scores.mean():.4f}")
print(f"Desv. Est.: {cv_scores.std():.4f}")

# Guardar métricas
metricas = pd.DataFrame({
    'Métrica': ['Accuracy_train', 'Accuracy_test', 'Precision_test',
                'Recall_test', 'F1_test', 'CV_media', 'CV_std'],
    'Valor': [accuracy_train, accuracy_test, precision, recall, f1,
              cv_scores.mean(), cv_scores.std()]
})
metricas.to_csv(RUTA_RESULTADOS / "metricas_clasificacion.csv", index=False)

# ============================================================
# 5. IMPORTANCIA DE VARIABLES
# ============================================================
print("\n" + "=" * 60)
print("5. IMPORTANCIA DE VARIABLES")
print("=" * 60)

importancias = pd.DataFrame({
    'Variable': variables_gasto,
    'Importancia': modelo.feature_importances_
}).sort_values('Importancia', ascending=False)

print(importancias.to_string(index=False))
importancias.to_csv(RUTA_RESULTADOS / "importancia_variables.csv", index=False)

# ============================================================
# 6. VISUALIZACIONES
# ============================================================
print("\n" + "=" * 60)
print("6. VISUALIZACIONES")
print("=" * 60)

# --- 6.1 Árbol de decisión ---
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(modelo, feature_names=variables_gasto, class_names=['Horeca', 'Retail'],
          filled=True, rounded=True, ax=ax, fontsize=10)
plt.title('Árbol de Decisión — Predicción del Canal', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "08_arbol_decision.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 08_arbol_decision.png")

# --- 6.2 Matriz de confusión ---
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Horeca', 'Retail'], yticklabels=['Horeca', 'Retail'])
ax.set_title('Matriz de Confusión', fontweight='bold')
ax.set_xlabel('Predicho')
ax.set_ylabel('Real')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "09_matriz_confusion.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 09_matriz_confusion.png")

# --- 6.3 Importancia de variables ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importancias['Variable'], importancias['Importancia'],
        color='seagreen', edgecolor='black')
ax.set_title('Importancia de las variables', fontweight='bold')
ax.set_xlabel('Importancia relativa')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "10_importancia_variables.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 10_importancia_variables.png")

# ============================================================
# 7. GUARDAR MODELO
# ============================================================
with open(RUTA_MODELOS / "modelo_clasificacion_tree.pkl", 'wb') as f:
    pickle.dump(modelo, f)
print(f"\n✅ Modelo guardado: modelo_clasificacion_tree.pkl")

print("\n" + "=" * 60)
print("✅ CLASIFICACIÓN COMPLETADA")
print("=" * 60)