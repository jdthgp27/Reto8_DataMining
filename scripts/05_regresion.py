# ============================================================
# RETO 8 — Fase 5: Regresión Lineal
# Predicción del gasto en Grocery a partir de otras categorías
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
print("1. CARGA DEL DATASET")
print("=" * 60)

df = pd.read_csv(RUTA_DATOS)

# Variable objetivo: Grocery (tiene correlación alta con Milk y Detergents_Paper)
variable_objetivo = 'Grocery'
variables_predictoras = ['Fresh', 'Milk', 'Frozen', 'Detergents_Paper', 'Delicassen']

X = df[variables_predictoras]
y = df[variable_objetivo]

print(f"Variable objetivo: {variable_objetivo}")
print(f"Predictoras: {variables_predictoras}")
print(f"X: {X.shape}, y: {y.shape}")

# ============================================================
# 2. DIVISIÓN TRAIN/TEST
# ============================================================
print("\n" + "=" * 60)
print("2. DIVISIÓN TRAIN/TEST")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

# ============================================================
# 3. ENTRENAR REGRESIÓN LINEAL
# ============================================================
print("\n" + "=" * 60)
print("3. ENTRENAR REGRESIÓN LINEAL")
print("=" * 60)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

print(f"\nIntercepto: {modelo.intercept_:.2f}")
print("\nCoeficientes:")
for var, coef in zip(variables_predictoras, modelo.coef_):
    print(f"  {var}: {coef:+.4f}")

# ============================================================
# 4. PREDICCIONES Y EVALUACIÓN
# ============================================================
print("\n" + "=" * 60)
print("4. EVALUACIÓN DEL MODELO")
print("=" * 60)

y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)

# Métricas
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"\n--- Métricas en TRAIN ---")
print(f"R²:   {r2_train:.4f}")
print(f"RMSE: {rmse_train:.2f}")
print(f"MAE:  {mae_train:.2f}")

print(f"\n--- Métricas en TEST ---")
print(f"R²:   {r2_test:.4f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"MAE:  {mae_test:.2f}")

# Cross-validation
cv_r2 = cross_val_score(modelo, X, y, cv=5, scoring='r2')
print(f"\n--- Cross-Validation R² (5-fold) ---")
print(f"Media: {cv_r2.mean():.4f}")
print(f"Desv. Est.: {cv_r2.std():.4f}")

# Guardar métricas
metricas = pd.DataFrame({
    'Métrica': ['R2_train', 'R2_test', 'RMSE_train', 'RMSE_test',
                'MAE_train', 'MAE_test', 'CV_R2_media', 'CV_R2_std'],
    'Valor': [r2_train, r2_test, rmse_train, rmse_test,
              mae_train, mae_test, cv_r2.mean(), cv_r2.std()]
})
metricas.to_csv(RUTA_RESULTADOS / "metricas_regresion.csv", index=False)

# Guardar coeficientes
coefs = pd.DataFrame({
    'Variable': variables_predictoras,
    'Coeficiente': modelo.coef_
})
coefs.to_csv(RUTA_RESULTADOS / "coeficientes_regresion.csv", index=False)

# ============================================================
# 5. VISUALIZACIONES
# ============================================================
print("\n" + "=" * 60)
print("5. VISUALIZACIONES")
print("=" * 60)

# --- 5.1 Real vs Predicho ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y_test, y_pred_test, alpha=0.6, color='steelblue', edgecolors='black')
# Línea ideal
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción perfecta')
ax.set_xlabel('Valores reales')
ax.set_ylabel('Valores predichos')
ax.set_title(f'Real vs Predicho — {variable_objetivo} (R²={r2_test:.4f})',
             fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "11_regresion_real_vs_predicho.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 11_regresion_real_vs_predicho.png")

# --- 5.2 Residuos ---
residuos = y_test - y_pred_test
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Residuos vs Predichos
axes[0].scatter(y_pred_test, residuos, alpha=0.6, color='coral', edgecolors='black')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Valores predichos')
axes[0].set_ylabel('Residuos')
axes[0].set_title('Residuos vs Valores predichos', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Histograma de residuos
axes[1].hist(residuos, bins=30, color='seagreen', edgecolor='white')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Residuos')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title('Distribución de residuos', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Análisis de residuos del modelo de regresión', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "12_analisis_residuos.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 12_analisis_residuos.png")

# --- 5.3 Coeficientes ---
fig, ax = plt.subplots(figsize=(10, 6))
colores = ['seagreen' if c > 0 else 'crimson' for c in coefs['Coeficiente']]
ax.barh(coefs['Variable'], coefs['Coeficiente'], color=colores, edgecolor='black')
ax.axvline(x=0, color='black', linewidth=1)
ax.set_title('Coeficientes de la regresión lineal', fontweight='bold')
ax.set_xlabel('Valor del coeficiente')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "13_coeficientes_regresion.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 13_coeficientes_regresion.png")

# ============================================================
# 6. GUARDAR MODELO
# ============================================================
with open(RUTA_MODELOS / "modelo_regresion_lineal.pkl", 'wb') as f:
    pickle.dump(modelo, f)
print(f"\n✅ Modelo guardado: modelo_regresion_lineal.pkl")

print("\n" + "=" * 60)
print("✅ REGRESIÓN COMPLETADA")
print("=" * 60)