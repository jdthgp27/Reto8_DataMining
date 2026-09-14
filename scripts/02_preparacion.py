# ============================================================
# RETO 8 — Fase 2: Preparación de datos
# Limpieza, normalización y codificación
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- Rutas ---
BASE = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE / "datos" / "raw" / "wholesale_customers.csv"
RUTA_LIMPIOS = BASE / "datos" / "limpios"
RUTA_FIGURAS = BASE / "salidas" / "figuras"
RUTA_RESULTADOS = BASE / "salidas" / "resultados"
RUTA_LIMPIOS.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ============================================================
# 1. CARGA
# ============================================================
print("=" * 60)
print("1. CARGA DEL DATASET")
print("=" * 60)

df = pd.read_csv(RUTA_DATOS)
print(f"Dimensiones originales: {df.shape}")

# ============================================================
# 2. DETECCIÓN Y TRATAMIENTO DE OUTLIERS
# ============================================================
print("\n" + "=" * 60)
print("2. DETECCIÓN DE OUTLIERS (método IQR)")
print("=" * 60)

variables_gasto = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

outliers_info = []
for var in variables_gasto:
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    n_outliers = ((df[var] < lim_inf) | (df[var] > lim_sup)).sum()
    outliers_info.append({
        'Variable': var,
        'Q1': round(Q1, 2),
        'Q3': round(Q3, 2),
        'IQR': round(IQR, 2),
        'Lim_inf': round(lim_inf, 2),
        'Lim_sup': round(lim_sup, 2),
        'N_outliers': n_outliers,
        '%_outliers': round(n_outliers / len(df) * 100, 2)
    })

df_outliers = pd.DataFrame(outliers_info)
print(df_outliers.to_string(index=False))
df_outliers.to_csv(RUTA_RESULTADOS / "outliers_por_variable.csv", index=False)

# --- Decisión: mantener outliers pero WINSORIZAR (cap) para clustering ---
print("\n--- Aplicando winsorización (cap a percentil 99) ---")
df_capped = df.copy()
for var in variables_gasto:
    limite_sup = df_capped[var].quantile(0.99)
    df_capped[var] = df_capped[var].clip(upper=limite_sup)

print(f"Dimensiones tras winsorización: {df_capped.shape}")

# ============================================================
# 3. NORMALIZACIÓN
# ============================================================
print("\n" + "=" * 60)
print("3. NORMALIZACIÓN DE VARIABLES")
print("=" * 60)

# --- 3.1 StandardScaler (media=0, std=1) ---
scaler_std = StandardScaler()
df_std = df_capped.copy()
df_std[variables_gasto] = scaler_std.fit_transform(df_capped[variables_gasto])

print("\n--- Variables estandarizadas (StandardScaler) ---")
print(df_std[variables_gasto].describe().round(3))

# --- 3.2 MinMaxScaler (rango 0-1) ---
scaler_minmax = MinMaxScaler()
df_minmax = df_capped.copy()
df_minmax[variables_gasto] = scaler_minmax.fit_transform(df_capped[variables_gasto])

print("\n--- Variables normalizadas (MinMaxScaler) ---")
print(df_minmax[variables_gasto].describe().round(3))

# ============================================================
# 4. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# ============================================================
print("\n" + "=" * 60)
print("4. CODIFICACIÓN DE VARIABLES CATEGÓRICAS")
print("=" * 60)

# One-Hot Encoding para Region (no ordinal)
df_encoded = pd.get_dummies(df_std, columns=['Region'], prefix='Region', drop_first=False)
print(f"\nColumnas tras One-Hot Encoding de Region: {df_encoded.columns.tolist()}")

# Etiquetas legibles para Channel
df_encoded['Channel_Label'] = df_encoded['Channel'].map({1: 'Horeca', 2: 'Retail'})
print(f"Channel etiquetado: {df_encoded['Channel_Label'].value_counts().to_dict()}")

# ============================================================
# 5. GUARDAR DATASETS
# ============================================================
print("\n" + "=" * 60)
print("5. GUARDAR DATASETS LIMPIOS")
print("=" * 60)

# Dataset estandarizado con codificación
df_encoded.to_csv(RUTA_LIMPIOS / "wholesale_preparado.csv", index=False)
print(f"✅ Guardado: wholesale_preparado.csv ({df_encoded.shape})")

# Dataset con MinMax para visualizaciones
df_minmax.to_csv(RUTA_LIMPIOS / "wholesale_minmax.csv", index=False)
print(f"✅ Guardado: wholesale_minmax.csv ({df_minmax.shape})")

# ============================================================
# 6. VISUALIZACIONES
# ============================================================
print("\n" + "=" * 60)
print("6. VISUALIZACIONES")
print("=" * 60)

# --- 6.1 Comparativa antes/después normalización ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, var in enumerate(variables_gasto):
    ax = axes[i // 3, i % 3]
    ax.hist(df_capped[var], bins=30, alpha=0.6, label='Original (capped)', color='steelblue')
    ax.hist(df_std[var], bins=30, alpha=0.6, label='Estandarizado', color='coral')
    ax.set_title(f'{var}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Distribución antes y después de la estandarización', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "04_normalizacion_comparativa.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 04_normalizacion_comparativa.png")

# ============================================================
# 7. RESUMEN
# ============================================================
print("\n" + "=" * 60)
print("✅ PREPARACIÓN COMPLETADA")
print("=" * 60)
print(f"\nDataset final: {df_encoded.shape}")
print(f"Variables numéricas: {len(variables_gasto)}")
print(f"Variables categóricas codificadas: {df_encoded.shape[1] - len(variables_gasto) - 1}")
print(f"Datasets guardados en: {RUTA_LIMPIOS}")