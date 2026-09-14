# ============================================================
# RETO 8 — Fase 1: Exploración inicial del dataset
# Dataset: Wholesale Customers (UCI)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Rutas ---
BASE = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE / "datos" / "raw" / "wholesale_customers.csv"
RUTA_FIGURAS = BASE / "salidas" / "figuras"
RUTA_RESULTADOS = BASE / "salidas" / "resultados"
RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

# --- Configuración de gráficos ---
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ============================================================
# 1. CARGA DEL DATASET
# ============================================================
print("=" * 60)
print("1. CARGA DEL DATASET")
print("=" * 60)

df = pd.read_csv(RUTA_DATOS)

print(f"\nDimensiones: {df.shape}")
print(f"Columnas: {df.columns.tolist()}")

print("\n--- Primeras 5 filas ---")
print(df.head())

print("\n--- Últimas 5 filas ---")
print(df.tail())

print("\n--- Tipos de datos ---")
print(df.dtypes)

print("\n--- Información general ---")
print(df.info())

# ============================================================
# 2. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================
print("\n" + "=" * 60)
print("2. ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 60)

print("\n--- Resumen estadístico ---")
print(df.describe().round(2))

# Guardar
df.describe().to_csv(RUTA_RESULTADOS / "estadisticas_descriptivas.csv")

# ============================================================
# 3. VALORES NULOS Y DUPLICADOS
# ============================================================
print("\n" + "=" * 60)
print("3. CALIDAD DE LOS DATOS")
print("=" * 60)

print("\n--- Valores nulos ---")
print(df.isnull().sum())

print(f"\n--- Filas duplicadas ---")
print(f"Total duplicados: {df.duplicated().sum()}")

# ============================================================
# 4. VARIABLES CATEGÓRICAS
# ============================================================
print("\n" + "=" * 60)
print("4. VARIABLES CATEGÓRICAS")
print("=" * 60)

print("\n--- Channel (1=Horeca, 2=Retail) ---")
print(df['Channel'].value_counts())

print("\n--- Region (1=Lisbon, 2=Oporto, 3=Other) ---")
print(df['Region'].value_counts())

# ============================================================
# 5. VISUALIZACIONES
# ============================================================
print("\n" + "=" * 60)
print("5. VISUALIZACIONES")
print("=" * 60)

# --- 5.1 Histogramas de todas las variables numéricas ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
variables = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

for i, var in enumerate(variables):
    ax = axes[i // 3, i % 3]
    ax.hist(df[var], bins=30, color='steelblue', edgecolor='white')
    ax.set_title(f'Distribución de {var}', fontweight='bold')
    ax.set_xlabel('Gasto (unidades monetarias)')
    ax.set_ylabel('Frecuencia')

plt.suptitle('Distribución de las variables de gasto', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "01_distribucion_variables.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 01_distribucion_variables.png")

# --- 5.2 Matriz de correlación ---
fig, ax = plt.subplots(figsize=(10, 8))
matriz_corr = df.corr().round(3)
sns.heatmap(matriz_corr, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=1, ax=ax,
            cbar_kws={'label': 'Correlación'})
ax.set_title('Matriz de correlación', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "02_matriz_correlacion.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 02_matriz_correlacion.png")

# --- 5.3 Boxplots por canal ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, var in enumerate(variables):
    ax = axes[i // 3, i % 3]
    sns.boxplot(x='Channel', y=var, data=df, ax=ax, hue='Channel', palette='Set2', legend=False)
    ax.set_title(f'{var} por Canal', fontweight='bold')
    ax.set_xlabel('Channel (1=Horeca, 2=Retail)')

plt.suptitle('Distribución de gasto por canal de compra', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "03_boxplots_por_canal.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 03_boxplots_por_canal.png")

# ============================================================
# 6. RESUMEN
# ============================================================
print("\n" + "=" * 60)
print("✅ EXPLORACIÓN COMPLETADA")
print("=" * 60)
print(f"\nDataset: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"Valores nulos: {df.isnull().sum().sum()}")
print(f"Duplicados: {df.duplicated().sum()}")
print(f"Figuras guardadas en: {RUTA_FIGURAS}")