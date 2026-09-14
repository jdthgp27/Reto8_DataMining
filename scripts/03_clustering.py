# ============================================================
# RETO 8 — Fase 3: Clustering con K-means
# Segmentación de clientes mayoristas
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pickle

BASE = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE / "datos" / "limpios" / "wholesale_preparado.csv"
RUTA_FIGURAS = BASE / "salidas" / "figuras"
RUTA_RESULTADOS = BASE / "salidas" / "resultados"
RUTA_MODELOS = BASE / "modelos"
RUTA_MODELOS.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ============================================================
# 1. CARGA
# ============================================================
print("=" * 60)
print("1. CARGA DEL DATASET PREPARADO")
print("=" * 60)

df = pd.read_csv(RUTA_DATOS)
variables_gasto = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

X = df[variables_gasto]
print(f"Dimensiones: {X.shape}")

# ============================================================
# 2. DETERMINAR K ÓPTIMO
# ============================================================
print("\n" + "=" * 60)
print("2. DETERMINAR NÚMERO ÓPTIMO DE CLUSTERS")
print("=" * 60)

inertias = []
silhouettes = []
davies = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, labels))
    davies.append(davies_bouldin_score(X, labels))
    print(f"K={k}: Inercia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.4f}, Davies-Bouldin={davies[-1]:.4f}")

# Guardar métricas
metricas_k = pd.DataFrame({
    'K': list(K_range),
    'Inertia': inertias,
    'Silhouette': silhouettes,
    'Davies_Bouldin': davies
})
metricas_k.to_csv(RUTA_RESULTADOS / "metricas_kmeans.csv", index=False)

# ============================================================
# 3. VISUALIZACIÓN DE MÉTRICAS
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Codo
axes[0].plot(K_range, inertias, marker='o', color='steelblue', linewidth=2)
axes[0].set_title('Método del Codo (Inercia)', fontweight='bold')
axes[0].set_xlabel('Número de clusters (K)')
axes[0].set_ylabel('Inercia')
axes[0].grid(True, alpha=0.3)

# Silhouette
axes[1].plot(K_range, silhouettes, marker='s', color='coral', linewidth=2)
axes[1].set_title('Silhouette Score', fontweight='bold')
axes[1].set_xlabel('Número de clusters (K)')
axes[1].set_ylabel('Silhouette')
axes[1].grid(True, alpha=0.3)

# Davies-Bouldin
axes[2].plot(K_range, davies, marker='^', color='seagreen', linewidth=2)
axes[2].set_title('Davies-Bouldin Index', fontweight='bold')
axes[2].set_xlabel('Número de clusters (K)')
axes[2].set_ylabel('Davies-Bouldin')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Selección del número óptimo de clusters', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "05_seleccion_k.png", dpi=100, bbox_inches='tight')
plt.close()
print("\n✅ Guardado: 05_seleccion_k.png")

# ============================================================
# 4. ENTRENAR K-MEANS FINAL (K=3)
# ============================================================
print("\n" + "=" * 60)
print("4. ENTRENAR K-MEANS CON K=3")
print("=" * 60)

K_OPTIMO = 3  # Basado en Silhouette y Davies-Bouldin
kmeans_final = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X)

print(f"\nDistribución de clientes por cluster:")
print(df['Cluster'].value_counts().sort_index())

# Guardar modelo
with open(RUTA_MODELOS / "modelo_clustering_kmeans.pkl", 'wb') as f:
    pickle.dump(kmeans_final, f)
print(f"\n✅ Modelo guardado: modelo_clustering_kmeans.pkl")

# ============================================================
# 5. CARACTERIZAR CLUSTERS
# ============================================================
print("\n" + "=" * 60)
print("5. CARACTERIZACIÓN DE CLUSTERS")
print("=" * 60)

# Media por cluster (en escala original)
df_original = pd.read_csv(BASE / "datos" / "raw" / "wholesale_customers.csv")
df_original['Cluster'] = df['Cluster']
perfil = df_original.groupby('Cluster')[variables_gasto].mean().round(2)
print("\n--- Perfil medio por cluster (escala original) ---")
print(perfil.to_string())
perfil.to_csv(RUTA_RESULTADOS / "perfil_clusters.csv")

# Tamaño por cluster
tamanio = df['Cluster'].value_counts().sort_index().to_frame('N_Clientes')
tamanio['%'] = (tamanio['N_Clientes'] / len(df) * 100).round(2)
print("\n--- Tamaño de cada cluster ---")
print(tamanio)
tamanio.to_csv(RUTA_RESULTADOS / "tamanio_clusters.csv")

# ============================================================
# 6. VISUALIZACIÓN DE CLUSTERS
# ============================================================
print("\n" + "=" * 60)
print("6. VISUALIZACIONES DE CLUSTERS")
print("=" * 60)

# --- 6.1 Scatter plot de 2 variables ---
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(df_original['Fresh'], df_original['Grocery'],
                     c=df_original['Cluster'], cmap='viridis',
                     s=50, alpha=0.7, edgecolors='black')
ax.set_xlabel('Fresh')
ax.set_ylabel('Grocery')
ax.set_title(f'Clusters de clientes (K={K_OPTIMO})', fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "06_clusters_scatter.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 06_clusters_scatter.png")

# --- 6.2 Radar / Bar comparativo de perfiles ---
fig, ax = plt.subplots(figsize=(12, 6))
perfil.T.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='white')
ax.set_title('Perfil de gasto medio por cluster', fontweight='bold')
ax.set_xlabel('Categoría de producto')
ax.set_ylabel('Gasto medio')
ax.legend(title='Cluster')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "07_perfil_clusters.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 07_perfil_clusters.png")

# ============================================================
# 7. EVALUACIÓN FINAL
# ============================================================
print("\n" + "=" * 60)
print("7. EVALUACIÓN FINAL DEL CLUSTERING")
print("=" * 60)

sil_final = silhouette_score(X, df['Cluster'])
davies_final = davies_bouldin_score(X, df['Cluster'])
inertia_final = kmeans_final.inertia_

print(f"\nSilhouette Score: {sil_final:.4f}")
print(f"Davies-Bouldin Index: {davies_final:.4f}")
print(f"Inercia: {inertia_final:.2f}")

# Guardar resultado
pd.DataFrame({
    'Métrica': ['Silhouette', 'Davies-Bouldin', 'Inercia', 'K_optimo'],
    'Valor': [sil_final, davies_final, inertia_final, K_OPTIMO]
}).to_csv(RUTA_RESULTADOS / "evaluacion_clustering.csv", index=False)

# Guardar dataset con clusters
df.to_csv(BASE / "datos" / "limpios" / "wholesale_con_clusters.csv", index=False)
print(f"\n✅ Guardado: wholesale_con_clusters.csv")

print("\n" + "=" * 60)
print("✅ CLUSTERING COMPLETADO")
print("=" * 60)