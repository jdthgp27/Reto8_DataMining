# ============================================================
# RETO 8 — Fase 6: Evaluación comparativa de modelos
# Consolida métricas de clustering, clasificación y regresión
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RUTA_RESULTADOS = BASE / "salidas" / "resultados"
RUTA_FIGURAS = BASE / "salidas" / "figuras"

sns.set_style("whitegrid")

# ============================================================
# 1. CARGAR MÉTRICAS
# ============================================================
print("=" * 60)
print("EVALUACIÓN COMPARATIVA DE MODELOS")
print("=" * 60)

# Clustering
clustering = pd.read_csv(RUTA_RESULTADOS / "evaluacion_clustering.csv")
print("\n--- CLUSTERING ---")
print(clustering.to_string(index=False))

# Clasificación
clasificacion = pd.read_csv(RUTA_RESULTADOS / "metricas_clasificacion.csv")
print("\n--- CLASIFICACIÓN ---")
print(clasificacion.to_string(index=False))

# Regresión
regresion = pd.read_csv(RUTA_RESULTADOS / "metricas_regresion.csv")
print("\n--- REGRESIÓN ---")
print(regresion.to_string(index=False))

# ============================================================
# 2. TABLA RESUMEN
# ============================================================
resumen = pd.DataFrame({
    'Técnica': ['Clustering (K-means)', 'Clasificación (Árbol)', 'Regresión (Lineal)'],
    'Métrica principal': ['Silhouette', 'F1-score', 'R²'],
    'Valor': [
        float(clustering[clustering['Métrica'] == 'Silhouette']['Valor'].values[0]),
        float(clasificacion[clasificacion['Métrica'] == 'F1_test']['Valor'].values[0]),
        float(regresion[regresion['Métrica'] == 'R2_test']['Valor'].values[0])
    ],
    'Interpretación': ['Clusters moderadamente separados',
                       'Excelente clasificación',
                       'Buena predicción']
})
print("\n--- RESUMEN ---")
print(resumen.to_string(index=False))
resumen.to_csv(RUTA_RESULTADOS / "resumen_modelos.csv", index=False)

# ============================================================
# 3. VISUALIZACIÓN COMPARATIVA
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Clustering
ax = axes[0]
metricas_clust = ['Silhouette', 'Davies-Bouldin']
valores_clust = [float(clustering[clustering['Métrica'] == m]['Valor'].values[0]) for m in metricas_clust]
ax.bar(metricas_clust, valores_clust, color=['seagreen', 'coral'], edgecolor='black')
ax.set_title('Clustering', fontweight='bold')
ax.set_ylabel('Valor')
for i, v in enumerate(valores_clust):
    ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
ax.grid(True, alpha=0.3)

# Clasificación
ax = axes[1]
metricas_clas = ['Accuracy_test', 'Precision_test', 'Recall_test', 'F1_test']
valores_clas = [float(clasificacion[clasificacion['Métrica'] == m]['Valor'].values[0]) for m in metricas_clas]
ax.bar(['Accuracy', 'Precision', 'Recall', 'F1'], valores_clas, color='steelblue', edgecolor='black')
ax.set_title('Clasificación', fontweight='bold')
ax.set_ylabel('Valor')
ax.set_ylim([0, 1.1])
for i, v in enumerate(valores_clas):
    ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
ax.grid(True, alpha=0.3)

# Regresión
ax = axes[2]
metricas_reg = ['R2_test']
valores_reg = [float(regresion[regresion['Métrica'] == m]['Valor'].values[0]) for m in metricas_reg]
ax.bar(['R²'], valores_reg, color='darkorange', edgecolor='black')
ax.set_title('Regresión', fontweight='bold')
ax.set_ylabel('Valor')
ax.set_ylim([0, 1.1])
for i, v in enumerate(valores_reg):
    ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('Evaluación comparativa de los 3 modelos', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "14_comparativa_modelos.png", dpi=100, bbox_inches='tight')
plt.close()
print("\n✅ Guardado: 14_comparativa_modelos.png")

print("\n" + "=" * 60)
print("✅ EVALUACIÓN COMPLETADA")
print("=" * 60)