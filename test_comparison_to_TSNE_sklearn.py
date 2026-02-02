import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from time import time
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from tSNH import tSNH # notre fonction crée


iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
X_scaled = StandardScaler().fit_transform(X) 
PERPLEXITY = 30.0
T = 1000       # Nombre d'itérations
ETA = 200.0   # Learning rate (taux d'apprentissage)
ALPHA = 0.5   # Momentum initial (géré par tSNH en 0.5/0.8)
D = 2         # Dimension de sortie
RANDOM_STATE = 42

print("--- Préparation des Exécutions ---")
print(f"Paramètres: Perp={PERPLEXITY}, T={T}, Init='random', Method='exact'")


# --- 1. Exécution de notre implémentation tSNH ---
print("\n[1/2] Exécution de notre Implémentation Manuelle (tSNH)...")
t0_manual = time()
# Note: Le paramètre 'alpha' est toujours requis par la signature, 
# même si la valeur est écrasée à l'intérieur de tSNH.
np.random.seed(RANDOM_STATE) 
# Cela assure que le Y initial de  tSNH (rng.normal) et le Y de Sklearn partent du même 'bruit' initial.	
Y_manual = tSNH(X_scaled, PERPLEXITY, T, ETA, ALPHA, D, random_state=RANDOM_STATE)
t1_manual = time()
print(f"-> tSNH terminé en {t1_manual - t0_manual:.4f} secondes.")


# --- 2. Exécution de l'implémentation Sklearn ---
print("\n[2/2] Exécution de l'Implémentation Sklearn (TSNE)...")
t0_sklearn = time()
tsne = TSNE(
	n_components=D,
	perplexity=PERPLEXITY,
	max_iter=T,
	random_state=RANDOM_STATE,
	init='random',      # Force l'initialisation aléatoire
	learning_rate=ETA,  
	method='exact',     # Force l'algorithme exact
	n_jobs=1            # Pour un comportement cohérent
)
Y_sklearn = tsne.fit_transform(X_scaled)
t1_sklearn = time()
print(f"-> Sklearn TSNE terminé en {t1_sklearn - t0_sklearn:.4f} secondes.")
print(f"\nDivergence KL finale (accessible après l'entraînement): {tsne.kl_divergence_:.4f}")
#c'est pour avoir la valeur de la divergence KL finale


# --- 3. Affichage des Résultats avec Matplotlib ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'Comparaison t-SNE (Perplexity={PERPLEXITY}, N_iter={T})', fontsize=16)

# Graphique de notre implémentation
scatter_manual = axes[0].scatter(Y_manual[:, 0], Y_manual[:, 1], c=y, cmap=plt.cm.get_cmap("viridis", 3), s=40 )
axes[0].set_title(" tSNH")
axes[0].set_xlabel("Dimension 1")
axes[0].set_ylabel("Dimension 2")
axes[0].grid(True, linestyle='--', alpha=0.6)

legend1=axes[0].legend(*scatter_manual.legend_elements(), title="Classes", loc="lower left", labels=target_names)
axes[0].add_artist(legend1)
# Graphique de l'implémentation Sklearn
scatter_sklearn = axes[1].scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=y, cmap=plt.cm.get_cmap("viridis", 3), s=40 )

axes[1].set_title("Sklearn TSNE (Exact, Random Init)")
axes[1].set_xlabel("Dimension 1")
axes[1].set_ylabel("Dimension 2")
axes[1].grid(True, linestyle='--', alpha=0.6)
	

legend2=axes[1].legend(*scatter_sklearn.legend_elements(), title="Classes", loc="lower left", labels=target_names)
axes[1].add_artist(legend2)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuster l'espacement pour le titre
plt.show()
