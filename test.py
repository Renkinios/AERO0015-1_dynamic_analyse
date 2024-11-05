import numpy as np

# Exemple de données
wd    = np.array([0, 1, 5, 10, 30, 40])  # Liste des vitesses
idx   = np.array([1, 3, 5, 7, 10, 12])  # Index actuel
idx_0 = np.array([1, 7, 5, 3, 12, 10])  # Index de référence

# Fonction pour appliquer les permutations inverses pour aligner `wd` en fonction de `idx` et `idx_0`
def apply_full_inverse_permutations(wd, idx, idx_0):
    # Trouver la correspondance des indices dans `idx` par rapport à `idx_0`
    target_order = np.argsort(np.argsort(idx_0))
    current_order = np.argsort(np.argsort(idx))
    
    # Trouver les permutations nécessaires
    permutation = [np.where(current_order == i)[0][0] for i in target_order]
    
    # Appliquer les permutations sur `wd` pour le remettre dans le bon ordre
    wd_corrected = wd[permutation]
    
    return wd_corrected

# Appliquer la fonction pour corriger `wd`
wd_corrige = apply_full_inverse_permutations(wd, idx, idx_0)

print("Vitesses initiales:", wd)
print("Vitesses corrigées:", wd_corrige)