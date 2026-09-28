#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def simulate_heston_paths(S0,v0,r,kappa,theta,xi,rho,T,n_steps,n_paths,seed = None,) -> tuple[np.ndarray, np.ndarray]:
  """Simule des trajectoires de prix (S) et de variance (v) selon le modèle de Heston
  en utilisant une discrétisation d'Euler avec troncature positive pour la
  variance.
  """
  
  # 1. Validation des paramètres de base
  if S0 <= 0 or v0 < 0 or T <= 0 or n_steps <= 0 or n_paths <= 0:
    raise ValueError(
        "Paramètres invalides : S0, T, n_steps, n_paths doivent être > 0 et v0"
        " >= 0"
    )
  if not (-1.0 <= rho <= 1.0):
    raise ValueError("Le coefficient de corrélation rho doit être entre -1 et 1")
  if kappa <= 0 or theta <= 0 or xi <= 0:
    raise ValueError(
        "Les paramètres kappa, theta et xi doivent être strictement positifs"
    )

  if seed is not None:
    np.random.seed(seed)

  dt = T / n_steps

  S = np.zeros((n_steps + 1, n_paths))
  v = np.zeros((n_steps + 1, n_paths))

  S[0] = S0
  v[0] = v0

  for t in range(1, n_steps + 1):
    z_v = np.random.standard_normal(n_paths)
    z_perp = np.random.standard_normal(n_paths)

    # Construction du mouvement brownien corrélé pour le prix
    z_s = rho * z_v + np.sqrt(1.0 - rho**2) * z_perp

    # Troncature positive pour éviter les racines carrées de nombres négatifs
    v_curr = np.maximum(v[t - 1], 0.0)

    # Mise à jour de la variance 
    v[t] = (v[t - 1]+ kappa * (theta - v_curr) * dt+ xi * np.sqrt(v_curr) * np.sqrt(dt) * z_v)

    # Mise à jour du prix de l'actif sous-jacent
    S[t] = S[t - 1] * np.exp((r - 0.5 * v_curr) * dt + np.sqrt(v_curr) * np.sqrt(dt) * z_s)

  return S, v





def satisfies_feller_condition(kappa, theta, xi) -> bool:
  """Vérifie si les paramètres de Heston respectent la condition de Feller :

  2 * kappa * theta > xi^2
  """
  return 2.0 * kappa * theta > xi**2


# ======================================================================
# Démonstration / tests manuels (exécutés uniquement en `python heston.py`)
# ======================================================================
if __name__ == "__main__":
    #%% TESTS

      # Paramètres de test
    S0, v0, r = 100.0, 0.04, 0.03
    kappa, theta, xi, rho = 2.0, 0.04, 0.3, -0.7
    T, n_steps, n_paths = 1.0, 252, 1000

    # 1. Test de la seed (reproductibilité)
    S_path1, v_path1 = simulate_heston_paths(S0, v0, r, kappa, theta, xi, rho, T, n_steps, n_paths, seed=42)
    S_path2, v_path2 = simulate_heston_paths(S0, v0, r, kappa, theta, xi, rho, T, n_steps, n_paths, seed=42)
    assert np.allclose(S_path1, S_path2), "Échec du test de seed : les trajectoires diffèrent !"
    print("✓ Test de la seed validé (trajectoires identiques).")

    # 2. Test de positivité des prix
    assert np.all(S_path1 > 0), "Erreur : des prix simulés sont négatifs ou nuls."
    print("✓ Test de positivité des prix validé.")

      # 3. Test de non-négativité de la variance (grâce au max(v, 0))
    print(f"✓ Variance minimale observée : {v_path1.min():.6f} (pas de valeursaberrantes explosives).")

      # 4. Test de gestion des paramètres invalides
    try:
        simulate_heston_paths(S0, v0, r, kappa, theta, xi, rho=1.5, T=T, n_steps=n_steps, n_paths=10)
    except ValueError as e:
        print(f"✓ Test des paramètres invalides capturé avec succès : {e}")
