#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from market_data import get_option_chain
from data_cleaning import clean_option_chain
from implied_vol import add_implied_volatility



def build_volatility_surface(df: pd.DataFrame) -> pd.DataFrame:
  """Prépare et retourne le DataFrame pour la surface de volatilité.

  Conserve uniquement les calculs réussis et calcule le log-moneyness.
  """
  # Filtrage strict sur les succès du solveur
  df_surf = df[df["iv_status"] == "success"].copy()

  if df_surf.empty:
    raise ValueError(
        "Aucune donnée d'IV valide ('success') trouvée dans le DataFrame."
    )

  # Calcul du log-moneyness : ln(K / S)
  df_surf["log_moneyness"] = np.log(df_surf["strike"] / df_surf["spot"])

  # Sélection des colonnes requises
  cols_to_keep = [
      "T",
      "strike",
      "log_moneyness",
      "implied_vol",
      "expiration",
      "option_type",
      "spot",
  ]
  existing_cols = [c for c in cols_to_keep if c in df_surf.columns]

  return df_surf[existing_cols].reset_index(drop=True)


def plot_volatility_surface_3d(surface_df: pd.DataFrame, save_path=None):
  """Trace la surface de volatilité en 3D (Scatter 3D)."""
  fig = plt.figure(figsize=(12, 8))
  ax = fig.add_subplot(projection="3d")

  x = surface_df["log_moneyness"]
  y = surface_df["T"]
  z = surface_df["implied_vol"] * 100  # Conversion en pourcentages

  # Nuage de points 3D
  img = ax.scatter(x, y, z, c=z, cmap="viridis", s=12, alpha=0.6)

  ax.set_title("Surface de Volatilité Implicite — Scatter 3D", fontsize=14)
  ax.set_xlabel(r"Log-Moneyness $\ln(K/S)$", fontsize=11, labelpad=10)
  ax.set_ylabel("Maturité $T$ (années)", fontsize=11, labelpad=10)
  ax.set_zlabel("Implied Volatility (%)", fontsize=11, labelpad=10)

  fig.colorbar(img, ax=ax, label="IV (%)", shrink=0.6, pad=0.1)
  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
  plt.show()
  
  
def plot_volatility_heatmap(
    surface_df: pd.DataFrame, n_bins_moneyness=30, n_bins_mat=20, save_path=None
):
  """Construit et affiche une Heatmap de la volatilité implicite."""
  df_hm = surface_df.copy()

  # Discrétisation en quantiles pour former une grille propre
  df_hm["moneyness_bin"] = pd.qcut(
      df_hm["log_moneyness"], q=n_bins_moneyness, duplicates="drop"
  )
  df_hm["maturity_bin"] = pd.qcut(
      df_hm["T"], q=n_bins_mat, duplicates="drop"
  )

  # Création de la grille pivot avec la médiane pour les doublons
  pivot = df_hm.pivot_table(
      index="maturity_bin",
      columns="moneyness_bin",
      values="implied_vol",
      aggfunc="median",
  )

  plt.figure(figsize=(12, 7))
  # origin='lower' pour avoir les maturités croissantes du bas vers le haut
  plt.imshow(
      pivot.values * 100,
      aspect="auto",
      origin="lower",
      cmap="viridis",
      interpolation="nearest",
  )

  plt.colorbar(label="Implied Volatility (%)")
  plt.title(
      "Heatmap de la Volatilité Implicite (Maturité vs Log-Moneyness)",
      fontsize=13,
  )
  plt.xlabel("Log-Moneyness (Bins)", fontsize=11)
  plt.ylabel("Maturité $T$ (Bins)", fontsize=11)

  # Nettoyage des ticks de l'axe X pour la lisibilité
  step_x = max(1, len(pivot.columns) // 10)
  plt.xticks(
      ticks=range(0, len(pivot.columns), step_x),
      labels=[str(col.mid)[:4] for col in pivot.columns[::step_x]],
      rotation=45,
  )

  step_y = max(1, len(pivot.index) // 10)
  plt.yticks(
      ticks=range(0, len(pivot.index), step_y),
      labels=[f"{idx.mid:.2f}" for idx in pivot.index[::step_y]],
  )

  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
  plt.show()


# ======================================================================
# Démonstration / tests manuels (exécutés uniquement en `python volatility_surface.py`)
# ======================================================================
if __name__ == "__main__":
    #%%

    r_market = 0.02
    q_market = 0.015
    cleaned_df = clean_option_chain(get_option_chain())[0]
    df_with_iv = add_implied_volatility(cleaned_df, r=r_market, q=q_market)

    # 1. Construction de la surface à partir du DataFrame avec IV
    surface_data = build_volatility_surface(df_with_iv)

    # 2. Affichage du scatter 3D (9.1)
    plot_volatility_surface_3d(surface_data)

    # 3. Affichage de la heatmap (9.2)
    #plot_volatility_heatmap(surface_data)
