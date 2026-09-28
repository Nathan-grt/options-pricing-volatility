#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from market_data import get_option_chain
from data_cleaning import clean_option_chain
from implied_vol import add_implied_volatility


def build_diagnostic_table(
    raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, df_with_iv: pd.DataFrame
) -> pd.DataFrame:
  """Construit le tableau de diagnostic quantitatif de la qualité des données

  exigé à l'étape 8.
  """
  diagnostics = []

  # 1. Taux de cotations supprimées
  total_raw = len(raw_df)
  total_clean = len(cleaned_df)
  suppressed_count = total_raw - total_clean
  suppression_rate = (
      (suppressed_count / total_raw) * 100 if total_raw > 0 else 0.0
  )
  diagnostics.append({
      "Indicateur": "Taux de cotations supprimées",
      "Valeur": f"{suppression_rate:.2f} % ({suppressed_count:,d} / {total_raw:,d} lignes)",
  })

  # 2. Spread relatif médian
  median_rel_spread = cleaned_df["relative_spread"].median() * 100
  diagnostics.append({
      "Indicateur": "Spread relatif médian",
      "Valeur": f"{median_rel_spread:.2f} %",
  })

  # 3. Taux de succès IV
  total_analyzed = len(df_with_iv)
  success_count = (df_with_iv["iv_status"] == "success").sum()
  success_rate = (
      (success_count / total_analyzed) * 100 if total_analyzed > 0 else 0.0
  )
  diagnostics.append({
      "Indicateur": "Taux de succès IV",
      "Valeur": f"{success_rate:.2f} % ({success_count:,d} réussites)",
  })

  # 4. Écart source IV / IV recalculée
  valid_iv = df_with_iv[df_with_iv["iv_status"] == "success"].copy()
  if "source_iv" in valid_iv.columns:
    valid_iv = valid_iv.dropna(subset=["source_iv"])
    abs_diff = (valid_iv["implied_vol"] - valid_iv["source_iv"]).abs()
    mean_diff = abs_diff.mean() * 100
    diagnostics.append({
        "Indicateur": "Écart source IV / IV recalculée",
        "Valeur": f"{mean_diff:.2f} points de vol (en moyenne)",
    })
  else:
    diagnostics.append({
        "Indicateur": "Écart source IV / IV recalculée",
        "Valeur": "N/A (source_iv absente)",
    })

  # 5. Nombre de strikes par maturité (moyenne)
  strikes_per_mat = (
      cleaned_df.groupby("expiration")["strike"].nunique().mean()
  )
  diagnostics.append({
      "Indicateur": "Nombre de strikes par maturité",
      "Valeur": f"{strikes_per_mat:.1f} strikes (en moyenne)",
  })

  # 6. Nombre de points par zone (ex: analyse de la fragilité par maturité)
  mat_counts = cleaned_df["expiration"].value_counts()
  min_zone_pts = mat_counts.min()
  max_zone_pts = mat_counts.max()
  diagnostics.append({
      "Indicateur": "Nombre de points par zone",
      "Valeur": f"Min: {min_zone_pts} pts, Max: {max_zone_pts} pts par échéance",
  })

  return pd.DataFrame(diagnostics)


# ======================================================================
# Démonstration / tests manuels (exécutés uniquement en `python diagnostics.py`)
# ======================================================================
if __name__ == "__main__":
    #%%

    # Génération du tableau
    raw_data = get_option_chain(ticker="SPY")
    cleaned_data, audit_table = clean_option_chain(raw_data)
    final_iv_data = add_implied_volatility(cleaned_data, r=0.02, q=0.015)

    diagnostic_table = build_diagnostic_table(raw_data, cleaned_data, final_iv_data)

    print("\n" + "=" * 80)
    print("TABLEAU DE DIAGNOSTIC QUANTITATIF DE LA QUALITÉ DES DONNÉES")
    print("=" * 80)
    print(diagnostic_table.to_string(index=False))
