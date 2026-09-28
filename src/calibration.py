#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from heston_pricing import heston_call_price
from heston_pricing import heston_put_price
from implied_vol import implied_volatility
from scipy.optimize import minimize


def _row_q(row, q):
  """q peut être un float ou le nom d'une colonne (ex. 'q_implied')."""
  return row[q] if isinstance(q, str) else q


def calibration_loss_iv(params, market_options_df, r, q=0.0, iv_col="implied_vol"):
  """MSE pondérée entre IV modèle (Heston) et IV de marché.

  iv_col : colonne d'IV de marché servant de cible. Par défaut 'implied_vol'
  (IV recalculée avec les MÊMES r et q que ceux utilisés pour inverser le prix
  Heston) : on compare ainsi des grandeurs cohérentes. 'source_iv' reste possible.
  """
  v0, kappa, theta, xi, rho = params

  # 1. Validation des contraintes physiques de base
  if v0 <= 0 or kappa <= 0 or theta <= 0 or xi <= 0 or not (-1.0 <= rho <= 1.0):
    return 1e10

  losses = []

  for _, row in market_options_df.iterrows():
    try:
      S0 = row["spot"]
      K = row["strike"]
      T = row["T"]
      iv_market = row[iv_col]
      q_i = _row_q(row, q)

      if pd.isna(iv_market) or T <= 0:
        continue

      opt_type = row.get("option_type", "call").lower()
      weight = row.get("weight", 1.0)

      # 2. Calcul du prix théorique Heston avec gestion d'exception
      if opt_type == "call":
        price_model = heston_call_price(S0, K, T, r, v0, kappa, theta, xi, rho, q_i)
      else:
        price_model = heston_put_price(S0, K, T, r, v0, kappa, theta, xi, rho, q_i)

      if np.isnan(price_model) or price_model <= 0:
        continue  # On ignore cette option si le modèle tousse

      # 3. Inversion de l'IV
      iv_model = implied_volatility(price_model, S0, K, T, r, opt_type, q_i)

      if np.isnan(iv_model):
        continue

      # 4. Écart quadratique pondéré
      loss_i = weight * (iv_model - iv_market) ** 2
      losses.append(loss_i)

    except Exception:
      # En cas d'erreur numérique sur un contrat, on passe au suivant
      continue

  # Si aucune option n'a pu être pricée correctement, on renvoie une grosse pénalité au lieu de NaN
  if len(losses) == 0:
    return 1e10

  return np.mean(losses)





def calibrate_heston(market_data, initial_guess, bounds, r, q=0.0, iv_col="implied_vol",
                     fd_eps=1e-4, maxiter=200, callback=None, fallback=True):
    """Calibre les paramètres de Heston (v0, kappa, theta, xi, rho)

    en minimisant l'écart d'IV par rapport au marché.

    fd_eps : pas des différences finies du gradient numérique de L-BFGS-B.
             Le défaut scipy (1e-8) est inférieur à la précision de l'inversion
             d'IV (xtol=1e-6) et à celle de l'intégration numérique : le gradient
             serait alors du bruit et l'optimiseur s'arrêterait prématurément.
    fallback : si L-BFGS-B s'arrête anormalement (échec de la recherche linéaire),
             on repart du point atteint avec Powell (sans gradient, avec bornes).
             Cause : calibration_loss_iv ignore les options dont l'IV modèle n'est
             pas inversible ; le nombre de termes de la moyenne change alors avec les
             paramètres, la perte est discontinue et le gradient numérique n'a plus
             de sens (observé sur la date de sell-off du 12/10/2022).
    """
    history = []

    # Fonction objective enveloppe pour scipy
    def objective(params):
        val = calibration_loss_iv(params, market_data, r, q, iv_col)
        history.append((*params, val))
        return val

    # Utilisation de L-BFGS-B qui accepte les bornes par paramètre
    result = minimize(
        objective,
        x0=initial_guess,
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": maxiter,
            "ftol": 1e-9,
            "eps": fd_eps,
            "disp": False,
        },
    )

    method_used = "L-BFGS-B"
    if fallback and not result.success:
        result_p = minimize(objective, x0=result.x, method="Powell", bounds=bounds,
                            options={"maxiter": 20, "xtol": 1e-3, "ftol": 1e-7, "disp": False})
        if result_p.fun < result.fun:
            result_p.nit = result.nit + result_p.nit
            result = result_p
        method_used = "L-BFGS-B + Powell"

    return {
        "success": result.success,
        "message": result.message,
        "method": method_used,
        "params": result.x,  # [v0, kappa, theta, xi, rho]
        "loss": result.fun,
        "n_iterations": result.nit,
        "n_evaluations": result.nfev,
        "history": pd.DataFrame(history, columns=["v0", "kappa", "theta", "xi", "rho", "loss"]),
    }


def validate_calibration(market_options_df, optimal_params, r, q=0.0, iv_col="implied_vol", verbose=True):
  """Valide la calibration de Heston en calculant les erreurs globales,

  par maturité et par log-moneyness (Étape 10).
  """
  v0, kappa, theta, xi, rho = optimal_params
  results = []

  #Calcul des prix et des IV du modèle pour chaque option
  for _, row in market_options_df.iterrows():
    S0 = row["spot"]
    K = row["strike"]
    T = row["T"]
    iv_market = row[iv_col]  # IV de marché (cible de la calibration)
    opt_type = row.get("option_type", "call").lower()
    q_i = _row_q(row, q)

    if pd.isna(iv_market) or T <= 0:
      continue

    # Prix théorique Heston
    if opt_type == "call":
      price_model = heston_call_price(S0, K, T, r, v0, kappa, theta, xi, rho, q_i)
    else:
      price_model = heston_put_price(S0, K, T, r, v0, kappa, theta, xi, rho, q_i)

    if np.isnan(price_model) or price_model <= 0:
      continue

    # Inversion de l'IV du modèle
    iv_model = implied_volatility(price_model, S0, K, T, r, opt_type, q_i)

    if np.isnan(iv_model):
      continue

    # Calcul des erreurs
    iv_error = iv_model - iv_market  # en décimales
    iv_error_pts = iv_error * 100  # en points de vol

    results.append({
        "expiration": row["expiration"],
        "option_type": opt_type,
        "T": T,
        "strike": K,
        "log_moneyness": np.log(K / S0),
        "iv_market": iv_market * 100,
        "iv_model": iv_model * 100,
        "iv_error_pts": iv_error_pts,
        "abs_iv_error_pts": abs(iv_error_pts),
        "sq_iv_error_pts": iv_error_pts**2,
    })

  df_val = pd.DataFrame(results)
  if df_val.empty:
    raise ValueError(
        "Aucune donnée valide n'a pu être évaluée pour la validation."
    )

  #Erreur Globale
  rmse_iv = np.sqrt(df_val["sq_iv_error_pts"].mean())
  mae_iv = df_val["abs_iv_error_pts"].mean()
  max_iv_error = df_val["abs_iv_error_pts"].max()

  if verbose: print("=" * 60)
  if verbose: print("RAPPORT D'ERREUR GLOBALE (en points de vol)")
  if verbose: print("=" * 60)
  if verbose: print(f"• RMSE de l'IV         : {rmse_iv:.4f} pts")
  if verbose: print(f"• MAE de l'IV          : {mae_iv:.4f} pts")
  if verbose: print(f"• Erreur maximale abs. : {max_iv_error:.4f} pts\n")

  # Erreur par Maturité
  # Découpage en tranches de maturité (ex: Court, Moyen, Long terme)
  df_val["maturity_bucket"] = pd.qcut(df_val["T"], q=3, labels=["Court terme", "Moyen terme", "Long terme"], duplicates="drop")

  mat_table = (
      df_val.groupby("maturity_bucket", observed=True)
      .agg(
          Nombre_de_points=("iv_error_pts", "count"),
          RMSE_IV=("sq_iv_error_pts", lambda x: np.sqrt(np.mean(x))),
          MAE_IV=("abs_iv_error_pts", "mean"),
      )
      .reset_index()
  )

  if verbose: print("=" * 60)
  if verbose: print("10.2 ERREUR PAR MATURITÉ")
  if verbose: print("=" * 60)
  if verbose: print(mat_table.to_string(index=False))
  if verbose: print("\n")

  #Erreur par Log-Moneyness
  # Découpage selon la position par rapport au strike (Ailes vs ATM)
  conditions = [
      df_val["log_moneyness"] < -0.05,
      (df_val["log_moneyness"] >= -0.05) & (df_val["log_moneyness"] <= 0.05),
      df_val["log_moneyness"] > 0.05,
  ]
  choices = ["Strikes bas (Puts OTM)", "Autour de l'ATM", "Strikes hauts (Calls OTM)"]
  df_val["moneyness_bucket"] = np.select(conditions, choices, default="Neutre")

  moneyness_table = (
      df_val.groupby("moneyness_bucket", observed=False)
      .agg(
          Nombre_de_points=("iv_error_pts", "count"),
          RMSE_IV=("sq_iv_error_pts", lambda x: np.sqrt(np.mean(x))),
          MAE_IV=("abs_iv_error_pts", "mean"),
      )
      .reset_index()
  )

  if verbose: print("=" * 60)
  if verbose: print("10.3 ERREUR PAR LOG-MONEYNESS (Skew et Ailes)")
  if verbose: print("=" * 60)
  if verbose: print(moneyness_table.to_string(index=False))
  if verbose: print("=" * 60)

  return df_val


# ----------------------------------------------------------------------
# Graphiques
# ----------------------------------------------------------------------
def plot_market_vs_heston_surfaces(df_val, save_path=None):
  fig = plt.figure(figsize=(16, 7))

  # 1. Surface de Marché
  ax1 = fig.add_subplot(121, projection="3d")
  img1 = ax1.scatter(
      df_val["log_moneyness"],
      df_val["T"],
      df_val["iv_market"],
      c=df_val["iv_market"],
      cmap="viridis",
      s=15,
      alpha=0.7,
  )
  ax1.set_title("Surface d'IV de Marché")
  ax1.set_xlabel("Log-Moneyness")
  ax1.set_ylabel("Maturité T")
  ax1.set_zlabel("IV (%)")

  # 2. Surface du Modèle Heston
  ax2 = fig.add_subplot(122, projection="3d")
  ax2.scatter(
      df_val["log_moneyness"],
      df_val["T"],
      df_val["iv_model"],
      c=df_val["iv_market"],
      cmap="viridis",
      vmin=df_val["iv_market"].min(),
      vmax=df_val["iv_market"].max(),
      s=15,
      alpha=0.7,
  )
  ax2.set_title("Surface d'IV du Modèle Heston")
  ax2.set_xlabel("Log-Moneyness")
  ax2.set_ylabel("Maturité T")
  ax2.set_zlabel("IV (%)")

  fig.colorbar(img1, ax=[ax1, ax2], label="IV (%)", shrink=0.6)
  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
  plt.show()
  

def plot_calibration_error_heatmap(df_val, save_path=None):
  plt.figure(figsize=(10, 6))

  # Erreur en points de pourcentage ou décimales
  error = df_val["iv_model"] - df_val["iv_market"]

  sc = plt.scatter(
      df_val["log_moneyness"],
      df_val["T"],
      c=error,
      cmap="coolwarm",
      s=20,
      vmin=-5,
      vmax=5,
  )
  plt.colorbar(sc, label="Erreur (IV Modèle - IV Marché en pts)")
  plt.axvline(0, color="black", linestyle="--", alpha=0.5, label="ATM")
  plt.title(
      "Cartographie des Erreurs de Calibration (Sur/Sous-estimation)",
      fontsize=13,
  )
  plt.xlabel("Log-Moneyness $\\ln(K/S)$")
  plt.ylabel("Maturité $T$ (années)")
  plt.legend()
  plt.grid(True, alpha=0.3)
  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
  plt.show()
  

def plot_smiling_comparison(df_val, n_maturities=3, save_path=None):
  # Sélectionner quelques maturités distinctes
  unique_ts = sorted(df_val["T"].unique())
  selected_ts = [
      unique_ts[int(i * (len(unique_ts) - 1) / (n_maturities - 1))]
      for i in range(n_maturities)
  ]

  fig, axes = plt.subplots(1, len(selected_ts), figsize=(16, 5), sharey=True)
  if len(selected_ts) == 1:
    axes = [axes]

  for ax, t_val in zip(axes, selected_ts):
    subset = df_val[df_val["T"] == t_val].sort_values("log_moneyness")

    # Points de marché
    ax.scatter(
        subset["log_moneyness"],
        subset["iv_market"],
        color="blue",
        label="Marché",
        s=25,
        alpha=0.7,
    )
    
    # Courbe Heston lissée (en regroupant par log_moneyness pour éviter les zigzags dus aux doublons Puts/Calls)
    subset_grouped = (
        subset.groupby("log_moneyness")["iv_model"].mean().reset_index()
    )
    
    
    
    # Courbe Heston
    ax.plot(
        subset_grouped["log_moneyness"],
        subset_grouped["iv_model"],
        color="red",
        linewidth=2,
        label="Heston",
    )

    ax.set_title(f"Maturité T $\\approx$ {t_val:.3f} ans")
    ax.set_xlabel("Log-Moneyness")
    ax.set_ylabel("IV (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
  plt.show()


# ======================================================================
# Démonstration / tests manuels (exécutés uniquement en `python calibration.py`)
# ======================================================================
if __name__ == "__main__":
    #%%
    #%% data test

    from market_data import get_option_chain
    from data_cleaning import clean_option_chain
    from implied_vol import add_implied_volatility

    r_market = 0.02
    q_market = 0.015
    cleaned_df = clean_option_chain(get_option_chain())[0]
    df_with_iv = add_implied_volatility(cleaned_df, r=r_market, q=q_market)

    #%% test


    # 1. On filtre pour stabiliser la calibration globale (c'est arbitraire)

    # Sélection d'un sous-ensemble multi-maturités propre et nettoyé des ailes extrêmes
    # Sélection d'un sous-ensemble propre, sans les maturités ultra-courtes ni les ailes extrêmes
    subset_market = df_with_iv[
        (df_with_iv["iv_status"] == "success")
        & (
            df_with_iv["T"].between(0.02, 0.5)
        )  # Entre ~1 semaine et 6 mois (exclut les 0-DTE instables)
        & (
            df_with_iv["strike"].between(
                0.85 * df_with_iv["spot"], 1.15 * df_with_iv["spot"]
            )
        )  # Strikes proches de la monnaie
    ].copy()


    # 2. Point de départ raisonnable (initial guess) : [v0, kappa, theta, xi, rho]
    initial_guess = [0.04, 2.0, 0.04, 0.3, -0.7]

    # 3. Bornes strictes test à ne pas dépasser (arbitraire)
    bounds = [
        (0.001, 1.0),  # v0
        (0.01, 10.0),  # kappa
        (0.001, 1.0),  # theta
        (0.01, 5.0),  # xi (vol-of-vol)
        (-0.99, 0.99),  # rho (corrélation)
    ]

    # 4. Lancement de la calibration

    calibration_result = calibrate_heston(
        subset_market, initial_guess, bounds, r=r_market, q=q_market
    )

    print("Résultat de la calibration :")
    print(f"Succès : {calibration_result['success']}")
    print(f"Fonction de perte finale (MSE IV) : {calibration_result['loss']:.6f}")
    print("Paramètres optimaux trouvés :")
    p = calibration_result["params"]
    print(
        f"  v0    = {p[0]:.4f}\n  kappa = {p[1]:.4f}\n  theta ="
        f" {p[2]:.4f}\n  xi    = {p[3]:.4f}\n  rho   = {p[4]:.4f}"
    )


    df_validation = validate_calibration(
        subset_market, calibration_result["params"], r=r_market, q=q_market
    )


    #plot_market_vs_heston_surfaces(df_validation)
    #plot_calibration_error_heatmap(df_validation)
    plot_smiling_comparison(df_validation)
