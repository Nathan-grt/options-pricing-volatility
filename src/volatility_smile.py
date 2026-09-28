#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from market_data import get_option_chain
from data_cleaning import clean_option_chain
from implied_vol import add_implied_volatility


def plot_volatility_smile(df: pd.DataFrame, expiration: str, use_moneyness: bool = False, save_path=None):
    """Affiche le Volatility Smile pour une date d'expiration donnée.

    Paramètres :
    - df : DataFrame contenant 'expiration', 'iv_status', 'strike', 'spot',
      'implied_vol', 'option_type'
    - expiration : Date d'expiration cible (format string ou datetime)
    - use_moneyness : Si True, affiche ln(K/S) en abscisse ; sinon le strike K
    """
    df_temp = df.copy()
    df_temp["expiration_str"] = pd.to_datetime(
        df_temp["expiration"]
    ).dt.strftime("%Y-%m-%d")
    target_exp_str = pd.to_datetime(expiration).strftime("%Y-%m-%d")

    # Filtrage sécurisé
    sub_df = df_temp[
        (df_temp["expiration_str"] == target_exp_str)
        & (df_temp["iv_status"] == "success")
    ].copy()

    if sub_df.empty:
        print(f"Aucune donnée d'IV valide pour l'expiration {expiration}.")
        return

    # 2. Calcul du log-moneyness : ln(K / S)
    sub_df["log_moneyness"] = np.log(sub_df["strike"] / sub_df["spot"])

    # Choix de la variable en abscisse
    x_col = "log_moneyness" if use_moneyness else "strike"
    x_label = r"Log-Moneyness $\ln(K/S)$" if use_moneyness else "Strike ($)"

    # 3. Séparation Calls et Puts pour observer d'éventuels écarts
    calls = sub_df[sub_df["option_type"].str.lower() == "call"].sort_values(
        by=x_col
    )
    puts = sub_df[sub_df["option_type"].str.lower() == "put"].sort_values(
        by=x_col
    )

    # 4. Tracé graphique
    plt.figure(figsize=(10, 6))

    # Points réels (scatter plot sans lissage forcé)
    if not calls.empty:
        plt.plot(
            calls[x_col],
            calls["implied_vol"] * 100,
            "o-",
            label="Calls",
            alpha=0.7,
            markersize=5,
        )

    if not puts.empty:
        plt.plot(
            puts[x_col],
            puts["implied_vol"] * 100,
            "s--",
            label="Puts",
            alpha=0.7,
            markersize=5,
        )

    # Ligne repère pour l'At-The-Money (ATM)
    spot_val = sub_df["spot"].iloc[0]
    atm_x = 0.0 if use_moneyness else spot_val
    plt.axvline(
        x=atm_x,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label=f"ATM (Spot ≈ {spot_val:.2f})",
    )

    plt.title(
        f"Volatility Smile — Expiration {expiration} (T ≈ {sub_df['T'].iloc[0]:.2f} ans)",
        fontsize=13,
    )
    plt.xlabel(x_label, fontsize=11)
    plt.ylabel("Implied Volatility (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ======================================================================
# Démonstration / tests manuels (exécutés uniquement en `python volatility_smile.py`)
# ======================================================================
if __name__ == "__main__":
    #%% Tests

    r_market = 0.02
    q_market = 0.015
    cleaned_df = clean_option_chain(get_option_chain())[0]
    df_with_iv = add_implied_volatility(cleaned_df, r=r_market, q=q_market)


    # Sélectionne la première échéance disponible dans ton DataFrame
    selected_exp = cleaned_df["expiration"].iloc[0]

    # Affichage avec l'axe Strike standard
    plot_volatility_smile(df_with_iv, expiration=selected_exp, use_moneyness=True)
