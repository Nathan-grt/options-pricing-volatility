#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from black_scholes import black_scholes_price


def simulate_terminal_prices(S, T, r, sigma, n_paths, q=0.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    z = np.random.standard_normal(n_paths)
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z

    return S * np.exp(drift + diffusion)


def price_monte_carlo(S, K, T, r, sigma, n_paths, option_type="call", q=0.0, seed=None):
    ST = simulate_terminal_prices(S, T, r, sigma, n_paths, q, seed)

    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")

    prices = np.exp(-r * T) * payoff
    mc_price = np.mean(prices)

    # Écart-type d'échantillon sans biais et erreur standard
    sample_std = np.std(prices, ddof=1)
    se = sample_std / np.sqrt(n_paths)

    # Intervalle de confiance à 95 % (z = 1.96)
    ci_low = mc_price - 1.96 * se
    ci_up = mc_price + 1.96 * se

    # Benchmark Black-Scholes avec dividende continu q
    bs_price = black_scholes_price(S, K, T, r, sigma, option_type, q=q)
    abs_error = abs(mc_price - bs_price)

    return {
        "mc_price": mc_price,
        "se": se,
        "ci_95": (ci_low, ci_up),
        "bs_price": bs_price,
        "abs_error": abs_error,
        "in_ci": ci_low <= bs_price <= ci_up,
    }


# ======================================================================
# Démonstration / tests manuels (exécutés uniquement en `python monte_carlo.py`)
# ======================================================================
if __name__ == "__main__":
    #%% Convergence vers Black scholes:


    S = 100
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20
    option_type = "call"

    paths_list = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]

    mc_prices = []
    ci_low = []
    ci_up = []
    errors = []

    # Prix théorique exact Black-Scholes
    bs_ref = black_scholes_price(S, K, T, r, sigma, option_type)

    # 1. Affichage du tableau formaté
    print(f"{'N paths':<12} {'MC price':<12} {'Error':<12}")
    print("-" * 36)

    for n in paths_list:
        res = price_monte_carlo(S, K, T, r, sigma, n, option_type=option_type, seed=42)

        mc_prices.append(res["mc_price"])
        ci_low.append(res["ci_95"][0])
        ci_up.append(res["ci_95"][1])
        errors.append(res["abs_error"])

        print(f"{n:<12,d} {res['mc_price']:<12.4f} {res['abs_error']:<12.4f}")      #place pour 12 charactères, 4 chiffres décimaux

    plt.figure(figsize=(10, 6))

    # Ligne théorique Black-Scholes
    plt.axhline(
        y=bs_ref,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Benchmark Black-Scholes ({bs_ref:.4f})",
    )

    # Prix Monte Carlo
    plt.plot(
        paths_list,
        mc_prices,
        marker="o",
        color="navy",
        label="Prix Monte Carlo",
        zorder=3,           #hauteur du "calque"
    )

    # Intervalle de confiance à 95 %
    plt.fill_between(
        paths_list,
        ci_low,
        ci_up,
        color="cornflowerblue",
        alpha=0.3,                  #transparence
        label="Intervalle de confiance à 95 %",
    )

    plt.xscale("log")  # Échelle logarithmiqu
    plt.title(
        f"Convergence du prix Monte Carlo vers Black-Scholes ({option_type.capitalize()})"
    )
    plt.xlabel("Nombre de simulations (N) [échelle log]")
    plt.ylabel("Prix de l'option")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)

    # Texte des paramètres formaté
    params_text = (
        f"Paramètres :\n"
        f"S = {S}\n"
        f"K = {K}\n"
        f"T = {T} an\n"
        f"r = {r * 100:.1f} %\n"
        f"$\\sigma$ = {sigma * 100:.1f} %"
    )

    # Ajout de l'encadré sur le graphique
    plt.text(
        0.03,
        0.95,
        params_text,  # Coordonnées (x, y) relatives : 3 % du bord gauche, 95 % du bas
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray",
            alpha=0.8,
        ),
    )

    plt.legend()
    plt.tight_layout()
    plt.show()
