#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm 
from black_scholes import d1
from black_scholes import d2
from black_scholes import black_scholes_price

    
def delta(S, K, T, r, sigma, option_type="call", q=0.0):
    val_d1 = d1(S, K, T, r, sigma, q)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(val_d1)
    else:
        return np.exp(-q * T) * (norm.cdf(val_d1) - 1.0)

def gamma(S, K, T, r, sigma, option_type="call", q=0.0):
    val_d1 = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(val_d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, option_type="call", q=0.0):
    val_d1 = d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(val_d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, option_type="call", q=0.0):
    val_d1 = d1(S, K, T, r, sigma, q)
    val_d2 = d2(S, K, T, r, sigma, q)

    term1 = -(S * np.exp(-q * T) * norm.pdf(val_d1) * sigma) / (2 * np.sqrt(T))

    if option_type == "call":
        return (
            term1
            + q * S * np.exp(-q * T) * norm.cdf(val_d1)
            - r * K * np.exp(-r * T) * norm.cdf(val_d2)
        )
    else:
        return (
            term1
            - q * S * np.exp(-q * T) * norm.cdf(-val_d1)
            + r * K * np.exp(-r * T) * norm.cdf(-val_d2)
        )


def rho(S, K, T, r, sigma, option_type="call", q=0.0):
    val_d2 = d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(val_d2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-val_d2)


def all_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    return {
        "delta": delta(S, K, T, r, sigma, option_type, q),
        "gamma": gamma(S, K, T, r, sigma, option_type, q),
        "vega": vega(S, K, T, r, sigma, option_type, q),
        "theta": theta(S, K, T, r, sigma, option_type, q),
        "rho": rho(S, K, T, r, sigma, option_type, q),
    }


# ======================================================================
# Démonstration / tests manuels (exécutés uniquement en `python greeks.py`)
# ======================================================================
if __name__ == "__main__":
    #%% Graphes

    #Price vs spot


    K = 100
    T = 1
    r = 0.05
    sigma = 0.20
    Smin = 0
    Smax = 200
    n = 100
    S = np.linspace(Smin, Smax, n + 1)

    Call_price = [black_scholes_price(s, K, T, r, sigma, option_type="call") for s in S]
    Put_price = [black_scholes_price(s, K, T, r, sigma, option_type="put") for s in S]

    # Tracé du graphique
    plt.figure(figsize=(9, 5))
    plt.plot(S, Call_price, label="Call", color="navy", lw=2)
    plt.plot(S, Put_price, label="Put", color="crimson", lw=2)
    plt.axvline(
        x=K,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Prix d'exercice (K={K})",
    )

    plt.title("Prix des options en fonction du sous-jacent (Black-Scholes)")
    plt.xlabel("Prix spot (S)")
    plt.ylabel("Prix de l'option")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()



    #Delta vs spot
    Delta_call=[delta(s, K, T, r, sigma, option_type="call") for s in S]
    Delta_put=[delta(s, K, T, r, sigma, option_type="put") for s in S]

    plt.figure(figsize=(9, 5))
    plt.plot(S, Delta_call, label="Call", color="navy", lw=2)
    plt.plot(S, Delta_put, label="Put", color="crimson", lw=2)
    plt.axvline(
        x=K,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Prix d'exercice (K={K})",
    )

    plt.title("Valeur du Delta en fonction du sous-jacent (Black-Scholes)")
    plt.xlabel("Prix spot (S)")
    plt.ylabel("Delta")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


    #Gamma vs spot
    Gamma=[gamma(s, K, T, r, sigma, option_type="call") for s in S]

    plt.figure(figsize=(9, 5))
    plt.plot(S, Gamma, label="Gamma (Gamma put = Gamma call", color="navy", lw=2)
    plt.axvline(
        x=K,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Prix d'exercice (K={K})",
    )

    plt.title("Valeur du Gamma en fonction du sous-jacent (Black-Scholes)")
    plt.xlabel("Prix spot (S)")
    plt.ylabel("Gamma")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


    #Vega vs spot
    Vega=[vega(s, K, T, r, sigma, option_type="call") for s in S]

    plt.figure(figsize=(9, 5))
    plt.plot(S, Vega, label="Vega (Vega put = Vega call", color="navy", lw=2)
    plt.axvline(
        x=K,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Prix d'exercice (K={K})",
    )

    plt.title("Valeur du vega en fonction du sous-jacent (Black-Scholes)")
    plt.xlabel("Prix spot (S)")
    plt.ylabel("Vega")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()





