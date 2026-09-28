#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pricing semi-analytique d'options européennes sous Heston (1993).

Formulation "little Heston trap" (Albrecher et al., 2007) de la fonction
caractéristique, numériquement stable pour les maturités longues.

Modification par rapport à la version initiale : ajout du rendement de
dividende continu ``q`` (défaut 0.0 -> résultats identiques à l'ancienne version).
Sans q, le prix Heston était calculé sur un forward S*e^{rT} alors que l'IV était
ensuite inversée avec q = 1.5 % : cela créait un biais artificiel (IV calls
surestimées, IV puts sous-estimées) dans la calibration.
"""

import numpy as np
from scipy.integrate import quad


def _heston_char_func(u, S0, v0, r, kappa, theta, xi, rho, T, q=0.0):
    """Fonction caractéristique de ln(S_T) sous Heston (mesure risque-neutre)."""
    i = 1j
    # Paramètres intermédiaires de Heston
    alpha = -u * u / 2.0 - i * u / 2.0
    beta = kappa - rho * xi * i * u
    gamma = xi**2 / 2.0

    # Discriminant
    d = np.sqrt(beta**2 - 4.0 * alpha * gamma)

    # Termes exponentiels et trigonométriques
    g = (beta - d) / (beta + d)
    exp_dt = np.exp(-d * T)

    C = ((r - q) * i * u * T) + (
        kappa
        * theta
        / (xi**2)
        * ((beta - d) * T - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g)))
    )
    D = (beta - d) / (xi**2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))

    return np.exp(C + D * v0 + i * u * np.log(S0))


def heston_call_price(S0, K, T, r, v0, kappa, theta, xi, rho, q=0.0):
    """Calcule le prix d'un Call européen sous le modèle de Heston

    par intégration semi-analytique de la fonction caractéristique.
    """
    # 1. Gestion des paramètres invalides
    if S0 <= 0 or K <= 0 or T <= 0 or v0 < 0:
        return np.nan
    if kappa <= 0 or theta <= 0 or xi <= 0 or not (-1.0 <= rho <= 1.0):
        return np.nan

    # Si l'option est déjà échue
    if T == 0:
        return max(S0 - K, 0.0)

    forward = S0 * np.exp((r - q) * T)

    # Intégrande pour P1
    def integrand_p1(u):
        val = np.exp(-1j * u * np.log(K)) * _heston_char_func(
            u - 1j, S0, v0, r, kappa, theta, xi, rho, T, q
        ) / (1j * u * forward)
        return val.real

    # Intégrande pour P2
    def integrand_p2(u):
        val = np.exp(-1j * u * np.log(K)) * _heston_char_func(
            u, S0, v0, r, kappa, theta, xi, rho, T, q
        ) / (1j * u)
        return val.real

    try:
        # Intégration numérique de 0 à l'infini, tronquée à une borne ADAPTATIVE.
        # Correction : la borne fixe de 100 était insuffisante aux maturités courtes
        # (|phi(100)| ~ 0.03 pour T = 0.05 an avec les paramètres calibrés ->
        # oscillations de plusieurs points d'IV dans les ailes).
        # Pour u grand, |phi(u)| ~ exp(-a u) avec a = (v0 + kappa*theta*T) sqrt(1-rho^2) / xi
        # (asymptotique de la fonction caractéristique) : on intègre jusqu'à |phi| ~ e^-25.
        decay = (v0 + kappa * theta * T) * np.sqrt(max(1.0 - rho**2, 1e-4)) / xi
        limit = float(np.clip(25.0 / max(decay, 1e-12), 100.0, 2000.0))
        int_p1, _ = quad(integrand_p1, 0, limit, limit=400)
        int_p2, _ = quad(integrand_p2, 0, limit, limit=400)

        P1 = 0.5 + (1.0 / np.pi) * int_p1
        P2 = 0.5 + (1.0 / np.pi) * int_p2

        # Prix du Call Heston
        price = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
        return max(float(price), 0.0)
    except Exception:
        return np.nan


def heston_put_price(S0, K, T, r, v0, kappa, theta, xi, rho, q=0.0):
    """Calcule le prix d'un Put européen via la parité Call-Put sous Heston.

    P = C - S0 * exp(-qT) + K * exp(-rT)
    """
    call_p = heston_call_price(S0, K, T, r, v0, kappa, theta, xi, rho, q)
    if np.isnan(call_p):
        return np.nan
    put_p = call_p - S0 * np.exp(-q * T) + K * np.exp(-r * T)
    return max(float(put_p), 0.0)
