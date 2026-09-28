#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pricing analytique Black-Scholes-Merton (avec dividende continu q)."""

import numpy as np
from scipy.stats import norm


def d1(S, K, T, r, sigma, q=0.0):
    numerateur = np.log(S / K) + (r - q + 1 / 2 * sigma**2) * T
    denominateur = sigma * np.sqrt(T)
    return numerateur / denominateur


def d2(S, K, T, r, sigma, q=0.0):
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def black_scholes_price(S, K, T, r, sigma, option_type="call", q=0.0):
    if option_type not in ("call", "put"):
        raise ValueError("Option_type doit être 'call' ou 'put'")
    if S < 0 or K <= 0 or sigma < 0:
        raise ValueError("S, K et sigma doivent être positifs")

    if T <= 0:
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    if S == 0:
        if option_type == "call":
            return 0.0
        else:
            return K * np.exp(-r * T)

    if sigma == 0:
        discounted_S = S * np.exp(-q * T)
        discounted_K = K * np.exp(-r * T)
        if option_type == "call":
            return max(discounted_S - discounted_K, 0.0)
        else:
            return max(discounted_K - discounted_S, 0.0)

    val_d1 = d1(S, K, T, r, sigma, q)
    val_d2 = d2(S, K, T, r, sigma, q)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(val_d1) - K * np.exp(
            -r * T
        ) * norm.cdf(val_d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-val_d2) - S * np.exp(
            -q * T
        ) * norm.cdf(-val_d1)
