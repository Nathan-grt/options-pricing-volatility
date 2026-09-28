#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulation GBM et backtest de delta-hedging d'un call/put vendu (short option)."""

import numpy as np
import pandas as pd
from scipy.stats import norm
from black_scholes import d1
from black_scholes import black_scholes_price
from greeks import delta


    

def simulate_gbm_paths(spot, maturity, volatility, rate, dividend_yield, n_steps, n_paths, seed=None): 
    if seed is not None:
        np.random.seed(seed)
    
    r=rate
    q=dividend_yield
    sigma=volatility
    T=maturity
    delta_T=T/n_steps
    
    # Initialisation d'une matrice (n_steps + 1 x n_paths) pour stocker les prix
    S = np.zeros((n_steps + 1, n_paths))
    S[0, :] = spot
     
    
    Z = np.random.standard_normal((n_steps, n_paths))
    
    drift = (r - q - 0.5 * sigma**2) * delta_T
    diffusion_coeff = sigma * np.sqrt(delta_T)
    
    for i in range (1,n_steps+1):
    
        diffusion = diffusion_coeff * Z[i - 1]
        S[i, :] = S[i - 1, :] * np.exp(drift + diffusion)
    
    return S

    


def _rebalance_step(rebalance_frequency):
    """'daily' / None -> rebalancement à chaque pas ; entier k -> tous les k pas."""
    if rebalance_frequency is None or isinstance(rebalance_frequency, str):
        return 1
    return max(1, int(rebalance_frequency))


def delta_hedging(spot_path,option_type,strike,maturity,rate,volatility,rebalance_frequency,transaction_cost,q=0.0,):
    """Delta-hedge d'une option VENDUE (on encaisse la prime, on achète Δ actions).

    Modifications par rapport à la version initiale :
    - rebalance_frequency est désormais utilisé : 'daily' = chaque pas du chemin,
      entier k = rebalancement tous les k pas (le prix/delta sont suivis à chaque pas).
    - les actions détenues perçoivent le dividende continu q (sinon biais négatif
      systématique du P&L d'environ q * Δ * S * T quand q > 0).
    - transaction_costs sont DÉJÀ déduits du cash : le P&L final est net de frais.
    """
    
    r=rate
    K=strike
    T=maturity  
        
    n_steps = len(spot_path) - 1
    dt = T / n_steps
    k_reb = _rebalance_step(rebalance_frequency)
    
    spots = []
    time_to_mats = []
    option_prices = []
    deltas = []
    shares_held = []
    cash = []
    trans_costs = []
    portfolio_values = []
    pnl = []
    
    curr_cash = 0.0
    curr_shares = 0.0
    
    for i in range(len(spot_path)):
        S = spot_path[i]
        # dernier pas forcé à 0 (évite un tau = 1e-16 dû à l'arrondi flottant)
        tau = 0.0 if i == n_steps else max(0.0, T - i * dt)
    
        if tau == 0.0:
            if option_type.lower() == "call":
                opt_price = max(0.0, S - K)
            else:
                opt_price = max(0.0, K - S)
            # correction : le delta d'un put ITM à l'échéance vaut -1 (et non +1)
            if option_type.lower() == "call":
                opt_delta = 1.0 if S > K else 0.0
            else:
                opt_delta = -1.0 if S < K else 0.0
        else:
            opt_price = black_scholes_price(S, K, tau, r, volatility, option_type, q)
            opt_delta = delta(S, K, tau, r, volatility, option_type, q)
    
    
        if i == 0:
            
            trade_shares = opt_delta
            tc = transaction_cost * abs(trade_shares * S)
            curr_shares = trade_shares
            curr_cash = opt_price - (curr_shares * S) - tc
        else:
            # intérêts sur le cash + dividendes perçus sur les actions détenues
            curr_cash = curr_cash * np.exp(r * dt) + curr_shares * spot_path[i - 1] * (np.exp(q * dt) - 1.0)
            if i % k_reb == 0 or tau == 0.0:
                trade_shares = opt_delta - curr_shares
            else:
                trade_shares = 0.0
            tc = transaction_cost * abs(trade_shares * S)
            curr_cash -= (trade_shares * S) + tc
            curr_shares = curr_shares + trade_shares
    
        port_val = curr_cash + (curr_shares * S) - opt_price

        cumulative_pnl = port_val
    
        spots.append(S)
        time_to_mats.append(tau)
        option_prices.append(opt_price)
        deltas.append(opt_delta)
        shares_held.append(curr_shares)
        cash.append(curr_cash)
        trans_costs.append(tc)
        portfolio_values.append(port_val)
        pnl.append(cumulative_pnl)

    result_df = pd.DataFrame({
        "spot": spots,
        "time_to_maturity": time_to_mats,
        "option_price": option_prices,
        "delta": deltas,
        "stock_shares": shares_held,
        "cash": cash,
        "transaction_costs": trans_costs,
        "portfolio_value": portfolio_values,
        "cumulative_pnl": pnl,
        })

    return result_df


# ---------------------------------------------------------------------------
# AJOUT : version vectorisée (toutes les trajectoires en même temps)
# Même logique comptable que delta_hedging (vérifié dans tests/test_hedging.py),
# ~100x plus rapide : indispensable pour les études de fréquence de
# rebalancement sur des grilles fines (ex. 252 x 13 pas x 5 000 chemins).
# ---------------------------------------------------------------------------
def _bs_price_delta_vec(S, K, tau, r, sigma, option_type, q):
    """Prix et delta BS vectorisés (tau > 0)."""
    sqrt_t = np.sqrt(tau)
    dd1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_t)
    dd2 = dd1 - sigma * sqrt_t
    disc_q, disc_r = np.exp(-q * tau), np.exp(-r * tau)
    if option_type == "call":
        price = S * disc_q * norm.cdf(dd1) - K * disc_r * norm.cdf(dd2)
        dlt = disc_q * norm.cdf(dd1)
    else:
        price = K * disc_r * norm.cdf(-dd2) - S * disc_q * norm.cdf(-dd1)
        dlt = disc_q * (norm.cdf(dd1) - 1.0)
    return price, dlt


def delta_hedging_vectorized(paths, option_type, strike, maturity, rate, volatility,
                             rebalance_frequency=1, transaction_cost=0.0, q=0.0,
                             return_paths=False):
    """Delta-hedge d'une option vendue sur une matrice de trajectoires (n_steps+1, n_paths).

    Retourne un dict : 'pnl' (P&L final net de frais), 'tc' (frais totaux),
    'premium' (prime encaissée), et si return_paths : 'pnl_path' (n_steps+1, n_paths).
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim == 1:
        paths = paths[:, None]
    n_steps = paths.shape[0] - 1
    dt = maturity / n_steps
    k_reb = _rebalance_step(rebalance_frequency)
    K, r = strike, rate

    S0 = paths[0]
    price0, delta0 = _bs_price_delta_vec(S0, K, maturity, r, volatility, option_type, q)
    shares = delta0.copy()
    tc_total = transaction_cost * np.abs(shares * S0)
    cash = price0 - shares * S0 - tc_total
    pnl_path = np.zeros_like(paths) if return_paths else None
    if return_paths:
        pnl_path[0] = cash + shares * S0 - price0

    for i in range(1, n_steps + 1):
        S = paths[i]
        tau = max(0.0, maturity - i * dt)
        cash = cash * np.exp(r * dt) + shares * paths[i - 1] * (np.exp(q * dt) - 1.0)
        if i == n_steps or tau <= 1e-12:
            if option_type == "call":
                opt, dlt = np.maximum(S - K, 0.0), np.where(S > K, 1.0, 0.0)
            else:
                opt, dlt = np.maximum(K - S, 0.0), np.where(S < K, -1.0, 0.0)
        else:
            opt, dlt = _bs_price_delta_vec(S, K, tau, r, volatility, option_type, q)
        if i % k_reb == 0 or i == n_steps:
            trade = dlt - shares
            tc = transaction_cost * np.abs(trade * S)
            cash -= trade * S + tc
            tc_total += tc
            shares = shares + trade
        if return_paths:
            pnl_path[i] = cash + shares * S - opt

    pnl = cash + shares * paths[-1] - opt
    out = {"pnl": pnl, "tc": tc_total, "premium": price0}
    if return_paths:
        out["pnl_path"] = pnl_path
    return out


# ======================================================================
# Démonstration / tests manuels (exécutés uniquement en `python hedging.py`)
# ======================================================================
if __name__ == "__main__":
    #%% Test 1 path
    paths=simulate_gbm_paths(100, 1, 0.2, 0.02, 0.015, 30, 1000, seed=42)
    single_path = paths[:, 0]  # Première trajectoire (colonne 0)

    df_hedge = delta_hedging(single_path,"call",100.0,1.0,0.02,0.20,"daily",0.001,0.015)

    print(df_hedge)

    #%% Test 1000 paths
    paths=simulate_gbm_paths(100, 1, 0.2, 0.02, 0.015, 252, 1000, seed=42)
    final_pnl_list = []
    final_pnl_net_list = []
    total_costs_list = []

    for j in range(paths.shape[1]):  
      single_path = paths[:, j]

      df_res = delta_hedging(
          spot_path=single_path,
          option_type="call",
          strike=100.0,
          maturity=1.0,
          rate=0.02,
          volatility=0.20,
          rebalance_frequency="daily",
          transaction_cost=0.00,
          q=0.015,
      )

      # Récupération des valeurs finales à la dernière ligne (T)
      final_pnl = df_res["cumulative_pnl"].iloc[-1]
      total_tc = df_res["transaction_costs"].sum()

      final_pnl_list.append(final_pnl)
      final_pnl_net_list.append(final_pnl - total_tc)  # P&L net des coûts
      total_costs_list.append(total_tc)

    # Conversion en tableaux numpy pour les statistiques
    pnl_net = np.array(final_pnl_net_list)

    # Affichage des métriques demandées
    print(f"P&L moyen sur les trajectoires : {np.mean(pnl_net):.4f}")
    print(f"Écart-type du P&L : {np.std(pnl_net):.4f}")
    print(
        f"Quantiles (5%, 50%, 95%) : {np.percentile(pnl_net, [5, 50, 95])}"
    )
    print(
        f"Proportion de trajectoires avec P&L négatif :"
        f" {np.mean(pnl_net < 0) * 100:.2f}%"
    )
