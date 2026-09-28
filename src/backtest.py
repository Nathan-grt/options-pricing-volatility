#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from black_scholes import black_scholes_price
from greeks import delta



def historical_backtest(
    spot_history,
    iv_history,
    dates_history,
    strike,
    maturity_initial,
    rate,
    option_type="call",
    transaction_cost=0.0,
    q=0.0,
    tau_history=None,
):
  """Effectue un backtest de couverture delta sur une trajectoire de prix historique réelle.

  - spot_history : tableau ou liste des prix réels de l'actif
  - dates_history : tableau ou liste des dates associées
  - maturity_initial : maturité totale initiale de l'option en années (ex: 1.0)
  - tau_history (AJOUT, optionnel) : maturité résiduelle réelle à chaque date
    (colonne T des données). Si fourni, remplace l'approximation
    tau = maturity_initial - i/252 qui dérive dès qu'il y a des jours fériés
    (T est en jours calendaires/365 alors que i compte des jours de bourse).
    Le pas d'actualisation du cash devient alors l'écart réel entre deux dates.
  """
  n_steps = len(spot_history)
  dt = 1.0 / 252.0  # Pas de temps journalier standard
  iv_history = pd.Series(iv_history, dtype=float).ffill().bfill().values  # IV manquantes

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

  for i in range(n_steps):
    S = spot_history[i]
    vol_du_jour = iv_history[i]
    # Le temps restant diminue à chaque pas historique
    if tau_history is not None:
      tau = max(0.0, float(tau_history[i]))
      dt_i = dt if i == 0 else max(0.0, float(tau_history[i - 1]) - tau)
    else:
      tau = max(0.0, maturity_initial - i * dt)
      dt_i = dt

    # 1. Calcul du prix et du delta de l'option à cette date historique
    if tau == 0.0:
      if option_type.lower() == "call":
        opt_price = max(0.0, S - strike)
      else:
        opt_price = max(0.0, strike - S)
      # correction : delta d'un put ITM à l'échéance = -1
      if option_type.lower() == "call":
        opt_delta = 1.0 if S > strike else 0.0
      else:
        opt_delta = -1.0 if S < strike else 0.0
    else:
      # Tu peux utiliser Black-Scholes (ou remplacer par heston_call_price / heston_delta si tu souhaites faire du backtest Heston !)
      opt_price = black_scholes_price(
          S, strike, tau, rate, vol_du_jour, option_type, q
      )
      opt_delta = delta(S, strike, tau, rate, vol_du_jour, option_type, q)

    # 2. Gestion de la trésorerie et des transactions
    if i == 0:
      trade_shares = opt_delta
      tc = transaction_cost * abs(trade_shares * S)
      curr_shares = trade_shares
      curr_cash = opt_price - (curr_shares * S) - tc
    else:
      # Actualisation du cash au taux sans risque + dividendes sur les actions détenues
      curr_cash = curr_cash * np.exp(rate * dt_i) + curr_shares * spot_history[i - 1] * (np.exp(q * dt_i) - 1.0)
      trade_shares = opt_delta - curr_shares
      tc = transaction_cost * abs(trade_shares * S)
      curr_cash -= (trade_shares * S) + tc
      curr_shares = opt_delta

    # 3. Valeur du portefeuille de couverture et P&L cumulé
    port_val = curr_cash + (curr_shares * S) - opt_price

    spots.append(S)
    time_to_mats.append(tau)
    option_prices.append(opt_price)
    deltas.append(opt_delta)
    shares_held.append(curr_shares)
    cash.append(curr_cash)
    trans_costs.append(tc)
    portfolio_values.append(port_val)
    pnl.append(port_val)

  # Création du DataFrame récapitulatif du backtest
  backtest_df = pd.DataFrame({
      "date": dates_history,
      "spot": spots,
      "time_to_maturity": time_to_mats,
      "vol_implicite": iv_history,
      "option_price": option_prices,
      "delta": deltas,
      "stock_shares": shares_held,
      "cash": cash,
      "transaction_costs": trans_costs,
      "portfolio_value": portfolio_values,
      "cumulative_pnl": pnl,
  })

  return backtest_df


# ======================================================================
# Démonstration / tests manuels (exécutés uniquement en `python backtest.py`)
# ======================================================================
if __name__ == "__main__":
    #%%
    from market_data import get_option_chain
    from data_cleaning import clean_option_chain

    # 1. Toutes les dates de cotation sont nécessaires pour un backtest historique
    #    (sans quote_date="all", get_option_chain ne renvoie qu'UNE date -> 1 seul pas)
    df_brut = get_option_chain(ticker="SPY", quote_date="all")
    df_clean, control_table = clean_option_chain(df_brut, max_relative_spread=0.5)
    df_calls = df_clean[df_clean["option_type"] == "call"]

    # 2. Contrat : call ATM émis le 1er jour de 2022, échéance 30/12/2022
    first_day = df_calls["quote_date"].min()
    exp_target = pd.Timestamp("2022-12-30")
    day0 = df_calls[(df_calls["quote_date"] == first_day) & (df_calls["expiration"] == exp_target)]
    strike_atm = day0.loc[(day0["strike"] - day0["spot"]).abs().idxmin(), "strike"]
    print(f"Contrat sélectionné -> Expiration : {exp_target.date()} | Strike : {strike_atm}")

    df_contrat = df_calls[(df_calls["expiration"] == exp_target) & (df_calls["strike"] == strike_atm)].sort_values("quote_date")

    # 3. Lancement du backtest historique
    df_resultats = historical_backtest(
        spot_history=df_contrat["spot"].values,
        iv_history=df_contrat["source_iv"].values,
        dates_history=df_contrat["quote_date"].values,
        strike=strike_atm,
        maturity_initial=df_contrat["T"].iloc[0],
        rate=0.02,
        option_type="call",
        transaction_cost=0.001,  # 10 bps de frais
        q=0.015,
        tau_history=df_contrat["T"].values,
    )

    print(df_resultats.tail())
    pnl_final = df_resultats["cumulative_pnl"].iloc[-1]
    print(f"\nP&L final de la stratégie sur données réelles : {pnl_final:.4f}")
