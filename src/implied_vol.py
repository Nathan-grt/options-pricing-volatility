#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from scipy.optimize import brentq
from black_scholes import black_scholes_price
from greeks import vega
import numpy as np
import pandas as pd
from market_data import get_option_chain
from data_cleaning import clean_option_chain


def check_arbitrage_bounds(market_price, S,K,T,r,option_type='call',q= 0.0,) -> bool:
    """Vérifie si le prix de marché respecte les bornes sans arbitrage."""
    discount_k = K * np.exp(-r * T)
    discount_s = S * np.exp(-q * T)

    if option_type == "call":
        lower_bound = max(discount_s - discount_k, 0.0)
        upper_bound = discount_s
    elif option_type == "put":
        lower_bound = max(discount_k - discount_s, 0.0)
        upper_bound = discount_k
    else:
        return False

    # Le prix doit être strictement à l'intérieur des bornes théoriques
    # Une tolérance epsilon évite les instabilités quand prix ~= valeur intrinsèque
    eps = 1e-4
    return (lower_bound + eps) < market_price < (upper_bound - eps)



def implied_volatility(market_price,S,K,T,r,option_type="call",q= 0.0,a=1e-4,b=3.0,return_status=False):
    # 1. Vérification des entrées de base
    if pd.isna(T) or T <= 0:
        res = (np.nan, "invalid_maturity")
        return res if return_status else res[0]
    
    if (pd.isna(market_price) or pd.isna(S) or pd.isna(K) or market_price <= 0 or S <= 0 or K <= 0):
        res = (np.nan, "invalid_price")
        return res if return_status else res[0]
    
    if not check_arbitrage_bounds(market_price, S, K, T, r, option_type, q):
        res = (np.nan, "arbitrage_bounds")
        return res if return_status else res[0]

    # Fonction d'écart : f(sigma) = BS(sigma) - P_market = 0
    def objective(sigma):
        return (black_scholes_price(S, K, T, r, sigma, option_type, q)- market_price)

    
    try:
        f_a = objective(a)
        f_b = objective(b)
        if f_a * f_b > 0:
            res = (np.nan, "no_bracket")
            # Même signe : pas de racine dans [1e-4, 3.0]
        else:
            iv = brentq(objective, a, b, xtol=1e-6, maxiter=100)
            res = (float(iv), "success")
    except (ValueError, RuntimeError):
        res = (np.nan, "solver_failed")
        
    return res if return_status else res[0]
    


# Solveur alternatif : Newton-Raphson
def implied_volatility_newton(market_price,S,K,T,r,option_type = "call",q = 0.0,sigma_init = 0.20,max_iter = 100,tol = 1e-6,) :
    """Implémente Newton-Raphson : sigma_{n+1} = sigma_n - (BS - P_mkt) / Vega.

    Retourne (iv, nb_iterations, succès).
    """
    if (T <= 0 or S <= 0 or K <= 0 or market_price <= 0 or np.isnan(market_price) ):
        return np.nan, 0, False

    if not check_arbitrage_bounds(market_price, S, K, T, r, option_type, q):
        return np.nan, 0, False

    sigma = sigma_init
    for i in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type, q)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma, i + 1, True

        Vega = vega(S, K, T, r, sigma, option_type, q)

        # Si le Vega est trop proche de 0 (options très OTM ou ITM), le pas diverge
        if Vega < 1e-8:
            return np.nan, i + 1, False

        sigma = sigma - diff / Vega

        # Sécurité : sigma ne doit pas devenir négatif ou absurde
        if sigma <= 0 or sigma > 5.0:
            return np.nan, i + 1, False
    return np.nan, max_iter, False




def add_implied_volatility(df, r, q= 0.0):
    """Ajoute les colonnes 'implied_vol' et 'iv_status'.

    q : soit un float (dividende continu commun), soit le NOM d'une colonne du
        DataFrame (ex. "q_implied" produit par add_implied_dividend) -> q par ligne.
    """
    df_out = df.copy()

    def process_row(row):
        q_row = row[q] if isinstance(q, str) else q
        return implied_volatility(market_price=row["mid"],S=row["spot"],K=row["strike"],T=row["T"],r=r,option_type=row["option_type"],q=q_row,return_status=True)

    results = df_out.apply(process_row, axis=1)

    df_out["implied_vol"] = [res[0] for res in results]
    df_out["iv_status"] = [res[1] for res in results]

    return df_out


# ---------------------------------------------------------------------------
# AJOUT : forward / dividende implicite par parité call-put
# ---------------------------------------------------------------------------
def estimate_implied_dividend(df, r, n_strikes=6, q_bounds=(-0.10, 0.10)):
    """Estime, pour chaque couple (quote_date, expiration), le forward implicite

    via la parité call-put  C - P = e^{-rT} (F - K)  =>  F = K + e^{rT} (C - P),
    en prenant la médiane sur les n_strikes les plus proches du spot, puis le
    rendement de dividende implicite q = r - ln(F/S) / T.

    Intérêt : un q fixe (1.5 %) ignore que SPY verse des dividendes DISCRETS
    (trimestriels) et que r a fortement varié en 2022 ; un forward mal spécifié
    décale systématiquement l'IV des calls vers le haut et celle des puts vers
    le bas (ou inversement). Le forward implicite rend les deux cohérentes.
    """
    keys = ["quote_date", "expiration"]
    cols = keys + ["strike", "mid", "spot", "T"]
    calls = df[df["option_type"] == "call"][cols]
    puts = df[df["option_type"] == "put"][keys + ["strike", "mid"]]
    m = calls.merge(puts, on=keys + ["strike"], suffixes=("_c", "_p"))
    m = m[m["T"] > 0]
    m["dist"] = (m["strike"] - m["spot"]).abs()
    m["F"] = m["strike"] + np.exp(r * m["T"]) * (m["mid_c"] - m["mid_p"])
    m = m.sort_values("dist").groupby(keys).head(n_strikes)
    out = m.groupby(keys).agg(forward=("F", "median"), spot=("spot", "first"), T=("T", "first"))
    out["q_implied"] = (r - np.log(out["forward"] / out["spot"]) / out["T"]).clip(*q_bounds)
    return out.reset_index()[keys + ["forward", "q_implied"]]


def add_implied_dividend(df, r, q_default=0.015, **kwargs):
    """Ajoute la colonne 'q_implied' (q par échéance ; q_default si non estimable)."""
    est = estimate_implied_dividend(df, r, **kwargs)
    out = df.merge(est, on=["quote_date", "expiration"], how="left")
    out["q_implied"] = out["q_implied"].fillna(q_default)
    return out


# ======================================================================
# Démonstration / tests manuels (exécutés uniquement en `python implied_vol.py`)
# ======================================================================
if __name__ == "__main__":
    #%% Comparaison méthodes Brentq et Newton

    S = 100.0
    K = 105.0
    T = 0.5  # 6 mois
    r = 0.03
    sigma_true = 0.225  # Volatilité cible de 22.5 %


    market_call = black_scholes_price(S, K, T, r, sigma_true, option_type="call")
    market_put = black_scholes_price(S, K, T, r, sigma_true, option_type="put")

    iv_call_brent = implied_volatility(market_call, S, K, T, r, option_type="call")
    iv_put_brent = implied_volatility(market_put, S, K, T, r, option_type="put")


    iv_call_newton, iter_call, success = implied_volatility_newton(market_call, S, K, T, r, option_type="call")


    print("--- Validation Synthétique ---")
    print(f"Volatilité théorique initiale : {sigma_true * 100:.2f} %")
    print(
        f"IV Call retrouvée (Brent)     : {iv_call_brent * 100:.4f} % (Erreur : {abs(iv_call_brent - sigma_true):.2e})"
    )
    print(
        f"IV Put retrouvée (Brent)      : {iv_put_brent * 100:.4f} % (Erreur : {abs(iv_put_brent - sigma_true):.2e})"
    )
    print(
        f"IV Call (Newton-Raphson)      : {iv_call_newton * 100:.4f} % en {iter_call} itérations"
    )

    # Test d'un prix aberrant (inférieur à la borne inférieure)
    aberrant_price = 0.001
    bad_iv = implied_volatility(aberrant_price, S, K, T, r, option_type="call")
    print(f"\nTest prix sous la borne théorique : résultat = {bad_iv} (attendu: nan)")


    #%% TESTS

    # Exemple d'exécution
    # r : taux sans risque estimé en 2022 (ex. 2% ou 3%)
    # q : rendement moyen du dividende de SPY (~1.5%)
    r_market = 0.02
    q_market = 0.015

    cleaned_df = clean_option_chain(get_option_chain())[0]

    df_with_iv = add_implied_volatility(cleaned_df, r=r_market, q=q_market)


    total_rows = len(df_with_iv)
    status_counts = df_with_iv["iv_status"].value_counts()
    success_count = (df_with_iv["iv_status"] == "success").sum()
    success_rate = (success_count / total_rows) * 100

    print("\n" + "=" * 45)
    print("RAPPORT DE CALCUL DE L'IMPLIED VOLATILITY")
    print("=" * 45)
    print(f"Total des options analysées       : {total_rows:,d}")
    print(f"Nombre d'options avec IV calculée : {success_count:,d}")
    print(f"Taux de succès du solveur         : {success_rate:.2f} %\n")

    print("Détail par statut :")
    print(status_counts.to_string())

    # Sous-ensemble filtré : uniquement les IV valides pour les statistiques
    df_success = df_with_iv[df_with_iv["iv_status"] == "success"].copy()

    print("\n" + "-" * 45)
    print("Distribution de implied_vol (statistiques descriptives) :")
    print(df_success["implied_vol"].describe().to_string())

    # Comparaison source_iv vs implied_vol
    if "source_iv" in df_success.columns:
        # Nettoyage des éventuels NaN dans la source
        valid_comparison = df_success.dropna(subset=["source_iv"])
        abs_diff = (valid_comparison["implied_vol"] - valid_comparison["source_iv"]).abs()

        print("\n" + "-" * 45)
        print("Comparaison entre source_iv et implied_vol :")
        print(f"Écart moyen absolu : {abs_diff.mean():.4f}")
        print(f"Écart médian       : {abs_diff.median():.4f}")
        print(f"Écart maximum      : {abs_diff.max():.4f}")

    # IV moyenne selon la maturité et le type d'option
    iv_by_mat_type = (
        df_success.groupby(["expiration", "option_type"])["implied_vol"]
        .agg(["count", "mean", "std"])
        .round(4)
    )
    print("\n" + "-" * 45)
    print("IV moyenne par expiration et type d'option :")
    print(iv_by_mat_type)
