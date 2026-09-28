"""Pipeline de préparation des données (ajout) : chargement -> nettoyage -> forward
implicite -> volatilité implicite, avec cache parquet dans data/processed/.

Utilisé par les notebooks 03 à 07 pour garantir un traitement identique partout.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from market_data import get_option_chain, DATA_DIR
from data_cleaning import clean_option_chain
from implied_vol import add_implied_dividend, add_implied_volatility

R_DEFAULT = 0.02      # taux sans risque (hypothèse du projet)
Q_DEFAULT = 0.015     # rendement de dividende moyen de SPY
CACHE_DIR = DATA_DIR / "processed"


def otm_mask(df: pd.DataFrame) -> pd.Series:
    """Options hors de la monnaie : puts K < S, calls K >= S (convention de marché)."""
    return ((df["option_type"] == "put") & (df["strike"] < df["spot"])) | (
        (df["option_type"] == "call") & (df["strike"] >= df["spot"])
    )


def load_chain_with_iv(quote_date: str, r: float = R_DEFAULT, q="implied",
                       max_relative_spread: float = 0.5, use_cache: bool = True):
    """Chaîne d'options nettoyée d'une date, avec 'q_implied', 'implied_vol', 'iv_status',
    'log_moneyness' et 'is_otm'.

    q : "implied" (forward implicite par parité call-put, recommandé) ou un float.
    Retourne (df_iv, raw_df, control_table).
    """
    tag = "impq" if q == "implied" else f"q{q}"
    cache = CACHE_DIR / f"chain_iv_{quote_date}_{tag}_r{r}_s{max_relative_spread}.parquet"
    raw = get_option_chain(quote_date=quote_date)
    cleaned, control = clean_option_chain(raw, max_relative_spread=max_relative_spread)
    if use_cache and cache.exists():
        return pd.read_parquet(cache), raw, control

    if q == "implied":
        cleaned = add_implied_dividend(cleaned, r, q_default=Q_DEFAULT)
        df = add_implied_volatility(cleaned, r=r, q="q_implied")
    else:
        df = add_implied_volatility(cleaned, r=r, q=q)
        df["q_implied"] = q
    df["log_moneyness"] = np.log(df["strike"] / df["spot"])
    df["is_otm"] = otm_mask(df)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df, raw, control
