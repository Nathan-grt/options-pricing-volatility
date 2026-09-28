#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Chargement des données d'options SPY 2022 (format OptionsDX, EOD 16h00).

Le fichier est cherché dans ``data/`` à la racine du repo (ou dans le dossier
indiqué par la variable d'environnement ``SPY_DATA_DIR``). Si un fichier
``spy_2022.parquet`` existe, il est utilisé en priorité (lecture ~10x plus rapide).
"""

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd


DATA_DIR = Path(os.environ.get("SPY_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
CSV_PATH = DATA_DIR / "spy_2022.csv"
PARQUET_PATH = DATA_DIR / "spy_2022.parquet"

# Jours fériés NYSE 2022 présents dans le fichier avec des cotations figées
# (spot quasi identique à la veille, volumes recopiés) -> à exclure des séries temporelles.
US_MARKET_HOLIDAYS_2022 = [
    "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20",
    "2022-07-04", "2022-09-05", "2022-11-24", "2022-12-26",
]


@lru_cache(maxsize=1)
def load_raw_data() -> pd.DataFrame:
    """Lit le fichier brut UNE seule fois par session (cache mémoire)."""
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
    elif CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, low_memory=False)
    else:
        raise FileNotFoundError(
            f"Données introuvables : placez spy_2022.csv dans {DATA_DIR} (voir data/README.md)"
        )
    # Nettoyage des espaces éventuels dans les colonnes et les dates
    df.columns = df.columns.str.strip().str.replace("[", "").str.replace("]", "")
    for col in ("QUOTE_DATE", "EXPIRE_DATE"):
        df[col] = df[col].astype(str).str.strip()
    return df


def get_expirations(ticker: str = "SPY") -> list[str]:
    """Retourne la liste des dates d'expiration disponibles au format YYYY-MM-DD."""
    expirations = sorted(load_raw_data()["EXPIRE_DATE"].unique())
    return expirations


def get_quote_dates() -> list[str]:
    """Retourne la liste triée des dates de cotation disponibles."""
    return sorted(load_raw_data()["QUOTE_DATE"].unique())


def get_underlying_history(drop_holidays: bool = True) -> pd.Series:
    """Série journalière du spot SPY (clôture 16h) indexée par date."""
    df = load_raw_data()
    if drop_holidays:
        df = df[~df["QUOTE_DATE"].isin(US_MARKET_HOLIDAYS_2022)]
    s = df.groupby("QUOTE_DATE")["UNDERLYING_LAST"].first()
    s.index = pd.to_datetime(s.index)
    return s.sort_index().rename("spot")


def get_option_chain(ticker: str = "SPY", expiration: str = None, quote_date: str = None,
                     drop_holidays: bool = True) -> pd.DataFrame:
    """Récupère les Calls et Puts pour une expiration et les structure au format attendu.

    - expiration : None = toutes les échéances
    - quote_date : None = première date du fichier (comportement d'origine),
      "all" = toutes les dates (nécessaire pour un backtest historique),
      ou une date "YYYY-MM-DD"
    """
    df = load_raw_data()

    # Filtre par expiration
    if expiration is not None:
        df = df[df["EXPIRE_DATE"].astype(str).str.strip() == str(expiration)]

    # Filtre optionnel par date de cotation (sinon on prend la première disponible)
    if quote_date == "all":
        if drop_holidays:
            df = df[~df["QUOTE_DATE"].isin(US_MARKET_HOLIDAYS_2022)]
    elif quote_date is not None:
        df = df[df["QUOTE_DATE"].astype(str).str.strip() == str(quote_date)]
    else:
        first_date = df["QUOTE_DATE"].iloc[0]
        df = df[df["QUOTE_DATE"] == first_date]

    # --- 1. Extraction des CALLS ---
    calls = pd.DataFrame(
        {
            "underlying": ticker,
            "quote_date": df["QUOTE_DATE"].astype(str).str.strip(),
            "expiration": df["EXPIRE_DATE"].astype(str).str.strip(),
            "option_type": "call",
            "strike": pd.to_numeric(df["STRIKE"], errors="coerce"),
            "bid": pd.to_numeric(df["C_BID"], errors="coerce"),
            "ask": pd.to_numeric(df["C_ASK"], errors="coerce"),
            "last_price": pd.to_numeric(df["C_LAST"], errors="coerce"),
            "volume": pd.to_numeric(df["C_VOLUME"], errors="coerce").fillna(0),
            "open_interest": pd.to_numeric(
                df.get("C_SIZE", 0), errors="coerce"
            ).fillna(0),
            "source_iv": pd.to_numeric(df["C_IV"], errors="coerce"),
            "spot": pd.to_numeric(df["UNDERLYING_LAST"], errors="coerce"),
        }
    )

    # --- 2. Extraction des PUTS ---
    puts = pd.DataFrame(
        {
            "underlying": ticker,
            "quote_date": df["QUOTE_DATE"].astype(str).str.strip(),
            "expiration": df["EXPIRE_DATE"].astype(str).str.strip(),
            "option_type": "put",
            "strike": pd.to_numeric(df["STRIKE"], errors="coerce"),
            "bid": pd.to_numeric(df["P_BID"], errors="coerce"),
            "ask": pd.to_numeric(df["P_ASK"], errors="coerce"),
            "last_price": pd.to_numeric(df["P_LAST"], errors="coerce"),
            "volume": pd.to_numeric(df["P_VOLUME"], errors="coerce").fillna(0),
            "open_interest": pd.to_numeric(
                df.get("P_SIZE", 0), errors="coerce"
            ).fillna(0),
            "source_iv": pd.to_numeric(df["P_IV"], errors="coerce"),
            "spot": pd.to_numeric(df["UNDERLYING_LAST"], errors="coerce"),
        }
    )

    # --- 3. Concaténation des deux sous-ensembles ---
    combined = pd.concat([calls, puts], ignore_index=True)

    # Ordre strict des colonnes spécifié par la consigne
    target_columns = [
        "underlying",
        "quote_date",
        "expiration",
        "option_type",
        "strike",
        "bid",
        "ask",
        "last_price",
        "volume",
        "open_interest",
        "source_iv",
        "spot",
    ]

    return combined[target_columns]