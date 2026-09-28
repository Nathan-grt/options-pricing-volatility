#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Nettoyage de la chaîne d'options + tableau de contrôle des filtres."""

import pandas as pd



def year_fraction(quote_date, expiration):
    return (expiration - quote_date).dt.days / 365.0


def clean_option_chain(df, max_relative_spread: float = 0.5, min_volume: int = 0):
    """Nettoie la chaîne d'options et renvoie (df_clean, control_table).

    min_volume : AJOUT optionnel, filtre de liquidité (0 = désactivé, comportement d'origine).
    Remarque : le fichier OptionsDX ne contient pas d'open interest (C_SIZE/P_SIZE sont
    des tailles bid x ask), donc aucun filtre d'open interest n'est appliqué.
    """
    df_clean=df.copy()
    df_clean["strike"]=pd.to_numeric(df_clean["strike"], errors="coerce")  #le coerce permet de ne pas renvoyer d'erreur si conversion impossible, mais met NaN
    df_clean["bid"]=pd.to_numeric(df_clean["bid"], errors="coerce")
    df_clean["ask"]=pd.to_numeric(df_clean["ask"], errors="coerce")
    df_clean["last_price"]=pd.to_numeric(df_clean["last_price"], errors="coerce")
    df_clean["quote_date"]=pd.to_datetime(df_clean["quote_date"], format='%Y-%m-%d')
    df_clean["expiration"]=pd.to_datetime(df_clean["expiration"], format='%Y-%m-%d')
    df_clean["volume"]=df_clean["volume"].astype('Int64')
    df_clean["open_interest"]=df_clean["open_interest"].astype('Int64')
    
    df_clean["mid"] = (df_clean["bid"] + df_clean["ask"]) / 2
    df_clean["spread"] = df_clean["ask"] - df_clean["bid"]
    df_clean["relative_spread"] = df_clean["spread"] / df_clean["mid"]
    df_clean["T"] = year_fraction(df_clean["quote_date"], df_clean["expiration"])

    
    log = []
    log.append({"Étape": "Données brutes", "Nombre de lignes": len(df_clean)})
    
    #suppression doublons
    df_clean = df_clean.drop_duplicates(subset=["quote_date", "expiration", "option_type", "strike"])
    log.append({"Étape": "Après suppression des doublons","Nombre de lignes": len(df_clean),})
    
    df_clean=df_clean[df_clean["bid"] >= 0]
    df_clean=df_clean[df_clean["ask"] > 0]
    df_clean=df_clean[df_clean["ask"] >= df_clean["bid"]]
    df_clean=df_clean[df_clean["mid"] > 0]
    df_clean=df_clean[df_clean["strike"] > 0]
    df_clean=df_clean[df_clean["spot"] > 0]
    
    log.append({"Étape": "Après vérification bid/ask","Nombre de lignes": len(df_clean),})
    
    df_clean=df_clean[df_clean["T"]>0]
    log.append({"Étape": "Après filtre de maturité","Nombre de lignes": len(df_clean),})
    
    df_clean=df_clean[df_clean["relative_spread"] <= max_relative_spread] #seuil arbitraire
    log.append({"Étape": "Après filtre de spread","Nombre de lignes": len(df_clean),})

    if min_volume > 0:
        df_clean = df_clean[df_clean["volume"] >= min_volume]
        log.append({"Étape": f"Après filtre de volume (>= {min_volume})","Nombre de lignes": len(df_clean),})
    
    
    control_table = pd.DataFrame(log)
    
    return df_clean, control_table
    
    

    


