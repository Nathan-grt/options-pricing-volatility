#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extrait l'année 2022 du fichier OptionsDX SPY 2020-2022 vers data/.

Usage (depuis la racine du repo) :
    python scripts/extract_spy_2022.py chemin/vers/spy_2020_2022.csv [--parquet]

Modifications par rapport au script d'origine : chemins d'entrée/sortie en
arguments, écriture dans data/, option --parquet (lecture ~10x plus rapide).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
input_csv = sys.argv[1] if len(sys.argv) > 1 else "spy_2020_2022.csv"
write_parquet = "--parquet" in sys.argv
out_dir = ROOT / "data"
out_dir.mkdir(exist_ok=True)

chunk_size = 100_000
filtered_chunks = []

for chunk in pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False):
    # 1. Nettoyage des noms de colonnes (retire crochets et espaces superflus)
    chunk.columns = (
        chunk.columns.str.strip().str.replace("[", "").str.replace("]", "")
    )

    # 2. Détection automatique de la colonne de date (en majuscules ou minuscules)
    date_col = next(
        (c for c in chunk.columns if c.upper() in ["QUOTE_DATE", "DATE"]), None
    )
    if date_col is None:
        raise ValueError(
            f"Colonne de date introuvable parmi : {list(chunk.columns)}"
        )

    # 3. Filtrage sur l'année 2022
    mask = chunk[date_col].astype(str).str.contains("2022")
    filtered_chunks.append(chunk[mask])


df_2022 = pd.concat(filtered_chunks, ignore_index=True)
df_2022.to_csv(out_dir / "spy_2022.csv", index=False)
if write_parquet:
    sys.path.insert(0, str(ROOT / "src"))
    import market_data  # noqa: E402  (nettoie les colonnes/dates de façon identique)
    market_data.load_raw_data().to_parquet(out_dir / "spy_2022.parquet", index=False)

print(f"Extraction terminée avec succès : {len(df_2022):,d} lignes pour 2022.")
