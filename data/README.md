# Données

Le fichier `spy_2022.csv` (~346 Mo, 1 146 980 lignes) **n'est pas versionné** : il dépasse la limite de 100 Mo de GitHub.

## Source

Chaîne d'options **SPY** de fin de journée (clôture 16h00 ET) fournie par **OptionsDX** (historique EOD 2020-2022), filtrée sur l'année 2022 avec `scripts/extract_spy_2022.py`.

## Reproduire

1. Télécharger l'historique SPY EOD 2020-2022 sur Kaggle (fichier `spy_2020_2022.csv`).
2. Depuis la racine du repo :
   ```bash
   python scripts/extract_spy_2022.py chemin/vers/spy_2020_2022.csv --parquet
   ```
   Le script écrit `data/spy_2022.csv` et, avec `--parquet`, `data/spy_2022.parquet` (80 Mo, lu en ~1 s contre ~10 s pour le CSV). `src/market_data.py` utilise le parquet s'il existe.
3. Un autre dossier peut être indiqué via la variable d'environnement `SPY_DATA_DIR`.

## Format (une ligne = une date × une échéance × un strike, call et put côte à côte)

| Colonne | Description |
|---|---|
| `QUOTE_DATE`, `QUOTE_TIME_HOURS` | date et heure de la cotation (16h00) |
| `UNDERLYING_LAST` | spot SPY |
| `EXPIRE_DATE`, `DTE` | échéance, jours restants |
| `STRIKE` | prix d'exercice |
| `C_BID`, `C_ASK`, `C_LAST`, `C_VOLUME`, `C_IV`, `C_DELTA`… | call : cotations, volume, IV et Greeks du fournisseur |
| `P_BID`, `P_ASK`, `P_LAST`, `P_VOLUME`, `P_IV`, `P_DELTA`… | put : idem |
| `C_SIZE`, `P_SIZE` | tailles bid × ask (texte "3 x 1") — **pas d'open interest** dans ce jeu de données |

## Particularités constatées

- **256 dates de cotation**, dont **9 jours fériés NYSE** (17/01, 21/02, 15/04, 30/05, 20/06, 04/07, 05/09, 24/11, 26/12) avec des cotations figées : spot quasi identique à la veille et volumes recopiés. Ils sont exclus des séries temporelles (`market_data.US_MARKET_HOLIDAYS_2022`), soit 247 séances utilisées.
- Le fichier n'est pas trié par date : la première ligne est au 01/08/2022, qui sert de date de référence par défaut dans `get_option_chain()`.
- Les IV du fournisseur (`C_IV`, `P_IV`) sont parfois vides et atteignent plus de 300 % pour les options 0DTE très OTM.
- `data/processed/` contient les caches parquet générés par `src/pipeline.py`, également ignorés par git.
