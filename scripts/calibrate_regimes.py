#!/usr/bin/env python3
"""Calibration de Heston sur les 4 dates de régime de 2022 (complément du notebook 05).

Usage (depuis la racine du repo) : python scripts/calibrate_regimes.py   (~25 min)
Même échantillonnage et même procédure que le notebook 05 : options OTM, 6 échéances
mensuelles, multi-start à 3 départs, repli Powell, validation hors échantillon.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calibration import calibrate_heston, validate_calibration  # noqa: E402
from heston import satisfies_feller_condition  # noqa: E402
from pipeline import load_chain_with_iv, R_DEFAULT  # noqa: E402
from plotting import set_style, savefig, save_table, PALETTE  # noqa: E402
from surface_tools import otm_slices, iv_at  # noqa: E402

r = R_DEFAULT
BOUNDS = [(0.001, 1.0), (0.01, 10.0), (0.001, 1.0), (0.01, 5.0), (-0.99, 0.99)]
X0_ORIGIN = [0.04, 2.0, 0.04, 0.3, -0.7]
REGIMES = {"Calme (03/01)": "2022-01-03", "Stress (16/06)": "2022-06-16",
           "Rebond (16/08)": "2022-08-16", "Sell-off (12/10)": "2022-10-12"}


def calibration_sample(df, strike_step=10, lm_range=(-0.25, 0.10), target_days=(21, 45, 80, 140, 230, 320)):
    d = df[(df.iv_status == "success") & df.is_otm & df.log_moneyness.between(*lm_range)].copy()
    exps = d.groupby("expiration")["T"].first()
    third_fridays = exps[[e.weekday() == 4 and 15 <= e.day <= 21 for e in exps.index]]
    chosen = sorted({third_fridays.index[np.argmin(np.abs(third_fridays.values - t / 365))] for t in target_days})
    return d[d.expiration.isin(chosen) & (d.strike % strike_step == 0)], d[d.expiration.isin(chosen)]


def run(date):
    df, _, _ = load_chain_with_iv(date)
    calib, full = calibration_sample(df)
    sl = otm_slices(df)
    v0m, thm = iv_at(sl, 0.0, 30 / 365) ** 2, iv_at(sl, 0.0, 1.0) ** 2
    starts = {"Guidé par le marché": [v0m, 2.0, thm, 0.5, -0.7], "Script d'origine": X0_ORIGIN,
              "Stress (κ, ξ élevés)": [v0m, 5.0, thm, 2.0, -0.6]}
    runs = {}
    for lab, x0 in starts.items():
        t0 = time.perf_counter()
        runs[lab] = calibrate_heston(calib, x0, BOUNDS, r=r, q="q_implied", iv_col="implied_vol")
        print(f"  {date} | {lab:<22} RMSE {100*np.sqrt(runs[lab]['loss']):.2f} pts "
              f"({runs[lab]['method']}, {time.perf_counter()-t0:.0f} s)", flush=True)
    best = min(runs, key=lambda k: runs[k]["loss"])
    return runs[best], best, full


if __name__ == "__main__":
    set_style()
    rows = []
    for name, date in REGIMES.items():
        res, best, full = run(date)
        oos = validate_calibration(full, res["params"], r=r, q="q_implied", verbose=False)
        p = res["params"]
        rows.append({"Régime": name, "Départ retenu": best, "v0": p[0], "√v0": np.sqrt(p[0]), "kappa": p[1],
                     "theta": p[2], "√θ": np.sqrt(p[2]), "xi": p[3], "rho": p[4],
                     "Feller": satisfies_feller_condition(p[1], p[2], p[3]),
                     "RMSE calib (pts)": 100 * np.sqrt(res["loss"]),
                     "RMSE hors-éch. (pts)": np.sqrt(oos.sq_iv_error_pts.mean())})
    tab = pd.DataFrame(rows)
    save_table(tab, "05_heston_params_by_regime", floatfmt=".3f")
    print(tab.to_string(index=False))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    cols = [PALETTE["teal"], PALETTE["crimson"], PALETTE["sky"], PALETTE["purple"]]
    for ax, (c_, lab) in zip(axes, [("√v0", "Vol instantanée √v₀"), ("√θ", "Vol de long terme √θ"),
                                    ("xi", "Vol-of-vol ξ"), ("rho", "Corrélation ρ")]):
        ax.bar(tab["Régime"].str.split(" ").str[0], tab[c_], color=cols)
        ax.set_title(lab); ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Paramètres de Heston calibrés selon le régime de marché", fontweight="bold")
    fig.tight_layout(); savefig(fig, "05_heston_params_by_regime.png")
