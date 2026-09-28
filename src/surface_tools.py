"""Outils d'interpolation de la surface de volatilité (ajout).

Convention : on travaille en log-moneyness FORWARD k = ln(K/F) et on interpole
- en k à maturité fixée (linéaire sur les options OTM),
- en T via la variance totale w = sigma^2 * T (linéaire en T), ce qui
  évite les arbitrages calendaires grossiers.
"""
import numpy as np
import pandas as pd


def otm_slices(df: pd.DataFrame, min_T: float = 1 / 365) -> dict:
    """Dictionnaire {T: DataFrame trié par k} des options OTM avec IV valide."""
    d = df[(df["iv_status"] == "success") & df["is_otm"] & (df["T"] >= min_T)].copy()
    fwd = d["forward"] if "forward" in d else d["spot"] * np.exp((0.02 - d["q_implied"]) * d["T"])
    d["k"] = np.log(d["strike"] / fwd)
    out = {}
    for T, g in d.groupby("T"):
        g = g.sort_values("k").drop_duplicates("k")
        if len(g) >= 5:
            out[T] = g
    return out


def iv_at(slices: dict, k: float, T_target: float) -> float:
    """IV interpolée à log-moneyness forward k et maturité T_target."""
    Ts = np.array(sorted(slices))
    if len(Ts) == 0 or T_target < Ts[0] or T_target > Ts[-1]:
        return np.nan

    def sigma_k(T):
        g = slices[T]
        if k < g["k"].iloc[0] or k > g["k"].iloc[-1]:
            return np.nan
        return float(np.interp(k, g["k"], g["implied_vol"]))

    i = np.searchsorted(Ts, T_target)
    if Ts[min(i, len(Ts) - 1)] == T_target:
        return sigma_k(T_target)
    T1, T2 = Ts[i - 1], Ts[i]
    s1, s2 = sigma_k(T1), sigma_k(T2)
    if np.isnan(s1) or np.isnan(s2):
        return np.nan
    w = s1**2 * T1 + (s2**2 * T2 - s1**2 * T1) * (T_target - T1) / (T2 - T1)
    return float(np.sqrt(max(w, 0) / T_target))


def term_structure(slices: dict, band: float = 0.05) -> pd.DataFrame:
    """Par échéance : IV ATM (k=0) et pente du smile dIV/dk (régression sur |k| < band)."""
    rows = []
    for T, g in slices.items():
        if g["k"].min() > 0 or g["k"].max() < 0:
            continue
        atm = float(np.interp(0.0, g["k"], g["implied_vol"]))
        loc = g[g["k"].abs() < band * max(1.0, np.sqrt(T / 0.25))]
        slope = np.polyfit(loc["k"], loc["implied_vol"], 1)[0] if len(loc) >= 4 else np.nan
        rows.append({"T": T, "expiration": g["expiration"].iloc[0], "atm_iv": atm, "skew_slope": slope})
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)
