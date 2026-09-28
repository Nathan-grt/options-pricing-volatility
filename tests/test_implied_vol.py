"""Inversion de la volatilité implicite (Brent / Newton) et forward implicite."""
import numpy as np
import pandas as pd
import pytest

from black_scholes import black_scholes_price as bs
from implied_vol import (implied_volatility, implied_volatility_newton,
                         check_arbitrage_bounds, estimate_implied_dividend)

GRID = [(K, T, s) for K in (70, 90, 100, 110, 140) for T in (0.05, 0.5, 2.0) for s in (0.1, 0.3, 0.8)]


@pytest.mark.parametrize("K,T,sigma", GRID)
@pytest.mark.parametrize("opt", ["call", "put"])
def test_brent_round_trip(K, T, sigma, opt):
    price = bs(100, K, T, 0.02, sigma, opt, 0.01)
    if not check_arbitrage_bounds(price, 100, K, T, 0.02, opt, 0.01):
        pytest.skip("prix trop proche de la valeur intrinsèque (vega ~ 0)")
    assert implied_volatility(price, 100, K, T, 0.02, opt, 0.01) == pytest.approx(sigma, abs=1e-5)


@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.4])
def test_newton_round_trip_atm(sigma):
    price = bs(100, 100, 1.0, 0.02, sigma, "call")
    iv, n_iter, ok = implied_volatility_newton(price, 100, 100, 1.0, 0.02, "call")
    assert ok and iv == pytest.approx(sigma, abs=1e-6) and n_iter < 10


def test_newton_can_fail_far_otm():
    # Deep OTM, maturité courte : vega ~ 0, Newton diverge depuis sigma_init=20 %
    price = bs(100, 150, 0.1, 0.02, 0.9, "call")
    iv_newton, _, ok = implied_volatility_newton(price, 100, 150, 0.1, 0.02, "call")
    iv_brent = implied_volatility(price, 100, 150, 0.1, 0.02, "call")
    assert not ok and np.isnan(iv_newton)
    assert iv_brent == pytest.approx(0.9, abs=1e-4)


def test_arbitrage_violation_returns_nan():
    # call K=90 à 5 $ alors que sa valeur intrinsèque actualisée vaut ~11.3 $ -> arbitrage
    iv, status = implied_volatility(5.0, 100, 90, 0.5, 0.03, "call", return_status=True)
    assert np.isnan(iv) and status == "arbitrage_bounds"
    # NB : un call OTM à 0.001 $ n'est PAS un arbitrage (borne basse = 0) -> IV très faible mais valide
    iv = implied_volatility(0.001, 100, 105, 0.5, 0.03, "call")
    assert 0 < iv < 0.05
    iv, status = implied_volatility(5, 100, 105, 0, 0.03, "call", return_status=True)
    assert status == "invalid_maturity"


def test_implied_dividend_recovers_true_q():
    S, r, q_true = 400.0, 0.03, 0.012
    rows = []
    for T in (0.1, 0.5, 1.0):
        for K in np.arange(360, 441, 5.0):
            for opt in ("call", "put"):
                rows.append({"quote_date": "d", "expiration": f"e{T}", "strike": K, "spot": S, "T": T,
                             "option_type": opt, "mid": bs(S, K, T, r, 0.2, opt, q_true)})
    est = estimate_implied_dividend(pd.DataFrame(rows), r)
    assert np.allclose(est["q_implied"], q_true, atol=1e-10)
