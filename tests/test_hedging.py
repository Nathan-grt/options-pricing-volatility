"""Tests du delta-hedging (cohérence comptable et propriétés théoriques)."""
import numpy as np
import pytest

from hedging import simulate_gbm_paths, delta_hedging, delta_hedging_vectorized


@pytest.mark.parametrize("opt", ["call", "put"])
@pytest.mark.parametrize("k", [1, 3])
def test_vectorized_matches_loop(opt, k):
    P = simulate_gbm_paths(100, 0.5, 0.25, 0.02, 0.015, 60, 15, seed=4)
    loop = [delta_hedging(P[:, j], opt, 100, 0.5, 0.02, 0.25, k, 0.001, 0.015)["cumulative_pnl"].iloc[-1]
            for j in range(P.shape[1])]
    vec = delta_hedging_vectorized(P, opt, 100, 0.5, 0.02, 0.25, k, 0.001, 0.015)["pnl"]
    assert np.allclose(loop, vec, atol=1e-10)


def test_initial_portfolio_is_minus_costs():
    P = simulate_gbm_paths(100, 1, 0.2, 0.02, 0.0, 10, 1, seed=0)[:, 0]
    df = delta_hedging(P, "call", 100, 1, 0.02, 0.2, "daily", 0.001)
    assert df["portfolio_value"].iloc[0] == pytest.approx(-df["transaction_costs"].iloc[0])


def test_pnl_unbiased_when_vol_correct():
    # vol de couverture = vol réalisée, sans frais : E[P&L] ~ 0 (y compris avec dividende)
    P = simulate_gbm_paths(100, 1, 0.2, 0.02, 0.015, 252, 4000, seed=11)
    pnl = delta_hedging_vectorized(P, "call", 100, 1, 0.02, 0.2, 1, 0.0, 0.015)["pnl"]
    assert abs(pnl.mean()) < 4 * pnl.std() / np.sqrt(len(pnl))


def test_hedging_error_scales_like_one_over_sqrt_n():
    P = simulate_gbm_paths(100, 1, 0.2, 0.02, 0.0, 1008, 3000, seed=5)
    std_daily = delta_hedging_vectorized(P, "call", 100, 1, 0.02, 0.2, 4, 0.0)["pnl"].std()   # 252 rebal.
    std_4x = delta_hedging_vectorized(P, "call", 100, 1, 0.02, 0.2, 1, 0.0)["pnl"].std()      # 1008 rebal.
    assert std_daily / std_4x == pytest.approx(2.0, rel=0.15)


def test_short_option_loses_when_realized_vol_higher():
    P = simulate_gbm_paths(100, 1, 0.30, 0.02, 0.0, 252, 2000, seed=2)
    pnl = delta_hedging_vectorized(P, "call", 100, 1, 0.02, 0.20, 1, 0.0)["pnl"]
    assert pnl.mean() < 0
