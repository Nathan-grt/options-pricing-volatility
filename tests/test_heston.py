"""Tests du modèle de Heston (simulation et pricing semi-analytique)."""
import numpy as np
import pytest

from black_scholes import black_scholes_price as bs
from heston import simulate_heston_paths, satisfies_feller_condition
from heston_pricing import heston_call_price, heston_put_price


def test_converges_to_black_scholes_when_volvol_vanishes():
    # xi -> 0 et v0 = theta : la variance est constante = v0 -> Black-Scholes
    for K in (80, 100, 120):
        h = heston_call_price(100, K, 1.0, 0.02, 0.04, 2.0, 0.04, 1e-3, -0.5, q=0.01)
        assert h == pytest.approx(bs(100, K, 1.0, 0.02, 0.2, "call", 0.01), rel=1e-3)


@pytest.mark.parametrize("K", [80, 100, 125])
def test_heston_put_call_parity(K):
    p = dict(S0=100, K=K, T=0.75, r=0.03, v0=0.05, kappa=1.5, theta=0.06, xi=0.6, rho=-0.7, q=0.01)
    c, pu = heston_call_price(**p), heston_put_price(**p)
    assert c - pu == pytest.approx(100 * np.exp(-0.01 * 0.75) - K * np.exp(-0.03 * 0.75), abs=1e-8)


def test_semi_analytic_vs_monte_carlo():
    S0, v0, r, kappa, theta, xi, rho, T, K = 100, 0.04, 0.02, 2.0, 0.04, 0.3, -0.7, 1.0, 100
    S, _ = simulate_heston_paths(S0, v0, r, kappa, theta, xi, rho, T, 250, 100_000, seed=3)
    disc = np.exp(-r * T) * np.maximum(S[-1] - K, 0)
    mc, se = disc.mean(), disc.std(ddof=1) / np.sqrt(len(disc))
    # tolérance : erreur statistique + petit biais de discrétisation d'Euler
    assert abs(mc - heston_call_price(S0, K, T, r, v0, kappa, theta, xi, rho)) < 4 * se + 0.05


def test_negative_rho_generates_skew():
    # rho < 0 -> IV des strikes bas > IV des strikes hauts (skew action)
    from implied_vol import implied_volatility
    ivs = []
    for K in (85, 115):
        price = heston_call_price(100, K, 0.5, 0.02, 0.04, 2.0, 0.04, 0.6, -0.8)
        ivs.append(implied_volatility(price, 100, K, 0.5, 0.02, "call"))
    assert ivs[0] > ivs[1] + 0.02


def test_simulation_properties():
    a = simulate_heston_paths(100, 0.04, 0.02, 2, 0.04, 0.3, -0.7, 1, 50, 200, seed=1)
    b = simulate_heston_paths(100, 0.04, 0.02, 2, 0.04, 0.3, -0.7, 1, 50, 200, seed=1)
    assert np.allclose(a[0], b[0]) and np.all(a[0] > 0)
    with pytest.raises(ValueError):
        simulate_heston_paths(100, 0.04, 0.02, 2, 0.04, 0.3, 1.5, 1, 50, 10)


def test_feller():
    assert satisfies_feller_condition(2.0, 0.04, 0.3)      # 0.16 > 0.09
    assert not satisfies_feller_condition(1.0, 0.04, 0.5)  # 0.08 < 0.25


def test_invalid_params_return_nan():
    assert np.isnan(heston_call_price(100, 100, 1, 0.02, 0.04, -1, 0.04, 0.3, -0.7))


def test_short_maturity_integration_truncation():
    """Régression : avec la borne fixe u <= 100, le prix à T = 0.05 an était faux
    d'environ 0.03 $ (oscillations de plusieurs points d'IV dans l'aile)."""
    from scipy.integrate import quad
    import heston_pricing as hp
    S, K, T, r = 410.72, 330.0, 0.049, 0.02
    p = (0.0488, 2.46, 0.097, 1.34, -0.589)
    lnK, fwd = np.log(K), S * np.exp(r * T)
    f1 = lambda u: (np.exp(-1j * u * lnK) * hp._heston_char_func(u - 1j, S, p[0], r, *p[1:], T) / (1j * u * fwd)).real
    f2 = lambda u: (np.exp(-1j * u * lnK) * hp._heston_char_func(u, S, p[0], r, *p[1:], T) / (1j * u)).real
    ref = S * (0.5 + quad(f1, 0, 5000, limit=2000)[0] / np.pi) - K * np.exp(-r * T) * (0.5 + quad(f2, 0, 5000, limit=2000)[0] / np.pi)
    assert heston_call_price(S, K, T, r, *p) == pytest.approx(ref, abs=1e-6)
