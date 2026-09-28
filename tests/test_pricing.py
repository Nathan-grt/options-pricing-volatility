"""Tests du moteur de pricing Black-Scholes et Monte Carlo."""
import numpy as np
import pytest

from black_scholes import black_scholes_price
from monte_carlo import price_monte_carlo

RNG = np.random.default_rng(0)
CASES = [
    (RNG.uniform(50, 150), RNG.uniform(50, 150), RNG.uniform(0.05, 3), RNG.uniform(0, 0.08),
     RNG.uniform(0.05, 0.8), RNG.uniform(0, 0.04))
    for _ in range(200)
]


def test_reference_value():
    # Valeur de référence classique (Hull) : S=K=100, T=1, r=5%, sigma=20%
    assert black_scholes_price(100, 100, 1, 0.05, 0.2, "call") == pytest.approx(10.4506, abs=1e-4)
    assert black_scholes_price(100, 100, 1, 0.05, 0.2, "put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
def test_put_call_parity(S, K, T, r, sigma, q):
    c = black_scholes_price(S, K, T, r, sigma, "call", q)
    p = black_scholes_price(S, K, T, r, sigma, "put", q)
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-8)


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES[:50])
def test_no_arbitrage_bounds(S, K, T, r, sigma, q):
    c = black_scholes_price(S, K, T, r, sigma, "call", q)
    assert max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) - 1e-10 <= c <= S * np.exp(-q * T) + 1e-10


def test_limits():
    assert black_scholes_price(120, 100, 0, 0.05, 0.2, "call") == 20
    assert black_scholes_price(80, 100, 0, 0.05, 0.2, "put") == 20
    # sigma = 0 -> valeur intrinsèque actualisée du forward
    assert black_scholes_price(100, 90, 1, 0.05, 0.0, "call") == pytest.approx(100 - 90 * np.exp(-0.05))


def test_monotonicity_in_vol():
    prices = [black_scholes_price(100, 100, 1, 0.02, s, "call") for s in np.linspace(0.05, 1, 20)]
    assert np.all(np.diff(prices) > 0)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        black_scholes_price(100, 100, 1, 0.02, 0.2, "digital")
    with pytest.raises(ValueError):
        black_scholes_price(100, -1, 1, 0.02, 0.2, "call")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_monte_carlo_converges_to_bs(option_type):
    res = price_monte_carlo(100, 105, 1.0, 0.03, 0.25, 400_000, option_type, q=0.01, seed=7)
    # le prix exact doit être dans l'IC à 95 % et l'erreur < 4 erreurs-types
    assert res["in_ci"]
    assert res["abs_error"] < 4 * res["se"]


def test_monte_carlo_error_decreases():
    se = [price_monte_carlo(100, 100, 1, 0.05, 0.2, n, seed=1)["se"] for n in (1_000, 100_000)]
    # l'erreur standard décroît en 1/sqrt(N) : ratio attendu ~ 10
    assert se[0] / se[1] == pytest.approx(10, rel=0.15)
