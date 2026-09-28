"""Greeks analytiques vs différences finies centrées."""
import numpy as np
import pytest

from black_scholes import black_scholes_price as bs
from greeks import delta, gamma, vega, theta, rho

PARAMS = [
    (100, 100, 1.0, 0.03, 0.2, 0.01),
    (100, 80, 0.5, 0.05, 0.35, 0.0),
    (100, 130, 2.0, 0.01, 0.15, 0.02),
    (410, 400, 0.1, 0.02, 0.25, 0.015),
]
H = 1e-3


@pytest.mark.parametrize("S,K,T,r,sigma,q", PARAMS)
@pytest.mark.parametrize("opt", ["call", "put"])
def test_delta_fd(S, K, T, r, sigma, q, opt):
    fd = (bs(S + H, K, T, r, sigma, opt, q) - bs(S - H, K, T, r, sigma, opt, q)) / (2 * H)
    assert delta(S, K, T, r, sigma, opt, q) == pytest.approx(fd, abs=1e-6)


@pytest.mark.parametrize("S,K,T,r,sigma,q", PARAMS)
@pytest.mark.parametrize("opt", ["call", "put"])
def test_gamma_fd(S, K, T, r, sigma, q, opt):
    h = 1e-2
    fd = (bs(S + h, K, T, r, sigma, opt, q) - 2 * bs(S, K, T, r, sigma, opt, q) + bs(S - h, K, T, r, sigma, opt, q)) / h**2
    assert gamma(S, K, T, r, sigma, opt, q) == pytest.approx(fd, rel=1e-4, abs=1e-7)


@pytest.mark.parametrize("S,K,T,r,sigma,q", PARAMS)
@pytest.mark.parametrize("opt", ["call", "put"])
def test_vega_fd(S, K, T, r, sigma, q, opt):
    h = 1e-5
    fd = (bs(S, K, T, r, sigma + h, opt, q) - bs(S, K, T, r, sigma - h, opt, q)) / (2 * h)
    assert vega(S, K, T, r, sigma, opt, q) == pytest.approx(fd, rel=1e-6)


@pytest.mark.parametrize("S,K,T,r,sigma,q", PARAMS)
@pytest.mark.parametrize("opt", ["call", "put"])
def test_theta_fd(S, K, T, r, sigma, q, opt):
    # theta = dV/dt = -dV/dT
    h = 1e-5
    fd = -(bs(S, K, T + h, r, sigma, opt, q) - bs(S, K, T - h, r, sigma, opt, q)) / (2 * h)
    assert theta(S, K, T, r, sigma, opt, q) == pytest.approx(fd, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("S,K,T,r,sigma,q", PARAMS)
@pytest.mark.parametrize("opt", ["call", "put"])
def test_rho_fd(S, K, T, r, sigma, q, opt):
    h = 1e-6
    fd = (bs(S, K, T, r + h, sigma, opt, q) - bs(S, K, T, r - h, sigma, opt, q)) / (2 * h)
    assert rho(S, K, T, r, sigma, opt, q) == pytest.approx(fd, rel=1e-5)


def test_gamma_vega_same_for_call_put():
    assert gamma(100, 95, 1, 0.02, 0.2, "call") == gamma(100, 95, 1, 0.02, 0.2, "put")
    assert vega(100, 95, 1, 0.02, 0.2, "call") == vega(100, 95, 1, 0.02, 0.2, "put")
