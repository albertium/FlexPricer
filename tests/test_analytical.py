"""
Tests for analytical functions
"""
import numpy as np

from flexpricer.analytical import price_bs_call, price_bs_put, price_call_with_phi, BlackScholesPhi


def test_black_scholes():
    """
    Benchmark against https://www.math.drexel.edu/~pg/fin/VanillaCalculator.html
    * Gamma from this website does not seem correct
    """
    s = 100
    k = 105
    r = 0.02
    q = 0.01
    sig = 0.2
    t = 0.25
    assert abs(price_bs_call(s, k, r, q, sig, t) - 2.13724) < 1e-5
    assert abs(price_bs_put(s, k, r, q, sig, t) - 6.86323) < 1e-5


def test_black_scholes_characteristic():
    """ Benchmark against forward and Black Scholes formula """
    s = 100
    k = 105
    r = 0.05
    q = 0.02
    sig = 0.3
    t = 0.25
    phi = BlackScholesPhi(r, q, sig, t).generate()
    assert phi(-1j) == 1  # We factor out the drift so we should get martingale here
    assert abs(price_call_with_phi(BlackScholesPhi, s, k, r, q, t, {'sig': sig})
               - price_bs_call(s, k, r, q, sig, t)) < 1e-10
