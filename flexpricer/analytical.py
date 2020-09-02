"""
This module host analytical solution of regular option pricing for the use of benchmarking
"""
import abc
from typing import TypeVar, Callable, Dict, Type
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate

from flexpricer.base_component import PricerComponent


NumericT = TypeVar('NumericT', float, np.ndarray, complex)


def price_bs_call(s: float, k: float, r: float, q: float, sig: float, t: float) -> float:
    total_vol = sig * np.sqrt(t)
    d1 = (np.log(s / k) + (r - q) * t) / total_vol + 0.5 * total_vol
    d2 = d1 - total_vol
    return s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


def price_bs_put(s: float, k: float, r: float, q: float, sig: float, t: float) -> float:
    forward = s * np.exp(-q * t) - k * np.exp(-r * t)
    return price_bs_call(s, k, r, q, sig, t) - forward


@dataclass
class PhiGenerator(PricerComponent, abc.ABC):

    @abc.abstractmethod
    def generate(self) -> Callable[[complex], complex]:
        """ Generate characteristic function """


@dataclass
class BlackScholesPhi(PhiGenerator):
    r: float
    q: float
    sig: float
    t: float

    def generate(self) -> Callable[[complex], complex]:
        def phi(u: complex) -> complex:
            return np.exp(- 0.5 * u * (u + 1j) * self.sig ** 2 * self.t)
        return phi


@dataclass
class HestonPhi(PhiGenerator):

    t: float
    v0: float  # Beginning variance
    vbar: float  # Long term variance level
    kappa: float  # Reversion
    eta: float  # vol of variance
    rho: float  # spot-variance correlation

    def generate(self) -> Callable[[complex], complex]:
        eta2 = self.eta ** 2

        def phi(u: complex) -> complex:
            aa = -u ** 2 / 2 - 1j * u / 2
            bb = self.kappa - self.rho * self.eta * 1j * u
            cc = eta2 / 2
            d = np.sqrt(bb ** 2 - 4 * aa * cc)
            rp = (bb + d) / eta2
            rm = (bb - d) / eta2
            r_ratio = rm / rp
            exp_d = np.exp(-d * self.t)
            big_d = rm * (1 - exp_d) / (1 - r_ratio * exp_d)
            big_c = self.kappa * (rm * self.t - 2 / eta2 * np.log((1 - r_ratio * exp_d) / (1 - r_ratio)))
            return np.exp(big_c * self.vbar + big_d * self.v0)

        return phi


def price_call_with_phi(generator_cls: Type[PhiGenerator], s: float, k: float, r: float, q: float, t: float,
                        params: Dict[str, float]) -> float:
    all_params = {**{'s': s, 'k': k, 'r': r, 'q': q, 't': t}, **params}
    phi = generator_cls(**{k: all_params[k] for k in generator_cls.parameters()}).generate()
    y = np.log(s / k) + (r - q) * t

    def wrapper(u: float) -> float:
        return (np.exp(1j * u * y) * phi(u - 1j / 2)).real / (u ** 2 + 0.25)

    # noinspection PyTypeChecker
    integration = integrate.quad(wrapper, 0, 1000)
    return s * np.exp(-q * t) - np.sqrt(s * k) * np.exp(-(r + q) * t / 2) / np.pi * integration[0]
