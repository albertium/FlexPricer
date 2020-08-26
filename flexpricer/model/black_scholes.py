"""
Black Scholes model
"""
from typing import List, Tuple, Dict
from dataclasses import dataclass
from jax import numpy as np

from flexpricer.model.base_model import Model


@dataclass
class BlackScholes(Model):

    spot: float
    rate: float
    dividend: float
    volatility: float

    def _get_required_schedule(self, expiration) -> Tuple[float]:
        return (expiration,)

    def _generate_slices(self, dts: Tuple[float], innovations: np.ndarray) -> List[Dict[str, np.ndarray]]:
        grid = []
        spot = np.array(self.spot)
        drift = self.rate - self.dividend - 0.5 * self.volatility ** 2
        numeraire = np.array(1.0)
        for dt, innovation in zip(dts, innovations):
            numeraire = numeraire * np.exp(self.rate * dt)
            spot = spot * np.exp(drift * dt + self.volatility * np.sqrt(dt) * innovation)
            grid.append({'spot': spot, 'numeraire': numeraire})

        return grid
