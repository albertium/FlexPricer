"""
Heston stochastic volatility model
"""
from typing import List, Tuple, Dict
from dataclasses import dataclass
from jax import numpy as np

from flexpricer.model.base_model import Model


@dataclass
class Heston(Model):

    spot: float
    rate: float
    dividend: float
    volatility: float
    vbar: float
    kappa: float
    eta: float

    def _get_required_schedule(self, expiration) -> Tuple[float]:
        return (expiration,)

    def _generate_slices(self, dts: Tuple[float], innovations: np.ndarray) -> List[Dict[str, np.ndarray]]:
        grid = []
        spot = np.array(self.spot)
        new_var = np.array(self.volatility ** 2)
        carry = self.rate - self.dividend
        numeraire = np.array(1.0)
        for dt, innovation in zip(dts, innovations):
            # Book keeping
            var = new_var

            # Numeraire
            numeraire = numeraire * np.exp(self.rate * dt)

            # Variance process
            v2 = np.maximum(var, 0)
            new_var = var - self.kappa * (v2 - self.vbar) * dt + self.eta * np.sqrt(v2 * dt) * innovation[0]

            # Spot process
            spot = spot * np.exp(carry * dt - (var + new_var) / 4 * dt + np.sqrt(v2 * dt) * innovation[1])
            grid.append({'spot': spot, 'variance': new_var, 'numeraire': numeraire})

        return grid
