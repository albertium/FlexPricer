"""
Arithmetic Black Scholes model.
* Use Geometric Black Scholes parameters as input and convert them internally
* Does not support rate and dividend
"""
from typing import List, Tuple, Dict
from dataclasses import dataclass
from jax import numpy as np

from flexpricer.model.base_model import Model


@dataclass
class ArithmeticBlackScholes(Model):

    spot: float
    volatility: float

    def _get_required_schedule(self, expiration) -> Tuple[float]:
        return (expiration,)

    def _generate_slices(self, dts: Tuple[float], innovations: np.ndarray) -> List[Dict[str, np.ndarray]]:
        grid = []
        spot = np.array(self.spot)
        numeraire = np.array(1.0)
        arithmetic_vol = self.volatility * self.spot
        for dt, innovation in zip(dts, innovations):
            spot = spot + arithmetic_vol * np.sqrt(dt) * innovation
            grid.append({'spot': spot, 'numeraire': numeraire})

        return grid
