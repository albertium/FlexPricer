"""
Vanilla European option
"""
from typing import Tuple
from dataclasses import dataclass, field
import jax.numpy as np

from flexpricer.instrument.base_instrument import Instrument, ForwardActionT, BackwardActionT


@dataclass
class VanillaEuropean(Instrument):

    # Instrument parameters
    strike: float
    expiration: float

    # Private variables
    _price: float = field(init=False, repr=False)

    def payoff(self, spot: np.ndarray) -> None:
        diff = spot - self.strike
        self._price = np.mean(diff * (np.tanh(6 * diff) + 1)) / 2

    def build_forward_events(self) -> Tuple[Tuple[float, ForwardActionT], ...]:
        return ((self.expiration, lambda variables: self.payoff(variables['spot'])),)

    def build_backward_events(self) -> Tuple[Tuple[float, BackwardActionT], ...]:
        return ((0, lambda prev, curr: self._price / prev['numeraire'] * curr['numeraire']),)
