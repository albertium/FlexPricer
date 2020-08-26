from typing import Tuple, Callable, Dict, Optional
from dataclasses import dataclass
import abc
import jax.numpy as np

from flexpricer.base_component import PricerComponent


# Dict is t's slice.
ForwardActionT = Callable[[Dict[str, np.ndarray]], Optional[float]]
# First dict is (t+1)'s slice and second dict is t's slice
BackwardActionT = Callable[[Dict[str, np.ndarray], Dict[str, np.ndarray]], Optional[float]]


@dataclass
class Instrument(PricerComponent, abc.ABC):

    @abc.abstractmethod
    def build_forward_events(self) -> Tuple[Tuple[float, ForwardActionT], ...]:
        """ Set up events for forward pass """

    @abc.abstractmethod
    def build_backward_events(self) -> Tuple[Tuple[float, BackwardActionT], ...]:
        """ Set up events for backward pass """
