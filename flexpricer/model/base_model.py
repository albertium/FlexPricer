from typing import Tuple, Dict, Callable, List
from dataclasses import dataclass, field
import abc

import jax.numpy as np
import jax as jx

from flexpricer.base_component import PricerComponent


@dataclass
class Model(PricerComponent, abc.ABC):

    _schedule: Tuple[float] = field(init=False, repr=False)
    _instrument_indices: Tuple[int] = field(init=False, repr=False)

    @property
    def schedule(self) -> Tuple[float]:
        return self._schedule

    def initialize(self, events: Tuple[Tuple[float, Callable], ...]) -> None:
        instrument_schedule = tuple(event[0] for event in events)
        expiration = events[-1][0]
        self._schedule = tuple(sorted(instrument_schedule + self._get_required_schedule(expiration)))

        # Figure out instrument schedule indices
        self._instrument_indices = tuple(self.schedule.index(time_point) for time_point in instrument_schedule)

    def populate_grids(self, num_paths: int, seed: int) -> List[Dict[str, np.ndarray]]:
        # Prepare dt
        dts = (self.schedule[0],) + tuple(curr - prev for curr, prev in zip(self.schedule[1:], self.schedule[:-1]))

        # Prepare innovations
        key = jx.random.PRNGKey(seed)
        innovations = jx.random.normal(key, (len(dts), num_paths), dtype=np.float32)

        # Simulate and organize data
        slices = self._generate_slices(dts, innovations)
        grids = []
        for idx in self._instrument_indices:
            slices[idx]['time'] = self.schedule[idx]
            grids.append(slices[idx])
        return grids

    @abc.abstractmethod
    def _get_required_schedule(self, expiration) -> Tuple[float]:
        """ Return required grid given expiration """

    @abc.abstractmethod
    def _generate_slices(self, dts: Tuple[float], innovations: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """ Generate paths for all variables. Convention is that we include the t=0 slice """
