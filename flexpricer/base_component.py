from typing import Tuple
from dataclasses import dataclass, fields


@dataclass
class PricerComponent:
    """ Base class that can be plugged into pricer. Both model and instrument subclass this """

    @classmethod
    def parameters(cls) -> Tuple[str, ...]:
        return tuple(f.name for f in fields(cls) if f.init)
