from typing import Dict, List
import jax as jx
import jax.numpy as np
from dataclasses import dataclass, fields, asdict


@dataclass(init=False)
class Component:

    @classmethod
    def parameters(cls):
        return tuple(f.name for f in fields(cls))


@dataclass
class Option(Component):

    k: float
    t: float

    def payoff(self, spot: float):
        diff = spot - self.k
        return diff / (1 + np.exp(-12 * diff))

    def price(self, spot, numeraire):
        return np.mean(self.payoff(spot) / numeraire)


@dataclass
class BlackScholes(Component):

    s: float
    r: float
    q: float
    sig: float
    t: float

    def generate_grid(self, seed: int) -> List[Dict[str, np.ndarray]]:
        key = jx.random.PRNGKey(seed)
        innovation = jx.random.normal(key, (200000,), dtype=np.float32)
        drift = (self.r - self.q - 0.5 * self.sig ** 2) * self.t
        spot = self.s * np.exp(drift + self.sig * np.sqrt(self.t) * innovation)
        numeraire = np.exp(self.r * self.t)
        return [{'spot': spot, 'numeraire': numeraire}]


def unit_price(sensitive_params: Dict[str, float], fixed_params: Dict[str, float], seed: int):
    all_params = {**sensitive_params, **fixed_params}
    option = Option(**{k: all_params[k] for k in Option.parameters()})
    model = BlackScholes(**{k: all_params[k] for k in BlackScholes.parameters()})

    grid = model.generate_grid(seed)[0]
    price = option.price(grid['spot'], grid['numeraire'])
    return price


def main():
    s = 100.0
    k = 100.0
    r = 0.0
    q = 0.0
    sig = 0.2
    t = 0.25
    d_fn = jx.value_and_grad(unit_price)

    gamma_fn = lambda x: unit_price({'s': x, 'k': k, 'r': r, 'q': q, 'sig': sig, 't': t}, {}, 0)
    d2_fn = jx.grad(jx.grad(gamma_fn))
    print(d_fn({'s': s, 'sig': sig, 't': t}, {'k': k, 'r': r, 'q': q}, 0))
    print(d2_fn(s))


if __name__ == '__main__':
    main()
