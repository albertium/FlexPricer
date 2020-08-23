import jax.numpy as np
import jax.scipy as sp
import jax as jx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


@jx.jit
def price_bs(s: float, k: float, r: float, q: float, sig: float, t: float) -> float:
    total_vol = sig * np.sqrt(t)
    d1 = (np.log(s / k) + (r - q) * t) / total_vol + 0.5 * total_vol
    d2 = d1 - total_vol
    return s * np.exp(-q * t) * sp.stats.norm.cdf(d1) - k * np.exp(-r * t) * sp.stats.norm.cdf(d2)


def main():
    s = 100.0
    k = 100.0
    r = 0.0
    q = 0.0
    sig = 0.2
    t = 0.5
    jacob_fn = jx.vmap(jx.value_and_grad(price_bs, (0, 4, 5)), in_axes=(None, 0, None, None, None, None))
    gamma_fn = jx.vmap(jx.grad(jx.grad(price_bs, 0), 0), in_axes=(None, 0, None, None, None, None))
    volga_fn = jx.vmap(jx.grad(jx.grad(price_bs, 4), 4), in_axes=(None, 0, None, None, None, None))

    strikes = np.linspace(50, 150, 200)
    prices, (deltas, vegas, rhos) = jacob_fn(s, strikes, r, q, sig, t)
    gammas = gamma_fn(s, strikes, r, q, sig, t)
    volgas = volga_fn(s, strikes, r, q, sig, t)
    # panel = pd.DataFrame({'strike': strikes, 'price': prices, 'delta': deltas, 'gamma': gammas, 'vega': vegas})
    fig = make_subplots(3, 2)
    # fig = go.Figure()
    for idx, data in enumerate([prices, deltas, gammas, vegas, volgas]):
        fig.add_trace(go.Scatter(x=strikes, y=data), row=idx // 2 + 1, col=idx % 2 + 1)
        # fig.add_trace(go.Scatter(x=strikes, y=data, yaxis='y' if row == 0 else f'y{row}'))
    fig.update_layout(height=1000, width=1000)
    fig.show()


if __name__ == '__main__':
    main()