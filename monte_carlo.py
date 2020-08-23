import jax.numpy as np
import jax as jx
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@jx.jit
def price_bs(s: float, k: float, r: float, q: float, sig: float, t: float, innovations: np.ndarray) -> float:
    n, m = innovations.shape
    dt = t / n
    total_vol = sig * np.sqrt(dt)
    var = sig ** 2
    spots = s * np.ones(m)
    for innovation in innovations:
        spots *= np.exp(-0.5 * var * dt + total_vol * innovation)

    return np.mean(np.maximum(spots - k, 0))


def main():
    s = 100.0
    k = 100.0
    r = 0.0
    q = 0.0
    sig = 0.2
    t = 0.5
    n, m = 15, 100000
    key = jx.random.PRNGKey(0)
    innovations = jx.random.normal(key, (n, m), dtype=np.float32)
    strikes = np.linspace(50, 150, 200)

    start = time.time()
    jacob_fn = jx.vmap(jx.value_and_grad(price_bs, (0, 4)), in_axes=(None, 0, None, None, None, None, None))
    prices, (deltas, vegas) = jacob_fn(s, strikes, r, q, sig, t, innovations)
    print(f'It takes {time.time() - start:.2f}s')

    fig = make_subplots(3, 2)
    # fig = go.Figure()
    for idx, data in enumerate([prices, deltas, vegas]):
        fig.add_trace(go.Scatter(x=strikes, y=data), row=idx // 2 + 1, col=idx % 2 + 1)
        # fig.add_trace(go.Scatter(x=strikes, y=data, yaxis='y' if row == 0 else f'y{row}'))
    fig.update_layout(height=1000, width=1000)
    fig.show()


if __name__ == '__main__':
    main()
