import jax as jx
import jax.numpy as np
import jax.scipy as sp
from functools import partial
import plotly.graph_objects as go
from flexpricer.engine import plot_lines


def price(params):
    s = params['s']
    k = params['k']
    sig = params['sig']
    t = params['t']
    total_vol = sig * np.sqrt(t)
    d1 = np.log(s / k) / total_vol + 0.5 * total_vol
    d2 = d1 - total_vol
    return s * sp.stats.norm.cdf(d1) - k * sp.stats.norm.cdf(d2)


def generate_d2(params: dict, name1, name2, vector_name):
    fixed_params = {k: v for k, v in params.items() if k not in (name1, name2, vector_name)}

    def wrapper(var1, var2, vector_var):
        return price({**var1, **var2, **vector_var, **fixed_params})

    dict1 = {name1: params[name1]}
    dict2 = {name2: params[name2]}
    d_fn = partial(jx.vmap(jx.grad(wrapper), in_axes=(None, None, 0)), dict1, dict2)
    d2_fn = partial(jx.vmap(jx.grad(lambda *args: jx.grad(wrapper)(*args)[name1], argnums=(1,)), in_axes=(None, None, 0)), dict1, dict2)
    return d_fn, d2_fn


def main():
    s = 100.0
    k = 100.0
    sig = 0.2
    t = 0.25
    d_fn, d2_fn = generate_d2({'s': s, 'k': k, 'sig': sig, 't': t}, 's', 'sig', 'k')
    strikes = np.linspace(50, 150, 200)
    d1 = d_fn({'k': strikes})['s']
    d2 = d2_fn({'k': strikes})[0]['sig']
    plot_lines(strikes, [('delta', d1), ('gamma', d2)])


if __name__ == '__main__':
    main()
