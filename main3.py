import jax as jx
import jax.numpy as np
import jax.scipy as sp
from functools import partial
import plotly.graph_objects as go
from flexpricer.engine import plot_lines


xs = np.linspace(-10, 10, 100)
ys = np.tanh(6 * xs)
fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=ys))
fig.show()

# if __name__ == '__main__':
#     fn = jx.grad(main)
#     print(fn(5.0))
