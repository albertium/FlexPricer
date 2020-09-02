import jax as jx
import jax.numpy as np
import jax.scipy as sp
from functools import partial
import plotly.graph_objects as go
from flexpricer.engine import plot_lines
from flexpricer.analytical import get_phi_black_scholes, price_bs_call, price_call_with_phi

s = 100
k = 100
r = 0.0
q = 0.0
sig = 0.2
t = 0.25
phi = get_phi_black_scholes(s, r, q, sig, t)
print(price_bs_call(s, k, r, q, sig, t))
print(price_call_with_phi(phi, s, k))
