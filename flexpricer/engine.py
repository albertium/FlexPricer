from typing import Dict, Type, List, Tuple, Callable
import jax.numpy as np
import jax as jx
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from flexpricer.model import Model
from flexpricer.instrument import Instrument


class Pricer:

    def __init__(self, model: Type[Model], instrument: Type[Instrument], num_paths: int = 100000) -> None:
        self.model_class = model
        self.instr_class = instrument
        self.num_paths = num_paths

    def unit_price(self, params: Dict[str, float], seed: int) -> float:
        # noinspection PyArgumentList
        model = self.model_class(**{k: params[k] for k in self.model_class.parameters()})
        # noinspection PyArgumentList
        instrument = self.instr_class(**{k: params[k] for k in self.instr_class.parameters()})

        # Retrieve events and initialize model
        forward_events = instrument.build_forward_events()
        backward_events = instrument.build_backward_events()
        model.initialize(forward_events)

        # Generate paths
        grids = model.populate_grids(self.num_paths, seed)
        last = len(grids) - 1

        # Forward pass
        for idx, (time_point, act) in enumerate(forward_events):
            this_slice = grids[idx]
            assert this_slice['time'] == time_point
            act(this_slice)

        # Backward pass
        for idx, (time_point, act) in enumerate(backward_events[:-1]):
            this_slice = grids[last - idx - 1]
            assert this_slice['time'] == time_point
            act(grids[last - idx], this_slice)

        _, payoff = backward_events[-1]
        price = payoff(grids[0], {'numeraire': np.array(1.0)})
        assert price is not None
        return price

    def generate_d1_fn(self, params: Dict[str, float], names: List[str], vector_name: str, seed: int) -> Callable:

        fixed_params = {k: v for k, v in params.items() if k not in names and k != vector_name}

        def wrapper(sensitives: Dict[str, float], vector_var: Dict[str, float]) -> float:
            return self.unit_price({**sensitives, **vector_var, **fixed_params}, seed)

        sensitive_params = {name: params[name] for name in names}
        return lambda x: jx.vmap(jx.value_and_grad(wrapper), in_axes=(None, 0))(sensitive_params, {vector_name: x})

    def generate_d2_fn(self, params: Dict[str, float], name1: str, name2: str, vector_name: str, seed: int) -> Callable:

        fixed_params = {k: v for k, v in params.items() if k not in (name1, name2, vector_name)}

        def wrapper(var1: Dict[str, float], var2: Dict[str, float], vector_var: Dict[str, float]) -> float:
            return self.unit_price({**var1, **var2, **vector_var, **fixed_params}, seed)

        if name1 == name2:
            idx = 0
            dict1 = {name1: params[name1]}
            dict2 = {}
        else:
            idx = 1
            dict1 = {name1: params[name1]}
            dict2 = {name2: params[name2]}

        v_fn = jx.vmap(jx.grad(lambda *args: jx.grad(wrapper)(*args)[name1], argnums=(idx,)), in_axes=(None, None, 0))
        return lambda x: v_fn(dict1, dict2, {vector_name: x})[0][name2]

    def profile_risk(self, params: Dict[str, float], seed: int) -> None:
        start = time.time()
        strikes = np.linspace(50, 150, 200)

        # 1st order greeks
        d_fn = self.generate_d1_fn(params, ['spot', 'volatility', 'expiration'], 'strike', seed)
        price, greek1 = d_fn(strikes)

        # 2nd order greeks
        gamma = self.generate_d2_fn(params, 'spot', 'spot', 'strike', seed)(strikes)
        volga = self.generate_d2_fn(params, 'volatility', 'volatility', 'strike', seed)(strikes)
        vanna = self.generate_d2_fn(params, 'spot', 'volatility', 'strike', seed)(strikes)
        vanna.block_until_ready()

        print(f'Evaluation takes {time.time() - start:.2f}s')

        plots = [
            ('price', price),
            ('delta', greek1['spot']),
            ('gamma', gamma),
            ('vega', greek1['volatility']),
            ('volga', volga),
            ('vanna', vanna),
        ]
        plot_lines(strikes, plots, num_cols=2)


def plot_lines(axis: np.ndarray, plots: List[Tuple[str, np.ndarray]], num_cols: int = 1):
    num_rows = int(np.ceil(len(plots) / num_cols))
    titles = [plot[0] for plot in plots]
    fig = make_subplots(num_rows, num_cols, subplot_titles=titles)
    for idx, (name, data) in enumerate(plots):
        fig.add_trace(go.Scatter(x=axis, y=data, name=name), row=idx // num_cols + 1, col=idx % num_cols + 1)
    fig.update_layout(height=1000, width=1000)
    fig.show()
