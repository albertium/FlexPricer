from flexpricer.model import BlackScholes, ArithmeticBlackScholes
from flexpricer.instrument import Vanilla, Digital
from flexpricer.engine import Pricer

from analytical import price_bs

s = 100.0
k = 100.0
r = 0.0
q = 0.0
sig = 0.3
t = 0.25

pricer = Pricer(ArithmeticBlackScholes, Digital, num_paths=1000000)
# print(pricer.unit_price({'spot': s, 'volatility': sig, 'expiration': t}, {'strike': k, 'rate': r, 'dividend': q}, 0))
pricer.profile_risk({'spot': s, 'volatility': sig, 'expiration': t, 'strike': k, 'rate': r, 'dividend': q, 'smooth': 2.5}, 0)
