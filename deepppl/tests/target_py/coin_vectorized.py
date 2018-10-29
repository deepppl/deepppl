import torch
from torch import tensor, randn
import pyro
import pyro.distributions as dist


def model(x):
    ___shape = {}
    ___shape['x'] = 10
    ___shape['theta'] = ()
    theta = pyro.sample('theta', dist.Uniform(0.0, 1.0))
    pyro.sample('theta' + '1', dist.Uniform(0.0, 1.0), obs=theta)
    pyro.sample('x' + '2', dist.Bernoulli(theta), obs=x)
