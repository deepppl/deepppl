import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def model(x):
    ___shape = {}
    ___shape['x'] = tensor(10)
    theta = pyro.sample('theta', dist.Uniform(tensor(0.0), tensor(1.0)))
    pyro.sample('theta', dist.Beta(tensor(10.0), tensor(10.0)), obs=theta)
    pyro.sample('x', dist.Bernoulli(theta), obs=x)