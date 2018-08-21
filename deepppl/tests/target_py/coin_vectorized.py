import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def model(x):
    ___shape = {}
    ___shape['x'] = tensor(10)
    theta = pyro.sample('theta', dist.Uniform(tensor(0.0), tensor(1.0)))
    pyro.sample('theta' + '1', dist.Uniform(tensor(0.0), tensor(1.0)), obs=theta)
    pyro.sample('x' + '2', dist.Bernoulli(theta), obs=x)
