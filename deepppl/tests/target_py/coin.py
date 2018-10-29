import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def model(x):
    ___shape = {}
    ___shape['x'] = 10
    theta = pyro.sample('theta', dist.Uniform(0.0, 1.0))
    pyro.sample('theta' + '1', dist.Uniform(0, 1), obs=theta)
    for i in range(1, 10 + 1):
        pyro.sample('x' + '{}'.format(i - 1) + '2', dist.Bernoulli(theta), obs=x[i - 1])
