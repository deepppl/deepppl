import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(N: 'int'=None, x: 'int[N]'=None):
    z: 'real' = pyro.sample('z', dist.Uniform(0.0, 1.0))
    pyro.sample('z' + '__1', dist.Beta(1, 1), obs=z)
    pyro.sample('x' + '__2', dist.Bernoulli(z), obs=x)
