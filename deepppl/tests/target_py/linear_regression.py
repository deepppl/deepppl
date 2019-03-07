import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(N: 'int'=None, x: 'real[N]'=None, y: 'real[N]'=None):
    alpha: 'real' = pyro.sample('alpha', ImproperUniform())
    beta: 'real' = pyro.sample('beta', ImproperUniform())
    sigma: 'real' = pyro.sample('sigma', LowerConstrainedImproperUniform(0.0))
    pyro.sample('y' + '__1', dist.Normal(alpha + beta * x, sigma), obs=y)
