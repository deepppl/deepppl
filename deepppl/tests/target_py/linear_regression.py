import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(N=None, x=None, y=None):
    ___shape = {}
    ___shape['N'] = ()
    ___shape['x'] = N
    ___shape['y'] = N
    ___shape['alpha'] = ()
    ___shape['beta'] = ()
    ___shape['sigma'] = ()
    alpha = pyro.sample('alpha', ImproperUniform())
    beta = pyro.sample('beta', ImproperUniform())
    sigma = pyro.sample('sigma', LowerConstrainedImproperUniform(0.0))
    pyro.sample('y' + '__1', dist.Normal(alpha + beta * x, sigma), obs=y)
