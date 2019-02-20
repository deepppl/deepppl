import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(K=None, N=None, x=None, y=None):
    ___shape = {}
    ___shape['N'] = ()
    ___shape['K'] = ()
    ___shape['x'] = ()
    ___shape['y'] = ()
    ___shape['alpha'] = ()
    ___shape['beta'] = ()
    ___shape['sigma'] = ()
    alpha = pyro.sample('alpha', ImproperUniform())
    beta = pyro.sample('beta', ImproperUniform())
    sigma = pyro.sample('sigma', LowerConstrainedImproperUniform(0.0))
    pyro.sample('y' + '1', dist.Normal(x * beta + alpha, sigma), obs=y)
