import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(K=None, N=None, x=None, y=None):
    ___shape = {}
    ___shape['K'] = ()
    ___shape['N'] = ()
    ___shape['x'] = ()
    ___shape['y'] = ()
    ___shape['beta'] = ()
    beta = pyro.sample('beta', ImproperUniform())
    pyro.sample('y' + '1', dist.Normal(x * beta, 1), obs=y)
