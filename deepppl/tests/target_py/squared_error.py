import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(K=None, N=None, x=None, y=None):
    ___shape = {}
    ___shape['N'] = ()
    ___shape['K'] = ()
    ___shape['y'] = N
    ___shape['x'] = N, K
    ___shape['beta'] = K
    beta = pyro.sample('beta', ImproperUniform(K))
    ___shape['squared_error'] = ()
    squared_error = dot_self(y - x * beta)
    pyro.sample('expr' + '__1', dist.Exponential(1.0), obs=--squared_error)


def generated_quantities(K=None, N=None, x=None, y=None):
    pyro.param('beta').item()
    pyro.param('squared_error').item()
    ___shape['sigma_squared'] = ()
    sigma_squared = squared_error / N
    return {'sigma_squared': sigma_squared}
