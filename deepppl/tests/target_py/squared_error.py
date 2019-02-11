import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(K, N, x, y):
    ___shape = {}
    ___shape['N'] = ()
    ___shape['K'] = ()
    ___shape['y'] = ()
    ___shape['x'] = ()
    ___shape['beta'] = ()
    beta = pyro.sample('beta', ImproperUniform())
    ___shape['squared_error'] = ()
    squared_error = dot_self(y - x * beta)
    pyro.sample('expr' + '1', dist.Exponential(1.0), obs=--squared_error)


def generated_quantities(K, N, x, y):
    pyro.param('beta').item()
    pyro.param('squared_error').item()
    ___shape['sigma_squared'] = ()
    sigma_squared = squared_error / N
    return {'sigma_squared': sigma_squared}
