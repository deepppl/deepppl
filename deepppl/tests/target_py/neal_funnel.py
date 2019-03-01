import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model():
    ___shape = {}
    ___shape['y_std'] = ()
    ___shape['x_std'] = ()
    y_std = pyro.sample('y_std', ImproperUniform())
    x_std = pyro.sample('x_std', ImproperUniform())
    ___shape['y'] = ()
    y = 3.0 * y_std
    ___shape['x'] = ()
    x = exp(y / 2) * x_std
    pyro.sample('y_std' + '__1', dist.Normal(0, 1), obs=y_std)
    pyro.sample('x_std' + '__2', dist.Normal(0, 1), obs=x_std)

def generated_quantities(parameters=None):
    x_std = parameters.x_std
    y_std = parameters.y_std
    ___shape['y'] = ()
    y = 3.0 * y_std
    ___shape['x'] = ()
    x = exp(y / 2) * x_std
    return {'y': y, 'x': x}
