import torch
from torch import tensor, randn
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model():
    ___shape = {}
    ___shape['theta'] = ()
    theta = pyro.sample('theta', ImproperUniform())
    pyro.sample('expr' + '1',
                dist.Exponential(1.0),
                obs=-(-0.5 * (theta - 1000.0) * (theta - 1000.0)))
