import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def model():
    ___shape = {}
    theta = pyro.sample('theta', ImproperUniform())
    pyro.sample('theta' + '1', dist.Normal(tensor(1000.0), tensor(1.0)), obs=theta)