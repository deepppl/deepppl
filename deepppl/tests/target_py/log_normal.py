import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def model():
    ___shape = {}
    theta = pyro.sample('theta', LowerConstrainedImproperUniform(0.0))
    pyro.sample('expr' + '1', dist.Normal(log(10.0), 1.0), obs=log(theta))
    pyro.sample('expr' + '2', dist.Exponential(1.0), obs=--log(fabs(theta)))