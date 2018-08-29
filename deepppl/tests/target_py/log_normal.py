import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def model():
    ___shape = {}
    theta = pyro.sample('theta', LowerConstrainedImproperUniform(tensor(0.0)))
    pyro.sample('expr' + '1', dist.Normal(log(tensor(10.0)), tensor(1.0)), obs=log(theta))
    pyro.sample('expr' + '2', dist.Exponential(tensor(1.0)), obs=--log(fabs(theta)))