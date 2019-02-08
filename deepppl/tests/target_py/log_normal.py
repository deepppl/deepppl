import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model():
    ___shape = {}
    ___shape['theta'] = ()
    theta = pyro.sample('theta', LowerConstrainedImproperUniform(0.0))
    pyro.sample('expr' + '1', dist.Normal(log(10.0), 1.0), obs=log(theta))
    pyro.sample('expr' + '2', dist.Exponential(1.0), obs=--log(fabs(theta)))
