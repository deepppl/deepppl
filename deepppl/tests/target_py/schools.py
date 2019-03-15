import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(J=None, sigma=None, y=None):
    mu = pyro.sample('mu', ImproperUniform(shape=1))
    tau = pyro.sample('tau', LowerConstrainedImproperUniform(zeros(1), shape=1)
        )
    eta = pyro.sample('eta', ImproperUniform(shape=J))
    theta = zeros(J)
    for j in range(1, J + 1):
        theta[j - 1] = mu[0 - 1] + tau[0 - 1] * eta[j - 1]
    pyro.sample('eta' + '__1', dist.Normal(zeros(J), ones(J)), obs=eta)
    pyro.sample('y' + '__2', dist.Normal(theta, sigma), obs=y)


def generated_quantities(J=None, sigma=None, y=None, parameters=None):
    eta = parameters['eta']
    mu = parameters['mu']
    tau = parameters['tau']
    theta = zeros(J)
    for j in range(1, J + 1):
        theta[j - 1] = mu[0 - 1] + tau[0 - 1] * eta[j - 1]
    return {'theta': theta}
