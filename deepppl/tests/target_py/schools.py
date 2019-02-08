import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(J, sigma, y):
    ___shape = {}
    ___shape['J'] = ()
    ___shape['y'] = J
    ___shape['sigma'] = J
    ___shape['mu'] = ()
    ___shape['tau'] = ()
    ___shape['eta'] = J
    ___shape['theta'] = J
    theta = zeros(___shape['theta'])
    for j in range(1, J + 1):
        theta[j - 1] = mu + tau * eta[j - 1]
    mu = pyro.sample('mu', ImproperUniform())
    tau = pyro.sample('tau', LowerConstrainedImproperUniform(0.0))
    eta = pyro.sample('eta', ImproperUniform())
    pyro.sample('eta' + '1', dist.Normal(0, 1), obs=eta)
    pyro.sample('y' + '2', dist.Normal(theta, sigma), obs=y)
