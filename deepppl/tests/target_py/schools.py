import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(J=None, sigma=None, y=None):
    ___shape = {}
    ___shape['J'] = ()
    ___shape['y'] = J
    ___shape['sigma'] = J
    ___shape['mu'] = 1
    ___shape['tau'] = 1
    ___shape['eta'] = J
    mu = pyro.sample('mu', ImproperUniform(1))
    tau = pyro.sample('tau', LowerConstrainedImproperUniform(0.0, 1))
    eta = pyro.sample('eta', ImproperUniform(J))
    ___shape['theta'] = J
    theta = zeros(___shape['theta'])
    for j in range(1, J + 1):
        theta[j - 1] = mu + tau * eta[j - 1]
    pyro.sample('eta' + '__1', dist.Normal(zeros(J), ones(J)), obs=eta)
    pyro.sample('y' + '__2', dist.Normal(theta, sigma), obs=y)
    return {'theta': theta, 'tau': tau, 'mu': mu, 'eta': eta}
