import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(N_mis=None, N_obs=None, y_obs=None):
    ___shape = {}
    ___shape['N_obs'] = ()
    ___shape['N_mis'] = ()
    ___shape['y_obs'] = N_obs
    ___shape['mu'] = ()
    ___shape['sigma'] = ()
    ___shape['y_mis'] = N_mis
    mu = pyro.sample('mu', ImproperUniform())
    sigma = pyro.sample('sigma', LowerConstrainedImproperUniform(0.0))
    y_mis = pyro.sample('y_mis', ImproperUniform(N_mis))
    pyro.sample('y_obs' + '__1', dist.Normal(mu, sigma), obs=y_obs)
    pyro.sample('y_mis' + '__2', dist.Normal(mu, sigma), obs=y_mis)
    return {'sigma': sigma, 'y_mis': y_mis, 'mu': mu}
