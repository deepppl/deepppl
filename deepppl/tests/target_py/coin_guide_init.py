import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def guide_(x):
    ___shape = {}
    ___shape['x'] = 10
    ___shape['theta'] = ()
    ___shape['alpha_q'] = ()
    ___shape['beta_q'] = ()
    alpha_q = pyro.param('alpha_q', tensor(15.0))
    beta_q = pyro.param('beta_q', tensor(15.0))
    theta = pyro.sample('theta', dist.Beta(alpha_q, beta_q))


def model(x):
    ___shape = {}
    ___shape['x'] = 10
    ___shape['theta'] = ()

    theta = pyro.sample('theta', dist.Uniform(0.0, 1.0))
    pyro.sample('theta' + '1', dist.Beta(10.0, 10.0), obs=theta
                )
    for i in range(1, 10 + 1):
        pyro.sample('x' + '{}'.format(i - 1) + '2', dist.Bernoulli(theta),
                    obs=x[i - 1])
