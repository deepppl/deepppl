import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def transformed_data(I=None, N=None, n=None, x1=None, x2=None):
    ___shape = {}
    ___shape['x1x2'] = I
    x1x2 = zeros(___shape['x1x2'])
    x1x2 = x1 * x2
    return {'x1x2': x1x2}


def model(I=None, N=None, n=None, x1=None, x2=None, transformed_data=None):
    x1x2 = transformed_data['x1x2']
    ___shape = {}
    ___shape['I'] = ()
    ___shape['n'] = I
    ___shape['N'] = I
    ___shape['x1'] = I
    ___shape['x2'] = I
    ___shape['alpha0'] = ()
    ___shape['alpha1'] = ()
    ___shape['alpha12'] = ()
    ___shape['alpha2'] = ()
    ___shape['tau'] = ()
    ___shape['b'] = I
    alpha0 = pyro.sample('alpha0', ImproperUniform())
    alpha1 = pyro.sample('alpha1', ImproperUniform())
    alpha12 = pyro.sample('alpha12', ImproperUniform())
    alpha2 = pyro.sample('alpha2', ImproperUniform())
    tau = pyro.sample('tau', ImproperUniform())
    b = pyro.sample('b', ImproperUniform(I))
    ___shape['sigma'] = ()
    sigma = 1.0 / sqrt(tau)
    pyro.sample('alpha0' + '__1', dist.Normal(0.0, 1000), obs=alpha0)
    pyro.sample('alpha1' + '__2', dist.Normal(0.0, 1000), obs=alpha1)
    pyro.sample('alpha2' + '__3', dist.Normal(0.0, 1000), obs=alpha2)
    pyro.sample('alpha12' + '__4', dist.Normal(0.0, 1000), obs=alpha12)
    pyro.sample('tau' + '__5', dist.Gamma(0.001, 0.001), obs=tau)
    pyro.sample('b' + '__6', dist.Normal(0.0, sigma), obs=b)
    pyro.sample('n' + '__7', binomial_logit(N, alpha0 + alpha1 * x1 + alpha2 *
                                            x2 + alpha12 * x1x2 + b), obs=n)

def generated_quantities(I=None, N=None, n=None, x1=None, x2=None,
    transformed_data=None, parameters=None):
    alpha0 = parameters.alpha0
    alpha1 = parameters.alpha1
    alpha12 = parameters.alpha12
    alpha2 = parameters.alpha2
    b = parameters.b
    tau = parameters.tau
    ___shape['sigma'] = ()
    sigma = 1.0 / sqrt(tau)
    return {'sigma': sigma}
