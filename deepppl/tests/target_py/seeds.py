import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def transformed_data(I: 'int'=None, N: 'int[I]'=None, n: 'int[I]'=None, x1:
    'real[I]'=None, x2: 'real[I]'=None):
    x1x2: 'real[I]' = zeros(I)
    x1x2: 'real[I]' = x1 * x2
    return {'x1x2': x1x2}


def model(I: 'int'=None, N: 'int[I]'=None, n: 'int[I]'=None, x1: 'real[I]'=
    None, x2: 'real[I]'=None, transformed_data=None):
    x1x2 = transformed_data['x1x2']
    alpha0: 'real' = pyro.sample('alpha0', ImproperUniform())
    alpha1: 'real' = pyro.sample('alpha1', ImproperUniform())
    alpha12: 'real' = pyro.sample('alpha12', ImproperUniform())
    alpha2: 'real' = pyro.sample('alpha2', ImproperUniform())
    tau: 'real' = pyro.sample('tau', ImproperUniform())
    b: 'real[I]' = pyro.sample('b', ImproperUniform(shape=I))
    sigma: 'real' = 1.0 / sqrt(tau)
    pyro.sample('alpha0' + '__1', dist.Normal(0.0, 1000), obs=alpha0)
    pyro.sample('alpha1' + '__2', dist.Normal(0.0, 1000), obs=alpha1)
    pyro.sample('alpha2' + '__3', dist.Normal(0.0, 1000), obs=alpha2)
    pyro.sample('alpha12' + '__4', dist.Normal(0.0, 1000), obs=alpha12)
    pyro.sample('tau' + '__5', dist.Gamma(0.001, 0.001), obs=tau)
    pyro.sample('b' + '__6', dist.Normal(zeros(I), sigma), obs=b)
    pyro.sample('n' + '__7', binomial_logit(N, alpha0 + alpha1 * x1 +
        alpha2 * x2 + alpha12 * x1x2 + b), obs=n)


def generated_quantities(I: 'int'=None, N: 'int[I]'=None, n: 'int[I]'=None,
    x1: 'real[I]'=None, x2: 'real[I]'=None, transformed_data=None,
    parameters=None):
    x1x2 = transformed_data['x1x2']
    alpha0 = parameters['alpha0']
    alpha1 = parameters['alpha1']
    alpha12 = parameters['alpha12']
    alpha2 = parameters['alpha2']
    b = parameters['b']
    tau = parameters['tau']
    sigma: 'real' = 1.0 / sqrt(tau)
    return {'sigma': sigma}
