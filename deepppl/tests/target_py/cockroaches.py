import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def transformed_data(N=None, exposure2=None, roach1=None, senior=None,
                     treatment=None, y=None):
    ___shape = {}
    ___shape['log_expo'] = N
    log_expo = zeros(___shape['log_expo'])
    log_expo = log(exposure2)
    return {'log_expo': log_expo}


def model(N=None, exposure2=None, roach1=None, senior=None, treatment=None,
          y=None, transformed_data=None):
    log_expo = transformed_data['log_expo']
    ___shape = {}
    ___shape['N'] = ()
    ___shape['exposure2'] = N
    ___shape['roach1'] = N
    ___shape['senior'] = N
    ___shape['treatment'] = N
    ___shape['y'] = N
    ___shape['beta'] = 4
    ___shape['lmbda'] = N
    ___shape['tau'] = ()
    beta = pyro.sample('beta', ImproperUniform(4))
    lmbda = pyro.sample('lmbda', ImproperUniform(N))
    tau = pyro.sample('tau', ImproperUniform())
    ___shape['sigma'] = ()
    sigma = 1.0 / sqrt(tau)
    pyro.sample('tau' + '__1', dist.Gamma(0.001, 0.001), obs=tau)
    pyro.sample('lmbda' + '__2', dist.Normal(0, sigma), obs=lmbda)
    pyro.sample('y' + '__3', poisson_log(log_expo + beta[1 - 1] + beta[2 - 1] *
                                       roach1 + beta[3 - 1] * treatment + beta[4 - 1] * senior + lmbda), obs=y
                )
    return {'tau': tau, 'lmbda': lmbda, 'sigma': sigma, 'beta': beta}
