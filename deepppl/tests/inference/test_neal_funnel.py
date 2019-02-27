import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist
from deepppl.utils.utils import *
from torch import exp


def model2():
    ___shape = {}
    ___shape['y_std'] = ()
    ___shape['x_std'] = ()
    y_std = pyro.sample('y_std', ImproperUniform())
    x_std = pyro.sample('x_std', ImproperUniform())
    ___shape['y'] = ()
    y = 3.0 * y_std
    ___shape['x'] = ()
    x = exp(y / 2) * x_std
    pyro.sample('y_std' + '1', dist.Normal(0, 1), obs=y_std)
    pyro.sample('x_std' + '2', dist.Normal(0, 1), obs=x_std)
    return (x, y)


import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor
import pyro
from pyro import distributions as dist
import numpy as np
from pyro.infer import mcmc
from pyro.infer.mcmc import MCMC, NUTS
import logging
import time
import pystan
from deepppl.utils.utils import ImproperUniform, LowerConstrainedImproperUniform
import matplotlib.pyplot as plt

import deepppl
import os
import pandas as pd

stan_model_file = 'deepppl/tests/good/neal_funnel.stan'
global_num_iterations = 1000
global_num_chains = 1
global_warmup_steps = 100


def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model)
    return mcmc.MCMC(nuts_kernel, **kwargs)


def test_neals_funnel():
    model = deepppl.DppplModel(model_file=stan_model_file)
    model._model = model2
    posterior = model.posterior(
        method=nuts,
        num_samples=global_num_iterations-global_warmup_steps,
        warmup_steps=global_warmup_steps)

    marginal = pyro.infer.EmpiricalMarginal(
        posterior.run(), sites=['x_std', 'y_std'])

    xy = [model._model() for _ in range(100)]
    x = [xi.item() for (xi, yi) in xy]
    y = [yi.item() for (xi, yi) in xy]

    # samples_fstan = [marginal() for _ in range(100)]
    # o = model.generated_quantities()
    # print(samples_fstan)
    # x_std_pystan, y_std_pystan = compare_with_stan_output()
    print(y)
    plt.scatter(x, y)
    plt.plot()
    plt.savefig('deepstan_neals_funnel.jpg')
    plt.close()


def compare_with_stan_output():
    stan_code = open(stan_model_file).read()
    t1 = time.time()

    # Compile and fit
    sm1 = pystan.StanModel(model_code=str(stan_code))
    fit_stan = sm1.sampling(iter=global_num_iterations, chains=global_num_chains,
                            warmup=global_warmup_steps)

    t2 = time.time()
    x_std = fit_stan.extract(permuted=True)['x_std']
    y_std = fit_stan.extract(permuted=True)['y_std']
    return x_std, y_std


if __name__ == "__main__":
    test_neals_funnel()
