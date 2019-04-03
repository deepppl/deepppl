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
global_num_iterations=3000
global_num_chains=1
global_warmup_steps = 300

def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model)
    return mcmc.MCMC(nuts_kernel, **kwargs)

@pytest.mark.xfail(strict=False, reason="There is a call to marginal that does not resolve.  Not sure what it is supposed to do.")
def test_neals_funnel():
    model = deepppl.DppplModel(model_file=stan_model_file)
    # model._model = model2
    posterior = model.posterior(
        method=nuts,
        num_samples=global_num_iterations-global_warmup_steps,
        warmup_steps=global_warmup_steps)


    # xy = [model._model() for _ in range(100)]
    # x = [xi.item() for (xi, yi) in xy]
    # y = [yi.item() for (xi, yi) in xy]

    def params_of_sample(s):
        return { 'x_std':s[0], 'y_std': s[1] }
    samples_fstan = [marginal() for _ in range(100)]
    xy = [ model.generated_quantities(parameters=params_of_sample(sample)) for sample in samples_fstan ]
    x = [ o['x'].item() for o in xy ]
    y = [ o['y'].item() for o in xy ]

    plt.scatter(x, y)
    plt.plot()
    plt.savefig('deepstan_neals_funnel.jpg')
    plt.close()

    x_std_pystan, y_std_pystan = compare_with_stan_output()
    assert len(x_std_pystan) == len(x)
    plt.scatter(x_std_pystan, y_std_pystan)
    plt.plot()
    plt.savefig('pystan_neals_funnel.jpg')
    plt.close()

    from scipy.stats import entropy
    hist1 = np.histogram(x, bins = 10)
    hist2 = np.histogram(x_std_pystan, bins = hist1[1])
    kl = entropy(hist1[0]+1, hist2[0]+1)
    skl = kl + entropy(hist2[0]+1, hist1[0]+1)
    print('kl for x is:{}'.format(skl))

    hist1 = np.histogram(y, bins = 10)
    hist2 = np.histogram(y_std_pystan, bins = hist1[1])
    skl = kl + entropy(hist2[0]+1, hist1[0]+1)
    print('kl for y is:{}'.format(skl))

def compare_with_stan_output():
    stan_code = open(stan_model_file).read()

    # Compile and fit
    sm1 = pystan.StanModel(model_code=str(stan_code))
    print("Compilation done")
    t1 = time.time()

    fit_stan = sm1.sampling(iter=global_num_iterations, chains=global_num_chains, 
                            warmup = global_warmup_steps)

    t2 = time.time()
    x = fit_stan.extract(permuted=True)['x']
    y = fit_stan.extract(permuted=True)['y']
    return x, y

if __name__ == "__main__":
    test_neals_funnel()
