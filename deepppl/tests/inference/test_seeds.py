I = 21
n = [10, 23, 23, 26, 17, 5, 53, 55, 32, 46, 10, 8, 10, 8, 23, 0, 3, 22, 15, 32, 3]
N = [39, 62, 81, 51, 39, 6, 74, 72, 51, 79, 13, 16, 30, 28, 45, 4, 12, 41, 30, 51, 7]
x1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
x2 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

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
import deepppl
import os
import pandas as pd
from torch import zeros, ones, sqrt
from deepppl.utils.utils import *

def transformed_data(I=None, N=None, n=None, x1=None, x2=None):
    ___shape = {}
    ___shape['x1x2'] = I
    x1x2 = zeros(___shape['x1x2'])
    x1x2 = x1 * x2
    return {'x1x2': x1x2}

def model2(I=None, N=None, n=None, x1=None, x2=None, transformed_data=None):
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
    pyro.sample('alpha0' + '1', dist.Normal(0.0, 1000), obs=alpha0)
    pyro.sample('alpha1' + '1', dist.Normal(0.0, 1000), obs=alpha1)
    pyro.sample('alpha2' + '3', dist.Normal(0.0, 1000), obs=alpha2)
    pyro.sample('alpha12' + '4', dist.Normal(0.0, 1000), obs=alpha12)
    pyro.sample('tau' + '5', dist.Gamma(0.001, 0.001), obs=tau)
    pyro.sample('b' + '6', dist.Normal(0.0, sigma), obs=b)
    pyro.sample('n' + '7', binomial_logit(N, alpha0 + alpha1 * x1 + alpha2 *
                                          x2 + alpha12 * x1x2 + b), obs=n)

stan_model_file = 'deepppl/tests/good/seeds.stan'
global_num_iterations=1000
global_num_chains=1
global_warmup_steps = 100

def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model)
    return mcmc.MCMC(nuts_kernel, **kwargs)

def test_seeds():
    model = deepppl.DppplModel(model_file=stan_model_file)
    model._model = model2
    posterior = model.posterior(
        method=nuts,
        num_samples=global_num_iterations-global_warmup_steps,
        warmup_steps=global_warmup_steps)

    local_n = torch.Tensor(n)
    local_N = torch.Tensor(N)
    local_x1 = torch.Tensor(x1)
    local_x2 = torch.Tensor(x2)

    marginal = pyro.infer.EmpiricalMarginal(posterior.run(I = I, n = local_n,
    N = local_N, x1 = local_x1, x2 = local_x2, transformed_data = 
    transformed_data(I = I, n = local_n, N = local_N, x1 = local_x1, x2 = local_x2)), 
        sites=['alpha0', 'alpha1', 'alpha12', 'alpha2', 'tau'])

    samples_fstan = [marginal() for _ in range(global_num_iterations-global_warmup_steps)]
    stack_samples = torch.stack(samples_fstan).numpy()

    params = ['alpha0', 'alpha1', 'alpha12', 'alpha2', 'tau']
    df = pd.DataFrame(stack_samples, columns=params)

    pystan_output, pystan_time = compare_with_stan_output()

    assert df.shape == pystan_output.shape
    from scipy.stats import entropy
    for column in params:
        #import pdb;pdb.set_trace()
        hist1 = np.histogram(df[column], bins = 10)
        hist2 = np.histogram(pystan_output[column], bins = hist1[1])
        kl = entropy(hist1[0]+1, hist2[0]+1)
        skl = kl + entropy(hist2[0]+1, hist1[0]+1)
        print('skl for column:{} is:{}'.format(column, skl))

def compare_with_stan_output():
    stan_code = open(stan_model_file).read()
    stan_code = """
        data {
            int I;
            int n[I];
            int N[I];
            vector[I] x1;
            vector[I] x2;
        }
        transformed data {
            vector[I] x1x2;
            x1x2 = x1 .* x2;
        }
        parameters {
            real alpha0;
            real alpha1;
            real alpha12;
            real alpha2;
            real tau;
            vector[I] b;
        }
        transformed parameters {
            real sigma = 1.0 / sqrt(tau);
        }
        model {
            alpha0 ~ normal(0.0,1000);
            alpha1 ~ normal(0.0,1000);
            alpha2 ~ normal(0.0,1000);
            alpha12 ~ normal(0.0,1000);
            tau ~ gamma(0.001,0.001);

            b ~ normal(0.0, sigma);
            n ~ binomial_logit (N, alpha0
                    + alpha1 * x1
                    + alpha2 * x2
                    + alpha12 * x1x2 + b);
        }

    """
    data = {
        'I': I, 
        'n': n, 
        'N': N, 
        'x1': x1,
        'x2': x2
    }

    # Compile and fit
    sm1 = pystan.StanModel(model_code=str(stan_code))
    print("Compilation done")
    t1 = time.time()

    fit_stan = sm1.sampling(data = data, iter=global_num_iterations, chains=global_num_chains, 
                            warmup = global_warmup_steps)

    t2 = time.time()
    alpha0 = fit_stan.extract(permuted=True)['alpha0']
    alpha1 = fit_stan.extract(permuted=True)['alpha1']
    alpha12 = fit_stan.extract(permuted=True)['alpha12']
    alpha2 = fit_stan.extract(permuted=True)['alpha2']
    tau = fit_stan.extract(permuted=True)['tau']

    alpha0 = np.reshape(alpha0, (alpha0.shape[0], 1))
    alpha1 = np.reshape(alpha1, (alpha1.shape[0], 1))
    alpha12 = np.reshape(alpha12, (alpha12.shape[0], 1))
    alpha2 = np.reshape(alpha2, (alpha2.shape[0], 1))
    tau = np.reshape(tau, (tau.shape[0], 1))
    marginal = np.concatenate((alpha0, alpha1, alpha12, alpha2, tau), axis = 1)
    params = ['alpha0', 'alpha1', 'alpha12', 'alpha2', 'tau']
    df = pd.DataFrame(marginal, columns=params)

    t2 = time.time()
    return df, t2-t1


if __name__ == "__main__":
    test_seeds()
