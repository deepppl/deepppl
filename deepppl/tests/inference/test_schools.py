import torch
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
from deepppl.tests.utils import skip_on_travis


import deepppl
import os
import pandas as pd

stan_model_file = 'deepppl/tests/good/schools.stan'
global_num_iterations=10000
global_num_chains=1
global_warmup_steps = 1000

def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model)
    return mcmc.MCMC(nuts_kernel, **kwargs)

@skip_on_travis
def test_schools():
    model = deepppl.DppplModel(model_file=stan_model_file)

    J = 8
    y = torch.Tensor([28., 8., -3., 7., -1., 1., 18., 12.])
    sigma = torch.Tensor([15., 10., 16., 11., 9., 11., 10., 18.])

    schools_dat = {'J': 8,
                'y': [28,  8, -3,  7, -1,  1, 18, 12],
                'sigma': [15, 10, 16, 11, 9, 11, 10, 18]}

    t1 = time.time()
    posterior = model.posterior(
        method=nuts,
        num_samples=global_num_iterations-global_warmup_steps,
        warmup_steps=global_warmup_steps).run(J=J, sigma=sigma, y=y)

    marginal = posterior.marginal(sites=["mu"])
    marginal = torch.cat(list(marginal.support(
        flatten=True).values()), dim=-1).cpu().numpy()
    
    marginal2 = posterior.marginal(sites=["tau"])
    marginal2 = torch.cat(list(marginal2.support(
        flatten=True).values()), dim=-1).cpu().numpy()


    marginal1 = posterior.marginal(sites=["eta"])
    marginal1 = torch.cat(list(marginal1.support(
        flatten=True).values()), dim=-1).cpu().numpy()
    t2 = time.time()
    marginal = np.reshape(marginal, (marginal.shape[0], 1))
    marginal2 = np.reshape(marginal2, (marginal2.shape[0], 1))
    marginal = np.concatenate([marginal, marginal2], axis = 1)

    marginal = np.concatenate([marginal, marginal1], axis = 1)
    params = ['mu', 'tau', 'eta[0]', 'eta[1]', 'eta[2]',
              'eta[3]', 'eta[4]', 'eta[5]', 'eta[6]', 'eta[7]']
    df = pd.DataFrame(marginal, columns=params)
    pystan_output, time_pystan, pystan_compilation_time = compare_with_stan_output(schools_dat)
    assert df.shape == pystan_output.shape
    from scipy.stats import entropy
    for column in params:
        #import pdb;pdb.set_trace()
        hist1 = np.histogram(df[column], bins = 10)
        hist2 = np.histogram(pystan_output[column], bins = hist1[1])
        kl = entropy(hist1[0]+1, hist2[0]+1)
        skl = kl + entropy(hist2[0]+1, hist1[0]+1)
        print('skl for column:{} is:{:.2f}'.format(column, skl))

    print("Time taken: deepstan:{:.2f}, pystan_compilation:{:.2f}, pystan:{:.2f}".format(t2-t1, pystan_compilation_time, time_pystan))
def compare_with_stan_output(data):
    stan_code = open(stan_model_file).read()
    stan_code = """
    data {
        int<lower=0> J; // number of schools
        real y[J]; // estimated treatment effects
        real<lower=0> sigma[J]; // s.e. of effect estimates
    }
    parameters {
        real mu;
        real<lower=0> tau;
        real eta[J];
    }
    transformed parameters {
        real theta[J];
        for (j in 1:J)
            theta[j] = mu + tau * eta[j];
    }
    model {
        eta ~ normal(0, 1);
        y ~ normal(theta, sigma);
    }
    """
    # Compile and fit
    t1 = time.time()
    sm1 = pystan.StanModel(model_code=str(stan_code))
    t2 = time.time()
    pystan_compilation_time = t2 - t1
    t1 = time.time()
    fit_stan = sm1.sampling(data=data, iter=global_num_iterations, chains=global_num_chains, warmup = global_warmup_steps)

    mu = fit_stan.extract(permuted=True)['mu']
    tau = fit_stan.extract(permuted=True)['tau']
    eta = fit_stan.extract(permuted=True)['eta']
    mu = np.reshape(mu, (mu.shape[0], 1))
    tau = np.reshape(tau, (tau.shape[0], 1))

    marginal = np.concatenate((mu, tau, eta), axis = 1)
    params = ['mu', 'tau', 'eta[0]', 'eta[1]', 'eta[2]',
              'eta[3]', 'eta[4]', 'eta[5]', 'eta[6]', 'eta[7]']
    df = pd.DataFrame(marginal, columns=params)

    t2 = time.time()
    return df, t2-t1, pystan_compilation_time

if __name__ == "__main__":
    test_schools()
