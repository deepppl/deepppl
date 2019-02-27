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


import deepppl
import os
import pandas as pd

stan_model_file = 'deepppl/tests/good/schools.stan'
global_num_iterations=3000
global_num_chains=1
global_warmup_steps = 300

def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model)
    return mcmc.MCMC(nuts_kernel, **kwargs)

def test_schools():
    model = deepppl.DppplModel(model_file=stan_model_file)

    J = 8
    y = torch.tensor([28., 8., -3., 7., -1., 1., 18., 12.])
    sigma = torch.tensor([15., 10., 16., 11., 9., 11., 10., 18.])

    schools_dat = {'J': 8,
                'y': [28,  8, -3,  7, -1,  1, 18, 12],
                'sigma': [15, 10, 16, 11, 9, 11, 10, 18]}

    t1 = time.time()
    posterior = model.posterior(
        method=nuts,
        num_samples=global_num_iterations-global_warmup_steps,
        warmup_steps=global_warmup_steps).run(J=J, sigma=sigma, y=y)

    marginal = posterior.marginal(sites=["mu", "tau", "eta"])
    marginal = torch.cat(list(marginal.support(
        flatten=True).values()), dim=-1).cpu().numpy()
    
    t2 = time.time()
    params = ['mu', 'tau', 'eta[0]', 'eta[1]', 'eta[2]',
              'eta[3]', 'eta[4]', 'eta[5]', 'eta[6]', 'eta[7]']
    df = pd.DataFrame(marginal, columns=params)
    pystan_output, time_pystan = compare_with_stan_output(schools_dat)
    assert df.shape == pystan_output.shape
    from scipy.stats import entropy
    for column in params:
        #import pdb;pdb.set_trace()
        hist1 = np.histogram(df[column], bins = 10)
        hist2 = np.histogram(pystan_output[column], bins = hist1[1])
        kl = entropy(hist1[0]+1, hist2[0]+1)
        skl = kl + entropy(hist2[0]+1, hist1[0]+1)
        print('skl for column:{} is:{}'.format(column, skl))

    print("Time taken: deepstan:{}, pystan:{}".format(t2-t1, time_pystan))
def compare_with_stan_output(data):
    # stan_code = open(stan_model_file).read()
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
    sm1 = pystan.StanModel(model_code=str(stan_code))
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
    return df, t2-t1

if __name__ == "__main__":
    test_schools()
