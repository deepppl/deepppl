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
global_num_iterations=1000
global_num_chains=1

def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model)
    return mcmc.MCMC(nuts_kernel, **kwargs)


def model2(J, sigma, y):
    eta = pyro.sample('eta', ImproperUniform(J))
    mu = pyro.sample('mu', ImproperUniform())
    # mu = pyro.sample('mu', dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
    tau = pyro.sample('tau', dist.HalfCauchy(scale=25 * torch.ones(1)))

    theta = mu + tau * eta

    return pyro.sample("obs", dist.Normal(theta, sigma), obs=y)


def test_schools():
    model = deepppl.DppplModel(model_file=stan_model_file)

    J = 8
    y = torch.tensor([28., 8., -3., 7., -1., 1., 18., 12.])
    sigma = torch.tensor([15., 10., 16., 11., 9., 11., 10., 18.])

    schools_dat = {'J': 8,
                'y': [28,  8, -3,  7, -1,  1, 18, 12],
                'sigma': [15, 10, 16, 11, 9, 11, 10, 18]}

    posterior = model.posterior(
        method=nuts,
        num_samples=1000,
        warmup_steps=300).run(J=J, sigma=sigma, y=y)

    marginal = posterior.marginal(sites=["mu", "tau", "eta"])
    marginal = torch.cat(list(marginal.support(
        flatten=True).values()), dim=-1).cpu().numpy()
    params = ['mu', 'tau', 'eta[0]', 'eta[1]', 'eta[2]',
              'eta[3]', 'eta[4]', 'eta[5]', 'eta[6]', 'eta[7]']
    df = pd.DataFrame(marginal, columns=params).transpose()
    df_summary = df.apply(pd.Series.describe, axis=1)[
        ["mean", "std", "25%", "50%", "75%"]]
    print(df_summary)
    compare_with_stan_output(schools_dat)

def compare_with_stan_output(data):
    #stan_code = open(stan_model_file).read()
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
    t1 = time.time()

    # Compile and fit
    sm1 = pystan.StanModel(model_code=str(stan_code))
    fit_stan = sm1.sampling(data=data, iter=global_num_iterations, chains=global_num_chains, warmup = 300)

    mu = fit_stan.extract(permuted=True)['mu'].mean()
    t2 = time.time()

    print(mu)

if __name__ == "__main__":
    test_schools()
