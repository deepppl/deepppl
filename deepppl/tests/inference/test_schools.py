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

from deepppl.utils.utils import ImproperUniform, LowerConstrainedImproperUniform


import deepppl
import os
import pandas as pd


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
    model = deepppl.DppplModel(model_file='deepppl/tests/good/schools.stan')

    J = 8
    y = torch.tensor([28., 8., -3., 7., -1., 1., 18., 12.])
    sigma = torch.tensor([15., 10., 16., 11., 9., 11., 10., 18.])

    posterior = model.posterior(
        method=nuts,
        num_samples=30,
        warmup_steps=3).run(J, sigma, y)

    marginal = posterior.marginal(sites=["mu", "tau", "eta"])
    print(marginal)
    marginal = torch.cat(list(marginal.support(
        flatten=True).values()), dim=-1).cpu().numpy()
    params = ['mu', 'tau', 'eta[0]', 'eta[1]', 'eta[2]',
              'eta[3]', 'eta[4]', 'eta[5]', 'eta[6]', 'eta[7]']
    df = pd.DataFrame(marginal, columns=params).transpose()
    df_summary = df.apply(pd.Series.describe, axis=1)[
        ["mean", "std", "25%", "50%", "75%"]]
    print(df_summary)


if __name__ == "__main__":
    test_schools()
