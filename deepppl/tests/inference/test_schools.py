import torch
import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor
import pyro
from pyro import distributions as dist
import numpy as np
from pyro.infer import mcmc

import deepppl
import os
import pandas as pd


def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model)
    return mcmc.MCMC(nuts_kernel, **kwargs)


def test_schools():
    model = deepppl.DppplModel(model_file='deepppl/tests/good/schools.stan')

    J = 8
    y = torch.tensor([28., 8., -3., 7., -1., 1., 18., 12.])
    sigma = torch.tensor([15., 10., 16., 11., 9., 11., 10., 18.])

    posterior = model.posterior(
        method=nuts,
        num_samples=30,
        warmup_steps=3).run(J, sigma, y)

    marginal_mu_tau = pyro.infer.EmpiricalMarginal(
        posterior.run(J, sigma, y), sites=['mu', 'tau'])
    marginal_eta = pyro.infer.EmpiricalMarginal(
        posterior.run(J, sigma, y), sites=["eta"])

    series = pd.Series([marginal_mu_tau() for _ in range(30)], name=r'$mu$')
    print(series)
    # assert np.abs(series.mean() - 1000) < 1
    # assert np.abs(series.std() - 1.0) < 0.1


if __name__ == "__main__":
    test_schools()
