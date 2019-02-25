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
    nuts_kernel = mcmc.NUTS(model, adapt_step_size = True)
    return mcmc.MCMC(nuts_kernel, **kwargs)

def test_logistic():
    model = deepppl.DppplModel(model_file = 'deepppl/tests/good/logistic.stan')

    # Add Data
    num_samples = 7
    num_features = 2
    #X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    X = np.array([[1, 1.329799263],[1, 1.272429321], [1, -1.539950042], [1, -0.928567035], [1, -0.294720447], [1, -0.005767173], [1, 2.404653389]])
    # y = 1 * x_0 + 2 * x_1 + 3
    X = torch.from_numpy(X).float()
    y = np.array([0, 1, 1, 1, 1, 1, 1])
    y = torch.from_numpy(y)

    posterior = model.posterior(
                method=nuts,
                num_samples=3000,
                warmup_steps=300)

    marginal = pyro.infer.EmpiricalMarginal(posterior.run(N=num_samples, M=num_features, y=y, x=X), sites='beta')

    series = pd.Series([marginal() for _ in range(3000)], name = r'$beta$')
    print(series)
    # assert np.abs(series.mean() - 1000) < 1
    # assert np.abs(series.std() - 1.0) < 0.1

if __name__ == "__main__":
    test_logistic()
