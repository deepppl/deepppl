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
    nuts_kernel = mcmc.NUTS(model, adapt_step_size=True)
    return mcmc.MCMC(nuts_kernel, **kwargs)


def test_kmeans():
    model = deepppl.DppplModel(model_file='deepppl/tests/good/kmeans.stan')

    num_samples = 6
    num_features = 2
    X = torch.tensor([[1., 1., 1., 4., 4., 4.], [2., 4., 0., 2., 4., 0.]])
    num_clusters = 2

    posterior = model.posterior(
        method=nuts,
        num_samples=30,
        warmup_steps=3)

    marginal = pyro.infer.EmpiricalMarginal(
        posterior.run(num_samples, num_features, num_clusters, X,
                      model._transformed_data(num_samples, num_features, num_clusters, X)),
        sites="mu")

    series = pd.Series([marginal()
                        for _ in range(3000)], name=r'$\mu$')
    print(series)
    # assert np.abs(series.mean() - 1000) < 1
    # assert np.abs(series.std() - 1.0) < 0.1


if __name__ == "__main__":
    test_kmeans()
