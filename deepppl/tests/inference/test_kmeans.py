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

def transformed_data(D, K, N, y):
    ___shape = {}
    ___shape['neg_log_K'] = ()
    neg_log_K = -np.log(K)
    return {'neg_log_K': neg_log_K}

def test_kmeans():
    model = deepppl.DppplModel(model_file = 'deepppl/tests/good/kmeans.stan')

    num_samples = 6
    num_features = 2
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    num_clusters = 2

    data = {'N': num_samples,
            'D': num_features,
            'K': num_clusters,
            'y': X}

    posterior = model.posterior(
                method=nuts, 
                num_samples=3000, 
                warmup_steps=300)

    marginal = pyro.infer.EmpiricalMarginal(posterior.run(num_samples, num_features, num_clusters, X, 
                transformed_data(num_samples, num_features, num_clusters, X)), sites='theta')

    series = pd.Series([marginal().item() for _ in range(3000)], name = r'$\theta$')
    print(series)
    # assert np.abs(series.mean() - 1000) < 1
    # assert np.abs(series.std() - 1.0) < 0.1 

if __name__ == "__main__":
    test_kmeans()