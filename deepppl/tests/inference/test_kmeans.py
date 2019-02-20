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
import time
import pystan
from sklearn.datasets import make_blobs

stan_model_file = 'deepppl/tests/good/kmeans.stan'
global_num_iterations=3000
global_num_chains=1

def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model, adapt_step_size=True)
    return mcmc.MCMC(nuts_kernel, **kwargs)


def test_kmeans():
    model = deepppl.DppplModel(model_file='deepppl/tests/good/kmeans.stan')

    num_samples = 6
    num_features = 2
    X = torch.tensor([[1., 2.], [1., 4.], [1., 0.], [4., 2.], [4., 4.], [4., 0.]])
    num_clusters = 2

    n_samples = 1500
    random_state = 170
    #X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    #X = torch.from_numpy(X)

    data = {'N': num_samples,
            'D': num_features,
            'K': num_clusters,
            'y': X}

    #X = torch.tensor([[1., 1., 1., 4., 4., 4.], [2., 4., 0., 2., 4., 0.]])
    #import pdb;pdb.set_trace()

    posterior = model.posterior(
        method=nuts,
        num_samples=30,
        warmup_steps=3)

    marginal = pyro.infer.EmpiricalMarginal(
        posterior.run(N=num_samples, D=num_features, K=num_clusters, y=X,
                      transformed_data=model._transformed_data(N=num_samples, D=num_features, K=num_clusters, y=X)),
        sites="mu")

    samples_fstan = [marginal()
                        for _ in range(3000)]
    stack_samples = torch.stack(samples_fstan)  
    import pdb;pdb.set_trace()                      
    #print(stack_samples.size())
    compare_with_stan_output(data)
    # assert np.abs(series.mean() - 1000) < 1
    # assert np.abs(series.std() - 1.0) < 0.1

def compare_with_stan_output(data):
    #stan_code = open(stan_model_file).read()
    t1 = time.time()
    stan_code = """
        data {
        int<lower=0> N;  // number of data points
        int<lower=1> D;  // number of dimensions
        int<lower=1> K;  // number of clusters
        vector[D] y[N];  // observations
        }
        transformed data {
        real<upper=0> neg_log_K;
        neg_log_K = -log(K);
        }
        parameters {
        vector[D] mu[K]; // cluster means
        }
        transformed parameters {
        real<upper=0> soft_z[N, K]; // log unnormalized clusters
        for (n in 1:N)
            for (k in 1:K)
            soft_z[n, k] = neg_log_K
                - 0.5 * dot_self(mu[k] - y[n]);
        }
        model {
        // prior
        for (k in 1:K)
                mu[k] ~ normal(0, 1);
        // likelihood
        for (n in 1:N)
                target += log_sum_exp(soft_z[n]);
        }
    """
    data['y'] = np.array([[1., 2], [1.,4], [1, 0], [4.,2], [4., 4.],[4, 0]])# [2., 4., 0., 2., 4., 0.]])
    data['K'] = 2
    print(data)
    # Compile and fit
    sm1 = pystan.StanModel(model_code=str(stan_code))
    fit_stan = sm1.sampling(data=data, iter=global_num_iterations, chains=global_num_chains, warmup = 300)

    mu = fit_stan.extract(permuted=True)['mu']
    print(mu.shape)
    t2 = time.time()



if __name__ == "__main__":
    test_kmeans()
