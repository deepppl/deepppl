import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist
from deepppl.utils.utils import *
from torch import zeros, ones
import pytest

def model2(K=None, M=None, N=None, V=None, alpha=None, beta=None, doc=None,
          w=None):
    ___shape = {}
    ___shape['K'] = ()
    ___shape['V'] = ()
    ___shape['M'] = ()
    ___shape['N'] = ()
    ___shape['w'] = N
    ___shape['doc'] = N
    ___shape['alpha'] = K
    ___shape['beta'] = V
    ___shape['theta'] = M, K
    ___shape['phi'] = K, V
    theta = pyro.sample('theta', ImproperUniform((M, K)))
    phi = pyro.sample('phi', ImproperUniform((K, V)))
    for m in range(1, M + 1):
        pyro.sample('theta' + '{}'.format(m - 1) + '1', dist.Dirichlet(
            alpha), obs=theta[m - 1])
    for k in range(1, K + 1):
        pyro.sample('phi' + '{}'.format(k - 1) + '2', dist.Dirichlet(beta),
                    obs=phi[k - 1])
    for n in range(1, N + 1):
        ___shape['gamma'] = K
        gamma = zeros(___shape['gamma'])
        #import pdb;pdb.set_trace()
        for k in range(1, K + 1):
            gamma[k - 1] = log(theta[doc[n - 1]-1, k-1]) + log(phi[k-1, w[n-1]-1])
        pyro.sample('expr' + str(n), dist.Exponential(1.0), obs=-log_sum_exp(
            gamma))


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

K = 2
V = 5
M = 25
N = 262
w = [4, 3, 5, 4, 3, 3, 3, 3, 3, 4, 5, 3, 4, 4, 5, 
3, 4, 4, 4, 3, 5, 4, 5, 2, 3, 3, 1, 5, 5, 1, 4, 
3, 1, 2, 5, 4, 4, 3, 5, 4, 2, 4, 5, 3, 4, 1, 4, 
4, 3, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 3, 1, 2, 2, 
4, 4, 5, 4, 5, 5, 4, 3, 5, 4, 4, 4, 2, 2, 1, 1, 
2, 1, 3, 1, 2, 1, 1, 1, 3, 2, 3, 3, 5, 4, 5, 4, 
3, 5, 4, 2, 2, 2, 1, 3, 2, 1, 3, 1, 3, 1, 1, 2, 
1, 2, 2, 4, 4, 4, 5, 5, 4, 4, 5, 4, 3, 3, 3, 1, 
3, 3, 4, 2, 1, 3, 4, 4, 5, 4, 4, 4, 3, 4, 3, 4, 
5, 1, 2, 1, 3, 2, 1, 1, 2, 3, 3, 3, 3, 4, 1, 4, 
4, 4, 4, 3, 4, 4, 1, 2, 2, 3, 3, 1, 1, 4, 1, 3, 
1, 5, 3, 2, 2, 1, 1, 2, 3, 3, 4, 4, 5, 3, 4, 3, 
1, 5, 5, 5, 3, 3, 4, 5, 3, 3, 3, 2, 3, 1, 3, 3, 
1, 3, 1, 5, 5, 5, 2, 2, 3, 3, 3, 1, 1, 5, 5, 5, 
3, 1, 5, 4, 1, 3, 3, 3, 3, 4, 2, 5, 1, 3, 5, 2, 
5, 5, 2, 1, 3, 3, 5, 3, 5, 3, 3, 5, 1, 2, 2, 1, 
1, 2, 1, 2, 3, 1, 1]
doc = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 
2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 
8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 
9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 
12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 
14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 
15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 
16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 
17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 
19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 
20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 
22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 
23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 
24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 
25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25]
alpha = [0.5, 0.5]
beta = [0.2, 0.2, 0.2, 0.2, 0.2]


stan_model_file = 'deepppl/tests/good/lda.stan'
global_num_iterations = 1000
global_warm_up_steps = 100
global_num_chains = 1

def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model, adapt_step_size=True)
    return mcmc.MCMC(nuts_kernel, **kwargs)

@pytest.mark.xfail(strict=False, reason="This currently fails with type inference.  Reasons not yet investigated.")
def test_lda_inference():
    model = deepppl.DppplModel(
        model_file=stan_model_file)
    model._model = model2

    posterior = model.posterior(
        method=nuts,
        num_samples=global_num_iterations-global_warm_up_steps,
        warmup_steps=global_warm_up_steps)
    global w, doc, alpha, beta
    w_local = torch.Tensor(w).type(torch.int32)
    print(w_local)
    doc_local = torch.Tensor(doc).type(torch.int32)
    alpha_local = torch.Tensor(alpha)
    beta_local = torch.Tensor(beta)

    t1 = time.time()
    marginal = pyro.infer.EmpiricalMarginal(posterior.run(
            K = K, V = V, M = M, N = N, w = w_local, doc = doc_local, alpha = alpha_local, beta = beta_local), 
            sites=['theta', 'phi']) 
    samples_fstan = [marginal() for _ in range(10)]
    stack_samples = torch.stack(samples_fstan)
    params = torch.mean(stack_samples, 0)
    t2 = time.time()

    time_taken = t2-t1
    print(params)
    compare_with_stan_output()

def compare_with_stan_output():
    data = {
    'K': K,
    'V': V,
    'M': M,
    'N': N,
    'w': w,
    'doc': doc,
    'alpha': alpha,
    'beta': beta
    }

    stan_code = open(stan_model_file).read()
    stan_code = """
        data {
        int<lower=2> K;               // num topics
        int<lower=2> V;               // num words
        int<lower=1> M;               // num docs
        int<lower=1> N;               // total word instances
        int<lower=1,upper=V> w[N];    // word n
        int<lower=1,upper=M> doc[N];  // doc ID for word n
        vector<lower=0>[K] alpha;     // topic prior
        vector<lower=0>[V] beta;      // word prior
        }
        parameters {
        simplex[K] theta[M];   // topic dist for doc m
        simplex[V] phi[K];     // word dist for topic k
        }
        model {
        for (m in 1:M)
            theta[m] ~ dirichlet(alpha);  // prior
        for (k in 1:K)
            phi[k] ~ dirichlet(beta);     // prior
        for (n in 1:N) {
            real gamma[K];
            for (k in 1:K)
            gamma[k] = log(theta[doc[n],k]) + log(phi[k,w[n]]);
            target += log_sum_exp(gamma);  // likelihood
        }
        }
"""
    t1 = time.time()

    # Compile and fit
    sm1 = pystan.StanModel(model_code=str(stan_code))
    fit_stan = sm1.sampling(data=data, iter=global_num_iterations, chains=global_num_chains, warmup = 30)
    theta = fit_stan.extract(permuted=True)['theta']
    print(theta)


if __name__ == "__main__":
    test_lda_inference()