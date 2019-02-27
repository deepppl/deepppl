import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist
from deepppl.utils.utils import *
from torch import zeros, ones, sqrt

def transformed_data(N=None, exposure2=None, roach1=None, senior=None,
                     treatment=None, y=None):
    ___shape = {}
    ___shape['log_expo'] = N
    log_expo = zeros(___shape['log_expo'])
    log_expo = log(exposure2)
    return {'log_expo': log_expo}


def model2(N=None, exposure2=None, roach1=None, senior=None, treatment=None,
          y=None, transformed_data=None):
    log_expo = transformed_data['log_expo']
    ___shape = {}
    ___shape['N'] = ()
    ___shape['exposure2'] = N
    ___shape['roach1'] = N
    ___shape['senior'] = N
    ___shape['treatment'] = N
    ___shape['y'] = N
    ___shape['beta'] = 4
    ___shape['lmbda'] = N
    ___shape['tau'] = ()
    beta = pyro.sample('beta', ImproperUniform(4))
    lmbda = pyro.sample('lmbda', ImproperUniform(N))
    tau = pyro.sample('tau', ImproperUniform())
    ___shape['sigma'] = ()
    sigma = 1.0 / sqrt(tau)
    pyro.sample('tau' + '1', dist.Gamma(0.001, 0.001), obs=tau)
    pyro.sample('lmbda' + '2', dist.Normal(0, sigma), obs=lmbda)
    pyro.sample('y' + '3', poisson_log(log_expo + beta[1 - 1] + beta[2 - 1] *
                                       roach1 + beta[3 - 1] * treatment + beta[4 - 1] * senior + lmbda), obs=y
                )

exposure2 = [0.8, 0.6, 1, 1, 1.1428571, 1, 
0.8, 1.1428571, 1, 1.1428571, 1, 1, 
1, 0.8, 1, 0.8, 0.8, 1, 
1, 1, 1, 1, 1, 0.8571429, 
1, 1, 0.8, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 
1, 0.8, 1.5714286, 1.4285714, 1, 1, 
1, 0.8, 1, 1, 1, 1, 
1, 1, 1.1428571, 1.1428571, 1, 1, 
0.8, 1, 0.9142857, 1, 1.2857143, 1, 
1, 0.7714286, 1, 1.4285714, 1, 0.7714286, 
1, 1, 1, 1, 1, 0.6, 
1, 1, 1, 1, 1.2857143, 1, 
1, 1, 1, 1, 1, 1.1428571, 
1, 1, 1, 0.8, 0.8, 1, 
1, 0.6857143, 1, 1, 1, 1, 
1, 1, 0.8, 0.8571429, 1, 1, 
1, 1, 1, 1, 0.4, 0.8, 
1.1428571, 1, 0.8, 1, 0.2, 1, 
1, 1, 1, 1, 1, 1, 
1, 1.1428571, 1.1428571, 1, 0.8, 1, 
0.8, 1, 1, 4.2857143, 0.8, 0.8, 
0.8, 1, 1, 1.2857143, 1, 1, 
1, 1, 1, 0.4571429, 1, 2.4285714, 
1.4285714, 1, 1, 1.1428571, 1, 1.1428571, 
1, 1, 1, 0.8, 1.1428571, 1, 
0.8, 1.5714286, 1, 1.1428571, 2.2857143, 1, 
0.5714286, 0.8, 0.8571429, 0.8, 0.8571429, 2.2857143, 
0.8, 1, 0.8571429, 1, 1.0285714, 0.6, 
0.8571429, 1, 0.8, 1, 1.4857143, 1, 
1, 1, 0.8, 0.8, 1, 1, 
0.8571429, 0.8, 1, 0.8, 1, 1, 
0.8, 1, 1, 1, 1.4285714, 1.7142857, 
0.7714286, 0.8, 1, 1.8571429, 1, 1, 
1, 1, 1, 1, 0.8, 1, 
1, 2.2857143, 0.8, 1.4285714, 1.0285714, 1.7142857, 
1.1428571, 1.1428571, 1, 1, 1.1428571, 0.8, 
1.4285714, 0.9142857, 1.1428571, 1.2857143, 0.6, 1, 
0.8, 1.1428571, 1, 1.4285714, 1, 1, 
1, 1, 1, 1, 1, 1, 
1.1428571, 0.8571429, 1, 1, 1, 1, 
1, 1, 1, 0.6, 1.0285714, 1, 
1, 0.6857143, 0.8, 1, 0.8, 1.4857143, 
1, 1, 1, 1]
senior = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
treatment = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 
1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
roach1 = [308, 331.25, 1.67, 3, 2, 0, 70, 64.56, 
1, 14, 138.25, 16, 97, 98, 44, 450, 
36.67, 75, 2, 342.5, 22.5, 4, 94, 132.13, 
80, 95, 34.13, 19, 25, 57.5, 3.11, 10.5, 
47, 2.63, 204, 0, 0, 246.25, 76.22, 15, 
14.88, 51, 15, 52.5, 13.75, 1.17, 1, 2, 
0, 2, 2, 30, 7, 18, 2, 266, 
174, 252, 0.88, 372.5, 42, 19, 263, 34, 
1.75, 213.5, 13.13, 154, 21, 220, 38.32, 352.5, 
7, 4, 59.5, 7.5, 112.88, 172, 13, 18, 
0, 27, 148, 32, 28, 0, 28, 0, 
14, 5, 0, 104, 27, 132, 258, 1, 
2, 6, 3, 3, 0, 0, 0, 1.25, 
0, 16, 68, 1, 18, 123.67, 2, 82.5, 
0, 1.25, 171, 91.88, 5, 7, 0, 4, 
53.75, 138.06, 1, 1.25, 28, 15, 0.88, 0, 
33, 136.25, 127.5, 2, 2, 0, 3, 46, 
68, 0, 49, 27.13, 45.5, 0, 4, 0, 
1, 0, 0, 10.5, 0, 3, 0, 0, 
0, 3.75, 0, 0, 0, 0.88, 2, 81, 
31, 10, 10.23, 0, 13, 1, 33.25, 0, 
53, 5, 157, 23.33, 6, 10, 100, 55, 
0, 16.25, 7.78, 53, 2, 73, 0, 0, 
0, 3.18, 5, 3, 0, 10, 1, 0, 
1, 12, 17, 16, 2.5, 0, 21.88, 173, 
111, 35, 0, 3, 2.1, 0, 49, 1.25, 
1, 1, 0, 0, 54, 4.2, 51.25, 30, 
196, 0, 2, 1.5, 96.25, 241.5, 140, 18, 
3, 0.54, 82, 19, 18.75, 51, 0, 1, 
0, 5.44, 0, 3, 0, 0, 0, 0, 
28.75, 0, 3, 2, 135, 0, 68.25, 1, 
0, 0, 0, 1, 1, 2.5, 51.25, 13.13, 
0, 0, 0, 0, 0, 0]
y = [153, 127, 7, 7, 0, 0, 73, 24, 2, 2, 0, 21, 
0, 179, 136, 104, 2, 5, 1, 203, 32, 1, 135, 59, 
29, 120, 44, 1, 2, 193, 13, 37, 2, 0, 3, 0, 
0, 15, 11, 19, 0, 19, 4, 122, 48, 0, 0, 3, 
0, 9, 0, 0, 0, 12, 0, 357, 11, 60, 0, 159, 
50, 48, 178, 4, 6, 0, 33, 127, 4, 63, 88, 5, 
0, 0, 62, 4, 150, 38, 0, 3, 1, 14, 77, 42, 
21, 1, 45, 0, 0, 0, 0, 0, 183, 28, 49, 1, 
0, 0, 3, 0, 0, 0, 0, 18, 0, 0, 5, 0, 
19, 5, 0, 27, 0, 0, 77, 1, 3, 2, 0, 0, 
22, 102, 0, 0, 0, 0, 0, 0, 0, 4, 12, 2, 
0, 0, 1, 0, 40, 0, 1, 2, 27, 0, 2, 0, 
0, 0, 0, 3, 1, 20, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 53, 69, 15, 0, 2, 4, 6, 8, 0, 
0, 0, 18, 38, 0, 2, 18, 34, 1, 109, 5, 15, 
0, 64, 0, 1, 0, 1, 3, 5, 7, 18, 1, 0, 
0, 3, 3, 0, 19, 0, 8, 26, 50, 15, 0, 19, 
5, 17, 121, 1, 0, 0, 0, 0, 4, 1, 14, 1, 
25, 0, 14, 0, 59, 243, 80, 69, 14, 9, 38, 37, 
48, 293, 7, 10, 19, 24, 91, 1, 0, 0, 0, 0, 
148, 3, 26, 12, 77, 0, 7, 0, 1, 0, 17, 0, 
7, 11, 6, 50, 1, 0, 0, 0, 171, 8]
N = 262

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

stan_model_file = 'deepppl/tests/good/cockroaches.stan'
global_num_iterations=1000
global_num_chains=1
global_warmup_steps = 100

def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model)
    return mcmc.MCMC(nuts_kernel, **kwargs)

def test_cockroaches():
    model = deepppl.DppplModel(model_file=stan_model_file)
    model._model = model2
    posterior = model.posterior(
        method=nuts,
        num_samples=global_num_iterations-global_warmup_steps,
        warmup_steps=global_warmup_steps)

    local_exposure2 = torch.Tensor(exposure2)
    local_roach1 = torch.Tensor(roach1)
    local_senior = torch.Tensor(senior)
    local_treatment = torch.Tensor(treatment)
    local_y = torch.Tensor(y)

    # marginal = pyro.infer.EmpiricalMarginal(posterior.run(N = N, 
    #             exposure2 = local_exposure2, roach1 = local_roach1, senior = local_senior,
    #             treatment = local_treatment, y = local_y, transformed_data = transformed_data(N = N, 
    #             exposure2 = local_exposure2, roach1 = local_roach1, senior = local_senior,
    #             treatment = local_treatment, y = local_y)), sites=['lmbda', 'tau'])

    posterior = posterior.run(N = N, 
                exposure2 = local_exposure2, roach1 = local_roach1, senior = local_senior,
                treatment = local_treatment, y = local_y, transformed_data = transformed_data(N = N, 
                exposure2 = local_exposure2, roach1 = local_roach1, senior = local_senior,
                treatment = local_treatment, y = local_y))
    marginal = posterior.marginal(sites=["lmbda"])

    #samples_fstan = [marginal() for _ in range(global_num_iterations-global_warmup_steps)]
    marginal = torch.cat(list(marginal.support(flatten=True).values()), dim=-1).cpu().numpy()
    mean_lambda = marginal.mean(axis = 0)

    marginal = posterior.marginal(sites=["tau"])
    tau = torch.cat(list(marginal.support(flatten=True).values()), dim=-1).cpu().numpy()

    #stack_samples = torch.stack(samples_fstan)
    import pdb;pdb.set_trace()

    pystan_mean_lambda, pystan_tau = compare_with_stan_output()

    from scipy.stats import entropy
    hist1 = np.histogram(tau, bins = 10)
    hist2 = np.histogram(pystan_tau, bins = hist1[1])
    kl = entropy(hist1[0]+1, hist2[0]+1)
    skl = kl + entropy(hist2[0]+1, hist1[0]+1)
    print('skl for tau is:{}'.format(skl))

    # mean_lambda = stack_samples.mean(dim=0)
    # mean_lambda_arr = mean_lambda[0, :]

    # from scipy.stats import entropy
    # hist1 = np.histogram(mean_lambda_arr.numpy(), bins = 10)
    # hist2 = np.histogram(pystan_mean_lambda, bins = hist1[1])
    # kl = entropy(hist1[0]+1, hist2[0]+1)
    # skl = kl + entropy(hist2[0]+1, hist1[0]+1)
    # print('skl for lambda is:{}'.format(skl))


def compare_with_stan_output():
    stan_code = open(stan_model_file).read()
    stan_code = """
        data {
            int N;
            vector[N] exposure2;
            vector[N] roach1;
            vector[N] senior;
            vector[N] treatment;
            int y[N];
        }
        transformed data {
            vector[N] log_expo;
            log_expo=log(exposure2);
        }
        parameters {
            vector[4] beta;
            vector[N] lmbda;
            real tau;
        }
        transformed parameters {
            real sigma = 1.0 / sqrt(tau);
        }
        model {
            tau ~ gamma(0.001, 0.001);
            lmbda ~ normal(0, sigma);
            y ~ poisson_log(log_expo + beta[1]
                + beta[2] * roach1
                + beta[3] * treatment
                + beta[4] * senior
                + lmbda);
        }    
    """
    data = {
        'N': N, 
        'exposure2': exposure2, 
        'roach1': roach1, 
        'senior': senior,
        'treatment': treatment, 
        'y': y        
    }

    # Compile and fit
    sm1 = pystan.StanModel(model_code=str(stan_code))
    print("Compilation done")
    t1 = time.time()

    fit_stan = sm1.sampling(data = data, iter=global_num_iterations, chains=global_num_chains, 
                            warmup = global_warmup_steps)

    t2 = time.time()
    mean_lambda = fit_stan.extract(permuted=True)['lmbda'].mean(axis = 0)
    tau = fit_stan.extract(permuted=True)['tau']

    return mean_lambda, tau

if __name__ == "__main__":
    test_cockroaches()
