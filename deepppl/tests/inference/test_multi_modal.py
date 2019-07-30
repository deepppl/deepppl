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
from deepppl.tests.utils import skip_on_travis
import deepppl
import os
import pandas as pd

stan_model_file = 'deepppl/tests/good/multimodal.stan'
global_num_iterations=3000
global_num_chains=1
global_warmup_steps = 300

def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model)
    return mcmc.MCMC(nuts_kernel, **kwargs)

@skip_on_travis
def test_multi_modal():
    model = deepppl.DppplModel(model_file=stan_model_file)
    #model._model = model2
    posterior = model.posterior(
        method=nuts,
        num_samples=global_num_iterations-global_warmup_steps,
        warmup_steps=global_warmup_steps)

    t1 = time.time()
    marginal = pyro.infer.EmpiricalMarginal(posterior.run(), 
        sites=['theta','cluster'])

    samples_fstan = [marginal() for _ in range(global_num_iterations-global_warmup_steps)]
    t2 = time.time()
    stack_samples = torch.stack(samples_fstan).numpy()

    pystan_output, pystan_time, pystan_compilation_time = compare_with_stan_output()

    assert stack_samples.shape == pystan_output.shape
    from scipy.stats import entropy
    hist1 = np.histogram(stack_samples[:, 0], bins = 10)
    hist2 = np.histogram(pystan_output[:, 0], bins = hist1[1])
    kl = entropy(hist1[0]+1, hist2[0]+1)
    skl = kl + entropy(hist2[0]+1, hist1[0]+1)
    print('skl for theta is:{:.4f}'.format(skl))

    hist1 = np.histogram(stack_samples[:, 1], bins = 10)
    hist2 = np.histogram(pystan_output[:, 1], bins = hist1[1])
    kl = entropy(hist1[0]+1, hist2[0]+1)
    skl = kl + entropy(hist2[0]+1, hist1[0]+1)
    print('skl for cluster is:{:.4f}'.format(skl))

    print("Time taken: deepstan:{:.2f}, pystan_compilation:{:.2f}, pystan:{:.2f}".format(t2-t1, pystan_compilation_time, pystan_time))

def compare_with_stan_output():
    stan_code = open(stan_model_file).read()

    # Compile and fit
    t1 = time.time()

    sm1 = pystan.StanModel(model_code=str(stan_code))
    print("Compilation done")
    t2 = time.time()
    pystan_compilation_time = t2-t1

    t1 = time.time()
    fit_stan = sm1.sampling(iter=global_num_iterations, chains=global_num_chains, 
                            warmup = global_warmup_steps)

    t2 = time.time()
    theta = fit_stan.extract(permuted=True)['theta']
    theta = np.reshape(theta, (theta.shape[0], 1))

    cluster = fit_stan.extract(permuted=True)['cluster']
    cluster = np.reshape(cluster, (cluster.shape[0], 1))

    marginal = np.concatenate((theta, cluster), axis = 1)

    t2 = time.time()
    return marginal, t2-t1, pystan_compilation_time

if __name__ == "__main__":
    test_multi_modal()
