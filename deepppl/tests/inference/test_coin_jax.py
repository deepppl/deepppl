import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor
import pyro
import numpy as np
from numpyro.hmc_util import initialize_model
from numpyro.mcmc import mcmc
from numpyro import distributions as dist
import jax.numpy as jnp
import jax.random as random
import logging
import time
import pystan
from deepppl.utils.utils import ImproperUniform, LowerConstrainedImproperUniform
import deepppl
import os
import pandas as pd


stan_model_file = 'deepppl/tests/good/coin.stan'
global_num_iterations=10000
global_num_chains=1
global_warmup_steps = 300

x = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
def run_inference(model, x):
    rng, rng_predict = random.split(random.PRNGKey(1+1))
    if global_num_chains > 1:
        rng = random.split(rng, global_num_chains)
        assert False, "don't know what to do"
    init_params, potential_fn, constrain_fn = initialize_model(rng, model, x)
    hmc_states = mcmc(global_warmup_steps, global_num_iterations, init_params,
                      sampler='hmc', potential_fn=potential_fn, constrain_fn=constrain_fn)
    return hmc_states

def test_coin():
    model = deepppl.NumPyroDPPLModel(model_file=stan_model_file)
    #model._model = model2

    local_x = jnp.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1])

    t1 = time.time()
    states = run_inference(model._model, x)
    samples_fstan = states['theta'][global_warmup_steps:]

    t2 = time.time()
    
    pystan_output, pystan_time, pystan_compilation_time = compare_with_stan_output()

    pystan_output = pystan_output.squeeze()
    assert samples_fstan.shape == pystan_output.shape
    from scipy.stats import entropy, ks_2samp
    hist1 = np.histogram(samples_fstan, bins = 10)
    hist2 = np.histogram(pystan_output, bins = hist1[1])
    kl = entropy(hist1[0]+1, hist2[0]+1)
    skl = kl + entropy(hist2[0]+1, hist1[0]+1)
    ks = ks_2samp(samples_fstan, pystan_output)
    print('skl for theta is:{:.6}'.format(skl))
    print(f'skl for theta is: {ks}')

    print("Time taken: deepstan:{:.2f}, pystan_compilation:{:.2f}, pystan:{:.2f}".format(t2-t1, pystan_compilation_time, pystan_time))

def compare_with_stan_output():
    stan_code = open(stan_model_file).read()
    data = {
        'x': x
    }

    # Compile and fit
    t1 = time.time()

    sm1 = pystan.StanModel(model_code=str(stan_code))
    print("Compilation done")
    t2 = time.time()
    pystan_compilation_time = t2-t1

    t1 = time.time()
    fit_stan = sm1.sampling(data = data, iter=global_num_iterations, chains=global_num_chains, 
                            warmup = global_warmup_steps)

    t2 = time.time()
    theta = fit_stan.extract(permuted=True)['theta']

    theta = np.reshape(theta, (theta.shape[0], 1))

    t2 = time.time()
    return theta, t2-t1, pystan_compilation_time


if __name__ == "__main__":
    test_coin()
