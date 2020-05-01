import torch
import pyro
import numpy as np
import logging
import time
import pystan
import deepppl
import os
import pandas as pd

stan_model_file = 'deepppl/tests/good/double_gaussian.stan'
global_num_iterations=10000
global_num_chains=1
global_warmup_steps = 300

def test_double_gaussian():
    model = deepppl.PyroModel(model_file=stan_model_file)
    mcmc = model.mcmc(global_num_iterations, global_warmup_steps)

    t1 = time.time()
    mcmc.run()
    samples_fstan = mcmc.get_samples()['theta']
    t2 = time.time()

    pystan_output, pystan_time, pystan_compilation_time = compare_with_stan_output()

    assert samples_fstan.shape == pystan_output.shape
    from scipy.stats import entropy, ks_2samp
    hist1 = np.histogram(samples_fstan, bins = 10)
    hist2 = np.histogram(pystan_output, bins = hist1[1])
    kl = entropy(hist1[0]+1, hist2[0]+1)
    skl = kl + entropy(hist2[0]+1, hist1[0]+1)
    ks = ks_2samp(samples_fstan, pystan_output)
    print('skl for theta is:{:.6f}'.format(skl))
    print(f'kolmogorov-smirnov for theta is: {ks}')

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

    t2 = time.time()
    return theta, t2-t1, pystan_compilation_time


if __name__ == "__main__":
    test_double_gaussian()
