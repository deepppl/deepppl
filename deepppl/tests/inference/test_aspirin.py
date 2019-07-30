
# Aspirin example from https://jrnold.github.io/bugs-examples-in-stan/aspirin.html#non-centered-parameterization
from matplotlib import pyplot as plt
from collections import defaultdict
import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist
from deepppl.utils.utils import *
from pyro.infer import mcmc, EmpiricalMarginal
from pyro.infer.mcmc import MCMC, NUTS
from torch import zeros, ones, sqrt
import pandas as pd
import numpy as np
from functools import partial
import pystan
import deepppl
import astor
from deepppl.tests.utils import skip_on_travis
from deepppl import dpplc
import time

def create_aspirin_data():
  aspirin = {'y' : np.array([2.77, 2.50, 1.84, 2.56, 2.31, -1.15]),
           'sd' : np.array([1.65, 1.31, 2.34, 1.67, 1.98, 0.90])}

  aspirin_data = {
    'y' : aspirin['y'],
    'N' : len(aspirin['y']),
    's' : aspirin['sd'],
    'mu_loc' : np.mean(aspirin['y']),
    'mu_scale' : 5 * np.std(aspirin['y']),
    'tau_scale' : 2.5 * np.std(aspirin['y']),
    'tau_df' : 4}
  aspirin_data_t = {k:torch.tensor(v).float() for (k,v) in aspirin_data.items()}
  aspirin_data_t['N'] = int(aspirin_data_t['N']) #this must be an integer
  return aspirin_data, aspirin_data_t




deep_stan_code = """
data {
  int N;
  real y[N];
  real s[N];
  real mu_loc;
  real mu_scale;
  real tau_scale;
  real tau_df;
}
parameters {
  real theta_raw[N];
  real mu;
  real tau;
}
transformed parameters {
  real theta[N];
  theta = tau * theta_raw + mu;
}
model {
  mu ~ normal(mu_loc, mu_scale);
  tau ~ student_t(tau_df, 0., tau_scale);
  theta_raw ~ normal(0., 1.);
  y ~ normal(theta, s);
}
generated quantities {
  real shrinkage[N];
  if (1==1) {
    real tau2;
    tau2 = pow(tau, 2.);
    for (i in 1:N) {
      real v;
      v = pow(s[i], 2);
      shrinkage[i] = v / (v + tau2);
    }
  }
}
"""

global_num_iterations=10000
global_num_chains=1
global_warmup_steps = 3000

def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model)
    return mcmc.MCMC(nuts_kernel, **kwargs)

def build_aspirin_df(samples):
    names= [f'theta[{i+1}]' for i in range(6)] + [f'shrinkage[{i+1}]' for i in range(6)]
    df = pd.concat([samples['theta'], samples['shrinkage']], axis=1, ignore_index=True)
    df.columns = names
    return df

@skip_on_travis
def test_aspirin():
    model = deepppl.DppplModel(model_code=deep_stan_code)
    aspirin_data, aspirin_data_t = create_aspirin_data()

    t1 = time.time()
    posterior = model.posterior(
        method=nuts,
        num_samples=global_num_iterations-global_warmup_steps,
        warmup_steps=global_warmup_steps).run(**aspirin_data_t)
    
    samples = model.run_generated(posterior, **aspirin_data_t)
    t2 = time.time()
    df = build_aspirin_df(samples)
    pystan_output, time_pystan, pystan_compilation_time = compare_with_stan_output(aspirin_data)
    assert df.shape == pystan_output.shape
    from scipy.stats import entropy, ks_2samp
    for column in df.columns:
        #import pdb;pdb.set_trace()
        hist1 = np.histogram(df[column], bins = 10)
        hist2 = np.histogram(pystan_output[column], bins = hist1[1])
        kl = entropy(hist1[0]+1, hist2[0]+1)
        skl = kl + entropy(hist2[0]+1, hist1[0]+1)
        ks = ks_2samp(df[column], pystan_output[column])
        print('skl for column:{} is:{:.6f}'.format(column, skl))
        print(f'kolmogorov-smirnov for column:{column} is: {ks}')

    print("Time taken: deepstan:{:.2f}, pystan_compilation:{:.2f}, pystan:{:.2f}".format(t2-t1, pystan_compilation_time, time_pystan))
def compare_with_stan_output(data):
    stan_code = """
    data {
      int N;
      vector[N] y;
      vector[N] s;
      real mu_loc;
      real mu_scale;
      real tau_scale;
      real tau_df;
    }
    parameters {
      vector[N] theta_raw;
      real mu;
      real tau;
    }
    transformed parameters {
      vector[N] theta;
      theta = tau * theta_raw + mu;
    }
    model {
      mu ~ normal(mu_loc, mu_scale);
      tau ~ student_t(tau_df, 0., tau_scale);
      theta_raw ~ normal(0., 1.);
      y ~ normal(theta, s);
    }
    generated quantities {
      vector[N] shrinkage;
      {
        real tau2;
        tau2 = pow(tau, 2.);
        for (i in 1:N) {
          real v;
          v = pow(s[i], 2);
          shrinkage[i] = v / (v + tau2);
        }
      }
    }
    """
    # Compile and fit
    t1 = time.time()
    sm1 = pystan.StanModel(model_code=str(stan_code))
    t2 = time.time()
    pystan_compilation_time = t2 - t1
    t1 = time.time()
    fit_stan = sm1.sampling(data=data, iter=global_num_iterations, chains=global_num_chains, warmup = global_warmup_steps)

    samples = fit_stan.extract(permuted=True)
    df = build_aspirin_df({k : pd.DataFrame(v) for k,v in samples.items()})
    t2 = time.time()
    return df, t2-t1, pystan_compilation_time

if __name__ == "__main__":
    test_aspirin()
