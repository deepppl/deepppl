import torch
import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor
import pyro
from pyro import distributions as dist
import numpy as np
import deepppl
import os
import pandas as pd

def test_gaussian_log_density_inference():
    model = deepppl.PyroModel(model_file = 'deepppl/tests/good/gaussian_log_density.stan')

    mcmc = model.mcmc(num_samples=3000, warmup_steps=300)
    mcmc.run()
    series = mcmc.get_samples()['theta']

    assert np.abs(series.mean() - 1000) < 1
    assert np.abs(series.std() - 1.0) < 0.1
    

if __name__ == "__main__":
    test_gaussian_log_density_inference()

