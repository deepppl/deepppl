import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def model():
    ___shape = {}
    theta = pyro.sample('theta', ImproperUniform())
    pyro.sample('expr' + '1', 
                dist.Exponential(tensor(1.0)), 
                obs=-(-tensor(0.5) * (theta - tensor(1000.0)) * (theta - tensor(1000.0))))