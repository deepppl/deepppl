import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def model():
    ___shape = {}
    if 1.0 == 0.0:
        a = 1.0