import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def model():
    ___shape = {}
    if tensor(1.0) == tensor(0.0):
        a = tensor(1.0)