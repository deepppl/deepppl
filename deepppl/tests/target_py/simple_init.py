import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model():
    ___shape = {}
    if 1.0 == 0.0:
        ___shape['a'] = ()
        a = 1.0
