import torch
from torch import tensor, randn
import pyro
import pyro.distributions as dist


def model():
    ___shape = {}
    if 1.0 == 0.0:
        ___shape['a'] = ()
        a = 1.0