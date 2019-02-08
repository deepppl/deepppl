import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(N, Ngrps, grp_index, p):
    ___shape = {}
    ___shape['N'] = ()
    ___shape['p'] = ()
    ___shape['Ngrps'] = ()
    ___shape['grp_index'] = N
    ___shape['sigmaGrp'] = ()
    ___shape['muGrp'] = ()
    sigmaGrp = pyro.sample('sigmaGrp', dist.Uniform(0.0001, 100.0))
    muGrp = pyro.sample('muGrp', dist.Uniform(-100, 1000.0))
    ___shape['grpi'] = ()
    for i in range(1, N + 1):
        grpi = grp_index[i - 1]
        pyro.sample('p' + '{}'.format(i - 1) + '1', dist.LogisticNormal(
            muGrp[grpi - 1], sigmaGrp[grpi - 1]), obs=p[i - 1])
