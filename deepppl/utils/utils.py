# /*
#  * Copyright 2018 IBM Corporation
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  * http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
#  */

from pyro import distributions as dist
from torch.distributions import constraints
import torch

# Utils to be imported by DppplModel


def categorical_logits(logits):
    return dist.Categorical(logits=logits)


def bernoulli_logit(logits):
    return dist.Bernoulli(logits=logits)


def binomial_logit(n, logits):
    return dist.Binomial(n, logits=logits)


def poisson_log(alpha):
    return dist.Poisson(torch.exp(alpha))


class ImproperUniform(dist.Normal):
    def __init__(self, shape=None):
        zeros = torch.zeros(shape) if shape else 0
        ones = torch.ones(shape) if shape else 1
        super(ImproperUniform, self).__init__(zeros, ones)

    def log_prob(self, x):
        return x.new_zeros(x.shape)


class LowerConstrainedImproperUniform(ImproperUniform):
    def __init__(self, lower_bound=0, shape=None):
        self.lower_bound = lower_bound
        super(LowerConstrainedImproperUniform, self).__init__(shape)
        self.support = constraints.greater_than(lower_bound)

    def log_prob(self, x):
        return x.new_zeros(x.shape)

    def sample(self):
        s = dist.Uniform(self.lower_bound, self.lower_bound + 2).sample()
        return s


class UpperConstrainedImproperUniform(ImproperUniform):
    def __init__(self, upper_bound=0, shape=None):
        self.upper_bound = upper_bound
        super(UpperConstrainedImproperUniform, self).__init__(shape)
        self.support = constraints.less_than(upper_bound)

    def log_prob(self, x):
        return x.new_zeros(x.shape)

    def sample(self):
        s = dist.Uniform(self.upper_bound - 2, self.upper_bound).sample()
        return s


def log(x):
    if isinstance(x, torch.Tensor):
        return torch.log(x)
    else:
        return torch.log(torch.tensor(x).float())


def dot_self(x):
    return torch.dot(x, x)


def log_sum_exp(x):
    return torch.logsumexp(x, 0)


def inv_logit(p):
    return torch.log(p / (1. - p))


hooks = {x.__name__: x for x in [
    bernoulli_logit,
    categorical_logits,
    binomial_logit,
    poisson_log,
    ImproperUniform,
    LowerConstrainedImproperUniform,
    UpperConstrainedImproperUniform,
    log,
    dot_self,
    log_sum_exp,
    inv_logit]

}
