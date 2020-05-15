
import time
from typing import Any, Callable, ClassVar, Dict, Optional, List
from dataclasses import dataclass, field

import pystan
from deepppl import PyroModel, NumPyroModel

from scipy.stats import entropy, ks_2samp
import numpy as np


def _ks(s1, s2):
    s, p = ks_2samp(s1, s2)
    return {'statistic': s, 'pvalue': p}


def _distance(pyro_samples, stan_samples, dist):
    if len(pyro_samples.shape) == 1:
        return dist(stan_samples, pyro_samples)
    if len(pyro_samples.shape) == 2:
        res = {}
        for i, (p, s) in enumerate(zip(pyro_samples.T, stan_samples.T)):
            res[i] = dist(p, s)
        return res
    # Don't know what to compute here. Too many dimensions.
    return {}


def _compare(res, ref, compare_params, dist):
    divergence = {}
    for k, a in res.items():
        assert k in ref, \
            f'{k} is not in Stan results'
        b = ref[k]
        assert a.shape == b.shape, \
            f'Shape mismatch for {k}, Pyro {a.shape}, Stan {b.shape}'
        if not compare_params or k in compare_params:
            divergence[k] = _distance(a, b, dist)
    return divergence


@dataclass
class TimeIt:
    name: str
    timers: Dict[str, float]

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *exc_info):
        self.timers[self.name] = time.perf_counter() - self.start


@dataclass
class Config:
    iterations: int = 100
    warmups: int = 10
    chains: int = 1
    thin: int = 2


@dataclass
class MCMCTest:
    name: str
    model_file: str
    data: Dict[str, Any] = field(default_factory=dict)
    compare_params: Optional[List[str]] = None
    config: Config = Config()
    with_pyro: bool = True
    with_numpyro: bool = True

    pyro_samples: Dict[str, Any] = field(init=False)
    numpyro_samples: Dict[str, Any] = field(init=False)
    stan_samples: Dict[str, Any] = field(init=False)
    timers: Dict[str, float] = field(init=False, default_factory=dict)
    divergences: Dict[str, Any] = field(init=False, default_factory=dict)

    def run_pyro(self):
        assert self.with_pyro or self.with_numpyro, \
            'Should run either Pyro or Numpyro'
        if self.with_pyro:
            with TimeIt('Pyro_Runtime', self.timers):
                model = PyroModel(model_file=self.model_file)
                mcmc = model.mcmc(self.config.iterations,
                                  self.config.warmups,
                                  num_chains=self.config.chains,
                                  thin=self.config.thin)
                mcmc.run(**self.data)
                self.pyro_samples = mcmc.get_samples()
        if self.with_numpyro:
            with TimeIt('NumPyro_Runtime', self.timers):
                model = NumPyroModel(model_file=self.model_file)
                mcmc = model.mcmc(self.config.iterations,
                                  self.config.warmups,
                                  num_chains=self.config.chains,
                                  thin=self.config.thin)
                mcmc.run(**self.data)
                self.numpyro_samples = mcmc.get_samples()

    def run_stan(self):
        with TimeIt('Stan_Compilation', self.timers):
            mcmc = pystan.StanModel(file=self.model_file)
        with TimeIt('Stan_Runtime', self.timers):
            fit = mcmc.sampling(data=self.data,
                                iter=self.config.iterations,
                                chains=self.config.chains,
                                warmup=self.config.warmups,
                                thin=self.config.thin)
            self.stan_samples = fit.extract(permuted=True)

    def compare(self):
        self.divergences = {'pyro': {}, 'numpyro': {}}
        if self.with_pyro:
            self.divergences['pyro']['ks'] = _compare(self.pyro_samples,
                                                      self.stan_samples,
                                                      self.compare_params,
                                                      _ks)
        if self.with_numpyro:
            self.divergences['numpyro']['ks'] = _compare(self.numpyro_samples,
                                                         self.stan_samples,
                                                         self.compare_params,
                                                         _ks)

    def run(self) -> Dict[str, Dict[str, Any]]:
        self.run_pyro()
        self.run_stan()
        self.compare()
        return {
            'divergences': self.divergences,
            'timers': self.timers
        }
