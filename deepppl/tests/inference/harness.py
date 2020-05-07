import time
from typing import Any, Callable, ClassVar, Dict, Optional, List
from dataclasses import dataclass, field

import pystan
from deepppl import PyroModel, NumPyroModel

from scipy.stats import entropy, ks_2samp
import numpy as np

def _skl(s1, s2, bins=10):
    hist1 = np.histogram(s1, bins) + 1
    hist2 = np.histogram(s2, bins) + 1
    return entropy(hist1[0], hist2[0]) + entropy(hist2[0], hist1[0])

def _distance(dist, pyro_samples, stan_samples):
    if len(pyro_samples.shape) == 1:
        return dist(stan_samples, pyro_samples)
    else:
        res = {}
        for i, (p, s) in enumerate(zip(pyro_samples.T, stan_samples.T)):
            res[i] = dist(p, s)
        return res
    


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
 
    
@dataclass
class MCMCTest:
    name: str
    model_file: str
    data: Dict[str, Any] = field(default_factory=dict) 
    compare_params: Optional[List[str]] = None
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
        if self.with_numpyro:
            with TimeIt('NumPyro_Runtime', self.timers):
                model = NumPyroModel(model_file=self.model_file)
                mcmc = model.mcmc(Config.iterations, Config.warmups)
                mcmc.run(**self.data)
        if self.with_pyro:
            with TimeIt('Pyro_Runtime', self.timers):
                model = PyroModel(model_file=self.model_file)
                mcmc = model.mcmc(Config.iterations, Config.warmups)
                mcmc.run(**self.data)
        self.pyro_samples = mcmc.get_samples()
                    
            
    def run_stan(self):
        with TimeIt('Stan_Compilation', self.timers):
            mcmc = pystan.StanModel(file=self.model_file)
        with TimeIt('Stan_Runtime', self.timers):
            fit = mcmc.sampling(data=self.data,
                                iter=Config.iterations,
                                chains=Config.chains, 
                                warmup=Config.warmups)
            self.stan_samples = fit.extract(permuted=True)
        

    def compare(self):
        for k in self.pyro_samples:
            assert k in self.stan_samples, \
                f'{k} is not in stan samples'
            p = self.pyro_samples[k]
            s = self.stan_samples[k]
            assert p.shape == s.shape, \
                f'Shape mismatch for {k}, Pyro {p.shape}, Stan {s.shape}'
            if not self.compare_params or k in self.compare_params:
                self.divergences[k] = _distance(ks_2samp, p, s)
            
    def run(self) -> Dict[str, Dict[str, Any]]:
        self.run_pyro()
        self.run_stan()
        self.compare()
        return {
            'divergences': self.divergences,
            'timers': self.timers
        }


