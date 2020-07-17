# DeepPPL

This directory contains the code associated to the paper entitled _Compiling Stan to Generative Probabilistic Languages_.
It contains a compiler for Stan an extended version of Stan with variational inference and deep probabilistic programming to Pyro.

## Installation

DeepPPL requires python 3.7, and ANTLR 4.7

```
$ make -C deepppl
$ pip install -r requirements.txt
$ pip install .
```

## Try

Notebook examples are available in the directory `examples`. 
To launch the notebooks use the following command:

```
$ jupyter notebook
```

To compile a Stan file and look at the generated Python code, you can use the following command:

```
$ python -m deepppl.dpplc --print --noinfer deepppl/tests/good/coin.stan
```


## Experiments: Comparison Stan/DeepStan

To reproduce the experiments comparing Stan and DeepStan run the `experiments` script.

```
$ python -m deepppl.tests.inference.experiments --help
usage: experiments.py [-h] [--logdir LOGDIR] [--iterations ITERATIONS]
                      [--warmups WARMUPS] [--thin THIN] [--runs N_RUNS]
                      [--no-run]

Compare the output of NUTS for DeepStan (with Pyro and Numpyro) and Stan on
the following experiments: coin, double normal, reparameterization, linear
regression, aspirin, roaches, 8 schools, seeds

optional arguments:
  -h, --help            show this help message and exit
  --logdir LOGDIR       Directory name to store the results
  --iterations ITERATIONS
                        Total number of iterations
  --warmups WARMUPS     Number of warmup steps (included in iterations)
  --thin THIN           Thining factor
  --runs N_RUNS         Number of run for each experiment
  --no-run              Analyse logdir without re-running the experiments
```

The default configuration is the one described in Section 6.1. and corresponds to:
```
--iterations 10000 --warmups 1000 --runs 10 --thin 10 --logdir logs
```

:warning: With the default configuration experiments may take a while to complete depending on your hardware.
Consider decreasing the number of iterations, warmup steps, and runs.

When the experiments complete, the directory specified with `logdir` (default `logs`) contains one file for each run of each experiments in the following format:

```python
{
    'divergences': { # distance between stan and deepstan results
        'numpyro': { # with numpyro runtime
            'ks': { # 2 samples KS test
                param1: {'statistic': ..., 'pvalue': ...}
                param2: {'statistic': ..., 'pvalue': ...}
            }
        }
    }
        'pyro': { # with pyro runtime
            'ks': { # 2 samples KS test
                param1: {'statistic': ..., 'pvalue': ...}
                param2: {'statistic': ..., 'pvalue': ...}
            }
        }
    }
    'timers': {
        'NumPyro_Runtime': ...,
        'Pyro_Runtime': ...,
        'Stan_Compilation': ...,
        'Stan_Runtime': ...
    }
}
```

The `experiments` script outputs a summary, averaging the results across all the log files.
The logs of our experiments are located in `logs_evaluation`.
To generate the summary:

```
$ python -m deepppl.tests.inference.experiments --no-run --log-dir logs_evaluation
```