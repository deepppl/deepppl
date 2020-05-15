# DeepPPL

This directory contains the code associated to the paper entitled _Compiling Stan to Generative Probabilistic Languages_.
It contains a compiler for Stan an extended version of Stan with variational inference and deep probabilistic programming to Pyro.

## Installation

```
cd deepppl
make
pip install -r requirements.txt
cd ..
pip install .
```

To create a symbolic link to the code use: `pip install -e ..` instead


## Try

Some examples of Python notebooks using the compiler are available in the directory `deepppl/examples`. To launch the notebooks use the following command:

```
jupyter notebook
```

To compile a Stan file and look at the generate Python code, you can use the following command:

```
python -m deepppl.dpplc --print --noinfer deepppl/tests/good/coin.stan
```


## Experiments: Comparison Stan/DeepStan

The directory `logs` contains the logs for the experiments used in the paper.
The logs are in the following formats:

```python
{
    'divergences': { # distance between stan and deepstan results
        'numpyro': { # with numpyro runtime
            'ks': { # 2 samples KS test
                param1: (KS, p_value)
                param2: (KS, p_value)
            },
            'skl': { # Symmetric KL
                param1: skl_value
                param2: skl_value
            }
        }
    }
        'pyro': { # with pyro runtime
            'ks': { # 2 samples KS test
                param1: (KS, p_value)
                param2: (KS, p_value)
            },
            'skl': { # Symmetric KL
                param1: skl_value
                param2: skl_value
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

### Run the experiments

To reproduce the experiments run the following command:

```
python -m deepppl.tests.inference.experiments
```

This will launch in parallel 5 runs of all experiments.
Note that skl values are computed on histograms with 10 bins (hence the infinite values).

Optional:
- change `logdir` variable
- change the configuration in `deepppl/tests/inference/harness.py` (`Config` class)


### Building a summary

To build a summary of the experiments, run the following command:

```
python parser.py
```

This will return a LaTex table with the execution time information for Stan, DeepStan/Pyro, and DeepStan/NumPyro, and the max KS across all the parameters (with the associated parameter name).
Everything is averaged over all the runs (typically 5).
