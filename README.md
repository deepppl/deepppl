[![Build Status](https://travis-ci.com/deepppl/deepppl.svg?branch=master)](https://travis-ci.com/deepppl/deepppl)
<img src="logo/logo.jpg" alt="logo" width="55px"/>

# DeepPPL
Deep Probabilistic Programming Language

Paper: https://arxiv.org/abs/1810.00873

## Installation
```
git clone https://github.com/deepppl/deepppl.git
cd deepppl/deepppl
make
pip install -r requirements.txt
pip install ..
```

To create a symbolic link to the code use: `pip install -e ..` instead

## Tests

Launch tests:
```
pytest -v
```

Tests without doing inference:
```
pytest -v -k 'not inference'
```

Compile a file:
```
python -m deepppl.dpplc --print --noinfer deepppl/tests/good/coin.stan
```
