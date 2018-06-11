# Dependencies

With python 3 on MacOS

```
pip install torch torchvision
brew install antlr
pip install antlr4-python3-runtime
pip install astpretty astor
pip install ipdb
```

# Test

```
python dpplc.py tests/good/coin.stan
```

Input:
```stan
data {
  int<lower=0,upper=1> x[10];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ uniform(0,1);
  for (i in 1:10)
    x[i] ~ bernoulli(theta);
}
```

Output:
```python
import torch
from torch.distributions import *
x = torch.zeros([10])
theta = torch.zeros([])
theta = Uniform(0, 1).sample()
for i in range(1 - 1, 10):
    x[i] = Bernoulli(theta).sample()
```
