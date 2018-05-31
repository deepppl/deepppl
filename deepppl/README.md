# Dependencies

With python 3 on MacOS

```
pip install torch torchvision
brew install antlr
pip install antlr4-python3-runtime
```

# Test

```
python test.py tests/good/coin.stan
```

Output:
```
import torch
from torch.distributions import Uniform, Bernoulli

# Data
x = torch.zeros([10])

# Parameters
theta = torch.zeros([])

# Model
theta = Uniform(0,1).sample()
for i in range(1,10):
    x[i] = Bernoulli(theta).sample()
```