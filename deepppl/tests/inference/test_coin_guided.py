# Adapted from https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb
import torch
import pyro

import deepppl
import os
import numpy as np

def test_coin_guided_inference():
    model = deepppl.DppplModel(model_file = 'deepppl/tests/good/coin_guide.stan')
    svi = model.svi(params={'lr':0.1})
    x = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    for step in range(1000):
        svi.step(x)
        if step % 100 == 0:
            print('.', end='')
    alpha_q = pyro.param("alpha_q").item()
    beta_q = pyro.param("beta_q").item()

    # The posterior distribution should be a Beta(10 + 2, 10 + 8)
    assert np.abs(alpha_q - (10 + 2)) < 2.0
    assert np.abs(beta_q - (10 + 8)) < 2.0