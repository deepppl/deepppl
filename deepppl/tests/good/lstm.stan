/*
 * Copyright 2018 IBM Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


data {
    int n_characters;
    int input[n_characters];
    int target[n_characters];
}

network {
    RNN rnn with parameters:
        encoder.weight;
        gru.weight_ih_l0;
        gru.weight_hh_l0;
        gru.bias_ih_l0;
        gru.bias_hh_l0;
        decoder.weight;
        decoder.bias;
}

prior
{
    rnn.encoder.weight ~  Normal(0, 1);
    rnn.gru.weight_ih_l0 ~ Normal(0, 1);
    rnn.gru.weight_hh_l0 ~ Normal(0, 1);
    rnn.gru.bias_ih_l0 ~  Normal(0, 1);
    rnn.gru.bias_hh_l0 ~  Normal(0, 1);
    rnn.decoder.weight ~  Normal(0, 1);
    rnn.decoder.bias ~  Normal(0, 1);
}

guide_parameters
{
    real ewl;
    real ews;
    real gw1l;
    real gw1s;
    real gw2l;
    real gw2s;
    real gb1l;
    real gb1s;
    real gb2l;
    real gb2s;
    real dwl;
    real dws;
    real dbl;
    real dbs;

}

guide {
    ewl = randn();
    ews = exp(randn());
    rnn.encoder.weight ~  Normal(ewl, ews);
    gw1l = randn();
    gw1s = exp(randn());
    rnn.gru.weight_ih_l0 ~ Normal(gw1l, gw1s);
    gw2l = randn();
    gw2s = exp(randn());
    rnn.gru.weight_hh_l0 ~ Normal(gw2l, gw2s);
    gb1l = randn();
    gb1s = exp(randn());
    rnn.gru.bias_ih_l0 ~  Normal(gb1l, gb1s);
    gb2l = randn();
    gb2s = exp(randn());
    rnn.gru.bias_hh_l0 ~  Normal(gb2l, gb2s);
    dwl = randn();
    dws = exp(randn());
    rnn.decoder.weight ~  Normal(dwl, dws);
    dbl = randn();
    dbs = exp(randn());
    rnn.decoder.bias ~  Normal(dbl, dbs);
}

model {
    int logits[n_characters];
    logits = rnn(input);
    target ~ Categorical(logits);
}


