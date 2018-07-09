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
    int batch_size;
    int <lower=0, upper=1> imgs[28,28,batch_size]; 
    int <lower=0, upper=10>  labels[batch_size];
}

network {
    MLP mlp with parameters:
        l1.weight;
        l1.bias;
        l2.weight;
        l2.bias;
}

prior
{
    mlp.l1.weight ~  Normal(0, 1);
    mlp.l1.bias ~ Normal(0, 1);
    mlp.l2.weight ~ Normal(0, 1);
}

guide_parameters
{
    real l1wloc;
    real l1wscale;
    real l1bloc;
    real l1bscale;
    real l2wloc;
    real l2wscale;
    real l2bloc;
    real l2bscale;
}

guide {
    l1wloc = randn(0,1);
    l1wscale = exp(randn());
    mlp.l1.weight ~  Normal(l1wloc, l1wscale);
    l1bloc = randn(0,1);
    l1bscale = exp(randn());
    mlp.l1.bias ~ Normal(l1bloc, l1bscale);
    l2wloc = randn(0,1);
    l2wscale = exp(randn());
    mlp.l2.weight ~ Normal(l2wloc, l2wscale);
    l2bloc = randn();
    l2bscale = exp(randn());
    mlp.l2.bias ~ Normal(l2bloc, l2bscale);
}

model {
    real logits[batch_size];
    logits = mlp(imgs);
    labels ~ Categorical(logits);
}


