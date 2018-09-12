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

networks {
    MLP mlp with parameters:
        l1.weight;
        l1.bias;
        l2.weight;
        l2.bias;
}

prior
{
    mlp.l1.weight ~  Normal(zeros(mlp.l1.weight$shape), ones(mlp.l1.weight$shape));
    mlp.l1.bias ~ Normal(zeros(mlp.l1.bias$shape), ones(mlp.l1.bias$shape));
    mlp.l2.weight ~ Normal(zeros(mlp.l2.weight$shape), ones(mlp.l2.weight$shape));
    mlp.l2.bias ~  Normal(zeros(mlp.l2.bias$shape), ones(mlp.l2.bias$shape));
}

variational parameters
{
    real l1wloc[mlp.l1.weight$shape];   
    real l1wscale[mlp.l1.weight$shape];
    real l1bloc[mlp.l1.bias$shape];
    real l1bscale[mlp.l1.bias$shape];
    real l2wloc[mlp.l2.weight$shape];
    real l2wscale[mlp.l2.weight$shape];
    real l2bloc[mlp.l2.bias$shape];
    real l2bscale[mlp.l2.bias$shape];
}

guide {
    l1wloc = randn(l1wloc$shape);
    l1wscale = exp(randn(l1wscale$length)); // <--- should be shape
    mlp.l1.weight ~  Normal(l1wloc, l1wscale);
    l1bloc = randn(l1bloc$shape);
    l1bscale = exp(randn(l1bscale$shape));
    mlp.l1.bias ~ Normal(l1bloc, l1bscale);
    l2wloc = randn(l2wloc$shape);
    l2wscale = exp(randn(l2wscale$shape));
    mlp.l2.weight ~ Normal(l2wloc, l2wscale);
    l2bloc = randn(l2bloc$shape);
    l2bscale = exp(randn(l2bscale$shape));
    mlp.l2.bias ~ Normal(l2bloc, l2bscale);
}

model {
    real logits[batch_size];
    logits = mlp(imgs);
    labels ~ Categorical(logits);
}


