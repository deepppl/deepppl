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


networks {
    Decoder decoder;
    Encoder encoder;
}

data {
    int nz;
    int<lower=0, upper=1> x[28, 28];
}

parameters {
    real z[_];
}

model {
    real mu[_, _];
    z ~ normal(0, 1);
    mu = decoder(z);
    x ~ bernoulli(mu);
}

guide {
    real encoded[2, _] = encoder(x);
    real mu_z[_] = encoded[1];
    real sigma_z[_] = encoded[2];
    z ~ normal(mu_z, sigma_z);
}
