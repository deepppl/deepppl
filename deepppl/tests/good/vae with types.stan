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
    int nz; int batch_size;
    int<lower=0, upper=1> x[28, 28, batch_size];
}

parameters {
    real z[nz, batch_size];
}

model {
    real mu[28, 28, batch_size];
    # zeros : 'a -> [dimension='a, component='real]
    # Normal : [dimension='a, component='b = real/int] -> [dimension='a, component='b'] -> [dimension='a, component='b]
    # Normal : [dimension='a, component='b = real/int] -> [dimension='a, component='b'] -> int -> [dimension=int, component=[dimension='a, component='b]]
    # TODO: implement matrix for the second one.


    # Normal : 
    z ~ Normal(zeros(nz), ones(nz), batch_size);
    mu = decoder(z);
    x ~ Bernoulli(mu);
}

guide {
    real encoded[2, nz, batch_size] = encoder(x);
    real mu_z[nz, batch_size] = encoded[1];
    real sigma_z[nz, batch_size] = encoded[2];
    z ~ Normal(mu_z, sigma_z);
}
