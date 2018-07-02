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
    int x;
}



parameters {
    int latent;
}

network {
    Decoder decoder;
    Encoder encoder;
}

guide {
    real encoded[2];
    real mu;
    real sigma;
    encoded = encoder(x);
    mu = encoded[1];
    sigma = encoded[2];
    latent ~ normal(mu, sigma);
}

model {
    int loc_img;
    latent ~ Normal(0, 1);
    loc_img = decoder(latent);
    x ~ Bernoulli(loc_img);
}


