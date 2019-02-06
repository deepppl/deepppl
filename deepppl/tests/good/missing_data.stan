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
int<lower=0> N_obs;
int<lower=0> N_mis;
real y_obs[N_obs];
}
parameters {
real mu;
real<lower=0> sigma;
real y_mis[N_mis];
}
model {
y_obs ~ normal(mu, sigma);
y_mis ~ normal(mu, sigma);
}
