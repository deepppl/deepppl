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
  int<lower=0> N;
  vector[N] y;
  vector[N] sigma_y;
} 
parameters {
  vector[N] eta;
  real mu_theta;
  real<lower=0,upper=100> sigma_eta;
  real xi;
} 
transformed parameters {
  vector[N] theta;
  theta = mu_theta + xi * eta;
}
model {
  mu_theta ~ normal(0, 100);
  sigma_eta ~ inv_gamma(1, 1); //prior distribution can be changed to uniform

  eta ~ normal(0, sigma_eta);
  xi ~ normal(0, 5);
  y ~ normal(theta,sigma_y);
}