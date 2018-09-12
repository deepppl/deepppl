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
  int<lower=0,upper=1> x[10];
}
parameters {
  real<lower=0,upper=1> theta;
}


variational parameters
{
  real<lower=0>  alpha_q;
  real<lower=0>  beta_q;
}

guide {

  alpha_q = 15.0;
  beta_q = 15.0;
  theta ~ beta(alpha_q, beta_q);
  for (i in 1:10)
    x[i] ~ bernoulli(theta);    // <---
}

model {
  theta ~ beta(10.0, 10.0);
  for (i in 1:10)
    x[i] ~ bernoulli(theta);
}
