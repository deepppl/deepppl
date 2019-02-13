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
int<lower=0> N;               // number of items
int<lower=0> M;               // number of predictors
int<lower=0,upper=1> y[N];           // outcomes
// row_vector[M] x[N];      // predictors
row_vector x[N, M];
}
parameters {
vector[M] beta;          // coefficients
}
model {
for (m in 1:M)
    beta[m] ~ cauchy(0.0, 2.5);

for (n in 1:N)
    y[n] ~ bernoulli(inv_logit(x[n] * beta));
}
