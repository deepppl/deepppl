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
int<lower=1> K;
vector[N] y;
matrix[N,K] x;
}
parameters {
vector[K] beta;
}
transformed parameters {
real<lower=0> squared_error;
squared_error = dot_self(y - x * beta);
}
model {
target += -squared_error;
}
generated quantities {
real<lower=0> sigma_squared;
sigma_squared = squared_error / N;
}
