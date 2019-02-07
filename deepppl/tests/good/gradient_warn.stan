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
    int N;
    vector[N] p;
    int Ngrps;
    int<lower=1, upper=Ngrps> grp_index[N];
}
parameters {
    vector<lower=0.0001, upper=100>[Ngrps] sigmaGrp;
    vector<lower=(-100), upper=1000>[Ngrps] muGrp;
}

model {
    int grpi;
    for (i in 1:N){
        grpi = grp_index[i];
        p[i] ~ logistic(muGrp[grpi], sigmaGrp[grpi]);
    }
}
