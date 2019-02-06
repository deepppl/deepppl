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

functions {

vector foo(int d) {
    vector[3] result = [10.1, 11*3.0, d]';
    return result;
}

row_vector bar() {
    row_vector[2] result = [7, 8];
    return result;
}

}
data {
real x;
real y;
}
transformed data {
vector[3] td_v1 = [ 21, 22, 23]';
row_vector[2] td_rv1 = [ 1, 2];
td_rv1 = [ x, y];
td_rv1 = [ x + y, x - y];
td_rv1 = [ x^2, y^2];
td_v1 = foo(1);
td_rv1 = bar();
}
parameters {
real z;
}
transformed parameters {
vector[3] tp_v1 = [ 41, 42, 43]';
row_vector[2] tp_rv1 = [ 1, x];
tp_v1 = foo(1);
tp_v1 = [ 51, y, z]';
tp_rv1 = [ y, z];
tp_rv1 = bar();
}
model {
z ~ normal(0,1);
}
generated quantities {
vector[3] gq_v1 = [1, x, y]';
row_vector[3] gq_rv1 = [1, x, y];
row_vector[3] gq_rv2 = [1, x, z];
gq_v1 = foo(1);
}
