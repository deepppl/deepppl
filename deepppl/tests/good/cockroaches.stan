data {
  int<lower=0> N; 
  vector[N] exposure2;
  vector[N] roach1;
  vector[N] senior;
  vector[N] treatment;
  int y[N];
}
transformed data {
  vector[N] log_expo;
  vector[N] sqrt_roach;

  log_expo = log(exposure2);
  sqrt_roach = sqrt(roach1);
}
parameters {
  vector[4] beta;
} 
model {
  beta[1] ~ normal(0, 5);
  beta[2] ~ normal(0, 2.5);
  beta[3] ~ normal(0, 2.5);
  beta[4] ~ normal(0, 2.5);
  y ~ poisson_log(log_expo + beta[1] + beta[2] * sqrt_roach + beta[3] * treatment
                  + beta[4] * senior);
}