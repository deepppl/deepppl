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
  log_expo = log(exposure2);
}
parameters {
  vector[4] beta;
  vector[N] lambda_;
  real<lower=0> tau;
} 
transformed parameters {
  real<lower=0> sigma;

  sigma = 1.0 / sqrt(tau);
}
model {
  tau ~ gamma(0.001, 0.001);
  lambda_ ~ normal(0, sigma);
   y ~ poisson_log(lambda_ + log_expo 
                   + beta[1] 
                   + beta[2] * roach1 
                   + beta[3] * senior 
                   + beta[4] * treatment);
}