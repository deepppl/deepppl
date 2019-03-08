data {
}
parameters {
  real cluster;
  real theta;
}
model {
  real mu;
  cluster ~ normal(0.0, 1.0);
  if (cluster > 0.0)
  {
    mu = 2.0;
  }
  else
  {
    mu = 0.0;
  }
  theta ~ normal(mu, 1.0);
}