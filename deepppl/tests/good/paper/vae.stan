// Figure 5
networks {
  Decoder decoder;
  Encoder encoder;
}

data {
    int nz;
    int<lower=0, upper=1> x[28, 28];
}
parameters {
    real z[_];
}
model {
  real mu[_, _];
  z ~ normal(0, 1);
  mu = decoder(z);
  x ~ bernoulli(mu);
}

guide {
  real encoded[2, nz] = encoder(x);
  real mu_z[_] = encoded[1];
  real sigma_z[_] = encoded[2];
  z ~ normal(mu_z, sigma_z);
}