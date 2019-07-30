data {
    int I;
    int n[I];
    int N[I];
    real x1[I];
    real x2[I];
}
transformed data {
    real x1x2[I];
    x1x2 = x1 .* x2;
}
parameters {
    real alpha0;
    real alpha1;
    real alpha12;
    real alpha2;
    real<lower=0> tau;
    real b[I];
}
transformed parameters {
    real sigma = 1.0 / sqrt(tau);
}
model {
    alpha0 ~ normal(0.0,1000);
    alpha1 ~ normal(0.0,1000);
    alpha2 ~ normal(0.0,1000);
    alpha12 ~ normal(0.0,1000);
    tau ~ gamma(0.001,0.001);

    b ~ normal(0.0, sigma);
    n ~ binomial_logit (N, alpha0
            + alpha1 * x1
            + alpha2 * x2
            + alpha12 * x1x2 + b);
}
