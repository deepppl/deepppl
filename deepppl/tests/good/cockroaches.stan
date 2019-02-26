data {
    int N;
    real exposure2[N];
    real roach1[N];
    real senior[N];
    real treatment[N];
    int y[N];
}
transformed data {
    real log_expo[N];
    log_expo=log(exposure2);
}
parameters {
    real beta[4];
    real lmbda[N];
    real tau;
}
transformed parameters {
    real sigma = 1.0 / sqrt(tau);
}
model {
    tau ~ gamma(0.001, 0.001);
    lmbda ~ normal(0, sigma);
    y ~ poisson_log(log_expo + beta[1]
        + beta[2] * roach1
        + beta[3] * treatment
        + beta[4] * senior
        + lmbda);
}