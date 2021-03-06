
networks {
    MLP mlp;
}

data {
    int batch_size;
    int <lower=0, upper=1> imgs[28,28,batch_size];
    int <lower=0, upper=10>  labels[batch_size];
}

parameters {
    real[] mlp.l1.weight;
    real[] mlp.l1.bias;
    real[] mlp.l2.weight;
    real[] mlp.l2.bias;
}

model {
    real logits[batch_size];
    mlp.l1.weight ~  normal(zeros(mlp.l1.weight$shape), ones(mlp.l1.weight$shape));
    mlp.l1.bias ~ normal(zeros(mlp.l1.bias$shape), ones(mlp.l1.bias$shape));
    mlp.l2.weight ~ normal(zeros(mlp.l2.weight$shape), ones(mlp.l2.weight$shape));
    mlp.l2.bias ~  normal(zeros(mlp.l2.bias$shape), ones(mlp.l2.bias$shape));

    logits = mlp(imgs);
    labels ~ categorical(logits);
}

guide parameters {
    real l1wloc[mlp.l1.weight$shape];
    real l1wscale[mlp.l1.weight$shape];
    real l1bloc[mlp.l1.bias$shape];
    real l1bscale[mlp.l1.bias$shape];
    real l2wloc[mlp.l2.weight$shape];
    real l2wscale[mlp.l2.weight$shape];
    real l2bloc[mlp.l2.bias$shape];
    real l2bscale[mlp.l2.bias$shape];
}

guide {
    l1wloc = randn(l1wloc$shape);
    l1wscale = exp(randn(l1wscale$length)); // <--- should be shape
    mlp.l1.weight ~  normal(l1wloc, l1wscale);
    l1bloc = randn(l1bloc$shape);
    l1bscale = exp(randn(l1bscale$shape));
    mlp.l1.bias ~ normal(l1bloc, l1bscale);
    l2wloc = randn(l2wloc$shape);
    l2wscale = exp(randn(l2wscale$shape));
    mlp.l2.weight ~ normal(l2wloc, l2wscale);
    l2bloc = randn(l2bloc$shape);
    l2bscale = exp(randn(l2bscale$shape));
    mlp.l2.bias ~ normal(l2bloc, l2bscale);
}
