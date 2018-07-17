import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor
import pyro
from pyro import distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.autograd import Variable

import torch.utils.data.dataloader as dataloader
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST

import deepppl
import os


batch_size, nx, nh, ny = 128, 28 * 28, 1024, 10
def build_mlp():
    # Model

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.l1 = torch.nn.Linear(nx, nh)
            self.l2 = torch.nn.Linear(nh, ny)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            h = self.relu(self.l1(x.view((-1, nx))))
            yhat = self.l2(h)
            return F.log_softmax(yhat, dim=-1)

    return MLP()



def loadData():
    train = MNIST(os.environ.get("DATA_DIR", '.') + "/data", train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),  # ToTensor does min-max normalization.
    ]), )
    test = MNIST(os.environ.get("DATA_DIR", '.') + "/data", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),  # ToTensor does min-max normalization.
    ]), )
    dataloader_args = dict(shuffle=True, batch_size=batch_size,
                        num_workers=3, pin_memory=False)
    train_loader = dataloader.DataLoader(train, **dataloader_args)
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    return train_loader, test_loader

def CategoricalLogits(logits):
    return dist.Categorical(logits=logits)

def predict(data, posterior):
    predictions = [model(data) for model in posterior]
    prediction = torch.stack(predictions).mean(dim=0)
    return prediction.argmax(dim=-1)

def test_mlp_inference():
    mlp = build_mlp()
    train_loader, test_loader = loadData()
    model = deepppl.DppplModel(model_file = 'deepppl/tests/good/mlp.stan', mlp=mlp, CategoricalLogits=CategoricalLogits)
    svi = model.svi(params = {'lr' : 0.01})

    for epoch in range(2):  # loop over the dataset multiple times
        for j, (imgs, lbls) in enumerate(train_loader, 0):
            # calculate the loss and take a gradient step
            loss = svi.step(batch_size, imgs, lbls)
    posterior = svi.posterior(30)

    for j, data in enumerate(test_loader):
        images, labels = data
        accuracy = (predict(images, posterior) == labels).type(torch.float).mean()
        assert accuracy > 0.8