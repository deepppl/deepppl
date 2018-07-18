import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor
import pyro
from pyro import distributions as dist
import numpy as np

import torch.utils.data.dataloader as dataloader
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST

import deepppl
import os

side = 28
batch_size, nx, nh, nz = 256, side * side, 1024, 4
def build_vae():
    # Model

    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.lh = nn.Linear(nz, nh)
            self.lx = nn.Linear(nh, nx)

        def forward(self, z):
            hidden = F.relu(self.lh(z))
            mu = self.lx(hidden)
            return F.sigmoid(mu.view(-1, 1, side, side))

    # define the PyTorch module that parameterizes the
    # diagonal gaussian distribution q(z|x)
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.lh = torch.nn.Linear(nx, nh)
            self.lz_mu = torch.nn.Linear(nh, nz)
            self.lz_sigma = torch.nn.Linear(nh, nz)
            self.softplus = nn.Softplus()

        def forward(self, x):
            x = x.view((-1, nx))
            hidden = F.relu(self.lh(x))
            z_mu = self.lz_mu(hidden)
            z_sigma = self.softplus(self.lz_sigma(hidden))
            return z_mu, z_sigma

    return Encoder(), Decoder()



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

def distance(x, y):
    return torch.norm(x - y, dim=1).data.numpy()


class Classifier:
    def __init__(self, encoder, train_loader):
        self.encoder = encoder
        images, labels = iter(train_loader).next()
        print('Classifier with {} labeled data'.format(len(labels)))
        classes = [[] for i in range(10)]
        for img, lbl in zip(images, labels):
            z = encoder(img)[0]
            classes[lbl].append(z)
        ## compute centroid for each classes
        self.classes = [torch.stack(cls).mean(dim=0) for cls in classes]
 
    def classify(self, img):
        z = self.encoder(img)[0]
        ## pick classes according to the minimum euclidean distance
        distances = [distance(z, self.classes[i]) for i in range(10)]
        return np.stack(distances).argmin(0)


def test_vae_inference():
    encoder, decoder = build_vae()
    train_loader, test_loader = loadData()
    model = deepppl.DppplModel(
                                model_file = 'deepppl/tests/good/vae.stan', 
                                encoder=encoder,
                                decoder=decoder)
    svi = model.svi(params = {'lr' : 0.01})

    for epoch in range(4):  # loop over the dataset multiple times
        for j, (imgs, _) in enumerate(train_loader, 0):
            # calculate the loss and take a gradient step
            k = len(imgs)
            loss = svi.step(k, nz, imgs)
    classifier = Classifier(encoder, train_loader)
    img, lbls = iter(test_loader).next()
    accuracy = (lbls.data.numpy() == classifier.classify(img)).mean()
    assert accuracy > 0.15