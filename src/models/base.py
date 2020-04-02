"""
base.py

Base Class for MNIST NN Model --> Implements all logic for downloading and processing datasets, implementing train
and validation logic, and handling optimization.
"""
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
import torch


class BaseMNIST(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseMNIST, self).__init__()

        # Save Hyper-Parameters
        self.hparams = hparams

        # Build Model
        self.build_model()

    # Data Loading
    def prepare_data(self):
        # Download Train and Test Data
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def train_dataloader(self):
        # Set Data Transforms
        mnist_mean, mnist_stdev = 0.1307, 0.3081
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mnist_mean,), (mnist_stdev,))])

        # Split Dataset into Train and Validation
        mnist_train = MNIST(self.hparams.data_dir, train=True, download=False, transform=transform)
        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

        # Create DataLoader
        return DataLoader(self.mnist_train, batch_size=self.hparams.bsz)

    def val_dataloader(self):
        # NOTE: Called after Train DataLoader
        return DataLoader(self.mnist_val, batch_size=self.hparams.bsz)

    def test_dataloader(self):
        # Set Data Transforms
        mnist_mean, mnist_stdev = 0.1307, 0.3081
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mnist_mean,), (mnist_stdev,))])

        # Create DataLoader
        mnist_test = MNIST(self.hparams.data_dir, train=False, download=False, transform=transform)
        return DataLoader(mnist_test, batch_size=self.hparams.bsz)

    # Training & Validation Steps
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)

        # Compute Loss and Accuracy
        loss = F.nll_loss(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {'loss': loss, 'acc': acc, 'log': {'train_loss': loss, 'train_acc': acc}}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)

        # Compute Loss and Accuracy
        loss = F.nll_loss(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {'val_loss': loss, 'val_acc': acc}

    def validation_end(self, outputs):
        # Outputs --> List of Individual Step Outputs!
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc, 'log': {'val_loss': avg_loss, 'val_acc': avg_acc}}

    # Optimizer
    def configure_optimizers(self):
        if self.hparams.opt == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.opt == 'sgd':
            opt = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        return opt

    def build_model(self):
        raise NotImplementedError("Needs to be implemented by sub-class!")

    def forward(self, x):
        raise NotImplementedError('Needs to be implemented by sub-class!')
