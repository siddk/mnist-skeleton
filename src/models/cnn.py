"""
cnn.py

Implements the MNIST Convolutional Neural Network Initialization and Forward Pass Logic.
"""
from models.base import BaseMNIST

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(BaseMNIST):
    def build_model(self):
        # Create Layers
        self.conv1 = nn.Conv2d(1, self.hparams.cnn_conv1, kernel_size=self.hparams.kernel_size)
        self.conv2 = nn.Conv2d(self.hparams.cnn_conv1, self.hparams.cnn_conv2, kernel_size=self.hparams.kernel_size)
        self.drop2d = nn.Dropout2d()
        self.fc1 = nn.Linear(320, self.hparams.cnn_ff)
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(self.hparams.cnn_ff, 10)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        # Feed through each Convolution Layer w/ Max-Pooling (Stride 2)
        conv1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        conv2 = F.relu(F.max_pool2d(self.drop2d(self.conv2(conv1)), 2))
        hidden = self.drop(F.relu(self.fc1(conv2.view(-1, 320))))
        return torch.log_softmax(self.fc2(hidden), dim=1)
