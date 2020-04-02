"""
feedforward.py

Implements the MNIST FeedForward Network Initialization and Forward Pass Logic.
"""
from models.base import BaseMNIST

import torch


class FeedForward(BaseMNIST):
    def build_model(self):
        # Create Layers
        self.layer_1 = torch.nn.Linear(28 * 28, self.hparams.ff_1)
        self.layer_2 = torch.nn.Linear(self.hparams.ff_1, self.hparams.ff_2)
        self.layer_3 = torch.nn.Linear(self.hparams.ff_2, 10)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        # Reshape [bsz, 1, 28, 28] --> [bsz, 28]
        x = x.view(-1, 28 * 28)

        # Feed through each Layer
        ff_1 = torch.relu(self.layer_1(x))
        ff_2 = torch.relu(self.layer_2(ff_1))

        return torch.log_softmax(self.layer_3(ff_2), dim=1)
