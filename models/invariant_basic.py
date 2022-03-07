import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import sys
sys.path.append('../layers/')
import equivariant_linear_pytorch as eq

class invariant_basic(nn.Module):
    def __init__(self, config, data):
        super(invariant_basic, self).__init__()
        self.config = config
        self.data = data
        self.build_model()

    def build_model(self):
        # build network architecture using config file
        self.equi_layers = nn.ModuleList()
        self.equi_layers.append(eq.layer_2_to_2(self.data.train_graphs[0].shape[0], self.config.architecture[0]))
        L = len(self.config.architecture)
        for layer in range(1, L):
            self.equi_layers.append(eq.layer_2_to_2(self.config.architecture[layer - 1], self.config.architecture[layer]))
        self.equi_layers.append(eq.layer_2_to_1(self.config.architecture[L - 1], 1024))
        self.fully1 = nn.Linear(1024, 512)
        self.fully2 = nn.Linear(512, 256)
        self.fully3 = nn.Linear(256, self.config.num_classes)

    def forward(self, inputs):
        outputs = inputs
        for layer in range(len(self.equi_layers)):
            outputs = self.equi_layers[layer](outputs)
        outputs = torch.sum(outputs, dim = 2)
        outputs = F.relu(self.fully1(outputs))
        outputs = F.relu(self.fully2(outputs))
        return F.log_softmax(self.fully3(outputs))
