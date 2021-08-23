"""
Testing procedure including the trainer class
"""
import sys
# import os
# path_to_TopKAST = os.path.join(os.getcwd(), "TopKAST")
# sys.path.insert(0, path_to_TopKAST)
sys.path.insert(0, "./TopKAST")

print(sys.path)
try:
    from TopKAST.topkast_linear import TopKastLinear
except ImportError:
    raise SystemExit("not found. check your relative path")
 
try:
    from TopKAST.topkast_loss import TopKastLoss
except ImportError:
    raise SystemExit("not found. check your relative path")    

try:
    from TopKAST.topkast_trainer import TopKastTrainer
except ImportError:
    raise SystemExit("not found. check your relative path")    

try:
    from .test_data import synthetic_dataset
except ImportError:
    raise SystemExit("not found. check your relative path")  

import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
import torch.nn as nn
import matplotlib.pyplot as plt

# Setup a small vanilla net to compare against
class RegularNet(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layer_in = nn.Linear(in_features, 128)
        self.activation = nn.ReLU()
        self.hidden1 = nn.Linear(128, 128)
        self.layer_out = nn.Linear(128, 1)

    def forward(self, X, sparse=True):
        y = self.layer_in(X)
        y = self.hidden1(self.activation(y))
        
        return self.layer_out(self.activation(y))
    
data = synthetic_dataset(256)
# data = boston_dataset()
# net = TopKastNet(2)
net = RegularNet(2)
loss = TopKastLoss(loss=nn.MSELoss, net=net, alpha=0.4)

# Instantiate a TopKast trainer
trainer = TopKastTrainer


