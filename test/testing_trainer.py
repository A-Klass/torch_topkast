"""
Testing procedure including the trainer class
"""
import sys
# import os
# path_to_TopKAST = os.path.join(os.getcwd(), "TopKAST")
# sys.path.insert(0, path_to_TopKAST)
sys.path.insert(0, "./TopKAST")
sys.path.insert(0, "./test")
print(sys.path)
try:
    from topkast_linear import TopKastLinear
except ImportError:
    raise SystemExit("not found. check your relative path")
 
try:
    from topkast_loss import TopKastLoss
except ImportError:
    raise SystemExit("not found. check your relative path")    

try:
    from topkast_trainer import TopKastTrainer
except ImportError:
    raise SystemExit("not found. check your relative path")    

try:
    from test_data import synthetic_dataset, boston_dataset
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

# Analogously, a TopKast net
class TopKastNet(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layer_in = TopKastLinear(
            in_features, 128, p_forward=0.6, p_backward=0.5)
        self.activation = nn.ReLU()
        self.hidden1 = TopKastLinear(
            128, 128, p_forward=0.7, p_backward=0.5)
        self.layer_out = TopKastLinear(
            128, 1,
            p_forward=0.6, p_backward=0.5)

    def forward(self, X, sparse=True):
        y = self.layer_in(X, sparse=sparse)
        y = self.hidden1(self.activation(y), sparse=sparse)
        
        return self.layer_out(self.activation(y), sparse=sparse)

#%% 
# Test with synthetic data sporting 2 features
data = synthetic_dataset(1024)
#%%
net = TopKastNet(2)
loss = TopKastLoss(loss=nn.MSELoss, net=net, alpha=0.4)

# Instantiate a TopKast trainer
trainer = TopKastTrainer(net,
                         loss,
                         num_epochs=50,
                         num_epochs_explore = 2,
                         update_every = 3,
                         batch_size = 5,
                         patience= 50,
                         data = data)
    
# and call training method
trainer.train()
#%% 
net = RegularNet(2)
loss = TopKastLoss(loss=nn.MSELoss, net=net, alpha=0.4)
trainer = TopKastTrainer(net,
                         loss,
                         num_epochs_explore = 2,
                         update_every = 3,
                         batch_size = 5,
                         patience= 20,
                         data = data)
trainer.train()

#%% now with boston which has 13 features
data = boston_dataset()
#%%
net = RegularNet(13)
loss = TopKastLoss(loss=nn.MSELoss, net=net, alpha=0.4)
trainer = TopKastTrainer(net,
                         loss,
                         num_epochs_explore = 2,
                         update_every = 3,
                         batch_size = 5,
                         patience= 20,
                         data = data)
trainer.train()
#%%
net = TopKastNet(13)
loss = TopKastLoss(loss=nn.MSELoss, net=net, alpha=0.4)
trainer = TopKastTrainer(net,
                         loss,
                         num_epochs_explore = 2,
                         update_every = 3,
                         batch_size = 5,
                         patience= 20,
                         data = data)
trainer.train()

trainer.eval()