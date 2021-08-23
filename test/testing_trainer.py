"""
Testing procedure including the trainer class
"""
#%%
from torch_topkast.topkast_linear import TopKastLinear
from torch_topkast.topkast_loss import TopKastLoss
from torch_topkast.topkast_trainer import TopKastTrainer
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
import torch.nn as nn
import matplotlib.pyplot as plt
#%%
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
data_synthetic = synthetic_dataset(1024)
#%%
net1 = TopKastNet(2) # synthetic data is 2 dimensional
loss1 = TopKastLoss(loss=nn.MSELoss, net=net1, alpha=0.4)

# Instantiate a TopKast trainer
trainer = TopKastTrainer(net1,
                         loss1,
                         num_epochs=50,
                         num_epochs_explore = 2,
                         update_every = 3,
                         batch_size = 5,
                         patience= 50,
                         data = data_synthetic)
# and call training method
trainer.train()
print("finished training RegularNet(2) for synthetic data")
#%% 
net2 = RegularNet(2)
loss2 = TopKastLoss(loss=nn.MSELoss, net=net2, alpha=0.4)
trainer = TopKastTrainer(net2,
                         loss2,
                         num_epochs_explore = 2,
                         update_every = 3,
                         batch_size = 5,
                         patience= 20,
                         data = data_synthetic)
trainer.train()
print("finished training RegularNet(2) for synthetic data")
#%% now with boston which has 13 features
# data_boston = boston_dataset()
# #%%
# net3 = RegularNet(13)
# loss3 = TopKastLoss(loss=nn.MSELoss, net=net3, alpha=0.4)
# trainer = TopKastTrainer(net3,
#                          loss3,
#                          num_epochs_explore = 2,
#                          update_every = 3,
#                          batch_size = 5,
#                          patience= 20,
#                          data = data_boston)
# trainer.train()
# print("finished training RegularNet(13) for boston data")
# #%%
# net4 = TopKastNet(13)
# loss4 = TopKastLoss(loss=nn.MSELoss, net=net4, alpha=0.4)
# trainer = TopKastTrainer(net4,
#                          loss4,
#                          num_epochs_explore = 2,
#                          update_every = 3,
#                          batch_size = 5,
#                          patience= 20,
#                          data = data_boston)
# trainer.train()
# print("finished training TopKastNet(13) for boston data")
# trainer.eval()