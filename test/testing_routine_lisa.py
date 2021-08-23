#%% import

# In order to use relative import such as:
# from TopKAST.topkast_linear import TopKastLinear, 
# first add the the package path to the PYTHONPATH.

# import sys
# sys.path.insert(0, "./TopKAST")

# try:
#     from topkast_linear import TopKastLinear
# except ImportError:
#     raise SystemExit("not found. check your relative path")
 
# try:
#     from topkast_loss import TopKastLoss
# except ImportError:
#     raise SystemExit("not found. check your relative path")    

from TopKAST.topkast_linear import TopKastLinear
from TopKAST.topkast_loss import TopKastLoss
from sklearn import datasets
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
import torch.nn as nn
import matplotlib.pyplot as plt

#%% get synthetic data

class synthetic_dataset():
    """Synthetic two-dimensional regression task"""
    
    def __init__(self, n_obs):
        self.n_observations = n_obs
        x = torch.arange(start=0, end=1, step=1/n_obs)
        self.features = torch.cat((x, torch.cos(x))).reshape(n_obs, 2)
        self.target = (self.features[:, 0] + self.features[:, 1] + 
                       torch.rand(n_obs))
        self.dataset = (self.features, self.target)
        
    def __len__(self):
        return self.n_observations
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.features[idx]
        target = self.target[idx]
        return data, target
        
#%% get boston housing data

class boston_dataset(Dataset):
    """Boston dataset."""

    def __init__(self):
        self.dataset = datasets.load_boston()

    def __len__(self):
        return len(self.dataset.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = torch.from_numpy(self.dataset.data[idx])
        target = torch.tensor(self.dataset.target[idx])
        return data, target

#%% define optimizer

def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

#%% define training routine

def train(data, net, num_epochs, num_epochs_explore, 
          update_every, loss, batch_size, split=[.7, .2, .1], 
          patience=5, lr=1e-3):
    
    if len(split) < 3:
        split.append(0)
    
    train_count, validation_count, test_count = np.round(
        np.multiply(data.__len__(), split)).astype(int)
    
    train_dataset, validation_dataset, test_dataset = \
    torch.utils.data.random_split(
        data, (train_count, validation_count, test_count), 
        generator=torch.Generator().manual_seed(42))
    
    train_dataset = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    losses_validation = np.zeros(num_epochs)
    losses_train = np.zeros(num_epochs)
    best_loss = np.inf
    best_epoch = 0
    # best_net = None

    for epoch in range(num_epochs):
        if epoch < num_epochs_explore:
            for layer in net.children():
                if isinstance(layer, TopKastLinear):
                    layer.update_active_param_set()
        else:
            if epoch % update_every == 0:
               for layer in net.children():
                   if isinstance(layer, TopKastLinear):
                        layer.update_active_param_set() 
                        
        for X, y in train_dataset:
            X = X.float()
            y = y.float().reshape(-1, 1)
            y_hat = net(X)
            loss_epoch = loss(y_hat, y)
            loss_epoch.sum().backward()
            sgd(net.parameters(), lr=lr, batch_size=batch_size)
            # print(torch.linalg.norm(net.layer_in.weight_vector))
            # print(torch.linalg.norm(net.layer_in.bias))
            # print(torch.linalg.norm(net.layer_in.sparse_weights.grad.to_dense()))
            losses_train[epoch] += loss_epoch / len(y)
            
        with torch.no_grad(): 
            losses_validation[epoch] = loss(
                net(validation_dataset[:][0].float(), sparse=False), 
                validation_dataset[:][1].float().reshape(-1, 1))
            
        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1}, \
                  loss {losses_validation[epoch]:f} \
                      train loss {losses_train[epoch]:f}')  
        
        # Compare this loss to the best current loss
        # If it's better save the current net and change best loss
        if losses_validation[epoch] < best_loss:
            best_epoch = epoch
            best_loss = losses_validation[epoch]
            # best_net = copy.deepcopy(net)

        # Check if we are patience epochs away from the current best epoch, 
        # if that's the case break the training loop
        if epoch - best_epoch > patience:
            break
    with torch.no_grad():
        test_loss = loss(
            net(test_dataset[:][0].float(), sparse=False), 
            test_dataset[:][1].float().reshape(-1, 1))

    return losses_validation[1:(best_epoch + patience)], \
        losses_train[1:(best_epoch + patience)], best_epoch, test_loss
        
#%% define topkast net

class TopKastNet(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layer_in = TopKastLinear(
            in_features, 128, p_forward=0.6, p_backward=0.5)
        self.activation = nn.ReLU()
        self.hidden1 = TopKastLinear(
            128, 128, p_forward=0.7, p_backward=0.5)
        # self.hidden2 = TopKastLinear(
        #     1024, 1024, p_forward=0.5, p_backward=0.4)
        self.layer_out = TopKastLinear(
            128, 1,
            p_forward=0.6, p_backward=0.5)

    def forward(self, X, sparse=True):
        y = self.layer_in(X, sparse=sparse)
        y = self.hidden1(self.activation(y), sparse=sparse)
        
        return self.layer_out(self.activation(y), sparse=sparse)

#%% define regular net

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

#%% set up vars

data = synthetic_dataset(1000)
# data = boston_dataset()
# net = TopKastNet(2)
net = RegularNet(2)
loss = TopKastLoss(loss=nn.MSELoss, net=net, alpha=0.4)

#%% run training

val_loss, train_loss, best_epoch, test_loss = train(
    data=data,
    net=net, 
    num_epochs=10000, 
    num_epochs_explore=100,
    update_every=10,
    loss=loss,
    # optimizer=optimizer, 
    batch_size=128,
    patience=100)

# %% plot results

plt.plot(range(len(val_loss)), val_loss, color="red", label="val_loss")
plt.plot(range(len(train_loss)), train_loss, color="blue", label="train_loss")
plt.legend(loc="upper right")
plt.show()

