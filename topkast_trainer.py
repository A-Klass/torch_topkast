"""
We need a class that does the training with burn in phase etc.
"""
import sys
sys.path.insert(0, "./TopKAST")
sys.path.insert(0, "./test")
try:
    from TopKAST.topkast_linear_commented import TopKastLinear
except ImportError:
    raise SystemExit("not found. check your relative path")

try:
    from TopKAST.topkast_loss import TopKastLoss
except ImportError:
    raise SystemExit("not found. check your relative path")

try:
    from test.boston_dataset import boston_dataset
except ImportError:
    raise SystemExit("not found. check your relative path")

from copy import error

import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For now it is not 100% clear whether we can utilize nn.optim optimizers
def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

class TopKastNet(nn.Module):
    """
    Build a neural net with TopKast layers.
    This is our vanilla template for easy testing
    """
    def __init__(self):
        super().__init__()
        self.layer_in = TopKastLinear(
            13, 128, p_forward=0.6, p_backward=0.5)
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

class TopKastTrainer():
    """
    This class executes the training procedure according to Jayakumar et al.(2021).
    Includes:
        - exploration phase (~burn in period): selecting different active sets A 
          and corresponding α at each iteration + performing one update step on θ 
          using gradients obtained from the loss f(α, x) and the regularizer
        - refinement phase: fixed set A
    """
    
    def __init__(self,
                 topkast_net: TopKastNet,
                 loss: TopKastLoss,
                 num_epochs: int = 50,
                 num_epochs_explore: int = None,
                 update_every: int = None,
                 batch_size: int = None,
                 train_val_test_split: list = [.7, .2, .1],
                 patience: int = None,
                 optimizer: nn.optim = None,
                 params_optimizer: dict = None,
              #  data = boston_dataset(),
                #  loss_args
                 ):
        """
        Args:
            topkast_net (TopKastNet): a neural net with TopKast sparse layers.
            loss (TopKastLoss):
            num_epochs (int): # of epochs for which training is run.
            num_epochs_explore: # of epochs for exploration phase
            update_every: do Top-K selection at every `update_every` epoch.
            batch_size (int): # of observations per batch.
            loss (TopKastLoss): loss function with regularization.
            train_val_test_split (list): split up data set in training, validation and test set.
            patience (int): early stopping if validation loss keeps not improving.
            optimizer (torch.nn.opim): optimizer from pytorch. not supported yet.
            params_optimizer (dict) : named dict of parameters to pass on to optimizer.
        """
        # Init definitions and asserts
        
        self.net = topkast_net
        
        self.num_epochs = num_epochs
    
        if num_epochs_explore is None:
            num_epochs_explore = max(int(1), int(num_epochs / 10))
        else: 
            assert num_epochs_explore <= num_epochs
        self.num_epochs_explore = num_epochs_explore
    
        # update weights every couple of epochs
        if update_every is None:
            update_every = max(1, int(num_epochs / 5))
        else:
            assert update_every >= 1
        self.update_every = update_every 
    
        # patience for early stopping
        if patience is None:
            patience = num_epochs
        self.patience = patience
    
        #####################
        # # dataset
        # hardcoded for now
        self.data = boston_dataset() 
    
        if batch_size is None:
            batch_size = max(1, int(self.data.__len__() / 50))
        self.batch_size = batch_size
        #####################
    
        # self.loss = loss(loss_args)
        self.loss = loss()
        
        if len(train_val_test_split) < 3:
            train_val_test_split.append(0)
        self.train_val_test_split = train_val_test_split
    
        self.train_count, self.validation_count, self.test_count = np.round(
            np.multiply(boston_dataset().__len__(), train_val_test_split)).astype(int)
        
        self.train_dataset, self.validation_dataset, self.test_dataset = \
        torch.utils.data.random_split(
            self.data,
            (self.train_count, self.validation_count, self.test_count), 
            generator=torch.Generator().manual_seed(42))

        self.train_dataset = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0)
        
        # parameters for the optimizer
        if optimizer is None:
            self.optimizer = sgd(self.net.parameters(), 
                                 lr=0.001, 
                                 batch_size=self.batch_size)
        else:
            # self.optimizer = optimizer(params_optimizer)
            raise ValueError('currently we only allow our self defined SGD')
            self.optimizer = optimizer
            
        self.losses_validation = np.zeros(self.num_epochs)
        self.losses_train = np.zeros(self.num_epochs)
        self.best_loss = np.inf
        self.best_epoch = 0
        self.best_net = None
    
    def training(self):
        for epoch in range(self.num_epochs):
            if epoch < self.num_epochs_explore:
                for layer in self.net.children():
                    if isinstance(layer, TopKastLinear):
                        layer.update_active_param_set()
            else:
                if epoch % self.update_every == 0:
                    for layer in self.net.children():
                        if isinstance(layer, TopKastLinear):
                            layer.update_active_param_set() 
            for X, y in self.train_dataset:
                X = X.float()
                y = y.float().reshape(-1, 1)
                y_hat = self.net(X)
                # optimizer.zero_grad()
                loss_epoch = self.loss(y_hat, y)
                loss_epoch.sum().backward()
                # print(torch.linalg.norm(net.layer_in.weight_vector))
                # optimizer.step()
                self.optimizer
                # for layer in net.children():
                #     if isinstance(layer, TopKastLinear):
                #         layer.update_backward_weights()
                # print(torch.linalg.norm(net.layer_in.weight_vector))
                # print(torch.linalg.norm(net.layer_in.bias))
                # print(torch.linalg.norm(net.layer_in.sparse_weights.grad.to_dense()))
                self.losses_train[epoch] += loss_epoch / len(y)
            with torch.no_grad(): 
                self.losses_validation[epoch] = self.loss(
                    self.net(self.validation_dataset[:][0].float(), sparse=False), 
                    self.validation_dataset[:][1].float().reshape(-1, 1))
            if (epoch + 1) % 10 == 0:
                print(f'epoch {epoch + 1}, loss {self.losses_validation[epoch]:f} train loss {self.losses_train[epoch]:f}')  
        
            # Compare this loss to the best current loss
            # If it's better save the current net and change best loss
            if self.losses_validation[epoch] < self.best_loss:
                self.best_epoch = epoch
                self.best_loss = self.losses_validation[epoch]
                # best_net = copy.deepcopy(net) # deepcopy doesn't work that way

            # Check if we are patience epochs away from the current best epoch, 
            # if that's the case break the training loop
            if epoch - self.best_epoch > self.patience:
                break
        with torch.no_grad():
            test_loss = self.loss(
                self.net(self.test_dataset[:][0].float(), sparse=False), 
                self.test_dataset[:][1].float().reshape(-1, 1))

        return self.losses_validation[1:(self.best_epoch + self.patience)], \
            self.losses_train[1:(self.best_epoch + self.patience)]
        
    def burn_in(self, epoch) -> None:
      if epoch < self.num_epochs_explore:
            for layer in self.children():
                if isinstance(layer, TopKastLinear):
                    layer.update_active_param_set()
      else:
            if epoch % self.update_every == 0:
               for layer in self.children():
                   if isinstance(layer, TopKastLinear):
                        layer.update_active_param_set()
