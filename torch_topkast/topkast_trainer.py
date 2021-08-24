"""
Class for training according to the authors 
With burn in phase etc.
"""
from torch_topkast.topkast_linear import TopKastLinear
from torch_topkast.topkast_loss import TopKastLoss

from copy import error

import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset

import matplotlib.pyplot as plt

# For now it is not 100% clear whether we can utilize nn.optim optimizers
def sgd_custom(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

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
                 topkast_net,
                 loss: TopKastLoss,
                 num_epochs: int = 50,
                 print_info_every: int = 10,
                 print_loss_history: bool = False,
                 num_epochs_explore: int = None,
                 update_every: int = None,
                 batch_size: int = None,
                 train_val_test_split: list = [.7, .2, .1],
                 patience: int = None,
                 optimizer: torch.optim = None,
                 seed: float = 0,
                 data: Dataset = None
                #  loss_args
                 ):
        """
        Args:
            topkast_net (TopKastNet): a neural net with TopKast sparse layers.
            loss (TopKastLoss):
            num_epochs (int): # of epochs for which training is run.
            print_info_every(int): print val/train loss es periodically.
            print_loss_history(bool): print training results.
            num_epochs_explore: # of epochs for exploration phase
            update_every: do Top-K selection at every `update_every` epoch.
            batch_size (int): # of observations per batch.
            loss (TopKastLoss): loss function with regularization.
            train_val_test_split (list): split up data set in training, validation and test set.
            patience (int): early stopping if validation loss keeps not improving.
            seed (float): Seed for the train/val/test split.
            optimizer (torch.nn.opim): optimizer from pytorch. not supported yet.
            data (torch.utils.data.Dataset): A dataset class with an overwritten `__getitem__()` Funktion.
        """
        # Init definitions and asserts
        
        self.net = topkast_net
        
        self.num_epochs = num_epochs

        self.print_info_every = print_info_every
        
        self.print_loss_history = print_loss_history
        
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
    
        assert data is not None
        self.data = data
        
        assert isinstance(self.data.__len__(), int)
        self.n_obs = self.data.__len__()
        
        if batch_size is None:
            batch_size = max(1, int(self.data.__len__() / 50))
        self.batch_size = batch_size
        #####################
    
        # self.loss = loss(loss_args)
        self.loss = loss
        
        if len(train_val_test_split) < 3:
            train_val_test_split.append(0)
        self.train_val_test_split = train_val_test_split
    
        self.train_count, self.validation_count, self.test_count = np.round(
            np.multiply(self.n_obs, train_val_test_split)).astype(int)
        
        self.train_dataset, self.validation_dataset, self.test_dataset = \
        random_split(
            self.data,
            (self.train_count, self.validation_count, self.test_count), 
            generator=torch.Generator().manual_seed(seed))

        self.train_dataset = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0)
        
        self.optimizer = optimizer
            
        self.losses_validation = np.zeros(self.num_epochs)
        self.losses_train = np.zeros(self.num_epochs)
        self.test_loss = None
        self.best_loss = np.inf
        self.best_epoch = 0
        self.best_net = None
    
    def _burn_in(self, epoch) -> None:
      if epoch < self.num_epochs_explore:
            for layer in self.net.children():
                if isinstance(layer, TopKastLinear):
                    layer.update_active_param_set()
      else:
            if epoch % self.update_every == 0:
               for layer in self.net.children():
                   if isinstance(layer, TopKastLinear):
                        layer.update_active_param_set()

    def _reset_justbwd_weights(self) -> None:
        for layer in self.net.children():
            if isinstance(layer, TopKastLinear):
                layer.reset_justbwd_weights()

    def train(self):
        for epoch in range(self.num_epochs):
            self._burn_in(epoch)
            for X, y in self.train_dataset:
                X = X.float()
                y = y.float().reshape(-1, 1)
                # Boston housing does not work here. can't figure out why self.net(X) is not working
                y_hat = self.net(X)
                self.optimizer.zero_grad()
                loss_epoch = self.loss(y_hat, y)
                loss_epoch.backward()
                self.optimizer.step()
                self._reset_justbwd_weights()
                # print(torch.linalg.norm(net.layer_in.weight_vector))
                # print(torch.linalg.norm(net.layer_in.bias))
                # print(torch.linalg.norm(net.layer_in.sparse_weights.grad.to_dense()))
                self.losses_train[epoch] += loss_epoch / len(y)
            with torch.no_grad(): 
                self.losses_validation[epoch] = self.loss(
                    self.net(self.validation_dataset[:][0].float(), sparse=False), 
                    self.validation_dataset[:][1].float().reshape(-1, 1))
            if (epoch + 1) % self.print_info_every == 0:
                print(f'epoch {epoch + 1}, val loss {self.losses_validation[epoch]:f} train loss {self.losses_train[epoch]:f}')  
        
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

        if self.print_loss_history:
            print(self.losses_validation[1:(self.best_epoch)])
            print(self.losses_train[1:(self.best_epoch)])

    def predict(self, data):
        with torch.no_grad():
            return self.net(data)
    
    def eval(self, test_data = None):
        """
        evaluate on test set
        """
        with torch.no_grad():
            test_loss = self.loss(
                self.net(self.test_dataset[:][0].float(), sparse=False), 
                self.test_dataset[:][1].float().reshape(-1, 1))
            print(f'test loss' % test_loss)
    
    def plot_loss(self):
        fig, axs = plt.subplots(2)
        axs[0].plot(range(len(self.losses_train)), self.losses_train)
        axs[0].set_title("training loss")
        axs[1].plot(range(len(self.losses_validation)), self.losses_validation, color="red")
        axs[1].set_title("validation loss")