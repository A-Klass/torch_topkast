"""
Class that executes the training procedure according to the authors 
(Jayakumar et al.(2021)).
Uses loss as defined in the class Topkastloss.
Compatible with the usual torch.optim optimizers.
"""

#%% Imports

from torch_topkast.topkast_linear import TopKastLinear
from torch_topkast.topkast_loss import TopKastLoss
import numpy as np
import torch 
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
from typing import Optional

#%%

class TopKastTrainer():
    """
    This class executes the training procedure according to the authors.
    Includes:
        - exploration phase (~burn in period): selecting different active sets A 
          and corresponding α at each iteration + performing one update step on θ 
          using gradients obtained from the loss f(α, x) and the regularizer
        - refinement phase: fixed set A
    """
    
    def __init__(self,
                 topkast_net: torch.nn.Module,
                 loss: TopKastLoss,
                 num_epochs: int=50,
                 print_info_every: int=10,
                 print_loss_history: bool=False,
                 num_epochs_explore: Optional[int]=None,
                 update_every: Optional[int]=None,
                 batch_size: Optional[int]=None,
                 train_val_test_split: list=[.7, .2, .1],
                 patience: Optional[int]=None,
                 optimizer: torch.optim.Optimizer=None,
                 seed: int=42,
                 data: Dataset=None,
                 device: Optional[torch.device]=None
                 ):
        """
        Args:
            topkast_net (torch.nn.Module): A neural net which may include 
            TopKast sparse layers.
            loss (TopKastLoss): Loss which treats sparse TopKast layers with an 
            according penalty.
            num_epochs (int): # of epochs for which training is run.
            print_info_every(int): Print val/train losses periodically.
            print_loss_history(bool): Print training results.
            num_epochs_explore: # of epochs for exploration phase
            update_every: Do Top-K selection at every `update_every` epoch.
            batch_size (int): # of observations per batch.
            train_val_test_split (list): Split up data set in training, 
            validation and test set.
            patience (int): Early stopping if validation loss does not improve 
            (substantially).
            seed (int): Seed for the train/val/test split.
            optimizer (torch.nn.optim): PyTorch optimizer.
            data (torch.utils.data.Dataset): A dataset class with a 
            `__getitem__()` function.
            device (torch.device): 'cpu' or 'cuda' depending on device to be 
            used.
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
    
        # Update weights every couple of epochs
        if update_every is None:
            update_every = max(1, int(num_epochs / 5))
        else:
            assert update_every >= 1
        self.update_every = update_every 
    
        # Patience for early stopping
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
        
        if device is None:
            # check whether GPU can be used for training
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
                print("GPU available. Training on", self.device)
            else:
                self.device = torch.device('cpu')
                print("Training on", self.device)
        else:
            self.device = device

        pin_memory = True if self.device != torch.device('cpu') else False

        self.train_dataset = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=0)
        
        self.optimizer = optimizer

        self.losses_validation = np.zeros(self.num_epochs)
        self.losses_train = np.zeros(self.num_epochs)
        self.test_loss = None
        self.best_loss = np.inf
        self.best_epoch = 0
        self.best_net = None
    
    ######################################################################
    # Helper functions
    ######################################################################
    def _update_sparse_layers(self):
        """
        Goes through all sparse TopKast layers
        and updates the active parameter set, accordingly.
        
        This is supposed to be called by `_burn_in()` and 
        `_update_periodically()`
        """
        for layer in self.net.children():
            if isinstance(layer, TopKastLinear):
                layer.update_active_param_set()
                
    def _check_and_update_forward_set(self, epoch):
        """
        Checks depending on the epoch if the active parameter set(s) 
        should be updated. 
        If we are in the exploration phase, we update the active 
        param sets in all the sparse layers in every epoch.
        After the exploration phase, we update the active sets according
        to how often the user has defined it (via `update_every`)
        """
        if epoch <= self.num_epochs_explore:
            self._update_sparse_layers()
        else:
            if epoch % self.update_every == 0:
                self._update_sparse_layers()
                
    def _reset_justbwd_weights(self) -> None:
        """
        Wrapper function for resetting B\A to zeros, since optim.step makes 
        these parameters nonzero.
        """
        for layer in self.net.children():
            if isinstance(layer, TopKastLinear):
                layer.reset_justbwd_weights()
                
    ######################################################################
    # Core functionality
    ######################################################################
    def train(self):
        for epoch in range(self.num_epochs):
            self._check_and_update_forward_set(epoch)
            for X, y in self.train_dataset:
                X = X.float().to(self.device)
                y = y.float().reshape(-1, 1).to(self.device)
                y_hat = self.net(X)
                self.optimizer.zero_grad()
                loss_epoch = self.loss(y_hat, y)
                loss_epoch.backward()
                self.optimizer.step()
                self._reset_justbwd_weights()
                self.losses_train[epoch] += loss_epoch / len(y)
            with torch.no_grad(): 
                self.losses_validation[epoch] = self.loss(
                    self.net(
                        self.validation_dataset[:][0].float().to(self.device), 
                        sparse=False), 
                    self.validation_dataset[:][1].float().to(
                        self.device).reshape(-1, 1))
            if (epoch + 1) % self.print_info_every == 0:
                print(f'epoch {epoch + 1}, val loss \
                      {self.losses_validation[epoch]:f} train loss \
                          {self.losses_train[epoch]:f}')  
        
            # Compare this loss to the best current loss
            # If it's better save the current net and change best loss
            if self.losses_validation[epoch] < self.best_loss:
                self.best_epoch = epoch
                self.best_loss = self.losses_validation[epoch]
                self.best_net = self.net

            # Check if we are patience epochs away from the current best epoch, 
            # if that's the case break the training loop
            if epoch - self.best_epoch > self.patience:
                break

        if self.print_loss_history:
            print(self.losses_validation[1:(self.best_epoch)])
            print(self.losses_train[1:(self.best_epoch)])

    def predict(self, data):
        """
        Wrapper for computing the prediction of the trained network with the 
        `data` provided.
        """
        with torch.no_grad():
            return self.net(data)
    
    def eval(self, test_data=None):
        """
        Computes and returns the loss for the test dataset or the provided 
        `test_data`. 
        """
        if test_data is None:
            with torch.no_grad():
                test_loss = self.loss(
                    self.net(
                        self.test_dataset[:][0].float().to(self.device), 
                        sparse=False), 
                    self.test_dataset[:][1].float().to(
                        self.device).reshape(-1, 1))
        else:
            with torch.no_grad():
                test_loss = self.loss(
                    self.net(
                        self.test_data.float().to(self.device), sparse=False), 
                    self.test_data.float().to(self.device).reshape(-1, 1))
                
        return test_loss
    
    def plot_loss(self):
        """
        Plots the two losses acquired through training.
        1st plot is of the train loss
        2nd plot is of the validation loss
        """
        fig, axs = plt.subplots(2)
        plt.subplots_adjust(hspace = .5)
        axs[0].plot(range(len(self.losses_train)), self.losses_train)
        axs[0].set_title('training loss')
        axs[1].plot(
            range(len(self.losses_validation)), 
            self.losses_validation, 
            color='red')
        axs[1].set_title('validation loss')
