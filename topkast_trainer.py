"""
We need a class that does the training with burn in phase etc.
"""
# a lot to still debug
from TopKAST.topkast_loss import TopKastLoss
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
from .boston_dataset import boston_dataset

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

class TopKastTrainer(topkast_net: TopKastNet,
                     num_epochs: int = 50,
                     num_epochs_explore: int = None,
                     update_every: int = None,
                     loss: TopKastLoss,
                     train_val_test_split: list = [.7, .2, .1],
                     patience: int,
                     params_optimizer,
                     batch_size: int = None,
                    #  data = boston_dataset(),
                     **loss_args):
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
    
    # dataset
    #######
    # hardcoded for now
    self.data = boston_dataset() 
    
    if batch_size is None:
        batch_size = max(1, int(self.data.__len__() / 50))
    self.batch_size = batch_size
    #######
    
    self.loss = loss(**loss_args)
    
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
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0)

    self.losses_validation = np.zeros(self.num_epochs)
    self.losses_train = np.zeros(self.num_epochs)
    self.best_loss = np.inf
    self.best_epoch = 0
    self.best_net = None

    for epoch in range(self.num_epochs):
        if epoch < self.num_epochs_explore:
            for layer in self.net.children():
                if isinstance(layer, TopKastLinear):
                    layer.update_active_param_set()
        else:
            if epoch % self.update_every == 0:
               for layer in net.children():
                   if isinstance(layer, TopKastLinear):
                        layer.update_active_param_set() 
        for X, y in train_dataset:
            X = X.float()
            y = y.float().reshape(-1, 1)
            y_hat = net(X)
            # optimizer.zero_grad()
            loss_epoch = loss(y_hat, y)
            loss_epoch.sum().backward()
            # print(torch.linalg.norm(net.layer_in.weight_vector))
            # optimizer.step()
            sgd(net.parameters(), lr=lr, batch_size=self.batch_size)
            # for layer in net.children():
            #     if isinstance(layer, TopKastLinear):
            #         layer.update_backward_weights()
            # print(torch.linalg.norm(net.layer_in.weight_vector))
            # print(torch.linalg.norm(net.layer_in.bias))
            # print(torch.linalg.norm(net.layer_in.sparse_weights.grad.to_dense()))
            losses_train[epoch] += loss_epoch / len(y)
        with torch.no_grad(): 
            losses_validation[epoch] = loss(
                net(validation_dataset[:][0].float(), sparse=False), 
                validation_dataset[:][1].float().reshape(-1, 1))
        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1}, loss {losses_validation[epoch]:f} train loss {losses_train[epoch]:f}')  
        
        # Compare this loss to the best current loss
        # If it's better save the current net and change best loss
        if losses_validation[epoch] < best_loss:
            best_epoch = epoch
            best_loss = losses_validation[epoch]
            best_net = copy.deepcopy(net)

        # Check if we are patience epochs away from the current best epoch, 
        # if that's the case break the training loop
        if epoch - best_epoch > patience:
            break
    with torch.no_grad():
        test_loss = loss(
            net(test_dataset[:][0].float(), sparse=False), 
            test_dataset[:][1].float().reshape(-1, 1))

    return best_net, losses_validation[1:(best_epoch + patience)], \
        losses_train[1:(best_epoch + patience)], best_epoch, test_loss
        
    def __init__(self, hidden, p_forward=0.9, p_backward=0.8, num_epochs_explore=500, update_every=100):
        super().__init__()
        self.layer_in = TopKastLinear(
            13, hidden, p_forward=p_forward, p_backward=p_backward)
        self.activation = nn.ReLU()
        self.hidden1 = TopKastLinear(
            hidden, hidden, p_forward=p_forward, p_backward=p_backward)
        # self.hidden2 = TopKastLinear(
        #     1024, 1024, p_forward=p_forward, p_backward=p_backward)
        self.layer_out = TopKastLinear(
            hidden, 1,
            p_forward=p_forward, p_backward=p_backward)
        self.num_epochs_explore = num_epochs_explore
        self.update_every = update_every
        
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
      
    def reset_justbwd_weights(self) -> None:
        for layer in self.children():
          if isinstance(layer, TopKastLinear):
            layer.reset_justbwd_weights()