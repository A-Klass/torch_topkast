#%%
from sklearn import datasets
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
import torch.nn as nn
from topkast_linear import TopKastLinear
from topkast_loss import TopKastLoss
import matplotlib.pyplot as plt

#%%
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
    
#%%    
def train(net, num_epochs, num_epochs_explore, update_every, loss, optimizer, 
          batch_size, boston=datasets.load_boston(), split=[.7, .2, .1], 
          patience=5):

    if len(split) < 3:
        split.append(0)

    train_count, validation_count, test_count = np.round(
        np.multiply(506, [.7, .2, .1])).astype(int)
    train_dataset, validation_dataset, test_dataset = \
    torch.utils.data.random_split(
        boston_dataset(), (train_count, validation_count,test_count), 
        generator=torch.Generator().manual_seed(42))

    train_dataset = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    losses_validation = np.zeros(num_epochs)
    losses_train = np.zeros(num_epochs)
    best_loss = np.inf
    best_epoch = 0
    best_net = None

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
            y = y.float().reshape((-1, 1))
            y_hat = net(X)
            optimizer.zero_grad()
            loss_epoch = loss(y_hat, y)
            loss_epoch.backward()
            optimizer.step()
            losses_train[epoch] += loss_epoch / len(y)
            
        losses_validation[epoch] = loss(
            net(validation_dataset[:][0].float(), sparse=False), 
            validation_dataset[:][1].float())
        if (epoch + 1) % 100 == 0:
            print(f'epoch {epoch + 1}, loss {losses_validation[epoch]:f}') 
        
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
    
    test_loss = loss(
        net(test_dataset[:][0].float(), sparse=False), 
        test_dataset[:][1].float())

    return best_net, losses_validation[1:(best_epoch + patience)], \
        losses_train[1:(best_epoch + patience)], best_epoch, test_loss

#%% Second Network
class TopKastNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_in = TopKastLinear(13, 128, 
                                      p_forward=0.6, p_backward=0.4)
        # self.layer_in = nn.Linear(13, 128)
        self.activation_1 = nn.ReLU()
        self.hidden = nn.Linear(128, 1)
        # self.hidden = TopKastLinear(128, 1,
        #                             p_forward=0.1, p_backward=0.05)
        # self.layer_out = TopKastLinear(128, 1,
        #                                p_forward=0.7, p_backward=0.3)

    def forward(self, X, sparse=True):
        y_1 = self.layer_in(self.activation_1(X), sparse=sparse)
        # y_2 = self.hidden(self.activation_2(y_1), sparse = sparse)
        # y_3 = self.layer_out(y_2, sparse = sparse)
        
        return self.hidden(y_1)

#%%
net = TopKastNet()
loss = TopKastLoss(loss = nn.MSELoss, net = net)
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

#%%
kast_net, val_loss, train_loss, best_epoch, test_loss = train(
    net=net, 
    num_epochs=100, 
    num_epochs_explore=30,
    update_every=10,
    loss=loss,
    optimizer=optimizer, 
    batch_size=128,
    patience=100)


# %%
plt.plot(range(len(val_loss)), val_loss, color = "red")
plt.plot(range(len(train_loss)), train_loss, color = "blue")
plt.show()
# %%
