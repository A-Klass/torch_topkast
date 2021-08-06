# %%
import torch
import torch.nn as nn
import numpy as np
from topkast_linear import TopKastLinear


#%%
class TopKastLoss(nn.Module):
    """Takes a normal torch.nn.loss and then adds the discounted (alpha) L2-norm of all
    active(!) parameters. Goes through all the layers of the net and looks for TopkLinear 
    Layers and only takes the appropriate weights."""
    def __init__(self, loss) -> None:
        super(TopKastLoss, self).__init__()
        self.loss = loss()
    
    def compute_norm_active_set(self, net):
        """Updates the forward and backward indices in self"""
        l2_norm = torch.tensor(0.)
        for child in net.children():
            # Go through all the parameters of the layers
            for name in child._parameters.keys():
                # Only continue if it's weights. Biases get skipped.
                if name != 'weight': continue
                # If it's a TopKastLinear layer it gets stopped. This is hard coded. 
                # If we have more Topk-Layers later we will need a better class system here.
                if isinstance(child, TopKastLinear):                    
                    l2_norm += torch.linalg.norm(child.sparse_weights())
                    l2_norm += torch.linalg.norm(child.sparse_weights(forward = False)) / child.D
                else:
                    l2_norm += torch.linalg.norm(child._parameters[name])
        
        return l2_norm
    
    def forward(self, y_hat, y, net):
        l = self.loss(y_hat, y) 
        l += self.compute_norm_active_set(net)

        return l

# %%
class net_a_bit_overkill(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_in = TopKastLinear(13, 128, 20, 30)
        self.activation_1 = nn.ReLU()
        self.hidden = TopKastLinear(128, 128, 50, 60)
        self.activation_2 = nn.ReLU()
        self.layer_out = nn.Linear(128, 1)

    def forward(self, X):
        return self.layer_out(self.activation_2(self.hidden(self.activation_1(self.layer_in(X)))))

net = net_a_bit_overkill()
# %%
loss = TopKastLoss(loss = nn.MSELoss)
loss(torch.rand(10), torch.rand(10), net)
# %%
