# %%
import torch
import torch.nn as nn
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
                    l2_norm += torch.linalg.norm(
                        child.sparse_weights(forward=False)) / child.d_fwd
                else:
                    l2_norm += torch.linalg.norm(child._parameters[name])
        
        return l2_norm
    
    def forward(self, y_hat, y, net):
        l = self.loss(y_hat, y) 
        l += self.compute_norm_active_set(net)

        return l

#%%
