#%% Imports

from torch_topkast.topkast_linear import TopKastLinear # type: ignore
import torch
import torch.nn as nn
from typing import Optional

#%%
class TopKastLoss(nn.Module):
    """
    Takes a standard torch.nn.loss and then adds the discounted (alpha) 
    L2-norm of all active(!) parameters. Goes through all the layers of the 
    net and looks for TopkLinear layers and only takes the appropriate weights.
    If the net doesn't have any TopKastLinear layers, this is just an 
    L2-weighted loss.
    """
    def __init__(self, 
                 loss: torch.nn.Module, 
                 net: torch.nn.Module, 
                 alpha: int=1, 
                 device: Optional[torch.device]=None) -> None:
        """
            Args:
                loss(torch.nn.Module): Loss function from torch
                net(TopKastNet): a net with TopKast layers
                alpha(float): penalty applied for active paramters in A. 
                Acts as regularization tool.
        """

        super(TopKastLoss, self).__init__()
        self.loss = loss()
        self.net = net
        assert alpha >= 0. and alpha <= 1.
        self.alpha = alpha
        self.device = device
    
    def compute_norm_active_set(self):
        """
        Computes the L2 norm for the active weights in the forward pass and 
        the weighted L2 norm of the weights that are added in the backward 
        pass.
        """
        
        penalty = torch.tensor(0., device=self.device)
        
        for child in self.net.children():
                         
            # If the loop encounters a TopKastLinear layer, it stops to compute 
            # the TopKast-specific penalty.
            # For a common layer, all weights are L2-penalized.
                
            if isinstance(child, TopKastLinear):
                penalty += torch.linalg.norm(
                    child.active_fwd_weights[child.set_fwd])
                child.active_fwd_weights.data[child.set_justbwd] = \
                    child.weight[child.idx_justbwd]
                penalty += (torch.linalg.norm(
                    child.active_fwd_weights[child.set_justbwd]) /  
                            (1 - child.p_forward))
                child.active_fwd_weights.data[child.set_justbwd] = 0.
            else:
                for name in child._parameters.keys():
                    if name != 'weight': continue
                    penalty += torch.linalg.norm(child._parameters[name])
        
        return penalty
    
    def forward(self, y_hat, y):
        l = self.loss(y_hat, y) 
        l += self.alpha * self.compute_norm_active_set()

        return l

#%%
