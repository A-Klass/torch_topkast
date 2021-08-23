# %%
# In order to use relative import such as:
# from TopKAST.topkast_linear import TopKastLinear, 
# first add the the package path to the PYTHONPATH.
import sys
sys.path.insert(0, "./TopKAST")

try:
    from topkast_linear import TopKastLinear
except ImportError:
    raise SystemExit("not found. check your relative path")
  
import torch
import torch.nn as nn
from topkast_linear import TopKastLinear

#%%
class TopKastLoss(nn.Module):
    """
    Takes a normal torch.nn.loss and then adds the discounted (alpha) 
    L2-norm of all active(!) parameters. Goes through all the layers of the net 
    and looks for TopkLinear layers and only takes the appropriate weights.
    """
    def __init__(self, loss, net, alpha=1) -> None:
        """
            Args:
                loss(torch.nn.Module): Loss function from torch
                net(TopKastNet): a net with TopKast layers
                alpha(float): penalty applied for active paramters in A. Acts as a
                              regularization tool.
        """

        super(TopKastLoss, self).__init__()
        self.loss = loss()
        self.net = net
        assert alpha >= 0 and alpha <= 1
        self.alpha = alpha
    
    def compute_norm_active_set(self):
        """
        Updates the forward and backward indices in self
        """
        
        penalty = torch.tensor(0.)
        
        for child in self.net.children():
                         
            # If the loop encounters a TopKastLinear layer, it stops to compute 
            # the TopKast-specific penalty.
            # TODO: adjust if further TopKast layers are added (proper class
            # system)
            # For a common layer, all weights are L2-penalized.
                
            if isinstance(child, TopKastLinear):
                penalty += torch.linalg.norm(child.active_fwd_weights[child.set_fwd])
                # penalty += torch.linalg.norm(child.active_fwd_weights[child.set_fwd])
                penalty += (torch.linalg.norm(child.active_fwd_weights[child.set_justbwd]) / # this is always 0: debug von vorn bis hinten. vllt kurz weights überschreiben oder so. 
                            (1 - child.p_forward))
                # penalty += (torch.linalg.norm(child.active_fwd_weights[child.set_justbwd]) / # this is always 0: debug von vorn bis hinten. vllt kurz weights überschreiben oder so. 
                #             (1 - child.p_forward))
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
