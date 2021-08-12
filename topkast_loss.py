# %%
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
    def __init__(self, loss) -> None:
        super(TopKastLoss, self).__init__()
        self.loss = loss()
    
    def compute_norm_active_set(self, net):
        """
        Updates the forward and backward indices in self
        """
        
        penalty = torch.tensor(0.)
        
        for child in net.children():
            
            # Go through all the parameters of the layers
            
            for name in child._parameters.keys():
                
                # Only continue if it's weights. Biases are skipped.
                
                if name != 'weight': continue
            
                # If it's a TopKastLinear layer it gets stopped. This is hard 
                # coded. If we have more Topk-Layers later we will need a 
                # better class system here.
                
                if isinstance(child, TopKastLinear):                    
                    penalty += torch.linalg.norm(
                        child.set_fwd().coalesce().values())
                    penalty += torch.linalg.norm(
                        (child.set_justbwd().coalesce().values() / 
                         (1 - child.p_forward)))
                else:
                    penalty += torch.linalg.norm(child._parameters[name])
        
        return penalty
    
    def forward(self, y_hat, y, net):
        
        l = self.loss(y_hat, y) 
        l += self.compute_norm_active_set(net)

        return l

#%%
