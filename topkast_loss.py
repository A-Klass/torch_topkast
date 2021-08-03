# %%
import torch
import torch.nn as nn
import numpy as np
from topk_linear import TopkLinear


#%%
class TopKastLoss(nn.Module):
    """Takes a normal torch.nn.loss and then adds the discounted (alpha) L2-norm of all
    active(!) parameters. Goes through all the layers of the net and looks for TopkLinear 
    Layers and only takes the appropriate weights."""
    def __init__(self, loss, alpha) -> None:
        super(TopKastLoss, self).__init__()
        self.loss = loss()
        self.alpha = alpha

    def compute_masked_params(self, net):
        # init param list which gets transformed into a 1d Tensor
        params = []
        # Go through all the layers
        for child in net.children():
            # Go through all the parameters of the layers
            for name in child._parameters.keys():
                # Only continue if it's weights. Biases get skipped.
                if name != 'weight': continue
                param = child._parameters[name].detach()
                # If it's a TopkLinear layer it gets stopped. This is hard coded. 
                # If we have more Topk-Layers later we will need a better class system here.
                if isinstance(child, TopkLinear):
                    # Compute the mask
                    topk_backward = child.topk_backward
                    topk_percentage = topk_backward / param.shape.numel()
                    if param.is_sparse:
                        threshold = np.quantile(param.values().detach(), topk_percentage)
                        mask = np.where(param.values().detach() >= threshold)
                    else:
                        threshold = np.quantile(param.reshape(-1).detach(), topk_percentage)
                        mask = np.where(param.detach() >= threshold)
                    # Apply the mask
                    params.append(param[mask].reshape(-1))
                else:
                    params.append(param.reshape(-1))
    
        return torch.cat(params)
    
    def forward(self, y_hat, y, net):
        # get the Tensor for all active weights in the backward pass.
        params = self.compute_masked_params(net)
        l = self.loss(y_hat, y)
        l += self.alpha * torch.linalg.norm(params)

        return l
# %% Don't run, minimal example
# loss = TopKastLoss(loss = nn.MSELoss, alpha = 0.1)
# loss(torch.rand(10), torch.rand(10), torch.rand(1000))
# %%
# class net_a_bit_overkill(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer_in = TopkLinear(13, 128, 20, 30)
#         self.activation_1 = nn.ReLU()
#         self.hidden = TopkLinear(128, 128, 50, 60)
#         self.activation_2 = nn.ReLU()
#         self.layer_out = nn.Linear(128, 1)

#     def forward(self, X):
#         return self.layer_out(self.activation_2(self.hidden(self.activation_1(self.layer_in(X)))))

# net = net_a_bit_overkill()
# # %%
# loss = TopKastLoss(loss = nn.MSELoss, alpha = 0.1)
# loss(torch.rand(10), torch.rand(10), net)
# %%
