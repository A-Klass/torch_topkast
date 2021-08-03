# %%
import torch
import torch.nn as nn


#%%
class TopKastLoss(nn.Module):
    def __init__(self, loss, alpha) -> None:
        super(TopKastLoss, self).__init__()
        self.loss = loss()
        self.alpha = alpha


    def forward(self, y_hat, y, params):
        params = params.detach().reshape(-1)
        l = self.loss(y_hat, y)
        l += self.alpha * torch.linalg.norm(params)

        return l
# %% Don't run, minimal example
# loss = TopKastLoss(loss = nn.MSELoss, alpha = 0.1)
# loss(torch.rand(10), torch.rand(10), torch.rand(1000))
# %%
