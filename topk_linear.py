#%%
import math
import warnings

import numpy as np
import torch
import torch.nn as nn

#%%
class topkTraining(torch.autograd.Function):
    """ Custom pytorch function to handle changing connections"""

    @staticmethod
    def forward(ctx, inputs, weights, bias, indices_forward, indices_backward):
        # output = torch_sparse.spmm(indices, weights, out_features, in_features, inputs.t()).t()
        weigths = torch.sparse.FloatTensor(
            indices_forward, weights[indices_forward], weights.shape)
        output = torch.addmm(bias, weigths, inputs.t()).t()

        # ctx.save_for_backward(inputs, weights, bias, indices_backward)
        # ctx.in1 = k
        # ctx.in2 = out_features
        # ctx.in3 = in_features
        # ctx.in4 = max_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights, indices = ctx.saved_tensors
        k = ctx.in1
        out_features = ctx.in2
        in_features = ctx.in3
        max_size = ctx.in4

        device = grad_output.device
        p_index = torch.LongTensor([1, 0])
        new_indices = torch.zeros_like(indices).to(device=device)
        new_indices[p_index] = indices

        

        return grad_input, None, None, None, None, None

#%%
class TopkLinear(nn.Module):
    "This is just the nn.Linear shell with added topk inputs."
    def __init__(self, in_features: int, out_features: int, topk_forward: int, topk_backward: int, 
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TopkLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        assert topk_backward >= topk_forward

        self.topk_forward = topk_forward
        self.topk_backward = topk_backward
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def compute_mask(self, topk):
        w = self.weight
        topk_percantage = topk / w.shape.numel()
        if w.is_sparse:
            threshold = np.quantile(w.values().detach(), topk_percantage)
            mask = torch.where(w.values().detach() > threshold)
        else:
            threshold = np.quantile(w.reshape(-1).detach(), topk_percantage)
            mask = torch.where(w.reshape(-1).detach() > threshold)
        return mask
    
    def forward(self, inputs):
        self.indices_forward = self.compute_mask(self.topk_forward)
        self.indices_backward = self.compute_mask(self.topk_backward)


        if self.training:
            output = topkTraining.apply(inputs, self.weight, self.bias, self.indices_forward,
                                        self.indices_backward)
        else:
            with torch.no_grad():
                weights = torch.sparse.FloatTensor(self.indices_forward, 
                                                  self.weight[self.indices_forward],
                                                  self.weight.shape)
                output = torch.mm(weights, inputs.t()).t()
        
        return output



#%%
layer = TopkLinear(10, 11, 4, 5)
x = torch.rand(10)
layer.training = False
layer(x)
# %%
(layer.weight * (layer.compute_mask(5) == False)).to_sparse()
# %%
