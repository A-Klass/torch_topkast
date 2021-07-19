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
    def forward(ctx, inputs, weights, bias, topk_forward, topk_backward):
        # Get the number of weights.
        number_weights = weigths.shape.numel()

        # Get number percentile of weigths to not set to zero
        topk_forward_percentage = topk_forward / number_weights
        topk_backward_percentage = topk_backward / number_weights

        mask_forward = compute_mask(weights, topk_forward_percentage)
        mask_backward = compute_mask(weights, topk_backward_percentage)

        # output = torch_sparse.spmm(indices, weights, out_features, in_features, inputs.t()).t()
        target = torch.sparse.FloatTensor(
            indices, weights, torch.Size([out_features, in_features]),
        ).to_dense()
        output = torch.mm(target, inputs.t()).t()

        ctx.save_for_backward(inputs, weights, indices)
        ctx.in1 = k
        ctx.in2 = out_features
        ctx.in3 = in_features
        ctx.in4 = max_size

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
    
    def compute_mask(w, topk = 0.5):
        if w.is_sparse:
            threshold = np.quantile(w.values().detach(), topk)
            mask = w.values().detach() > threshold
        else:
            threshold = np.quantile(w.reshape(-1).detach(), topk)
            mask = w.reshape(-1).detach() > threshold
        return mask.reshape(w.shape)

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
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        return topkTraining.apply(inputs, self.weights, self.bias, self.topk_forward,
                                  self.topk_backward)



#%%
