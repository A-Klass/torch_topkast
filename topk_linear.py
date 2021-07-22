#%%
import math
import warnings

import numpy as np
import torch
import torch.nn as nn

#%%
class topkTraining(torch.autograd.Function):
    """ 
    Custom pytorch function to handle changing connections and topkast
    ctx = context in which you can save stuff that doesn't need gradients.
    """

    @staticmethod
    def forward(ctx, inputs, weights, bias, indices_forward, indices_backward):
        # output = torch_sparse.spmm(indices, weights, out_features, in_features, inputs.t()).t()
        weigth = torch.sparse_coo_tensor(
            indices_forward, weights[indices_forward], weights.shape)
        output = torch.sparse.addmm(bias.unsqueeze(1), weigth, inputs.t()).t()

        ctx.save_for_backward(inputs, weights, bias)

        # ctx.save_for_backward(inputs, weights, bias, indices_backward)
        # ctx.in1 = k
        # ctx.in2 = out_features
        # ctx.in3 = in_features
        # ctx.in4 = max_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # inputs, weights, indices = ctx.saved_tensors
        # k = ctx.in1
        # out_features = ctx.in2
        # in_features = ctx.in3
        # max_size = ctx.in4

        # device = grad_output.device
        # p_index = torch.LongTensor([1, 0])
        # new_indices = torch.zeros_like(indices).to(device=device)
        # new_indices[p_index] = indices

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

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
        self.indices_forward = self.compute_mask(self.topk_forward)
        self.indices_backward = self.compute_mask(self.topk_backward)

    def compute_mask(self, topk):
        w = self.weight
        topk_percentage = topk / w.shape.numel()
        if w.is_sparse:
            threshold = np.quantile(w.values().detach(), topk_percentage)
            mask = np.where(w.values().detach() <= threshold)
        else:
            threshold = np.quantile(w.reshape(-1).detach(), topk_percentage)
            mask = np.where(w.detach() <= threshold)
        return mask
    
    def forward(self, inputs, sparse = True):
        self.indices_forward = self.compute_mask(self.topk_forward)
        self.indices_backward = self.compute_mask(self.topk_backward)

        if sparse:
            if self.training:
                # Sparse training
                output = topkTraining.apply(inputs, self.weight, self.bias, self.indices_forward,
                                            self.indices_backward)
            else:
                # Sparse forward pass without training
                with torch.no_grad():
                    weights = torch.sparse_coo_tensor(self.indices_forward, 
                                                    self.weight[self.indices_forward],
                                                    self.weight.shape)
                    output = torch.sparse.addmm(self.bias.unsqueeze(1), weights, inputs.t()).t()
        else:
            # Dense training is not possible, only a dense forward pass for prediction
            with torch.no_grad():
                output = torch.addmm(self.bias.unsqueeze(1), self.weight, inputs.t()).t()
        
        return output


#%%
# objective function
def objective(x):
	return x[0]**2.0 + x[1]**2.0

#%%
layer = TopkLinear(2, 1, 0, 0)
x = torch.rand((1, 2))
y = torch.tensor([objective(x_) for x_ in x])
# layer.training = False
# layer(x, sparse = False)
y_hat = layer(x)
loss = torch.nn.MSELoss()
l = loss(y_hat, y)
# %%
l.backward()
# %%
