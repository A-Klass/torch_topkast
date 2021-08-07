#%% Imports

import math
import numpy as np
import torch
import torch.nn as nn

#%% TopKast training function

class TopKastTraining(torch.autograd.Function):
    """ 
    Custom pytorch function to handle changing connections and topkast.
    ctx = context in which you can save stuff that doesn't need gradients.
    """

    @staticmethod
    def forward(ctx, inputs, weights, bias, indices_forward, indices_backward):
        
        # Compute sparse weight tensor
        weights_used = torch.sparse_coo_tensor(
            indices=indices_forward, 
            values=weights[indices_forward], 
            size=weights.shape)
        
        # Compute output as weighted sum of inputs plus bias term
        output = torch.sparse.addmm(
            mat=bias.unsqueeze(1), 
            mat1=weights_used, 
            mat2=inputs.t()).t()
        
        # Store values in saved tensors to access during backward()
        ctx.save_for_backward(inputs, weights, bias)
        
        # Store backward indices in context
        ctx.indices_backward = indices_backward

        return output

    @staticmethod
    def backward(ctx, grad_output):

        # Get saved tensors
        inputs, weights, bias = ctx.saved_tensors
        
        # Initialize gradients
        grad_inputs = grad_weights = grad_bias = None
        
        # Get backward indices from context
        indices_backward = ctx.indices_backward

        # Compute grad wrt inputs if necessary
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.mm(weights)
        
        # Compute grad wrt weights if necessary
        if ctx.needs_input_grad[1]:
            grad_weights = grad_output.t().mm(inputs)
            grad_weights = torch.sparse_coo_tensor(
                indices=indices_backward,
                values=grad_weights[indices_backward], 
                size=grad_weights.shape)
            
        # Compute grad wrt bias if necessary (and bias is specified)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).to_sparse()

        return grad_inputs, grad_weights, grad_bias, None, None

#%% TopKast linear layer

class TopKastLinear(nn.Module):
    """"
    Sparse adaptation of nn.Linear module with topkast.
    """
    
    def __init__(self, in_features: int, out_features: int, p_forward: float, 
                 p_backward: float, bias: bool=True, device=None, 
                 dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TopKastLinear, self).__init__()
        
        # Assert that active parameter set in backward pass is at least as 
        # large as in forward pass to allow for it to change
        assert p_forward >= p_backward
            
        # Initialize
        self.in_features = in_features
        self.out_features = out_features
        self.p_forward = p_forward
        self.p_backward = p_backward
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    # Define weight initialization (He et al., 2015)

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.indices_forward = self.compute_mask(self.p_forward)
        self.indices_backward = self.compute_mask(self.p_backward)
        
    # Define masking operations

    def compute_mask(self, p):
        w = self.weight
        if w.is_sparse:
            threshold = torch.quantile(w.values().detach().abs(), 1 - p)
            mask = np.where(w.values().detach().abs() >= threshold)
        else:
            threshold = torch.quantile(w.reshape(-1).detach().abs(), 1 - p)
            mask = np.where(w.detach().abs() >= threshold)
        return mask
    
    # Define forward pass
    
    def forward(self, inputs, sparse=True):
        
        self.indices_forward = self.compute_mask(self.p_forward)
        self.indices_backward = self.compute_mask(self.p_backward)

        if sparse:
            if self.training:
                # Sparse training
                output = TopKastTraining.apply(inputs, self.weight, 
                                               self.bias, 
                                               self.indices_forward,
                                               self.indices_backward)
            else:
                # Sparse forward pass without training
                with torch.no_grad():
                    weights = torch.sparse_coo_tensor(
                        indices=self.indices_forward, 
                        values=self.weight[self.indices_forward],
                        size=self.weight.shape)
                    output = torch.sparse.addmm(
                        mat=self.bias.unsqueeze(1), 
                        mat1=weights, 
                        mat2=inputs.t()).t()
        else:
            # Dense training is not possible, only a dense forward pass for 
            # prediction
            with torch.no_grad():
                output = torch.addmm(
                    mat=self.bias.unsqueeze(1), 
                    mat1=self.weight, 
                    mat2=inputs.t()).t()
        
        return output
    
    # Define field to access sparse weights
    
    def sparse_weights(self):
        weights = torch.sparse_coo_tensor(
            indices=self.indices_forward, 
            values=self.weight[self.indices_forward],
            size=self.weight.shape)
        return weights

# %%
