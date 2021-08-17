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
    def forward(ctx, inputs, sparse_weights, bias, indices_backward):
        
        # Compute output as weighted sum of inputs plus bias term
        
        output = torch.sparse.addmm(
            bias.unsqueeze(1), 
            sparse_weights, 
            inputs.t()).t()
        
        # Store values in saved tensors to access during backward()
        
        ctx.save_for_backward(inputs, sparse_weights, bias)
        
        # Store backward indices in context
        
        ctx.indices_backward = indices_backward

        return output

    @staticmethod
    def backward(ctx, grad_output):

        # Get saved tensors
        
        inputs, sparse_weights, bias = ctx.saved_tensors
        
        # Initialize gradients
        
        grad_inputs = grad_weights = grad_bias = None
        
        # Get backward indices from context
        
        indices_backward = ctx.indices_backward

        # Compute grad wrt inputs if necessary
        
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.mm(sparse_weights.to_dense())
            # grad_inputs = torch.sparse.mm(sparse_weights, grad_output.t()).t()
        
        # Compute grad wrt weights if necessary
        
        if ctx.needs_input_grad[1]:
            grad_weights = grad_output.t().mm(inputs)
            grad_weights = torch.sparse_coo_tensor(
                indices=indices_backward,
                values=grad_weights[indices_backward], 
                size=grad_weights.shape)
            
        # Compute grad wrt bias if necessary (and bias is specified)
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_inputs, grad_weights, grad_bias, None

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
        
        # Perform basic input checks
        
        for i in [in_features, out_features]:
            assert type(i) == int, 'integer input required'
            assert i > 0, 'inputs must be > 0'
        for i in [p_forward, p_backward]:
            assert type(i) == float, 'float input required'
            assert i > 0, 'inputs must be between 0 and 1'
            assert i <= 1, 'inputs must be between 0 and 1'
        assert p_forward > p_backward
        assert type(bias) == bool
            
        # Initialize
        
        self.in_features, self.out_features = in_features, out_features
        self.p_forward, self.p_backward = p_forward, p_backward
        self.weight = torch.empty(
            (out_features, in_features), **factory_kwargs)
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        self.update_active_param_set()
        
    # Define weight initialization (He et al., 2015)

    def reset_parameters(self) -> None:
        
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
  
    # Define masking operations

    @staticmethod
    def compute_mask(matrix, p):
                
        if matrix.is_sparse:
            threshold = torch.quantile(matrix.values().detach().abs(), p)
            mask = np.where(matrix.values().detach().abs() >= threshold)
        else:
            threshold = torch.quantile(matrix.reshape(-1).detach().abs(), p)
            mask = np.where(matrix.detach().abs() >= threshold)
            
        return mask
    
    # Compute set difference between forward and backward active sets
    # Rather lengthy because np.where's ordering by index must be circumvented
     
    def compute_justbwd(self):
        
        f, b = self.indices_forward, self.indices_backward
        tuples_fwd, tuples_bwd = [], []
        
        for r, c in zip(f[0], f[1]):
            tuples_fwd.append([r, c])
        for r, c in zip(b[0], b[1]):
            tuples_bwd.append([r, c])

        setdiff = lambda x, y: [x_ for x_ in x if x_ not in y]
        just_bwd = np.array(setdiff(tuples_bwd, tuples_fwd))
        
        return just_bwd[:, 0], just_bwd[:, 1]
    
    # Define update step for active set
    
    def update_active_param_set(self) -> None:
        self.indices_forward = self.compute_mask(self.weight, self.p_forward)
        self.indices_backward = self.compute_mask(self.weight, self.p_backward)
        self.just_backward = self.compute_justbwd()
        
        self.sparse_weights = torch.sparse_coo_tensor(
            indices=self.indices_forward, 
            values=self.weight[self.indices_forward], #/ (1 - self.p_forward),
            size=self.weight.shape,
            requires_grad=True)
        
        self.set_fwd = self.weight[self.indices_forward]
        self.set_bwd = self.weight[self.indices_backward]
        self.set_justbwd = self.weight[self.just_backward]
    
    # Define forward pass
    
    def forward(self, inputs, sparse=True):
        if sparse:
            if self.training:
                # Sparse training
                output = TopKastTraining.apply(
                    inputs, 
                    self.sparse_weights, 
                    self.bias,
                    self.indices_backward)
            else:
                # Sparse forward pass without training
                with torch.no_grad():
                    output = torch.sparse.addmm(
                        self.bias.unsqueeze(1), 
                        self.sparse_weights, 
                        inputs.t()).t()
        else:
            # Dense training is not possible, only a dense forward pass for 
            # prediction
            with torch.no_grad():
                output = torch.addmm(
                    self.bias.unsqueeze(1), 
                    self.weight, 
                    inputs.t()).t()
        
        return output
        
# %%
