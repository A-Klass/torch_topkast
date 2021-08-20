#%% Imports

import math
import numpy as np
import torch
import torch.nn as nn
import torch_sparse

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
        self.weight = torch.empty((out_features, in_features), **factory_kwargs)
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        self.weight_vector = None
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
            mask = torch.where(matrix.values().detach().abs() >= threshold)
        else:
            threshold = torch.quantile(matrix.reshape(-1).detach().abs(), p)
            mask = torch.where(matrix.detach().abs() >= threshold)
            
        return mask
    
    # Compute set difference between forward and backward active sets
    # Rather lengthy because np.where's ordering by index must be circumvented
     
    def compute_justbwd(self):
        
        f = torch.zeros_like(self.weight)
        b = torch.zeros_like(self.weight)
        
        f[self.indices_forward] = 1
        b[self.indices_backward] = 1
        
        return torch.where(b - f == 1)
    
    # Define update step for active set
    
    def update_active_param_set(self) -> None:
        if self.weight_vector is not None:
            self.weight[self.indices_backward] = self.weight_vector.detach()
        
        self.indices_forward = self.compute_mask(self.weight, self.p_forward)
        self.indices_backward = self.compute_mask(self.weight, self.p_backward)
        self.just_backward = self.compute_justbwd()
        
        self.weight_vector = nn.Parameter(
            torch.cat((self.weight[self.indices_forward].detach(),
                       torch.zeros(len(self.just_backward[0])).detach())))
        self.indices = (torch.cat((self.indices_forward[0], self.just_backward[0])), 
                        torch.cat((self.indices_forward[1], self.just_backward[1])))
        
        self.set_fwd = range(len(self.indices_forward[0]))
        # self.set_bwd = self.weight[self.indices_backward]
        self.set_justbwd = range(len(self.indices_forward[0]), len(self.weight_vector))
    
    # Define forward pass
    
    def forward(self, inputs, sparse=True):
        if sparse:
            if self.training:
                # Sparse training
                output = torch_sparse.spmm(
                    self.indices, 
                    self.weight_vector, 
                    self.out_features, 
                    self.in_features, 
                    inputs.t()).t()
                output += self.bias
            else:
                # Sparse forward pass without training
                with torch.no_grad():
                    output = torch_sparse.spmm(
                    self.indices, 
                    self.weight_vector, 
                    self.out_features, 
                    self.in_features, 
                    inputs.t()).t()
                    output += self.bias
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
