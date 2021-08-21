"""Top K sparse implementations for torch.nn.Modules.

As of now, we provide a sparse version of a linear nn.Module.

Intended usage of TopKastLinear is much like any torch layer module:

torch.nn.Sequential(
    torch.nn.TopKastLinear(13,13),
    torch.nn.ReLU())
"""

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
    
    Includes a forward sparse with selecting Top K weights in 
    the layer, updating the active parameter set A
    We distinguish between parameter sets with different flavors:
     - Forward set (A): Weights used to compute output of model in
       forward pass.
     - Backward set (B): Weights used and updated in backward pass. 
     - Complete set of weights of the "dense" model(Θ).
    It holds: Θ⊃B⊃A. Θ\B are not used.
    A and B denote subsets of Θ (=all parameters) and refer to the
    respective active parameter sets. 
       
    """
    
    def __init__(self,
                 in_features: int, 
                 out_features: int, 
                 p_forward: float, 
                 p_backward: float, 
                 include_bias: bool=True, 
                 device='cpu', 
                 dtype=None) -> None:
        """ Initialize the layer
        
        Args:
            in_features (int): input dimension
            out_features (int): output dimension
            p_forward (float): forward sparsity as percentage
            p_backward (float): backward sparsity as percentage
            include_bias (bool): whether to add bias term
            device (str): either 'cpu' or 'cuda'
            dtype: do we need this?

        """
        
        # Input checks
        
        for i in [in_features, out_features]:
            if i < 0:
                raise ValueError(i % "must be > 0")
        for j in [p_forward, p_backward]:
            if j < 0:
                raise ValueError(j % "must be > 0")
            if j > 1:
                raise ValueError(j % "must be <=1")
            
        # Usually, you would want something like: values that make up 
        # the top 5 % (by magnitude) such that the sparsity is 95%.
        # If the "forward sparsity" is 95% and we backpropagate for
        # a superset B⊃A then the "backward sparsity" must be lower.
        assert p_forward > p_backward 
        
        assert device == "cpu" or device == "cuda"
        if device == "cuda":
            assert torch.cuda.is_available()
            
        factory_kwargs = {'device': device, 'dtype': dtype}
 
        super(TopKastLinear, self).__init__()
        
        self.in_features, self.out_features = in_features, out_features
        self.p_forward, self.p_backward = p_forward, p_backward
        
        # Dense tensor with full dimensionality to store weights in:
        self.weight = torch.empty((out_features, in_features), **factory_kwargs)
        
        if include_bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        self.weight_vector = None
        self.update_active_param_set()
        
    # Define weight initialization (He et al., 2015)

    def reset_parameters(self) -> None:
        
        torch.nn.init.kaiming_uniform_(self.weight,
                                       a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
  
    
    def norm(tensor: torch.Tensor, norm: str='abs'):
        assert norm == 'abs' or norm == 'euclidean'
        if norm == 'abs':
            norm_tensor = tensor.abs()
        else:
            norm_tensor = tensor.square()
        return norm_tensor
    
    # Masking operations
    @staticmethod
    def compute_mask(matrix,
                     K: float,
                     norm: str='abs'):
        """
        Get the indices of `matrix` values that belong to
        the `K` biggest absolute values in this matrix
        (as in: top 1 % of the layer, by weight norm).
        Support for Euclidean norm may be added later on (depending
        on the project's progress)
        
        Args:
            matrix (torch.Tensor): weight matrix
            K (float): self.p_forward; p-quantile
            
        Returns:
            torch.Tensor with indices
        """
        
        if matrix.is_sparse:
            threshold = torch.quantile(matrix.values().detach().norm(norm), K)
            mask = torch.where(matrix.values().detach().norm(norm) >= threshold)
        else:
            threshold = torch.quantile(matrix.reshape(-1).detach().norm(norm), K)
            mask = torch.where(matrix.detach().norm(norm)>= threshold)
        return mask
    
    def compute_just_bwd(self):
        """
        Compute set difference between forward set (A) and backward set (B).
        Supposed to be called within update_active_param_set() which 
        sets the indices by computing the mask for self.idx_fwd and self.idx_bwd,
        thus creating self.idx_fwd and self.idx_bwd
        
        Input:
            The mask from compute_mask(matrix, K)
            
        Returns:
            torch.Tensor cpntaining indices of B\A
        """
        
        assert self.idx_fwd is not None and self.idx_bwd is not None, \
            "make sure that the indices are assigned within \
                update_active_param_set() before calling this function."
        
        A = torch.zeros_like(self.weight)
        B = torch.zeros_like(self.weight)
        
        A[self.idx_fwd] = 1
        B[self.idx_bwd] = 1
        
        return torch.where(B - A == 1)
    
    # Update step for active set
    def update_active_param_set(self) -> None:
        # when not calling for first time, then update 
        # all parameters affected in the backward pass
        if self.weight_vector is not None:
            self.weight[self.idx_bwd] = self.weight_vector.detach()
        
        self.idx_fwd = self.compute_mask(self.weight, self.p_forward)
        self.idx_bwd = self.compute_mask(self.weight, self.p_backward)
        self.just_backward = self.compute_just_bwd()
        
        self.weight_vector = nn.Parameter(
            torch.cat((self.weight[self.indices_forward].detach(),
                       torch.zeros(len(self.just_backward[0])).detach())))
        self.indices = (torch.cat((self.indices_forward[0], self.just_backward[0])), 
                        torch.cat((self.indices_forward[1], self.just_backward[1])))
        
        self.set_fwd = self.weight[self.indices_forward]
        self.set_bwd = self.weight[self.indices_backward]
        self.set_justbwd = self.weight[self.just_backward]
    
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
