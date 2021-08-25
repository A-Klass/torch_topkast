"""Top K sparse implementations for torch.nn.Modules.

As of now, we provide a sparse version of a linear nn.Module.

Intended usage of TopKastLinear is much like any torch layer module:

torch.nn.Sequential(
    torch.nn.TopKastLinear(13,13),
    torch.nn.ReLU())
"""

#%% Imports

import math
import torch
import torch.nn as nn
from torch_sparse import spmm

#%% TopKast linear layer
class TopKastLinear(nn.Module):
    """"
    Sparse adaptation of nn.Linear module with topkast (https://arxiv.org/abs/2106.03517).
    
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
                 bias: bool=True, 
                 device: torch.device = None, 
                 dtype=None) -> None:
        """ Initialize the layer
        Note:
            We refer to p_forward, p_backward as forward and backward
            sparsity (as in this percentage we set to zero), respectively.
            Since there are more parameters affected in the backward pass 
            than in the forward pass, we enforce p_forward >= p_backward.
                
        Args:
            in_features (int): input dimension (# of batch size)
            out_features (int): output dimension (# of columns in layer)
            p_forward (float): Forward sparsity as percentage. 
                               How many parameters (in %) are set to 0.
            p_backward (float): Backward sparsity as percentage.
                                How many parameters (in %) are set to 0.
            bias (bool): whether to add bias term
            device (str): either 'cpu' or 'cuda'
        """
        
        # Input checks
        
        for i in [in_features, out_features]:
            if i < 0:
                raise ValueError(i % "must be > 0")
        for j in [p_forward, p_backward]:
            if j < 0:
                raise ValueError(j % "must be >= 0")
            if j >= 1:
                raise ValueError(j % "must be < 1")
            
        # Usually, you would want something like: values that make up 
        # the top 5 % (by magnitude) such that the sparsity is 95%.
        # If the "forward sparsity" is 95% and we backpropagate for
        # a superset B⊃A then the "backward sparsity" must be lower.
        assert p_forward >= p_backward
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.device = device
        self.dtype = dtype
 
        super(TopKastLinear, self).__init__()
        
        self.in_features, self.out_features = in_features, out_features
        self.p_forward, self.p_backward = p_forward, p_backward
        
        # Dense tensor with full dimensionality to store weights in:
        self.weight = torch.empty((out_features, in_features), **factory_kwargs)
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        self.active_fwd_weights = None
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
            
    # Masking operations
    @staticmethod
    def compute_mask(matrix,
                     p: float):
        """
        Get the indices of `matrix` values that belong to
        the `p` biggest absolute values in this matrix
        (as in: top 1 % of the layer, by weight norm).
        In the paper this refers to D or D+M, respectively.
        
        Args:
            matrix (torch.Tensor): weight matrix
            p(float): self.p_forward; p-quantile
            
        Returns:
            mask as torch.Tensor tuple containing indices of
            matrix' biggest values
        """
        threshold = torch.quantile(torch.abs(matrix), p)
        return torch.where(torch.abs(matrix) >= threshold)
    
    def compute_just_bwd(self):
        """
        Compute set difference between forward set (A) and backward set (B).
        Supposed to be called within update_active_param_set() which 
        sets the indices by computing the mask for self.idx_fwd and self.idx_bwd,
        thus creating self.idx_fwd and self.idx_bwd
        
        Input:
            The mask from compute_mask(matrix, K)
            
        Returns:
            torch.Tensor containing indices of B\A
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
        """
        Updates the dense (complete) weight tensor with 
        newly learned weights from B.
        Computes the masks to get the subsets of active parameters
        (sets A, B, and B\A) in terms of indices. 
        """
        # when not calling for first time, then update 
        # all parameters affected in the backward pass
        if self.active_fwd_weights is not None:
            self.weight[self.idx_fwd] = self.active_fwd_weights.detach()[self.set_fwd]       
        self.idx_fwd = self.compute_mask(self.weight, self.p_forward)
        self.idx_bwd = self.compute_mask(self.weight, self.p_backward)
        self.idx_justbwd = self.compute_just_bwd()
        
        # The vector of active weights for the forward pass contains the 
        # ones from A as well as placeholders for B\A with value=0.00.
        # We do this since for a sparse coo tensor it is impossible
        # to update values in-place.
        self.active_fwd_weights = nn.Parameter(
            torch.cat((self.weight[self.idx_fwd].detach(),
                       torch.zeros(len(self.idx_justbwd[0]), device=self.device)))) # paddings for B\A
        self.indices = torch.cat((torch.cat((self.idx_fwd[0], self.idx_justbwd[0])).reshape(1,-1), 
                        torch.cat((self.idx_fwd[1], self.idx_justbwd[1])).reshape(1,-1)), 0).to(self.device)
        
        self.set_fwd = range(len(self.idx_fwd[0]))
        self.set_justbwd = range(len(self.idx_fwd[0]), len(self.active_fwd_weights))        

    def reset_justbwd_weights(self) -> None:
        """
        Updates weight matrix for B\A and resets the corresponding weights in 
        the active_fwd_weights.
        """
        self.weight[self.idx_justbwd] += self.active_fwd_weights.detach()[self.set_justbwd]
        with torch.no_grad():
            self.active_fwd_weights[self.set_justbwd] = 0

    # Define forward pass
    def forward(self, inputs, sparse=True):
        if sparse:
            if self.training:
                # Sparse training
                # breakpoint()
                output = spmm(
                    self.indices, 
                    self.active_fwd_weights, 
                    self.out_features, 
                    self.in_features, 
                    inputs.t()).t()
                output += self.bias
            else:
                # Sparse forward pass without training
                with torch.no_grad():
                    output = spmm(
                    self.indices, 
                    self.active_fwd_weights, 
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
