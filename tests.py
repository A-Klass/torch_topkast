#%% Imports

import numpy as np
import topkast_linear as tk
import topkast_loss as tkl
import torch
import torch.nn as nn
import unittest

#%%

class TestTopKastLinear(unittest.TestCase):
    
    def test_weights_are_initialized_nonzero(self):
        
        layer_tkl = tk.TopKastLinear(
            in_features=1000, 
            out_features=10, 
            p_forward=0.6,
            p_backward=0.4,
            bias=True)
        
        layer_tkl.reset_parameters()
        
        for param in [layer_tkl.weight, layer_tkl.bias]:
            sum_elements = param.detach().numpy().flatten().sum()
            self.assertNotAlmostEqual(sum_elements, 0)
            
    def test_weights_have_grads(self):
        
        layer_tkl = tk.TopKastLinear(
            in_features=1000, 
            out_features=10, 
            p_forward=0.6,
            p_backward=0.4,
            bias=True)
        
        self.assertTrue(layer_tkl.sparse_weights.requires_grad)
        self.assertTrue(layer_tkl.bias.requires_grad)
            
    def test_masking_achieves_right_sparsity(self):
        
        layer_tkl = tk.TopKastLinear(
            in_features=1000, 
            out_features=10, 
            p_forward=0.6,
            p_backward=0.4,
            bias=True)
        
        m = layer_tkl.compute_mask(layer_tkl.p_forward)
        # TODO find better way to define tol
        tol = layer_tkl.p_forward * 0.01
        
        self.assertGreaterEqual(
            1 - (len(m[0]) / layer_tkl.weight.numel()), 
            layer_tkl.p_forward - tol)
        self.assertLessEqual(
            1 - (len(m[0]) / layer_tkl.weight.numel()),
            layer_tkl.p_forward + tol)   
        
    # def test_justbwd_is_diff_between_bwd_and_fwd(self):
        
    #     layer_tkl = tk.TopKastLinear(
    #         in_features=1000, 
    #         out_features=10, 
    #         p_forward=0.6,
    #         p_backward=0.4,
    #         bias=True)       
        
    #     fwd = layer_tkl.set_fwd().to_dense()
    #     bwd = layer_tkl.set_bwd().to_dense()
    #     justbwd = layer_tkl.set_justbwd().to_dense()
    #     is_identical = (justbwd == bwd - fwd)
        
    #     self.assertTrue(all(is_identical.flatten()))
        
#%%

class TestTopKastLoss(unittest.TestCase):
    
    def test_penalty_is_l2_for_non_topkast(self):
        
        class NN(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_in = nn.Linear(10, 128)
                self.activation = nn.ReLU()
                self.hidden = nn.Linear(128, 1)
            def forward(self, x):
                return self.activation(
                    self.hidden(self.activation(self.layer_in(x))))

        net = NN()
        loss_tk = tkl.TopKastLoss(loss=nn.MSELoss, net=net)
        
        penalty = loss_tk.compute_norm_active_set()
        standard_norm = (torch.linalg.norm(net.layer_in.weight) + 
                          torch.linalg.norm(net.hidden.weight))
        
        self.assertEqual(
            np.round(penalty.detach().numpy()), 
            np.round(standard_norm.detach().numpy())) 
        
    # def test_penalty_is_l2_for_topkast(self):
        
    #     class NN(nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.layer_in = tk.TopKastLinear(10, 128, 0.6, 0.4)
    #             self.activation = nn.ReLU()
    #             self.hidden = tk.TopKastLinear(128, 1, 0.6, 0.4)
    #         def forward(self, x):
    #             return self.activation(
    #                 self.hidden(self.activation(self.layer_in(x))))

    #     net = NN()
    #     loss_tk = tkl.TopKastLoss(loss=nn.MSELoss, net=net)
        
    #     penalty = loss_tk.compute_norm_active_set()
        
    #     w_in_fwd, w_h_fwd, w_in_jbwd, w_h_jbwd = [
    #         w.to_dense() 
    #         for w in [net.layer_in.set_fwd(), net.hidden.set_fwd(),
    #                   net.layer_in.set_justbwd(), net.hidden.set_justbwd()]]
        
    #     coeff_in = 1 - net.layer_in.p_forward
    #     coeff_h = 1 - net.hidden.p_forward        
        
    #     standard_norm = (
    #         torch.linalg.norm(w_in_fwd) + torch.linalg.norm(w_h_fwd) +
    #         torch.linalg.norm(w_in_jbwd) / coeff_in + 
    #         torch.linalg.norm(w_h_jbwd) / coeff_h)   
        
    #     self.assertEqual(
    #         np.round(penalty.detach().numpy()), 
    #         np.round(standard_norm.detach().numpy()))
        
#     def loss_is_differentiable(self):
        
#         class NN(nn.Module):
#             def __init__(self):
#                 super().__init__()
#                 self.layer_in = tk.TopKastLinear(10, 128, 0.6, 0.4)
#                 self.activation = nn.ReLU()
#                 self.hidden = tk.Linear(128, 1)
#             def forward(self, x):
#                 return self.activation(
#                     self.hidden(self.activation(self.layer_in(x))))
            
#         net = nn()
#         loss_tk = tkl.TopKastLoss(loss=nn.MSELoss)
            
#         l = loss_tk(torch.rand(10), torch.rand(10), net)
#         l.sum().backward()
        
#         for i in net.children():

#             self.assertEqual(i.weight.grad.shape[0], i.in_features)
#             self.assertEqual(i.weight.grad.shape[1], i.out_features)
#             self.assertNotNone(i.bias.grad)       
            
#     def gradient_has_right_sparsity(self):
        
#         class NN(nn.Module):
#             def __init__(self):
#                 super().__init__()
#                 self.layer_in = tk.TopKastLinear(10, 128, 0.6, 0.4)
#                 self.activation = nn.ReLU()
#                 self.hidden = tk.Linear(128, 1)
#             def forward(self, x):
#                 return self.activation(
#                     self.hidden(self.activation(self.layer_in(x))))
            
#         net = nn()
#         loss_tk = tkl.TopKastLoss(loss=nn.MSELoss)
        
#         l = loss_tk(torch.rand(10), torch.rand(10), net)
#         l.sum().backward()
        
#         is_zero = (net.layer_in.weight.grad == 0.)
#         # TODO find better way to define tol
#         tol = net.layer_in.p_backward * 0.01
        
#         self.assertGreaterEqual(
#             is_zero.sum() / is_zero.numel(),
#             torch.tensor(net.layer_in.p_backward - tol))
#         self.assertLessEqual(
#             is_zero.sum() / is_zero.numel(),
#             torch.tensor(net.layer_in.p_backward + tol))

#%%
        
if __name__ == '__main__':
    unittest.main()

#%% 

# TODO
# test that active set actually changes over iterations

#%%

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_in = tk.TopKastLinear(10, 128, 0.6, 0.4)
        self.activation = nn.ReLU()
        self.hidden = tk.TopKastLinear(128, 1, 0.6, 0.4)
    def forward(self, x):
        return self.activation(
            self.hidden(self.activation(self.layer_in(x))))

net = NN()
loss_tk = tkl.TopKastLoss(loss=nn.MSELoss, net=net)

penalty = loss_tk.compute_norm_active_set()
for child in net.children():
    for name in child._parameters.keys():
        print(name)