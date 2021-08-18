#%% Usage:
    # 1. Make sure that nose is installed: pip install nose
    # 2. From the Anaconda terminal, call: nosetests test/tests.py

#%% Imports

import numpy as np
import topkast_linear as tk
import topkast_loss as tkl
import torch
import torch.nn as nn
import unittest

#%% 

def make_test_layer():
    return tk.TopKastLinear(
        in_features=1000, 
        out_features=2, 
        p_forward=0.6,
        p_backward=0.4,
        bias=True)

def make_test_net():
    class NN(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_in = tk.TopKastLinear(10, 128, 0.6, 0.4)
                self.activation = nn.ReLU()
                self.layer_out = nn.Linear(128, 1)
            def forward(self, x):
                return self.activation(
                    self.layer_out(self.activation(self.layer_in(x))))
            
    return NN()

#%%

class TestTopKastLinear(unittest.TestCase):
    
    def test_weights_are_initialized_nonempty(self):
        
        layer_tkl = make_test_layer()
        layer_tkl.reset_parameters()
        
        for param in [layer_tkl.weight, layer_tkl.bias]:
            sum_elements = param.detach().numpy().flatten().sum()
            self.assertNotAlmostEqual(sum_elements, 0)
            
    def test_weights_have_grads(self):
        
        layer_tkl = make_test_layer()
        
        self.assertTrue(layer_tkl.sparse_weights.requires_grad)
        self.assertTrue(layer_tkl.bias.requires_grad)
            
    def test_masking_achieves_right_sparsity(self):
        
        layer_tkl = make_test_layer()
        
        m = layer_tkl.compute_mask(layer_tkl.weight, layer_tkl.p_forward)
        # TODO find better way to define tol
        tol = layer_tkl.p_forward * 0.01
        
        self.assertGreaterEqual(
            1 - (len(m[0]) / layer_tkl.weight.numel()), 
            layer_tkl.p_forward - tol)
        self.assertLessEqual(
            1 - (len(m[0]) / layer_tkl.weight.numel()),
            layer_tkl.p_forward + tol)   
        
    def test_justbwd_is_diff_between_bwd_and_fwd(self):
        
        layer_tkl = make_test_layer()
        
        fwd = np.sort(layer_tkl.set_fwd)
        bwd = np.sort(layer_tkl.set_bwd)
        justbwd = np.sort(layer_tkl.set_justbwd)
        
        tol = layer_tkl.weight.numel() * 0.001
        if tol < 0:
            tol = 1
        
        self.assertTrue(len(bwd) - len(fwd) == len(justbwd))
        self.assertTrue(len(justbwd) == len(np.intersect1d(justbwd, bwd)))
        self.assertLessEqual(len(np.intersect1d(justbwd, fwd)), tol)
        
#%%

class TestTopKastLoss(unittest.TestCase):
    
    def test_penalty_is_l2(self):

        net = make_test_net()
        loss_tk = tkl.TopKastLoss(loss=nn.MSELoss, net=net)
        
        penalty = loss_tk.compute_norm_active_set()
        
        standard_norm_in = (
            torch.linalg.norm(net.layer_in.set_fwd) + 
            (torch.linalg.norm(net.layer_in.set_justbwd) / 
            (1 - net.layer_in.p_forward)))
        
        standard_norm_out = torch.linalg.norm(net.layer_out.weight)
        
        standard_norm = standard_norm_in + standard_norm_out
        
        self.assertEqual(
            penalty.detach().numpy(), standard_norm.detach().numpy())
        
    def loss_is_differentiable(self):
        
        net = make_test_net()
        loss_tk = tkl.TopKastLoss(loss=nn.MSELoss, net=net)
        
        l = loss_tk(torch.rand(10), torch.rand(10), net)
        l.sum().backward()
        
        for i in net.children():

            self.assertEqual(i.weight.grad.shape[0], i.in_features)
            self.assertEqual(i.weight.grad.shape[1], i.out_features)
            self.assertNotNone(i.bias.grad)       
            
    def gradient_has_right_sparsity(self):
        
        net = make_test_net()
        loss_tk = tkl.TopKastLoss(loss=nn.MSELoss, net=net)
        
        l = loss_tk(torch.rand(10), torch.rand(10), net)
        l.sum().backward()
        
        is_zero = (net.layer_in.weight.grad == 0.)
        # TODO find better way to define tol
        tol = net.layer_in.p_backward * 0.01
        
        self.assertGreaterEqual(
            is_zero.sum() / is_zero.numel(),
            torch.tensor(net.layer_in.p_backward - tol))
        self.assertLessEqual(
            is_zero.sum() / is_zero.numel(),
            torch.tensor(net.layer_in.p_backward + tol))

#%%
        
if __name__ == '__main__':
    unittest.main()

#%% 

# TODO
# test that active set actually changes over iterations

#%%