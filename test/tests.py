#%% Usage:
    # 1. Make sure that nose is installed: pip install nose
    # 2. From the Anaconda terminal, call: nosetests test/tests.py

#%% Imports

import numpy as np
from torch_topkast.topkast_linear import TopKastLinear
from torch_topkast.topkast_loss import TopKastLoss
from torch_topkast.topkast_trainer import TopKastTrainer
import torch
import torch.nn as nn
import torch.optim
import unittest

#%% set testing params

# tests only run for certain magnitudes: for very small inputs/nets, chances 
# are that no updates occur

input_features = 5
hidden_neurons = 8

#%% define test objects

def make_test_layer():
    return TopKastLinear(
        in_features=input_features, 
        out_features=hidden_neurons, 
        p_forward=0.6,
        p_backward=0.4,
        bias=True)

def make_test_net():
    class NN(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_in = TopKastLinear(
                    input_features, hidden_neurons, 0.6, 0.4)
                self.activation = nn.ReLU()
                self.layer_out = nn.Linear(hidden_neurons, 1)
            def forward(self, x):
                return self.activation(
                    self.layer_out(self.activation(self.layer_in(x))))
            
    return NN()

#%% define test runs

class TestTopKastLinear(unittest.TestCase):
    
    def test_weights_are_initialized_nonzero(self):
        
        layer_tkl = make_test_layer()
        layer_tkl.reset_parameters()
        
        for param in [layer_tkl.weight, layer_tkl.bias]:
            self.assertFalse(all(param.detach().numpy().flatten() == 0.))
            
    def test_weights_have_grads(self):
        
        layer_tkl = make_test_layer()
        
        self.assertTrue(layer_tkl.active_fwd_weights.requires_grad)
        self.assertTrue(layer_tkl.bias.requires_grad)
            
    def test_masking_achieves_right_sparsity(self):
        
        layer_tkl = make_test_layer()
        
        m = layer_tkl.compute_mask(layer_tkl.weight, layer_tkl.p_forward)
        tol = layer_tkl.p_forward * 0.01
        
        self.assertGreaterEqual(
            1 - (len(m[0]) / layer_tkl.weight.numel()), 
            layer_tkl.p_forward - tol)
        self.assertLessEqual(
            1 - (len(m[0]) / layer_tkl.weight.numel()),
            layer_tkl.p_forward + tol)   
        
    def test_right_weights_are_masked(self):
        
        layer_tkl = make_test_layer()
                
        layer_tkl.weight = torch.arange(
            layer_tkl.weight.numel(), dtype=torch.float32)
        layer_tkl.weight = layer_tkl.weight.reshape(
            layer_tkl.out_features, layer_tkl.in_features)
        
        m = layer_tkl.compute_mask(layer_tkl.weight, layer_tkl.p_forward)
        
        self.assertEqual(
            layer_tkl.weight[m][0].item(),
            layer_tkl.weight.numel() * layer_tkl.p_forward)
        self.assertEqual(
            layer_tkl.weight[m][m[0].numel() - 1].item(),
            layer_tkl.weight.numel() - 1.)
        
    def test_justbwd_has_right_size(self):
        
        layer_tkl = make_test_layer()
        
        fwd = layer_tkl.idx_fwd
        bwd = layer_tkl.idx_bwd
        justbwd = layer_tkl.idx_justbwd
        
        self.assertTrue(
            bwd[0].shape[0] - fwd[0].shape[0] == justbwd[0].shape[0])
        
    def test_dense_weights_are_updated(self):
        
        net = make_test_net()
        w_before = net.layer_in.weight.clone()
        loss_tk = TopKastLoss(loss=nn.MSELoss, net=net)
        
        n_obs = 10
        X = torch.rand(n_obs * input_features).reshape(
            n_obs, input_features).float()
        y = torch.rand(n_obs).float().reshape(-1, 1)
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
        
        for i in range(3):
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss_tk(y_hat, y)
            l.sum().backward()
            optimizer.step()
            net.layer_in.update_active_param_set()
            net.layer_in.reset_justbwd_weights()
            i += 1

        self.assertFalse(
            torch.equal(net.layer_in.weight, w_before))
        
    def test_sparse_weights_are_updated(self):
        
        net = make_test_net()
        w_before = net.layer_in.active_fwd_weights.clone()
        loss_tk = TopKastLoss(loss=nn.MSELoss, net=net)
        
        n_obs = 10
        X = torch.rand(n_obs * input_features).reshape(
            n_obs, input_features).float()
        y = torch.rand(n_obs).float().reshape(-1, 1)
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
        
        for i in range(10):
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss_tk(y_hat, y)
            l.sum().backward()
            optimizer.step()
            net.layer_in.update_active_param_set()
            net.layer_in.reset_justbwd_weights()
            i += 1

        self.assertFalse(
            torch.equal(net.layer_in.active_fwd_weights, w_before))
        
    def test_forward_active_set_is_updated(self):
        
        net = make_test_net()
        idx_fwd_before = net.layer_in.idx_fwd
        loss_tk = TopKastLoss(loss=nn.MSELoss, net=net)
        
        n_obs = 10
        X = torch.rand(n_obs * input_features).reshape(
            n_obs, input_features).float()
        y = torch.rand(n_obs).float().reshape(-1, 1)
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
        
        for i in range(3):
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss_tk(y_hat, y)
            l.sum().backward()
            optimizer.step()
            net.layer_in.update_active_param_set()
            net.layer_in.weight += torch.rand_like(net.layer_in.weight)
            net.layer_in.reset_justbwd_weights()
            i += 1
            
        a = torch.zeros_like(net.layer_in.weight)
        b = torch.zeros_like(net.layer_in.weight)
        a[net.layer_in.idx_fwd] = 1
        b[idx_fwd_before] = 1
            
        self.assertFalse(torch.equal(a, b))

#%%

# class TestTopKastLoss(unittest.TestCase):
    
#     def test_penalty_is_l2(self):

#         net = make_test_net()
#         loss_tk = tkl.TopKastLoss(loss=nn.MSELoss, net=net)
        
#         penalty = loss_tk.compute_norm_active_set()
        
#         standard_norm_in = (
#             torch.linalg.norm(net.layer_in.set_fwd) + 
#             (torch.linalg.norm(net.layer_in.set_justbwd) / 
#             (1 - net.layer_in.p_forward)))
        
#         standard_norm_out = torch.linalg.norm(net.layer_out.weight)
        
#         standard_norm = standard_norm_in + standard_norm_out
        
#         self.assertEqual(
#             penalty.detach().numpy(), standard_norm.detach().numpy())
        
#     def loss_is_differentiable(self):
        
#         net = make_test_net()
#         loss_tk = tkl.TopKastLoss(loss=nn.MSELoss, net=net)
        
#         l = loss_tk(torch.rand(10), torch.rand(10), net)
#         l.sum().backward()
        
#         for i in net.children():

#             self.assertEqual(i.weight.grad.shape[0], i.in_features)
#             self.assertEqual(i.weight.grad.shape[1], i.out_features)
#             self.assertNotNone(i.bias.grad)       
            
#     def gradient_has_right_sparsity(self):
        
#         net = make_test_net()
#         loss_tk = tkl.TopKastLoss(loss=nn.MSELoss, net=net)
        
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