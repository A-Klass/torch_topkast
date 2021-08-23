#%% Usage:
    # 1. Make sure that nose is installed: pip install nose
    # 2. From the Anaconda terminal, call: nosetests test/tests.py

#%% Imports

import sys
sys.path.insert(0, "./TopKAST")

try:
    from topkast_linear import TopKastLinear
except ImportError:
    raise SystemExit("not found. check your relative path")
 
try:
    from topkast_loss import TopKastLoss
except ImportError:
    raise SystemExit("not found. check your relative path")  
import torch
import torch.nn as nn
import unittest

#%% set testing params

# tests only run for certain magnitudes: for very small inputs/nets, chances 
# are that no updates occur

input_features = 10
hidden_neurons = 128

#%% define test objects

def make_test_layer():
    return TopKastLinear(
        in_features=1000, 
        out_features=2, 
        p_forward=0.6,
        p_backward=0.2,
        bias=True)

def make_test_net():
    class NN(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_in = TopKastLinear(
                    input_features, hidden_neurons, 0.8, 0.5, True)
                self.activation = nn.ReLU()
                self.layer_out = nn.Linear(hidden_neurons, 1)
            def forward(self, x):
                return self.activation(
                    self.layer_out(self.activation(self.layer_in(x))))
            
    return NN()

#%% define optimizer

def sgd(params, lr=0.1):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()

#%% define test runs

class TestTopKastLinear(unittest.TestCase):
    
    def test_weights_are_initialized_nonempty(self):
        
        layer_tkl = make_test_layer()
        layer_tkl.reset_parameters()
        
        for param in [layer_tkl.weight, layer_tkl.bias]:
            sum_elements = param.detach().numpy().flatten().sum()
            self.assertNotAlmostEqual(sum_elements, 0)
            
    def test_weights_have_grads(self):
        
        layer_tkl = make_test_layer()
        
        self.assertTrue(layer_tkl.weight_vector.requires_grad)
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
        
        n_obs = 1000
        X = torch.rand(n_obs * input_features).reshape(
            n_obs, input_features).float()
        y = torch.rand(n_obs).float().reshape(-1, 1)
        
        for i in range(10):
            y_hat = net(X)
            l = loss_tk(y_hat, y)
            l.sum().backward()
            sgd(net.parameters())
            net.layer_in.update_active_param_set()

        self.assertFalse(
            torch.equal(net.layer_in.weight, w_before))
        
    def test_sparse_weights_are_updated(self):
              
        net = make_test_net()
        w_sparse_before = net.layer_in.weight_vector.clone()
        loss_tk = TopKastLoss(loss=nn.MSELoss, net=net)
        
        n_obs = 1000
        X = torch.rand(n_obs * input_features).reshape(
            n_obs, input_features).float()
        y = torch.rand(n_obs).float().reshape(-1, 1)
        
        for i in range(100):
            y_hat = net(X)
            l = loss_tk(y_hat, y)
            l.sum().backward()
            sgd(net.parameters())
            net.layer_in.update_active_param_set()
        
        self.assertFalse(
            torch.equal(net.layer_in.weight_vector, w_sparse_before))
        
    # def test_active_sets_are_updated(self):
        
    #     net = make_test_net()
    #     fwd_before = net.layer_in.idx_fwd
    #     bwd_before = net.layer_in.idx_bwd
    #     loss_tk = tkl.TopKastLoss(loss=nn.MSELoss, net=net)
        
    #     n_obs = 1000
    #     X = torch.rand(n_obs * input_features).reshape(
    #         n_obs, input_features).float()
    #     y = torch.rand(n_obs).float().reshape(-1, 1)
    #     y_hat = net(X)
        
    #     for i in range(100):
    #         y_hat = net(X)
    #         l = loss_tk(y_hat, y)
    #         l.sum().backward()
    #         sgd(net.parameters())
    #         net.layer_in.update_active_param_set()
        
    #     in_fwd = out_fwd = in_bwd = out_bwd = \
    #         torch.zeros_like(net.layer_in.weight)
        
    #     in_fwd[fwd_before] = 1
    #     out_fwd[net.layer_in.idx_fwd] = 1
    #     in_bwd[bwd_before] = 1
    #     out_bwd[net.layer_in.idx_bwd] = 1        
        
    #     self.assertFalse(
    #         torch.equal(in_fwd, out_fwd) and torch.equal(in_bwd, out_bwd))

#%%

class TestTopKastLoss(unittest.TestCase):
    
    def test_topkast_penalty_is_not_zero(self):

        net = make_test_net()
        
        norm_fwd = torch.linalg.norm(
            net.layer_in.weight_vector[net.layer_in.set_fwd])
        norm_justbwd = (
            torch.linalg.norm(
                net.layer_in.weight[net.layer_in.idx_justbwd]) / 
            (1 - net.layer_in.p_forward))
        
        self.assertNotEqual(norm_fwd.detach().numpy(), 0)  
        self.assertNotEqual(norm_justbwd.detach().numpy(), 0)  
    
    # def test_penalty_is_l2(self):

    #     net = make_test_net()
    #     loss_tk = TopKastLoss(loss=nn.MSELoss, net=net)
        
    #     penalty = loss_tk.compute_norm_active_set()
        
    #     standard_norm_in = (
    #         torch.linalg.norm(
    #             net.layer_in.weight_vector[net.layer_in.set_fwd]) + 
    #         (torch.linalg.norm(
    #             net.layer_in.weight[net.layer_in.idx_justbwd]) / 
    #             (1 - net.layer_in.p_forward)))
        
    #     standard_norm_out = torch.linalg.norm(net.layer_out.weight)
        
    #     standard_norm = standard_norm_in + standard_norm_out
        
    #     self.assertEqual(
    #         penalty.detach().numpy(), standard_norm.detach().numpy())
        
    def loss_is_differentiable(self):
        
        net = make_test_net()
        loss_tk = TopKastLoss(loss=nn.MSELoss, net=net)
        
        n_obs = 1000
        X = torch.rand(n_obs * input_features).reshape(
            n_obs, input_features).float()
        y = torch.rand(n_obs).float().reshape(-1, 1)
        
        y_hat = net(X)
        l = loss_tk(y_hat, y)
        l.sum().backward()
        
        self.assertIsNotNone(net.layer_in.weight_vector.grad)
        self.assertIsNotNone(net.layer_in.bias.grad)
        self.assertIsNotNone(net.layer_out.weight.grad)
        self.assertIsNotNone(net.layer_out.bias.grad)  

#%%
        
if __name__ == '__main__':
    unittest.main()

#%% 