#%% Imports

import topkast_linear as tk
import torch
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
            self.assertIsInstance(param, torch.nn.parameter.Parameter)
            sum_elements = param.detach().numpy().flatten().sum()
            self.assertNotAlmostEqual(sum_elements, 0)
            
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
        
    def test_justbwd_is_diff_between_bwd_and_fwd(self):
        
        layer_tkl = tk.TopKastLinear(
            in_features=1000, 
            out_features=10, 
            p_forward=0.6,
            p_backward=0.4,
            bias=True)       
        
        fwd = layer_tkl.set_fwd().to_dense()
        bwd = layer_tkl.set_bwd().to_dense()
        justbwd = layer_tkl.set_justbwd().to_dense()
        is_identical = (justbwd == bwd - fwd)
        
        self.assertTrue(all(is_identical.flatten()))
        
#%%
        
if __name__ == '__main__':
    unittest.main()
    
#%%

layer = tk.TopKastLinear(
            in_features=6, 
            out_features=2, 
            p_forward=0.6,
            p_backward=0.4,
            bias=True)

#%%

layer.set_fwd(), layer.set_bwd(), layer.set_justbwd()
layer.forward(torch.rand(4, 6))

#%% Unit test: class

# class TestClass(unittest.TestCase):
#     def test_is_topklinear(self):
#         self.assertIsInstance(test_layer, tk.TopKastLinear)
        
#%% Unit test: arguments

# really necessary? this could go on forever........

# class TestArgs(unittest.TestCase):
#     def test_count_infeatures(self):
#         self.assertRaises(AssertionError, tk.TopKastLinear, 10.5, 1, 0.6, 0.4)
#         self.assertRaises(AssertionError, tk.TopKastLinear, -10, 1, 0.6, 0.4)
#     def test_count_outfeatures(self):
#         self.assertRaises(AssertionError, tk.TopKastLinear, 10, 1.5, 0.6, 0.4)
#         self.assertRaises(AssertionError, tk.TopKastLinear, 10, 0, 0.6, 0.4)
#     def test_count_topkforward(self):
#         self.assertRaises(AssertionError, tk.TopKastLinear, 10, 1.5, 1.6, 0.4)
#         self.assertRaises(AssertionError, tk.TopKastLinear, 10, 1.5, -0.6, 0.4)
#     def test_count_topkbackward(self):
#         self.assertRaises(AssertionError, tk.TopKastLinear, 10, 1.5, 0.6, 1.4)
#         self.assertRaises(AssertionError, tk.TopKastLinear, 10, 1.5, 0.6, -0.4)
#     def test_fwsparsity_geq_bwsparsity(self):
#         self.assertRaises(AssertionError, tk.TopKastLinear, 10, 1, 0.6, 0.8)
#     def test_bool_bias(self):
#         self.assertRaises(AssertionError, tk.TopKastLinear, 10, 1, 0.6, 0.4, 1)
    
#%% Unit test: bias & weights

# class TestWeightsBias(unittest.TestCase):
#     def test_has_right_size_weights(self):
#         self.assertTrue(test_layer.weight.numel() == test_layer.in_features)
#     def test_has_right_size_bias(self):    
#         self.assertTrue(test_layer.bias.numel() == 1)
#     def test_has_grad_weights(self):
#         self.assertTrue(test_layer.weight.requires_grad)
#     def test_has_grad_bias(self):
#         self.assertTrue(test_layer.bias.requires_grad)  
    
#%% Unit test: forward sparsity

# class TestSparsity(unittest.TestCase):
#     def test_has_right_forward_sparsity(self):
#         d = test_layer.weight.numel()
#         s = test_layer.sparse_weights().coalesce().values().numel()
#         self.assertAlmostEqual(s, (1 - test_layer.p_forward) * d)
#     def test_has_right_backward_sparsity(self):
#         d = test_layer.weight.numel()
#         s = test_layer.sparse_weights(forward=False).coalesce().values().numel()
#         self.assertAlmostEqual(s, (1 - test_layer.p_backward) * d)
    
#%% Unit test: output

# class TestOutput(unittest.TestCase):
#     def test_has_right_size(self):
#         x = torch.rand(1, test_layer.in_features)
#         self.assertTrue(test_layer(x).numel() == test_layer.out_features)

#%%

# layer1 = TopkLinear(5, 4, 1, 2)
# layer2 = TopkLinear(4, 1, 1, 2)
# x = torch.rand((3, 5))
# y = torch.tensor([objective(x_) for x_ in x])
# # layer.training = False
# # layer(x, sparse = False)
# y_hat = layer2(layer1(x))
# # loss = TopKastLoss(loss, net, alpha)
# # l = loss(y_hat, y)
# # %%
# l.sum().backward()
# # %%
# layer1.weight.grad, layer2.weight.grad

#%% TODO

# test that active set actually changes over iterations

# %%
# class net_a_bit_overkill(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer_in = TopKastLinear(13, 128, 20, 30)
#         self.activation_1 = nn.ReLU()
#         self.hidden = TopKastLinear(128, 128, 50, 60)
#         self.activation_2 = nn.ReLU()
#         self.layer_out = nn.Linear(128, 1)

#     def forward(self, X):
#         return self.layer_out(self.activation_2(self.hidden(self.activation_1(self.layer_in(X)))))

# net = net_a_bit_overkill()
# # %%
# loss = TopKastLoss(loss = nn.MSELoss)
# loss(torch.rand(10), torch.rand(10), net)
# %%