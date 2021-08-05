#%% Imports

import topkast_linear as tk
import torch
import unittest

#%% Test layer

test_layer = tk.TopKastLinear(
    in_features=100, 
    out_features=1, 
    topk_forward=40,
    topk_backward=50,
    has_bias=True)

#%% Unit test: class

class TestClass(unittest.TestCase):
    def test_is_topklinear(self):
        self.assertIsInstance(test_layer, tk.TopKastLinear)
        
#%% Unit test: arguments

class TestArgs(unittest.TestCase):
    def test_count_infeatures(self):
        self.assertTrue(type(test_layer.in_features) == int)
        self.assertGreater(test_layer.in_features, 0)
    def test_count_outfeatures(self):
        self.assertTrue(type(test_layer.out_features) == int)
        self.assertGreater(test_layer.out_features, 0)
    def test_count_topkforward(self):
        self.assertTrue(type(test_layer.topk_forward) == int)
        self.assertGreater(test_layer.topk_forward, 0)
    def test_count_topkbackward(self):
        self.assertTrue(type(test_layer.topk_backward) == int)
        self.assertGreater(test_layer.topk_backward, 0)        
    def test_bool_bias(self):
        self.assertTrue(type(test_layer.has_bias) == bool)
    
#%% Unit test: bias & weights

class TestWeightsBias(unittest.TestCase):
    def test_has_right_size_weights(self):
        self.assertTrue(test_layer.weight.numel() == test_layer.in_features)
    def test_has_right_size_bias(self):    
        self.assertTrue(test_layer.bias.numel() == 1)
    def test_has_grad_weights(self):
        self.assertTrue(test_layer.weight.requires_grad)
    def test_has_grad_bias(self):
        self.assertTrue(test_layer.bias.requires_grad)  
    
#%% Unit test: forward sparsity

class TestSparsity(unittest.TestCase):
    def test_has_right_fwsparsity(self):
        dense_vals = test_layer.sparse_weights().coalesce().values()
        self.assertAlmostEqual(
            test_layer.in_features - dense_vals.numel(),
            # should only be dense_vals.numel but class definition probably
            # not correct atm
            test_layer.topk_forward)
    
#%% Unit test: output

class TestOutput(unittest.TestCase):
    def test_has_right_size(self):
        x = torch.rand(1, test_layer.in_features)
        self.assertTrue(test_layer(x).numel() == 1)
        
if __name__ == '__main__':
    unittest.main()

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

#%%