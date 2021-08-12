#%% Imports

import topkast_linear as tk
import torch
import unittest

#%% Test layer

test_layer = tk.TopKastLinear(
    in_features=10, 
    out_features=1, 
    p_forward=0.6,
    p_backward=0.4,
    bias=True)

#%% Unit test: class

class TestClass(unittest.TestCase):
    def test_is_topklinear(self):
        self.assertIsInstance(test_layer, tk.TopKastLinear)
        
#%% Unit test: arguments

# really necessary? this could go on forever........

class TestArgs(unittest.TestCase):
    def test_count_infeatures(self):
        self.assertRaises(AssertionError, tk.TopKastLinear, 100.5, 1, 0.6, 0.4)
        self.assertRaises(AssertionError, tk.TopKastLinear, -100, 1, 0.6, 0.4)
    def test_count_outfeatures(self):
        self.assertRaises(AssertionError, tk.TopKastLinear, 100, 1.5, 0.6, 0.4)
        self.assertRaises(AssertionError, tk.TopKastLinear, 100, 0, 0.6, 0.4)
    def test_count_topkforward(self):
        self.assertRaises(AssertionError, tk.TopKastLinear, 100, 1.5, 1.6, 0.4)
        self.assertRaises(AssertionError, tk.TopKastLinear, 100, 1.5, -0.6, 0.4)
    def test_count_topkbackward(self):
        self.assertRaises(AssertionError, tk.TopKastLinear, 100, 1.5, 0.6, 1.4)
        self.assertRaises(AssertionError, tk.TopKastLinear, 100, 1.5, 0.6, -0.4)
    def test_fwsparsity_geq_bwsparsity(self):
        self.assertRaises(AssertionError, tk.TopKastLinear, 100, 1, 0.6, 0.8)
    def test_bool_bias(self):
        self.assertRaises(AssertionError, tk.TopKastLinear, 100, 1, 0.6, 0.4, 1)
    
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
    def test_has_right_forward_sparsity(self):
        d = test_layer.weight.numel()
        s = test_layer.sparse_weights().coalesce().values().numel()
        self.assertAlmostEqual(s, (1 - test_layer.p_forward) * d)
    def test_has_right_backward_sparsity(self):
        d = test_layer.weight.numel()
        s = test_layer.sparse_weights(forward=False).coalesce().values().numel()
        self.assertAlmostEqual(s, (1 - test_layer.p_backward) * d)
    
#%% Unit test: output

class TestOutput(unittest.TestCase):
    def test_has_right_size(self):
        x = torch.rand(1, test_layer.in_features)
        self.assertTrue(test_layer(x).numel() == test_layer.out_features)
        
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

#%% TODO

# test that active set actually changes over iterations