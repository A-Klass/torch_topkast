#%% Imports

import topk_linear as tk
import torch
import unittest

#%% Test layer

test_layer = tk.TopkLinear(
    in_features=100, 
    out_features=1, 
    topk_forward=40,
    topk_backward=50)

#%% Unit test: class

class TestClass(unittest.TestCase):
    def test_is_topklinear(self):
        self.assertIsInstance(test_layer, tk.TopkLinear)
    
#%% Unit test: bias & weights

class TestWeightsBias(unittest.TestCase):
    def test_has_right_size(self):
        self.assertTrue(test_layer.weight.numel() == test_layer.in_features)
        self.assertTrue(test_layer.bias.numel() == 1)
    def test_has_grad(self):
        self.assertTrue(test_layer.weight.requires_grad)
        self.assertTrue(test_layer.bias.requires_grad)  
    
#%% Unit test: forward sparsity

class TestSparsity(unittest.TestCase):
    def test_has_right_fwsparsity(self):
        dense_vals = test_layer.sparse_weights().coalesce().values()
        self.assertAlmostEqual(
            test_layer.in_features - dense_vals.numel(), 
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