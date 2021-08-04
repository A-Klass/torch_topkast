#%% Imports

# import os
# os.chdir('/home/lisa-wm/Documents/2_uni/3_sem/applied_dl/appl_deepl')
import topk_linear as tk

#%% Unit test: 

layer_1 = tk.TopKastLinear(
    in_features=5, 
    out_features=1, 
    topk_forward=1,
    topk_backward=2)

print(layer_1.bias)
print(layer_1.weight)
print(layer_1.sparse_weights())

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