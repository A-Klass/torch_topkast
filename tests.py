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