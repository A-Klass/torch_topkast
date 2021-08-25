# Course Applied Deep Learning with TensorFlow and PyTorch

![img](figs/srsly_wtf.png)
## Implementation of Top-K Always Sparse Training in Pytorch

This repository is going to contain a package for an implementation of [Top-KAST: Top-K Always Sparse Training](https://arxiv.org/abs/2106.03517v1). Top-KAST is a generic method to train fully sparse neural networks. We follow the original ideas from the authors' Top-KAST paper as closely as possible.

### Top-K Layers

Selecting the active parameter subset is done by identifying the Top-K biggest weights per layer (by either Euclidean or absolute-vale norm). We provide a sparse adaptation for an nn.Linear module. Other types of layers may be added later on.

## Installation

Local installation:
1. Make sure that PyTorch is installed (see https://pytorch.org/get-started/locally/).
2. Clone this repository.
3. Set working directory to where the repository has been copied to (`cd appl_deepl`).
4. Run `pip install .` .
### Testing

Testing is carried out by training on a synthetic two-dimensional example and the Boston housing data set.

### Example

```py
from torch_topkast.topkast_linear import TopKastLinear
from torch_topkast.topkast_loss import TopKastLoss
from torch_topkast.topkast_trainer import TopKastTrainer
import torch
import torch.nn as nn
from test_data import boston_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TopKastNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_in = TopKastLinear(
            13, 128, p_forward=0.6, p_backward=0.5, device=device)
        self.activation = nn.ReLU()
        self.hidden1 = TopKastLinear(
            128, 128, p_forward=0.7, p_backward=0.5, device=device)
        self.layer_out = TopKastLinear(
            128, 1, p_forward=0.6, p_backward=0.5, device=device)

    def forward(self, X, sparse=True):
        y = self.layer_in(X, sparse=sparse)
        y = self.hidden1(self.activation(y), sparse=sparse)
        
        return self.layer_out(self.activation(y), sparse=sparse)

net = TopKastNet().to(device)
loss = TopKastLoss(loss=nn.MSELoss, net=net, alpha=0.4, device=device)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-05)
# Instantiate a TopKast trainer
trainer = TopKastTrainer(net,
                         loss,
                         num_epochs=200,
                         num_epochs_explore = 100,
                         update_every = 3,
                         batch_size = 128,
                         patience = 20,
                         optimizer = optimizer,
                         data = boston_dataset,
                         device = device)
trainer.train()
trainer.plot_loss()
print(f'The test loss is: {trainer.eval()}')
```

### Benchmarks

We compare runtimes and VRAM usage to demonstrate the benefits of our implementation.
### Dependencies

- Python >=3.7 
- NumPy >= 1.19.5
- PyTorch >= 1.8.1
- pytorch-sparse >= 0.6.11. Installing this is dependent on your local OS/PyTorch/CUDA combination, see (https://github.com/rusty1s/pytorch_sparse).
