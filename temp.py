import torch
import numpy as np

a = torch.tensor([[0, 0, 1, 0], [1, 2, 0, 0], [0, 0, 0, 0]], dtype = torch.float64)
sp = a._to_sparse_csr()
vec = torch.randn(4, 1, dtype=torch.float64)
sp.matmul(vec)