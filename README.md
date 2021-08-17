# Course_Deep_Learning_Pytorch_Tensorflow

## Implementation of Top-K Always Sparse Training in Pytorch

This repository is going to contain a package for an implementation of [Top-KAST: Top-K Always Sparse Training](https://arxiv.org/abs/2106.03517v1), by [Lisa Wimmer](https://github.com/lisa-wm), [Sven Lorenz](https://github.com/SvenLorenz), and Andreas Klaß. Top-KAST is a generic method to train fully sparse neural networks. We follow the original ideas from the Top-KAST's authors as closely as possible.

### Top-K Layers

Selecting the active parameter subset is done by identifying the Top-K absolute biggest weights per layer. We provide a sparse adaptation for an nn.Linear module. Other types of layers may be added later on.
### Testing

Testing is carried out by training on a synthetic example (tbd) and the Boston housing data set
### Example

tbd.
### Benchmarks

We compare runtimes and (V)RAM usage for simple vanilla networks to demonstrate the benefits of our implementation.

### Dependencies

- Python >=3.7 
- NumPy >= 1.19.5
- PyTorch >= 1.8.1
- pytorch-sparse >= 0.6.11

## Installation

tbd