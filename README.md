# Self-Pruning Neural Network

## Overview

This project implements a neural network that learns to prune its own weights using learnable sigmoid gates and L1 sparsity regularization.

## Features

* Learnable pruning mechanism
* Sparsity regularization
* CIFAR-10 training
* Analysis of sparsity vs accuracy trade-off

## How to Run

```bash
pip install torch torchvision matplotlib
python self_pruning_network.py
```

## Results

The model shows increasing sparsity with higher λ values while maintaining reasonable accuracy.

## Files

* self_pruning_network.py → main code
* report.md → explanation and results
* gate_distribution.png → visualization
