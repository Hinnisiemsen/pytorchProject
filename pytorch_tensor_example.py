#!/usr/bin/env python3
import torch

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create two tensors
tensor1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
tensor2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)

# Perform a basic operation: matrix addition
result = tensor1 + tensor2
print("Tensor 1:")
print(tensor1)
print("Tensor 2:")
print(tensor2)
print("Result of Tensor Addition:")
print(result)

# Perform matrix multiplication
product = torch.matmul(tensor1, tensor2)
print("Result of Tensor Multiplication:")
print(product)

