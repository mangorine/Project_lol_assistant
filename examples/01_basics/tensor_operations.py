"""
Basic Tensor Operations in PyTorch
This script demonstrates fundamental tensor creation and manipulation operations.
"""

import torch
import numpy as np


def create_tensors():
    """Demonstrate different ways to create tensors."""
    print("=" * 50)
    print("Creating Tensors")
    print("=" * 50)
    
    # Create tensor from list
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(f"From list:\n{x_data}\n")
    
    # Create tensor from numpy array
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print(f"From numpy:\n{x_np}\n")
    
    # Create tensor with ones
    x_ones = torch.ones_like(x_data)
    print(f"Ones tensor:\n{x_ones}\n")
    
    # Create tensor with random values
    x_rand = torch.rand_like(x_data, dtype=torch.float)
    print(f"Random tensor:\n{x_rand}\n")
    
    # Create tensor with specific shape
    shape = (2, 3)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)
    
    print(f"Random tensor with shape {shape}:\n{rand_tensor}")
    print(f"Ones tensor with shape {shape}:\n{ones_tensor}")
    print(f"Zeros tensor with shape {shape}:\n{zeros_tensor}\n")


def tensor_attributes():
    """Demonstrate tensor attributes."""
    print("=" * 50)
    print("Tensor Attributes")
    print("=" * 50)
    
    tensor = torch.rand(3, 4)
    print(f"Tensor:\n{tensor}\n")
    print(f"Shape: {tensor.shape}")
    print(f"Datatype: {tensor.dtype}")
    print(f"Device: {tensor.device}\n")


def tensor_operations():
    """Demonstrate basic tensor operations."""
    print("=" * 50)
    print("Tensor Operations")
    print("=" * 50)
    
    # Matrix multiplication
    tensor1 = torch.rand(2, 3)
    tensor2 = torch.rand(3, 2)
    result = torch.matmul(tensor1, tensor2)
    print(f"Matrix multiplication:\n{tensor1}\n@\n{tensor2}\n=\n{result}\n")
    
    # Element-wise operations
    tensor = torch.rand(2, 3)
    print(f"Original tensor:\n{tensor}")
    print(f"Multiplied by 2:\n{tensor * 2}")
    print(f"Added 10:\n{tensor + 10}\n")
    
    # Aggregation operations
    print(f"Sum: {tensor.sum()}")
    print(f"Mean: {tensor.mean()}")
    print(f"Max: {tensor.max()}")
    print(f"Min: {tensor.min()}\n")


def tensor_indexing():
    """Demonstrate tensor indexing and slicing."""
    print("=" * 50)
    print("Tensor Indexing and Slicing")
    print("=" * 50)
    
    tensor = torch.arange(12).reshape(3, 4)
    print(f"Tensor:\n{tensor}\n")
    
    print(f"First row: {tensor[0]}")
    print(f"First column: {tensor[:, 0]}")
    print(f"Last column: {tensor[:, -1]}")
    print(f"Middle 2x2: {tensor[1:3, 1:3]}\n")


def gpu_operations():
    """Demonstrate GPU operations if available."""
    print("=" * 50)
    print("GPU Operations")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print("CUDA is available!")
        device = torch.device("cuda")
        tensor = torch.rand(3, 3)
        tensor_gpu = tensor.to(device)
        print(f"Tensor on GPU: {tensor_gpu.device}")
        
        # Operations on GPU
        result = tensor_gpu @ tensor_gpu.T
        print(f"Result device: {result.device}\n")
    else:
        print("CUDA is not available. Using CPU.\n")


def main():
    """Run all tensor operation examples."""
    print("\n" + "=" * 50)
    print("PyTorch Tensor Operations Tutorial")
    print("=" * 50 + "\n")
    
    create_tensors()
    tensor_attributes()
    tensor_operations()
    tensor_indexing()
    gpu_operations()
    
    print("=" * 50)
    print("Tutorial Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
