"""
Automatic Differentiation with Autograd
This script demonstrates PyTorch's automatic differentiation capabilities.
"""

import torch


def basic_autograd():
    """Demonstrate basic autograd functionality."""
    print("=" * 50)
    print("Basic Autograd")
    print("=" * 50)
    
    # Create tensors with gradient tracking
    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=True)
    
    # Perform operations
    z = x ** 2 + y ** 3
    print(f"x = {x.item()}, y = {y.item()}")
    print(f"z = x^2 + y^3 = {z.item()}\n")
    
    # Compute gradients
    z.backward()
    
    print(f"dz/dx = 2x = {x.grad.item()}")
    print(f"dz/dy = 3y^2 = {y.grad.item()}\n")


def gradient_computation():
    """Demonstrate gradient computation for a more complex function."""
    print("=" * 50)
    print("Gradient Computation")
    print("=" * 50)
    
    # Create a tensor with gradient tracking
    x = torch.randn(3, requires_grad=True)
    print(f"Input tensor x:\n{x}\n")
    
    # Perform operations
    y = x + 2
    z = y * y * 2
    out = z.mean()
    
    print(f"Output: {out.item()}\n")
    
    # Compute gradients
    out.backward()
    print(f"Gradients dx:\n{x.grad}\n")


def no_grad_context():
    """Demonstrate torch.no_grad() context manager."""
    print("=" * 50)
    print("No Gradient Context")
    print("=" * 50)
    
    x = torch.randn(2, 2, requires_grad=True)
    print(f"x requires grad: {x.requires_grad}")
    
    with torch.no_grad():
        y = x * 2
        print(f"y requires grad (inside no_grad): {y.requires_grad}")
    
    z = x * 2
    print(f"z requires grad (outside no_grad): {z.requires_grad}\n")


def detach_example():
    """Demonstrate tensor detachment."""
    print("=" * 50)
    print("Detach Example")
    print("=" * 50)
    
    x = torch.randn(2, 2, requires_grad=True)
    print(f"x requires grad: {x.requires_grad}")
    
    y = x.detach()
    print(f"y (detached) requires grad: {y.requires_grad}")
    print(f"y shares storage with x: {y.data_ptr() == x.data_ptr()}\n")


def gradient_accumulation():
    """Demonstrate gradient accumulation."""
    print("=" * 50)
    print("Gradient Accumulation")
    print("=" * 50)
    
    x = torch.tensor([2.0], requires_grad=True)
    
    # First backward pass
    y1 = x ** 2
    y1.backward()
    print(f"After first backward: x.grad = {x.grad}")
    
    # Second backward pass (gradients accumulate)
    y2 = x ** 3
    y2.backward()
    print(f"After second backward: x.grad = {x.grad}")
    
    # Reset gradients
    x.grad.zero_()
    print(f"After zero_(): x.grad = {x.grad}\n")


def higher_order_gradients():
    """Demonstrate computing higher-order gradients."""
    print("=" * 50)
    print("Higher Order Gradients")
    print("=" * 50)
    
    x = torch.tensor([2.0], requires_grad=True)
    
    # First derivative
    y = x ** 3
    dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
    print(f"y = x^3, x = {x.item()}")
    print(f"dy/dx = 3x^2 = {dy_dx.item()}")
    
    # Second derivative
    d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
    print(f"d²y/dx² = 6x = {d2y_dx2.item()}\n")


def main():
    """Run all autograd examples."""
    print("\n" + "=" * 50)
    print("PyTorch Autograd Tutorial")
    print("=" * 50 + "\n")
    
    basic_autograd()
    gradient_computation()
    no_grad_context()
    detach_example()
    gradient_accumulation()
    higher_order_gradients()
    
    print("=" * 50)
    print("Tutorial Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
