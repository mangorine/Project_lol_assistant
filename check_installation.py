"""
Check PyTorch Installation
This script verifies that PyTorch is properly installed and checks available features.
"""

import sys


def check_pytorch():
    """Check if PyTorch is installed and display version information."""
    try:
        import torch
        print("✓ PyTorch is installed")
        print(f"  Version: {torch.__version__}")
        return True
    except ImportError:
        print("✗ PyTorch is not installed")
        print("  Install with: pip install torch")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print("✓ CUDA is available")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("ℹ CUDA is not available (CPU only)")
            print("  This is fine for learning! GPU is not required.")
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")


def check_torchvision():
    """Check if torchvision is installed."""
    try:
        import torchvision
        print("✓ torchvision is installed")
        print(f"  Version: {torchvision.__version__}")
        return True
    except ImportError:
        print("✗ torchvision is not installed")
        print("  Install with: pip install torchvision")
        return False


def check_numpy():
    """Check if NumPy is installed."""
    try:
        import numpy as np
        print("✓ NumPy is installed")
        print(f"  Version: {np.__version__}")
        return True
    except ImportError:
        print("✗ NumPy is not installed")
        print("  Install with: pip install numpy")
        return False


def check_matplotlib():
    """Check if Matplotlib is installed."""
    try:
        import matplotlib
        print("✓ Matplotlib is installed")
        print(f"  Version: {matplotlib.__version__}")
        return True
    except ImportError:
        print("ℹ Matplotlib is not installed (optional)")
        print("  Install with: pip install matplotlib")
        return False


def test_basic_operations():
    """Test basic PyTorch operations."""
    try:
        import torch
        print("\nTesting basic operations...")
        
        # Create a tensor
        x = torch.randn(3, 3)
        print("✓ Tensor creation works")
        
        # Matrix multiplication
        y = torch.matmul(x, x.T)
        print("✓ Matrix operations work")
        
        # Autograd
        x = torch.tensor([2.0], requires_grad=True)
        y = x ** 2
        y.backward()
        print("✓ Autograd works")
        
        # Neural network
        model = torch.nn.Linear(10, 5)
        output = model(torch.randn(2, 10))
        print("✓ Neural network modules work")
        
        return True
    except Exception as e:
        print(f"✗ Error in basic operations: {e}")
        return False


def main():
    """Run all installation checks."""
    print("=" * 70)
    print("  PyTorch Installation Check")
    print("=" * 70)
    
    print("\nChecking dependencies...")
    print("-" * 70)
    
    pytorch_ok = check_pytorch()
    print()
    
    if not pytorch_ok:
        print("\n" + "=" * 70)
        print("  Please install PyTorch first!")
        print("  Run: pip install -r requirements.txt")
        print("=" * 70)
        sys.exit(1)
    
    check_cuda()
    print()
    check_torchvision()
    print()
    check_numpy()
    print()
    check_matplotlib()
    
    # Test basic operations
    test_ok = test_basic_operations()
    
    print("\n" + "=" * 70)
    if test_ok:
        print("  ✓ All checks passed! You're ready to start learning PyTorch!")
        print("\n  Next steps:")
        print("  1. Run: python examples/quickstart.py")
        print("  2. Start with: python examples/01_basics/tensor_operations.py")
    else:
        print("  ✗ Some checks failed. Please check the errors above.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
