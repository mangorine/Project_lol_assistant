"""
PyTorch Quick Start Guide
This script provides a quick overview of all available examples.
"""

import os
import sys


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(number, title, description, script_path):
    """Print an example section."""
    print(f"\n{number}. {title}")
    print(f"   {description}")
    print(f"   Run: python {script_path}")


def main():
    """Display quick start guide."""
    print_header("PyTorch Learning Repository - Quick Start Guide")
    
    print("\n📚 Welcome to the PyTorch Learning Repository!")
    print("\nThis repository contains hands-on examples to help you learn PyTorch")
    print("from basics to advanced concepts.\n")
    
    print("🎯 GETTING STARTED:")
    print("   1. Make sure you have installed dependencies: pip install -r requirements.txt")
    print("   2. Choose an example below and run it")
    print("   3. Read the code to understand what's happening")
    print("   4. Modify and experiment!")
    
    print_header("Available Examples")
    
    print("\n┌─ BASICS ─────────────────────────────────────────────────────────┐")
    print_section(
        "1",
        "Tensor Operations",
        "Learn PyTorch tensors, operations, indexing, and GPU usage",
        "examples/01_basics/tensor_operations.py"
    )
    print_section(
        "2",
        "Automatic Differentiation",
        "Understand autograd, gradients, and backpropagation",
        "examples/01_basics/autograd_example.py"
    )
    print("└──────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ NEURAL NETWORKS ────────────────────────────────────────────────┐")
    print_section(
        "3",
        "Building Neural Networks",
        "Create feedforward, multi-layer, and convolutional networks",
        "examples/02_neural_networks/simple_nn.py"
    )
    print("└──────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ TRAINING ───────────────────────────────────────────────────────┐")
    print_section(
        "4",
        "Training Loop",
        "Implement complete training with validation and model saving",
        "examples/03_training/training_loop.py"
    )
    print("└──────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ ADVANCED ───────────────────────────────────────────────────────┐")
    print_section(
        "5",
        "Custom Datasets",
        "Create custom datasets and efficient data loaders",
        "examples/04_advanced/custom_dataset.py"
    )
    print_section(
        "6",
        "Transfer Learning",
        "Use pretrained models and fine-tuning techniques",
        "examples/04_advanced/transfer_learning.py"
    )
    print("└──────────────────────────────────────────────────────────────────┘")
    
    print_header("Recommended Learning Path")
    print("\n  Week 1: Start with examples 1-2 (Basics)")
    print("  Week 2: Move to example 3 (Neural Networks)")
    print("  Week 3: Learn example 4 (Training)")
    print("  Week 4+: Explore examples 5-6 (Advanced) and create your own projects")
    
    print_header("Tips for Learning")
    print("\n  • Run each example and observe the output")
    print("  • Read through the code carefully")
    print("  • Modify parameters and see what changes")
    print("  • Try to break things - that's how you learn!")
    print("  • Experiment with your own ideas")
    
    print_header("Need Help?")
    print("\n  📖 Check the README.md for detailed documentation")
    print("  🌐 Visit PyTorch documentation: https://pytorch.org/docs/")
    print("  💬 Join PyTorch forums: https://discuss.pytorch.org/")
    
    print("\n" + "=" * 70)
    print("  Happy Learning! 🚀")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
