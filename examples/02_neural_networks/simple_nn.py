"""
Simple Neural Network Examples
This script demonstrates how to build basic neural networks in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """A simple feedforward neural network."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MultiLayerNet(nn.Module):
    """A multi-layer neural network with dropout."""
    
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.5):
        super(MultiLayerNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ConvNet(nn.Module):
    """A simple convolutional neural network."""
    
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Input: (batch_size, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # (batch_size, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (batch_size, 64, 7, 7)
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def demonstrate_simple_net():
    """Demonstrate SimpleNet architecture and forward pass."""
    print("=" * 50)
    print("Simple Neural Network")
    print("=" * 50)
    
    model = SimpleNet(input_size=10, hidden_size=20, output_size=2)
    print(f"Model architecture:\n{model}\n")
    
    # Create dummy input
    x = torch.randn(5, 10)  # Batch of 5 samples, 10 features each
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}\n")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}\n")


def demonstrate_multi_layer_net():
    """Demonstrate MultiLayerNet with multiple hidden layers."""
    print("=" * 50)
    print("Multi-Layer Neural Network")
    print("=" * 50)
    
    model = MultiLayerNet(
        input_size=784,
        hidden_sizes=[256, 128, 64],
        output_size=10,
        dropout_prob=0.3
    )
    print(f"Model architecture:\n{model}\n")
    
    # Create dummy input (flattened 28x28 image)
    x = torch.randn(32, 784)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}\n")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}\n")


def demonstrate_conv_net():
    """Demonstrate ConvNet for image classification."""
    print("=" * 50)
    print("Convolutional Neural Network")
    print("=" * 50)
    
    model = ConvNet(num_classes=10)
    print(f"Model architecture:\n{model}\n")
    
    # Create dummy input (batch of MNIST-like images)
    x = torch.randn(16, 1, 28, 28)  # 16 images, 1 channel, 28x28 pixels
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}\n")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}\n")


def demonstrate_layer_access():
    """Demonstrate accessing and modifying layers."""
    print("=" * 50)
    print("Layer Access and Modification")
    print("=" * 50)
    
    model = SimpleNet(10, 20, 2)
    
    # Access layers
    print("Layers:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    print()
    
    # Freeze first layer
    for param in model.fc1.parameters():
        param.requires_grad = False
    
    print("After freezing fc1:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    print()


def main():
    """Run all neural network examples."""
    print("\n" + "=" * 50)
    print("PyTorch Neural Networks Tutorial")
    print("=" * 50 + "\n")
    
    demonstrate_simple_net()
    demonstrate_multi_layer_net()
    demonstrate_conv_net()
    demonstrate_layer_access()
    
    print("=" * 50)
    print("Tutorial Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
