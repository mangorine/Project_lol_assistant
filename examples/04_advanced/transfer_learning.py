"""
Transfer Learning Example
This script demonstrates transfer learning concepts in PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
import os


class PretrainedModel(nn.Module):
    """Simulate a pretrained model."""
    
    def __init__(self, num_features=512):
        super(PretrainedModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, num_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(num_features, 1000)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FineTunedModel(nn.Module):
    """Model adapted for a new task using transfer learning."""
    
    def __init__(self, pretrained_model, num_classes, freeze_features=True):
        super(FineTunedModel, self).__init__()
        
        # Use pretrained features
        self.features = pretrained_model.features
        
        # Freeze feature extractor if specified
        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Replace classifier with new one
        num_features = 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def demonstrate_feature_extraction():
    """Demonstrate feature extraction approach."""
    print("=" * 50)
    print("Feature Extraction")
    print("=" * 50)
    
    # Create pretrained model
    pretrained = PretrainedModel()
    print("Pretrained model created\n")
    
    # Adapt for new task (e.g., 5 classes)
    model = FineTunedModel(pretrained, num_classes=5, freeze_features=True)
    
    # Check which parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}\n")
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}\n")


def demonstrate_fine_tuning():
    """Demonstrate fine-tuning approach."""
    print("=" * 50)
    print("Fine-Tuning")
    print("=" * 50)
    
    # Create pretrained model
    pretrained = PretrainedModel()
    
    # Adapt for new task with unfrozen features
    model = FineTunedModel(pretrained, num_classes=5, freeze_features=False)
    
    # Check trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Use different learning rates for different parts
    optimizer = optim.Adam([
        {'params': model.features.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ])
    
    print("Optimizer with different learning rates:")
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"  Group {i}: LR = {param_group['lr']}\n")


def demonstrate_layer_unfreezing():
    """Demonstrate gradual layer unfreezing."""
    print("=" * 50)
    print("Gradual Layer Unfreezing")
    print("=" * 50)
    
    # Create model
    pretrained = PretrainedModel()
    model = FineTunedModel(pretrained, num_classes=5, freeze_features=True)
    
    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Initially trainable: {count_trainable_params(model):,}")
    
    # Unfreeze last layer of features
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    
    print(f"After unfreezing last layer: {count_trainable_params(model):,}")
    
    # Unfreeze more layers
    for param in model.features.parameters():
        param.requires_grad = True
    
    print(f"After unfreezing all: {count_trainable_params(model):,}\n")


def demonstrate_layer_analysis():
    """Demonstrate analyzing and accessing model layers."""
    print("=" * 50)
    print("Layer Analysis")
    print("=" * 50)
    
    pretrained = PretrainedModel()
    model = FineTunedModel(pretrained, num_classes=5, freeze_features=True)
    
    print("Model structure:")
    for name, module in model.named_children():
        print(f"\n{name}:")
        if hasattr(module, '__iter__') and not isinstance(module, nn.Linear):
            for idx, layer in enumerate(module):
                print(f"  [{idx}] {layer}")
        else:
            print(f"  {module}")
    print()


def demonstrate_partial_loading():
    """Demonstrate loading partial model weights."""
    print("=" * 50)
    print("Partial Weight Loading")
    print("=" * 50)
    
    # Create models
    model1 = PretrainedModel()
    model2 = FineTunedModel(model1, num_classes=5)
    
    # Save model1 weights
    save_path = os.path.join(tempfile.gettempdir(), "pretrained_weights.pth")
    torch.save(model1.state_dict(), save_path)
    print(f"Saved pretrained weights to {save_path}")
    
    # Load only feature weights into model2
    pretrained_dict = torch.load(save_path)
    model2_dict = model2.state_dict()
    
    # Filter out classifier weights
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model2_dict and 'classifier' not in k}
    
    print(f"\nLoading {len(pretrained_dict)} weight tensors")
    print(f"Model has {len(model2_dict)} weight tensors total")
    
    # Update and load
    model2_dict.update(pretrained_dict)
    model2.load_state_dict(model2_dict, strict=False)
    print("Partial weights loaded successfully\n")


def main():
    """Run all transfer learning examples."""
    print("\n" + "=" * 50)
    print("PyTorch Transfer Learning Tutorial")
    print("=" * 50 + "\n")
    
    demonstrate_feature_extraction()
    demonstrate_fine_tuning()
    demonstrate_layer_unfreezing()
    demonstrate_layer_analysis()
    demonstrate_partial_loading()
    
    print("=" * 50)
    print("Tutorial Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
