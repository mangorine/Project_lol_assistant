"""
Training Loop Implementation
This script demonstrates how to implement a complete training loop in PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleClassifier(nn.Module):
    """A simple classifier for demonstration."""
    
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def create_dummy_dataset(num_samples=1000, input_size=20, num_classes=3):
    """Create a dummy dataset for demonstration."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def demonstrate_training():
    """Demonstrate a complete training loop."""
    print("=" * 50)
    print("Training Loop Demonstration")
    print("=" * 50)
    
    # Hyperparameters
    input_size = 20
    num_classes = 3
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.001
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create datasets
    train_dataset = create_dummy_dataset(800, input_size, num_classes)
    val_dataset = create_dummy_dataset(200, input_size, num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, criterion, and optimizer
    model = SimpleClassifier(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Criterion: {criterion.__class__.__name__}\n")
    
    # Training loop
    print("Starting training...")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Print statistics
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 50)
    
    print("Training complete!\n")


def demonstrate_learning_rate_scheduling():
    """Demonstrate learning rate scheduling."""
    print("=" * 50)
    print("Learning Rate Scheduling")
    print("=" * 50)
    
    model = SimpleClassifier(20, 3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Different schedulers
    scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    print("Learning rate schedule (StepLR):")
    for epoch in range(15):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: LR = {current_lr:.6f}")
        scheduler_step.step()
    print()


def demonstrate_model_saving():
    """Demonstrate model saving and loading."""
    print("=" * 50)
    print("Model Saving and Loading")
    print("=" * 50)
    
    model = SimpleClassifier(20, 3)
    
    # Save model
    save_path = "/tmp/model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Load model
    new_model = SimpleClassifier(20, 3)
    new_model.load_state_dict(torch.load(save_path))
    print(f"Model loaded from {save_path}")
    
    # Verify
    x = torch.randn(1, 20)
    output1 = model(x)
    output2 = new_model(x)
    print(f"Outputs match: {torch.allclose(output1, output2)}\n")


def main():
    """Run all training examples."""
    print("\n" + "=" * 50)
    print("PyTorch Training Loop Tutorial")
    print("=" * 50 + "\n")
    
    demonstrate_training()
    demonstrate_learning_rate_scheduling()
    demonstrate_model_saving()
    
    print("=" * 50)
    print("Tutorial Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
