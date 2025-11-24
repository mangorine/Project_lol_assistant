"""
Custom Dataset and DataLoader
This script demonstrates how to create custom datasets in PyTorch.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CustomDataset(Dataset):
    """
    A custom dataset class.
    
    This demonstrates the three required methods:
    - __init__: Initialize the dataset
    - __len__: Return the size of the dataset
    - __getitem__: Return a single sample from the dataset
    """
    
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: Input data (numpy array or tensor)
            labels: Labels (numpy array or tensor)
            transform: Optional transform to be applied to samples
        """
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Return a single sample from the dataset.
        
        Args:
            idx: Index of the sample to return
            
        Returns:
            Tuple of (sample, label)
        """
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


class TimeSeriesDataset(Dataset):
    """A dataset for time series data with sliding windows."""
    
    def __init__(self, data, window_size, stride=1):
        """
        Args:
            data: Time series data (numpy array or tensor)
            window_size: Size of the sliding window
            stride: Step size for sliding the window
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride
        
        # Calculate number of windows
        self.num_windows = (len(data) - window_size) // stride + 1
    
    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        # Return window and next value (for prediction)
        window = self.data[start_idx:end_idx]
        target = self.data[end_idx] if end_idx < len(self.data) else self.data[-1]
        
        return torch.FloatTensor(window), torch.FloatTensor([target])


def demonstrate_custom_dataset():
    """Demonstrate custom dataset usage."""
    print("=" * 50)
    print("Custom Dataset")
    print("=" * 50)
    
    # Create dummy data
    data = np.random.randn(100, 10)
    labels = np.random.randint(0, 3, 100)
    
    # Create dataset
    dataset = CustomDataset(data, labels)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a single sample
    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Label: {label}\n")
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Iterate through batches
    for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Data shape: {batch_data.shape}")
        print(f"  Labels shape: {batch_labels.shape}")
        if batch_idx == 2:  # Show only first 3 batches
            break
    print()


def demonstrate_time_series_dataset():
    """Demonstrate time series dataset."""
    print("=" * 50)
    print("Time Series Dataset")
    print("=" * 50)
    
    # Create dummy time series data
    time_series = np.sin(np.linspace(0, 4 * np.pi, 100))
    
    # Create dataset with sliding windows
    dataset = TimeSeriesDataset(time_series, window_size=10, stride=5)
    
    print(f"Time series length: {len(time_series)}")
    print(f"Number of windows: {len(dataset)}")
    
    # Get a sample
    window, target = dataset[0]
    print(f"Window shape: {window.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Window values: {window[:5]}...")
    print(f"Target value: {target.item():.4f}\n")
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    batch_data, batch_targets = next(iter(dataloader))
    print(f"Batch data shape: {batch_data.shape}")
    print(f"Batch targets shape: {batch_targets.shape}\n")


def demonstrate_transforms():
    """Demonstrate data transformations."""
    print("=" * 50)
    print("Data Transformations")
    print("=" * 50)
    
    # Define a simple transform
    def normalize(x):
        return (x - x.mean()) / (x.std() + 1e-8)
    
    # Create dataset with transform
    data = np.random.randn(50, 5)
    labels = np.random.randint(0, 2, 50)
    
    dataset_no_transform = CustomDataset(data, labels, transform=None)
    dataset_with_transform = CustomDataset(data, labels, transform=normalize)
    
    # Compare samples
    sample1, _ = dataset_no_transform[0]
    sample2, _ = dataset_with_transform[0]
    
    print("Without transform:")
    print(f"  Mean: {sample1.mean():.4f}, Std: {sample1.std():.4f}")
    print("\nWith normalization transform:")
    print(f"  Mean: {sample2.mean():.4f}, Std: {sample2.std():.4f}\n")


def demonstrate_dataloader_features():
    """Demonstrate various DataLoader features."""
    print("=" * 50)
    print("DataLoader Features")
    print("=" * 50)
    
    # Create dataset
    data = np.random.randn(100, 10)
    labels = np.random.randint(0, 3, 100)
    dataset = CustomDataset(data, labels)
    
    # Different DataLoader configurations
    print("1. Basic DataLoader:")
    loader1 = DataLoader(dataset, batch_size=16)
    print(f"   Batch size: {loader1.batch_size}, Shuffle: {False}")
    
    print("\n2. Shuffled DataLoader:")
    loader2 = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"   Batch size: {loader2.batch_size}, Shuffle: {True}")
    
    print("\n3. Multi-worker DataLoader:")
    loader3 = DataLoader(dataset, batch_size=16, num_workers=2)
    print(f"   Batch size: {loader3.batch_size}, Workers: {2}")
    
    print("\n4. DataLoader with drop_last:")
    loader4 = DataLoader(dataset, batch_size=16, drop_last=True)
    print(f"   Batch size: {loader4.batch_size}, Drop last: {True}")
    print(f"   Number of batches: {len(loader4)}\n")


def main():
    """Run all custom dataset examples."""
    print("\n" + "=" * 50)
    print("PyTorch Custom Dataset Tutorial")
    print("=" * 50 + "\n")
    
    demonstrate_custom_dataset()
    demonstrate_time_series_dataset()
    demonstrate_transforms()
    demonstrate_dataloader_features()
    
    print("=" * 50)
    print("Tutorial Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
