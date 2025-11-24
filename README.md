# PyTorch Learning & Experimentation

A comprehensive repository for learning and experimenting with PyTorch, covering everything from basic tensor operations to advanced deep learning concepts.

## 📚 Table of Contents

- [Getting Started](#getting-started)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Examples](#examples)
- [Learning Path](#learning-path)
- [Resources](#resources)

## 🚀 Getting Started

This repository is designed to help you learn PyTorch through hands-on examples. Each script is self-contained and demonstrates specific concepts with clear explanations.

### Quick Start

Run the quick start guide to see all available examples:
```bash
python examples/quickstart.py
```

### Prerequisites

- Python 3.8 or higher
- Basic understanding of Python programming
- Familiarity with machine learning concepts (helpful but not required)

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/mangorine/Deepl_fun.git
cd Deepl_fun
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python check_installation.py
```

## 📁 Repository Structure

```
Deepl_fun/
├── examples/
│   ├── 01_basics/              # Fundamental PyTorch concepts
│   │   ├── tensor_operations.py
│   │   └── autograd_example.py
│   ├── 02_neural_networks/     # Building neural networks
│   │   └── simple_nn.py
│   ├── 03_training/            # Training models
│   │   └── training_loop.py
│   └── 04_advanced/            # Advanced topics
│       ├── custom_dataset.py
│       └── transfer_learning.py
├── requirements.txt
└── README.md
```

## 🎯 Examples

### 1. Basic Concepts (01_basics/)

#### Tensor Operations
Learn the fundamentals of PyTorch tensors:
- Creating tensors in different ways
- Tensor attributes (shape, dtype, device)
- Basic operations (addition, multiplication, matrix operations)
- Indexing and slicing
- GPU operations

Run the example:
```bash
python examples/01_basics/tensor_operations.py
```

#### Automatic Differentiation (Autograd)
Understand PyTorch's automatic differentiation engine:
- Computing gradients automatically
- Gradient tracking with `requires_grad`
- Using `torch.no_grad()` context
- Detaching tensors
- Gradient accumulation
- Higher-order gradients

Run the example:
```bash
python examples/01_basics/autograd_example.py
```

### 2. Neural Networks (02_neural_networks/)

#### Building Neural Networks
Learn how to construct neural networks using `nn.Module`:
- Simple feedforward networks
- Multi-layer networks with dropout
- Convolutional neural networks (CNNs)
- Accessing and modifying layers
- Counting parameters

Run the example:
```bash
python examples/02_neural_networks/simple_nn.py
```

### 3. Training Models (03_training/)

#### Complete Training Loop
Master the training process:
- Creating datasets and dataloaders
- Implementing training loops
- Validation and evaluation
- Learning rate scheduling
- Model saving and loading
- Tracking metrics

Run the example:
```bash
python examples/03_training/training_loop.py
```

### 4. Advanced Topics (04_advanced/)

#### Custom Datasets
Learn to create custom datasets:
- Implementing the Dataset class
- Time series datasets
- Data transformations
- DataLoader configurations
- Multi-worker data loading

Run the example:
```bash
python examples/04_advanced/custom_dataset.py
```

#### Transfer Learning
Understand transfer learning techniques:
- Feature extraction
- Fine-tuning pretrained models
- Gradual layer unfreezing
- Using different learning rates for different layers
- Partial weight loading

Run the example:
```bash
python examples/04_advanced/transfer_learning.py
```

## 📖 Learning Path

Recommended order for working through the examples:

1. **Start with Basics** (Week 1)
   - Tensor operations
   - Autograd and gradients

2. **Build Neural Networks** (Week 2)
   - Simple neural networks
   - Different architectures

3. **Train Models** (Week 3)
   - Training loops
   - Optimization techniques

4. **Advanced Techniques** (Week 4+)
   - Custom datasets
   - Transfer learning
   - Experiment with your own ideas!

## 🎓 Key Concepts Covered

- **Tensors**: Multi-dimensional arrays, the fundamental data structure in PyTorch
- **Autograd**: Automatic differentiation for computing gradients
- **nn.Module**: Base class for all neural network modules
- **Optimizers**: Algorithms for updating model parameters (SGD, Adam, etc.)
- **Loss Functions**: Measuring model performance (CrossEntropyLoss, MSE, etc.)
- **DataLoader**: Efficient data loading and batching
- **Transfer Learning**: Leveraging pretrained models for new tasks

## 🔧 Running Examples

Each example script can be run independently:

```bash
python examples/<category>/<script_name>.py
```

All scripts are self-contained and will print detailed output explaining what's happening at each step.

## 📊 What You'll Learn

By working through these examples, you'll gain:

- ✅ Solid understanding of PyTorch fundamentals
- ✅ Ability to build custom neural networks
- ✅ Skills to train and evaluate models
- ✅ Knowledge of advanced techniques like transfer learning
- ✅ Practical experience with real-world ML workflows

## 🤝 Contributing

Feel free to add your own examples and experiments! Some ideas:
- More advanced architectures (Transformers, GANs, etc.)
- Computer vision projects
- Natural language processing examples
- Reinforcement learning experiments

## 📚 Additional Resources

- [Official PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [Deep Learning Book](https://www.deeplearningbook.org/)

## 📝 License

This project is open source and available for educational purposes.

## 🌟 Next Steps

After completing these examples, try:
1. Implementing a model for your own dataset
2. Participating in a Kaggle competition
3. Building a complete ML project from scratch
4. Exploring cutting-edge architectures and techniques

Happy Learning! 🚀