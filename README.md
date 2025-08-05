# Federated Learning with Neural Network Pruning

This repository contains implementations of federated learning and neural network pruning techniques for deep learning models.

## Project Structure

- `fine.py` - Neural network training and magnitude-based weight pruning on Fashion MNIST dataset
- `FL.ipynb` - Federated learning implementation for heart disease classification
- `magnitude_prining-finetuning.ipynb` - Comprehensive neural network pruning analysis

## Key Features

### Neural Network Pruning (`fine.py`)
- Deep neural network with 20+ layers (1000 neurons each)
- Fashion MNIST classification (10 classes)
- Magnitude-based weight pruning with configurable pruning percentages
- Performance evaluation across different pruning levels

### Federated Learning (`FL.ipynb`)
- Multi-client federated learning simulation
- Heart disease dataset classification
- Client data distribution across 5 simulated clients
- Federated averaging for model aggregation

### Pruning Analysis (`magnitude_prining-finetuning.ipynb`)
- Extended pruning experiments with fine-tuning
- Comprehensive accuracy analysis under different pruning conditions
- Visualization of pruning effects on model performance

## Requirements

- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Usage

1. Run `fine.py` for neural network training and pruning experiments
2. Execute `FL.ipynb` for federated learning simulations
3. Use `magnitude_prining-finetuning.ipynb` for detailed pruning analysis

## Results

The implementations demonstrate:
- Neural network compression through weight pruning
- Federated learning with distributed data
- Trade-offs between model size and accuracy
- Performance evaluation across different pruning strategies 