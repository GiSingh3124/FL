# Experiments Guide

## Neural Network Pruning Experiments

### Architecture Details
- Input: 784 features (28x28 pixel images)
- Hidden layers: 20 dense layers with 1000 neurons each
- Output: 10 classes (Fashion MNIST categories)
- Activation: ReLU for hidden layers, Softmax for output
- Optimizer: Adam with learning rate 0.0001

### Pruning Methodology
1. **Weight Extraction**: Extract all weights from trained model
2. **Magnitude Sorting**: Sort weights by absolute magnitude
3. **Pruning Application**: Remove smallest magnitude weights
4. **Performance Evaluation**: Test accuracy after pruning

### Pruning Levels Tested
- 0% (baseline)
- 10%, 20%, 30%, 40%, 50%
- 60%, 70%, 80%, 90%

## Federated Learning Experiments

### Client Setup
- 5 simulated clients
- Data distribution: Equal split of training data
- Local training: 3 epochs per communication round
- Communication rounds: 10

### Model Architecture
- Input layer: 13 features (heart disease dataset)
- Hidden layers: 3 dense layers with dropout
- Output: Binary classification (sigmoid activation)

### Federated Averaging
- Weight scaling based on client data size
- Global model aggregation after local training
- Performance evaluation on test data

## Performance Metrics

### Classification Accuracy
- Training accuracy progression
- Validation accuracy monitoring
- Test set final evaluation

### Pruning Analysis
- Accuracy vs pruning percentage
- Model compression ratios
- Computational efficiency gains

### Federated Learning Metrics
- Global model accuracy
- Client-specific performance
- Communication efficiency 