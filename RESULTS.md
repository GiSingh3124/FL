# Results Summary

## Neural Network Pruning Results

### Baseline Performance
- Model accuracy: 88.25% on Fashion MNIST test set
- Architecture: 20+ dense layers with 1000 neurons each
- Training: 30 epochs with early stopping

### Pruning Impact Analysis

| Pruning % | Accuracy | Performance Change |
|-----------|----------|-------------------|
| 0%        | 88.25%   | Baseline          |
| 10%       | 88.17%   | -0.08%            |
| 20%       | 88.22%   | -0.03%            |
| 30%       | 88.08%   | -0.17%            |
| 40%       | 88.38%   | +0.13%            |
| 50%       | 88.26%   | +0.01%            |
| 60%       | 88.40%   | +0.15%            |
| 70%       | 88.06%   | -0.19%            |
| 80%       | 66.34%   | -21.91%           |
| 90%       | 26.27%   | -61.98%           |

### Key Findings
- Model maintains performance up to 60% pruning
- Significant degradation beyond 70% pruning
- Optimal pruning range: 40-60%

## Federated Learning Results

### Training Progression
- Communication rounds: 10
- Clients: 5 simulated clients
- Local epochs: 3 per round

### Performance Metrics
- Global model convergence observed
- Client-specific accuracy variations
- Balanced accuracy improvements over rounds

### Model Compression Benefits
- Reduced model size through pruning
- Maintained accuracy with 60% weight removal
- Computational efficiency gains
- Memory usage optimization

## Technical Insights

### Pruning Effectiveness
- Magnitude-based pruning preserves important weights
- Gradual performance degradation pattern
- Critical threshold at 70% pruning level

### Federated Learning Stability
- Consistent global model updates
- Client diversity handling
- Scalable architecture design 