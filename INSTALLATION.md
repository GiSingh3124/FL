# Installation Guide

## Prerequisites

- Python 3.7 or higher
- pip package manager

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd FL-main
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install tensorflow==2.8.0
pip install numpy==1.21.0
pip install pandas==1.3.0
pip install matplotlib==3.4.0
pip install scikit-learn==1.0.0
pip install jupyter==1.0.0
```

4. Verify installation:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Data Preparation

For federated learning experiments:
- Place heart.csv dataset in the project directory
- Ensure proper data formatting for client distribution

For neural network pruning:
- Fashion MNIST dataset is automatically downloaded by TensorFlow

## Environment Variables

Set the following paths in your scripts:
- Model save directory
- Data directory paths
- Output directory for results 