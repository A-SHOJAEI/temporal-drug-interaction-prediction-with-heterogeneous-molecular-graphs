# Temporal Drug Interaction Prediction with Heterogeneous Molecular Graphs

A production-ready machine learning system that predicts adverse drug-drug interactions by modeling molecular structures as heterogeneous graphs with temporal attention mechanisms that capture how interaction risks evolve based on metabolite formation over time.

## Overview

This system jointly learns from Tox21 toxicity assays and constructs a dynamic interaction graph where edges represent potential metabolic pathways, enabling pharmacovigilance teams to identify dangerous drug combinations before clinical trials.

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd temporal-drug-interaction-prediction-with-heterogeneous-molecular-graphs

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

```python
from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs import (
    TemporalDrugInteractionGNN, DrugInteractionDataLoader, TemporalDrugTrainer
)

# Load configuration
config = load_config("configs/default.yaml")

# Initialize data loader and load dataset
data_loader = DrugInteractionDataLoader(config.data)
train_loader, val_loader, test_loader, metadata = data_loader.load_complete_dataset()

# Initialize and train model
trainer = TemporalDrugTrainer(config)
history = trainer.train(train_loader, val_loader)

# Evaluate model
test_metrics = trainer.evaluate(test_loader)
```

### Command Line Interface

```bash
# Train model
python scripts/train.py --config configs/default.yaml --num-epochs 50

# Evaluate model
python scripts/evaluate.py checkpoints/best_model.pt --plot-curves --attention-analysis

# Quick training with dry run
python scripts/train.py --dry-run --debug
```

## Architecture

### Core Components

- **Heterogeneous Graph Neural Networks**: Models drugs, metabolites, and targets as different node types
- **Temporal Attention Mechanism**: Captures evolving interaction risks over time
- **Metabolite Pathway Predictor**: Predicts metabolic pathways and their temporal dynamics
- **Multi-task Learning**: Joint optimization for interaction prediction, toxicity assessment, and pathway modeling

### Model Pipeline

1. **Molecular Preprocessing**: SMILES → Graph representations with RDKit
2. **Heterogeneous Graph Construction**: Multi-relational graph with drugs, metabolites, targets
3. **Temporal Modeling**: Time-dependent attention for dynamic interactions
4. **Multi-task Prediction**: Simultaneous prediction of interactions, pathways, and toxicity

## Key Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Interaction AUROC | 0.88 | TBD* |
| Early Detection Recall@K | 0.75 | TBD* |
| Metabolite Pathway Accuracy | 0.82 | TBD* |
| Cross-task Transfer Improvement | 0.15 | TBD* |

*Results depend on training completion and dataset quality.

## Technical Features

### Data Processing
- **Molecular Graph Preprocessing**: Automated SMILES to graph conversion
- **Heterogeneous Graph Construction**: Multi-relational temporal graphs
- **Feature Engineering**: 265-dimensional molecular descriptors
- **Temporal Dynamics**: Pharmacokinetic modeling with exponential decay

### Model Architecture
- **Graph Attention Networks**: Multi-head attention for heterogeneous node types
- **Temporal Attention**: Positional encoding with learnable decay parameters
- **Hierarchical Modeling**: Node → Graph → Interaction prediction pipeline
- **Regularization**: Dropout, batch normalization, gradient clipping

### Training Infrastructure
- **MLflow Integration**: Experiment tracking and model versioning
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Checkpointing**: Automatic model saving and resumption
- **Multi-GPU Support**: Distributed training capabilities

### Evaluation Framework
- **Comprehensive Metrics**: AUROC, AUPRC, recall@k, temporal consistency
- **Cross-validation**: Stratified splitting for robust evaluation
- **Attention Analysis**: Interpretability through attention weights
- **Statistical Significance**: Bootstrap confidence intervals

## Project Structure

```
temporal-drug-interaction-prediction-with-heterogeneous-molecular-graphs/
├── src/temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Neural network architectures
│   ├── training/       # Training pipeline and utilities
│   ├── evaluation/     # Metrics and evaluation tools
│   └── utils/          # Configuration and utilities
├── tests/              # Comprehensive test suite
├── scripts/            # Training and evaluation scripts
├── configs/            # Configuration files
├── notebooks/          # Jupyter notebooks for exploration
└── requirements.txt    # Dependencies
```

## Configuration

The system uses YAML configuration files for reproducible experiments:

```yaml
model:
  hidden_dim: 256
  num_layers: 4
  num_heads: 8
  temporal_attention_dim: 128

training:
  batch_size: 32
  learning_rate: 1e-3
  num_epochs: 100
  early_stopping_patience: 15

data:
  molecular_feature_dim: 128
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

## Advanced Usage

### Custom Drug Pair Evaluation

```python
# Evaluate specific drug combinations
drug_pairs = [("aspirin", "warfarin"), ("caffeine", "diphenhydramine")]
predictions = model.predict_interactions(drug_pairs, drug_features)
```

### Attention Weight Analysis

```python
# Extract attention weights for interpretability
outputs = model(graph_data, return_attention=True)
attention_weights = outputs['attention_weights']
```

### Temporal Dynamics Modeling

```python
# Model drug concentration over time
time_points = torch.linspace(0, 24, 25)  # 24 hours
temporal_graph = graph_constructor.add_temporal_dynamics(graph, time_points)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs

# Run specific test categories
pytest tests/ -m "not slow"  # Skip slow tests
pytest tests/ -m "gpu"       # GPU-only tests
```

## Dependencies

### Core Requirements
- PyTorch ≥ 2.0.0
- PyTorch Geometric ≥ 2.4.0
- RDKit ≥ 2022.9.1
- NetworkX ≥ 2.8
- NumPy ≥ 1.21.0
- Pandas ≥ 1.3.0
- Scikit-learn ≥ 1.1.0

### Optional Dependencies
- MLflow ≥ 2.0.0 (experiment tracking)
- DeepChem ≥ 2.7.0 (molecular datasets)
- OGB ≥ 1.3.6 (graph benchmarks)
- Jupyter ≥ 1.0.0 (notebooks)

## Performance Optimization

### Memory Optimization
- **Gradient Checkpointing**: Reduces memory usage for large models
- **Mixed Precision**: FP16 training for faster computation
- **Batch Size Scaling**: Automatic batch size optimization

### Computational Efficiency
- **Graph Sampling**: Subgraph sampling for large graphs
- **Distributed Training**: Multi-GPU and multi-node support
- **Model Compression**: Quantization and pruning support

## Production Deployment

### Model Serving
```python
# Load trained model
model = TemporalDrugInteractionGNN.load_from_checkpoint("checkpoints/best_model.pt")

# Predict drug interactions
risk_score = model.predict_interaction_risk("drug1_smiles", "drug2_smiles")
```

### API Integration
```python
# REST API endpoint
@app.post("/predict-interaction")
def predict_interaction(drug1: str, drug2: str):
    return {"interaction_probability": model.predict(drug1, drug2)}
```

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.