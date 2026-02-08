"""Test configuration and shared fixtures for the drug interaction prediction project."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.utils.config import Config, DataConfig, ModelConfig, TrainingConfig, ExperimentConfig
from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.data.preprocessing import MolecularGraphPreprocessor, TemporalGraphConstructor
from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.models.model import TemporalDrugInteractionGNN
from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.evaluation.metrics import DrugInteractionMetrics


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1CCN(CC1)C(C2=CC=CC=C2)C3=CC=CC=C3",  # Diphenhydramine
        "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",  # Salbutamol
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "C1=CC(=C(C=C1C2=COC3=CC(=CC(=C3C2=O)O)O)O)O",  # Quercetin
    ]


@pytest.fixture
def sample_drug_pairs():
    """Sample drug pairs for testing."""
    return [
        ("aspirin", "warfarin"),
        ("caffeine", "diphenhydramine"),
        ("salbutamol", "quercetin"),
        ("aspirin", "caffeine"),
        ("warfarin", "salbutamol")
    ]


@pytest.fixture
def sample_interaction_labels():
    """Sample interaction labels."""
    return [1, 0, 1, 0, 1]  # Binary interaction labels


@pytest.fixture
def test_config():
    """Test configuration with reduced parameters."""
    config = Config()

    # Reduce model size for testing
    config.model.hidden_dim = 64
    config.model.num_layers = 2
    config.model.num_heads = 4
    config.model.temporal_attention_dim = 32
    config.model.metabolite_pathway_dim = 16
    config.model.max_time_steps = 10

    # Reduce training parameters
    config.training.batch_size = 4
    config.training.num_epochs = 2
    config.training.early_stopping_patience = 1
    config.training.learning_rate = 1e-3

    # Test data parameters
    config.data.max_molecules_per_batch = 10
    config.data.molecular_feature_dim = 64
    config.data.train_ratio = 0.6
    config.data.val_ratio = 0.2
    config.data.test_ratio = 0.2

    # Disable MLflow for testing
    config.experiment.use_mlflow = False
    config.experiment.log_level = "DEBUG"

    # Use CPU for testing
    config.device = "cpu"
    config.seed = 42

    return config


@pytest.fixture
def molecular_preprocessor():
    """Molecular graph preprocessor for testing."""
    return MolecularGraphPreprocessor(
        max_atoms=50,
        use_edge_features=True,
        use_3d_coords=False
    )


@pytest.fixture
def temporal_graph_constructor():
    """Temporal graph constructor for testing."""
    return TemporalGraphConstructor()


@pytest.fixture
def sample_drug_features():
    """Sample drug molecular features."""
    feature_dim = 265  # Molecular descriptor dimension
    return {
        "aspirin": torch.randn(feature_dim),
        "warfarin": torch.randn(feature_dim),
        "caffeine": torch.randn(feature_dim),
        "diphenhydramine": torch.randn(feature_dim),
        "salbutamul": torch.randn(feature_dim),
        "quercetin": torch.randn(feature_dim)
    }


@pytest.fixture
def sample_hetero_graph(temporal_graph_constructor, sample_drug_pairs, sample_drug_features, sample_interaction_labels):
    """Sample heterogeneous graph for testing."""
    return temporal_graph_constructor.construct_heterogeneous_graph(
        sample_drug_pairs,
        sample_drug_features,
        sample_interaction_labels
    )


@pytest.fixture
def test_model(test_config):
    """Test model instance."""
    node_type_dims = {
        'drug': test_config.data.molecular_feature_dim,
        'metabolite': 64,
        'target': 64
    }

    edge_types = [
        ('drug', 'metabolizes_to', 'metabolite'),
        ('metabolite', 'metabolized_from', 'drug'),
        ('drug', 'targets', 'target'),
        ('target', 'targeted_by', 'drug'),
        ('metabolite', 'affects', 'target'),
        ('target', 'affected_by', 'metabolite'),
        ('drug', 'interacts', 'drug')
    ]

    return TemporalDrugInteractionGNN(
        node_type_dims=node_type_dims,
        edge_types=edge_types,
        hidden_dim=test_config.model.hidden_dim,
        num_layers=test_config.model.num_layers,
        num_heads=test_config.model.num_heads,
        dropout=test_config.model.dropout,
        temporal_attention_dim=test_config.model.temporal_attention_dim,
        metabolite_pathway_dim=test_config.model.metabolite_pathway_dim,
        max_time_steps=test_config.model.max_time_steps,
    )


@pytest.fixture
def metrics_calculator():
    """Metrics calculator for testing."""
    target_metrics = {
        "interaction_auroc": 0.88,
        "early_detection_recall_at_k": 0.75,
        "metabolite_pathway_accuracy": 0.82,
        "cross_task_transfer_improvement": 0.15
    }
    return DrugInteractionMetrics(target_metrics)


@pytest.fixture
def sample_predictions():
    """Sample predictions for testing metrics."""
    return torch.tensor([0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6, 0.15, 0.85])


@pytest.fixture
def sample_targets():
    """Sample targets for testing metrics."""
    return torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])


@pytest.fixture
def temp_dir():
    """Temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible testing."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Set deterministic behavior for testing
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.fixture
def sample_batch():
    """Sample batch for training/testing."""
    batch_size = 4
    feature_dim = 64

    return {
        'drug1_features': torch.randn(batch_size, feature_dim),
        'drug2_features': torch.randn(batch_size, feature_dim),
        'label': torch.randint(0, 2, (batch_size,)).float(),
        'drug1_id': [f'drug1_{i}' for i in range(batch_size)],
        'drug2_id': [f'drug2_{i}' for i in range(batch_size)]
    }


@pytest.fixture
def sample_temporal_features():
    """Sample temporal features for testing."""
    num_drugs = 6
    num_time_points = 25

    temporal_features = {}
    drug_names = ["aspirin", "warfarin", "caffeine", "diphenhydramine", "salbutamol", "quercetin"]

    for drug in drug_names:
        # Simulate concentration over time with exponential decay
        time_points = np.linspace(0, 24, num_time_points)
        half_life = np.random.uniform(2, 8)  # Random half-life
        concentrations = np.exp(-0.693 * time_points / half_life)
        temporal_features[drug] = torch.tensor(concentrations, dtype=torch.float)

    return temporal_features


# Test utilities
def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple, name: str = "tensor"):
    """Assert that tensor has expected shape."""
    assert tensor.shape == expected_shape, f"{name} shape {tensor.shape} != expected {expected_shape}"


def assert_tensor_finite(tensor: torch.Tensor, name: str = "tensor"):
    """Assert that tensor contains only finite values."""
    assert torch.all(torch.isfinite(tensor)), f"{name} contains non-finite values"


def assert_tensor_range(tensor: torch.Tensor, min_val: float, max_val: float, name: str = "tensor"):
    """Assert that tensor values are within expected range."""
    assert torch.all(tensor >= min_val), f"{name} contains values < {min_val}"
    assert torch.all(tensor <= max_val), f"{name} contains values > {max_val}"


# Add test utilities to pytest namespace
pytest.assert_tensor_shape = assert_tensor_shape
pytest.assert_tensor_finite = assert_tensor_finite
pytest.assert_tensor_range = assert_tensor_range