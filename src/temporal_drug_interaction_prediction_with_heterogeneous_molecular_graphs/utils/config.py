"""Configuration management for drug interaction prediction."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml


@dataclass
class ModelConfig:
    """Configuration for the temporal drug interaction GNN model."""

    # Model architecture
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    temporal_attention_dim: int = 128
    metabolite_pathway_dim: int = 64

    # Heterogeneous graph settings
    node_types: List[str] = field(default_factory=lambda: ["drug", "metabolite", "target"])
    edge_types: List[str] = field(default_factory=lambda: [
        "drug_metabolite", "metabolite_drug", "drug_target", "target_drug",
        "metabolite_target", "target_metabolite", "drug_drug"
    ])

    # Temporal modeling
    time_encoding_dim: int = 32
    max_time_steps: int = 100
    temporal_decay_rate: float = 0.95


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    early_stopping_patience: int = 15
    gradient_clip_norm: float = 1.0

    # Loss function weights
    interaction_loss_weight: float = 1.0
    toxicity_loss_weight: float = 0.5
    metabolite_pathway_loss_weight: float = 0.3
    temporal_consistency_weight: float = 0.2

    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 1000

    # Regularization
    use_dropout: bool = True
    use_batch_norm: bool = True
    label_smoothing: float = 0.1


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Dataset paths
    tox21_path: Optional[str] = None
    ogbl_ppa_path: Optional[str] = None
    custom_interaction_data: Optional[str] = None

    # Data splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Preprocessing
    max_molecules_per_batch: int = 1000
    molecular_feature_dim: int = 128
    use_edge_features: bool = True
    normalize_features: bool = True

    # Data augmentation
    random_walk_length: int = 10
    num_random_walks: int = 20
    temporal_jitter_std: float = 0.1


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and logging."""

    experiment_name: str = "temporal_drug_interaction_prediction"
    run_name: Optional[str] = None
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"

    # MLflow settings
    use_mlflow: bool = True
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "drug_interaction_prediction"

    # Logging
    log_level: str = "INFO"
    log_frequency: int = 100
    save_checkpoint_frequency: int = 5

    # Evaluation
    eval_frequency: int = 1
    save_predictions: bool = True
    compute_attention_weights: bool = True


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Device and reproducibility
    device: str = "auto"
    seed: int = 42
    num_workers: int = 4

    # Target metrics (from specification)
    target_metrics: Dict[str, float] = field(default_factory=lambda: {
        "interaction_auroc": 0.88,
        "early_detection_recall_at_k": 0.75,
        "metabolite_pathway_accuracy": 0.82,
        "cross_task_transfer_improvement": 0.15
    })

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def _dataclass_to_dict(obj: Any) -> Any:
            if hasattr(obj, '__dataclass_fields__'):
                return {k: _dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [_dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _dataclass_to_dict(v) for k, v in obj.items()}
            else:
                return obj

        return _dataclass_to_dict(self)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

        logging.info(f"Configuration saved to {path}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {})),
            device=config_dict.get('device', 'auto'),
            seed=config_dict.get('seed', 42),
            num_workers=config_dict.get('num_workers', 4),
            target_metrics=config_dict.get('target_metrics', {
                "interaction_auroc": 0.88,
                "early_detection_recall_at_k": 0.75,
                "metabolite_pathway_accuracy": 0.82,
                "cross_task_transfer_improvement": 0.15
            })
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from file or use defaults.

    Args:
        config_path: Path to configuration file. If None, uses default config.

    Returns:
        Configuration object.

    Raises:
        FileNotFoundError: If specified config file doesn't exist.
    """
    if config_path is None:
        logging.info("Using default configuration")
        return Config()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logging.info(f"Loading configuration from {config_path}")
    return Config.load(config_path)


def setup_logging(config: Config) -> None:
    """Set up logging configuration.

    Args:
        config: Configuration object containing logging settings.
    """
    log_dir = Path(config.experiment.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, config.experiment.log_level.upper())

    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler
    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Suppress noisy loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)

    logging.info(f"Random seeds set to {seed}")