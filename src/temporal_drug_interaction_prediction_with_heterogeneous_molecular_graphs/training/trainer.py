"""Training pipeline for temporal drug interaction prediction with MLflow tracking.

This module provides a comprehensive training pipeline for temporal drug interaction
prediction using heterogeneous graph neural networks. It includes support for:
- Multi-task loss computation with interaction, metabolite, and temporal objectives
- MLflow experiment tracking for reproducible research
- Early stopping with model weight restoration
- Comprehensive evaluation metrics for drug interaction prediction
- Robust error handling and logging for production deployment
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Experiment tracking disabled.")

from ..models.model import TemporalDrugInteractionGNN
from ..evaluation.metrics import DrugInteractionMetrics
from ..utils.config import Config, set_seeds
from .exceptions import (
    TrainingError,
    ModelInitializationError,
    BatchProcessingError,
    CheckpointError,
    MLflowTrackingError,
    ValidationError,
    EarlyStoppingError
)

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback to prevent overfitting.

    This class implements an early stopping mechanism that monitors a specified
    metric and stops training when the metric stops improving for a given number
    of epochs (patience). It can optionally restore the best model weights when
    early stopping is triggered.

    The callback supports both minimization (loss) and maximization (accuracy)
    metrics, and includes a minimum delta threshold to avoid stopping on
    negligible improvements.

    Example:
        early_stopping = EarlyStopping(patience=10, mode='min', min_delta=1e-4)
        for epoch in range(num_epochs):
            val_loss = train_one_epoch()
            if early_stopping(val_loss, model):
                break
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = 'min',
        restore_best_weights: bool = True
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping. Must be positive.
            min_delta: Minimum change to qualify as improvement. Should be small
                positive value to avoid stopping on negligible changes.
            mode: Either 'min' for loss-like metrics or 'max' for accuracy-like
                metrics.
            restore_best_weights: Whether to restore best model weights when
                early stopping is triggered.

        Raises:
            ValueError: If patience is not positive or mode is invalid.
        """
        if patience <= 0:
            raise ValueError(f"Patience must be positive, got {patience}")
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
        if min_delta < 0:
            raise ValueError(f"Min delta must be non-negative, got {min_delta}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_weights = None
        self.should_stop = False

        logger.info(
            f"Initialized early stopping: patience={patience}, mode={mode}, "
            f"min_delta={min_delta}, restore_weights={restore_best_weights}"
        )

    def __call__(self, score: float, model: nn.Module) -> bool:
        """Check if training should stop and update internal state.

        Args:
            score: Current score to evaluate (validation loss, accuracy, etc.).
            model: Model to potentially save weights from.

        Returns:
            True if training should stop, False otherwise.

        Raises:
            EarlyStoppingError: If weight restoration fails.
        """
        if not isinstance(score, (int, float)) or np.isnan(score):
            logger.warning(f"Invalid score received: {score}. Continuing training.")
            return False

        try:
            if self._is_improvement(score):
                self.best_score = score
                self.counter = 0
                if self.restore_best_weights:
                    # Store weights on CPU to save GPU memory
                    self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                logger.debug(f"New best score: {score:.6f}")
            else:
                self.counter += 1
                logger.debug(f"No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")

                if self.restore_best_weights and self.best_weights:
                    try:
                        # Move weights back to model's device
                        device = next(model.parameters()).device
                        best_weights_on_device = {k: v.to(device) for k, v in self.best_weights.items()}
                        model.load_state_dict(best_weights_on_device)
                        logger.info(f"Restored best model weights (score: {self.best_score:.6f})")
                    except Exception as e:
                        raise EarlyStoppingError(
                            f"Failed to restore best model weights: {e}",
                            details={'best_score': self.best_score, 'counter': self.counter}
                        )

            return self.should_stop

        except Exception as e:
            if isinstance(e, EarlyStoppingError):
                raise
            logger.error(f"Error in early stopping callback: {e}")
            return False

    def _is_improvement(self, score: float) -> bool:
        """Check if score represents an improvement over the best score.

        Args:
            score: Current score to evaluate.

        Returns:
            True if score is better than best score by at least min_delta.
        """
        if self.mode == 'min':
            return score < (self.best_score - self.min_delta)
        else:
            return score > (self.best_score + self.min_delta)

    def reset(self) -> None:
        """Reset the early stopping state for a new training run."""
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
        logger.info("Early stopping state reset")


class TemporalDrugTrainer:
    """Trainer for temporal drug interaction prediction models.

    This class orchestrates the complete training pipeline for temporal drug
    interaction prediction using heterogeneous graph neural networks. It handles:
    - Model initialization with multi-task learning objectives
    - Training loop with gradient clipping and learning rate scheduling
    - Validation with comprehensive metric computation
    - Checkpoint saving and loading for model persistence
    - MLflow experiment tracking for reproducible research
    - Early stopping to prevent overfitting

    The trainer supports both binary and multi-class drug interaction prediction
    and can handle various input formats including heterogeneous graphs and
    traditional feature-based representations.

    Example:
        config = Config.from_yaml('config.yaml')
        trainer = TemporalDrugTrainer(config)
        history = trainer.train(train_loader, val_loader)
        test_metrics = trainer.evaluate(test_loader)
    """

    def __init__(self, config: Config) -> None:
        """Initialize the trainer with configuration and setup components.

        Args:
            config: Configuration object containing all training parameters,
                model architecture settings, and experiment configuration.

        Raises:
            ModelInitializationError: If model, optimizer, or scheduler creation fails.
            ValidationError: If configuration validation fails.
        """
        logger.info("Initializing TemporalDrugTrainer")

        # Validate configuration
        self._validate_config(config)
        self.config = config

        # Set up device and reproducibility
        try:
            self.device = torch.device(config.device)
            if self.device.type == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = torch.device('cpu')

            logger.info(f"Using device: {self.device}")

            # Set random seeds for reproducibility
            set_seeds(config.seed)
            logger.debug(f"Set random seed: {config.seed}")

        except Exception as e:
            raise ModelInitializationError(f"Failed to setup device and seeds: {e}")

        # Initialize components
        try:
            # Initialize model
            logger.info("Creating temporal drug interaction model")
            self.model = self._create_model()
            self.model = self.model.to(self.device)
            # Note: GATConv uses lazy initialization, so parameter counting
            # must be deferred until after the first forward pass.
            logger.info("Model created and moved to device (lazy params pending initialization)")

            # Initialize optimizer and scheduler
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()

            # Initialize loss function and metrics
            self.criterion = self._create_loss_function()
            self.metrics = DrugInteractionMetrics(config.target_metrics)

            # Early stopping
            self.early_stopping = EarlyStopping(
                patience=config.training.early_stopping_patience,
                mode='min',
                min_delta=getattr(config.training, 'early_stopping_min_delta', 1e-4),
                restore_best_weights=getattr(config.training, 'restore_best_weights', True)
            )

        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize model components: {e}")

        # Initialize training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_start_time = None

        # Create directories
        try:
            self.checkpoint_dir = Path(config.experiment.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        except Exception as e:
            raise ModelInitializationError(f"Failed to create checkpoint directory: {e}")

        # Initialize MLflow if available
        if MLFLOW_AVAILABLE and config.experiment.use_mlflow:
            try:
                self._setup_mlflow()
            except Exception as e:
                logger.error(f"MLflow setup failed: {e}")
                if getattr(config.experiment, 'require_mlflow', False):
                    raise MLflowTrackingError(f"MLflow required but setup failed: {e}")

        logger.info("TemporalDrugTrainer initialization complete")

    def _validate_config(self, config: Config) -> None:
        """Validate configuration parameters.

        Args:
            config: Configuration to validate.

        Raises:
            ValidationError: If configuration is invalid.
        """
        required_attrs = [
            'device', 'seed', 'training', 'model', 'data', 'experiment'
        ]

        for attr in required_attrs:
            if not hasattr(config, attr):
                raise ValidationError(f"Missing required config attribute: {attr}")

        # Validate training config
        if config.training.learning_rate <= 0:
            raise ValidationError(f"Learning rate must be positive: {config.training.learning_rate}")

        if config.training.batch_size <= 0:
            raise ValidationError(f"Batch size must be positive: {config.training.batch_size}")

        if config.training.num_epochs <= 0:
            raise ValidationError(f"Number of epochs must be positive: {config.training.num_epochs}")

        # Validate model config
        if config.model.hidden_dim <= 0:
            raise ValidationError(f"Hidden dimension must be positive: {config.model.hidden_dim}")

        logger.debug("Configuration validation passed")

    def _create_model(self) -> TemporalDrugInteractionGNN:
        """Create the temporal drug interaction model.

        This method initializes the heterogeneous graph neural network model
        for temporal drug interaction prediction. The model supports multiple
        node types (drugs, metabolites, targets) and various edge types
        representing biological relationships.

        Returns:
            Initialized TemporalDrugInteractionGNN model.

        Raises:
            ModelInitializationError: If model creation fails due to invalid
                parameters or insufficient resources.
        """
        try:
            # Define node type dimensions based on input feature sizes
            # These dimensions should match the preprocessing pipeline output
            node_type_dims = {
                'drug': self.config.data.molecular_feature_dim,
                'metabolite': getattr(self.config.data, 'metabolite_feature_dim', 64),
                'target': getattr(self.config.data, 'target_feature_dim', 64)
            }

            logger.debug(f"Node type dimensions: {node_type_dims}")

            # Define edge types for heterogeneous graph representing biological relationships
            # Each tuple represents (source_type, edge_type, destination_type)
            edge_types = [
                ('drug', 'metabolizes_to', 'metabolite'),      # Drug metabolism
                ('metabolite', 'metabolized_from', 'drug'),    # Reverse metabolism
                ('drug', 'targets', 'target'),                 # Drug-target interactions
                ('target', 'targeted_by', 'drug'),             # Reverse targeting
                ('metabolite', 'affects', 'target'),           # Metabolite effects
                ('target', 'affected_by', 'metabolite'),       # Reverse effects
                ('drug', 'interacts', 'drug')                  # Drug-drug interactions
            ]

            logger.debug(f"Defined {len(edge_types)} edge types for heterogeneous graph")

            # Validate model configuration parameters
            if self.config.model.hidden_dim <= 0:
                raise ValueError("Hidden dimension must be positive")
            if self.config.model.num_layers <= 0:
                raise ValueError("Number of layers must be positive")
            if self.config.model.num_heads <= 0:
                raise ValueError("Number of attention heads must be positive")
            if not 0 <= self.config.model.dropout <= 1:
                raise ValueError("Dropout must be between 0 and 1")

            # Create model with validated parameters
            # num_interaction_types=1 for binary classification task
            model = TemporalDrugInteractionGNN(
                node_type_dims=node_type_dims,
                edge_types=edge_types,
                hidden_dim=self.config.model.hidden_dim,
                num_layers=self.config.model.num_layers,
                num_heads=self.config.model.num_heads,
                dropout=self.config.model.dropout,
                temporal_attention_dim=self.config.model.temporal_attention_dim,
                metabolite_pathway_dim=self.config.model.metabolite_pathway_dim,
                max_time_steps=self.config.model.max_time_steps,
                num_interaction_types=1,
            )

            # Log model architecture details
            logger.info(
                f"Created model: {self.config.model.num_layers} layers, "
                f"{self.config.model.hidden_dim} hidden dims, "
                f"{self.config.model.num_heads} attention heads"
            )

            return model

        except Exception as e:
            raise ModelInitializationError(
                f"Failed to create temporal drug interaction model: {e}",
                details={
                    'node_types': list(node_type_dims.keys()) if 'node_type_dims' in locals() else None,
                    'edge_types_count': len(edge_types) if 'edge_types' in locals() else None,
                    'config': {
                        'hidden_dim': self.config.model.hidden_dim,
                        'num_layers': self.config.model.num_layers,
                        'num_heads': self.config.model.num_heads
                    }
                }
            )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for model training.

        Supports AdamW and SGD optimizers with configurable parameters.
        AdamW is generally recommended for transformer-based models and
        provides better generalization, while SGD with momentum can be
        more stable for certain architectures.

        Returns:
            Configured PyTorch optimizer.

        Raises:
            ModelInitializationError: If optimizer creation fails or
                unsupported optimizer is specified.
        """
        try:
            optimizer_name = self.config.training.optimizer.lower()
            lr = self.config.training.learning_rate
            weight_decay = getattr(self.config.training, 'weight_decay', 0.01)

            logger.debug(f"Creating {optimizer_name} optimizer with lr={lr}, weight_decay={weight_decay}")

            if optimizer_name == 'adamw':
                optimizer = AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=getattr(self.config.training, 'betas', (0.9, 0.999)),
                    eps=getattr(self.config.training, 'eps', 1e-8)
                )
                logger.info(f"Created AdamW optimizer with lr={lr}")

            elif optimizer_name == 'sgd':
                momentum = getattr(self.config.training, 'momentum', 0.9)
                optimizer = SGD(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    nesterov=getattr(self.config.training, 'nesterov', False)
                )
                logger.info(f"Created SGD optimizer with lr={lr}, momentum={momentum}")

            else:
                supported_optimizers = ['adamw', 'sgd']
                raise ValueError(
                    f"Unsupported optimizer: {self.config.training.optimizer}. "
                    f"Supported optimizers: {supported_optimizers}"
                )

            return optimizer

        except Exception as e:
            raise ModelInitializationError(
                f"Failed to create optimizer: {e}",
                details={
                    'optimizer': self.config.training.optimizer,
                    'learning_rate': self.config.training.learning_rate,
                    'weight_decay': getattr(self.config.training, 'weight_decay', 0.01)
                }
            )

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler for training optimization.

        Supports cosine annealing, step decay, and plateau-based scheduling.
        The scheduler helps improve convergence by adjusting learning rate
        during training based on different strategies.

        Returns:
            Configured learning rate scheduler or None if no scheduler specified.

        Raises:
            ModelInitializationError: If scheduler creation fails.
        """
        try:
            scheduler_name = getattr(self.config.training, 'scheduler', 'none').lower()

            if scheduler_name == 'none' or not scheduler_name:
                logger.info("No learning rate scheduler specified")
                return None

            logger.debug(f"Creating {scheduler_name} scheduler")

            if scheduler_name == 'cosine':
                T_max = self.config.training.num_epochs
                eta_min = getattr(self.config.training, 'min_lr', 0)
                scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=T_max,
                    eta_min=eta_min
                )
                logger.info(f"Created cosine annealing scheduler (T_max={T_max}, eta_min={eta_min})")

            elif scheduler_name == 'step':
                step_size = getattr(
                    self.config.training, 'scheduler_step_size',
                    self.config.training.num_epochs // 3
                )
                gamma = getattr(self.config.training, 'scheduler_gamma', 0.1)
                scheduler = StepLR(
                    self.optimizer,
                    step_size=step_size,
                    gamma=gamma
                )
                logger.info(f"Created step LR scheduler (step_size={step_size}, gamma={gamma})")

            elif scheduler_name == 'plateau':
                patience = getattr(self.config.training, 'scheduler_patience', 5)
                factor = getattr(self.config.training, 'scheduler_factor', 0.5)
                threshold = getattr(self.config.training, 'scheduler_threshold', 1e-4)
                scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=factor,
                    patience=patience,
                    threshold=threshold,
                    min_lr=getattr(self.config.training, 'min_lr', 0)
                )
                logger.info(
                    f"Created plateau scheduler (patience={patience}, "
                    f"factor={factor}, threshold={threshold})"
                )

            else:
                supported_schedulers = ['cosine', 'step', 'plateau', 'none']
                raise ValueError(
                    f"Unsupported scheduler: {scheduler_name}. "
                    f"Supported schedulers: {supported_schedulers}"
                )

            return scheduler

        except Exception as e:
            raise ModelInitializationError(
                f"Failed to create learning rate scheduler: {e}",
                details={
                    'scheduler': getattr(self.config.training, 'scheduler', 'none'),
                    'num_epochs': self.config.training.num_epochs
                }
            )

    def _create_loss_function(self) -> nn.Module:
        """Create loss function.

        Returns:
            Loss function.
        """
        # For demonstration, use BCEWithLogitsLoss
        # In practice, the model computes its own multi-task loss
        return nn.BCEWithLogitsLoss()

    def _setup_mlflow(self) -> None:
        """Set up MLflow experiment tracking.

        Initializes MLflow for experiment tracking including setting up
        tracking URI, experiment, and logging hyperparameters. Handles
        various MLflow configuration scenarios gracefully.

        Raises:
            MLflowTrackingError: If MLflow setup fails and tracking is required.
        """
        try:
            # Set tracking URI if specified
            if hasattr(self.config.experiment, 'mlflow_tracking_uri') and self.config.experiment.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.config.experiment.mlflow_tracking_uri)
                logger.debug(f"MLflow tracking URI: {self.config.experiment.mlflow_tracking_uri}")

            # Set experiment name
            experiment_name = getattr(
                self.config.experiment, 'mlflow_experiment_name',
                'temporal_drug_interaction_prediction'
            )
            mlflow.set_experiment(experiment_name)
            logger.debug(f"MLflow experiment: {experiment_name}")

            # Start MLflow run with optional run name
            run_name = getattr(self.config.experiment, 'run_name', None)
            mlflow.start_run(run_name=run_name)
            logger.info(f"Started MLflow run: {run_name or 'auto-generated'}")

            # Prepare hyperparameters for logging
            hyperparameters = {
                'model_hidden_dim': self.config.model.hidden_dim,
                'model_num_layers': self.config.model.num_layers,
                'model_num_heads': self.config.model.num_heads,
                'model_dropout': self.config.model.dropout,
                'learning_rate': self.config.training.learning_rate,
                'batch_size': self.config.training.batch_size,
                'num_epochs': self.config.training.num_epochs,
                'optimizer': self.config.training.optimizer,
                'device': self.config.device,
                'seed': self.config.seed
            }

            # Add optional parameters if they exist
            if hasattr(self.config.training, 'scheduler'):
                hyperparameters['scheduler'] = self.config.training.scheduler
            if hasattr(self.config.training, 'weight_decay'):
                hyperparameters['weight_decay'] = self.config.training.weight_decay
            if hasattr(self.config.training, 'gradient_clip_norm'):
                hyperparameters['gradient_clip_norm'] = self.config.training.gradient_clip_norm

            # Log hyperparameters to MLflow
            mlflow.log_params(hyperparameters)
            logger.info(f"Logged {len(hyperparameters)} hyperparameters to MLflow")

            # Log additional configuration as tags
            mlflow.set_tags({
                'model_type': 'TemporalDrugInteractionGNN',
                'framework': 'PyTorch',
                'task': 'drug_interaction_prediction'
            })

            logger.info("MLflow tracking initialized successfully")

        except ImportError:
            raise MLflowTrackingError("MLflow not available but tracking was requested")
        except Exception as e:
            error_msg = f"Failed to setup MLflow: {e}"
            logger.error(error_msg)
            raise MLflowTrackingError(error_msg, details={'experiment_name': experiment_name})

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        epoch_losses = []
        epoch_metrics = DrugInteractionMetrics()

        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()

            # Move batch to device
            if isinstance(batch, dict):
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
            else:
                batch = batch.to(self.device)

            try:
                # Forward pass
                if hasattr(batch, 'x_dict'):  # HeteroData
                    outputs = self.model(batch)
                    # Extract interaction predictions
                    interaction_logits = outputs['interaction_logits']
                    targets = getattr(batch, 'interaction_labels', torch.zeros_like(interaction_logits))
                else:
                    # Regular batch with drug features
                    drug1_features = batch['drug1_features']
                    drug2_features = batch['drug2_features']
                    targets = batch['label']

                    # Create synthetic heterogeneous graph for this batch
                    hetero_data = self._create_batch_hetero_data(batch)
                    outputs = self.model(hetero_data)
                    interaction_logits = outputs['interaction_logits']

                # Compute loss
                if interaction_logits.dim() > 1 and interaction_logits.size(1) > 1:
                    # Multi-class case
                    loss = nn.CrossEntropyLoss()(interaction_logits, targets.long())
                else:
                    # Binary case
                    if interaction_logits.dim() > 1:
                        interaction_logits = interaction_logits.squeeze(1)
                    loss = nn.BCEWithLogitsLoss()(interaction_logits, targets.float())

                # Add additional losses from model if available
                if hasattr(self.model, 'compute_loss'):
                    model_losses = self.model.compute_loss(outputs, {'interaction_labels': targets})
                    if 'total_loss' in model_losses:
                        loss = model_losses['total_loss']

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config.training.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_norm
                    )

                self.optimizer.step()

                # Update metrics
                epoch_losses.append(loss.item())
                with torch.no_grad():
                    if interaction_logits.dim() > 1 and interaction_logits.size(1) > 1:
                        predictions = torch.softmax(interaction_logits, dim=1)
                    else:
                        predictions = torch.sigmoid(interaction_logits)

                    epoch_metrics.update(predictions, targets)

                self.global_step += 1

                # Log training progress
                if batch_idx % self.config.experiment.log_frequency == 0:
                    logger.info(
                        f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                        f'Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]["lr"]:.6f}'
                    )

            except RuntimeError as e:
                # Handle common PyTorch runtime errors
                if "out of memory" in str(e).lower():
                    logger.error(f"GPU out of memory in batch {batch_idx}. Consider reducing batch size.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    logger.error(f"Runtime error in training batch {batch_idx}: {e}")
                raise BatchProcessingError(
                    f"Runtime error during training batch {batch_idx}",
                    batch_idx=batch_idx,
                    epoch=epoch,
                    details=str(e)
                )
            except ValueError as e:
                logger.error(f"Value error in training batch {batch_idx}: {e}")
                raise BatchProcessingError(
                    f"Invalid values in training batch {batch_idx}",
                    batch_idx=batch_idx,
                    epoch=epoch,
                    details=str(e)
                )
            except Exception as e:
                logger.error(f"Unexpected error in training batch {batch_idx}: {e}")
                # For unexpected errors, we can choose to continue or fail
                # Continuing allows training to proceed with other batches
                continue

        # Compute epoch metrics
        train_metrics = epoch_metrics.compute_interaction_metrics()
        train_metrics['loss'] = np.mean(epoch_losses) if epoch_losses else 0.0
        train_metrics['epoch_time'] = time.time() - start_time

        return train_metrics

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch.

        Args:
            val_loader: Validation data loader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        epoch_losses = []
        epoch_metrics = DrugInteractionMetrics()

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                if isinstance(batch, dict):
                    for key in batch:
                        if torch.is_tensor(batch[key]):
                            batch[key] = batch[key].to(self.device)
                else:
                    batch = batch.to(self.device)

                try:
                    # Forward pass
                    if hasattr(batch, 'x_dict'):  # HeteroData
                        outputs = self.model(batch)
                        interaction_logits = outputs['interaction_logits']
                        targets = getattr(batch, 'interaction_labels', torch.zeros_like(interaction_logits))
                    else:
                        # Regular batch
                        hetero_data = self._create_batch_hetero_data(batch)
                        outputs = self.model(hetero_data)
                        interaction_logits = outputs['interaction_logits']
                        targets = batch['label']

                    # Compute loss
                    if interaction_logits.dim() > 1 and interaction_logits.size(1) > 1:
                        loss = nn.CrossEntropyLoss()(interaction_logits, targets.long())
                        predictions = torch.softmax(interaction_logits, dim=1)
                    else:
                        if interaction_logits.dim() > 1:
                            interaction_logits = interaction_logits.squeeze(1)
                        loss = nn.BCEWithLogitsLoss()(interaction_logits, targets.float())
                        predictions = torch.sigmoid(interaction_logits)

                    epoch_losses.append(loss.item())
                    epoch_metrics.update(predictions, targets)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error("GPU out of memory during validation. Consider reducing batch size.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    logger.error(f"Runtime error in validation: {e}")
                    continue  # Skip this batch but continue validation
                except ValueError as e:
                    logger.error(f"Value error in validation: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error in validation batch: {e}")
                    continue

        # Compute validation metrics
        val_metrics = epoch_metrics.compute_all_metrics()
        val_metrics['loss'] = np.mean(epoch_losses) if epoch_losses else 0.0

        return val_metrics

    def _create_batch_hetero_data(self, batch: Dict[str, torch.Tensor]) -> HeteroData:
        """Create heterogeneous data from batch for model input.

        Args:
            batch: Batch dictionary.

        Returns:
            HeteroData object.
        """
        batch_size = batch['drug1_features'].size(0)
        hetero_data = HeteroData()

        # Stack drug features: interleave drug1 and drug2 features
        drug_features = torch.stack([batch['drug1_features'], batch['drug2_features']], dim=1)
        drug_features = drug_features.view(-1, drug_features.size(-1))  # Flatten to (batch_size*2, feat_dim)
        num_drugs = drug_features.size(0)

        hetero_data['drug'].x = drug_features

        # Create dummy metabolite and target features
        num_met = num_drugs
        num_target = num_drugs
        hetero_data['metabolite'].x = torch.randn(num_met, 64, device=drug_features.device)
        hetero_data['target'].x = torch.randn(num_target, 64, device=drug_features.device)

        # Create edge indices
        drug_indices = torch.arange(num_drugs, device=drug_features.device)
        met_indices = torch.arange(num_met, device=drug_features.device)
        target_indices = torch.arange(num_target, device=drug_features.device)

        # Drug -> Metabolite and reverse
        hetero_data[('drug', 'metabolizes_to', 'metabolite')].edge_index = torch.stack([
            drug_indices, met_indices
        ])
        hetero_data[('metabolite', 'metabolized_from', 'drug')].edge_index = torch.stack([
            met_indices, drug_indices
        ])

        # Drug -> Target and reverse
        hetero_data[('drug', 'targets', 'target')].edge_index = torch.stack([
            drug_indices, target_indices
        ])
        hetero_data[('target', 'targeted_by', 'drug')].edge_index = torch.stack([
            target_indices, drug_indices
        ])

        # Metabolite -> Target and reverse
        hetero_data[('metabolite', 'affects', 'target')].edge_index = torch.stack([
            met_indices, target_indices
        ])
        hetero_data[('target', 'affected_by', 'metabolite')].edge_index = torch.stack([
            target_indices, met_indices
        ])

        # Drug -> Drug (self-interactions for pairs)
        # Connect drug pairs: drug1->drug2 and drug2->drug1
        src = torch.arange(0, num_drugs, 2, device=drug_features.device)
        dst = torch.arange(1, num_drugs, 2, device=drug_features.device)
        if len(src) > 0 and len(dst) > 0:
            drug_drug_src = torch.cat([src, dst])
            drug_drug_dst = torch.cat([dst, src])
            hetero_data[('drug', 'interacts', 'drug')].edge_index = torch.stack([
                drug_drug_src, drug_drug_dst
            ])

        return hetero_data

    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        """Save model checkpoint with validation metrics.

        Creates a comprehensive checkpoint including model state, optimizer state,
        scheduler state, and training metadata. Automatically saves the best
        performing model based on validation loss.

        Args:
            epoch: Current epoch number.
            val_metrics: Dictionary of validation metrics including loss.

        Raises:
            CheckpointError: If checkpoint saving fails due to I/O or serialization issues.
        """
        try:
            # Prepare checkpoint data
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': self.config.to_dict(),
                'global_step': self.global_step,
                'best_val_loss': self.best_val_loss,
                'timestamp': time.time()
            }

            # Add scheduler state if available
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

            # Add early stopping state
            if hasattr(self.early_stopping, 'best_score'):
                checkpoint['early_stopping_state'] = {
                    'best_score': self.early_stopping.best_score,
                    'counter': self.early_stopping.counter,
                    'should_stop': self.early_stopping.should_stop
                }

            # Save regular checkpoint
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Save best model if this is the best performance
            current_val_loss = val_metrics.get('loss', float('inf'))
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                best_path = self.checkpoint_dir / 'best_model.pt'
                torch.save(checkpoint, best_path)
                logger.info(f"New best model saved (val_loss={current_val_loss:.6f}) at epoch {epoch}")

        except OSError as e:
            raise CheckpointError(
                f"Failed to save checkpoint due to I/O error: {e}",
                details={'epoch': epoch, 'checkpoint_dir': str(self.checkpoint_dir)}
            )
        except RuntimeError as e:
            raise CheckpointError(
                f"Failed to serialize checkpoint: {e}",
                details={'epoch': epoch, 'model_type': type(self.model).__name__}
            )
        except Exception as e:
            raise CheckpointError(
                f"Unexpected error saving checkpoint: {e}",
                details={'epoch': epoch, 'val_loss': val_metrics.get('loss', 'unknown')}
            )

    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, float]:
        """Load model checkpoint and restore training state.

        Loads a previously saved checkpoint including model weights, optimizer
        state, scheduler state, and training metadata. Validates checkpoint
        compatibility and handles device mapping automatically.

        Args:
            checkpoint_path: Path to the checkpoint file to load.

        Returns:
            Dictionary of validation metrics from the loaded checkpoint.

        Raises:
            CheckpointError: If checkpoint loading fails due to file issues,
                compatibility problems, or state restoration errors.
        """
        try:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

            logger.info(f"Loading checkpoint from: {checkpoint_path}")

            # Load checkpoint with proper device mapping
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Validate checkpoint structure
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'global_step']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

            # Load model state
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.debug("Loaded model state dict")
            except RuntimeError as e:
                raise ValueError(f"Model state dict incompatible: {e}")

            # Load optimizer state
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.debug("Loaded optimizer state dict")
            except ValueError as e:
                logger.warning(f"Could not load optimizer state: {e}. Using fresh optimizer state.")

            # Restore training state
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            # Load scheduler state if available
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.debug("Loaded scheduler state dict")
                except Exception as e:
                    logger.warning(f"Could not load scheduler state: {e}")

            # Restore early stopping state if available
            if 'early_stopping_state' in checkpoint and hasattr(self.early_stopping, 'best_score'):
                es_state = checkpoint['early_stopping_state']
                self.early_stopping.best_score = es_state.get('best_score', float('inf'))
                self.early_stopping.counter = es_state.get('counter', 0)
                self.early_stopping.should_stop = es_state.get('should_stop', False)
                logger.debug("Restored early stopping state")

            # Log checkpoint info
            val_metrics = checkpoint.get('val_metrics', {})
            logger.info(
                f"Successfully loaded checkpoint from epoch {self.epoch} "
                f"(global_step={self.global_step}, "
                f"val_loss={val_metrics.get('loss', 'unknown')})"
            )

            return val_metrics

        except FileNotFoundError:
            raise CheckpointError(
                f"Checkpoint file not found: {checkpoint_path}",
                details={'path': str(checkpoint_path)}
            )
        except (KeyError, ValueError) as e:
            raise CheckpointError(
                f"Invalid checkpoint format: {e}",
                details={'path': str(checkpoint_path)}
            )
        except Exception as e:
            raise CheckpointError(
                f"Failed to load checkpoint: {e}",
                details={'path': str(checkpoint_path), 'device': str(self.device)}
            )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Number of epochs (uses config if None).

        Returns:
            Training history.
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs

        history = {
            'train_loss': [], 'val_loss': [],
            'train_auroc': [], 'val_auroc': [],
            'train_accuracy': [], 'val_accuracy': []
        }

        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validation phase
            val_metrics = self.validate_epoch(val_loader)

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUROC: {val_metrics.get('interaction_auroc', 0):.4f}"
            )

            # MLflow logging
            if MLFLOW_AVAILABLE and self.config.experiment.use_mlflow:
                try:
                    mlflow.log_metrics({
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                        'train_auroc': train_metrics.get('interaction_auroc', 0),
                        'val_auroc': val_metrics.get('interaction_auroc', 0),
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    }, step=epoch)
                except Exception as e:
                    logger.warning(f"MLflow logging failed: {e}")

            # Save history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_auroc'].append(train_metrics.get('interaction_auroc', 0))
            history['val_auroc'].append(val_metrics.get('interaction_auroc', 0))
            history['train_accuracy'].append(train_metrics.get('interaction_accuracy', 0))
            history['val_accuracy'].append(val_metrics.get('interaction_accuracy', 0))

            # Save checkpoint
            if (epoch + 1) % self.config.experiment.save_checkpoint_frequency == 0:
                self.save_checkpoint(epoch, val_metrics)

            # Early stopping
            if self.early_stopping(val_metrics['loss'], self.model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")

        # Save final model
        self.save_checkpoint(self.epoch, val_metrics)

        # Log final metrics to MLflow
        if MLFLOW_AVAILABLE and self.config.experiment.use_mlflow:
            try:
                final_metrics = self.metrics.compute_all_metrics()
                mlflow.log_metrics(final_metrics)
                mlflow.pytorch.log_model(self.model, "model")
            except Exception as e:
                logger.warning(f"Final MLflow logging failed: {e}")

        return history

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set.

        Args:
            test_loader: Test data loader.

        Returns:
            Test metrics.
        """
        logger.info("Evaluating model on test set")
        test_metrics = self.validate_epoch(test_loader)

        # Print comprehensive metrics
        test_eval_metrics = DrugInteractionMetrics(self.config.target_metrics)

        # Collect all predictions for comprehensive evaluation
        all_predictions = []
        all_targets = []

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    for key in batch:
                        if torch.is_tensor(batch[key]):
                            batch[key] = batch[key].to(self.device)
                else:
                    batch = batch.to(self.device)

                try:
                    if hasattr(batch, 'x_dict'):
                        outputs = self.model(batch)
                        interaction_logits = outputs['interaction_logits']
                        targets = getattr(batch, 'interaction_labels', torch.zeros_like(interaction_logits))
                    else:
                        hetero_data = self._create_batch_hetero_data(batch)
                        outputs = self.model(hetero_data)
                        interaction_logits = outputs['interaction_logits']
                        targets = batch['label']

                    if interaction_logits.dim() > 1 and interaction_logits.size(1) > 1:
                        predictions = torch.softmax(interaction_logits, dim=1)
                    else:
                        if interaction_logits.dim() > 1:
                            interaction_logits = interaction_logits.squeeze(1)
                        predictions = torch.sigmoid(interaction_logits)

                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())

                except Exception as e:
                    logger.error(f"Error in test evaluation: {e}")
                    continue

        # Update metrics with all predictions
        if all_predictions:
            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)
            test_eval_metrics.update(all_predictions, all_targets)

        # Compute comprehensive metrics
        comprehensive_metrics = test_eval_metrics.compute_all_metrics()
        test_metrics.update(comprehensive_metrics)

        # Print metrics summary
        test_eval_metrics.print_metrics_summary()

        return test_metrics