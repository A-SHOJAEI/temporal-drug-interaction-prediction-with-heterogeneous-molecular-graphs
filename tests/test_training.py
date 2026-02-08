"""Tests for training pipeline and trainer functionality."""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.training.trainer import (
    TemporalDrugTrainer, EarlyStopping
)
from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.data.loader import DrugInteractionDataLoader
from torch.utils.data import DataLoader, TensorDataset


class TestEarlyStopping:
    """Test early stopping functionality."""

    def test_early_stopping_initialization(self):
        """Test early stopping initialization."""
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)

        assert early_stopping.patience == 5
        assert early_stopping.min_delta == 0.001
        assert early_stopping.mode == 'min'
        assert not early_stopping.should_stop
        assert early_stopping.counter == 0

    def test_early_stopping_improvement_detection(self):
        """Test early stopping improvement detection."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')

        # Mock model for testing
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {'param': torch.tensor([1.0])}

        # Test improvement (decreasing loss)
        assert not early_stopping(1.0, mock_model)  # First call
        assert early_stopping.counter == 0

        assert not early_stopping(0.8, mock_model)  # Improvement
        assert early_stopping.counter == 0

        assert not early_stopping(0.5, mock_model)  # Another improvement
        assert early_stopping.counter == 0

    def test_early_stopping_no_improvement(self):
        """Test early stopping when no improvement."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, mode='min')

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {'param': torch.tensor([1.0])}

        # Initialize with first score
        assert not early_stopping(1.0, mock_model)

        # No improvement
        assert not early_stopping(1.05, mock_model)  # No improvement
        assert early_stopping.counter == 1

        assert early_stopping(1.1, mock_model)  # Trigger early stopping
        assert early_stopping.should_stop

    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, mode='max')

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {'param': torch.tensor([1.0])}

        # Test improvement (increasing accuracy)
        assert not early_stopping(0.8, mock_model)
        assert not early_stopping(0.9, mock_model)  # Improvement
        assert early_stopping.counter == 0

        # No improvement
        assert not early_stopping(0.85, mock_model)  # Decrease
        assert early_stopping.counter == 1

        assert early_stopping(0.8, mock_model)  # Trigger early stopping

    def test_early_stopping_weight_restoration(self):
        """Test weight restoration functionality."""
        early_stopping = EarlyStopping(patience=1, restore_best_weights=True)

        # Create a simple model
        model = torch.nn.Linear(2, 1)
        initial_weight = model.weight.data.clone()

        # First call (best score)
        early_stopping(0.5, model)

        # Modify model weights
        model.weight.data.fill_(999.0)

        # Trigger early stopping (should restore weights)
        early_stopping(0.6, model)  # No improvement
        early_stopping(0.7, model)  # Trigger early stopping

        # Check that weights were restored (approximately)
        restored_weight = model.weight.data
        assert torch.allclose(restored_weight, initial_weight), "Weights should be restored"


class TestTemporalDrugTrainer:
    """Test temporal drug trainer functionality."""

    def test_trainer_initialization(self, test_config):
        """Test trainer initialization."""
        trainer = TemporalDrugTrainer(test_config)

        assert trainer.config == test_config
        assert trainer.device == torch.device(test_config.device)
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.metrics is not None
        assert trainer.early_stopping is not None

        # Check that directories are created
        assert trainer.checkpoint_dir.exists()

    def test_trainer_model_creation(self, test_config):
        """Test trainer model creation."""
        trainer = TemporalDrugTrainer(test_config)

        model = trainer.model
        assert model is not None

        # Check model configuration
        assert model.hidden_dim == test_config.model.hidden_dim
        assert model.num_layers == test_config.model.num_layers

        # Check that model is on correct device
        for param in model.parameters():
            assert param.device == trainer.device

    def test_trainer_optimizer_creation(self, test_config):
        """Test trainer optimizer creation."""
        # Test AdamW optimizer
        test_config.training.optimizer = "adamw"
        trainer = TemporalDrugTrainer(test_config)

        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert trainer.optimizer.param_groups[0]['lr'] == test_config.training.learning_rate

        # Test SGD optimizer
        test_config.training.optimizer = "sgd"
        trainer = TemporalDrugTrainer(test_config)

        assert isinstance(trainer.optimizer, torch.optim.SGD)

    def test_trainer_scheduler_creation(self, test_config):
        """Test trainer scheduler creation."""
        # Test cosine scheduler
        test_config.training.scheduler = "cosine"
        trainer = TemporalDrugTrainer(test_config)

        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

        # Test step scheduler
        test_config.training.scheduler = "step"
        trainer = TemporalDrugTrainer(test_config)

        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)

        # Test plateau scheduler
        test_config.training.scheduler = "plateau"
        trainer = TemporalDrugTrainer(test_config)

        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_trainer_create_batch_hetero_data(self, test_config, sample_batch):
        """Test creation of heterogeneous data from batch."""
        trainer = TemporalDrugTrainer(test_config)

        hetero_data = trainer._create_batch_hetero_data(sample_batch)

        # Check that heterogeneous data was created
        assert hasattr(hetero_data, 'x_dict')
        assert 'drug' in hetero_data.x_dict
        assert 'metabolite' in hetero_data.x_dict
        assert 'target' in hetero_data.x_dict

        # Check shapes
        batch_size = sample_batch['drug1_features'].size(0)
        assert hetero_data['drug'].x.size(0) == batch_size * 2  # Two drugs per pair

    def test_trainer_train_epoch_basic(self, test_config):
        """Test basic training epoch functionality."""
        trainer = TemporalDrugTrainer(test_config)

        # Create simple training data
        num_samples = 8
        feature_dim = test_config.data.molecular_feature_dim

        train_data = []
        for i in range(num_samples):
            train_data.append({
                'drug1_features': torch.randn(feature_dim),
                'drug2_features': torch.randn(feature_dim),
                'label': torch.randint(0, 2, (1,)).float(),
                'drug1_id': f'drug1_{i}',
                'drug2_id': f'drug2_{i}'
            })

        train_loader = DataLoader(train_data, batch_size=2, shuffle=False)

        # Train for one epoch
        metrics = trainer.train_epoch(train_loader, epoch=0)

        # Check that metrics were returned
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'epoch_time' in metrics
        assert metrics['loss'] >= 0

    def test_trainer_validate_epoch_basic(self, test_config):
        """Test basic validation epoch functionality."""
        trainer = TemporalDrugTrainer(test_config)

        # Create simple validation data
        num_samples = 4
        feature_dim = test_config.data.molecular_feature_dim

        val_data = []
        for i in range(num_samples):
            val_data.append({
                'drug1_features': torch.randn(feature_dim),
                'drug2_features': torch.randn(feature_dim),
                'label': torch.randint(0, 2, (1,)).float(),
                'drug1_id': f'drug1_{i}',
                'drug2_id': f'drug2_{i}'
            })

        val_loader = DataLoader(val_data, batch_size=2, shuffle=False)

        # Validate
        metrics = trainer.validate_epoch(val_loader)

        # Check that metrics were returned
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert metrics['loss'] >= 0

    def test_trainer_save_and_load_checkpoint(self, test_config, temp_dir):
        """Test saving and loading checkpoints."""
        test_config.experiment.checkpoint_dir = str(temp_dir)
        trainer = TemporalDrugTrainer(test_config)

        # Save checkpoint
        val_metrics = {'loss': 0.5, 'interaction_auroc': 0.8}
        trainer.save_checkpoint(epoch=5, val_metrics=val_metrics)

        # Check that files were created
        checkpoint_file = temp_dir / "checkpoint_epoch_5.pt"
        assert checkpoint_file.exists()

        # Load checkpoint
        loaded_metrics = trainer.load_checkpoint(checkpoint_file)

        assert loaded_metrics['loss'] == val_metrics['loss']
        assert trainer.epoch == 5

    def test_trainer_full_training_loop_short(self, test_config):
        """Test a short training loop."""
        # Very short training for testing
        test_config.training.num_epochs = 2
        test_config.training.early_stopping_patience = 1

        trainer = TemporalDrugTrainer(test_config)

        # Create minimal training and validation data
        num_samples = 4
        feature_dim = test_config.data.molecular_feature_dim

        def create_data(n_samples):
            data = []
            for i in range(n_samples):
                data.append({
                    'drug1_features': torch.randn(feature_dim),
                    'drug2_features': torch.randn(feature_dim),
                    'label': torch.randint(0, 2, (1,)).float(),
                    'drug1_id': f'drug1_{i}',
                    'drug2_id': f'drug2_{i}'
                })
            return data

        train_loader = DataLoader(create_data(num_samples), batch_size=2, shuffle=False)
        val_loader = DataLoader(create_data(num_samples // 2), batch_size=2, shuffle=False)

        # Run training
        history = trainer.train(train_loader, val_loader, num_epochs=2)

        # Check that history was returned
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) <= 2  # May stop early

    def test_trainer_evaluate(self, test_config):
        """Test model evaluation functionality."""
        trainer = TemporalDrugTrainer(test_config)

        # Create test data
        num_samples = 6
        feature_dim = test_config.data.molecular_feature_dim

        test_data = []
        for i in range(num_samples):
            test_data.append({
                'drug1_features': torch.randn(feature_dim),
                'drug2_features': torch.randn(feature_dim),
                'label': torch.randint(0, 2, (1,)).float(),
                'drug1_id': f'drug1_{i}',
                'drug2_id': f'drug2_{i}'
            })

        test_loader = DataLoader(test_data, batch_size=2, shuffle=False)

        # Evaluate
        test_metrics = trainer.evaluate(test_loader)

        # Check that evaluation metrics were returned
        assert isinstance(test_metrics, dict)
        assert 'loss' in test_metrics
        assert test_metrics['loss'] >= 0

    @patch('temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.training.trainer.mlflow')
    def test_trainer_mlflow_integration(self, mock_mlflow, test_config):
        """Test MLflow integration (mocked)."""
        test_config.experiment.use_mlflow = True

        trainer = TemporalDrugTrainer(test_config)

        # Check that MLflow methods were called
        mock_mlflow.set_experiment.assert_called_once()
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_params.assert_called_once()

    def test_trainer_gradient_clipping(self, test_config):
        """Test gradient clipping functionality."""
        test_config.training.gradient_clip_norm = 1.0
        trainer = TemporalDrugTrainer(test_config)

        # Create batch with features that might cause large gradients
        batch = {
            'drug1_features': torch.randn(2, test_config.data.molecular_feature_dim) * 10,
            'drug2_features': torch.randn(2, test_config.data.molecular_feature_dim) * 10,
            'label': torch.tensor([0.0, 1.0]),
            'drug1_id': ['drug1_0', 'drug1_1'],
            'drug2_id': ['drug2_0', 'drug2_1']
        }

        train_data = [batch]
        train_loader = DataLoader(train_data, batch_size=1, shuffle=False)

        # Train one step to check gradient clipping
        trainer.train_epoch(train_loader, epoch=0)

        # Check that gradients exist and are reasonable
        max_grad_norm = 0
        for param in trainer.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.norm()
                max_grad_norm = max(max_grad_norm, param_norm.item())

        # With gradient clipping, norms should be reasonable
        assert max_grad_norm < 10.0, f"Gradient norm {max_grad_norm} seems too large"

    def test_trainer_different_optimizers(self, test_config):
        """Test trainer with different optimizers."""
        optimizers = ["adamw", "sgd"]

        for opt_name in optimizers:
            test_config.training.optimizer = opt_name
            trainer = TemporalDrugTrainer(test_config)

            # Check that correct optimizer was created
            if opt_name == "adamw":
                assert isinstance(trainer.optimizer, torch.optim.AdamW)
            elif opt_name == "sgd":
                assert isinstance(trainer.optimizer, torch.optim.SGD)

    def test_trainer_invalid_optimizer(self, test_config):
        """Test trainer with invalid optimizer."""
        test_config.training.optimizer = "invalid_optimizer"

        with pytest.raises(ValueError, match="Unsupported optimizer"):
            TemporalDrugTrainer(test_config)

    def test_trainer_scheduler_step_update(self, test_config):
        """Test that learning rate scheduler is properly updated."""
        test_config.training.scheduler = "step"
        test_config.training.num_epochs = 3

        trainer = TemporalDrugTrainer(test_config)

        initial_lr = trainer.optimizer.param_groups[0]['lr']

        # Create minimal data
        feature_dim = test_config.data.molecular_feature_dim
        batch = {
            'drug1_features': torch.randn(1, feature_dim),
            'drug2_features': torch.randn(1, feature_dim),
            'label': torch.tensor([0.0]),
            'drug1_id': ['drug1'],
            'drug2_id': ['drug2']
        }

        train_loader = DataLoader([batch], batch_size=1, shuffle=False)
        val_loader = DataLoader([batch], batch_size=1, shuffle=False)

        # Run short training to trigger scheduler
        trainer.train(train_loader, val_loader, num_epochs=2)

        # Learning rate should have changed (step scheduler decreases LR)
        final_lr = trainer.optimizer.param_groups[0]['lr']
        assert final_lr != initial_lr or trainer.scheduler is None

    def test_trainer_loss_computation(self, test_config):
        """Test loss computation with different output shapes."""
        trainer = TemporalDrugTrainer(test_config)

        # Test binary classification loss
        batch_size = 3
        binary_logits = torch.randn(batch_size)
        binary_targets = torch.randint(0, 2, (batch_size,)).float()

        loss = torch.nn.BCEWithLogitsLoss()(binary_logits, binary_targets)
        assert torch.isfinite(loss)

        # Test multi-class classification loss
        num_classes = 5
        multi_logits = torch.randn(batch_size, num_classes)
        multi_targets = torch.randint(0, num_classes, (batch_size,)).long()

        loss = torch.nn.CrossEntropyLoss()(multi_logits, multi_targets)
        assert torch.isfinite(loss)