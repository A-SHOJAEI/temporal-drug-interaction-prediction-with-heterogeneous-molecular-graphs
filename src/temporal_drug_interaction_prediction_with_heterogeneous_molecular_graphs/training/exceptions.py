"""Custom exceptions for training pipeline."""

from typing import Any, Optional


class TrainingError(Exception):
    """Base exception for training-related errors.

    This serves as the base class for all training-specific exceptions,
    providing a common interface for error handling in the training pipeline.
    """

    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        """Initialize training error.

        Args:
            message: Human-readable error message.
            details: Additional error details or context.
        """
        super().__init__(message)
        self.message = message
        self.details = details


class ModelInitializationError(TrainingError):
    """Raised when model initialization fails.

    This exception is thrown when there are issues creating or initializing
    the model, optimizer, scheduler, or other training components.
    """
    pass


class BatchProcessingError(TrainingError):
    """Raised when batch processing fails during training or validation.

    This exception is thrown when there are issues processing individual
    batches, such as tensor shape mismatches, device errors, or forward
    pass failures.
    """

    def __init__(
        self,
        message: str,
        batch_idx: Optional[int] = None,
        epoch: Optional[int] = None,
        details: Optional[Any] = None
    ) -> None:
        """Initialize batch processing error.

        Args:
            message: Human-readable error message.
            batch_idx: Index of the problematic batch.
            epoch: Epoch number when error occurred.
            details: Additional error details.
        """
        super().__init__(message, details)
        self.batch_idx = batch_idx
        self.epoch = epoch


class CheckpointError(TrainingError):
    """Raised when checkpoint operations fail.

    This exception is thrown when there are issues saving or loading
    model checkpoints, including file I/O errors and state dict mismatches.
    """
    pass


class MLflowTrackingError(TrainingError):
    """Raised when MLflow experiment tracking fails.

    This exception is thrown when there are issues with MLflow operations,
    such as logging metrics, parameters, or artifacts.
    """
    pass


class ValidationError(TrainingError):
    """Raised when validation fails due to configuration or data issues.

    This exception is thrown when validation operations encounter
    configuration problems, data format issues, or metric computation errors.
    """
    pass


class EarlyStoppingError(TrainingError):
    """Raised when early stopping mechanism encounters issues.

    This exception is thrown when the early stopping callback fails
    to save or restore model weights properly.
    """
    pass