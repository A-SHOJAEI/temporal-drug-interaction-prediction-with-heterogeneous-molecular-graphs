#!/usr/bin/env python3
"""Training script for temporal drug interaction prediction model."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.data.loader import DrugInteractionDataLoader
from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.training.trainer import TemporalDrugTrainer
from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.utils.config import Config, load_config, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train temporal drug interaction prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs"
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow experiment tracking"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run with small dataset"
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
        print(f"✓ Loaded configuration from {args.config}")
    except FileNotFoundError:
        print(f"✗ Configuration file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        sys.exit(1)

    # Override config with command line arguments
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.device != "auto":
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.no_mlflow:
        config.experiment.use_mlflow = False

    # Update output directories
    if args.output_dir:
        config.experiment.log_dir = str(Path(args.output_dir) / "logs")
        config.experiment.checkpoint_dir = str(Path(args.output_dir) / "checkpoints")

    # Setup logging
    if args.debug:
        config.experiment.log_level = "DEBUG"
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("Starting temporal drug interaction prediction training")
    logger.info(f"Configuration: {config.to_dict()}")

    try:
        # Initialize data loader
        logger.info("Initializing data loader...")
        data_loader = DrugInteractionDataLoader(config.data)

        # Load and preprocess data
        logger.info("Loading dataset...")
        train_loader, val_loader, test_loader, metadata = data_loader.load_complete_dataset()

        logger.info(f"Dataset loaded successfully:")
        logger.info(f"  - Training samples: {len(train_loader.dataset)}")
        logger.info(f"  - Validation samples: {len(val_loader.dataset)}")
        logger.info(f"  - Test samples: {len(test_loader.dataset)}")
        logger.info(f"  - Feature dimension: {metadata['feature_dim']}")
        logger.info(f"  - Temporal dimension: {metadata['temporal_dim']}")

        # For dry run, use smaller dataset
        if args.dry_run:
            logger.info("Performing dry run with reduced dataset")
            config.training.num_epochs = 2
            # Limit batches for dry run
            train_loader.dataset.drug_pairs = train_loader.dataset.drug_pairs[:100]
            train_loader.dataset.labels = train_loader.dataset.labels[:100]
            val_loader.dataset.drug_pairs = val_loader.dataset.drug_pairs[:50]
            val_loader.dataset.labels = val_loader.dataset.labels[:50]

        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = TemporalDrugTrainer(config)

        # Load checkpoint if specified
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            if checkpoint_path.exists():
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                trainer.load_checkpoint(checkpoint_path)
            else:
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                sys.exit(1)

        # Train the model
        logger.info("Starting training...")
        print("\n" + "="*60)
        print("🚀 TRAINING TEMPORAL DRUG INTERACTION PREDICTION MODEL")
        print("="*60)
        print(f"📊 Dataset: {metadata['num_molecules']} molecules, {metadata['num_interactions']} interactions")
        print(f"🔧 Model: {config.model.hidden_dim}D hidden, {config.model.num_layers} layers")
        print(f"⚡ Device: {config.device}")
        print(f"📈 Epochs: {config.training.num_epochs}")
        print(f"🎯 Target Metrics:")
        for metric, target in config.target_metrics.items():
            print(f"  • {metric}: {target:.3f}")
        print("="*60 + "\n")

        history = trainer.train(train_loader, val_loader)

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)

        # Print final results
        print("\n" + "="*60)
        print("🎯 TRAINING COMPLETED - FINAL RESULTS")
        print("="*60)

        # Compare with target metrics
        target_comparison = {}
        for metric_name, target_value in config.target_metrics.items():
            current_value = test_metrics.get(metric_name, 0.0)
            achieved = current_value >= target_value
            status = "✅ ACHIEVED" if achieved else "❌ NOT ACHIEVED"
            gap = current_value - target_value

            target_comparison[metric_name] = {
                'target': target_value,
                'achieved_value': current_value,
                'gap': gap,
                'status': status
            }

            print(f"📊 {metric_name}:")
            print(f"  Target: {target_value:.3f} | Achieved: {current_value:.3f} | Gap: {gap:+.3f} | {status}")

        # Overall success rate
        achieved_count = sum(1 for comp in target_comparison.values() if "✅" in comp['status'])
        success_rate = achieved_count / len(target_comparison) * 100

        print(f"\n🏆 OVERALL SUCCESS RATE: {success_rate:.1f}% ({achieved_count}/{len(target_comparison)} metrics achieved)")

        if success_rate >= 75:
            print("🎉 EXCELLENT PERFORMANCE! Most target metrics achieved.")
        elif success_rate >= 50:
            print("👍 GOOD PERFORMANCE! Several target metrics achieved.")
        else:
            print("⚠️  PERFORMANCE NEEDS IMPROVEMENT. Consider tuning hyperparameters.")

        print("="*60)

        # Save final configuration and results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Save config
        config.save(output_dir / "final_config.yaml")

        # Save results
        results = {
            'final_metrics': test_metrics,
            'target_comparison': target_comparison,
            'training_history': history,
            'metadata': metadata,
            'success_rate': success_rate
        }

        import json
        with open(output_dir / "training_results.json", 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_types(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, (list, tuple)):
                    return [convert_types(x) for x in obj]
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                return obj

            json.dump(convert_types(results), f, indent=2)

        logger.info(f"Training completed successfully. Results saved to {output_dir}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()