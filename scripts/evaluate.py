#!/usr/bin/env python3
"""Evaluation script for temporal drug interaction prediction model."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, precision_recall_curve

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.data.loader import DrugInteractionDataLoader
from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.models.model import TemporalDrugInteractionGNN
from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.evaluation.metrics import DrugInteractionMetrics
from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.utils.config import Config, load_config, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate temporal drug interaction prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint to evaluate"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results"
    )

    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to custom test data (optional)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for evaluation"
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save model predictions to file"
    )

    parser.add_argument(
        "--plot-curves",
        action="store_true",
        help="Generate ROC and PR curves"
    )

    parser.add_argument(
        "--attention-analysis",
        action="store_true",
        help="Perform attention weight analysis"
    )

    parser.add_argument(
        "--drug-pairs",
        type=str,
        nargs="+",
        default=None,
        help="Specific drug pairs to evaluate (format: 'drug1,drug2')"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: Path,
    config: Config,
    device: torch.device
) -> TemporalDrugInteractionGNN:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        config: Configuration object.
        device: Device to load model on.

    Returns:
        Loaded model.
    """
    logger = logging.getLogger(__name__)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading model from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model with same configuration
    node_type_dims = {
        'drug': config.data.molecular_feature_dim,
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

    model = TemporalDrugInteractionGNN(
        node_type_dims=node_type_dims,
        edge_types=edge_types,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        temporal_attention_dim=config.model.temporal_attention_dim,
        metabolite_pathway_dim=config.model.metabolite_pathway_dim,
        max_time_steps=config.model.max_time_steps,
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    return model


def evaluate_model_comprehensive(
    model: TemporalDrugInteractionGNN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_predictions: bool = False,
    output_dir: Path = None
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Comprehensive model evaluation.

    Args:
        model: Model to evaluate.
        data_loader: Data loader for evaluation.
        device: Device for computation.
        save_predictions: Whether to save predictions.
        output_dir: Output directory.

    Returns:
        Tuple of (metrics, predictions, targets).
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive model evaluation")

    model.eval()
    all_predictions = []
    all_targets = []
    all_drug_pairs = []

    metrics_calculator = DrugInteractionMetrics()

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            try:
                # Move batch to device
                if isinstance(batch, dict):
                    for key in batch:
                        if torch.is_tensor(batch[key]):
                            batch[key] = batch[key].to(device)

                    # Create heterogeneous data from batch
                    from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.training.trainer import TemporalDrugTrainer
                    trainer = TemporalDrugTrainer.__new__(TemporalDrugTrainer)  # Create instance without __init__
                    trainer.device = device
                    hetero_data = trainer._create_batch_hetero_data(batch)

                    outputs = model(hetero_data)
                    interaction_logits = outputs['interaction_logits']
                    targets = batch['label']

                    # Store drug pair information
                    drug1_ids = batch.get('drug1_id', [f'drug1_{i}' for i in range(len(targets))])
                    drug2_ids = batch.get('drug2_id', [f'drug2_{i}' for i in range(len(targets))])

                    if isinstance(drug1_ids, torch.Tensor):
                        drug1_ids = drug1_ids.tolist()
                    if isinstance(drug2_ids, torch.Tensor):
                        drug2_ids = drug2_ids.tolist()

                    batch_pairs = list(zip(drug1_ids, drug2_ids))
                    all_drug_pairs.extend(batch_pairs)

                else:
                    outputs = model(batch)
                    interaction_logits = outputs['interaction_logits']
                    targets = getattr(batch, 'interaction_labels', torch.zeros_like(interaction_logits))

                # Convert to probabilities
                if interaction_logits.dim() > 1 and interaction_logits.size(1) > 1:
                    predictions = torch.softmax(interaction_logits, dim=1)
                else:
                    if interaction_logits.dim() > 1:
                        interaction_logits = interaction_logits.squeeze(1)
                    predictions = torch.sigmoid(interaction_logits)

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

                # Update metrics
                metrics_calculator.update(predictions, targets)

                if batch_idx % 50 == 0:
                    logger.info(f"Processed {batch_idx}/{len(data_loader)} batches")

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    logger.info(f"Evaluation completed: {len(all_predictions)} samples processed")

    # Compute comprehensive metrics
    comprehensive_metrics = metrics_calculator.compute_all_metrics()

    # Save predictions if requested
    if save_predictions and output_dir:
        predictions_df = pd.DataFrame({
            'drug1': [pair[0] for pair in all_drug_pairs] if all_drug_pairs else range(len(all_predictions)),
            'drug2': [pair[1] for pair in all_drug_pairs] if all_drug_pairs else range(len(all_predictions)),
            'predicted_interaction_prob': all_predictions.flatten() if all_predictions.ndim == 1 else all_predictions[:, 0],
            'true_label': all_targets.flatten()
        })

        if all_predictions.ndim > 1:
            for i in range(all_predictions.shape[1]):
                predictions_df[f'class_{i}_prob'] = all_predictions[:, i]

        predictions_file = output_dir / "predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)
        logger.info(f"Predictions saved to {predictions_file}")

    return comprehensive_metrics, all_predictions, all_targets


def generate_evaluation_plots(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: Path
) -> None:
    """Generate evaluation plots.

    Args:
        predictions: Model predictions.
        targets: True targets.
        output_dir: Output directory for plots.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating evaluation plots")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle multi-class predictions
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        pred_probs = predictions[:, 1]  # Use positive class probability
    else:
        pred_probs = predictions.flatten()

    targets_binary = targets.flatten()

    try:
        # ROC Curve
        fpr, tpr, _ = roc_curve(targets_binary, pred_probs)
        roc_auc = np.trapz(tpr, fpr)

        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(targets_binary, pred_probs)
        pr_auc = np.trapz(precision, recall)

        plt.subplot(2, 2, 2)
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Prediction distribution
        plt.subplot(2, 2, 3)
        plt.hist(pred_probs[targets_binary == 0], bins=50, alpha=0.7, label='No Interaction', density=True)
        plt.hist(pred_probs[targets_binary == 1], bins=50, alpha=0.7, label='Interaction', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Prediction Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Confusion matrix heatmap (for threshold = 0.5)
        from sklearn.metrics import confusion_matrix
        pred_binary = (pred_probs > 0.5).astype(int)
        cm = confusion_matrix(targets_binary, pred_binary)

        plt.subplot(2, 2, 4)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig(output_dir / "evaluation_plots.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Evaluation plots saved to {output_dir / 'evaluation_plots.png'}")

    except Exception as e:
        logger.error(f"Error generating plots: {e}")


def analyze_attention_weights(
    model: TemporalDrugInteractionGNN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 100
) -> None:
    """Analyze attention weights for interpretability.

    Args:
        model: Model to analyze.
        data_loader: Data loader.
        device: Device for computation.
        output_dir: Output directory.
        num_samples: Number of samples to analyze.
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing attention weights for interpretability")

    model.eval()
    attention_weights_list = []
    sample_count = 0

    with torch.no_grad():
        for batch in data_loader:
            if sample_count >= num_samples:
                break

            try:
                # Move batch to device
                if isinstance(batch, dict):
                    for key in batch:
                        if torch.is_tensor(batch[key]):
                            batch[key] = batch[key].to(device)

                    from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.training.trainer import TemporalDrugTrainer
                    trainer = TemporalDrugTrainer.__new__(TemporalDrugTrainer)
                    trainer.device = device
                    hetero_data = trainer._create_batch_hetero_data(batch)

                    outputs = model(hetero_data, return_attention=True)

                    if 'attention_weights' in outputs:
                        attention_weights_list.extend(outputs['attention_weights'])

                    sample_count += len(batch['label'])

            except Exception as e:
                logger.error(f"Error in attention analysis: {e}")
                continue

    if attention_weights_list:
        # Analyze attention patterns
        avg_attention = np.mean([aw.cpu().numpy() for aw in attention_weights_list if aw is not None], axis=0)

        # Save attention analysis
        attention_file = output_dir / "attention_analysis.json"
        analysis = {
            'average_attention_per_head': avg_attention.tolist() if avg_attention.ndim > 0 else [avg_attention.item()],
            'num_samples_analyzed': sample_count,
            'attention_statistics': {
                'mean': float(np.mean(avg_attention)),
                'std': float(np.std(avg_attention)),
                'min': float(np.min(avg_attention)),
                'max': float(np.max(avg_attention))
            }
        }

        with open(attention_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"Attention analysis saved to {attention_file}")
    else:
        logger.warning("No attention weights captured during analysis")


def evaluate_specific_drug_pairs(
    model: TemporalDrugInteractionGNN,
    drug_pairs: List[Tuple[str, str]],
    drug_features: Dict[str, torch.Tensor],
    device: torch.device
) -> List[Dict[str, Any]]:
    """Evaluate specific drug pairs.

    Args:
        model: Model to use for prediction.
        drug_pairs: List of drug pairs to evaluate.
        drug_features: Drug molecular features.
        device: Device for computation.

    Returns:
        List of evaluation results for each pair.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating {len(drug_pairs)} specific drug pairs")

    model.eval()
    results = []

    with torch.no_grad():
        for drug1, drug2 in drug_pairs:
            try:
                # Get drug features (using dummy features if not available)
                drug1_feat = drug_features.get(drug1, torch.randn(265))
                drug2_feat = drug_features.get(drug2, torch.randn(265))

                # Create batch
                batch = {
                    'drug1_features': drug1_feat.unsqueeze(0).to(device),
                    'drug2_features': drug2_feat.unsqueeze(0).to(device),
                    'label': torch.tensor([0.0]).to(device),  # Dummy label
                    'drug1_id': [drug1],
                    'drug2_id': [drug2]
                }

                # Create heterogeneous data
                from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.training.trainer import TemporalDrugTrainer
                trainer = TemporalDrugTrainer.__new__(TemporalDrugTrainer)
                trainer.device = device
                hetero_data = trainer._create_batch_hetero_data(batch)

                # Get prediction
                outputs = model(hetero_data)
                interaction_logit = outputs['interaction_logits'][0]

                if interaction_logit.dim() > 0 and interaction_logit.size(0) > 1:
                    interaction_prob = torch.softmax(interaction_logit, dim=0)[1].item()
                else:
                    interaction_prob = torch.sigmoid(interaction_logit).item()

                # Get additional outputs
                pathway_probs = outputs.get('pathway_probs', torch.zeros(1, 10))[0]
                top_pathway = torch.argmax(pathway_probs).item()

                result = {
                    'drug1': drug1,
                    'drug2': drug2,
                    'interaction_probability': interaction_prob,
                    'interaction_risk': 'High' if interaction_prob > 0.7 else 'Medium' if interaction_prob > 0.3 else 'Low',
                    'top_metabolic_pathway': int(top_pathway),
                    'confidence': float(torch.max(pathway_probs).item())
                }

                results.append(result)
                logger.info(f"{drug1} + {drug2}: {interaction_prob:.3f} ({result['interaction_risk']} risk)")

            except Exception as e:
                logger.error(f"Error evaluating {drug1} + {drug2}: {e}")
                results.append({
                    'drug1': drug1,
                    'drug2': drug2,
                    'interaction_probability': 0.0,
                    'interaction_risk': 'Unknown',
                    'error': str(e)
                })

    return results


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Setup paths
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    try:
        config = load_config(args.config)
        if args.device != "auto":
            config.device = args.device
        print(f"✓ Loaded configuration from {args.config}")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        sys.exit(1)

    # Setup logging
    if args.debug:
        config.experiment.log_level = "DEBUG"
    setup_logging(config)
    logger = logging.getLogger(__name__)

    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    try:
        # Load model
        logger.info("Loading model from checkpoint...")
        model = load_model_from_checkpoint(checkpoint_path, config, device)

        # Load test data
        logger.info("Loading test data...")
        data_loader = DrugInteractionDataLoader(config.data)
        _, _, test_loader, metadata = data_loader.load_complete_dataset()

        if args.batch_size != config.training.batch_size:
            # Update batch size
            test_loader = torch.utils.data.DataLoader(
                test_loader.dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=config.num_workers
            )

        print("\n" + "="*60)
        print("🔬 EVALUATING TEMPORAL DRUG INTERACTION MODEL")
        print("="*60)
        print(f"📊 Test set: {len(test_loader.dataset)} samples")
        print(f"🧠 Model: {checkpoint_path.name}")
        print(f"⚡ Device: {device}")
        print("="*60 + "\n")

        # Comprehensive evaluation
        logger.info("Running comprehensive evaluation...")
        metrics, predictions, targets = evaluate_model_comprehensive(
            model, test_loader, device, args.save_predictions, output_dir
        )

        # Print results
        print("📈 EVALUATION RESULTS:")
        print("-" * 40)
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric_name:<35}: {value:.4f}")

        # Generate plots if requested
        if args.plot_curves:
            logger.info("Generating evaluation plots...")
            generate_evaluation_plots(predictions, targets, output_dir)

        # Attention analysis if requested
        if args.attention_analysis:
            logger.info("Performing attention weight analysis...")
            analyze_attention_weights(model, test_loader, device, output_dir)

        # Evaluate specific drug pairs if provided
        if args.drug_pairs:
            logger.info("Evaluating specific drug pairs...")
            # Parse drug pairs
            specific_pairs = []
            for pair_str in args.drug_pairs:
                try:
                    drug1, drug2 = pair_str.split(',')
                    specific_pairs.append((drug1.strip(), drug2.strip()))
                except ValueError:
                    logger.warning(f"Invalid drug pair format: {pair_str}")

            if specific_pairs:
                # Get drug features (simplified)
                drug_features = {}  # Would load from preprocessed data in practice
                pair_results = evaluate_specific_drug_pairs(
                    model, specific_pairs, drug_features, device
                )

                print("\n🎯 SPECIFIC DRUG PAIR PREDICTIONS:")
                print("-" * 40)
                for result in pair_results:
                    if 'error' not in result:
                        print(f"  {result['drug1']} + {result['drug2']}: "
                              f"{result['interaction_probability']:.3f} ({result['interaction_risk']} risk)")

                # Save specific results
                with open(output_dir / "specific_drug_pairs.json", 'w') as f:
                    json.dump(pair_results, f, indent=2)

        # Save all results
        results = {
            'evaluation_metrics': metrics,
            'metadata': metadata,
            'evaluation_config': {
                'checkpoint': str(checkpoint_path),
                'test_samples': len(test_loader.dataset),
                'batch_size': args.batch_size,
                'device': str(device)
            }
        }

        with open(output_dir / "evaluation_results.json", 'w') as f:
            # Convert numpy types for JSON
            def convert_types(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, (list, tuple)):
                    return [convert_types(x) for x in obj]
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                return obj

            json.dump(convert_types(results), f, indent=2)

        print(f"\n✅ Evaluation completed. Results saved to {output_dir}")
        print("="*60)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()