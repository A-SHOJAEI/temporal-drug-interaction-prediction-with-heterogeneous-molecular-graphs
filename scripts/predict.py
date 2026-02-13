#!/usr/bin/env python3
"""Inference script for temporal drug interaction prediction."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.data.preprocessing import MolecularPreprocessor
from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.models.model import TemporalDrugInteractionGNN
from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.utils.config import Config, load_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict drug-drug interactions using trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to trained model checkpoint"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--drug1",
        type=str,
        default=None,
        help="SMILES string for first drug"
    )

    parser.add_argument(
        "--drug2",
        type=str,
        default=None,
        help="SMILES string for second drug"
    )

    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="JSON file with drug pairs (format: [{\"drug1\": \"SMILES1\", \"drug2\": \"SMILES2\"}])"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="predictions.json",
        help="Output file for predictions"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Interaction probability threshold for classification"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def load_model(checkpoint_path: Path, config: Config, device: torch.device) -> TemporalDrugInteractionGNN:
    """Load trained model from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

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

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def preprocess_smiles(smiles: str, preprocessor: MolecularPreprocessor) -> Optional[torch.Tensor]:
    """Convert SMILES to molecular features."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        features = preprocessor.extract_features(mol)
        return torch.tensor(features, dtype=torch.float32)
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None


def predict_interaction(
    model: TemporalDrugInteractionGNN,
    drug1_features: torch.Tensor,
    drug2_features: torch.Tensor,
    device: torch.device
) -> Dict[str, float]:
    """Predict interaction for a single drug pair."""
    model.eval()

    with torch.no_grad():
        batch = {
            'drug1_features': drug1_features.unsqueeze(0).to(device),
            'drug2_features': drug2_features.unsqueeze(0).to(device),
            'label': torch.tensor([0.0]).to(device),
            'drug1_id': ['drug1'],
            'drug2_id': ['drug2']
        }

        # Create heterogeneous data
        from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.training.trainer import TemporalDrugTrainer
        trainer = TemporalDrugTrainer.__new__(TemporalDrugTrainer)
        trainer.device = device
        hetero_data = trainer._create_batch_hetero_data(batch)

        outputs = model(hetero_data)
        interaction_logit = outputs['interaction_logits'][0]

        # Convert to probability
        if interaction_logit.dim() > 0 and interaction_logit.size(0) > 1:
            interaction_prob = torch.softmax(interaction_logit, dim=0)[1].item()
        else:
            interaction_prob = torch.sigmoid(interaction_logit).item()

        # Get pathway predictions
        pathway_probs = outputs.get('pathway_probs', torch.zeros(1, 10))[0]
        top_pathway = torch.argmax(pathway_probs).item()
        pathway_confidence = torch.max(pathway_probs).item()

        # Get temporal predictions
        temporal_preds = outputs.get('temporal_preds', torch.zeros(1, 1))[0]

        return {
            'interaction_probability': float(interaction_prob),
            'interaction_predicted': bool(interaction_prob > 0.5),
            'top_metabolic_pathway': int(top_pathway),
            'pathway_confidence': float(pathway_confidence),
            'temporal_score': float(temporal_preds.mean().item()) if temporal_preds.numel() > 0 else 0.0
        }


def main():
    """Main prediction function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    if args.device != "auto":
        config.device = args.device

    device = torch.device(config.device)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Load model
    checkpoint_path = Path(args.checkpoint)
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, config, device)
    print(f"Model loaded successfully on {device}")

    # Initialize preprocessor
    preprocessor = MolecularPreprocessor()

    # Collect drug pairs
    drug_pairs = []

    if args.drug1 and args.drug2:
        drug_pairs.append({
            'drug1_smiles': args.drug1,
            'drug2_smiles': args.drug2,
            'name': f"pair_1"
        })

    if args.input_file:
        with open(args.input_file, 'r') as f:
            input_data = json.load(f)
            for i, pair in enumerate(input_data):
                drug_pairs.append({
                    'drug1_smiles': pair['drug1'],
                    'drug2_smiles': pair['drug2'],
                    'name': pair.get('name', f"pair_{i+1}")
                })

    if not drug_pairs:
        # Use example drug pairs if none provided
        print("No drug pairs provided. Using example SMILES...")
        drug_pairs = [
            {
                'drug1_smiles': 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # Ibuprofen
                'drug2_smiles': 'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
                'name': 'Ibuprofen+Aspirin'
            },
            {
                'drug1_smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
                'drug2_smiles': 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # Ibuprofen
                'name': 'Caffeine+Ibuprofen'
            }
        ]

    # Make predictions
    print(f"\nPredicting interactions for {len(drug_pairs)} drug pairs...\n")
    results = []

    for pair_data in drug_pairs:
        drug1_smiles = pair_data['drug1_smiles']
        drug2_smiles = pair_data['drug2_smiles']
        name = pair_data['name']

        # Preprocess
        drug1_features = preprocess_smiles(drug1_smiles, preprocessor)
        drug2_features = preprocess_smiles(drug2_smiles, preprocessor)

        if drug1_features is None or drug2_features is None:
            print(f"❌ Failed to process {name}")
            results.append({
                'name': name,
                'drug1_smiles': drug1_smiles,
                'drug2_smiles': drug2_smiles,
                'error': 'Invalid SMILES'
            })
            continue

        # Predict
        try:
            prediction = predict_interaction(model, drug1_features, drug2_features, device)

            result = {
                'name': name,
                'drug1_smiles': drug1_smiles,
                'drug2_smiles': drug2_smiles,
                **prediction
            }
            results.append(result)

            # Print result
            status = "⚠️  INTERACTION" if prediction['interaction_predicted'] else "✅ NO INTERACTION"
            prob = prediction['interaction_probability']
            print(f"{status} | {name}")
            print(f"  Probability: {prob:.4f}")
            print(f"  Top pathway: {prediction['top_metabolic_pathway']} (confidence: {prediction['pathway_confidence']:.4f})")
            print(f"  Temporal score: {prediction['temporal_score']:.4f}")
            print()

        except Exception as e:
            print(f"❌ Error predicting {name}: {e}")
            results.append({
                'name': name,
                'drug1_smiles': drug1_smiles,
                'drug2_smiles': drug2_smiles,
                'error': str(e)
            })

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Predictions saved to {output_path}")

    # Summary statistics
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        avg_prob = np.mean([r['interaction_probability'] for r in valid_results])
        num_interactions = sum(1 for r in valid_results if r['interaction_predicted'])
        print(f"\n📊 Summary:")
        print(f"  Total pairs: {len(drug_pairs)}")
        print(f"  Valid predictions: {len(valid_results)}")
        print(f"  Predicted interactions: {num_interactions} ({100*num_interactions/len(valid_results):.1f}%)")
        print(f"  Average probability: {avg_prob:.4f}")


if __name__ == "__main__":
    main()
