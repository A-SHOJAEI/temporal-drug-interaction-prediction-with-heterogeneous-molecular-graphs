"""Tests for model components and functionality."""

import pytest
import torch
import numpy as np

from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.models.model import (
    TemporalDrugInteractionGNN,
    TemporalAttentionModule,
    HeterogeneousGATLayer,
    MetabolitePathwayPredictor,
    DrugInteractionPredictor,
    PositionalEncoding
)
from torch_geometric.data import HeteroData


class TestPositionalEncoding:
    """Test positional encoding functionality."""

    def test_positional_encoding_initialization(self):
        """Test positional encoding initialization."""
        d_model = 64
        max_len = 100

        pos_enc = PositionalEncoding(d_model, max_len)

        assert pos_enc.pe.size() == (1, max_len, d_model)
        pytest.assert_tensor_finite(pos_enc.pe, "positional_encoding")

    def test_positional_encoding_forward(self):
        """Test positional encoding forward pass."""
        d_model = 64
        seq_len = 20
        batch_size = 4

        pos_enc = PositionalEncoding(d_model, max_len=100)

        x = torch.randn(batch_size, seq_len, d_model)
        output = pos_enc(x)

        pytest.assert_tensor_shape(output, (batch_size, seq_len, d_model), "pos_enc_output")
        pytest.assert_tensor_finite(output, "pos_enc_output")

        # Check that output is different from input (encoding was added)
        assert not torch.equal(x, output)

    def test_positional_encoding_different_lengths(self):
        """Test positional encoding with different sequence lengths."""
        d_model = 32
        pos_enc = PositionalEncoding(d_model, max_len=50)

        for seq_len in [5, 10, 25, 40]:
            x = torch.randn(2, seq_len, d_model)
            output = pos_enc(x)
            pytest.assert_tensor_shape(output, (2, seq_len, d_model), f"pos_enc_output_len_{seq_len}")


class TestTemporalAttentionModule:
    """Test temporal attention module functionality."""

    def test_temporal_attention_initialization(self):
        """Test temporal attention module initialization."""
        hidden_dim = 64
        num_heads = 4
        max_time_steps = 50

        attention = TemporalAttentionModule(
            hidden_dim, num_heads, max_time_steps=max_time_steps
        )

        assert attention.hidden_dim == hidden_dim
        assert attention.num_heads == num_heads
        assert attention.head_dim == hidden_dim // num_heads

    def test_temporal_attention_forward_basic(self):
        """Test basic temporal attention forward pass."""
        hidden_dim = 64
        num_heads = 4
        batch_size = 2
        seq_len = 10

        attention = TemporalAttentionModule(hidden_dim, num_heads)

        x = torch.randn(batch_size, seq_len, hidden_dim)
        output = attention(x)

        pytest.assert_tensor_shape(output, (batch_size, seq_len, hidden_dim), "attention_output")
        pytest.assert_tensor_finite(output, "attention_output")

    def test_temporal_attention_with_temporal_features(self):
        """Test temporal attention with temporal features."""
        hidden_dim = 32
        num_heads = 2
        batch_size = 3
        seq_len = 8

        attention = TemporalAttentionModule(hidden_dim, num_heads)

        x = torch.randn(batch_size, seq_len, hidden_dim)
        temporal_features = torch.randn(batch_size, seq_len, 5)

        output = attention(x, temporal_features)

        pytest.assert_tensor_shape(output, (batch_size, seq_len, hidden_dim), "attention_output_temporal")
        pytest.assert_tensor_finite(output, "attention_output_temporal")

    def test_temporal_attention_with_mask(self):
        """Test temporal attention with time mask."""
        hidden_dim = 64
        num_heads = 4
        batch_size = 2
        seq_len = 12

        attention = TemporalAttentionModule(hidden_dim, num_heads)

        x = torch.randn(batch_size, seq_len, hidden_dim)
        # Create mask (1 for valid positions, 0 for invalid)
        time_mask = torch.ones(batch_size, seq_len)
        time_mask[:, 8:] = 0  # Mask last 4 positions

        output = attention(x, time_mask=time_mask)

        pytest.assert_tensor_shape(output, (batch_size, seq_len, hidden_dim), "attention_output_masked")
        pytest.assert_tensor_finite(output, "attention_output_masked")


class TestMetabolitePathwayPredictor:
    """Test metabolite pathway predictor functionality."""

    def test_metabolite_pathway_predictor_initialization(self):
        """Test metabolite pathway predictor initialization."""
        drug_dim = 128
        metabolite_dim = 64
        pathway_dim = 32
        num_pathways = 8

        predictor = MetabolitePathwayPredictor(
            drug_dim, metabolite_dim, pathway_dim, num_pathways
        )

        assert predictor.num_pathways == num_pathways

    def test_metabolite_pathway_predictor_forward(self):
        """Test metabolite pathway predictor forward pass."""
        drug_dim = 64
        metabolite_dim = 32
        pathway_dim = 16
        num_pathways = 5
        batch_size = 3

        predictor = MetabolitePathwayPredictor(
            drug_dim, metabolite_dim, pathway_dim, num_pathways
        )

        drug_features = torch.randn(batch_size, drug_dim)
        pathway_probs, metabolite_features, temporal_metabolites = predictor(drug_features)

        # Check output shapes
        pytest.assert_tensor_shape(pathway_probs, (batch_size, num_pathways), "pathway_probs")
        pytest.assert_tensor_shape(metabolite_features, (batch_size, metabolite_dim), "metabolite_features")
        pytest.assert_tensor_shape(temporal_metabolites, (batch_size, metabolite_dim), "temporal_metabolites")

        # Check pathway probabilities sum to 1
        pathway_sums = torch.sum(pathway_probs, dim=1)
        assert torch.allclose(pathway_sums, torch.ones_like(pathway_sums), atol=1e-5)

        # Check finite values
        pytest.assert_tensor_finite(pathway_probs, "pathway_probs")
        pytest.assert_tensor_finite(metabolite_features, "metabolite_features")

    def test_metabolite_pathway_predictor_temporal(self):
        """Test metabolite pathway predictor with temporal dynamics."""
        drug_dim = 64
        metabolite_dim = 32
        batch_size = 2
        time_steps = 10

        predictor = MetabolitePathwayPredictor(drug_dim, metabolite_dim)

        drug_features = torch.randn(batch_size, drug_dim)
        time_steps_tensor = torch.linspace(0, 24, time_steps).unsqueeze(0).repeat(batch_size, 1)

        pathway_probs, metabolite_features, temporal_metabolites = predictor(
            drug_features, time_steps_tensor
        )

        # Check temporal output shape
        pytest.assert_tensor_shape(temporal_metabolites, (batch_size, time_steps, metabolite_dim), "temporal_metabolites")
        pytest.assert_tensor_finite(temporal_metabolites, "temporal_metabolites")


class TestDrugInteractionPredictor:
    """Test drug interaction predictor functionality."""

    def test_drug_interaction_predictor_initialization(self):
        """Test drug interaction predictor initialization."""
        drug_dim = 128
        metabolite_dim = 64
        target_dim = 64
        hidden_dim = 256

        predictor = DrugInteractionPredictor(
            drug_dim, metabolite_dim, target_dim, hidden_dim
        )

        assert predictor.drug_proj.in_features == drug_dim
        assert predictor.metabolite_proj.in_features == metabolite_dim
        assert predictor.target_proj.in_features == target_dim

    def test_drug_interaction_predictor_forward(self):
        """Test drug interaction predictor forward pass."""
        drug_dim = 64
        metabolite_dim = 32
        target_dim = 32
        hidden_dim = 128
        num_interaction_types = 3
        batch_size = 4

        predictor = DrugInteractionPredictor(
            drug_dim, metabolite_dim, target_dim, hidden_dim, num_interaction_types
        )

        # Create input features
        drug1_features = torch.randn(batch_size, drug_dim)
        drug2_features = torch.randn(batch_size, drug_dim)
        metabolite1_features = torch.randn(batch_size, metabolite_dim)
        metabolite2_features = torch.randn(batch_size, metabolite_dim)
        target1_features = torch.randn(batch_size, target_dim)
        target2_features = torch.randn(batch_size, target_dim)

        interaction_logits = predictor(
            drug1_features, drug2_features,
            metabolite1_features, metabolite2_features,
            target1_features, target2_features
        )

        pytest.assert_tensor_shape(interaction_logits, (batch_size, num_interaction_types), "interaction_logits")
        pytest.assert_tensor_finite(interaction_logits, "interaction_logits")


class TestHeterogeneousGATLayer:
    """Test heterogeneous GAT layer functionality."""

    def test_heterogeneous_gat_layer_initialization(self):
        """Test heterogeneous GAT layer initialization."""
        in_dim = {'drug': 128, 'metabolite': 64, 'target': 64}
        out_dim = 256
        edge_types = [('drug', 'interacts', 'drug'), ('drug', 'metabolizes', 'metabolite')]
        num_heads = 4

        gat_layer = HeterogeneousGATLayer(in_dim, out_dim, edge_types, num_heads)

        assert gat_layer.out_dim == out_dim
        assert gat_layer.num_heads == num_heads

    def test_heterogeneous_gat_layer_forward(self, sample_hetero_graph):
        """Test heterogeneous GAT layer forward pass."""
        in_dim = {'drug': 265, 'metabolite': 64, 'target': 64}  # Match fixture dimensions
        out_dim = 128
        edge_types = list(sample_hetero_graph.edge_index_dict.keys())

        gat_layer = HeterogeneousGATLayer(in_dim, out_dim, edge_types, num_heads=2)

        # Forward pass
        output_dict = gat_layer(
            sample_hetero_graph.x_dict,
            sample_hetero_graph.edge_index_dict
        )

        # Check outputs
        for node_type in sample_hetero_graph.x_dict.keys():
            assert node_type in output_dict
            pytest.assert_tensor_shape(
                output_dict[node_type],
                (sample_hetero_graph.x_dict[node_type].size(0), out_dim),
                f"{node_type}_output"
            )
            pytest.assert_tensor_finite(output_dict[node_type], f"{node_type}_output")


class TestTemporalDrugInteractionGNN:
    """Test complete temporal drug interaction GNN model."""

    def test_model_initialization(self, test_config):
        """Test model initialization."""
        node_type_dims = {
            'drug': test_config.data.molecular_feature_dim,
            'metabolite': 64,
            'target': 64
        }

        edge_types = [
            ('drug', 'metabolizes_to', 'metabolite'),
            ('metabolite', 'metabolized_from', 'drug'),
            ('drug', 'targets', 'target'),
            ('target', 'targeted_by', 'drug')
        ]

        model = TemporalDrugInteractionGNN(
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

        assert model.hidden_dim == test_config.model.hidden_dim
        assert model.num_layers == test_config.model.num_layers
        assert len(model.gat_layers) == test_config.model.num_layers

        # Check parameter count
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 1000, "Model should have reasonable number of parameters"

    def test_model_forward_pass(self, test_model, sample_hetero_graph):
        """Test model forward pass."""
        test_model.eval()

        with torch.no_grad():
            outputs = test_model(sample_hetero_graph)

        # Check required outputs
        required_keys = ['interaction_logits', 'pathway_probs', 'metabolite_features', 'temporal_metabolites']
        for key in required_keys:
            assert key in outputs, f"Missing output: {key}"

        # Check output shapes and values
        pytest.assert_tensor_finite(outputs['interaction_logits'], "interaction_logits")
        pytest.assert_tensor_finite(outputs['pathway_probs'], "pathway_probs")
        pytest.assert_tensor_finite(outputs['metabolite_features'], "metabolite_features")

    def test_model_forward_with_attention(self, test_model, sample_hetero_graph):
        """Test model forward pass with attention weights."""
        test_model.eval()

        with torch.no_grad():
            outputs = test_model(sample_hetero_graph, return_attention=True)

        assert 'attention_weights' in outputs, "Should return attention weights when requested"

    def test_model_training_mode(self, test_model, sample_hetero_graph):
        """Test model in training mode."""
        test_model.train()

        # Forward pass should work in training mode
        outputs = test_model(sample_hetero_graph)

        # Check gradients can be computed
        loss = outputs['interaction_logits'].sum()
        loss.backward()

        # Check that gradients were computed
        for param in test_model.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Gradients should be computed"

    def test_model_compute_loss(self, test_model, sample_hetero_graph):
        """Test model loss computation."""
        outputs = test_model(sample_hetero_graph)

        # Create dummy targets
        targets = {
            'interaction_labels': torch.randint(0, 2, (outputs['interaction_logits'].size(0),)).float(),
            'pathway_labels': torch.randint(0, 10, (outputs['pathway_probs'].size(0),)).long(),
            'temporal_targets': torch.randn_like(outputs['temporal_metabolites'])
        }

        losses = test_model.compute_loss(outputs, targets)

        # Check loss components
        assert 'total_loss' in losses
        assert isinstance(losses['total_loss'], torch.Tensor)
        assert losses['total_loss'].item() >= 0, "Loss should be non-negative"
        pytest.assert_tensor_finite(losses['total_loss'], "total_loss")

    def test_model_predict_interactions(self, test_model):
        """Test model interaction prediction functionality."""
        test_model.eval()

        drug_pairs = [("aspirin", "warfarin"), ("caffeine", "diphenhydramine")]
        drug_features = {
            "aspirin": torch.randn(265),
            "warfarin": torch.randn(265),
            "caffeine": torch.randn(265),
            "diphenhydramine": torch.randn(265)
        }

        with torch.no_grad():
            predictions = test_model.predict_interactions(drug_pairs, drug_features)

        pytest.assert_tensor_shape(predictions, (len(drug_pairs),), "interaction_predictions")
        pytest.assert_tensor_range(predictions, 0.0, 1.0, "interaction_predictions")

    def test_model_different_batch_sizes(self, test_model):
        """Test model with different batch sizes."""
        test_model.eval()

        for batch_size in [1, 2, 4]:
            # Create sample data with different batch sizes
            hetero_data = HeteroData()
            hetero_data['drug'].x = torch.randn(batch_size * 2, 64)
            hetero_data['metabolite'].x = torch.randn(batch_size * 2, 64)
            hetero_data['target'].x = torch.randn(batch_size * 2, 64)

            # Add minimal edges
            drug_indices = torch.arange(batch_size * 2)
            hetero_data[('drug', 'metabolizes_to', 'metabolite')].edge_index = torch.stack([
                drug_indices, drug_indices
            ])

            with torch.no_grad():
                outputs = test_model(hetero_data)

            assert outputs is not None, f"Forward pass failed for batch size {batch_size}"

    def test_model_parameter_updates(self, test_model, sample_hetero_graph):
        """Test that model parameters are updated during training."""
        test_model.train()

        # Get initial parameters
        initial_params = {}
        for name, param in test_model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.clone()

        # Forward and backward pass
        outputs = test_model(sample_hetero_graph)
        loss = outputs['interaction_logits'].sum()
        loss.backward()

        # Update parameters (simulate optimizer step)
        with torch.no_grad():
            for param in test_model.parameters():
                if param.requires_grad and param.grad is not None:
                    param.data -= 0.01 * param.grad  # Simple SGD update

        # Check that parameters were updated
        updated_count = 0
        for name, param in test_model.named_parameters():
            if param.requires_grad and name in initial_params:
                if not torch.equal(param, initial_params[name]):
                    updated_count += 1

        assert updated_count > 0, "At least some parameters should be updated"

    def test_model_reproducibility(self, test_config):
        """Test model reproducibility with same seed."""
        torch.manual_seed(42)

        node_type_dims = {
            'drug': test_config.data.molecular_feature_dim,
            'metabolite': 64,
            'target': 64
        }

        edge_types = [('drug', 'metabolizes_to', 'metabolite')]

        model1 = TemporalDrugInteractionGNN(
            node_type_dims=node_type_dims,
            edge_types=edge_types,
            hidden_dim=test_config.model.hidden_dim,
            num_layers=test_config.model.num_layers,
            num_heads=test_config.model.num_heads,
            dropout=0.0,  # Disable dropout for reproducibility
            temporal_attention_dim=test_config.model.temporal_attention_dim,
            metabolite_pathway_dim=test_config.model.metabolite_pathway_dim,
            max_time_steps=test_config.model.max_time_steps,
        )

        torch.manual_seed(42)

        model2 = TemporalDrugInteractionGNN(
            node_type_dims=node_type_dims,
            edge_types=edge_types,
            hidden_dim=test_config.model.hidden_dim,
            num_layers=test_config.model.num_layers,
            num_heads=test_config.model.num_heads,
            dropout=0.0,  # Disable dropout for reproducibility
            temporal_attention_dim=test_config.model.temporal_attention_dim,
            metabolite_pathway_dim=test_config.model.metabolite_pathway_dim,
            max_time_steps=test_config.model.max_time_steps,
        )

        # Check that models have identical parameters
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert name1 == name2
            assert torch.equal(param1, param2), f"Parameters {name1} differ between models"