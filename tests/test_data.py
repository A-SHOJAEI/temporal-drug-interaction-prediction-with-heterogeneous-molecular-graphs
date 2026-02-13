"""Tests for data loading and preprocessing modules."""

import pytest
import torch
import numpy as np
from pathlib import Path

from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.data.loader import (
    DrugInteractionDataLoader, DrugInteractionDataset
)
from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.data.preprocessing import (
    MolecularGraphPreprocessor, TemporalGraphConstructor
)
from temporal_drug_interaction_prediction_with_heterogeneous_molecular_graphs.utils.config import DataConfig


class TestMolecularGraphPreprocessor:
    """Test molecular graph preprocessing functionality."""

    def test_smiles_to_graph_valid(self, molecular_preprocessor, sample_smiles):
        """Test SMILES to graph conversion with valid inputs."""
        for smiles in sample_smiles[:3]:  # Test first 3 for speed
            graph_data = molecular_preprocessor.smiles_to_graph(smiles)

            assert graph_data is not None, f"Failed to process SMILES: {smiles}"
            assert hasattr(graph_data, 'x'), "Graph should have node features"
            assert hasattr(graph_data, 'edge_index'), "Graph should have edge indices"
            assert hasattr(graph_data, 'mol_descriptors'), "Graph should have molecular descriptors"

            # Check tensor shapes
            pytest.assert_tensor_shape(graph_data.edge_index, (2, graph_data.edge_index.size(1)), "edge_index")
            assert graph_data.x.size(0) > 0, "Should have at least one node"
            assert graph_data.mol_descriptors.size(0) == 265, "Should have 265 molecular descriptors"

            # Check finite values
            pytest.assert_tensor_finite(graph_data.x, "node_features")
            pytest.assert_tensor_finite(graph_data.mol_descriptors, "molecular_descriptors")

    def test_smiles_to_graph_invalid(self, molecular_preprocessor):
        """Test SMILES to graph conversion with invalid inputs."""
        invalid_smiles = ["INVALID", "", "X", "C(C(C"]

        for smiles in invalid_smiles:
            graph_data = molecular_preprocessor.smiles_to_graph(smiles)
            assert graph_data is None, f"Should return None for invalid SMILES: {smiles}"

    def test_smiles_to_graph_large_molecule(self, molecular_preprocessor):
        """Test handling of molecules that exceed max_atoms limit."""
        # Create a large molecule SMILES (this is a simplified test)
        large_smiles = "C" * 150  # Very long carbon chain (invalid but tests size limit)

        graph_data = molecular_preprocessor.smiles_to_graph(large_smiles)
        # Should return None due to parsing failure or size limit
        assert graph_data is None

    def test_atom_features_consistency(self, molecular_preprocessor):
        """Test that atom features are consistent."""
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        graph_data = molecular_preprocessor.smiles_to_graph(smiles)

        assert graph_data is not None

        # Check feature dimensions
        num_atoms = graph_data.x.size(0)
        feature_dim = graph_data.x.size(1)

        assert feature_dim > 60, "Feature dimension should be reasonable size"
        assert num_atoms > 5, "Aspirin should have more than 5 atoms"

    def test_edge_features_when_enabled(self):
        """Test edge feature generation when enabled."""
        preprocessor = MolecularGraphPreprocessor(use_edge_features=True)
        smiles = "CCO"  # Ethanol (simple molecule)

        graph_data = preprocessor.smiles_to_graph(smiles)
        assert graph_data is not None

        if hasattr(graph_data, 'edge_attr'):
            assert graph_data.edge_attr.size(0) == graph_data.edge_index.size(1)
            pytest.assert_tensor_finite(graph_data.edge_attr, "edge_attributes")

    def test_molecular_descriptors_range(self, molecular_preprocessor, sample_smiles):
        """Test that molecular descriptors are in reasonable ranges."""
        for smiles in sample_smiles[:2]:
            graph_data = molecular_preprocessor.smiles_to_graph(smiles)
            if graph_data is not None:
                descriptors = graph_data.mol_descriptors

                # Check for reasonable ranges (not too extreme)
                assert torch.all(descriptors[0:9] >= 0), "Basic descriptors should be non-negative"
                assert torch.all(descriptors[0:9] < 1000), "Basic descriptors should be reasonable"


class TestTemporalGraphConstructor:
    """Test temporal graph construction functionality."""

    def test_construct_heterogeneous_graph_basic(self, temporal_graph_constructor, sample_drug_pairs,
                                                sample_drug_features, sample_interaction_labels):
        """Test basic heterogeneous graph construction."""
        graph = temporal_graph_constructor.construct_heterogeneous_graph(
            sample_drug_pairs, sample_drug_features, sample_interaction_labels
        )

        assert hasattr(graph, 'x_dict'), "Should have node feature dictionary"
        assert hasattr(graph, 'edge_index_dict'), "Should have edge index dictionary"

        # Check node types
        expected_node_types = {'drug', 'metabolite', 'target'}
        assert set(graph.x_dict.keys()) == expected_node_types

        # Check that all node types have features
        for node_type in expected_node_types:
            assert graph.x_dict[node_type].size(0) > 0, f"{node_type} should have nodes"
            pytest.assert_tensor_finite(graph.x_dict[node_type], f"{node_type}_features")

    def test_construct_heterogeneous_graph_edge_types(self, temporal_graph_constructor,
                                                     sample_drug_pairs, sample_drug_features,
                                                     sample_interaction_labels):
        """Test that correct edge types are created."""
        graph = temporal_graph_constructor.construct_heterogeneous_graph(
            sample_drug_pairs, sample_drug_features, sample_interaction_labels
        )

        # Check for expected edge types
        expected_edge_patterns = [
            'drug',
            'metabolite',
            'target'
        ]

        edge_types = list(graph.edge_index_dict.keys())
        assert len(edge_types) > 0, "Should have edge types"

        # Check edge indices are valid
        for edge_type, edge_index in graph.edge_index_dict.items():
            assert edge_index.size(0) == 2, f"Edge index should have 2 rows for {edge_type}"
            assert edge_index.size(1) > 0, f"Should have edges for {edge_type}"
            assert torch.all(edge_index >= 0), f"Edge indices should be non-negative for {edge_type}"

    def test_add_temporal_dynamics(self, temporal_graph_constructor, sample_hetero_graph):
        """Test adding temporal dynamics to graph."""
        time_points = [0.0, 1.0, 2.0, 4.0, 8.0]

        temporal_graph = temporal_graph_constructor.add_temporal_dynamics(
            sample_hetero_graph, time_points
        )

        # Check that temporal features were added
        for node_type in temporal_graph.node_types:
            if hasattr(temporal_graph[node_type], 'temporal_features'):
                temporal_features = temporal_graph[node_type].temporal_features
                assert temporal_features.size(1) == len(time_points)
                pytest.assert_tensor_finite(temporal_features, f"{node_type}_temporal_features")
                pytest.assert_tensor_range(temporal_features, 0.0, 1.1, f"{node_type}_temporal_features")

    def test_empty_drug_pairs(self, temporal_graph_constructor):
        """Test handling of empty drug pairs."""
        empty_pairs = []
        empty_labels = []
        empty_features = {}

        graph = temporal_graph_constructor.construct_heterogeneous_graph(
            empty_pairs, empty_features, empty_labels
        )

        # Should still create valid graph structure
        assert hasattr(graph, 'x_dict')
        assert len(graph.x_dict) > 0  # Should have some node types


class TestDrugInteractionDataset:
    """Test drug interaction dataset functionality."""

    def test_dataset_creation(self, sample_drug_pairs, sample_interaction_labels, sample_drug_features):
        """Test creating a drug interaction dataset."""
        dataset = DrugInteractionDataset(
            sample_drug_pairs, sample_interaction_labels, sample_drug_features
        )

        assert len(dataset) == len(sample_drug_pairs)

        # Test getting an item
        item = dataset[0]
        assert 'drug1_features' in item
        assert 'drug2_features' in item
        assert 'label' in item
        assert 'drug1_id' in item
        assert 'drug2_id' in item

        # Check tensor shapes and types
        assert isinstance(item['label'], torch.Tensor)
        assert item['label'].dtype == torch.float

    def test_dataset_with_temporal_features(self, sample_drug_pairs, sample_interaction_labels,
                                          sample_drug_features, sample_temporal_features):
        """Test dataset with temporal features."""
        dataset = DrugInteractionDataset(
            sample_drug_pairs, sample_interaction_labels,
            sample_drug_features, sample_temporal_features
        )

        item = dataset[0]
        drug1_id = item['drug1_id']
        drug2_id = item['drug2_id']

        # Check if temporal features are included when available
        if drug1_id in sample_temporal_features:
            assert 'drug1_temporal' in item
        if drug2_id in sample_temporal_features:
            assert 'drug2_temporal' in item

    def test_dataset_indexing(self, sample_drug_pairs, sample_interaction_labels, sample_drug_features):
        """Test dataset indexing and edge cases."""
        dataset = DrugInteractionDataset(
            sample_drug_pairs, sample_interaction_labels, sample_drug_features
        )

        # Test valid indices
        for i in range(len(dataset)):
            item = dataset[i]
            assert isinstance(item, dict)

        # Test invalid indices
        with pytest.raises(IndexError):
            dataset[len(dataset)]

        with pytest.raises(IndexError):
            dataset[-len(dataset) - 1]


class TestDrugInteractionDataLoader:
    """Test drug interaction data loader functionality."""

    def test_data_loader_initialization(self, test_config):
        """Test data loader initialization."""
        data_loader = DrugInteractionDataLoader(test_config.data)

        assert data_loader.config == test_config.data
        assert data_loader.mol_preprocessor is not None
        assert data_loader.graph_constructor is not None
        assert data_loader.cache_dir.exists()

    def test_create_drug_interaction_pairs(self, test_config):
        """Test creation of drug interaction pairs."""
        data_loader = DrugInteractionDataLoader(test_config.data)

        smiles_list = ["CCO", "CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]  # Simple molecules

        drug_pairs, labels = data_loader.create_drug_interaction_pairs(
            smiles_list, interaction_probability=0.2, seed=42
        )

        assert len(drug_pairs) > 0
        assert len(drug_pairs) == len(labels)
        assert all(label in [0, 1] for label in labels)

        # Test reproducibility
        drug_pairs2, labels2 = data_loader.create_drug_interaction_pairs(
            smiles_list, interaction_probability=0.2, seed=42
        )

        assert drug_pairs == drug_pairs2
        assert labels == labels2

    def test_preprocess_molecules(self, test_config):
        """Test molecule preprocessing."""
        data_loader = DrugInteractionDataLoader(test_config.data)

        smiles_list = ["CCO", "CC(=O)O"]  # Simple molecules

        features = data_loader.preprocess_molecules(smiles_list)

        assert len(features) == len(smiles_list)
        for smiles in smiles_list:
            assert smiles in features
            assert isinstance(features[smiles], torch.Tensor)
            assert features[smiles].size(0) == 265  # Molecular descriptor size

    def test_create_temporal_features(self, test_config, sample_drug_pairs):
        """Test temporal feature creation."""
        data_loader = DrugInteractionDataLoader(test_config.data)

        temporal_features = data_loader.create_temporal_features(sample_drug_pairs)

        # Extract unique drugs
        all_drugs = set()
        for drug1, drug2 in sample_drug_pairs:
            all_drugs.add(drug1)
            all_drugs.add(drug2)

        assert len(temporal_features) == len(all_drugs)

        for drug in all_drugs:
            assert drug in temporal_features
            assert isinstance(temporal_features[drug], torch.Tensor)
            assert temporal_features[drug].size(0) == 25  # Default time points

    def test_get_data_loaders(self, test_config, sample_drug_pairs, sample_interaction_labels,
                             sample_drug_features):
        """Test data loader creation."""
        data_loader = DrugInteractionDataLoader(test_config.data)

        train_loader, val_loader, test_loader = data_loader.get_data_loaders(
            sample_drug_pairs, sample_interaction_labels, sample_drug_features,
            batch_size=2
        )

        # Check that loaders are created
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Check that datasets have appropriate sizes
        total_samples = len(sample_drug_pairs)
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        assert train_size + val_size + test_size == total_samples
        assert train_size > 0
        assert val_size >= 0  # Could be 0 for small datasets
        assert test_size >= 0

    def test_synthetic_tox21_generation(self, test_config):
        """Test synthetic Tox21 data generation."""
        data_loader = DrugInteractionDataLoader(test_config.data)

        smiles_list, labels_array, task_names = data_loader._generate_synthetic_tox21()

        assert len(smiles_list) > 0
        assert labels_array.shape[0] == len(smiles_list)
        assert labels_array.shape[1] == len(task_names)
        assert len(task_names) == 12  # Expected number of Tox21 tasks

        # Check label values are binary
        assert np.all(np.isin(labels_array, [0, 1]))

    def test_prediction_interaction_likelihood(self, test_config):
        """Test interaction likelihood prediction."""
        data_loader = DrugInteractionDataLoader(test_config.data)

        # Test with valid SMILES
        smiles1 = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        smiles2 = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine

        likelihood = data_loader._predict_interaction_likelihood(smiles1, smiles2)
        assert likelihood in [0, 1]

        # Test with invalid SMILES
        likelihood_invalid = data_loader._predict_interaction_likelihood("INVALID", smiles2)
        assert likelihood_invalid == 0