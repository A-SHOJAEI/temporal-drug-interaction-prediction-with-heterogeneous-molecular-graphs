"""Temporal Drug Interaction Prediction with Heterogeneous Molecular Graphs.

A production-ready machine learning system that predicts adverse drug-drug
interactions by modeling molecular structures as heterogeneous graphs with
temporal attention mechanisms.
"""

__version__ = "1.0.0"
__author__ = "Drug Interaction Prediction Team"

from .models.model import TemporalDrugInteractionGNN
from .data.loader import DrugInteractionDataLoader
from .training.trainer import TemporalDrugTrainer
from .evaluation.metrics import DrugInteractionMetrics

__all__ = [
    "TemporalDrugInteractionGNN",
    "DrugInteractionDataLoader",
    "TemporalDrugTrainer",
    "DrugInteractionMetrics",
]