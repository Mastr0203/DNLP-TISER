"""
TISER Data Processing Module

This module provides utilities for loading, preprocessing, and managing
the TISER (Temporal Information Semantic Extraction and Reasoning) dataset.

Main components:
- preprocessing: Hierarchical stratified sampling for dataset reduction
- tiser_dataset: Dataset classes and loaders
"""

from src.data.preprocessing import (
    ContextKeyExtractor,
    HierarchicalSampler,
    TISERPreprocessor,
    preprocess_tiser_split
)

from src.data.tiser_dataset import (
    TiserExample,
    TiserFileLoader,
    TiserDataset,
    load_tiser_file
)

__all__ = [
    # Preprocessing utilities
    'ContextKeyExtractor',
    'HierarchicalSampler',
    'TISERPreprocessor',
    'preprocess_tiser_split',
    
    # Dataset loading and handling
    'TiserExample',
    'TiserFileLoader',
    'TiserDataset',
    'load_tiser_file',
]

