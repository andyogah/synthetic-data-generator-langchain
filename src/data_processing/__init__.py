"""
Data Processing Module for Synthetic Data Generator

This module provides comprehensive data processing capabilities including
document loading, text processing, embedding generation, and synthetic data creation.
"""

from .pipeline import DataProcessingPipeline
from .loader import DataLoader
from .embedder import Embedder
from .generator import SyntheticDataGenerator
from .text_processor import process_text_with_pii_protection, clean_text
from .pii_protection import PIIProtector, NISTCompliantPIIProtector

__all__ = [
    'DataProcessingPipeline',
    'DataLoader', 
    'Embedder',
    'SyntheticDataGenerator',
    'process_text_with_pii_protection',
    'clean_text',
    'PIIProtector',
    'NISTCompliantPIIProtector'
]