"""
Vector Store Module for Synthetic Data Generator

This module provides vector storage capabilities supporting multiple backends
including Azure Cognitive Search and FAISS for local development.
"""

from .vector_store_factory import VectorStoreFactory
from .base_vector_store import BaseVectorStore

__all__ = ['VectorStoreFactory', 'BaseVectorStore']