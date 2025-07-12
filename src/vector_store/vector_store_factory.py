"""
Vector Store Factory

This module provides a factory pattern implementation for creating
different types of vector stores based on configuration.
"""

import logging
from typing import Optional
from langchain.embeddings.base import Embeddings
from .base_vector_store import BaseVectorStore
from .langchain_vector_store import LangChainVectorStore

logger = logging.getLogger(__name__)


class MockVectorStore(BaseVectorStore):
    """
    Mock vector store implementation for development and testing.
    
    This is a simple in-memory implementation that can be used
    when proper vector stores are not configured.
    """
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        
    def add_documents(self, documents, embeddings):
        """Add documents and embeddings to memory"""
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        logger.info(f"Added {len(documents)} documents to mock vector store")
        
    def similarity_search_by_vector(self, embedding, k=5):
        """Return first k documents (mock implementation)"""
        return self.documents[:k]
        
    def similarity_search(self, query, k=5):
        """Return first k documents (mock implementation)"""
        return self.documents[:k]
        
    def clear(self):
        """Clear all stored documents"""
        self.documents.clear()
        self.embeddings.clear()
        logger.info("Cleared mock vector store")


class VectorStoreFactory:
    """
    Factory for creating vector store instances.
    
    This factory creates appropriate vector store implementations
    based on the specified type and configuration.
    """
    
    def __init__(self):
        self.store_types = {
            'mock': MockVectorStore,
            'custom': MockVectorStore,  # Default to mock for now
        }
    
    def create_vector_store(self, store_type: str = "mock", embeddings: Optional[Embeddings] = None, **kwargs) -> BaseVectorStore:
        """
        Create a vector store instance.
        
        Args:
            store_type: Type of vector store to create
            embeddings: Embeddings instance for vector store
            **kwargs: Additional configuration parameters
            
        Returns:
            Vector store instance
            
        Raises:
            ValueError: If unsupported store type is specified
        """
        if store_type not in self.store_types:
            logger.warning(f"Unsupported vector store type: {store_type}, using mock instead")
            store_type = "mock"
        
        store_class = self.store_types[store_type]
        
        try:
            if store_type == "mock":
                return store_class()
            elif store_type == "langchain":
                return LangChainVectorStore(store_type="azure_search", embeddings=embeddings)
            elif store_type == "faiss":
                return LangChainVectorStore(store_type="faiss", embeddings=embeddings)
            else:
                return store_class(**kwargs)
        except (ImportError, TypeError, ValueError, AttributeError) as e:
            logger.error(f"Failed to create vector store of type {store_type}: {e}")
            logger.info("Falling back to mock vector store")
            return MockVectorStore()
    
    def register_store_type(self, name: str, store_class: type):
        """
        Register a new vector store type.
        
        Args:
            name: Name of the store type
            store_class: Class implementing BaseVectorStore
        """
        self.store_types[name] = store_class
        logger.info(f"Registered vector store type: {name}")