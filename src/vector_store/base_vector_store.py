"""
Base Vector Store Interface

This module defines the abstract base class for vector storage implementations,
providing a consistent interface across different vector database backends.
"""

from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document


class BaseVectorStore(ABC):
    """
    Abstract base class for vector storage implementations.
    
    This class defines the interface that all vector store implementations
    must follow, ensuring consistency across different backends.
    """
    
    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of documents to store
            embeddings: Corresponding embedding vectors
        """
        pass
    
    @abstractmethod
    def similarity_search_by_vector(self, embedding: List[float], k: int = 5) -> List[Document]:
        """
        Search for similar documents using a query embedding vector.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents using a text query.
        
        Args:
            query: Text query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        pass
    
    def clear(self) -> None:
        """
        Clear all documents from the vector store.
        
        Default implementation - can be overridden by specific implementations.
        """
        pass