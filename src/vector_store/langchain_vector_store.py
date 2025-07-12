"""
LangChain Vector Store Implementation

This module provides a unified interface for different LangChain vector stores,
supporting both Azure Cognitive Search and FAISS backends.
"""

import logging
from typing import List, Optional
from langchain.vectorstores import AzureSearch, FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from src.config.settings import settings

from .base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class LangChainVectorStore(BaseVectorStore):
    """
    Unified LangChain vector store implementation.
    
    This class provides a consistent interface for different LangChain
    vector store backends including Azure Cognitive Search and FAISS.
    """
    
    def __init__(self, store_type: str = "faiss", embeddings: Optional[Embeddings] = None, **kwargs):
        """
        Initialize the LangChain vector store.
        
        Args:
            store_type: Type of vector store ("azure_search" or "faiss")
            embeddings: Embeddings instance for the vector store
            **kwargs: Additional configuration parameters
        """
        self.store_type = store_type
        self.embeddings = embeddings
        self.vector_store = None
        
        try:
            if store_type == "azure_search":
                self._init_azure_search(**kwargs)
            elif store_type == "faiss":
                self._init_faiss(**kwargs)
            else:
                raise ValueError(f"Unsupported store type: {store_type}")
                
        except ImportError as e:
            logger.error(f"Failed to import required dependencies for {store_type}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize {store_type} vector store: {e}")
            raise
    
    def _init_azure_search(self, **kwargs):
        """Initialize Azure Cognitive Search vector store"""
        try:
            # Get Azure Search configuration from settings first, then kwargs
            search_endpoint = kwargs.get('search_endpoint') or settings.azure_search_endpoint
            search_key = kwargs.get('search_key') or settings.azure_search_api_key
            index_name = kwargs.get('index_name') or settings.azure_search_index_name
            
            if not search_endpoint or not search_key:
                raise ValueError("Azure Search endpoint and key are required")
            
            self.vector_store = AzureSearch(
                azure_search_endpoint=search_endpoint,
                azure_search_key=search_key,
                index_name=index_name,
                embedding_function=self.embeddings.embed_query if self.embeddings else None
            )
            
            logger.info("Initialized Azure Search vector store")
            
        except ImportError:
            logger.error("Azure Search dependencies not available")
            raise
    
    def _init_faiss(self, **kwargs):
        """Initialize FAISS vector store"""
        try:
            if not self.embeddings:
                raise ValueError("Embeddings instance is required for FAISS")
            
            # Initialize empty FAISS index
            self.vector_store = None  # Will be created when first documents are added
            
            logger.info("Initialized FAISS vector store")
            
        except ImportError:
            logger.error("FAISS dependencies not available")
            raise
    
    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> None:
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of documents to store
            embeddings: Corresponding embedding vectors
        """
        if not documents:
            return
        
        try:
            if self.store_type == "azure_search":
                # Azure Search handles embeddings internally
                self.vector_store.add_documents(documents)
                
            elif self.store_type == "faiss":
                if self.vector_store is None:
                    # Create initial FAISS index
                    self.vector_store = FAISS.from_documents(documents, self.embeddings)
                else:
                    # Add to existing index
                    self.vector_store.add_documents(documents)
            
            logger.info(f"Added {len(documents)} documents to {self.store_type} vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise
    
    def similarity_search_by_vector(self, embedding: List[float], k: int = 5) -> List[Document]:
        """
        Search for similar documents using a query embedding vector.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized or empty")
            return []
        
        try:
            if self.store_type == "azure_search":
                # Azure Search may not support direct vector search
                # Fall back to similarity search with empty query
                return self.vector_store.similarity_search("", k=k)
                
            elif self.store_type == "faiss":
                return self.vector_store.similarity_search_by_vector(embedding, k=k)
            
        except Exception as e:
            logger.error(f"Similarity search by vector failed: {e}")
            return []
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents using a text query.
        
        Args:
            query: Text query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized or empty")
            return []
        
        try:
            return self.vector_store.similarity_search(query, k=k)
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the vector store"""
        if hasattr(self.vector_store, 'delete'):
            self.vector_store.delete(document_ids)
    
    def save_local(self, path: str) -> None:
        """Save vector store locally (for FAISS)"""
        if self.store_type == "faiss" and self.vector_store is not None:
            self.vector_store.save_local(path)
    
    def load_local(self, path: str) -> None:
        """Load vector store from local path (for FAISS)"""
        if self.store_type == "faiss":
            self.vector_store = FAISS.load_local(path, self.embeddings)
    
    def clear(self) -> None:
        """
        Clear all documents from the vector store.
        """
        try:
            if self.store_type == "faiss":
                self.vector_store = None
                logger.info("Cleared FAISS vector store")
            elif self.store_type == "azure_search":
                # Azure Search doesn't support clearing, would need to delete index
                logger.warning("Azure Search clearing not implemented")
                
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
