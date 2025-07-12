"""
Text Embedding Module for Synthetic Data Generator

This module provides comprehensive text embedding capabilities using multiple
embedding providers including Azure OpenAI and HuggingFace. It supports
both document and query embeddings with flexible configuration options.

The module implements efficient embedding generation with caching and
serialization capabilities for large-scale document processing.

Example:
    Basic usage of the embedder:
    
    >>> from embedder import Embedder
    >>> embedder = Embedder(embedding_type="azure")
    >>> embeddings = embedder.create_embeddings(["Sample text", "Another text"])
    >>> query_embedding = embedder.create_query_embedding("Query text")
"""

import pickle
from typing import List
from langchain.embeddings import AzureOpenAIEmbeddings, HuggingFaceEmbeddings
from src.config.settings import settings


class Embedder:
    """
    Advanced text embedding class with multiple provider support.
    
    This class provides flexible text embedding capabilities supporting both
    Azure OpenAI and HuggingFace embedding models. It includes features for
    document embedding, query embedding, and embedding persistence.
    
    Attributes:
        embedding_type (str): Type of embedding provider ("azure" or "huggingface")
        model_name (str): Name of the embedding model
        embeddings: Embedding instance from LangChain
        
    Example:
        >>> # Using Azure OpenAI embeddings
        >>> embedder = Embedder(embedding_type="azure")
        >>> embeddings = embedder.create_embeddings(["Document text"])
        >>> 
        >>> # Using HuggingFace embeddings
        >>> embedder = Embedder(embedding_type="huggingface", model_name="all-MiniLM-L6-v2")
        >>> embeddings = embedder.create_embeddings(["Document text"])
    """
    
    def __init__(self, embedding_type: str = "huggingface", model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the Embedder with specified embedding provider and model.
        
        Args:
            embedding_type (str, optional): Type of embedding provider to use.
                Supported values: "azure", "huggingface". Defaults to "huggingface".
            model_name (str, optional): Name of the embedding model to use.
                For Azure: deployment name, for HuggingFace: model name.
                Defaults to 'all-MiniLM-L6-v2'.
        """
        self.embedding_type = embedding_type
        
        if embedding_type == "azure":
            self.embeddings = AzureOpenAIEmbeddings(
                deployment=settings.azure_openai_embedding_deployment,
                model=settings.azure_openai_embedding_model,
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

    def create_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of text chunks (documents).
        
        This method generates vector embeddings for multiple text documents,
        suitable for storage in vector databases and similarity search operations.
        
        Args:
            chunks (List[str]): List of text chunks to embed
            
        Returns:
            List[List[float]]: List of embedding vectors, one for each input chunk
        """
        return self.embeddings.embed_documents(chunks)

    def create_query_embedding(self, query: str) -> List[float]:
        """
        Create embedding for a single query text.
        
        This method generates a vector embedding for a query string, optimized
        for similarity search against document embeddings.
        
        Args:
            query (str): Query text to embed
            
        Returns:
            List[float]: Embedding vector for the query
        """
        return self.embeddings.embed_query(query)

    def save_embeddings(self, embeddings: List[List[float]], file_path: str):
        """
        Save embeddings to disk using pickle serialization.
        
        This method provides persistent storage for computed embeddings,
        enabling reuse and avoiding recomputation for large document sets.
        
        Args:
            embeddings (List[List[float]]): List of embedding vectors to save
            file_path (str): Path where embeddings should be saved
        """
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)

    def load_embeddings(self, file_path: str) -> List[List[float]]:
        """
        Load embeddings from disk using pickle deserialization.
        
        This method loads previously saved embeddings from disk, enabling
        efficient reuse of computed embeddings.
        
        Args:
            file_path (str): Path to the saved embeddings file
            
        Returns:
            List[List[float]]: List of loaded embedding vectors
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)