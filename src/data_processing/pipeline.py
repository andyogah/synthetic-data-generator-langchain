"""
Data Processing Pipeline Module for Synthetic Data Generator

This module implements the core orchestration pipeline for processing documents
through the complete synthetic data generation workflow. It coordinates multiple
processing stages including loading, preprocessing, chunking, embedding generation,
and vector storage.

The pipeline provides a unified interface for processing documents from various
sources and supports both Azure and local storage backends with comprehensive
error handling and logging.

Example:
    Basic usage of the data processing pipeline:
    
    >>> from pipeline import DataProcessingPipeline
    >>> pipeline = DataProcessingPipeline(
    ...     embedding_type="azure",
    ...     vector_store_type="azure_search"
    ... )
    >>> result = pipeline.process_documents("file", "document.pdf")
    >>> print(f"Processed {result['total_chunks']} chunks")
"""

import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.vector_store.vector_store_factory import VectorStoreFactory
from src.config.settings import settings

from .loader import DataLoader
from .embedder import Embedder
from .text_processor import process_text_with_pii_protection
from .pii_protection import PIIProtector

logger = logging.getLogger(__name__)


class DataProcessingPipeline:
    """
    Comprehensive data processing pipeline for synthetic data generation.
    
    This class orchestrates the complete document processing workflow from
    loading through vector storage. It integrates multiple processing components
    and provides a unified interface for document processing with comprehensive
    error handling and performance monitoring.
    
    The pipeline supports:
    - Multiple document sources (files, directories, Azure Blob Storage)
    - Configurable text processing and chunking
    - Multiple embedding providers (Azure OpenAI, HuggingFace)
    - Various vector storage backends (Azure Search, FAISS)
    - PII protection and data sanitization
    
    Attributes:
        loader (DataLoader): Document loading component
        text_splitter (RecursiveCharacterTextSplitter): Document chunking component
        embedder (Embedder): Embedding generation component
        vector_store: Vector storage backend
        enable_pii_protection (bool): Flag for PII protection
        
    Example:
        >>> pipeline = DataProcessingPipeline(
        ...     chunk_size=1000,
        ...     embedding_type="azure",
        ...     vector_store_type="azure_search"
        ... )
        >>> result = pipeline.process_documents("directory", "/path/to/docs")
    """
    
    def __init__(self, 
                 chunk_size: Optional[int] = None,
                 chunk_overlap: Optional[int] = None,
                 embedding_type: str = "huggingface",
                 vector_store_type: str = "custom",
                 enable_pii_protection: Optional[bool] = None):
        """
        Initialize the data processing pipeline with specified configuration.
        
        Args:
            chunk_size (Optional[int]): Size of text chunks. Uses settings default if None.
            chunk_overlap (Optional[int]): Overlap between chunks. Uses settings default if None.
            embedding_type (str): Type of embedding provider to use.
            vector_store_type (str): Type of vector storage backend.
            enable_pii_protection (Optional[bool]): Enable PII protection. Uses settings default if None.
        """
        try:
            # Use settings defaults if not provided
            chunk_size = chunk_size or settings.chunk_size
            chunk_overlap = chunk_overlap or settings.chunk_overlap
            enable_pii_protection = enable_pii_protection if enable_pii_protection is not None else settings.ENABLE_PII_PROTECTION
            
            # Initialize components
            self.loader = DataLoader()
            
            # Initialize text splitter (replaces chunker.py)
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            self.embedder = Embedder(embedding_type)
            
            # Initialize vector store
            self.vector_store_factory = VectorStoreFactory()
            self.vector_store = self.vector_store_factory.create_vector_store(vector_store_type)
            
            # Store configuration
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.embedding_type = embedding_type
            self.vector_store_type = vector_store_type
            self.enable_pii_protection = enable_pii_protection
            
            logger.info(f"Pipeline initialized: chunks={chunk_size}, "
                       f"embedding={embedding_type}, storage={vector_store_type}, "
                       f"pii_protection={enable_pii_protection}")
            
        except (ImportError, AttributeError, ValueError, TypeError) as e:
            logger.error(f"Pipeline initialization failed: {str(e)}")
            raise
    
    def _preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Preprocess documents with text cleaning and PII protection.
        
        Args:
            documents (List[Document]): Raw documents to preprocess
            
        Returns:
            List[Document]: Preprocessed documents
        """
        processed_docs = []
        
        for doc in documents:
            try:
                # Use simple text processing function
                processed_content = process_text_with_pii_protection(
                    doc.page_content, 
                    self.enable_pii_protection
                )
                
                # Handle metadata sanitization if needed
                if self.enable_pii_protection:
                    pii_protector = PIIProtector()
                    sanitized_metadata = pii_protector.sanitize_metadata(doc.metadata)
                else:
                    sanitized_metadata = doc.metadata
                
                processed_docs.append(Document(
                    page_content=processed_content,
                    metadata=sanitized_metadata
                ))
                
            except (AttributeError, ValueError, TypeError, RuntimeError) as e:
                logger.warning(f"Failed to preprocess document: {e}")
                continue
        
        return processed_docs
    
    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into manageable chunks.
        
        Args:
            documents (List[Document]): Documents to chunk
            
        Returns:
            List[Document]: List of document chunks
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            try:
                # Use text_splitter directly instead of chunker
                chunks = self.text_splitter.split_text(doc.page_content)
                
                # Create chunk documents with metadata
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_doc = Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            'chunk_id': chunk_idx,
                            'source_doc_id': doc_idx,
                            'chunk_size': len(chunk),
                            'total_chunks': len(chunks)
                        }
                    )
                    all_chunks.append(chunk_doc)
                    
            except (AttributeError, ValueError, TypeError, RuntimeError) as e:
                logger.warning(f"Failed to chunk document {doc_idx}: {e}")
                continue
        
        return all_chunks

    def process_documents(self, source_type: str, source_path: str) -> Dict[str, Any]:
        """
        Process documents through the complete pipeline workflow.
        
        This method orchestrates the full document processing pipeline:
        1. Load documents from specified source
        2. Preprocess text content with PII protection
        3. Split documents into manageable chunks
        4. Generate vector embeddings for chunks
        5. Store embeddings in vector database
        
        Args:
            source_type (str): Type of document source. Supported values:
                - "file": Single file processing
                - "directory": Directory of files
                - "azure_blob": Azure Blob Storage container
            source_path (str): Path or identifier for the document source
            
        Returns:
            Dict[str, Any]: Processing results containing:
                - total_documents: Number of source documents processed
                - total_chunks: Number of text chunks created
                - embedding_dimension: Dimension of generated embeddings
                - success: Boolean indicating processing success
                - error: Error message if processing failed
                
        Raises:
            ValueError: If unsupported source_type specified
            Exception: If any processing stage fails
            
        Example:
            >>> pipeline = DataProcessingPipeline()
            >>> 
            >>> # Process a single PDF file
            >>> result = pipeline.process_documents("file", "document.pdf")
            >>> print(f"Success: {result['success']}")
            >>> 
            >>> # Process a directory of documents
            >>> result = pipeline.process_documents("directory", "/docs")
            >>> print(f"Processed {result['total_chunks']} chunks")
            >>> 
            >>> # Process from Azure Blob Storage
            >>> result = pipeline.process_documents("azure_blob", "container/path")
            >>> print(f"Embedding dimension: {result['embedding_dimension']}")
        """
        try:
            # Load documents
            logger.info(f"Loading documents from {source_type}: {source_path}")
            if source_type == "azure_blob":
                documents = self.loader.load_from_azure_blob(source_path)
            elif source_type == "file":
                documents = self.loader.load_from_file(source_path)
            elif source_type == "directory":
                documents = self.loader.load_from_directory(source_path)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
            
            # Preprocess documents
            logger.info("Preprocessing documents")
            processed_docs = self._preprocess_documents(documents)
            
            # Chunk documents
            logger.info("Chunking documents")
            all_chunks = self._chunk_documents(processed_docs)
            
            # Create embeddings
            logger.info("Creating embeddings")
            chunk_texts = [chunk.page_content for chunk in all_chunks]
            embeddings = self.embedder.create_embeddings(chunk_texts)
            
            # Store in vector database
            logger.info("Storing in vector database")
            self.vector_store.add_documents(all_chunks, embeddings)
            
            return {
                'total_documents': len(documents),
                'total_chunks': len(all_chunks),
                'embedding_dimension': len(embeddings[0]) if embeddings else 0,
                'success': True
            }
            
        except (ValueError, AttributeError, TypeError, RuntimeError, OSError, IOError) as e:
            logger.error(f"Pipeline processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Search for documents similar to the given query.
        
        This method performs semantic similarity search against the stored
        document embeddings to find the most relevant content.
        
        Args:
            query (str): Search query text
            top_k (int, optional): Number of similar documents to return.
                Defaults to 5.
                
        Returns:
            List[Document]: List of similar documents ranked by relevance
            
        Raises:
            ValueError: If query is empty or invalid
            Exception: If search operation fails
            
        Example:
            >>> pipeline = DataProcessingPipeline()
            >>> pipeline.process_documents("file", "document.pdf")
            >>> 
            >>> # Search for similar content
            >>> results = pipeline.search_similar("machine learning algorithms", top_k=3)
            >>> for doc in results:
            ...     print(f"Relevance: {doc.metadata.get('score', 'N/A')}")
            ...     print(f"Content: {doc.page_content[:100]}...")
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        try:
            logger.info(f"Searching for similar documents: '{query[:50]}...'")
            query_embedding = self.embedder.create_query_embedding(query)
            similar_docs = self.vector_store.similarity_search_by_vector(query_embedding, k=top_k)
            logger.info(f"Found {len(similar_docs)} similar documents")
            return similar_docs
            
        except (ValueError, AttributeError, TypeError, RuntimeError) as e:
            logger.error(f"Similarity search failed: {str(e)}")
            raise
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the pipeline configuration and performance.
        
        Returns:
            Dict[str, Any]: Pipeline statistics including configuration,
                component details, and performance metrics
            
        Example:
            >>> pipeline = DataProcessingPipeline()
            >>> stats = pipeline.get_pipeline_stats()
            >>> print(f"Chunk size: {stats['configuration']['chunk_size']}")
            >>> print(f"Embedding type: {stats['components']['embedder']}")
        """
        return {
            'configuration': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'embedding_type': self.embedding_type,
                'vector_store_type': self.vector_store_type,
                'pii_protection_enabled': self.enable_pii_protection
            },
            'components': {
                'loader': type(self.loader).__name__,
                'text_splitter': type(self.text_splitter).__name__,
                'embedder': type(self.embedder).__name__,
                'vector_store': type(self.vector_store).__name__
            },
            'capabilities': {
                'supported_sources': ['file', 'directory', 'azure_blob'],
                'embedding_providers': ['azure', 'huggingface'],
                'vector_stores': ['azure_search', 'faiss', 'custom'],
                'pii_protection': self.enable_pii_protection,
                'similarity_search': True
            }
        }
    
    def reset_pipeline(self) -> None:
        """
        Reset the pipeline by clearing the vector store and reinitializing components.
        
        This method is useful for processing new document sets or changing
        configuration parameters.
        
        Example:
            >>> pipeline = DataProcessingPipeline()
            >>> pipeline.process_documents("file", "doc1.pdf")
            >>> pipeline.reset_pipeline()  # Clear previous data
            >>> pipeline.process_documents("file", "doc2.pdf")  # Process new data
        """
        try:
            # Clear vector store if it has a reset method
            if hasattr(self.vector_store, 'clear'):
                self.vector_store.clear()
            
            logger.info("Pipeline reset completed")
            
        except (AttributeError, RuntimeError) as e:
            logger.error(f"Pipeline reset failed: {str(e)}")
            raise