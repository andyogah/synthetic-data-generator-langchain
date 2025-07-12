"""
Document Loading Module for Synthetic Data Generator

This module provides comprehensive document loading capabilities supporting
multiple file formats and storage backends. It integrates with LangChain
document loaders to handle various document types and provides unified
access to local files, directories, and Azure Blob Storage.

The module implements robust error handling, metadata preservation, and
flexible configuration options for different document sources.

Example:
    Basic usage of the document loader:
    
    >>> from loader import DataLoader
    >>> loader = DataLoader()
    >>> 
    >>> # Load a single PDF file
    >>> documents = loader.load_from_file("document.pdf")
    >>> 
    >>> # Load all documents from a directory
    >>> documents = loader.load_from_directory("/path/to/documents")
    >>> 
    >>> # Load from Azure Blob Storage
    >>> documents = loader.load_from_azure_blob("container/path")
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.schema import Document
from langchain.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader, 
    JSONLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    DirectoryLoader
)
from langchain.document_loaders.azure_blob_storage_container import AzureBlobStorageContainerLoader

# Azure SDK imports with error handling
try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    BlobServiceClient = None
    AZURE_AVAILABLE = False

# Settings import with fallback
try:
    from ..config.settings import settings
except ImportError:
    # Fallback for when relative import fails
    try:
        from src.config.settings import settings
    except ImportError:
        # Create a minimal settings object as fallback
        class MockSettings:
            azure_storage_connection_string = None
        settings = MockSettings()

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Comprehensive document loading class with multi-format and multi-source support.
    
    This class provides a unified interface for loading documents from various
    sources and formats. It supports local files, directories, and Azure Blob Storage
    with automatic format detection and appropriate loader selection.
    
    Supported formats:
    - PDF documents (.pdf)
    - Text files (.txt, .md)
    - CSV files (.csv)
    - JSON files (.json)
    - Word documents (.docx, .doc)
    - Excel spreadsheets (.xlsx, .xls)
    - PowerPoint presentations (.pptx, .ppt)
    
    Supported sources:
    - Local files and directories
    - Azure Blob Storage containers
    
    Attributes:
        supported_extensions (Dict[str, type]): Mapping of file extensions to loaders
        azure_connection_string (Optional[str]): Azure Blob Storage connection string
        
    Example:
        >>> loader = DataLoader()
        >>> 
        >>> # Load various document types
        >>> pdf_docs = loader.load_from_file("report.pdf")
        >>> csv_docs = loader.load_from_file("data.csv")
        >>> json_docs = loader.load_from_file("config.json")
        >>> 
        >>> # Load from Azure Blob Storage
        >>> azure_docs = loader.load_from_azure_blob("container/documents")
    """
    
    def __init__(self, azure_connection_string: Optional[str] = None):
        """
        Initialize the DataLoader with optional Azure Blob Storage configuration.
        
        Args:
            azure_connection_string (Optional[str]): Azure Blob Storage connection string.
                If not provided, will attempt to load from settings.
        """
        self.azure_connection_string = (
            azure_connection_string or 
            settings.azure_storage_connection_string or 
            os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        )
        
        # Define supported file extensions and their corresponding loaders
        self.supported_extensions = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.md': TextLoader,
            '.csv': CSVLoader,
            '.json': JSONLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader,
            '.pptx': UnstructuredPowerPointLoader,
            '.ppt': UnstructuredPowerPointLoader
        }
        
        logger.info(f"DataLoader initialized with support for {len(self.supported_extensions)} file types")
        if self.azure_connection_string:
            logger.info("Azure Blob Storage integration enabled")
    
    def load_from_file(self, file_path: str, **kwargs) -> List[Document]:
        """
        Load documents from a single file with automatic format detection.
        
        This method automatically detects the file format based on extension
        and uses the appropriate LangChain loader to process the document.
        
        Args:
            file_path (str): Path to the file to load
            **kwargs: Additional arguments passed to the specific loader
            
        Returns:
            List[Document]: List of loaded documents (may contain multiple pages/sections)
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If the file format is not supported
            Exception: If document loading fails
            
        Example:
            >>> loader = DataLoader()
            >>> 
            >>> # Load a PDF document
            >>> pdf_docs = loader.load_from_file("report.pdf")
            >>> print(f"Loaded {len(pdf_docs)} pages")
            >>> 
            >>> # Load a CSV with custom delimiter
            >>> csv_docs = loader.load_from_file("data.csv", delimiter=";")
            >>> 
            >>> # Load a JSON with specific content key
            >>> json_docs = loader.load_from_file("data.json", jq_schema=".content")
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        try:
            logger.info(f"Loading document: {file_path}")
            
            # Get appropriate loader class
            loader_class = self.supported_extensions[file_extension]
            
            # Initialize loader with file path
            loader = loader_class(file_path, **kwargs)
            
            # Load documents
            documents = loader.load()
            
            # Enhance metadata with file information
            for doc in documents:
                doc.metadata.update({
                    'source_file': file_path,
                    'file_name': Path(file_path).name,
                    'file_extension': file_extension,
                    'file_size': os.path.getsize(file_path),
                    'loader_type': loader_class.__name__
                })
            
            logger.info(f"Successfully loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {str(e)}")
            raise
    
    def load_from_directory(self, 
                           directory_path: str, 
                           recursive: bool = True,
                           file_pattern: Optional[str] = None,
                           **kwargs) -> List[Document]:
        """
        Load documents from a directory with optional recursive scanning.
        
        This method scans a directory for supported document types and loads
        all found documents. It supports recursive directory traversal and
        file pattern filtering.
        
        Args:
            directory_path (str): Path to the directory to scan
            recursive (bool, optional): Whether to scan subdirectories recursively.
                Defaults to True.
            file_pattern (Optional[str]): Glob pattern to filter files.
                If None, loads all supported files.
            **kwargs: Additional arguments passed to individual loaders
            
        Returns:
            List[Document]: List of all loaded documents from the directory
            
        Raises:
            FileNotFoundError: If the directory doesn't exist
            Exception: If directory scanning or loading fails
            
        Example:
            >>> loader = DataLoader()
            >>> 
            >>> # Load all supported documents from directory
            >>> all_docs = loader.load_from_directory("/path/to/documents")
            >>> 
            >>> # Load only PDF files recursively
            >>> pdf_docs = loader.load_from_directory(
            ...     "/path/to/documents",
            ...     file_pattern="**/*.pdf"
            ... )
            >>> 
            >>> # Load files from current directory only
            >>> current_docs = loader.load_from_directory(
            ...     "/path/to/documents",
            ...     recursive=False
            ... )
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not os.path.isdir(directory_path):
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        try:
            logger.info(f"Loading documents from directory: {directory_path}")
            
            # Use DirectoryLoader for efficient directory scanning
            if file_pattern:
                loader = DirectoryLoader(
                    directory_path,
                    glob=file_pattern,
                    recursive=recursive,
                    **kwargs
                )
            else:
                # Create glob patterns for all supported extensions
                supported_patterns = [f"**/*{ext}" if recursive else f"*{ext}" 
                                    for ext in self.supported_extensions.keys()]
                
                all_documents = []
                for pattern in supported_patterns:
                    try:
                        loader = DirectoryLoader(
                            directory_path,
                            glob=pattern,
                            recursive=recursive,
                            **kwargs
                        )
                        documents = loader.load()
                        all_documents.extend(documents)
                    except Exception as e:
                        logger.warning(f"Failed to load files with pattern {pattern}: {e}")
                        continue
                
                # Enhance metadata for all documents
                for doc in all_documents:
                    doc.metadata.update({
                        'source_directory': directory_path,
                        'recursive_scan': recursive,
                        'loader_type': 'DirectoryLoader'
                    })
                
                logger.info(f"Successfully loaded {len(all_documents)} documents from directory")
                return all_documents
            
            # Load documents using the configured loader
            documents = loader.load()
            
            # Enhance metadata
            for doc in documents:
                doc.metadata.update({
                    'source_directory': directory_path,
                    'file_pattern': file_pattern,
                    'recursive_scan': recursive,
                    'loader_type': 'DirectoryLoader'
                })
            
            logger.info(f"Successfully loaded {len(documents)} documents from directory")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load documents from directory {directory_path}: {str(e)}")
            raise
    
    def load_from_azure_blob(self, 
                            container_name: str,
                            blob_prefix: Optional[str] = None,
                            connection_string: Optional[str] = None,
                            **kwargs) -> List[Document]:
        """
        Load documents from Azure Blob Storage container.
        
        This method connects to Azure Blob Storage and loads documents from
        a specified container. It supports blob prefix filtering and automatic
        format detection for supported file types.
        
        Args:
            container_name (str): Name of the Azure Blob Storage container
            blob_prefix (Optional[str]): Prefix to filter blobs (e.g., "documents/")
            connection_string (Optional[str]): Azure Storage connection string.
                If not provided, uses the instance connection string.
            **kwargs: Additional arguments passed to the Azure loader
            
        Returns:
            List[Document]: List of loaded documents from Azure Blob Storage
            
        Raises:
            ValueError: If no Azure connection string is available
            Exception: If Azure blob loading fails
            
        Example:
            >>> loader = DataLoader(azure_connection_string="DefaultEndpointsProtocol=https;...")
            >>> 
            >>> # Load all documents from container
            >>> all_docs = loader.load_from_azure_blob("documents")
            >>> 
            >>> # Load documents with specific prefix
            >>> filtered_docs = loader.load_from_azure_blob(
            ...     "documents",
            ...     blob_prefix="reports/2024/"
            ... )
        """
        # Use provided connection string or instance default
        conn_str = connection_string or self.azure_connection_string
        
        if not conn_str:
            raise ValueError("Azure connection string is required for blob storage access")
        
        try:
            logger.info(f"Loading documents from Azure Blob Storage: {container_name}")
            
            # Initialize Azure Blob Storage loader
            loader = AzureBlobStorageContainerLoader(
                conn_str=conn_str,
                container=container_name,
                prefix=blob_prefix,
                **kwargs
            )
            
            # Load documents
            documents = loader.load()
            
            # Enhance metadata with Azure-specific information
            for doc in documents:
                doc.metadata.update({
                    'source_type': 'azure_blob',
                    'container_name': container_name,
                    'blob_prefix': blob_prefix,
                    'loader_type': 'AzureBlobStorageContainerLoader'
                })
            
            logger.info(f"Successfully loaded {len(documents)} documents from Azure Blob Storage")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load documents from Azure Blob Storage: {str(e)}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List[str]: List of supported file extensions
            
        Example:
            >>> loader = DataLoader()
            >>> formats = loader.get_supported_formats()
            >>> print(f"Supported formats: {', '.join(formats)}")
        """
        return list(self.supported_extensions.keys())
    
    def validate_file_format(self, file_path: str) -> bool:
        """
        Validate if a file format is supported.
        
        Args:
            file_path (str): Path to the file to validate
            
        Returns:
            bool: True if the file format is supported, False otherwise
            
        Example:
            >>> loader = DataLoader()
            >>> if loader.validate_file_format("document.pdf"):
            ...     print("PDF format is supported")
            >>> if not loader.validate_file_format("document.xyz"):
            ...     print("XYZ format is not supported")
        """
        file_extension = Path(file_path).suffix.lower()
        return file_extension in self.supported_extensions
    
    def get_loader_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the loader configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing loader statistics and capabilities
            
        Example:
            >>> loader = DataLoader()
            >>> stats = loader.get_loader_stats()
            >>> print(f"Supported formats: {stats['supported_formats_count']}")
            >>> print(f"Azure enabled: {stats['azure_blob_enabled']}")
        """
        return {
            'supported_formats_count': len(self.supported_extensions),
            'supported_formats': list(self.supported_extensions.keys()),
            'azure_blob_enabled': bool(self.azure_connection_string),
            'supported_sources': ['file', 'directory', 'azure_blob'] if self.azure_connection_string else ['file', 'directory'],
            'loader_classes': [loader.__name__ for loader in self.supported_extensions.values()],
            'capabilities': {
                'recursive_directory_scan': True,
                'file_pattern_filtering': True,
                'metadata_enhancement': True
            }
        }
    
    def test_azure_connection(self) -> bool:
        """
        Test Azure Blob Storage connection.
        
        Returns:
            bool: True if connection is successful, False otherwise
            
        Example:
            >>> loader = DataLoader()
            >>> if loader.test_azure_connection():
            ...     print("Azure connection successful")
            >>> else:
            ...     print("Azure connection failed")
        """
        if not AZURE_AVAILABLE:
            logger.warning("Azure SDK not available - install azure-storage-blob package")
            return False
            
        if not self.azure_connection_string:
            logger.warning("No Azure connection string configured")
            return False
        
        try:
            # Test connection by listing containers
            blob_service_client = BlobServiceClient.from_connection_string(self.azure_connection_string)
            containers = list(blob_service_client.list_containers(max_results=1))
            
            logger.info("Azure Blob Storage connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"Azure Blob Storage connection test failed: {str(e)}")
            return False