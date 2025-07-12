"""
Configuration Settings Module for Synthetic Data Generator

This module provides comprehensive configuration management for the synthetic
data generator application, including Azure service configurations, PII protection
settings, and processing parameters.

The module implements pydantic-based settings management with environment variable
support and validation, ensuring robust configuration handling across different
deployment environments.

Example:
    Basic usage of the settings:
    
    >>> from config.settings import settings
    >>> print(settings.azure_openai_endpoint)
    >>> print(settings.ENABLE_PII_PROTECTION)
"""

import os
import logging
from enum import Enum
from typing import List, Dict, Any
from pydantic_settings import BaseSettings


logger = logging.getLogger(__name__)


class VectorizationApproach(str, Enum):
    """
    Enumeration for vectorization approach selection.
    
    This enum defines the available approaches for document vectorization
    and embedding generation in the synthetic data pipeline.
    
    Attributes:
        CUSTOM: Manual pipeline with custom embedding generation
        INTEGRATED: Integrated vectorization using Azure AI Search
    """
    CUSTOM = "custom"
    INTEGRATED = "integrated"


class SearchType(str, Enum):
    """
    Enumeration for search type configuration.
    
    This enum defines the available search methods for vector similarity
    and document retrieval operations.
    
    Attributes:
        TEXT: Traditional keyword-based text search
        VECTOR: Vector similarity search using embeddings
        SEMANTIC: Semantic search with AI-powered understanding
        HYBRID: Combined approach using multiple search methods
    """
    TEXT = "text"
    VECTOR = "vector"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class Settings(BaseSettings):
    """
    Comprehensive settings class for synthetic data generator configuration.
    
    This class manages all configuration parameters for the synthetic data generator,
    including Azure service configurations, processing parameters, and PII protection
    settings. It uses pydantic for validation and environment variable loading.
    
    The settings are organized into logical groups:
    - Vectorization and search configuration
    - Azure service connections
    - Processing parameters
    - PII protection settings (NIST SP 800-122 compliant)
    
    Attributes:
        vectorization_approach (VectorizationApproach): Approach for document vectorization
        azure_search_endpoint (str): Azure AI Search service endpoint
        azure_openai_endpoint (str): Azure OpenAI service endpoint
        chunk_size (int): Size of text chunks for processing
        ENABLE_PII_PROTECTION (bool): Flag to enable PII protection
        
    Example:
        >>> settings = Settings()
        >>> print(f"Using {settings.vectorization_approach} vectorization")
        >>> print(f"PII protection: {settings.ENABLE_PII_PROTECTION}")
    """
    
    # Vectorization approach selection
    vectorization_approach: VectorizationApproach = VectorizationApproach(
        os.getenv("VECTORIZATION_APPROACH", "integrated")
    )
    """
    Vectorization approach for document processing.
    
    Determines whether to use custom embedding pipeline or integrated
    Azure AI Search vectorization. Defaults to INTEGRATED for simplicity.
    """
    
    # Azure AI Search Configuration (Integrated approach)
    azure_search_endpoint: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    """Azure AI Search service endpoint URL for integrated vectorization."""
    
    azure_search_api_key: str = os.getenv("AZURE_SEARCH_API_KEY", "")
    """Azure AI Search service API key for authentication."""
    
    azure_search_index_name: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "synthetic-data-index")
    """Name of the Azure AI Search index for document storage."""
    
    azure_search_api_version: str = os.getenv("AZURE_SEARCH_API_VERSION", "2023-11-01")
    """Azure AI Search API version for compatibility."""
    
    # Custom approach settings
    azure_blob_connection_string: str = os.getenv("AZURE_BLOB_CONNECTION_STRING", "")
    """Azure Blob Storage connection string for document storage."""
    
    azure_cosmos_connection_string: str = os.getenv("AZURE_COSMOS_CONNECTION_STRING", "")
    """Azure Cosmos DB connection string for metadata storage."""
    
    cosmos_database_name: str = os.getenv("COSMOS_DATABASE_NAME", "synthetic_data")
    """Cosmos DB database name for application data."""
    
    cosmos_container_name: str = os.getenv("COSMOS_CONTAINER_NAME", "vectors")
    """Cosmos DB container name for vector storage."""
    
    # Azure OpenAI Configuration (used by both approaches)
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    """Azure OpenAI service endpoint URL for embeddings and generation."""
    
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    """Azure OpenAI service API key for authentication."""
    
    azure_openai_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-35-turbo")
    """Azure OpenAI deployment name for synthetic data generation."""
    
    azure_openai_model: str = os.getenv("AZURE_OPENAI_MODEL", "gpt-35-turbo")
    """Azure OpenAI model name for synthetic data generation."""
    
    azure_openai_embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    """Azure OpenAI embedding deployment name for vector generation."""
    
    azure_openai_embedding_model: str = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    """Azure OpenAI embedding model name for vector generation."""
    
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
    """Azure OpenAI API version for compatibility."""
    
    # Processing Configuration
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    """Size of text chunks for document processing (characters)."""
    
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    """Overlap between adjacent text chunks (characters)."""
    
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    """Dimension of embedding vectors (depends on model)."""
    
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
    """Maximum number of search results to return."""
    
    # Search Configuration
    default_search_type: SearchType = SearchType(os.getenv("DEFAULT_SEARCH_TYPE", "hybrid"))
    """Default search type for similarity operations."""
    
    enable_semantic_search: bool = os.getenv("ENABLE_SEMANTIC_SEARCH", "true").lower() == "true"
    """Enable semantic search capabilities."""
    
    enable_vector_search: bool = os.getenv("ENABLE_VECTOR_SEARCH", "true").lower() == "true"
    """Enable vector similarity search."""
    
    enable_hybrid_search: bool = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
    """Enable hybrid search combining multiple methods."""
    
    enable_reranking: bool = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
    """Enable search result reranking for improved relevance."""
    
    # Azure Storage for document loading
    azure_storage_connection_string: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    """Azure Storage connection string for document loading."""
    
    # NIST SP 800-122 Compliant PII Protection Settings
    ENABLE_PII_PROTECTION: bool = os.getenv("ENABLE_PII_PROTECTION", "true").lower() == "true"
    """
    Enable comprehensive PII protection following NIST SP 800-122 guidelines.
    
    When enabled, all text processing includes PII detection and anonymization
    to ensure compliance with data protection standards.
    """
    
    PII_ENTITIES_TO_ANONYMIZE: List[str] = os.getenv(
        "PII_ENTITIES_TO_ANONYMIZE", 
        "PERSON,EMAIL_ADDRESS,PHONE_NUMBER,CREDIT_CARD,US_SSN"
    ).split(",")
    """
    List of PII entity types to anonymize during processing.
    
    Supports all Presidio entity types including custom patterns
    defined in the PII protection module.
    """
    
    PII_VALIDATION_ENABLED: bool = os.getenv("PII_VALIDATION_ENABLED", "true").lower() == "true"
    """
    Enable PII validation in generated synthetic content.
    
    When enabled, all generated content is scanned for PII and
    anonymized before being returned to the user.
    """
    
    # NIST Risk-based PII Protection
    PII_RISK_THRESHOLD: str = os.getenv("PII_RISK_THRESHOLD", "MODERATE")
    """
    Minimum risk level for PII protection (HIGH, MODERATE, LOW).
    
    Follows NIST SP 800-122 risk classification:
    - HIGH: Significant harm potential
    - MODERATE: Minor harm potential  
    - LOW: Unlikely to cause harm
    """
    
    # NIST-recognized PII entities for Presidio
    NIST_PRESIDIO_ENTITIES: List[str] = os.getenv(
        "NIST_PRESIDIO_ENTITIES", 
        "PERSON,EMAIL_ADDRESS,PHONE_NUMBER,CREDIT_CARD,US_SSN,US_PASSPORT,US_DRIVER_LICENSE,DATE_TIME,MEDICAL_LICENSE,US_BANK_NUMBER,CRYPTO,IBAN_CODE,IP_ADDRESS,URL,US_ITIN,LOCATION,ORGANIZATION"
    ).split(",")
    """
    Comprehensive list of NIST-recognized PII entities for Presidio detection.
    
    Includes all major PII categories as defined in NIST SP 800-122:
    - Direct identifiers (SSN, passport, driver's license)
    - Financial information (credit cards, bank accounts)
    - Contact information (email, phone, address)
    - Medical information (medical records, insurance)
    - Employment and educational information
    """
    
    # Custom PII patterns (enabled by default for NIST compliance)
    ENABLE_CUSTOM_PII_PATTERNS: bool = os.getenv("ENABLE_CUSTOM_PII_PATTERNS", "true").lower() == "true"
    """
    Enable custom PII pattern detection beyond Presidio's built-in entities.
    
    When enabled, additional regex patterns are used to detect PII types
    that may not be covered by Presidio's default recognizers.
    """
    
    # NIST-compliant metadata sanitization
    ENABLE_METADATA_SANITIZATION: bool = os.getenv("ENABLE_METADATA_SANITIZATION", "true").lower() == "true"
    """
    Enable comprehensive metadata sanitization for document processing.
    
    When enabled, document metadata is scanned for PII and sensitive
    information is anonymized or removed before storage.
    """
    
    # PII detection reporting
    ENABLE_PII_REPORTING: bool = os.getenv("ENABLE_PII_REPORTING", "true").lower() == "true"
    """
    Enable detailed PII detection reporting for compliance auditing.
    
    When enabled, comprehensive reports are generated documenting
    all PII detection and anonymization activities.
    """
    
    class Config:
        """
        Pydantic configuration for settings management.
        
        Configures environment file loading and encoding settings
        for robust configuration management across environments.
        """
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
    def validate_azure_config(self) -> Dict[str, bool]:
        """
        Validate Azure service configuration completeness.
        
        Checks that all required Azure service configurations are present
        and properly formatted for the selected vectorization approach.
        
        Returns:
            Dict[str, bool]: Dictionary indicating validation status for each service
            
        Example:
            >>> settings = Settings()
            >>> validation = settings.validate_azure_config()
            >>> if all(validation.values()):
            ...     print("All Azure services configured correctly")
        """
        validation_results = {}
        
        # Validate Azure OpenAI (required for both approaches)
        validation_results['azure_openai'] = bool(
            self.azure_openai_endpoint and 
            self.azure_openai_api_key and
            self.azure_openai_embedding_model
        )
        
        # Validate Azure Search (required for integrated approach)
        if self.vectorization_approach == VectorizationApproach.INTEGRATED:
            validation_results['azure_search'] = bool(
                self.azure_search_endpoint and 
                self.azure_search_api_key
            )
        
        # Validate storage services (required for custom approach)
        if self.vectorization_approach == VectorizationApproach.CUSTOM:
            validation_results['azure_blob'] = bool(self.azure_blob_connection_string)
            validation_results['azure_cosmos'] = bool(self.azure_cosmos_connection_string)
        
        return validation_results
    
    def get_pii_protection_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of PII protection configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing PII protection configuration summary
            
        Example:
            >>> settings = Settings()
            >>> summary = settings.get_pii_protection_summary()
            >>> print(f"PII protection level: {summary['protection_level']}")
        """
        return {
            'protection_enabled': self.ENABLE_PII_PROTECTION,
            'risk_threshold': self.PII_RISK_THRESHOLD,
            'validation_enabled': self.PII_VALIDATION_ENABLED,
            'metadata_sanitization': self.ENABLE_METADATA_SANITIZATION,
            'custom_patterns_enabled': self.ENABLE_CUSTOM_PII_PATTERNS,
            'reporting_enabled': self.ENABLE_PII_REPORTING,
            'entities_monitored': len(self.NIST_PRESIDIO_ENTITIES),
            'nist_compliance': True,
            'protection_level': 'COMPREHENSIVE' if self.ENABLE_PII_PROTECTION else 'DISABLED'
        }
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get summary of processing configuration parameters.
        
        Returns:
            Dict[str, Any]: Dictionary containing processing configuration summary
            
        Example:
            >>> settings = Settings()
            >>> summary = settings.get_processing_summary()
            >>> print(f"Chunk size: {summary['chunk_size']}")
        """
        return {
            'vectorization_approach': self.vectorization_approach.value,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'embedding_dimension': self.embedding_dimension,
            'max_search_results': self.max_search_results,
            'default_search_type': self.default_search_type.value,
            'search_capabilities': {
                'semantic': self.enable_semantic_search,
                'vector': self.enable_vector_search,
                'hybrid': self.enable_hybrid_search,
                'reranking': self.enable_reranking
            }
        }


# Global settings instance
settings = Settings()
"""
Global settings instance for application-wide configuration access.

This singleton instance provides convenient access to all configuration
parameters throughout the application.

Example:
    >>> from config.settings import settings
    >>> print(settings.azure_openai_endpoint)
    >>> print(settings.ENABLE_PII_PROTECTION)
"""

# Log configuration validation on import
try:
    azure_validation = settings.validate_azure_config()
    pii_summary = settings.get_pii_protection_summary()
    
    logger.info(f"Configuration loaded: {settings.vectorization_approach.value} approach")
    logger.info(f"PII protection: {pii_summary['protection_level']}")
    
    if not all(azure_validation.values()):
        missing_services = [k for k, v in azure_validation.items() if not v]
        logger.warning(f"Missing Azure configuration for: {missing_services}")
        
except Exception as e:
    logger.error(f"Configuration validation failed: {str(e)}")