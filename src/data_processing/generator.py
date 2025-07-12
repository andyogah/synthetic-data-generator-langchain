"""
Synthetic Data Generation Module

This module provides advanced synthetic data generation capabilities using
Azure OpenAI and LangChain framework. It implements PII-safe synthetic data
creation with NIST SP 800-122 compliant validation and anonymization.

The module supports context-aware generation, maintaining original data
characteristics while ensuring generated content is synthetic and PII-free.

Example:
    Basic usage of the synthetic data generator:
    
    >>> from generator import SyntheticDataGenerator
    >>> generator = SyntheticDataGenerator(model_type="azure")
    >>> synthetic_data = generator.generate_from_context(
    ...     context="Sample business text",
    ...     num_samples=5
    ... )
"""

import json
from typing import List, Dict, Any
from langchain_openai import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.schema import Document
from src.config.settings import settings

from .pii_protection import PIIProtector


class SyntheticDataGenerator:
    """
    Advanced synthetic data generator with PII protection and context preservation.
    
    This class implements sophisticated synthetic data generation using Azure OpenAI
    and LangChain framework, with built-in PII protection following NIST SP 800-122
    guidelines. It maintains original data characteristics while ensuring generated
    content is synthetic and safe.
    
    Attributes:
        model_type (str): Type of language model to use ("azure" supported)
        llm: Language model instance for text generation
        pii_protector (PIIProtector): PII protection instance for validation
        generation_chain (LLMChain): LangChain chain for structured generation
        
    Example:
        >>> generator = SyntheticDataGenerator(
        ...     model_type="azure",
        ...     enable_pii_validation=True
        ... )
        >>> results = generator.generate_from_context("Sample text", num_samples=3)
    """
    
    def __init__(self, model_type: str = "azure", enable_pii_validation: bool = True):
        """
        Initialize the SyntheticDataGenerator with specified model and PII settings.
        
        Args:
            model_type (str, optional): Type of language model to use.
                Currently supports "azure" for Azure OpenAI. Defaults to "azure".
            enable_pii_validation (bool, optional): Enable PII validation in generated
                content. Defaults to True for security compliance.
                
        Raises:
            ValueError: If unsupported model_type is specified
            
        Example:
            >>> generator = SyntheticDataGenerator(
            ...     model_type="azure",
            ...     enable_pii_validation=True
            ... )
        """
        self.model_type = model_type
        
        if model_type == "azure":
            self.llm = AzureOpenAI(
                deployment_name=settings.azure_openai_deployment,
                model_name=settings.azure_openai_model,
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                temperature=0.7
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.pii_protector = PIIProtector() if enable_pii_validation else None
        
        self.generation_prompt = PromptTemplate(
            input_variables=["context", "style", "length"],
            template="""
            Based on the following context, generate synthetic data that maintains the same style and structure:
            
            Context: {context}
            
            Style: {style}
            Length: {length}
            
            Generate new content that:
            1. Follows the same format and structure as the context
            2. Maintains the same writing style and tone
            3. Contains similar but not identical information
            4. Is approximately {length} in length
            
            Generated content:
            """
        )
        
        self.generation_chain = LLMChain(
            llm=self.llm,
            prompt=self.generation_prompt
        )
    
    def generate_from_context(self, 
                             context: str, 
                             style: str = "professional",
                             length: str = "medium",
                             num_samples: int = 1) -> List[str]:
        """
        Generate synthetic data from provided context with specified parameters.
        
        This method creates synthetic data that maintains the style and structure
        of the original context while generating entirely new content. All generated
        content is validated for PII if validation is enabled.
        
        Args:
            context (str): Source text to use as generation context
            style (str, optional): Writing style for generation ("professional", 
                "casual", "technical", "creative"). Defaults to "professional".
            length (str, optional): Target length for generated content ("short", 
                "medium", "long"). Defaults to "medium".
            num_samples (int, optional): Number of synthetic samples to generate.
                Defaults to 1.
                
        Returns:
            List[str]: List of generated synthetic text samples
            
        Example:
            >>> generator = SyntheticDataGenerator()
            >>> context = "This is a professional business report about quarterly results."
            >>> synthetic_data = generator.generate_from_context(
            ...     context=context,
            ...     style="professional",
            ...     length="medium",
            ...     num_samples=3
            ... )
            >>> print(len(synthetic_data))  # 3
        """
        generated_samples = []
        
        for _ in range(num_samples):
            try:
                result = self.generation_chain.run(
                    context=context,
                    style=style,
                    length=length
                )
                generated_samples.append(result.strip())
            except (ValueError, TypeError, ConnectionError, TimeoutError) as e:
                print(f"Error generating sample: {e}")
                continue
        
        # Validate generated content for PII
        if self.pii_protector:
            validated_samples = []
            for sample in generated_samples:
                pii_detected = self.pii_protector.detect_pii(sample)
                if pii_detected:
                    # Log warning and anonymize
                    print(f"PII detected in generated sample: {pii_detected}")
                    sample = self.pii_protector.anonymize_text(sample)
                validated_samples.append(sample)
            return validated_samples
        
        return generated_samples
    
    def generate_from_documents(self, 
                               documents: List[Document],
                               samples_per_doc: int = 1,
                               style: str = "professional",
                               length: str = "medium") -> List[Dict[str, Any]]:
        """
        Generate synthetic data from multiple LangChain documents.
        
        This method processes multiple documents and generates synthetic content
        for each, maintaining document metadata and generation parameters for
        traceability and analysis.
        
        Args:
            documents (List[Document]): List of LangChain Document objects
            samples_per_doc (int, optional): Number of samples per document.
                Defaults to 1.
            style (str, optional): Writing style for generation. Defaults to "professional".
            length (str, optional): Target length for generated content. Defaults to "medium".
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing:
                - original_metadata: Original document metadata
                - generated_content: Generated synthetic content
                - generation_params: Parameters used for generation
                
        Example:
            >>> documents = [Document(page_content="Sample text", metadata={"source": "doc1"})]
            >>> generator = SyntheticDataGenerator()
            >>> results = generator.generate_from_documents(
            ...     documents=documents,
            ...     samples_per_doc=2,
            ...     style="professional"
            ... )
            >>> print(len(results))  # 2 (2 samples from 1 document)
        """
        synthetic_data = []
        
        for doc in documents:
            samples = self.generate_from_context(
                context=doc.page_content,
                style=style,
                length=length,
                num_samples=samples_per_doc
            )
            
            for sample in samples:
                synthetic_data.append({
                    'original_metadata': doc.metadata,
                    'generated_content': sample,
                    'generation_params': {
                        'style': style,
                        'length': length
                    }
                })
        
        return synthetic_data
    
    def save_synthetic_data(self, data: List[Dict[str, Any]], output_path: str):
        """
        Save generated synthetic data to JSON file.
        
        Args:
            data (List[Dict[str, Any]]): Generated synthetic data to save
            output_path (str): Path where data should be saved
            
        Example:
            >>> generator = SyntheticDataGenerator()
            >>> data = generator.generate_from_context("Sample", num_samples=5)
            >>> generator.save_synthetic_data(data, "synthetic_data.json")
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
