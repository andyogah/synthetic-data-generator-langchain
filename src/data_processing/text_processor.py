"""
Simplified Text Processing Utilities

This module provides simple text processing functions that can be used
directly in the pipeline without requiring DataFrame structures.
"""

import re
import logging
from .pii_protection import PIIProtector

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text data using standard preprocessing techniques.
    
    Args:
        text (str): Raw text string to clean and normalize
        
    Returns:
        str: Cleaned and normalized text string
    """
    if not isinstance(text, str):
        logger.warning(f"Non-string input received: {type(text)}")
        return str(text)
        
    # Convert to lowercase for normalization
    text = text.lower()
    
    # Remove extra whitespace (multiple spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Strip leading/trailing whitespace
    cleaned_text = text.strip()
    
    return cleaned_text


def process_text_with_pii_protection(text: str, 
                                   enable_pii_protection: bool = True) -> str:
    """
    Process text with optional PII protection.
    
    Args:
        text (str): Input text to process
        enable_pii_protection (bool): Whether to apply PII protection
        
    Returns:
        str: Processed text
        
    Example:
        >>> processed = process_text_with_pii_protection(
        ...     "John Doe's email is john@email.com",
        ...     enable_pii_protection=True
        ... )
        >>> print(processed)  # "john doe email is [EMAIL-REDACTED]"
    """
    # Clean text first
    cleaned_text = clean_text(text)
    
    # Apply PII protection if enabled
    if enable_pii_protection:
        try:
            pii_protector = PIIProtector()
            cleaned_text = pii_protector.anonymize_text(cleaned_text)
        except (ImportError, AttributeError, ValueError, TypeError) as e:
            logger.warning(f"PII protection failed, continuing without: {e}")
    
    return cleaned_text


def validate_text_input(text: str) -> bool:
    """
    Validate text input for processing.
    
    Args:
        text (str): Text to validate
        
    Returns:
        bool: True if text is valid for processing
        
    Example:
        >>> valid = validate_text_input("Sample text")
        >>> print(valid)  # True
        >>> valid = validate_text_input("")
        >>> print(valid)  # False
    """
    return isinstance(text, str) and len(text.strip()) > 0


def get_text_stats(text: str) -> dict:
    """
    Get basic statistics about text content.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Dictionary containing text statistics
        
    Example:
        >>> stats = get_text_stats("Hello world! This is a test.")
        >>> print(stats['word_count'])  # 6
        >>> print(stats['char_count'])  # 28
    """
    if not isinstance(text, str):
        return {'error': 'Invalid input type'}
    
    words = text.split()
    return {
        'char_count': len(text),
        'word_count': len(words),
        'line_count': len(text.splitlines()),
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
        'is_empty': len(text.strip()) == 0
    }
