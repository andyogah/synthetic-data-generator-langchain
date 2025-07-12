"""
Dependency Setup Script for Synthetic Data Generator

This script handles the installation of additional dependencies that require
special handling, such as spaCy models and Azure SDK components.
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def install_spacy_model():
    """Install the English spaCy model for PII detection."""
    try:
        logger.info("Installing spaCy English model...")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        logger.info("spaCy English model installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install spaCy model: {e}")
        logger.info("You can install it manually with: python -m spacy download en_core_web_sm")


def verify_azure_dependencies():
    """Verify that Azure dependencies are properly installed."""
    azure_packages = [
        'azure.storage.blob',
        'azure.search.documents', 
        'azure.cosmos',
        'azure.identity'
    ]
    
    missing_packages = []
    
    for package in azure_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is not available")
    
    if missing_packages:
        logger.error(f"Missing Azure packages: {missing_packages}")
        logger.info("Install with: pip install azure-storage-blob azure-search-documents azure-cosmos azure-identity")
    else:
        logger.info("All Azure dependencies are available")
    
    return len(missing_packages) == 0


def verify_langchain_dependencies():
    """Verify that LangChain dependencies are properly installed."""
    langchain_packages = [
        'langchain',
        'langchain_openai',
        'langchain_community'
    ]
    
    missing_packages = []
    
    for package in langchain_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is not available")
    
    if missing_packages:
        logger.error(f"Missing LangChain packages: {missing_packages}")
        logger.info("Install with: pip install langchain langchain-openai langchain-community")
    else:
        logger.info("All LangChain dependencies are available")
    
    return len(missing_packages) == 0


def main():
    """Main setup function."""
    logger.info("Setting up dependencies for Synthetic Data Generator...")
    
    # Verify core dependencies
    azure_ok = verify_azure_dependencies()
    langchain_ok = verify_langchain_dependencies()
    
    # Install spaCy model
    install_spacy_model()
    
    if azure_ok and langchain_ok:
        logger.info("✓ All dependencies are properly configured!")
    else:
        logger.warning("⚠ Some dependencies are missing. Please install them manually.")
        logger.info("Run: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
