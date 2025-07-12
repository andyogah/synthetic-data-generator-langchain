"""
Main Entry Point for Synthetic Data Generator

This module provides the main entry point for running the synthetic
data generation pipeline with example usage.
"""

import logging
import sys
from pathlib import Path
from data_processing.pipeline import DataProcessingPipeline
from config.settings import settings

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """
    Main function demonstrating the synthetic data generation pipeline.
    """
    try:
        logger.info("Starting Synthetic Data Generator")
        
        # Initialize pipeline using settings
        pipeline = DataProcessingPipeline(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            embedding_type="huggingface",  # Default to HuggingFace for local testing
            vector_store_type="mock",      # Use mock store for demo
            enable_pii_protection=settings.ENABLE_PII_PROTECTION
        )
        
        # Example: Process documents (you can modify this for your use case)
        logger.info("Pipeline initialized successfully")
        
        # Get pipeline stats
        stats = pipeline.get_pipeline_stats()
        logger.info(f"Pipeline configuration: {stats['configuration']}")
        
        # Example usage - uncomment and modify as needed:
        # result = pipeline.process_documents("directory", "/path/to/your/documents")
        # if result['success']:
        #     logger.info(f"Processed {result['total_chunks']} chunks from {result['total_documents']} documents")
        #     
        #     # Initialize generator with settings
        #     # generator = SyntheticDataGenerator(
        #     #     model_type="azure", 
        #     #     enable_pii_validation=settings.PII_VALIDATION_ENABLED
        #     # )
        #     
        #     # Generate synthetic data
        #     # synthetic_data = generator.generate_from_context("Sample context", num_samples=3)
        #     # logger.info(f"Generated {len(synthetic_data)} synthetic samples")
        
        logger.info("Synthetic Data Generator completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()