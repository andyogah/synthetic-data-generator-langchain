# Synthetic Data Generator Langchain

## Overview

### The Challenge: The Data Access Paradox
In government agencies and high-security organizations, developers face a critical chicken-and-egg problem: **you need data to build proof-of-concepts (POCs), proof-of-value (POVs), and minimum viable products (MVPs), but you can't access the real data due to security, privacy, and clearance restrictions.**

This creates an impossible situation:
- **Security teams** won't release sensitive data for development
- **Privacy regulations** (GDPR, HIPAA, government classifications) prevent data sharing
- **Clearance requirements** may exclude developers from accessing classified datasets
- **Development timelines** can't wait for lengthy security approval processes
- **Innovation stalls** without realistic data to test and validate solutions

### The Solution: Contextually-Aware Synthetic Data Generation
The Synthetic Data Generator breaks this deadlock by creating **realistic, contextually-aware synthetic datasets** that preserve the statistical properties and semantic relationships of original data while ensuring complete privacy and security compliance.

**Key Value Propositions:**
- ✅ **Accelerate Development**: Build POCs and MVPs without waiting for data access approvals
- ✅ **Maintain Privacy**: Generate synthetic data that contains no actual PII or sensitive information
- ✅ **Preserve Context**: Maintain semantic relationships and statistical properties of original datasets
- ✅ **Enable Innovation**: Allow developers to work with realistic data regardless of clearance level
- ✅ **Compliance Ready**: Built-in PII protection ensures regulatory compliance from day one

### Technical Excellence
The Synthetic Data Generator leverages LangChain framework, advanced data processing techniques, embedding methods, and Azure OpenAI to generate new content that resembles the original data in both content and context, while providing enterprise-grade security and privacy protection.

## Features
- Load original data from various sources using LangChain document loaders
- Process data through intelligent chunking and embedding with LangChain
- Store embeddings in vector databases (Azure Cognitive Search, FAISS)
- Generate synthetic content using Azure OpenAI integration
- Support for multiple search methods with LangChain vector stores
- Built-in PII protection using Presidio analyzer

## Azure Products Used
- **Azure OpenAI Service**: For embeddings and synthetic content generation
- **Azure Cognitive Search**: For vector search capabilities (optional)

## Workflow Architecture

### Custom Embeddings Approach
This codebase implements a **custom embeddings pipeline** with manual orchestration:

#### 1. **Custom Embeddings Generation**
- **Using `Embedder` class** with Azure OpenAI embeddings
- **Manual control** over embedding model selection and parameters
- **Flexible embedding strategies** supporting Azure OpenAI

#### 2. **Manual Pipeline Orchestration**
Documents are processed through separate, coordinated steps:

```
Load → Preprocess → Chunk → Embed → Store
```

**Step-by-Step Workflow:**
1. **Load**: Multi-format document loading via LangChain loaders
2. **Preprocess**: Text cleaning and PII protection
3. **Chunk**: Intelligent text splitting with RecursiveCharacterTextSplitter
4. **Embed**: Vector generation using Azure OpenAI
5. **Store**: Vector storage in Azure Cognitive Search or FAISS

#### 3. **Separate Vector Store Management**
- **`LangChainVectorStore`** handles vector operations independently
- **Unified interface** for Azure Search and FAISS backends
- **Manual embedding injection** into vector storage

### Pipeline Flow Diagram
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   DataLoader│───▶│Preprocessor │───▶│   Chunker   │
│ (LangChain) │    │(PII Protection)│  │ (LangChain) │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│Vector Store │◀───│  Embedder   │◀───│  Documents  │
│(LangChain)  │    │(Azure OpenAI)│   │  (Chunked)  │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Output    │◀───│  Generator  │◀───│   Search    │
│(Synthetic)  │    │(Azure OpenAI)│   │(Similarity) │
└─────────────┘    └─────────────┘    └─────────────┘
```

## PII Protection Features

### Built-in PII Safeguards
- **Pre-processing Anonymization**: PII detection and anonymization using Presidio
- **Configurable Protection**: Customizable PII entities and protection levels
- **Generation Validation**: Post-generation PII detection and filtering

### PII Protection Layers
1. **Input Layer**: Document content anonymization using Presidio
2. **Storage Layer**: Vector embeddings contain no readable PII
3. **Output Layer**: Generated content validation and cleaning

### Supported PII Types
- Personal names (PERSON)
- Email addresses (EMAIL_ADDRESS)
- Phone numbers (PHONE_NUMBER)
- Social Security Numbers (US_SSN)
- Credit card numbers (CREDIT_CARD)
- Custom patterns via configuration

## Project Structure
```
synthetic-data-generator/
├── src/
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── pipeline.py        # Manual orchestration
│   │   ├── chunker.py         # LangChain text splitting
│   │   ├── embedder.py        # Azure OpenAI embeddings
│   │   ├── loader.py          # LangChain document loaders
│   │   ├── preprocessor.py    # Text cleaning & PII protection
│   │   └── generator.py       # Synthetic data generation
│   ├── vector_store/
│   │   ├── __init__.py
│   │   ├── base_vector_store.py
│   │   ├── langchain_vector_store.py  # Unified vector interface
│   │   └── vector_store_factory.py    # Vector store creation
│   └── utils/
│       ├── __init__.py
│       └── pii_protector.py   # PII detection and anonymization
├── .env.example
├── requirements.txt
└── README.md
```

## Technology Stack
- **LangChain**: Core framework for document processing, embeddings, and LLM integration
- **Azure OpenAI**: For embeddings and synthetic data generation
- **Azure Cognitive Search**: Vector database for semantic search (optional)
- **FAISS**: Alternative vector database for local development
- **Presidio**: PII detection and anonymization
- **Python**: Core programming language

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd synthetic-data-generator
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install all Python dependencies
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Copy the `.env.example` file and configure your Azure services:

```bash
cp .env.example .env
```

Edit `.env` with your Azure credentials:
```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-35-turbo
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Azure Cognitive Search (Optional)
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your_search_admin_key
AZURE_SEARCH_INDEX_NAME=documents

# Vector Store Configuration
VECTOR_STORE_TYPE=azure_search  # or 'faiss' for local development
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Usage

### Quick Start
```bash
# Run the main application
python src/main.py
```

### Using Individual Components

#### 1. Document Loading
```python
from src.data_processing.loader import DataLoader

loader = DataLoader()
documents = loader.load_documents("path/to/your/documents")
```

#### 2. Data Processing Pipeline
```python
from src.data_processing.pipeline import DataProcessingPipeline

pipeline = DataProcessingPipeline()
results = pipeline.process_documents(documents)
```

#### 3. Synthetic Data Generation
```python
from src.data_processing.generator import SyntheticDataGenerator

generator = SyntheticDataGenerator()
synthetic_data = generator.generate_synthetic_data(
    context="Your context here",
    num_samples=10
)
```

## Core Components

### Data Processing (LangChain-Powered)
- **Loader**: Multi-format document loading (PDF, CSV, JSON, TXT, DOCX)
- **Preprocessor**: Text cleaning and PII protection using Presidio
- **Chunker**: Intelligent text splitting with overlap using RecursiveCharacterTextSplitter
- **Embedder**: Azure OpenAI embeddings integration
- **Generator**: Synthetic data generation using Azure OpenAI with prompt templates
- **Pipeline**: End-to-end orchestration with manual control

### Vector Store (LangChain Integration)
- **Base Vector Store**: Abstract interface for vector operations
- **LangChain Vector Store**: Unified interface for Azure Search and FAISS
- **Vector Store Factory**: Factory pattern for creating vector store instances

### PII Protection
- **PIIProtector**: Presidio-based PII detection and anonymization
- **Configurable Entities**: Support for multiple PII types
- **Anonymization**: Replace PII with placeholders or remove entirely

## Configuration
Configuration settings are managed through the `config/settings.py` module, which loads environment variables and provides centralized configuration management.

## Key Benefits of LangChain Integration
- **Unified Interface**: Consistent API across different vector stores and LLMs
- **Rich Ecosystem**: Access to multiple document loaders and embedding models
- **Azure Integration**: Native support for Azure OpenAI and Azure Cognitive Search
- **Prompt Management**: Structured prompt templates for consistent generation
- **Flexibility**: Easy switching between different backends

## Troubleshooting

### Azure OpenAI Connection Issues
If you encounter connection errors:

```bash
# Verify your environment variables are set
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('Endpoint:', os.getenv('AZURE_OPENAI_ENDPOINT'))
print('API Key:', 'Set' if os.getenv('AZURE_OPENAI_API_KEY') else 'Not Set')
"
```

### Import Path Issues
If you encounter module import errors:

```bash
# Run from the project root directory
cd /c:/Users/andre/ai-models/synthetic-data-generator
python src/main.py
```

### Missing Dependencies
If you encounter missing package errors:

```bash
# Reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.