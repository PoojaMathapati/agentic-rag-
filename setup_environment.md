# Agentic RAG Setup Guide

## Dependencies Installation

Your agentic RAG application requires several Python packages. Here are the installation steps:

### Method 1: Using pip (Recommended)
```bash
pip install sentence-transformers torch numpy langchain langchain-community langchain-openai langchain-text-splitters langchain-core langgraph faiss-cpu duckduckgo-search
```

### Method 2: Using requirements.txt
```bash
pip install -r requirements.txt
```

### Method 3: Individual package installation
```bash
pip install sentence-transformers
pip install torch
pip install langchain
pip install langchain-community
pip install langchain-openai
pip install langchain-text-splitters
pip install langchain-core
pip install langgraph
pip install faiss-cpu
pip install duckduckgo-search
```

## Azure OpenAI Configuration

Your application uses Azure OpenAI. You need to set these environment variables:

```bash
# Set these environment variables
AZURE_OPENAI_CHAT_DEPLOYMENT=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_KEY=your_api_key
```

### Windows PowerShell:
```powershell
$env:AZURE_OPENAI_CHAT_DEPLOYMENT="your_deployment_name"
$env:AZURE_OPENAI_API_VERSION="2024-02-15-preview"
$env:AZURE_OPENAI_ENDPOINT="your_azure_endpoint"
$env:AZURE_OPENAI_API_KEY="your_api_key"
```

### Windows Command Prompt:
```cmd
set AZURE_OPENAI_CHAT_DEPLOYMENT=your_deployment_name
set AZURE_OPENAI_API_VERSION=2024-02-15-preview
set AZURE_OPENAI_ENDPOINT=your_azure_endpoint
set AZURE_OPENAI_API_KEY=your_api_key
```

## Testing the Installation

Run the test script to verify all dependencies are installed:
```bash
python test_imports.py
```

## Running the Application

Once all dependencies are installed and environment variables are set:
```bash
python agentic_RAG.py
```

## Troubleshooting

1. **ModuleNotFoundError**: Make sure you're using the correct Python environment
2. **Azure OpenAI errors**: Verify your environment variables are set correctly
3. **FAISS issues**: Try installing `faiss-cpu` instead of `faiss-gpu` if you don't have CUDA
4. **PowerShell issues**: Try using Command Prompt instead of PowerShell
