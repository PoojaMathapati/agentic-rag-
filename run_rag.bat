@echo off
echo Setting Azure OpenAI environment variables...
set AZURE_OPENAI_ENDPOINT=https://genai-trigent-openai.openai.azure.com/
set AZURE_OPENAI_API_KEY=51ba5d46601c477b844d3883af93463c
set AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
set AZURE_OPENAI_API_VERSION=2025-01-01-preview

echo.
echo Starting Interactive Agentic RAG System...
echo You can ask questions and get answers from your knowledge base!
echo.
python agentic_RAG.py
