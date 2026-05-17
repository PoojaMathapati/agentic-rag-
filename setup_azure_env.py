#!/usr/bin/env python3
"""
Azure OpenAI Environment Setup Script
This script helps you configure the required environment variables for Azure OpenAI.
"""

import os
import sys

def setup_azure_environment():
    """Interactive setup for Azure OpenAI environment variables."""
    
    print("🔧 Azure OpenAI Environment Setup")
    print("=" * 50)
    print("This script will help you set up the required environment variables.")
    print("You'll need your Azure OpenAI credentials from the Azure portal.")
    print()
    
    # Check if environment variables already exist
    existing_vars = {
        'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
        'AZURE_OPENAI_API_KEY': os.getenv('AZURE_OPENAI_API_KEY'),
        'AZURE_OPENAI_CHAT_DEPLOYMENT': os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT'),
        'AZURE_OPENAI_API_VERSION': os.getenv('AZURE_OPENAI_API_VERSION')
    }
    
    print("Current environment variables:")
    for var, value in existing_vars.items():
        status = "✓ Set" if value else "✗ Not set"
        print(f"  {var}: {status}")
    
    print("\n" + "=" * 50)
    print("To set these variables manually, use one of these methods:")
    print()
    
    print("🔹 Method 1: PowerShell (Windows)")
    print("$env:AZURE_OPENAI_ENDPOINT=\"https://your-resource.openai.azure.com/\"")
    print("$env:AZURE_OPENAI_API_KEY=\"your-api-key\"")
    print("$env:AZURE_OPENAI_CHAT_DEPLOYMENT=\"your-chat-deployment-name\"")
    print("$env:AZURE_OPENAI_API_VERSION=\"2024-02-15-preview\"")
    
    print("\n🔹 Method 2: Command Prompt (Windows)")
    print("set AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
    print("set AZURE_OPENAI_API_KEY=your-api-key")
    print("set AZURE_OPENAI_CHAT_DEPLOYMENT=your-chat-deployment-name")
    print("set AZURE_OPENAI_API_VERSION=2024-02-15-preview")
    
    print("\n🔹 Method 3: Create .env file")
    print("Create a file named '.env' in this directory with:")
    print("AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
    print("AZURE_OPENAI_API_KEY=your-api-key")
    print("AZURE_OPENAI_CHAT_DEPLOYMENT=your-chat-deployment-name")
    print("AZURE_OPENAI_API_VERSION=2024-02-15-preview")
    
    print("\n" + "=" * 50)
    print("💡 Where to find your Azure OpenAI credentials:")
    print("1. Go to https://portal.azure.com")
    print("2. Navigate to your Azure OpenAI resource")
    print("3. Go to 'Keys and Endpoint' section")
    print("4. Copy the endpoint URL and one of the keys")
    print("5. Go to 'Model deployments' to find your deployment name")
    
    return all(existing_vars.values())

if __name__ == "__main__":
    if setup_azure_environment():
        print("\n✅ All Azure OpenAI environment variables are configured!")
        print("You can now run: python agentic_RAG.py")
    else:
        print("\n❌ Some environment variables are missing.")
        print("Please set them using one of the methods above, then try again.")
        sys.exit(1)
