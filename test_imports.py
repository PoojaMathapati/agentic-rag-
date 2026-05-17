#!/usr/bin/env python3
"""Test script to verify all dependencies are installed correctly."""

import sys

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name} failed to import: {e}")
        return False

def main():
    print("Testing imports for Agentic RAG application...")
    print("-" * 50)
    
    modules_to_test = [
        ("sentence_transformers", "sentence-transformers"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("langchain", "LangChain"),
        ("langchain_community", "LangChain Community"),
        ("langchain_openai", "LangChain OpenAI"),
        ("langchain_text_splitters", "LangChain Text Splitters"),
        ("langchain_core", "LangChain Core"),
        ("langgraph", "LangGraph"),
        ("faiss", "FAISS"),
    ]
    
    success_count = 0
    total_count = len(modules_to_test)
    
    for module, package in modules_to_test:
        if test_import(module, package):
            success_count += 1
    
    print("-" * 50)
    print(f"Import test results: {success_count}/{total_count} successful")
    
    if success_count == total_count:
        print("🎉 All dependencies are installed correctly!")
        return True
    else:
        print("❌ Some dependencies are missing. Please install them.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
