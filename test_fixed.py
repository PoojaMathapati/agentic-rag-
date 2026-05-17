#!/usr/bin/env python3
"""
Test script to verify the Pydantic compatibility fix
"""

print("🧪 Testing Pydantic Compatibility Fix...")
print("="*50)

# Test 1: Import LangChain packages
try:
    from langchain_openai import AzureChatOpenAI
    print("✅ AzureChatOpenAI import: SUCCESS")
except Exception as e:
    print(f"❌ AzureChatOpenAI import: FAILED - {e}")

try:
    from langchain_community.vectorstores import FAISS
    print("✅ FAISS import: SUCCESS")
except Exception as e:
    print(f"❌ FAISS import: FAILED - {e}")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ HuggingFaceEmbeddings import: SUCCESS")
except Exception as e:
    print(f"❌ HuggingFaceEmbeddings import: FAILED - {e}")

try:
    from langgraph.graph import StateGraph, END
    print("✅ LangGraph import: SUCCESS")
except Exception as e:
    print(f"❌ LangGraph import: FAILED - {e}")

# Test 2: Package versions
print("\n📦 Package Versions:")
import subprocess
packages = ['langchain', 'langchain-openai', 'langchain-community', 'pydantic']
for pkg in packages:
    try:
        result = subprocess.run(['pip', 'show', pkg], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = [line for line in result.stdout.split('\n') if line.startswith('Version:')]
            if version_line:
                version = version_line[0].split(':')[1].strip()
                print(f"  {pkg}: {version}")
    except:
        print(f"  {pkg}: Not found")

print("\n🎉 Compatibility Test Complete!")
print("If all imports show SUCCESS, your environment is fixed!")
