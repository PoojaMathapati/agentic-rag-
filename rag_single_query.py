#!/usr/bin/env python3
"""
Single-query version of the Agentic RAG system.
Usage: python rag_single_query.py "Your question here"
"""

import sys
import os
from agentic_RAG import main, load_docs_from_dir, split_docs, load_faiss_index, build_graph_with_web, init_state, get_llm
from langgraph.checkpoint.memory import MemorySaver
from pathlib import Path

def run_single_query(question: str):
    """Run a single query through the RAG system."""
    
    print("🤖 Agentic RAG System - Single Query Mode")
    print("=" * 50)
    
    # Check Azure OpenAI configuration
    try:
        get_llm()
        print("✅ Azure OpenAI configuration validated")
    except Exception as e:
        print(f"❌ Azure OpenAI configuration error: {e}")
        return
    
    # Load documents and index
    DOCS_DIR = "./docs"
    INDEX_DIR = "./faiss_index_hf"
    
    docs = load_docs_from_dir(DOCS_DIR)
    vs = load_faiss_index(INDEX_DIR)
    kb_available = vs is not None
    
    if kb_available:
        print(f"📚 Knowledge base loaded: {len(docs)} documents")
    else:
        print("⚠️  No knowledge base found, will use web search")
    
    # Build and run workflow
    app = build_graph_with_web(vs, checkpointer=MemorySaver())
    app = app.with_config(configurable={"thread_id": "single_query"})
    
    print(f"\n🔍 Processing: {question}")
    print("=" * 50)
    
    try:
        state = init_state(question, thread_id="single_query")
        state["kb_available"] = kb_available
        result = app.invoke(state)
        
        print("\n" + "=" * 50)
        print("🎯 FINAL ANSWER")
        print("=" * 50)
        print(result.get("answer", "No answer generated"))
        print("\n" + "=" * 50)
        print(f"📊 Status: {result.get('final_status', 'unknown')}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python rag_single_query.py \"Your question here\"")
        print("\nExample:")
        print('python rag_single_query.py "What are the benefits of business process automation?"')
        sys.exit(1)
    
    question = sys.argv[1]
    run_single_query(question)

if __name__ == "__main__":
    main()
