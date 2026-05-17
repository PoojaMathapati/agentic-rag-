#!/usr/bin/env python3
"""
Test script to demonstrate KB vs Web segregation in the Agentic RAG system.
This script tests both knowledge base queries and web search queries.
"""

import os
from agentic_RAG import main, load_docs_from_dir, split_docs, load_faiss_index, build_graph_with_web, init_state, get_llm
from langgraph.checkpoint.memory import MemorySaver
from pathlib import Path

def test_query(question: str, app, kb_available: bool):
    """Test a single query and show the routing."""
    print("\n" + "=" * 80)
    print(f"🔍 TESTING QUERY: {question}")
    print("=" * 80)
    
    try:
        state = init_state(question, thread_id="test")
        state["kb_available"] = kb_available
        result = app.invoke(state)
        
        # Extract key information
        route = result.get("route", "unknown")
        final_status = result.get("final_status", "unknown")
        source_type = result.get("source_type", "Unknown")
        kb_files = result.get("kb_files_used", [])
        
        print(f"\n📊 RESULT SUMMARY:")
        print(f"   Route taken: {route}")
        print(f"   Answer source: {source_type}")
        print(f"   Final status: {final_status}")
        if kb_files:
            print(f"   KB files used: {', '.join(kb_files)}")
        
        return route, final_status, source_type
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return "error", "error", "error"

def main():
    """Main test function."""
    print("🧪 KB vs Web Segregation Test")
    print("=" * 50)
    
    # Check Azure OpenAI configuration
    try:
        get_llm()
        print("✅ Azure OpenAI configuration validated")
    except Exception as e:
        print(f"❌ Azure OpenAI configuration error: {e}")
        print("Please set your environment variables and try again.")
        return
    
    # Load documents and build system
    DOCS_DIR = "./docs"
    INDEX_DIR = "./faiss_index_hf"
    
    docs = load_docs_from_dir(DOCS_DIR)
    vs = load_faiss_index(INDEX_DIR)
    kb_available = vs is not None
    
    print(f"📚 Documents loaded: {len(docs)}")
    print(f"🗂️  Vector database available: {'✅ Yes' if kb_available else '❌ No'}")
    
    # Build workflow
    app = build_graph_with_web(vs, checkpointer=MemorySaver())
    app = app.with_config(configurable={"thread_id": "test"})
    
    # Test queries
    print("\n" + "=" * 50)
    print("🧪 RUNNING TESTS")
    print("=" * 50)
    
    # Knowledge Base queries (should route to KB)
    kb_queries = [
        "What are the benefits of business process automation?",
        "How did the investment firm improve efficiency?",
        "What security measures are important for RPA?",
        "Tell me about cost reduction through automation",
        "How does RPA help with operational efficiency?"
    ]
    
    # Web queries (should route to Web)
    web_queries = [
        "What is the current weather in New York?",
        "Who won the latest Nobel Prize in Physics?", 
        "What are the latest developments in quantum computing?",
        "Tell me about recent space missions to Mars",
        "What is the current price of Bitcoin?"
    ]
    
    print("\n📚 TESTING KNOWLEDGE BASE QUERIES:")
    print("-" * 40)
    kb_results = []
    for query in kb_queries[:3]:  # Test first 3 to save time
        route, status, source = test_query(query, app, kb_available)
        kb_results.append((query, route, status, source))
    
    print("\n🌐 TESTING WEB SEARCH QUERIES:")
    print("-" * 40)
    web_results = []
    for query in web_queries[:3]:  # Test first 3 to save time
        route, status, source = test_query(query, app, kb_available)
        web_results.append((query, route, status, source))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n📚 KNOWLEDGE BASE QUERIES:")
    for query, route, status, source in kb_results:
        correct = "✅" if route == "kb" and "kb" in status else "❌"
        print(f"   {correct} {query[:50]}... → Route: {route}, Status: {status}")
    
    print("\n🌐 WEB SEARCH QUERIES:")
    for query, route, status, source in web_results:
        correct = "✅" if route == "web" and "web" in status else "❌"
        print(f"   {correct} {query[:50]}... → Route: {route}, Status: {status}")
    
    # Calculate accuracy
    kb_correct = sum(1 for _, route, status, _ in kb_results if route == "kb" and "kb" in status)
    web_correct = sum(1 for _, route, status, _ in web_results if route == "web" and "web" in status)
    
    total_tests = len(kb_results) + len(web_results)
    total_correct = kb_correct + web_correct
    accuracy = (total_correct / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n🎯 OVERALL ACCURACY: {accuracy:.1f}% ({total_correct}/{total_tests})")
    print(f"   KB routing accuracy: {kb_correct}/{len(kb_results)}")
    print(f"   Web routing accuracy: {web_correct}/{len(web_results)}")
    
    if accuracy >= 80:
        print("\n🎉 Excellent! Your system correctly segregates KB and Web queries!")
    elif accuracy >= 60:
        print("\n👍 Good performance! Some fine-tuning may improve accuracy.")
    else:
        print("\n⚠️  The routing system may need adjustment.")

if __name__ == "__main__":
    main()
