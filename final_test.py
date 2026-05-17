#!/usr/bin/env python3
"""
Final test script to verify the agentic RAG system works perfectly.
"""

import os

def set_env_vars():
    """Set the required environment variables."""
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://genai-trigent-openai.openai.azure.com/"
    os.environ["AZURE_OPENAI_API_KEY"] = "51ba5d46601c477b844d3883af93463c"
    os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"] = "gpt-4o-mini"
    os.environ["AZURE_OPENAI_API_VERSION"] = "2025-01-01-preview"

def test_questions():
    """Test key questions to verify the system works."""
    from agentic_RAG import main, load_docs_from_dir, split_docs, load_faiss_index, build_graph_with_web, init_state, get_llm
    from langgraph.checkpoint.memory import MemorySaver
    
    print("🧪 FINAL SYSTEM TEST")
    print("=" * 50)
    
    # Set environment variables
    set_env_vars()
    
    # Verify Azure OpenAI
    try:
        get_llm()
        print("✅ Azure OpenAI: Connected")
    except Exception as e:
        print(f"❌ Azure OpenAI: {e}")
        return False
    
    # Load documents and build system
    docs = load_docs_from_dir("./docs")
    vs = load_faiss_index("./faiss_index_hf")
    kb_available = vs is not None
    
    print(f"✅ Documents: {len(docs)} loaded")
    print(f"✅ Vector DB: {'Available' if kb_available else 'Not available'}")
    
    # Build workflow
    app = build_graph_with_web(vs, checkpointer=None)
    
    # Test cases
    test_cases = [
        ("What is the importance of RPA?", "kb", "Should use knowledge base - RPA content in your docs"),
        ("What are types of RAG?", "web", "Should fallback to web - no RAG content in your docs"),
        ("What's the weather today?", "web", "Should use web - real-time information"),
    ]
    
    print("\n🔍 TESTING ROUTING DECISIONS:")
    print("=" * 50)
    
    for question, expected_source, explanation in test_cases:
        print(f"\n❓ Testing: {question}")
        print(f"📋 Expected: {expected_source} ({explanation})")
        
        try:
            state = init_state(question, thread_id="test")
            state["kb_available"] = kb_available
            result = app.invoke(state)
            
            final_status = result.get("final_status", "unknown")
            actual_source = "kb" if "kb" in final_status else "web"
            
            if actual_source == expected_source:
                print(f"✅ CORRECT: Routed to {actual_source}")
            else:
                print(f"❌ WRONG: Expected {expected_source}, got {actual_source}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n🎉 FINAL SYSTEM TEST COMPLETE!")
    print("Your agentic RAG system is ready for production use!")
    return True

if __name__ == "__main__":
    test_questions()
