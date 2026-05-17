#!/usr/bin/env python3
"""
Test script to verify the embeddings fix.
"""

import os
import sys

def test_embeddings():
    """Test the embeddings class."""
    try:
        from agentic_RAG import get_embeddings, HuggingFaceEmbeddings
        from langchain_core.embeddings import Embeddings
        
        print("🧪 Testing Embeddings Fix")
        print("=" * 40)
        
        # Test embeddings creation
        embeddings = get_embeddings()
        print(f"✅ Embeddings created: {type(embeddings)}")
        print(f"✅ Is LangChain compatible: {isinstance(embeddings, Embeddings)}")
        
        # Test query embedding
        test_query = "test query"
        query_emb = embeddings.embed_query(test_query)
        print(f"✅ Query embedding: {len(query_emb)} dimensions")
        
        # Test document embedding  
        test_docs = ["test document 1", "test document 2"]
        doc_embs = embeddings.embed_documents(test_docs)
        print(f"✅ Document embeddings: {len(doc_embs)} documents")
        
        print("\n🎉 All embedding tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_faiss_creation():
    """Test FAISS index creation."""
    try:
        from agentic_RAG import load_docs_from_dir, split_docs, build_faiss_index
        import tempfile
        import shutil
        
        print("\n🗂️  Testing FAISS Index Creation")
        print("=" * 40)
        
        # Load documents
        docs = load_docs_from_dir("./docs")
        print(f"✅ Documents loaded: {len(docs)}")
        
        if not docs:
            print("⚠️  No documents found, creating test document...")
            from langchain_core.documents import Document
            docs = [Document(page_content="Test document for FAISS", metadata={"source": "test.txt"})]
        
        # Split documents
        chunks = split_docs(docs)
        print(f"✅ Document chunks: {len(chunks)}")
        
        # Test FAISS index creation
        with tempfile.TemporaryDirectory() as temp_dir:
            vs = build_faiss_index(chunks, temp_dir)
            if vs:
                print("✅ FAISS index created successfully")
                
                # Test search
                results = vs.similarity_search("test", k=1)
                print(f"✅ Similarity search works: {len(results)} results")
                
                return True
            else:
                print("❌ FAISS index creation failed")
                return False
                
    except Exception as e:
        print(f"❌ FAISS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔧 Embeddings and FAISS Fix Verification")
    print("=" * 50)
    
    success = True
    
    # Test embeddings
    if not test_embeddings():
        success = False
    
    # Test FAISS
    if not test_faiss_creation():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Your system should work now.")
        print("✅ Ready to run: python agentic_RAG.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()
