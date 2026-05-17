#!/usr/bin/env python3
"""Test script to validate RAG components without requiring Azure OpenAI."""

import os
from pathlib import Path
from agentic_RAG import (
    load_docs_from_dir, split_docs, get_embeddings, 
    build_faiss_index, load_faiss_index
)

def test_document_loading():
    """Test document loading from the docs directory."""
    print("🔍 Testing Document Loading...")
    docs_dir = "./docs"
    docs = load_docs_from_dir(docs_dir)
    
    if not docs:
        print("❌ No documents found in docs directory")
        return False
    
    print(f"✅ Loaded {len(docs)} documents")
    for i, doc in enumerate(docs):
        filename = os.path.basename(doc.metadata.get('source', 'unknown'))
        content_preview = doc.page_content[:100].replace('\n', ' ')
        print(f"   [{i+1}] {filename}: {content_preview}...")
    
    return True

def test_document_splitting():
    """Test document chunking."""
    print("\n📝 Testing Document Splitting...")
    docs_dir = "./docs"
    docs = load_docs_from_dir(docs_dir)
    
    if not docs:
        print("❌ No documents to split")
        return False
    
    chunks = split_docs(docs)
    print(f"✅ Created {len(chunks)} chunks from {len(docs)} documents")
    
    # Show sample chunks
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk.page_content[:150].replace('\n', ' ')
        print(f"   Chunk {i+1}: {preview}...")
    
    return True

def test_embeddings():
    """Test embedding generation."""
    print("\n🧠 Testing Embeddings...")
    
    try:
        embeddings = get_embeddings()
        print(f"✅ Embedding model loaded: {embeddings.model_name}")
        
        # Test query embedding
        test_query = "What is business process automation?"
        query_emb = embeddings.embed_query(test_query)
        
        if query_emb:
            print(f"✅ Query embedding generated: {len(query_emb)} dimensions")
        else:
            print("❌ Failed to generate query embedding")
            return False
        
        # Test document embedding
        test_docs = ["Business process automation improves efficiency.", "RPA helps reduce costs."]
        doc_embs = embeddings.embed_documents(test_docs)
        
        if doc_embs and len(doc_embs) == 2:
            print(f"✅ Document embeddings generated: {len(doc_embs)} documents")
        else:
            print("❌ Failed to generate document embeddings")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False

def test_faiss_index():
    """Test FAISS index creation and loading."""
    print("\n🗂️  Testing FAISS Index...")
    
    docs_dir = "./docs"
    index_dir = "./test_faiss_index"
    
    # Load and split documents
    docs = load_docs_from_dir(docs_dir)
    if not docs:
        print("❌ No documents for indexing")
        return False
    
    chunks = split_docs(docs)
    print(f"📚 Processing {len(chunks)} chunks...")
    
    # Build index
    try:
        vs = build_faiss_index(chunks, index_dir)
        if vs:
            print("✅ FAISS index built successfully")
        else:
            print("❌ Failed to build FAISS index")
            return False
    except Exception as e:
        print(f"❌ FAISS index creation failed: {e}")
        return False
    
    # Test search
    try:
        test_query = "business automation benefits"
        results = vs.similarity_search(test_query, k=3)
        print(f"✅ Similarity search returned {len(results)} results")
        
        for i, result in enumerate(results):
            preview = result.page_content[:100].replace('\n', ' ')
            source = os.path.basename(result.metadata.get('source', 'unknown'))
            print(f"   [{i+1}] {source}: {preview}...")
            
    except Exception as e:
        print(f"❌ Similarity search failed: {e}")
        return False
    
    # Test loading index
    try:
        vs_loaded = load_faiss_index(index_dir)
        if vs_loaded:
            print("✅ FAISS index loaded successfully")
        else:
            print("❌ Failed to load FAISS index")
            return False
    except Exception as e:
        print(f"❌ FAISS index loading failed: {e}")
        return False
    
    # Cleanup test index
    import shutil
    if Path(index_dir).exists():
        shutil.rmtree(index_dir)
        print("🧹 Cleaned up test index")
    
    return True

def test_retrieval_pipeline():
    """Test the complete retrieval pipeline."""
    print("\n🔄 Testing Complete Retrieval Pipeline...")
    
    docs_dir = "./docs"
    index_dir = "./faiss_index_hf"
    
    # Check if existing index exists
    if Path(index_dir).exists():
        print("📁 Using existing FAISS index...")
        try:
            vs = load_faiss_index(index_dir)
            if not vs:
                print("❌ Failed to load existing index")
                return False
        except Exception as e:
            print(f"❌ Error loading existing index: {e}")
            return False
    else:
        print("🔨 Building new FAISS index...")
        docs = load_docs_from_dir(docs_dir)
        if not docs:
            print("❌ No documents found")
            return False
        
        chunks = split_docs(docs)
        vs = build_faiss_index(chunks, index_dir)
        if not vs:
            print("❌ Failed to build index")
            return False
    
    # Test various queries
    test_queries = [
        "business process automation",
        "RPA security measures", 
        "investment consulting firm benefits",
        "cost reduction through automation"
    ]
    
    print("🔍 Testing queries...")
    for i, query in enumerate(test_queries):
        try:
            results = vs.similarity_search(query, k=2)
            print(f"   [{i+1}] '{query}' → {len(results)} results")
            if results:
                best_result = results[0]
                preview = best_result.page_content[:80].replace('\n', ' ')
                source = os.path.basename(best_result.metadata.get('source', 'unknown'))
                print(f"       Best: {source} - {preview}...")
        except Exception as e:
            print(f"   [{i+1}] Query failed: {e}")
            return False
    
    print("✅ Retrieval pipeline test completed successfully!")
    return True

def main():
    """Run all tests."""
    print("🧪 RAG Components Test Suite")
    print("=" * 50)
    
    tests = [
        ("Document Loading", test_document_loading),
        ("Document Splitting", test_document_splitting), 
        ("Embeddings", test_embeddings),
        ("FAISS Index", test_faiss_index),
        ("Retrieval Pipeline", test_retrieval_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All RAG components are working correctly!")
        print("Your system is ready for Azure OpenAI integration.")
    else:
        print("⚠️  Some components need attention before proceeding.")
    
    return passed == len(results)

if __name__ == "__main__":
    main()
