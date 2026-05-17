import os, glob, time, uuid, math, statistics as stats, re, pathlib
from typing import List, Dict, Any, Optional, Tuple, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import numpy as np
import torch
from pathlib import Path

# LangChain / LangGraph
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import AzureChatOpenAI
# from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

# Custom HuggingFace Embeddings class using sentence-transformers
from langchain_core.embeddings import Embeddings

class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-base-en"):
        print(f"[info] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            print(f"[error] Failed to embed documents: {e}")
            return []
    
    def embed_query(self, text: str) -> List[float]:
        if not text.strip():
            return []
        try:
            embedding = self.model.encode([text], show_progress_bar=False)
            return embedding[0].tolist()
        except Exception as e:
            print(f"[error] Failed to embed query: {e}")
            return []
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        return self.embed_query(text)


# ===================================================================
# 1. STATE + HELPERS
# ===================================================================

class RAGState(TypedDict):
    thread_id: str
    question: str
    rewritten: Optional[str]
    docs: List[Any]
    filtered_docs: List[Any]
    citations: List[str]
    answer: Optional[str]
    attempts: int
    need_more_raw: bool
    history: List[Any]
    path: List[str]
    cached_hit: bool
    kb_available: bool
    route: Optional[str]
    grounded: Optional[bool]
    contradiction: Optional[bool]
    final_status: Optional[str]
    source_type: Optional[str]
    kb_files_used: List[str]

def init_state(question: str, thread_id: str = "default") -> RAGState:
    return RAGState(
        thread_id=thread_id,
        question=question,
        rewritten=None,
        docs=[],
        filtered_docs=[],
        citations=[],
        answer=None,
        attempts=0,
        need_more_raw=False,
        history=[],
        path=[],
        cached_hit=False,
        kb_available=False,
        route=None,
        grounded=None,
        contradiction=None,
        final_status=None,
        source_type=None,
        kb_files_used=[]
    )

def debug_header(title: str):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def print_docs(docs, limit: int = 3):
    for i, d in enumerate(docs[:limit], 1):
        meta = getattr(d, "metadata", {}) or {}
        src  = os.path.basename(str(meta.get("source", "unknown")))
        snippet = (d.page_content or "")[:160].replace("\n", " ")
        print(f"  [{i}] {src} | {snippet}...")

# ===================================================================
# 2. LLM + EMBEDDINGS
# ===================================================================
def get_llm() -> AzureChatOpenAI:
    # Check for required environment variables
    required_vars = {
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_CHAT_DEPLOYMENT": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        print(f"\n❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please run 'python setup_azure_env.py' for setup instructions.")
        raise ValueError(f"Missing Azure OpenAI environment variables: {missing_vars}")
    
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        temperature=0.2,
        max_tokens=800,
    )

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

# ===================================================================
# 3. DOCUMENT PIPELINE (load + split + index)
# ===================================================================
def load_docs_from_dir(path: str) -> List[Document]:
    docs: List[Document] = []
    
    # Load from docs directory
    for pat in ("**/*.txt", "**/*.md"):
        for fp in glob.glob(os.path.join(path, pat), recursive=True):
            docs.extend(TextLoader(fp, encoding="utf-8").load())
    
    # Also load files in the root directory (like file_1.txt, file_2.txt)
    root_files = ["file_1.txt", "file_2.txt"]
    for filename in root_files:
        if os.path.exists(filename):
            try:
                docs.extend(TextLoader(filename, encoding="utf-8").load())
                print(f"[info] Loaded {filename} from root directory")
            except Exception as e:
                print(f"[warn] Failed to load {filename}: {e}")
    
    return docs

def split_docs(docs: List[Document], chunk_size: int = 1200, chunk_overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

from typing import Optional

def build_faiss_index(chunks: List[Document], index_dir: str) -> Optional[FAISS]:
    """Return a FAISS index or None if no chunks."""
    if not chunks:
        print("[info] No local documents found. Skipping FAISS build; KB retrieval will be disabled.")
        return None
    
    print(f"[info] Building FAISS index with {len(chunks)} chunks...")
    embeddings = get_embeddings()
    
    try:
        # Create FAISS index with proper embeddings
        vs = FAISS.from_documents(chunks, embeddings)
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        vs.save_local(index_dir)
        print(f"[info] FAISS index saved to {index_dir}")
        return vs
    except Exception as e:
        print(f"[error] Failed to build FAISS index: {e}")
        print(f"[debug] Embeddings type: {type(embeddings)}")
        return None

def load_faiss_index(index_dir: str) -> Optional[FAISS]:
    """Return a FAISS index if present, else None."""
    if not Path(index_dir).exists():
        print(f"[info] No existing FAISS index found at {index_dir}")
        return None
    
    print(f"[info] Loading FAISS index from {index_dir}...")
    embeddings = get_embeddings()
    try:
        vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        print(f"[info] FAISS index loaded successfully")
        return vs
    except Exception as e:
        print(f"[warn] Failed to load FAISS index at {index_dir}: {e}")
        print(f"[debug] Embeddings type: {type(embeddings)}")
        print("[info] Will rebuild index from documents...")
        return None

# ===================================================================
# 4. NODES
# ===================================================================
def node_rewrite(state: RAGState) -> RAGState:
    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the user's question to be concise and retrieval-friendly."),
            MessagesPlaceholder("history"),
            ("human", "Original question: {q}\nGive only the rewritten query.")
        ])
        chain = prompt | llm | (lambda m: m.content.strip())
        rewritten = chain.invoke({"q": state["question"], "history": []})
        debug_header("STEP 1: QUERY REWRITE")
        print("Original :", state["question"])
        print("Rewritten:", rewritten)
        new_state = dict(state)
        new_state["rewritten"] = rewritten
        return new_state
    except Exception as e:
        print(f"[error] Query rewrite failed: {e}")
        print("[info] Using original query")
        debug_header("STEP 1: QUERY REWRITE (FAILED)")
        print("Original :", state["question"])
        print("Rewritten:", state["question"])
        new_state = dict(state)
        new_state["rewritten"] = state["question"]
        return new_state

def node_router(state: RAGState) -> RAGState:
    kb_available = bool(state.get("kb_available", False))
    if not kb_available:
        route = "web"
        judge = "forced:web (no KB)"
    else:
        try:
            llm = get_llm()
            
            # Smart routing with clear decision criteria
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a router for a RAG system. Decide whether to search the knowledge base (kb) or web.

ROUTING STRATEGY: Always try 'kb' first unless the query CLEARLY needs real-time information.

Choose 'web' ONLY for:
- Real-time data: weather, current prices, live scores
- Recent events: "latest news", "today's", "current", "now", "recent"
- Time-sensitive: "what happened today", "breaking news"

Choose 'kb' for:
- Questions about RPA, automation, business processes
- Investment consulting and firm operations  
- Security measures and governance
- Cost reduction and efficiency topics
- Content that matches your uploaded documents

Choose 'web' for:
- Technical topics NOT in your documents (RAG, ML, AI concepts)
- General knowledge and explanations
- Academic or technical concepts

Examples:
✅ kb: "What is RPA?", "Types of automation", "How does business automation work?"
✅ web: "Today's weather", "Latest news", "Current Bitcoin price", "Types of RAG", "How does machine learning work?"

Default to 'kb' when uncertain. Reply only 'kb' or 'web'."""),
                ("human", "QUERY: {q}")
            ])
            
            out = (llm.invoke(prompt.format_messages(q=state["question"])).content or "").strip().lower()
            route = "kb" if out.startswith("kb") else "web"
            judge = out
        except Exception as e:
            print(f"[error] Router decision failed: {e}")
            print("[info] Defaulting to KB route")
            route = "kb"
            judge = "error:defaulted to kb"

    debug_header("STEP 1.5: ROUTER")
    print(f"Route: {route}  (judge: {judge})")
    print(f"Knowledge Base Available: {'✅ Yes' if kb_available else '❌ No'}")
    if route == "kb":
        print("📚 Will search your documents for the answer")
    else:
        print("🌐 Will search the web for the answer")
    new = dict(state)
    new["route"] = route
    return new


def node_retrieve(vs: Optional[FAISS], k: int = 6):
    if vs is None:
        # Return a no-op node that logs and continues
        def _noop(state: RAGState) -> RAGState:
            debug_header(f"STEP 2: RETRIEVE (attempt {state.get('attempts',0)+1})")
            print("KB disabled (no FAISS index). Skipping retrieval.")
            ns = dict(state)
            ns["docs"] = []
            ns["attempts"] = state.get("attempts", 0) + 1
            return ns
        return _noop

    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    def _inner(state: RAGState) -> RAGState:
        q = state.get("rewritten") or state["question"]
        docs = retriever.invoke(q)
        debug_header(f"STEP 2: RETRIEVE (attempt {state.get('attempts',0)+1})")
        print("Query used:", q)
        print_docs(docs)
        ns = dict(state)
        ns["docs"] = docs
        ns["attempts"] = state.get("attempts", 0) + 1
        return ns
    return _inner


# ---------- STEP 2.5: VALIDATION ----------
VAL_EMB_SIM_THRESH = 0.35
VAL_LEX_OVERLAP_MIN = 2
VAL_LLM_WEIGHT = 0.5
VAL_HEUR_WEIGHT = 0.5

KB_AVAILABLE_DEFAULT = True


def node_validate_retrieval(state: RAGState) -> RAGState:
    q = (state.get("rewritten") or state["question"]).strip()
    docs: List[Document] = state.get("docs", [])
    debug_header("STEP 2.5: VALIDATE RETRIEVAL")
    if not docs:
        print("No docs retrieved."); return state
    
    embeddings = get_embeddings()
    q_emb = embeddings.embed_query(q)
    if not q_emb:  # Handle embedding failure
        print("[warn] Query embedding failed, skipping validation")
        return state
        
    results = []
    for i, d in enumerate(docs):
        try:
            d_emb = embeddings.embed_query(d.page_content)
            if not d_emb:  # Handle embedding failure
                sim = 0.0
            else:
                sim = np.dot(q_emb, d_emb)/(np.linalg.norm(q_emb)*np.linalg.norm(d_emb))
            
            lex = len(set(q.lower().split()) & set(d.page_content.lower().split()))
            
            # Simplified scoring to reduce LLM calls
            heur_score = 0.6*max(0.0, sim) + 0.4*(1.0 if lex >= VAL_LEX_OVERLAP_MIN else 0.0)
            final_score = heur_score  # Use heuristic only for efficiency
            
            d.metadata["val_score"] = round(final_score,3)
            results.append((d, final_score))
            
        except Exception as e:
            print(f"[warn] Validation failed for doc {i}: {e}")
            # Keep document with low score if validation fails
            d.metadata["val_score"] = 0.1
            results.append((d, 0.1))
    
    results.sort(key=lambda x:x[1], reverse=True)
    for i,(d,s) in enumerate(results,1):
        src = os.path.basename(str(d.metadata.get('source', 'local')))
        print(f"  [{i}] {s:.2f} | {src}")
    new_state = dict(state); new_state["docs"]=[d for d,_ in results]
    return new_state

def node_grade_docs(state: RAGState) -> RAGState:
    q = state.get("rewritten") or state["question"]
    docs = state.get("docs", [])
    kept = []
    
    debug_header("STEP 3: GRADE CHUNKS")
    
    # More strict grading with LLM assistance
    for i, d in enumerate(docs):
        try:
            val_score = d.metadata.get("val_score", 0.0)
            
            # Higher threshold for better quality
            if val_score > 0.5:  # Increased threshold
                kept.append(d)
                print(f"  [{i+1}] Kept (score: {val_score:.2f})")
            elif val_score > 0.3:  # Moderate relevance - additional LLM check
                try:
                    llm = get_llm()
                    relevance_check = llm.invoke(ChatPromptTemplate.from_messages([
                        ("system", "Does this document chunk help answer the question? Reply 'YES' or 'NO' only."),
                        ("human", "QUESTION: {q}\n\nCHUNK: {chunk}")
                    ]).format_messages(q=q, chunk=d.page_content[:500])).content.strip().upper()
                    
                    if relevance_check.startswith("YES"):
                        kept.append(d)
                        print(f"  [{i+1}] Kept (score: {val_score:.2f}, LLM: relevant)")
                    else:
                        print(f"  [{i+1}] Filtered out (score: {val_score:.2f}, LLM: not relevant)")
                except:
                    # If LLM check fails, use score threshold
                    kept.append(d)
                    print(f"  [{i+1}] Kept (score: {val_score:.2f}, LLM check failed)")
            else:
                print(f"  [{i+1}] Filtered out (score: {val_score:.2f})")
        except Exception as e:
            print(f"[warn] Grading failed for doc {i}: {e}")
    
    # Only keep top documents if we have many
    if len(kept) > 4:
        kept = kept[:4]  # Limit to top 4 most relevant
        print(f"[info] Limited to top {len(kept)} documents")
    
    need_more = len(kept) < 1  # Need at least 1 good document
    print(f"Relevant kept: {len(kept)} | need_more={need_more}")
    new = dict(state); new["filtered_docs"]=kept; new["need_more_raw"]=need_more
    return new

def decide_after_grade_or_web(state: RAGState) -> str:
    debug_header("STEP 3.5: DECIDE NEXT")
    attempts = state.get("attempts", 0)
    need_more = state.get("need_more_raw", False)
    kept_docs = len(state.get("filtered_docs", []))
    route = state.get("route", "kb")
    
    # Smart decision: Quality over quantity
    if kept_docs >= 1:
        # Always do semantic relevance check for better decisions
        try:
            llm = get_llm()
            
            sample_content = ""
            for doc in state.get("filtered_docs", []):
                sample_content += f"Document: {doc.page_content[:400]}\n\n"
            
            question = state.get("question", "")
            
            # Comprehensive relevance evaluation
            relevance_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are evaluating if documents can answer a user's question.

Respond 'YES' if:
- Documents contain information that directly answers the question
- Documents have relevant content even if not perfectly matching
- The topic/subject matter aligns with the question

Respond 'NO' if:
- Documents are about completely different topics
- No relevant information is present
- The question requires information not in the documents

Be strict but fair in your evaluation."""),
                ("human", "QUESTION: {question}\n\nDOCUMENTS:\n{documents}\n\nCan these documents adequately answer the question? Reply YES or NO only.")
            ])
            
            relevance_check = llm.invoke(relevance_prompt.format_messages(
                question=question,
                documents=sample_content[:2000]
            )).content.strip().upper()
            
            if relevance_check.startswith("YES"):
                print(f"✅ Found {kept_docs} relevant documents - generating answer from knowledge base")
                return "final"
            else:
                print(f"❌ Found {kept_docs} documents but they don't answer the question - trying web search")
                return "websearch"
                
        except Exception as e:
            print(f"[error] Relevance check failed: {e}")
            # Conservative fallback: try web search if uncertain
            print(f"⚠️  Relevance check failed - trying web search for safety")
            return "websearch"
    
    # If KB search didn't find enough and we haven't tried web yet
    if route == "kb" and kept_docs < 2 and attempts >= 1:
        print(f"KB search found only {kept_docs} relevant documents - trying web search as fallback")
        return "websearch"
    
    # Standard retry logic
    if need_more and attempts >= 1:
        return "websearch"
    if need_more:
        return "requery"
    
    return "final"

def node_websearch(state: RAGState) -> RAGState:
    q = state.get("rewritten") or state["question"]
    debug_header("STEP W: WEB SEARCH")
    print("Query:", q)
    
    try:
        search_tool = DuckDuckGoSearchRun()
        results = search_tool.run(q)
        print("Results preview:", results[:300])
        doc = Document(page_content=results, metadata={"source":"web"})
        new = dict(state)
        new["filtered_docs"] = [doc]
        return new
    except Exception as e:
        print(f"[error] Web search failed: {e}")
        print("[info] Proceeding without web search results")
        doc = Document(
            page_content=f"Web search failed for query: {q}. Please try a different approach.",
            metadata={"source": "web_error"}
        )
        new = dict(state)
        new["filtered_docs"] = [doc]
        return new

def node_generate(state: RAGState) -> RAGState:
    q = state.get("rewritten") or state["question"]
    ctx_docs = state.get("filtered_docs") or []
    
    debug_header("STEP 4: GENERATE")
    
    if not ctx_docs:
        answer = "I don't have enough information to answer your question. Please try rephrasing or asking about a different topic."
    else:
        ctx = "\n\n".join(f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in ctx_docs)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question based ONLY on the provided context. Always cite your sources. If the context doesn't contain enough information, say so clearly."),
            ("human", "Question: {q}\n\nContext:\n{ctx}")
        ])
        
        try:
            llm = get_llm()
            answer = llm.invoke(prompt.format_messages(q=q, ctx=ctx)).content
        except Exception as e:
            print(f"[error] Answer generation failed: {e}")
            answer = f"I apologize, but I encountered an error while generating an answer to your question: {q}. Please try again."
    
    print("Answer preview:", answer[:300])
    new = dict(state)
    new["answer"] = answer
    return new

def node_groundedness_check(state: RAGState) -> RAGState:
    ans = (state.get("answer") or "").strip()
    ctx_docs = state.get("filtered_docs") or []
    
    debug_header("STEP 4.5: GROUNDEDNESS CHECK")
    
    if not ans:
        print("Verdict: ungrounded (no answer)")
        new = dict(state)
        new["grounded"] = False
        return new
    
    if not ctx_docs:
        print("Verdict: ungrounded (no context)")
        new = dict(state)
        new["grounded"] = False
        return new
    
    try:
        ctx = "\n".join(d.page_content for d in ctx_docs)
        llm = get_llm()
        verdict = llm.invoke(ChatPromptTemplate.from_messages([
            ("system", "Reply 'grounded' if ANSWER is supported by CONTEXT, else 'ungrounded'."),
            ("human", "Q: {q}\n\nCONTEXT:\n{ctx}\n\nANSWER:\n{ans}")
        ]).format_messages(q=state["question"], ctx=ctx, ans=ans)).content.strip().lower()
        
        grounded = verdict.startswith("grounded")
        print("Verdict:", verdict)
        
    except Exception as e:
        print(f"[error] Groundedness check failed: {e}")
        print("Verdict: grounded (defaulted due to error)")
        grounded = True  # Default to grounded if check fails
    
    new = dict(state)
    new["grounded"] = grounded
    return new

def node_contradiction_check(state: RAGState) -> RAGState:
    debug_header("STEP 4.6: CONTRADICTION CHECK")
    
    answer = state.get("answer", "").strip()
    filtered_docs = state.get("filtered_docs", [])
    
    if not answer or not filtered_docs:
        print("No answer or sources to check for contradictions")
        new_state = dict(state)
        new_state["contradiction"] = False
        return new_state
    
    try:
        llm = get_llm()
        
        # Create context from source documents
        context = "\n\n".join([
            f"Source {i+1}: {doc.page_content[:500]}" 
            for i, doc in enumerate(filtered_docs)
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the ANSWER for internal contradictions or conflicts with the SOURCE CONTEXT.

Look for:
- Contradictory facts or numbers
- Conflicting statements within the answer
- Information that conflicts with the sources
- Logical inconsistencies

Reply 'contradiction' if you find any conflicts, 'consistent' if no contradictions."""),
            ("human", "QUESTION: {question}\n\nSOURCE CONTEXT:\n{context}\n\nANSWER:\n{answer}")
        ])
        
        result = llm.invoke(prompt.format_messages(
            question=state["question"],
            context=context,
            answer=answer
        )).content.strip().lower()
        
        has_contradiction = result.startswith("contradiction")
        
        if has_contradiction:
            print("⚠️  Potential contradiction detected in answer")
            print("🔍 Recommendation: Review answer for conflicting information")
        else:
            print("✅ No contradictions found - answer is internally consistent")
        
        new_state = dict(state)
        new_state["contradiction"] = has_contradiction
        return new_state
        
    except Exception as e:
        print(f"[error] Contradiction check failed: {e}")
        print("⚠️  Proceeding without contradiction validation")
        new_state = dict(state)
        new_state["contradiction"] = False
        return new_state

def node_finalize_status(state: RAGState) -> RAGState:
    debug_header("FINAL STATUS")
    
    filtered_docs = state.get("filtered_docs", [])
    grounded = state.get("grounded", False)
    route = state.get("route", "unknown")
    
    # Count sources by type
    kb_sources = []
    web_sources = []
    
    for doc in filtered_docs:
        source = doc.metadata.get("source", "unknown")
        if source == "web":
            web_sources.append(source)
        elif source not in ["web", "web_error"]:
            kb_sources.append(os.path.basename(source))
    
    # Determine status based on source and grounding
    if web_sources:
        status = "answered_from_web"
        source_icon = "🌐"
        source_desc = "Web Search"
    elif kb_sources:
        status = "answered_from_kb" 
        source_icon = "📚"
        source_desc = "Knowledge Base"
    else:
        status = "no_sources"
        source_icon = "❓"
        source_desc = "No Sources"
    
    # Only mark as low confidence if actually ungrounded or contradictory
    if not grounded:
        status = "low_confidence"
        source_icon = "⚠️"
    elif state.get("contradiction", False):
        status = "low_confidence"
        source_icon = "⚠️"
    
    print(f"Status: {status}")
    
    # Clear and accurate route reporting
    if web_sources and route == "kb":
        print(f"🔄 Route taken: {route} → web (smart fallback)")
        print(f"📋 Decision: Tried KB first, used Web as fallback for better results")
    elif web_sources:
        print(f"🌐 Route taken: {route} (direct web search)")
        print(f"📋 Decision: Routed directly to web for real-time information")
    else:
        print(f"📚 Route taken: {route} (knowledge base)")
        print(f"📋 Decision: Found relevant information in your documents")
    
    print(f"Answer source: {source_icon} {source_desc}")
    
    if kb_sources:
        print(f"Knowledge Base files used: {', '.join(set(kb_sources))}")
    if web_sources:
        print(f"Web search performed: Yes")
    
    print(f"Total sources: {len(filtered_docs)}")
    
    # Consistent confidence reporting
    is_confident = grounded and not state.get("contradiction", False)
    confidence_level = "High" if is_confident else "Low"
    print(f"Answer confidence: {confidence_level}")
    
    new = dict(state)
    new["final_status"] = status
    new["source_type"] = source_desc
    new["kb_files_used"] = list(set(kb_sources))
    return new

# ===================================================================
# 5. GRAPH BUILDER
# ===================================================================
def build_graph_with_web(vs: Optional[FAISS], checkpointer=None):
    g = StateGraph(RAGState)

    # Nodes
    g.add_node("rewrite",    node_rewrite)              # STEP 1
    g.add_node("router",     node_router)               # STEP 1.5
    g.add_node("retrieve",   node_retrieve(vs))         # STEP 2
    g.add_node("validate",   node_validate_retrieval)   # STEP 2.5
    g.add_node("grade",      node_grade_docs)           # STEP 3
    g.add_node("web",        node_websearch)            # STEP W
    g.add_node("generate",   node_generate)             # STEP 4
    g.add_node("grounded",   node_groundedness_check)   # STEP 4.5
    g.add_node("contradict", node_contradiction_check)  # STEP 4.6
    g.add_node("finalize",   node_finalize_status)      # FINAL

    # Entry + edges with direct web routing
    g.set_entry_point("rewrite")
    g.add_edge("rewrite", "router")
    
    # Direct routing: KB goes through full pipeline, Web goes directly to search
    g.add_conditional_edges("router", lambda s: s.get("route", "kb"),
                            {"kb": "retrieve", "web": "web"})
    
    # KB pathway (full pipeline)
    g.add_edge("retrieve", "validate")
    g.add_edge("validate", "grade")
    g.add_conditional_edges(
        "grade",
        decide_after_grade_or_web,   # returns 'requery' | 'websearch' | 'final'
        {"requery": "retrieve", "websearch": "web", "final": "generate"},
    )
    
    # Both KB and Web routes converge at generate
    g.add_edge("web", "generate")
    g.add_edge("generate", "grounded")
    
    # After generation, check grounding
    g.add_conditional_edges(
        "grounded",
        lambda s: "contradict" if s.get("grounded") else "web",
        {"contradict": "contradict", "web": "web"},
    )
    g.add_edge("contradict", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer) if checkpointer else g.compile()


# ===================================================================
# 6. MAIN DEMO
# ===================================================================
def main():
    """Main function to run the agentic RAG system."""
    DOCS_DIR = "./docs"
    INDEX_DIR = "./faiss_index_hf"  # Updated to match your existing index
    Path(DOCS_DIR).mkdir(exist_ok=True)

    print("🤖 Agentic RAG System Starting...")
    print("=" * 50)
    
    # Check if Azure OpenAI is configured
    try:
        get_llm()  # This will raise an error if not configured
        print("✅ Azure OpenAI configuration validated")
    except Exception as e:
        print(f"❌ Azure OpenAI configuration error: {e}")
        print("\n📋 Please set your environment variables:")
        print("   - AZURE_OPENAI_ENDPOINT")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - AZURE_OPENAI_CHAT_DEPLOYMENT")
        print("\n💡 Run 'python setup_azure_env.py' for detailed instructions")
        return

    # Build / load index if docs exist
    docs = load_docs_from_dir(DOCS_DIR)
    chunks = split_docs(docs) if docs else []
    vs = None
    
    if chunks:
        print(f"📚 Found {len(docs)} documents, creating {len(chunks)} chunks")
        vs = build_faiss_index(chunks, INDEX_DIR)
    else:
        # Try load an existing index if already built earlier
        vs = load_faiss_index(INDEX_DIR)

    kb_available = vs is not None
    if not kb_available:
        print("⚠️  Running without a KB index; will use web search for all queries.")
    else:
        print("✅ Knowledge base ready for queries")

    print("\n🚀 Building agentic workflow...")
    # Use in-memory checkpointer without persistence to avoid serialization issues
    app = build_graph_with_web(vs, checkpointer=None)
    # app = app.with_config(configurable={"thread_id": "demo_thread"})

    # Interactive question input
    print("\n" + "=" * 60)
    print("🤖 INTERACTIVE AGENTIC RAG SYSTEM")
    print("=" * 60)
    
    # Show sample questions for inspiration
    sample_questions = [
        "Tell me about business process automation",
        "What are the benefits of RPA implementation?", 
        "How can investment consulting firms improve their operations?",
        "What security measures should be considered for RPA?",
        "How did the investment firm reduce processing time?",
        "What are the cost savings from automation?"
    ]
    
    print("\n💡 Sample questions you can ask:")
    for i, q in enumerate(sample_questions, 1):
        print(f"   {i}. {q}")
    
    print(f"\n📚 Knowledge Base: {len(docs) if docs else 0} documents loaded")
    print("🌐 Web Search: Available as fallback")
    print("\n" + "=" * 60)
    
    while True:
        try:
            print("\n❓ Enter your question (or 'quit' to exit):")
            question = input("You: ").strip()
            
            if not question:
                print("⚠️  Please enter a question.")
                continue
                
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Agentic RAG System!")
                break
            
            print("\n" + "=" * 60)
            print(f"🔍 Processing: {question}")
            print("=" * 60)
            
            state = init_state(question, thread_id="session")
            state["kb_available"] = kb_available
            result = app.invoke(state)

            print("\n" + "=" * 60)
            print("🎯 FINAL ANSWER")
            print("=" * 60)
            print(result.get("answer", "No answer generated"))
            
            print("\n" + "=" * 60)
            print("📊 ANSWER DETAILS")
            print("=" * 60)
            source_type = result.get('source_type', 'Unknown')
            final_status = result.get('final_status', 'unknown')
            kb_files = result.get('kb_files_used', [])
            
            if final_status == "answered_from_kb":
                print("✅ KNOWLEDGE BASE ANSWER")
                print(f"📚 Source: Your uploaded documents")
                if kb_files:
                    print(f"📄 Files used: {', '.join(kb_files)}")
            elif final_status == "answered_from_web":
                print("✅ WEB SEARCH ANSWER") 
                print(f"🌐 Source: Internet search results")
                print("ℹ️  Information may be more current but less specific to your documents")
            else:
                print(f"ℹ️  Status: {final_status}")
            
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing your question: {e}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main()
