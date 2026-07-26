import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import json
import numpy as np
import requests
import argparse
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

PDF_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_TOP_K = 4


def log(message: str, level: str = "INFO"):
    """Simple logging function."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {level} - {message}")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF file using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        return text.strip()
    except Exception as e:
        log(f"Error reading PDF {pdf_path}: {str(e)}", "ERROR")
        return ""


def load_pdf_documents(pdf_dir: Path) -> List[Dict[str, Any]]:
    """Load all PDF documents from directory."""
    documents = []
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        log(f"No PDF files found in {pdf_dir}", "WARNING")
        return documents
    
    log(f"Found {len(pdf_files)} PDF file(s)")
    
    for pdf_file in pdf_files:
        log(f"Loading PDF: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        if text:
            documents.append({
                "page_content": text,
                "metadata": {
                    "source": pdf_file.name,
                    "page": 1,
                    "file_path": str(pdf_file)
                }
            })
            log(f"Loaded {len(text)} characters from {pdf_file.name}")
    
    log(f"Total documents loaded: {len(documents)}")
    return documents


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks with overlap."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        if end < text_length:
            last_space = chunk.rfind(' ')
            if last_space > chunk_size // 2:
                end = start + last_space
                chunk = text[start:end]
        
        chunks.append(chunk)
        start = end - overlap
        
        if start >= text_length:
            break
    
    return chunks


def split_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split documents into chunks."""
    split_docs = []
    
    for doc in documents:
        chunks = split_text(doc["page_content"])
        for i, chunk in enumerate(chunks):
            split_docs.append({
                "page_content": chunk,
                "metadata": {
                    **doc["metadata"],
                    "chunk": i + 1,
                    "total_chunks": len(chunks)
                }
            })
    
    log(f"Splitted {len(documents)} document(s) into {len(split_docs)} chunk(s)")
    return split_docs


class SimpleEmbeddings:
    """Simple embedding using character-level hashing for demonstration."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def embed_query(self, text: str) -> List[float]:
        """Create a simple embedding using character frequency."""
        return self._create_embedding(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts."""
        return [self._create_embedding(text) for text in texts]
    
    def _create_embedding(self, text: str) -> List[float]:
        """Create a simple embedding."""
        embedding = [0.0] * self.dimension
        
        for i, char in enumerate(text.lower()):
            idx = i % self.dimension
            embedding[idx] += ord(char) / 256
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding


class SimpleVectorStore:
    """Simple vector store using numpy for similarity search."""
    
    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.index_path = persist_dir / "index.json"
        self.embeddings: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add documents and embeddings."""
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
    
    def save(self):
        """Save to disk."""
        data = {
            "embeddings": self.embeddings,
            "documents": self.documents
        }
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        log(f"Vector store saved to {self.index_path}")
    
    def load(self) -> bool:
        """Load from disk."""
        if not self.index_path.exists():
            return False
        
        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.embeddings = data.get("embeddings", [])
            self.documents = data.get("documents", [])
            log(f"Loaded {len(self.documents)} documents from vector store")
            return True
        except Exception as e:
            log(f"Error loading vector store: {str(e)}", "ERROR")
            return False
    
    def similarity_search(self, query_embedding: List[float], k: int = 4) -> List[Dict[str, Any]]:
        """Find similar documents."""
        if not self.embeddings:
            return []
        
        query_vec = np.array(query_embedding)
        doc_vecs = np.array(self.embeddings)
        
        similarities = np.dot(doc_vecs, query_vec) / (
            np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)
        )
        
        top_indices = np.argsort(similarities)[::-1][:k]
        results = [self.documents[i] for i in top_indices if similarities[i] > 0.1]
        return results


class DeepSeekLLM:
    """Simple DeepSeek LLM client."""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
    
    def invoke(self, prompt: str) -> str:
        """Call DeepSeek API to generate response."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise ValueError(f"API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"]


def format_context(documents: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into context string."""
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.get("metadata", {}).get("source", "unknown")
        chunk = doc.get("metadata", {}).get("chunk", "unknown")
        content = doc.get("page_content", "")
        context_parts.append(f"[Document {i}]\nSource: {source}\nChunk: {chunk}\nContent:\n{content}\n")
    return "\n".join(context_parts)


def initialize_rag(recreate: bool = False):
    """Initialize RAG system."""
    index_path = VECTOR_STORE_DIR / "index.json"
    
    embedding_model = SimpleEmbeddings(dimension=384)
    
    if recreate or not index_path.exists():
        log("Creating new vector store from PDF documents")
        
        documents = load_pdf_documents(PDF_DIR)
        if not documents:
            log("No documents loaded", "ERROR")
            sys.exit(1)
        
        split_docs = split_documents(documents)
        texts = [doc["page_content"] for doc in split_docs]
        
        log(f"Creating embeddings for {len(texts)} chunks")
        embeddings = embedding_model.embed_documents(texts)
        
        vector_store = SimpleVectorStore(VECTOR_STORE_DIR)
        vector_store.add_documents(split_docs, embeddings)
        vector_store.save()
    else:
        log("Loading existing vector store")
        vector_store = SimpleVectorStore(VECTOR_STORE_DIR)
        if not vector_store.load():
            log("Failed to load vector store, recreating", "WARNING")
            return initialize_rag(recreate=True)
    
    if DEEPSEEK_API_KEY:
        llm = DeepSeekLLM(DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_API_BASE)
        log(f"DeepSeek LLM initialized: {DEEPSEEK_MODEL}")
    else:
        llm = None
        log("DEEPSEEK_API_KEY not found. Will use retrieval only mode.", "WARNING")
    
    def retriever(query: str) -> List[Dict[str, Any]]:
        query_embedding = embedding_model.embed_query(query)
        return vector_store.similarity_search(query_embedding, k=RETRIEVAL_TOP_K)
    
    return llm, retriever


def generate_answer(llm, retriever, question: str) -> Dict[str, Any]:
    """Generate answer for a question."""
    log(f"Processing question: {question[:50]}...")
    
    retrieved_docs = retriever(question)
    
    if not retrieved_docs:
        log("No documents retrieved", "WARNING")
        if llm:
            answer = llm.invoke(question)
            return {"answer": answer, "context": "", "sources": []}
        else:
            return {"answer": "No relevant documents found.", "context": "", "sources": []}
    
    context = format_context(retrieved_docs)
    
    if llm:
        prompt = f"""基于以下上下文信息回答问题。如果上下文没有相关信息，请回答"无法从提供的文档中找到答案"。

上下文:
{context}

问题: {question}
"""
        
        answer = llm.invoke(prompt)
    else:
        answer = "LLM not configured. Here are the relevant document snippets:\n\n" + context
    
    return {
        "answer": answer,
        "context": context,
        "sources": [doc["metadata"]["source"] for doc in retrieved_docs]
    }


def main():
    parser = argparse.ArgumentParser(description="PDF RAG System with DeepSeek")
    parser.add_argument("--recreate", action="store_true", help="Recreate the vector store")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--query", type=str, help="Query to answer")
    
    args = parser.parse_args()
    
    if not args.chat and not args.query:
        parser.print_help()
        sys.exit(1)
    
    try:
        llm, retriever = initialize_rag(recreate=args.recreate)
        
        if args.query:
            result = generate_answer(llm, retriever, args.query)
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nSources: {', '.join(result['sources'])}")
        
        if args.chat:
            print("\n=== RAG Chat Mode ===")
            print("Type 'exit' to quit\n")
            
            while True:
                question = input("You: ").strip()
                
                if question.lower() in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                result = generate_answer(llm, retriever, question)
                print(f"\nAnswer:\n{result['answer']}")
                print(f"\nSources: {', '.join(result['sources'])}")
                print("\n" + "="*50 + "\n")
    
    except Exception as e:
        log(f"Error: {str(e)}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
