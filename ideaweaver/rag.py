"""
RAG (Retrieval-Augmented Generation) Module using LangChain and Vector Databases
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from dataclasses import dataclass, asdict
import click

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, DirectoryLoader, 
    UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

# Vector database and embeddings
import chromadb
from sentence_transformers import SentenceTransformer

# Cloud Vector Database imports
try:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Advanced RAG imports
import re
import nltk
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


@dataclass
class KnowledgeBaseConfig:
    """Configuration for a knowledge base"""
    name: str
    description: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: str = "recursive"  # recursive, token, semantic
    vector_store: str = "chroma"  # chroma, qdrant_local, qdrant_cloud
    created_at: str = ""
    updated_at: str = ""
    document_count: int = 0
    
    # Advanced RAG features
    use_hybrid_search: bool = False
    hybrid_alpha: float = 0.5  # Weight for semantic vs keyword search (0=keyword, 1=semantic)
    use_reranking: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    reranker_top_k: int = 20  # Number of docs to retrieve before reranking
    semantic_similarity_threshold: float = 0.8  # For semantic chunking
    
    # Cloud Vector Database Configuration
    qdrant_url: Optional[str] = None  # For Qdrant Cloud: https://xyz.qdrant.io
    qdrant_api_key: Optional[str] = None  # Qdrant Cloud API key
    qdrant_collection_name: Optional[str] = None  # Collection name in Qdrant
    qdrant_prefer_grpc: bool = False  # Use gRPC for better performance
    qdrant_timeout: int = 60  # Connection timeout in seconds
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeBaseConfig':
        return cls(**data)


class HybridSearcher:
    """Hybrid search combining semantic and keyword search"""
    
    def __init__(self, embedding_model: str, alpha: float = 0.5):
        """
        Initialize hybrid searcher
        
        Args:
            embedding_model: Sentence transformer model for embeddings
            alpha: Weight for combining scores (0=keyword only, 1=semantic only)
        """
        self.alpha = alpha
        self.embedder = SentenceTransformer(embedding_model)
        self.bm25 = None
        self.corpus_embeddings = None
        self.documents = []
        
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents for hybrid search"""
        self.documents = documents
        
        # Prepare texts for BM25
        tokenized_corpus = []
        texts = []
        
        for doc in documents:
            # Simple tokenization for BM25
            tokens = self._tokenize(doc.page_content)
            tokenized_corpus.append(tokens)
            texts.append(doc.page_content)
        
        # Initialize BM25
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Create embeddings for semantic search
        self.corpus_embeddings = self.embedder.encode(texts, convert_to_tensor=True)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.bm25 or self.corpus_embeddings is None:
            raise ValueError("Documents not indexed. Call index_documents() first.")
        
        # BM25 scores
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Semantic scores
        query_embedding = self.embedder.encode([query], convert_to_tensor=True)
        semantic_scores = cosine_similarity(
            query_embedding.cpu().numpy(), 
            self.corpus_embeddings.cpu().numpy()
        )[0]
        
        # Normalize scores to [0, 1]
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
        
        # Combine scores
        hybrid_scores = (1 - self.alpha) * bm25_scores + self.alpha * semantic_scores
        
        # Get top k results
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(hybrid_scores[idx])))
        
        return results


class CrossEncoderReranker:
    """Cross-encoder reranking for improved relevance"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """Initialize cross-encoder reranker"""
        self.model_name = model_name
        self.model = None
        
    def _load_model(self):
        """Lazy load the cross-encoder model"""
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of (document, relevance_score) tuples
        """
        self._load_model()
        
        # Prepare query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Sort by score and return top k
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:top_k]


class SemanticChunker:
    """Semantic chunking based on content similarity"""
    
    def __init__(self, embedding_model: str, similarity_threshold: float = 0.8):
        """
        Initialize semantic chunker
        
        Args:
            embedding_model: Model for computing embeddings
            similarity_threshold: Threshold for splitting chunks
        """
        self.embedder = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
    
    def chunk_text(self, text: str, min_chunk_size: int = 100, max_chunk_size: int = 1000) -> List[str]:
        """
        Chunk text based on semantic similarity
        
        Args:
            text: Text to chunk
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            
        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # Get sentence embeddings
        embeddings = self.embedder.encode(sentences)
        
        # Find semantic boundaries
        chunks = []
        current_chunk = [sentences[0]]
        current_size = len(sentences[0])
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = cosine_similarity(
                embeddings[i-1:i], 
                embeddings[i:i+1]
            )[0][0]
            
            sentence_len = len(sentences[i])
            
            # Decide whether to continue current chunk or start new one
            should_split = (
                similarity < self.similarity_threshold or  # Low semantic similarity
                current_size + sentence_len > max_chunk_size  # Size limit
            )
            
            if should_split and current_size >= min_chunk_size:
                # Start new chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_size = sentence_len
            else:
                # Continue current chunk
                current_chunk.append(sentences[i])
                current_size += sentence_len
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        # Use regex for basic sentence splitting
        sentences = re.split(r'[.!?]+', text)
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences


class RAGManager:
    """
    Core RAG functionality manager using LangChain and vector databases
    """
    
    def __init__(self, base_dir: str = "./rag_data", verbose: bool = False):
        """
        Initialize RAG Manager
        
        Args:
            base_dir: Base directory for storing RAG data
            verbose: Enable verbose logging
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        
        # Create subdirectories
        self.kb_dir = self.base_dir / "knowledge_bases"
        self.kb_dir.mkdir(exist_ok=True)
        
        self.config_dir = self.base_dir / "configs"
        self.config_dir.mkdir(exist_ok=True)
        
        if self.verbose:
            click.echo(f"📁 RAG data directory: {self.base_dir}")
    
    def create_knowledge_base(self, name: str, description: str = "", 
                            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                            chunk_size: int = 512, chunk_overlap: int = 50) -> KnowledgeBaseConfig:
        """
        Create a new knowledge base
        
        Args:
            name: Knowledge base name
            description: Description of the knowledge base
            embedding_model: Embedding model to use
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            KnowledgeBaseConfig object
        """
        kb_path = self.kb_dir / name
        
        if kb_path.exists():
            raise ValueError(f"Knowledge base '{name}' already exists")
        
        # Create knowledge base directory
        kb_path.mkdir(parents=True)
        
        # Create config
        import datetime
        config = KnowledgeBaseConfig(
            name=name,
            description=description,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            created_at=datetime.datetime.now().isoformat(),
            updated_at=datetime.datetime.now().isoformat()
        )
        
        # Save config
        config_path = self.config_dir / f"{name}.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Initialize vector store
        self._init_vector_store(name, embedding_model)
        
        if self.verbose:
            click.echo(f"✅ Created knowledge base: {name}")
            click.echo(f"📁 Path: {kb_path}")
            click.echo(f"🧠 Embedding model: {embedding_model}")
        
        return config
    
    def list_knowledge_bases(self) -> List[KnowledgeBaseConfig]:
        """List all knowledge bases"""
        kbs = []
        
        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    kbs.append(KnowledgeBaseConfig.from_dict(config_data))
            except (json.JSONDecodeError, TypeError) as e:
                if self.verbose:
                    click.echo(f"⚠️  Warning: Could not load config {config_file}: {e}")
        
        return kbs
    
    def get_knowledge_base(self, name: str) -> Optional[KnowledgeBaseConfig]:
        """Get knowledge base configuration"""
        config_path = self.config_dir / f"{name}.json"
        
        if not config_path.exists():
            return None
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                return KnowledgeBaseConfig.from_dict(config_data)
        except (json.JSONDecodeError, TypeError):
            return None
    
    def delete_knowledge_base(self, name: str) -> bool:
        """Delete a knowledge base"""
        kb_path = self.kb_dir / name
        config_path = self.config_dir / f"{name}.json"
        
        if not kb_path.exists() and not config_path.exists():
            if self.verbose:
                click.echo(f"❌ Knowledge base '{name}' not found")
            return False
        
        # Remove directory and config
        if kb_path.exists():
            shutil.rmtree(kb_path)
        if config_path.exists():
            config_path.unlink()
        
        if self.verbose:
            click.echo(f"🗑️  Deleted knowledge base: {name}")
        
        return True
    
    def _init_vector_store(self, kb_name: str, embedding_model: str):
        """Initialize vector store for knowledge base"""
        config = self.get_knowledge_base(kb_name)
        if not config:
            raise ValueError(f"Knowledge base '{kb_name}' not found")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use CPU for now
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if config.vector_store == "chroma":
            # ChromaDB (local)
            persist_directory = str(self.kb_dir / kb_name / "chroma_db")
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            
        elif config.vector_store == "qdrant_cloud":
            # Qdrant Cloud
            if not QDRANT_AVAILABLE:
                raise ImportError("Qdrant not available. Install with: pip install qdrant-client langchain-qdrant")
            
            if not config.qdrant_url or not config.qdrant_api_key:
                raise ValueError("Qdrant Cloud requires qdrant_url and qdrant_api_key")
            
            # Initialize Qdrant client
            client = QdrantClient(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
                timeout=config.qdrant_timeout,
                prefer_grpc=config.qdrant_prefer_grpc
            )
            
            collection_name = config.qdrant_collection_name or f"kb_{kb_name}"
            
            # Create collection if it doesn't exist
            try:
                client.get_collection(collection_name)
                if self.verbose:
                    click.echo(f"🔗 Using existing Qdrant collection: {collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                vector_size = 384  # Default for all-MiniLM-L6-v2
                if "mpnet" in embedding_model:
                    vector_size = 768
                elif "large" in embedding_model:
                    vector_size = 1024
                
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                if self.verbose:
                    click.echo(f"🆕 Created Qdrant collection: {collection_name}")
            
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings
            )
            
        elif config.vector_store == "qdrant_local":
            # Qdrant Local
            if not QDRANT_AVAILABLE:
                raise ImportError("Qdrant not available. Install with: pip install qdrant-client langchain-qdrant")
            
            # Local Qdrant instance (requires Qdrant server running locally)
            client = QdrantClient(
                host="localhost",
                port=6333
            )
            
            collection_name = config.qdrant_collection_name or f"kb_{kb_name}"
            
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings
            )
            
        else:
            raise ValueError(f"Unsupported vector store: {config.vector_store}")
        
        return vectorstore
    
    def ingest_documents(self, kb_name: str, source_path: str, 
                        file_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Ingest documents into knowledge base
        
        Args:
            kb_name: Knowledge base name
            source_path: Path to documents (file or directory)
            file_types: List of file types to include (e.g., ['pdf', 'txt', 'md'])
            
        Returns:
            Dictionary with ingestion results
        """
        config = self.get_knowledge_base(kb_name)
        if not config:
            raise ValueError(f"Knowledge base '{kb_name}' not found")
        
        source_path = Path(source_path)
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")
        
        # Default file types
        if file_types is None:
            file_types = ['pdf', 'txt', 'md', 'docx']
        
        documents = []
        
        if source_path.is_file():
            # Single file
            documents = self._load_single_file(source_path)
        else:
            # Directory
            documents = self._load_directory(source_path, file_types)
        
        if not documents:
            if self.verbose:
                click.echo("⚠️  No documents found to ingest")
            return {"documents_processed": 0, "chunks_created": 0}
        
        # Split documents into chunks
        chunks = self._split_documents(documents, config)
        
        if not chunks:
            if self.verbose:
                click.echo("⚠️  No chunks created from documents")
            return {"documents_processed": len(documents), "chunks_created": 0}
        
        # Add to vector store
        vectorstore = self._init_vector_store(kb_name, config.embedding_model)
        vectorstore.add_documents(chunks)
        
        # Update config
        config.document_count += len(documents)
        import datetime
        config.updated_at = datetime.datetime.now().isoformat()
        
        # Save updated config
        config_path = self.config_dir / f"{kb_name}.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        if self.verbose:
            click.echo(f"✅ Ingested {len(documents)} documents")
            click.echo(f"📄 Created {len(chunks)} chunks")
            click.echo(f"🔢 Total documents in KB: {config.document_count}")
        
        return {
            "documents_processed": len(documents),
            "chunks_created": len(chunks),
            "total_documents": config.document_count
        }
    
    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Load a single file"""
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif extension == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif extension == '.md':
                loader = UnstructuredMarkdownLoader(str(file_path))
            elif extension == '.docx':
                loader = UnstructuredWordDocumentLoader(str(file_path))
            else:
                if self.verbose:
                    click.echo(f"⚠️  Unsupported file type: {extension}")
                return []
            
            return loader.load()
            
        except Exception as e:
            if self.verbose:
                click.echo(f"❌ Error loading {file_path}: {e}")
            return []
    
    def _load_directory(self, dir_path: Path, file_types: List[str]) -> List[Document]:
        """Load all files in directory"""
        documents = []
        
        for file_type in file_types:
            pattern = f"**/*.{file_type}"
            
            try:
                if file_type == 'pdf':
                    loader = DirectoryLoader(
                        str(dir_path), 
                        glob=pattern, 
                        loader_cls=PyPDFLoader,
                        show_progress=self.verbose
                    )
                elif file_type == 'txt':
                    loader = DirectoryLoader(
                        str(dir_path), 
                        glob=pattern, 
                        loader_cls=TextLoader,
                        loader_kwargs={'encoding': 'utf-8'},
                        show_progress=self.verbose
                    )
                elif file_type == 'md':
                    loader = DirectoryLoader(
                        str(dir_path), 
                        glob=pattern, 
                        loader_cls=UnstructuredMarkdownLoader,
                        show_progress=self.verbose
                    )
                elif file_type == 'docx':
                    loader = DirectoryLoader(
                        str(dir_path), 
                        glob=pattern, 
                        loader_cls=UnstructuredWordDocumentLoader,
                        show_progress=self.verbose
                    )
                else:
                    continue
                
                docs = loader.load()
                documents.extend(docs)
                
                if self.verbose:
                    click.echo(f"📄 Loaded {len(docs)} {file_type.upper()} files")
                    
            except Exception as e:
                if self.verbose:
                    click.echo(f"❌ Error loading {file_type} files: {e}")
        
        return documents
    
    def _split_documents(self, documents: List[Document], config: KnowledgeBaseConfig) -> List[Document]:
        """Split documents into chunks"""
        
        if config.chunking_strategy == "semantic":
            # Use semantic chunking
            semantic_chunker = SemanticChunker(
                embedding_model=config.embedding_model,
                similarity_threshold=config.semantic_similarity_threshold
            )
            
            all_chunks = []
            for doc in documents:
                text_chunks = semantic_chunker.chunk_text(
                    doc.page_content,
                    min_chunk_size=config.chunk_size // 2,
                    max_chunk_size=config.chunk_size * 2
                )
                
                for i, chunk_text in enumerate(text_chunks):
                    chunk_doc = Document(
                        page_content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_id": f"{len(all_chunks)}_{i}",
                            "chunk_strategy": "semantic",
                            "original_doc_id": id(doc)
                        }
                    )
                    all_chunks.append(chunk_doc)
            
            return all_chunks
            
        elif config.chunking_strategy == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        elif config.chunking_strategy == "token":
            splitter = TokenTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
        else:
            # Default to recursive
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
        
        chunks = splitter.split_documents(documents)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "kb_name": config.name,
                "chunk_size": len(chunk.page_content),
                "chunk_strategy": config.chunking_strategy
            })
        
        return chunks
    
    def query(self, kb_name: str, question: str, top_k: int = 5, 
              llm_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the knowledge base with advanced features
        
        Args:
            kb_name: Knowledge base name
            question: Question to ask
            top_k: Number of top documents to retrieve
            llm_model: LLM model for generation (optional)
            
        Returns:
            Dictionary with query results
        """
        config = self.get_knowledge_base(kb_name)
        if not config:
            raise ValueError(f"Knowledge base '{kb_name}' not found")
        
        # Initialize vector store
        vectorstore = self._init_vector_store(kb_name, config.embedding_model)
        
        # Determine retrieval strategy
        if config.use_hybrid_search or config.use_reranking:
            # Advanced retrieval with hybrid search and/or reranking
            retrieved_docs = self._advanced_retrieval(
                vectorstore, question, config, top_k
            )
        else:
            # Standard semantic retrieval
            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
            retrieved_docs = retriever.invoke(question)
            retrieved_docs = [(doc, 1.0) for doc in retrieved_docs]  # Add dummy scores
        
        result = {
            "question": question,
            "retrieved_documents": [],
            "answer": None,
            "kb_name": kb_name,
            "top_k": top_k,
            "search_method": self._get_search_method_description(config)
        }
        
        # Format retrieved documents
        for i, (doc, score) in enumerate(retrieved_docs):
            result["retrieved_documents"].append({
                "rank": i + 1,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": float(score)
            })
        
        # Generate answer if LLM is provided
        if llm_model and retrieved_docs:
            try:
                answer = self._generate_answer(question, [doc for doc, _ in retrieved_docs], llm_model)
                result["answer"] = answer
            except Exception as e:
                if self.verbose:
                    click.echo(f"⚠️  Warning: Could not generate answer: {e}")
                result["answer"] = "Error generating answer with LLM"
        
        if self.verbose:
            click.echo(f"🔍 Retrieved {len(retrieved_docs)} documents using {result['search_method']}")
            if result["answer"]:
                click.echo(f"💡 Generated answer using {llm_model}")
        
        return result
    
    def _advanced_retrieval(self, vectorstore, question: str, config: KnowledgeBaseConfig, 
                          top_k: int) -> List[Tuple[Document, float]]:
        """Advanced retrieval with hybrid search and reranking"""
        
        # Determine initial retrieval count
        initial_k = config.reranker_top_k if config.use_reranking else top_k
        
        if config.use_hybrid_search:
            # Get all documents from vector store for hybrid search
            all_docs = self._get_all_documents_from_vectorstore(vectorstore)
            
            # Initialize hybrid searcher
            hybrid_searcher = HybridSearcher(
                embedding_model=config.embedding_model,
                alpha=config.hybrid_alpha
            )
            hybrid_searcher.index_documents(all_docs)
            
            # Perform hybrid search
            doc_scores = hybrid_searcher.search(question, top_k=initial_k)
            
        else:
            # Standard semantic search
            retriever = vectorstore.as_retriever(search_kwargs={"k": initial_k})
            docs = retriever.invoke(question)
            doc_scores = [(doc, 1.0) for doc in docs]  # Add dummy scores
        
        # Apply reranking if enabled
        if config.use_reranking and doc_scores:
            reranker = CrossEncoderReranker(model_name=config.reranker_model)
            documents = [doc for doc, _ in doc_scores]
            doc_scores = reranker.rerank(question, documents, top_k=top_k)
        else:
            # Trim to final top_k if no reranking
            doc_scores = doc_scores[:top_k]
        
        return doc_scores
    
    def _get_all_documents_from_vectorstore(self, vectorstore) -> List[Document]:
        """Get all documents from vector store for hybrid search"""
        try:
            # Try to get all documents from Chroma
            collection = vectorstore._collection
            results = collection.get(include=["documents", "metadatas"])
            
            documents = []
            for i, (doc_text, metadata) in enumerate(zip(results["documents"], results["metadatas"])):
                doc = Document(page_content=doc_text, metadata=metadata or {})
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            if self.verbose:
                click.echo(f"⚠️  Warning: Could not retrieve all documents for hybrid search: {e}")
            return []
    
    def _get_search_method_description(self, config: KnowledgeBaseConfig) -> str:
        """Get description of search method used"""
        methods = []
        
        if config.use_hybrid_search:
            methods.append(f"Hybrid Search (α={config.hybrid_alpha})")
        else:
            methods.append("Semantic Search")
            
        if config.use_reranking:
            methods.append(f"Cross-Encoder Reranking ({config.reranker_model})")
            
        return " + ".join(methods)
    
    def _generate_answer(self, question: str, documents: List[Document], 
                        llm_model: str) -> str:
        """Generate answer using LLM and retrieved documents"""
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Simple prompt template
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            # Use HuggingFace pipeline for generation
            from transformers import pipeline
            
            generator = pipeline(
                "text-generation",
                model=llm_model,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                return_full_text=False
            )
            
            response = generator(prompt)
            answer = response[0]['generated_text'].strip()
            
            return answer
            
        except Exception as e:
            raise RuntimeError(f"Error generating answer: {e}")
    
    def get_stats(self, kb_name: str) -> Dict[str, Any]:
        """Get statistics for a knowledge base"""
        config = self.get_knowledge_base(kb_name)
        if not config:
            raise ValueError(f"Knowledge base '{kb_name}' not found")
        
        # Get vector store stats
        vectorstore = self._init_vector_store(kb_name, config.embedding_model)
        
        try:
            # Try to get collection info (Chroma-specific)
            collection = vectorstore._collection
            chunk_count = collection.count()
        except:
            chunk_count = "Unknown"
        
        stats = {
            "name": config.name,
            "description": config.description,
            "embedding_model": config.embedding_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "chunking_strategy": config.chunking_strategy,
            "created_at": config.created_at,
            "updated_at": config.updated_at,
            "document_count": config.document_count,
            "chunk_count": chunk_count,
            "vector_store": config.vector_store
        }
        
        return stats 