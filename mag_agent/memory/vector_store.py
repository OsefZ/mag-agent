import litellm
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from mag_agent.exceptions import LLMError
from mag_agent.utils.logging_config import logger

class LiteLLMEmbeddings(Embeddings):
    """LangChain-compatible embedding interface using LiteLLM.
    
    Provides consistent embedding generation through LiteLLM library
    for semantic similarity computations.
    """
    
    def __init__(self, model: str = "openai/text-embedding-3-small"):
        self.model = model
        logger.info(f"Initialized LiteLLMEmbeddings with model: {model}")
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple text documents.
        
        Args:
            texts: Text documents to encode as vectors
            
        Returns:
            List of high-dimensional embedding vectors
        """
        try:
            logger.debug(f"Embedding {len(texts)} documents with LiteLLM")
            response = litellm.embedding(
                model=self.model,
                input=texts
            )
            embeddings = [data["embedding"] for data in response.data]
            logger.debug(f"Successfully embedded {len(embeddings)} documents")
            return embeddings
        except Exception as e:
            logger.error(f"LiteLLM embedding request failed: {e}")
            raise LLMError(f"Embedding failed: {e}")
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding vector for search query.
        
        Args:
            text: Query string to encode
            
        Returns:
            High-dimensional embedding vector
        """
        try:
            logger.debug("Embedding query text")
            response = litellm.embedding(
                model=self.model,
                input=[text]
            )
            embedding = response.data[0]["embedding"]
            logger.debug("Successfully embedded query")
            return embedding
        except Exception as e:
            logger.error(f"LiteLLM query embedding failed: {e}")
            raise LLMError(f"Query embedding failed: {e}")

class MAGVectorStore:
    """In-memory vector store for semantic similarity search.
    
    Wraps LangChain's InMemoryVectorStore with LiteLLM embeddings to provide
    efficient semantic search capabilities for code retrieval.
    """
    
    def __init__(self, embedding_model: str = "openai/text-embedding-3-small"):
        """Initialize vector store with specified embedding model.
        
        Args:
            embedding_model: Model identifier for text embedding generation
        """
        self.embeddings = LiteLLMEmbeddings(model=embedding_model)
        self.vector_store = InMemoryVectorStore(self.embeddings)
        logger.info("Initialized MAGVectorStore with InMemoryVectorStore")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Store documents in vector index for semantic search.
        
        Args:
            documents: Text documents with metadata to index
            
        Returns:
            List of document IDs
        """
        logger.debug(f"Adding {len(documents)} documents to vector store")
        document_ids = self.vector_store.add_documents(documents)
        logger.info(f"Added {len(document_ids)} documents to vector store")
        return document_ids
    
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """
        Add texts directly to the vector store.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts
            
        Returns:
            List of document IDs
        """
        logger.debug(f"Adding {len(texts)} texts to vector store")
        document_ids = self.vector_store.add_texts(texts, metadatas)
        logger.info(f"Added {len(document_ids)} texts to vector store")
        return document_ids
    
    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        """
        Search for the most similar documents to the query.
        
        Args:
            query: Query string
            k: Number of top results to return
            
        Returns:
            List of most similar documents
        """
        logger.debug(f"Performing similarity search with k={k}")
        results = self.vector_store.similarity_search(query, k=k)
        logger.debug(f"Found {len(results)} similar documents")
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 8) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Query string
            k: Number of top results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        logger.debug(f"Performing similarity search with scores, k={k}")
        results = self.vector_store.similarity_search_with_score(query, k=k)
        logger.debug(f"Found {len(results)} similar documents with scores")
        return results
    
    def cosine_similarity_test(self) -> bool:
        """
        Sanity test: verify that cosine similarity returns 1.0 for identical vectors.
        
        Returns:
            True if test passes
        """
        logger.info("Running cosine similarity sanity test...")
        
        test_text = "This is a test document for cosine similarity verification."
        
        # Add the test document
        test_doc = Document(page_content=test_text, metadata={"test": "cosine_similarity"})
        self.add_documents([test_doc])
        
        # Search for the exact same text
        results = self.similarity_search_with_score(test_text, k=1)
        
        if not results:
            logger.error("Cosine similarity test failed: No results returned")
            return False
        
        doc, score = results[0]
        logger.info(f"Cosine similarity test result: score={score}")
        
        # Check if the score is approximately 1.0 (accounting for floating point precision)
        if abs(score - 1.0) < 1e-6:
            logger.info("✅ Cosine similarity test PASSED")
            return True
        else:
            logger.error(f"❌ Cosine similarity test FAILED: expected ~1.0, got {score}")
            return False
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the vector store.
        
        Returns:
            Number of documents
        """
        # InMemoryVectorStore doesn't have a direct count method, so we'll track it
        # Simple document count implementation
        try:
            # Try to get all documents by searching for a very common word
            all_docs = self.similarity_search("the", k=10000)  # Large k to get all docs
            return len(all_docs)
        except:
            logger.warning("Unable to get document count, returning 0")
            return 0