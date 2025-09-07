import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document

from mag_agent.llm.client import LiteLLMClient
from mag_agent.memory.vector_store import MAGVectorStore
from mag_agent.utils.logging_config import logger
from mag_agent.utils.swe_bench_loader import SWEBenchTask

# Import RepoBench completion mixin
sys.path.insert(0, str(Path(__file__).parent.parent))
from systems.repobench_completion_mixin import RepoBenchCompletionMixin

class VectorMAGAgent(RepoBenchCompletionMixin):
    """Vector-based MAG Agent using semantic similarity for code retrieval.
    
    This implementation serves as a baseline comparison to the graph-based approach,
    using embedding-based similarity search to find relevant code context for
    completion and generation tasks.
    """
    
    def __init__(self, model: str = "openai/gpt-4o-mini", embedding_model: str = "openai/text-embedding-3-small"):
        """Initialize the vector-based agent with LLM and embedding models.
        
        Args:
            model: Language model name for text generation
            embedding_model: Model for generating text embeddings for similarity search
        """
        logger.info("Initializing VectorMAGAgent...")
        self.llm_client = LiteLLMClient(enable_tracking=True)
        self.vector_store = MAGVectorStore(embedding_model=embedding_model)
        self.model = model
        self.embedding_model = embedding_model
        logger.info("VectorMAGAgent initialized successfully.")
    
    def index_codebase(self, repo_path: str, file_paths: List[str] = None) -> None:
        """Build vector embeddings index for repository codebase.
        
        Args:
            repo_path: Root directory of the repository to index
            file_paths: Specific files to include, or None to auto-discover code files
        """
        logger.info(f"Indexing codebase at: {repo_path}")
        
        if file_paths is None:
            # Auto-discover code files in repository
            file_paths = self._find_code_files(repo_path)
        
        documents = []
        for file_path in file_paths:
            full_path = os.path.join(repo_path, file_path)
            
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Break large files into manageable chunks
                    chunks = self._split_code_into_chunks(content, file_path)
                    
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "file_path": file_path,
                                "full_path": full_path,
                                "chunk_index": i,
                                "total_chunks": len(chunks)
                            }
                        )
                        documents.append(doc)
                        
                except Exception as e:
                    logger.warning(f"Failed to read {full_path}: {e}")
            else:
                logger.warning(f"File not found: {full_path}")
        
        if documents:
            self.vector_store.add_documents(documents)
            logger.info(f"Indexed {len(documents)} code chunks from {len(file_paths)} files")
        else:
            logger.warning("No documents were indexed")
    
    def _find_code_files(self, repo_path: str) -> List[str]:
        """Discover code files in repository directory tree.
        
        Args:
            repo_path: Root directory to search
            
        Returns:
            List of relative file paths for code files
        """
        code_files = []
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.go', '.rs', '.rb'}
        
        for root, dirs, files in os.walk(repo_path):
            # Exclude directories that don't contain source code
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}]
            
            for file in files:
                if any(file.endswith(ext) for ext in code_extensions):
                    rel_path = os.path.relpath(os.path.join(root, file), repo_path)
                    code_files.append(rel_path)
        
        logger.debug(f"Found {len(code_files)} code files")
        return code_files
    
    def _split_code_into_chunks(self, content: str, file_path: str, chunk_size: int = 2000) -> List[str]:
        """Divide source code into smaller chunks for efficient vector search.
        
        Args:
            content: Raw file content to split
            file_path: Source file identifier for logging
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of code text chunks
        """
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > chunk_size and current_chunk:
                # Chunk size exceeded, save current buffer
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Save final chunk if any content remains
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        logger.debug(f"Split {file_path} into {len(chunks)} chunks")
        return chunks
    
    def run_task(self, task: SWEBenchTask) -> Dict[str, Any]:
        """Execute bug-fixing task using vector-based context retrieval.
        
        Args:
            task: Task specification with problem description and target files
            
        Returns:
            Dictionary containing generated solution and execution metadata
        """
        logger.info(f"--- Starting Vector-based Task: {task.instance_id} ---")
        logger.info(f"Problem: {task.problem_statement}")
        
        # Monitor token consumption for cost analysis
        total_prompt_tokens = 0
        
        # Build vector index if not already populated
        if self.vector_store.get_document_count() == 0:
            logger.info("Indexing codebase...")
            
            # Locate repository directory, checking alternative paths if needed
            repo_path = task.repo_path
            if not os.path.exists(repo_path):
                # Check common development directory locations
                alternative_paths = [
                    "./data/sample_repo",  # For sample tasks
                    f"./data/{task.raw_data.get('repo', '').replace('/', '_')}"
                ]
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        repo_path = alt_path
                        logger.info(f"Using alternative repo path: {repo_path}")
                        break
            
            self.index_codebase(repo_path, task.file_paths_to_edit)
        
        # Retrieve relevant code using semantic similarity search
        logger.info("Searching for relevant code chunks...")
        relevant_docs = self.vector_store.similarity_search(
            query=task.problem_statement,
            k=8
        )
        
        # Build context string from retrieved code chunks
        code_context = ""
        for i, doc in enumerate(relevant_docs):
            file_path = doc.metadata.get('file_path', 'unknown')
            chunk_info = f"(chunk {doc.metadata.get('chunk_index', 0) + 1}/{doc.metadata.get('total_chunks', 1)})"
            code_context += f"--- {file_path} {chunk_info} ---\n{doc.page_content}\n\n"
        
        # Generate code patch using LLM with retrieved context
        logger.info("Generating solution using vector-retrieved context...")
        prompt = (
            f"Problem: {task.problem_statement}\n\n"
            f"Files to modify: {', '.join(task.file_paths_to_edit)}\n\n"
            "Relevant code context:\n"
            f"{code_context}\n\n"
            "Generate a unified diff patch to fix the issue. Use standard git patch format:\n"
            "--- a/file.py\n+++ b/file.py\n@@ -start,count +start,count @@\n"
            "Include 3-5 context lines around changes. Generate only the patch."
        )
        
        messages = [
            {"role": "system", "content": "Generate unified diff patches to fix code issues. Output only the patch, no explanations."},
            {"role": "user", "content": prompt}
        ]
        
        # Estimate token usage for cost tracking
        prompt_tokens = len(prompt.split())
        total_prompt_tokens += prompt_tokens
        
        response = self.llm_client.generate(self.model, messages)
        generated_code = response['choices'][0]['message']['content']
        
        # Use precise token count if provided by API response
        if 'usage' in response:
            total_prompt_tokens = response['usage'].get('prompt_tokens', prompt_tokens)
        
        logger.info(f"\n--- Generated Code Patch ---\n{generated_code}\n--------------------------")
        logger.info("--- Vector-based Task Finished ---")
        
        return {
            "instance_id": task.instance_id,
            "problem_statement": task.problem_statement,
            "generated_code": generated_code,
            "relevant_chunks": len(relevant_docs),
            "prompt_tokens": total_prompt_tokens,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "mode": "vector"
        }
    
    def run_repobench_task(self, task) -> Dict[str, Any]:
        """Execute code completion task using vector-based context retrieval.
        
        Args:
            task: Task object containing incomplete code and file metadata
            
        Returns:
            Dictionary with completion prediction and usage statistics
        """
        logger.info(f"Running RepoBench task for {task.file_path}")
        
        # Use recent code lines as search query for context retrieval
        code_lines = task.cropped_code.split('\n')
        query = '\n'.join(code_lines[-5:]) if len(code_lines) > 5 else task.cropped_code
        
        # Find similar code snippets using vector search
        vector_results = self.vector_store.search(query, k=10)
        
        # Format retrieved code snippets into context string
        context_str = "\n\n".join([
            f"# From {doc.metadata.get('source', 'unknown')}:\n{doc.page_content}"
            for doc in vector_results
        ])
        
        # Generate completion using retrieved context
        result = self.run_repobench_completion(task, context_str)
        
        # Ensure compatibility with evaluation framework
        result['generated_code'] = result['prediction']
        result['vector_retrieval_results'] = [doc.metadata for doc in vector_results]
        
        return result
    
    def run_cosine_similarity_test(self) -> bool:
        """Validate vector store functionality with similarity test.
        
        Returns:
            True if similarity search is working correctly
        """
        return self.vector_store.cosine_similarity_test()