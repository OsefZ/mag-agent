"""MAGAgent implementation for graph-based code completion.

This agent uses a memory-augmented graph structure to provide relevant
code context for next-line prediction tasks in software repositories.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
from mag_agent.llm.client import LiteLLMClient
from mag_agent.memory.graph_store import GraphStore
from mag_agent.utils.logging_config import logger
import time

# Import RepoBench completion mixin
sys.path.insert(0, str(Path(__file__).parent.parent))
from systems.repobench_completion_mixin import RepoBenchCompletionMixin

class MAGAgent(RepoBenchCompletionMixin):
    """Memory-Augmented Graph Agent for code completion tasks.
    
    Combines graph-based memory storage with LLM generation to provide
    contextually relevant code completions by retrieving related functions,
    classes, and code patterns from a repository's structural representation.
    """
    
    def __init__(self, model: str = "openai/gpt-4o-mini", cache_size: int = 10000, max_context_tokens: int = 3000):
        """Initialize the MAG Agent with graph memory and LLM client.
        
        Args:
            model: Name of the language model to use for code generation
            cache_size: Maximum number of entries to cache in graph memory
            max_context_tokens: Token limit for context window management
        """
        self.llm_client = LiteLLMClient()
        self.model = model
        self.graph_store = GraphStore(cache_size=cache_size, max_context_tokens=max_context_tokens)
        
        logger.info(f"MAGAgent initialized with model: {model}")
    
    def run_repobench_task(self, task) -> Dict[str, Any]:
        """Execute a code completion task using graph-based context retrieval.
        
        Args:
            task: Task object containing source code, file path, and repository information
            
        Returns:
            Dictionary containing the generated prediction and usage metrics
        """
        logger.info(f"Processing completion task for {task.file_path}")
        
        # Create retrieval plan for graph-based context extraction
        plan = {
            'files_to_inspect': [task.file_path],
            'functions_to_analyze': [],
            'approach': f"Complete next line in {task.file_path}"
        }
        
        # Retrieve relevant context from graph memory
        graph_context = self.graph_store.get_subgraph_by_plan(plan, use_compression=True)
        
        # Convert graph data to formatted context string
        context_str = self._build_graph_code_context(graph_context)
        
        # Generate completion using retrieved context
        result = self.run_repobench_completion(task, context_str)
        
        # Ensure compatibility with evaluation framework
        result['generated_code'] = result['prediction']
        
        return result
    
    def _build_graph_code_context(self, graph_context: Dict[str, Any]) -> str:
        """Convert graph context data into formatted code context string.
        
        Processes functions, classes, and related code snippets from the graph
        memory store and formats them for use in code completion prompts.
        
        Args:
            graph_context: Dictionary containing extracted code elements from graph
            
        Returns:
            Formatted string containing relevant code context
        """
        if not graph_context or not any(graph_context.values()):
            return "# No relevant context available"

        context_parts = []
        
        # Track processed content to prevent duplicates
        added_content = set()

        # Extract function definitions and implementations
        for func_data in graph_context.get('functions', []):
            if isinstance(func_data, dict) and func_data.get('code'):
                content = func_data['code']
                if content not in added_content:
                    file_path = func_data.get('file_path', func_data.get('location', 'unknown'))
                    func_name = func_data.get('name', 'unknown')
                    context_parts.append(f"# From: {file_path}\n# Function: {func_name}")
                    context_parts.append(content[:1500])  # Truncate long functions
                    added_content.add(content)

        # Extract class definitions and methods
        for class_data in graph_context.get('classes', []):
            if isinstance(class_data, dict) and class_data.get('code'):
                content = class_data['code']
                if content not in added_content:
                    file_path = class_data.get('file_path', class_data.get('location', 'unknown'))
                    class_name = class_data.get('name', 'unknown')
                    context_parts.append(f"# From: {file_path}\n# Class: {class_name}")
                    context_parts.append(content[:1500])  # Truncate long classes
                    added_content.add(content)
        
        # Include semantically related functions
        for func_data in graph_context.get('related_functions', []):
            if isinstance(func_data, dict) and func_data.get('code'):
                content = func_data['code']
                if content not in added_content:
                    file_path = func_data.get('file_path', func_data.get('location', 'unknown'))
                    func_name = func_data.get('name', 'unknown')
                    context_parts.append(f"# Related from: {file_path}\n# Function: {func_name}")
                    context_parts.append(content[:1500])
                    added_content.add(content)

        if not context_parts:
            return "# No code context available from graph retrieval"
            
        return "\n\n---\n\n".join(context_parts)