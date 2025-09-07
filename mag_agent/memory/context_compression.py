"""
Minimal context compression implementation for MAG Agent memory system.
Simplified for RepoBench evaluation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ContextItem:
    """A single context item with metadata."""
    content: str
    source_type: str = "unknown"
    file_path: str = ""
    relevance_score: float = 1.0
    metadata: Optional[Dict[str, Any]] = None

class HierarchicalContextCompressor:
    """Simple context compression using token limits."""
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
    
    def compress_context(self, context_items: List[ContextItem]) -> str:
        """Compress context items into a single string within token limit."""
        if not context_items:
            return "# No relevant context found"
        
        # Sort by relevance score (highest first)
        sorted_items = sorted(context_items, key=lambda x: x.relevance_score, reverse=True)
        
        context_parts = []
        current_tokens = 0
        
        for item in sorted_items:
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            item_tokens = len(item.content) // 4
            
            if current_tokens + item_tokens > self.max_tokens:
                break
            
            if item.file_path:
                context_parts.append(f"# From: {item.file_path}")
            context_parts.append(item.content)
            current_tokens += item_tokens
        
        return "\n\n---\n\n".join(context_parts)

class AdaptiveContextSelector:
    """Simple context selector based on relevance."""
    
    def __init__(self, target_tokens: int = 2000):
        self.target_tokens = target_tokens
    
    def select_relevant_context(self, context_items: List[ContextItem], query: str = "") -> List[ContextItem]:
        """Select most relevant context items."""
        if not context_items:
            return []
        
        # Simple selection: return items sorted by relevance score
        return sorted(context_items, key=lambda x: x.relevance_score, reverse=True)
    
    def estimate_relevance(self, item: ContextItem, query: str) -> float:
        """Estimate relevance score for context item."""
        # Simple keyword matching
        if not query:
            return item.relevance_score
        
        query_words = set(query.lower().split())
        content_words = set(item.content.lower().split())
        
        # Calculate intersection ratio
        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words) if query_words else 0.0