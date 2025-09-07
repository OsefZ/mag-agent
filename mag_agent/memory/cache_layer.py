"""
Minimal cache layer implementation for MAG Agent memory system.
Simplified for RepoBench evaluation.
"""

import time
from typing import Dict, Any, Optional
from collections import OrderedDict

class QueryCacheManager:
    """Simple LRU cache for graph queries."""
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.cache = OrderedDict()
    
    def get(self, query_key: str) -> Optional[Any]:
        """Get cached result if exists."""
        if query_key in self.cache:
            self.cache.move_to_end(query_key)
            return self.cache[query_key]
        return None
    
    def put(self, query_key: str, result: Any):
        """Cache query result."""
        if query_key in self.cache:
            self.cache.move_to_end(query_key)
        elif len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        self.cache[query_key] = result
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()

class MaterializedViewManager:
    """Simple view manager for commonly accessed subgraphs."""
    
    def __init__(self):
        self.views = {}
    
    def create_view(self, view_name: str, query: str):
        """Create a materialized view."""
        self.views[view_name] = {"query": query, "created_at": time.time()}
    
    def get_view(self, view_name: str) -> Optional[Dict]:
        """Get materialized view if exists."""
        return self.views.get(view_name)
    
    def refresh_view(self, view_name: str):
        """Refresh materialized view."""
        if view_name in self.views:
            self.views[view_name]["updated_at"] = time.time()

class PerformanceTracker:
    """Simple performance tracking for queries."""
    
    def __init__(self):
        self.query_times = []
        self.total_queries = 0
    
    def track_query(self, duration_ms: float):
        """Track query execution time."""
        self.query_times.append(duration_ms)
        self.total_queries += 1
        
        # Keep only last 100 query times
        if len(self.query_times) > 100:
            self.query_times = self.query_times[-100:]
    
    def get_average_time(self) -> float:
        """Get average query time."""
        return sum(self.query_times) / len(self.query_times) if self.query_times else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_queries": self.total_queries,
            "average_time_ms": self.get_average_time(),
            "recent_queries": len(self.query_times)
        }