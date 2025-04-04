import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
import hashlib

class MemorySelector:
    """Dynamic memory selection system that retrieves relevant memories for current task."""
    
    def __init__(self, agent_memory, config: Dict[str, Any]):
        self.agent_memory = agent_memory
        self.config = config
        self.selection_stats = {
            "total_selections": 0,
            "avg_memories_selected": 0,
            "selection_time": 0,
            "cache_hits": 0
        }
        self.selection_cache = {}
        self.cache_expiry = config.get("cache_expiry_seconds", 60)
        
    def select(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Select relevant memories for current context.
        
        Args:
            context: Contains task, intent, and constraints
            
        Returns:
            List of selected memories
        """
        start_time = time.time()
        self.selection_stats["total_selections"] += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(context)
        if cache_key in self.selection_cache:
            cache_entry = self.selection_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_expiry:
                self.selection_stats["cache_hits"] += 1
                return cache_entry["memories"]
        
        # Extract context elements
        task = context.get("task", {})
        intent = context.get("intent", {})
        constraints = context.get("constraints", {})
        
        # Determine memory count limits
        memory_limit = constraints.get("memory_limit", 
                                      self.config.get("default_memory_limit", 10))
        
        # Get task-specific memories
        task_memories = self._get_task_memories(task, memory_limit)
        
        # Get intent-aligned memories
        intent_memories = self._get_intent_memories(intent, task, memory_limit)
        
        # Get recent successful memories
        recent_memories = self._get_recent_successful_memories(memory_limit)
        
        # Combine and deduplicate memories
        all_memories = {}
        for memory in task_memories + intent_memories + recent_memories:
            if memory["id"] not in all_memories:
                all_memories[memory["id"]] = memory
        
        # Apply selection strategy to combined memories
        selected = self._apply_selection_strategy(
            list(all_memories.values()), 
            context, 
            memory_limit
        )
        
        # Update statistics
        selection_time = time.time() - start_time
        self.selection_stats["selection_time"] = (
            (self.selection_stats["selection_time"] * 
             (self.selection_stats["total_selections"] - 1) + 
             selection_time) / self.selection_stats["total_selections"]
        )
        self.selection_stats["avg_memories_selected"] = (
            (self.selection_stats["avg_memories_selected"] * 
             (self.selection_stats["total_selections"] - 1) + 
             len(selected)) / self.selection_stats["total_selections"]
        )
        
        # Cache result
        self.selection_cache[cache_key] = {
            "memories": selected,
            "timestamp": time.time()
        }
        
        # Clean cache occasionally
        if self.selection_stats["total_selections"] % 100 == 0:
            self._clean_cache()
        
        return selected
    
    def _get_task_memories(self, task: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Get memories related to current task type and content."""
        task_type = task.get("type", "unknown")
        task_context = {"type": task_type}
        
        # Query by context for exact task type matches
        type_matches = self.agent_memory.query_by_context(task_context, limit=limit)
        
        # If task has content field, try to find similar memories
        content_matches = []
        if "content" in task and hasattr(self.agent_memory, "query_by_vector"):
            content_vector = self._get_content_vector(task["content"])
            if content_vector is not None:
                content_matches = self.agent_memory.query_by_vector(
                    content_vector, "task_content", limit=limit)
        
        # Combine but prioritize type matches
        combined = type_matches.copy()
        for memory in content_matches:
            if memory["id"] not in [m["id"] for m in combined]:
                combined.append(memory)
                if len(combined) >= limit:
                    break
        
        return combined[:limit]
    
    def _get_intent_memories(self, 
                           intent: Any, 
                           task: Dict[str, Any], 
                           limit: int) -> List[Dict[str, Any]]:
        """Get memories aligned with current intent vector."""
        # First, extract intent dimensions
        intent_dims = {}
        if hasattr(intent, "get_vector_as_dict"):
            intent_dims = intent.get_vector_as_dict()
        elif isinstance(intent, dict):
            intent_dims = intent
        
        # Find the dominant intent dimensions
        sorted_dims = sorted(
            intent_dims.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Use top dimensions to find memories
        memories = []
        for dim_name, dim_value in sorted_dims[:3]:  # Use top 3 dimensions
            if dim_value > 0.5:  # Only use significant dimensions
                # Find memories where this dimension was strong
                dim_memories = self._get_memories_by_dimension(dim_name, limit)
                
                # Add to results without duplicates
                for memory in dim_memories:
                    if memory["id"] not in [m["id"] for m in memories]:
                        memories.append(memory)
                        if len(memories) >= limit:
                            break
            
            if len(memories) >= limit:
                break
        
        return memories[:limit]
    
    def _get_memories_by_dimension(self, dimension: str, limit: int) -> List[Dict[str, Any]]:
        """Get memories related to a specific intent dimension."""
        # This implementation depends on how memory is tagged with intent dimensions
        # Simplified implementation
        
        # For coherence dimension
        if dimension == "coherence":
            return self.agent_memory.query_by_context(
                {"metadata.coherence_related": True}, limit=limit)
        
        # For knowledge dimension
        elif dimension == "knowledge":
            return self.agent_memory.query_by_context(
                {"type": "factual"}, limit=limit)
        
        # For exploration dimension
        elif dimension == "exploration":
            return self.agent_memory.query_by_context(
                {"metadata.novelty": True}, limit=limit)
        
        # Default: just return some memories
        return self.agent_memory.query_by_context({}, limit=limit)
    
    def _get_recent_successful_memories(self, limit: int) -> List[Dict[str, Any]]:
        """Get recent memories with successful outcomes."""
        return self.agent_memory.query_by_context(
            {"metadata.success": True}, limit=limit)
    
    def _apply_selection_strategy(self, 
                                memories: List[Dict[str, Any]], 
                                context: Dict[str, Any],
                                limit: int) -> List[Dict[str, Any]]:
        """Apply selection strategy to choose best memories."""
        if not memories:
            return []
        
        # Score memories based on relevance to current context
        scored_memories = []
        for memory in memories:
            score = self._score_memory(memory, context)
            scored_memories.append((memory, score))
        
        # Sort by score (descending)
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Take top memories
        return [m for m, _ in scored_memories[:limit]]
    
    def _score_memory(self, memory: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Score a memory's relevance to current context."""
        score = 0.0
        
        # Base score components
        if "metadata" in memory:
            # Success bonus
            if memory["metadata"].get("success", False):
                score += 0.3
            
            # Recency bonus (higher for more recent)
            if "created_at" in memory:
                age_seconds = time.time() - memory["created_at"]
                recency_factor = max(0, 1.0 - (age_seconds / 
                                              (self.config.get("recency_window", 604800))))  # Default 1 week
                score += 0.2 * recency_factor
            
            # Reward score contribution
            if "reward" in memory["metadata"]:
                score += 0.2 * memory["metadata"]["reward"]
        
        # Task similarity
        task_similarity = self._calculate_task_similarity(
            memory.get("context", {}), context.get("task", {}))
        score += 0.3 * task_similarity
        
        return score
    
    def _calculate_task_similarity(self, 
                                 memory_context: Dict[str, Any], 
                                 current_task: Dict[str, Any]) -> float:
        """Calculate similarity between memory context and current task."""
        # Basic type matching
        if memory_context.get("type") == current_task.get("type"):
            return 0.7
        
        # More sophisticated methods would use embeddings or semantic similarity
        # This is a simplified implementation
        return 0.3
    
    def _get_content_vector(self, content: Any) -> Optional[np.ndarray]:
        """Get vector embedding for content (simplified implementation)."""
        # In a real implementation, this would use an embedding model
        # For this example, we return None to indicate no vector is available
        return None
    
    def _generate_cache_key(self, context: Dict[str, Any]) -> str:
        """Generate a cache key from context."""
        # Only use stable parts of context for caching
        cache_parts = {
            "task_type": context.get("task", {}).get("type", "unknown"),
            "task_id": context.get("task", {}).get("id", "unknown")
        }
        
        # Add simplified intent if available
        if "intent" in context:
            intent = context["intent"]
            if hasattr(intent, "get_vector_as_dict"):
                top_dims = sorted(
                    intent.get_vector_as_dict().items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                cache_parts["intent_dims"] = str(top_dims)
        
        # Generate hash
        key_str = str(cache_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _clean_cache(self) -> None:
        """Clean expired entries from cache."""
        now = time.time()
        expired_keys = [
            key for key, entry in self.selection_cache.items()
            if now - entry["timestamp"] > self.cache_expiry
        ]
        
        for key in expired_keys:
            del self.selection_cache[key]
