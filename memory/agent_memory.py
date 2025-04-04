import hashlib
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

class AgentMemory:
    """Centralized memory store that preserves experience across mutations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.memories = []
        self.memory_by_id = {}
        self.memory_by_context = defaultdict(list)
        self.memory_by_task_hash = defaultdict(list)
        self.memory_by_type = defaultdict(list)
        self.vector_store = None  # Will be initialized when vector db is available
        
        self.config = config
        self.memory_retention = config.get("memory_retention", 10000)
        self.max_memory_entropy = config.get("max_memory_entropy", 0.75)
        self.prioritization_weights = config.get("prioritization_weights", {
            "recency": 0.3,
            "success": 0.4,
            "relevance": 0.3,
            "novelty": 0.2,
            "doctrinal_alignment": 0.3
        })
        
    def store(self, 
             memory_content: Any,
             memory_type: str,
             task_context: Dict[str, Any],
             metadata: Dict[str, Any],
             vectors: Optional[Dict[str, np.ndarray]] = None) -> str:
        """Store a memory with full metadata and indexing."""
        # Generate memory ID and calculate hashes
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        context_hash = self._hash_dict(task_context)
        content_hash = self._hash_content(memory_content)
        
        # Create timestamp
        timestamp = time.time()
        
        # Create memory object
        memory = {
            "id": memory_id,
            "content": memory_content,
            "type": memory_type,
            "context": task_context,
            "context_hash": context_hash,
            "content_hash": content_hash,
            "metadata": metadata,
            "vectors": vectors,
            "created_at": timestamp,
            "last_accessed": timestamp,
            "access_count": 0,
            "success_rate": metadata.get("success", False),
            "entropy": metadata.get("entropy", 0.5),
            "doctrine_version": metadata.get("doctrine_version"),
            "parent_memory_id": metadata.get("parent_memory_id"),
            "mutation_id": metadata.get("mutation_id"),
            "suppressed": False,
            "suppression_reason": None
        }
        
        # Add to collections
        self.memories.append(memory)
        self.memory_by_id[memory_id] = memory
        self.memory_by_context[context_hash].append(memory_id)
        
        task_hash = self._hash_dict({"type": task_context.get("type", "unknown")})
        self.memory_by_task_hash[task_hash].append(memory_id)
        self.memory_by_type[memory_type].append(memory_id)
        
        # Add to vector store if vectors provided
        if vectors and self.vector_store:
            for key, vector in vectors.items():
                self.vector_store.add(memory_id, vector, {"field": key})
        
        # Enforce memory limits if needed
        if len(self.memories) > self.memory_retention:
            self._prune_memories()
            
        return memory_id
        
    def retrieve(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory by ID and update access metrics."""
        memory = self.memory_by_id.get(memory_id)
        
        if memory:
            # Update access metadata
            memory["last_accessed"] = time.time()
            memory["access_count"] += 1
            
        return memory
    
    def query_by_context(self, context: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Find memories with similar context."""
        context_hash = self._hash_dict(context)
        memory_ids = self.memory_by_context.get(context_hash, [])
        
        # If exact match not found, find partial matches
        if not memory_ids:
            # Find most similar contexts
            context_str = json.dumps(context, sort_keys=True)
            similarity_scores = []
            
            for hash_key in self.memory_by_context:
                # Get a sample memory from this context
                sample_id = self.memory_by_context[hash_key][0]
                sample_memory = self.memory_by_id[sample_id]
                sample_context_str = json.dumps(sample_memory["context"], sort_keys=True)
                
                # Simple string similarity (in real implementation use better metrics)
                score = self._simple_similarity(context_str, sample_context_str)
                similarity_scores.append((hash_key, score))
            
            # Sort by similarity and take top matches
            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            top_hashes = [h for h, _ in similarity_scores[:limit]]
            
            # Collect memory IDs from top matching contexts
            for hash_key in top_hashes:
                memory_ids.extend(self.memory_by_context[hash_key])
        
        # Retrieve actual memories
        memories = [self.retrieve(mid) for mid in memory_ids[:limit]]
        return [m for m in memories if m and not m["suppressed"]]
    
    def query_by_vector(self, vector: np.ndarray, field: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find memories with similar vector embeddings."""
        if not self.vector_store:
            return []
            
        results = self.vector_store.search(vector, filter={"field": field}, limit=limit)
        memories = [self.retrieve(result["id"]) for result in results]
        return [m for m in memories if m and not m["suppressed"]]
    
    def suppress_memory(self, memory_id: str, reason: str) -> bool:
        """Suppress a memory to prevent it from being used."""
        memory = self.memory_by_id.get(memory_id)
        if not memory:
            return False
            
        memory["suppressed"] = True
        memory["suppression_reason"] = reason
        memory["suppressed_at"] = time.time()
        return True
    
    def _prune_memories(self) -> None:
        """Remove lowest-value memories when capacity is reached."""
        # Score all memories
        scored_memories = []
        for memory in self.memories:
            score = self._calculate_memory_value(memory)
            scored_memories.append((memory["id"], score))
        
        # Sort by score ascending (lowest value first)
        scored_memories.sort(key=lambda x: x[1])
        
        # Remove lowest-scored memories to get back under limit
        to_remove = max(1, int(len(self.memories) * 0.05))  # Remove at least 1, up to 5%
        for i in range(to_remove):
            if i < len(scored_memories):
                memory_id = scored_memories[i][0]
                self._remove_memory(memory_id)
    
    def _remove_memory(self, memory_id: str) -> None:
        """Remove a memory from all indexes."""
        if memory_id not in self.memory_by_id:
            return
            
        memory = self.memory_by_id[memory_id]
        
        # Remove from all indexes
        self.memories = [m for m in self.memories if m["id"] != memory_id]
        self.memory_by_context[memory["context_hash"]] = [
            mid for mid in self.memory_by_context[memory["context_hash"]] 
            if mid != memory_id
        ]
        self.memory_by_type[memory["type"]] = [
            mid for mid in self.memory_by_type[memory["type"]] 
            if mid != memory_id
        ]
        
        # Remove from task hash index
        task_hash = self._hash_dict({"type": memory["context"].get("type", "unknown")})
        if task_hash in self.memory_by_task_hash:
            self.memory_by_task_hash[task_hash] = [
                mid for mid in self.memory_by_task_hash[task_hash] 
                if mid != memory_id
            ]
        
        # Remove from vector store
        if self.vector_store and memory.get("vectors"):
            for key in memory["vectors"]:
                self.vector_store.delete(memory_id, filter={"field": key})
                
        # Finally remove from ID index
        del self.memory_by_id[memory_id]
    
    def _calculate_memory_value(self, memory: Dict[str, Any]) -> float:
        """Calculate the value of a memory for retention decisions."""
        # Extract base metrics
        recency = 1.0 - min(1.0, (time.time() - memory["last_accessed"]) / 
                             (self.config.get("recency_horizon", 604800)))  # 1 week default
        success = float(memory["success_rate"])
        access_factor = min(1.0, memory["access_count"] / 10.0)  # Cap at 10 accesses
        entropy_factor = 1.0 - memory["entropy"]  # Lower entropy is better
        
        # Calculate weighted score
        weights = self.prioritization_weights
        score = (
            weights["recency"] * recency +
            weights["success"] * success +
            weights["novelty"] * entropy_factor +
            0.1 * access_factor  # Small bonus for frequently accessed memories
        )
        
        # Apply penalties
        if memory["suppressed"]:
            score *= 0.2  # Heavy penalty for suppressed memories
            
        return score
    
    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Create a stable hash from a dictionary."""
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()
    
    def _hash_content(self, content: Any) -> str:
        """Hash any content type."""
        if isinstance(content, dict):
            return self._hash_dict(content)
        elif isinstance(content, (list, tuple)):
            return hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()
        else:
            return hashlib.md5(str(content).encode()).hexdigest()
    
    def _simple_similarity(self, str1: str, str2: str) -> float:
        """
        Simple string similarity measure.
        In production, use proper semantic similarity measures.
        """
        # Just a placeholder for a better similarity function
        # For real use, implement or use proper similarity metrics
        if len(str1) == 0 or len(str2) == 0:
            return 0.0
        
        # Simple jaccard similarity on character 3-grams
        def get_ngrams(s, n=3):
            return [s[i:i+n] for i in range(max(0, len(s)-n+1))]
            
        ngrams1 = set(get_ngrams(str1))
        ngrams2 = set(get_ngrams(str2))
        
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)
