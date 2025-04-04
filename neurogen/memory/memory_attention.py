import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
import time

class MemoryAttention:
    """Weights and filters memories for coherent and contradiction-free injection."""
    
    def __init__(self, memory_cycle_guard, config: Dict[str, Any]):
        self.memory_cycle_guard = memory_cycle_guard
        self.config = config
        self.attention_stats = {
            "total_weightings": 0,
            "avg_input_size": 0,
            "avg_output_size": 0,
            "contradictions_detected": 0
        }
        
    def weight_and_filter(self, 
                         memories: List[Dict[str, Any]], 
                         context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Weight memories by relevance and filter out contradictions.
        
        Args:
            memories: List of candidate memories
            context: Current execution context
            
        Returns:
            Filtered and weighted memories
        """
        if not memories:
            return []
            
        start_time = time.time()
        self.attention_stats["total_weightings"] += 1
        self.attention_stats["avg_input_size"] = (
            (self.attention_stats["avg_input_size"] * 
             (self.attention_stats["total_weightings"] - 1) + 
             len(memories)) / self.attention_stats["total_weightings"]
        )
        
        # Check each memory against cycle guard
        if self.memory_cycle_guard:
            memories = [m for m in memories if not 
                        self.memory_cycle_guard.is_suppressed(m["id"])]
        
        # Calculate weights for each memory
        weighted_memories = []
        for memory in memories:
            weight = self._calculate_weight(memory, context)
            weighted_memories.append((memory, weight))
        
        # Sort by weight (descending)
        weighted_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out contradictions
        filtered_memories = self._filter_contradictions(
            [m for m, _ in weighted_memories], context)
        
        self.attention_stats["avg_output_size"] = (
            (self.attention_stats["avg_output_size"] * 
             (self.attention_stats["total_weightings"] - 1) + 
             len(filtered_memories)) / self.attention_stats["total_weightings"]
        )
        
        # Return filtered memories with weights attached
        for memory in filtered_memories:
            memory["attention_weight"] = next(
                w for m, w in weighted_memories if m["id"] == memory["id"])
        
        return filtered_memories
    
    def _calculate_weight(self, memory: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate attention weight for a memory based on context."""
        weight = 0.0
        
        # Get weighting components based on config
        weights = self.config.get("weight_factors", {
            "age": 0.2,
            "task_similarity": 0.3,
            "reward": 0.2,
            "doctrine_version": 0.15,
            "repetition": 0.15
        })
        
        # Age weighting (prefer newer memories)
        if "created_at" in memory:
            age_seconds = time.time() - memory["created_at"]
            max_age = self.config.get("max_memory_age_seconds", 2592000)  # 30 days default
            age_factor = max(0, 1.0 - (age_seconds / max_age))
            weight += weights["age"] * age_factor
        
        # Task similarity
        if "context" in memory and "task" in context:
            similarity = self._task_similarity(memory["context"], context["task"])
            weight += weights["task_similarity"] * similarity
        
        # Reward score
        if "metadata" in memory and "reward" in memory["metadata"]:
            reward = memory["metadata"]["reward"]
            weight += weights["reward"] * reward
        
        # Doctrine version compatibility
        if "metadata" in memory and "doctrine_version" in memory["metadata"]:
            # Check if memory was created with current doctrine
            current_doctrine = context.get("doctrine", {}).get("version_id", "unknown")
            memory_doctrine = memory["metadata"]["doctrine_version"]
            
            if current_doctrine == memory_doctrine:
                weight += weights["doctrine_version"]  # Full weight if exact match
            else:
                # Partial weight if different but not incompatible
                # This would need more sophisticated version comparison
                weight += weights["doctrine_version"] * 0.5
        
        # Repetition penalty
        if "access_count" in memory:
            access_count = memory["access_count"]
            repetition_factor = max(0, 1.0 - (access_count / 
                                             self.config.get("repetition_threshold", 10)))
            weight += weights["repetition"] * repetition_factor
        
        return weight
    
    def _filter_contradictions(self, 
                             memories: List[Dict[str, Any]], 
                             context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter out contradictory memories from the set.
        
        Args:
            memories: List of memories to check
            context: Current execution context
            
        Returns:
            List with contradictions removed
        """
        if len(memories) <= 1:
            return memories
            
        # Sort by weight if available
        memories = sorted(
            memories, 
            key=lambda m: m.get("attention_weight", 0), 
            reverse=True
        )
        
        result = []
        contradictions = 0
        
        # Keep track of key claims
        claims = set()
        
        for memory in memories:
            # Check if memory contradicts any previously accepted memory
            if self._is_contradictory(memory, claims):
                contradictions += 1
                continue
                
            # Extract claims from this memory
            memory_claims = self._extract_claims(memory)
            claims.update(memory_claims)
            
            # Add to result
            result.append(memory)
        
        if contradictions > 0:
            self.attention_stats["contradictions_detected"] += contradictions
            
        return result
    
    def _is_contradictory(self, memory: Dict[str, Any], claims: Set[str]) -> bool:
        """Check if memory contradicts any existing claims."""
        # Extract claims from this memory
        memory_claims = self._extract_claims(memory)
        
        # Check for direct contradictions (simplified)
        for claim in memory_claims:
            # Very simple contradiction check - in reality would be more sophisticated
            if claim.startswith("not_") and claim[4:] in claims:
                return True
                
            if not claim.startswith("not_") and f"not_{claim}" in claims:
                return True
                
        return False
    
    def _extract_claims(self, memory: Dict[str, Any]) -> Set[str]:
        """Extract claims from a memory for contradiction checking."""
        # This is a simplistic implementation
        # In a real system, this would use more sophisticated NLP or logic extraction
        
        claims = set()
        
        # Try to extract claims from memory content
        content = memory.get("content", {})
        
        # For execution results, look at success/failure
        if memory.get("type") == "execution_result":
            if content.get("success", False):
                claims.add("success")
            else:
                claims.add("not_success")
                
        # Look for explicit claims in memory metadata
        if "metadata" in memory and "claims" in memory["metadata"]:
            claims.update(memory["metadata"]["claims"])
            
        return claims
    
    def _task_similarity(self, memory_context: Dict[str, Any], task: Dict[str, Any]) -> float:
        """Calculate similarity between memory context and current task."""
        # Basic similarity check (type match)
        if memory_context.get("type") == task.get("type"):
            return 0.8
            
        # Check for partial matches in content if available
        if "content" in memory_context and "content" in task:
            # In a real implementation, use semantic similarity
            # This is a simplistic approach
            memory_content_str = str(memory_context["content"]).lower()
            task_content_str = str(task["content"]).lower()
            
            # Check for shared words
            memory_words = set(memory_content_str.split())
            task_words = set(task_content_str.split())
            
            common_words = memory_words.intersection(task_words)
            
            if common_words:
                return 0.5 * (len(common_words) / 
                             max(len(memory_words), len(task_words)))
        
        # Default low similarity
        return 0.2
