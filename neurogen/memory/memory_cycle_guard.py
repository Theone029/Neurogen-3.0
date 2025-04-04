import time
from typing import Dict, List, Any, Optional, Set, Tuple
import hashlib

class MemoryCycleGuard:
    """Prevents recursive memory loops and stale pattern reuse."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_usage = {}  # memory_id -> usage stats
        self.cycle_detection = {}  # cycle signature -> occurrences
        self.suppressed = set()  # set of suppressed memory_ids
        self.suppression_reasons = {}  # memory_id -> reason
        
        # Configuration
        self.reuse_threshold = config.get("reuse_threshold", 5)
        self.recency_window = config.get("recency_window", 100)  # cycles
        self.novelty_threshold = config.get("novelty_threshold", 0.3)
        self.cyclic_pattern_threshold = config.get("cyclic_pattern_threshold", 3)
        
        # Stats
        self.stats = {
            "suppressions": 0,
            "cycles_detected": 0,
            "reuse_suppressions": 0,
            "novelty_suppressions": 0,
            "validation_decay_suppressions": 0
        }
        
    def check_memory(self, memory_id: str, context: Dict[str, Any]) -> bool:
        """
        Check if memory should be allowed or suppressed.
        
        Args:
            memory_id: Memory identifier
            context: Current execution context
            
        Returns:
            True if memory is allowed, False if suppressed
        """
        # If already suppressed, maintain suppression
        if memory_id in self.suppressed:
            return False
            
        # Initialize tracking if this is a new memory
        if memory_id not in self.memory_usage:
            self.memory_usage[memory_id] = {
                "access_count": 0,
                "recent_accesses": [],  # list of cycle ids
                "recent_contexts": [],  # list of context hashes
                "novelty_scores": [],  # list of novelty scores
                "validation_history": [],  # list of validation results
                "last_access": time.time()
            }
        
        # Update usage stats
        usage = self.memory_usage[memory_id]
        usage["access_count"] += 1
        usage["last_access"] = time.time()
        
        # Add current cycle to recent accesses
        cycle_id = context.get("cycle_id", str(int(time.time())))
        usage["recent_accesses"].append(cycle_id)
        
        # Trim history if needed
        if len(usage["recent_accesses"]) > self.recency_window:
            usage["recent_accesses"] = usage["recent_accesses"][-self.recency_window:]
            usage["recent_contexts"] = usage["recent_contexts"][-self.recency_window:]
            usage["novelty_scores"] = usage["novelty_scores"][-self.recency_window:]
            usage["validation_history"] = usage["validation_history"][-self.recency_window:]
        
        # Check conditions for suppression
        
        # 1. Reuse threshold check
        if usage["access_count"] > self.reuse_threshold:
            # Check recency - if used too many times in recent cycles
            recent_count = len(usage["recent_accesses"])
            if recent_count > self.reuse_threshold * 0.8:  # 80% of threshold in recent window
                self._suppress_memory(memory_id, "excessive_reuse", context)
                self.stats["reuse_suppressions"] += 1
                return False
        
        # 2. Check for cyclic patterns
        context_hash = self._hash_context(context)
        usage["recent_contexts"].append(context_hash)
        
        cycle_signature = self._detect_cycle(memory_id, context_hash)
        if cycle_signature:
            # Increment occurrence counter
            if cycle_signature not in self.cycle_detection:
                self.cycle_detection[cycle_signature] = 0
            self.cycle_detection[cycle_signature] += 1
            
            # Check if cycle threshold exceeded
            if self.cycle_detection[cycle_signature] >= self.cyclic_pattern_threshold:
                self._suppress_memory(memory_id, "cyclic_pattern", context)
                self.stats["cycles_detected"] += 1
                return False
        
        # 3. Novelty check - if we have output and previous outputs to compare
        if "output" in context and "previous_outputs" in context:
            novelty_score = self._calculate_novelty(
                context["output"], 
                context["previous_outputs"]
            )
            usage["novelty_scores"].append(novelty_score)
            
            # Check if novelty is consistently low
            if (len(usage["novelty_scores"]) >= 3 and
                all(score < self.novelty_threshold for score in usage["novelty_scores"][-3:])):
                self._suppress_memory(memory_id, "low_novelty", context)
                self.stats["novelty_suppressions"] += 1
                return False
        
        # 4. Validation decay check
        if "validation_result" in context:
            validation = context["validation_result"]
            usage["validation_history"].append(validation)
            
            # Check if validation success is declining
            if len(usage["validation_history"]) >= 3:
                recent_validation = usage["validation_history"][-3:]
                # If was successful but now failing
                if recent_validation[0] and not recent_validation[-1]:
                    self._suppress_memory(memory_id, "validation_decay", context)
                    self.stats["validation_decay_suppressions"] += 1
                    return False
        
        # Memory passes all checks
        return True
    
    def is_suppressed(self, memory_id: str) -> bool:
        """Check if a memory is currently suppressed."""
        return memory_id in self.suppressed
    
    def get_suppression_reason(self, memory_id: str) -> Optional[str]:
        """Get the reason a memory was suppressed."""
        return self.suppression_reasons.get(memory_id)
    
    def reactivate_memory(self, memory_id: str) -> bool:
        """
        Attempt to reactivate a suppressed memory.
        
        Returns:
            True if successfully reactivated, False if wasn't suppressed
        """
        if memory_id in self.suppressed:
            self.suppressed.remove(memory_id)
            if memory_id in self.suppression_reasons:
                del self.suppression_reasons[memory_id]
            return True
        return False
    
    def clear_cyclic_detection(self) -> None:
        """Reset cyclic pattern detection counters."""
        self.cycle_detection = {}
    
    def _suppress_memory(self, memory_id: str, reason: str, context: Dict[str, Any]) -> None:
        """Suppress a memory with a given reason."""
        self.suppressed.add(memory_id)
        self.suppression_reasons[memory_id] = reason
        self.stats["suppressions"] += 1
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create a stable hash from context for cycle detection."""
        # Extract stable elements from context
        stable_elements = {
            "task_type": context.get("task", {}).get("type", "unknown"),
            "cycle_id": context.get("cycle_id", "unknown"),
            "intent_type": str(type(context.get("intent", None)))
        }
        
        # Convert to string and hash
        context_str = str(stable_elements)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def _detect_cycle(self, memory_id: str, context_hash: str) -> Optional[str]:
        """
        Detect if a cyclic pattern is forming.
        
        Returns:
            Cycle signature if detected, None otherwise
        """
        usage = self.memory_usage[memory_id]
        contexts = usage["recent_contexts"]
        
        # Need at least 4 contexts to detect meaningful cycles
        if len(contexts) < 4:
            return None
            
        # Check for repeating patterns in context access
        # (A,B,C,A,B,C) pattern detection
        for pattern_length in range(2, len(contexts) // 2 + 1):
            # Get latest pattern and check if it repeats
            latest_pattern = contexts[-pattern_length:]
            previous_pattern = contexts[-(pattern_length*2):-pattern_length]
            
            if latest_pattern == previous_pattern:
                # Found a repeating pattern
                pattern_sig = f"{memory_id}:{'.'.join(latest_pattern)}"
                return pattern_sig
                
        return None
    
    def _calculate_novelty(self, current_output: Any, previous_outputs: List[Any]) -> float:
        """
        Calculate novelty score between current output and previous outputs.
        
        Returns:
            Novelty score (0-1), higher is more novel
        """
        # Simple implementation - in production would use semantic comparison
        # or more sophisticated difference metrics
        
        if not previous_outputs:
            return 1.0  # Maximum novelty if no previous outputs
            
        # Convert outputs to strings for comparison
        current_str = str(current_output)
        
        # Calculate similarity to most recent previous output
        most_recent = str(previous_outputs[-1])
        
        # Simple Jaccard similarity on character trigrams
        current_trigrams = self._get_trigrams(current_str)
        recent_trigrams = self._get_trigrams(most_recent)
        
        if not current_trigrams or not recent_trigrams:
            return 0.5  # Neutral score for empty strings
            
        # Calculate Jaccard similarity
        intersection = len(current_trigrams.intersection(recent_trigrams))
        union = len(current_trigrams.union(recent_trigrams))
        
        if union == 0:
            return 0.5
            
        similarity = intersection / union
        
        # Convert similarity to novelty (1 - similarity)
        return 1.0 - similarity
    
    def _get_trigrams(self, text: str) -> Set[str]:
        """Generate character trigrams from text."""
        return set(text[i:i+3] for i in range(len(text)-2))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cycle guard statistics."""
        return {
            "total_memories_tracked": len(self.memory_usage),
            "suppressed_memories": len(self.suppressed),
            "suppressions": self.stats["suppressions"],
            "cycles_detected": self.stats["cycles_detected"],
            "suppression_reasons": {
                "reuse": self.stats["reuse_suppressions"],
                "novelty": self.stats["novelty_suppressions"],
                "validation_decay": self.stats["validation_decay_suppressions"],
                "cycles": self.stats["cycles_detected"]
            }
        }
