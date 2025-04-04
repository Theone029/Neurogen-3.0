import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import deque

class MutationMemory:
    """Tracks failed mutations and their contexts to prevent retry loops."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Core memory stores
        self.failures = {}  # plan_hash -> failure records
        self.chains = {}    # mutation_id -> chain history
        self.pattern_store = {}  # error_pattern -> count
        
        # LRU caches for fast lookups
        self.recent_failures = deque(maxlen=config.get("recent_failure_cache", 100))
        self.recent_mutations = deque(maxlen=config.get("recent_mutation_cache", 200))
        
        # Pattern matching configuration
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.max_chain_length = config.get("max_chain_length", 5)
        self.pattern_expiry = config.get("pattern_expiry_seconds", 3600)  # 1 hour
        
        # Statistics
        self.stats = {
            "total_failures": 0,
            "unique_failures": 0,
            "retry_preventions": 0,
            "pattern_matches": 0,
            "successful_mutations": 0
        }
    
    def record_failure(self, 
                     plan: Dict[str, Any],
                     error: Dict[str, Any],
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record a mutation failure for future reference.
        
        Args:
            plan: The plan that failed
            error: Error details
            context: Execution context
            
        Returns:
            Failure record details
        """
        self.stats["total_failures"] += 1
        
        # Generate plan hash
        plan_hash = self._hash_plan(plan)
        
        # Extract mutation metadata if available
        mutation_id = plan.get("mutation_metadata", {}).get("mutation_id", "original")
        parent_id = plan.get("mutation_metadata", {}).get("parent_id", None)
        mutation_strategy = plan.get("mutation_metadata", {}).get("strategy", None)
        
        # Create failure record
        failure_record = {
            "timestamp": time.time(),
            "plan_hash": plan_hash,
            "error_type": error.get("type", "unknown"),
            "error_message": error.get("message", ""),
            "error_trace": error.get("traceback", None),
            "mutation_id": mutation_id,
            "parent_id": parent_id,
            "strategy": mutation_strategy,
            "task_type": context.get("task", {}).get("type", "unknown"),
            "mutations": []  # Will track subsequent mutations that try to fix this
        }
        
        # Extract error pattern
        error_pattern = self._extract_error_pattern(error)
        failure_record["error_pattern"] = error_pattern
        
        # Store in failure dictionary
        if plan_hash not in self.failures:
            self.failures[plan_hash] = []
            self.stats["unique_failures"] += 1
            
        self.failures[plan_hash].append(failure_record)
        
        # Add to recent failures cache
        self.recent_failures.append((plan_hash, error_pattern, time.time()))
        
        # Update mutation chain if this is part of a chain
        if parent_id:
            self._update_mutation_chain(mutation_id, parent_id, False)
        
        # Update pattern store
        if error_pattern:
            if error_pattern not in self.pattern_store:
                self.pattern_store[error_pattern] = {
                    "count": 0,
                    "last_seen": time.time(),
                    "strategies": {}
                }
            
            pattern_data = self.pattern_store[error_pattern]
            pattern_data["count"] += 1
            pattern_data["last_seen"] = time.time()
            
            if mutation_strategy:
                if mutation_strategy not in pattern_data["strategies"]:
                    pattern_data["strategies"][mutation_strategy] = {"attempts": 0, "successes": 0}
                pattern_data["strategies"][mutation_strategy]["attempts"] += 1
        
        return failure_record
    
    def record_success(self, 
                     plan: Dict[str, Any],
                     output: Any,
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record a successful mutation to improve future mutations.
        
        Args:
            plan: The successful plan
            output: Output produced
            context: Execution context
            
        Returns:
            Success record details
        """
        # Extract mutation metadata if available
        mutation_id = plan.get("mutation_metadata", {}).get("mutation_id", None)
        parent_id = plan.get("mutation_metadata", {}).get("parent_id", None)
        mutation_strategy = plan.get("mutation_metadata", {}).get("strategy", None)
        
        if not mutation_id or not parent_id:
            # Not a mutation, nothing to record
            return {"success": True, "recorded": False}
        
        self.stats["successful_mutations"] += 1
        
        # Update mutation chain
        chain_record = self._update_mutation_chain(mutation_id, parent_id, True)
        
        # Update pattern store if this strategy fixed a known error pattern
        if mutation_strategy and parent_id:
            # Find the parent's error pattern
            parent_error_pattern = None
            parent_plan_hash = None
            
            # Search for parent in recent failures
            for ph, ep, _ in self.recent_failures:
                # Check if this failure has the parent ID
                for failure in self.failures.get(ph, []):
                    if failure.get("mutation_id") == parent_id:
                        parent_error_pattern = failure.get("error_pattern")
                        parent_plan_hash = ph
                        break
                
                if parent_error_pattern:
                    break
            
            # If we found the parent's error pattern, update success stats
            if parent_error_pattern and parent_error_pattern in self.pattern_store:
                pattern_data = self.pattern_store[parent_error_pattern]
                
                if mutation_strategy in pattern_data["strategies"]:
                    pattern_data["strategies"][mutation_strategy]["successes"] += 1
                
                # Add to the failure record's mutations list
                if parent_plan_hash in self.failures:
                    for failure in self.failures[parent_plan_hash]:
                        if failure.get("mutation_id") == parent_id:
                            failure["mutations"].append({
                                "mutation_id": mutation_id,
                                "strategy": mutation_strategy,
                                "success": True,
                                "timestamp": time.time()
                            })
                            break
        
        return {
            "success": True,
            "recorded": True,
            "mutation_id": mutation_id,
            "chain_length": chain_record.get("chain_length", 1) if chain_record else 1
        }
    
    def find_similar_failures(self, 
                           error: Dict[str, Any], 
                           plan_hash: str) -> List[Dict[str, Any]]:
        """
        Find similar failures to avoid repeating failed mutations.
        
        Args:
            error: Current error details
            plan_hash: Hash of the current plan
            
        Returns:
            List of similar failure records
        """
        # Extract error pattern
        error_pattern = self._extract_error_pattern(error)
        
        # Start with exact matches for this plan
        matches = []
        exact_matches = self.failures.get(plan_hash, [])
        if exact_matches:
            matches.extend(exact_matches)
        
        # Look for similar error patterns across all failures
        if error_pattern:
            # First check exact pattern matches
            for failure_hash, failures in self.failures.items():
                if failure_hash == plan_hash:
                    continue  # Already added
                    
                for failure in failures:
                    if failure.get("error_pattern") == error_pattern:
                        matches.append(failure)
                        self.stats["pattern_matches"] += 1
            
            # If we don't have enough matches, look for similar patterns
            if len(matches) < 3:  # Arbitrary threshold
                for failure_hash, failures in self.failures.items():
                    if failure_hash == plan_hash:
                        continue
                        
                    for failure in failures:
                        failure_pattern = failure.get("error_pattern", "")
                        if failure_pattern and failure not in matches:
                            similarity = self._pattern_similarity(error_pattern, failure_pattern)
                            if similarity >= self.similarity_threshold:
                                matches.append(failure)
                                self.stats["pattern_matches"] += 1
        
        return matches
    
    def should_retry(self, 
                   error: Dict[str, Any], 
                   plan_hash: str, 
                   parent_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Determine if a mutation should be retried based on history.
        
        Args:
            error: Current error details
            plan_hash: Hash of the current plan
            parent_id: Optional parent mutation ID
            
        Returns:
            (should_retry, reason) tuple
        """
        # Check chain length limits
        if parent_id and self._get_chain_length(parent_id) >= self.max_chain_length:
            self.stats["retry_preventions"] += 1
            return False, "mutation_chain_too_long"
        
        # Extract error pattern
        error_pattern = self._extract_error_pattern(error)
        
        # Check for repeated identical failures
        if plan_hash in self.failures:
            identical_count = 0
            
            for failure in self.failures[plan_hash]:
                if failure.get("error_pattern") == error_pattern:
                    identical_count += 1
                    
                    # If we've seen this exact error multiple times, don't retry
                    if identical_count >= 3:  # Arbitrary threshold
                        self.stats["retry_preventions"] += 1
                        return False, "repeated_identical_failure"
        
        # Check if this error pattern has been attempted many times with no success
        if error_pattern in self.pattern_store:
            pattern_data = self.pattern_store[error_pattern]
            
            # If pattern is very common and has low success rate, avoid retrying
            if pattern_data["count"] > 10:
                total_attempts = sum(s["attempts"] for s in pattern_data["strategies"].values())
                total_successes = sum(s["successes"] for s in pattern_data["strategies"].values())
                
                if total_attempts > 0 and total_successes / total_attempts < 0.1:
                    # Less than 10% success rate for this pattern
                    self.stats["retry_preventions"] += 1
                    return False, "low_success_rate_pattern"
        
        # Check if there's a promising strategy for this error pattern
        recommended_strategy = self._recommend_strategy(error_pattern)
        
        # Allow retry
        return True, recommended_strategy
    
    def count_chain_mutations(self, mutation_id: str) -> int:
        """Count how many mutations are in this chain."""
        return self._get_chain_length(mutation_id)
    
    def get_mutation_success_rate(self, strategy: Optional[str] = None) -> float:
        """
        Get success rate for mutations, optionally filtered by strategy.
        
        Args:
            strategy: Optional strategy name to filter by
            
        Returns:
            Success rate (0-1)
        """
        if strategy:
            # Get success rate for specific strategy
            attempts = 0
            successes = 0
            
            for pattern_data in self.pattern_store.values():
                if strategy in pattern_data["strategies"]:
                    strategy_data = pattern_data["strategies"][strategy]
                    attempts += strategy_data["attempts"]
                    successes += strategy_data["successes"]
            
            return successes / attempts if attempts > 0 else 0.0
        else:
            # Overall success rate
            return (self.stats["successful_mutations"] / 
                   max(1, self.stats["total_failures"]))
    
    def cleanup_expired_patterns(self) -> int:
        """Clean up expired error patterns to prevent memory bloat."""
        now = time.time()
        expired_count = 0
        
        # Remove expired patterns
        expired_patterns = []
        for pattern, data in self.pattern_store.items():
            if now - data["last_seen"] > self.pattern_expiry:
                expired_patterns.append(pattern)
                expired_count += 1
        
        for pattern in expired_patterns:
            del self.pattern_store[pattern]
        
        return expired_count
    
    def _hash_plan(self, plan: Dict[str, Any]) -> str:
        """Create a stable hash of a plan."""
        # Clone plan and remove non-structural elements
        plan_copy = dict(plan)
        if "mutation_metadata" in plan_copy:
            del plan_copy["mutation_metadata"]
        
        # Simplistic hash creation - in production use more robust approach
        plan_str = str(sorted(plan_copy.items()))
        return hashlib.md5(plan_str.encode()).hexdigest()
    
    def _extract_error_pattern(self, error: Dict[str, Any]) -> Optional[str]:
        """Extract a pattern that captures the essence of an error."""
        if not error:
            return None
            
        # Start with error type
        pattern_parts = [error.get("type", "unknown")]
        
        # Add error message, but clean it up first
        message = error.get("message", "")
        if message:
            # Remove specific values that would prevent pattern matching
            # This is a simplified example - real implementation would be more sophisticated
            message = message.split("\n")[0]  # Only use first line
            
            # Replace specific numbers, hashes, timestamps
            import re
            message = re.sub(r'\b[0-9a-f]{7,40}\b', 'HASH', message)
            message = re.sub(r'\b\d+\b', 'NUM', message)
            
            pattern_parts.append(message)
        
        # Additional context about where the error occurred, if available
        if "validator" in error and error["validator"]:
            pattern_parts.append("validator")
            
        if "meta_judge" in error and error["meta_judge"]:
            pattern_parts.append("doctrine")
        
        # Combine parts into pattern
        return "::".join(pattern_parts)
    
    def _pattern_similarity(self, pattern1: str, pattern2: str) -> float:
        """Calculate similarity between two error patterns."""
        # Very basic implementation - in production use more sophisticated 
        # text similarity algorithm
        
        if not pattern1 or not pattern2:
            return 0.0
            
        # Split into parts
        parts1 = pattern1.split("::")
        parts2 = pattern2.split("::")
        
        # Compare types (first part)
        type_match = parts1[0] == parts2[0]
        if not type_match:
            return 0.0  # Different error types are not similar
            
        # If we only have types, they're similar but not very
        if len(parts1) == 1 and len(parts2) == 1:
            return 0.6
            
        # Compare messages if available
        message_similarity = 0.0
        if len(parts1) > 1 and len(parts2) > 1:
            msg1, msg2 = parts1[1], parts2[1]
            
            # Calculate Jaccard similarity of words
            words1 = set(msg1.split())
            words2 = set(msg2.split())
            
            if words1 and words2:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                
                message_similarity = intersection / union
        
        # Overall similarity is combination of type match and message similarity
        return 0.4 + (0.6 * message_similarity) if type_match else 0.0
    
    def _update_mutation_chain(self, 
                           mutation_id: str, 
                           parent_id: str, 
                           success: bool) -> Dict[str, Any]:
        """Update the tracked mutation chain."""
        # Create or update chain record
        if parent_id not in self.chains:
            self.chains[parent_id] = {
                "children": set(),
                "success": False,
                "chain_length": 1,
                "last_updated": time.time()
            }
        
        # Add current mutation to parent's children
        parent_chain = self.chains[parent_id]
        parent_chain["children"].add(mutation_id)
        parent_chain["last_updated"] = time.time()
        
        # Create record for current mutation
        if mutation_id not in self.chains:
            chain_length = parent_chain["chain_length"] + 1
            self.chains[mutation_id] = {
                "children": set(),
                "success": success,
                "parent_id": parent_id,
                "chain_length": chain_length,
                "last_updated": time.time()
            }
        else:
            # Update existing record
            self.chains[mutation_id]["success"] = success
            self.chains[mutation_id]["last_updated"] = time.time()
            
        # Add to recent mutations cache
        self.recent_mutations.append((mutation_id, parent_id, success, time.time()))
        
        return self.chains[mutation_id]
    
    def _get_chain_length(self, mutation_id: str) -> int:
        """Get the length of a mutation chain."""
        if mutation_id not in self.chains:
            return 1
            
        return self.chains[mutation_id]["chain_length"]
    
    def _recommend_strategy(self, error_pattern: str) -> Optional[str]:
        """Recommend a mutation strategy based on past successes."""
        if not error_pattern or error_pattern not in self.pattern_store:
            return None
            
        pattern_data = self.pattern_store[error_pattern]
        if not pattern_data["strategies"]:
            return None
            
        # Calculate success rates for each strategy
        strategy_scores = {}
        for strategy, data in pattern_data["strategies"].items():
            attempts = data["attempts"]
            successes = data["successes"]
            
            # Only consider strategies with multiple attempts
            if attempts >= 2:
                # Calculate score based on success rate
                success_rate = successes / attempts
                strategy_scores[strategy] = success_rate
        
        # Find the strategy with the highest success rate
        if strategy_scores:
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
            if best_strategy[1] > 0:  # Only recommend if there's some success
                return best_strategy[0]
        
        return None
