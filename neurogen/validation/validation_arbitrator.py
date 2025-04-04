from typing import Dict, Any, Optional
import time
import hashlib

class ValidationArbitrator:
    """
    Mediator between validator and meta_judge that resolves validation conflicts 
    and determines if outputs should be accepted.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.arbitration_stats = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "conflicts": 0
        }
        
    def arbitrate(self, 
                validator_result: Dict[str, Any], 
                meta_judge_result: Dict[str, Any],
                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Arbitrate between validator and meta_judge results.
        
        Args:
            validator_result: Result from validator
            meta_judge_result: Result from meta_judge
            context: Additional context for decision making
            
        Returns:
            Arbitration decision
        """
        self.arbitration_stats["total"] += 1
        arbitration_id = hashlib.md5(f"arb:{time.time()}".encode()).hexdigest()[:12]
        
        # Extract validation results
        validator_valid = validator_result.get("valid", False)
        doctrine_valid = meta_judge_result.get("valid", False)
        
        # Initialize result
        result = {
            "arbitration_id": arbitration_id,
            "verdict": "reject",  # Default to reject
            "conflict": False,
            "conflict_details": None,
            "reason": None
        }
        
        # Simple case: both agree
        if validator_valid and doctrine_valid:
            result["verdict"] = "accept"
            result["reason"] = "Both validator and doctrine checks passed"
            self.arbitration_stats["accepted"] += 1
            return result
            
        if not validator_valid and not doctrine_valid:
            result["verdict"] = "reject"
            result["reason"] = "Both validator and doctrine checks failed"
            self.arbitration_stats["rejected"] += 1
            return result
        
        # Conflict case
        self.arbitration_stats["conflicts"] += 1
        result["conflict"] = True
        
        # Extract error details
        validator_error = validator_result.get("error", "Unknown validator error")
        doctrine_error = meta_judge_result.get("error", "Unknown doctrine error")
        
        result["conflict_details"] = {
            "validator": validator_valid,
            "doctrine": doctrine_valid,
            "validator_error": validator_error,
            "doctrine_error": doctrine_error
        }
        
        # Apply conflict resolution strategy based on configuration
        strategy = self.config.get("conflict_strategy", "conservative")
        
        if strategy == "conservative":
            # Conservative approach: reject if either validation fails
            result["verdict"] = "reject"
            result["reason"] = "Conservative strategy rejects on any validation failure"
            
            if validator_valid:
                result["reason"] = f"Doctrine check failed: {doctrine_error}"
            else:
                result["reason"] = f"Validator check failed: {validator_error}"
                
            self.arbitration_stats["rejected"] += 1
            
        elif strategy == "validator_priority":
            # Prioritize validator over doctrine
            result["verdict"] = "accept" if validator_valid else "reject"
            result["reason"] = "Validator result prioritized over doctrine"
            
            if not validator_valid:
                result["reason"] = f"Validator check failed: {validator_error}"
                self.arbitration_stats["rejected"] += 1
            else:
                self.arbitration_stats["accepted"] += 1
                
        elif strategy == "doctrine_priority":
            # Prioritize doctrine over validator
            result["verdict"] = "accept" if doctrine_valid else "reject"
            result["reason"] = "Doctrine result prioritized over validator"
            
            if not doctrine_valid:
                result["reason"] = f"Doctrine check failed: {doctrine_error}"
                self.arbitration_stats["rejected"] += 1
            else:
                self.arbitration_stats["accepted"] += 1
                
        elif strategy == "context_aware":
            # Use context to make a more nuanced decision
            result = self._context_aware_decision(
                validator_valid, 
                doctrine_valid,
                validator_error,
                doctrine_error,
                context,
                result
            )
            
        else:
            # Default to reject on conflict
            result["verdict"] = "reject"
            result["reason"] = "Unknown conflict resolution strategy, defaulting to reject"
            self.arbitration_stats["rejected"] += 1
            
        # Record stats
        if result["verdict"] == "accept":
            self.arbitration_stats["accepted"] += 1
        else:
            self.arbitration_stats["rejected"] += 1
            
        return result
    
    def _context_aware_decision(self,
                             validator_valid: bool,
                             doctrine_valid: bool,
                             validator_error: str,
                             doctrine_error: str,
                             context: Dict[str, Any],
                             result: Dict[str, Any]) -> Dict[str, Any]:
        """Make a context-aware arbitration decision."""
        # Extract relevant context
        task_type = context.get("task", {}).get("type", "unknown")
        mutation_count = context.get("mutation_count", 0)
        
        # Different strategies based on task type
        if task_type == "creative" and validator_valid and not doctrine_valid:
            # For creative tasks, be more permissive about doctrine if validator passes
            if "intent" in context:
                # Check if exploration is prioritized in intent
                exploration_value = 0.0
                intent = context["intent"]
                
                if hasattr(intent, "get_dimension_value"):
                    exploration_value = intent.get_dimension_value("exploration")
                elif isinstance(intent, dict) and "exploration" in intent:
                    exploration_value = intent["exploration"]
                
                if exploration_value > 0.6:  # High exploration intent
                    result["verdict"] = "accept"
                    result["reason"] = "Accepted: Creative task with high exploration intent, validator passed"
                    return result
        
        elif task_type == "analytical" and not validator_valid and doctrine_valid:
            # For analytical tasks, be more strict about validation
            result["verdict"] = "reject"
            result["reason"] = f"Rejected: Analytical task requires validator pass: {validator_error}"
            return result
        
        # Consider mutation count in decision
        if mutation_count >= self.config.get("max_mutations", 3):
            # We've tried several mutations already, be more lenient
            if validator_valid and not doctrine_valid:
                # Minor doctrine issues might be acceptable after multiple mutations
                if "minor" in doctrine_error.lower() or "non-critical" in doctrine_error.lower():
                    result["verdict"] = "accept"
                    result["reason"] = f"Accepted after {mutation_count} mutations with minor doctrine issue"
                    return result
            
            elif not validator_valid and doctrine_valid:
                # Minor validation issues might be acceptable after multiple mutations
                if "warning" in validator_error.lower() or "non-critical" in validator_error.lower():
                    result["verdict"] = "accept"
                    result["reason"] = f"Accepted after {mutation_count} mutations with minor validation issue"
                    return result
        
        # Default to rejection on conflict
        result["verdict"] = "reject"
        if validator_valid:
            result["reason"] = f"Rejected due to doctrine issue: {doctrine_error}"
        else:
            result["reason"] = f"Rejected due to validation issue: {validator_error}"
            
        return result
