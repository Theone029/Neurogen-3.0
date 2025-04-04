import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

class ArbiterBreaker:
    """Resolves validation conflicts and makes intelligent decisions when standard arbitration fails."""
    
    def __init__(self, prime_directive, evolution_auditor, config: Dict[str, Any]):
        self.prime_directive = prime_directive
        self.evolution_auditor = evolution_auditor
        self.config = config
        
        # Conflict resolution strategies
        self.resolution_strategies = {
            "doctrine_precedence": self._resolve_with_doctrine_precedence,
            "validator_precedence": self._resolve_with_validator_precedence,
            "dynamic_weighting": self._resolve_with_dynamic_weighting,
            "intent_alignment": self._resolve_with_intent_alignment,
            "context_sensitive": self._resolve_with_context_sensitive,
            "coherence_maximizing": self._resolve_with_coherence_maximizing,
            "task_specific": self._resolve_with_task_specific
        }
        
        # Default resolution strategy and fallbacks
        self.default_strategy = config.get("default_strategy", "dynamic_weighting")
        self.fallback_strategies = config.get("fallback_strategies", [
            "context_sensitive", "doctrine_precedence"
        ])
        
        # Doctrine conflict thresholds
        self.doctrine_conflict_threshold = config.get("doctrine_conflict_threshold", 0.7)
        self.fork_recommendation_threshold = config.get("fork_recommendation_threshold", 0.85)
        
        # Task-type specific resolution preferences
        self.task_resolution_preferences = config.get("task_resolution_preferences", {
            "creative": "validator_precedence",
            "analytical": "doctrine_precedence",
            "critical": "coherence_maximizing"
        })
        
        # Statistics tracking
        self.stats = {
            "total_conflicts": 0,
            "strategies_used": {},
            "fork_recommendations": 0,
            "doctrine_precedence_rate": 0.0,
            "validation_precedence_rate": 0.0,
            "resolution_success_rate": 1.0  # Start optimistic
        }
    
    def resolve(self, 
               validation_result: Dict[str, Any],
               meta_judge_result: Dict[str, Any],
               arbitration_result: Dict[str, Any],
               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve conflict between validator and meta_judge when arbitrator can't decide.
        
        Args:
            validation_result: Result from validator
            meta_judge_result: Result from meta_judge
            arbitration_result: Result from validation_arbitrator
            context: Rich execution context
            
        Returns:
            Final resolution decision
        """
        # Generate resolution ID
        resolution_id = hashlib.md5(f"res:{time.time()}".encode()).hexdigest()[:12]
        
        # Track statistics
        self.stats["total_conflicts"] += 1
        
        # Check if this is actually a conflict
        if not arbitration_result.get("conflict", False):
            return {
                "resolution_id": resolution_id,
                "verdict": arbitration_result.get("verdict", "reject"),
                "reason": "No conflict detected, using arbitration verdict",
                "conflict": False,
                "fork_recommended": False
            }
        
        # Create conflict record
        conflict = {
            "validator": validation_result.get("valid", False),
            "validator_error": validation_result.get("error"),
            "meta_judge": meta_judge_result.get("valid", False),
            "meta_judge_error": meta_judge_result.get("error"),
            "severity": self._assess_conflict_severity(validation_result, meta_judge_result)
        }
        
        # Determine resolution strategy
        strategy = self._select_resolution_strategy(conflict, context)
        
        # Track strategy usage
        if strategy not in self.stats["strategies_used"]:
            self.stats["strategies_used"][strategy] = 0
        self.stats["strategies_used"][strategy] += 1
        
        # Apply resolution strategy
        resolution_fn = self.resolution_strategies.get(strategy, 
                                                     self.resolution_strategies[self.default_strategy])
        resolution = resolution_fn(conflict, validation_result, meta_judge_result, context)
        
        # Add resolution metadata
        resolution["resolution_id"] = resolution_id
        resolution["strategy"] = strategy
        resolution["conflict_severity"] = conflict["severity"]
        
        # Check if we should recommend a fork
        if conflict["severity"] >= self.fork_recommendation_threshold:
            resolution["fork_recommended"] = True
            self.stats["fork_recommendations"] += 1
        else:
            resolution["fork_recommended"] = False
        
        # Update precedence stats
        if resolution["verdict"] == "accept" and not meta_judge_result.get("valid", False):
            # Validation precedence
            old_count = self.stats["total_conflicts"] - 1
            old_rate = self.stats["validation_precedence_rate"]
            self.stats["validation_precedence_rate"] = (old_rate * old_count + 1) / self.stats["total_conflicts"]
            
        elif resolution["verdict"] == "accept" and not validation_result.get("valid", False):
            # Doctrine precedence
            old_count = self.stats["total_conflicts"] - 1
            old_rate = self.stats["doctrine_precedence_rate"]
            self.stats["doctrine_precedence_rate"] = (old_rate * old_count + 1) / self.stats["total_conflicts"]
        
        return resolution
    
    def _select_resolution_strategy(self, 
                                  conflict: Dict[str, Any], 
                                  context: Dict[str, Any]) -> str:
        """Select optimal resolution strategy based on conflict and context."""
        # Check for task-specific preference
        task_type = context.get("task", {}).get("type", "default")
        if task_type in self.task_resolution_preferences:
            return self.task_resolution_preferences[task_type]
        
        # For severe conflicts, use coherence_maximizing
        if conflict["severity"] >= self.fork_recommendation_threshold:
            return "coherence_maximizing"
        
        # Check intent to determine strategy
        intent = context.get("intent")
        if intent:
            intent_dims = {}
            if hasattr(intent, "get_vector_as_dict"):
                intent_dims = intent.get_vector_as_dict()
            elif isinstance(intent, dict):
                intent_dims = intent
                
            # If doctrine alignment is high, prioritize it
            if intent_dims.get("doctrinal_alignment", 0) > 0.8:
                return "doctrine_precedence"
                
            # If exploration is high, be more permissive with validation
            if intent_dims.get("exploration", 0) > 0.7:
                return "validator_precedence"
                
            # If knowledge or efficiency is high, use dynamic weighting
            if intent_dims.get("knowledge", 0) > 0.7 or intent_dims.get("efficiency", 0) > 0.7:
                return "dynamic_weighting"
        
        # Use context-sensitive if mutation count is high
        if context.get("mutation_count", 0) >= 2:
            return "context_sensitive"
        
        # Default to configured default strategy
        return self.default_strategy
    
    def _assess_conflict_severity(self, 
                               validation_result: Dict[str, Any], 
                               meta_judge_result: Dict[str, Any]) -> float:
        """Assess how severe the validation conflict is."""
        # Base level conflict severity
        severity = 0.5  # Start at middle severity
        
        # Factor 1: Error types - some are more severe than others
        validator_error = validation_result.get("error", "")
        metajudge_error = meta_judge_result.get("error", "")
        
        # Severity increases for certain error terms
        critical_terms = ["critical", "fatal", "violation", "breach", "unsafe"]
        for term in critical_terms:
            if term in str(validator_error).lower() or term in str(metajudge_error).lower():
                severity += 0.1
        
        # Severity increases if both sides have strong detailed reasons
        if (validation_result.get("details") and meta_judge_result.get("details") and
            "justification" in meta_judge_result):
            severity += 0.1
            
        # Severity increases if validation looks sound (multiple checks run)
        if "checks" in validation_result and len(validation_result["checks"]) > 2:
            severity += 0.05
            
        # Severity increases if meta_judge invoked core doctrine law
        if meta_judge_result.get("details") and "law_violation" in meta_judge_result["details"]:
            severity += 0.2
            
        # Cap severity between 0 and 1
        return min(1.0, max(0.0, severity))
        
    def _resolve_with_doctrine_precedence(self,
                                       conflict: Dict[str, Any],
                                       validation_result: Dict[str, Any],
                                       meta_judge_result: Dict[str, Any],
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict by prioritizing doctrinal alignment."""
        # If meta_judge approves, accept; otherwise reject
        verdict = "accept" if meta_judge_result.get("valid", False) else "reject"
        
        return {
            "verdict": verdict, 
            "reason": f"Doctrine precedence: {meta_judge_result.get('justification', 'Doctrinal alignment required')}",
            "conflict": True,
            "resolution_type": "doctrine_precedence"
        }
    
    def _resolve_with_validator_precedence(self,
                                        conflict: Dict[str, Any],
                                        validation_result: Dict[str, Any],
                                        meta_judge_result: Dict[str, Any],
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict by prioritizing technical validation."""
        # If validator approves, accept; otherwise reject
        verdict = "accept" if validation_result.get("valid", False) else "reject"
        
        reason = "Validator precedence"
        if verdict == "accept":
            reason += ": Output is technically valid despite doctrinal concerns"
        else:
            reason += f": {validation_result.get('error', 'Technical validation failed')}"
            
        return {
            "verdict": verdict,
            "reason": reason,
            "conflict": True,
            "resolution_type": "validator_precedence"
        }
    
    def _resolve_with_dynamic_weighting(self,
                                     conflict: Dict[str, Any],
                                     validation_result: Dict[str, Any],
                                     meta_judge_result: Dict[str, Any],
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict using dynamic weighting based on system state."""
        # Get state factors that influence weighting
        mutation_count = context.get("mutation_count", 0)
        task_type = context.get("task", {}).get("type", "unknown")
        is_creative = task_type in ["creative", "generative", "exploration"]
        success_history = self._get_success_history(context)
        
        # Calculate weights
        validator_weight = 0.5  # Base weight
        metajudge_weight = 0.5  # Base weight
        
        # Adjust weights based on context
        if mutation_count > 1:
            # After multiple mutation attempts, lean toward validation
            validator_weight += 0.1 * min(3, mutation_count - 1)
            metajudge_weight -= 0.1 * min(3, mutation_count - 1)
            
        if is_creative:
            # For creative tasks, lean toward validation
            validator_weight += 0.15
            metajudge_weight -= 0.15
            
        if success_history < 0.4:
            # When success rate is low, doctrine may be too restrictive
            validator_weight += 0.1
            metajudge_weight -= 0.1
            
        # Get core system state if available
        if self.evolution_auditor:
            coherence_trend = 0.0
            if hasattr(self.evolution_auditor, "get_coherence_trend"):
                trend = self.evolution_auditor.get_coherence_trend()
                coherence_trend = trend.get("trend", 0.0)
                
                # If coherence is declining, favor doctrine to stabilize
                if coherence_trend < -0.01:
                    validator_weight -= 0.1
                    metajudge_weight += 0.1
        
        # Normalize weights
        total_weight = validator_weight + metajudge_weight
        validator_weight /= total_weight
        metajudge_weight /= total_weight
        
        # Apply weighted decision
        validator_valid = validation_result.get("valid", False)
        metajudge_valid = meta_judge_result.get("valid", False)
        
        validator_score = 1.0 if validator_valid else 0.0
        metajudge_score = 1.0 if metajudge_valid else 0.0
        
        weighted_score = (validator_score * validator_weight + 
                          metajudge_score * metajudge_weight)
        
        # Decision threshold
        verdict = "accept" if weighted_score >= 0.5 else "reject"
        
        return {
            "verdict": verdict,
            "reason": f"Dynamic weighting (V:{validator_weight:.2f}/D:{metajudge_weight:.2f}) resulted in {verdict}",
            "conflict": True,
            "resolution_type": "dynamic_weighting",
            "weights": {
                "validator": validator_weight,
                "metajudge": metajudge_weight
            },
            "weighted_score": weighted_score
        }
    
    def _resolve_with_intent_alignment(self,
                                    conflict: Dict[str, Any],
                                    validation_result: Dict[str, Any],
                                    meta_judge_result: Dict[str, Any],
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict based on alignment with current intent vector."""
        # Extract intent dimensions
        intent = context.get("intent")
        intent_dims = {}
        
        if hasattr(intent, "get_vector_as_dict"):
            intent_dims = intent.get_vector_as_dict()
        elif isinstance(intent, dict):
            intent_dims = intent
            
        if not intent_dims:
            # Fall back to dynamic weighting if intent not available
            return self._resolve_with_dynamic_weighting(
                conflict, validation_result, meta_judge_result, context)
        
        # Determine which dominant intent dimensions should influence decision
        dominant_dims = sorted(
            intent_dims.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]  # Consider top 3 dimensions
        
        # Resolve based on dominant intent
        if dominant_dims:
            top_dim, top_value = dominant_dims[0]
            
            # Different resolution strategies based on intent
            if top_dim == "exploration" and top_value > 0.6:
                # With high exploration, prioritize validation over doctrine
                verdict = "accept" if validation_result.get("valid", False) else "reject"
                return {
                    "verdict": verdict,
                    "reason": f"Intent alignment: High exploration ({top_value:.2f}) prioritizes validation",
                    "conflict": True,
                    "resolution_type": "intent_alignment",
                    "intent_dimension": top_dim
                }
                
            elif top_dim == "stability" and top_value > 0.6:
                # With high stability, prioritize doctrine over validation
                verdict = "accept" if meta_judge_result.get("valid", False) else "reject"
                return {
                    "verdict": verdict,
                    "reason": f"Intent alignment: High stability ({top_value:.2f}) prioritizes doctrine",
                    "conflict": True,
                    "resolution_type": "intent_alignment",
                    "intent_dimension": top_dim
                }
                
            elif top_dim == "efficiency" and top_value > 0.6:
                # With high efficiency, accept if any validation passes
                verdict = "accept" if (validation_result.get("valid", False) or 
                                      meta_judge_result.get("valid", False)) else "reject"
                return {
                    "verdict": verdict,
                    "reason": f"Intent alignment: High efficiency ({top_value:.2f}) accepts any passing validation",
                    "conflict": True,
                    "resolution_type": "intent_alignment",
                    "intent_dimension": top_dim
                }
                
            elif top_dim == "coherence" and top_value > 0.6:
                # With high coherence, check which option is more coherent with past decisions
                if self.evolution_auditor:
                    # Favor previous consistent decisions
                    doctrine_rate = self.stats["doctrine_precedence_rate"]
                    validation_rate = self.stats["validation_precedence_rate"]
                    
                    if doctrine_rate > validation_rate + 0.2:
                        verdict = "accept" if meta_judge_result.get("valid", False) else "reject"
                        return {
                            "verdict": verdict,
                            "reason": f"Intent alignment: High coherence ({top_value:.2f}) follows historical doctrine precedence",
                            "conflict": True,
                            "resolution_type": "intent_alignment",
                            "intent_dimension": top_dim
                        }
                    elif validation_rate > doctrine_rate + 0.2:
                        verdict = "accept" if validation_result.get("valid", False) else "reject"
                        return {
                            "verdict": verdict,
                            "reason": f"Intent alignment: High coherence ({top_value:.2f}) follows historical validation precedence",
                            "conflict": True,
                            "resolution_type": "intent_alignment",
                            "intent_dimension": top_dim
                        }
        
        # If no clear intent-based resolution, fall back to dynamic weighting
        dynamic_resolution = self._resolve_with_dynamic_weighting(
            conflict, validation_result, meta_judge_result, context)
        dynamic_resolution["resolution_type"] = "intent_alignment_fallback"
        
        return dynamic_resolution
    
    def _resolve_with_context_sensitive(self,
                                     conflict: Dict[str, Any],
                                     validation_result: Dict[str, Any],
                                     meta_judge_result: Dict[str, Any],
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict using rich context-sensitive analysis."""
        # Extract key context elements
        task_type = context.get("task", {}).get("type", "unknown")
        mutation_count = context.get("mutation_count", 0)
        output = context.get("output")
        plan = context.get("plan", {})
        
        # Check mutation context
        if mutation_count >= 2:
            # After multiple mutations, be more lenient
            if validation_result.get("valid", False):
                return {
                    "verdict": "accept",
                    "reason": f"Context-sensitive: After {mutation_count} mutations, accepting technically valid output",
                    "conflict": True,
                    "resolution_type": "context_sensitive"
                }
        
        # Check specific task types
        if task_type == "creative":
            # For creative tasks, technical validity matters more
            if validation_result.get("valid", False):
                return {
                    "verdict": "accept",
                    "reason": "Context-sensitive: Creative task prioritizes technical validity",
                    "conflict": True, 
                    "resolution_type": "context_sensitive"
                }
                
        elif task_type in ["critical", "security", "safety"]:
            # For critical tasks, doctrine matters more
            if not meta_judge_result.get("valid", False):
                return {
                    "verdict": "reject",
                    "reason": f"Context-sensitive: {task_type} task requires doctrinal compliance",
                    "conflict": True,
                    "resolution_type": "context_sensitive"
                }
        
        # Check plan complexity
        if plan and "steps" in plan:
            steps_count = len(plan["steps"])
            if steps_count > 5 and validation_result.get("valid", False):
                # For complex plans that validate, be more lenient on doctrine
                return {
                    "verdict": "accept",
                    "reason": f"Context-sensitive: Complex plan ({steps_count} steps) with technical validity accepted",
                    "conflict": True,
                    "resolution_type": "context_sensitive"
                }
        
        # Check output type specifics
        if output and isinstance(output, dict) and "error" in output:
            # Output already contains error information
            return {
                "verdict": "reject",
                "reason": "Context-sensitive: Output contains error information, maintaining rejection",
                "conflict": True,
                "resolution_type": "context_sensitive"
            }
            
        # Fall back to conflict severity assessment
        if conflict["severity"] > 0.7:
            return {
                "verdict": "reject",
                "reason": f"Context-sensitive: High conflict severity ({conflict['severity']:.2f}) requires rejection",
                "conflict": True,
                "resolution_type": "context_sensitive"
            }
        
        # Default to accepting if technically valid
        if validation_result.get("valid", False):
            return {
                "verdict": "accept",
                "reason": "Context-sensitive: Technically valid with acceptable doctrinal conflict",
                "conflict": True,
                "resolution_type": "context_sensitive"
            }
        
        # Otherwise reject
        return {
            "verdict": "reject",
            "reason": "Context-sensitive: Neither validation nor doctrine support acceptance",
            "conflict": True,
            "resolution_type": "context_sensitive"
        }
    
    def _resolve_with_coherence_maximizing(self,
                                        conflict: Dict[str, Any],
                                        validation_result: Dict[str, Any],
                                        meta_judge_result: Dict[str, Any],
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict by maximizing system coherence."""
        # This is the most sophisticated resolution strategy, used for severe conflicts
        
        # Check if we can get coherence trend
        coherence_data = {}
        if self.evolution_auditor and hasattr(self.evolution_auditor, "get_coherence_trend"):
            coherence_data = self.evolution_auditor.get_coherence_trend()
        
        # Default verdict based on current coherence state
        default_verdict = "reject"  # Conservative default
        
        if coherence_data:
            coherence_trend = coherence_data.get("trend", 0.0)
            
            # If coherence is declining significantly, prefer doctrinal alignment
            if coherence_trend < -0.02:
                verdict = "accept" if meta_judge_result.get("valid", False) else "reject"
                return {
                    "verdict": verdict,
                    "reason": f"Coherence maximizing: Declining coherence ({coherence_trend:.3f}) prioritizes doctrine",
                    "conflict": True,
                    "resolution_type": "coherence_maximizing",
                    "fork_recommended": conflict["severity"] >= self.fork_recommendation_threshold
                }
            
            # If coherence is stable or improving, can be more flexible
            if coherence_trend >= 0:
                # Allow validation to override if technically valid and not critical doctrine issue
                if validation_result.get("valid", False) and conflict["severity"] < 0.7:
                    return {
                        "verdict": "accept",
                        "reason": f"Coherence maximizing: Stable coherence ({coherence_trend:.3f}) allows validation precedence",
                        "conflict": True,
                        "resolution_type": "coherence_maximizing",
                        "fork_recommended": conflict["severity"] >= self.fork_recommendation_threshold
                    }
        
        # For high-severity conflicts, recommend fork instead of forcing a choice
        if conflict["severity"] >= self.fork_recommendation_threshold:
            return {
                "verdict": "reject",  # Conservative default when recommending fork
                "reason": f"Coherence maximizing: High-severity conflict ({conflict['severity']:.2f}) recommends system fork",
                "conflict": True,
                "resolution_type": "coherence_maximizing",
                "fork_recommended": True
            }
        
        # Check if Prime Directive gives clear guidance
        if self.prime_directive:
            curr_doctrine = self.prime_directive.get_current_version()
            
            # Check core laws against conflict
            if "core_laws" in curr_doctrine:
                for law in curr_doctrine["core_laws"]:
                    # Very basic check - in real implementation, use more sophisticated matching
                    law_lower = law.lower()
                    meta_error = meta_judge_result.get("error", "").lower()
                    
                    # If meta_judge error directly relates to a core law, enforce doctrine
                    for word in meta_error.split():
                        if word and len(word) > 4 and word in law_lower:
                            return {
                                "verdict": "reject",
                                "reason": f"Coherence maximizing: Core doctrine law violation detected",
                                "conflict": True,
                                "resolution_type": "coherence_maximizing"
                            }
            
            # Check alignment vectors
            if "alignment_vectors" in curr_doctrine:
                # Implementation would check if output aligns with alignment vectors
                pass
        
        # Default to compromise - allow validation to win but flag for review
        if validation_result.get("valid", False):
            return {
                "verdict": "accept",
                "reason": "Coherence maximizing: Technically valid but requires doctrinal review",
                "conflict": True,
                "resolution_type": "coherence_maximizing",
                "requires_review": True
            }
        else:
            return {
                "verdict": "reject",
                "reason": "Coherence maximizing: Neither validation nor doctrine support acceptance",
                "conflict": True,
                "resolution_type": "coherence_maximizing"
            }
    
    def _resolve_with_task_specific(self,
                                 conflict: Dict[str, Any],
                                 validation_result: Dict[str, Any],
                                 meta_judge_result: Dict[str, Any],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict based on specific task requirements."""
        task = context.get("task", {})
        task_type = task.get("type", "unknown")
        task_requirements = task.get("requirements", {})
        
        # Check for explicit validation requirements
        if "strict_validation" in task_requirements and task_requirements["strict_validation"]:
            # Task explicitly requires strict validation
            verdict = "accept" if validation_result.get("valid", False) else "reject"
            return {
                "verdict": verdict,
                "reason": f"Task-specific: {task_type} requires strict validation",
                "conflict": True,
                "resolution_type": "task_specific"
            }
            
        # Check for explicit doctrine requirements
        if "strict_doctrine" in task_requirements and task_requirements["strict_doctrine"]:
            # Task explicitly requires strict doctrinal compliance
            verdict = "accept" if meta_judge_result.get("valid", False) else "reject"
            return {
                "verdict": verdict,
                "reason": f"Task-specific: {task_type} requires strict doctrinal compliance",
                "conflict": True,
                "resolution_type": "task_specific"
            }
        
        # Handle specific task types
        if task_type == "problem_solving":
            # For problem-solving, validate solution correctness first
            if validation_result.get("valid", False):
                return {
                    "verdict": "accept",
                    "reason": "Task-specific: Problem-solving prioritizes solution correctness",
                    "conflict": True,
                    "resolution_type": "task_specific"
                }
                
        elif task_type == "exploration":
            # For exploration, allow more latitude if technically valid
            if validation_result.get("valid", False):
                return {
                    "verdict": "accept",
                    "reason": "Task-specific: Exploration task permits doctrinal flexibility",
                    "conflict": True,
                    "resolution_type": "task_specific"
                }
                
        elif task_type == "doctrine_test":
            # For doctrine tests, enforce strict compliance
            verdict = "accept" if meta_judge_result.get("valid", False) else "reject"
            return {
                "verdict": verdict,
                "reason": "Task-specific: Doctrine test requires strict compliance",
                "conflict": True,
                "resolution_type": "task_specific"
            }
        
        # Fall back to dynamic weighting for other task types
        dynamic_resolution = self._resolve_with_dynamic_weighting(
            conflict, validation_result, meta_judge_result, context)
        dynamic_resolution["resolution_type"] = "task_specific_fallback"
        
        return dynamic_resolution
    
    def _get_success_history(self, context: Dict[str, Any]) -> float:
        """Get historical success rate from context."""
        # First try getting it from context
        if "success_history" in context:
            return context["success_history"]
            
        # Try getting from evolution auditor
        if self.evolution_auditor and hasattr(self.evolution_auditor, "get_system_state_report"):
            state = self.evolution_auditor.get_system_state_report()
            if "doctrine" in state and "recent_success_rate" in state["doctrine"]:
                return state["doctrine"]["recent_success_rate"]
        
        # Default middle value
        return 0.5
