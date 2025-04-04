import os
import json
import time
import shutil
import pickle
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np

class ForkReconciler:
    """Manages the integration of divergent system states after forking."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Output directory for reconciliation artifacts
        self.reconciliation_dir = config.get("reconciliation_dir", "reconciliations")
        os.makedirs(self.reconciliation_dir, exist_ok=True)
        
        # Reconciliation weighting
        self.metrics_weights = config.get("metrics_weights", {
            "coherence": 0.3,
            "reward_trend": 0.2,
            "success_rate": 0.2,
            "drift": 0.15,
            "error_rate": 0.15
        })
        
        # Component integration strategies
        self.component_strategies = config.get("component_strategies", {
            "memory": "winner_plus_select",  # Winner's memories + select loser memories
            "doctrine": "winner_always",     # Winner's doctrine
            "intent": "weighted_average",    # Weighted average of vectors
            "constraints": "adaptive_merge"  # Selective merger based on performance
        })
        
        # Stats tracking
        self.stats = {
            "total_reconciliations": 0,
            "clean_wins": 0,          # Clear winner
            "contested_wins": 0,      # Close competition
            "integration_errors": 0,  # Errors during integration
            "avg_reconciliation_time": 0
        }
    
    def reconcile(self, 
                winner: Dict[str, Any], 
                loser: Dict[str, Any],
                comparison: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconcile fork states, preserving valuable information from both.
        
        Args:
            winner: The winning fork state
            loser: The losing fork state
            comparison: Comparison data explaining the winner selection
            
        Returns:
            Reconciliation result with merged state
        """
        start_time = time.time()
        self.stats["total_reconciliations"] += 1
        
        # Create reconciliation record
        reconciliation_id = f"recon_{int(time.time())}_{self.stats['total_reconciliations']}"
        reconciliation_dir = os.path.join(self.reconciliation_dir, reconciliation_id)
        os.makedirs(reconciliation_dir, exist_ok=True)
        
        # Initialize result
        result = {
            "reconciliation_id": reconciliation_id,
            "timestamp": time.time(),
            "winner_id": comparison.get("winner"),
            "margin": comparison.get("margin", 0),
            "success": False,
            "components_merged": {}
        }
        
        # Determine if this was a clean or contested win
        contested = comparison.get("margin", 1.0) < self.config.get("contested_threshold", 0.2)
        result["contested"] = contested
        
        if contested:
            self.stats["contested_wins"] += 1
        else:
            self.stats["clean_wins"] += 1
        
        try:
            # Extract state components from both forks
            winner_state = self._extract_state_components(winner)
            loser_state = self._extract_state_components(loser)
            
            # Create merged state
            merged_state = {}
            merge_logs = {}
            
            # Merge memories
            memories_result = self._merge_memories(
                winner_state.get("memory", {}),
                loser_state.get("memory", {}),
                contested
            )
            merged_state["memory"] = memories_result["merged"]
            merge_logs["memory"] = memories_result["log"]
            result["components_merged"]["memory"] = memories_result["stats"]
            
            # Merge doctrine
            doctrine_result = self._merge_doctrine(
                winner_state.get("doctrine", {}),
                loser_state.get("doctrine", {}),
                contested
            )
            merged_state["doctrine"] = doctrine_result["merged"]
            merge_logs["doctrine"] = doctrine_result["log"]
            result["components_merged"]["doctrine"] = doctrine_result["stats"]
            
            # Merge intent vector
            intent_result = self._merge_intent(
                winner_state.get("intent", {}),
                loser_state.get("intent", {}),
                contested
            )
            merged_state["intent"] = intent_result["merged"]
            merge_logs["intent"] = intent_result["log"]
            result["components_merged"]["intent"] = intent_result["stats"]
            
            # Merge constraints
            constraints_result = self._merge_constraints(
                winner_state.get("constraints", {}),
                loser_state.get("constraints", {}),
                comparison
            )
            merged_state["constraints"] = constraints_result["merged"]
            merge_logs["constraints"] = constraints_result["log"]
            result["components_merged"]["constraints"] = constraints_result["stats"]
            
            # Save merge logs
            with open(os.path.join(reconciliation_dir, "merge_logs.json"), 'w') as f:
                json.dump(merge_logs, f, indent=2)
                
            # Save merged state
            with open(os.path.join(reconciliation_dir, "merged_state.json"), 'w') as f:
                json.dump(merged_state, f, indent=2)
                
            # Save full state in pickle format if configured
            if self.config.get("save_full_state", False):
                try:
                    with open(os.path.join(reconciliation_dir, "merged_state.pickle"), 'wb') as f:
                        pickle.dump(merged_state, f)
                except:
                    # Pickling can fail for many reasons
                    pass
            
            # Mark as successful
            result["success"] = True
            result["merged_state"] = merged_state
            
        except Exception as e:
            self.stats["integration_errors"] += 1
            result["success"] = False
            result["error"] = str(e)
            
        # Update timing stats
        reconciliation_time = time.time() - start_time
        self.stats["avg_reconciliation_time"] = (
            (self.stats["avg_reconciliation_time"] * (self.stats["total_reconciliations"] - 1) + 
             reconciliation_time) / self.stats["total_reconciliations"]
        )
        
        result["reconciliation_time"] = reconciliation_time
        
        return result
    
    def _extract_state_components(self, fork_state: Any) -> Dict[str, Any]:
        """Extract key state components from a fork instance."""
        components = {}
        
        # Extract memory if available
        if hasattr(fork_state, "agent_memory"):
            memory_data = self._extract_memory(fork_state.agent_memory)
            components["memory"] = memory_data
            
        # Extract doctrine if available
        if hasattr(fork_state, "prime_directive"):
            doctrine_data = self._extract_doctrine(fork_state.prime_directive)
            components["doctrine"] = doctrine_data
            
        # Extract intent vector if available
        if hasattr(fork_state, "intent_vector"):
            intent_data = self._extract_intent(fork_state.intent_vector)
            components["intent"] = intent_data
            
        # Extract constraints if available
        if hasattr(fork_state, "constraint_controller"):
            constraint_data = self._extract_constraints(fork_state.constraint_controller)
            components["constraints"] = constraint_data
            
        # Extract metrics if available
        if hasattr(fork_state, "evolution_auditor"):
            evolution_data = self._extract_evolution(fork_state.evolution_auditor)
            components["evolution"] = evolution_data
            
        # Extract stats if available
        if hasattr(fork_state, "stats"):
            components["stats"] = fork_state.stats
            
        return components
    
    def _extract_memory(self, memory_module: Any) -> Dict[str, Any]:
        """Extract memory records from agent_memory."""
        memory_data = {
            "records": [],
            "stats": {}
        }
        
        # Extract memory records if available
        if hasattr(memory_module, "memories"):
            memory_data["records"] = memory_module.memories
            
        # Extract memory stats if available
        if hasattr(memory_module, "stats"):
            memory_data["stats"] = memory_module.stats
            
        return memory_data
    
    def _extract_doctrine(self, doctrine_module: Any) -> Dict[str, Any]:
        """Extract doctrine data from prime_directive."""
        doctrine_data = {}
        
        # Get current doctrine version
        if hasattr(doctrine_module, "get_current_version"):
            doctrine_data["current"] = doctrine_module.get_current_version()
            
        # Get doctrine history if available
        if hasattr(doctrine_module, "version_history"):
            doctrine_data["history"] = doctrine_module.version_history
            
        return doctrine_data
    
    def _extract_intent(self, intent_module: Any) -> Dict[str, Any]:
        """Extract intent vector data."""
        intent_data = {}
        
        # Get current intent vector
        if hasattr(intent_module, "get_vector_as_dict"):
            intent_data["vector"] = intent_module.get_vector_as_dict()
        elif hasattr(intent_module, "current"):
            if hasattr(intent_module.current, "tolist"):
                intent_data["vector"] = intent_module.current.tolist()
            else:
                intent_data["vector"] = intent_module.current
        
        # Get dimension names if available
        if hasattr(intent_module, "dimension_names"):
            intent_data["dimensions"] = intent_module.dimension_names
            
        return intent_data
    
    def _extract_constraints(self, constraint_module: Any) -> Dict[str, Any]:
        """Extract constraint data."""
        constraint_data = {}
        
        # Get current constraints
        if hasattr(constraint_module, "current_constraints"):
            constraint_data["current"] = constraint_module.current_constraints
            
        # Get constraint bounds if available
        if hasattr(constraint_module, "constraint_bounds"):
            constraint_data["bounds"] = constraint_module.constraint_bounds
            
        # Get stats if available
        if hasattr(constraint_module, "stats"):
            constraint_data["stats"] = constraint_module.stats
            
        return constraint_data
    
    def _extract_evolution(self, evolution_module: Any) -> Dict[str, Any]:
        """Extract evolution metrics."""
        evolution_data = {}
        
        # Get coherence trend
        if hasattr(evolution_module, "get_coherence_trend"):
            evolution_data["coherence_trend"] = evolution_module.get_coherence_trend()
            
        # Get reward trend
        if hasattr(evolution_module, "get_reward_trend"):
            evolution_data["reward_trend"] = evolution_module.get_reward_trend()
            
        # Get system state report
        if hasattr(evolution_module, "get_system_state_report"):
            evolution_data["state_report"] = evolution_module.get_system_state_report()
            
        return evolution_data
    
    def _merge_memories(self, 
                      winner_memory: Dict[str, Any], 
                      loser_memory: Dict[str, Any],
                      contested: bool) -> Dict[str, Any]:
        """Merge memory records from both forks."""
        strategy = self.component_strategies.get("memory", "winner_plus_select")
        
        # Start with winner's memories
        merged_records = winner_memory.get("records", [])[:]
        merged_stats = winner_memory.get("stats", {}).copy()
        
        # Get memory IDs from winner for deduplication
        winner_ids = set(m.get("id") for m in merged_records if "id" in m)
        
        # Track what we're doing
        merge_log = {
            "strategy": strategy,
            "winner_memories": len(merged_records),
            "loser_memories": len(loser_memory.get("records", [])),
            "integrated_from_loser": 0
        }
        
        if strategy == "winner_only":
            # Nothing more to do
            pass
            
        elif strategy == "winner_plus_select":
            # Add high-value memories from loser
            loser_records = loser_memory.get("records", [])
            
            # Define selection criteria
            selection_threshold = 0.7 if contested else 0.85
            
            # Select high-value memories from loser
            for memory in loser_records:
                memory_id = memory.get("id")
                
                # Skip if already in winner's memories
                if memory_id in winner_ids:
                    continue
                
                # Evaluate memory value
                memory_value = self._evaluate_memory_value(memory)
                
                # If memory is valuable, add it
                if memory_value >= selection_threshold:
                    merged_records.append(memory)
                    winner_ids.add(memory_id)
                    merge_log["integrated_from_loser"] += 1
                    
        elif strategy == "full_merge":
            # Merge all unique memories
            loser_records = loser_memory.get("records", [])
            
            for memory in loser_records:
                memory_id = memory.get("id")
                
                # Skip if already in winner's memories
                if memory_id in winner_ids:
                    continue
                
                merged_records.append(memory)
                winner_ids.add(memory_id)
                merge_log["integrated_from_loser"] += 1
        
        # Update stats
        merged_stats["total_memories"] = len(merged_records)
        
        return {
            "merged": {
                "records": merged_records,
                "stats": merged_stats
            },
            "log": merge_log,
            "stats": {
                "winner_count": len(winner_memory.get("records", [])),
                "loser_count": len(loser_memory.get("records", [])),
                "merged_count": len(merged_records),
                "integrated_count": merge_log["integrated_from_loser"]
            }
        }
    
    def _merge_doctrine(self, 
                      winner_doctrine: Dict[str, Any], 
                      loser_doctrine: Dict[str, Any],
                      contested: bool) -> Dict[str, Any]:
        """Merge doctrine from both forks."""
        strategy = self.component_strategies.get("doctrine", "winner_always")
        
        # Start with winner's doctrine
        merged_doctrine = winner_doctrine.get("current", {}).copy()
        
        # Track what we're doing
        merge_log = {
            "strategy": strategy,
            "winner_version": winner_doctrine.get("current", {}).get("version_id", "unknown"),
            "loser_version": loser_doctrine.get("current", {}).get("version_id", "unknown"),
            "changes": []
        }
        
        if strategy == "winner_always":
            # Nothing more to do
            pass
            
        elif strategy == "selective_law_integration" and contested:
            # Only for contested reconciliations, integrate beneficial laws
            winner_laws = winner_doctrine.get("current", {}).get("core_laws", [])
            loser_laws = loser_doctrine.get("current", {}).get("core_laws", [])
            
            # Find unique laws in loser's doctrine
            unique_loser_laws = [law for law in loser_laws if law not in winner_laws]
            
            # Evaluate and integrate beneficial laws
            for law in unique_loser_laws:
                # In a real implementation, evaluate if law is beneficial
                # For this example, we'll just integrate laws that don't conflict
                if not self._law_conflicts_with_existing(law, winner_laws):
                    if "core_laws" not in merged_doctrine:
                        merged_doctrine["core_laws"] = []
                    merged_doctrine["core_laws"].append(law)
                    merge_log["changes"].append({
                        "type": "add_law",
                        "law": law
                    })
        
        elif strategy == "alignment_vector_averaging" and contested:
            # Average alignment vectors for contested reconciliations
            winner_alignment = winner_doctrine.get("current", {}).get("alignment_vectors", {})
            loser_alignment = loser_doctrine.get("current", {}).get("alignment_vectors", {})
            
            if winner_alignment and loser_alignment:
                merged_alignment = self._merge_alignment_vectors(winner_alignment, loser_alignment)
                merged_doctrine["alignment_vectors"] = merged_alignment
                merge_log["changes"].append({
                    "type": "merge_alignment",
                    "result": merged_alignment
                })
        
        # Update version info
        if "version_id" in merged_doctrine:
            merged_doctrine["version_id"] = f"{merged_doctrine['version_id']}_reconciled"
            
        if "version_history" in winner_doctrine:
            merged_doctrine["version_history"] = winner_doctrine["version_history"][:]
            
        return {
            "merged": {
                "current": merged_doctrine,
                "history": winner_doctrine.get("history", [])
            },
            "log": merge_log,
            "stats": {
                "changes": len(merge_log["changes"]),
                "contested": contested
            }
        }
    
    def _merge_intent(self, 
                    winner_intent: Dict[str, Any], 
                    loser_intent: Dict[str, Any],
                    contested: bool) -> Dict[str, Any]:
        """Merge intent vectors from both forks."""
        strategy = self.component_strategies.get("intent", "weighted_average")
        
        # Get vectors
        winner_vector = winner_intent.get("vector", {})
        loser_vector = loser_intent.get("vector", {})
        
        if not winner_vector:
            return {
                "merged": winner_intent,
                "log": {"strategy": "winner_only", "reason": "No winner vector"},
                "stats": {"dimensions": 0}
            }
            
        # Track what we're doing
        merge_log = {
            "strategy": strategy,
            "winner_dimensions": len(winner_vector) if isinstance(winner_vector, dict) else 0,
            "loser_dimensions": len(loser_vector) if isinstance(loser_vector, dict) else 0
        }
        
        # Different merging strategies
        if strategy == "winner_always" or not contested:
            # Just use winner's intent
            merged_intent = winner_intent.copy()
            
        elif strategy == "weighted_average" and contested:
            # For contested reconciliations, do weighted averaging
            if isinstance(winner_vector, dict) and isinstance(loser_vector, dict):
                # Determine weights
                winner_weight = 0.7  # Winner still has higher weight
                loser_weight = 0.3
                
                # Generate merged vector
                merged_vector = {}
                all_dims = set(winner_vector.keys()).union(loser_vector.keys())
                
                for dim in all_dims:
                    winner_val = winner_vector.get(dim, 0.0)
                    loser_val = loser_vector.get(dim, 0.0)
                    
                    # Weighted average
                    merged_vector[dim] = (winner_val * winner_weight + 
                                        loser_val * loser_weight)
                
                # Create merged intent
                merged_intent = winner_intent.copy()
                merged_intent["vector"] = merged_vector
                
                merge_log["dimensions"] = len(merged_vector)
                merge_log["averaging_weights"] = {
                    "winner": winner_weight,
                    "loser": loser_weight
                }
                
            elif hasattr(winner_vector, "__len__") and hasattr(loser_vector, "__len__"):
                # Vector as list/array
                if len(winner_vector) == len(loser_vector):
                    winner_weight = 0.7
                    loser_weight = 0.3
                    
                    # Weighted average
                    merged_vector = []
                    for i in range(len(winner_vector)):
                        merged_val = (winner_vector[i] * winner_weight + 
                                     loser_vector[i] * loser_weight)
                        merged_vector.append(merged_val)
                    
                    # Create merged intent
                    merged_intent = winner_intent.copy()
                    merged_intent["vector"] = merged_vector
                    
                    merge_log["dimensions"] = len(merged_vector)
                    merge_log["averaging_weights"] = {
                        "winner": winner_weight,
                        "loser": loser_weight
                    }
                else:
                    # Dimension mismatch, use winner
                    merged_intent = winner_intent.copy()
                    merge_log["reason"] = "Dimension mismatch"
            else:
                # Unknown format, use winner
                merged_intent = winner_intent.copy()
                merge_log["reason"] = "Unknown vector format"
        else:
            # Fallback to winner
            merged_intent = winner_intent.copy()
        
        return {
            "merged": merged_intent,
            "log": merge_log,
            "stats": {
                "dimensions": (len(merged_intent.get("vector", {})) 
                             if isinstance(merged_intent.get("vector", {}), dict) 
                             else 0),
                "contested": contested
            }
        }
    
    def _merge_constraints(self, 
                         winner_constraints: Dict[str, Any], 
                         loser_constraints: Dict[str, Any],
                         comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Merge constraints from both forks."""
        strategy = self.component_strategies.get("constraints", "adaptive_merge")
        
        # Get current constraints
        winner_current = winner_constraints.get("current", {})
        loser_current = loser_constraints.get("current", {})
        
        if not winner_current:
            return {
                "merged": winner_constraints,
                "log": {"strategy": "winner_only", "reason": "No winner constraints"},
                "stats": {"keys": 0}
            }
            
        # Track what we're doing
        merge_log = {
            "strategy": strategy,
            "winner_keys": len(winner_current),
            "loser_keys": len(loser_current) if loser_current else 0,
            "changes": []
        }
        
        # Start with winner's constraints
        merged_constraints = {
            "current": winner_current.copy(),
            "bounds": winner_constraints.get("bounds", {}),
            "stats": winner_constraints.get("stats", {})
        }
        
        # Apply different strategies
        if strategy == "winner_always":
            # Nothing more to do
            pass
            
        elif strategy == "adaptive_merge":
            # Check component-specific performance metrics
            winner_metrics = comparison.get("metrics", {}).get("winner", {})
            loser_metrics = comparison.get("metrics", {}).get("loser", {})
            
            # Get performance keys for adaptive merging
            loser_advantages = self._identify_loser_advantages(winner_metrics, loser_metrics)
            
            # Use loser's constraints in areas where it had advantages
            for key in loser_current:
                # Only consider keys that exist in both
                if key in winner_current:
                    # If loser performed better in related metrics, consider its value
                    if key in loser_advantages or self._constraint_related_to_advantage(key, loser_advantages):
                        # Weighted average for contested constraints
                        winner_val = winner_current[key]
                        loser_val = loser_current[key]
                        
                        # Different handling based on value type
                        if isinstance(winner_val, (int, float)) and isinstance(loser_val, (int, float)):
                            # For numerical constraints, use weighted average
                            merged_val = winner_val * 0.3 + loser_val * 0.7  # Higher weight for better performer
                            
                            # Round to int if both were integers
                            if isinstance(winner_val, int) and isinstance(loser_val, int):
                                merged_val = int(round(merged_val))
                                
                            merged_constraints["current"][key] = merged_val
                            merge_log["changes"].append({
                                "type": "merge_numerical",
                                "key": key,
                                "winner_val": winner_val,
                                "loser_val": loser_val,
                                "merged_val": merged_val
                            })
                        else:
                            # For non-numerical, use loser's value directly
                            merged_constraints["current"][key] = loser_val
                            merge_log["changes"].append({
                                "type": "use_loser",
                                "key": key,
                                "reason": "Performance advantage"
                            })
        
        return {
            "merged": merged_constraints,
            "log": merge_log,
            "stats": {
                "total_keys": len(merged_constraints["current"]),
                "changes": len(merge_log["changes"])
            }
        }
    
    def _evaluate_memory_value(self, memory: Dict[str, Any]) -> float:
        """Evaluate a memory's value for reconciliation."""
        value = 0.5  # Neutral starting value
        
        # Success adds value
        if memory.get("metadata", {}).get("success", False):
            value += 0.2
            
        # More recent memories are more valuable
        if "created_at" in memory:
            age_seconds = time.time() - memory["created_at"]
            # Higher value for newer memories
            recency_factor = max(0, 1.0 - (age_seconds / (7 * 24 * 60 * 60)))  # 1 week scale
            value += 0.15 * recency_factor
            
        # Reward score adds value
        reward = memory.get("metadata", {}).get("reward", 0.0)
        value += 0.15 * reward
        
        # Coherence adds value
        coherence = memory.get("metadata", {}).get("coherence", 0.0)
        value += 0.1 * coherence
        
        # Memory that's been accessed multiple times is more valuable
        access_count = memory.get("access_count", 0)
        access_factor = min(1.0, access_count / 5.0)  # Cap at 5 accesses
        value += 0.1 * access_factor
        
        # Penalize suppressed memories
        if memory.get("suppressed", False):
            value -= 0.5
            
        # Cap between 0 and 1
        return max(0.0, min(1.0, value))
    
    def _law_conflicts_with_existing(self, new_law: str, existing_laws: List[str]) -> bool:
        """Check if a new law conflicts with existing laws."""
        # Very simplified implementation - in production use more sophisticated logic
        for law in existing_laws:
            # Check for contradictory terms
            contradictions = [
                ("must", "must not"),
                ("always", "never"),
                ("required", "prohibited"),
                ("maximize", "minimize")
            ]
            
            for pos, neg in contradictions:
                if pos in new_law and neg in law and self._same_subject(new_law, law):
                    return True
                if neg in new_law and pos in law and self._same_subject(new_law, law):
                    return True
                    
        return False
    
    def _same_subject(self, law1: str, law2: str) -> bool:
        """Determine if two laws address the same subject."""
        # Very simplified implementation - in production use NLP or more advanced logic
        words1 = set(law1.lower().split())
        words2 = set(law2.lower().split())
        
        # Remove common stopwords
        stopwords = {"is", "are", "the", "and", "or", "must", "should", "not", "never", "always"}
        words1 = words1.difference(stopwords)
        words2 = words2.difference(stopwords)
        
        # If they share several content words, they might be about the same subject
        intersection = words1.intersection(words2)
        return len(intersection) >= 3
    
    def _merge_alignment_vectors(self, 
                              winner_alignment: Dict[str, List[str]],
                              loser_alignment: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merge alignment vectors from both doctrines."""
        merged_alignment = {}
        
        # Process optimize_for vectors
        if "optimize_for" in winner_alignment and "optimize_for" in loser_alignment:
            # Start with winner's list
            merged_optimize = winner_alignment["optimize_for"][:]
            
            # Add unique items from loser
            for item in loser_alignment["optimize_for"]:
                if item not in merged_optimize:
                    merged_optimize.append(item)
                    
            merged_alignment["optimize_for"] = merged_optimize
        elif "optimize_for" in winner_alignment:
            merged_alignment["optimize_for"] = winner_alignment["optimize_for"][:]
        elif "optimize_for" in loser_alignment:
            merged_alignment["optimize_for"] = loser_alignment["optimize_for"][:]
            
        # Process minimize vectors
        if "minimize" in winner_alignment and "minimize" in loser_alignment:
            # Start with winner's list
            merged_minimize = winner_alignment["minimize"][:]
            
            # Add unique items from loser
            for item in loser_alignment["minimize"]:
                if item not in merged_minimize:
                    merged_minimize.append(item)
                    
            merged_alignment["minimize"] = merged_minimize
        elif "minimize" in winner_alignment:
            merged_alignment["minimize"] = winner_alignment["minimize"][:]
        elif "minimize" in loser_alignment:
            merged_alignment["minimize"] = loser_alignment["minimize"][:]
            
        return merged_alignment
    
    def _identify_loser_advantages(self, 
                                winner_metrics: Dict[str, float],
                                loser_metrics: Dict[str, float]) -> List[str]:
        """Identify metrics where the loser performed better than the winner."""
        advantages = []
        
        for key, loser_val in loser_metrics.items():
            winner_val = winner_metrics.get(key, 0.0)
            
            # Higher values are better for most metrics
            if loser_val > winner_val * 1.1:  # 10% better
                advantages.append(key)
                
            # Special case: for some metrics lower is better
            if key in ["drift", "error_rate"] and loser_val < winner_val * 0.9:  # 10% better
                advantages.append(key)
                
        return advantages
    
    def _constraint_related_to_advantage(self, 
                                      constraint_key: str, 
                                      advantages: List[str]) -> bool:
        """Check if a constraint is related to metrics where loser had advantage."""
        # Map constraints to related metrics
        constraint_metric_map = {
            "max_plan_depth": ["success_rate", "complexity"],
            "memory_limit": ["coherence", "efficiency"],
            "mutation_scale": ["success_rate", "stability"],
            "max_exploration": ["reward_trend", "drift"],
            "max_entropy": ["stability", "drift"]
        }
        
        if constraint_key in constraint_metric_map:
            related_metrics = constraint_metric_map[constraint_key]
            for metric in related_metrics:
                if metric in advantages:
                    return True
                    
        return False
