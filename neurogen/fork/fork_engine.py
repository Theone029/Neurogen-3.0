import os
import json
import time
import uuid
import shutil
import pickle
from typing import Dict, List, Any, Optional, Tuple, Callable

class ForkEngine:
    """Manages recursive system divergence through controlled forking."""
    
    def __init__(self, config: Dict[str, Any], neurogen_factory: Callable):
        self.config = config
        self.neurogen_factory = neurogen_factory
        
        # Fork tracking
        self.active_forks = {}
        self.fork_history = []
        
        # Fork working directories
        self.forks_dir = config.get("forks_dir", "forks")
        os.makedirs(self.forks_dir, exist_ok=True)
        
        # Fork metrics
        self.reconciliation_metrics = config.get("reconciliation_metrics", [
            "coherence",
            "reward_trend",
            "success_rate",
            "drift",
            "error_rate"
        ])
        
        # Fork thresholds
        self.fork_timeout = config.get("fork_timeout_seconds", 600)  # 10 minutes
        self.evaluation_cycles = config.get("evaluation_cycles", 5)
        
        # Stats
        self.stats = {
            "total_forks": 0,
            "successful_reconciliations": 0,
            "failed_reconciliations": 0,
            "active_forks": 0
        }
    
    def fork(self, reason: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a fork of the current system state.
        
        Args:
            reason: Reason for forking
            context: Current system context
            
        Returns:
            Fork creation result
        """
        # Generate fork IDs
        fork_a_id = f"fork_a_{uuid.uuid4().hex[:8]}"
        fork_b_id = f"fork_b_{uuid.uuid4().hex[:8]}"
        
        # Generate parent ID
        parent_id = context.get("system_id", str(uuid.uuid4().hex[:8]))
        
        # Create fork record
        fork_record = {
            "fork_a_id": fork_a_id,
            "fork_b_id": fork_b_id,
            "parent_id": parent_id,
            "reason": reason,
            "timestamp": time.time(),
            "status": "initializing",
            "fork_a_status": "pending",
            "fork_b_status": "pending",
            "reconciliation_status": "pending",
            "context": {
                "task_type": context.get("task", {}).get("type", "unknown"),
                "doctrine_version": context.get("doctrine", {}).get("version_id", "unknown"),
                "cycle_id": context.get("cycle_id", "unknown")
            }
        }
        
        # Create fork directories
        fork_a_dir = os.path.join(self.forks_dir, fork_a_id)
        fork_b_dir = os.path.join(self.forks_dir, fork_b_id)
        os.makedirs(fork_a_dir, exist_ok=True)
        os.makedirs(fork_b_dir, exist_ok=True)
        
        # Save fork configuration
        with open(os.path.join(fork_a_dir, "fork_config.json"), 'w') as f:
            json.dump({
                "fork_id": fork_a_id,
                "fork_type": "continuation",
                "parent_id": parent_id,
                "reason": reason,
                "timestamp": time.time()
            }, f)
            
        with open(os.path.join(fork_b_dir, "fork_config.json"), 'w') as f:
            json.dump({
                "fork_id": fork_b_id,
                "fork_type": "divergent",
                "parent_id": parent_id,
                "reason": reason,
                "timestamp": time.time()
            }, f)
        
        # Initialize fork states
        try:
            # Serialize current state if available
            if "state" in context:
                with open(os.path.join(fork_a_dir, "initial_state.pickle"), 'wb') as f:
                    pickle.dump(context["state"], f)
                with open(os.path.join(fork_b_dir, "initial_state.pickle"), 'wb') as f:
                    pickle.dump(context["state"], f)
            
            # Create fork A (continuation path)
            fork_a = self._create_fork(fork_a_id, "continuation", context)
            
            # Create fork B (divergent path)
            fork_b = self._create_fork(fork_b_id, "divergent", context)
            
            # Update fork status
            fork_record["status"] = "active"
            fork_record["fork_a_status"] = "active"
            fork_record["fork_b_status"] = "active"
            
            # Register active forks
            self.active_forks[fork_a_id] = fork_a
            self.active_forks[fork_b_id] = fork_b
            
            # Update stats
            self.stats["total_forks"] += 1
            self.stats["active_forks"] += 2
            
            # Add to history
            self.fork_history.append(fork_record)
            
            return {
                "success": True,
                "fork_a_id": fork_a_id,
                "fork_b_id": fork_b_id,
                "parent_id": parent_id,
                "reason": reason
            }
            
        except Exception as e:
            # Clean up on failure
            shutil.rmtree(fork_a_dir, ignore_errors=True)
            shutil.rmtree(fork_b_dir, ignore_errors=True)
            
            fork_record["status"] = "failed"
            fork_record["error"] = str(e)
            self.fork_history.append(fork_record)
            
            return {
                "success": False,
                "error": str(e),
                "reason": reason
            }
    
    def evaluate_forks(self, fork_pair_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate active forks and potentially reconcile.
        
        Args:
            fork_pair_id: Optional specific fork pair to evaluate
            
        Returns:
            Evaluation results
        """
        # Find fork pairs ready for evaluation
        fork_pairs = self._find_fork_pairs(fork_pair_id)
        
        if not fork_pairs:
            return {"success": False, "error": "No eligible fork pairs found"}
            
        results = []
        
        for fork_record in fork_pairs:
            fork_a_id = fork_record["fork_a_id"]
            fork_b_id = fork_record["fork_b_id"]
            
            # Check if forks are still active
            if fork_a_id not in self.active_forks or fork_b_id not in self.active_forks:
                fork_record["status"] = "incomplete"
                continue
                
            fork_a = self.active_forks[fork_a_id]
            fork_b = self.active_forks[fork_b_id]
            
            # Collect metrics from both forks
            fork_a_metrics = self._collect_fork_metrics(fork_a)
            fork_b_metrics = self._collect_fork_metrics(fork_b)
            
            # Compare forks
            comparison = self._compare_forks(fork_a_metrics, fork_b_metrics)
            
            # Determine winner
            winner_id, loser_id = (fork_a_id, fork_b_id) if comparison["winner"] == "a" else (fork_b_id, fork_a_id)
            winner = self.active_forks[winner_id]
            loser = self.active_forks[loser_id]
            
            # Update fork record
            fork_record["evaluation"] = {
                "fork_a_metrics": fork_a_metrics,
                "fork_b_metrics": fork_b_metrics,
                "comparison": comparison,
                "winner": winner_id,
                "loser": loser_id,
                "timestamp": time.time()
            }
            
            # Reconcile forks
            reconciliation = self._reconcile_forks(winner, loser, comparison)
            
            fork_record["reconciliation"] = reconciliation
            fork_record["status"] = "reconciled" if reconciliation["success"] else "failed_reconciliation"
            fork_record["reconciliation_status"] = "complete" if reconciliation["success"] else "failed"
            
            # Update stats
            if reconciliation["success"]:
                self.stats["successful_reconciliations"] += 1
            else:
                self.stats["failed_reconciliations"] += 1
                
            self.stats["active_forks"] -= 2  # Both forks no longer active
            
            # Remove from active forks
            if fork_a_id in self.active_forks:
                del self.active_forks[fork_a_id]
            if fork_b_id in self.active_forks:
                del self.active_forks[fork_b_id]
                
            results.append({
                "fork_a_id": fork_a_id,
                "fork_b_id": fork_b_id,
                "winner": winner_id,
                "success": reconciliation["success"]
            })
        
        return {
            "success": True,
            "evaluations": results,
            "count": len(results)
        }
    
    def get_fork_status(self, fork_id: str) -> Dict[str, Any]:
        """Get status and metrics for a specific fork."""
        # Check active forks
        if fork_id in self.active_forks:
            fork = self.active_forks[fork_id]
            metrics = self._collect_fork_metrics(fork)
            
            return {
                "status": "active",
                "fork_id": fork_id,
                "metrics": metrics,
                "cycles_completed": metrics.get("cycles_completed", 0)
            }
        
        # Check fork history
        for record in self.fork_history:
            if record["fork_a_id"] == fork_id:
                return {
                    "status": record["fork_a_status"],
                    "fork_id": fork_id,
                    "pair_id": record["fork_b_id"],
                    "reason": record["reason"],
                    "metrics": record.get("evaluation", {}).get("fork_a_metrics") if "evaluation" in record else {}
                }
            elif record["fork_b_id"] == fork_id:
                return {
                    "status": record["fork_b_status"],
                    "fork_id": fork_id,
                    "pair_id": record["fork_a_id"],
                    "reason": record["reason"],
                    "metrics": record.get("evaluation", {}).get("fork_b_metrics") if "evaluation" in record else {}
                }
        
        return {"status": "unknown", "fork_id": fork_id}
    
    def get_fork_stats(self) -> Dict[str, Any]:
        """Get statistics about fork operations."""
        return {
            "total_forks": self.stats["total_forks"],
            "active_forks": self.stats["active_forks"],
            "successful_reconciliations": self.stats["successful_reconciliations"],
            "failed_reconciliations": self.stats["failed_reconciliations"],
            "average_fork_lifespan": self._calculate_average_fork_lifespan()
        }
    
    def _create_fork(self, 
                   fork_id: str, 
                   fork_type: str, 
                   context: Dict[str, Any]) -> Any:
        """Create and initialize a fork instance."""
        # Path for fork data
        fork_dir = os.path.join(self.forks_dir, fork_id)
        
        # Create NEUROGEN instance for this fork
        fork_instance = self.neurogen_factory()
        
        # Apply fork-specific configuration
        if fork_type == "continuation":
            # Continuation fork maintains current behavior
            pass
        elif fork_type == "divergent":
            # Divergent fork explores alternative approach
            self._configure_divergent_fork(fork_instance, context)
        
        # Save initial state
        self._save_fork_state(fork_instance, fork_dir, "initial")
        
        return fork_instance
    
    def _configure_divergent_fork(self, 
                               fork_instance: Any, 
                               context: Dict[str, Any]) -> None:
        """Configure a divergent fork with alternative parameters."""
        # Get system context
        drift = context.get("drift", {})
        
        # Apply divergent configuration based on reason
        if "intent" in context:
            # Adjust intent vector if available
            intent = context["intent"]
            if hasattr(intent, "update"):
                # Increase exploration dimension
                exploration_boost = {
                    "exploration": 0.8,
                    "stability": 0.4
                }
                intent.update(exploration_boost, context.get("task", {}))
        
        # Adjust constraint controller if available
        if hasattr(fork_instance, "constraint_controller") and fork_instance.constraint_controller:
            controller = fork_instance.constraint_controller
            
            # Loosen constraints for exploration
            current = controller.current_constraints.copy()
            
            # Increase plan depth
            if "max_plan_depth" in current:
                current["max_plan_depth"] += 2
            
            # Allow more memory
            if "memory_limit" in current:
                current["memory_limit"] += 3
            
            # Increase mutation scale
            if "mutation_scale" in current:
                current["mutation_scale"] = min(0.8, current["mutation_scale"] * 1.5)
            
            # Apply changes
            controller.current_constraints = current
    
    def _save_fork_state(self, fork_instance: Any, fork_dir: str, label: str) -> None:
        """Save current state of a fork."""
        # Create state snapshot
        if hasattr(fork_instance, "get_system_state"):
            state = fork_instance.get_system_state()
        else:
            state = {"timestamp": time.time()}
            
        # Save state as JSON
        state_file = os.path.join(fork_dir, f"{label}_state.json")
        with open(state_file, 'w') as f:
            json.dump(state, f)
            
        # Save pickled instance if configured to do so
        if self.config.get("save_instance_pickle", False):
            try:
                instance_file = os.path.join(fork_dir, f"{label}_instance.pickle")
                with open(instance_file, 'wb') as f:
                    pickle.dump(fork_instance, f)
            except:
                # Pickling can fail for many reasons, just log and continue
                print(f"Warning: Could not pickle fork instance {fork_dir}")
    
    def _find_fork_pairs(self, specific_pair_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find fork pairs ready for evaluation."""
        eligible_pairs = []
        
        for record in self.fork_history:
            # Skip non-active forks
            if record["status"] != "active":
                continue
                
            # Check if specific pair requested
            if specific_pair_id:
                if record["fork_a_id"] == specific_pair_id or record["fork_b_id"] == specific_pair_id:
                    eligible_pairs.append(record)
                    break
                continue
            
            # Check if both forks are active
            fork_a_id = record["fork_a_id"]
            fork_b_id = record["fork_b_id"]
            
            if fork_a_id in self.active_forks and fork_b_id in self.active_forks:
                # Check if forks have run long enough
                fork_a = self.active_forks[fork_a_id]
                fork_b = self.active_forks[fork_b_id]
                
                fork_a_cycles = self._get_cycle_count(fork_a)
                fork_b_cycles = self._get_cycle_count(fork_b)
                
                # Only evaluate if both forks have completed enough cycles
                if fork_a_cycles >= self.evaluation_cycles and fork_b_cycles >= self.evaluation_cycles:
                    eligible_pairs.append(record)
        
        return eligible_pairs
    
    def _get_cycle_count(self, fork: Any) -> int:
        """Get number of execution cycles for a fork."""
        if hasattr(fork, "stats") and "total_loops" in fork.stats:
            return fork.stats["total_loops"]
        if hasattr(fork, "execution_loop") and hasattr(fork.execution_loop, "stats"):
            return fork.execution_loop.stats.get("total_loops", 0)
        return 0
    
    def _collect_fork_metrics(self, fork: Any) -> Dict[str, Any]:
        """Collect performance metrics from a fork."""
        metrics = {
            "cycles_completed": self._get_cycle_count(fork),
            "timestamp": time.time()
        }
        
        # Get system state if available
        if hasattr(fork, "get_system_state"):
            state = fork.get_system_state()
            
            # Extract key metrics
            if "coherence" in state:
                metrics["coherence"] = state["coherence"]
                
            if "drift" in state:
                metrics["drift"] = state["drift"]
                
            if "memory_stats" in state:
                metrics["memory_usage"] = state["memory_stats"].get("total_memories", 0)
                
            if "stats" in state:
                metrics["success_rate"] = (
                    state["stats"].get("successful_loops", 0) / 
                    max(1, state["stats"].get("total_loops", 1))
                )
                metrics["mutation_rate"] = (
                    state["stats"].get("mutations", 0) / 
                    max(1, state["stats"].get("total_loops", 1))
                )
        
        # Evolution auditor metrics
        if hasattr(fork, "evolution_auditor"):
            evolution = fork.evolution_auditor
            
            if hasattr(evolution, "get_reward_trend"):
                reward_trend = evolution.get_reward_trend()
                metrics["reward_trend"] = reward_trend.get("trend", 0.0)
                
            if hasattr(evolution, "get_coherence_trend"):
                coherence_trend = evolution.get_coherence_trend()
                metrics["coherence_trend"] = coherence_trend.get("
