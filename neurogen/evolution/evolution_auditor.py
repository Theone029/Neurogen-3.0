import numpy as np
import time
import json
from typing import Dict, List, Any, Optional, Tuple
import datetime
from collections import deque

class EvolutionAuditor:
    """Tracks and audits NEUROGEN's evolution over time across multiple dimensions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # History tracking
        self.cycle_history = deque(maxlen=config.get("max_cycle_history", 1000))
        self.doctrine_history = deque(maxlen=config.get("max_doctrine_history", 100))
        self.intent_history = deque(maxlen=config.get("max_intent_history", 500))
        self.coherence_history = deque(maxlen=config.get("max_coherence_history", 500))
        
        # Baseline data
        self.baseline_state = None
        self.baseline_timestamp = None
        
        # Analysis windows for different metrics
        self.short_window = config.get("short_window", 10)
        self.medium_window = config.get("medium_window", 50)
        self.long_window = config.get("long_window", 200)
        
        # Intervention thresholds
        self.intervention_thresholds = config.get("intervention_thresholds", {
            "coherence_drop": 0.3,
            "drift_critical": 0.6,
            "reward_collapse": -0.05,
            "doctrine_mutations_max": 3,  # per 100 cycles
            "evolution_divergence": 0.7
        })
        
        # Component weights for coherence calculation
        self.coherence_weights = config.get("coherence_weights", {
            "reward": 0.3,
            "drift": 0.25,
            "doctrine_alignment": 0.25,
            "memory_consistency": 0.2
        })
        
        # Stats
        self.stats = {
            "total_cycles": 0,
            "doctrine_changes": 0,
            "interventions_signaled": 0,
            "avg_coherence": 0.0
        }
    
    def record_cycle(self, cycle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record a complete execution cycle with all metrics.
        
        Args:
            cycle_data: Complete data from execution cycle
            
        Returns:
            Cycle analysis with coherence score
        """
        self.stats["total_cycles"] += 1
        
        # Extract core components
        task = cycle_data.get("task", {})
        intent = cycle_data.get("intent", None)
        success = cycle_data.get("success", False)
        validation = cycle_data.get("validation", {})
        doctrine = cycle_data.get("doctrine", {})
        drift = cycle_data.get("drift", {})
        reward = cycle_data.get("reward", {})
        
        # Create cycle record
        cycle_record = {
            "timestamp": time.time(),
            "cycle_id": cycle_data.get("cycle_id", str(self.stats["total_cycles"])),
            "task_type": task.get("type", "unknown"),
            "success": success,
            "intent_vector": self._extract_intent_vector(intent),
            "drift": drift.get("drift", 0.0) if isinstance(drift, dict) else 0.0,
            "reward": reward.get("reward", 0.0) if isinstance(reward, dict) else 0.0,
            "doctrine_version": doctrine.get("version_id", "unknown") if isinstance(doctrine, dict) else "unknown",
            "mutation_count": cycle_data.get("mutation_count", 0)
        }
        
        # Set baseline state if not already set
        if self.baseline_state is None:
            self._set_baseline(cycle_data)
        
        # Calculate coherence score
        coherence = self._calculate_coherence(cycle_data)
        cycle_record["coherence"] = coherence
        
        # Update historical tracking
        self.cycle_history.append(cycle_record)
        
        # Record intent vector if available
        if intent is not None:
            self._record_intent(intent, cycle_record["cycle_id"])
        
        # Check for intervention signals
        intervention_signals = self._generate_intervention_signals(cycle_record)
        
        # Update stats
        self.stats["avg_coherence"] = (
            (self.stats["avg_coherence"] * (self.stats["total_cycles"] - 1) + coherence) / 
            self.stats["total_cycles"]
        )
        
        # Return analysis
        return {
            "coherence": coherence,
            "cycle_id": cycle_record["cycle_id"],
            "intervention_signals": intervention_signals
        }
    
    def record_doctrine_change(self, 
                              old_version: str, 
                              new_version: str, 
                              justification: str) -> Dict[str, Any]:
        """
        Record a doctrine version change with justification.
        
        Args:
            old_version: Previous doctrine version ID
            new_version: New doctrine version ID
            justification: Reason for the change
            
        Returns:
            Analysis of the doctrine change
        """
        self.stats["doctrine_changes"] += 1
        
        # Create doctrine change record
        change_record = {
            "timestamp": time.time(),
            "old_version": old_version,
            "new_version": new_version,
            "justification": justification,
            "cycle_id": str(self.stats["total_cycles"]) if self.cycle_history else "unknown"
        }
        
        # Add to history
        self.doctrine_history.append(change_record)
        
        # Calculate doctrine mutation rate
        recent_window = min(100, self.stats["total_cycles"])
        if recent_window > 0:
            recent_changes = sum(1 for record in self.doctrine_history
                                if record["cycle_id"] != "unknown" and
                                int(record["cycle_id"]) > self.stats["total_cycles"] - recent_window)
            mutation_rate = recent_changes / recent_window
        else:
            mutation_rate = 0
        
        # Check if we're mutating doctrine too frequently
        max_rate = self.intervention_thresholds["doctrine_mutations_max"] / 100
        excessive_mutation = mutation_rate > max_rate
        
        # Return analysis
        return {
            "mutation_rate": mutation_rate,
            "excessive_mutation": excessive_mutation,
            "total_changes": self.stats["doctrine_changes"]
        }
    
    def get_latest_coherence(self) -> Optional[float]:
        """Get the most recent coherence score."""
        if not self.coherence_history:
            return None
        return self.coherence_history[-1]
    
    def get_coherence_trend(self, window_size: int = None) -> Dict[str, Any]:
        """
        Get coherence trend analysis.
        
        Args:
            window_size: Optional override for analysis window size
            
        Returns:
            Coherence trend analysis
        """
        if not self.coherence_history:
            return {"trend": 0.0, "stability": 1.0, "current": None}
            
        # Use specified window or default to medium window
        window = window_size if window_size is not None else self.medium_window
        
        # Get coherence values
        values = list(self.coherence_history)[-min(window, len(self.coherence_history)):]
        
        if len(values) < 2:
            return {"trend": 0.0, "stability": 1.0, "current": values[-1] if values else None}
            
        # Calculate trend using linear regression
        x = np.arange(len(values))
        y = np.array(values)
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate stability (inverse of volatility)
        stability = 1.0 - min(1.0, np.std(values) * 2)
        
        return {
            "trend": float(slope),
            "stability": float(stability),
            "current": values[-1],
            "min": min(values),
            "max": max(values),
            "samples": len(values)
        }
    
    def get_reward_trend(self, window_size: int = None) -> Dict[str, Any]:
        """Get reward trend analysis."""
        if not self.cycle_history:
            return {"trend": 0.0, "stability": 1.0, "current": None}
            
        # Use specified window or default to medium window
        window = window_size if window_size is not None else self.medium_window
        
        # Get reward values
        values = [record["reward"] for record in list(self.cycle_history)[-min(window, len(self.cycle_history)):]]
        
        if len(values) < 2:
            return {"trend": 0.0, "stability": 1.0, "current": values[-1] if values else None}
            
        # Calculate trend using linear regression
        x = np.arange(len(values))
        y = np.array(values)
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate stability (inverse of volatility)
        stability = 1.0 - min(1.0, np.std(values) * 2)
        
        return {
            "trend": float(slope),
            "stability": float(stability),
            "current": values[-1] if values else None,
            "min": min(values),
            "max": max(values),
            "samples": len(values)
        }
    
    def get_drift_trend(self, window_size: int = None) -> Dict[str, Any]:
        """Get drift trend analysis."""
        if not self.cycle_history:
            return {"trend": 0.0, "stability": 1.0, "current": None}
            
        # Use specified window or default to medium window
        window = window_size if window_size is not None else self.medium_window
        
        # Get drift values
        values = [record["drift"] for record in list(self.cycle_history)[-min(window, len(self.cycle_history)):]]
        
        if len(values) < 2:
            return {"trend": 0.0, "stability": 1.0, "current": values[-1] if values else None}
            
        # Calculate trend using linear regression
        x = np.arange(len(values))
        y = np.array(values)
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate stability (inverse of volatility)
        stability = 1.0 - min(1.0, np.std(values) * 2)
        
        return {
            "trend": float(slope),
            "stability": float(stability),
            "current": values[-1] if values else None,
            "min": min(values),
            "max": max(values),
            "samples": len(values)
        }
    
    def get_intent_evolution(self, dimension: Optional[str] = None) -> Dict[str, Any]:
        """
        Get intent vector evolution analysis.
        
        Args:
            dimension: Optional specific intent dimension to analyze
            
        Returns:
            Intent evolution analysis
        """
        if not self.intent_history:
            return {"trend": {}, "divergence": 0.0}
            
        # If we have no intent records, return empty analysis
        intent_records = list(self.intent_history)
        if not intent_records:
            return {"trend": {}, "divergence": 0.0}
            
        # Get most recent intent record
        recent_intent = intent_records[-1]["vector"]
        
        # Get oldest intent record for comparison
        baseline_intent = intent_records[0]["vector"] if len(intent_records) > 1 else recent_intent
        
        # Calculate divergence from baseline
        if isinstance(recent_intent, dict) and isinstance(baseline_intent, dict):
            # For dictionary representation
            all_dimensions = set(recent_intent.keys()).union(baseline_intent.keys())
            squared_diffs = []
            
            for dim in all_dimensions:
                recent_val = recent_intent.get(dim, 0.0)
                baseline_val = baseline_intent.get(dim, 0.0)
                squared_diffs.append((recent_val - baseline_val) ** 2)
                
            divergence = (sum(squared_diffs) / len(squared_diffs)) ** 0.5 if squared_diffs else 0.0
            
        elif hasattr(recent_intent, "__len__") and hasattr(baseline_intent, "__len__"):
            # For list/array representation
            if len(recent_intent) == len(baseline_intent):
                # Euclidean distance
                squared_diffs = [(a - b) ** 2 for a, b in zip(recent_intent, baseline_intent)]
                divergence = (sum(squared_diffs) / len(squared_diffs)) ** 0.5 if squared_diffs else 0.0
            else:
                divergence = 0.5  # Default for mismatched dimensions
        else:
            divergence = 0.0
        
        # Analyze trends for each dimension
        dimension_trends = {}
        
        if dimension is not None:
            # Analyze specific dimension
            if isinstance(recent_intent, dict) and dimension in recent_intent:
                # Extract dimension values over time
                dim_values = [
                    record["vector"].get(dimension, 0.0) 
                    for record in intent_records 
                    if isinstance(record["vector"], dict) and dimension in record["vector"]
                ]
                
                if len(dim_values) >= 2:
                    # Calculate trend
                    x = np.arange(len(dim_values))
                    slope, intercept = np.polyfit(x, dim_values, 1)
                    dimension_trends[dimension] = float(slope)
        else:
            # Analyze all dimensions
            if isinstance(recent_intent, dict):
                # Get all dimensions that appear in at least half the records
                dimension_counts = {}
                for record in intent_records:
                    if isinstance(record["vector"], dict):
                        for dim in record["vector"]:
                            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
                
                common_dimensions = [
                    dim for dim, count in dimension_counts.items()
                    if count >= len(intent_records) // 2
                ]
                
                # Calculate trend for each common dimension
                for dim in common_dimensions:
                    dim_values = [
                        record["vector"].get(dim, 0.0) 
                        for record in intent_records 
                        if isinstance(record["vector"], dict) and dim in record["vector"]
                    ]
                    
                    if len(dim_values) >= 2:
                        # Calculate trend
                        x = np.arange(len(dim_values))
                        slope, intercept = np.polyfit(x, dim_values, 1)
                        dimension_trends[dim] = float(slope)
        
        return {
            "trend": dimension_trends,
            "divergence": float(divergence),
            "dimensions": len(recent_intent) if hasattr(recent_intent, "__len__") else 0
        }
    
    def get_system_state_report(self) -> Dict[str, Any]:
        """Generate a comprehensive system state report."""
        # Get trend analyses
        coherence_trend = self.get_coherence_trend()
        reward_trend = self.get_reward_trend()
        drift_trend = self.get_drift_trend()
        intent_evolution = self.get_intent_evolution()
        
        # Calculate days since baseline
        days_since_baseline = 0
        if self.baseline_timestamp:
            seconds_elapsed = time.time() - self.baseline_timestamp
            days_since_baseline = seconds_elapsed / (24 * 60 * 60)
        
        # Generate report
        report = {
            "timestamp": time.time(),
            "cycles": self.stats["total_cycles"],
            "days_since_baseline": days_since_baseline,
            "coherence": {
                "current": coherence_trend["current"],
                "trend": coherence_trend["trend"],
                "stability": coherence_trend["stability"]
            },
            "reward": {
                "current": reward_trend["current"],
                "trend": reward_trend["trend"],
                "stability": reward_trend["stability"]
            },
            "drift": {
                "current": drift_trend["current"],
                "trend": drift_trend["trend"],
                "stability": drift_trend["stability"]
            },
            "intent": {
                "divergence": intent_evolution["divergence"],
                "trends": intent_evolution["trend"]
            },
            "doctrine": {
                "version": self.doctrine_history[-1]["new_version"] if self.doctrine_history else "unknown",
                "changes": self.stats["doctrine_changes"],
                "recent_change_rate": self._calculate_doctrine_change_rate()
            },
            "intervention_signals": self._generate_system_intervention_signals()
        }
        
        return report
    
    def _set_baseline(self, cycle_data: Dict[str, Any]) -> None:
        """Set baseline state for future comparison."""
        self.baseline_state = {
            "intent": self._extract_intent_vector(cycle_data.get("intent")),
            "doctrine_version": cycle_data.get("doctrine", {}).get("version_id", "unknown"),
            "cycle_id": cycle_data.get("cycle_id", "0"),
            "task_type": cycle_data.get("task", {}).get("type", "unknown")
        }
        self.baseline_timestamp = time.time()
    
    def _extract_intent_vector(self, intent: Any) -> Dict[str, float]:
        """Extract intent vector from various possible formats."""
        if intent is None:
            return {}
            
        if hasattr(intent, "get_vector_as_dict"):
            return intent.get_vector_as_dict()
            
        if isinstance(intent, dict):
            # Check if it's already a vector representation
            if all(isinstance(v, (int, float)) for v in intent.values()):
                return intent
                
        if hasattr(intent, "tolist"):
            # Convert numpy array to list
            return {f"dim_{i}": float(v) for i, v in enumerate(intent.tolist())}
            
        if isinstance(intent, (list, tuple)):
            return {f"dim_{i}": float(v) for i, v in enumerate(intent)}
            
        # Unknown format
        return {}
    
    def _record_intent(self, intent: Any, cycle_id: str) -> None:
        """Record intent vector for tracking intent evolution."""
        vector = self._extract_intent_vector(intent)
        
        if vector:
            intent_record = {
                "timestamp": time.time(),
                "cycle_id": cycle_id,
                "vector": vector
            }
            self.intent_history.append(intent_record)
    
    def _calculate_coherence(self, cycle_data: Dict[str, Any]) -> float:
        """
        Calculate system coherence as composite metric.
        
        Args:
            cycle_data: Complete data from execution cycle
            
        Returns:
            Coherence score (0-1)
        """
        # Extract components for coherence calculation
        reward = cycle_data.get("reward", {}).get("reward", 0.0) if isinstance(cycle_data.get("reward"), dict) else 0.0
        drift = cycle_data.get("drift", {}).get("drift", 0.0) if isinstance(cycle_data.get("drift"), dict) else 0.0
        doctrine_valid = cycle_data.get("doctrine", {}).get("valid", True) if isinstance(cycle_data.get("doctrine"), dict) else True
        
        # Calculate memory consistency if available
        memory_consistency = 0.5  # Default neutral value
        if "memory_links_used" in cycle_data:
            memory_count = len(cycle_data["memory_links_used"])
            if memory_count > 0:
                # More memories is generally better up to a point
                memory_consistency = min(1.0, memory_count / 5)  # Cap at 5 memories
        
        # Calculate coherence components
        reward_component = reward  # Higher reward = higher coherence
        drift_component = 1.0 - drift  # Lower drift = higher coherence
        doctrine_component = 1.0 if doctrine_valid else 0.3
        memory_component = memory_consistency
        
        # Combine components using weights
        coherence = (
            self.coherence_weights["reward"] * reward_component +
            self.coherence_weights["drift"] * drift_component +
            self.coherence_weights["doctrine_alignment"] * doctrine_component +
            self.coherence_weights["memory_consistency"] * memory_component
        )
        
        # Cap between 0 and 1
        coherence = max(0.0, min(1.0, coherence))
        
        # Record coherence
        self.coherence_history.append(coherence)
        
        return coherence
    
    def _generate_intervention_signals(self, cycle_record: Dict[str, Any]) -> Dict[str, bool]:
        """Generate intervention signals based on current cycle."""
        signals = {
            "rollback_recommended": False,
            "fork_recommended": False,
            "freeze_recommended": False,
            "audit_recommended": False
        }
        
        # Check coherence drop
        if len(self.coherence_history) >= 2:
            prev_coherence = self.coherence_history[-2]
            current_coherence = cycle_record["coherence"]
            
            coherence_drop = prev_coherence - current_coherence
            if coherence_drop > self.intervention_thresholds["coherence_drop"]:
                signals["freeze_recommended"] = True
                signals["audit_recommended"] = True
                self.stats["interventions_signaled"] += 1
        
        # Check drift critical level
        if cycle_record["drift"] > self.intervention_thresholds["drift_critical"]:
            signals["fork_recommended"] = True
            self.stats["interventions_signaled"] += 1
        
        # Check reward collapse
        if len(self.cycle_history) >= self.short_window:
            recent_cycles = list(self.cycle_history)[-self.short_window:]
            rewards = [c["reward"] for c in recent_cycles]
            
            if len(rewards) >= 2:
                x = np.arange(len(rewards))
                slope, _ = np.polyfit(x, rewards, 1)
                
                if slope < self.intervention_thresholds["reward_collapse"]:
                    signals["rollback_recommended"] = True
                    self.stats["interventions_signaled"] += 1
        
        return signals
    
    def _generate_system_intervention_signals(self) -> Dict[str, Any]:
        """Generate system-wide intervention signals based on all metrics."""
        signals = {
            "rollback_recommended": False,
            "fork_recommended": False,
            "freeze_recommended": False,
            "audit_recommended": False,
            "reasons": []
        }
        
        # Check coherence trend
        coherence_trend = self.get_coherence_trend()
        if coherence_trend["trend"] < -0.01 and coherence_trend["stability"] < 0.7:
            signals["audit_recommended"] = True
            signals["reasons"].append("Declining coherence with high volatility")
        
        # Check reward collapse
        reward_trend = self.get_reward_trend()
        if reward_trend["trend"] < self.intervention_thresholds["reward_collapse"]:
            signals["rollback_recommended"] = True
            signals["reasons"].append("Sustained reward collapse")
        
        # Check drift spike
        drift_trend = self.get_drift_trend()
        if drift_trend["current"] and drift_trend["current"] > self.intervention_thresholds["drift_critical"]:
            signals["fork_recommended"] = True
            signals["reasons"].append("Critical drift level exceeded")
        
        # Check intent divergence
        intent_evolution = self.get_intent_evolution()
        if intent_evolution["divergence"] > self.intervention_thresholds["evolution_divergence"]:
            signals["fork_recommended"] = True
            signals["reasons"].append("Intent vector critical divergence")
        
        # Check doctrine mutation rate
        doctrine_change_rate = self._calculate_doctrine_change_rate()
        if doctrine_change_rate > (self.intervention_thresholds["doctrine_mutations_max"] / 100):
            signals["freeze_recommended"] = True
            signals["reasons"].append("Excessive doctrine mutation rate")
        
        # Set intervention count
        if any(signals.values()):
            self.stats["interventions_signaled"] += 1
        
        return signals
    
    def _calculate_doctrine_change_rate(self) -> float:
        """Calculate recent doctrine change rate (changes per cycle)."""
        if not self.doctrine_history or self.stats["total_cycles"] == 0:
            return 0.0
            
        # Count changes in last 100 cycles
        recent_window = min(100, self.stats["total_cycles"])
        if recent_window == 0:
            return 0.0
            
        recent_changes = sum(1 for record in self.doctrine_history
                            if record["cycle_id"] != "unknown" and
                            int(record["cycle_id"]) > self.stats["total_cycles"] - recent_window)
        
        return recent_changes / recent_window
