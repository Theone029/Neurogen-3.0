import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

class AdaptiveConstraintController:
    """Dynamically manages system complexity constraints based on performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Base constraints
        self.base_constraints = config.get("base_constraints", {
            "max_plan_depth": 5,
            "memory_limit": 8,
            "mutation_scale": 0.3,
            "max_exploration": 0.6,
            "max_entropy": 0.5
        })
        
        # Current constraints (starts at base values)
        self.current_constraints = self.base_constraints.copy()
        
        # Constraint history
        self.constraints_history = deque(maxlen=config.get("history_size", 100))
        
        # Bounds for each constraint
        self.constraint_bounds = config.get("constraint_bounds", {
            "max_plan_depth": (3, 10),
            "memory_limit": (3, 20),
            "mutation_scale": (0.1, 0.8),
            "max_exploration": (0.3, 0.9),
            "max_entropy": (0.2, 0.8)
        })
        
        # Adaptation parameters
        self.adaptation_rate = config.get("adaptation_rate", 0.1)
        self.volatility_damping = config.get("volatility_damping", 0.7)
        
        # Task-specific constraint overrides
        self.task_constraints = config.get("task_constraints", {})
        
        # Statistics
        self.stats = {
            "adjustments": 0,
            "expansions": 0,
            "contractions": 0,
            "task_overrides": 0
        }
    
    def get_constraints(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get constraints for current execution context.
        
        Args:
            context: Current execution context with task, intent, etc.
            
        Returns:
            Dictionary of constraints
        """
        # Extract context elements
        task = context.get("task", {})
        intent = context.get("intent", {})
        reward_signal = context.get("reward_signal", None)
        drift = context.get("drift", None)
        
        # Start with current constraints
        constraints = self.current_constraints.copy()
        
        # Apply task-specific overrides
        task_type = task.get("type", "default")
        if task_type in self.task_constraints:
            for key, value in self.task_constraints[task_type].items():
                constraints[key] = value
                self.stats["task_overrides"] += 1
        
        # Apply intent-based adjustments
        constraints = self._adjust_for_intent(constraints, intent)
        
        # Apply pressure-based adjustments if available
        if reward_signal and hasattr(reward_signal, "get_constraint_pressure"):
            pressure = reward_signal.get_constraint_pressure(context)
            constraints = self._adjust_for_pressure(constraints, pressure)
        
        # Track constraints
        self.constraints_history.append({
            "constraints": constraints.copy(),
            "task_type": task_type,
            "timestamp": time.time()
        })
        
        return constraints
    
    def update_constraints(self, 
                         performance_metrics: Dict[str, Any],
                         drift_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update current constraints based on system performance.
        
        Args:
            performance_metrics: System performance data
            drift_data: Optional drift measurements
            
        Returns:
            Updated constraints
        """
        self.stats["adjustments"] += 1
        
        # Extract key metrics
        reward = performance_metrics.get("reward", 0.5)
        reward_slope = performance_metrics.get("reward_slope", 0.0)
        entropy = performance_metrics.get("entropy", 0.5)
        success_rate = performance_metrics.get("success_rate", 0.5)
        
        # Extract drift if available
        drift_value = 0.0
        drift_slope = 0.0
        if drift_data:
            drift_value = drift_data.get("drift", 0.0)
            drift_slope = drift_data.get("trend", 0.0)
        
        # Determine if system should expand or contract
        expand = (
            (reward > 0.7 and reward_slope >= 0) or
            (success_rate > 0.8 and reward_slope > 0.01) or
            (drift_value < 0.2 and drift_slope < 0.01)
        )
        
        contract = (
            (reward < 0.4 and reward_slope <= 0) or
            (success_rate < 0.5 and reward_slope < 0) or
            (drift_value > 0.5 or drift_slope > 0.05) or
            (entropy > 0.7)
        )
        
        # Update each constraint
        new_constraints = self.current_constraints.copy()
        
        if expand and not contract:
            # Expand constraints
            self.stats["expansions"] += 1
            for key, value in new_constraints.items():
                if key in self.constraint_bounds:
                    bound_min, bound_max = self.constraint_bounds[key]
                    
                    # Different adjustment logic based on constraint type
                    if key in ["max_plan_depth", "memory_limit"]:
                        # Integer constraints - step up
                        new_value = min(bound_max, value + 1)
                    else:
                        # Float constraints - scale up
                        new_value = min(bound_max, value * (1 + self.adaptation_rate))
                        
                    new_constraints[key] = new_value
        
        elif contract and not expand:
            # Contract constraints
            self.stats["contractions"] += 1
            for key, value in new_constraints.items():
                if key in self.constraint_bounds:
                    bound_min, bound_max = self.constraint_bounds[key]
                    
                    # Different adjustment logic based on constraint type
                    if key in ["max_plan_depth", "memory_limit"]:
                        # Integer constraints - step down
                        new_value = max(bound_min, value - 1)
                    else:
                        # Float constraints - scale down
                        new_value = max(bound_min, value * (1 - self.adaptation_rate))
                        
                    new_constraints[key] = new_value
        
        # Special handling for mutation scale based on entropy
        if entropy > 0.6:
            # Reduce mutation scale when entropy is high
            new_constraints["mutation_scale"] = max(
                self.constraint_bounds["mutation_scale"][0],
                new_constraints["mutation_scale"] * 0.9
            )
        elif entropy < 0.3 and success_rate < 0.6:
            # Increase mutation scale when entropy is low but success is also low
            new_constraints["mutation_scale"] = min(
                self.constraint_bounds["mutation_scale"][1],
                new_constraints["mutation_scale"] * 1.1
            )
        
        # Apply volatility damping to smooth changes
        for key in new_constraints:
            if key in self.current_constraints:
                damped_value = (
                    self.volatility_damping * self.current_constraints[key] +
                    (1 - self.volatility_damping) * new_constraints[key]
                )
                new_constraints[key] = damped_value
        
        # Update current constraints
        self.current_constraints = new_constraints
        
        return new_constraints
    
    def get_mutation_scale(self, context: Dict[str, Any]) -> float:
        """Get the current mutation scale for mutation operations."""
        constraints = self.get_constraints(context)
        return constraints.get("mutation_scale", 0.3)
    
    def get_memory_limit(self, context: Dict[str, Any]) -> int:
        """Get the current memory limit for memory selection."""
        constraints = self.get_constraints(context)
        return int(constraints.get("memory_limit", 8))
    
    def get_plan_depth(self, context: Dict[str, Any]) -> int:
        """Get the maximum plan depth for the planner."""
        constraints = self.get_constraints(context)
        return int(constraints.get("max_plan_depth", 5))
    
    def _adjust_for_intent(self, 
                         constraints: Dict[str, Any], 
                         intent: Any) -> Dict[str, Any]:
        """Adjust constraints based on intent vector."""
        # Extract intent dimensions if available
        intent_dims = {}
        if hasattr(intent, "get_vector_as_dict"):
            intent_dims = intent.get_vector_as_dict()
        elif isinstance(intent, dict):
            intent_dims = intent
        
        if not intent_dims:
            return constraints
            
        # Copy constraints to avoid modifying the input
        adjusted = constraints.copy()
        
        # Adjust based on exploration dimension
        if "exploration" in intent_dims:
            exploration = intent_dims["exploration"]
            
            # High exploration intent allows more complexity
            if exploration > 0.7:
                if "max_plan_depth" in adjusted:
                    adjusted["max_plan_depth"] = min(
                        self.constraint_bounds["max_plan_depth"][1],
                        adjusted["max_plan_depth"] + 1
                    )
                    
                if "memory_limit" in adjusted:
                    adjusted["memory_limit"] = min(
                        self.constraint_bounds["memory_limit"][1],
                        adjusted["memory_limit"] + 2
                    )
                    
                if "max_exploration" in adjusted:
                    adjusted["max_exploration"] = min(
                        self.constraint_bounds["max_exploration"][1],
                        adjusted["max_exploration"] * 1.2
                    )
            
            # Low exploration prefers more constraint
            elif exploration < 0.3:
                if "max_plan_depth" in adjusted:
                    adjusted["max_plan_depth"] = max(
                        self.constraint_bounds["max_plan_depth"][0],
                        adjusted["max_plan_depth"] - 1
                    )
                    
                if "memory_limit" in adjusted:
                    adjusted["memory_limit"] = max(
                        self.constraint_bounds["memory_limit"][0],
                        adjusted["memory_limit"] - 1
                    )
                    
                if "max_exploration" in adjusted:
                    adjusted["max_exploration"] = max(
                        self.constraint_bounds["max_exploration"][0],
                        adjusted["max_exploration"] * 0.8
                    )
        
        # Adjust based on stability dimension
        if "stability" in intent_dims:
            stability = intent_dims["stability"]
            
            # High stability intent reduces entropy and mutation
            if stability > 0.7:
                if "max_entropy" in adjusted:
                    adjusted["max_entropy"] = max(
                        self.constraint_bounds["max_entropy"][0],
                        adjusted["max_entropy"] * 0.8
                    )
                    
                if "mutation_scale" in adjusted:
                    adjusted["mutation_scale"] = max(
                        self.constraint_bounds["mutation_scale"][0],
                        adjusted["mutation_scale"] * 0.8
                    )
            
            # Low stability allows more mutation
            elif stability < 0.3:
                if "mutation_scale" in adjusted:
                    adjusted["mutation_scale"] = min(
                        self.constraint_bounds["mutation_scale"][1],
                        adjusted["mutation_scale"] * 1.2
                    )
        
        return adjusted
    
    def _adjust_for_pressure(self, 
                           constraints: Dict[str, Any], 
                           pressure: float) -> Dict[str, Any]:
        """Adjust constraints based on pressure signal."""
        # Copy constraints to avoid modifying the input
        adjusted = constraints.copy()
        
        # High pressure means tighter constraints
        if pressure > 0.7:
            # Reduce complexity
            if "max_plan_depth" in adjusted:
                adjusted["max_plan_depth"] = max(
                    self.constraint_bounds["max_plan_depth"][0],
                    int(adjusted["max_plan_depth"] * 0.8)
                )
                
            if "memory_limit" in adjusted:
                adjusted["memory_limit"] = max(
                    self.constraint_bounds["memory_limit"][0],
                    int(adjusted["memory_limit"] * 0.8)
                )
                
            if "max_entropy" in adjusted:
                adjusted["max_entropy"] = max(
                    self.constraint_bounds["max_entropy"][0],
                    adjusted["max_entropy"] * 0.8
                )
                
            if "mutation_scale" in adjusted:
                adjusted["mutation_scale"] = max(
                    self.constraint_bounds["mutation_scale"][0],
                    adjusted["mutation_scale"] * 0.7
                )
        
        # Low pressure allows more complexity
        elif pressure < 0.3:
            # Increase complexity
            if "max_plan_depth" in adjusted:
                adjusted["max_plan_depth"] = min(
                    self.constraint_bounds["max_plan_depth"][1],
                    int(adjusted["max_plan_depth"] * 1.2) + 1
                )
                
            if "memory_limit" in adjusted:
                adjusted["memory_limit"] = min(
                    self.constraint_bounds["memory_limit"][1],
                    int(adjusted["memory_limit"] * 1.2) + 2
                )
                
            if "max_entropy" in adjusted:
                adjusted["max_entropy"] = min(
                    self.constraint_bounds["max_entropy"][1],
                    adjusted["max_entropy"] * 1.2
                )
                
            if "mutation_scale" in adjusted:
                adjusted["mutation_scale"] = min(
                    self.constraint_bounds["mutation_scale"][1],
                    adjusted["mutation_scale"] * 1.3
                )
        
        return adjusted
    
    def get_constraint_trend(self, constraint_key: str, window_size: int = 10) -> Dict[str, Any]:
        """Get trend data for a specific constraint."""
        if not self.constraints_history or constraint_key not in self.current_constraints:
            return {"trend": 0.0, "volatility": 0.0, "current": self.current_constraints.get(constraint_key, 0.0)}
            
        # Get recent values
        history = list(self.constraints_history)[-min(window_size, len(self.constraints_history)):]
        values = [h["constraints"].get(constraint_key, 0.0) for h in history if constraint_key in h["constraints"]]
        
        if not values:
            return {"trend": 0.0, "volatility": 0.0, "current": self.current_constraints.get(constraint_key, 0.0)}
            
        # Calculate trend (slope)
        x = np.arange(len(values))
        if len(x) > 1:
            slope = np.polyfit(x, values, 1)[0]
        else:
            slope = 0.0
            
        # Calculate volatility
        volatility = np.std(values) if len(values) > 1 else 0.0
        
        return {
            "trend": float(slope),
            "volatility": float(volatility),
            "current": self.current_constraints.get(constraint_key, 0.0),
            "history": values
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            "adjustments": self.stats["adjustments"],
            "expansions": self.stats["expansions"],
            "contractions": self.stats["contractions"],
            "task_overrides": self.stats["task_overrides"],
            "current_constraints": self.current_constraints
        }
