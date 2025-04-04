import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

class RewardSignalRouter:
    """Routes feedback signals to shape system evolution and mutation pressure."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Reward history
        self.reward_history = deque(maxlen=config.get("history_size", 100))
        self.reward_by_task_type = {}
        self.reward_by_component = {}
        
        # Pressure routing configuration
        self.pressure_weights = config.get("pressure_weights", {
            "mutation": 0.4,
            "memory": 0.3, 
            "constraint": 0.2,
            "doctrine": 0.1
        })
        
        # Reward calculation weights
        self.reward_weights = config.get("reward_weights", {
            "success": 0.5,
            "validation_quality": 0.2,
            "entropy_reduction": 0.15,
            "execution_efficiency": 0.1,
            "memory_utilization": 0.05
        })
        
        # Adaptive parameters
        self.adaptive_pressure = config.get("adaptive_pressure", True)
        self.pressure_decay = config.get("pressure_decay", 0.95)
        self.initial_pressure = config.get("initial_pressure", 0.5)
        
        # Current pressure levels for each route
        self.current_pressure = {
            "mutation": self.initial_pressure,
            "memory": self.initial_pressure,
            "constraint": self.initial_pressure,
            "doctrine": self.initial_pressure
        }
        
        # Statistics
        self.stats = {
            "total_rewards": 0,
            "avg_reward": 0.0,
            "reward_slope": 0.0,
            "pressure_adjustments": 0
        }
    
    def calculate_reward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate reward based on execution results.
        
        Args:
            context: Execution context with success, validation, drift, etc.
            
        Returns:
            Reward calculation with component breakdown
        """
        # Extract context elements
        success = context.get("success", False)
        validation = context.get("validation", {})
        doctrine = context.get("doctrine", {})
        drift = context.get("drift", {})
        execution_time = context.get("execution_time")
        mutation_count = context.get("mutation_count", 0)
        task = context.get("task", {})
        
        # Initialize reward components
        components = {}
        
        # Success component
        components["success"] = 1.0 if success else 0.0
        
        # Validation quality component
        if validation:
            validation_score = 0.0
            if validation.get("valid", False):
                # Score perfect validation higher
                check_count = len(validation.get("checks", []))
                passed_checks = sum(1 for check in validation.get("checks", []) 
                                  if check.get("passed", False))
                
                if check_count > 0:
                    validation_score = passed_checks / check_count
            components["validation_quality"] = validation_score
        else:
            components["validation_quality"] = 0.0
            
        # Doctrine alignment component
        if doctrine:
            doctrine_score = 1.0 if doctrine.get("valid", False) else 0.0
            components["doctrine_alignment"] = doctrine_score
        else:
            components["doctrine_alignment"] = 0.0
            
        # Entropy reduction component
        if drift and "drift" in drift:
            # Lower drift is better
            entropy_reduction = max(0, 1.0 - drift["drift"])
            components["entropy_reduction"] = entropy_reduction
        else:
            components["entropy_reduction"] = 0.5  # Neutral if no drift info
            
        # Execution efficiency component
        if execution_time and success:
            # Calculate efficiency based on task type average times
            task_type = task.get("type", "default")
            if task_type in self.reward_by_task_type:
                avg_time = self.reward_by_task_type[task_type].get("avg_time", execution_time)
                # Better than average is good
                efficiency = min(1.0, avg_time / execution_time) if execution_time > 0 else 0.5
            else:
                efficiency = 0.5  # Neutral for first execution
                
            components["execution_efficiency"] = efficiency
        else:
            components["execution_efficiency"] = 0.0
            
        # Memory utilization component
        if "memory_links_used" in context:
            memory_count = len(context["memory_links_used"])
            memory_target = self.config.get("optimal_memory_usage", 5)
            
            # Score optimal memory usage highest
            if memory_count == 0:
                memory_score = 0.2  # Penalize not using memory
            elif memory_count <= memory_target:
                memory_score = 0.5 + (memory_count / memory_target) * 0.5
            else:
                # Penalize excessive memory usage, but not as severely
                over_usage = (memory_count - memory_target) / memory_target
                memory_score = 1.0 - min(0.5, over_usage * 0.1)
                
            components["memory_utilization"] = memory_score
        else:
            components["memory_utilization"] = 0.0
            
        # Mutation efficiency component
        if mutation_count > 0:
            if success:
                # Successful mutations are good, but fewer is better
                mutation_eff = 1.0 / mutation_count
                components["mutation_efficiency"] = mutation_eff
            else:
                # Failed mutations are bad
                components["mutation_efficiency"] = 0.0
        else:
            # No mutations needed is good
            components["mutation_efficiency"] = 1.0 if success else 0.0
        
        # Calculate weighted reward
        total_reward = 0.0
        weight_sum = 0.0
        
        for component, value in components.items():
            weight = self.reward_weights.get(component, 0.0)
            total_reward += value * weight
            weight_sum += weight
            
        # Normalize reward
        if weight_sum > 0:
            total_reward /= weight_sum
            
        # Cap reward at 1.0
        total_reward = min(1.0, max(0.0, total_reward))
        
        # Record reward
        self._record_reward(total_reward, components, task)
        
        # Route pressure signals based on reward
        if self.adaptive_pressure:
            self._adjust_pressure(total_reward, components)
            
        # Return complete reward information
        return {
            "reward": total_reward,
            "components": components,
            "timestamp": time.time()
        }
    
    def get_mutation_pressure(self, context: Dict[str, Any]) -> float:
        """Get current mutation pressure value."""
        task_type = context.get("task", {}).get("type", "default")
        
        # Check for task-specific overrides
        if task_type in self.reward_by_task_type:
            task_data = self.reward_by_task_type[task_type]
            if "mutation_pressure" in task_data:
                return task_data["mutation_pressure"]
                
        # Use global mutation pressure
        base_pressure = self.current_pressure["mutation"]
        
        # Adjust based on task success rate
        if task_type in self.reward_by_task_type:
            success_rate = self.reward_by_task_type[task_type].get("success_rate", 0.5)
            
            # Higher pressure when success rate is low
            if success_rate < 0.3:
                pressure_boost = (0.3 - success_rate) * 2.0  # Up to 0.6 boost
                return min(1.0, base_pressure + pressure_boost)
                
            # Lower pressure when success rate is high
            if success_rate > 0.8:
                pressure_reduction = (success_rate - 0.8) * 0.5  # Up to 0.1 reduction
                return max(0.1, base_pressure - pressure_reduction)
                
        return base_pressure
    
    def get_memory_pressure(self, context: Dict[str, Any]) -> float:
        """Get current memory selection pressure value."""
        task_type = context.get("task", {}).get("type", "default")
        
        # Similar logic to mutation pressure
        if task_type in self.reward_by_task_type:
            task_data = self.reward_by_task_type[task_type]
            if "memory_pressure" in task_data:
                return task_data["memory_pressure"]
                
        return self.current_pressure["memory"]
    
    def get_constraint_pressure(self, context: Dict[str, Any]) -> float:
        """Get current constraint adjustment pressure value."""
        return self.current_pressure["constraint"]
    
    def get_doctrine_pressure(self, context: Dict[str, Any]) -> float:
        """Get current doctrine evolution pressure value."""
        return self.current_pressure["doctrine"]
    
    def get_reward_slope(self, window_size: int = 10) -> float:
        """Calculate the slope of recent rewards."""
        if len(self.reward_history) < 2:
            return 0.0
            
        # Get recent rewards
        recent = list(self.reward_history)[-min(window_size, len(self.reward_history)):]
        
        if len(recent) < 2:
            return 0.0
            
        # Calculate slope using simple linear regression
        x = np.arange(len(recent))
        y = np.array([r["reward"] for r in recent])
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Update stats
        self.stats["reward_slope"] = slope
        
        return slope
    
    def _record_reward(self, 
                     reward: float, 
                     components: Dict[str, float],
                     task: Dict[str, Any]) -> None:
        """Record reward in history and update statistics."""
        # Create reward record
        record = {
            "reward": reward,
            "components": components,
            "timestamp": time.time(),
            "task_type": task.get("type", "default")
        }
        
        # Add to history
        self.reward_history.append(record)
        
        # Update overall stats
        self.stats["total_rewards"] += 1
        self.stats["avg_reward"] = (
            (self.stats["avg_reward"] * (self.stats["total_rewards"] - 1) + reward) / 
            self.stats["total_rewards"]
        )
        
        # Update task-specific stats
        task_type = task.get("type", "default")
        if task_type not in self.reward_by_task_type:
            self.reward_by_task_type[task_type] = {
                "count": 0,
                "avg_reward": 0.0,
                "success_rate": 0.0,
                "avg_time": 0.0
            }
            
        task_stats = self.reward_by_task_type[task_type]
        task_stats["count"] += 1
        task_stats["avg_reward"] = (
            (task_stats["avg_reward"] * (task_stats["count"] - 1) + reward) / 
            task_stats["count"]
        )
        
        # Update success rate
        success = components.get("success", 0.0) > 0.5
        prev_successes = task_stats["success_rate"] * (task_stats["count"] - 1)
        task_stats["success_rate"] = (prev_successes + (1 if success else 0)) / task_stats["count"]
        
        # Update average execution time
        if "execution_efficiency" in components and task.get("execution_time"):
            prev_time_sum = task_stats["avg_time"] * (task_stats["count"] - 1)
            task_stats["avg_time"] = (prev_time_sum + task.get("execution_time", 0)) / task_stats["count"]
        
        # Update component-specific stats
        for component, value in components.items():
            if component not in self.reward_by_component:
                self.reward_by_component[component] = {
                    "count": 0,
                    "avg_value": 0.0
                }
                
            comp_stats = self.reward_by_component[component]
            comp_stats["count"] += 1
            comp_stats["avg_value"] = (
                (comp_stats["avg_value"] * (comp_stats["count"] - 1) + value) / 
                comp_stats["count"]
            )
    
    def _adjust_pressure(self, reward: float, components: Dict[str, float]) -> None:
        """Adjust pressure signals based on reward feedback."""
        self.stats["pressure_adjustments"] += 1
        
        # Calculate reward delta (improvement or decline)
        if len(self.reward_history) >= 2:
            previous = self.reward_history[-2]["reward"]
            delta = reward - previous
        else:
            delta = 0.0
            
        # Get reward slope for trend analysis
        slope = self.get_reward_slope()
        
        # Apply pressure adjustments
        
        # 1. Mutation pressure
        if "mutation_efficiency" in components:
            mutation_eff = components["mutation_efficiency"]
            
            # Increase pressure when efficiency is low or reward is declining
            if mutation_eff < 0.5 or delta < -0.1:
                self.current_pressure["mutation"] = min(
                    1.0, 
                    self.current_pressure["mutation"] * 1.1
                )
            # Decrease pressure when efficiency is high and reward is stable/increasing
            elif mutation_eff > 0.8 and delta >= 0:
                self.current_pressure["mutation"] = max(
                    0.1,
                    self.current_pressure["mutation"] * 0.95
                )
        
        # 2. Memory pressure
        if "memory_utilization" in components:
            memory_util = components["memory_utilization"]
            
            # Adjust memory pressure based on utilization
            if memory_util < 0.4:
                # Increase pressure to use more memories
                self.current_pressure["memory"] = min(
                    1.0,
                    self.current_pressure["memory"] * 1.05
                )
            elif memory_util > 0.8:
                # Decrease pressure if memory utilization is good
                self.current_pressure["memory"] = max(
                    0.2,
                    self.current_pressure["memory"] * 0.98
                )
        
        # 3. Constraint pressure
        if "entropy_reduction" in components:
            entropy_red = components["entropy_reduction"]
            
            # Adjust constraint pressure based on entropy
            if entropy_red < 0.4:
                # Increase constraint pressure (tighten constraints)
                self.current_pressure["constraint"] = min(
                    1.0,
                    self.current_pressure["constraint"] * 1.1
                )
            elif entropy_red > 0.7:
                # Decrease constraint pressure (loosen constraints)
                self.current_pressure["constraint"] = max(
                    0.1,
                    self.current_pressure["constraint"] * 0.95
                )
        
        # 4. Doctrine pressure
        if "doctrine_alignment" in components:
            doctrine_align = components["doctrine_alignment"]
            
            # Adjust doctrine pressure based on alignment
            if doctrine_align < 0.5:
                # High pressure might indicate need for doctrine evolution
                if self.current_pressure["doctrine"] > 0.7:
                    # Keep high pressure as signal for potential doctrine update
                    pass
                else:
                    # Increase pressure to signal potential doctrine issues
                    self.current_pressure["doctrine"] = min(
                        0.8,  # Cap lower than others to minimize doctrine changes
                        self.current_pressure["doctrine"] * 1.1
                    )
            else:
                # Gradually reduce doctrine pressure when alignment is good
                self.current_pressure["doctrine"] = max(
                    0.1,
                    self.current_pressure["doctrine"] * 0.9
                )
        
        # Apply decay to all pressures
        for key in self.current_pressure:
            # Apply decay to gradually reduce pressure over time if not actively adjusted
            self.current_pressure[key] *= self.pressure_decay
            # Ensure minimum pressure
            self.current_pressure[key] = max(0.1, self.current_pressure[key])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reward and pressure statistics."""
        return {
            "reward": {
                "total_rewards": self.stats["total_rewards"],
                "avg_reward": self.stats["avg_reward"],
                "reward_slope": self.stats["reward_slope"],
                "task_types": len(self.reward_by_task_type)
            },
            "pressure": {
                "current": self.current_pressure,
                "adjustments": self.stats["pressure_adjustments"]
            }
        }
