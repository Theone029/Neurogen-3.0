import numpy as np
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

class IntentVector:
    """Evolving high-dimensional representation of system goals and directive priorities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Core dimensions with initial values
        self.dimensions = config.get("dimensions", {
            "exploration": 0.5,      # Exploration vs exploitation
            "coherence": 0.7,        # Coherence vs novelty
            "stability": 0.6,        # Stability vs adaptation
            "knowledge": 0.6,        # Knowledge acquisition vs application
            "efficiency": 0.5,       # Efficiency vs thoroughness
            "abstraction": 0.4,      # Abstraction vs concreteness
            "doctrinal_alignment": 0.8  # Alignment with prime directive
        })
        
        # Dimension properties (min, max, volatility, decay)
        self.dimension_properties = config.get("dimension_properties", {
            dim: {"min": 0.1, "max": 0.9, "volatility": 0.1, "decay": 0.001}
            for dim in self.dimensions
        })
        
        # Initialize current vector
        self.current = np.array(list(self.dimensions.values()))
        
        # Historical vectors for tracking evolution
        self.history = deque(maxlen=config.get("history_size", 100))
        self.history.append({
            "vector": self.dimensions.copy(),
            "timestamp": time.time(),
            "reason": "initialization"
        })
        
        # Pressure points - dimensions under external pressure
        self.pressure_points = {}
        
        # Stats tracking
        self.stats = {
            "updates": 0,
            "significant_shifts": 0,
            "avg_magnitude": 0.0,
            "dimension_volatility": {d: 0.0 for d in self.dimensions}
        }
    
    def update(self, 
              shift: Dict[str, float], 
              context: Dict[str, Any], 
              reason: str = "standard_update") -> Dict[str, Any]:
        """
        Update the intent vector with new pressures/shifts.
        
        Args:
            shift: Dictionary of dimension shifts (dimension: delta)
            context: Current execution context
            reason: Reason for this update
            
        Returns:
            Update report with changes
        """
        self.stats["updates"] += 1
        
        # Track pre-update state
        pre_update = self.dimensions.copy()
        
        # Calculate adaptive rate based on context
        base_rate = self.config.get("base_adaptation_rate", 0.1)
        adaptive_rate = self._calculate_adaptive_rate(base_rate, context)
        
        # Process each dimension in the shift
        changes = {}
        significant_shift = False
        
        for dim, delta in shift.items():
            if dim not in self.dimensions:
                # Skip unknown dimensions
                continue
                
            # Get current value and properties
            current = self.dimensions[dim]
            properties = self.dimension_properties.get(dim, 
                {"min": 0.1, "max": 0.9, "volatility": 0.1, "decay": 0.001})
            
            # Apply volatility modifier based on dimension properties
            dim_volatility = properties["volatility"]
            actual_delta = delta * adaptive_rate * dim_volatility
            
            # Calculate new value with bounds
            new_value = max(properties["min"], 
                          min(properties["max"], current + actual_delta))
            
            # Check if change is significant
            change_magnitude = abs(new_value - current)
            if change_magnitude > 0.05:  # 5% threshold for significant change
                significant_shift = True
                
            # Record the change
            changes[dim] = {
                "previous": current,
                "new": new_value,
                "delta": new_value - current,
                "magnitude": change_magnitude
            }
            
            # Apply the change
            self.dimensions[dim] = new_value
        
        # Update numpy array representation
        self.current = np.array(list(self.dimensions.values()))
        
        # Add to history if there were changes
        if changes:
            entry = {
                "vector": self.dimensions.copy(),
                "timestamp": time.time(),
                "reason": reason,
                "changes": changes
            }
            self.history.append(entry)
            
        # Update stats
        if significant_shift:
            self.stats["significant_shifts"] += 1
            
        # Calculate average magnitude
        if changes:
            avg_mag = sum(c["magnitude"] for c in changes.values()) / len(changes)
            self.stats["avg_magnitude"] = (
                (self.stats["avg_magnitude"] * (self.stats["updates"] - 1) + avg_mag) / 
                self.stats["updates"]
            )
            
            # Update dimension volatility stats
            for dim in changes:
                current_vol = self.stats["dimension_volatility"][dim]
                new_vol = (current_vol * (self.stats["updates"] - 1) + 
                         changes[dim]["magnitude"]) / self.stats["updates"]
                self.stats["dimension_volatility"][dim] = new_vol
                
        # Create report
        report = {
            "previous": pre_update,
            "current": self.dimensions.copy(),
            "changes": changes,
            "significant_shift": significant_shift,
            "adaptive_rate": adaptive_rate,
            "reason": reason
        }
        
        return report
    
    def apply_pressure(self, 
                     dimension: str, 
                     pressure: float, 
                     duration: int,
                     source: str = "unspecified") -> bool:
        """
        Apply sustained pressure to a dimension over time.
        
        Args:
            dimension: The dimension to pressure
            pressure: Amount of pressure (-1.0 to 1.0)
            duration: Number of cycles to maintain pressure
            source: Source of the pressure
            
        Returns:
            Success flag
        """
        if dimension not in self.dimensions:
            return False
            
        # Record pressure point
        self.pressure_points[dimension] = {
            "pressure": pressure,
            "remaining": duration,
            "source": source,
            "created_at": time.time()
        }
        
        return True
    
    def release_pressure(self, dimension: str) -> bool:
        """Release pressure on a specific dimension."""
        if dimension in self.pressure_points:
            del self.pressure_points[dimension]
            return True
        return False
    
    def cycle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a cycle update, applying decay and pressures.
        
        Args:
            context: Current execution context
            
        Returns:
            Cycle update report
        """
        # Calculate decay influence
        decayed_dimensions = {}
        for dim, value in self.dimensions.items():
            properties = self.dimension_properties.get(dim, {})
            decay_rate = properties.get("decay", 0.001)
            
            # Calculate homeostasis point (default 0.5)
            homeostasis = properties.get("homeostasis", 0.5)
            
            # Decay toward homeostasis
            if abs(value - homeostasis) > 0.01:  # Only decay if significantly different
                direction = -1 if value > homeostasis else 1
                decay_amount = decay_rate * abs(value - homeostasis)
                
                # Record dimension for decay
                decayed_dimensions[dim] = direction * decay_amount
        
        # Process active pressure points
        pressured_dimensions = {}
        expired_pressure = []
        
        for dim, pressure_data in self.pressure_points.items():
            # Skip dimensions that don't exist
            if dim not in self.dimensions:
                expired_pressure.append(dim)
                continue
                
            # Apply pressure
            pressure_value = pressure_data["pressure"]
            pressure_amount = pressure_value * 0.05  # 5% influence per cycle
            
            pressured_dimensions[dim] = pressure_amount
            
            # Reduce remaining duration
            pressure_data["remaining"] -= 1
            
            # Check if pressure has expired
            if pressure_data["remaining"] <= 0:
                expired_pressure.append(dim)
        
        # Remove expired pressure points
        for dim in expired_pressure:
            if dim in self.pressure_points:
                del self.pressure_points[dim]
        
        # Combine decay and pressure effects
        combined_shift = {}
        
        # First apply decay to all dimensions
        for dim, decay in decayed_dimensions.items():
            combined_shift[dim] = decay
            
        # Then apply pressure, overriding decay
        for dim, pressure in pressured_dimensions.items():
            # Add to existing shift or create new entry
            if dim in combined_shift:
                combined_shift[dim] += pressure
            else:
                combined_shift[dim] = pressure
        
        # Apply the combined shift
        if combined_shift:
            return self.update(combined_shift, context, "cycle_update")
        
        return {"changes": {}, "significant_shift": False}
    
    def get_dominant_intent(self) -> Tuple[str, float]:
        """Get the currently dominant intent dimension."""
        if not self.dimensions:
            return ("none", 0.0)
            
        # Find dimension with highest value
        dominant = max(self.dimensions.items(), key=lambda x: x[1])
        return dominant
    
    def get_vector_as_dict(self) -> Dict[str, float]:
        """Get the current intent vector as a dictionary."""
        return self.dimensions.copy()
    
    def get_dimension(self, dimension: str, default: float = 0.5) -> float:
        """Get the current value of a specific dimension."""
        return self.dimensions.get(dimension, default)
    
    def get_evolution_report(self) -> Dict[str, Any]:
        """Generate a report on the evolution of the intent vector."""
        if len(self.history) < 2:
            return {
                "evolution": "insufficient_history",
                "dimensions": len(self.dimensions),
                "updates": self.stats["updates"]
            }
            
        # Get first and most recent entries
        first = self.history[0]
        latest = self.history[-1]
        
        # Calculate overall evolution
        evolution = {}
        for dim in self.dimensions:
            if dim in first["vector"] and dim in latest["vector"]:
                evolution[dim] = {
                    "start": first["vector"][dim],
                    "current": latest["vector"][dim],
                    "change": latest["vector"][dim] - first["vector"][dim],
                    "volatility": self.stats["dimension_volatility"].get(dim, 0.0)
                }
        
        # Calculate significant dimension shifts
        significant_shifts = []
        for dim, data in evolution.items():
            if abs(data["change"]) > 0.2:  # 20% shift threshold
                significant_shifts.append({
                    "dimension": dim,
                    "change": data["change"],
                    "from": data["start"],
                    "to": data["current"]
                })
        
        # Sort by magnitude of change
        significant_shifts.sort(key=lambda x: abs(x["change"]), reverse=True)
        
        return {
            "evolution": evolution,
            "significant_shifts": significant_shifts,
            "time_period": latest["timestamp"] - first["timestamp"],
            "updates": self.stats["updates"],
            "dominant_intent": self.get_dominant_intent()[0]
        }
    
    def export_to_json(self) -> str:
        """Export the current intent state to JSON."""
        export_data = {
            "dimensions": self.dimensions,
            "properties": self.dimension_properties,
            "dominant": self.get_dominant_intent()[0],
            "history_size": len(self.history),
            "pressures": {k: v.copy() for k, v in self.pressure_points.items()},
            "stats": self.stats.copy(),
            "exported_at": time.time()
        }
        
        return json.dumps(export_data, indent=2)
    
    def _calculate_adaptive_rate(self, base_rate: float, context: Dict[str, Any]) -> float:
        """Calculate adaptive rate based on context."""
        # Start with base rate
        rate = base_rate
        
        # Adjust based on mutation count (more mutations → lower adaptation)
        mutation_count = context.get("mutation_count", 0)
        if mutation_count > 0:
            rate *= (1.0 - 0.1 * min(mutation_count, 3))  # Max 30% reduction
            
        # Adjust based on success (failure → higher adaptation)
        if "success" in context:
            success = context["success"]
            if not success:
                rate *= 1.2  # 20% increase for failures
            else:
                rate *= 0.9  # 10% decrease for successes
                
        # Adjust based on volatility goal
        volatility_goal = context.get("intent_volatility_goal", None)
        if volatility_goal is not None:
            current_volatility = self.stats["avg_magnitude"]
            if volatility_goal > current_volatility:
                rate *= 1.1  # Increase if we want more volatility
            else:
                rate *= 0.9  # Decrease if we want less volatility
        
        # Cap rate within reasonable bounds
        return max(0.01, min(0.5, rate))
