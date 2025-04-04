import numpy as np
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple

class DriftAuditor:
    """Monitors system drift from baseline state across multiple dimensions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.baseline = None
        self.drift_history = []
        self.component_history = {}
        self.drift_thresholds = config.get("drift_thresholds", {
            "plan_structure": 0.3,
            "output_entropy": 0.4,
            "memory_usage": 0.35,
            "intent_vector": 0.25
        })
        
    def set_baseline(self, state: Dict[str, Any]) -> None:
        """Set baseline state for drift comparison."""
        self.baseline = self._extract_drift_signatures(state)
        
    def measure_drift(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Measure drift from baseline across multiple dimensions.
        
        Args:
            current_state: Current system state data
            
        Returns:
            Drift analysis with overall score and component breakdown
        """
        # Generate unique ID for this measurement
        drift_id = hashlib.md5(f"drift:{time.time()}".encode()).hexdigest()[:12]
        
        # If no baseline set, use this as baseline
        if self.baseline is None:
            self.set_baseline(current_state)
            return {
                "drift_id": drift_id,
                "drift": 0.0,
                "components": {},
                "is_baseline": True,
                "timestamp": time.time()
            }
        
        # Extract signatures from current state
        current_signatures = self._extract_drift_signatures(current_state)
        
        # Calculate drift for each component
        components = {}
        for component, signature in current_signatures.items():
            if component in self.baseline:
                baseline_signature = self.baseline[component]
                component_drift = self._calculate_component_drift(baseline_signature, signature, component)
                
                components[component] = {
                    "drift": component_drift,
                    "threshold": self.drift_thresholds.get(component, 0.3),
                    "critical": component_drift > self.drift_thresholds.get(component, 0.3)
                }
                
                # Track component history
                if component not in self.component_history:
                    self.component_history[component] = []
                    
                self.component_history[component].append({
                    "timestamp": time.time(),
                    "drift": component_drift
                })
        
        # Calculate overall drift as weighted combination
        weights = self.config.get("component_weights", {
            "plan_structure": 0.3,
            "output_entropy": 0.2,
            "memory_usage": 0.25,
            "intent_vector": 0.25
        })
        
        total_drift = 0.0
        total_weight = 0.0
        
        for component, data in components.items():
            if component in weights:
                total_drift += data["drift"] * weights[component]
                total_weight += weights[component]
        
        # Normalize
        if total_weight > 0:
            total_drift /= total_weight
        
        # Create drift record
        drift_record = {
            "drift_id": drift_id,
            "drift": total_drift,
            "components": components,
            "timestamp": time.time(),
            "critical": total_drift > self.config.get("critical_total_drift", 0.4)
        }
        
        # Store in history
        self.drift_history.append(drift_record)
        
        # Trim history if needed
        max_history = self.config.get("max_history_size", 1000)
        if len(self.drift_history) > max_history:
            self.drift_history = self.drift_history[-max_history:]
            
        return drift_record
    
    def get_drift_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """Calculate drift trend over recent history."""
        if not self.drift_history or len(self.drift_history) < 2:
            return {"trend": 0.0, "stable": True}
            
        # Get recent drift measurements
        recent = self.drift_history[-min(window_size, len(self.drift_history)):]
        drift_values = [r["drift"] for r in recent]
        
        # Calculate trend (slope)
        x = np.arange(len(drift_values))
        if len(x) > 1:
            slope, _, _, _, _ = np.polyfit(x, drift_values, 1, full=True)
        else:
            slope = 0.0
            
        # Calculate volatility (standard deviation)
        volatility = np.std(drift_values) if len(drift_values) > 1 else 0.0
        
        # Determine stability
        stable = (
            abs(slope) < self.config.get("max_trend_slope", 0.05) and
            volatility < self.config.get("max_volatility", 0.1)
        )
        
        return {
            "trend": float(slope),
            "volatility": float(volatility),
            "stable": stable,
            "samples": len(drift_values),
            "current": drift_values[-1] if drift_values else 0.0,
            "max": max(drift_values) if drift_values else 0.0
        }
    
    def should_trigger_fork(self) -> Tuple[bool, str]:
        """Determine if drift is significant enough to trigger a fork."""
        if not self.drift_history:
            return False, "Insufficient history"
            
        current_drift = self.drift_history[-1]["drift"]
        trend = self.get_drift_trend()
        
        # Check if current drift exceeds fork threshold
        if current_drift > self.config.get("fork_threshold", 0.6):
            return True, f"Total drift {current_drift:.2f} exceeds fork threshold"
            
        # Check if drift trend is rapidly increasing
        if trend["trend"] > self.config.get("critical_trend_threshold", 0.1):
            return True, f"Drift increasing rapidly with slope {trend['trend']:.3f}"
            
        # Check component-specific triggers
        components = self.drift_history[-1]["components"]
        for component, data in components.items():
            if data["drift"] > self.config.get(f"{component}_fork_threshold", 0.7):
                return True, f"Component {component} drift {data['drift']:.2f} exceeds threshold"
                
        return False, "Drift within acceptable parameters"
    
    def _extract_drift_signatures(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract signatures from state for drift comparison."""
        signatures = {}
        
        # Plan structure signature
        if "plan" in state:
            plan = state["plan"]
            signatures["plan_structure"] = self._extract_plan_signature(plan)
            
        # Output entropy signature
        if "output" in state:
            output = state["output"]
            signatures["output_entropy"] = self._extract_output_signature(output)
            
        # Memory usage signature
        if "memories_used" in state:
            memories = state["memories_used"]
            signatures["memory_usage"] = self._extract_memory_signature(memories)
            
        # Intent vector signature
        if "intent" in state:
            intent = state["intent"]
            signatures["intent_vector"] = self._extract_intent_signature(intent)
            
        return signatures
    
    def _extract_plan_signature(self, plan: Any) -> Dict[str, Any]:
        """Extract structural signature from a plan."""
        if not isinstance(plan, dict):
            return {"type": "invalid"}
            
        # Create a simplified representation of plan structure
        structure = {}
        
        # Count steps if present
        if "steps" in plan and isinstance(plan["steps"], list):
            structure["step_count"] = len(plan["steps"])
            
            # Analyze step types
            step_types = {}
            for step in plan["steps"]:
                if isinstance(step, dict) and "type" in step:
                    step_type = step["type"]
                    if step_type not in step_types:
                        step_types[step_type] = 0
                    step_types[step_type] += 1
                    
            structure["step_types"] = step_types
            
        # Count constraints if present
        if "constraints" in plan and isinstance(plan["constraints"], dict):
            structure["constraint_count"] = len(plan["constraints"])
            
        # Check for mutation
        if "mutation_metadata" in plan:
            structure["is_mutation"] = True
            
        return structure
    
    def _extract_output_signature(self, output: Any) -> Dict[str, Any]:
        """Extract entropy signature from output."""
        signature = {"type": "unknown"}
        
        if isinstance(output, dict):
            signature["type"] = "dict"
            signature["key_count"] = len(output)
            signature["depth"] = self._calculate_dict_depth(output)
            
        elif isinstance(output, list):
            signature["type"] = "list"
            signature["item_count"] = len(output)
            
        elif isinstance(output, str):
            signature["type"] = "string"
            signature["length"] = len(output)
            signature["word_count"] = len(output.split())
            
        return signature
    
    def _extract_memory_signature(self, memories: List[str]) -> Dict[str, Any]:
        """Extract signature from memory usage patterns."""
        return {
            "count": len(memories),
            "unique": len(set(memories))
        }
    
    def _extract_intent_signature(self, intent: Any) -> Dict[str, Any]:
        """Extract signature from intent vector."""
        if hasattr(intent, "tolist"):
            # NumPy array or similar
            return {"vector": intent.tolist()}
            
        elif hasattr(intent, "get_vector_as_dict"):
            # Intent object with dimension accessor
            return {"vector": intent.get_vector_as_dict()}
            
        elif isinstance(intent, dict):
            # Dictionary representation
            return {"vector": intent}
            
        else:
            # Unknown format
            return {"vector": str(intent)}
    
    def _calculate_component_drift(self, baseline: Dict[str, Any], current: Dict[str, Any], component: str) -> float:
        """Calculate drift for a specific component."""
        if component == "plan_structure":
            return self._calculate_plan_drift(baseline, current)
            
        elif component == "output_entropy":
            return self._calculate_output_drift(baseline, current)
            
        elif component == "memory_usage":
            return self._calculate_memory_drift(baseline, current)
            
        elif component == "intent_vector":
            return self._calculate_intent_drift(baseline, current)
            
        # Default generic calculation
        return 0.5  # Middle value indicates uncertain drift
    
    def _calculate_plan_drift(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> float:
        """Calculate drift between plan structures."""
        drift = 0.0
        
        # Compare step counts
        if "step_count" in baseline and "step_count" in current:
            baseline_steps = baseline["step_count"]
            current_steps = current["step_count"]
            
            if baseline_steps > 0:
                step_diff = abs(baseline_steps - current_steps) / baseline_steps
                drift += step_diff * 0.3  # 30% of drift from step count difference
            
        # Compare step type distribution
        if "step_types" in baseline and "step_types" in current:
            baseline_types = set(baseline["step_types"].keys())
            current_types = set(current["step_types"].keys())
            
            all_types = baseline_types.union(current_types)
            if all_types:
                # Calculate Jaccard distance for step types
                jaccard = 1.0 - len(baseline_types.intersection(current_types)) / len(all_types)
                drift += jaccard * 0.4  # 40% of drift from step type changes
                
                # Calculate distribution difference for common types
                common_types = baseline_types.intersection(current_types)
                if common_types:
                    type_diffs = 0.0
                    for t in common_types:
                        baseline_count = baseline["step_types"].get(t, 0)
                        current_count = current["step_types"].get(t, 0)
                        
                        if baseline_count > 0:
                            type_diffs += abs(baseline_count - current_count) / baseline_count
                            
                    type_drift = type_diffs / len(common_types) if common_types else 0.0
                    drift += type_drift * 0.2  # 20% from distribution changes
        
        # Compare constraint count
        if "constraint_count" in baseline and "constraint_count" in current:
            baseline_constraints = baseline["constraint_count"]
            current_constraints = current["constraint_count"]
            
            if baseline_constraints > 0:
                constraint_diff = abs(baseline_constraints - current_constraints) / baseline_constraints
                drift += constraint_diff * 0.1  # 10% from constraint changes
                
        return min(1.0, drift)  # Cap at 1.0
    
    def _calculate_output_drift(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> float:
        """Calculate drift between output signatures."""
        # If types differ, that's a major drift
        if baseline["type"] != current["type"]:
            return 0.7  # High drift for type changes
            
        drift = 0.0
        
        if baseline["type"] == "dict":
            # Compare key count
            if "key_count" in baseline and "key_count" in current:
                baseline_keys = baseline["key_count"]
                current_keys = current["key_count"]
                
                if baseline_keys > 0:
                    key_diff = abs(baseline_keys - current_keys) / baseline_keys
                    drift += key_diff * 0.5
                    
            # Compare depth
            if "depth" in baseline and "depth" in current:
                baseline_depth = baseline["depth"] 
                current_depth = current["depth"]
                
                if baseline_depth > 0:
                    depth_diff = abs(baseline_depth - current_depth) / baseline_depth
                    drift += depth_diff * 0.5
                    
        elif baseline["type"] == "list":
            # Compare item count
            if "item_count" in baseline and "item_count" in current:
                baseline_items = baseline["item_count"]
                current_items = current["item_count"]
                
                if baseline_items > 0:
                    item_diff = abs(baseline_items - current_items) / baseline_items
                    drift += item_diff
                    
        elif baseline["type"] == "string":
            # Compare string length
            if "length" in baseline and "length" in current:
                baseline_len = baseline["length"]
                current_len = current["length"]
                
                if baseline_len > 0:
                    len_diff = abs(baseline_len - current_len) / baseline_len
                    drift += len_diff * 0.5
                    
            # Compare word count
            if "word_count" in baseline and "word_count" in current:
                baseline_words = baseline["word_count"]
                current_words = current["word_count"]
                
                if baseline_words > 0:
                    word_diff = abs(baseline_words - current_words) / baseline_words
                    drift += word_diff * 0.5
                    
        return min(1.0, drift)
    
    def _calculate_memory_drift(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> float:
        """Calculate drift in memory usage patterns."""
        drift = 0.0
        
        # Compare memory count
        if "count" in baseline and "count" in current:
            baseline_count = baseline["count"]
            current_count = current["count"]
            
            if baseline_count > 0:
                count_diff = abs(baseline_count - current_count) / max(1, baseline_count)
                drift += count_diff * 0.7
                
        # Compare unique memory proportion
        if "count" in baseline and "unique" in baseline and "count" in current and "unique" in current:
            if baseline["count"] > 0 and current["count"] > 0:
                baseline_unique_ratio = baseline["unique"] / baseline["count"]
                current_unique_ratio = current["unique"] / current["count"]
                
                unique_diff = abs(baseline_unique_ratio - current_unique_ratio)
                drift += unique_diff * 0.3
                
        return min(1.0, drift)
    
    def _calculate_intent_drift(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> float:
        """Calculate drift between intent vectors."""
        if "vector" not in baseline or "vector" not in current:
            return 0.5  # Uncertain drift if vector not available
            
        baseline_vector = baseline["vector"]
        current_vector = current["vector"]
        
        # Handle different vector representations
        if isinstance(baseline_vector, list) and isinstance(current_vector, list):
            # Vector as list
            if len(baseline_vector) != len(current_vector):
                return 0.8  # High drift for dimension changes
                
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(baseline_vector, current_vector))
            magnitude_a = sum(a * a for a in baseline_vector) ** 0.5
            magnitude_b = sum(b * b for b in current_vector) ** 0.5
            
            if magnitude_a == 0 or magnitude_b == 0:
                return 1.0  # Maximum drift if either vector is zero
                
            similarity = dot_product / (magnitude_a * magnitude_b)
            return 1.0 - max(0, min(1, similarity))
            
        elif isinstance(baseline_vector, dict) and isinstance(current_vector, dict):
            # Vector as dictionary
            all_dims = set(baseline_vector.keys()).union(current_vector.keys())
            
            # Convert to common format
            vec_a = [baseline_vector.get(dim, 0.0) for dim in all_dims]
            vec_b = [current_vector.get(dim, 0.0) for dim in all_dims]
            
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
            magnitude_a = sum(a * a for a in vec_a) ** 0.5
            magnitude_b = sum(b * b for b in vec_b) ** 0.5
            
            if magnitude_a == 0 or magnitude_b == 0:
                return 1.0
                
            similarity = dot_product / (magnitude_a * magnitude_b)
            return 1.0 - max(0, min(1, similarity))
        
        # Fallback for incomparable formats
        return 0.6
    
    def _calculate_dict_depth(self, d: Dict[str, Any], current_depth: int = 1) -> int:
        """Calculate the maximum depth of a nested dictionary."""
        if not isinstance(d, dict) or not d:
            return current_depth
            
        return max(
            self._calculate_dict_depth(v, current_depth + 1) if isinstance(v, dict) else current_depth
            for k, v in d.items()
        )
