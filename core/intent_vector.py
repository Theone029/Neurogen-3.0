import numpy as np
from typing import Dict, Any, List, Optional, Union
import time

class IntentVector:
    """Dynamic motivational vector that guides system behavior and alignment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dimensions = config.get("dimensions", {
            "coherence": 0.8,
            "knowledge": 0.7,
            "stability": 0.6,
            "exploration": 0.4,
            "efficiency": 0.5,
            "doctrinal_alignment": 0.9
        })
        
        # Initialize baseline vector from config
        self.baseline = np.array([self.dimensions[d] for d in sorted(self.dimensions.keys())])
        self.current = self.baseline.copy()
        self.dimension_names = sorted(self.dimensions.keys())
        
        # Tracking history
        self.history = []
        self.task_type_averages = {}
    
    def update(self, task: Dict[str, Any], doctrine: Dict[str, Any]) -> np.ndarray:
        """
        Update the intent vector based on task and current doctrine.
        
        Args:
            task: The current task to adjust intent for
            doctrine: Current Prime Directive state
            
        Returns:
            Updated intent vector
        """
        # Start with baseline
        updated = self.baseline.copy()
        
        # Adjust based on task type
        task_type = task.get("type", "default")
        task_adjustments = self._get_task_adjustments(task_type, task)
        
        for i, dim in enumerate(self.dimension_names):
            if dim in task_adjustments:
                updated[i] *= task_adjustments[dim]
        
        # Adjust based on doctrine constraints
        if "alignment_vectors" in doctrine:
            optimize_for = doctrine["alignment_vectors"].get("optimize_for", [])
            minimize = doctrine["alignment_vectors"].get("minimize", [])
            
            for dim in optimize_for:
                if dim in self.dimension_names:
                    idx = self.dimension_names.index(dim)
                    updated[idx] = min(1.0, updated[idx] * 1.2)  # Boost by 20%
                    
            for dim in minimize:
                if dim in self.dimension_names:
                    idx = self.dimension_names.index(dim)
                    updated[idx] = max(0.1, updated[idx] * 0.8)  # Reduce by 20%
        
        # Normalize to maintain overall magnitude
        original_magnitude = np.linalg.norm(self.baseline)
        updated_magnitude = np.linalg.norm(updated)
        if updated_magnitude > 0:
            updated = updated * (original_magnitude / updated_magnitude)
        
        # Record the update
        self.current = updated
        self._record_update(task, updated)
        
        return updated
    
    def get_current(self) -> np.ndarray:
        """Get the current intent vector."""
        return self.current
    
    def get_dimension_value(self, dimension: str) -> float:
        """Get the current value for a specific dimension."""
        if dimension in self.dimension_names:
            idx = self.dimension_names.index(dimension)
            return self.current[idx]
        return 0.0
    
    def get_vector_as_dict(self) -> Dict[str, float]:
        """Get the current intent vector as a dictionary."""
        return {dim: self.current[i] for i, dim in enumerate(self.dimension_names)}
    
    def calculate_drift(self, comparison_vector: Optional[np.ndarray] = None) -> float:
        """Calculate drift from baseline or provided vector."""
        if comparison_vector is None:
            comparison_vector = self.baseline
            
        # Compute cosine similarity
        dot_product = np.dot(self.current, comparison_vector)
        norm_product = np.linalg.norm(self.current) * np.linalg.norm(comparison_vector)
        
        if norm_product == 0:
            return 1.0  # Maximum drift if either vector is zero
            
        similarity = dot_product / norm_product
        # Convert to distance (0 = same, 2 = opposite)
        return 1.0 - similarity
    
    def _get_task_adjustments(self, task_type: str, task: Dict[str, Any]) -> Dict[str, float]:
        """Get adjustments for a given task type."""
        # Default adjustments
        adjustments = {}
        
        # Task type specific adjustments
        if task_type == "problem_solving":
            adjustments = {
                "knowledge": 1.2,
                "coherence": 1.1,
                "exploration": 0.9
            }
        elif task_type == "creative":
            adjustments = {
                "exploration": 1.3,
                "coherence": 0.9,
                "efficiency": 0.8
            }
        elif task_type == "analytical":
            adjustments = {
                "knowledge": 1.2,
                "stability": 1.1,
                "coherence": 1.1,
                "exploration": 0.8
            }
        elif task_type == "doctrinal":
            adjustments = {
                "doctrinal_alignment": 1.3,
                "coherence": 1.2,
                "stability": 1.1,
                "exploration": 0.7
            }
        
        # Task specific adjustments (override type-based ones)
        if "intent_adjustments" in task:
            for dim, value in task["intent_adjustments"].items():
                if dim in self.dimension_names:
                    adjustments[dim] = value
        
        return adjustments
    
