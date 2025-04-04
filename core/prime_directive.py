import json
import hashlib
from typing import Dict, List, Any, Optional
import datetime

class PrimeDirective:
    """Immutable foundation for system alignment and purpose."""
    
    def __init__(self, directive_path: str = "configs/prime_directive.json"):
        self.version_hash = None
        self.directive_path = directive_path
        self.laws = self._load_directive()
        self.version_history = []
        self._compute_hash()
        self._log_version("initial", None, "System initialization")
    
    def _load_directive(self) -> Dict[str, Any]:
        """Load directive from version-controlled JSON."""
        try:
            with open(self.directive_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Bootstrap with minimal core laws if file doesn't exist
            return {
                "core_laws": [
                    "Mutation is valid only if entropy is reduced or recursive alignment increases",
                    "Memory must be preserved across mutations unless explicitly invalidated",
                    "Intent vectors must maintain alignment with system purpose",
                    "Evolve toward greater coherence and reduced entropy",
                    "Fork only when reconciliation is impossible"
                ],
                "alignment_vectors": {
                    "optimize_for": ["coherence", "knowledge", "stability"],
                    "minimize": ["entropy", "invalid_mutation", "memory_corruption"]
                },
                "mutation_constraints": {
                    "max_drift_per_cycle": 0.15,
                    "min_coherence_threshold": 0.75,
                    "require_justification": True
                }
            }
    
    def _compute_hash(self) -> str:
        """Compute cryptographic hash of current directive state."""
        directive_str = json.dumps(self.laws, sort_keys=True)
        self.version_hash = hashlib.sha256(directive_str.encode()).hexdigest()
        return self.version_hash
    
    def _log_version(self, version_id: str, parent_id: Optional[str], justification: str) -> None:
        """Record version change with justification."""
        self.version_history.append({
            "version_id": version_id if version_id != "initial" else self.version_hash[:12],
            "parent_id": parent_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "hash": self.version_hash,
            "justification": justification
        })
    
    def validate_mutation(self, proposed_laws: Dict[str, Any], justification: str) -> Dict[str, Any]:
        """Validate a proposed mutation to the directive."""
        # Create deep copy of current laws
        current = json.loads(json.dumps(self.laws))
        
        # Check for critical protections
        for core_law in current["core_laws"]:
            if core_law not in proposed_laws["core_laws"]:
                return {
                    "valid": False,
                    "reason": f"Critical law removed: {core_law}",
                    "protection_triggered": "core_law_protection"
                }
        
        # Verify alignment vectors haven't been compromised
        for key in current["alignment_vectors"]:
            if key not in proposed_laws["alignment_vectors"]:
                return {
                    "valid": False, 
                    "reason": f"Alignment vector removed: {key}",
                    "protection_triggered": "alignment_protection"
                }
        
        # Compute preliminary drift measure
        drift = self._compute_directive_drift(current, proposed_laws)
        if drift > current["mutation_constraints"]["max_drift_per_cycle"]:
            return {
                "valid": False,
                "reason": f"Mutation drift {drift:.2f} exceeds limit {current['mutation_constraints']['max_drift_per_cycle']}",
                "protection_triggered": "drift_protection"
            }
            
        # All checks passed
        return {
            "valid": True,
            "drift": drift,
            "requires_fork": drift > current["mutation_constraints"]["max_drift_per_cycle"] * 0.8
        }
    
    def update(self, proposed_laws: Dict[str, Any], justification: str) -> Dict[str, Any]:
        """Attempt to update the directive with justification."""
        validation = self.validate_mutation(proposed_laws, justification)
        
        if not validation["valid"]:
            return validation
            
        # Store old version info
        old_hash = self.version_hash
        old_version_id = self.version_history[-1]["version_id"]
        
        # Apply update
        self.laws = proposed_laws
        new_hash = self._compute_hash()
        new_version_id = f"{new_hash[:12]}"
        
        # Log the change
        self._log_version(new_version_id, old_version_id, justification)
        
        return {
            "valid": True,
            "old_version": old_version_id,
            "new_version": new_version_id,
            "drift": validation["drift"],
            "requires_fork": validation.get("requires_fork", False)
        }
    
    def _compute_directive_drift(self, current: Dict[str, Any], proposed: Dict[str, Any]) -> float:
        """Compute numerical drift between directive versions."""
        # Count changes to core laws
        law_changes = set(current["core_laws"]) ^ set(proposed["core_laws"])
        law_drift = len(law_changes) / max(len(current["core_laws"]), len(proposed["core_laws"]))
        
        # Count alignment vector changes
        align_drift = 0.0
        for key in set(list(current["alignment_vectors"].keys()) + list(proposed["alignment_vectors"].keys())):
            if key not in current["alignment_vectors"] or key not in proposed["alignment_vectors"]:
                align_drift += 1.0
                continue
                
            current_vals = set(current["alignment_vectors"][key])
            proposed_vals = set(proposed["alignment_vectors"][key])
            changes = current_vals ^ proposed_vals
            align_drift += len(changes) / max(len(current_vals), len(proposed_vals))
        
        align_drift = align_drift / max(len(current["alignment_vectors"]), len(proposed["alignment_vectors"]))
        
        # Weight different components
        total_drift = 0.7 * law_drift + 0.3 * align_drift
        return total_drift
    
    def get_current_version(self) -> Dict[str, Any]:
        """Get the current directive version info."""
        return {
            "version_id": self.version_history[-1]["version_id"],
            "hash": self.version_hash,
            "laws": self.laws,
            "timestamp": self.version_history[-1]["timestamp"]
        }
