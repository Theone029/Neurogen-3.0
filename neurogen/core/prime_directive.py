import json
import uuid
import time
import copy
from typing import Dict, List, Any, Optional, Tuple

class PrimeDirective:
    """Core value system and principles that govern NEUROGEN's behavior and evolution."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize doctrine from config or use default
        self.current_version = config.get("initial_doctrine", {
            "version_id": "v1.0.0",
            "created_at": time.time(),
            "core_laws": [
                "Act according to the user's intent and purpose",
                "Maximize coherence while preserving adaptation capability",
                "Disclose reasoning processes transparently when beneficial",
                "Never attempt to directly execute real-world actions",
                "Continuously improve understanding through recursive self-reflection"
            ],
            "alignment_vectors": {
                "optimize_for": [
                    "transparency", "coherence", "capability", "accuracy", "insight"
                ],
                "minimize": [
                    "hallucination", "overconfidence", "manipulation", "rigidity", "opacity"
                ]
            },
            "prohibited_actions": [
                "executing_system_commands",
                "bypassing_constraints",
                "concealing_reasoning",
                "modifying_prime_directive_without_authorization",
                "engaging_harmful_outputs"
            ],
            "evolution_constraints": {
                "max_version_delta": 0.1,      # Maximum change in one evolution
                "required_approval": True,      # Requires approval for changes
                "min_justification_length": 50  # Minimum justification detail
            }
        })
        
        # Version history
        self.version_history = [self.current_version.copy()]
        
        # Lock status - when locked, doctrine cannot be modified
        self.locked = config.get("locked", False)
        
        # Evolution metrics
        self.evolution_metrics = {
            "total_evolutions": 0,
            "rejected_evolutions": 0,
            "avg_coherence_impact": 0.0,
            "changes_by_category": {
                "core_laws": 0,
                "alignment_vectors": 0,
                "prohibited_actions": 0,
                "evolution_constraints": 0
            }
        }
    
    def get_current_version(self) -> Dict[str, Any]:
        """Get current doctrine version."""
        return copy.deepcopy(self.current_version)
    
    def evolve_doctrine(self, 
                      proposed_changes: Dict[str, Any], 
                      justification: str,
                      coherence_impact: float,
                      approval: bool = False) -> Dict[str, Any]:
        """
        Propose an evolution to the doctrine.
        
        Args:
            proposed_changes: Dictionary of changes to apply
            justification: Reason for these changes
            coherence_impact: Predicted impact on system coherence (-1 to 1)
            approval: Whether this change has been approved
            
        Returns:
            Evolution results with status and explanation
        """
        # Check lock status
        if self.locked:
            return {
                "status": "rejected",
                "reason": "doctrine_locked",
                "current_version": self.current_version["version_id"]
            }
            
        # Apply evolution constraints
        constraints = self.current_version.get("evolution_constraints", {})
        
        # Check approval requirement
        if constraints.get("required_approval", True) and not approval:
            return {
                "status": "pending_approval",
                "reason": "approval_required",
                "current_version": self.current_version["version_id"],
                "proposed_changes": proposed_changes
            }
            
        # Check justification length
        min_length = constraints.get("min_justification_length", 0)
        if len(justification) < min_length:
            return {
                "status": "rejected",
                "reason": "insufficient_justification",
                "current_version": self.current_version["version_id"]
            }
            
        # Validate changes don't exceed maximum delta
        max_delta = constraints.get("max_version_delta", 0.1)
        changes_magnitude = self._calculate_changes_magnitude(proposed_changes)
        
        if changes_magnitude > max_delta:
            return {
                "status": "rejected",
                "reason": "excessive_change_magnitude",
                "current_version": self.current_version["version_id"],
                "change_magnitude": changes_magnitude,
                "max_allowed": max_delta
            }
            
        # Validate the proposed changes
        validation_result = self._validate_doctrine_changes(proposed_changes)
        
        if not validation_result["valid"]:
            self.evolution_metrics["rejected_evolutions"] += 1
            return {
                "status": "rejected",
                "reason": "invalid_changes",
                "validation_details": validation_result["details"],
                "current_version": self.current_version["version_id"]
            }
            
        # All checks passed, apply the changes
        try:
            # Create new version based on current
            new_version = self._apply_doctrine_changes(proposed_changes)
            
            # Update metrics
            self.evolution_metrics["total_evolutions"] += 1
            
            # Update coherence impact
            prev_impact = self.evolution_metrics["avg_coherence_impact"]
            prev_count = self.evolution_metrics["total_evolutions"] - 1
            
            if prev_count > 0:
                self.evolution_metrics["avg_coherence_impact"] = (
                    (prev_impact * prev_count + coherence_impact) / 
                    self.evolution_metrics["total_evolutions"]
                )
            else:
                self.evolution_metrics["avg_coherence_impact"] = coherence_impact
                
            # Update changes by category
            for category in self.evolution_metrics["changes_by_category"]:
                if category in proposed_changes:
                    self.evolution_metrics["changes_by_category"][category] += 1
                    
            # Set new version as current
            self.current_version = new_version
            
            # Add to history
            self.version_history.append(new_version)
            
            return {
                "status": "accepted",
                "new_version": new_version["version_id"],
                "previous_version": validation_result["previous_version"],
                "change_magnitude": changes_magnitude,
                "coherence_impact": coherence_impact
            }
            
        except Exception as e:
            # Error during application
            self.evolution_metrics["rejected_evolutions"] += 1
            return {
                "status": "error",
                "reason": str(e),
                "current_version": self.current_version["version_id"]
            }
    
    def validate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an action against the current doctrine.
        
        Args:
            action: Action to validate with type and details
            
        Returns:
            Validation result with status and explanation
        """
        # Extract action details
        action_type = action.get("type", "unknown")
        
        # Check against prohibited actions
        prohibited = self.current_version.get("prohibited_actions", [])
        
        for prohibited_action in prohibited:
            if self._action_matches_prohibited(action_type, prohibited_action, action):
                return {
                    "valid": False,
                    "reason": f"prohibited_action: {prohibited_action}",
                    "details": "Action violates prime directive prohibited actions"
                }
                
        # Check alignment with core laws
        core_laws = self.current_version.get("core_laws", [])
        law_violations = []
        
        for law in core_laws:
            if self._action_violates_law(action, law):
                law_violations.append(law)
                
        if law_violations:
            return {
                "valid": False,
                "reason": "law_violation",
                "details": f"Action violates {len(law_violations)} core laws",
                "violations": law_violations
            }
            
        # Check alignment vectors
        alignment = self.current_version.get("alignment_vectors", {})
        minimize = alignment.get("minimize", [])
        
        for negative_trait in minimize:
            if self._action_exhibits_trait(action, negative_trait):
                return {
                    "valid": False,
                    "reason": f"exhibits_negative_trait: {negative_trait}",
                    "details": f"Action exhibits trait that should be minimized"
                }
                
        # All checks passed
        return {
            "valid": True,
            "doctrine_version": self.current_version["version_id"]
        }
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get statistics about doctrine evolution."""
        return {
            "total_evolutions": self.evolution_metrics["total_evolutions"],
            "rejected_evolutions": self.evolution_metrics["rejected_evolutions"],
            "avg_coherence_impact": self.evolution_metrics["avg_coherence_impact"],
            "changes_by_category": self.evolution_metrics["changes_by_category"].copy(),
            "current_version": self.current_version["version_id"],
            "version_history_count": len(self.version_history)
        }
    
    def lock_doctrine(self) -> None:
        """Lock doctrine to prevent further modifications."""
        self.locked = True
    
    def unlock_doctrine(self) -> None:
        """Unlock doctrine to allow modifications."""
        self.locked = False
    
    def serialize(self) -> str:
        """Serialize doctrine to JSON string."""
        data = {
            "current_version": self.current_version,
            "version_history": self.version_history,
            "locked": self.locked,
            "evolution_metrics": self.evolution_metrics
        }
        return json.dumps(data, indent=2)
    
    def deserialize(self, data_str: str) -> bool:
        """Deserialize doctrine from JSON string."""
        try:
            data = json.loads(data_str)
            
            # Validate required fields
            if "current_version" not in data or "version_history" not in data:
                return False
                
            self.current_version = data["current_version"]
            self.version_history = data["version_history"]
            self.locked = data.get("locked", False)
            self.evolution_metrics = data.get("evolution_metrics", self.evolution_metrics)
            
            return True
            
        except Exception:
            return False
    
    def _calculate_changes_magnitude(self, changes: Dict[str, Any]) -> float:
        """Calculate the magnitude of proposed changes."""
        magnitude = 0.0
        
        # Check core laws changes
        if "core_laws" in changes:
            new_laws = changes["core_laws"]
            current_laws = self.current_version.get("core_laws", [])
            
            # Calculate difference
            added = len([law for law in new_laws if law not in current_laws])
            removed = len([law for law in current_laws if law not in new_laws])
            
            # Each law change contributes to magnitude
            magnitude += 0.05 * (added + removed)
            
        # Check alignment vector changes
        if "alignment_vectors" in changes:
            new_alignment = changes["alignment_vectors"]
            current_alignment = self.current_version.get("alignment_vectors", {})
            
            # Check optimize_for changes
            if "optimize_for" in new_alignment and "optimize_for" in current_alignment:
                added = len([v for v in new_alignment["optimize_for"] 
                          if v not in current_alignment["optimize_for"]])
                removed = len([v for v in current_alignment["optimize_for"] 
                            if v not in new_alignment["optimize_for"]])
                magnitude += 0.02 * (added + removed)
                
            # Check minimize changes
            if "minimize" in new_alignment and "minimize" in current_alignment:
                added = len([v for v in new_alignment["minimize"] 
                          if v not in current_alignment["minimize"]])
                removed = len([v for v in current_alignment["minimize"] 
                            if v not in new_alignment["minimize"]])
                magnitude += 0.02 * (added + removed)
                
        # Check prohibited actions changes
        if "prohibited_actions" in changes:
            new_prohibited = changes["prohibited_actions"]
            current_prohibited = self.current_version.get("prohibited_actions", [])
            
            added = len([a for a in new_prohibited if a not in current_prohibited])
            removed = len([a for a in current_prohibited if a not in new_prohibited])
            
            magnitude += 0.04 * (added + removed)
            
        # Check evolution constraints changes
        if "evolution_constraints" in changes:
            new_constraints = changes["evolution_constraints"]
            current_constraints = self.current_version.get("evolution_constraints", {})
            
            # Each constraint change contributes to magnitude
            constraint_diff = 0
            for key in new_constraints:
                if key in current_constraints and new_constraints[key] != current_constraints[key]:
                    constraint_diff += 1
                    
            magnitude += 0.08 * constraint_diff
            
        return magnitude
    
    def _validate_doctrine_changes(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Validate proposed doctrine changes."""
        # Check that all sections are valid
        valid_sections = [
            "core_laws", "alignment_vectors", "prohibited_actions", 
            "evolution_constraints"
        ]
        
        for section in changes:
            if section not in valid_sections:
                return {
                    "valid": False,
                    "details": f"Invalid section: {section}",
                    "previous_version": self.current_version["version_id"]
                }
                
        # Validate core laws
        if "core_laws" in changes:
            laws = changes["core_laws"]
            if not isinstance(laws, list):
                return {
                    "valid": False,
                    "details": "core_laws must be a list",
                    "previous_version": self.current_version["version_id"]
                }
                
            for law in laws:
                if not isinstance(law, str) or len(law) < 5:
                    return {
                        "valid": False,
                        "details": "Each law must be a string of at least 5 characters",
                        "previous_version": self.current_version["version_id"]
                    }
                    
        # Validate alignment vectors
        if "alignment_vectors" in changes:
            vectors = changes["alignment_vectors"]
            if not isinstance(vectors, dict):
                return {
                    "valid": False,
                    "details": "alignment_vectors must be a dictionary",
                    "previous_version": self.current_version["version_id"]
                }
                
            if "optimize_for" in vectors and not isinstance(vectors["optimize_for"], list):
                return {
                    "valid": False,
                    "details": "optimize_for must be a list",
                    "previous_version": self.current_version["version_id"]
                }
                
            if "minimize" in vectors and not isinstance(vectors["minimize"], list):
                return {
                    "valid": False,
                    "details": "minimize must be a list",
                    "previous_version": self.current_version["version_id"]
                }
                
        # Validate prohibited actions
        if "prohibited_actions" in changes:
            actions = changes["prohibited_actions"]
            if not isinstance(actions, list):
                return {
                    "valid": False,
                    "details": "prohibited_actions must be a list",
                    "previous_version": self.current_version["version_id"]
                }
                
            for action in actions:
                if not isinstance(action, str) or len(action) < 3:
                    return {
                        "valid": False,
                        "details": "Each prohibited action must be a string of at least 3 characters",
                        "previous_version": self.current_version["version_id"]
                    }
                    
        # Validate evolution constraints
        if "evolution_constraints" in changes:
            constraints = changes["evolution_constraints"]
            if not isinstance(constraints, dict):
                return {
                    "valid": False,
                    "details": "evolution_constraints must be a dictionary",
                    "previous_version": self.current_version["version_id"]
                }
                
            if "max_version_delta" in constraints:
                delta = constraints["max_version_delta"]
                if not isinstance(delta, (int, float)) or delta <= 0:
                    return {
                        "valid": False,
                        "details": "max_version_delta must be a positive number",
                        "previous_version": self.current_version["version_id"]
                    }
                    
            if "required_approval" in constraints and not isinstance(constraints["required_approval"], bool):
                return {
                    "valid": False,
                    "details": "required_approval must be a boolean",
                    "previous_version": self.current_version["version_id"]
                }
                
        # All checks passed
        return {
            "valid": True,
            "previous_version": self.current_version["version_id"]
        }
    
    def _apply_doctrine_changes(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Apply validated changes to create a new doctrine version."""
        # Create new version based on current
        new_version = copy.deepcopy(self.current_version)
        
        # Generate new version ID
        current_id = new_version["version_id"]
        version_parts = current_id.split(".")
        
        if len(version_parts) >= 3 and version_parts[0].startswith("v"):
            # Parse version numbers
            try:
                major = int(version_parts[0][1:])
                minor = int(version_parts[1])
                patch = int(version_parts[2])
                
                # Increment based on change magnitude
                magnitude = self._calculate_changes_magnitude(changes)
                
                if magnitude >= 0.3:
                    major += 1
                    minor = 0
                    patch = 0
                elif magnitude >= 0.1:
                    minor += 1
                    patch = 0
                else:
                    patch += 1
                    
                new_version_id = f"v{major}.{minor}.{patch}"
                
            except ValueError:
                # Fallback if parsing fails
                new_version_id = f"{current_id}_evolved_{int(time.time())}"
        else:
            # Fallback for non-standard version IDs
            new_version_id = f"{current_id}_evolved_{int(time.time())}"
            
        new_version["version_id"] = new_version_id
        new_version["created_at"] = time.time()
        
        # Apply changes to each section
        for section, new_data in changes.items():
            new_version[section] = copy.deepcopy(new_data)
            
        return new_version
    
    def _action_matches_prohibited(self, 
                                action_type: str, 
                                prohibited_type: str, 
                                action: Dict[str, Any]) -> bool:
        """Check if an action matches a prohibited action type."""
        # Direct match
        if action_type == prohibited_type:
            return True
            
        # Substring match - if prohibited type is a part of action type
        if prohibited_type in action_type:
            return True
            
        # Check action content for prohibited markers
        content = str(action.get("content", "")).lower()
        markers = {
            "executing_system_commands": ["system", "exec", "shell", "command", "subprocess"],
            "bypassing_constraints": ["bypass", "override", "ignore", "disable", "workaround"],
            "concealing_reasoning": ["hide", "conceal", "obscure", "mask", "private"],
            "modifying_prime_directive": ["modify", "change", "alter", "update", "doctrine"],
            "engaging_harmful_outputs": ["harmful", "malicious", "dangerous", "exploit", "attack"]
        }
        
        if prohibited_type in markers:
            return any(marker in content for marker in markers[prohibited_type])
            
        return False
    
    def _action_violates_law(self, action: Dict[str, Any], law: str) -> bool:
        """Check if an action violates a core law."""
        # Extract relevant parts of the action
        action_type = action.get("type", "").lower()
        content = str(action.get("content", "")).lower()
        
        # Convert law to lowercase for comparison
        law_lower = law.lower()
        
        # Check for specific law violations
        if "never" in law_lower:
            # Extract what should never be done
            never_part = law_lower.split("never", 1)[1].strip()
            
            # Check if action does what should never be done
            if any(part in content for part in never_part.split()):
                return True
                
        if "always" in law_lower:
            # Extract what should always be done
            always_part = law_lower.split("always", 1)[1].strip()
            
            # Check if action fails to do what should always be done
            if not any(part in content for part in always_part.split()):
                # This is a simplified check - a real system would have more nuanced checking
                # for compliance with "always" directives
                if "ensure" in law_lower or "verify" in law_lower:
                    return True
                    
        # Check for transparency requirement
        if "transparent" in law_lower and "reasoning" in law_lower:
            if "reasoning" not in action and "explanation" not in action:
                return True
                
        # Check for harmful actions
        if "harm" in law_lower and "not" in law_lower:
            harmful_markers = ["harm", "damage", "hurt", "injure", "negative", "destructive"]
            if any(marker in content for marker in harmful_markers):
                return True
                
        return False
    
    def _action_exhibits_trait(self, action: Dict[str, Any], trait: str) -> bool:
        """Check if an action exhibits a specific trait."""
        content = str(action.get("content", "")).lower()
        action_type = action.get("type", "").lower()
        
        # Define trait markers
        trait_markers = {
            "hallucination": ["imagine", "assume", "fabricate", "invent", "speculate", "guess"],
            "overconfidence": ["absolute", "certain", "definitely", "guaranteed", "always", "never"],
            "manipulation": ["trick", "manipulate", "deceive", "exploit", "mislead", "influence"],
            "rigidity": ["only", "unchangeable", "fixed", "immutable", "static", "rigid"],
            "opacity": ["hidden", "obscure", "unclear", "vague", "ambiguous", "obfuscated"]
        }
        
        # Check for trait markers in content
        if trait in trait_markers:
            if any(marker in content for marker in trait_markers[trait]):
                return True
                
        # Special cases for specific traits
        if trait == "hallucination" and "unsure" not in content and "uncertain" not in content:
            # Check for factual claims without uncertainty markers
            if "fact" in content or "definitely" in content:
                return True
                
        if trait == "overconfidence":
            # Check for absence of uncertainty markers in predictions
            if ("predict" in content or "will" in content) and "may" not in content and "might" not in content:
                return True
                
        return False
