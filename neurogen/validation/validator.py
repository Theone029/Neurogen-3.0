import traceback
import hashlib
import json
from typing import Dict, Any, Tuple, List, Optional, Callable
from collections import defaultdict

class Validator:
    """Execution and logic verification system that validates NEUROGEN outputs."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validators = {}
        self.static_rules = {}
        self.validation_stats = defaultdict(lambda: {"pass": 0, "fail": 0, "errors": defaultdict(int)})
        
    def register_validator(self, output_type: str, validator_fn: Callable) -> None:
        """Register a validation function for a specific output type."""
        self.validators[output_type] = validator_fn
        
    def register_static_rule(self, rule_name: str, rule_fn: Callable) -> None:
        """Register a static validation rule."""
        self.static_rules[rule_name] = rule_fn
        
    def validate(self, output: Any, output_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an output based on its type and context."""
        validation_id = hashlib.md5(f"{output_type}:{time.time()}".encode()).hexdigest()[:12]
        
        # Initialize validation result
        result = {
            "valid": False,
            "validation_id": validation_id,
            "output_type": output_type,
            "checks": [],
            "details": {},
            "error": None
        }
        
        try:
            # Structure validation
            structure_check = self._validate_structure(output, output_type)
            result["checks"].append({"type": "structure", "passed": structure_check[0]})
            
            if not structure_check[0]:
                result["details"]["structure_error"] = structure_check[1]
                result["error"] = f"Structure validation failed: {structure_check[1]}"
                self.validation_stats[output_type]["fail"] += 1
                self.validation_stats[output_type]["errors"]["structure"] += 1
                return result
            
            # Type-specific validation
            if output_type in self.validators:
                try:
                    validator_fn = self.validators[output_type]
                    validation_result = validator_fn(output, context)
                    
                    result["checks"].append({"type": "type_specific", "passed": validation_result[0]})
                    
                    if not validation_result[0]:
                        result["details"]["validation_error"] = validation_result[1]
                        result["error"] = f"Type validation failed: {validation_result[1]}"
                        self.validation_stats[output_type]["fail"] += 1
                        self.validation_stats[output_type]["errors"]["type_specific"] += 1
                        return result
                except Exception as e:
                    result["checks"].append({"type": "type_specific", "passed": False, "error": str(e)})
                    result["details"]["validator_exception"] = traceback.format_exc()
                    result["error"] = f"Validator exception: {str(e)}"
                    self.validation_stats[output_type]["fail"] += 1
                    self.validation_stats[output_type]["errors"]["validator_exception"] += 1
                    return result
            
            # Static rule validation
            for rule_name, rule_fn in self.static_rules.items():
                try:
                    rule_result = rule_fn(output, context)
                    result["checks"].append({
                        "type": "static_rule", 
                        "rule": rule_name, 
                        "passed": rule_result[0]
                    })
                    
                    if not rule_result[0]:
                        result["details"][f"rule_{rule_name}"] = rule_result[1]
                        result["error"] = f"Rule '{rule_name}' failed: {rule_result[1]}"
                        self.validation_stats[output_type]["fail"] += 1
                        self.validation_stats[output_type]["errors"][f"rule_{rule_name}"] += 1
                        return result
                except Exception as e:
                    result["checks"].append({
                        "type": "static_rule", 
                        "rule": rule_name, 
                        "passed": False, 
                        "error": str(e)
                    })
                    result["details"][f"rule_{rule_name}_exception"] = traceback.format_exc()
                    result["error"] = f"Rule exception in '{rule_name}': {str(e)}"
                    self.validation_stats[output_type]["fail"] += 1
                    self.validation_stats[output_type]["errors"][f"rule_{rule_name}_exception"] += 1
                    return result
            
            # All validations passed
            result["valid"] = True
            self.validation_stats[output_type]["pass"] += 1
            return result
            
        except Exception as e:
            result["error"] = f"Validation system error: {str(e)}"
            result["details"]["system_exception"] = traceback.format_exc()
            self.validation_stats[output_type]["fail"] += 1
            self.validation_stats[output_type]["errors"]["system"] += 1
            return result
    
    def _validate_structure(self, output: Any, output_type: str) -> Tuple[bool, Optional[str]]:
        """Validate the basic structure of an output."""
        # Check if output is None
        if output is None:
            return False, "Output is None"
            
        # Type-specific structure validation
        if output_type == "json":
            if not isinstance(output, (dict, list)):
                return False, f"Expected dict or list, got {type(output).__name__}"
                
        elif output_type == "text":
            if not isinstance(output, str):
                return False, f"Expected string, got {type(output).__name__}"
                
        elif output_type == "plan":
            if not isinstance(output, dict):
                return False, f"Expected dict, got {type(output).__name__}"
            
            # Check required plan keys
            required_keys = ["steps", "goal"]
            missing = [k for k in required_keys if k not in output]
            if missing:
                return False, f"Plan missing required keys: {', '.join(missing)}"
                
            # Validate steps structure
            if not isinstance(output.get("steps"), list):
                return False, "Plan steps must be a list"
                
        elif output_type == "vector":
            # Check if output can be interpreted as a vector
            try:
                import numpy as np
                np.array(output)
            except:
                return False, "Cannot convert to numpy array"
                
        # If we get here, structure is valid
        return True, None
