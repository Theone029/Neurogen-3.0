import random
import uuid
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np

class Mutator:
    """Guided self-mutation engine that evolves the system through controlled variation."""
    
    def __init__(self, mutation_memory, reward_router, constraint_controller, config: Dict[str, Any]):
        self.mutation_memory = mutation_memory
        self.reward_router = reward_router
        self.constraint_controller = constraint_controller
        
        self.config = config
        self.max_retries = config.get("max_retries", 3)
        self.mutation_strategies = self._initialize_mutation_strategies()
        self.mutation_stats = {
            "total_attempts": 0,
            "successful_mutations": 0,
            "retry_exhaustion": 0,
            "strategy_usage": {s: 0 for s in self.mutation_strategies}
        }
    
    def _initialize_mutation_strategies(self) -> Dict[str, Callable]:
        """Initialize mutation strategy functions."""
        return {
            "parameter_adjustment": self._mutate_parameters,
            "structure_modification": self._mutate_structure,
            "component_replacement": self._mutate_component,
            "context_augmentation": self._mutate_context,
            "constraint_relaxation": self._mutate_constraints
        }
    
    def mutate(self, 
              failed_output: Any, 
              error_trace: Dict[str, Any], 
              plan: Dict[str, Any],
              context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a mutation based on failure analysis."""
        self.mutation_stats["total_attempts"] += 1
        
        # Generate mutation ID
        mutation_id = f"mut_{uuid.uuid4().hex[:8]}"
        parent_id = plan.get("mutation_id", "original")
        
        # Check mutation history to avoid repeated failures
        if self.mutation_memory:
            similar_failures = self.mutation_memory.find_similar_failures(
                error=error_trace, 
                plan_hash=self._hash_plan(plan)
            )
            
            if similar_failures:
                # Use failure history to guide mutation
                strategy_weights = self._calculate_strategy_weights(similar_failures)
            else:
                # No history, use default weights
                strategy_weights = {s: 1.0 for s in self.mutation_strategies}
        else:
            strategy_weights = {s: 1.0 for s in self.mutation_strategies}
        
        # Get current mutation scale from constraint controller
        if self.constraint_controller:
            mutation_scale = self.constraint_controller.get_mutation_scale(context)
        else:
            mutation_scale = self.config.get("default_mutation_scale", 0.3)
        
        # Choose mutation strategy based on weights
        total_weight = sum(strategy_weights.values())
        normalized_weights = {s: w/total_weight for s, w in strategy_weights.items()}
        
        strategies = list(normalized_weights.keys())
        weights = [normalized_weights[s] for s in strategies]
        
        # Select strategy with probability proportional to weight
        selected_strategy = random.choices(strategies, weights=weights, k=1)[0]
        self.mutation_stats["strategy_usage"][selected_strategy] += 1
        
        # Apply the selected mutation strategy
        mutation_fn = self.mutation_strategies[selected_strategy]
        mutated_plan = mutation_fn(
            plan=plan, 
            error=error_trace, 
            scale=mutation_scale,
            context=context
        )
        
        # Enrich with mutation metadata
        mutated_plan["mutation_metadata"] = {
            "mutation_id": mutation_id,
            "parent_id": parent_id,
            "strategy": selected_strategy,
            "scale": mutation_scale,
            "timestamp": time.time(),
            "error_type": error_trace.get("type", "unknown"),
            "attempt": self._count_mutation_chain(parent_id) + 1
        }
        
        # If using reward router, apply mutation pressure
        if self.reward_router:
            pressure = self.reward_router.get_mutation_pressure(context)
            mutated_plan["mutation_metadata"]["pressure"] = pressure
        
        return mutated_plan
    
    def _mutate_parameters(self, plan: Dict[str, Any], error: Dict[str, Any], 
                         scale: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate numerical parameters within the plan."""
        mutated = self._deep_copy(plan)
        
        # Find numerical parameters in the plan
        params = self._extract_parameters(mutated)
        
        # Choose parameters to mutate based on scale
        num_to_mutate = max(1, int(len(params) * scale))
        params_to_mutate = random.sample(params, min(num_to_mutate, len(params)))
        
        # Apply mutations to selected parameters
        for param_path in params_to_mutate:
            current_value = self._get_nested_value(mutated, param_path)
            
            # Skip if not a number
            if not isinstance(current_value, (int, float)):
                continue
                
            # Determine mutation range based on value
            if isinstance(current_value, int):
                # For integers, modify by at least 1
                delta = max(1, int(abs(current_value) * scale * random.uniform(0.1, 0.5)))
                new_value = current_value + random.choice([-1, 1]) * delta
            else:
                # For floats, apply percentage change
                delta = abs(current_value) * scale * random.uniform(0.1, 0.5)
                new_value = current_value + random.choice([-1, 1]) * delta
                
            # Set the new value
            self._set_nested_value(mutated, param_path, new_value)
            
        return mutated
    
    def _mutate_structure(self, plan: Dict[str, Any], error: Dict[str, Any], 
                        scale: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """Modify the structure of the plan (add/remove/reorder components)."""
        mutated = self._deep_copy(plan)
        
        # Find list structures in the plan
        lists = self._extract_lists(mutated)
        
        # Choose lists to mutate based on scale
        num_to_mutate = max(1, int(len(lists) * scale))
        lists_to_mutate = random.sample(lists, min(num_to_mutate, len(lists)))
        
        # Apply structure mutations to selected lists
        for list_path in lists_to_mutate:
            current_list = self._get_nested_value(mutated, list_path)
            
            # Skip if not a list or empty
            if not isinstance(current_list, list) or not current_list:
                continue
                
            # Choose a mutation operation
            operation = random.choice(["reorder", "remove", "duplicate"])
            
            if operation == "reorder" and len(current_list) > 1:
                # Shuffle elements
                new_list = current_list.copy()
                random.shuffle(new_list)
                self._set_nested_value(mutated, list_path, new_list)
                
            elif operation == "remove" and len(current_list) > 1:
                # Remove a random element
                idx = random.randrange(len(current_list))
                new_list = current_list.copy()
                new_list.pop(idx)
                self._set_nested_value(mutated, list_path, new_list)
                
            elif operation == "duplicate" and current_list:
                # Duplicate a random element
                idx = random.randrange(len(current_list))
                new_list = current_list.copy()
                new_list.append(current_list[idx])
                self._set_nested_value(mutated, list_path, new_list)
                
        return mutated
    
    def _mutate_component(self, plan: Dict[str, Any], error: Dict[str, Any], 
                        scale: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """Replace a component of the plan with an alternative."""
        mutated = self._deep_copy(plan)
        
        # This is highly domain-specific and depends on plan structure
        # Implementation would need knowledge of components and alternatives
        
        # For demonstration, we'll implement a simplified version
        # that replaces string values that match error keywords
        
        if "message" in error:
            error_keywords = self._extract_keywords(error["message"])
            if error_keywords:
                # Find string values in plan that match error keywords
                strings = self._extract_strings(mutated)
                
                # Filter to those containing error keywords
                matches = []
                for path in strings:
                    value = self._get_nested_value(mutated, path)
                    if any(kw.lower() in value.lower() for kw in error_keywords):
                        matches.append(path)
                
                # Choose strings to replace
                num_to_replace = max(1, int(len(matches) * scale))
                to_replace = random.sample(matches, min(num_to_replace, len(matches)))
                
                # Replace with alternatives (simplified example)
                for path in to_replace:
                    current = self._get_nested_value(mutated, path)
                    
                    # Simple replacement - in reality would use more intelligent alternatives
                    for kw in error_keywords:
                        if kw.lower() in current.lower():
                            # Replace keyword with alternative
                            alternatives = self._get_alternatives_for_keyword(kw)
                            if alternatives:
                                alt = random.choice(alternatives)
                                new_value = current.replace(kw, alt)
                                self._set_nested_value(mutated, path, new_value)
        
        return mutated
    
    def _mutate_context(self, plan: Dict[str, Any], error: Dict[str, Any], 
                       scale: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the plan with additional context information."""
        mutated = self._deep_copy(plan)
        
        # Check if context augmentation is needed
        if "context" not in mutated:
            mutated["context"] = {}
            
        # Identify missing context that might help
        missing_context = self._identify_missing_context(mutated, error)
        
        # Add missing context elements
        for key, value in missing_context.items():
            if key not in mutated["context"]:
                mutated["context"][key] = value
                
        return mutated
    
    def _mutate_constraints(self, plan: Dict[str, Any], error: Dict[str, Any], 
                          scale: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """Modify constraints within the plan to allow more flexibility."""
        mutated = self._deep_copy(plan)
        
        # Check if constraints exist
        if "constraints" not in mutated:
            return mutated  # Nothing to mutate
            
        constraints = mutated["constraints"]
        
        # Identify constraints that might be causing the error
        if "type" in error and "message" in error:
            error_type = error["type"]
            error_msg = error["message"]
            
            # Look for constraints that might match the error
            for key in list(constraints.keys()):
                should_relax = False
                
                # Check if constraint keyword appears in error
                if key.lower() in error_msg.lower():
                    should_relax = True
                    
                # For validation errors, relax related constraints
                if error_type == "validation" and key in ["validate", "verify", "check"]:
                    should_relax = True
                    
                # If we should relax this constraint
                if should_relax:
                    if isinstance(constraints[key], bool) and constraints[key]:
                        # Disable boolean constraint
                        mutated["constraints"][key] = False
                    elif isinstance(constraints[key], (int, float)) and constraints[key] > 0:
                        # Decrease numerical constraint by scale factor
                        mutated["constraints"][key] *= (1.0 - scale)
                    elif isinstance(constraints[key], list) and constraints[key]:
                        # Remove a random constraint from the list
                        idx = random.randrange(len(constraints[key]))
                        mutated["constraints"][key].pop(idx)
        
        return mutated
    
    def _calculate_strategy_weights(self, similar_failures: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate strategy weights based on past failure patterns."""
        # Default weights
        weights = {s: 1.0 for s in self.mutation_strategies}
        
        # Count successes for each strategy
        strategy_success = {s: 0 for s in self.mutation_strategies}
        strategy_attempts = {s: 0 for s in self.mutation_strategies}
        
        for failure in similar_failures:
            if "mutations" in failure:
                for mutation in failure["mutations"]:
                    strategy = mutation.get("strategy")
                    success = mutation.get("success", False)
                    
                    if strategy in self.mutation_strategies:
                        strategy_attempts[strategy] += 1
                        if success:
                            strategy_success[strategy] += 1
        
        # Calculate success rates and adjust weights
        for strategy in self.mutation_strategies:
            attempts = strategy_attempts[strategy]
            if attempts > 0:
                success_rate = strategy_success[strategy] / attempts
                # Boost successful strategies
                weights[strategy] = 1.0 + success_rate * 2.0
            
        return weights
    
    def _extract_parameters(self, plan: Dict[str, Any]) -> List[List[str]]:
        """Extract paths to all numerical parameters in the plan."""
        params = []
        self._find_parameters(plan, [], params)
        return params
    
    def _find_parameters(self, obj: Any, path: List[str], result: List[List[str]]) -> None:
        """Recursively find all numerical parameters in a nested structure."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (int, float)):
                    result.append(path + [key])
                elif isinstance(value, (dict, list)):
                    self._find_parameters(value, path + [key], result)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                if isinstance(value, (int, float)):
                    result.append(path + [str(i)])
                elif isinstance(value, (dict, list)):
                    self._find_parameters(value, path + [str(i)], result)
    
    def _extract_lists(self, plan: Dict[str, Any]) -> List[List[str]]:
        """Extract paths to all lists in the plan."""
        lists = []
        self._find_lists(plan, [], lists)
        return lists
    
    def _find_lists(self, obj: Any, path: List[str], result: List[List[str]]) -> None:
        """Recursively find all lists in a nested structure."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, list):
                    result.append(path + [key])
                if isinstance(value, (dict, list)):
                    self._find_lists(value, path + [key], result)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                if isinstance(value, (dict, list)):
                    self._find_lists(value, path + [str(i)], result)
    
    def _extract_strings(self, plan: Dict[str, Any]) -> List[List[str]]:
        """Extract paths to all string values in the plan."""
        strings = []
        self._find_strings(plan, [], strings)
        return strings
    
    def _find_strings(self, obj: Any, path: List[str], result: List[List[str]]) -> None:
        """Recursively find all strings in a nested structure."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    result.append(path + [key])
                elif isinstance(value, (dict, list)):
                    self._find_strings(value, path + [key], result)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                if isinstance(value, str):
                    result.append(path + [str(i)])
                elif isinstance(value, (dict, list)):
                    self._find_strings(value, path + [str(i)], result)
    
    def _get_nested_value(self, obj: Any, path: List[str]) -> Any:
        """Get a value from a nested structure using a path."""
        for key in path:
            if isinstance(obj, dict):
                if key not in obj:
                    return None
                obj = obj[key]
            elif isinstance(obj, list):
                idx = int(key)
                if idx >= len(obj):
                    return None
                obj = obj[idx]
            else:
                return None
        return obj
    
    def _set_nested_value(self, obj: Any, path: List[str], value: Any) -> None:
        """Set a value in a nested structure using a path."""
        for i, key in enumerate(path[:-1]):
            if isinstance(obj, dict):
                if key not in obj:
                    obj[key] = {} if isinstance(path[i+1], str) else []
                obj = obj[key]
            elif isinstance(obj, list):
                idx = int(key)
                if idx >= len(obj):
                    # Extend list if needed
                    obj.extend([None] * (idx - len(obj) + 1))
                if obj[idx] is None:
                    obj[idx] = {} if isinstance(path[i+1], str) else []
                obj = obj[idx]
        
        # Set the final value
        last_key = path[-1]
        if isinstance(obj, dict):
            obj[last_key] = value
        elif isinstance(obj, list):
            idx = int(last_key)
            if idx >= len(obj):
                obj.extend([None] * (idx - len(obj) + 1))
            obj[idx] = value
    
    def _hash_plan(self, plan: Dict[str, Any]) -> str:
        """Create a hash of the plan structure for comparison."""
        # This is a simplified implementation
        import hashlib
        import json
        
        # Remove mutation metadata for comparison
        plan_copy = self._deep_copy(plan)
        if "mutation_metadata" in plan_copy:
            del plan_copy["mutation_metadata"]
            
        plan_str = json.dumps(plan_copy, sort_keys=True)
        return hashlib.md5(plan_str.encode()).hexdigest()
    
    def _extract_keywords(self, error_message: str) -> List[str]:
        """Extract potential keywords from an error message."""
        # Very simplified implementation
        # In a real system, use NLP techniques for better extraction
        words = error_message.split()
        return [w for w in words if len(w) > 4 and w.isalpha()]
    
    def _identify_missing_context(self, plan: Dict[str, Any], error: Dict[str, Any]) -> Dict[str, Any]:
        """Identify context information that might be missing based on the error."""
        # This is highly domain-specific
        # Simplified implementation that looks for keywords in error
        missing = {}
        
        if "message" in error:
            msg = error["message"].lower()
            
            # Check for common missing context patterns
            if "permission" in msg or "access" in msg:
                missing["permissions"] = ["read", "write"]
                
            if "timeout" in msg or "too long" in msg:
                missing["timeout"] = 120  # seconds
                
            if "format" in msg or "syntax" in msg:
                missing["required_format"] = "json"
                
            if "missing" in msg and "parameter" in msg:
                # Try to extract parameter name
                words = msg.split()
                if "parameter" in words:
                    idx = words.index("parameter")
                    if idx + 1 < len(words):
                        param = words[idx + 1].strip("'\".,;:")
                        missing["required_parameters"] = [param]
        
        return missing
    
    def _get_alternatives_for_keyword(self, keyword: str) -> List[str]:
        """Get alternative terms for a keyword."""
        # This would be better implemented with a synonym dictionary or word embedding model
        # Simple example implementation
        alternatives = {
            "error": ["failure", "exception", "issue", "problem"],
            "create": ["generate", "make", "build", "construct"],
            "delete": ["remove", "erase", "eliminate", "destroy"],
            "update": ["modify", "change", "alter", "revise"],
            "query": ["search", "find", "lookup", "retrieve"],
            "parameter": ["argument", "variable", "value", "input"],
            "function": ["method", "procedure", "routine", "operation"],
            "validate": ["verify", "check", "confirm", "test"]
        }
        
        # Convert to lowercase for comparison
        kw_lower = keyword.lower()
        
        # Return alternatives if found
        for k, alts in alternatives.items():
            if k == kw_lower:
                return alts
        
        # Return empty list if no alternatives found
        return []
    
    def _count_mutation_chain(self, parent_id: str) -> int:
        """Count how many mutations have already been attempted in this chain."""
        if self.mutation_memory:
            return self.mutation_memory.count_chain_mutations(parent_id)
        return 0
    
    def _deep_copy(self, obj: Any) -> Any:
        """Create a deep copy of an object."""
        import copy
        return copy.deepcopy(obj)
