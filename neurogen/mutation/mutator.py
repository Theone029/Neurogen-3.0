import uuid
import time
import copy
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class Mutator:
    """
    Strategic mutation engine that modifies plans, outputs, and execution behavior
    based on reward signals and environmental pressure.
    """
    
    def __init__(self, mutation_memory, reward_router, constraint_controller, config: Dict[str, Any]):
        self.mutation_memory = mutation_memory
        self.reward_router = reward_router
        self.constraint_controller = constraint_controller
        self.config = config
        
        # Mutation strategies
        self.mutation_strategies = {
            "plan_restructure": self._plan_restructure_mutation,
            "step_refinement": self._step_refinement_mutation,
            "constraint_adaptation": self._constraint_adaptation_mutation,
            "memory_injection": self._memory_injection_mutation,
            "output_transformation": self._output_transformation_mutation,
            "error_correction": self._error_correction_mutation,
            "concept_substitution": self._concept_substitution_mutation,
            "divergent_exploration": self._divergent_exploration_mutation
        }
        
        # Strategy selection weights (will be adapted over time)
        self.strategy_weights = config.get("strategy_weights", {
            "plan_restructure": 0.15,
            "step_refinement": 0.2,
            "constraint_adaptation": 0.1,
            "memory_injection": 0.15,
            "output_transformation": 0.1,
            "error_correction": 0.15,
            "concept_substitution": 0.1,
            "divergent_exploration": 0.05
        })
        
        # Mutation parameters
        self.params = {
            "min_mutations": config.get("min_mutations", 1),
            "max_mutations": config.get("max_mutations", 3),
            "target_success_rate": config.get("target_success_rate", 0.7),
            "entropy_reduction_weight": config.get("entropy_reduction_weight", 0.4),
            "reward_improvement_weight": config.get("reward_improvement_weight", 0.6)
        }
        
        # Stats tracking
        self.stats = {
            "total_mutations": 0,
            "successful_mutations": 0,
            "strategy_usage": {name: 0 for name in self.mutation_strategies},
            "average_improvements": {name: 0.0 for name in self.mutation_strategies},
            "strategy_success_rates": {name: 0.0 for name in self.mutation_strategies},
            "mutation_chains": {
                "length_1": 0,
                "length_2": 0,
                "length_3+": 0
            }
        }
    
    def mutate(self, 
              plan: Dict[str, Any], 
              error: Dict[str, Any], 
              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a strategic mutation to improve execution outcome.
        
        Args:
            plan: The plan that failed and needs mutation
            error: Error information from validation or execution
            context: Execution context with constraints, memory, and intent
            
        Returns:
            Mutated plan with mutation metadata
        """
        mutation_id = f"mut_{uuid.uuid4().hex[:8]}"
        parent_id = plan.get("mutation_metadata", {}).get("mutation_id", "original")
        
        # Track stats
        self.stats["total_mutations"] += 1
        
        # Record mutation chain length
        if parent_id == "original":
            self.stats["mutation_chains"]["length_1"] += 1
        elif parent_id.startswith("mut_"):
            chain_length = self.mutation_memory.count_chain_mutations(parent_id) + 1
            if chain_length == 2:
                self.stats["mutation_chains"]["length_2"] += 1
            else:
                self.stats["mutation_chains"]["length_3+"] += 1
        
        # Check if we should try to mutate
        should_retry, recommended_strategy = (True, None)
        if self.mutation_memory:
            should_retry, recommended_strategy = self.mutation_memory.should_retry(error, self._hash_plan(plan), parent_id)
        
        if not should_retry:
            # Return failure indicator when mutation isn't advisable
            return {
                "abort_mutation": True,
                "reason": "Mutation cycle detected or excessive chain length",
                "mutation_id": mutation_id,
                "parent_id": parent_id
            }
        
        # Determine mutation scale based on context
        mutation_scale = self._determine_mutation_scale(error, context)
        
        # Select mutation strategy
        strategy_name = self._select_mutation_strategy(plan, error, context, recommended_strategy)
        strategy_fn = self.mutation_strategies[strategy_name]
        
        # Track strategy usage
        self.stats["strategy_usage"][strategy_name] += 1
        
        # Apply the selected mutation strategy
        mutated_plan = strategy_fn(plan, error, context, mutation_scale)
        
        # Add mutation metadata
        mutated_plan["mutation_metadata"] = {
            "mutation_id": mutation_id,
            "parent_id": parent_id,
            "timestamp": time.time(),
            "strategy": strategy_name,
            "scale": mutation_scale,
            "error_type": error.get("type", "unknown"),
            "chain_length": self.mutation_memory.count_chain_mutations(parent_id) + 1 if self.mutation_memory else 1
        }
        
        return mutated_plan
    
    def update_strategy_weights(self, 
                              mutation_id: str, 
                              success: bool, 
                              reward_delta: float) -> None:
        """
        Update strategy weights based on mutation outcomes.
        
        Args:
            mutation_id: ID of the mutation to update
            success: Whether the mutation led to successful execution
            reward_delta: Change in reward from pre-mutation to post-mutation
        """
        # Get mutation metadata
        if not hasattr(self, "recent_mutations"):
            self.recent_mutations = {}
            return
            
        if mutation_id not in self.recent_mutations:
            return
            
        mutation_data = self.recent_mutations[mutation_id]
        strategy = mutation_data.get("strategy")
        
        if not strategy or strategy not in self.strategy_weights:
            return
            
        # Update success rate statistics
        current_success = self.stats["strategy_success_rates"][strategy]
        current_uses = self.stats["strategy_usage"][strategy]
        
        # New success rate is weighted average of old rate and new outcome
        new_success_rate = ((current_success * current_uses) + (1 if success else 0)) / (current_uses + 1)
        self.stats["strategy_success_rates"][strategy] = new_success_rate
        
        # Update average improvement statistics
        current_improvement = self.stats["average_improvements"][strategy]
        new_improvement = ((current_improvement * current_uses) + reward_delta) / (current_uses + 1)
        self.stats["average_improvements"][strategy] = new_improvement
        
        # Adjust weights based on success and reward improvement
        learning_rate = self.config.get("strategy_learning_rate", 0.05)
        
        # Increase weight for successful and high-reward strategies
        if success and reward_delta > 0:
            adjustment = learning_rate * (1 + reward_delta)
            self.strategy_weights[strategy] *= (1 + adjustment)
        
        # Decrease weight for unsuccessful strategies
        elif not success:
            adjustment = learning_rate * (1 + abs(reward_delta))
            self.strategy_weights[strategy] *= (1 - adjustment)
        
        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        for s in self.strategy_weights:
            self.strategy_weights[s] /= total_weight
            
        # Track successful mutations
        if success:
            self.stats["successful_mutations"] += 1
    
    def _determine_mutation_scale(self, 
                                error: Dict[str, Any], 
                                context: Dict[str, Any]) -> float:
        """Determine appropriate mutation scale based on error and context."""
        # Default scale
        base_scale = 0.3
        
        # Get scale from constraint controller if available
        if self.constraint_controller:
            base_scale = self.constraint_controller.get_mutation_scale(context)
        
        # Adjust based on error severity
        error_severity = self._calculate_error_severity(error)
        severity_factor = 0.5 + (error_severity * 0.5)  # 0.5 to 1.0 based on severity
        
        # Adjust based on mutation pressure from reward router
        pressure_factor = 1.0
        if self.reward_router:
            pressure = self.reward_router.get_mutation_pressure(context)
            pressure_factor = 0.7 + (pressure * 0.6)  # 0.7 to 1.3 based on pressure
        
        # Calculate adjusted scale
        adjusted_scale = base_scale * severity_factor * pressure_factor
        
        # Cap within reasonable bounds
        return max(0.1, min(1.0, adjusted_scale))
    
    def _calculate_error_severity(self, error: Dict[str, Any]) -> float:
        """Calculate severity of error for scaling mutation."""
        severity = 0.5  # Default moderate severity
        
        # Check error type
        error_type = error.get("type", "")
        if error_type in ["critical", "fatal", "security"]:
            severity = 0.9
        elif error_type in ["validation", "structure"]:
            severity = 0.7
        elif error_type in ["warning", "minor"]:
            severity = 0.3
            
        # Check for specific error patterns indicating severe issues
        error_msg = str(error.get("message", "")).lower()
        critical_terms = ["crash", "invalid state", "security breach", "violated constraint"]
        if any(term in error_msg for term in critical_terms):
            severity = max(severity, 0.8)
            
        # Check if this is a repeated error (more severe)
        if error.get("repeat_count", 0) > 0:
            severity = min(1.0, severity + 0.1 * error.get("repeat_count", 0))
            
        return severity
    
    def _select_mutation_strategy(self, 
                               plan: Dict[str, Any], 
                               error: Dict[str, Any], 
                               context: Dict[str, Any],
                               recommended_strategy: Optional[str] = None) -> str:
        """Select appropriate mutation strategy based on error and context."""
        # If a strategy was recommended by mutation memory, use it
        if recommended_strategy and recommended_strategy in self.mutation_strategies:
            return recommended_strategy
            
        # Get normalized weights
        weights = self.strategy_weights.copy()
        
        # Adjust weights based on error type
        error_type = error.get("type", "")
        
        if error_type == "validation":
            # Validation errors benefit from output transformation
            weights["output_transformation"] *= 1.5
            weights["error_correction"] *= 1.3
            
        elif error_type == "execution":
            # Execution errors benefit from plan restructuring
            weights["plan_restructure"] *= 1.5
            weights["step_refinement"] *= 1.3
            
        elif error_type == "doctrine":
            # Doctrine errors benefit from concept substitution
            weights["concept_substitution"] *= 1.5
            weights["constraint_adaptation"] *= 1.3
            
        # Adjust for specific error patterns
        error_msg = str(error.get("message", "")).lower()
        
        if "missing" in error_msg or "not found" in error_msg:
            weights["memory_injection"] *= 1.5
            
        if "inconsistent" in error_msg or "contradiction" in error_msg:
            weights["concept_substitution"] *= 1.4
            
        if "complex" in error_msg or "insufficient" in error_msg:
            weights["plan_restructure"] *= 1.3
            
        # Adjust for contextual factors
        mutation_count = context.get("mutation_count", 0)
        
        # For multiple failed mutations, try more divergent approaches
        if mutation_count >= 2:
            weights["divergent_exploration"] *= (1.0 + 0.5 * mutation_count)
            
        # Normalize weights
        total = sum(weights.values())
        normalized_weights = {k: v/total for k, v in weights.items()}
        
        # Select strategy probabilistically based on weights
        strategies = list(normalized_weights.keys())
        probabilities = list(normalized_weights.values())
        
        return random.choices(strategies, weights=probabilities, k=1)[0]
    
    def _plan_restructure_mutation(self, 
                                plan: Dict[str, Any], 
                                error: Dict[str, Any],
                                context: Dict[str, Any],
                                scale: float) -> Dict[str, Any]:
        """
        Restructure the overall plan architecture by adding, removing, or
        reordering steps.
        """
        # Create a copy of the plan to mutate
        mutated_plan = copy.deepcopy(plan)
        
        # Extract steps for modification
        steps = mutated_plan.get("steps", [])
        if not steps:
            # No steps to restructure
            return mutated_plan
            
        # Determine number of mutations based on scale
        num_mutations = 1 + int(scale * 2)  # 1-3 mutations depending on scale
        
        for _ in range(num_mutations):
            # Choose a restructuring operation based on scale
            operations = ["reorder", "add", "remove", "split", "merge"]
            operation_weights = [0.3, 0.3, 0.2, 0.1, 0.1]
            
            # For higher scales, prefer more significant changes
            if scale > 0.6:
                operation_weights = [0.2, 0.3, 0.2, 0.15, 0.15]
                
            operation = random.choices(operations, weights=operation_weights, k=1)[0]
            
            if operation == "reorder" and len(steps) >= 2:
                # Reorder steps while preserving dependencies
                valid_reorder = False
                for _ in range(5):  # Try up to 5 times to find valid reordering
                    # Select two positions to swap
                    i, j = random.sample(range(len(steps)), 2)
                    
                    # Check if reordering would break dependencies
                    if self._is_valid_reordering(steps, i, j):
                        # Swap the steps
                        steps[i], steps[j] = steps[j], steps[i]
                        valid_reorder = True
                        break
                        
                if not valid_reorder:
                    # If no valid reordering found, try another operation
                    continue
                    
            elif operation == "add":
                # Add a new step
                # First, determine a good insertion point
                dependencies = self._extract_dependencies(steps)
                insertion_points = self._find_insertion_points(dependencies)
                
                if insertion_points:
                    insert_at = random.choice(insertion_points)
                    
                    # Create a new step
                    new_step = self._generate_new_step(steps, context, insert_at)
                    
                    # Insert the new step
                    steps.insert(insert_at, new_step)
                    
            elif operation == "remove" and len(steps) > 2:
                # Remove a step, but ensure critical steps remain
                removable_steps = [i for i, step in enumerate(steps)
                                 if not self._is_critical_step(step)]
                
                if removable_steps:
                    remove_idx = random.choice(removable_steps)
                    removed_step = steps.pop(remove_idx)
                    
                    # Update dependencies in remaining steps
                    self._update_dependencies_after_removal(steps, removed_step)
                    
            elif operation == "split" and len(steps) > 0:
                # Split a complex step into multiple simpler steps
                splittable_steps = [i for i, step in enumerate(steps)
                                  if self._is_splittable_step(step)]
                
                if splittable_steps:
                    split_idx = random.choice(splittable_steps)
                    split_step = steps[split_idx]
                    
                    # Generate replacement steps
                    replacement_steps = self._split_step(split_step)
                    
                    # Replace the original step with the split steps
                    steps.pop(split_idx)
                    for i, new_step in enumerate(replacement_steps):
                        steps.insert(split_idx + i, new_step)
                        
            elif operation == "merge" and len(steps) >= 2:
                # Merge consecutive related steps
                mergeable_pairs = self._find_mergeable_steps(steps)
                
                if mergeable_pairs:
                    merge_idx = random.choice(mergeable_pairs)
                    
                    # Merge the steps
                    merged_step = self._merge_steps(steps[merge_idx], steps[merge_idx+1])
                    
                    # Replace the original steps with the merged step
                    steps.pop(merge_idx)
                    steps.pop(merge_idx)  # Index shifts after first pop
                    steps.insert(merge_idx, merged_step)
        
        # Update the plan with modified steps
        mutated_plan["steps"] = steps
        
        # Update step IDs if needed
        self._ensure_unique_step_ids(mutated_plan)
        
        return mutated_plan
    
    def _step_refinement_mutation(self, 
                               plan: Dict[str, Any], 
                               error: Dict[str, Any],
                               context: Dict[str, Any],
                               scale: float) -> Dict[str, Any]:
        """
        Refine individual steps while preserving overall plan structure.
        """
        # Create a copy of the plan to mutate
        mutated_plan = copy.deepcopy(plan)
        
        # Extract steps
        steps = mutated_plan.get("steps", [])
        if not steps:
            return mutated_plan
            
        # Identify problematic steps based on error
        problematic_steps = self._identify_problematic_steps(steps, error)
        
        # If no specific problematic steps identified, use some heuristics
        if not problematic_steps and steps:
            # Target steps with certain characteristics
            candidates = []
            for i, step in enumerate(steps):
                score = 0
                
                # Complex steps are good candidates
                if len(str(step)) > 200:
                    score += 2
                    
                # Steps with many inputs are good candidates
                if len(step.get("inputs", [])) > 2:
                    score += 1
                    
                # Last steps are often responsible for errors
                if i == len(steps) - 1:
                    score += 2
                    
                candidates.append((i, score))
                
            # Select steps with higher scores
            candidates.sort(key=lambda x: x[1], reverse=True)
            problematic_steps = [idx for idx, _ in candidates[:max(1, int(len(steps) * 0.3))]]
        
        # Number of steps to refine based on scale
        num_refinements = max(1, int(len(problematic_steps) * scale))
        
        # Select steps to refine
        steps_to_refine = random.sample(problematic_steps, min(num_refinements, len(problematic_steps)))
        
        for step_idx in steps_to_refine:
            step = steps[step_idx]
            
            # Choose refinement operations based on step properties
            operations = []
            
            # Add potential operations based on step properties
            if "action" in step:
                operations.append("modify_action")
                
            if "inputs" in step and step["inputs"]:
                operations.append("adjust_inputs")
                
            if "type" in step:
                operations.append("change_type")
                
            if not operations:
                operations = ["general_refinement"]
                
            # Select operation
            operation = random.choice(operations)
            
            if operation == "modify_action":
                # Modify the action description
                current_action = step.get("action", "")
                
                # Different modifications based on scale
                if scale < 0.4:
                    # Minor clarification
                    step["action"] = self._clarify_action(current_action)
                elif scale < 0.7:
                    # Moderate change
                    step["action"] = self._enhance_action(current_action)
                else:
                    # Significant change
                    step["action"] = self._transform_action(current_action, context)
                    
            elif operation == "adjust_inputs":
                # Modify the step inputs
                current_inputs = step.get("inputs", [])
                available_steps = [s.get("id") for s in steps[:step_idx] if "id" in s]
                
                if scale < 0.5 and current_inputs and available_steps:
                    # Replace one input with another available step
                    if current_inputs and available_steps:
                        replace_idx = random.randrange(len(current_inputs))
                        new_input = random.choice(available_steps)
                        current_inputs[replace_idx] = new_input
                else:
                    # More significant changes
                    if random.random() < 0.5 and available_steps:
                        # Add a new input
                        new_input = random.choice(available_steps)
                        if new_input not in current_inputs:
                            current_inputs.append(new_input)
                    else:
                        # Remove an input
                        if len(current_inputs) > 1:
                            remove_idx = random.randrange(len(current_inputs))
                            current_inputs.pop(remove_idx)
                
                step["inputs"] = current_inputs
                
            elif operation == "change_type":
                # Change the step type
                current_type = step.get("type", "")
                
                # Get compatible types
                compatible_types = self._get_compatible_step_types(current_type)
                
                if compatible_types:
                    # Select a new type
                    new_type = random.choice(compatible_types)
                    step["type"] = new_type
                    
                    # Update action to match new type if needed
                    if "action" in step:
                        step["action"] = self._adapt_action_to_type(step["action"], new_type)
                    
            elif operation == "general_refinement":
                # General improvements to step
                for key, value in step.items():
                    if isinstance(value, str) and len(value) > 10:
                        if random.random() < scale:
                            step[key] = self._refine_text(value, scale)
        
        # Update the plan with refined steps
        mutated_plan["steps"] = steps
        
        return mutated_plan
    
    def _constraint_adaptation_mutation(self, 
                                     plan: Dict[str, Any], 
                                     error: Dict[str, Any],
                                     context: Dict[str, Any],
                                     scale: float) -> Dict[str, Any]:
        """
        Adapt plans to better work within system constraints.
        """
        # Create a copy of the plan to mutate
        mutated_plan = copy.deepcopy(plan)
        
        # Check if plan has constraints section
        if "constraints" not in mutated_plan:
            mutated_plan["constraints"] = {}
            
        # Get current constraints from context
        context_constraints = context.get("constraints", {})
        
        # Extract error patterns related to constraints
        constraint_issues = self._extract_constraint_issues(error)
        
        # Update plan constraints based on context and errors
        plan_constraints = mutated_plan["constraints"]
        
        # Strategies for constraint adaptation
        strategies = ["relax", "enforce", "rebalance"]
        strategy = random.choices(strategies, weights=[0.4, 0.3, 0.3], k=1)[0]
        
        if strategy == "relax" and constraint_issues:
            # Relax constraints that are causing issues
            for constraint in constraint_issues:
                if constraint in context_constraints:
                    current_value = context_constraints[constraint]
                    
                    # Modify the constraint value based on type
                    if isinstance(current_value, (int, float)):
                        # For numeric constraints, relax by increasing or decreasing
                        direction = self._determine_constraint_direction(constraint)
                        if direction > 0:
                            # Higher is more relaxed
                            new_value = current_value * (1 + scale * 0.3)
                        else:
                            # Lower is more relaxed
                            new_value = current_value * (1 - scale * 0.3)
                            
                        # Update constraint
                        plan_constraints[constraint] = new_value
                        
                    elif isinstance(current_value, bool):
                        # For boolean constraints, may flip if highly problematic
                        if scale > 0.7:
                            plan_constraints[constraint] = not current_value
        
        elif strategy == "enforce":
            # Add or tighten constraints to prevent error recurrence
            # Identify key constraints that should be enforced
            key_constraints = self._identify_key_constraints(error)
            
            for constraint in key_constraints:
                if constraint in context_constraints:
                    current_value = context_constraints[constraint]
                    
                    # Modify constraint based on type
                    if isinstance(current_value, (int, float)):
                        # Tighten by moving in the constraining direction
                        direction = self._determine_constraint_direction(constraint)
                        if direction > 0:
                            # Higher is more constrained
                            new_value = current_value * (1 + scale * 0.2)
                        else:
                            # Lower is more constrained
                            new_value = current_value * (1 - scale * 0.2)
                            
                        # Update constraint
                        plan_constraints[constraint] = new_value
                    
                    elif isinstance(current_value, bool):
                        # For boolean constraints, set to constraining value
                        if constraint.startswith("allow_") or constraint.startswith("enable_"):
                            plan_constraints[constraint] = False
                        elif constraint.startswith("require_") or constraint.startswith("force_"):
                            plan_constraints[constraint] = True
        
        elif strategy == "rebalance":
            # Rebalance constraints to optimize for the current situation
            # Identify constraints that should be rebalanced
            rebalance_candidates = self._identify_rebalance_candidates(context_constraints, error)
            
            # Sample a subset of constraints to rebalance
            num_to_rebalance = max(1, int(len(rebalance_candidates) * scale))
            selected = random.sample(rebalance_candidates, min(num_to_rebalance, len(rebalance_candidates)))
            
            for constraint in selected:
                current_value = context_constraints.get(constraint)
                
                if isinstance(current_value, (int, float)):
                    # Adjust by a random factor within range determined by scale
                    factor = 1 + random.uniform(-0.2, 0.2) * scale
                    new_value = current_value * factor
                    
                    # Update constraint
                    plan_constraints[constraint] = new_value
        
        # If plan has a settings section, adapt it based on constraints
        if "settings" in mutated_plan:
            settings = mutated_plan["settings"]
            self._adapt_settings_to_constraints(settings, plan_constraints, scale)
            
        # Potentially add constraint justification
        if plan_constraints and scale > 0.5:
            mutated_plan["constraint_justification"] = self._generate_constraint_justification(plan_constraints)
        
        return mutated_plan
    
    def _memory_injection_mutation(self, 
                                plan: Dict[str, Any], 
                                error: Dict[str, Any],
                                context: Dict[str, Any],
                                scale: float) -> Dict[str, Any]:
        """
        Inject relevant memories and knowledge into the plan.
        """
        # Create a copy of the plan to mutate
        mutated_plan = copy.deepcopy(plan)
        
        # Extract available memories from context
        available_memories = context.get("memories", [])
        
        if not available_memories:
            # No memories to inject
            return mutated_plan
            
        # Analyze the plan to identify memory injection points
        memory_injection_points = self._identify_memory_injection_points(mutated_plan, error)
        
        if not memory_injection_points:
            # No suitable injection points found
            return mutated_plan
            
        # Score memories for relevance to the current error
        scored_memories = self._score_memories_for_injection(available_memories, error, mutated_plan)
        
        # Number of memories to inject based on scale
        num_injections = 1 + int(scale * 2)  # 1-3 injections
        
        # Track injected memories
        injected_memory_ids = []
        
        # Perform injections
        injections_performed = 0
        
        for _ in range(num_injections):
            if not memory_injection_points or not scored_memories:
                break
                
            # Select an injection point
            injection_point = random.choice(memory_injection_points)
            memory_injection_points.remove(injection_point)
            
            # Select a memory to inject
            if not scored_memories:
                continue
                
            memory_idx = 0
            if len(scored_memories) > 1:
                # Use weighted selection based on scores
                weights = [score for _, score in scored_memories]
                memory_idx = random.choices(range(len(scored_memories)), weights=weights, k=1)[0]
                
            memory, _ = scored_memories.pop(memory_idx)
            
            # Inject the memory
            success = self._inject_memory(mutated_plan, injection_point, memory, scale)
            
            if success:
                injections_performed += 1
                injected_memory_ids.append(memory.get("id", "unknown"))
        
        # Add memory injection metadata if injections were performed
        if injections_performed > 0:
            if "memory_metadata" not in mutated_plan:
                mutated_plan["memory_metadata"] = {}
                
            mutated_plan["memory_metadata"]["injected_memories"] = injected_memory_ids
            mutated_plan["memory_metadata"]["injection_count"] = injections_performed
        
        return mutated_plan
    
    def _output_transformation_mutation(self, 
                                      plan: Dict[str, Any], 
                                      error: Dict[str, Any],
                                      context: Dict[str, Any],
                                      scale: float) -> Dict[str, Any]:
        """
        Transform plan's output generation to address validation issues.
        """
        # Create a copy of the plan to mutate
        mutated_plan = copy.deepcopy(plan)
        
        # Check if the error is related to output validation
        is_output_error = self._is_output_validation_error(error)
        
        # Extract output related information
        output_type = plan.get("expected_output_type", context.get("task", {}).get("output_type", "text"))
        output_format = plan.get("output_format", {})
        
        # Initialize or extract output transformation section
        if "output_transformation" not in mutated_plan:
            mutated_plan["output_transformation"] = {}
            
        output_transformation = mutated_plan["output_transformation"]
        
        # Determine transformation strategy based on error and output type
        strategies = ["restructure", "filter", "enrich", "simplify", "validate"]
        
        # Weight strategies based on error and output type
        strategy_weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Default equal weights
        
        if is_output_error:
            # For output validation errors, prioritize validation and restructuring
            strategy_weights = [0.3, 0.1, 0.2, 0.1, 0.3]
            
        # Adjust based on output type
        if output_type == "json":
            strategy_weights[0] += 0.1  # Increase restructure weight
            strategy_weights[4] += 0.1  # Increase validate weight
        elif output_type == "text":
            strategy_weights[2] += 0.1  # Increase enrich weight
            strategy_weights[3] += 0.1  # Increase simplify weight
        elif output_type == "code":
            strategy_weights[1] += 0.1  # Increase filter weight
            strategy_weights[4] += 0.1  # Increase validate weight
            
        # Select strategies based on scale
        num_strategies = 1 + int(scale * 2)  # 1-3 strategies
        selected_strategies = []
        
        for _ in range(num_strategies):
            if not strategies:
                break
                
            strategy_idx = random.choices(range(len(strategies)), weights=strategy_weights, k=1)[0]
            selected_strategies.append(strategies.pop(strategy_idx))
            strategy_weights.pop(strategy_idx)
            
        # Apply selected transformation strategies
        for strategy in selected_strategies:
            if strategy == "restructure":
                # Restructure output format
                output_transformation["restructure"] = self._generate_output_restructure(
                    output_type, output_format, error, scale)
                    
            elif strategy == "filter":
                # Add content filtering
                output_transformation["filter"] = self._generate_output_filter(
                    output_type, error, scale)
                    
            elif strategy == "enrich":
                # Add content enrichment
                output_transformation["enrich"] = self._generate_output_enrichment(
                    output_type, context, scale)
                    
            elif strategy == "simplify":
                # Add simplification rules
                output_transformation["simplify"] = self._generate_output_simplification(
                    output_type, error, scale)
                    
            elif strategy == "validate":
                # Add validation rules
                output_transformation["validate"] = self._generate_output_validation(
                    output_type, error, scale)
        
        # Update output format if needed
        if "restructure" in output_transformation and scale > 0.5:
            mutated_plan["output_format"] = self._update_output_format(
                output_format, output_transformation["restructure"])
        
        return mutated_plan
    
    def _error_correction_mutation(self, 
                                plan: Dict[str, Any], 
                                error: Dict[str, Any],
                                context: Dict[str, Any],
                                scale: float) -> Dict[str, Any]:
        """
        Apply targeted fixes to directly address specific error patterns.
        """
        # Create a copy of the plan to mutate
        mutated_plan = copy.deepcopy(plan)
        
        # Extract error details
        error_type = error.get("type", "unknown")
        error_message = str(error.get("message", ""))
        error_details = error.get("details", {})
        
        # Track mutations made
        corrections = []
        
        # Handle different error types
        if error_type == "validation":
            # Fix validation errors
            if "structure" in error_message.lower() or "schema" in error_message.lower():
                # Structure validation errors
                corrections.extend(self._fix_structure_validation(mutated_plan, error_details, scale))
                
            elif "type" in error_message.lower():
                # Type validation errors
                corrections.extend(self._fix_type_validation(mutated_plan, error_details, scale))
                
            elif "constraint" in error_message.lower():
                # Constraint validation errors
                corrections.extend(self._fix_constraint_validation(mutated_plan, error_details, scale))
                
        elif error_type == "execution":
            # Fix execution errors
            if "missing" in error_message.lower() or "not found" in error_message.lower():
                # Missing resource/step errors
                corrections.extend(self._fix_missing_resource(mutated_plan, error_details, scale))
                
            elif "timeout" in error_message.lower():
                # Timeout errors
                corrections.extend(self._fix_timeout(mutated_plan, error_details, scale))
                
            elif "dependency" in error_message.lower():
                # Dependency errors
                corrections.extend(self._fix_dependency(mutated_plan, error_details, scale))
                
        elif error_type == "doctrine":
            # Fix doctrine violations
            if "prohibited" in error_message.lower() or "forbidden" in error_message.lower():
                # Prohibited content
                corrections.extend(self._fix_prohibited_content(mutated_plan, error_details, scale))
                
            elif "required" in error_message.lower():
                # Missing required elements
                corrections.extend(self._fix_required_element(mutated_plan, error_details, scale))
                
            elif "alignment" in error_message.lower():
                # Alignment issues
                corrections.extend(self._fix_alignment(mutated_plan, error_details, scale))
                
        # Apply general error corrections if no specific fixes were found
        if not corrections:
            corrections.extend(self._apply_general_fixes(mutated_plan, error, scale))
            
        # Add error correction metadata
        if corrections:
            if "correction_metadata" not in mutated_plan:
                mutated_plan["correction_metadata"] = {}
                
            mutated_plan["correction_metadata"]["corrections"] = corrections
            mutated_plan["correction_metadata"]["error_type"] = error_type
            
        return mutated_plan
    
    def _concept_substitution_mutation(self, 
                                    plan: Dict[str, Any], 
                                    error: Dict[str, Any],
                                    context: Dict[str, Any],
                                    scale: float) -> Dict[str, Any]:
        """
        Replace concepts and approaches with doctrine-aligned alternatives.
        """
        # Create a copy of the plan to mutate
        mutated_plan = copy.deepcopy(plan)
        
        # Extract doctrine from context
        doctrine = context.get("doctrine", {})
        
        # Identify problematic concepts
        problematic_concepts = self._identify_problematic_concepts(plan, error, doctrine)
        
        if not problematic_concepts:
            # No specific concepts identified, use general approach
            return self._general_concept_substitution(mutated_plan, doctrine, scale)
            
        # Number of concepts to substitute based on scale
        num_substitutions = max(1, int(len(problematic_concepts) * scale))
        
        # Select concepts to substitute
        concepts_to_substitute = random.sample(
            problematic_concepts, 
            min(num_substitutions, len(problematic_concepts))
        )
        
        # Perform substitutions
        substitutions = []
        
        for concept in concepts_to_substitute:
            # Find suitable replacement
            replacement = self._find_concept_replacement(concept, doctrine)
            
            if replacement:
                # Apply substitution throughout the plan
                count = self._substitute_concept(mutated_plan, concept, replacement)
                
                if count > 0:
                    substitutions.append({
                        "original": concept,
                        "replacement": replacement,
                        "count": count
                    })
        
        # Add substitution metadata
        if substitutions:
            if "substitution_metadata" not in mutated_plan:
                mutated_plan["substitution_metadata"] = {}
                
            mutated_plan["substitution_metadata"]["substitutions"] = substitutions
            
        return mutated_plan
    
    def _divergent_exploration_mutation(self, 
                                     plan: Dict[str, Any], 
                                     error: Dict[str, Any],
                                     context: Dict[str, Any],
                                     scale: float) -> Dict[str, Any]:
        """
        Generate creative alternative approaches when conventional mutations fail.
        """
        # Create a copy of the plan to mutate
        mutated_plan = copy.deepcopy(plan)
        
        # For divergent exploration, make more substantial changes
        # The scale factor determines how radical the changes will be
        
        if scale < 0.3:
            # Mild exploration - combine two strategies
            strategies = ["plan_restructure", "step_refinement", "memory_injection"]
            selected = random.sample(strategies, 2)
            
            for strategy in selected:
                if strategy == "plan_restructure":
                    mutated_plan = self._plan_restructure_mutation(
                        mutated_plan, error, context, scale * 1.2)
                elif strategy == "step_refinement":
                    mutated_plan = self._step_refinement_mutation(
                        mutated_plan, error, context, scale * 1.2)
                elif strategy == "memory_injection":
                    mutated_plan = self._memory_injection_mutation(
                        mutated_plan, error, context, scale * 1.2)
                    
        elif scale < 0.7:
            # Moderate exploration - try alternative problem framing
            # Change the core approach while preserving the goal
            
            # Extract goal and core elements
            goal = mutated_plan.get("goal", "Complete the task")
            
            # Generate alternative problem framing
            alt_framing = self._generate_alternative_framing(goal, context, error)
            
            # Apply the alternative framing
            if "framing" not in mutated_plan:
                mutated_plan["framing"] = {}
                
            mutated_plan["framing"]["alternative"] = alt_framing
            
            # Generate new steps based on alternative framing
            new_steps = self._generate_steps_from_framing(alt_framing, context)
            if new_steps:
                mutated_plan["steps"] = new_steps
            
            # Apply step refinement to new steps
            mutated_plan = self._step_refinement_mutation(
                mutated_plan, error, context, 0.5)
                
        else:
            # Radical exploration - fundamentally different approach
            # This might involve:
            # 1. Inverting the problem
            # 2. Using completely different techniques
            # 3. Restructuring the entire solution approach
            
            # Start with a drastically different plan structure
            if random.random() < 0.7:
                # Option 1: Invert the problem-solving approach
                inverted_approach = self._generate_inverted_approach(mutated_plan, context)
                
                # Apply the inverted approach
                for key, value in inverted_approach.items():
                    mutated_plan[key] = value
                    
            else:
                # Option 2: Cross-domain approach from a different problem domain
                cross_domain = self._generate_cross_domain_approach(mutated_plan, context)
                
                # Apply the cross-domain approach
                for key, value in cross_domain.items():
                    mutated_plan[key] = value
            
        # Add exploration metadata
        if "exploration_metadata" not in mutated_plan:
            mutated_plan["exploration_metadata"] = {}
            
        mutated_plan["exploration_metadata"]["exploration_level"] = "mild" if scale < 0.3 else "moderate" if scale < 0.7 else "radical"
        mutated_plan["exploration_metadata"]["divergence_factor"] = scale
        
        return mutated_plan
    
    # Helper methods for mutation strategies
    
    def _hash_plan(self, plan: Dict[str, Any]) -> str:
        """Create a hash of the plan for identification."""
        # Simple implementation - in a real system, use a more robust approach
        plan_str = str(sorted([f"{k}:{v}" for k, v in plan.items() if k != "mutation_metadata"]))
        
        import hashlib
        return hashlib.md5(plan_str.encode()).hexdigest()
    
    def _extract_dependencies(self, steps: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract step dependencies from a plan."""
        dependencies = {}
        
        for step in steps:
            step_id = step.get("id")
            if not step_id:
                continue
                
            inputs = step.get("inputs", [])
            dependencies[step_id] = inputs
            
        return dependencies
    
    def _is_valid_reordering(self, steps: List[Dict[str, Any]], i: int, j: int) -> bool:
        """Check if reordering steps i and j would preserve dependencies."""
        # Extract step IDs
        step_ids = [step.get("id") for step in steps]
        
        # Extract dependencies
        dependencies = {}
        for idx, step in enumerate(steps):
            step_id = step.get("id")
            if not step_id:
                continue
                
            dependencies[step_id] = step.get("inputs", [])
            
        # Create a dependency graph
        graph = {step_id: set() for step_id in step_ids if step_id}
        for step_id, inputs in dependencies.items():
            for input_id in inputs:
                if input_id in graph:
                    graph[input_id].add(step_id)
                    
        # Check if reordering would create a cycle
        if i > j:
            i, j = j, i  # Ensure i < j
            
        step_i_id = step_ids[i]
        step_j_id = step_ids[j]
        
        if not step_i_id or not step_j_id:
            return False
            
        # Check if step j depends on step i
        if step_i_id in dependencies.get(step_j_id, []):
            return False
            
        # Check if step i depends on step j
        if step_j_id in dependencies.get(step_i_id, []):
            return False
            
        return True
    
    def _find_insertion_points(self, dependencies: Dict[str, List[str]]) -> List[int]:
        """Find valid insertion points for new steps."""
        # Simple implementation - in a real system, use a more sophisticated approach
        # Return indices between existing steps
        return list(range(len(dependencies) + 1))
    
    def _generate_new_step(self, steps: List[Dict[str, Any]], context: Dict[str, Any], insert_at: int) -> Dict[str, Any]:
        """Generate a new step for insertion."""
        # Generate a unique step ID
        existing_ids = [step.get("id") for step in steps if "id" in step]
        step_id = f"new_step_{len(existing_ids) + 1}"
        
        while step_id in existing_ids:
            step_id = f"new_step_{len(existing_ids) + 2}"
            
        # Determine available inputs (steps before the insertion point)
        available_inputs = [step.get("id") for step in steps[:insert_at] if "id" in step]
        
        # Determine step type based on context and surrounding steps
        step_types = ["analysis", "processing", "transformation", "validation", "integration"]
        
        if insert_at == 0:
            # First step is usually analysis
            step_type = "analysis"
        elif insert_at == len(steps):
            # Last step is usually validation or integration
            step_type = random.choice(["validation", "integration"])
        else:
            # Middle steps can be any type
            step_type = random.choice(step_types)
            
        # Generate inputs
        num_inputs = min(len(available_inputs), 2)
        inputs = random.sample(available_inputs, num_inputs) if available_inputs and num_inputs > 0 else []
        
        # Generate action based on type
        action = self._generate_action_for_type(step_type, context)
        
        # Create the new step
        new_step = {
            "id": step_id,
            "type": step_type,
            "action": action
        }
        
        if inputs:
            new_step["inputs"] = inputs
            
        return new_step
    
    def _generate_action_for_type(self, step_type: str, context: Dict[str, Any]) -> str:
        """Generate an action description for a step type."""
        task_type = context.get("task", {}).get("type", "unknown")
        
        # Generate action based on step type and task type
        if step_type == "analysis":
            return f"Analyze {task_type} requirements and extract key elements"
        elif step_type == "processing":
            return f"Process input data for {task_type} operation"
        elif step_type == "transformation":
            return f"Transform intermediate results into structured format"
        elif step_type == "validation":
            return f"Validate results against {task_type} requirements"
        elif step_type == "integration":
            return f"Integrate components into final {task_type} solution"
        else:
            return f"Perform {step_type} operation for {task_type} task"
    
    def _is_critical_step(self, step: Dict[str, Any]) -> bool:
        """Determine if a step is critical and shouldn't be removed."""
        # Critical step types
        critical_types = ["analysis", "execution", "integration", "validation"]
        
        # Check step type
        if "type" in step and step["type"] in critical_types:
            return True
            
        # Check step action for critical keywords
        critical_keywords = ["critical", "essential", "required", "necessary", "key"]
        if "action" in step:
            action = step["action"].lower()
            if any(keyword in action for keyword in critical_keywords):
                return True
                
        return False
    
    def _update_dependencies_after_removal(self, steps: List[Dict[str, Any]], removed_step: Dict[str, Any]) -> None:
        """Update dependencies after removing a step."""
        removed_id = removed_step.get("id")
        if not removed_id:
            return
            
        # Find steps that depended on the removed step
        for step in steps:
            if "inputs" in step:
                inputs = step["inputs"]
                if removed_id in inputs:
                    # Remove the dependency
                    inputs.remove(removed_id)
                    
                    # If the removed step had inputs, add those as replacements
                    if "inputs" in removed_step:
                        for input_id in removed_step["inputs"]:
                            if input_id not in inputs:
                                inputs.append(input_id)
    
    def _is_splittable_step(self, step: Dict[str, Any]) -> bool:
        """Determine if a step can be split into multiple steps."""
        # Check if the step has a complex action
        if "action" in step:
            action = step["action"]
            
            # Check for compound actions with multiple parts
            if " and " in action or ";" in action or "," in action:
                return True
                
            # Check for long actions
            if len(action) > 100:
                return True
                
        return False
    
    def _split_step(self, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a complex step into multiple simpler steps."""
        action = step.get("action", "")
        step_id = step.get("id", "unknown")
        step_type = step.get("type", "unknown")
        inputs = step.get("inputs", [])
        
        # Split action into parts
        parts = []
        
        if " and " in action:
            parts = action.split(" and ")
        elif ";" in action:
            parts = action.split(";")
        elif "," in action:
            parts = action.split(",")
        else:
            # Split long action roughly in half
            midpoint = len(action) // 2
            # Find a good split point near the midpoint
            split_point = action.find(". ", midpoint)
            if split_point != -1:
                parts = [action[:split_point+1], action[split_point+2:]]
            else:
                parts = [action]
        
        # Create new steps
        new_steps = []
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
                
            new_id = f"{step_id}_{i+1}"
            
            # Determine step type based on content
            new_type = step_type
            if "analysis" in part.lower():
                new_type = "analysis"
            elif "process" in part.lower() or "transform" in part.lower():
                new_type = "processing"
            elif "valid" in part.lower() or "check" in part.lower():
                new_type = "validation"
                
            # Create new step
            new_step = {
                "id": new_id,
                "type": new_type,
                "action": part
            }
            
            # Set inputs - first step uses original inputs
            if i == 0:
                if inputs:
                    new_step["inputs"] = inputs
            else:
                # Later steps depend on previous step
                prev_id = f"{step_id}_{i}"
                new_step["inputs"] = [prev_id]
                
            new_steps.append(new_step)
            
        return new_steps
    
    def _find_mergeable_steps(self, steps: List[Dict[str, Any]]) -> List[int]:
        """Find pairs of consecutive steps that can be merged."""
        mergeable = []
        
        for i in range(len(steps) - 1):
            step1 = steps[i]
            step2 = steps[i+1]
            
            # Check if steps can be merged
            if self._can_merge_steps(step1, step2):
                mergeable.append(i)
                
        return mergeable
    
    def _can_merge_steps(self, step1: Dict[str, Any], step2: Dict[str, Any]) -> bool:
        """Determine if two steps can be merged."""
        # Check if step2 only depends on step1
        step1_id = step1.get("id")
        step2_inputs = step2.get("inputs", [])
        
        if step1_id and step2_inputs == [step1_id]:
            return True
            
        # Check if steps have the same type
        if "type" in step1 and "type" in step2 and step1["type"] == step2["type"]:
            return True
            
        return False
    
    def _merge_steps(self, step1: Dict[str, Any], step2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two steps into a single step."""
        # Create new merged step
        merged = {
            "id": step1.get("id", "merged"),
            "type": step1.get("type", step2.get("type", "merged"))
        }
        
        # Merge actions
        action1 = step1.get("action", "")
        action2 = step2.get("action", "")
        
        if action1 and action2:
            merged["action"] = f"{action1} and {action2}"
        elif action1:
            merged["action"] = action1
        elif action2:
            merged["action"] = action2
            
        # Merge inputs (excluding dependencies between the merged steps)
        inputs = step1.get("inputs", []).copy()
        for input_id in step2.get("inputs", []):
            if input_id != step1.get("id") and input_id not in inputs:
                inputs.append(input_id)
                
        if inputs:
            merged["inputs"] = inputs
            
        return merged
    
    def _ensure_unique_step_ids(self, plan: Dict[str, Any]) -> None:
        """Ensure all steps have unique IDs."""
        steps = plan.get("steps", [])
        
        # Find and fix duplicate IDs
        seen_ids = set()
        for i, step in enumerate(steps):
            if "id" in step:
                step_id = step["id"]
                
                if step_id in seen_ids:
                    # Generate a new unique ID
                    new_id = f"{step_id}_{i}"
                    
                    # Update this step's ID
                    step["id"] = new_id
                    
                    # Update references to this step
                    self._update_step_references(steps, step_id, new_id)
                    
                seen_ids.add(step["id"])
            else:
                # Generate an ID if missing
                step["id"] = f"step_{i}"
                seen_ids.add(step["id"])
    
    def _update_step_references(self, steps: List[Dict[str, Any]], old_id: str, new_id: str) -> None:
        """Update references to a step ID in other steps."""
        for step in steps:
            if "inputs" in step:
                inputs = step["inputs"]
                if old_id in inputs:
                    # Replace old ID with new ID
                    inputs[inputs.index(old_id)] = new_id
    
    def _identify_problematic_steps(self, steps: List[Dict[str, Any]], error: Dict[str, Any]) -> List[int]:
        """Identify steps that are likely causing the error."""
        problematic_indices = []
        error_msg = str(error.get("message", "")).lower()
        
        # Check for specific step mentions in error
        for i, step in enumerate(steps):
            step_id = step.get("id", "")
            
            if step_id and step_id in error_msg:
                problematic_indices.append(i)
                
        # Check for error in final step
        if "final" in error_msg or "output" in error_msg:
            problematic_indices.append(len(steps) - 1)
            
        # Check for specific step types mentioned in error
        for i, step in enumerate(steps):
            step_type = step.get("type", "")
            
            if step_type and step_type in error_msg:
                problematic_indices.append(i)
                
        # Deduplicate
        return list(set(problematic_indices))
    
    def _clarify_action(self, action: str) -> str:
        """Make minor clarifications to an action description."""
        if not action:
            return action
            
        # Add clarity qualifiers
        clarifications = [
            "clearly ", "explicitly ", "specifically ", "precisely ",
            "thoroughly ", "carefully ", "methodically "
        ]
        
        # Find a good insertion point
        words = action.split()
        if len(words) >= 2:
            # Insert after first verb
            insert_at = 1
            
            # Create clarified action
            words.insert(insert_at, random.choice(clarifications))
            return " ".join(words)
        
        return action
    
    def _enhance_action(self, action: str) -> str:
        """Make moderate enhancements to an action description."""
        if not action:
            return action
            
        # Add descriptive elements or objectives
        enhancements = [
            " while ensuring consistency",
            " with focus on key elements",
            " to improve clarity and structure",
            " using systematic approach",
            " to identify critical patterns",
            " with validation at each step"
        ]
        
        return action + random.choice(enhancements)
    
    def _transform_action(self, action: str, context: Dict[str, Any]) -> str:
        """Significantly transform an action description."""
        if not action:
            return action
            
        # Extract key verbs and objects
        words = action.lower().split()
        verbs = ["analyze", "process", "transform", "validate", "generate", "extract", "implement"]
        
        # Find main verb or use default
        main_verb = next((word for word in words if word in verbs), "process")
        
        # Get transformer verbs based on context
        task_type = context.get("task", {}).get("type", "unknown")
        
        if task_type == "creative":
            transformer_verbs = ["design", "create", "synthesize", "generate", "compose"]
        elif task_type == "analytical":
            transformer_verbs = ["analyze", "dissect", "evaluate", "investigate", "examine"]
        elif task_type == "processing":
            transformer_verbs = ["transform", "convert", "process", "restructure", "organize"]
        else:
            transformer_verbs = ["develop", "implement", "construct", "establish", "formulate"]
            
        # Replace main verb
        new_verb = random.choice(transformer_verbs)
        
        # Create transformed action
        if main_verb in words:
            for i, word in enumerate(words):
                if word == main_verb:
                    words[i] = new_verb
                    break
                    
            transformed = " ".join(words)
            
            # Add transformative element
            transformations = [
                " using advanced pattern recognition",
                " through multi-stage optimization",
                " with structured decomposition technique",
                " following systematic validation protocol",
                " using coherence-maximizing approach"
            ]
            
            return transformed + random.choice(transformations)
        
        return action
    
    def _get_compatible_step_types(self, current_type: str) -> List[str]:
        """Get compatible step types for type mutation."""
        type_compatibility = {
            "analysis": ["examination", "investigation", "assessment"],
            "processing": ["transformation", "manipulation", "conversion"],
            "validation": ["verification", "evaluation", "testing"],
            "generation": ["creation", "synthesis", "production"],
            "integration": ["combination", "incorporation", "merging"]
        }
        
        # If current type is in compatibility map, return compatible types
        if current_type in type_compatibility:
            return type_compatibility[current_type]
            
        # For unknown types, return general types
        return ["analysis", "processing", "validation", "integration"]
    
    def _adapt_action_to_type(self, action: str, new_type: str) -> str:
        """Adapt an action description to match a new step type."""
        if not action:
            return action
