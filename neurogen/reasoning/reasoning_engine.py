import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Callable
import json
import numpy as np

class ReasoningEngine:
    """Strategic reasoning system for generating, evaluating, and refining solutions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Reasoning strategies
        self.strategies = {
            "deductive": self._deductive_reasoning,
            "inductive": self._inductive_reasoning,
            "abductive": self._abductive_reasoning,
            "analogical": self._analogical_reasoning,
            "counterfactual": self._counterfactual_reasoning,
            "causal": self._causal_reasoning,
            "multi_step": self._multi_step_reasoning
        }
        
        # Strategy selection weights
        self.strategy_weights = config.get("strategy_weights", {
            "deductive": 0.3,
            "inductive": 0.2,
            "abductive": 0.15,
            "analogical": 0.15,
            "counterfactual": 0.1,
            "causal": 0.1,
            "multi_step": 0.0  # Dynamically activated when needed
        })
        
        # Reasoning trace storage
        self.reasoning_traces = {}
        
        # Adaptive parameters
        self.adaptive_params = {
            "entropy_threshold": config.get("entropy_threshold", 0.6),
            "confidence_threshold": config.get("confidence_threshold", 0.7),
            "max_reasoning_steps": config.get("max_reasoning_steps", 5),
            "abduction_penalty": config.get("abduction_penalty", 0.3)
        }
        
        # Stats tracking
        self.stats = {
            "strategies_used": {s: 0 for s in self.strategies},
            "avg_steps": 0,
            "avg_confidence": 0,
            "total_reasonings": 0
        }
    
    def reason(self, 
              problem: Dict[str, Any], 
              memories: List[Dict[str, Any]], 
              constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform strategic reasoning on a problem.
        
        Args:
            problem: Problem definition and context
            memories: Relevant memories to inform reasoning
            constraints: Constraints on the reasoning process
            
        Returns:
            Reasoning results with traces and justification
        """
        # Generate reasoning ID
        reasoning_id = f"reason_{uuid.uuid4().hex[:10]}"
        start_time = time.time()
        self.stats["total_reasonings"] += 1
        
        # Extract problem elements
        problem_type = problem.get("type", "general")
        goal = problem.get("goal", "Solve the problem")
        context = problem.get("context", {})
        inputs = problem.get("inputs", {})
        
        # Initialize reasoning trace
        trace = []
        
        # Determine appropriate reasoning strategy
        strategy_name, confidence = self._select_reasoning_strategy(problem, memories, constraints)
        self.stats["strategies_used"][strategy_name] += 1
        
        trace.append({
            "step": 0,
            "operation": "strategy_selection",
            "strategy": strategy_name,
            "confidence": confidence
        })
        
        # Get the strategy function
        strategy_fn = self.strategies.get(strategy_name, self._deductive_reasoning)
        
        # Apply the reasoning strategy
        reasoning_result = strategy_fn(
            problem=problem,
            memories=memories, 
            constraints=constraints,
            trace=trace
        )
        
        # Get reasoning metadata
        steps = len(trace) - 1  # Subtract the strategy selection step
        
        # Final result structure
        result = {
            "reasoning_id": reasoning_id,
            "strategy": strategy_name,
            "confidence": reasoning_result.get("confidence", 0.0),
            "solution": reasoning_result.get("solution"),
            "justification": reasoning_result.get("justification", ""),
            "trace": trace,
            "metadata": {
                "steps": steps,
                "time_taken": time.time() - start_time,
                "problem_type": problem_type,
                "entropy": reasoning_result.get("entropy", 0.5)
            }
        }
        
        # Store reasoning trace
        self.reasoning_traces[reasoning_id] = result
        
        # Update stats
        self.stats["avg_steps"] = ((self.stats["avg_steps"] * (self.stats["total_reasonings"] - 1) + 
                                  steps) / self.stats["total_reasonings"])
        
        self.stats["avg_confidence"] = ((self.stats["avg_confidence"] * (self.stats["total_reasonings"] - 1) + 
                                      reasoning_result.get("confidence", 0.0)) / self.stats["total_reasonings"])
        
        return result
    
    def _select_reasoning_strategy(self, 
                                 problem: Dict[str, Any], 
                                 memories: List[Dict[str, Any]],
                                 constraints: Dict[str, Any]) -> Tuple[str, float]:
        """Select best reasoning strategy for the problem."""
        problem_type = problem.get("type", "general")
        
        # Check for explicit strategy request
        if "reasoning_strategy" in problem:
            requested = problem["reasoning_strategy"]
            if requested in self.strategies:
                return requested, 1.0
                
        # Get base weights from config
        weights = self.strategy_weights.copy()
        
        # Apply heuristic adjustment based on problem type
        if problem_type == "logical_deduction":
            weights["deductive"] *= 2.0
            weights["inductive"] *= 0.5
            
        elif problem_type == "pattern_recognition":
            weights["inductive"] *= 2.0
            weights["analogical"] *= 1.5
            
        elif problem_type == "root_cause_analysis":
            weights["abductive"] *= 2.0
            weights["causal"] *= 1.8
            
        elif problem_type == "counterfactual_analysis":
            weights["counterfactual"] *= 2.5
            
        elif problem_type == "complex_problem":
            weights["multi_step"] = 1.0  # Activate multi-step reasoning
            
        # Adjust based on available memories
        if memories:
            # More memories favor inductive and analogical reasoning
            memory_factor = min(1.5, len(memories) / 5)
            weights["inductive"] *= memory_factor
            weights["analogical"] *= memory_factor
        else:
            # Fewer memories favor deductive reasoning
            weights["deductive"] *= 1.5
            
        # Adjust based on constraints
        if "reasoning_depth" in constraints:
            depth = constraints["reasoning_depth"]
            if depth >= 3:
                weights["multi_step"] = max(weights.get("multi_step", 0), 0.8)
                
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Get strategy pairs (strategy, weight)
        strategy_pairs = list(normalized_weights.items())
        
        # Sort by weight in descending order
        strategy_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top strategy and confidence
        return strategy_pairs[0][0], strategy_pairs[0][1]
    
    def _deductive_reasoning(self, 
                           problem: Dict[str, Any], 
                           memories: List[Dict[str, Any]],
                           constraints: Dict[str, Any],
                           trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform deductive reasoning (premises to conclusion)."""
        # Extract premises and rules from problem
        premises = problem.get("premises", [])
        rules = problem.get("rules", [])
        
        # If no explicit premises, extract from problem context
        if not premises and "context" in problem:
            premises = self._extract_premises(problem["context"])
            
        # If no explicit rules, try to extract from memories
        if not rules and memories:
            rules = self._extract_rules_from_memories(memories)
            
        # Add extracted information to trace
        trace.append({
            "step": len(trace),
            "operation": "extract_premises",
            "premises": premises
        })
        
        trace.append({
            "step": len(trace),
            "operation": "extract_rules",
            "rules": rules
        })
        
        # Apply rules to premises
        conclusions = []
        confidence = 1.0
        
        for rule in rules:
            # Parse rule
            if isinstance(rule, dict) and "if" in rule and "then" in rule:
                condition = rule["if"]
                conclusion = rule["then"]
                rule_confidence = rule.get("confidence", 0.9)
                
                # Check if condition is met by premises
                if self._condition_satisfied(condition, premises):
                    conclusions.append(conclusion)
                    confidence *= rule_confidence
                    
                    trace.append({
                        "step": len(trace),
                        "operation": "apply_rule",
                        "rule": rule,
                        "conclusion": conclusion,
                        "confidence": rule_confidence
                    })
            elif isinstance(rule, str):
                # Simple string rule, try to apply directly
                simple_conclusion = self._apply_simple_rule(rule, premises)
                if simple_conclusion:
                    conclusions.append(simple_conclusion)
                    confidence *= 0.8  # Lower confidence for simple rules
                    
                    trace.append({
                        "step": len(trace),
                        "operation": "apply_simple_rule",
                        "rule": rule,
                        "conclusion": simple_conclusion,
                        "confidence": 0.8
                    })
        
        # Generate final solution and justification
        if conclusions:
            # Combine conclusions
            solution = self._combine_conclusions(conclusions)
            justification = self._generate_deductive_justification(premises, rules, conclusions)
            
            # Calculate entropy - lower for deductive reasoning with more rules
            entropy = max(0.1, 0.5 - 0.05 * len(rules))
            
            trace.append({
                "step": len(trace),
                "operation": "generate_solution",
                "solution": solution,
                "confidence": confidence
            })
            
            return {
                "solution": solution,
                "confidence": confidence,
                "justification": justification,
                "entropy": entropy
            }
        else:
            # Fallback to general inference
            solution = self._generate_general_inference(premises)
            
            trace.append({
                "step": len(trace),
                "operation": "fallback_inference",
                "solution": solution,
                "confidence": 0.4
            })
            
            return {
                "solution": solution,
                "confidence": 0.4,
                "justification": "Derived through general inference from available premises.",
                "entropy": 0.7
            }
    
    def _inductive_reasoning(self, 
                          problem: Dict[str, Any], 
                          memories: List[Dict[str, Any]],
                          constraints: Dict[str, Any],
                          trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform inductive reasoning (examples to general rule)."""
        # Extract examples from problem and memories
        examples = problem.get("examples", [])
        
        # Extract additional examples from memories
        if memories:
            memory_examples = self._extract_examples_from_memories(memories, problem)
            examples.extend(memory_examples)
            
        trace.append({
            "step": len(trace),
            "operation": "collect_examples",
            "examples_count": len(examples)
        })
        
        if not examples:
            # Insufficient data for induction
            trace.append({
                "step": len(trace),
                "operation": "insufficient_examples",
                "fallback": "Use deductive reasoning instead"
            })
            
            # Fallback to deductive reasoning
            return self._deductive_reasoning(problem, memories, constraints, trace)
            
        # Identify patterns in examples
        patterns = self._identify_patterns(examples)
        
        trace.append({
            "step": len(trace),
            "operation": "identify_patterns",
            "patterns": patterns
        })
        
        # Generate rules from patterns
        rules = []
        for pattern in patterns:
            rule = self._generate_rule_from_pattern(pattern, examples)
            if rule:
                rules.append(rule)
                
        trace.append({
            "step": len(trace),
            "operation": "generate_rules",
            "rules": rules
        })
        
        # Apply rules to generate conclusion
        if rules:
            # Test rules against examples for confidence
            confidence = self._test_rules_against_examples(rules, examples)
            
            # Generate solution by applying the best rule
            best_rule = max(rules, key=lambda r: r.get("confidence", 0))
            solution = self._apply_rule_to_problem(best_rule, problem)
            
            # Generate justification
            justification = self._generate_inductive_justification(examples, patterns, rules, solution)
            
            # Calculate entropy - higher for inductive (less certain)
            entropy = 0.4 + (0.3 / (1 + len(examples) / 10))
            
            trace.append({
                "step": len(trace),
                "operation": "generate_solution",
                "solution": solution,
                "confidence": confidence
            })
            
            return {
                "solution": solution,
                "confidence": confidence,
                "justification": justification,
                "entropy": entropy
            }
        else:
            # Failed to generate rules
            trace.append({
                "step": len(trace),
                "operation": "fallback_to_similarity",
                "reason": "No rules generated"
            })
            
            # Fallback to similarity-based solution
            solution = self._similarity_based_solution(examples, problem)
            
            return {
                "solution": solution,
                "confidence": 0.5,
                "justification": "Derived through similarity matching with examples.",
                "entropy": 0.7
            }
    
    def _abductive_reasoning(self, 
                          problem: Dict[str, Any], 
                          memories: List[Dict[str, Any]],
                          constraints: Dict[str, Any],
                          trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform abductive reasoning (effect to best explanation)."""
        # Extract observations from problem
        observations = problem.get("observations", [])
        
        # If no explicit observations, extract from problem context
        if not observations and "context" in problem:
            observations = self._extract_observations(problem["context"])
            
        trace.append({
            "step": len(trace),
            "operation": "extract_observations",
            "observations": observations
        })
        
        # Generate possible explanations
        explanations = self._generate_explanations(observations, memories)
        
        trace.append({
            "step": len(trace),
            "operation": "generate_explanations",
            "explanations_count": len(explanations)
        })
        
        if not explanations:
            # No explanations generated
            trace.append({
                "step": len(trace),
                "operation": "fallback_to_deductive",
                "reason": "No explanations generated"
            })
            
            # Fallback to deductive reasoning
            return self._deductive_reasoning(problem, memories, constraints, trace)
            
        # Rank explanations by plausibility
        ranked_explanations = self._rank_explanations(explanations, observations, memories)
        
        trace.append({
            "step": len(trace),
            "operation": "rank_explanations",
            "top_explanations": ranked_explanations[:3] if len(ranked_explanations) > 3 else ranked_explanations
        })
        
        # Select best explanation as solution
        best_explanation = ranked_explanations[0] if ranked_explanations else None
        
        if best_explanation:
            # Calculate confidence (penalized for abduction)
            base_confidence = best_explanation.get("plausibility", 0.7)
            confidence = base_confidence * (1 - self.adaptive_params["abduction_penalty"])
            
            # Generate justification
            justification = self._generate_abductive_justification(observations, best_explanation, ranked_explanations)
            
            # Calculate entropy - higher for abductive (more uncertain)
            entropy = 0.5 + (0.2 / (1 + len(observations) / 5))
            
            solution = best_explanation.get("explanation", "No viable explanation found")
            
            trace.append({
                "step": len(trace),
                "operation": "select_best_explanation",
                "solution": solution,
                "confidence": confidence
            })
            
            return {
                "solution": solution,
                "confidence": confidence,
                "justification": justification,
                "entropy": entropy,
                "alternative_explanations": [e.get("explanation") for e in ranked_explanations[1:3]] if len(ranked_explanations) > 1 else []
            }
        else:
            # Failed to find viable explanation
            trace.append({
                "step": len(trace),
                "operation": "fallback_to_deductive",
                "reason": "No viable explanations"
            })
            
            # Fallback to deductive
            return self._deductive_reasoning(problem, memories, constraints, trace)
    
    def _analogical_reasoning(self, 
                           problem: Dict[str, Any], 
                           memories: List[Dict[str, Any]],
                           constraints: Dict[str, Any],
                           trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform analogical reasoning (source to target mapping)."""
        # Extract target problem
        target = problem.get("target", problem.get("context", {}))
        
        trace.append({
            "step": len(trace),
            "operation": "extract_target",
            "target": target
        })
        
        # Find source analogs from memories
        source_analogs = self._find_source_analogs(target, memories)
        
        trace.append({
            "step": len(trace),
            "operation": "find_source_analogs",
            "sources_count": len(source_analogs)
        })
        
        if not source_analogs:
            # No good analogs found
            trace.append({
                "step": len(trace),
                "operation": "fallback_to_inductive",
                "reason": "No source analogs found"
            })
            
            # Fallback to inductive reasoning
            return self._inductive_reasoning(problem, memories, constraints, trace)
            
        # Determine best source analog
        best_source = self._determine_best_analog(source_analogs, target)
        
        trace.append({
            "step": len(trace),
            "operation": "select_best_analog",
            "source": best_source
        })
        
        # Map relations from source to target
        mappings = self._map_relations(best_source, target)
        
        trace.append({
            "step": len(trace),
            "operation": "map_relations",
            "mappings": mappings
        })
        
        # Generate candidate solution through analogical transfer
        if mappings:
            solution = self._transfer_solution(best_source, target, mappings)
            
            # Calculate confidence based on mapping quality
            mapping_coverage = len(mappings) / max(1, len(best_source.get("elements", [])))
            confidence = 0.5 + (0.4 * mapping_coverage)
            
            # Generate justification
            justification = self._generate_analogical_justification(best_source, target, mappings, solution)
            
            # Calculate entropy
            similarity = best_source.get("similarity", 0.5)
            entropy = 0.5 - (0.3 * similarity)
            
            trace.append({
                "step": len(trace),
                "operation": "generate_solution",
                "solution": solution,
                "confidence": confidence
            })
            
            return {
                "solution": solution,
                "confidence": confidence,
                "justification": justification,
                "entropy": entropy,
                "source_analog": best_source
            }
        else:
            # No viable mappings
            trace.append({
                "step": len(trace),
                "operation": "fallback_to_deductive",
                "reason": "No viable mappings"
            })
            
            # Fallback to deductive
            return self._deductive_reasoning(problem, memories, constraints, trace)
    
    def _counterfactual_reasoning(self, 
                               problem: Dict[str, Any], 
                               memories: List[Dict[str, Any]],
                               constraints: Dict[str, Any],
                               trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform counterfactual reasoning (what-if analysis)."""
        # Extract the factual case
        factual_case = problem.get("factual_case", problem.get("context", {}))
        
        # Identify candidate intervention points
        intervention_points = self._identify_intervention_points(factual_case)
        
        trace.append({
            "step": len(trace),
            "operation": "identify_intervention_points",
            "points": intervention_points
        })
        
        if not intervention_points:
            # No intervention points found
            trace.append({
                "step": len(trace),
                "operation": "fallback_to_causal",
                "reason": "No intervention points identified"
            })
            
            # Fallback to causal reasoning
            return self._causal_reasoning(problem, memories, constraints, trace)
            
        # Generate counterfactual scenarios
        counterfactuals = self._generate_counterfactuals(factual_case, intervention_points)
        
        trace.append({
            "step": len(trace),
            "operation": "generate_counterfactuals",
            "counterfactuals_count": len(counterfactuals)
        })
        
        # Evaluate counterfactual outcomes
        evaluations = self._evaluate_counterfactuals(counterfactuals, factual_case, memories)
        
        trace.append({
            "step": len(trace),
            "operation": "evaluate_counterfactuals",
            "evaluations_count": len(evaluations)
        })
        
        # Select most informative counterfactual
        best_counterfactual = self._select_best_counterfactual(evaluations)
        
        if best_counterfactual:
            # Generate insight from counterfactual
            solution = self._generate_counterfactual_insight(best_counterfactual, factual_case)
            
            # Calculate confidence
            confidence = best_counterfactual.get("confidence", 0.6)
            
            # Generate justification
            justification = self._generate_counterfactual_justification(
                factual_case, best_counterfactual, solution)
            
            # Calculate entropy
            entropy = 0.4 + (0.2 / max(1, len(evaluations)))
            
            trace.append({
                "step": len(trace),
                "operation": "generate_solution",
                "solution": solution,
                "confidence": confidence
            })
            
            return {
                "solution": solution,
                "confidence": confidence,
                "justification": justification,
                "entropy": entropy,
                "counterfactuals": [c.get("scenario") for c in evaluations[:3]] if evaluations else []
            }
        else:
            # No viable counterfactual
            trace.append({
                "step": len(trace),
                "operation": "fallback_to_causal",
                "reason": "No viable counterfactuals"
            })
            
            # Fallback to causal
            return self._causal_reasoning(problem, memories, constraints, trace)
    
    def _causal_reasoning(self, 
                       problem: Dict[str, Any], 
                       memories: List[Dict[str, Any]],
                       constraints: Dict[str, Any],
                       trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform causal reasoning (cause-effect analysis)."""
        # Extract events/observations
        events = problem.get("events", [])
        
        # If no explicit events, try to extract from context
        if not events and "context" in problem:
            events = self._extract_events(problem["context"])
            
        trace.append({
            "step": len(trace),
            "operation": "extract_events",
            "events_count": len(events)
        })
        
        if not events:
            # No events to analyze
            trace.append({
                "step": len(trace),
                "operation": "fallback_to_deductive",
                "reason": "No events to analyze"
            })
            
            # Fallback to deductive
            return self._deductive_reasoning(problem, memories, constraints, trace)
            
        # Construct causal graph from events and memories
        causal_graph = self._construct_causal_graph(events, memories)
        
        trace.append({
            "step": len(trace),
            "operation": "construct_causal_graph",
            "nodes": len(causal_graph.get("nodes", [])),
            "edges": len(causal_graph.get("edges", []))
        })
        
        # Identify key causal factors
        causal_factors = self._identify_causal_factors(causal_graph, events)
        
        trace.append({
            "step": len(trace),
            "operation": "identify_causal_factors",
            "factors": causal_factors
        })
        
        if causal_factors:
            # Generate causal explanation
            solution = self._generate_causal_explanation(causal_factors, causal_graph, events)
            
            # Calculate confidence
            graph_completeness = len(causal_graph.get("edges", [])) / max(1, len(events) * 2)
            confidence = 0.4 + (0.5 * graph_completeness)
            
            # Generate justification
            justification = self._generate_causal_justification(events, causal_factors, causal_graph)
            
            # Calculate entropy
            entropy = 0.4 + (0.3 / (1 + len(causal_factors)))
            
            trace.append({
                "step": len(trace),
                "operation": "generate_solution",
                "solution": solution,
                "confidence": confidence
            })
            
            return {
                "solution": solution,
                "confidence": confidence,
                "justification": justification,
                "entropy": entropy,
                "causal_factors": causal_factors
            }
        else:
            # No causal factors identified
            trace.append({
                "step": len(trace),
                "operation": "fallback_to_abductive",
                "reason": "No causal factors identified"
            })
            
            # Fallback to abductive
            return self._abductive_reasoning(problem, memories, constraints, trace)
    
    def _multi_step_reasoning(self, 
                           problem: Dict[str, Any], 
                           memories: List[Dict[str, Any]],
                           constraints: Dict[str, Any],
                           trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform multi-step reasoning using different strategies."""
        # Initialize reasoning chain
        reasoning_chain = []
        chain_results = []
        subproblems = []
        
        # Decompose problem into subproblems
        if "subproblems" in problem:
            subproblems = problem["subproblems"]
        else:
            subproblems = self._decompose_problem(problem)
            
        trace.append({
            "step": len(trace),
            "operation": "decompose_problem",
            "subproblems_count": len(subproblems)
        })
        
        # If decomposition failed, fallback to single-strategy reasoning
        if not subproblems:
            # Try deductive as fallback
            trace.append({
                "step": len(trace),
                "operation": "fallback_to_deductive",
                "reason": "Problem decomposition failed"
            })
            
            return self._deductive_reasoning(problem, memories, constraints, trace)
            
        # Process each subproblem with appropriate strategy
        for i, subproblem in enumerate(subproblems):
            # Select best strategy for this subproblem
            subproblem_strategy, _ = self._select_reasoning_strategy(subproblem, memories, constraints)
            
            # Get strategy function
            strategy_fn = self.strategies.get(subproblem_strategy)
            if strategy_fn and strategy_fn != self._multi_step_reasoning:  # Prevent recursion
                # Execute strategy on subproblem
                subtrace = []
                subresult = strategy_fn(subproblem, memories, constraints, subtrace)
                
                # Add to chain
                reasoning_chain.append({
                    "step": i,
                    "strategy": subproblem_strategy,
                    "subproblem": subproblem,
                    "result": subresult,
                    "trace": subtrace
                })
                
                chain_results.append(subresult.get("solution"))
                
                trace.append({
                    "step": len(trace),
                    "operation": f"solve_subproblem_{i}",
                    "strategy": subproblem_strategy,
                    "confidence": subresult.get("confidence", 0)
                })
        
        # Integrate results from reasoning chain
        if chain_results:
            # Calculate overall confidence (average of steps)
            confidence_values = [step.get("result", {}).get("confidence", 0) 
                               for step in reasoning_chain]
            overall_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
            
            # Penalize for multiple steps
            confidence_penalty = 0.05 * len(reasoning_chain)
            adjusted_confidence = max(0.1, overall_confidence - confidence_penalty)
            
            # Create integrated solution
            solution = self._integrate_reasoning_chain(reasoning_chain, problem)
            
            # Generate justification
            justification = self._generate_multi_step_justification(reasoning_chain, solution)
            
            # Calculate entropy (increases with steps)
            entropy = 0.3 + (0.1 * len(reasoning_chain))
            
            trace.append({
                "step": len(trace),
                "operation": "integrate_results",
                "solution": solution,
                "confidence": adjusted_confidence
            })
            
            return {
                "solution": solution,
                "confidence": adjusted_confidence,
                "justification": justification,
                "entropy": entropy,
                "reasoning_chain": reasoning_chain
            }
        else:
            # Chain failed, fall back to most appropriate single strategy
            fallback_strategy, _ = self._select_reasoning_strategy(problem, memories, constraints)
            fallback_fn = self.strategies.get(fallback_strategy, self._deductive_reasoning)
            
            trace.append({
                "step": len(trace),
                "operation": "fallback_to_single_strategy",
                "strategy": fallback_strategy
            })
            
            # Use fallback but make sure it's not multi_step to avoid infinite recursion
            if fallback_fn == self._multi_step_reasoning:
                fallback_fn = self._deductive_reasoning
                
            return fallback_fn(problem, memories, constraints, trace)
    
    # Helper methods for reasoning strategies
    
    def _extract_premises(self, context: Any) -> List[str]:
        """Extract premises from problem context."""
        premises = []
        
        if isinstance(context, dict):
            # Extract key-value pairs as premises
            for key, value in context.items():
                premises.append(f"{key}: {value}")
        elif isinstance(context, list):
            # Use list items as premises
            premises = context
        elif isinstance(context, str):
            # Try to split string on separators
            for separator in ['. ', '; ', '\n']:
                if separator in context:
                    premises = [p.strip() for p in context.split(separator) if p.strip()]
                    break
            
            # If no separators found, use whole string
            if not premises:
                premises = [context]
                
        return premises
    
    def _extract_rules_from_memories(self, memories: List[Dict[str, Any]]) -> List[Any]:
        """Extract rules from memory."""
        rules = []
        
        for memory in memories:
            content = memory.get("content", {})
            
            # Check for explicit rules in memory
            if isinstance(content, dict) and "rules" in content:
                memory_rules = content["rules"]
                if isinstance(memory_rules, list):
                    rules.extend(memory_rules)
            
            # Check for rules in simplified form
            elif isinstance(content, dict) and "if" in content and "then" in content:
                rules.append(content)
                
            # Extract rules from strings
            elif isinstance(content, str) and "if" in content.lower() and "then" in content.lower():
                # Basic rule extraction from text
                rule_parts = content.split("then", 1)
                if len(rule_parts) == 2 and "if" in rule_parts[0].lower():
                    condition = rule_parts[0].replace("if", "", 1).strip()
                    conclusion = rule_parts[1].strip()
                    rules.append({"if": condition, "then": conclusion, "confidence": 0.7})
        
        return rules
    
    def _condition_satisfied(self, condition: Any, premises: List[str]) -> bool:
        """Check if a condition is satisfied by the premises."""
        if isinstance(condition, str):
            # Check if condition string is contained in any premise
            condition_lower = condition.lower()
            for premise in premises:
                if isinstance(premise, str) and condition_lower in premise.lower():
                    return True
                    
            return False
        elif isinstance(condition, dict) and "any" in condition:
            # Check if any subcondition is met
            subconditions = condition["any"]
            return any(self._condition_satisfied(subcond, premises) for subcond in subconditions)
        elif isinstance(condition, dict) and "all" in condition:
            # Check if all subconditions are met
            subconditions = condition["all"]
            return all(self._condition_satisfied(subcond, premises) for subcond in subconditions)
        
        return False
    
    def _apply_simple_rule(self, rule: str, premises: List[str]) -> Optional[str]:
        """Apply a simple string rule to premises."""
        rule_lower = rule.lower()
        
        # Check for "if...then" structure
        if "if" in rule_lower and "then" in rule_lower:
            parts = rule.split("then", 1)
            if len(parts) == 2:
                condition = parts[0].replace("if", "", 1).strip()
                conclusion = parts[1].strip()
                
                # Check if condition is met
                for premise in premises:
                    if isinstance(premise, str) and condition.lower() in premise.lower():
                        return conclusion
        
        return None
    
    def _combine_conclusions(self, conclusions: List[Any]) -> Any:
        """Combine multiple conclusions into a single solution."""
        if not conclusions:
            return None
            
        if len(conclusions) == 1:
            return conclusions[0]
            
        # If all conclusions are strings, join them
        if all(isinstance(c, str) for c in conclusions):
            return ". ".join(conclusions)
            
        # If all conclusions are dicts, merge them
        if all(isinstance(c, dict) for c in conclusions):
            merged = {}
            for conclusion in conclusions:
                merged.update(conclusion)
            return merged
            
        # Otherwise, return as a list
        return conclusions
    
    def _generate_deductive_justification(self, 
                                       premises: List[str], 
                                       rules: List[Any], 
                                       conclusions: List[Any]) -> str:
        """Generate justification for deductive reasoning."""
        justification = "Deductive reasoning applied:\n"
        
        # Add premises
        if premises:
            justification += "Starting with premises:\n"
            for i, premise in enumerate(premises[:3]):  # Limit to first 3 for brevity
                justification += f"- {premise}\n"
            if len(premises) > 3:
                justification += f"- (and {len(premises) - 3} more premises)\n"
        
        # Add rules applied
        if rules:
            justification += "Applying rules:\n"
            for i, rule in enumerate(rules[:3]):  # Limit to first 3
                if isinstance(rule, dict) and "if" in rule and "then" in rule:
                    justification += f"- If {rule['if']} then {rule['then']}\n"
                else:
                    justification += f"- {rule}\n"
            if len(rules) > 3:
                justification += f"- (and {len(rules) - 3} more rules)\n"
        
        # Add conclusions
        if conclusions:
            justification += "Reaching conclusions:\n"
            for i, conclusion in enumerate(conclusions[:3]):  # Limit to first 3
                justification += f"- {conclusion}\n"
            if len(conclusions) > 3:
                justification += f"- (and {len(conclusions) - 3} more conclusions)\n"
                
        return justification
    
    def _generate_general_inference(self, premises: List[str]) -> Any:
        """Generate a general inference when explicit rules fail."""
        if not premises:
            return "Insufficient premises for inference"
            
        # Very basic inference approach - in a real system this would be more sophisticated
        if len(premises) == 1:
            return f"Based solely on the premise: {premises[0]}"
        else:
            return f"Inference based on {len(premises)} premises, primarily: {premises[0]}"
    
    def _extract_examples_from_memories(self, 
                                     memories: List[Dict[str, Any]], 
                                     problem: Dict[str, Any]) -> List[Any]:
        """Extract relevant examples from memories."""
        examples = []
        problem_type = problem.get("type", "unknown")
        
        for memory in memories:
            content = memory.get("content", {})
            
            # Check for explicit examples
            if isinstance(content, dict) and "examples" in content:
                memory_examples = content["examples"]
                if isinstance(memory_examples, list):
                    examples.extend(memory_examples)
            
            # Check for examples in execution results
            elif (isinstance(content, dict) and "type" in content 
                  and content.get("type") == problem_type):
                # Extract as example if relevant to problem type
                example = {"input": content.get("input"), "output": content.get("output")}
                examples.append(example)
        
        return examples
    
    def _identify_patterns(self, examples: List[Any]) -> List[Dict[str, Any]]:
        """Identify patterns in examples."""
        patterns = []
        
        # Check if examples have consistent structure
        if not examples or not all(isinstance(e, dict) for e in examples):
            return patterns
            
        # Identify common keys
        common_keys = set.intersection(*[set(e.keys()) for e in examples])
        
        for key in common_keys:
            values = [e[key] for e in examples]
            
            # Check for consistent type
            if all(isinstance(v, (int, float)) for v in values):
                # Numerical pattern
                avg = sum(values) / len(values)
                variance = sum((v - avg) ** 2 for v in values) / len(values)
                
                if variance < 0.1 * avg:  # Low variance indicates pattern
                    patterns.append({
                        "type": "constant",
                        "field": key,
                        "value": avg,
                        "confidence": 0.9
                    })
                else:
                    # Check for trend
                    if len(values) >= 3:
                        x = list(range(len(values)))
                        # Simple linear regression
                        m, b = np.polyfit(x, values, 1)
                        if abs(m) > 0.01:  # Non-zero slope indicates trend
                            patterns.append({
                                "type": "trend",
                                "field": key,
                                "slope": m,
                                "intercept": b,
                                "confidence": 0.7
                            })
            
            # Check for categorical patterns
            elif all(isinstance(v, str) for v in values):
                # Check for most common value
                value_counts = {}
                for v in values:
                    value_counts[v] = value_counts.get(v, 0) + 1
                
                most_common = max(value_counts.items(), key=lambda x: x[1])
                if most_common[1] > len(values) * 0.7:  # >70% occurrence
                    patterns.append({
                        "type": "category",
                        "field": key,
                        "value": most_common[0],
                        "confidence": most_common[1] / len(values)
                    })
        
        return patterns
    
    def _generate_rule_from_pattern(self, 
                                 pattern: Dict[str, Any], 
                                 examples: List[Any]) -> Optional[Dict[str, Any]]:
        """Generate a rule from an identified pattern."""
        if pattern["type"] == "constant":
            return {
                "type": "equality",
                "field": pattern["field"],
                "value": pattern["value"],
                "confidence": pattern["confidence"]
            }
        elif pattern["type"] == "trend":
            return {
                "type": "trend",
                "field": pattern["field"],
                "slope": pattern["slope"],
                "intercept": pattern["intercept"],
                "confidence": pattern["confidence"]
            }
        elif pattern["type"] == "category":
            return {
                "type": "category",
                "field": pattern["field"],
                "value": pattern["value"],
                "confidence": pattern["confidence"]
            }
        
        return None
    
    def _test_rules_against_examples(self, 
                                   rules: List[Dict[str, Any]], 
                                   examples: List[Any]) -> float:
        """Test generated rules against examples to calculate confidence."""
        if not rules or not examples:
            return 0.4  # Default moderate confidence
            
        # Count matches for each rule
        rule_matches = []
        for rule in rules:
            matches = 0
            for example in examples:
                if self._rule_matches_example(rule, example):
                    matches += 1
            
            match_rate = matches / len(examples)
            rule_matches.append(match_rate)
        
        # Overall confidence is average match rate
        return sum(rule_matches) / len(rule_matches)
    
    def _rule_matches_example(self, rule: Dict[str, Any], example: Dict[str, Any]) -> bool:
        """Check if a rule matches an example."""
        if rule["type"] == "equality":
            if rule["field"] in example:
                return abs(example[rule["field"]] - rule["value"]) < 0.1 * rule["value"]
        elif rule["type"] == "trend":
            # Would need to know example position in sequence
            return True  # Simplification
        elif rule["type"] == "category":
            if rule["field"] in example:
                return example[rule["field"]] == rule["value"]
        
        return False
    
    def _apply_rule_to_problem(self, rule: Dict[str, Any], problem: Dict[str, Any]) -> Any:
        """Apply a rule to generate a solution for the problem."""
        if "field" not in rule:
            return "Unable to apply rule to problem"
            
        field = rule["field"]
        
        # Check if the field exists in the problem
        problem_content = problem.get("content", {})
        if field not in problem_content:
            return f"Rule applies to {field} but problem lacks this field"
            
        # Apply rule based on type
        if rule["type"] == "equality":
            return f"{field} value is approximately {rule['value']}"
        elif rule["type"] == "trend":
            slope = rule["slope"]
            direction = "increases" if slope > 0 else "decreases"
            return f"{field} {direction} at a rate of {abs(slope):.2f} per increment"
        elif rule["type"] == "category":
            return f"{field} is typically '{rule['value']}'"
        
        return "Unable to determine pattern application"
    
    def _generate_inductive_justification(self, 
                                      examples: List[Any], 
                                      patterns: List[Dict[str, Any]], 
                                      rules: List[Dict[str, Any]], 
                                      solution: Any) -> str:
        """Generate justification for inductive reasoning."""
        justification = "Inductive reasoning applied:\n"
        
        # Add examples summary
        justification += f"Based on {len(examples)} examples, "
        
        # Add patterns identified
        if patterns:
            justification += "identified patterns:\n"
            for pattern in patterns[:3]:  # Limit to first 3
                if pattern["type"] == "constant":
                    justification += f"- {pattern['field']} is consistently around {pattern['value']}\n"
                elif pattern["type"] == "trend":
                    direction = "increases" if pattern["slope"] > 0 else "decreases"
                    justification += f"- {pattern['field']} {direction} at a rate of {abs(pattern['slope']):.2f}\n"
                elif pattern["type"] == "category":
                    justification += f"- {pattern['field']} is typically '{pattern['value']}'\n"
            
            if len(patterns) > 3:
                justification += f"- (and {len(patterns) - 3} more patterns)\n"
        else:
            justification += "but no clear patterns were identified.\n"
            
        # Add rule application
        if rules:
            justification += "Applied rules to derive solution"
        else:
            justification += "Insufficient rules derived from patterns"
            
        return justification
    
    def _similarity_based_solution(self, examples: List[Any], problem: Dict[str, Any]) -> Any:
        """Generate solution based on similarity to examples when rule generation fails."""
        if not examples:
            return "No examples available for similarity comparison"
            
        # Find most similar example
        problem_content = problem.get("content", {})
        most_similar = None
        highest_similarity = -1
        
        for example in examples:
            similarity = self._calculate_similarity(example, problem_content)
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar = example
                
        if most_similar and highest_similarity > 0.5:
            if "output" in most_similar:
                return most_similar["output"]
            else:
                return f"Most similar example: {most_similar}"
        else:
            return "No sufficiently similar examples found"
    
    def _calculate_similarity(self, example: Any, problem: Any) -> float:
        """Calculate similarity between example and problem."""
        # Simple implementation - would be more sophisticated in a real system
        if isinstance(example, dict) and isinstance(problem, dict):
            # Count matching keys
            common_keys = set(example.keys()).intersection(set(problem.keys()))
            total_keys = set(example.keys()).union(set(problem.keys()))
            
            if not total_keys:
                return 0
                
            # Check value similarity for common keys
            value_similarities = []
            for key in common_keys:
                if key in example and key in problem:
                    if example[key] == problem[key]:
                        value_similarities.append(1.0)
                    else:
                        value_similarities.append(0.5)  # Partial match
            
            # Overall similarity
            key_similarity = len(common_keys) / len(total_keys)
            value_similarity = sum(value_similarities) / max(1, len(common_keys))
            
            return 0.5 * key_similarity + 0.5 * value_similarity
        elif isinstance(example, str) and isinstance(problem, str):
            # Simple string similarity
            words1 = set(example.lower().split())
            words2 = set(problem.lower().split())
            
            common_words = words1.intersection(words2)
            total_words = words1.union(words2)
            
            if not total_words:
                return 0
                
            return len(common_words) / len(total_words)
        else:
            return 0.0  # Different types
    
    def _extract_observations(self, context: Any) -> List[str]:
        """Extract observations from problem context."""
        observations = []
        
        if isinstance(context, dict):
            if "observations" in context:
                return context["observations"] if isinstance(context["observations"], list) else [context["observations"]]
            
            # Extract key observations from dictionary
            for key, value in context.items():
                observations.append(f"{key}: {value}")
        elif isinstance(context, list):
            observations = context
        elif isinstance(context, str):
            # Split by sentence or line
            for separator in ['. ', '; ', '\n']:
                if separator in context:
                    observations = [obs.strip() for obs in context.split(separator) if obs.strip()]
                    break
            
            if not observations:
                observations = [context]
                
        return observations
    
    def _generate_explanations(self, 
                            observations: List[str], 
                            memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate possible explanations for observations."""
        explanations = []
        
        # First, check memories for similar observations and their explanations
        for memory in memories:
            content = memory.get("content", {})
            memory_observations = []
            memory_explanation = None
            
            # Extract observations and explanations from memory
            if isinstance(content, dict):
                if "observations" in content:
                    memory_observations = content["observations"]
                    if "explanation" in content:
                        memory_explanation = content["explanation"]
                elif "symptoms" in content:
                    memory_observations = content["symptoms"]
                    if "diagnosis" in content:
                        memory_explanation = content["diagnosis"]
                elif "problem" in content:
                    memory_observations = content["problem"]
                    if "solution" in content:
                        memory_explanation = content["solution"]
            
            # Check similarity with current observations
            if memory_observations and memory_explanation:
                similarity = self._calculate_observations_similarity(
                    observations, memory_observations)
                
                if similarity > 0.6:  # Threshold for considering memory
                    explanations.append({
                        "explanation": memory_explanation,
                        "source": "memory",
                        "similarity": similarity,
                        "plausibility": 0.7 * similarity  # Scale based on similarity
                    })
        
        # Generate additional candidate explanations if needed
        if len(explanations) < 3:
            # Causal chain analysis for observations
            causal_explanation = self._generate_causal_explanation_from_observations(observations)
            if causal_explanation:
                explanations.append({
                    "explanation": causal_explanation,
                    "source": "causal_analysis",
                    "plausibility": 0.6
                })
                
            # Pattern-based explanation
            pattern_explanation = self._generate_pattern_explanation(observations)
            if pattern_explanation:
                explanations.append({
                    "explanation": pattern_explanation,
                    "source": "pattern_analysis",
                    "plausibility": 0.5
                })
        
        return explanations
    
    def _calculate_observations_similarity(self, 
                                        observations1: List[str], 
                                        observations2: List[str]) -> float:
        """Calculate similarity between two sets of observations."""
        # Convert to sets of words for comparison
        words1 = set()
        for obs in observations1:
            words1.update(str(obs).lower().split())
        
        words2 = set()
        for obs in observations2:
            words2.update(str(obs).lower().split())
        
        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _generate_causal_explanation_from_observations(self, observations: List[str]) -> Optional[str]:
        """Generate a causal explanation from observations."""
        if not observations:
            return None
            
        # Very simplified implementation - in real system would use more sophisticated analysis
        common_themes = set()
        for obs in observations:
            # Extract key terms
            words = str(obs).lower().split()
            common_themes.update([w for w in words if len(w) > 4])
            
        if common_themes:
            top_themes = list(common_themes)[:3]
            return f"Causal explanation based on key factors: {', '.join(top_themes)}"
        
        return None
    
    def _generate_pattern_explanation(self, observations: List[str]) -> Optional[str]:
        """Generate pattern-based explanation from observations."""
        if len(observations) < 2:
            return None
            
        # Check for time-based patterns
        time_indicators = ["first", "then", "after", "before", "finally", "initially"]
        has_temporal = False
        
        for obs in observations:
            if any(indicator in str(obs).lower() for indicator in time_indicators):
                has_temporal = True
                break
                
        if has_temporal:
            return "Pattern shows a temporal sequence of events"
            
        # Check for conditional patterns
        conditional_indicators = ["if", "when", "unless", "only if"]
        has_conditional = False
        
        for obs in observations:
            if any(indicator in str(obs).lower() for indicator in conditional_indicators):
                has_conditional = True
                break
                
        if has_conditional:
            return "Pattern shows conditional relationships"
        
        return "No clear pattern identified in observations"
    
    def _rank_explanations(self, 
                        explanations: List[Dict[str, Any]], 
                        observations: List[str],
                        memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank explanations by plausibility."""
        if not explanations:
            return []
            
        # Calculate coverage for each explanation
        for explanation in explanations:
            coverage = self._calculate_explanation_coverage(
                explanation["explanation"], observations)
            
            # Update plausibility based on coverage
            explanation["coverage"] = coverage
            explanation["plausibility"] = (explanation.get("plausibility", 0.5) + coverage) / 2
            
        # Calculate coherence with memories
        for explanation in explanations:
            coherence = self._calculate_explanation_coherence(
                explanation["explanation"], memories)
            
            # Update plausibility based on coherence
            explanation["coherence"] = coherence
            explanation["plausibility"] = (explanation["plausibility"] + coherence) / 2
        
        # Sort by plausibility
        return sorted(explanations, key=lambda x: x.get("plausibility", 0), reverse=True)
    
    def _calculate_explanation_coverage(self, explanation: Any, observations: List[str]) -> float:
        """Calculate how well an explanation covers the observations."""
        # Convert explanation to string for comparison
        explanation_str = str(explanation).lower()
        
        # Count observations that are covered by the explanation
        covered = 0
        for obs in observations:
            obs_str = str(obs).lower()
            
            # Simple coverage check - observation terms appear in explanation
            terms = [t for t in obs_str.split() if len(t) > 3]
            
            if terms:
                # Calculate what percentage of key terms are in explanation
                term_coverage = sum(1 for t in terms if t in explanation_str) / len(terms)
                if term_coverage > 0.3:  # At least 30% of terms covered
                    covered += term_coverage
            
        # Calculate overall coverage
        return covered / max(1, len(observations))
    
    def _calculate_explanation_coherence(self, explanation: Any, memories: List[Dict[str, Any]]) -> float:
        """Calculate how coherent an explanation is with existing memories."""
        if not memories:
            return 0.5  # Neutral coherence when no memories
            
        # Convert explanation to string for comparison
        explanation_str = str(explanation).lower()
        
        # Check similarity with memory content
        similarity_scores = []
        
        for memory in memories:
            content = memory.get("content", {})
            
            if isinstance(content, dict):
                # Check for explanations in memory
                memory_explanations = []
                
                if "explanation" in content:
                    memory_explanations.append(content["explanation"])
                if "solution" in content:
                    memory_explanations.append(content["solution"])
                if "diagnosis" in content:
                    memory_explanations.append(content["diagnosis"])
                    
                # Calculate similarity with each memory explanation
                for mem_exp in memory_explanations:
                    mem_exp_str = str(mem_exp).lower()
                    
                    # Simple word overlap similarity
                    words1 = set(explanation_str.split())
                    words2 = set(mem_exp_str.split())
                    
                    if words1 and words2:
                        intersection = words1.intersection(words2)
                        union = words1.union(words2)
                        
                        similarity = len(intersection) / len(union)
                        similarity_scores.append(similarity)
        
        # Return average similarity
        if similarity_scores:
            return sum(similarity_scores) / len(similarity_scores)
        else:
            return 0.5  # Neutral coherence
    
    def _generate_abductive_justification(self, 
                                      observations: List[str],
                                      best_explanation: Dict[str, Any],
                                      ranked_explanations: List[Dict[str, Any]]) -> str:
        """Generate justification for abductive reasoning."""
        justification = "Abductive reasoning applied:\n"
        
        # Add observations summary
        justification += f"Based on {len(observations)} observations, including:\n"
        for i, obs in enumerate(observations[:3]):  # Limit to first 3
            justification += f"- {obs}\n"
        if len(observations) > 3:
            justification += f"- (and {len(observations) - 3} more observations)\n"
            
        # Add explanation metrics
        if "coverage" in best_explanation:
            justification += f"Selected explanation covers {best_explanation['coverage']:.0%} of observations.\n"
            
        if "coherence" in best_explanation:
            justification += f"Explanation has {best_explanation['coherence']:.0%} coherence with existing knowledge.\n"
            
        if "source" in best_explanation:
            justification += f"Explanation derived from {best_explanation['source']}.\n"
            
        # Add alternative explanations summary
        if len(ranked_explanations) > 1:
            justification += f"Selected from {len(ranked_explanations)} candidate explanations.\n"
            
            # Add top alternative
            alt = ranked_explanations[1]
            justification += f"Top alternative: {alt.get('explanation')[:50]}... (plausibility: {alt.get('plausibility', 0):.0%})\n"
        
        return justification
    
    def _find_source_analogs(self, 
                          target: Dict[str, Any], 
                          memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find source analogs in memory for analogical reasoning."""
        analogs = []
        
        for memory in memories:
            content = memory.get("content", {})
            
            # Skip memories that don't have structured content
            if not isinstance(content, dict):
                continue
                
            # Calculate structural similarity
            similarity = self._calculate_structural_similarity(content, target)
            
            if similarity > 0.4:  # Threshold for considering as analog
                # Extract elements and relations
                elements, relations = self._extract_elements_and_relations(content)
                
                analogs.append({
                    "content": content,
                    "similarity": similarity,
                    "elements": elements,
                    "relations": relations,
                    "memory_id": memory.get("id", "unknown")
                })
        
        # Sort by similarity
        analogs.sort(key=lambda x: x["similarity"], reverse=True)
        
        return analogs
    
    def _calculate_structural_similarity(self, source: Dict[str, Any], target: Dict[str, Any]) -> float:
        """Calculate structural similarity between source and target."""
        if not source or not target:
            return 0
            
        # Calculate key overlap
        source_keys = set(source.keys())
        target_keys = set(target.keys())
        
        key_overlap = len(source_keys.intersection(target_keys)) / len(source_keys.union(target_keys))
        
        # Calculate value type similarity
        type_similarity = 0
        common_keys = source_keys.intersection(target_keys)
        
        if common_keys:
            type_matches = 0
            for key in common_keys:
                source_type = type(source[key]).__name__
                target_type = type(target[key]).__name__
                
                if source_type == target_type:
                    type_matches += 1
                    
            type_similarity = type_matches / len(common_keys)
        
        # Combine similarities
        return 0.7 * key_overlap + 0.3 * type_similarity
    
    def _extract_elements_and_relations(self, content: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Extract elements and relations from content."""
        elements = []
        relations = []
        
        # Extract elements (keys and values)
        for key, value in content.items():
            if isinstance(value, (str, int, float, bool)):
                elements.append(key)
                elements.append(str(value))
                
                # Add relation between key and value
                relations.append({
                    "type": "has_value",
                    "source": key,
                    "target": str(value)
                })
            elif isinstance(value, dict):
                # Nested dictionary
                elements.append(key)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (str, int, float, bool)):
                        elements.append(sub_key)
                
