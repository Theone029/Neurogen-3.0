import time
import traceback
from typing import Dict, List, Any, Optional, Tuple, Callable
import json

class Executor:
    """Executes plans to generate outputs based on task requirements."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "avg_execution_time": 0,
            "errors_by_step_type": {}
        }
        
        # Initialize execution handlers
        self.step_handlers = self._initialize_step_handlers()
        
    def execute(self, plan: Dict[str, Any], task: Dict[str, Any]) -> Any:
        """
        Execute a plan to generate output.
        
        Args:
            plan: Execution plan with steps
            task: Original task
            
        Returns:
            Generated output
        """
        start_time = time.time()
        self.execution_stats["total_executions"] += 1
        
        # Initialize execution context
        context = {
            "task": task,
            "plan": plan,
            "steps_completed": 0,
            "steps_failed": 0,
            "step_outputs": {},
            "execution_start": start_time,
            "errors": []
        }
        
        try:
            # Extract plan steps
            steps = plan.get("steps", [])
            if not steps:
                raise ValueError("Plan contains no steps to execute")
            
            # Store intermediate outputs for step dependencies
            outputs = {"task": task}
            
            # Execute each step in sequence
            for i, step in enumerate(steps):
                step_id = step.get("id", f"step_{i}")
                step_type = step.get("type", "unknown")
                step_inputs = step.get("inputs", [])
                
                # Prepare inputs for this step
                step_input_data = {}
                for input_key in step_inputs:
                    if input_key in outputs:
                        step_input_data[input_key] = outputs[input_key]
                    else:
                        context["errors"].append(f"Missing input '{input_key}' for step '{step_id}'")
                
                # Execute the step
                try:
                    step_start = time.time()
                    step_output = self._execute_step(step, step_input_data, context)
                    step_duration = time.time() - step_start
                    
                    # Store the output for dependent steps
                    outputs[step_id] = step_output
                    
                    # Update context
                    context["steps_completed"] += 1
                    context["step_outputs"][step_id] = {
                        "output": step_output,
                        "duration": step_duration
                    }
                    
                except Exception as e:
                    # Record step failure
                    error_msg = str(e)
                    error_trace = traceback.format_exc()
                    
                    context["steps_failed"] += 1
                    context["errors"].append({
                        "step_id": step_id,
                        "step_type": step_type,
                        "error": error_msg,
                        "traceback": error_trace
                    })
                    
                    # Update error statistics
                    if step_type not in self.execution_stats["errors_by_step_type"]:
                        self.execution_stats["errors_by_step_type"][step_type] = 0
                    self.execution_stats["errors_by_step_type"][step_type] += 1
                    
                    # Check if we should continue or abort
                    if self.config.get("continue_on_step_error", False):
                        # Provide a default output for dependent steps
                        outputs[step_id] = {"error": error_msg}
                    else:
                        # Abort execution
                        raise RuntimeError(f"Execution aborted at step '{step_id}': {error_msg}")
            
            # Generate final output based on last step or plan's output instructions
            if "output_step" in plan:
                output_step = plan["output_step"]
                if output_step in outputs:
                    final_output = outputs[output_step]
                else:
                    raise ValueError(f"Specified output step '{output_step}' not found in outputs")
            else:
                # Use the last step's output by default
                last_step = steps[-1]["id"]
                final_output = outputs[last_step]
            
            # Format the output if needed
            if "output_format" in plan:
                final_output = self._format_output(final_output, plan["output_format"])
            
            # Update success statistics
            self.execution_stats["successful_executions"] += 1
            
            return final_output
            
        except Exception as e:
            # Record execution failure
            context["execution_error"] = str(e)
            context["execution_error_trace"] = traceback.format_exc()
            
            # Construct error output
            error_output = {
                "error": str(e),
                "context": {
                    "steps_completed": context["steps_completed"],
                    "steps_failed": context["steps_failed"]
                }
            }
            
            # Raise exception or return error output based on configuration
            if self.config.get("raise_execution_errors", True):
                raise
            else:
                return error_output
                
        finally:
            # Update timing statistics
            execution_time = time.time() - start_time
            self.execution_stats["avg_execution_time"] = (
                (self.execution_stats["avg_execution_time"] * 
                 (self.execution_stats["total_executions"] - 1) + 
                 execution_time) / self.execution_stats["total_executions"]
            )
    
    def _execute_step(self, 
                    step: Dict[str, Any], 
                    inputs: Dict[str, Any], 
                    context: Dict[str, Any]) -> Any:
        """Execute an individual plan step."""
        step_type = step.get("type", "unknown")
        
        # Get appropriate handler for this step type
        if step_type in self.step_handlers:
            handler = self.step_handlers[step_type]
            return handler(step, inputs, context)
        else:
            # Use generic handler for unknown step types
            return self._generic_step_handler(step, inputs, context)
    
    def _initialize_step_handlers(self) -> Dict[str, Callable]:
        """Initialize handlers for different step types."""
        handlers = {}
        
        # Analysis handler
        handlers["analysis"] = self._analysis_handler
        
        # Memory retrieval handler
        handlers["memory_retrieval"] = self._memory_retrieval_handler
        handlers["memory_application"] = self._memory_application_handler
        handlers["memory_inspiration"] = self._memory_inspiration_handler
        
        # Planning and execution handlers
        handlers["planning"] = self._planning_handler
        handlers["execution"] = self._execution_handler
        handlers["implementation"] = self._implementation_handler
        
        # Creative thinking handlers
        handlers["divergent_thinking"] = self._divergent_thinking_handler
        handlers["evaluation"] = self._evaluation_handler
        handlers["refinement"] = self._refinement_handler
        
        # Analytical handlers
        handlers["decomposition"] = self._decomposition_handler
        handlers["component_analysis"] = self._component_analysis_handler
        handlers["synthesis"] = self._synthesis_handler
        handlers["conclusion"] = self._conclusion_handler
        
        # Sequential processing handlers
        handlers["pipeline_creation"] = self._pipeline_creation_handler
        handlers["stage_configuration"] = self._stage_configuration_handler
        handlers["pipeline_execution"] = self._pipeline_execution_handler
        
        # Add more handlers as needed
        
        return handlers
    
    # Step type handlers
    def _analysis_handler(self, 
                       step: Dict[str, Any], 
                       inputs: Dict[str, Any], 
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analysis steps."""
        task = inputs.get("task", {})
        
        # Extract key elements from task
        analysis = {
            "goal": task.get("goal", ""),
            "requirements": self._extract_requirements(task),
            "constraints": task.get("constraints", {}),
            "key_elements": self._identify_key_elements(task)
        }
        
        return analysis
    
    def _memory_retrieval_handler(self, 
                               step: Dict[str, Any], 
                               inputs: Dict[str, Any], 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory retrieval steps."""
        task = inputs.get("task", {})
        memories = inputs.get("memories", [])
        
        if not memories:
            return {"retrieved": [], "relevance": {}}
            
        # Extract relevant content from memories
        retrieved = []
        relevance = {}
        
        for memory in memories:
            # Skip suppressed memories
            if memory.get("suppressed", False):
                continue
                
            memory_id = memory.get("id", "unknown")
            
            # Extract content based on memory type
            content = None
            if memory.get("type") == "execution_result":
                # For execution results, get the output
                if "content" in memory and "output" in memory["content"]:
                    content = memory["content"]["output"]
            else:
                # For other types, use content directly
                content = memory.get("content")
                
            if content:
                retrieved.append({
                    "id": memory_id,
                    "content": content,
                    "type": memory.get("type", "unknown")
                })
                
                # Calculate simple relevance score
                relevance[memory_id] = memory.get("attention_weight", 0.5)
        
        return {
            "retrieved": retrieved,
            "relevance": relevance
        }
    
    def _memory_application_handler(self, 
                                 step: Dict[str, Any], 
                                 inputs: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory application steps."""
        understand = inputs.get("understand", {})
        memories = inputs.get("memories", [])
        
        # Extract insights from memories
        insights = []
        
        for memory in memories:
            # Skip suppressed memories
            if memory.get("suppressed", False):
                continue
                
            memory_id = memory.get("id", "unknown")
            
            # Get relevant content from memory
            content = memory.get("content", {})
            if isinstance(content, dict) and "output" in content:
                content = content["output"]
                
            # Generate insight from memory content
            if content:
                insights.append({
                    "memory_id": memory_id,
                    "pattern": self._extract_pattern(content),
                    "relevance": memory.get("attention_weight", 0.5)
                })
        
        return {
            "insights": insights,
            "applied_count": len(insights),
            "understanding": understand
        }
    
    def _memory_inspiration_handler(self, 
                                 step: Dict[str, Any], 
                                 inputs: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle creative inspiration from memories."""
        define = inputs.get("define", {})
        memories = inputs.get("memories", [])
        
        # Extract creative elements from memories
        inspirations = []
        
        for memory in memories:
            # Skip suppressed memories
            if memory.get("suppressed", False):
                continue
                
            memory_id = memory.get("id", "unknown")
            
            # Get relevant content from memory
            content = memory.get("content", {})
            if isinstance(content, dict) and "output" in content:
                content = content["output"]
                
            # Extract creative elements
            if content:
                inspirations.append({
                    "memory_id": memory_id,
                    "elements": self._extract_creative_elements(content),
                    "novelty": memory.get("metadata", {}).get("novelty", 0.5)
                })
        
        return {
            "inspirations": inspirations,
            "boundaries": define.get("boundaries", {}),
            "themes": self._identify_themes(inspirations)
        }
    
    def _planning_handler(self, 
                       step: Dict[str, Any], 
                       inputs: Dict[str, Any], 
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle solution planning steps."""
        analysis = inputs.get("analyze", {})
        retrieved = inputs.get("retrieve", {"retrieved": []})
        
        # Develop solution approach
        approach = {
            "strategy": self._determine_solution_strategy(analysis, retrieved),
            "components": self._identify_solution_components(analysis),
            "methodology": self._determine_methodology(analysis)
        }
        
        # Apply insights from memories if available
        if "retrieved" in retrieved and retrieved["retrieved"]:
            approach["memory_applications"] = self._apply_memory_insights(
                retrieved["retrieved"], approach)
        
        return approach
    
    def _execution_handler(self, 
                        step: Dict[str, Any], 
                        inputs: Dict[str, Any], 
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle solution execution steps."""
        approach = inputs.get("formulate", {})
        
        # Execute solution based on approach
        result = {
            "strategy_executed": approach.get("strategy", "default"),
            "components_implemented": {}
        }
        
        # Implement each component
        for component_name, component_spec in approach.get("components", {}).items():
            result["components_implemented"][component_name] = self._implement_component(
                component_name, component_spec)
        
        # Apply methodology
        result["methodology_applied"] = approach.get("methodology", "default")
        
        return result
    
    def _implementation_handler(self, 
                             step: Dict[str, Any], 
                             inputs: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle solution implementation steps."""
        develop = inputs.get("develop", {})
        
        # Implement solution
        implementation = {
            "approach": develop.get("strategy", "default"),
            "elements": {}
        }
        
        # Implement each element of the solution
        for element_name, element_spec in develop.get("components", {}).items():
            implementation["elements"][element_name] = self._implement_element(
                element_name, element_spec)
        
        return implementation
    
    def _divergent_thinking_handler(self, 
                                 step: Dict[str, Any], 
                                 inputs: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle divergent thinking steps."""
        define = inputs.get("define", {})
        inspire = inputs.get("inspire", {"inspirations": []})
        
        # Generate multiple alternative solutions
        alternatives_count = step.get("alternatives_count", 3)
        alternatives = []
        
        for i in range(alternatives_count):
            # Generate a unique alternative
            alternative = self._generate_alternative(
                define, inspire, i, alternatives)
            alternatives.append(alternative)
        
        return {
            "alternatives": alternatives,
            "boundaries": define.get("boundaries", {}),
            "themes_explored": inspire.get("themes", [])
        }
    
    def _evaluation_handler(self, 
                         step: Dict[str, Any], 
                         inputs: Dict[str, Any], 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle evaluation steps."""
        diverge = inputs.get("diverge", {"alternatives": []})
        define = inputs.get("define", {})
        
        # Evaluate each alternative
        evaluations = []
        best_score = -1
        best_index = -1
        
        for i, alternative in enumerate(diverge.get("alternatives", [])):
            # Generate evaluation criteria if not defined
            if "criteria" not in define:
                criteria = self._generate_evaluation_criteria(define)
            else:
                criteria = define["criteria"]
                
            # Evaluate against criteria
            scores = {}
            total_score = 0
            weight_sum = 0
            
            for criterion in criteria:
                criterion_name = criterion["name"]
                weight = criterion.get("weight", 1.0)
                score = self._evaluate_criterion(alternative, criterion)
                
                scores[criterion_name] = score
                total_score += score * weight
                weight_sum += weight
            
            # Calculate weighted average
            avg_score = total_score / weight_sum if weight_sum > 0 else 0
            
            evaluations.append({
                "alternative_index": i,
                "scores": scores,
                "average_score": avg_score,
                "strengths": self._identify_strengths(alternative, scores),
                "weaknesses": self._identify_weaknesses(alternative, scores)
            })
            
            # Track best alternative
            if avg_score > best_score:
                best_score = avg_score
                best_index = i
        
        return {
            "evaluations": evaluations,
            "best_alternative_index": best_index,
            "criteria_used": criteria if "criteria" in locals() else []
        }
    
    def _refinement_handler(self, 
                         step: Dict[str, Any], 
                         inputs: Dict[str, Any], 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle refinement steps."""
        evaluate = inputs.get("evaluate", {})
        
        # Get the best alternative
        best_index = evaluate.get("best_alternative_index", 0)
        alternatives = inputs.get("diverge", {"alternatives": []}).get("alternatives", [])
        
        if not alternatives:
            return {"error": "No alternatives available for refinement"}
            
        best_alternative = alternatives[best_index] if best_index < len(alternatives) else alternatives[0]
        
        # Get evaluation details
        evaluations = evaluate.get("evaluations", [])
        best_evaluation = next((e for e in evaluations if e.get("alternative_index") == best_index), 
                             evaluations[0] if evaluations else {})
        
        # Refine the best alternative
        refined = self._refine_alternative(
            best_alternative, 
            best_evaluation.get("strengths", []),
            best_evaluation.get("weaknesses", [])
        )
        
        return {
            "original": best_alternative,
            "refined": refined,
            "improvements": self._identify_improvements(best_alternative, refined)
        }
    
    def _decomposition_handler(self, 
                            step: Dict[str, Any], 
                            inputs: Dict[str, Any], 
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle problem decomposition steps."""
        gather = inputs.get("gather", {})
        historical = inputs.get("historical", {"insights": []})
        
        # Decompose problem into components
        components = self._decompose_problem(gather)
        
        # Apply historical insights if available
        if "insights" in historical and historical["insights"]:
            components = self._apply_historical_insights(components, historical["insights"])
        
        return {
            "components": components,
            "relationships": self._identify_component_relationships(components),
            "hierarchy": self._establish_component_hierarchy(components)
        }
    
    def _component_analysis_handler(self, 
                                 step: Dict[str, Any], 
                                 inputs: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle component analysis steps."""
        decompose = inputs.get("decompose", {})
        components = decompose.get("components", {})
        
        # Analyze each component independently
        analyses = {}
        
        for component_name, component in components.items():
            analyses[component_name] = self._analyze_component(
                component_name, component, decompose)
        
        return {
            "component_analyses": analyses,
            "cross_cutting_concerns": self._identify_cross_cutting_concerns(analyses),
            "key_findings": self._extract_key_findings(analyses)
        }
    
    def _synthesis_handler(self, 
                        step: Dict[str, Any], 
                        inputs: Dict[str, Any], 
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle synthesis steps."""
        component_analyses = inputs.get("analyze_components", {}).get("component_analyses", {})
        
        # Synthesize findings into cohesive understanding
        synthesis = {
            "integrated_model": self._create_integrated_model(component_analyses),
            "key_insights": self._extract_key_insights(component_analyses),
            "contradictions": self._identify_contradictions(component_analyses),
            "supporting_evidence": self._compile_supporting_evidence(component_analyses)
        }
        
        return synthesis
    
    def _conclusion_handler(self, 
                         step: Dict[str, Any], 
                         inputs: Dict[str, Any], 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle conclusion steps."""
        synthesize = inputs.get("synthesize", {})
        framework = inputs.get("framework", {})
        
        # Draw final conclusions
        conclusions = {
            "main_findings": self._extract_main_findings(synthesize),
            "implications": self._identify_implications(synthesize, framework),
            "recommendations": self._generate_recommendations(synthesize, framework),
            "limitations": self._identify_limitations(
