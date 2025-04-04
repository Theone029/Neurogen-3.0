import time
import uuid
from typing import Dict, List, Any, Optional, Tuple

class Planner:
    """Generates execution plans based on task, memory, and intent."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.planning_stats = {
            "total_plans": 0,
            "avg_planning_time": 0,
            "avg_steps_per_plan": 0
        }
        
    def generate_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an execution plan for the current task.
        
        Args:
            context: Contains task, intent, constraints, memories, and doctrine
            
        Returns:
            Execution plan with steps and metadata
        """
        start_time = time.time()
        self.planning_stats["total_plans"] += 1
        
        # Extract context elements
        task = context.get("task", {})
        intent = context.get("intent", {})
        constraints = context.get("constraints", {})
        memories = context.get("memories", [])
        doctrine = context.get("doctrine", {})
        
        # Generate plan ID
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        
        # Determine plan depth based on constraints
        max_depth = constraints.get("max_plan_depth", 
                               self.config.get("default_max_depth", 5))
        
        # Choose planning strategy based on task type
        task_type = task.get("type", "default")
        plan_strategy = self._select_planning_strategy(task_type, intent)
        
        # Generate plan based on selected strategy
        plan = self._generate_plan_by_strategy(
            plan_strategy, 
            task, 
            intent,
            memories, 
            doctrine,
            max_depth
        )
        
        # Add plan metadata
        plan["plan_id"] = plan_id
        plan["created_at"] = time.time()
        plan["strategy"] = plan_strategy
        plan["task_id"] = task.get("id", "unknown")
        plan["doctrine_version"] = doctrine.get("version_id", "unknown")
        plan["memory_count"] = len(memories)
        plan["memory_ids"] = [m["id"] for m in memories] if memories else []
        plan["constraints"] = constraints
        
        # Generate justification
        plan["justification"] = self._generate_justification(
            task, intent, plan_strategy, doctrine, memories)
        
        # Update stats
        planning_time = time.time() - start_time
        self.planning_stats["avg_planning_time"] = (
            (self.planning_stats["avg_planning_time"] * 
             (self.planning_stats["total_plans"] - 1) + 
             planning_time) / self.planning_stats["total_plans"]
        )
        
        steps_count = len(plan.get("steps", []))
        self.planning_stats["avg_steps_per_plan"] = (
            (self.planning_stats["avg_steps_per_plan"] * 
             (self.planning_stats["total_plans"] - 1) + 
             steps_count) / self.planning_stats["total_plans"]
        )
        
        return plan
    
    def _select_planning_strategy(self, 
                               task_type: str, 
                               intent: Any) -> str:
        """Select planning strategy based on task and intent."""
        # Extract intent dimensions if available
        intent_dims = {}
        if hasattr(intent, "get_vector_as_dict"):
            intent_dims = intent.get_vector_as_dict()
        elif isinstance(intent, dict):
            intent_dims = intent
        
        # Strategy selection based on task type and intent
        if task_type == "problem_solving":
            # For problem-solving, use structured approach
            return "structured"
            
        elif task_type == "creative":
            # For creative tasks, prefer exploration
            if intent_dims.get("exploration", 0) > 0.6:
                return "divergent"
            else:
                return "balanced"
                
        elif task_type == "analytical":
            # For analytical tasks, use detail-oriented approach
            return "analytical"
            
        elif task_type == "sequential":
            # For sequential tasks, use step-by-step approach
            return "sequential"
            
        # Default strategy
        return "balanced"
    
    def _generate_plan_by_strategy(self,
                                 strategy: str,
                                 task: Dict[str, Any],
                                 intent: Any,
                                 memories: List[Dict[str, Any]],
                                 doctrine: Dict[str, Any],
                                 max_depth: int) -> Dict[str, Any]:
        """Generate plan using the selected strategy."""
        # Base plan structure
        plan = {
            "goal": task.get("goal", "Complete task successfully"),
            "steps": [],
            "expected_output_type": task.get("output_type", "text")
        }
        
        # Apply strategy-specific planning
        if strategy == "structured":
            plan["steps"] = self._generate_structured_steps(task, memories, max_depth)
            
        elif strategy == "divergent":
            plan["steps"] = self._generate_divergent_steps(task, memories, max_depth)
            plan["fallbacks"] = self._generate_fallbacks(task)
            
        elif strategy == "analytical":
            plan["steps"] = self._generate_analytical_steps(task, memories, max_depth)
            plan["validation_criteria"] = self._generate_validation_criteria(task)
            
        elif strategy == "sequential":
            plan["steps"] = self._generate_sequential_steps(task, memories, max_depth)
            
        else:  # balanced
            plan["steps"] = self._generate_balanced_steps(task, memories, max_depth)
        
        # Add memory utilization plan
        plan["memory_utilization"] = self._plan_memory_utilization(memories, strategy)
        
        # Add output format specification if needed
        if "output_format" in task:
            plan["output_format"] = task["output_format"]
        elif "output_type" in task:
            plan["output_format"] = self._determine_output_format(task["output_type"])
            
        return plan
    
    def _generate_structured_steps(self, 
                                task: Dict[str, Any], 
                                memories: List[Dict[str, Any]],
                                max_depth: int) -> List[Dict[str, Any]]:
        """Generate structured steps for problem-solving tasks."""
        steps = []
        
        # Step 1: Analyze task requirements
        steps.append({
            "id": "analyze",
            "type": "analysis",
            "action": "Analyze task requirements and constraints",
            "inputs": ["task"]
        })
        
        # Step 2: Retrieve relevant information from memories
        if memories:
            steps.append({
                "id": "retrieve",
                "type": "memory_retrieval",
                "action": "Retrieve relevant information from memories",
                "inputs": ["task", "memories"],
                "memory_ids": [m["id"] for m in memories[:5]]  # Use top 5 memories
            })
        
        # Step 3: Formulate solution approach
        steps.append({
            "id": "formulate",
            "type": "planning",
            "action": "Formulate solution approach",
            "inputs": ["analyze", "retrieve"] if memories else ["analyze"]
        })
        
        # Step 4: Execute solution
        steps.append({
            "id": "execute",
            "type": "execution",
            "action": "Execute solution approach",
            "inputs": ["formulate"]
        })
        
        # Step 5: Verify solution
        steps.append({
            "id": "verify",
            "type": "validation",
            "action": "Verify solution against requirements",
            "inputs": ["execute", "analyze"]
        })
        
        return steps[:max_depth]  # Limit to max depth
    
    def _generate_divergent_steps(self, 
                               task: Dict[str, Any], 
                               memories: List[Dict[str, Any]],
                               max_depth: int) -> List[Dict[str, Any]]:
        """Generate steps for creative, exploratory tasks."""
        steps = []
        
        # Step 1: Define creative space
        steps.append({
            "id": "define",
            "type": "boundary_setting",
            "action": "Define creative possibilities and constraints",
            "inputs": ["task"]
        })
        
        # Step 2: Explore inspirations from memories
        if memories:
            steps.append({
                "id": "inspire",
                "type": "memory_inspiration",
                "action": "Draw inspiration from relevant memories",
                "inputs": ["define", "memories"],
                "memory_ids": [m["id"] for m in memories[:3]]
            })
        
        # Step 3: Generate multiple alternatives
        steps.append({
            "id": "diverge",
            "type": "divergent_thinking",
            "action": "Generate multiple alternative solutions",
            "inputs": ["define", "inspire"] if memories else ["define"],
            "alternatives_count": 3
        })
        
        # Step 4: Evaluate alternatives
        steps.append({
            "id": "evaluate",
            "type": "evaluation",
            "action": "Evaluate alternatives against creative criteria",
            "inputs": ["diverge", "define"]
        })
        
        # Step 5: Refine best option
        steps.append({
            "id": "refine",
            "type": "refinement",
            "action": "Refine the best alternative into final solution",
            "inputs": ["evaluate"]
        })
        
        return steps[:max_depth]
    
    def _generate_analytical_steps(self, 
                                task: Dict[str, Any], 
                                memories: List[Dict[str, Any]],
                                max_depth: int) -> List[Dict[str, Any]]:
        """Generate detailed analytical steps."""
        steps = []
        
        # Step 1: Define analytical framework
        steps.append({
            "id": "framework",
            "type": "framework_definition",
            "action": "Define analytical framework and metrics",
            "inputs": ["task"]
        })
        
        # Step 2: Gather relevant data points
        steps.append({
            "id": "gather",
            "type": "data_gathering",
            "action": "Gather relevant data points and context",
            "inputs": ["framework"]
        })
        
        # Step 3: Apply historical insights from memories
        if memories:
            steps.append({
                "id": "historical",
                "type": "memory_analysis",
                "action": "Apply insights from historical data",
                "inputs": ["gather", "memories"],
                "memory_ids": [m["id"] for m in memories[:5]]
            })
        
        # Step 4: Decompose problem into components
        steps.append({
            "id": "decompose",
            "type": "decomposition",
            "action": "Break down problem into analyzable components",
            "inputs": ["gather", "historical"] if memories else ["gather"]
        })
        
        # Step 5: Analyze each component
        steps.append({
            "id": "analyze_components",
            "type": "component_analysis",
            "action": "Analyze each component independently",
            "inputs": ["decompose"]
        })
        
        # Step 6: Synthesize findings
        steps.append({
            "id": "synthesize",
            "type": "synthesis",
            "action": "Combine component analyses into cohesive understanding",
            "inputs": ["analyze_components"]
        })
        
        # Step 7: Draw conclusions
        steps.append({
            "id": "conclude",
            "type": "conclusion",
            "action": "Draw final conclusions from analysis",
            "inputs": ["synthesize", "framework"]
        })
        
        return steps[:max_depth]
    
    def _generate_sequential_steps(self, 
                                task: Dict[str, Any], 
                                memories: List[Dict[str, Any]],
                                max_depth: int) -> List[Dict[str, Any]]:
        """Generate steps for sequential processing tasks."""
        steps = []
        
        # Step 1: Identify sequence requirements
        steps.append({
            "id": "identify",
            "type": "requirement_identification",
            "action": "Identify sequential processing requirements",
            "inputs": ["task"]
        })
        
        # Step 2: Create processing pipeline
        steps.append({
            "id": "pipeline",
            "type": "pipeline_creation",
            "action": "Create processing pipeline with defined stages",
            "inputs": ["identify"]
        })
        
        # Step 3: Configure each stage
        steps.append({
            "id": "configure",
            "type": "stage_configuration",
            "action": "Configure parameters for each pipeline stage",
            "inputs": ["pipeline"]
        })
        
        # Step 4: Apply memory-based optimizations
        if memories:
            steps.append({
                "id": "optimize",
                "type": "memory_optimization",
                "action": "Apply optimizations based on historical runs",
                "inputs": ["configure", "memories"],
                "memory_ids": [m["id"] for m in memories[:3]]
            })
        
        # Step 5: Execute pipeline
        steps.append({
            "id": "execute",
            "type": "pipeline_execution",
            "action": "Execute sequential processing pipeline",
            "inputs": ["optimize"] if memories else ["configure"]
        })
        
        # Step 6: Collect results
        steps.append({
            "id": "collect",
            "type": "result_collection",
            "action": "Collect and organize processing results",
            "inputs": ["execute"]
        })
        
        return steps[:max_depth]
    
    def _generate_balanced_steps(self, 
                              task: Dict[str, Any], 
                              memories: List[Dict[str, Any]],
                              max_depth: int) -> List[Dict[str, Any]]:
        """Generate balanced steps for general-purpose tasks."""
        steps = []
        
        # Step 1: Understand task context
        steps.append({
            "id": "understand",
            "type": "context_understanding",
            "action": "Understand task context and requirements",
            "inputs": ["task"]
        })
        
        # Step 2: Apply relevant memories
        if memories:
            steps.append({
                "id": "apply_memory",
                "type": "memory_application",
                "action": "Apply insights from relevant memories",
                "inputs": ["understand", "memories"],
                "memory_ids": [m["id"] for m in memories[:4]]
            })
        
        # Step 3: Develop solution
        steps.append({
            "id": "develop",
            "type": "solution_development",
            "action": "Develop balanced solution approach",
            "inputs": ["understand", "apply_memory"] if memories else ["understand"]
        })
        
        # Step 4: Implement solution
        steps.append({
            "id": "implement",
            "type": "implementation",
            "action": "Implement solution approach",
            "inputs": ["develop"]
        })
        
        # Step 5: Review and refine
        steps.append({
            "id": "review",
            "type": "review",
            "action": "Review and refine solution",
            "inputs": ["implement", "understand"]
        })
        
        return steps[:max_depth]
    
    def _generate_fallbacks(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate fallback approaches for divergent planning."""
        fallbacks = []
        
        # Fallback 1: Simplify approach
        fallbacks.append({
            "id": "simplify",
            "condition": "complexity_too_high",
            "action": "Reduce complexity by focusing on core elements"
        })
        
        # Fallback 2: Use more conventional approach
        fallbacks.append({
            "id": "conventional",
            "condition": "too_novel",
            "action": "Shift to more conventional solution patterns"
        })
        
        # Fallback 3: Increase constraints
        fallbacks.append({
            "id": "constrain",
            "condition": "too_divergent",
            "action": "Add more constraints to guide creative process"
        })
        
        return fallbacks
    
    def _generate_validation_criteria(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate validation criteria for analytical plans."""
        criteria = []
        
        # Criterion 1: Completeness
        criteria.append({
            "id": "completeness",
            "description": "All aspects of the problem are addressed",
            "threshold": 0.8
        })
        
        # Criterion 2: Logical consistency
        criteria.append({
            "id": "consistency",
            "description": "Analysis maintains logical consistency throughout",
            "threshold": 0.9
        })
        
        # Criterion 3: Evidence support
        criteria.append({
            "id": "evidence",
            "description": "Conclusions are supported by sufficient evidence",
            "threshold": 0.85
        })
        
        # Criterion 4: Applicability
        criteria.append({
            "id": "applicability",
            "description": "Analysis is applicable to the original problem",
            "threshold": 0.8
        })
        
        return criteria
    
    def _plan_memory_utilization(self, 
                              memories: List[Dict[str, Any]], 
                              strategy: str) -> Dict[str, Any]:
        """Plan how memories will be utilized in execution."""
        if not memories:
            return {"count": 0}
        
        memory_plan = {
            "count": len(memories),
            "ids": [m["id"] for m in memories],
            "utilization": {}
        }
        
        # Determine utilization strategy based on planning strategy
        if strategy == "structured":
            memory_plan["utilization"]["pattern"] = "reference"
            memory_plan["utilization"]["description"] = "Use memories as reference examples"
            
        elif strategy == "divergent":
            memory_plan["utilization"]["pattern"] = "inspiration"
            memory_plan["utilization"]["description"] = "Use memories as creative inspiration"
            
        elif strategy == "analytical":
            memory_plan["utilization"]["pattern"] = "evidence"
            memory_plan["utilization"]["description"] = "Use memories as analytical evidence"
            
        elif strategy == "sequential":
            memory_plan["utilization"]["pattern"] = "configuration"
            memory_plan["utilization"]["description"] = "Use memories for pipeline configuration"
            
        else:  # balanced
            memory_plan["utilization"]["pattern"] = "guidance"
            memory_plan["utilization"]["description"] = "Use memories for general guidance"
        
        return memory_plan
    
    def _determine_output_format(self, output_type: str) -> Dict[str, Any]:
        """Determine output format based on type."""
        if output_type == "json":
            return {
                "type": "json",
                "schema_enforced": True
            }
        elif output_type == "text":
            return {
                "type": "text",
                "structure": "free-form"
            }
        elif output_type == "code":
            return {
                "type": "code",
                "syntax_checking": True
            }
        elif output_type == "plan":
            return {
                "type": "plan",
                "requires_steps": True
            }
        else:
            return {
                "type": output_type
            }
    
    def _generate_justification(self,
                             task: Dict[str, Any],
                             intent: Any,
                             strategy: str,
                             doctrine: Dict[str, Any],
                             memories: List[Dict[str, Any]]) -> str:
        """Generate justification for plan based on context."""
        # Extract intent dimensions if available
        intent_dims = {}
        if hasattr(intent, "get_vector_as_dict"):
            intent_dims = intent.get_vector_as_dict()
        elif isinstance(intent, dict):
            intent_dims = intent
        
        # Get top dimensions
        top_dims = sorted(
            intent_dims.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2] if intent_dims else []
        
        # Build justification
        justification = f"Plan using {strategy} strategy"
        
        # Add intent justification
        if top_dims:
            dims_str = ", ".join([f"{d[0]} ({d[1]:.2f})" for d in top_dims])
            justification += f" aligned with intent dimensions: {dims_str}"
        
        # Add memory justification
        if memories:
            justification += f", utilizing {len(memories)} relevant memories"
        
        # Add doctrine reference
        if "core_laws" in doctrine and doctrine["core_laws"]:
            first_law = doctrine["core_laws"][0]
            justification += f", in accordance with doctrine: '{first_law[:50]}...'"
        
        return justification
