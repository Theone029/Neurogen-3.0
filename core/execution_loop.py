import time
import uuid
from typing import Dict, Any, Optional, List, Tuple
import traceback
import json

class ExecutionLoop:
    """Central recursive orchestrator for the NEUROGEN architecture."""
    
    def __init__(self, 
                intent_vector,
                planner,
                memory_selector,
                memory_attention,
                agent_memory,
                validator,
                meta_judge,
                validation_arbitrator,
                mutator,
                mutation_memory,
                drift_auditor,
                reward_signal_router,
                constraint_controller,
                evolution_auditor,
                prime_directive,
                executor,
                config: Dict[str, Any]):
        
        # Core components
        self.intent_vector = intent_vector
        self.planner = planner
        self.memory_selector = memory_selector
        self.memory_attention = memory_attention
        self.agent_memory = agent_memory
        self.validator = validator
        self.meta_judge = meta_judge
        self.validation_arbitrator = validation_arbitrator
        self.mutator = mutator
        self.mutation_memory = mutation_memory
        self.drift_auditor = drift_auditor
        self.reward_router = reward_signal_router
        self.constraint_controller = constraint_controller
        self.evolution_auditor = evolution_auditor
        self.prime_directive = prime_directive
        self.executor = executor
        
        # Configuration
        self.config = config
        self.max_retries = config.get("max_retries", 3)
        self.fork_threshold = config.get("fork_threshold", 0.85)
        
        # Runtime state
        self.loop_count = 0
        self.current_task = None
        self.current_intent = None
        self.current_loop_id = None
        
        # Statistics
        self.stats = {
            "total_loops": 0,
            "successful_loops": 0,
            "validation_failures": 0,
            "mutations": 0,
            "doctrine_changes": 0
        }
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task through the recursive execution loop.
        
        Args:
            task: The task to process with goals, constraints, etc.
            
        Returns:
            Complete result with output and all metadata
        """
        # Initialize loop
        self.current_task = task
        self.current_loop_id = f"loop_{uuid.uuid4().hex[:8]}"
        self.loop_count += 1
        self.stats["total_loops"] += 1
        
        loop_start_time = time.time()
        mutation_count = 0
        execution_trace = []
        
        try:
            # 1. Update intent vector
            self.current_intent = self.intent_vector.update(task, self.prime_directive.get_current_version())
            intent_record = {
                "timestamp": time.time(),
                "vector": self.current_intent.tolist() if hasattr(self.current_intent, "tolist") else self.current_intent,
                "task_type": task.get("type", "unknown")
            }
            execution_trace.append({"step": "intent_updated", "data": intent_record})
            
            # 2. Get constraints for this task
            if self.constraint_controller:
                constraints = self.constraint_controller.get_constraints({
                    "task": task,
                    "intent": self.current_intent,
                    "loop_id": self.current_loop_id
                })
            else:
                constraints = {}
            
            # 3. Select relevant memories
            if self.memory_selector:
                memory_candidates = self.memory_selector.select({
                    "task": task,
                    "intent": self.current_intent,
                    "constraints": constraints
                })
                
                # Weight and filter memories
                if self.memory_attention and memory_candidates:
                    weighted_memories = self.memory_attention.weight_and_filter(
                        memory_candidates,
                        {"task": task, "intent": self.current_intent}
                    )
                else:
                    weighted_memories = memory_candidates
                    
                memory_ids = [m["id"] for m in weighted_memories] if weighted_memories else []
            else:
                weighted_memories = []
                memory_ids = []
            
            execution_trace.append({"step": "memories_selected", "count": len(memory_ids)})
            
            # 4. Generate plan
            plan_context = {
                "task": task,
                "intent": self.current_intent,
                "constraints": constraints,
                "memories": weighted_memories,
                "doctrine": self.prime_directive.get_current_version(),
                "loop_id": self.current_loop_id
            }
            
            plan = self.planner.generate_plan(plan_context)
            execution_trace.append({"step": "plan_generated"})
            
            # 5. Execute plan
            output = None
            validation_result = None
            doctrine_result = None
            
            # Loop for mutation retries
            success = False
            while not success and mutation_count <= self.max_retries:
                try:
                    # Execute the plan
                    execution_start = time.time()
                    output = self.executor.execute(plan, task)
                    execution_time = time.time() - execution_start
                    
                    execution_trace.append({
                        "step": "execution_completed", 
                        "time": execution_time,
                        "mutation_attempt": mutation_count
                    })
                    
                    # 6. Validate output
                    validation_result = self.validator.validate(
                        output=output,
                        output_type=task.get("output_type", "unknown"),
                        context={"task": task, "plan": plan}
                    )
                    
                    # 7. Check doctrinal alignment
                    doctrine_result = self.meta_judge.evaluate(
                        output=output,
                        intent=self.current_intent,
                        doctrine=self.prime_directive.get_current_version(),
                        context={"task": task, "plan": plan}
                    )
                    
                    # 8. Get validation verdict
                    arbitration_result = self.validation_arbitrator.arbitrate(
                        validator_result=validation_result,
                        meta_judge_result=doctrine_result,
                        context={
                            "task": task,
                            "plan": plan,
                            "output": output,
                            "intent": self.current_intent,
                            "mutation_count": mutation_count
                        }
                    )
                    
                    execution_trace.append({
                        "step": "validation_completed",
                        "validator": validation_result["valid"],
                        "doctrine": doctrine_result["valid"],
                        "arbitration": arbitration_result["verdict"]
                    })
                    
                    # Check if we succeeded
                    if arbitration_result["verdict"] == "accept":
                        success = True
                        self.stats["successful_loops"] += 1
                        break
                    else:
                        # Mark failure and prepare for mutation
                        self.stats["validation_failures"] += 1
                        
                        # Create error trace for mutation
                        error_trace = {
                            "type": "validation_failure",
                            "validator": not validation_result["valid"],
                            "meta_judge": not doctrine_result["valid"],
                            "reason": arbitration_result.get("reason", "Unknown validation failure"),
                            "details": {
                                "validator_details": validation_result.get("details", {}),
                                "doctrine_details": doctrine_result.get("details", {})
                            }
                        }
                        
                        # Log failure to mutation memory
                        if self.mutation_memory:
                            self.mutation_memory.record_failure(
                                plan=plan,
                                error=error_trace,
                                context={"task": task, "intent": self.current_intent}
                            )
                        
                        # Attempt mutation if we haven't exceeded retries
                        if mutation_count < self.max_retries:
                            mutation_count += 1
                            self.stats["mutations"] += 1
                            
                            # Generate mutated plan
                            plan = self.mutator.mutate(
                                failed_output=output,
                                error_trace=error_trace,
                                plan=plan,
                                context={"task": task, "intent": self.current_intent}
                            )
                            
                            execution_trace.append({
                                "step": "plan_mutated",
                                "mutation_id": plan.get("mutation_metadata", {}).get("mutation_id", "unknown"),
                                "strategy": plan.get("mutation_metadata", {}).get("strategy", "unknown")
                            })
                        else:
                            # No more retries
                            execution_trace.append({"step": "mutation_retries_exhausted"})
                
                except Exception as e:
                    # Execution error
                    error_trace = {
                        "type": "execution_error",
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                        "mutation_count": mutation_count
                    }
                    
                    execution_trace.append({"step": "execution_error", "error": str(e)})
                    
                    # Log error to mutation memory
                    if self.mutation_memory:
                        self.mutation_memory.record_failure(
                            plan=plan,
                            error=error_trace,
                            context={"task": task, "intent": self.current_intent}
                        )
                    
                    # Attempt mutation if we haven't exceeded retries
                    if mutation_count < self.max_retries:
                        mutation_count += 1
                        self.stats["mutations"] += 1
                        
                        # Generate mutated plan
                        plan = self.mutator.mutate(
                            failed_output=None,  # No output for execution error
                            error_trace=error_trace,
                            plan=plan,
                            context={"task": task, "intent": self.current_intent}
                        )
                        
                        execution_trace.append({
                            "step": "plan_mutated_after_error",
                            "mutation_id": plan.get("mutation_metadata", {}).get("mutation_id", "unknown"),
                            "strategy": plan.get("mutation_metadata", {}).get("strategy", "unknown")
                        })
                    else:
                        # No more retries
                        execution_trace.append({"step": "mutation_retries_exhausted"})
            
            # 9. Calculate drift
            if self.drift_auditor:
                drift_result = self.drift_auditor.measure_drift({
                    "plan": plan,
                    "output": output,
                    "memories_used": memory_ids,
                    "intent": self.current_intent,
                    "success": success
                })
            else:
                drift_result = {"drift": 0.0, "components": {}}
            
            execution_trace.append({"step": "drift_measured", "drift": drift_result["drift"]})
            
            # 10. Calculate reward
            if self.reward_router:
                reward_result = self.reward_router.calculate_reward({
                    "task": task,
                    "plan": plan,
                    "output": output,
                    "success": success,
                    "validation": validation_result,
                    "doctrine": doctrine_result,
                    "drift": drift_result,
                    "execution_time": execution_time if success else None,
                    "mutation_count": mutation_count
                })
            else:
                reward_result = {"reward": 1.0 if success else 0.0, "components": {}}
            
            execution_trace.append({"step": "reward_calculated", "reward": reward_result["reward"]})
            
            # 11. Update evolution auditor
            if self.evolution_auditor:
                coherence_result = self.evolution_auditor.record_cycle({
                    "task": task,
                    "intent": self.current_intent,
                    "plan": plan,
                    "output": output,
                    "success": success,
                    "validation": validation_result,
                    "doctrine": doctrine_result,
                    "drift": drift_result,
                    "reward": reward_result,
                    "mutation_count": mutation_count,
                    "memories_used": memory_ids,
                    "execution_trace": execution_trace
                })
            else:
                coherence_result = {"coherence": None}
            
            execution_trace.append({"step": "evolution_recorded"})
            
            # 12. Store memory of this execution
            if self.agent_memory:
                memory_content = {
                    "task": task,
                    "plan": plan,
                    "output": output,
                    "success": success,
                    "execution_trace": execution_trace
                }
                
                memory_id = self.agent_memory.store(
                    memory_content=memory_content,
                    memory_type="execution_result",
                    task_context=task,
                    metadata={
                        "success": success,
                        "reward": reward_result["reward"],
                        "drift": drift_result["drift"],
                        "mutation_count": mutation_count,
                        "doctrine_version": self.prime_directive.get_current_version()["version_id"],
                        "coherence": coherence_result.get("coherence")
                    }
                )
                
                execution_trace.append({"step": "memory_stored", "memory_id": memory_id})
            
            # 13. Prepare final result
            result = {
                "loop_id": self.current_loop_id,
                "task_id": task.get("id", "unknown"),
                "output": output,
                "success": success,
                "execution_time": time.time() - loop_start_time,
                "justification_trace": self._build_justification_trace(
                    plan, validation_result, doctrine_result),
                "memory_links_used": memory_ids,
                "mutation_id": plan.get("mutation_metadata", {}).get("mutation_id") if mutation_count > 0 else None,
                "mutation_count": mutation_count,
                "reward_score": reward_result["reward"],
                "drift_signature": drift_result["drift"],
                "coherence_score": coherence_result.get("coherence"),
                "doctrine_version": self.prime_directive.get_current_version()["version_id"],
                "execution_trace": execution_trace
            }
            
            return result
            
        except Exception as e:
            # Critical loop failure
            error_trace = {
                "type": "critical_loop_failure",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
            
            execution_trace.append({"step": "critical_failure", "error": str(e)})
            
            # Try to store failure memory if possible
            if self.agent_memory:
                try:
                    self.agent_memory.store(
                        memory_content={"error": error_trace, "task": task},
                        memory_type="critical_error",
                        task_context=task,
                        metadata={"success": False, "critical": True}
                    )
                except:
                    pass  # Don't let memory storage failure cascade
            
            # Return error result
            return {
                "loop_id": self.current_loop_id,
                "task_id": task.get("id", "unknown"),
                "success": False,
                "critical_error": True,
                "error": str(e),
                "execution_time": time.time() - loop_start_time,
                "execution_trace": execution_trace
            }
    
    def _build_justification_trace(self, plan, validation_result, doctrine_result):
        """Build a justification trace for the execution."""
        trace = []
        
        # Add plan justification
        if "justification" in plan:
            trace.append(f"plan: {plan['justification']}")
            
        # Add validation justification
        if validation_result and validation_result.get("valid"):
            trace.append(f"validator: passed")
        
        # Add doctrine justification
        if doctrine_result and doctrine_result.get("valid"):
            if "justification" in doctrine_result:
                trace.append(f"doctrine: {doctrine_result['justification']}")
            else:
                trace.append("doctrine: aligned")
                
        # Add prime directive reference
        trace.append(f"Prime Directive: {self.prime_directive.get_current_version()['version_id']}")
        
        return trace
