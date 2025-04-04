import time
import uuid
import traceback
import threading
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable

class ExecutionLoop:
    """
    Core execution loop that orchestrates all NEUROGEN components and manages
    the cognitive cycle through adaptive feedback loops.
    """
    
    def __init__(self, config: Dict[str, Any], components: Dict[str, Any] = None):
        self.config = config
        self.components = components or {}
        
        # System ID
        self.system_id = config.get("system_id", f"neurogen_{uuid.uuid4().hex[:8]}")
        
        # Task queue and execution state
        self.task_queue = []
        self.current_task = None
        self.is_running = False
        self.pause_requested = False
        self.should_terminate = False
        
        # Execution context
        self.context = {
            "system_id": self.system_id,
            "cycle_id": 0,
            "state": {}
        }
        
        # Execution flags
        self.flags = {
            "fork_required": False,
            "rollback_required": False,
            "mutation_required": False,
            "doctrine_evolution_pending": False
        }
        
        # Cycle timing
        self.cycle_interval = config.get("cycle_interval", 1.0)  # seconds
        self.min_cycle_time = config.get("min_cycle_time", 0.1)  # seconds
        self.max_cycle_time = config.get("max_cycle_time", 10.0)  # seconds
        
        # Component shortcuts (initialized in setup)
        self.intent_vector = None
        self.prime_directive = None
        self.planner = None
        self.memory_selector = None
        self.reward_router = None
        self.constraint_controller = None
        self.mutator = None
        self.evolution_auditor = None
        self.fork_engine = None
        self.executor = None
        self.validator = None
        self.meta_judge = None
        self.arbiter = None
        self.agent_memory = None
        
        # Stats tracking
        self.stats = {
            "total_loops": 0,
            "successful_loops": 0,
            "mutation_loops": 0,
            "error_loops": 0,
            "avg_cycle_time": 0.0,
            "critical_errors": 0,
            "forks": 0,
            "rollbacks": 0
        }
        
        # Execution threads and locks
        self.main_thread = None
        self.async_loop = None
        self.mutex = threading.Lock()
        
        # Callbacks
        self.on_cycle_complete = None
        self.on_error = None
        
    def setup(self) -> bool:
        """Set up the execution loop and initialize components."""
        # Set up component shortcuts
        if "intent_vector" in self.components:
            self.intent_vector = self.components["intent_vector"]
            
        if "prime_directive" in self.components:
            self.prime_directive = self.components["prime_directive"]
            
        if "planner" in self.components:
            self.planner = self.components["planner"]
            
        if "memory_selector" in self.components:
            self.memory_selector = self.components["memory_selector"]
            
        if "reward_router" in self.components:
            self.reward_router = self.components["reward_router"]
            
        if "constraint_controller" in self.components:
            self.constraint_controller = self.components["constraint_controller"]
            
        if "mutator" in self.components:
            self.mutator = self.components["mutator"]
            
        if "evolution_auditor" in self.components:
            self.evolution_auditor = self.components["evolution_auditor"]
            
        if "fork_engine" in self.components:
            self.fork_engine = self.components["fork_engine"]
            
        if "executor" in self.components:
            self.executor = self.components["executor"]
            
        if "validator" in self.components:
            self.validator = self.components["validator"]
            
        if "meta_judge" in self.components:
            self.meta_judge = self.components["meta_judge"]
            
        if "arbiter" in self.components:
            self.arbiter = self.components["arbiter"]
            
        if "agent_memory" in self.components:
            self.agent_memory = self.components["agent_memory"]
            
        # Initialize async event loop
        self.async_loop = asyncio.new_event_loop()
        
        # Return success if critical components are available
        return (self.planner is not None and 
                self.executor is not None and 
                self.intent_vector is not None and
                self.prime_directive is not None)
    
    def start(self, initial_task: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start the execution loop in a separate thread.
        
        Args:
            initial_task: Optional initial task to process
            
        Returns:
            Success flag
        """
        if self.is_running:
            return False
            
        # Set up if not already done
        if not self.setup():
            return False
            
        # Add initial task if provided
        if initial_task:
            self.add_task(initial_task)
            
        # Start the main execution thread
        self.is_running = True
        self.main_thread = threading.Thread(target=self._run_loop)
        self.main_thread.daemon = True
        self.main_thread.start()
        
        return True
    
    def stop(self) -> None:
        """Stop the execution loop gracefully."""
        self.should_terminate = True
        
        # Wait for thread to terminate
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5.0)
    
    def pause(self) -> bool:
        """Pause the execution loop."""
        if not self.is_running:
            return False
            
        self.pause_requested = True
        return True
    
    def resume(self) -> bool:
        """Resume the execution loop."""
        if not self.is_running:
            return False
            
        self.pause_requested = False
        return True
    
    def add_task(self, task: Dict[str, Any]) -> str:
        """
        Add a task to the execution queue.
        
        Args:
            task: Task description with type, content, etc.
            
        Returns:
            Task ID
        """
        # Generate task ID if not provided
        if "id" not in task:
            task["id"] = f"task_{uuid.uuid4().hex[:8]}"
            
        # Add timestamp if not provided
        if "created_at" not in task:
            task["created_at"] = time.time()
            
        # Add to queue
        with self.mutex:
            self.task_queue.append(task)
            
        return task["id"]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        with self.mutex:
            status = {
                "is_running": self.is_running,
                "paused": self.pause_requested,
                "queue_size": len(self.task_queue),
                "current_task": self.current_task["id"] if self.current_task else None,
                "cycle_id": self.context["cycle_id"],
                "stats": self.stats.copy(),
                "flags": self.flags.copy()
            }
            
        return status
    
    def register_callback(self, event_type: str, callback: Callable) -> bool:
        """Register a callback for specific events."""
        if event_type == "cycle_complete":
            self.on_cycle_complete = callback
            return True
        elif event_type == "error":
            self.on_error = callback
            return True
            
        return False
    
    def _run_loop(self) -> None:
        """Main execution loop function."""
        try:
            while not self.should_terminate:
                # Check if paused
                if self.pause_requested:
                    time.sleep(0.1)
                    continue
                    
                # Check for tasks
                if not self.current_task and not self.task_queue:
                    time.sleep(0.1)
                    continue
                    
                # Start cycle timer
                cycle_start = time.time()
                self.stats["total_loops"] += 1
                
                # Increment cycle ID
                self.context["cycle_id"] += 1
                cycle_id = self.context["cycle_id"]
                
                # Get next task if needed
                if not self.current_task and self.task_queue:
                    with self.mutex:
                        self.current_task = self.task_queue.pop(0)
                
                # Execute cycle
                try:
                    cycle_result = self._execute_cycle()
                    
                    # Handle cycle result
                    if cycle_result["success"]:
                        self.stats["successful_loops"] += 1
                        
                        # Check if task is complete
                        if cycle_result.get("task_complete", False):
                            self.current_task = None
                            
                    elif cycle_result.get("mutation_applied", False):
                        self.stats["mutation_loops"] += 1
                        
                    else:
                        self.stats["error_loops"] += 1
                        
                        # Check error severity
                        if cycle_result.get("critical_error", False):
                            self.stats["critical_errors"] += 1
                            
                            # Handle critical error based on flags
                            if self.flags["fork_required"]:
                                self._handle_fork()
                            elif self.flags["rollback_required"]:
                                self._handle_rollback()
                            else:
                                # Move to next task on critical error
                                self.current_task = None
                                
                    # Fire cycle complete callback
                    if self.on_cycle_complete:
                        try:
                            self.on_cycle_complete(cycle_id, cycle_result)
                        except Exception:
                            pass  # Ignore callback errors
                            
                except Exception as e:
                    # Handle unhandled exceptions
                    error_info = {
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "cycle_id": cycle_id,
                        "task_id": self.current_task["id"] if self.current_task else None
                    }
                    
                    self.stats["error_loops"] += 1
                    self.stats["critical_errors"] += 1
                    
                    # Fire error callback
                    if self.on_error:
                        try:
                            self.on_error(error_info)
                        except Exception:
                            pass  # Ignore callback errors
                            
                    # Move to next task on unhandled error
                    self.current_task = None
                
                # Update cycle time stats
                cycle_time = time.time() - cycle_start
                self.stats["avg_cycle_time"] = (
                    (self.stats["avg_cycle_time"] * (self.stats["total_loops"] - 1) + 
                     cycle_time) / self.stats["total_loops"]
                )
                
                # Adjust cycle interval if adaptive timing is enabled
                if self.config.get("adaptive_timing", True):
                    self._adjust_cycle_interval(cycle_time)
                    
                # Sleep if cycle was faster than interval
                sleep_time = max(0, self.cycle_interval - cycle_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            # Critical loop failure
            error_info = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "critical": True
            }
            
            # Fire error callback
            if self.on_error:
                try:
                    self.on_error(error_info)
                except Exception:
                    pass  # Ignore callback errors
                    
        finally:
            # Mark as not running
            self.is_running = False
    
    def _execute_cycle(self) -> Dict[str, Any]:
        """Execute a single cognitive cycle."""
        task = self.current_task
        context = self.context.copy()
        
        # Add task to context
        context["task"] = task
        
        # Initialize cycle state
        cycle_state = {
            "success": False,
            "output": None,
            "error": None,
            "task_complete": False,
            "mutation_applied": False,
            "critical_error": False
        }
        
        try:
            # 1. Update Intent Vector
            if self.intent_vector:
                intent_update = self.intent_vector.cycle(context)
                context["intent"] = self.intent_vector
                
                # Check for significant intent shift
                if intent_update.get("significant_shift", False):
                    # Log significant intent shift for analysis
                    cycle_state["intent_shift"] = intent_update["changes"]
            
            # 2. Select Constraints
            if self.constraint_controller:
                constraints = self.constraint_controller.get_constraints(context)
                context["constraints"] = constraints
            
            # 3. Select Relevant Memories
            memories = []
            if self.memory_selector and self.agent_memory:
                memories = self.memory_selector.select(context)
                context["memories"] = memories
            
            # 4. Generate Plan
            plan = None
            if self.planner:
                plan = self.planner.generate_plan(context)
                context["plan"] = plan
            else:
                # Critical error - can't continue without plan
                cycle_state["critical_error"] = True
                cycle_state["error"] = "No planner available"
                return cycle_state
            
            # 5. Execute Plan
            output = None
            execution_success = False
            if self.executor:
                output = self.executor.execute(plan, task)
                context["output"] = output
                
                # Check execution result 
                if isinstance(output, dict) and "error" in output:
                    execution_success = False
                    cycle_state["error"] = output["error"]
                else:
                    execution_success = True
            else:
                # Critical error - can't continue without executor
                cycle_state["critical_error"] = True
                cycle_state["error"] = "No executor available"
                return cycle_state
            
            # 6. Validate Output
            validation_result = {"valid": execution_success}
            if self.validator and execution_success:
                validation_result = self.validator.validate(output, context)
                context["validation"] = validation_result
            
            # 7. Check Doctrinal Alignment
            meta_result = {"valid": True}
            if self.meta_judge and execution_success:
                meta_result = self.meta_judge.evaluate(output, context)
                context["meta_judge"] = meta_result
            
            # 8. Resolve Validation Conflicts
            valid_output = execution_success
            if execution_success:
                if self.arbiter:
                    # Get arbitration between validator and meta-judge
                    arbitration = self.arbiter.arbitrate(
                        validation_result, meta_result, context)
                    context["arbitration"] = arbitration
                    
                    # Use arbitration verdict
                    valid_output = arbitration.get("verdict", "reject") == "accept"
                else:
                    # Without arbiter, both must agree
                    valid_output = validation_result.get("valid", False) and meta_result.get("valid", False)
            
            # 9. Handle Invalid Output
            mutation_applied = False
            if not valid_output:
                # Prepare error information
                error_info = {
                    "type": "validation" if not validation_result.get("valid", False) else "doctrine",
                    "validation": not validation_result.get("valid", False),
                    "doctrine": not meta_result.get("valid", False),
                    "message": validation_result.get("error", meta_result.get("error", "Invalid output")),
                    "details": {**validation_result, **meta_result}
                }
                
                # Save error to context
                context["error"] = error_info
                cycle_state["error"] = error_info
                
                # Check if mutation is appropriate
                should_mutate = self._should_apply_mutation(context)
                
                if should_mutate and self.mutator:
                    # Generate and apply mutation
                    mutation_count = context.get("mutation_count", 0)
                    context["mutation_count"] = mutation_count + 1
                    
                    # Apply mutation
                    mutated_plan = self.mutator.mutate(plan, error_info, context)
                    
                    if not mutated_plan.get("abort_mutation", False):
                        # Update plan with mutation
                        context["plan"] = mutated_plan
                        mutation_applied = True
                        cycle_state["mutation_applied"] = True
                        
                        # Don't mark as failed if mutation was applied
                        cycle_state["success"] = False
                        cycle_state["task_complete"] = False
                        return cycle_state
                    else:
                        # Mutation aborted - mark as critical error
                        cycle_state["critical_error"] = True
                        cycle_state["error"] = "Mutation aborted: " + mutated_plan.get("reason", "Unknown reason")
                        return cycle_state
                else:
                    # No mutation - mark as failed
                    cycle_state["success"] = False
                    cycle_state["task_complete"] = True
                    return cycle_state
            
            # 10. Process Valid Output
            if valid_output:
                # Store successful output
                cycle_state["success"] = True
                cycle_state["output"] = output
                cycle_state["task_complete"] = True
                
                # Memory formation for successful execution
                if self.agent_memory:
                    memory_entry = {
                        "id": f"mem_{uuid.uuid4().hex[:8]}",
                        "type": "execution_result",
                        "task_id": task["id"],
                        "cycle_id": context["cycle_id"],
                        "content": {
                            "task": task,
                            "output": output
                        },
                        "metadata": {
                            "success": True,
                            "created_at": time.time()
                        }
                    }
                    
                    # Add memory
                    self.agent_memory.add(memory_entry)
            
            # 11. Calculate Reward Signal
            context["success"] = valid_output
            
            if self.reward_router:
                reward_signal = self.reward_router.calculate_reward(context)
                context["reward"] = reward_signal
                cycle_state["reward"] = reward_signal
                
                # Update mutation strategy weights if mutation was used
                if mutation_applied and "mutation_metadata" in context["plan"]:
                    mutation_id = context["plan"]["mutation_metadata"]["mutation_id"]
                    self.mutator.update_strategy_weights(
                        mutation_id, valid_output, reward_signal.get("reward", 0))
            
            # 12. Record Evolution Metrics
            if self.evolution_auditor:
                audit_result = self.evolution_auditor.record_cycle(context)
                context["coherence"] = audit_result["coherence"]
                cycle_state["coherence"] = audit_result["coherence"]
                
                # Check for intervention signals
                if audit_result.get("intervention_signals", {}).get("fork_recommended", False):
                    self.flags["fork_required"] = True
                    
                if audit_result.get("intervention_signals", {}).get("rollback_recommended", False):
                    self.flags["rollback_required"] = True
            
            # 13. Update Constraints
            if self.constraint_controller and self.reward_router:
                # Update constraints based on performance
                performance_metrics = {
                    "reward": context.get("reward", {}).get("reward", 0.5),
                    "success_rate": self.stats["successful_loops"] / max(1, self.stats["total_loops"]),
                    "mutation_rate": self.stats["mutation_loops"] / max(1, self.stats["total_loops"]),
                    "coherence": context.get("coherence", 0.5)
                }
                
                updated_constraints = self.constraint_controller.update_constraints(
                    performance_metrics, context.get("drift", None))
                
                cycle_state["constraint_update"] = updated_constraints
                
            return cycle_state
            
        except Exception as e:
            # Handle cycle execution error
            error_info = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "cycle_id": context["cycle_id"],
                "task_id": task["id"] if task else None
            }
            
            cycle_state["critical_error"] = True
            cycle_state["error"] = error_info
            
            # Fire error callback
            if self.on_error:
                try:
                    self.on_error(error_info)
                except Exception:
                    pass  # Ignore callback errors
                    
            return cycle_state
    
    def _should_apply_mutation(self, context: Dict[str, Any]) -> bool:
        """Determine if mutation should be applied."""
        # Get current mutation count
        mutation_count = context.get("mutation_count", 0)
        
        # Check if we've reached maximum mutations
        max_mutations = self.config.get("max_mutations_per_task", 3)
        if mutation_count >= max_mutations:
            return False
            
        # Get error information
        error = context.get("error", {})
        error_type = error.get("type", "unknown")
        
        # Always mutate validation errors on first attempt
        if error_type == "validation" and mutation_count == 0:
            return True
            
        # Check if doctrine violations can be mutated
        if error_type == "doctrine":
            doctrine_error = error.get("doctrine", False)
            validation_error = error.get("validation", False)
            
            # Only mutation if it's purely a doctrine error (not also validation)
            if doctrine_error and not validation_error:
                return mutation_count < 2  # Limit doctrine mutations
        
        # Check if there's a pressure signal from reward router
        if self.reward_router:
            mutation_pressure = self.reward_router.get_mutation_pressure(context)
            pressure_threshold = 0.7 - (0.1 * mutation_count)  # Threshold decreases with more mutations
            
            if mutation_pressure > pressure_threshold:
                return True
        
        # Default behavior
        return mutation_count < 2  # Allow up to 2 mutations by default
    
    def _handle_fork(self) -> None:
        """Handle fork requirement."""
        if not self.fork_engine or not self.current_task:
            return
            
        # Create fork with current context
        fork_result = self.fork_engine.fork(
            "coherence_intervention", self.context)
            
        if fork_result.get("success", False):
            self.stats["forks"] += 1
            self.flags["fork_required"] = False
            
            # Clear current task as it's being handled by forks
            self.current_task = None
    
    def _handle_rollback(self) -> None:
        """Handle rollback requirement."""
        # Simple implementation - just abandon current task
        self.current_task = None
        self.stats["rollbacks"] += 1
        self.flags["rollback_required"] = False
    
    def _adjust_cycle_interval(self, last_cycle_time: float) -> None:
        """Adjust cycle interval based on performance."""
        # Get target interval from config
        target_interval = self.config.get("target_cycle_interval", 1.0)
        
        # Adjust based on last cycle time
        if last_cycle_time > target_interval * 1.5:
            # Cycle took too long, increase interval
            self.cycle_interval = min(
                self.max_cycle_time,
                self.cycle_interval * 1.1
            )
        elif last_cycle_time < target_interval * 0.5:
            # Cycle was very fast, decrease interval
            self.cycle_interval = max(
                self.min_cycle_time,
                self.cycle_interval * 0.9
            )
        else:
            # Gradually move toward target
            self.cycle_interval = (
                0.9 * self.cycle_interval + 
                0.1 * target_interval
            )
