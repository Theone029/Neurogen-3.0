import json
import os
from typing import Dict, Any, List, Optional, Callable

class Neurogen:
    """
    Main NEUROGEN system that integrates all components into a unified architecture.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize NEUROGEN with configuration."""
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize core components
        self.prime_directive = self._init_prime_directive()
        self.agent_memory = self._init_agent_memory()
        self.intent_vector = self._init_intent_vector()
        
        # Initialize validation components
        self.validator = self._init_validator()
        self.meta_judge = self._init_meta_judge()
        self.validation_arbitrator = self._init_validation_arbitrator()
        
        # Initialize mutation components
        self.mutation_memory = self._init_mutation_memory()
        self.mutator = self._init_mutator()
        
        # Initialize drift components
        self.drift_auditor = self._init_drift_auditor()
        self.constraint_controller = self._init_constraint_controller()
        
        # Initialize memory components
        self.memory_selector = self._init_memory_selector()
        self.memory_attention = self._init_memory_attention()
        
        # Initialize evolution components
        self.reward_router = self._init_reward_router()
        self.evolution_auditor = self._init_evolution_auditor()
        
        # Initialize planning components
        self.planner = self._init_planner()
        self.executor = self._init_executor()
        
        # Initialize forking components
        self.fork_engine = self._init_fork_engine()
        
        # Initialize execution loop
        self.execution_loop = self._init_execution_loop()
        
        # System state
        self.initialized = True
        self.stats = {
            "tasks_processed": 0,
            "mutations_applied": 0,
            "forks_triggered": 0,
            "doctrine_updates": 0
        }
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task through the recursive execution loop."""
        self.stats["tasks_processed"] += 1
        
        # Validate task structure
        if not self._validate_task(task):
            return {
                "success": False,
                "error": "Invalid task structure",
                "output": None
            }
        
        # Process task through execution loop
        result = self.execution_loop.process(task)
        
        # Update stats based on result
        if "mutation_count" in result and result["mutation_count"] > 0:
            self.stats["mutations_applied"] += 1
            
        return result
    
    def update_doctrine(self, proposed_doctrine: Dict[str, Any], justification: str) -> Dict[str, Any]:
        """Update the Prime Directive with justification."""
        update_result = self.prime_directive.update(proposed_doctrine, justification)
        
        if update_result["valid"]:
            self.stats["doctrine_updates"] += 1
            
            # Notify evolution auditor
            if self.evolution_auditor:
                self.evolution_auditor.record_doctrine_change(
                    update_result["old_version"],
                    update_result["new_version"],
                    justification
                )
                
        return update_result
    
    def fork_system(self, reason: str) -> Dict[str, Any]:
        """Fork the system due to irreconcilable conflict."""
        if not self.fork_engine:
            return {
                "success": False,
                "error": "Fork engine not initialized",
                "reason": reason
            }
            
        fork_result = self.fork_engine.fork(reason)
        
        if fork_result["success"]:
            self.stats["forks_triggered"] += 1
            
        return fork_result
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state for monitoring."""
        return {
            "stats": self.stats,
            "doctrine": self.prime_directive.get_current_version() if self.prime_directive else None,
            "intent": self.intent_vector.get_vector_as_dict() if self.intent_vector else None,
            "drift": self.drift_auditor.get_drift_trend() if self.drift_auditor else None,
            "coherence": self.evolution_auditor.get_latest_coherence() if self.evolution_auditor else None,
            "memory_stats": self.agent_memory.get_stats() if self.agent_memory else None
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "system": {
                "name": "NEUROGEN",
                "version": "3.0",
                "max_mutations": 3,
                "max_memory": 10000,
                "fork_threshold": 0.7
            },
            "components": {
                "prime_directive": {"path": "configs/prime_directive.json"},
                "agent_memory": {"capacity": 10000},
                "validator": {"conflict_strategy": "conservative"},
                "intent_vector": {
                    "dimensions": {
                        "coherence": 0.8,
                        "knowledge": 0.7,
                        "stability": 0.6,
                        "exploration": 0.4,
                        "efficiency": 0.5,
                        "doctrinal_alignment": 0.9
                    }
                }
            }
        }
    
    def _validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate task structure."""
        if not isinstance(task, dict):
            return False
            
        # Require minimal fields
        if "id" not in task:
            task["id"] = f"task_{hash(str(task))}"
            
        if "type" not in task:
            task["type"] = "default"
            
        return True
    
    # Component initialization methods
    def _init_prime_directive(self):
        from neurogen.core.prime_directive import PrimeDirective
        config = self.config["components"].get("prime_directive", {})
        return PrimeDirective(config.get("path", "configs/prime_directive.json"))
    
    def _init_agent_memory(self):
        from neurogen.memory.agent_memory import AgentMemory
        config = self.config["components"].get("agent_memory", {})
        return AgentMemory(config)
    
    def _init_intent_vector(self):
        from neurogen.core.intent_vector import IntentVector
        config = self.config["components"].get("intent_vector", {})
        return IntentVector(config)
    
    def _init_validator(self):
        from neurogen.validation.validator import Validator
        config = self.config["components"].get("validator", {})
        return Validator(config)
    
    def _init_meta_judge(self):
        from neurogen.validation.meta_judge import MetaJudge
        config = self.config["components"].get("meta_judge", {})
        return MetaJudge(config)
    
    def _init_validation_arbitrator(self):
        from neurogen.validation.validation_arbitrator import ValidationArbitrator
        config = self.config["components"].get("validation_arbitrator", {})
        return ValidationArbitrator(config)
    
    def _init_mutation_memory(self):
        # Implementation depends on mutation_memory module
        return None
    
    def _init_mutator(self):
        from neurogen.mutation.mutator import Mutator
        config = self.config["components"].get("mutator", {})
        return Mutator(
            self.mutation_memory,
            self.reward_router,
            self.constraint_controller,
            config
        )
    
    def _init_drift_auditor(self):
        from neurogen.drift.drift_auditor import DriftAuditor
        config = self.config["components"].get("drift_auditor", {})
        return DriftAuditor(config)
    
    def _init_constraint_controller(self):
        # Implementation depends on constraint_controller module
        return None
    
    def _init_memory_selector(self):
        # Implementation depends on memory_selector module
        return None
    
    def _init_memory_attention(self):
        # Implementation depends on memory_attention module
        return None
    
    def _init_reward_router(self):
        # Implementation depends on reward_router module
        return None
    
    def _init_evolution_auditor(self):
        # Implementation depends on evolution_auditor module
        return None
    
    def _init_planner(self):
        # Implementation depends on planner module
        return None
    
    def _init_executor(self):
        # Implementation depends on executor module
        return None
    
    def _init_fork_engine(self):
        # Implementation depends on fork_engine module
        return None
    
    def _init_execution_loop(self):
        from neurogen.core.execution_loop import ExecutionLoop
        config = self.config["components"].get("execution_loop", {})
        return ExecutionLoop(
            self.intent_vector,
            self.planner,
            self.memory_selector,
            self.memory_attention,
            self.agent_memory,
            self.validator,
            self.meta_judge,
            self.validation_arbitrator,
            self.mutator,
            self.mutation_memory,
            self.drift_auditor,
            self.reward_router,
            self.constraint_controller,
            self.evolution_auditor,
            self.prime_directive,
            self.executor,
            config
        )
