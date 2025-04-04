import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set, Callable

class ConstraintEnforcer:
    """
    Advanced constraint enforcement system that actively prevents constraint violations
    through predictive analysis and intervention across all system operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Constraint enforcement rules
        self.enforcement_rules = config.get("enforcement_rules", {})
        
        # Default rules if not provided
        if not self.enforcement_rules:
            self.enforcement_rules = self._generate_default_rules()
        
        # Prevention policies
        self.prevention_policies = {
            "memory": self._memory_constraint_policy,
            "execution": self._execution_constraint_policy,
            "mutation": self._mutation_constraint_policy,
            "planning": self._planning_constraint_policy,
            "doctrine": self._doctrine_constraint_policy,
            "interface": self._interface_constraint_policy,
            "resource": self._resource_constraint_policy
        }
        
        # Violation handlers
        self.violation_handlers = {
            "reject": self._handle_rejection,
            "modify": self._handle_modification,
            "log": self._handle_logging,
            "escalate": self._handle_escalation
        }
        
        # Enforcement statistics
        self.stats = {
            "enforcements": 0,
            "violations": 0,
            "rejections": 0,
            "modifications": 0,
            "escalations": 0,
            "domain_stats": {domain: 0 for domain in self.prevention_policies.keys()}
        }
        
        # Recent violations for pattern detection
        self.recent_violations = []
        self.max_violation_history = config.get("max_violation_history", 100)
        
        # Registered constraint callbacks
        self.violation_callbacks = []
        
        # Runtime adaptation parameters
        self.adaptation_factor = config.get("adaptation_factor", 0.1)
        self.constraint_strictness = config.get("constraint_strictness", 0.7)
        
        # Constraint rule verification
        if config.get("verify_rules_at_init", True):
            self._verify_constraint_rules()
    
    def enforce(self, 
               domain: str, 
               operation: Dict[str, Any], 
               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce constraints on an operation within a specific domain.
        
        Args:
            domain: The domain of operation (memory, execution, etc.)
            operation: Operation details
            context: Current execution context
            
        Returns:
            Enforcement result with modifications or rejections
        """
        self.stats["enforcements"] += 1
        
        # Get domain-specific policy
        policy_fn = self.prevention_policies.get(domain, self._default_constraint_policy)
        
        # Apply policy to operation
        result = policy_fn(operation, context)
        
        # Extract violation info
        violations = result.get("violations", [])
        
        # Track domain stats
        if domain in self.stats["domain_stats"]:
            self.stats["domain_stats"][domain] += 1
        
        # Process violations if any
        if violations:
            self.stats["violations"] += len(violations)
            
            # Store recent violations
            for violation in violations:
                self._record_violation(domain, violation, operation)
                
            # Determine most severe handling method
            handling_methods = [v.get("handling", "reject") for v in violations]
            
            # Priority: escalate > reject > modify > log
            if "escalate" in handling_methods:
                handler = self.violation_handlers["escalate"]
                self.stats["escalations"] += 1
            elif "reject" in handling_methods:
                handler = self.violation_handlers["reject"]
                self.stats["rejections"] += 1
            elif "modify" in handling_methods:
                handler = self.violation_handlers["modify"]
                self.stats["modifications"] += 1
            else:
                handler = self.violation_handlers["log"]
                
            # Apply handler to result
            handler_result = handler(violations, result, operation, context)
            
            # Fire violation callbacks
            for callback in self.violation_callbacks:
                try:
                    callback(domain, violations, handler_result)
                except Exception:
                    pass  # Don't let callback errors propagate
                    
            return handler_result
        
        # No violations
        return {
            "allowed": True,
            "original": operation,
            "modified": False,
            "operation": operation,
            "violations": []
        }
    
    def register_violation_callback(self, callback: Callable) -> int:
        """Register a callback for constraint violations."""
        self.violation_callbacks.append(callback)
        return len(self.violation_callbacks)
    
    def update_strictness(self, strictness: float) -> None:
        """Update constraint enforcement strictness (0.0-1.0)."""
        self.constraint_strictness = max(0.0, min(1.0, strictness))
    
    def add_enforcement_rule(self, 
                          domain: str, 
                          rule_name: str, 
                          rule: Dict[str, Any]) -> bool:
        """Add a new enforcement rule."""
        if not self._verify_rule(rule):
            return False
            
        if domain not in self.enforcement_rules:
            self.enforcement_rules[domain] = {}
            
        self.enforcement_rules[domain][rule_name] = rule
        return True
    
    def remove_enforcement_rule(self, domain: str, rule_name: str) -> bool:
        """Remove an enforcement rule."""
        if domain in self.enforcement_rules and rule_name in self.enforcement_rules[domain]:
            del self.enforcement_rules[domain][rule_name]
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get constraint enforcement statistics."""
        return {
            "total_enforcements": self.stats["enforcements"],
            "total_violations": self.stats["violations"],
            "rejection_rate": self.stats["rejections"] / max(1, self.stats["enforcements"]),
            "modification_rate": self.stats["modifications"] / max(1, self.stats["enforcements"]),
            "escalation_rate": self.stats["escalations"] / max(1, self.stats["enforcements"]),
            "domain_stats": self.stats["domain_stats"],
            "active_rules_count": sum(len(rules) for rules in self.enforcement_rules.values()),
            "current_strictness": self.constraint_strictness
        }
    
    def analyze_violation_patterns(self) -> Dict[str, Any]:
        """Analyze recent violation patterns."""
        if not self.recent_violations:
            return {"patterns": [], "hotspots": {}}
            
        # Count violations by rule
        rule_counts = {}
        for violation in self.recent_violations:
            rule = violation.get("rule", "unknown")
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
            
        # Count violations by domain
        domain_counts = {}
        for violation in self.recent_violations:
            domain = violation.get("domain", "unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
        # Identify patterns in sequential violations
        patterns = []
        if len(self.recent_violations) >= 3:
            # Look for repeating sequences of rule violations
            rule_sequence = [v.get("rule", "unknown") for v in self.recent_violations[-10:]]
            
            # Simple pattern detection (repeating pairs or triplets)
            for i in range(len(rule_sequence) - 2):
                if rule_sequence[i] == rule_sequence[i+2] and rule_sequence[i+1] == rule_sequence[i+3]:
                    patterns.append({
                        "type": "repeating_pair",
                        "rules": [rule_sequence[i], rule_sequence[i+1]],
                        "frequency": 2  # Simplified - would count actual occurrences in full implementation
                    })
        
        # Identify hotspots (domains with highest violation rates)
        hotspots = {}
        if domain_counts:
            # Get top 3 domains by violation count
            sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
            for domain, count in sorted_domains[:3]:
                hotspots[domain] = count
        
        return {
            "patterns": patterns,
            "hotspots": hotspots,
            "rule_frequency": rule_counts
        }
    
    def _generate_default_rules(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Generate default constraint enforcement rules."""
        return {
            "memory": {
                "memory_recursion_limit": {
                    "condition": "memory_depth > 3",
                    "handling": "reject",
                    "message": "Memory recursion depth exceeds limit",
                    "severity": "high"
                },
                "memory_token_limit": {
                    "condition": "memory_tokens > 8192",
                    "handling": "modify",
                    "modification": "truncate_memory",
                    "message": "Memory token count exceeds limit",
                    "severity": "medium"
                }
            },
            "execution": {
                "max_execution_time": {
                    "condition": "estimated_time > 300",  # 5 minutes
                    "handling": "reject",
                    "message": "Estimated execution time exceeds limit",
                    "severity": "high"
                },
                "prohibited_operations": {
                    "condition": "contains_prohibited_operation",
                    "handling": "escalate",
                    "message": "Operation contains prohibited elements",
                    "severity": "critical"
                }
            },
            "mutation": {
                "mutation_chain_limit": {
                    "condition": "mutation_chain_length > 5",
                    "handling": "reject",
                    "message": "Mutation chain exceeded maximum length",
                    "severity": "high"
                },
                "high_risk_mutation": {
                    "condition": "risk_score > 0.8",
                    "handling": "modify",
                    "modification": "reduce_mutation_scope",
                    "message": "High risk mutation detected",
                    "severity": "high"
                }
            },
            "planning": {
                "plan_depth_limit": {
                    "condition": "plan_depth > 7",
                    "handling": "modify",
                    "modification": "simplify_plan",
                    "message": "Plan depth exceeds maximum",
                    "severity": "medium"
                },
                "plan_complexity_limit": {
                    "condition": "plan_complexity > 50",
                    "handling": "modify",
                    "modification": "reduce_plan_complexity",
                    "message": "Plan complexity exceeds maximum",
                    "severity": "medium"
                }
            },
            "doctrine": {
                "doctrine_modification_limit": {
                    "condition": "doctrine_mutation_rate > 0.1",
                    "handling": "reject",
                    "message": "Doctrine modification rate exceeds threshold",
                    "severity": "critical"
                },
                "doctrine_consistency": {
                    "condition": "doctrine_consistency_check == false",
                    "handling": "escalate",
                    "message": "Doctrine modification creates inconsistency",
                    "severity": "critical"
                }
            },
            "interface": {
                "rate_limit": {
                    "condition": "requests_per_minute > 60",
                    "handling": "reject",
                    "message": "Interface rate limit exceeded",
                    "severity": "medium"
                },
                "privileged_command": {
                    "condition": "command_privilege_level > user_privilege_level",
                    "handling": "reject",
                    "message": "Insufficient privileges for command",
                    "severity": "high"
                }
            },
            "resource": {
                "memory_usage_limit": {
                    "condition": "system_memory_usage > 0.9",
                    "handling": "modify",
                    "modification": "reduce_memory_footprint",
                    "message": "System memory usage exceeds threshold",
                    "severity": "high"
                },
                "cpu_usage_limit": {
                    "condition": "system_cpu_usage > 0.8",
                    "handling": "modify",
                    "modification": "defer_processing",
                    "message": "System CPU usage exceeds threshold",
                    "severity": "medium"
                }
            }
        }
    
    def _verify_constraint_rules(self) -> bool:
        """Verify that all constraint rules are valid."""
        for domain, rules in self.enforcement_rules.items():
            for rule_name, rule in rules.items():
                if not self._verify_rule(rule):
                    # Invalid rule - remove it
                    self.enforcement_rules[domain].pop(rule_name)
                    
        return True
    
    def _verify_rule(self, rule: Dict[str, Any]) -> bool:
        """Verify that a constraint rule is valid."""
        # Check required fields
        required_fields = ["condition", "handling", "message", "severity"]
        for field in required_fields:
            if field not in rule:
                return False
                
        # Check handling method
        if rule["handling"] not in self.violation_handlers:
            return False
            
        # If handling is modify, check modification field
        if rule["handling"] == "modify" and "modification" not in rule:
            return False
            
        # Check severity level
        if rule["severity"] not in ["low", "medium", "high", "critical"]:
            return False
            
        return True
    
    def _record_violation(self, domain: str, violation: Dict[str, Any], operation: Dict[str, Any]) -> None:
        """Record a constraint violation."""
        # Create violation record
        violation_record = {
            "timestamp": time.time(),
            "domain": domain,
            "rule": violation.get("rule", "unknown"),
            "severity": violation.get("severity", "medium"),
            "message": violation.get("message", ""),
            "operation_id": operation.get("id", "unknown"),
            "operation_type": operation.get("type", "unknown")
        }
        
        # Add to recent violations
        self.recent_violations.append(violation_record)
        
        # Trim if exceeded max size
        while len(self.recent_violations) > self.max_violation_history:
            self.recent_violations.pop(0)
    
    # Domain-specific constraint policies
    
    def _memory_constraint_policy(self, 
                               operation: Dict[str, Any], 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply memory domain constraints."""
        violations = []
        
        # Get rules for this domain
        rules = self.enforcement_rules.get("memory", {})
        
        # Apply each rule
        for rule_name, rule in rules.items():
            # Memory-specific context extraction
            memory_depth = self._calculate_memory_depth(operation, context)
            memory_tokens = self._estimate_token_count(operation)
            memory_cyclic = self._check_memory_cyclicity(operation, context)
            
            # Enhanced context for condition evaluation
            eval_context = {
                "memory_depth": memory_depth,
                "memory_tokens": memory_tokens,
                "memory_cyclic": memory_cyclic,
                "operation": operation,
                "context": context,
                "strictness": self.constraint_strictness
            }
            
            # Evaluate condition
            if self._evaluate_condition(rule["condition"], eval_context):
                # Condition matched - add violation
                violations.append({
                    "rule": rule_name,
                    "handling": rule["handling"],
                    "message": rule["message"],
                    "severity": rule["severity"],
                    "modification": rule.get("modification"),
                    "context": {
                        "memory_depth": memory_depth,
                        "memory_tokens": memory_tokens,
                        "memory_cyclic": memory_cyclic
                    }
                })
        
        return {
            "violations": violations,
            "operation": operation
        }
    
    def _execution_constraint_policy(self, 
                                  operation: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply execution domain constraints."""
        violations = []
        
        # Get rules for this domain
        rules = self.enforcement_rules.get("execution", {})
        
        # Apply each rule
        for rule_name, rule in rules.items():
            # Execution-specific context extraction
            estimated_time = self._estimate_execution_time(operation, context)
            contains_prohibited = self._check_prohibited_operations(operation, context)
            resource_impact = self._estimate_resource_impact(operation, context)
            
            # Enhanced context for condition evaluation
            eval_context = {
                "estimated_time": estimated_time,
                "contains_prohibited_operation": contains_prohibited,
                "resource_impact": resource_impact,
                "operation": operation,
                "context": context,
                "strictness": self.constraint_strictness
            }
            
            # Evaluate condition
            if self._evaluate_condition(rule["condition"], eval_context):
                # Condition matched - add violation
                violations.append({
                    "rule": rule_name,
                    "handling": rule["handling"],
                    "message": rule["message"],
                    "severity": rule["severity"],
                    "modification": rule.get("modification"),
                    "context": {
                        "estimated_time": estimated_time,
                        "contains_prohibited": contains_prohibited,
                        "resource_impact": resource_impact
                    }
                })
        
        return {
            "violations": violations,
            "operation": operation
        }
    
    def _mutation_constraint_policy(self, 
                                 operation: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mutation domain constraints."""
        violations = []
        
        # Get rules for this domain
        rules = self.enforcement_rules.get("mutation", {})
        
        # Extract mutation-specific information
        mutation_chain_length = self._get_mutation_chain_length(operation, context)
        risk_score = self._calculate_mutation_risk(operation, context)
        
        # Apply each rule
        for rule_name, rule in rules.items():
            # Enhanced context for condition evaluation
            eval_context = {
                "mutation_chain_length": mutation_chain_length,
                "risk_score": risk_score,
                "operation": operation,
                "context": context,
                "strictness": self.constraint_strictness
            }
            
            # Evaluate condition
            if self._evaluate_condition(rule["condition"], eval_context):
                # Condition matched - add violation
                violations.append({
                    "rule": rule_name,
                    "handling": rule["handling"],
                    "message": rule["message"],
                    "severity": rule["severity"],
                    "modification": rule.get("modification"),
                    "context": {
                        "mutation_chain_length": mutation_chain_length,
                        "risk_score": risk_score
                    }
                })
        
        return {
            "violations": violations,
            "operation": operation
        }
    
    def _planning_constraint_policy(self, 
                                 operation: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply planning domain constraints."""
        violations = []
        
        # Get rules for this domain
        rules = self.enforcement_rules.get("planning", {})
        
        # Extract planning-specific information
        plan_depth = self._calculate_plan_depth(operation)
        plan_complexity = self._calculate_plan_complexity(operation)
        
        # Apply each rule
        for rule_name, rule in rules.items():
            # Enhanced context for condition evaluation
            eval_context = {
                "plan_depth": plan_depth,
                "plan_complexity": plan_complexity,
                "operation": operation,
                "context": context,
                "strictness": self.constraint_strictness
            }
            
            # Evaluate condition
            if self._evaluate_condition(rule["condition"], eval_context):
                # Condition matched - add violation
                violations.append({
                    "rule": rule_name,
                    "handling": rule["handling"],
                    "message": rule["message"],
                    "severity": rule["severity"],
                    "modification": rule.get("modification"),
                    "context": {
                        "plan_depth": plan_depth,
                        "plan_complexity": plan_complexity
                    }
                })
        
        return {
            "violations": violations,
            "operation": operation
        }
    
    def _doctrine_constraint_policy(self, 
                                 operation: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply doctrine domain constraints."""
        violations = []
        
        # Get rules for this domain
        rules = self.enforcement_rules.get("doctrine", {})
        
        # Extract doctrine-specific information
        doctrine_mutation_rate = self._calculate_doctrine_mutation_rate(context)
        doctrine_consistency_check = self._verify_doctrine_consistency(operation, context)
        
        # Apply each rule
        for rule_name, rule in rules.items():
            # Enhanced context for condition evaluation
            eval_context = {
                "doctrine_mutation_rate": doctrine_mutation_rate,
                "doctrine_consistency_check": doctrine_consistency_check,
                "operation": operation,
                "context": context,
                "strictness": self.constraint_strictness
            }
            
            # Evaluate condition
            if self._evaluate_condition(rule["condition"], eval_context):
                # Condition matched - add violation
                violations.append({
                    "rule": rule_name,
                    "handling": rule["handling"],
                    "message": rule["message"],
                    "severity": rule["severity"],
                    "modification": rule.get("modification"),
                    "context": {
                        "doctrine_mutation_rate": doctrine_mutation_rate,
                        "doctrine_consistency_check": doctrine_consistency_check
                    }
                })
        
        return {
            "violations": violations,
            "operation": operation
        }
    
    def _interface_constraint_policy(self, 
                                  operation: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply interface domain constraints."""
        violations = []
        
        # Get rules for this domain
        rules = self.enforcement_rules.get("interface", {})
        
        # Extract interface-specific information
        requests_per_minute = self._calculate_request_rate(context)
        command_privilege_level = self._get_command_privilege(operation)
        user_privilege_level = self._get_user_privilege(context)
        
        # Apply each rule
        for rule_name, rule in rules.items():
            # Enhanced context for condition evaluation
            eval_context = {
                "requests_per_minute": requests_per_minute,
                "command_privilege_level": command_privilege_level,
                "user_privilege_level": user_privilege_level,
                "operation": operation,
                "context": context,
                "strictness": self.constraint_strictness
            }
            
            # Evaluate condition
            if self._evaluate_condition(rule["condition"], eval_context):
                # Condition matched - add violation
                violations.append({
                    "rule": rule_name,
                    "handling": rule["handling"],
                    "message": rule["message"],
                    "severity": rule["severity"],
                    "modification": rule.get("modification"),
                    "context": {
                        "requests_per_minute": requests_per_minute,
                        "command_privilege_level": command_privilege_level,
                        "user_privilege_level": user_privilege_level
                    }
                })
        
        return {
            "violations": violations,
            "operation": operation
        }
    
    def _resource_constraint_policy(self, 
                                 operation: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply resource domain constraints."""
        violations = []
        
        # Get rules for this domain
        rules = self.enforcement_rules.get("resource", {})
        
        # Extract resource-specific information
        system_memory_usage = context.get("system_memory_usage", 0.5)  # Default to 50% if not provided
        system_cpu_usage = context.get("system_cpu_usage", 0.5)  # Default to 50% if not provided
        
        # Apply each rule
        for rule_name, rule in rules.items():
            # Enhanced context for condition evaluation
            eval_context = {
                "system_memory_usage": system_memory_usage,
                "system_cpu_usage": system_cpu_usage,
                "operation": operation,
                "context": context,
                "strictness": self.constraint_strictness
            }
            
            # Evaluate condition
            if self._evaluate_condition(rule["condition"], eval_context):
                # Condition matched - add violation
                violations.append({
                    "rule": rule_name,
                    "handling": rule["handling"],
                    "message": rule["message"],
                    "severity": rule["severity"],
                    "modification": rule.get("modification"),
                    "context": {
                        "system_memory_usage": system_memory_usage,
                        "system_cpu_usage": system_cpu_usage
                    }
                })
        
        return {
            "violations": violations,
            "operation": operation
        }
    
    def _default_constraint_policy(self, 
                                operation: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Default constraint policy for unknown domains."""
        # No specific checks for unknown domains
        return {
            "violations": [],
            "operation": operation
        }
    
    # Violation handlers
    
    def _handle_rejection(self, 
                       violations: List[Dict[str, Any]], 
                       result: Dict[str, Any],
                       operation: Dict[str, Any],
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle violations by rejecting the operation."""
        # Find the most severe violation
        most_severe = max(violations, key=lambda v: self._severity_score(v["severity"]))
        
        return {
            "allowed": False,
            "original": operation,
            "modified": False,
            "operation": None,
            "violations": violations,
            "reason": most_severe["message"]
        }
    
    def _handle_modification(self, 
                          violations: List[Dict[str, Any]], 
                          result: Dict[str, Any],
                          operation: Dict[str, Any],
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle violations by modifying the operation."""
        # Start with original operation
        modified_operation = operation.copy()
        modifications_applied = []
        
        # Apply modifications for each violation
        for violation in violations:
            if violation["handling"] == "modify":
                modification_type = violation["modification"]
                
                # Apply specific modification
                if modification_type == "truncate_memory":
                    modified_operation = self._apply_truncate_memory(modified_operation, violation["context"])
                    modifications_applied.append("truncate_memory")
                    
                elif modification_type == "reduce_mutation_scope":
                    modified_operation = self._apply_reduce_mutation_scope(modified_operation, violation["context"])
                    modifications_applied.append("reduce_mutation_scope")
                    
                elif modification_type == "simplify_plan":
                    modified_operation = self._apply_simplify_plan(modified_operation, violation["context"])
                    modifications_applied.append("simplify_plan")
                    
                elif modification_type == "reduce_plan_complexity":
                    modified_operation = self._apply_reduce_plan_complexity(modified_operation, violation["context"])
                    modifications_applied.append("reduce_plan_complexity")
                    
                elif modification_type == "reduce_memory_footprint":
                    modified_operation = self._apply_reduce_memory_footprint(modified_operation, violation["context"])
                    modifications_applied.append("reduce_memory_footprint")
                    
                elif modification_type == "defer_processing":
                    modified_operation = self._apply_defer_processing(modified_operation, violation["context"])
                    modifications_applied.append("defer_processing")
        
        return {
            "allowed": True,
            "original": operation,
            "modified": True,
            "operation": modified_operation,
            "violations": violations,
            "modifications": modifications_applied
        }
    
    def _handle_logging(self, 
                     violations: List[Dict[str, Any]], 
                     result: Dict[str, Any],
                     operation: Dict[str, Any],
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle violations by logging them but allowing operation."""
        return {
            "allowed": True,
            "original": operation,
            "modified": False,
            "operation": operation,
            "violations": violations,
            "warning": "Violations logged but operation allowed"
        }
    
    def _handle_escalation(self, 
                        violations: List[Dict[str, Any]], 
                        result: Dict[str, Any],
                        operation: Dict[str, Any],
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle violations by escalating to higher authority."""
        # Find the most severe violation
        most_severe = max(violations, key=lambda v: self._severity_score(v["severity"]))
        
        # Create escalation record
        escalation = {
            "timestamp": time.time(),
            "violations": violations,
            "operation": operation,
            "context_summary": {
                "cycle_id": context.get("cycle_id"),
                "task_id": context.get("task", {}).get("id"),
                "system_id": context.get("system_id")
            },
            "severity": most_severe["severity"],
            "primary_reason": most_severe["message"]
        }
        
        # In a full implementation, this would notify human supervisors
        # or trigger an emergency protocol
        
        return {
            "allowed": False,
            "original": operation,
            "modified": False,
            "operation": None,
            "violations": violations,
            "reason": most_severe["message"],
            "escalated": True,
            "escalation_id": hashlib.md5(str(escalation).encode()).hexdigest()[:8]
        }
    
    # Helper methods for condition evaluation
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition string with the given context."""
        try:
            # Simple cases - direct comparison
            if " > " in condition:
                left, right = condition.split(" > ", 1)
                left_val = self._resolve_variable(left, context)
                right_val = self._resolve_variable(right, context)
                return left_val > right_val
                
            elif " < " in condition:
                left, right = condition.split(" < ", 1)
                left_val = self._resolve_variable(left, context)
                right_val = self._resolve_variable(right, context)
                return left_val < right_val
                
            elif " == " in condition:
                left, right = condition.split(" == ", 1)
                left_val = self._resolve_variable(left, context)
                right_val = self._resolve_variable(right, context)
                return left_val == right_val
                
            elif " != " in condition:
                left, right = condition.split(" != ", 1)
                left_val = self._resolve_variable(left, context)
                right_val = self._resolve_variable(right, context)
                return left_val != right_val
                
            # Direct variable reference (assumed boolean)
            else:
                return bool(self._resolve_variable(condition, context))
                
        except Exception:
            # On any error, default to False
            return False
    
    def _resolve_variable(self, var_name: str, context: Dict[str, Any]) -> Any:
        """Resolve a variable name to its value in context."""
        var_name = var_name.strip()
        
        # Handle numeric literals
        try:
            if "." in var_name:
                return float(var_name)
            else:
                return int(var_name)
        except ValueError:
            pass
        
        # Handle boolean literals
        if var_name.lower() == "true":
            return True
        elif var_name.lower() == "false":
            return False
            
        # Handle context variables
        if var_name in context:
            return context[var_name]
            
        # Not found
        return None
    
    def _severity_score(self, severity: str) -> int:
        """Convert severity string to numeric score."""
        scores = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4
        }
        return scores.get(severity, 0)
    
    # Helper methods for context extraction
    
    def _calculate_memory_depth(self, operation: Dict[str, Any], context: Dict[str, Any]) -> int:
        """Calculate recursive memory depth."""
        # Default implementation - would be more sophisticated in real system
        memory_links = operation.get("memory_links", [])
        return len(memory_links)
    
    def _estimate_token_count(self, operation: Dict[str, Any]) -> int:
        """Estimate token count of an operation."""
        # Simplified implementation - would use a real tokenizer in production
        operation_str = str(operation)
        # Approximate 4 chars per token
        return len(operation_str) // 4
    
    def _check_memory_cyclicity(self, operation: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if operation creates memory cycles."""
        # Simplified check - would use graph analysis in production
        memory_id = operation.get("id", "")
        memory_links = operation.get("memory_links", [])
        return memory_id in memory_links
    
    def _estimate_execution_time(self, operation: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Estimate execution time for an operation in seconds."""
        # Simplified implementation - would use historical data in production
        if "estimated_time" in operation:
            return operation["estimated_time"]
            
        # Default estimates based on operation type
        op_type = operation.get("type", "")
        
        if op_type == "mutation":
            return 10.0  # Mutations take about 10 seconds
        elif op_type == "planning":
            return 5.0  # Planning takes about 5 seconds
        elif op_type == "execution":
            return 15.0  # Execution takes about 15 seconds
        elif op_type == "validation":
            return 2.0  # Validation takes about 2 seconds
            
        # Default estimate
        return 5.0
    
    def _check_prohibited_operations(self, operation: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if operation contains prohibited elements."""
        # Simplified implementation - would use more sophisticated pattern matching
        
        # Check operation type
        op_type = operation.get("type", "").lower()
        prohibited_types = ["system_command", "file_write", "network_access", "doctrine_override"]
        if op_type in prohibited_types:
            return True
            
        # Check content for prohibited patterns
        if "content" in operation:
            content = str(operation["content"]).lower()
            prohibited_patterns = ["exec(", "system(", "os.system", "subprocess", "file.write", "open(", "__import__"]
            
            for pattern in prohibited_patterns:
                if pattern in content:
                    return True
        
        return False
    
    def _estimate_resource_impact(self, operation: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Estimate resource impact of an operation (0-1 scale)."""
        # Simplified implementation - would use more sophisticated modeling
        op_type = operation.get("type", "").lower()
        
        # Base impact by operation type
        if op_type == "mutation":
            base_impact = 0.7  # Mutations are resource intensive
        elif op_type == "planning":
            base_impact = 0.5  # Planning is moderately intensive
        elif op_type == "execution":
            base_impact = 0.6  # Execution is fairly intensive
        else:
            base_impact = 0.3  # Other operations less intensive
            
        # Adjust based on context
        system_load = context.get("system_cpu_usage", 0.5)
        
        # Final impact considers current system load
        return min(1.0, base_impact * (1 + system_load))
    
    def _get_mutation_chain_length(self, operation: Dict[str, Any], context: Dict[str, Any]) -> int:
        """Get length of mutation chain for operation."""
        # Direct extraction if available
        if "mutation_metadata" in operation:
            return operation["mutation_metadata"].get("chain_length", 1)
            
        # From context
        return context.get("mutation_count", 0)
    
    def _calculate_mutation_risk(self, operation: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate risk score for a mutation operation (0-1 scale)."""
        # Base risk factors
        base_risk = 0.3
        
        # Add risk based on mutation chain length
        chain_length = self._get_mutation_chain_length(operation, context)
        chain_risk = min(0.5, 0.1 * chain_length)  # Up to 0.5 for long chains
        
        # Add risk based on mutation scope
        mutation_scale = operation.get("mutation_metadata", {}).get("scale", 0.5)
        scale_risk = mutation_scale * 0.4  # Up to 0.4 for full scale
        
        # Add risk if critical components are affected
        affects_doctrine = "doctrine" in str(operation).lower()
        affects_core = "core" in str(operation).lower()
        component_risk = 0.2 if affects_doctrine or affects_core else 0.0
        
        # Calculate total risk
        risk = base_risk + chain_risk + scale_risk + component_risk
        
        # Cap at 1.0
        return min(1.0, risk)
    
    def _calculate_plan_depth(self, operation: Dict[str, Any]) -> int:
        """Calculate depth of a plan structure."""
        # If not a plan, depth is 0
        if "steps" not in operation:
            return 0
            
        steps = operation.get("steps", [])
        
        # Find maximum nesting
        max_nested = 0
        for step in steps:
            if "substeps" in step:
                nested_depth = 1 + self._calculate_plan_depth({"steps": step["substeps"]})
                max_nested = max(max_nested, nested_depth)
        
        # Total depth is 1 (current level) plus maximum nesting
        return 1 + max_nested
    
    def _calculate_plan_complexity(self, operation: Dict[str, Any]) -> float:
        """Calculate complexity score of a plan (heuristic scale)."""
        # If not a plan, complexity is 0
        if "steps" not in operation:
            return 0
            
        steps = operation.get("steps", [])
        
        # Base complexity from step count
        complexity = len(steps) * 5.0
        
        # Add complexity for each step
        for step in steps:
            # Add for dependencies
            if "inputs" in step:
                complexity += len(step["inputs"]) * 2.0
                
            # Add for substeps
            if "substeps" in step:
                complexity += self._calculate_plan_complexity({"steps": step["substeps"]})
                
            # Add for complex actions
            if "action" in step:
                action_length = len(str(step["action"]))
                complexity += action_length / 50.0  # 1 point per 50 chars
        
        return complexity
    
    def _calculate_doctrine_mutation_rate(self, context: Dict[str, Any]) -> float:
        """Calculate recent doctrine mutation rate."""
        # Try to get from context first
        if "doctrine" in context and "stats" in context["doctrine"]:
            doctrine_stats = context["doctrine"]["stats"]
            if "recent_change_rate" in doctrine_stats:
                return doctrine_stats["recent_change_rate"]
                
        # Default estimate based on context
        total_cycles = context.get("cycle_id", 100)
        doctrine_changes = context.get("doctrine_changes", 0)
        
        if total_cycles > 0:
            return doctrine_changes / total_cycles
        return 0.0
    
    def _verify_doctrine_consistency(self, operation: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Verify that doctrine changes maintain consistency."""
        # If not a doctrine operation, always consistent
        if operation.get("type") != "doctrine_evolution":
            return True
            
        # Direct check if available
        if "consistency_check" in operation:
            return operation["consistency_check"]
            
        # Simple check for contradictions
        if "changes" in operation:
            changes = operation["changes"]
            
            # Check for contradictions in core laws
            if "core_laws" in changes:
                core_laws = changes["core_laws"]
                return self._check_law_consistency(core_laws)
        
        # Default to true if no specific checks failed
        return True
    
    def _check_law_consistency(self, laws: List[str]) -> bool:
        """Check for internal consistency in a list of laws."""
        # Simplified implementation - would use more sophisticated logic
        # Look for direct contradictions
        for i, law1 in enumerate(laws):
            for j, law2 in enumerate(laws):
                if i != j and self._are_laws_contradictory(law1, law2):
                    return False
                    
        return True
    
    def _are_laws_contradictory(self, law1: str, law2: str) -> bool:
        """Check if two laws directly contradict each other."""
        # Simplified implementation - would use NLP/logical analysis
        
        # Check for direct opposites
        opposites = [
            ("always", "never"),
            ("must", "must not"),
            ("required", "prohibited"),
            ("maximize", "minimize")
        ]
        
        law1_lower = law1.lower()
        law2_lower = law2.lower()
        
        for pos, neg in opposites:
            if pos in law1_lower and neg in law2_lower:
                # Check if they refer to the same subject
                return self._have_common_subject(law1_lower, law2_lower)
                
            if neg in law1_lower and pos in law2_lower:
                # Check if they refer to the same subject
                return self._have_common_subject(law1_lower, law2_lower)
                
        return False
    
    def _have_common_subject(self, text1: str, text2: str) -> bool:
        """Check if two texts refer to the same subject."""
        # Simplified implementation - would use NLP
        
        # Extract words excluding common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "in", "to", "for"}
        
        words1 = set(word for word in text1.split() if word not in stop_words)
        words2 = set(word for word in text2.split() if word not in stop_words)
        
        # Check overlap
        common_words = words1.intersection(words2)
        return len(common_words) >= 3  # At least 3 common significant words
    
    def _calculate_request_rate(self, context: Dict[str, Any]) -> float:
        """Calculate current request rate."""
        # Try to get from context
        if "interface_stats" in context:
            return context["interface_stats"].get("requests_per_minute", 0)
            
        # Default implementation would track timestamps of recent requests
        return 10.0  # Default assumption of 10 rpm
    
    def _get_command_privilege(self, operation: Dict[str, Any]) -> int:
        """Get privilege level required for a command."""
        # Direct extraction if available
        if "privilege_required" in operation:
            return operation["privilege_required"]
            
        # Determine based on operation type and content
        op_type = operation.get("type", "").lower()
        
        # Higher privilege operations
        if op_type in ["system", "admin", "configure", "doctrine_evolution"]:
            return 3  # Admin level
            
        # Moderate privilege operations
        if op_type in ["mutation", "fork", "checkpoint"]:
            return 2  # Power user level
            
        # Default privilege level
        return 1  # Standard user level
    
    def _get_user_privilege(self, context: Dict[str, Any]) -> int:
        """Get user's privilege level."""
        # Try to extract from context
        if "user" in context:
            return context["user"].get("privilege_level", 1)
            
        # Default to standard user
        return 1
    
    # Modification handlers for constraint violations
    
    def _apply_truncate_memory(self, operation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply memory truncation modification."""
        # Create a copy of the operation
        modified = operation.copy()
        
        # Extract target token count
        token_limit = 4096  # Default token limit
        
        # If we have context with current token count
        if "memory_tokens" in context:
            current_tokens = context["memory_tokens"]
            # Leave room for about 20% of the limit
            target_tokens = int(token_limit * 0.8)
            
            if current_tokens > target_tokens:
                # Truncate content proportionally
                if "content" in modified and isinstance(modified["content"], str):
                    content = modified["content"]
                    # Simple character-based approximation (4 chars per token)
                    target_chars = target_tokens * 4
                    modified["content"] = content[:target_chars] + "... [truncated]"
                
                # Truncate other large fields if needed
                for field, value in list(modified.items()):
                    if isinstance(value, str) and len(value) > 1000:
                        # Truncate long string fields
                        modified[field] = value[:1000] + "... [truncated]"
                
        return modified
    
    def _apply_reduce_mutation_scope(self, operation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mutation scope reduction."""
        # Create a copy of the operation
        modified = operation.copy()
        
        # Extract risk score for proportional reduction
        risk_score = context.get("risk_score", 0.8)
        
        # Adjust mutation scale
        if "mutation_metadata" in modified:
            metadata = modified["mutation_metadata"]
            if "scale" in metadata:
                # Reduce scale proportionally to risk
                reduction_factor = 0.5 + (0.5 * (1 - risk_score))
                metadata["scale"] *= reduction_factor
                
        # Reduce scope of changes
        if "changes" in modified:
            changes = modified["changes"]
            
            # If there are multiple change types, focus on the least risky
            if isinstance(changes, dict) and len(changes) > 1:
                # Risk ranking of change types (lowest to highest)
                risk_ranking = [
                    "output_transformation",
                    "parameter_adjustment",
                    "step_refinement",
                    "plan_restructure",
                    "constraint_adaptation",
                    "doctrine_modification"
                ]
                
                # Remove higher risk changes
                keys_to_remove = []
                for key in changes:
                    if key in risk_ranking[3:]:  # Higher risk changes
                        keys_to_remove.append(key)
                        
                # Remove highest risk changes first until we've reduced enough
                keys_to_remove.sort(key=lambda k: -risk_ranking.index(k) if k in risk_ranking else -100)
                
                # Remove based on risk score
                num_to_remove = int(len(keys_to_remove) * min(0.9, risk_score))
                for i in range(num_to_remove):
                    if i < len(keys_to_remove):
                        changes.pop(keys_to_remove[i], None)
        
        return modified
    
    def _apply_simplify_plan(self, operation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply plan simplification."""
        # Create a copy of the operation
        modified = operation.copy()
        
        # Get current plan depth
        current_depth = context.get("plan_depth", 7)
        target_depth = 4  # Target maximum depth
        
        # Only process if this is a plan
        if "steps" in modified:
            steps = modified["steps"]
            
            if current_depth > target_depth:
                # Flatten the plan structure
                modified["steps"] = self._flatten_plan_structure(steps, current_depth - target_depth)
        
        return modified
    
    def _flatten_plan_structure(self, steps: List[Dict[str, Any]], levels_to_flatten: int) -> List[Dict[str, Any]]:
        """Flatten a nested plan structure by the specified number of levels."""
        if levels_to_flatten <= 0 or not steps:
            return steps
            
        flattened = []
        
        for step in steps:
            # Add current step
            flattened.append(step.copy())
            
            # If it has substeps, flatten them
            if "substeps" in step:
                substeps = step["substeps"]
                
                if levels_to_flatten == 1:
                    # Just add substeps at this level and remove from parent
                    flattened.extend(substeps)
                    flattened[-len(substeps)-1].pop("substeps")
                else:
                    # Recursively flatten deeper
                    flat_substeps = self._flatten_plan_structure(substeps, levels_to_flatten - 1)
                    flattened.extend(flat_substeps)
                    flattened[-len(flat_substeps)-1].pop("substeps")
        
        return flattened
    
    def _apply_reduce_plan_complexity(self, operation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply plan complexity reduction."""
        # Create a copy of the operation
        modified = operation.copy()
        
        # Get current complexity
        current_complexity = context.get("plan_complexity", 50)
        target_complexity = 40  # Target maximum complexity
        
        # Only process if this is a plan
        if "steps" in modified and current_complexity > target_complexity:
            steps = modified["steps"]
            
            # Simplify steps while maintaining core functionality
            # 1. Merge similar consecutive steps
            merged_steps = self._merge_similar_steps(steps)
            
            # 2. Remove optional steps
            if len(merged_steps) > 3:  # Keep at least 3 steps
                final_steps = self._remove_optional_steps(merged_steps)
                
                # Only use simplified plan if we actually reduced complexity
                if len(final_steps) < len(steps):
                    modified["steps"] = final_steps
            else:
                modified["steps"] = merged_steps
                
            # 3. Simplify step actions
            for step in modified["steps"]:
                if "action" in step and isinstance(step["action"], str) and len(step["action"]) > 100:
                    # Simplify verbose actions
                    step["action"] = self._simplify_action_text(step["action"])
        
        return modified
    
    def _merge_similar_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge similar consecutive steps."""
        if len(steps) <= 1:
            return steps
            
        merged = []
        i = 0
        
        while i < len(steps):
            current = steps[i]
            
            # Check if next step can be merged with current
            if i < len(steps) - 1:
                next_step = steps[i+1]
                
                # Check if steps are similar and can be merged
                if (current.get("type") == next_step.get("type") and
                   "action" in current and "action" in next_step):
                    # Create merged step
                    merged_step = current.copy()
                    merged_step["action"] = f"{current['action']} and {next_step['action']}"
                    
                    # Combine inputs
                    if "inputs" in current or "inputs" in next_step:
                        inputs1 = current.get("inputs", [])
                        inputs2 = next_step.get("inputs", [])
                        merged_step["inputs"] = list(set(inputs1 + inputs2))
                    
                    merged.append(merged_step)
                    i += 2  # Skip both steps
                    continue
            
            # No merge, add current step as is
            merged.append(current)
            i += 1
            
        return merged
    
    def _remove_optional_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove optional steps while preserving core functionality."""
        # Identify essential steps
        essential_indices = set()
        
        # First and last steps are essential
        if steps:
            essential_indices.add(0)
            essential_indices.add(len(steps) - 1)
        
        # Steps with critical types are essential
        critical_types = ["analysis", "execution", "validation", "integration"]
        for i, step in enumerate(steps):
            if step.get("type") in critical_types:
                essential_indices.add(i)
                
        # Steps that other steps depend on are essential
        dependencies = {}
        for i, step in enumerate(steps):
            if "inputs" in step:
                for input_id in step["inputs"]:
                    # Find which step produces this input
                    for j, other_step in enumerate(steps):
                        if other_step.get("id") == input_id:
                            dependencies.setdefault(i, set()).add(j)
                            essential_indices.add(j)  # Mark as essential
        
        # Create a new list with only essential steps
        essential_steps = [step for i, step in enumerate(steps) if i in essential_indices]
        
        return essential_steps
    
    def _simplify_action_text(self, action: str) -> str:
        """Simplify verbose action text."""
        if len(action) <= 100:
            return action
            
        # Try to preserve key information while shortening
        sentences = action.split(". ")
        if len(sentences) > 1:
            # Keep first and last sentence
            if len(sentences) > 3:
                return f"{sentences[0]}. ... {sentences[-1]}"
            else:
                return f"{sentences[0]}. {sentences[-1]}"
                
        # Just truncate with ellipsis
        return action[:97] + "..."
    
    def _apply_reduce_memory_footprint(self, operation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply memory footprint reduction."""
        # Create a copy of the operation
        modified = operation.copy()
        
        # Remove non-essential verbose data
        for field in list(modified.keys()):
            value = modified[field]
            
            # Remove debug information
            if field.startswith("debug_") or field.endswith("_debug"):
                del modified[field]
                continue
                
            # Truncate history lists
            if field.endswith("_history") and isinstance(value, list) and len(value) > 5:
                modified[field] = value[-5:]  # Keep only most recent 5
                continue
                
            # Truncate long metadata
            if field == "metadata" and isinstance(value, dict):
                for meta_key in list(value.keys()):
                    if meta_key.startswith("detailed_") or meta_key.endswith("_trace"):
                        del value[meta_key]
                continue
        
        return modified
    
    def _apply_defer_processing(self, operation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply processing deferral."""
        # Create a copy of the operation
        modified = operation.copy()
        
        # Add deferral metadata
        modified["defer"] = True
        modified["defer_reason"] = "Resource constraint violation"
        modified["defer_timestamp"] = time.time()
        modified["defer_priority"] = "low"
        
        # Reduce immediate processing requirements
        if "processing_flags" in modified:
            flags = modified["processing_flags"]
            
            # Disable resource-intensive options
            for intensive_flag in ["parallelized", "high_precision", "exhaustive", "optimized"]:
                if intensive_flag in flags:
                    flags[intensive_flag] = False
            
            # Enable resource-conserving options
            for conservative_flag in ["cached", "simplified", "reduced"]:
                flags[conservative_flag] = True
        
        return modified
