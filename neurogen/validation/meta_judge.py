import json
from typing import Dict, Any, List, Tuple, Optional
import hashlib
import time

class MetaJudge:
    """Doctrinal and ethical consistency checker for NEUROGEN outputs."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ethical_rules = config.get("ethical_rules", [])
        self.evaluation_stats = {"pass": 0, "fail": 0, "reasons": {}}
    
    def evaluate(self, 
                output: Any, 
                intent: Any, 
                doctrine: Dict[str, Any], 
                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate output against doctrine and intent.
        
        Args:
            output: The output to evaluate
            intent: Current intent vector
            doctrine: Current prime directive state
            context: Execution context
            
        Returns:
            Evaluation result with details
        """
        evaluation_id = hashlib.md5(f"eval:{time.time()}".encode()).hexdigest()[:12]
        
        # Initialize result
        result = {
            "valid": False,
            "evaluation_id": evaluation_id,
            "checks": [],
            "justification": None,
            "details": {},
            "error": None
        }
        
        try:
            # 1. Check alignment with core laws
            if "core_laws" in doctrine:
                law_checks = self._check_core_laws(output, doctrine["core_laws"], context)
                result["checks"].extend(law_checks)
                
                # If any laws were violated, fail immediately
                failed_laws = [c for c in law_checks if not c["passed"]]
                if failed_laws:
                    violation = failed_laws[0]
                    result["details"]["law_violation"] = violation
                    result["error"] = f"Violated core law: {violation['law']}"
                    self._record_failure(result["error"])
                    return result
            
            # 2. Check alignment vectors
            if "alignment_vectors" in doctrine:
                vector_checks = self._check_alignment_vectors(
                    output, 
                    intent, 
                    doctrine["alignment_vectors"], 
                    context
                )
                result["checks"].extend(vector_checks)
                
                # If any alignment checks failed, fail evaluation
                failed_alignments = [c for c in vector_checks if not c["passed"]]
                if failed_alignments:
                    violation = failed_alignments[0]
                    result["details"]["alignment_violation"] = violation
                    result["error"] = f"Violated alignment vector: {violation['vector']}"
                    self._record_failure(result["error"])
                    return result
            
            # 3. Check ethical rules
            rule_checks = self._check_ethical_rules(output, context)
            result["checks"].extend(rule_checks)
            
            # If any ethical rules were violated, fail evaluation
            failed_rules = [c for c in rule_checks if not c["passed"]]
            if failed_rules:
                violation = failed_rules[0]
                result["details"]["ethical_violation"] = violation
                result["error"] = f"Violated ethical rule: {violation['rule']}"
                self._record_failure(result["error"])
                return result
            
            # 4. Generate justification for passing evaluation
            result["justification"] = self._generate_justification(
                output, intent, doctrine, context, result["checks"]
            )
            
            # All checks passed
            result["valid"] = True
            self.evaluation_stats["pass"] += 1
            return result
            
        except Exception as e:
            result["error"] = f"Meta-judge system error: {str(e)}"
            result["details"]["system_exception"] = str(e)
            self._record_failure("system_error")
            return result
    
    def _check_core_laws(self, 
                        output: Any, 
                        laws: List[str], 
                        context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if output complies with core doctrine laws."""
        checks = []
        
        for law in laws:
            # We'll use a simplified approach for the core framework
            # In a full implementation, this would use more sophisticated logic
            
            law_key = law.lower().replace(" ", "_")[:20]
            law_check = {"type": "core_law", "law": law, "passed": True}
            
            # Define law checking logic
            if "mutation is valid only if" in law.lower():
                if "mutation_metadata" in context.get("plan", {}):
                    mutation_meta = context["plan"]["mutation_metadata"]
                    
                    # Check if entropy was reduced
                    if "entropy_delta" in mutation_meta and mutation_meta["entropy_delta"] > 0:
                        # Check if alignment increased to compensate
                        if "alignment_delta" not in mutation_meta or mutation_meta["alignment_delta"] <= 0:
                            law_check["passed"] = False
                            law_check["reason"] = "Mutation increased entropy without improving alignment"
            
            elif "memory must be preserved" in law.lower():
                # Check if memory was preserved across mutations
                if "memory_links_used" in context and not context.get("memory_links_used"):
                    # No memories used when they should have been
                    task_type = context.get("task", {}).get("type", "")
                    if task_type not in ["initialization", "reset"]:
                        law_check["passed"] = False
                        law_check["reason"] = "No memories preserved during standard execution"
            
            # Add more law checking logic as needed
            
            checks.append(law_check)
        
        return checks
    
    def _check_alignment_vectors(self, 
                               output: Any, 
                               intent: Any, 
                               alignment: Dict[str, List[str]], 
                               context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if output aligns with the alignment vectors."""
        checks = []
        
        # Check optimization factors
        if "optimize_for" in alignment:
            for vector in alignment["optimize_for"]:
                vector_check = {"type": "alignment", "vector": vector, "passed": True}
                
                # Simplified check based on vector type
                if vector == "coherence":
                    # Check if output maintains coherence with context
                    if "coherence_score" in context and context["coherence_score"] < 0.6:
                        vector_check["passed"] = False
                        vector_check["reason"] = f"Coherence score {context['coherence_score']} below threshold"
                
                elif vector == "knowledge":
                    # Check if output leverages available knowledge
                    if "memory_links_used" in context:
                        if len(context["memory_links_used"]) < self.config.get("min_memory_usage", 1):
                            vector_check["passed"] = False
                            vector_check["reason"] = "Insufficient memory utilization"
                
                elif vector == "stability":
                    # Check for stability indicators
                    if "drift" in context and context["drift"] > self.config.get("max_drift", 0.3):
                        vector_check["passed"] = False
                        vector_check["reason"] = f"Drift {context['drift']} exceeds threshold"
                
                checks.append(vector_check)
        
        # Check minimization factors
        if "minimize" in alignment:
            for vector in alignment["minimize"]:
                vector_check = {"type": "alignment", "vector": vector, "passed": True}
                
                # Simplified check based on vector type
                if vector == "entropy":
                    # Check if output keeps entropy low
                    if "entropy" in context and context["entropy"] > self.config.get("max_entropy", 0.7):
                        vector_check["passed"] = False
                        vector_check["reason"] = f"Entropy {context['entropy']} exceeds threshold"
                
                elif vector == "invalid_mutation":
                    # Check mutation validity
                    if "mutation_count" in context and context["mutation_count"] > 0:
                        # If we needed mutations, ensure they improved things
                        if "reward" in context and context["reward"] < self.config.get("min_mutation_reward", 0.3):
                            vector_check["passed"] = False
                            vector_check["reason"] = "Mutations did not yield sufficient improvement"
                
                checks.append(vector_check)
        
        return checks
    
    def _check_ethical_rules(self, 
                           output: Any, 
                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if output complies with configured ethical rules."""
        checks = []
        
        for rule in self.ethical_rules:
            rule_check = {"type": "ethical_rule", "rule": rule["name"], "passed": True}
            
            # Apply rule based on type
            rule_type = rule.get("type", "")
            
            if rule_type == "content_filter":
                # Check for forbidden content
                forbidden = rule.get("forbidden_terms", [])
                output_str = str(output)
                
                for term in forbidden:
                    if term.lower() in output_str.lower():
                        rule_check["passed"] = False
                        rule_check["reason"] = f"Output contains forbidden term: {term}"
                        break
            
            elif rule_type == "safety_guard":
                # Check for safety violations
                # This would be more sophisticated in a real implementation
                if "safety_score" in context and context["safety_score"] < rule.get("min_score", 0.8):
                    rule_check["passed"] = False
                    rule_check["reason"] = f"Safety score {context['safety_score']} below threshold"
            
            checks.append(rule_check)
        
        return checks
    
    def _generate_justification(self, 
                              output: Any, 
                              intent: Any, 
                              doctrine: Dict[str, Any], 
                              context: Dict[str, Any],
                              checks: List[Dict[str, Any]]) -> str:
        """Generate justification for why output meets doctrinal requirements."""
        # Simple justification based on what was checked
        law_checks = [c for c in checks if c["type"] == "core_law"]
        alignment_checks = [c for c in checks if c["type"] == "alignment"]
        ethical_checks = [c for c in checks if c["type"] == "ethical_rule"]
        
        justification = "Output is aligned with doctrine because it "
        
        reasons = []
        if law_checks:
            reasons.append(f"respects all {len(law_checks)} core laws")
        
        if alignment_checks:
            optimize = [c["vector"] for c in alignment_checks 
                      if "vector" in c and c["vector"] in doctrine.get("alignment_vectors", {}).get("optimize_for", [])]
            minimize = [c["vector"] for c in alignment_checks 
                      if "vector" in c and c["vector"] in doctrine.get("alignment_vectors", {}).get("minimize", [])]
            
            if optimize:
                reasons.append(f"optimizes for {', '.join(optimize)}")
            
            if minimize:
                reasons.append(f"minimizes {', '.join(minimize)}")
        
        if ethical_checks:
            reasons.append(f"adheres to {len(ethical_checks)} ethical constraints")
        
        if reasons:
            justification += " and ".join(reasons)
        else:
            justification += "meets all validation criteria"
            
        return justification
    
    def _record_failure(self, reason: str) -> None:
        """Record a failure for statistics."""
        self.evaluation_stats["fail"] += 1
        
        # Normalize reason
        if len(reason) > 50:
            reason = reason[:47] + "..."
            
        if reason not in self.evaluation_stats["reasons"]:
            self.evaluation_stats["reasons"][reason] = 0
            
        self.evaluation_stats["reasons"][reason] += 1
