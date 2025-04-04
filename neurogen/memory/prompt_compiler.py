import time
from typing import Dict, List, Any, Optional, Union
import json
import re

class PromptCompiler:
    """Compiles prompts with embedded memory, context, and doctrine for LLM inference."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Prompt templates
        self.templates = config.get("templates", {})
        if not self.templates:
            # Set default template if none provided
            self.templates["default"] = (
                "Task: {task}\n\n"
                "{memory_section}"
                "{doctrine_section}"
                "{constraints_section}"
                "Generate a response that accomplishes this task."
            )
        
        # Memory formatting templates
        self.memory_format = config.get("memory_format", {
            "prefix": "Relevant context from memory:\n",
            "item": "- {content}\n",
            "suffix": "\n"
        })
        
        # Doctrine formatting
        self.doctrine_format = config.get("doctrine_format", {
            "prefix": "System values and guidelines:\n",
            "core_laws": "- {law}\n",
            "alignment": "Focus on: {focus}\nAvoid: {avoid}\n",
            "suffix": "\n"
        })
        
        # Constraints formatting
        self.constraints_format = config.get("constraints_format", {
            "prefix": "Constraints:\n",
            "item": "- {constraint}: {value}\n",
            "suffix": "\n"
        })
        
        # Configuration for token management
        self.token_limits = config.get("token_limits", {
            "total": 4000,
            "memory": 1500,
            "doctrine": 500,
            "constraints": 300,
            "task": 1000
        })
        
        # Token counting function (simplified)
        self.count_tokens = config.get("token_counter", self._default_token_counter)
        
        # Stats tracking
        self.stats = {
            "prompts_compiled": 0,
            "avg_prompt_tokens": 0,
            "avg_memory_count": 0,
            "template_usage": {}
        }
    
    def compile(self, 
               task: Dict[str, Any], 
               memories: List[Dict[str, Any]], 
               doctrine: Dict[str, Any],
               constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile a prompt with memories, doctrine, and constraints.
        
        Args:
            task: Task description and parameters
            memories: Relevant memories to include
            doctrine: Current doctrine version
            constraints: Execution constraints
            
        Returns:
            Compiled prompt with metadata
        """
        # Start compilation timestamp
        start_time = time.time()
        self.stats["prompts_compiled"] += 1
        
        # Select appropriate template
        template_key = task.get("template", "default")
        if template_key not in self.templates:
            template_key = "default"
            
        # Track template usage
        if template_key not in self.stats["template_usage"]:
            self.stats["template_usage"][template_key] = 0
        self.stats["template_usage"][template_key] += 1
        
        template = self.templates[template_key]
        
        # Process task content
        task_content = self._format_task(task)
        task_tokens = self.count_tokens(task_content)
        
        # Determine token budgets
        available_tokens = self.token_limits["total"] - task_tokens
        memory_budget = min(available_tokens * 0.6, self.token_limits["memory"])
        doctrine_budget = min(available_tokens * 0.2, self.token_limits["doctrine"])
        constraints_budget = min(available_tokens * 0.1, self.token_limits["constraints"])
        
        # Process memory section
        memory_section, memory_metadata = self._format_memories(memories, memory_budget)
        
        # Process doctrine section
        doctrine_section, doctrine_metadata = self._format_doctrine(doctrine, doctrine_budget)
        
        # Process constraints section
        constraints_section, constraints_metadata = self._format_constraints(constraints, constraints_budget)
        
        # Fill in template
        prompt = template.format(
            task=task_content,
            memory_section=memory_section,
            doctrine_section=doctrine_section,
            constraints_section=constraints_section
        )
        
        # Calculate token usage
        total_tokens = self.count_tokens(prompt)
        
        # Update statistics
        self.stats["avg_prompt_tokens"] = (
            (self.stats["avg_prompt_tokens"] * (self.stats["prompts_compiled"] - 1) + 
             total_tokens) / self.stats["prompts_compiled"]
        )
        
        self.stats["avg_memory_count"] = (
            (self.stats["avg_memory_count"] * (self.stats["prompts_compiled"] - 1) + 
             memory_metadata["count"]) / self.stats["prompts_compiled"]
        )
        
        # Compile metadata
        metadata = {
            "timestamp": time.time(),
            "compilation_time": time.time() - start_time,
            "template": template_key,
            "token_count": total_tokens,
            "memory": memory_metadata,
            "doctrine": doctrine_metadata,
            "constraints": constraints_metadata
        }
        
        return {
            "prompt": prompt,
            "metadata": metadata
        }
    
    def _format_task(self, task: Dict[str, Any]) -> str:
        """Format task into prompt text."""
        task_type = task.get("type", "general")
        content = task.get("content", "")
        goal = task.get("goal", "")
        
        # Basic task formatting
        if isinstance(content, str):
            formatted = content
        elif isinstance(content, dict):
            # Convert structured content to text
            try:
                formatted = json.dumps(content, indent=2)
            except:
                formatted = str(content)
        else:
            formatted = str(content)
            
        # Add goal if specified
        if goal:
            formatted = f"{formatted}\n\nGoal: {goal}"
            
        # Add task type if needed
        if "include_task_type" in task and task["include_task_type"]:
            formatted = f"Task Type: {task_type}\n\n{formatted}"
            
        return formatted
    
    def _format_memories(self, 
                       memories: List[Dict[str, Any]], 
                       token_budget: int) -> Tuple[str, Dict[str, Any]]:
        """Format memories into prompt text within token budget."""
        if not memories:
            return "", {"count": 0, "tokens": 0}
            
        # Start with prefix
        formatted = self.memory_format["prefix"]
        formatted_tokens = self.count_tokens(formatted)
        
        # Sort memories by relevance or weight
        sorted_memories = sorted(
            memories, 
            key=lambda m: m.get("attention_weight", 0), 
            reverse=True
        )
        
        # Add memories until budget is reached
        included_memories = []
        memory_ids = []
        
        for memory in sorted_memories:
            # Extract content
            content = memory.get("content", "")
            
            # For execution results, extract output
            if memory.get("type") == "execution_result" and isinstance(content, dict):
                content = content.get("output", content)
                
            # Format to string if needed
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, indent=2)
                except:
                    content = str(content)
            
            # Truncate if needed
            max_content_length = self.config.get("max_memory_content_length", 500)
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
                
            # Format memory item
            memory_text = self.memory_format["item"].format(content=content)
            memory_tokens = self.count_tokens(memory_text)
            
            # Check if adding this would exceed budget
            if formatted_tokens + memory_tokens > token_budget:
                break
                
            # Add to prompt
            formatted += memory_text
            formatted_tokens += memory_tokens
            
            # Track for metadata
            included_memories.append(memory)
            memory_ids.append(memory.get("id", "unknown"))
        
        # Add suffix
        formatted += self.memory_format["suffix"]
        formatted_tokens = self.count_tokens(formatted)
        
        # If no memories included, return empty string
        if not included_memories:
            return "", {"count": 0, "tokens": 0}
            
        metadata = {
            "count": len(included_memories),
            "tokens": formatted_tokens,
            "memory_ids": memory_ids
        }
        
        return formatted, metadata
    
    def _format_doctrine(self, 
                       doctrine: Dict[str, Any], 
                       token_budget: int) -> Tuple[str, Dict[str, Any]]:
        """Format doctrine into prompt text within token budget."""
        if not doctrine:
            return "", {"tokens": 0}
            
        # Start with prefix
        formatted = self.doctrine_format["prefix"]
        formatted_tokens = self.count_tokens(formatted)
        
        # Track what we include
        included_laws = 0
        included_alignments = False
        
        # Add core laws
        if "core_laws" in doctrine:
            core_laws = doctrine["core_laws"]
            for law in core_laws:
                law_text = self.doctrine_format["core_laws"].format(law=law)
                law_tokens = self.count_tokens(law_text)
                
                if formatted_tokens + law_tokens > token_budget:
                    break
                    
                formatted += law_text
                formatted_tokens += law_tokens
                included_laws += 1
        
        # Add alignment vectors if space permits
        if "alignment_vectors" in doctrine and formatted_tokens < token_budget:
            alignment = doctrine["alignment_vectors"]
            
            focus_items = alignment.get("optimize_for", [])
            avoid_items = alignment.get("minimize", [])
            
            focus_text = ", ".join(focus_items)
            avoid_text = ", ".join(avoid_items)
            
            alignment_text = self.doctrine_format["alignment"].format(
                focus=focus_text, avoid=avoid_text)
            alignment_tokens = self.count_tokens(alignment_text)
            
            if formatted_tokens + alignment_tokens <= token_budget:
                formatted += alignment_text
                formatted_tokens += alignment_tokens
                included_alignments = True
        
        # Add suffix
        formatted += self.doctrine_format["suffix"]
        formatted_tokens = self.count_tokens(formatted)
        
        metadata = {
            "tokens": formatted_tokens,
            "laws_included": included_laws,
            "alignment_included": included_alignments,
            "doctrine_version": doctrine.get("version_id", "unknown")
        }
        
        return formatted, metadata
    
    def _format_constraints(self, 
                          constraints: Dict[str, Any], 
                          token_budget: int) -> Tuple[str, Dict[str, Any]]:
        """Format constraints into prompt text within token budget."""
        if not constraints:
            return "", {"count": 0, "tokens": 0}
            
        # Start with prefix
        formatted = self.constraints_format["prefix"]
        formatted_tokens = self.count_tokens(formatted)
        
        # Track what we include
        included_constraints = 0
        included_keys = []
        
        # Add constraints
        for key, value in constraints.items():
            constraint_text = self.constraints_format["item"].format(
                constraint=key, value=value)
            constraint_tokens = self.count_tokens(constraint_text)
            
            if formatted_tokens + constraint_tokens > token_budget:
                break
                
            formatted += constraint_text
            formatted_tokens += constraint_tokens
            included_constraints += 1
            included_keys.append(key)
        
        # Add suffix
        formatted += self.constraints_format["suffix"]
        formatted_tokens = self.count_tokens(formatted)
        
        metadata = {
            "count": included_constraints,
            "tokens": formatted_tokens,
            "included_keys": included_keys
        }
        
        return formatted, metadata
    
    def _default_token_counter(self, text: str) -> int:
        """Simple default token counter, assumes ~4 chars per token."""
        return len(text) // 4
    
    def register_template(self, name: str, template: str) -> None:
        """Register a new prompt template."""
        self.templates[name] = template
        
    def register_token_counter(self, counter_function: Callable[[str], int]) -> None:
        """Register a more accurate token counting function."""
        self.count_tokens = counter_function
