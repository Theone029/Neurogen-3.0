#!/usr/bin/env python3
import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to Python path if running as script
if __name__ == "__main__":
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from neurogen.core.neurogen import NeuroGen

def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging system."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    logging.basicConfig(level=level, format=log_format)
    logging.info(f"Logging initialized at level {log_level}")

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or use default."""
    default_config = {
        "system_id": "neurogen_test",
        "debug_mode": True,
        "max_cycles": 10,
        "intent_vector": {
            "dimensions": {
                "exploration": 0.4,
                "coherence": 0.8,
                "stability": 0.7,
                "knowledge": 0.6,
                "efficiency": 0.5,
                "abstraction": 0.4,
                "doctrinal_alignment": 0.8
            }
        },
        "mutation": {
            "strategy_weights": {
                "plan_restructure": 0.15,
                "step_refinement": 0.2,
                "constraint_adaptation": 0.1,
                "memory_injection": 0.15,
                "output_transformation": 0.1,
                "error_correction": 0.15,
                "concept_substitution": 0.1,
                "divergent_exploration": 0.05
            },
            "max_mutations_per_task": 3,
            "min_reward_for_persistence": 0.3
        },
        "constraints": {
            "max_memory_depth": 3,
            "max_execution_time": 300,
            "max_fork_count": 2,
            "max_plan_complexity": 50
        },
        "execution": {
            "cycle_interval": 0.5,
            "adaptive_timing": True
        }
    }
    
    if not config_path:
        logging.info("Using default configuration")
        return default_config
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.warning(f"Failed to load config: {str(e)}. Using default configuration.")
        return default_config

def load_task(task_path: Optional[str] = None) -> Dict[str, Any]:
    """Load task from file or use default test task."""
    default_task = {
        "id": "test_task_001",
        "type": "default",
        "goal": "Simulate recursive mutation loop",
        "context": {
            "intent": {
                "coherence": 0.8,
                "exploration": 0.4
            }
        }
    }
    
    if not task_path:
        logging.info("Using default test task")
        return default_task
        
    try:
        with open(task_path, 'r') as f:
            task = json.load(f)
        logging.info(f"Task loaded from {task_path}")
        return task
    except Exception as e:
        logging.warning(f"Failed to load task: {str(e)}. Using default test task.")
        return default_task

def save_results(result: Dict[str, Any], output_path: Optional[str] = None) -> None:
    """Save results to file if path provided."""
    if not output_path:
        return
        
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logging.info(f"Results saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save results: {str(e)}")

def print_system_stats(neurogen_instance: NeuroGen) -> None:
    """Print key system statistics after execution."""
    stats = {}
    
    # Extract stats from key components if available
    if hasattr(neurogen_instance, "execution_loop") and neurogen_instance.execution_loop:
        exec_stats = neurogen_instance.execution_loop.stats
        stats["execution"] = {
            "total_cycles": exec_stats.get("total_loops", 0),
            "successful_cycles": exec_stats.get("successful_loops", 0),
            "mutation_cycles": exec_stats.get("mutation_loops", 0),
            "error_cycles": exec_stats.get("error_loops", 0),
            "avg_cycle_time": round(exec_stats.get("avg_cycle_time", 0), 4)
        }
    
    if hasattr(neurogen_instance, "mutator") and neurogen_instance.mutator:
        mut_stats = neurogen_instance.mutator.stats
        stats["mutation"] = {
            "total_mutations": mut_stats.get("total_mutations", 0),
            "successful_mutations": mut_stats.get("successful_mutations", 0),
            "success_rate": round(mut_stats.get("successful_mutations", 0) / 
                             max(1, mut_stats.get("total_mutations", 1)), 2)
        }
    
    if hasattr(neurogen_instance, "intent_vector") and neurogen_instance.intent_vector:
        stats["intent"] = neurogen_instance.intent_vector.get_vector_as_dict()
    
    # Print formatted stats
    print("\n=== NEUROGEN SYSTEM STATISTICS ===")
    for section, data in stats.items():
        print(f"\n{section.upper()}:")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {data}")

def run_neurogen(
    config_path: Optional[str] = None,
    task_path: Optional[str] = None,
    output_path: Optional[str] = None,
    log_level: str = "INFO",
    visualize: bool = False,
    interactive: bool = False,
    trace_execution: bool = False
) -> Dict[str, Any]:
    """Run NEUROGEN with provided configuration and task."""
    # Setup logging
    setup_logging(log_level)
    
    # Load configuration
    config = load_config(config_path)
    
    # Set trace mode if requested
    if trace_execution:
        config["debug_mode"] = True
        config["trace_execution"] = True
    
    # Initialize the system
    start_time = time.perf_counter()
    neurogen = NeuroGen(config)
    init_time = time.perf_counter() - start_time
    logging.info(f"NEUROGEN initialized in {init_time:.3f} seconds")
    
    # Load task
    task = load_task(task_path)
    
    # Enable visualization if requested
    if visualize and hasattr(neurogen, "enable_visualization"):
        neurogen.enable_visualization()
    
    # Run in interactive mode if requested
    if interactive:
        from neurogen.interface.cli_interface import NeuroGenCLI
        cli = NeuroGenCLI(neurogen)
        cli.cmdloop()
        return {"mode": "interactive"}
    
    # Process task
    logging.info(f"Processing task: {task.get('id', 'unknown')}")
    process_start = time.perf_counter()
    result = neurogen.process(task)
    process_time = time.perf_counter() - process_start
    logging.info(f"Task processing completed in {process_time:.3f} seconds")
    
    # Add timing information to result
    result["timing"] = {
        "initialization": init_time,
        "processing": process_time,
        "total": init_time + process_time
    }
    
    # Print system statistics
    print_system_stats(neurogen)
    
    # Save results if output path provided
    save_results(result, output_path)
    
    # Clean shutdown
    neurogen.shutdown()
    
    return result

def print_banner():
    """Print NEUROGEN banner."""
    banner = """
    ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗  ██████╗ ███████╗███╗   ██╗
    ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔════╝ ██╔════╝████╗  ██║
    ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██║  ███╗█████╗  ██╔██╗ ██║
    ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██║   ██║██╔══╝  ██║╚██╗██║
    ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝╚██████╔╝███████╗██║ ╚████║
    ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═══╝
                                                                            
    Recursive Intelligence Architecture
    """
    print(banner)

def main():
    """Main entry point for NEUROGEN test runner."""
    parser = argparse.ArgumentParser(description="NEUROGEN Test Runner")
    parser.add_argument("-c", "--config", help="Path to configuration file")
    parser.add_argument("-t", "--task", help="Path to task file")
    parser.add_argument("-o", "--output", help="Path to save output results")
    parser.add_argument("-l", "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("-v", "--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--trace", action="store_true", help="Enable execution tracing")
    
    args = parser.parse_args()
    
    print_banner()
    
    try:
        result = run_neurogen(
            config_path=args.config,
            task_path=args.task,
            output_path=args.output,
            log_level=args.log_level,
            visualize=args.visualize,
            interactive=args.interactive,
            trace_execution=args.trace
        )
        
        if not args.interactive:
            print("\n=== NEUROGEN OUTPUT ===")
            if isinstance(result, dict) and "output" in result:
                print(json.dumps(result["output"], indent=2))
            else:
                print(json.dumps(result, indent=2))
                
        return 0
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Error running NEUROGEN: {str(e)}")
        logging.debug(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
