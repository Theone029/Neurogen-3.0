import cmd
import sys
import os
import json
import logging
import time
import threading
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable

class NeuroGenCLI(cmd.Cmd):
    """
    Command-line interface for the NEUROGEN system providing advanced control,
    monitoring, and diagnostic capabilities with minimal overhead.
    """
    
    intro = """
╔═══════════════════════════════════════════════╗
║ NEUROGEN Command Interface                    ║
║ Advanced Recursive Intelligence Architecture  ║
║                                               ║
║ Type 'help' or '?' for command list           ║
║ Type 'quit' or Ctrl-D to exit                 ║
╚═══════════════════════════════════════════════╝
"""
    prompt = "neurogen> "
    
    def __init__(self, neurogen_instance=None):
        super().__init__()
        self.neurogen = neurogen_instance
        self.event_listeners = {}
        self.output_format = "text"  # Options: text, json
        self.async_loop = None
        self.event_thread = None
        self.event_thread_running = False
        self.log_level = logging.INFO
        self.max_history = 100
        self.command_history = []
        self.visualization_enabled = False
        self.auto_status_interval = 0  # 0 = disabled
        self.auto_status_timer = None
        
        # Configure logging
        self._setup_logging()
        
        # Start event thread
        self._start_event_thread()
    
    # Core system commands
    
    def do_init(self, arg):
        """Initialize NEUROGEN system with config file path: init [config_path]"""
        args = arg.split()
        config_path = args[0] if args else "config/neurogen_config.json"
        
        if not os.path.exists(config_path):
            self.perror(f"Config file not found: {config_path}")
            return
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            if self.neurogen is None:
                from neurogen.core.neurogen import NeuroGen
                self.neurogen = NeuroGen(config)
                self.poutput("NEUROGEN system initialized successfully")
            else:
                self.neurogen.reconfigure(config)
                self.poutput("NEUROGEN system reconfigured successfully")
                
        except Exception as e:
            self.perror(f"Initialization error: {str(e)}")
    
    def do_start(self, arg):
        """Start NEUROGEN system execution: start [initial_task_path]"""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized. Use 'init' first.")
            return
            
        args = arg.split()
        task_path = args[0] if args else None
        
        initial_task = None
        if task_path:
            try:
                with open(task_path, 'r') as f:
                    initial_task = json.load(f)
            except Exception as e:
                self.perror(f"Error loading task: {str(e)}")
                return
        
        try:
            success = self.neurogen.start(initial_task)
            if success:
                self.poutput("NEUROGEN system started successfully")
                
                # Register for core events
                self.neurogen.execution_loop.register_callback(
                    "cycle_complete", self._handle_cycle_event)
                self.neurogen.execution_loop.register_callback(
                    "error", self._handle_error_event)
            else:
                self.perror("Failed to start NEUROGEN system")
        except Exception as e:
            self.perror(f"Start error: {str(e)}")
    
    def do_stop(self, arg):
        """Stop NEUROGEN system execution: stop"""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        try:
            self.neurogen.stop()
            self.poutput("NEUROGEN system stopped")
        except Exception as e:
            self.perror(f"Stop error: {str(e)}")
    
    def do_pause(self, arg):
        """Pause NEUROGEN system execution: pause"""
        if not self.neurogen or not self.neurogen.execution_loop:
            self.perror("NEUROGEN system not running")
            return
            
        try:
            success = self.neurogen.execution_loop.pause()
            if success:
                self.poutput("NEUROGEN system paused")
            else:
                self.perror("Failed to pause NEUROGEN system")
        except Exception as e:
            self.perror(f"Pause error: {str(e)}")
    
    def do_resume(self, arg):
        """Resume paused NEUROGEN system execution: resume"""
        if not self.neurogen or not self.neurogen.execution_loop:
            self.perror("NEUROGEN system not running")
            return
            
        try:
            success = self.neurogen.execution_loop.resume()
            if success:
                self.poutput("NEUROGEN system resumed")
            else:
                self.perror("Failed to resume NEUROGEN system")
        except Exception as e:
            self.perror(f"Resume error: {str(e)}")
    
    def do_status(self, arg):
        """Show current NEUROGEN system status: status [component]"""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        args = arg.split()
        component = args[0] if args else None
        
        try:
            if component:
                # Show status for specific component
                self._show_component_status(component)
            else:
                # Show overall system status
                status = self.neurogen.get_status()
                self._format_and_print(status)
                
        except Exception as e:
            self.perror(f"Status error: {str(e)}")
    
    def do_task(self, arg):
        """Add a task to the execution queue: task [task_file_path]"""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        args = arg.split()
        
        if not args:
            self.perror("Task path required")
            return
            
        task_path = args[0]
        
        try:
            with open(task_path, 'r') as f:
                task = json.load(f)
                
            task_id = self.neurogen.execution_loop.add_task(task)
            self.poutput(f"Task added to queue with ID: {task_id}")
            
        except Exception as e:
            self.perror(f"Task error: {str(e)}")
    
    # Memory commands
    
    def do_memory(self, arg):
        """Memory operations: memory [list|get|purge|stats] [args]"""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        args = arg.split()
        if not args:
            self.perror("Memory operation required: list, get, purge, or stats")
            return
            
        operation = args[0]
        
        try:
            if operation == "list":
                # List memories, optionally filtered
                filter_by = args[1] if len(args) > 1 else None
                self._list_memories(filter_by)
                
            elif operation == "get":
                # Get specific memory by ID
                if len(args) < 2:
                    self.perror("Memory ID required")
                    return
                    
                memory_id = args[1]
                self._get_memory(memory_id)
                
            elif operation == "purge":
                # Purge memories
                confirm = len(args) > 1 and args[1] == "--confirm"
                if not confirm:
                    self.poutput("Warning: This will delete all memories.")
                    self.poutput("Use 'memory purge --confirm' to proceed.")
                    return
                    
                self._purge_memories()
                
            elif operation == "stats":
                # Show memory statistics
                self._show_memory_stats()
                
            else:
                self.perror(f"Unknown memory operation: {operation}")
                
        except Exception as e:
            self.perror(f"Memory error: {str(e)}")
    
    # Doctrine commands
    
    def do_doctrine(self, arg):
        """Doctrine operations: doctrine [show|evolve|lock|unlock|history]"""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        args = arg.split()
        if not args:
            self.perror("Doctrine operation required")
            return
            
        operation = args[0]
        
        try:
            if operation == "show":
                # Show current doctrine
                self._show_doctrine()
                
            elif operation == "evolve":
                # Evolve doctrine with changes from file
                if len(args) < 2:
                    self.perror("Evolution file path required")
                    return
                    
                changes_path = args[1]
                self._evolve_doctrine(changes_path)
                
            elif operation == "lock":
                # Lock doctrine
                self._lock_doctrine()
                
            elif operation == "unlock":
                # Unlock doctrine
                self._unlock_doctrine()
                
            elif operation == "history":
                # Show doctrine evolution history
                self._show_doctrine_history()
                
            else:
                self.perror(f"Unknown doctrine operation: {operation}")
                
        except Exception as e:
            self.perror(f"Doctrine error: {str(e)}")
    
    # Mutation commands
    
    def do_mutation(self, arg):
        """Mutation operations: mutation [stats|strategies|reset]"""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        args = arg.split()
        if not args:
            self.perror("Mutation operation required")
            return
            
        operation = args[0]
        
        try:
            if operation == "stats":
                # Show mutation statistics
                self._show_mutation_stats()
                
            elif operation == "strategies":
                # Show mutation strategies
                self._show_mutation_strategies()
                
            elif operation == "reset":
                # Reset mutation statistics
                self._reset_mutation_stats()
                
            else:
                self.perror(f"Unknown mutation operation: {operation}")
                
        except Exception as e:
            self.perror(f"Mutation error: {str(e)}")
    
    # Fork commands
    
    def do_fork(self, arg):
        """Fork operations: fork [list|status|evaluate|create]"""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        args = arg.split()
        if not args:
            self.perror("Fork operation required")
            return
            
        operation = args[0]
        
        try:
            if operation == "list":
                # List active forks
                self._list_forks()
                
            elif operation == "status":
                # Show fork status
                if len(args) < 2:
                    self.perror("Fork ID required")
                    return
                    
                fork_id = args[1]
                self._show_fork_status(fork_id)
                
            elif operation == "evaluate":
                # Evaluate forks
                fork_pair_id = args[1] if len(args) > 1 else None
                self._evaluate_forks(fork_pair_id)
                
            elif operation == "create":
                # Create fork
                if len(args) < 2:
                    self.perror("Reason required")
                    return
                    
                reason = args[1]
                self._create_fork(reason)
                
            else:
                self.perror(f"Unknown fork operation: {operation}")
                
        except Exception as e:
            self.perror(f"Fork error: {str(e)}")
    
    # Intent commands
    
    def do_intent(self, arg):
        """Intent operations: intent [show|update|reset|history]"""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        args = arg.split()
        if not args:
            self.perror("Intent operation required")
            return
            
        operation = args[0]
        
        try:
            if operation == "show":
                # Show current intent vector
                self._show_intent()
                
            elif operation == "update":
                # Update intent dimension
                if len(args) < 3:
                    self.perror("Dimension and value required")
                    return
                    
                dimension = args[1]
                try:
                    value = float(args[2])
                except ValueError:
                    self.perror("Value must be a float")
                    return
                    
                self._update_intent(dimension, value)
                
            elif operation == "reset":
                # Reset intent to defaults
                self._reset_intent()
                
            elif operation == "history":
                # Show intent evolution history
                self._show_intent_history()
                
            else:
                self.perror(f"Unknown intent operation: {operation}")
                
        except Exception as e:
            self.perror(f"Intent error: {str(e)}")
    
    # Constraint commands
    
    def do_constraint(self, arg):
        """Constraint operations: constraint [list|set|reset|enforce]"""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        args = arg.split()
        if not args:
            self.perror("Constraint operation required")
            return
            
        operation = args[0]
        
        try:
            if operation == "list":
                # List current constraints
                self._list_constraints()
                
            elif operation == "set":
                # Set constraint value
                if len(args) < 3:
                    self.perror("Constraint name and value required")
                    return
                    
                name = args[1]
                value_str = args[2]
                
                # Parse value based on type
                if value_str.lower() == "true":
                    value = True
                elif value_str.lower() == "false":
                    value = False
                else:
                    try:
                        if "." in value_str:
                            value = float(value_str)
                        else:
                            value = int(value_str)
                    except ValueError:
                        value = value_str
                
                self._set_constraint(name, value)
                
            elif operation == "reset":
                # Reset constraints to defaults
                self._reset_constraints()
                
            elif operation == "enforce":
                # Test constraint enforcement
                if len(args) < 3:
                    self.perror("Domain and operation file required")
                    return
                    
                domain = args[1]
                operation_file = args[2]
                self._test_constraint_enforcement(domain, operation_file)
                
            else:
                self.perror(f"Unknown constraint operation: {operation}")
                
        except Exception as e:
            self.perror(f"Constraint error: {str(e)}")
    
    # System monitoring commands
    
    def do_monitor(self, arg):
        """Monitoring operations: monitor [start|stop|interval] [seconds]"""
        args = arg.split()
        if not args:
            self.perror("Monitor operation required")
            return
            
        operation = args[0]
        
        if operation == "start":
            # Start automatic status monitoring
            interval = int(args[1]) if len(args) > 1 else 5
            self._start_monitoring(interval)
            
        elif operation == "stop":
            # Stop automatic status monitoring
            self._stop_monitoring()
            
        elif operation == "interval":
            # Change monitoring interval
            if len(args) < 2:
                self.perror("Interval in seconds required")
                return
                
            try:
                interval = int(args[1])
                self._set_monitor_interval(interval)
            except ValueError:
                self.perror("Interval must be an integer")
                
        else:
            self.perror(f"Unknown monitor operation: {operation}")
    
    # Event commands
    
    def do_event(self, arg):
        """Event operations: event [listen|unlisten|list|trigger]"""
        args = arg.split()
        if not args:
            self.perror("Event operation required")
            return
            
        operation = args[0]
        
        try:
            if operation == "listen":
                # Register event listener
                if len(args) < 2:
                    self.perror("Event type required")
                    return
                    
                event_type = args[1]
                self._register_event_listener(event_type)
                
            elif operation == "unlisten":
                # Unregister event listener
                if len(args) < 2:
                    self.perror("Event type required")
                    return
                    
                event_type = args[1]
                self._unregister_event_listener(event_type)
                
            elif operation == "list":
                # List registered event listeners
                self._list_event_listeners()
                
            elif operation == "trigger":
                # Manually trigger event (for testing)
                if len(args) < 3:
                    self.perror("Event type and data file required")
                    return
                    
                event_type = args[1]
                data_file = args[2]
                self._trigger_test_event(event_type, data_file)
                
            else:
                self.perror(f"Unknown event operation: {operation}")
                
        except Exception as e:
            self.perror(f"Event error: {str(e)}")
    
    # System configuration
    
    def do_config(self, arg):
        """Configuration operations: config [show|set|save|load]"""
        args = arg.split()
        if not args:
            self.perror("Config operation required")
            return
            
        operation = args[0]
        
        try:
            if operation == "show":
                # Show current configuration
                self._show_config()
                
            elif operation == "set":
                # Set configuration value
                if len(args) < 3:
                    self.perror("Path and value required")
                    return
                    
                path = args[1]
                value_str = args[2]
                
                # Parse value based on type
                if value_str.lower() == "true":
                    value = True
                elif value_str.lower() == "false":
                    value = False
                else:
                    try:
                        if "." in value_str:
                            value = float(value_str)
                        else:
                            value = int(value_str)
                    except ValueError:
                        value = value_str
                
                self._set_config(path, value)
                
            elif operation == "save":
                # Save current configuration to file
                file_path = args[1] if len(args) > 1 else "config/neurogen_config.json"
                self._save_config(file_path)
                
            elif operation == "load":
                # Load configuration from file
                if len(args) < 2:
                    self.perror("Config file path required")
                    return
                    
                file_path = args[1]
                self._load_config(file_path)
                
            else:
                self.perror(f"Unknown config operation: {operation}")
                
        except Exception as e:
            self.perror(f"Config error: {str(e)}")
    
    # Debug and diagnostic commands
    
    def do_debug(self, arg):
        """Debug operations: debug [level|trace|dump|profile|history]"""
        args = arg.split()
        if not args:
            self.perror("Debug operation required")
            return
            
        operation = args[0]
        
        try:
            if operation == "level":
                # Set debug level
                if len(args) < 2:
                    self.perror("Debug level required")
                    return
                    
                level = args[1].upper()
                self._set_debug_level(level)
                
            elif operation == "trace":
                # Enable/disable execution tracing
                if len(args) < 2:
                    self.perror("Enable/disable required (on/off)")
                    return
                    
                enable = args[1].lower() in ("on", "true", "yes", "1")
                self._set_execution_tracing(enable)
                
            elif operation == "dump":
                # Dump system state to file
                file_path = args[1] if len(args) > 1 else f"neurogen_dump_{int(time.time())}.json"
                self._dump_system_state(file_path)
                
            elif operation == "profile":
                # Show performance profile
                self._show_performance_profile()
                
            elif operation == "history":
                # Show command history
                self._show_command_history()
                
            else:
                self.perror(f"Unknown debug operation: {operation}")
                
        except Exception as e:
            self.perror(f"Debug error: {str(e)}")
    
    # Utility commands
    
    def do_output(self, arg):
        """Set output format: output [text|json]"""
        if not arg:
            self.poutput(f"Current output format: {self.output_format}")
            return
            
        format_type = arg.strip().lower()
        
        if format_type in ("text", "json"):
            self.output_format = format_type
            self.poutput(f"Output format set to {format_type}")
        else:
            self.perror("Valid formats: text, json")
    
    def do_log(self, arg):
        """Log a message: log [info|warning|error] message"""
        args = arg.split()
        if len(args) < 2:
            self.perror("Log level and message required")
            return
            
        level = args[0].lower()
        message = " ".join(args[1:])
        
        if level == "info":
            logging.info(message)
        elif level == "warning":
            logging.warning(message)
        elif level == "error":
            logging.error(message)
        else:
            self.perror("Valid log levels: info, warning, error")
    
    def do_clear(self, arg):
        """Clear the screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def do_version(self, arg):
        """Show NEUROGEN version information"""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        try:
            version_info = self.neurogen.get_version_info()
            self._format_and_print(version_info)
        except Exception as e:
            self.perror(f"Version error: {str(e)}")
            if hasattr(self.neurogen, "version"):
                self.poutput(f"Version: {self.neurogen.version}")
    
    def do_exit(self, arg):
        """Exit the NEUROGEN command interface"""
        self._cleanup()
        return True
        
    def do_quit(self, arg):
        """Exit the NEUROGEN command interface"""
        return self.do_exit(arg)
    
    def do_EOF(self, arg):
        """Exit on Ctrl-D"""
        self.poutput("\nExiting...")
        return self.do_exit(arg)
    
    # Helper methods
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_format = '%(asctime)s [%(levelname)s] %(message)s'
        logging.basicConfig(
            level=self.log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("neurogen_cli.log")
            ]
        )
    
    def _start_event_thread(self):
        """Start the async event thread."""
        if self.event_thread is not None:
            return
            
        self.event_thread_running = True
        self.event_thread = threading.Thread(target=self._run_event_loop)
        self.event_thread.daemon = True
        self.event_thread.start()
    
    def _run_event_loop(self):
        """Run the async event loop."""
        self.async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.async_loop)
        
        try:
            while self.event_thread_running:
                self.async_loop.run_until_complete(asyncio.sleep(0.1))
        finally:
            self.async_loop.close()
    
    def _cleanup(self):
        """Clean up resources before exit."""
        # Stop the event thread
        self.event_thread_running = False
        if self.event_thread:
            self.event_thread.join(timeout=1.0)
            
        # Stop the monitoring timer
        self._stop_monitoring()
        
        # Stop the NEUROGEN system if running
        if self.neurogen:
            try:
                self.neurogen.stop()
            except Exception:
                pass
    
    def _format_and_print(self, data):
        """Format and print data according to the current output format."""
        if self.output_format == "json":
            json_str = json.dumps(data, indent=2)
            self.poutput(json_str)
        else:
            # Text format
            if isinstance(data, dict):
                self._print_dict(data)
            elif isinstance(data, list):
                self._print_list(data)
            else:
                self.poutput(str(data))
    
    def _print_dict(self, data, indent=0):
        """Print a dictionary in text format."""
        for key, value in data.items():
            if isinstance(value, dict):
                self.poutput(" " * indent + f"{key}:")
                self._print_dict(value, indent + 2)
            elif isinstance(value, list):
                self.poutput(" " * indent + f"{key}:")
                self._print_list(value, indent + 2)
            else:
                self.poutput(" " * indent + f"{key}: {value}")
    
    def _print_list(self, data, indent=0):
        """Print a list in text format."""
        for i, item in enumerate(data):
            if isinstance(item, dict):
                self.poutput(" " * indent + f"{i}:")
                self._print_dict(item, indent + 2)
            elif isinstance(item, list):
                self.poutput(" " * indent + f"{i}:")
                self._print_list(item, indent + 2)
            else:
                self.poutput(" " * indent + f"- {item}")
    
    def _handle_cycle_event(self, cycle_id, cycle_result):
        """Handle cycle completion event."""
        if "cycle_complete" not in self.event_listeners:
            return
            
        event_data = {
            "type": "cycle_complete",
            "cycle_id": cycle_id,
            "success": cycle_result.get("success", False),
            "timestamp": time.time()
        }
        
        self.poutput(f"[EVENT] Cycle {cycle_id} completed: {'Success' if cycle_result.get('success', False) else 'Failure'}")
        
        # Trigger async event processing
        if self.async_loop:
            asyncio.run_coroutine_threadsafe(
                self._process_event("cycle_complete", event_data), 
                self.async_loop
            )
    
    def _handle_error_event(self, error_info):
        """Handle error event."""
        if "error" not in self.event_listeners:
            return
            
        event_data = {
            "type": "error",
            "error": error_info.get("error", "Unknown error"),
            "critical": error_info.get("critical", False),
            "timestamp": time.time()
        }
        
        self.poutput(f"[ERROR] {error_info.get('error', 'Unknown error')}")
        
        # Trigger async event processing
        if self.async_loop:
            asyncio.run_coroutine_threadsafe(
                self._process_event("error", event_data), 
                self.async_loop
            )
    
    async def _process_event(self, event_type, event_data):
        """Process an event asynchronously."""
        # Notify event listeners
        if event_type in self.event_listeners:
            for listener in self.event_listeners[event_type]:
                try:
                    if asyncio.iscoroutinefunction(listener):
                        await listener(event_data)
                    else:
                        listener(event_data)
                except Exception as e:
                    logging.error(f"Error in event listener: {str(e)}")
    
    def _show_component_status(self, component):
        """Show status for a specific component."""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        try:
            if component == "memory":
                if not hasattr(self.neurogen, "agent_memory"):
                    self.perror("Memory component not available")
                    return
                    
                status = self.neurogen.agent_memory.get_stats()
                
            elif component == "execution":
                if not hasattr(self.neurogen, "execution_loop"):
                    self.perror("Execution loop not available")
                    return
                    
                status = self.neurogen.execution_loop.get_status()
                
            elif component == "intent":
                if not hasattr(self.neurogen, "intent_vector"):
                    self.perror("Intent vector not available")
                    return
                    
                status = self.neurogen.intent_vector.get_vector_as_dict()
                
            elif component == "doctrine":
                if not hasattr(self.neurogen, "prime_directive"):
                    self.perror("Prime directive not available")
                    return
                    
                status = self.neurogen.prime_directive.get_evolution_stats()
                
            elif component == "mutation":
                if not hasattr(self.neurogen, "mutator"):
                    self.perror("Mutator not available")
                    return
                    
                status = self.neurogen.mutator.stats
                
            elif component == "evolution":
                if not hasattr(self.neurogen, "evolution_auditor"):
                    self.perror("Evolution auditor not available")
                    return
                    
                status = self.neurogen.evolution_auditor.get_system_state_report()
                
            else:
                self.perror(f"Unknown component: {component}")
                return
                
            self._format_and_print(status)
            
        except Exception as e:
            self.perror(f"Status error: {str(e)}")
    
    def _list_memories(self, filter_by):
        """List memories, optionally filtered."""
        if not hasattr(self.neurogen, "agent_memory"):
            self.perror("Memory component not available")
            return
            
        memories = self.neurogen.agent_memory.get_all_memories()
        
        if filter_by:
            # Apply filter
            filtered = []
            for memory in memories:
                content_str = str(memory.get("content", "")).lower()
                if filter_by.lower() in content_str:
                    filtered.append(memory)
            memories = filtered
            
        # Format for display
        memory_list = []
        for memory in memories:
            memory_list.append({
                "id": memory.get("id", "unknown"),
                "type": memory.get("type", "unknown"),
                "created_at": memory.get("metadata", {}).get("created_at", 0),
                "access_count": memory.get("access_count", 0)
            })
            
        self._format_and_print(memory_list)
    
    def _get_memory(self, memory_id):
        """Get specific memory by ID."""
        if not hasattr(self.neurogen, "agent_memory"):
            self.perror("Memory component not available")
            return
            
        memory = self.neurogen.agent_memory.get_memory(memory_id)
        if memory:
            self._format_and_print(memory)
        else:
            self.perror(f"Memory not found: {memory_id}")
    
    def _purge_memories(self):
        """Purge all memories."""
        if not hasattr(self.neurogen, "agent_memory"):
            self.perror("Memory component not available")
            return
            
        self.neurogen.agent_memory.clear()
        self.poutput("All memories purged")
    
    def _show_memory_stats(self):
        """Show memory statistics."""
        if not hasattr(self.neurogen, "agent_memory"):
            self.perror("Memory component not available")
            return
            
        stats = self.neurogen.agent_memory.get_stats()
        self._format_and_print(stats)
    
    def _show_doctrine(self):
        """Show current doctrine."""
        if not hasattr(self.neurogen, "prime_directive"):
            self.perror("Prime directive not available")
            return
            
        doctrine = self.neurogen.prime_directive.get_current_version()
        self._format_and_print(doctrine)
    
    def _evolve_doctrine(self, changes_path):
        """Evolve doctrine with changes from file."""
        if not hasattr(self.neurogen, "prime_directive"):
            self.perror("Prime directive not available")
            return
            
        try:
            with open(changes_path, 'r') as f:
                changes = json.load(f)
                
            justification = changes.pop("justification", "CLI-initiated evolution")
            coherence_impact = changes.pop("coherence_impact", 0.0)
            approval = True  # CLI evolution is pre-approved
            
            result = self.neurogen.prime_directive.evolve_doctrine(
                changes, justification, coherence_impact, approval)
                
            self._format_and_print(result)
            
        except Exception as e:
            self.perror(f"Evolution error: {str(e)}")
    
    def _lock_doctrine(self):
        """Lock doctrine."""
        if not hasattr(self.neurogen, "prime_directive"):
            self.perror("Prime directive not available")
            return
            
        self.neurogen.prime_directive.lock_doctrine()
        self.poutput("Doctrine locked")
    
    def _unlock_doctrine(self):
        """Unlock doctrine."""
        if not hasattr(self.neurogen, "prime_directive"):
            self.perror("Prime directive not available")
            return
            
        self.neurogen.prime_directive.unlock_doctrine()
        self.poutput("Doctrine unlocked")
    
    def _show_doctrine_history(self):
        """Show doctrine evolution history."""
        if not hasattr(self.neurogen, "prime_directive"):
            self.perror("Prime directive not available")
            return
            
        history = self.neurogen.prime_directive.version_history
        
        # Format for display
        history_display = []
        for version in history:
            history_display.append({
                "version_id": version.get("version_id", "unknown"),
                "created_at": version.get("created_at", 0),
                "core_laws_count": len(version.get("core_laws", []))
            })
            
        self._format_and_print(history_display)
    
    def _show_mutation_stats(self):
        """Show mutation statistics."""
        if not hasattr(self.neurogen, "mutator"):
            self.perror("Mutator not available")
            return
            
        stats = self.neurogen.mutator.stats
        self._format_and_print(stats)
    
    def _show_mutation_strategies(self):
        """Show mutation strategies."""
        if not hasattr(self.neurogen, "mutator"):
            self.perror("Mutator not available")
            return
            
        strategies = self.neurogen.mutator.strategy_weights
        self._format_and_print(strategies)
    
    def _reset_mutation_stats(self):
        """Reset mutation statistics."""
        if not hasattr(self.neurogen, "mutator"):
            self.perror("Mutator not available")
            return
            
        # Reset stats
        self.neurogen.mutator.stats = {
            "total_mutations": 0,
            "successful_mutations": 0,
            "strategy_usage": {name: 0 for name in self.neurogen.mutator.mutation_strategies},
            "average_improvements": {name: 0.0 for name in self.neurogen.mutator.mutation_strategies},
            "strategy_success_rates": {name: 0.0 for name in self.neurogen.mutator.mutation_strategies},
            "mutation_chains": {
                "length_1": 0,
                "length_2": 0,
                "length_3+": 0
            }
        }
        
        self.poutput("Mutation statistics reset")
    
    def _list_forks(self):
        """List active forks."""
        if not hasattr(self.neurogen, "fork_engine"):
            self.perror("Fork engine not available")
            return
            
        stats = self.neurogen.fork_engine.get_fork_stats()
        active_forks = stats.get("active_forks", 0)
        
        self.poutput(f"Active forks: {active_forks}")
        
        # List active forks
        if active_forks > 0:
            forks = []
            for fork_id in self.neurogen.fork_engine.active_forks:
                status = self.neurogen.fork_engine.get_fork_status(fork_id)
                forks.append(status)
                
            self._format_and_print(forks)
    
    def _show_fork_status(self, fork_id):
        """Show fork status."""
        if not hasattr(self.neurogen, "fork_engine"):
            self.perror("Fork engine not available")
            return
            
        status = self.neurogen.fork_engine.get_fork_status(fork_id)
        self._format_and_print(status)
    
    def _evaluate_forks(self, fork_pair_id):
        """Evaluate forks."""
        if not hasattr(self.neurogen, "fork_engine"):
            self.perror("Fork engine not available")
            return
            
        result = self.neurogen.fork_engine.evaluate_forks(fork_pair_id)
        self._format_and_print(result)
    
    def _create_fork(self, reason):
        """Create fork."""
        if not hasattr(self.neurogen, "fork_engine"):
            self.perror("Fork engine not available")
            return
            
        context = self.neurogen.execution_loop.context
        result = self.neurogen.fork_engine.fork(reason, context)
        self._format_and_print(result)
    
    def _show_intent(self):
        """Show current intent vector."""
        if not hasattr(self.neurogen, "intent_vector"):
            self.perror("Intent vector not available")
            return
            
        intent = self.neurogen.intent_vector.get_vector_as_dict()
        self._format_and_print(intent)
    
    def _update_intent(self, dimension, value):
        """Update intent dimension."""
        if not hasattr(self.neurogen, "intent_vector"):
            self.perror("Intent vector not available")
            return
            
        # Check if dimension exists
        if dimension not in self.neurogen.intent_vector.dimensions:
            self.perror(f"Unknown dimension: {dimension}")
            return
            
        # Create shift dictionary with single dimension
        shift = {dimension: value - self.neurogen.intent_vector.dimensions[dimension]}
        
        # Apply update
        context = self.neurogen.execution_loop.context
        result = self.neurogen.intent_vector.update(shift, context, "CLI update")
        
        self.poutput(f"Updated {dimension} to {self.neurogen.intent_vector.dimensions[dimension]}")
    
    def _reset_intent(self):
        """Reset intent to defaults."""
        if not hasattr(self.neurogen, "intent_vector"):
            self.perror("Intent vector not available")
            return
            
        # Get default values from config
        config = self.neurogen.config.get("intent_vector", {})
        default_dimensions = config.get("dimensions", {
            "exploration": 0.5,
            "coherence": 0.7,
            "stability": 0.6,
            "knowledge": 0.6,
            "efficiency": 0.5,
            "abstraction": 0.4,
            "doctrinal_alignment": 0.8
        })
        
        # Create shift to reset all dimensions
        shift = {}
        for dim, default_val in default_dimensions.items():
            if dim in self.neurogen.intent_vector.dimensions:
                current = self.neurogen.intent_vector.dimensions[dim]
                shift[dim] = default_val - current
        
        # Apply update
        context = self.neurogen.execution_loop.context
        self.neurogen.intent_vector.update(shift, context, "CLI reset")
        
        self.poutput("Intent vector reset to defaults")
    
    def _show_intent_history(self):
        """Show intent evolution history."""
        if not hasattr(self.neurogen, "intent_vector"):
            self.perror("Intent vector not available")
            return
            
        # Get evolution report
        report = self.neurogen.intent_vector.get_evolution_report()
        self._format_and_print(report)
    
    def _list_constraints(self):
        """List current constraints."""
        if not hasattr(self.neurogen, "constraint_controller"):
            self.perror("Constraint controller not available")
            return
            
        constraints = self.neurogen.constraint_controller.get_constraints({})
        self._format_and_print(constraints)
    
    def _set_constraint(self, name, value):
        """Set constraint value."""
        if not hasattr(self.neurogen, "constraint_controller"):
            self.perror("Constraint controller not available")
            return
            
        # Check if constraint exists
        current = self.neurogen.constraint_controller.current_constraints
        if name not in current:
            self.perror(f"Unknown constraint: {name}")
            return
            
        # Update constraint
        current[name] = value
        self.neurogen.constraint_controller.current_constraints = current
        
        self.poutput(f"Constraint {name} set to {value}")
    
    def _reset_constraints(self):
        """Reset constraints to defaults."""
        if not hasattr(self.neurogen, "constraint_controller"):
            self.perror("Constraint controller not available")
            return
            
        self.neurogen.constraint_controller.reset_constraints()
        self.poutput("Constraints reset to defaults")
    
    def _test_constraint_enforcement(self, domain, operation_file):
        """Test constraint enforcement."""
        if not hasattr(self.neurogen, "constraint_enforcer"):
            self.perror("Constraint enforcer not available")
            return
            
        try:
            with open(operation_file, 'r') as f:
                operation = json.load(f)
                
            context = self.neurogen.execution_loop.context
            result = self.neurogen.constraint_enforcer.enforce(domain, operation, context)
            
            self._format_and_print(result)
            
        except Exception as e:
            self.perror(f"Enforcement error: {str(e)}")
    
    def _start_monitoring(self, interval):
        """Start automatic status monitoring."""
        if self.auto_status_timer:
            self.poutput("Monitoring already active")
            return
            
        self.auto_status_interval = interval
        
        # Define timer function
        def status_timer():
            if self.neurogen:
                status = self.neurogen.get_status()
                self.poutput("\n=== Status Update ===")
                self._format_and_print(status)
                
            # Reschedule if still active
            if self.auto_status_interval > 0:
                self.auto_status_timer = threading.Timer(
                    self.auto_status_interval, status_timer)
                self.auto_status_timer.daemon = True
                self.auto_status_timer.start()
        
        # Start initial timer
        self.auto_status_timer = threading.Timer(interval, status_timer)
        self.auto_status_timer.daemon = True
        self.auto_status_timer.start()
        
        self.poutput(f"Monitoring started with {interval}s interval")
    
    def _stop_monitoring(self):
        """Stop automatic status monitoring."""
        if self.auto_status_timer:
            self.auto_status_timer.cancel()
            self.auto_status_timer = None
            self.auto_status_interval = 0
            self.poutput("Monitoring stopped")
        else:
            self.poutput("Monitoring not active")
    
    def _set_monitor_interval(self, interval):
        """Change monitoring interval."""
        if interval <= 0:
            self._stop_monitoring()
            return
            
        if self.auto_status_timer:
            # Stop current timer
            self.auto_status_timer.cancel()
            self.auto_status_timer = None
            
            # Restart with new interval
            self._start_monitoring(interval)
        else:
            # Just update the interval for future use
            self.auto_status_interval = interval
            self.poutput(f"Monitor interval set to {interval}s (monitoring not active)")
    
    def _register_event_listener(self, event_type):
        """Register event listener."""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
            
        # Simple event handler
        def event_handler(event_data):
            self.poutput(f"\n[EVENT] {event_type}: {event_data}")
            
        self.event_listeners[event_type].append(event_handler)
        self.poutput(f"Event listener registered for {event_type}")
    
    def _unregister_event_listener(self, event_type):
        """Unregister event listener."""
        if event_type in self.event_listeners:
            self.event_listeners[event_type] = []
            self.poutput(f"Event listeners unregistered for {event_type}")
        else:
            self.perror(f"No listeners for event type: {event_type}")
    
    def _list_event_listeners(self):
        """List registered event listeners."""
        listeners = {}
        for event_type, handlers in self.event_listeners.items():
            listeners[event_type] = len(handlers)
            
        self._format_and_print(listeners)
    
    def _trigger_test_event(self, event_type, data_file):
        """Manually trigger event for testing."""
        try:
            with open(data_file, 'r') as f:
                event_data = json.load(f)
                
            event_data["type"] = event_type
            event_data["timestamp"] = time.time()
            
            if self.async_loop:
                asyncio.run_coroutine_threadsafe(
                    self._process_event(event_type, event_data), 
                    self.async_loop
                )
                
            self.poutput(f"Test event triggered: {event_type}")
            
        except Exception as e:
            self.perror(f"Event trigger error: {str(e)}")
    
    def _show_config(self):
        """Show current configuration."""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        self._format_and_print(self.neurogen.config)
    
    def _set_config(self, path, value):
        """Set configuration value."""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        # Navigate to the target config location
        path_parts = path.split(".")
        target = self.neurogen.config
        
        # Navigate to the parent object
        for i, part in enumerate(path_parts[:-1]):
            if part not in target:
                target[part] = {}
            target = target[part]
            
        # Set the value
        target[path_parts[-1]] = value
        
        self.poutput(f"Config {path} set to {value}")
    
    def _save_config(self, file_path):
        """Save current configuration to file."""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        try:
            with open(file_path, 'w') as f:
                json.dump(self.neurogen.config, f, indent=2)
                
            self.poutput(f"Configuration saved to {file_path}")
            
        except Exception as e:
            self.perror(f"Save config error: {str(e)}")
    
    def _load_config(self, file_path):
        """Load configuration from file."""
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
                
            if self.neurogen:
                self.neurogen.reconfigure(config)
                self.poutput("Configuration loaded and applied")
            else:
                self.perror("NEUROGEN system not initialized. Use 'init' with the config file.")
                
        except Exception as e:
            self.perror(f"Load config error: {str(e)}")
    
    def _set_debug_level(self, level):
        """Set debug level."""
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        if level not in levels:
            self.perror(f"Unknown debug level: {level}")
            self.poutput("Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL")
            return
            
        # Set log level
        self.log_level = levels[level]
        logging.getLogger().setLevel(self.log_level)
        
        self.poutput(f"Debug level set to {level}")
    
    def _set_execution_tracing(self, enable):
        """Enable/disable execution tracing."""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        if hasattr(self.neurogen, "set_execution_tracing"):
            self.neurogen.set_execution_tracing(enable)
            self.poutput(f"Execution tracing {'enabled' if enable else 'disabled'}")
        else:
            self.perror("Execution tracing not supported")
    
    def _dump_system_state(self, file_path):
        """Dump system state to file."""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        try:
            # Collect state data from all components
            state = {
                "timestamp": time.time(),
                "version": getattr(self.neurogen, "version", "unknown"),
                "system_id": getattr(self.neurogen, "system_id", "unknown"),
                "execution_state": {}
            }
            
            # Add execution loop state
            if hasattr(self.neurogen, "execution_loop"):
                state["execution_state"] = self.neurogen.execution_loop.get_status()
                
            # Add intent vector
            if hasattr(self.neurogen, "intent_vector"):
                state["intent"] = self.neurogen.intent_vector.get_vector_as_dict()
                
            # Add doctrine
            if hasattr(self.neurogen, "prime_directive"):
                state["doctrine"] = self.neurogen.prime_directive.get_current_version()
                
            # Add constraints
            if hasattr(self.neurogen, "constraint_controller"):
                state["constraints"] = self.neurogen.constraint_controller.get_constraints({})
                
            # Add evolution metrics
            if hasattr(self.neurogen, "evolution_auditor"):
                state["evolution"] = self.neurogen.evolution_auditor.get_system_state_report()
                
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            self.poutput(f"System state dumped to {file_path}")
            
        except Exception as e:
            self.perror(f"Dump error: {str(e)}")
    
    def _show_performance_profile(self):
        """Show performance profile."""
        if not self.neurogen:
            self.perror("NEUROGEN system not initialized")
            return
            
        profile = {}
        
        # Execution loop timing
        if hasattr(self.neurogen, "execution_loop"):
            exec_stats = self.neurogen.execution_loop.stats
            profile["execution"] = {
                "avg_cycle_time": exec_stats.get("avg_cycle_time", 0),
                "total_cycles": exec_stats.get("total_loops", 0),
                "successful_cycles": exec_stats.get("successful_loops", 0),
                "success_rate": (
                    exec_stats.get("successful_loops", 0) / 
                    max(1, exec_stats.get("total_loops", 1))
                )
            }
            
        # Memory performance
        if hasattr(self.neurogen, "agent_memory"):
            memory_stats = self.neurogen.agent_memory.get_stats()
            profile["memory"] = {
                "total_memories": memory_stats.get("total_memories", 0),
                "avg_access_time": memory_stats.get("avg_access_time", 0),
                "avg_attention_score": memory_stats.get("avg_attention_score", 0)
            }
            
        # Mutation performance
        if hasattr(self.neurogen, "mutator"):
            mutation_stats = self.neurogen.mutator.stats
            profile["mutation"] = {
                "total_mutations": mutation_stats.get("total_mutations", 0),
                "success_rate": (
                    mutation_stats.get("successful_mutations", 0) / 
                    max(1, mutation_stats.get("total_mutations", 1))
                ),
                "avg_improvements": mutation_stats.get("avg_magnitude", 0)
            }
            
        self._format_and_print(profile)
    
    def _show_command_history(self):
        """Show command history."""
        for i, cmd in enumerate(self.command_history[-self.max_history:]):
            self.poutput(f"{i+1}: {cmd}")
    
    def postcmd(self, stop, line):
        """Called after a command is executed."""
        if line.strip():
            self.command_history.append(line.strip())
            # Trim if too long
            while len(self.command_history) > self.max_history:
                self.command_history.pop(0)
                
        return stop
    
    def perror(self, errmsg):
        """Print error message."""
        logging.error(errmsg)
        super().perror(errmsg)

def main():
    """Main entry point."""
    cli = NeuroGenCLI()
    try:
        cli.cmdloop()
    except KeyboardInterrupt:
        cli.poutput("\nExiting...")
    finally:
        cli._cleanup()

if __name__ == "__main__":
    main()
