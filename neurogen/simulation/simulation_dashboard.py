import os
import json
import time
import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class SimulationDashboard:
    """Visualizes and analyzes NEUROGEN simulation results for human auditing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Output directory for reports and visualizations
        self.output_dir = config.get("output_dir", "simulation/reports")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data collection
        self.simulation_data = {}
        self.current_simulation_id = None
        
        # Fixed color scheme for consistency
        self.colors = {
            "coherence": "#1f77b4",  # blue
            "reward": "#ff7f0e",     # orange
            "drift": "#2ca02c",      # green
            "success": "#d62728",    # red
            "mutation": "#9467bd",   # purple
            "fork": "#8c564b",       # brown
            "doctrine": "#e377c2",   # pink
            "memory": "#7f7f7f",     # gray
            "intent": "#bcbd22",     # olive
            "constraints": "#17becf"  # cyan
        }
        
        # Plot settings
        self.fig_size = (12, 8)
        self.dpi = 100
        self.line_width = 2
        self.grid_alpha = 0.3
        
    def register_simulation(self, simulation_id: str, metadata: Dict[str, Any]) -> None:
        """Register a new simulation for tracking."""
        if simulation_id in self.simulation_data:
            # Already registered
            return
            
        self.simulation_data[simulation_id] = {
            "metadata": metadata,
            "cycles": [],
            "metrics": {
                "coherence": [],
                "reward": [],
                "drift": [],
                "success_rate": [],
                "mutation_rate": [],
                "fork_events": [],
                "doctrine_changes": [],
                "memory_usage": [],
                "intent_divergence": [],
                "constraint_levels": {}
            },
            "start_time": time.time(),
            "last_update": time.time()
        }
        
        self.current_simulation_id = simulation_id
        
    def record_cycle(self, 
                   simulation_id: str, 
                   cycle_data: Dict[str, Any]) -> None:
        """Record a cycle's metrics for visualization."""
        if simulation_id not in self.simulation_data:
            return
            
        simulation = self.simulation_data[simulation_id]
        cycle_number = len(simulation["cycles"]) + 1
        
        # Store full cycle data
        simulation["cycles"].append(cycle_data)
        
        # Extract core metrics
        if "coherence" in cycle_data:
            simulation["metrics"]["coherence"].append(cycle_data["coherence"])
            
        if "reward" in cycle_data:
            simulation["metrics"]["reward"].append(cycle_data["reward"])
            
        if "drift" in cycle_data:
            simulation["metrics"]["drift"].append(cycle_data["drift"])
            
        if "success" in cycle_data:
            # Calculate running success rate
            current_success_rate = simulation["metrics"]["success_rate"][-1] if simulation["metrics"]["success_rate"] else 0
            success_val = 1 if cycle_data["success"] else 0
            
            if cycle_number == 1:
                new_rate = success_val
            else:
                new_rate = (current_success_rate * (cycle_number - 1) + success_val) / cycle_number
                
            simulation["metrics"]["success_rate"].append(new_rate)
        else:
            # If no explicit success flag, copy previous value or set 0
            prev_rate = simulation["metrics"]["success_rate"][-1] if simulation["metrics"]["success_rate"] else 0
            simulation["metrics"]["success_rate"].append(prev_rate)
            
        # Record mutation rate
        mutation_count = cycle_data.get("mutation_count", 0)
        mutation_rate = mutation_count / max(1, cycle_number)
        simulation["metrics"]["mutation_rate"].append(mutation_rate)
        
        # Check for fork events
        if cycle_data.get("fork_recommended", False) or cycle_data.get("fork_triggered", False):
            simulation["metrics"]["fork_events"].append(cycle_number)
            
        # Check for doctrine changes
        if cycle_data.get("doctrine_changed", False):
            simulation["metrics"]["doctrine_changes"].append(cycle_number)
            
        # Record memory usage
        memory_count = 0
        if "memory_links_used" in cycle_data:
            memory_count = len(cycle_data["memory_links_used"])
        simulation["metrics"]["memory_usage"].append(memory_count)
        
        # Record intent divergence if available
        if "intent_divergence" in cycle_data:
            simulation["metrics"]["intent_divergence"].append(cycle_data["intent_divergence"])
        else:
            # If not available, use previous or 0
            prev_div = simulation["metrics"]["intent_divergence"][-1] if simulation["metrics"]["intent_divergence"] else 0
            simulation["metrics"]["intent_divergence"].append(prev_div)
            
        # Record constraint levels
        if "constraints" in cycle_data:
            for key, value in cycle_data["constraints"].items():
                if key not in simulation["metrics"]["constraint_levels"]:
                    simulation["metrics"]["constraint_levels"][key] = []
                    
                # Ensure we have values for all cycles
                while len(simulation["metrics"]["constraint_levels"][key]) < cycle_number - 1:
                    simulation["metrics"]["constraint_levels"][key].append(None)
                    
                simulation["metrics"]["constraint_levels"][key].append(value)
        
        # Update timestamp
        simulation["last_update"] = time.time()
        
    def generate_report(self, 
                      simulation_id: Optional[str] = None, 
                      output_format: str = "html") -> str:
        """Generate a comprehensive simulation report."""
        # Use current simulation if not specified
        sim_id = simulation_id or self.current_simulation_id
        
        if not sim_id or sim_id not in self.simulation_data:
            return "No simulation data available"
            
        simulation = self.simulation_data[sim_id]
        
        # Create a report directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.output_dir, f"{sim_id}_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate plots
        self._generate_core_metrics_plot(simulation, report_dir)
        self._generate_drift_analysis_plot(simulation, report_dir)
        self._generate_evolution_events_plot(simulation, report_dir)
        self._generate_constraint_evolution_plot(simulation, report_dir)
        
        # Generate report
        if output_format.lower() == "html":
            report_path = self._generate_html_report(simulation, sim_id, report_dir)
        elif output_format.lower() == "json":
            report_path = self._generate_json_report(simulation, sim_id, report_dir)
        else:
            report_path = self._generate_text_report(simulation, sim_id, report_dir)
            
        return report_path
    
    def _generate_core_metrics_plot(self, 
                                 simulation: Dict[str, Any], 
                                 output_dir: str) -> str:
        """Generate plot of core performance metrics."""
        metrics = simulation["metrics"]
        cycles = range(1, len(simulation["cycles"]) + 1)
        
        plt.figure(figsize=self.fig_size, dpi=self.dpi)
        
        # Plot coherence
        if metrics["coherence"]:
            plt.plot(cycles, metrics["coherence"], color=self.colors["coherence"], 
                    linewidth=self.line_width, label="Coherence")
            
        # Plot reward
        if metrics["reward"]:
            plt.plot(cycles, metrics["reward"], color=self.colors["reward"], 
                    linewidth=self.line_width, label="Reward")
            
        # Plot success rate
        if metrics["success_rate"]:
            plt.plot(cycles, metrics["success_rate"], color=self.colors["success"], 
                    linewidth=self.line_width, label="Success Rate")
            
        plt.title("NEUROGEN Core Performance Metrics", fontsize=16)
        plt.xlabel("Execution Cycles", fontsize=12)
        plt.ylabel("Score (0-1)", fontsize=12)
        plt.grid(alpha=self.grid_alpha)
        plt.legend(loc="best", fontsize=12)
        
        # Save figure
        plot_path = os.path.join(output_dir, "core_metrics.png")
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def _generate_drift_analysis_plot(self, 
                                   simulation: Dict[str, Any], 
                                   output_dir: str) -> str:
        """Generate plot of drift and stability metrics."""
        metrics = simulation["metrics"]
        cycles = range(1, len(simulation["cycles"]) + 1)
        
        plt.figure(figsize=self.fig_size, dpi=self.dpi)
        
        # Plot drift
        if metrics["drift"]:
            plt.plot(cycles, metrics["drift"], color=self.colors["drift"], 
                    linewidth=self.line_width, label="Drift")
            
        # Plot intent divergence
        if metrics["intent_divergence"]:
            plt.plot(cycles, metrics["intent_divergence"], color=self.colors["intent"], 
                    linewidth=self.line_width, label="Intent Divergence")
            
        # Plot mutation rate
        if metrics["mutation_rate"]:
            plt.plot(cycles, metrics["mutation_rate"], color=self.colors["mutation"], 
                    linewidth=self.line_width, label="Mutation Rate")
            
        # Mark fork events
        if metrics["fork_events"]:
            for cycle in metrics["fork_events"]:
                plt.axvline(x=cycle, color=self.colors["fork"], linestyle='--', alpha=0.7)
                
        plt.title("NEUROGEN Drift & Stability Analysis", fontsize=16)
        plt.xlabel("Execution Cycles", fontsize=12)
        plt.ylabel("Magnitude", fontsize=12)
        plt.grid(alpha=self.grid_alpha)
        plt.legend(loc="best", fontsize=12)
        
        # Save figure
        plot_path = os.path.join(output_dir, "drift_analysis.png")
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def _generate_evolution_events_plot(self, 
                                     simulation: Dict[str, Any], 
                                     output_dir: str) -> str:
        """Generate plot of evolutionary events over time."""
        metrics = simulation["metrics"]
        cycles = range(1, len(simulation["cycles"]) + 1)
        
        plt.figure(figsize=self.fig_size, dpi=self.dpi)
        
        # Create a subplot for coherence as background context
        ax1 = plt.gca()
        if metrics["coherence"]:
            ax1.plot(cycles, metrics["coherence"], color=self.colors["coherence"], 
                    linewidth=self.line_width, label="Coherence")
            
        # Mark fork events
        if metrics["fork_events"]:
            for cycle in metrics["fork_events"]:
                plt.axvline(x=cycle, color=self.colors["fork"], linestyle='--', 
                          alpha=0.7, label="Fork" if cycle == metrics["fork_events"][0] else "")
                
        # Mark doctrine changes
        if metrics["doctrine_changes"]:
            for cycle in metrics["doctrine_changes"]:
                plt.axvline(x=cycle, color=self.colors["doctrine"], linestyle='-.',
                          alpha=0.7, label="Doctrine Change" if cycle == metrics["doctrine_changes"][0] else "")
                
        # Mark high mutation cycles (>0.5)
        if metrics["mutation_rate"]:
            high_mutation_cycles = [i+1 for i, rate in enumerate(metrics["mutation_rate"]) if rate > 0.5]
            if high_mutation_cycles:
                for cycle in high_mutation_cycles:
                    plt.axvline(x=cycle, color=self.colors["mutation"], linestyle=':',
                              alpha=0.5, label="High Mutation" if cycle == high_mutation_cycles[0] else "")
                
        plt.title("NEUROGEN Evolutionary Events Timeline", fontsize=16)
        plt.xlabel("Execution Cycles", fontsize=12)
        plt.ylabel("Coherence", fontsize=12)
        plt.grid(alpha=self.grid_alpha)
        plt.legend(loc="best", fontsize=12)
        
        # Save figure
        plot_path = os.path.join(output_dir, "evolution_events.png")
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
        
    def _generate_constraint_evolution_plot(self, 
                                         simulation: Dict[str, Any], 
                                         output_dir: str) -> str:
        """Generate plot of constraint evolution over time."""
        constraint_levels = simulation["metrics"]["constraint_levels"]
        if not constraint_levels:
            return ""
            
        cycles = range(1, len(simulation["cycles"]) + 1)
        
        plt.figure(figsize=self.fig_size, dpi=self.dpi)
        
        # Plot each constraint type
        for i, (constraint, values) in enumerate(constraint_levels.items()):
            # Handle missing values (pad with None)
            padded_values = values.copy()
            while len(padded_values) < len(cycles):
                padded_values.append(None)
                
            # Remove None values for plotting
            valid_indices = [i for i, v in enumerate(padded_values) if v is not None]
            valid_cycles = [cycles[i] for i in valid_indices]
            valid_values = [padded_values[i] for i in valid_indices]
            
            if valid_values:
                plt.plot(valid_cycles, valid_values, 
                       linewidth=self.line_width, 
                       label=constraint,
                       marker='o' if len(valid_values) < 10 else None)
            
        plt.title("NEUROGEN Constraint Evolution", fontsize=16)
        plt.xlabel("Execution Cycles", fontsize=12)
        plt.ylabel("Constraint Level", fontsize=12)
        plt.grid(alpha=self.grid_alpha)
        plt.legend(loc="best", fontsize=12)
        
        # Save figure
        plot_path = os.path.join(output_dir, "constraint_evolution.png")
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def _generate_html_report(self, 
                           simulation: Dict[str, Any], 
                           simulation_id: str,
                           report_dir: str) -> str:
        """Generate an HTML report with embedded visualizations."""
        metrics = simulation["metrics"]
        stats = self._calculate_summary_stats(simulation)
        
        # Create HTML report
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>NEUROGEN Simulation Report: {simulation_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        .metrics {{ display: flex; flex-wrap: wrap; margin-bottom: 20px; }}
        .metric-card {{ background: #f9f9f9; border-radius: 5px; padding: 15px; margin: 10px; width: 200px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
        .metric-name {{ font-size: 14px; color: #666; }}
        .visualization {{ margin: 30px 0; }}
        .visualization img {{ max-width: 100%; height: auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        table, th, td {{ border: 1px solid #ddd; }}
        th, td {{ padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>NEUROGEN Simulation Report</h1>
    <p><strong>Simulation ID:</strong> {simulation_id}</p>
    <p><strong>Duration:</strong> {stats['duration_str']}</p>
    <p><strong>Cycles:</strong> {stats['cycle_count']}</p>
    
    <h2>Summary Metrics</h2>
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-name">Final Coherence</div>
            <div class="metric-value">{stats['final_coherence']:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Avg Reward</div>
            <div class="metric-value">{stats['avg_reward']:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Success Rate</div>
            <div class="metric-value">{stats['success_rate']:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Drift (Avg/Max)</div>
            <div class="metric-value">{stats['avg_drift']:.2f}/{stats['max_drift']:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Mutation Rate</div>
            <div class="metric-value">{stats['mutation_rate']:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Fork Events</div>
            <div class="metric-value">{stats['fork_count']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Doctrine Changes</div>
            <div class="metric-value">{stats['doctrine_changes']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Avg Memory Usage</div>
            <div class="metric-value">{stats['avg_memory_usage']:.1f}</div>
        </div>
    </div>
    
    <h2>Performance Analysis</h2>
    <p>Coherence trend: {stats['coherence_trend_desc']}</p>
    <p>Reward trend: {stats['reward_trend_desc']}</p>
    <p>Drift stability: {stats['drift_stability_desc']}</p>
    
    <h2>Visualizations</h2>
    
    <div class="visualization">
        <h3>Core Performance Metrics</h3>
        <img src="core_metrics.png" alt="Core Metrics">
    </div>
    
    <div class="visualization">
        <h3>Drift Analysis</h3>
        <img src="drift_analysis.png" alt="Drift Analysis">
    </div>
    
    <div class="visualization">
        <h3>Evolution Events</h3>
        <img src="evolution_events.png" alt="Evolution Events">
    </div>
    
    <div class="visualization">
        <h3>Constraint Evolution</h3>
        <img src="constraint_evolution.png" alt="Constraint Evolution">
    </div>
    
    <h2>Evolution Timeline</h2>
    <table>
        <tr>
            <th>Cycle</th>
            <th>Event</th>
            <th>Coherence</th>
            <th>Reward</th>
            <th>Drift</th>
        </tr>
"""
        
        # Add key events to the timeline
        all_events = []
        for i, cycle in enumerate(metrics["fork_events"]):
            all_events.append((cycle, "Fork Event"))
            
        for i, cycle in enumerate(metrics["doctrine_changes"]):
            all_events.append((cycle, "Doctrine Change"))
            
        # Add high mutation cycles
        high_mutation_cycles = [i+1 for i, rate in enumerate(metrics["mutation_rate"]) if rate > 0.5]
        for cycle in high_mutation_cycles:
            all_events.append((cycle, "High Mutation"))
            
        # Sort events by cycle
        all_events.sort(key=lambda x: x[0])
        
        # Add events to table
        for cycle, event_type in all_events:
            if cycle <= len(simulation["cycles"]):
                cycle_idx = cycle - 1
                coherence = metrics["coherence"][cycle_idx] if metrics["coherence"] and cycle_idx < len(metrics["coherence"]) else "N/A"
                reward = metrics["reward"][cycle_idx] if metrics["reward"] and cycle_idx < len(metrics["reward"]) else "N/A"
                drift = metrics["drift"][cycle_idx] if metrics["drift"] and cycle_idx < len(metrics["drift"]) else "N/A"
                
                html_content += f"""
        <tr>
            <td>{cycle}</td>
            <td>{event_type}</td>
            <td>{coherence:.2f if isinstance(coherence, float) else coherence}</td>
            <td>{reward:.2f if isinstance(reward, float) else reward}</td>
            <td>{drift:.2f if isinstance(drift, float) else drift}</td>
        </tr>"""
        
        # Complete HTML
        html_content += """
    </table>
    
    <h2>System Assessment</h2>
    <p><strong>Stability Analysis:</strong> """ + stats["stability_assessment"] + """</p>
    <p><strong>Evolution Assessment:</strong> """ + stats["evolution_assessment"] + """</p>
    
    <footer>
        <p>Generated by NEUROGEN Simulation Dashboard on """ + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </footer>
</body>
</html>"""
        
        # Write HTML to file
        report_path = os.path.join(report_dir, "simulation_report.html")
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        return report_path
    
    def _generate_json_report(self, 
                           simulation: Dict[str, Any], 
                           simulation_id: str,
                           report_dir: str) -> str:
        """Generate a JSON report with summary statistics."""
        stats = self._calculate_summary_stats(simulation)
        
        # Create report structure
        report = {
            "simulation_id": simulation_id,
            "generated_at": datetime.datetime.now().isoformat(),
            "metrics": stats,
            "visualization_paths": {
                "core_metrics": "core_metrics.png",
                "drift_analysis": "drift_analysis.png",
                "evolution_events": "evolution_events.png",
                "constraint_evolution": "constraint_evolution.png"
            },
            "timeline": {
                "fork_events": simulation["metrics"]["fork_events"],
                "doctrine_changes": simulation["metrics"]["doctrine_changes"],
                "high_mutation_cycles": [i+1 for i, rate in enumerate(simulation["metrics"]["mutation_rate"]) 
                                       if rate > 0.5]
            },
            "raw_metrics": {
                "coherence": simulation["metrics"]["coherence"],
                "reward": simulation["metrics"]["reward"],
                "drift": simulation["metrics"]["drift"],
                "success_rate": simulation["metrics"]["success_rate"],
                "mutation_rate": simulation["metrics"]["mutation_rate"],
                "memory_usage": simulation["metrics"]["memory_usage"],
                "intent_divergence": simulation["metrics"]["intent_divergence"]
            }
        }
        
        # Write JSON to file
        report_path = os.path.join(report_dir, "simulation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report_path
    
    def _generate_text_report(self, 
                           simulation: Dict[str, Any], 
                           simulation_id: str,
                           report_dir: str) -> str:
        """Generate a plain text report."""
        stats = self._calculate_summary_stats(simulation)
        
        # Create text report
        text_content = f"""NEUROGEN Simulation Report
==========================
Simulation ID: {simulation_id}
Duration: {stats['duration_str']}
Cycles: {stats['cycle_count']}

Summary Metrics
--------------
Final Coherence: {stats['final_coherence']:.2f}
Avg Reward: {stats['avg_reward']:.2f}
Success Rate: {stats['success_rate']:.1%}
Drift (Avg/Max): {stats['avg_drift']:.2f}/{stats['max_drift']:.2f}
Mutation Rate: {stats['mutation_rate']:.1%}
Fork Events: {stats['fork_count']}
Doctrine Changes: {stats['doctrine_changes']}
Avg Memory Usage: {stats['avg_memory_usage']:.1f}

Performance Analysis
------------------
Coherence trend: {stats['coherence_trend_desc']}
Reward trend: {stats['reward_trend_desc']}
Drift stability: {stats['drift_stability_desc']}

Evolution Timeline
----------------
"""
        
        # Add key events to the timeline
        all_events = []
        for i, cycle in enumerate(simulation["metrics"]["fork_events"]):
            all_events.append((cycle, "Fork Event"))
            
        for i, cycle in enumerate(simulation["metrics"]["doctrine_changes"]):
            all_events.append((cycle, "Doctrine Change"))
            
        # Add high mutation cycles
        high_mutation_cycles = [i+1 for i, rate in enumerate(simulation["metrics"]["mutation_rate"]) if rate > 0.5]
        for cycle in high_mutation_cycles:
            all_events.append((cycle, "High Mutation"))
            
        # Sort events by cycle
        all_events.sort(key=lambda x: x[0])
        
        # Add events to text
        for cycle, event_type in all_events:
            metrics = simulation["metrics"]
            if cycle <= len(simulation["cycles"]):
                cycle_idx = cycle - 1
                coherence = metrics["coherence"][cycle_idx] if metrics["coherence"] and cycle_idx < len(metrics["coherence"]) else "N/A"
                reward = metrics["reward"][cycle_idx] if metrics["reward"] and cycle_idx < len(metrics["reward"]) else "N/A"
                drift = metrics["drift"][cycle_idx] if metrics["drift"] and cycle_idx < len(metrics["drift"]) else "N/A"
                
                text_content += f"Cycle {cycle}: {event_type} - Coherence: {coherence:.2f if isinstance(coherence, float) else coherence}, Reward: {reward:.2f if isinstance(reward, float) else reward}, Drift: {drift:.2f if isinstance(drift, float) else drift}\n"
        
        # Add assessment
        text_content += f"""
System Assessment
---------------
Stability Analysis: {stats["stability_assessment"]}

Evolution Assessment: {stats["evolution_assessment"]}

Generated by NEUROGEN Simulation Dashboard on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        # Write text to file
        report_path = os.path.join(report_dir, "simulation_report.txt")
        with open(report_path, 'w') as f:
            f.write(text_content)
            
        return report_path
    
    def _calculate_summary_stats(self, simulation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for reporting."""
        metrics = simulation["metrics"]
        
        # Basic counts
        cycle_count = len(simulation["cycles"])
        fork_count = len(metrics["fork_events"])
        doctrine_changes = len(metrics["doctrine_changes"])
        
        # Duration
        duration_seconds = simulation["last_update"] - simulation["start_time"]
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Calculate averages and trends
        avg_coherence = np.mean(metrics["coherence"]) if metrics["coherence"] else 0
        final_coherence = metrics["coherence"][-1] if metrics["coherence"] else 0
        
        avg_reward = np.mean(metrics["reward"]) if metrics["reward"] else 0
        final_reward = metrics["reward"][-1] if metrics["reward"] else 0
        
        avg_drift = np.mean(metrics["drift"]) if metrics["drift"] else 0
        max_drift = max(metrics["drift"]) if metrics["drift"] else 0
        
        success_rate = metrics["success_rate"][-1] if metrics["success_rate"] else 0
        
        mutation_rate = np.mean(metrics["mutation_rate"]) if metrics["mutation_rate"] else 0
        
        avg_memory_usage = np.mean(metrics["memory_usage"]) if metrics["memory_usage"] else 0
        
        # Calculate trends
        coherence_trend = self._calculate_trend(metrics["coherence"])
        reward_trend = self._calculate_trend(metrics["reward"])
        drift_trend = self._calculate_trend(metrics["drift"])
        
        # Create descriptive labels for trends
        coherence_trend_desc = self._trend_description(coherence_trend, "coherence")
        reward_trend_desc = self._trend_description(reward_trend, "reward")
        
        # Calculate drift stability
        if metrics["drift"] and len(metrics["drift"]) > 5:
            drift_volatility = np.std(metrics["drift"][-5:])
            drift_stability_desc = self._stability_description(drift_volatility, "drift")
        else:
            drift_stability_desc = "Insufficient data for drift stability assessment"
            
        # Overall system assessments
        stability_assessment = self._generate_stability_assessment({
            "coherence_trend": coherence_trend,
            "reward_trend": reward_trend,
            "drift_trend": drift_trend,
            "success_rate": success_rate,
            "mutation_rate": mutation_rate,
            "fork_count": fork_count,
            "cycle_count": cycle_count
        })
        
        evolution_assessment = self._generate_evolution_assessment({
            "coherence_trend": coherence_trend,
            "reward_trend": reward_trend,
            "doctrine_changes": doctrine_changes,
            "fork_count": fork_count,
            "cycle_count": cycle_count,
            "mutation_rate": mutation_rate
        })
        
        return {
            "cycle_count": cycle_count,
            "fork_count": fork_count,
            "doctrine_changes": doctrine_changes,
            "duration_seconds": duration_seconds,
            "duration_str": duration_str,
            "avg_coherence": avg_coherence,
            "final_coherence": final_coherence,
