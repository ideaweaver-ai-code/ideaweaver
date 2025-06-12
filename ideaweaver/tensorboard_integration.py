#!/usr/bin/env python3
"""
TensorBoard Integration for IdeaWeaver CLI

This module provides TensorBoard experiment tracking to replace 
Weights & Biases functionality. Works perfectly with Python 3.13!
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

class TensorBoardTracker:
    """
    TensorBoard experiment tracker that provides similar functionality to W&B.
    
    ✅ Python 3.13 Compatible
    ✅ Local Experiment Tracking  
    ✅ No External Dependencies
    ✅ Rich Visualizations
    """
    
    def __init__(self, 
                 project_name: str = "ideaweaver-experiments",
                 experiment_name: str = None,
                 log_dir: str = None,
                 tags: List[str] = None,
                 config: Dict[str, Any] = None):
        """
        Initialize TensorBoard tracker.
        
        Args:
            project_name: Project name for organizing experiments
            experiment_name: Specific experiment name
            log_dir: Custom log directory (auto-generated if None)
            tags: List of tags for the experiment
            config: Configuration dictionary to log
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard not available. Install with: pip install tensorboard")
        
        self.project_name = project_name
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tags = tags or []
        self.config = config or {}
        self.start_time = time.time()
        
        # Create log directory
        if log_dir is None:
            base_dir = Path("tensorboard_logs") / project_name
            self.log_dir = base_dir / self.experiment_name
        else:
            self.log_dir = Path(log_dir)
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Log initial metadata
        self._log_experiment_metadata()
        
        print(f"🚀 TensorBoard tracker initialized")
        print(f"   📁 Log directory: {self.log_dir}")
        print(f"   🏷️  Experiment: {self.experiment_name}")
        print(f"   🌐 View with: tensorboard --logdir=tensorboard_logs")
    
    def _log_experiment_metadata(self):
        """Log experiment metadata and configuration."""
        # Log configuration as text
        if self.config:
            config_text = "\n".join([f"{k}: {v}" for k, v in self.config.items()])
            self.writer.add_text("experiment/config", config_text, 0)
        
        # Log tags
        if self.tags:
            self.writer.add_text("experiment/tags", ", ".join(self.tags), 0)
        
        # Log start time
        self.writer.add_text("experiment/start_time", 
                           datetime.fromtimestamp(self.start_time).isoformat(), 0)
        
        # Save config to JSON for external access
        metadata = {
            "experiment_name": self.experiment_name,
            "project_name": self.project_name,
            "start_time": self.start_time,
            "tags": self.tags,
            "config": self.config
        }
        
        with open(self.log_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def log_metric(self, name: str, value: Union[int, float], step: int = None):
        """Log a single metric value."""
        if step is None:
            step = int(time.time() - self.start_time)
        
        self.writer.add_scalar(name, value, step)
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: int = None):
        """Log multiple metrics at once."""
        if step is None:
            step = int(time.time() - self.start_time)
        
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(name, value, step)
    
    def log_evaluation_results(self, results: Dict[str, Any], prefix: str = "eval"):
        """
        Log evaluation results from lm-evaluation-harness.
        
        Args:
            results: Results dictionary from evaluator
            prefix: Prefix for metric names
        """
        print(f"📊 Logging evaluation results to TensorBoard...")
        
        # Log overall metrics
        if "results" in results:
            for task_name, task_results in results["results"].items():
                for metric_name, metric_value in task_results.items():
                    if isinstance(metric_value, (int, float)):
                        metric_path = f"{prefix}/{task_name}/{metric_name}"
                        self.log_metric(metric_path, metric_value)
        
        # Log model information
        if "model_info" in results:
            model_info = results["model_info"]
            if isinstance(model_info, dict):
                for key, value in model_info.items():
                    if isinstance(value, (int, float)):
                        self.log_metric(f"model/{key}", value)
        
        # Log evaluation config
        if "config" in results:
            config_text = json.dumps(results["config"], indent=2)
            self.writer.add_text("evaluation/config", config_text, 0)
        
        # Create summary table
        self._create_evaluation_summary_table(results)
        
        print(f"   ✅ Logged {len(results.get('results', {}))} task results")
    
    def _create_evaluation_summary_table(self, results: Dict[str, Any]):
        """Create a summary table of evaluation results."""
        if "results" not in results:
            return
        
        # Create markdown table
        table_lines = ["| Task | Metric | Value |", "|------|--------|-------|"]
        
        for task_name, task_results in results["results"].items():
            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, (int, float)):
                    table_lines.append(f"| {task_name} | {metric_name} | {metric_value:.4f} |")
        
        table_text = "\n".join(table_lines)
        self.writer.add_text("evaluation/summary_table", table_text, 0)
    
    def log_model_comparison(self, comparison_results: Dict[str, Dict], experiment_id: str = None):
        """
        Log model comparison results.
        
        Args:
            comparison_results: Dictionary with model_name -> results mapping
            experiment_id: Optional experiment identifier
        """
        print(f"🔄 Logging model comparison to TensorBoard...")
        
        # Create comparison metrics
        all_tasks = set()
        all_metrics = set()
        
        # Collect all tasks and metrics
        for model_results in comparison_results.values():
            if "results" in model_results:
                for task_name, task_results in model_results["results"].items():
                    all_tasks.add(task_name)
                    all_metrics.update(task_results.keys())
        
        # Log individual model results
        for model_name, model_results in comparison_results.items():
            clean_model_name = model_name.replace("/", "_").replace("-", "_")
            
            if "results" in model_results:
                for task_name, task_results in model_results["results"].items():
                    for metric_name, metric_value in task_results.items():
                        if isinstance(metric_value, (int, float)):
                            metric_path = f"comparison/{clean_model_name}/{task_name}/{metric_name}"
                            self.log_metric(metric_path, metric_value)
        
        # Create comparison table
        self._create_comparison_table(comparison_results, all_tasks, all_metrics)
        
        print(f"   ✅ Logged comparison for {len(comparison_results)} models")
    
    def _create_comparison_table(self, comparison_results: Dict, all_tasks: set, all_metrics: set):
        """Create a comprehensive comparison table."""
        # Create detailed comparison table
        table_lines = ["| Model | Task | Metric | Value |", "|-------|------|--------|-------|"]
        
        for model_name, model_results in comparison_results.items():
            clean_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
            
            if "results" in model_results:
                for task_name, task_results in model_results["results"].items():
                    for metric_name, metric_value in task_results.items():
                        if isinstance(metric_value, (int, float)):
                            table_lines.append(f"| {clean_model_name} | {task_name} | {metric_name} | {metric_value:.4f} |")
        
        table_text = "\n".join(table_lines)
        self.writer.add_text("comparison/detailed_table", table_text, 0)
        
        # Create best performer summary
        self._create_best_performer_summary(comparison_results)
    
    def _create_best_performer_summary(self, comparison_results: Dict):
        """Create a summary of best performing models."""
        best_performers = {}
        
        # Find best performer for each task/metric combination
        for model_name, model_results in comparison_results.items():
            if "results" in model_results:
                for task_name, task_results in model_results["results"].items():
                    for metric_name, metric_value in task_results.items():
                        if isinstance(metric_value, (int, float)):
                            key = f"{task_name}_{metric_name}"
                            if key not in best_performers or metric_value > best_performers[key][1]:
                                best_performers[key] = (model_name, metric_value)
        
        # Create best performer table
        summary_lines = ["| Task_Metric | Best Model | Value |", "|-------------|------------|-------|"]
        
        for key, (model_name, value) in best_performers.items():
            clean_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
            summary_lines.append(f"| {key} | {clean_model_name} | {value:.4f} |")
        
        summary_text = "\n".join(summary_lines)
        self.writer.add_text("comparison/best_performers", summary_text, 0)
    
    def log_system_info(self, system_info: Dict[str, Any]):
        """Log system information and environment details."""
        for key, value in system_info.items():
            if isinstance(value, (int, float)):
                self.log_metric(f"system/{key}", value)
            elif isinstance(value, str):
                self.writer.add_text(f"system/{key}", value, 0)
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information."""
        for key, value in dataset_info.items():
            if isinstance(value, (int, float)):
                self.log_metric(f"dataset/{key}", value)
            elif isinstance(value, str):
                self.writer.add_text(f"dataset/{key}", value, 0)
    
    def finalize(self, final_metrics: Dict[str, Any] = None):
        """Finalize the experiment and close the writer."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Log final metrics
        if final_metrics:
            for name, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    self.log_metric(f"final/{name}", value)
        
        # Log experiment summary
        summary = {
            "duration_seconds": duration,
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "status": "completed"
        }
        
        self.writer.add_text("experiment/summary", 
                           json.dumps(summary, indent=2), 0)
        
        # Close writer
        self.writer.close()
        
        print(f"✅ Experiment completed!")
        print(f"   ⏱️  Duration: {duration:.2f} seconds")
        print(f"   📁 Logs saved to: {self.log_dir}")
        print(f"   🌐 View with: tensorboard --logdir=tensorboard_logs")
    
    def get_log_dir(self) -> str:
        """Get the log directory path."""
        return str(self.log_dir)
    
    def get_experiment_url(self, host: str = "localhost", port: int = 6006) -> str:
        """Get the TensorBoard URL for this experiment."""
        return f"http://{host}:{port}/#timeseries&_experimentKindsFilter=ALL"

class TensorBoardEvaluationReporter:
    """Enhanced reporting for TensorBoard evaluation results."""
    
    @staticmethod
    def create_evaluation_report(results: Dict[str, Any], 
                               tracker: TensorBoardTracker,
                               model_name: str) -> str:
        """
        Create a comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            tracker: TensorBoard tracker instance
            model_name: Name of the evaluated model
            
        Returns:
            Path to the generated report file
        """
        report_path = tracker.log_dir / f"evaluation_report_{model_name.replace('/', '_')}.md"
        
        with open(report_path, "w") as f:
            f.write(f"# Model Evaluation Report\n\n")
            f.write(f"**Model**: {model_name}\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n")
            f.write(f"**TensorBoard Logs**: {tracker.get_log_dir()}\n\n")
            
            # Summary table
            if "results" in results:
                f.write("## Evaluation Results\n\n")
                f.write("| Task | Metric | Value |\n")
                f.write("|------|--------|---------|\n")
                
                for task_name, task_results in results["results"].items():
                    for metric_name, metric_value in task_results.items():
                        if isinstance(metric_value, (int, float)):
                            f.write(f"| {task_name} | {metric_name} | {metric_value:.4f} |\n")
            
            # Configuration
            if "config" in results:
                f.write("\n## Configuration\n\n")
                f.write("```json\n")
                f.write(json.dumps(results["config"], indent=2))
                f.write("\n```\n")
            
            # TensorBoard instructions
            f.write("\n## View Results\n\n")
            f.write("To view detailed results in TensorBoard:\n\n")
            f.write("```bash\n")
            f.write("tensorboard --logdir=tensorboard_logs\n")
            f.write("```\n\n")
            f.write(f"Then open: http://localhost:6006\n\n")
        
        return str(report_path)

def create_tensorboard_tracker(model_name: str = None, 
                             experiment_name: str = None,
                             config: Dict[str, Any] = None) -> TensorBoardTracker:
    """
    Factory function to create a TensorBoard tracker for IdeaWeaver evaluation.
    
    Args:
        model_name: Name of the model being evaluated
        experiment_name: Custom experiment name
        config: Configuration to log
        
    Returns:
        Configured TensorBoard tracker
    """
    if not experiment_name and model_name:
        # Create experiment name from model name
        clean_model_name = model_name.replace("/", "_").replace("-", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{clean_model_name}_{timestamp}"
    
    return TensorBoardTracker(
        project_name="ideaweaver-evaluation",
        experiment_name=experiment_name,
        config=config or {}
    )

def check_tensorboard_availability() -> bool:
    """Check if TensorBoard is available."""
    return TENSORBOARD_AVAILABLE

def get_tensorboard_installation_instructions() -> str:
    """Get installation instructions for TensorBoard."""
    return """
TensorBoard is not installed. To install:

pip install tensorboard

For Python 3.13 compatibility, TensorBoard is the recommended choice over MLflow.
""" 