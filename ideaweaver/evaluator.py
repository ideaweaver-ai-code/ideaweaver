"""
LLM Evaluation Module using EleutherAI lm-evaluation-harness and Weights & Biases
"""

import os
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import wandb
import click
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMEvaluator:
    """
    A comprehensive LLM evaluation class using lm-evaluation-harness and wandb
    """
    
    def __init__(self, 
                 model_path: str,
                 wandb_project: str = "llm-evaluation",
                 wandb_entity: Optional[str] = None,
                 device: str = "auto",
                 batch_size: int = 8,
                 verbose: bool = False):
        """
        Initialize the LLM Evaluator
        
        Args:
            model_path: Path to the model (local path or HuggingFace model ID)
            wandb_project: Weights & Biases project name
            wandb_entity: Weights & Biases entity (optional)
            device: Device to run evaluation on ('auto', 'cuda', 'cpu')
            batch_size: Batch size for evaluation
            verbose: Enable verbose logging
        """
        self.model_path = model_path
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Verify model exists
        self._verify_model()
        
        # Initialize wandb
        self._init_wandb()
    
    def _verify_model(self):
        """Verify that the model exists and is accessible"""
        try:
            if os.path.exists(self.model_path):
                # Local model
                if self.verbose:
                    click.echo(f"✅ Local model found: {self.model_path}")
            else:
                # Try to load from HuggingFace
                from transformers import AutoTokenizer
                AutoTokenizer.from_pretrained(self.model_path)
                if self.verbose:
                    click.echo(f"✅ HuggingFace model accessible: {self.model_path}")
        except Exception as e:
            raise ValueError(f"Model not found or inaccessible: {self.model_path}. Error: {str(e)}")
    
    def _init_wandb(self):
        """Initialize Weights & Biases"""
        try:
            # Check if wandb is already initialized
            if wandb.run is not None:
                wandb.finish()
            
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                config={
                    "model_path": self.model_path,
                    "device": self.device,
                    "batch_size": self.batch_size
                },
                tags=["llm-evaluation", "lm-eval-harness"]
            )
            
            if self.verbose:
                click.echo(f"✅ Wandb initialized: {self.wandb_run.url}")
                
        except Exception as e:
            if self.verbose:
                click.echo(f"⚠️  Warning: Wandb initialization failed: {str(e)}")
            self.wandb_run = None
    
    def evaluate_on_benchmarks(self, 
                              tasks: List[str],
                              num_fewshot: Optional[int] = None,
                              limit: Optional[int] = None,
                              output_path: Optional[str] = None) -> Dict:
        """
        Evaluate the model on specified benchmarks using lm-evaluation-harness
        
        Args:
            tasks: List of benchmark tasks to evaluate on
            num_fewshot: Number of few-shot examples (default: task-specific)
            limit: Limit number of samples for testing (optional)
            output_path: Path to save detailed results (optional)
            
        Returns:
            Dictionary containing evaluation results
        """
        
        # Create temporary output directory if not specified
        if output_path is None:
            output_path = tempfile.mkdtemp(prefix="llm_eval_")
        else:
            os.makedirs(output_path, exist_ok=True)
        
        if self.verbose:
            click.echo(f"📊 Starting evaluation on tasks: {', '.join(tasks)}")
            click.echo(f"📁 Output path: {output_path}")
        
        # Build lm_eval command
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={self.model_path}",
            "--tasks", ",".join(tasks),
            "--device", self.device,
            "--batch_size", str(self.batch_size),
            "--output_path", output_path,
            "--log_samples"
        ]
        
        # Add optional parameters
        if num_fewshot is not None:
            cmd.extend(["--num_fewshot", str(num_fewshot)])
        
        if limit is not None:
            cmd.extend(["--limit", str(limit)])
        
        # Add wandb integration
        if self.wandb_run:
            wandb_args = f"project={self.wandb_project}"
            if self.wandb_entity:
                wandb_args += f",entity={self.wandb_entity}"
            cmd.extend(["--wandb_args", wandb_args])
        
        if self.verbose:
            click.echo(f"🔧 Running command: {' '.join(cmd)}")
        
        try:
            # Run evaluation
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            end_time = time.time()
            
            if self.verbose:
                click.echo(f"✅ Evaluation completed in {end_time - start_time:.2f} seconds")
                click.echo("📋 Evaluation output:")
                click.echo(result.stdout)
            
            # Load and parse results
            results = self._parse_results(output_path, expected_tasks=tasks)
            
            # Log additional metrics to wandb
            if self.wandb_run:
                self._log_to_wandb(results, tasks, end_time - start_time)
            
            return results
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Evaluation failed: {e.stderr}"
            if self.verbose:
                click.echo(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def _find_all_results_files(self, output_path):
        import fnmatch
        matches = []
        for root, dirs, files in os.walk(output_path):
            for filename in files:
                if fnmatch.fnmatch(filename, "results*.json"):
                    matches.append(os.path.join(root, filename))
        return matches

    def _parse_results(self, output_path: str, expected_tasks: Optional[List[str]] = None) -> Dict:
        """Parse evaluation results from output directory, matching expected tasks if provided."""
        import os
        import json

        # Use os.walk to find all results*.json files, even in hidden directories
        matches = self._find_all_results_files(output_path)
        results_file = None
        selected_results = None
        normalized_expected = [t.strip().lower() for t in expected_tasks] if expected_tasks else []
        found_debug = []

        # Try to find a results file that contains all expected tasks (case/whitespace-insensitive)
        if matches and expected_tasks:
            for file in sorted(matches, key=os.path.getmtime, reverse=True):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    if "results" in data:
                        result_keys = [k.strip().lower() for k in data["results"].keys()]
                        found_debug.append((file, result_keys))
                        if all(t in result_keys for t in normalized_expected):
                            results_file = file
                            selected_results = data
                            break
                except Exception:
                    continue

        # If not found, fall back to the most recent results file, but print debug info
        if not selected_results and matches:
            results_file = max(matches, key=os.path.getmtime)
            try:
                with open(results_file, 'r') as f:
                    selected_results = json.load(f)
                if self.verbose:
                    click.echo(f"⚠️  No exact match for expected tasks. Falling back to most recent results file: {results_file}")
                    click.echo(f"Expected tasks: {normalized_expected}")
                    click.echo(f"Found in files:")
                    for fname, keys in found_debug:
                        click.echo(f"  {fname}: {keys}")
            except Exception:
                selected_results = None

        if selected_results:
            if self.verbose:
                click.echo(f"✅ Found and parsed results from: {results_file}")
                click.echo(f"DEBUG: Parsed results: {json.dumps(selected_results, indent=2)}")
            # If no metrics, raise error
            if not selected_results.get("results") or not any(isinstance(v, dict) and v for v in selected_results["results"].values()):
                raise RuntimeError("No valid metrics found in the results file. Please check your evaluation output.")
            return selected_results

        # If no results file found, raise error
        raise RuntimeError(f"No results*.json file found in {output_path} for the requested tasks. Please check that evaluation completed successfully and results were written.")
    
    def _log_to_wandb(self, results: Dict, tasks: List[str], duration: float):
        """Log evaluation results to Weights & Biases"""
        if not self.wandb_run:
            return
        
        try:
            # Log summary metrics
            summary_metrics = {
                "evaluation_duration": duration,
                "num_tasks": len(tasks),
                "tasks_evaluated": tasks
            }
            
            # Extract and log task-specific metrics
            for task_name, task_results in results.get("results", {}).items():
                for metric_name, metric_value in task_results.items():
                    if isinstance(metric_value, (int, float)):
                        summary_metrics[f"{task_name}_{metric_name}"] = metric_value
            
            # Log all metrics
            self.wandb_run.log(summary_metrics)
            
            # Create a summary table of results
            task_data = []
            for task_name, task_results in results.get("results", {}).items():
                task_row = {"task": task_name}
                for metric_name, metric_value in task_results.items():
                    if isinstance(metric_value, (int, float)):
                        task_row[metric_name] = metric_value
                task_data.append(task_row)
            
            if task_data:
                results_table = wandb.Table(data=task_data, columns=list(task_data[0].keys()))
                self.wandb_run.log({"evaluation_results": results_table})
            
            if self.verbose:
                click.echo("✅ Results logged to Weights & Biases")
                
        except Exception as e:
            if self.verbose:
                click.echo(f"⚠️  Warning: Failed to log to wandb: {str(e)}")
    
    def run_comprehensive_evaluation(self, 
                                   benchmark_suite: str = "standard",
                                   custom_tasks: Optional[List[str]] = None,
                                   num_fewshot: Optional[int] = None,
                                   limit: Optional[int] = None,
                                   output_path: Optional[str] = None) -> Dict:
        """
        Run a comprehensive evaluation using predefined benchmark suites
        
        Args:
            benchmark_suite: Predefined benchmark suite ('standard', 'reasoning', 'knowledge', 'custom')
            custom_tasks: Custom list of tasks (used when benchmark_suite='custom')
            num_fewshot: Number of few-shot examples
            limit: Limit number of samples for testing
            output_path: Path to save results
            
        Returns:
            Dictionary containing evaluation results
        """
        
        # Define benchmark suites
        benchmark_suites = {
            "standard": [
                "hellaswag",
                "arc_easy",
                "arc_challenge", 
                "winogrande",
                "piqa"
            ],
            "reasoning": [
                "hellaswag",
                "arc_challenge",
                "winogrande",
                "mathqa",
                "gsm8k"
            ],
            "knowledge": [
                "arc_easy",
                "arc_challenge",
                "mmlu",
                "truthfulqa_mc1",
                "truthfulqa_mc2"
            ],
            "comprehensive": [
                "hellaswag",
                "arc_easy", 
                "arc_challenge",
                "winogrande",
                "piqa",
                "mmlu",
                "truthfulqa_mc1",
                "gsm8k"
            ]
        }
        
        if benchmark_suite == "custom":
            if not custom_tasks:
                raise ValueError("custom_tasks must be provided when using 'custom' benchmark suite")
            tasks = custom_tasks
        elif benchmark_suite in benchmark_suites:
            tasks = benchmark_suites[benchmark_suite]
        else:
            available_suites = list(benchmark_suites.keys()) + ["custom"]
            raise ValueError(f"Unknown benchmark suite: {benchmark_suite}. Available: {available_suites}")
        
        if self.verbose:
            click.echo(f"🎯 Running {benchmark_suite} evaluation suite")
            click.echo(f"📝 Tasks: {', '.join(tasks)}")
        
        return self.evaluate_on_benchmarks(
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            output_path=output_path
        )
    
    def compare_models(self, 
                      model_paths: List[str],
                      tasks: List[str],
                      num_fewshot: Optional[int] = None,
                      limit: Optional[int] = None) -> Dict:
        """
        Compare multiple models on the same benchmarks
        
        Args:
            model_paths: List of model paths to compare
            tasks: List of tasks to evaluate on
            num_fewshot: Number of few-shot examples
            limit: Limit number of samples for testing
            
        Returns:
            Dictionary containing comparison results
        """
        
        comparison_results = {}
        
        for model_path in model_paths:
            if self.verbose:
                click.echo(f"🔄 Evaluating model: {model_path}")
            
            # Create new evaluator for each model
            evaluator = LLMEvaluator(
                model_path=model_path,
                wandb_project=self.wandb_project,
                wandb_entity=self.wandb_entity,
                device=self.device,
                batch_size=self.batch_size,
                verbose=self.verbose
            )
            
            results = evaluator.evaluate_on_benchmarks(
                tasks=tasks,
                num_fewshot=num_fewshot,
                limit=limit
            )
            
            comparison_results[model_path] = results
            
            # Close wandb run for this model
            if evaluator.wandb_run:
                evaluator.wandb_run.finish()
        
        # Log comparison to main wandb run
        if self.wandb_run:
            self._log_comparison_to_wandb(comparison_results, tasks)
        
        return comparison_results
    
    def _log_comparison_to_wandb(self, comparison_results: Dict, tasks: List[str]):
        """Log model comparison results to wandb"""
        if not self.wandb_run:
            return
        
        try:
            # Create comparison table
            comparison_data = []
            
            for model_path, results in comparison_results.items():
                model_row = {"model": model_path}
                
                for task_name in tasks:
                    task_results = results.get("results", {}).get(task_name, {})
                    for metric_name, metric_value in task_results.items():
                        if isinstance(metric_value, (int, float)):
                            model_row[f"{task_name}_{metric_name}"] = metric_value
                
                comparison_data.append(model_row)
            
            if comparison_data:
                comparison_table = wandb.Table(data=comparison_data, columns=list(comparison_data[0].keys()))
                self.wandb_run.log({"model_comparison": comparison_table})
                
                if self.verbose:
                    click.echo("✅ Model comparison logged to Weights & Biases")
                    
        except Exception as e:
            if self.verbose:
                click.echo(f"⚠️  Warning: Failed to log comparison to wandb: {str(e)}")
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available evaluation tasks from lm-evaluation-harness"""
        try:
            result = subprocess.run(
                ["lm_eval", "--list_tasks"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Parse the output to extract task names
            lines = result.stdout.strip().split('\n')
            tasks = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('Available'):
                    tasks.append(line)
            
            return sorted(tasks)
            
        except subprocess.CalledProcessError as e:
            if self.verbose:
                click.echo(f"⚠️  Warning: Could not get available tasks: {str(e)}")
            return []
    
    def finish(self):
        """Clean up and finish wandb run"""
        if self.wandb_run:
            self.wandb_run.finish()
            if self.verbose:
                click.echo("✅ Wandb run finished")


def create_evaluation_report(results: Dict, output_file: str = "evaluation_report.md"):
    """
    Create a markdown report from evaluation results
    
    Args:
        results: Evaluation results dictionary
        output_file: Output markdown file path
    """
    
    with open(output_file, 'w') as f:
        f.write("# LLM Evaluation Report\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if "config" in results:
            f.write("## Configuration\n\n")
            for key, value in results["config"].items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
        
        if "results" in results:
            f.write("## Results Summary\n\n")
            f.write("| Task | Metric | Value |\n")
            f.write("|------|--------|-------|\n")
            
            for task_name, task_results in results["results"].items():
                for metric_name, metric_value in task_results.items():
                    if isinstance(metric_value, (int, float)):
                        f.write(f"| {task_name} | {metric_name} | {metric_value:.4f} |\n")
            
            f.write("\n")
        
        f.write("## Detailed Results\n\n")
        f.write("```json\n")
        f.write(json.dumps(results, indent=2))
        f.write("\n```\n")
    
    click.echo(f"📄 Evaluation report saved to: {output_file}") 